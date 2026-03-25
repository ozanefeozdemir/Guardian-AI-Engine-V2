"""
CIC-IDS Flow Tracker
====================
Canlı ağ trafiğinden flow-bazlı feature extraction.
CIC-IDS 2017/2018 formatında 79 feature üretir.

Kullanım:
    tracker = CICFlowTracker(timeout=120.0, on_flow_ready=callback)
    sniff(prn=tracker.process_packet, store=False)
"""

from scapy.all import IP, TCP, UDP
import numpy as np
import time as _time


class CICFlowTracker:
    """
    Paketleri 5-tuple bazlı flow'lara gruplar ve CIC-IDS formatında
    79 feature çıkarır. Flow kapandığında (FIN/RST veya timeout)
    on_flow_ready callback'ini çağırır.
    """

    # --- Aktif/Idle eşik değeri (saniye) ---
    ACTIVE_TIMEOUT = 5.0  # 5 saniyeden uzun sessizlik → idle

    def __init__(self, timeout=120.0, on_flow_ready=None):
        """
        Args:
            timeout: Saniye cinsinden flow timeout süresi.
            on_flow_ready: Flow kapandığında çağrılacak callback.
                           Signature: callback(features_dict, src_ip, dst_ip)
        """
        self.active_flows = {}
        self.timeout = timeout
        self.on_flow_ready = on_flow_ready
        self._last_timeout_check = 0

    # ──────────────────────────────────────────────
    #  FLOW IDENTIFICATION
    # ──────────────────────────────────────────────

    def get_flow_id(self, pkt):
        """5-tuple bazlı flow ID üret. Mevcut flow varsa yönünü belirle."""
        if IP not in pkt or (TCP not in pkt and UDP not in pkt):
            return None, None, None

        src_ip, dst_ip, proto = pkt[IP].src, pkt[IP].dst, pkt[IP].proto
        src_port = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport
        dst_port = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport

        fwd_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}"
        bwd_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{proto}"

        if fwd_key in self.active_flows:
            return fwd_key, "fwd", None
        elif bwd_key in self.active_flows:
            return bwd_key, "bwd", None
        else:
            # Yeni flow meta bilgisi
            meta = {
                'src_ip': src_ip, 'dst_ip': dst_ip,
                'src_port': src_port, 'dst_port': dst_port,
                'protocol': proto,
            }
            return fwd_key, "new", meta

    # ──────────────────────────────────────────────
    #  PACKET PROCESSING
    # ──────────────────────────────────────────────

    def process_packet(self, pkt):
        """Ana giriş noktası. Her yakalanan paket buraya gelir."""
        flow_id, direction, meta = self.get_flow_id(pkt)
        if not flow_id:
            return

        current_time = float(pkt.time)
        pkt_len = len(pkt)

        # Header uzunluğu (IP + TCP/UDP)
        # Not: Scapy'de constructor ile oluşturulan paketlerde ihl/dataofs None olabilir
        ip_ihl = pkt[IP].ihl if pkt[IP].ihl is not None else 5
        ip_header_len = ip_ihl * 4
        if TCP in pkt:
            tcp_dataofs = pkt[TCP].dataofs if pkt[TCP].dataofs is not None else 5
            transport_header_len = tcp_dataofs * 4
        else:
            transport_header_len = 8  # UDP header sabit 8 byte
        header_len = ip_header_len + transport_header_len

        # Payload (segment) boyutu
        payload_len = max(0, pkt_len - header_len)

        # ── YENİ AKIŞ ──
        if direction == "new":
            self.active_flows[flow_id] = self._create_flow(
                pkt, current_time, pkt_len, header_len, payload_len, meta
            )
            direction = "fwd"

        # ── MEVCUT AKIŞI GÜNCELLE ──
        else:
            flow = self.active_flows[flow_id]
            flow['last_time'] = current_time
            flow['all_timestamps'].append(current_time)
            flow['all_pkt_lengths'].append(pkt_len)

            # Active / Idle tracking
            gap = current_time - flow['_prev_pkt_time']
            if gap > self.ACTIVE_TIMEOUT:
                flow['idle_times'].append(gap)
            else:
                flow['active_times'].append(gap)
            flow['_prev_pkt_time'] = current_time

            if direction == "fwd":
                flow['fwd_timestamps'].append(current_time)
                flow['fwd_pkt_lengths'].append(pkt_len)
                flow['fwd_header_len'] += header_len
                flow['fwd_seg_sizes'].append(payload_len)
                if payload_len > 0:
                    flow['fwd_data_pkts'] += 1
            elif direction == "bwd":
                flow['bwd_timestamps'].append(current_time)
                flow['bwd_pkt_lengths'].append(pkt_len)
                flow['bwd_header_len'] += header_len
                flow['bwd_seg_sizes'].append(payload_len)
                if flow['init_bwd_win_bytes'] == -1 and TCP in pkt:
                    flow['init_bwd_win_bytes'] = pkt[TCP].window

        # ── TCP BAYRAKLARI ──
        if TCP in pkt:
            flags = pkt[TCP].flags
            flow = self.active_flows[flow_id]
            if 'F' in flags: flow['fin_cnt'] += 1
            if 'S' in flags: flow['syn_cnt'] += 1
            if 'R' in flags: flow['rst_cnt'] += 1
            if 'P' in flags: flow['psh_cnt'] += 1
            if 'A' in flags: flow['ack_cnt'] += 1
            if 'U' in flags: flow['urg_cnt'] += 1
            if 'E' in flags: flow['ece_cnt'] += 1
            if 'C' in flags: flow['cwe_cnt'] += 1

            # Yön bazlı PSH/URG
            if direction == "fwd":
                if 'P' in flags: flow['fwd_psh_cnt'] += 1
                if 'U' in flags: flow['fwd_urg_cnt'] += 1
            elif direction == "bwd":
                if 'P' in flags: flow['bwd_psh_cnt'] += 1
                if 'U' in flags: flow['bwd_urg_cnt'] += 1

            # FIN veya RST → flow'u kapat
            if 'F' in flags or 'R' in flags:
                return self._close_flow(flow_id)

        # ── TIMEOUT KONTROLÜ (her 5 saniyede) ──
        if current_time - self._last_timeout_check > 5.0:
            self._last_timeout_check = current_time
            self.check_timeouts(current_time)

    # ──────────────────────────────────────────────
    #  FLOW CREATION
    # ──────────────────────────────────────────────

    def _create_flow(self, pkt, current_time, pkt_len, header_len, payload_len, meta):
        """Yeni flow state dict'i oluştur."""
        return {
            # Meta
            'src_ip': meta['src_ip'],
            'dst_ip': meta['dst_ip'],
            'src_port': meta['src_port'],
            'dst_port': meta['dst_port'],
            'protocol': meta['protocol'],

            # Zaman
            'start_time': current_time,
            'last_time': current_time,
            '_prev_pkt_time': current_time,

            # Paket ve boyut listeleri
            'fwd_pkt_lengths': [pkt_len],
            'bwd_pkt_lengths': [],
            'all_pkt_lengths': [pkt_len],
            'fwd_timestamps': [current_time],
            'bwd_timestamps': [],
            'all_timestamps': [current_time],

            # Segment boyutları
            'fwd_seg_sizes': [payload_len],
            'bwd_seg_sizes': [],
            'fwd_data_pkts': 1 if payload_len > 0 else 0,

            # Başlık ve Window
            'fwd_header_len': header_len,
            'bwd_header_len': 0,
            'init_fwd_win_bytes': pkt[TCP].window if TCP in pkt else 0,
            'init_bwd_win_bytes': -1,

            # TCP bayrak sayaçları (toplam)
            'fin_cnt': 0, 'syn_cnt': 0, 'rst_cnt': 0, 'psh_cnt': 0,
            'ack_cnt': 0, 'urg_cnt': 0, 'ece_cnt': 0, 'cwe_cnt': 0,

            # Yön bazlı PSH/URG
            'fwd_psh_cnt': 0, 'bwd_psh_cnt': 0,
            'fwd_urg_cnt': 0, 'bwd_urg_cnt': 0,

            # Active / Idle
            'active_times': [],
            'idle_times': [],
        }

    # ──────────────────────────────────────────────
    #  TIMEOUT MANAGEMENT
    # ──────────────────────────────────────────────

    def check_timeouts(self, current_time):
        """Timeout'a uğramış flow'ları tespit edip kapat."""
        expired = [
            fid for fid, f in self.active_flows.items()
            if (current_time - f['last_time']) > self.timeout
        ]
        for fid in expired:
            self._close_flow(fid)

    # ──────────────────────────────────────────────
    #  FLOW CLOSING & FEATURE EXPORT
    # ──────────────────────────────────────────────

    def _close_flow(self, flow_id):
        """Flow'u kapat, feature'ları çıkar ve callback'i çağır."""
        if flow_id not in self.active_flows:
            return None

        features = self.extract_features(flow_id)
        flow = self.active_flows.pop(flow_id)

        if self.on_flow_ready:
            self.on_flow_ready(features, flow['src_ip'], flow['dst_ip'])

        return features

    def flush_all(self):
        """Tüm aktif flow'ları zorla kapat (shutdown için)."""
        for fid in list(self.active_flows.keys()):
            self._close_flow(fid)

    # ──────────────────────────────────────────────
    #  HELPER CALCULATIONS
    # ──────────────────────────────────────────────

    @staticmethod
    def _calc_iat(timestamps):
        """Zaman damgalarından IAT dizisi üret."""
        if len(timestamps) < 2:
            return [0.0]
        return np.diff(timestamps).tolist()

    @staticmethod
    def _safe_stat(data, stat):
        """Boş listelerde hata almamak için güvenli istatistik."""
        if not data:
            return 0.0
        arr = np.array(data, dtype=np.float64)
        if stat == 'mean': return float(np.mean(arr))
        if stat == 'std':  return float(np.std(arr))
        if stat == 'max':  return float(np.max(arr))
        if stat == 'min':  return float(np.min(arr))
        if stat == 'tot':  return float(np.sum(arr))
        if stat == 'var':  return float(np.var(arr))
        return 0.0

    # ──────────────────────────────────────────────
    #  FEATURE EXTRACTION (84 features, CIC-IDS 2017 TrafficLab format)
    # ──────────────────────────────────────────────

    def extract_features(self, flow_id):
        """
        CIC-IDS 2017 TrafficLab formatında 84 feature üretir.
        cols.txt + cols_2017_trafficlab.txt'deki tüm kolonları kapsar.
        """
        flow = self.active_flows[flow_id]

        # Temel hesaplamalar
        flow_duration = max((flow['last_time'] - flow['start_time']) * 1e6, 1.0)  # mikrosaniye
        fwd_iat = self._calc_iat(flow['fwd_timestamps'])
        bwd_iat = self._calc_iat(flow['bwd_timestamps'])
        flow_iat = self._calc_iat(flow['all_timestamps'])

        tot_fwd_pkts = len(flow['fwd_pkt_lengths'])
        tot_bwd_pkts = len(flow['bwd_pkt_lengths'])
        tot_pkts = tot_fwd_pkts + tot_bwd_pkts
        tot_bytes = sum(flow['all_pkt_lengths'])

        # Minimum segment boyutu (fwd)
        fwd_seg_min = min(flow['fwd_seg_sizes']) if flow['fwd_seg_sizes'] else 0

        features = {
            # ── Kimlik (CIC-IDS 2017 TrafficLab) ──
            'Flow ID':          f"{flow['src_ip']}-{flow['dst_ip']}-{flow['src_port']}-{flow['dst_port']}-{flow['protocol']}",
            'Source IP':        flow['src_ip'],
            'Source Port':      flow['src_port'],
            'Destination IP':   flow['dst_ip'],

            # ── Temel ──
            'Dst Port':         flow['dst_port'],
            'Protocol':         flow['protocol'],
            'Timestamp':        flow['start_time'],
            'Flow Duration':    flow_duration,
            'Tot Fwd Pkts':     tot_fwd_pkts,
            'Tot Bwd Pkts':     tot_bwd_pkts,
            'TotLen Fwd Pkts':  sum(flow['fwd_pkt_lengths']),
            'TotLen Bwd Pkts':  sum(flow['bwd_pkt_lengths']),

            # ── Fwd Paket Len İstatistikleri ──
            'Fwd Pkt Len Max':  self._safe_stat(flow['fwd_pkt_lengths'], 'max'),
            'Fwd Pkt Len Min':  self._safe_stat(flow['fwd_pkt_lengths'], 'min'),
            'Fwd Pkt Len Mean': self._safe_stat(flow['fwd_pkt_lengths'], 'mean'),
            'Fwd Pkt Len Std':  self._safe_stat(flow['fwd_pkt_lengths'], 'std'),

            # ── Bwd Paket Len İstatistikleri ──
            'Bwd Pkt Len Max':  self._safe_stat(flow['bwd_pkt_lengths'], 'max'),
            'Bwd Pkt Len Min':  self._safe_stat(flow['bwd_pkt_lengths'], 'min'),
            'Bwd Pkt Len Mean': self._safe_stat(flow['bwd_pkt_lengths'], 'mean'),
            'Bwd Pkt Len Std':  self._safe_stat(flow['bwd_pkt_lengths'], 'std'),

            # ── Hız ve Oranlar ──
            'Flow Byts/s':      (tot_bytes / flow_duration) * 1e6,
            'Flow Pkts/s':      (tot_pkts / flow_duration) * 1e6,

            # ── Flow IAT ──
            'Flow IAT Mean':    self._safe_stat(flow_iat, 'mean'),
            'Flow IAT Std':     self._safe_stat(flow_iat, 'std'),
            'Flow IAT Max':     self._safe_stat(flow_iat, 'max'),
            'Flow IAT Min':     self._safe_stat(flow_iat, 'min'),

            # ── Fwd IAT ──
            'Fwd IAT Tot':      self._safe_stat(fwd_iat, 'tot'),
            'Fwd IAT Mean':     self._safe_stat(fwd_iat, 'mean'),
            'Fwd IAT Std':      self._safe_stat(fwd_iat, 'std'),
            'Fwd IAT Max':      self._safe_stat(fwd_iat, 'max'),
            'Fwd IAT Min':      self._safe_stat(fwd_iat, 'min'),

            # ── Bwd IAT ──
            'Bwd IAT Tot':      self._safe_stat(bwd_iat, 'tot'),
            'Bwd IAT Mean':     self._safe_stat(bwd_iat, 'mean'),
            'Bwd IAT Std':      self._safe_stat(bwd_iat, 'std'),
            'Bwd IAT Max':      self._safe_stat(bwd_iat, 'max'),
            'Bwd IAT Min':      self._safe_stat(bwd_iat, 'min'),

            # ── Yön Bazlı Bayraklar ──
            'Fwd PSH Flags':    flow['fwd_psh_cnt'],
            'Bwd PSH Flags':    flow['bwd_psh_cnt'],
            'Fwd URG Flags':    flow['fwd_urg_cnt'],
            'Bwd URG Flags':    flow['bwd_urg_cnt'],

            # ── Başlık Uzunlukları ──
            'Fwd Header Len':   flow['fwd_header_len'],
            'Bwd Header Len':   flow['bwd_header_len'],

            # ── Paket Hızları ──
            'Fwd Pkts/s':       (tot_fwd_pkts / flow_duration) * 1e6,
            'Bwd Pkts/s':       (tot_bwd_pkts / flow_duration) * 1e6,

            # ── Genel Paket Len İstatistikleri ──
            'Pkt Len Min':      self._safe_stat(flow['all_pkt_lengths'], 'min'),
            'Pkt Len Max':      self._safe_stat(flow['all_pkt_lengths'], 'max'),
            'Pkt Len Mean':     self._safe_stat(flow['all_pkt_lengths'], 'mean'),
            'Pkt Len Std':      self._safe_stat(flow['all_pkt_lengths'], 'std'),
            'Pkt Len Var':      self._safe_stat(flow['all_pkt_lengths'], 'var'),

            # ── TCP Bayrak Sayaçları ──
            'FIN Flag Cnt':     flow['fin_cnt'],
            'SYN Flag Cnt':     flow['syn_cnt'],
            'RST Flag Cnt':     flow['rst_cnt'],
            'PSH Flag Cnt':     flow['psh_cnt'],
            'ACK Flag Cnt':     flow['ack_cnt'],
            'URG Flag Cnt':     flow['urg_cnt'],
            'CWE Flag Count':   flow['cwe_cnt'],
            'ECE Flag Cnt':     flow['ece_cnt'],

            # ── Oranlar ve Ortalamalar ──
            'Down/Up Ratio':    tot_bwd_pkts / tot_fwd_pkts if tot_fwd_pkts > 0 else 0,
            'Pkt Size Avg':     tot_bytes / tot_pkts if tot_pkts > 0 else 0,
            'Fwd Seg Size Avg': self._safe_stat(flow['fwd_seg_sizes'], 'mean'),
            'Bwd Seg Size Avg': self._safe_stat(flow['bwd_seg_sizes'], 'mean'),

            # ── Bulk İstatistikleri (CICFlowMeter'da genellikle 0) ──
            'Fwd Byts/b Avg':   0, 'Fwd Pkts/b Avg':  0, 'Fwd Blk Rate Avg': 0,
            'Bwd Byts/b Avg':   0, 'Bwd Pkts/b Avg':  0, 'Bwd Blk Rate Avg': 0,

            # ── Subflow (= toplam değerlerle aynı, tek subflow var) ──
            'Subflow Fwd Pkts': tot_fwd_pkts,
            'Subflow Fwd Byts': sum(flow['fwd_pkt_lengths']),
            'Subflow Bwd Pkts': tot_bwd_pkts,
            'Subflow Bwd Byts': sum(flow['bwd_pkt_lengths']),

            # ── Window & Segment ──
            'Init Fwd Win Byts': flow['init_fwd_win_bytes'],
            'Init Bwd Win Byts': flow['init_bwd_win_bytes'] if flow['init_bwd_win_bytes'] != -1 else 0,
            'Fwd Act Data Pkts': flow['fwd_data_pkts'],
            'Fwd Header Length.1': flow['fwd_header_len'],  # CIC-IDS 2017 duplikat kolon
            'Fwd Seg Size Min':  fwd_seg_min,

            # ── Active / Idle Süreleri ──
            'Active Mean':      self._safe_stat(flow['active_times'], 'mean'),
            'Active Std':       self._safe_stat(flow['active_times'], 'std'),
            'Active Max':       self._safe_stat(flow['active_times'], 'max'),
            'Active Min':       self._safe_stat(flow['active_times'], 'min'),
            'Idle Mean':        self._safe_stat(flow['idle_times'], 'mean'),
            'Idle Std':         self._safe_stat(flow['idle_times'], 'std'),
            'Idle Max':         self._safe_stat(flow['idle_times'], 'max'),
            'Idle Min':         self._safe_stat(flow['idle_times'], 'min'),
        }

        return features


if __name__ == "__main__":
    from scapy.all import sniff
    import json

    tracker = CICFlowTracker(
        timeout=120.0,
        on_flow_ready=lambda feats, src, dst: print(
            f"\n[FLOW] {src} → {dst}\n{json.dumps(feats, indent=2)}"
        )
    )

    print("Sniffing 200 packets...")
    sniff(prn=tracker.process_packet, count=200)
    tracker.flush_all()