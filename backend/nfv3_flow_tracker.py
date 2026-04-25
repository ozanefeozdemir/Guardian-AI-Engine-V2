"""
NF-v3 Flow Tracker
==================
Canli ag trafigindan flow-bazli feature extraction.
NF-v3 (NetFlow) formatinda 53 feature uretir.

FlowGuard modeli (canavar-model) bu feature'lari bekler.
CICFlowTracker ile ayni mimariyi kullanir ama farkli feature seti uretir.

Kullanim:
    tracker = NFv3FlowTracker(timeout=120.0, on_flow_ready=callback)
    sniff(prn=tracker.process_packet, store=False)

Urettigi feature'lar (53 adet):
    - PROTOCOL, L7_PROTO (protokol)
    - IN_BYTES, IN_PKTS, OUT_BYTES, OUT_PKTS (hacim)
    - TCP_FLAGS, CLIENT_TCP_FLAGS, SERVER_TCP_FLAGS (bayraklar)
    - FLOW_DURATION_MILLISECONDS, DURATION_IN, DURATION_OUT (sure)
    - MIN_TTL, MAX_TTL (TTL)
    - LONGEST_FLOW_PKT, SHORTEST_FLOW_PKT, MIN_IP_PKT_LEN, MAX_IP_PKT_LEN
    - SRC_TO_DST_SECOND_BYTES, DST_TO_SRC_SECOND_BYTES (oran)
    - SRC_TO_DST_AVG_THROUGHPUT, DST_TO_SRC_AVG_THROUGHPUT
    - RETRANSMITTED_IN_BYTES/PKTS, RETRANSMITTED_OUT_BYTES/PKTS
    - NUM_PKTS_UP_TO_128_BYTES .. NUM_PKTS_1024_TO_1514_BYTES
    - TCP_WIN_MAX_IN, TCP_WIN_MAX_OUT
    - ICMP_TYPE, ICMP_IPV4_TYPE
    - DNS_QUERY_ID, DNS_QUERY_TYPE, DNS_TTL_ANSWER
    - FTP_COMMAND_RET_CODE
    - SRC_TO_DST_IAT_* , DST_TO_SRC_IAT_* (8 IAT feature)
    - 6 port bucket feature (SRC/DST WELL_KNOWN/REGISTERED/EPHEMERAL)
"""

from scapy.all import IP, TCP, UDP, ICMP, DNS, Raw
import numpy as np


class NFv3FlowTracker:
    """
    Paketleri 5-tuple bazli flow'lara gruplar ve NF-v3 formatinda
    53 feature cikarir. Flow kapandiginda (FIN/RST veya timeout)
    on_flow_ready callback'ini cagirir.
    """

    def __init__(self, timeout=120.0, on_flow_ready=None):
        """
        Args:
            timeout: Saniye cinsinden flow timeout suresi.
            on_flow_ready: Flow kapandiginda cagirilacak callback.
                           Signature: callback(features_dict, src_ip, dst_ip)
        """
        self.active_flows = {}
        self.timeout = timeout
        self.on_flow_ready = on_flow_ready
        self._last_timeout_check = 0

    # ---------------------------------------------------------------
    #  FLOW IDENTIFICATION
    # ---------------------------------------------------------------

    def get_flow_id(self, pkt):
        """5-tuple bazli flow ID uret. Mevcut flow varsa yonunu belirle."""
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
            meta = {
                'src_ip': src_ip, 'dst_ip': dst_ip,
                'src_port': src_port, 'dst_port': dst_port,
                'protocol': proto,
            }
            return fwd_key, "new", meta

    # ---------------------------------------------------------------
    #  PACKET PROCESSING
    # ---------------------------------------------------------------

    def process_packet(self, pkt):
        """Ana giris noktasi. Her yakalanan paket buraya gelir."""
        flow_id, direction, meta = self.get_flow_id(pkt)
        if not flow_id:
            return

        current_time = float(pkt.time)
        # Fix: NetFlow v3 measures L3 payload, so we use IP len instead of raw L2 frame size
        pkt_len = pkt[IP].len if IP in pkt else len(pkt)
        ttl = pkt[IP].ttl if IP in pkt else 0

        # -- YENI AKIS --
        is_first_packet = False
        if direction == "new":
            # Orphaned (Yarım) Akış Koruması
            # Model datasetleri (CIC/UNSW) tam akışlardan oluşur. 
            # Canlı ortamda dinlemeye ortadan başladığımızda gelen saf ACK/PSH paketleri 
            # SYN barındırmadığı için, NIDS tarafından 0 süreli sahte Port Scan olarak algılanır.
            if TCP in pkt and 'S' not in str(pkt[TCP].flags):
                return
                
            self.active_flows[flow_id] = self._create_flow(
                pkt, current_time, pkt_len, ttl, meta
            )
            direction = "fwd"
            is_first_packet = True

        # -- MEVCUT AKISI GUNCELLE --
        flow = self.active_flows[flow_id]
        flow['last_time'] = current_time

        if not is_first_packet:
            # TTL (ilk paket _create_flow'da eklendi)
            flow['ttl_values'].append(ttl)
            
            # Paket boyutu dagilimi (ilk paket _create_flow'da eklendi)
            if pkt_len <= 128:
                flow['pkts_up_to_128'] += 1
            elif pkt_len <= 256:
                flow['pkts_128_to_256'] += 1
            elif pkt_len <= 512:
                flow['pkts_256_to_512'] += 1
            elif pkt_len <= 1024:
                flow['pkts_512_to_1024'] += 1
            else:
                flow['pkts_1024_to_1514'] += 1

            # Yon bazli metrikler
            if direction == "fwd":
                flow['in_bytes'] += pkt_len
                flow['in_pkts'] += 1
                flow['fwd_pkt_lengths'].append(pkt_len)
                flow['fwd_timestamps'].append(current_time)
            elif direction == "bwd":
                flow['out_bytes'] += pkt_len
                flow['out_pkts'] += 1
                flow['bwd_pkt_lengths'].append(pkt_len)
                flow['bwd_timestamps'].append(current_time)

        if not is_first_packet:
            if direction == "fwd":
                if TCP in pkt:
                    win = pkt[TCP].window
                    if win > flow['tcp_win_max_in']:
                        flow['tcp_win_max_in'] = win
                    
                    seq = pkt[TCP].seq
                    if seq in flow['_fwd_seen_seqs']:
                        flow['retransmitted_in_bytes'] += pkt_len
                        flow['retransmitted_in_pkts'] += 1
                    else:
                        flow['_fwd_seen_seqs'].add(seq)
            elif direction == "bwd":
                if TCP in pkt:
                    win = pkt[TCP].window
                    if win > flow['tcp_win_max_out']:
                        flow['tcp_win_max_out'] = win

                    seq = pkt[TCP].seq
                    if seq in flow['_bwd_seen_seqs']:
                        flow['retransmitted_out_bytes'] += pkt_len
                        flow['retransmitted_out_pkts'] += 1
                    else:
                        flow['_bwd_seen_seqs'].add(seq)

        # -- TCP BAYRAKLARI --
        if TCP in pkt:
            flags_int = int(pkt[TCP].flags)
            flow['tcp_flags_all'] |= flags_int
            if direction == "fwd":
                flow['client_tcp_flags'] |= flags_int
            elif direction == "bwd":
                flow['server_tcp_flags'] |= flags_int

            # FIN veya RST -> flow'u kapat
            flags = pkt[TCP].flags
            if 'F' in flags or 'R' in flags:
                return self._close_flow(flow_id)

        # -- ICMP bilgileri --
        if ICMP in pkt:
            flow['icmp_type'] = pkt[ICMP].type
            flow['icmp_code'] = pkt[ICMP].code

        # -- DNS bilgileri --
        if DNS in pkt and pkt.haslayer(DNS):
            dns = pkt[DNS]
            if dns.qr == 0:  # Query
                flow['dns_query_id'] = dns.id
                if dns.qdcount > 0 and dns.qd:
                    flow['dns_query_type'] = dns.qd.qtype
            elif dns.qr == 1:  # Response
                if dns.ancount > 0 and dns.an:
                    flow['dns_ttl_answer'] = dns.an.ttl

        # -- TIMEOUT KONTROLU (her 5 saniyede) --
        if current_time - self._last_timeout_check > 5.0:
            self._last_timeout_check = current_time
            self.check_timeouts(current_time)

    # ---------------------------------------------------------------
    #  FLOW CREATION
    # ---------------------------------------------------------------

    def _create_flow(self, pkt, current_time, pkt_len, ttl, meta):
        """Yeni flow state dict'i olustur."""
        flow = {
            # Meta
            'src_ip': meta['src_ip'],
            'dst_ip': meta['dst_ip'],
            'src_port': meta['src_port'],
            'dst_port': meta['dst_port'],
            'protocol': meta['protocol'],

            # L7 Proto (basit tahmin: port bazli)
            'l7_proto': self._guess_l7_proto(meta['dst_port'], meta['protocol']),

            # Zaman
            'start_time': current_time,
            'last_time': current_time,

            # Hacim (fwd = src->dst = IN, bwd = dst->src = OUT)
            'in_bytes': pkt_len,
            'in_pkts': 1,
            'out_bytes': 0,
            'out_pkts': 0,

            # Paket boyutlari
            'fwd_pkt_lengths': [pkt_len],
            'bwd_pkt_lengths': [],
            'fwd_timestamps': [current_time],
            'bwd_timestamps': [],

            # TTL
            'ttl_values': [ttl],

            # TCP bayraklari (bitwise OR)
            'tcp_flags_all': 0,
            'client_tcp_flags': 0,
            'server_tcp_flags': 0,

            # TCP Window
            'tcp_win_max_in': pkt[TCP].window if TCP in pkt else 0,
            'tcp_win_max_out': 0,

            # Retransmission
            'retransmitted_in_bytes': 0,
            'retransmitted_in_pkts': 0,
            'retransmitted_out_bytes': 0,
            'retransmitted_out_pkts': 0,

            # Retransmission tespiti icin gorulmus seq numaralari
            '_fwd_seen_seqs': set(),
            '_bwd_seen_seqs': set(),

            # Paket boyutu dagilimi
            'pkts_up_to_128': 1 if pkt_len <= 128 else 0,
            'pkts_128_to_256': 1 if 128 < pkt_len <= 256 else 0,
            'pkts_256_to_512': 1 if 256 < pkt_len <= 512 else 0,
            'pkts_512_to_1024': 1 if 512 < pkt_len <= 1024 else 0,
            'pkts_1024_to_1514': 1 if 1024 < pkt_len <= 1514 else 0,

            # ICMP
            'icmp_type': 0,
            'icmp_code': 0,

            # DNS
            'dns_query_id': 0,
            'dns_query_type': 0,
            'dns_ttl_answer': 0,

            # FTP
            'ftp_command_ret_code': 0,
        }

        # Ilk paketin TCP bayraklarini kaydet
        if TCP in pkt:
            flags_int = int(pkt[TCP].flags)
            flow['tcp_flags_all'] = flags_int
            flow['client_tcp_flags'] = flags_int
            flow['_fwd_seen_seqs'].add(pkt[TCP].seq)

        return flow

    # ---------------------------------------------------------------
    #  TIMEOUT MANAGEMENT
    # ---------------------------------------------------------------

    def check_timeouts(self, current_time):
        """Timeout'a ugramis flow'lari tespit edip kapat."""
        expired = [
            fid for fid, f in self.active_flows.items()
            if (current_time - f['last_time']) > self.timeout
        ]
        for fid in expired:
            self._close_flow(fid)

    # ---------------------------------------------------------------
    #  FLOW CLOSING & FEATURE EXPORT
    # ---------------------------------------------------------------

    def _close_flow(self, flow_id):
        """Flow'u kapat, feature'lari cikar ve callback'i cagir."""
        if flow_id not in self.active_flows:
            return None

        features = self.extract_features(flow_id)
        flow = self.active_flows.pop(flow_id)

        if self.on_flow_ready:
            self.on_flow_ready(features, flow['src_ip'], flow['dst_ip'])

        return features

    def flush_all(self):
        """Tum aktif flow'lari zorla kapat (shutdown icin)."""
        for fid in list(self.active_flows.keys()):
            self._close_flow(fid)

    # ---------------------------------------------------------------
    #  HELPERS
    # ---------------------------------------------------------------

    @staticmethod
    def _calc_iat_stats(timestamps):
        """Timestamp listesinden IAT istatistikleri hesapla.

        Returns:
            (iat_min, iat_max, iat_avg, iat_stddev) in milliseconds
        """
        if len(timestamps) < 2:
            return 0.0, 0.0, 0.0, 0.0

        iats = np.diff(timestamps) * 1000.0  # saniye -> milisaniye
        return (
            float(np.min(iats)),
            float(np.max(iats)),
            float(np.mean(iats)),
            float(np.std(iats)),
        )

    @staticmethod
    def _guess_l7_proto(dst_port, protocol):
        """Hedef porta ve protokole gore L7 proto tahmini.

        Basit best-effort haritalama. Gercek L7 DPI yapmiyor.
        nDPI proto numaralari kullanilir.
        """
        port_map = {
            80: 7,      # HTTP
            443: 91,    # TLS/HTTPS
            53: 5,      # DNS
            22: 92,     # SSH
            21: 1,      # FTP_CONTROL
            20: 1,      # FTP_DATA
            25: 3,      # SMTP
            110: 4,     # POP3
            143: 9,     # IMAP
            23: 23,     # Telnet
            3389: 88,   # RDP
            8080: 7,    # HTTP alt
            8443: 91,   # HTTPS alt
        }
        if dst_port in port_map:
            return port_map[dst_port]
        if protocol == 1:   # ICMP
            return 81
        if protocol == 17:  # UDP
            return 67       # generic UDP
        return 0  # Unknown

    # ---------------------------------------------------------------
    #  FEATURE EXTRACTION (53 NF-v3 features)
    # ---------------------------------------------------------------

    def extract_features(self, flow_id):
        """
        NF-v3 formatinda 53 feature uretir.

        Feature sirasi model/canavar-model/src/data/features.py'deki
        get_feature_names(include_engineered=True) ile birebir aynidir.

        Returns:
            dict: FlowGuardProvider.predict() icin hazir feature dict
                  (L4_SRC_PORT ve L4_DST_PORT dahil, port bucketing
                   provider tarafinda yapilir)
        """
        flow = self.active_flows[flow_id]

        # Temel hesaplamalar
        flow_duration_ms = (flow['last_time'] - flow['start_time']) * 1000.0
        flow_duration_sec = flow_duration_ms / 1000.0

        # Fwd/Bwd sureler (ms)
        if flow['fwd_timestamps'] and len(flow['fwd_timestamps']) > 1:
            duration_in = (flow['fwd_timestamps'][-1] - flow['fwd_timestamps'][0]) * 1000.0
        else:
            duration_in = 0.0

        if flow['bwd_timestamps'] and len(flow['bwd_timestamps']) > 1:
            duration_out = (flow['bwd_timestamps'][-1] - flow['bwd_timestamps'][0]) * 1000.0
        else:
            duration_out = 0.0

        # Paket boyutu istatistikleri
        all_pkt_lengths = flow['fwd_pkt_lengths'] + flow['bwd_pkt_lengths']
        longest_pkt = max(all_pkt_lengths) if all_pkt_lengths else 0
        shortest_pkt = min(all_pkt_lengths) if all_pkt_lengths else 0
        min_ip_pkt = shortest_pkt
        max_ip_pkt = longest_pkt

        # Throughput (bytes/second)
        if flow_duration_sec > 0:
            src_to_dst_bytes_per_sec = flow['in_bytes'] / flow_duration_sec
            dst_to_src_bytes_per_sec = flow['out_bytes'] / flow_duration_sec
            
            # Average throughput (bits/second)
            src_to_dst_throughput = (flow['in_bytes'] * 8) / flow_duration_sec
            dst_to_src_throughput = (flow['out_bytes'] * 8) / flow_duration_sec
        else:  
            # Fix: Avoid infinity explosion on single packet or ultra-fast flows
            src_to_dst_bytes_per_sec = 0.0
            dst_to_src_bytes_per_sec = 0.0
            src_to_dst_throughput = 0.0
            dst_to_src_throughput = 0.0

        # IAT istatistikleri
        s2d_iat_min, s2d_iat_max, s2d_iat_avg, s2d_iat_std = self._calc_iat_stats(flow['fwd_timestamps'])
        d2s_iat_min, d2s_iat_max, d2s_iat_avg, d2s_iat_std = self._calc_iat_stats(flow['bwd_timestamps'])

        features = {
            # --- Identity (port bucketing icin, model girdisi degil) ---
            'L4_SRC_PORT': flow['src_port'],
            'L4_DST_PORT': flow['dst_port'],

            # --- 47 ham NF-v3 feature ---

            # Protokol
            'PROTOCOL': flow['protocol'],
            'L7_PROTO': flow['l7_proto'],

            # Hacim
            'IN_BYTES': flow['in_bytes'],
            'IN_PKTS': flow['in_pkts'],
            'OUT_BYTES': flow['out_bytes'],
            'OUT_PKTS': flow['out_pkts'],

            # TCP bayraklari
            'TCP_FLAGS': flow['tcp_flags_all'],
            'CLIENT_TCP_FLAGS': flow['client_tcp_flags'],
            'SERVER_TCP_FLAGS': flow['server_tcp_flags'],

            # Sure
            'FLOW_DURATION_MILLISECONDS': flow_duration_ms,
            'DURATION_IN': duration_in,
            'DURATION_OUT': duration_out,

            # TTL
            'MIN_TTL': min(flow['ttl_values']) if flow['ttl_values'] else 0,
            'MAX_TTL': max(flow['ttl_values']) if flow['ttl_values'] else 0,

            # Paket boyutlari
            'LONGEST_FLOW_PKT': longest_pkt,
            'SHORTEST_FLOW_PKT': shortest_pkt,
            'MIN_IP_PKT_LEN': min_ip_pkt,
            'MAX_IP_PKT_LEN': max_ip_pkt,

            # Oran / throughput
            'SRC_TO_DST_SECOND_BYTES': src_to_dst_bytes_per_sec,
            'DST_TO_SRC_SECOND_BYTES': dst_to_src_bytes_per_sec,
            'SRC_TO_DST_AVG_THROUGHPUT': src_to_dst_throughput,
            'DST_TO_SRC_AVG_THROUGHPUT': dst_to_src_throughput,

            # Retransmission
            'RETRANSMITTED_IN_BYTES': flow['retransmitted_in_bytes'],
            'RETRANSMITTED_IN_PKTS': flow['retransmitted_in_pkts'],
            'RETRANSMITTED_OUT_BYTES': flow['retransmitted_out_bytes'],
            'RETRANSMITTED_OUT_PKTS': flow['retransmitted_out_pkts'],

            # Paket boyutu dagilimi
            'NUM_PKTS_UP_TO_128_BYTES': flow['pkts_up_to_128'],
            'NUM_PKTS_128_TO_256_BYTES': flow['pkts_128_to_256'],
            'NUM_PKTS_256_TO_512_BYTES': flow['pkts_256_to_512'],
            'NUM_PKTS_512_TO_1024_BYTES': flow['pkts_512_to_1024'],
            'NUM_PKTS_1024_TO_1514_BYTES': flow['pkts_1024_to_1514'],

            # TCP Window
            'TCP_WIN_MAX_IN': flow['tcp_win_max_in'],
            'TCP_WIN_MAX_OUT': flow['tcp_win_max_out'],

            # ICMP
            'ICMP_TYPE': flow['icmp_type'],
            'ICMP_IPV4_TYPE': flow['icmp_code'],

            # DNS
            'DNS_QUERY_ID': flow['dns_query_id'],
            'DNS_QUERY_TYPE': flow['dns_query_type'],
            'DNS_TTL_ANSWER': flow['dns_ttl_answer'],

            # FTP
            'FTP_COMMAND_RET_CODE': flow['ftp_command_ret_code'],

            # IAT (Inter-Arrival Time)
            'SRC_TO_DST_IAT_MIN': s2d_iat_min,
            'SRC_TO_DST_IAT_MAX': s2d_iat_max,
            'SRC_TO_DST_IAT_AVG': s2d_iat_avg,
            'SRC_TO_DST_IAT_STDDEV': s2d_iat_std,
            'DST_TO_SRC_IAT_MIN': d2s_iat_min,
            'DST_TO_SRC_IAT_MAX': d2s_iat_max,
            'DST_TO_SRC_IAT_AVG': d2s_iat_avg,
            'DST_TO_SRC_IAT_STDDEV': d2s_iat_std,

            # Not: Port bucket feature'lari (6 adet) FlowGuardProvider._preprocess()
            # tarafinda hesaplanir. Burada L4_SRC_PORT ve L4_DST_PORT olarak gonderiyoruz.
        }

        return features


if __name__ == "__main__":
    from scapy.all import sniff
    import json

    tracker = NFv3FlowTracker(
        timeout=120.0,
        on_flow_ready=lambda feats, src, dst: print(
            f"\n[FLOW] {src} -> {dst}\n"
            f"  PROTOCOL={feats['PROTOCOL']} L7={feats['L7_PROTO']}\n"
            f"  IN: {feats['IN_BYTES']}B / {feats['IN_PKTS']}pkts  "
            f"OUT: {feats['OUT_BYTES']}B / {feats['OUT_PKTS']}pkts\n"
            f"  Duration: {feats['FLOW_DURATION_MILLISECONDS']:.1f}ms  "
            f"TTL: {feats['MIN_TTL']}-{feats['MAX_TTL']}\n"
            f"  TCP Flags: {feats['TCP_FLAGS']}  "
            f"Client: {feats['CLIENT_TCP_FLAGS']}  "
            f"Server: {feats['SERVER_TCP_FLAGS']}\n"
            f"  Features count: {len(feats)}"
        )
    )

    print("Sniffing 200 packets (NF-v3 mode)...")
    sniff(prn=tracker.process_packet, count=200)
    tracker.flush_all()
