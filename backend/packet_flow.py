from scapy.all import *
import numpy as np
import pandas as pd

class CICFlowTracker:
    def __init__(self, timeout=120.0):
        self.active_flows = {}
        self.timeout = timeout

    def get_flow_id(self, pkt):
        if IP not in pkt or (TCP not in pkt and UDP not in pkt):
            return None, None
        
        src_ip, dst_ip, proto = pkt[IP].src, pkt[IP].dst, pkt[IP].proto
        src_port = pkt[TCP].sport if TCP in pkt else pkt[UDP].sport
        dst_port = pkt[TCP].dport if TCP in pkt else pkt[UDP].dport

        fwd_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}"
        bwd_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{proto}"

        if fwd_key in self.active_flows: return fwd_key, "fwd"
        elif bwd_key in self.active_flows: return bwd_key, "bwd"
        else: return fwd_key, "new"

    def process_packet(self, pkt):
        flow_id, direction = self.get_flow_id(pkt)
        if not flow_id: return
        
        current_time = float(pkt.time)
        pkt_len = len(pkt)
        
        # Başlık uzunluğu hesaplama (IP Header + TCP/UDP Header)
        header_len = (pkt[IP].ihl * 4)
        if TCP in pkt: header_len += (pkt[TCP].dataofs * 4)
        elif UDP in pkt: header_len += 8

        # --- YENİ AKIŞ OLUŞTURMA ---
        if direction == "new":
            self.active_flows[flow_id] = {
                'start_time': current_time,
                'last_time': current_time,
                
                # Paket ve Boyut Listeleri (IAT ve Std hesapları için)
                'fwd_pkt_lengths': [pkt_len], 'bwd_pkt_lengths': [],
                'all_pkt_lengths': [pkt_len],
                'fwd_timestamps': [current_time], 'bwd_timestamps': [],
                'all_timestamps': [current_time],
                
                # Başlık ve Window Boyutları
                'fwd_header_len': header_len, 'bwd_header_len': 0,
                'init_fwd_win_bytes': pkt[TCP].window if TCP in pkt else 0,
                'init_bwd_win_bytes': -1, # Henüz Bwd paketi gelmedi
                
                # Bayrak Sayaçları (Sadece TCP için)
                'fin_cnt': 0, 'syn_cnt': 0, 'rst_cnt': 0, 'psh_cnt': 0,
                'ack_cnt': 0, 'urg_cnt': 0, 'ece_cnt': 0, 'cwe_cnt': 0,
            }
            direction = "fwd"
        
        # --- MEVCUT AKIŞI GÜNCELLEME ---
        else:
            flow = self.active_flows[flow_id]
            flow['last_time'] = current_time
            flow['all_timestamps'].append(current_time)
            flow['all_pkt_lengths'].append(pkt_len)
            
            if direction == "fwd":
                flow['fwd_timestamps'].append(current_time)
                flow['fwd_pkt_lengths'].append(pkt_len)
                flow['fwd_header_len'] += header_len
            elif direction == "bwd":
                flow['bwd_timestamps'].append(current_time)
                flow['bwd_pkt_lengths'].append(pkt_len)
                flow['bwd_header_len'] += header_len
                # İlk Bwd paketiyse Init Win Bytes değerini al
                if flow['init_bwd_win_bytes'] == -1 and TCP in pkt:
                    flow['init_bwd_win_bytes'] = pkt[TCP].window

        # --- TCP BAYRAKLARINI SAYMA ---
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
            
            # Kapanış kontrolü
            if 'F' in flags or 'R' in flags:
                return self.export_and_close_flow(flow_id)

    # --- YARDIMCI HESAPLAMA FONKSİYONLARI ---
    def calc_iat(self, timestamps):
        """Zaman damgalarından paketler arası varış süresi (IAT) dizisi üretir"""
        if len(timestamps) < 2: return [0]
        return np.diff(timestamps).tolist()

    def safe_stat(self, data_list, stat_type):
        """Boş listelerde hata almamak için güvenli istatistik hesaplar"""
        if not data_list: return 0.0
        if stat_type == 'mean': return float(np.mean(data_list))
        if stat_type == 'std': return float(np.std(data_list))
        if stat_type == 'max': return float(np.max(data_list))
        if stat_type == 'min': return float(np.min(data_list))
        if stat_type == 'tot': return float(np.sum(data_list))
        return 0.0

    def extract_features(self, flow_id):
        flow = self.active_flows[flow_id]
        
        # Süre ve IAT Hesaplamaları
        flow_duration = max((flow['last_time'] - flow['start_time']) * 1e6, 1) # Mikrosaniye cinsinden (CIC formatı)
        fwd_iat = self.calc_iat(flow['fwd_timestamps'])
        bwd_iat = self.calc_iat(flow['bwd_timestamps'])
        flow_iat = self.calc_iat(flow['all_timestamps'])
        
        tot_fwd_pkts = len(flow['fwd_pkt_lengths'])
        tot_bwd_pkts = len(flow['bwd_pkt_lengths'])
        tot_pkts = tot_fwd_pkts + tot_bwd_pkts
        tot_bytes = sum(flow['all_pkt_lengths'])

        # Sözlüğü CIC-IDS-2017 formatında inşa ediyoruz
        features = {
            'Flow Duration': flow_duration,
            'Tot Fwd Pkts': tot_fwd_pkts,
            'Tot Bwd Pkts': tot_bwd_pkts,
            'TotLen Fwd Pkts': sum(flow['fwd_pkt_lengths']),
            'TotLen Bwd Pkts': sum(flow['bwd_pkt_lengths']),
            
            # Boyut İstatistikleri (Fwd, Bwd ve All)
            'Fwd Pkt Len Max': self.safe_stat(flow['fwd_pkt_lengths'], 'max'),
            'Fwd Pkt Len Min': self.safe_stat(flow['fwd_pkt_lengths'], 'min'),
            'Fwd Pkt Len Mean': self.safe_stat(flow['fwd_pkt_lengths'], 'mean'),
            'Fwd Pkt Len Std': self.safe_stat(flow['fwd_pkt_lengths'], 'std'),
            'Bwd Pkt Len Max': self.safe_stat(flow['bwd_pkt_lengths'], 'max'),
            'Bwd Pkt Len Min': self.safe_stat(flow['bwd_pkt_lengths'], 'min'),
            'Bwd Pkt Len Mean': self.safe_stat(flow['bwd_pkt_lengths'], 'mean'),
            'Bwd Pkt Len Std': self.safe_stat(flow['bwd_pkt_lengths'], 'std'),
            'Pkt Len Min': self.safe_stat(flow['all_pkt_lengths'], 'min'),
            'Pkt Len Max': self.safe_stat(flow['all_pkt_lengths'], 'max'),
            'Pkt Len Mean': self.safe_stat(flow['all_pkt_lengths'], 'mean'),
            'Pkt Len Std': self.safe_stat(flow['all_pkt_lengths'], 'std'),
            'Pkt Len Var': np.var(flow['all_pkt_lengths']) if flow['all_pkt_lengths'] else 0,
            
            # Hız ve Oranlar
            'Flow Byts/s': (tot_bytes / flow_duration) * 1e6,
            'Flow Pkts/s': (tot_pkts / flow_duration) * 1e6,
            'Fwd Pkts/s': (tot_fwd_pkts / flow_duration) * 1e6,
            'Bwd Pkts/s': (tot_bwd_pkts / flow_duration) * 1e6,
            'Down/Up Ratio': tot_bwd_pkts / tot_fwd_pkts if tot_fwd_pkts > 0 else 0,
            
            # IAT İstatistikleri
            'Flow IAT Mean': self.safe_stat(flow_iat, 'mean'),
            'Flow IAT Std': self.safe_stat(flow_iat, 'std'),
            'Flow IAT Max': self.safe_stat(flow_iat, 'max'),
            'Flow IAT Min': self.safe_stat(flow_iat, 'min'),
            'Fwd IAT Tot': self.safe_stat(fwd_iat, 'tot'),
            'Fwd IAT Mean': self.safe_stat(fwd_iat, 'mean'),
            'Fwd IAT Max': self.safe_stat(fwd_iat, 'max'),
            'Bwd IAT Tot': self.safe_stat(bwd_iat, 'tot'),
            'Bwd IAT Mean': self.safe_stat(bwd_iat, 'mean'),
            'Bwd IAT Max': self.safe_stat(bwd_iat, 'max'),

            # TCP Bayrakları
            'FIN Flag Cnt': flow['fin_cnt'],
            'SYN Flag Cnt': flow['syn_cnt'],
            'RST Flag Cnt': flow['rst_cnt'],
            'PSH Flag Cnt': flow['psh_cnt'],
            'ACK Flag Cnt': flow['ack_cnt'],
            'URG Flag Cnt': flow['urg_cnt'],
            'CWE Flag Count': flow['cwe_cnt'],
            'ECE Flag Cnt': flow['ece_cnt'],
            
            # Başlık ve Window Boyutları
            'Fwd Header Len': flow['fwd_header_len'],
            'Bwd Header Len': flow['bwd_header_len'],
            'Init Fwd Win Byts': flow['init_fwd_win_bytes'],
            'Init Bwd Win Byts': flow['init_bwd_win_bytes'] if flow['init_bwd_win_bytes'] != -1 else 0
        }
        return features

    def export_and_close_flow(self, flow_id):
        features = self.extract_features(flow_id)
        
        # Veriyi Pandas DataFrame'e çeviriyoruz (Modelin tam istediği format)
        df = pd.DataFrame([features])
        import json
        print(f"\n{json.dumps(features, indent=4)}")
        # print(df.to_string()) # İstersen ekrana yazdırabilirsin
        
        # ==========================================
        # MODEL PREDICTION BURAYA GELECEK
        # ornek: sonuc = model.predict(df)
        # if sonuc == 1: print("SALDIRI TESPİT EDİLDİ!")
        # ==========================================
        
        del self.active_flows[flow_id]
        return df

# --- KULLANIM ---
tracker = CICFlowTracker()

def packet_callback(pkt):
    tracker.process_packet(pkt)

# Canlı trafiği dinle (Örnek olarak 200 paket)
sniff(prn=packet_callback, count=200)