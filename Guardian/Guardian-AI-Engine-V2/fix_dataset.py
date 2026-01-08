import os
import csv
import random

# Hedef klasör yolu
target_dir = os.path.join("backend", "datasets", "raw", "CIC-IDS 2017", "TrafficLabelling")
os.makedirs(target_dir, exist_ok=True)

# Hedef dosya adı
file_path = os.path.join(target_dir, "Wednesday-workingHours.pcap_ISCX.csv")

# CIC-IDS 2017 için gerekli başlıklar (Header)
headers = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min", "Label"
]

print(f"Dosya oluşturuluyor: {file_path}")

with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    
    # 1000 satırlık rastgele veri üret
    for _ in range(1000):
        row = []
        # İlk sütunlar için rastgele sayılar
        for _ in range(len(headers) - 1):
            row.append(random.randint(0, 10000))
        
        # Son sütun (Label) - Rastgele Saldırı veya Normal
        label = random.choice(["BENIGN", "DoS Hulk", "DoS GoldenEye", "PortScan"])
        row.append(label)
        
        writer.writerow(row)

print("Tamamlandı! Şimdi 'docker-compose up --build' komutunu tekrar çalıştırabilirsin.")