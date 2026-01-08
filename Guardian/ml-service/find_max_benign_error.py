import pandas as pd
import joblib
import os
import torch
import torch.nn as nn
from fastparquet import ParquetFile  # fastparquet kullanıyoruz
from tqdm import tqdm
import warnings

# 'analysis_engine' klasöründen GuardianEngine Sınıfını ve mimarisini import et
from analysis_engine.engine import GuardianEngine, Autoencoder

# Uyarıları bastır
warnings.filterwarnings('ignore')

# --- 1. Yapılandırma ---
ASSETS_DIR = "analysis_engine/assets"
PROCESSED_DATA_DIR = "analysis_engine/processed"

# === ÖNEMLİ ===
# Sadece "BENIGN" verileri içeren dosyayı kullanıyoruz
BENIGN_FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, "X_benign_scaled.parquet")

print("--- Guardian: Maksimum BENIGN Hatasını Bulma ---")

# --- 2. Motoru ve Modeli Yükle ---
try:
    print("Guardian Engine (Model, Scaler, Eşik) yükleniyor...")
    engine = GuardianEngine(assets_path=ASSETS_DIR)
    
    # Motoru kullanmayacağız, sadece içindeki model, eşik ve cihaz bilgisi lazım
    model = engine.model
    device = engine.device
    current_threshold = engine.threshold
    
    print(f"Motor yüklendi. Mevcut Eşik Değeri (Karşılaştırma için): {current_threshold}")
    
except FileNotFoundError as e:
    print(f"HATA: Gerekli asset dosyası bulunamadı: {e}")
    exit()

# --- 3. Batch (Parça Parça) Okuma ve Hata Tespiti ---
print(f"Sadece BENIGN paketleri içeren '{BENIGN_FEATURES_FILE}' işlenecek...")

pf_features = ParquetFile(BENIGN_FEATURES_FILE)

# Takip edilecek en yüksek hatayı sıfırdan başlat
max_benign_error = 0.0

total_rows = pf_features.count()
num_row_groups = len(pf_features.row_groups)
print(f"Toplam {total_rows} BENIGN satır, {num_row_groups} parça halinde işlenecek...")

# Kayıp (Hata) fonksiyonu
criterion = nn.MSELoss(reduction='none') 

# fastparquet iteratörü kullan
batch_iterator = pf_features.iter_row_groups()

for scaled_features_df in tqdm(batch_iterator, total=num_row_groups):
    
    # Veri zaten "BENIGN" ve "Scale edilmiş"
    input_tensor = torch.tensor(scaled_features_df.values, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        reconstructed = model(input_tensor)
        
        # Her bir satır için hatayı hesapla
        errors = criterion(reconstructed, input_tensor)
        errors_per_sample = torch.mean(errors, dim=1) # Her satırın ortalama hatası
        
        # Bu parçadaki (batch) maksimum hatayı bul
        batch_max_error = torch.max(errors_per_sample).item()
        
        # Genel maksimum hatayı güncelle
        if batch_max_error > max_benign_error:
            max_benign_error = batch_max_error

print("...Tüm BENIGN veri seti başarıyla işlendi.")

# --- 4. Sonuç ---
print("\n--- Sonuç Raporu ---")
print(f"Mevcut Eşik Değeri (Referans): {current_threshold:.10f}")
print(f"Maksimum BENIGN Hatası (TN):    {max_benign_error:.10f}")

# Öneri
new_suggested_threshold = max_benign_error * 1.01 # Maksimum hatadan %1 daha büyük
print(f"\nÖNERİLEN YENİ EŞİK DEĞERİ (Maksimum Hata * 1.01): {new_suggested_threshold:.10f}")
print("Değerlendirme tamamlandı.")