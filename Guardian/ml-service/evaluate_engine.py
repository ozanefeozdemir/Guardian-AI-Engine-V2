import pandas as pd
import joblib
import os
import torch
import torch.nn as nn
import pyarrow.parquet as pq # pyarrow ile batch okuma
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import warnings
import gc # Garbage collector

# 'analysis_engine' klasöründen mimarileri import et
from analysis_engine.engine import Autoencoder, DNNClassifier

# Uyarıları bastır
warnings.filterwarnings('ignore')

# --- 1. Yapılandırma ---
ASSETS_DIR = "analysis_engine/assets"
PROCESSED_DATA_DIR = "analysis_engine/processed"

FEATURES_FILE = os.path.join(PROCESSED_DATA_DIR, "X_all_scaled.parquet")
LABELS_FILE = os.path.join(PROCESSED_DATA_DIR, "y_all_encoded.parquet")
LE_FILE = os.path.join(ASSETS_DIR, "label_encoder.joblib")

# Modelleri yükleyeceğimiz yer
MODEL_AE_PATH = os.path.join(ASSETS_DIR, "guardian_autoencoder.pth")
MODEL_DNN_PATH = os.path.join(ASSETS_DIR, "guardian_classifier_dnn.pth") # <-- DNN modelini kullanıyoruz
THRESHOLD_PATH = os.path.join(ASSETS_DIR, "reconstruction_threshold.joblib")

# Mimarileri yüklemek için parametreler
INPUT_DIM = 69
LATENT_DIM = 8
OUTPUT_DIM = 15 # 'Aşama 2 (DNN)' notebook'undaki çıktı boyutu

# pyarrow batch boyutu
PYARROW_BATCH_SIZE = 65536

print("--- Guardian NİHAİ (AE + DNN) Model Tam Değerlendirme ---")

# --- 2. Tüm Modelleri ve Ayarları Yükle ---
try:
    print("Cihaz belirleniyor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    # Aşama 1 (Nöbetçi) modelini yükle
    print("Aşama 1 (Autoencoder) modeli yükleniyor...")
    model_ae = Autoencoder(INPUT_DIM, LATENT_DIM).to(device)
    model_ae.load_state_dict(torch.load(MODEL_AE_PATH, map_location=device))
    model_ae.eval()
    criterion_ae = nn.MSELoss(reduction='none')

    # Aşama 2 (DNN Analist) modelini yükle
    print("Aşama 2 (DNN Sınıflandırıcı) modeli yükleniyor...")
    model_dnn = DNNClassifier(INPUT_DIM, OUTPUT_DIM).to(device)
    model_dnn.load_state_dict(torch.load(MODEL_DNN_PATH, map_location=device))
    model_dnn.eval()

    # Ayarları yükle
    print("Ayarlar (Eşik, Etiketler) yükleniyor...")
    threshold_data = joblib.load(THRESHOLD_PATH)
    threshold = threshold_data['reconstruction_threshold'] # Güvenlik ağı eşiği (0.118...)
    le = joblib.load(LE_FILE)
    benign_label_numeric = le.transform(['BENIGN'])[0]
    class_names = le.classes_ # Raporlama için

    print(f"Aşama 1 Eşik Değeri: {threshold:.10f}")
    print(f"'BENIGN' etiketinin sayısal karşılığı: {benign_label_numeric}")

    print(f"Etiket dosyası ({LABELS_FILE}) hafızaya yükleniyor...")
    # Etiketler küçük olduğu için hafızaya alabiliriz
    all_labels_series = pd.read_parquet(LABELS_FILE)['Label']
    print("Etiketler yüklendi.")

except FileNotFoundError as e:
    print(f"HATA: Gerekli asset dosyası bulunamadı: {e}")
    exit()
except Exception as e:
     print(f"Model yüklenirken hata: {e}")
     exit()

# --- 3. Batch (Parça Parça) Okuma ve NİHAİ Değerlendirme (pyarrow) ---
print(f"Özellik dosyası ({FEATURES_FILE}) pyarrow ile işlenecek...")

y_true_list = [] # Gerçek etiketler (0 veya 1)
y_pred_list = [] # Nihai Model tahmini (0 veya 1)

try:
    # pyarrow ile dosyaları aç
    features_table = pq.read_table(FEATURES_FILE)
    labels_table = pq.read_table(LABELS_FILE) # Tekrar okuyoruz (indeks eşleşmesi için)
    total_rows = features_table.num_rows
    print(f"Toplam {total_rows} satır işlenecek...")

    feature_batches = features_table.to_batches(max_chunksize=PYARROW_BATCH_SIZE)
    label_batches = labels_table.to_batches(max_chunksize=PYARROW_BATCH_SIZE) # Gerçek etiketleri almak için

    current_batch_start_index = 0 # Batch'lerin indeksini takip etmek için

    # Batch'ler üzerinde döngü
    for batch_features, batch_labels_arrow in tqdm(zip(feature_batches, label_batches), total=-(total_rows // -PYARROW_BATCH_SIZE), desc="Değerlendirme"):

        # a. Gerçek Etiketleri (y_true) hazırla (0 = Normal, 1 = Anomali)
        batch_labels_pd = batch_labels_arrow.to_pandas()
        actual_anomalies = batch_labels_pd['Label'].apply(lambda x: 1 if x != benign_label_numeric else 0)
        y_true_list.extend(actual_anomalies)

        # b. Tahminleri (y_pred) al (İKİ AŞAMALI - DNN)

        # pyarrow Batch'ini Pandas'a, NumPy'e, sonra Tensor'e çevir
        batch_features_pd = batch_features.to_pandas()
        input_np = batch_features_pd.values
        input_tensor = torch.tensor(input_np, dtype=torch.float32).to(device)

        with torch.no_grad():
            # Aşama 1 (Nöbetçi) Tahmini: Anormallik Skoru
            reconstructed = model_ae(input_tensor)
            errors = criterion_ae(reconstructed, input_tensor)
            # errors_per_sample tensörünü CPU'da tutalım
            errors_per_sample = torch.mean(errors, dim=1).cpu()

            # Aşama 2 (DNN Analist) Tahmini
            outputs_dnn = model_dnn(input_tensor)
            _, predicted_class_ids_tensor = torch.max(outputs_dnn, 1)
            # predicted_class_ids'i NumPy array olarak CPU'da tutalım
            predicted_class_ids = predicted_class_ids_tensor.cpu().numpy()

        # --- c. Nihai Karar Mekanizması ---
        final_predictions = []
        for i in range(len(batch_features_pd)): # Batch boyutu kadar döngü

            is_known_attack = (predicted_class_ids[i] != benign_label_numeric)
            # errors_per_sample artık bir tensor, .item() kullanabiliriz veya index ile erişebiliriz
            is_zero_day = (errors_per_sample[i].item() > threshold)
            is_benign_according_to_dnn = (predicted_class_ids[i] == benign_label_numeric)

            # Durum A veya B ise Anomali (1), değilse Normal (0)
            if is_known_attack or (is_benign_according_to_dnn and is_zero_day):
                final_predictions.append(1) # Anomali
            else:
                final_predictions.append(0) # Normal

        y_pred_list.extend(final_predictions)

        # Belleği temizle
        del batch_features, batch_labels_arrow, batch_labels_pd, batch_features_pd
        del input_np, input_tensor, reconstructed, errors, errors_per_sample
        del outputs_dnn, predicted_class_ids_tensor, predicted_class_ids, final_predictions
        gc.collect()

    # Döngü bitti, tabloları serbest bırak
    del features_table, labels_table, feature_batches, label_batches
    gc.collect()

except Exception as e:
    print(f"\nDeğerlendirme sırasında HATA: {e}")
    exit()


print("...Tüm veri seti başarıyla işlendi.")

# --- 5. Raporlama ---
print("\n--- Guardian NİHAİ (AE + DNN) Model Performans Raporu ---")

if len(y_true_list) == len(y_pred_list) and len(y_true_list) > 0:
    all_possible_labels = [0, 1] # Sadece Normal(0)/Anomali(1) ikili sınıflandırma raporu

    cm = confusion_matrix(y_true_list, y_pred_list, labels=all_possible_labels)
    print("\nKonfüzyon Matrisi (Confusion Matrix):")
    print("                     Tahmin: Normal (0)   |   Tahmin: Anomali (1)")
    print("--------------------------------------------------------------------")
    print(f"Gerçek: Normal (0)   |   {cm[0][0]:<18} (TN) |   {cm[0][1]:<18} (FP - Yanlış Alarm)")
    print(f"Gerçek: Anomali (1)  |   {cm[1][0]:<18} (FN - Kaçan Saldırı) |   {cm[1][1]:<18} (TP - Yakalanan Saldırı)")

    print("\nSınıflandırma Raporu (Precision, Recall, F_1-Score):")
    # 0 = Normal (BENIGN), 1 = Anomali (Saldırı)
    print(classification_report(y_true_list, y_pred_list, labels=all_possible_labels, target_names=['Normal (0)', 'Anomali (1)'], zero_division=0))

    accuracy = accuracy_score(y_true_list, y_pred_list)
    print(f"\nGenel Doğruluk (Accuracy): {accuracy * 100:.2f}%")
else:
    print("HATA: Gerçek ve Tahmin etiket listeleri eşleşmiyor veya boş.")

print("Değerlendirme tamamlandı.")