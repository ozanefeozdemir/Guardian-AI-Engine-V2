import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import redis
import json
import os
import warnings
import gc

# Uyarıları bastır
warnings.filterwarnings('ignore')

# --- 1. Model Mimarileri ---
INPUT_DIM = 69
LATENT_DIM = 8
OUTPUT_DIM = 15 

class Autoencoder(nn.Module):
    # ... (Autoencoder class definition - kısaltıldı) ...
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim), nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DNNClassifier(nn.Module):
    # ... (DNNClassifier class definition - kısaltıldı) ...
    def __init__(self, input_dim, output_dim):
        super(DNNClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.network(x)

print("Guardian Engine Classes (AE & DNN) defined.")

# --- 2. Guardian Analiz Motoru Sınıfı (Final) ---
class GuardianEngine:
    def __init__(self, assets_path="."):
        print("Guardian Engine (Two-Stage - DNN) initializing...")
        
        # Dosya yolları
        model_ae_path = os.path.join(assets_path, "guardian_autoencoder.pth")
        model_dnn_path = os.path.join(assets_path, "guardian_classifier_dnn.pth")
        scaler_path = os.path.join(assets_path, "min_max_scaler.joblib")
        threshold_path = os.path.join(assets_path, "reconstruction_threshold.joblib")
        le_path = os.path.join(assets_path, "label_encoder.joblib")

        # Cihazı belirle
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modelleri yükle
        self.model_ae = Autoencoder(INPUT_DIM, LATENT_DIM).to(self.device)
        self.model_ae.load_state_dict(torch.load(model_ae_path, map_location=self.device))
        self.model_ae.eval()
        self.criterion_ae = nn.MSELoss()
        
        self.model_dnn = DNNClassifier(INPUT_DIM, OUTPUT_DIM).to(self.device)
        self.model_dnn.load_state_dict(torch.load(model_dnn_path, map_location=self.device))
        self.model_dnn.eval()

        # Yardımcıları yükle
        self.scaler = joblib.load(scaler_path)
        self.scaler_columns = self.scaler.feature_names_in_
        self.le = joblib.load(le_path)
        self.benign_label_numeric = self.le.transform(['BENIGN'])[0]
        self.benign_label_string = 'BENIGN'
        self.threshold = joblib.load(threshold_path)['reconstruction_threshold']
        
        print(f"Anomali eşik değeri yüklendi (Aşama 1): {self.threshold:.10f}")

        # Redis bağlantısı
        try:
            self.redis_client = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"),
                                            port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
        except redis.exceptions.ConnectionError:
            self.redis_client = None

    def preprocess(self, packet_features_df):
        # ... (preprocess method - no changes) ...
        try:
            ordered_df = packet_features_df[self.scaler_columns]
            scaled_data = self.scaler.transform(ordered_df)
            tensor_data = torch.tensor(scaled_data, dtype=torch.float32).to(self.device)
            return tensor_data
        except KeyError as e:
            print(f"ERROR: Preprocess sırasında sütun hatası: {e}. Beklenen sütunlar: {list(self.scaler_columns)}")
            raise
        except Exception as e:
            print(f"ERROR: Preprocess sırasında genel hata: {e}")
            raise

    def analyze_packet(self, packet_features_df):
        # ... (analyze_packet method - no changes) ...
        try:
            input_tensor = self.preprocess(packet_features_df)
            with torch.no_grad():
                reconstructed = self.model_ae(input_tensor)
                loss = self.criterion_ae(reconstructed, input_tensor)
                reconstruction_error = loss.item()
                outputs_dnn = self.model_dnn(input_tensor)
                _, predicted_class_id_tensor = torch.max(outputs_dnn, 1)
                predicted_class_id = predicted_class_id_tensor.item()
                predicted_class_name = self.le.inverse_transform([predicted_class_id])[0]
            anomaly_type = predicted_class_name
            is_anomaly = False
            if predicted_class_id != self.benign_label_numeric:
                is_anomaly = True
            is_zero_day_anomaly = (reconstruction_error > self.threshold)
            if is_zero_day_anomaly and not is_anomaly:
                is_anomaly = True
                anomaly_type = "UNKNOWN (Zero-Day)"
            if is_anomaly:
                self.publish_to_redis(
                    error=reconstruction_error, threshold=self.threshold,
                    classification=predicted_class_name,
                    is_zero_day=(anomaly_type == "UNKNOWN (Zero-Day)"),
                    packet_features_df=packet_features_df
                )
            return is_anomaly, anomaly_type, reconstruction_error
        except Exception as e:
            print(f"ERROR: Exception during analysis: {e}")
            return False, "ERROR", 0.0

    def publish_to_redis(self, error, threshold, classification, is_zero_day, packet_features_df):
        if not self.redis_client: return
        try:
            original_data_json = packet_features_df.to_json(orient='records')
            alert_message = {
                "type": "ANOMALY_DETECTED", "timestamp": pd.Timestamp.now().isoformat(),
                "classification_result": classification, "is_known_attack": (classification != self.benign_label_string),
                "is_zero_day_anomaly": is_zero_day, "reconstruction_error": error,
                "error_threshold": threshold, "packet_data": json.loads(original_data_json)[0]
            }
            self.redis_client.publish("guardian_alerts", json.dumps(alert_message))
        except Exception:
            pass # Hata durumunda Redis yayınını sessizce atla

# --- 3. Test Alanı (GÜNCEL ve GÜVENİLİR TESTLER) ---
if __name__ == "__main__":
    ASSETS_DIR = "analysis_engine/assets"
    PROCESSED_DATA_DIR = "analysis_engine/processed"

    try:
        engine = GuardianEngine(assets_path=ASSETS_DIR)
        print("\n--- Analysis Engine Test Starting (Final - AI Consistency Check) ---")

        # --- Helper Function to Get Raw Packet Data ---
        def get_raw_packet(label_name, index=0):
            """Gets a raw (inverse transformed) packet for a specific label."""
            try:
                all_features_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "X_all_scaled.parquet"))
                all_labels_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "y_all_encoded.parquet"))
            except FileNotFoundError:
                raise # Dosya yoksa dur

            # <<< DÜZELTME 2: Etiket adındaki boşluk sorununu çöz >>>
            # Sadece LE'deki sınıfların adlarını kullan
            if label_name not in engine.le.classes_:
                return None
            
            label_numeric = engine.le.transform([label_name])[0]
            indices = all_labels_df[all_labels_df['Label'] == label_numeric].index
            
            if len(indices) == 0:
                return None

            selected_index = indices[index % len(indices)]
            scaled_row = all_features_df.iloc[selected_index:selected_index+1]
            
            raw_df = pd.DataFrame(
                engine.scaler.inverse_transform(scaled_row),
                columns=engine.scaler_columns
            )
            return raw_df
        
        # --- Test Verileri Listesi (Sadece LE'de Olanları Kullan) ---
        TEST_CASES = [
            # Normal
            ('BENIGN', 0, 'Normal'),
            ('BENIGN', 50, 'Normal'),
            
            # Bilinen Başarılı Saldırılar (Daha Sık Yakalananlar)
            ('DDoS', 10, 'Saldırı'),
            ('DoS Hulk', 5, 'Saldırı'),
            ('FTP-Patator', 10, 'Saldırı'),
            ('SSH-Patator', 10, 'Saldırı'),
            
            # Sinsi ve Nadir
            ('PortScan', 100, 'Saldırı'),
            ('Heartbleed', 0, 'Saldırı'),
            
            # Simülasyon
            ('BENIGN', 0, 'Zero-Day Simülasyonu'),
        ]

        # --- Testleri Çalıştır ---
        for i, (label, index, expected_type) in enumerate(TEST_CASES):
            raw_packet = get_raw_packet(label, index)
            if raw_packet is None:
                print(f"Test {i+1} atlandı: '{label}' etiketi veri setinde bulunamadı.")
                continue

            print(f"\nTest {i+1} ({label}) - Paket Index {index} gönderiliyor...")
            
            if expected_type == 'Zero-Day Simülasyonu':
                # Gürültü ekleyip AE eşiğini aşmasını sağla
                noisy_scaled_array = raw_packet.values + np.random.normal(0, 5, size=raw_packet.shape)
                raw_test_df = pd.DataFrame(engine.scaler.inverse_transform(noisy_scaled_array), columns=engine.scaler_columns)
                expected_result = "Anomaly: True, Tip: UNKNOWN"
            else:
                raw_test_df = raw_packet
                expected_result = "Anomaly: True" if expected_type == 'Saldırı' else "Anomaly: False"
            
            is_anomaly, a_type, error = engine.analyze_packet(raw_test_df)
            print(f"Result -> Anomaly: {is_anomaly}, Tip: {a_type}, AE Error: {error:.10f}")
            print(f"Beklenti: {expected_result}")
            print("-" * 40)


    except FileNotFoundError as e:
        print(f"\nERROR: Test için gerekli dosyalar bulunamadı: {e}")
    except Exception as e:
        print(f"Genel bir hata oluştu: {e}")
    finally:
        gc.collect()