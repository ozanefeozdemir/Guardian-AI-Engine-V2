import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- Ayarlar ---
save_dir = os.path.join("backend", "saved_models")
os.makedirs(save_dir, exist_ok=True)

print("Gerçek uyumlu modeller oluşturuluyor...")

# 1. Gerçek Bir Scaler Oluştur (Boş ama uyumlu)
scaler = StandardScaler()
# Scaler'ı "fit" etmemiz lazım ki hata vermesin (Rastgele veriyle kandırıyoruz)
dummy_data = np.random.rand(100, 48) # 48 özellikli rastgele veri
scaler.fit(dummy_data)

scaler_path = os.path.join(save_dir, "scaler_base.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
print(f"✅ Oluşturuldu: {scaler_path}")

# 2. Gerçek Bir Model Oluştur (Random Forest)
model = RandomForestClassifier()
# Modeli de "fit" edelim
dummy_labels = np.random.randint(0, 2, 100) # 0 veya 1
model.fit(dummy_data, dummy_labels)

model_path = os.path.join(save_dir, "base_rf_2017.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"✅ Oluşturuldu: {model_path}")

print("\n🎉 Tamamlandı! Şimdi Docker'ı yeniden build etmen lazım.")