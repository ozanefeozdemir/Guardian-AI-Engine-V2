import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Docker içi yol
save_dir = "/app/backend/saved_models"
os.makedirs(save_dir, exist_ok=True)

print(f"Modeller {save_dir} konumuna oluşturuluyor...")

# 1. SCALER (DÜZELTİLDİ: scaler_2017 -> scaler_base)
scaler = StandardScaler()
dummy_data = np.random.rand(100, 78)
scaler.fit(dummy_data)

# Engine'in beklediği isim: scaler_base.pkl
with open(os.path.join(save_dir, "scaler_base.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler OK (scaler_base.pkl)")

# 2. MODEL (Bu zaten doğruydu)
model = RandomForestClassifier()
dummy_labels = np.random.randint(0, 2, 100)
model.fit(dummy_data, dummy_labels)

with open(os.path.join(save_dir, "base_rf_2017.pkl"), "wb") as f:
    pickle.dump(model, f)
print("✅ Model OK (base_rf_2017.pkl)")