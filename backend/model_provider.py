"""
Model Provider Abstraction Layer
================================
Pluggable model provider pattern for Guardian AI Engine.
Allows swapping ML models without modifying the analysis engine.

Usage:
    provider = get_model_provider("placeholder")  # or "legacy", "custom"
    provider.load()
    result = provider.predict(raw_features_dict)
"""

import os
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class BaseModelProvider(ABC):
    """
    Abstract base class for all model providers.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Model dosyasını yükle.
        Bu metod engine başlatılırken bir kez çağrılır.
        """
        pass

    @abstractmethod
    def predict(self, features: dict) -> dict:
        """
        Ham feature dictionary alıp tahmin sonucu döndür.
        
        Args:
            features: Raw feature dictionary (CIC-IDS format veya mapped format)
            
        Returns:
            dict with keys:
                - "is_attack": bool
                - "confidence": float (0.0 - 1.0, attack olasılığı)
                - "attack_type": str ("Benign", "Malicious", vb.)
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """Model yüklenmiş ve tahmin yapmaya hazır mı?"""
        pass

    def get_info(self) -> dict:
        """Provider hakkında bilgi döndür (debug/logging için)."""
        return {
            "provider": self.__class__.__name__,
            "ready": self.is_ready(),
        }


class PlaceholderModelProvider(BaseModelProvider):


    def __init__(self):
        self._ready = False

    def load(self) -> None:
        logger.warning("[PlaceholderProvider] No real model loaded. All predictions will return Benign/0.0.")
        print("[Provider] ⚠️  PLACEHOLDER MODE — Gerçek model yüklenmedi.")
        print("[Provider] ⚠️  Tüm tahminler 'Benign' olarak dönecek.")
        self._ready = True  # "Hazır" ama gerçek tahmin yapmıyor

    def predict(self, features: dict) -> dict:
        return {
            "is_attack": False,
            "confidence": 0.0,
            "attack_type": "ModelNotLoaded",
        }

    def is_ready(self) -> bool:
        return self._ready



class LegacySklearnProvider(BaseModelProvider):
    """
    Mevcut sklearn RandomForest + StandardScaler pipeline.
    Geliştirme ve test amaçlı korunuyor.
    """

    MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "base_rf_2017.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "scaler_base.pkl")

    def __init__(self, adapt_file: str = None):
        self.model = None
        self.extractor = None
        self._ready = False
        self.adapt_file = adapt_file

    def load(self) -> None:
        import joblib
        import numpy as np
        import pandas as pd
        from feature_extractor import FeatureExtractor, MAPPING

        print(f"[LegacyProvider] Loading model from {self.MODEL_PATH}...")

        if not os.path.exists(self.MODEL_PATH) or not os.path.exists(self.SCALER_PATH):
            raise FileNotFoundError(
                f"Model files not found: {self.MODEL_PATH}, {self.SCALER_PATH}"
            )

        scaler = joblib.load(self.SCALER_PATH)
        self.model = joblib.load(self.MODEL_PATH)

        # --- Online Adaptation (from original analyze_engine.py) ---
        if self.adapt_file and os.path.exists(self.adapt_file):
            print(f"\n[LegacyProvider] ADAPTATION START: Retraining on 5% of {os.path.basename(self.adapt_file)}")
            try:
                calib_skip_lambda = lambda x: x > 0 and x % 20 != 0
                df_calib = pd.read_csv(self.adapt_file, skiprows=calib_skip_lambda)
                df_calib.columns = df_calib.columns.str.strip()
                df_calib = df_calib.rename(columns=MAPPING)

                if 'Destination Port' in df_calib.columns:
                    df_calib = df_calib[pd.to_numeric(df_calib['Destination Port'], errors='coerce').notnull()]

                if 'Label' not in df_calib.columns:
                    raise ValueError("Label column missing in calibration data")

                y_calib = np.where(df_calib['Label'].astype(str).str.lower() == 'benign', 0, 1)
                X_calib = df_calib.drop(columns=['Label'])

                valid_cols = [c for c in MAPPING.values() if c in X_calib.columns]
                X_calib = X_calib[valid_cols]

                for col in X_calib.columns:
                    X_calib[col] = pd.to_numeric(X_calib[col], errors='coerce').astype('float32')
                X_calib.replace([np.inf, -np.inf], 0, inplace=True)
                X_calib.fillna(0, inplace=True)

                X_calib_scaled = scaler.transform(X_calib)

                unique_classes = np.unique(y_calib)
                n_attack = np.sum(y_calib == 1)
                print(f"[LegacyProvider] Calibration Sample: {len(y_calib)} rows (Attacks: {n_attack})")

                if len(unique_classes) < 2:
                    print(f"[LegacyProvider] SKIP: Data has only ONE class ({unique_classes}).")
                else:
                    current_trees = self.model.n_estimators
                    new_trees = current_trees + 50
                    self.model.n_estimators = new_trees
                    self.model.fit(X_calib_scaled, y_calib)
                    print(f"[LegacyProvider] Model adapted: {current_trees} -> {new_trees} trees.")

            except Exception as e:
                print(f"[LegacyProvider] ADAPTATION ERROR: {e}")
        else:
            print("[LegacyProvider] No adaptation dataset. Using base model.")

        self.extractor = FeatureExtractor(scaler)
        self._ready = True
        print("[LegacyProvider] Model ready.")

    def predict(self, features: dict) -> dict:
        if not self._ready:
            raise RuntimeError("Model not loaded. Call load() first.")

        X_input = self.extractor.transform(features)
        probs = self.model.predict_proba(X_input)[0]
        p_attack = float(probs[1])

        from analyze_engine import THRESHOLD
        is_attack = p_attack > THRESHOLD

        return {
            "is_attack": is_attack,
            "confidence": p_attack,
            "attack_type": "Malicious" if is_attack else "Benign",
        }

    def is_ready(self) -> bool:
        return self._ready



class CustomModelProvider(BaseModelProvider):
    """
    🚧 ARKADAŞIN MODELİ İÇİN HAZIR İSKELET 🚧
    
    Bu sınıf, arkadaşınızın modeli geldiğinde doldurulacak.
    Aşağıdaki TODO'ları takip edin.
    
    Entegrasyon adımları:
        1. Model dosyasını saved_models/ altına koyun
        2. load() metodunda modeli yükleyin
        3. predict() metodunda tahmin yapın
        4. MODEL_PROVIDER=custom olarak ayarlayın
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self._ready = False
        # TODO: Model dosyasının yolunu belirle
        self.model_path = model_path or os.path.join(
            BASE_DIR, "saved_models", "custom_model.pkl"  # TODO: Dosya uzantısını güncelle
        )

    def load(self) -> None:
        """
        TODO: Arkadaşın modelini yükle.
        
        Örnekler:
            # PyTorch:
            # import torch
            # self.model = torch.load(self.model_path)
            # self.model.eval()
            
            # TensorFlow/Keras:
            # from tensorflow import keras
            # self.model = keras.models.load_model(self.model_path)
            
            # XGBoost:
            # import xgboost as xgb
            # self.model = xgb.Booster()
            # self.model.load_model(self.model_path)
            
            # ONNX Runtime:
            # import onnxruntime as ort
            # self.model = ort.InferenceSession(self.model_path)
            
            # Joblib (sklearn-compatible):
            # import joblib
            # self.model = joblib.load(self.model_path)
        """
        raise NotImplementedError(
            "CustomModelProvider.load() henüz implemente edilmedi. "
            "Arkadaşın modeli geldiğinde bu metodu doldurun."
        )

    def predict(self, features: dict) -> dict:
        """
        TODO: Arkadaşın modeliyle tahmin yap.
        
        Args:
            features: Raw feature dictionary. Anahtarlar CIC-IDS format
                      veya mapped format olabilir.
        
        Returns:
            dict: Aşağıdaki anahtarları MUTLAKA döndürmelisiniz:
                - "is_attack": bool     → Saldırı mı?
                - "confidence": float   → 0.0 ile 1.0 arası güven skoru
                - "attack_type": str    → "Benign", "DDoS", "PortScan" vb.
        
        Implement ederken dikkat edilecekler:
            1. features dict'indeki verileri modelin beklediği formata çevirin
            2. Gerekli preprocessing (normalization, scaling) yapın
            3. Model çıktısını yukarıdaki dict formatına dönüştürün
        """
        raise NotImplementedError(
            "CustomModelProvider.predict() henüz implemente edilmedi. "
            "Arkadaşın modeli geldiğinde bu metodu doldurun."
        )

    def is_ready(self) -> bool:
        return self._ready


PROVIDER_REGISTRY = {
    "placeholder": PlaceholderModelProvider,
    "legacy": LegacySklearnProvider,
    "custom": CustomModelProvider,
}


def get_model_provider(provider_name: str = None, **kwargs) -> BaseModelProvider:
    """
    Provider adına göre uygun model provider'ı döndür.
    
    Args:
        provider_name: "placeholder", "legacy", "custom"
                       None ise MODEL_PROVIDER env var'ından okunur.
        **kwargs: Provider'a iletilecek ek parametreler.
    
    Returns:
        BaseModelProvider instance (henüz load() çağrılmamış)
    """
    if provider_name is None:
        provider_name = os.getenv("MODEL_PROVIDER", "legacy")

    provider_name = provider_name.lower().strip()

    if provider_name not in PROVIDER_REGISTRY:
        available = ", ".join(PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown provider: '{provider_name}'. Available: {available}"
        )

    provider_class = PROVIDER_REGISTRY[provider_name]
    return provider_class(**kwargs)
