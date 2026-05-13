"""
Model Provider Abstraction Layer
================================
Pluggable model provider pattern for Guardian AI Engine.
Allows swapping ML models without modifying the analysis engine.

Usage:
    provider = get_model_provider("placeholder")  # or "legacy", "custom", "guardian", "flowguard"
    provider.load()
    result = provider.predict(raw_features_dict)
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from collections import deque

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
    GuardianHybrid PyTorch Model Provider
    
    model/sunum-model/ klasöründeki Conv1d + LSTM tabanlı hybrid modeli yükler.
    Sliding window buffer ile 10-adımlık sekans oluşturup modele verir.
    5 sınıf: Benign, DDoS, PortScan, WebAttack, Botnet
    
    Kullanım:
        1. Model eğitimi: python model/sunum-model/train.py --phase all
           → backend/saved_models/guardian_complete.pth + backend/saved_models/guardian_scaler.pkl
        2. Engine: python backend/analyze_engine.py --provider guardian
    """

    CLASS_NAMES = {0: "Benign", 1: "DDoS", 2: "PortScan", 3: "WebAttack", 4: "Botnet"}
    SEQ_LEN = 10

    # Varsayılan dosya yolları (backend/saved_models/ altında)
    PROJECT_ROOT = os.path.dirname(BASE_DIR)  # backend/ → proje kökü
    DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "guardian_complete.pth")
    DEFAULT_SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "guardian_scaler.pkl")

    def __init__(self, model_path: str = None, scaler_path: str = None):
        self.model = None
        self.scaler = None
        self.device = None
        self.feature_columns = None  # Scaler'dan alınacak feature sıralaması
        self._ready = False
        self._window = deque(maxlen=self.SEQ_LEN)  # Sliding window buffer

        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.scaler_path = scaler_path or self.DEFAULT_SCALER_PATH

    def load(self) -> None:
        """
        GuardianHybrid modelini ve MinMaxScaler'ı yükler.
        """
        import torch
        import joblib

        # — Scaler —
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Scaler dosyası bulunamadı: {self.scaler_path}\n"
                "Önce modeli eğitin: python model/sunum-model/train.py --phase all"
            )
        self.scaler = joblib.load(self.scaler_path)
        print(f"[GuardianProvider] Scaler yüklendi: {self.scaler_path}")

        # Feature sıralamasını scaler'dan al
        if hasattr(self.scaler, 'feature_names_in_'):
            self.feature_columns = list(self.scaler.feature_names_in_)
        else:
            self.feature_columns = None
            print("[GuardianProvider] ⚠️  Scaler'da feature_names_in_ yok, " 
                  "sıralama garanti edilemez.")

        # — Model —
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {self.model_path}\n"
                "Önce modeli eğitin: python model/sunum-model/train.py --phase all"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # GuardianHybrid sınıfını import et
        # model/sunum-model/ klasörünü sys.path'e ekle
        model_dir = os.path.join(self.PROJECT_ROOT, "model", "sunum-model")
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from model import GuardianHybrid

        # input_dim = feature sayısı
        n_features = len(self.feature_columns) if self.feature_columns else self.scaler.n_features_in_
        self.model = GuardianHybrid(
            input_dim=n_features,
            seq_len=self.SEQ_LEN,
            latent_dim=32,
            n_classes=len(self.CLASS_NAMES)
        ).to(self.device)

        # Ağırlıkları yükle
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self._ready = True
        print(f"[GuardianProvider] Model yüklendi: {self.model_path}")
        print(f"[GuardianProvider] Device: {self.device} | Features: {n_features} | Classes: {len(self.CLASS_NAMES)}")

    def _preprocess(self, features: dict):
        """
        Ham feature dict → normalize edilmiş 1-D numpy array.
        """
        import numpy as np
        import pandas as pd
        from feature_extractor import MAPPING

        df = pd.DataFrame([features])
        df = df.rename(columns=MAPPING)

        # Feature sıralamasını uygula
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[self.feature_columns]
        
        # Sayısala çevir, NaN/Inf temizle
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)

        # MinMaxScaler ile normalize et
        scaled = self.scaler.transform(df)
        return scaled[0]  # (n_features,)

    def predict(self, features: dict) -> dict:
        """
        Feature dict alıp sliding window'a ekler.
        Pencere dolduğunda (10 adım) model ile tahmin yapar.
        Pencere henüz dolmamışsa → Benign varsayımıyla erken döner.
        """
        import torch
        import numpy as np

        if not self._ready:
            raise RuntimeError("Model yüklenmedi. Önce load() çağırın.")

        # 1. Preprocess & buffer'a ekle
        vec = self._preprocess(features)
        self._window.append(vec)

        # 2. Pencere dolmadıysa erken dön
        if len(self._window) < self.SEQ_LEN:
            return {
                "is_attack": False,
                "confidence": 0.0,
                "attack_type": f"Warmup ({len(self._window)}/{self.SEQ_LEN})",
            }

        # 3. Sliding window → tensor (1, SEQ_LEN, n_features)
        seq = np.array(list(self._window), dtype=np.float32)
        tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)

        # 4. Model inference
        with torch.no_grad():
            probs = self.model(tensor, mode='classify')  # (1, n_classes)

        probs_np = probs.cpu().numpy()[0]  # (n_classes,)
        predicted_class = int(np.argmax(probs_np))
        confidence = float(probs_np[predicted_class])

        is_attack = predicted_class != 0  # 0 = Benign
        attack_type = self.CLASS_NAMES.get(predicted_class, "Unknown")

        # Confidence: saldırı olasılığı (1 - benign prob)
        attack_confidence = float(1.0 - probs_np[0])

        return {
            "is_attack": is_attack,
            "confidence": attack_confidence,
            "attack_type": attack_type,
        }

    def is_ready(self) -> bool:
        return self._ready


class FlowGuardProvider(BaseModelProvider):
    """
    FlowGuard provider — model/canavar-model is the single source of truth.

    Preprocess is NOT reimplemented here. The provider imports the exact
    helper functions from src.data.preprocess and calls them in the same
    order as the training pipeline, so a feature dict produced live goes
    through byte-identical math as a row from the training CSV.

    Decision rule mirrors canavar-model evaluation code: logits.argmax(dim=1).
    No tunable threshold — softmax probability is reported only for UI/logging.

    Inputs (raw_features dict): all NF-v3 columns as they appear in the
    training CSV (IPV4_*, L4_*, FLOW_START_MILLISECONDS, FLOW_END_MILLISECONDS
    included; preprocess will drop them). Missing columns are filled with 0
    to match the inference-mode path in preprocess.preprocess_dataset.

    Files:
      - backend/saved_models/hardened_model.pt
      - backend/saved_models/flowguard_stats.npz
    """

    ENCODER_CONFIG = {
        'model': {
            'encoder': {
                'input_dim': 53,
                'model_dim': 128,
                'num_heads': 4,
                'num_layers': 4,
                'feedforward_dim': 512,
                'dropout': 0.1,
            },
            'classification_head': {
                'output_type': 'binary',
                'hidden_dims': [64],
            },
        }
    }
    NUM_DOMAINS = 4  # unsw, bot, ton, cic

    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    CANAVAR_SRC = os.path.join(PROJECT_ROOT, 'model', 'canavar-model')
    DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "hardened_model.pt")
    DEFAULT_STATS_PATH = os.path.join(BASE_DIR, "saved_models", "flowguard_stats.npz")

    def __init__(self, model_path: str = None, stats_path: str = None):
        self.model = None
        self.device = None
        self.stats = None  # PreprocessingStats instance from canavar-model
        self._ready = False
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.stats_path = stats_path or self.DEFAULT_STATS_PATH

    def _ensure_canavar_on_path(self):
        if self.CANAVAR_SRC not in sys.path:
            sys.path.insert(0, self.CANAVAR_SRC)

    def load(self) -> None:
        import torch

        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(
                f"FlowGuard stats not found: {self.stats_path}\n"
                "Source: model/canavar-model/data/processed/<dataset>_stats.npz"
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"FlowGuard checkpoint not found: {self.model_path}\n"
                "Source: model/canavar-model/checkpoints/phase5/hardened_model.pt"
            )

        self._ensure_canavar_on_path()
        from src.data.preprocess import PreprocessingStats
        from src.models.flowguard import FlowGuard

        self.stats = PreprocessingStats.load(self.stats_path)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FlowGuard(self.ENCODER_CONFIG)
        # hardened_model.pt includes DomainDiscriminator keys
        self.model.enable_domain_discriminator(num_domains=self.NUM_DOMAINS)
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._ready = True
        print(f"[FlowGuardProvider] Loaded model={self.model_path} | "
              f"device={self.device} | features={len(self.stats.feature_names)} | "
              f"log_transform={len(self.stats.log_transform_columns)}")

    def _build_vector(self, raw_features: dict):
        """Apply canavar-model preprocess in the exact training order.

        Single code path with training: same _bucket_ports, _handle_inf_and_nan,
        _log_transform, _zscore_normalize functions are invoked.
        """
        import numpy as np
        import pandas as pd
        self._ensure_canavar_on_path()
        from src.data.preprocess import (
            _bucket_ports,
            _handle_inf_and_nan,
            _log_transform,
            _zscore_normalize,
            _IDENTITY_COLUMNS,
            _TIMESTAMP_COLUMNS,
        )

        df = pd.DataFrame([raw_features])
        df.columns = df.columns.str.strip()

        if 'L4_SRC_PORT' in df.columns:
            df = pd.concat([df, _bucket_ports(df['L4_SRC_PORT'], 'SRC_PORT')], axis=1)
        if 'L4_DST_PORT' in df.columns:
            df = pd.concat([df, _bucket_ports(df['L4_DST_PORT'], 'DST_PORT')], axis=1)

        cols_to_drop = [c for c in _IDENTITY_COLUMNS + _TIMESTAMP_COLUMNS if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        df = _handle_inf_and_nan(df)

        # Inference-mode alignment: ensure every training feature column exists.
        feature_cols = self.stats.feature_names
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        _log_transform(df, self.stats.log_transform_columns)
        df, _, _ = _zscore_normalize(
            df,
            feature_cols,
            means=self.stats.feature_means,
            stds=self.stats.feature_stds,
        )

        return df[feature_cols].values.astype(np.float32)

    def predict(self, features: dict) -> dict:
        import torch
        if not self._ready:
            raise RuntimeError("Model not loaded. Call load() first.")

        vec = self._build_vector(features)  # (1, 53)
        tensor = torch.from_numpy(vec).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            # canavar-model decision rule: argmax over class logits.
            pred = int(logits.argmax(dim=1).item())
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        is_attack = pred == 1
        return {
            'is_attack': is_attack,
            'confidence': float(probs[1]),  # p_attack — UI only, not the decision
            'attack_type': 'Malicious' if is_attack else 'Benign',
        }

    def is_ready(self) -> bool:
        return self._ready

    def get_info(self) -> dict:
        info = super().get_info()
        info.update({
            'model': 'FlowGuard Phase 5 (Hardened)',
            'architecture': 'TransformerEncoder(53->128) + ClassificationHead(128->2)',
            'features': '53 NF-v3 features',
            'classification': 'Binary (Benign/Attack), argmax decision',
            'checkpoint': self.model_path,
        })
        return info


PROVIDER_REGISTRY = {
    "placeholder": PlaceholderModelProvider,
    "legacy": LegacySklearnProvider,
    "custom": CustomModelProvider,
    "guardian": CustomModelProvider,
    "flowguard": FlowGuardProvider,
}

def get_model_provider(provider_name: str = None, **kwargs) -> BaseModelProvider:
    """
    Provider adına göre uygun model provider'ı döndür.
    
    Args:
        provider_name: "placeholder", "legacy", "custom", "guardian", "flowguard"
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
