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
    FlowGuard Transformer-based NIDS Provider
    ===========================================
    model/canavar-model/ icindeki Phase 5 hardened FlowGuard modelini yukler.
    53 NF-v3 (NetFlow) feature ile binary siniflandirma yapar: Benign (0) / Attack (1).

    Mimari:
        FlowTransformerEncoder(53->128) + ClassificationHead(128->2) + DomainDiscriminator(128->4)
        Her feature bir token olarak islenir, CLS token eklenir, 4-layer Transformer.

    Preprocessing Pipeline:
        1. Raw feature dict -> 53 NF-v3 feature secimi
        2. Port bucketing (SRC/DST port -> 6 binary bucket)
        3. Log1p transform (heavy-tailed feature'lar)
        4. Z-score normalizasyon (egitimden kaydedilen mean/std)

    Checkpoint: backend/saved_models/hardened_model.pt
    Stats:      backend/saved_models/flowguard_stats.npz

    Kullanım:
        python analyze_engine.py --mode live --provider flowguard
    """

    # ── Model Hyperparameters ──
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
    NUM_DOMAINS = 4  # 4 dataset ile eğitildi (unsw, bot, ton, cic)

    # ── Dosya Yolları ──
    PROJECT_ROOT = os.path.dirname(BASE_DIR)  # backend/ → proje kökü
    DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "hardened_model.pt")
    DEFAULT_STATS_PATH = os.path.join(BASE_DIR, "saved_models", "flowguard_stats.npz")

    # ── Port Bucket Sınırları ──
    _PORT_BUCKETS = {
        'WELL_KNOWN': (0, 1023),
        'REGISTERED': (1024, 49151),
        'EPHEMERAL': (49152, 65535),
    }

    # ── Log Transform Uygulanacak Feature'lar ──
    _LOG_TRANSFORM_CANDIDATES = [
        'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
        'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
        'SRC_TO_DST_SECOND_BYTES', 'DST_TO_SRC_SECOND_BYTES',
        'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
        'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS',
        'SRC_TO_DST_AVG_THROUGHPUT', 'DST_TO_SRC_AVG_THROUGHPUT',
        'DNS_TTL_ANSWER',
        'SRC_TO_DST_IAT_MIN', 'SRC_TO_DST_IAT_MAX',
        'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV',
        'DST_TO_SRC_IAT_MIN', 'DST_TO_SRC_IAT_MAX',
        'DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV',
    ]

    def __init__(self, model_path: str = None, stats_path: str = None):
        self.model = None
        self.device = None
        self._ready = False

        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.stats_path = stats_path or self.DEFAULT_STATS_PATH

        # Preprocessing stats (eğitimden yüklenir)
        self.feature_names = None   # 53 feature sırası
        self.feature_means = None   # Z-score mean
        self.feature_stds = None    # Z-score std
        self.log_transform_columns = None  # Hangi feature'lara log1p uygulandı

    def load(self) -> None:
        """
        FlowGuard hardened modelini ve preprocessing stats'ını yükler.

        Yükleme sırası kritik:
          1. Stats (.npz) yükle
          2. FlowGuard model oluştur (config dict ile)
          3. DomainDiscriminator'ı etkinleştir (checkpoint'taki key'ler için)
          4. State dict yükle
          5. Eval moduna al
        """
        import torch
        import numpy as np
        import json

        # ── 1. Preprocessing Stats ──
        if not os.path.exists(self.stats_path):
            raise FileNotFoundError(
                f"FlowGuard stats dosyası bulunamadı: {self.stats_path}\n"
                "Bu dosya model eğitimi sırasında üretilir.\n"
                "Kaynak: model/canavar-model/data/processed/<dataset>_stats.npz"
            )

        stats = np.load(self.stats_path, allow_pickle=True)
        self.feature_names = stats['feature_names'].tolist()
        self.feature_means = stats['feature_means'].astype(np.float64)
        self.feature_stds = stats['feature_stds'].astype(np.float64)
        self.log_transform_columns = stats['log_transform_columns'].tolist()
        print(f"[FlowGuardProvider] Stats yuklendi: {self.stats_path}")
        print(f"[FlowGuardProvider]   Features: {len(self.feature_names)} | "
              f"Log-transform: {len(self.log_transform_columns)} feature")

        # ── 2. Model ──
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"FlowGuard model dosyası bulunamadı: {self.model_path}\n"
                "Bu dosya Phase 5 hardening sonrası üretilir.\n"
                "Kaynak: model/canavar-model/checkpoints/phase5/hardened_model.pt"
            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # FlowGuard sınıfını import et
        canavar_src = os.path.join(self.PROJECT_ROOT, 'model', 'canavar-model')
        if canavar_src not in sys.path:
            sys.path.insert(0, canavar_src)
        from src.models.flowguard import FlowGuard

        # ── 3. Model oluştur + DomainDiscriminator etkinleştir ──
        self.model = FlowGuard(self.ENCODER_CONFIG)

        # KRİTİK: hardened_model.pt checkpoint'u DomainDiscriminator
        # katmanlarını da içerir. Bu çağrı yapılmazsa load_state_dict
        # "Unexpected key(s): domain_disc.*" hatası verir.
        self.model.enable_domain_discriminator(num_domains=self.NUM_DOMAINS)

        # ── 4. State dict yükle ──
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._ready = True
        print(f"[FlowGuardProvider] Model yuklendi: {self.model_path}")
        print(f"[FlowGuardProvider] Device: {self.device} | "
              f"Encoder: Transformer(53->128) | Classes: 2 (Benign/Attack)")

    def _bucket_port(self, port_value) -> dict:
        """
        Port numarasını 3 binary bucket'a çevir.

        Args:
            port_value: Port numarası (int/float/str)

        Returns:
            dict: {'WELL_KNOWN': 0/1, 'REGISTERED': 0/1, 'EPHEMERAL': 0/1}
        """
        try:
            port = int(float(port_value))
        except (ValueError, TypeError):
            port = 0

        result = {}
        for bucket_name, (lo, hi) in self._PORT_BUCKETS.items():
            result[bucket_name] = 1 if lo <= port <= hi else 0
        return result

    def _preprocess(self, features: dict):
        """
        Raw NF-v3 feature dict → normalized (1, 53) numpy array.

        Pipeline (eğitimle aynı sıra):
          1. 47 ham NF-v3 feature'ı seç
          2. Port bucketing → 6 ek feature ekle
          3. log1p(|x|) transform (heavy-tailed feature'lar)
          4. Z-score normalize (mean/std)
        """
        import numpy as np

        # ── 1. Feature vektörü oluştur (feature_names sırasına göre) ──
        feature_vector = np.zeros(len(self.feature_names), dtype=np.float64)

        # Port bucket'ları hesapla
        src_port_buckets = self._bucket_port(features.get('L4_SRC_PORT', 0))
        dst_port_buckets = self._bucket_port(features.get('L4_DST_PORT', 0))

        # Fix: We're zeroing out port bucket features to prevent the model from
        # automatically classifying outbound traffic (from ephemeral ports) as an attack.
        port_bucket_map = {
            'SRC_PORT_WELL_KNOWN': 0.0,
            'SRC_PORT_REGISTERED': 0.0,
            'SRC_PORT_EPHEMERAL': 0.0,
            'DST_PORT_WELL_KNOWN': 0.0,
            'DST_PORT_REGISTERED': 0.0,
            'DST_PORT_EPHEMERAL': 0.0,
        }

        for i, fname in enumerate(self.feature_names):
            # Port bucket feature'ları
            if fname in port_bucket_map:
                feature_vector[i] = port_bucket_map[fname]
                continue

            # Ham NF-v3 feature'ları
            raw_val = features.get(fname, 0.0)
            try:
                val = float(raw_val)
            except (ValueError, TypeError):
                val = 0.0

            # Inf/NaN temizle
            if np.isinf(val) or np.isnan(val):
                val = 0.0

            feature_vector[i] = val

        # ── 2. Log1p transform (eğitimle aynı feature'lar) ──
        for col_name in self.log_transform_columns:
            if col_name in self.feature_names:
                idx = self.feature_names.index(col_name)
                feature_vector[idx] = np.log1p(np.abs(feature_vector[idx]))

        # ── 3. Z-score normalize ──
        eps = 1e-8
        feature_vector = (feature_vector - self.feature_means) / (self.feature_stds + eps)

        return feature_vector.astype(np.float32)

    def predict(self, features: dict) -> dict:
        """
        NF-v3 feature dict alıp binary tahmin yapar.

        Args:
            features: NF-v3 formatında feature dict
                      (L4_SRC_PORT, PROTOCOL, IN_BYTES, vb.)

        Returns:
            dict:
                - "is_attack": bool
                - "confidence": float (0.0-1.0, saldırı olasılığı)
                - "attack_type": "Benign" | "Malicious"
        """
        import torch
        import numpy as np

        if not self._ready:
            raise RuntimeError("Model yüklenmedi. Önce load() çağırın.")

        # 1. Preprocess
        vec = self._preprocess(features)

        # 2. Tensor oluştur (1, 53)
        tensor = torch.from_numpy(vec).unsqueeze(0).to(self.device)

        # 3. Model inference
        with torch.no_grad():
            logits = self.model(tensor)  # (1, 2)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [p_benign, p_attack]

        p_benign = float(probs[0])
        p_attack = float(probs[1])

        # THRESHOLD = 0.50 (Canavar model, softmax kullandigi icin 0.50 optimum karar siniridir)
        # Bazi false positive'leri engellemek icin 0.40'dan 0.50'ye yukseltildi.
        is_attack = p_attack > 0.50

        return {
            'is_attack': is_attack,
            'confidence': p_attack,
            'attack_type': 'Malicious' if is_attack else 'Benign',
        }

    def is_ready(self) -> bool:
        return self._ready

    def get_info(self) -> dict:
        """Provider hakkında detaylı bilgi."""
        info = super().get_info()
        info.update({
            'model': 'FlowGuard Phase 5 (Hardened)',
            'architecture': 'TransformerEncoder(53->128) + ClassificationHead(128->2)',
            'features': '53 NF-v3 features',
            'classification': 'Binary (Benign/Attack)',
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
