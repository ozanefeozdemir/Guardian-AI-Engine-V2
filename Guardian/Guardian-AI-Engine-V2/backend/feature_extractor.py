import pandas as pd
import numpy as np

# Feature Mapping from CIC-IDS 2017/2018 to our internal feature set
MAPPING = {
    'Dst Port': 'Destination Port',
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'Tot Bwd Pkts': 'Total Backward Packets',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
    'Fwd Pkt Len Max': 'Fwd Packet Length Max',
    'Fwd Pkt Len Min': 'Fwd Packet Length Min',
    'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
    'Fwd Pkt Len Std': 'Fwd Packet Length Std',
    'Bwd Pkt Len Max': 'Bwd Packet Length Max',
    'Bwd Pkt Len Min': 'Bwd Packet Length Min',
    'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
    'Bwd Pkt Len Std': 'Bwd Packet Length Std',
    'Flow Byts/s': 'Flow Bytes/s',
    'Flow Pkts/s': 'Flow Packets/s',
    'Fwd IAT Tot': 'Fwd IAT Total',
    'Bwd IAT Tot': 'Bwd IAT Total',
    'Fwd Pkts/s': 'Fwd Packets/s',
    'Bwd Pkts/s': 'Bwd Packets/s',
    'Pkt Len Min': 'Min Packet Length',
    'Pkt Len Max': 'Max Packet Length',
    'Pkt Len Mean': 'Packet Length Mean',
    'Pkt Len Std': 'Packet Length Std',
    'Pkt Len Var': 'Packet Length Variance',
    'FIN Flag Cnt': 'FIN Flag Count',
    'SYN Flag Cnt': 'SYN Flag Count',
    'RST Flag Cnt': 'RST Flag Count',
    'PSH Flag Cnt': 'PSH Flag Count',
    'ACK Flag Cnt': 'ACK Flag Count',
    'URG Flag Cnt': 'URG Flag Count',
    'ECE Flag Cnt': 'ECE Flag Count',
    'Pkt Size Avg': 'Average Packet Size',
    'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
    'Bwd Seg Size Avg': 'Avg Bwd Segment Size',
    'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
    'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk',
    'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate',
    'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk',
    'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk',
    'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
    'Subflow Fwd Pkts': 'Subflow Fwd Packets',
    'Subflow Fwd Byts': 'Subflow Fwd Bytes',
    'Subflow Bwd Pkts': 'Subflow Bwd Packets',
    'Subflow Bwd Byts': 'Subflow Bwd Bytes',
    'Init Fwd Win Byts': 'Init_Win_bytes_forward',
    'Init Bwd Win Byts': 'Init_Win_bytes_backward',
    'Fwd Act Data Pkts': 'act_data_pkt_fwd',
    'Fwd Seg Size Min': 'min_seg_size_forward'
}

# The ordered list of features expected by the model (78 features approx)
# We derive this dynamically or hardcode it. 
# For safety, let's trust the MAPPING values + any others that might be common.
# In train scripts, we used: list(MAPPING.values())
ORDERED_FEATURES = list(MAPPING.values())

class FeatureExtractor:
    def __init__(self, scaler):
        """
        Args:
            scaler: A fitted sklearn StandardScaler/MinMaxScaler.
        """
        self.scaler = scaler

    def transform(self, input_data: dict) -> np.ndarray:
        """
        Transforms a dictionary of raw features (e.g., from JSON) into 
        the scaled numpy array expected by the model.
        """
        # 1. Convert dict to DataFrame (single row)
        # We assume input keys match the RAW CSV headers (e.g. 'Dst Port') 
        # OR the mapped headers (e.g. 'Destination Port').
        # We'll normalize to Mapped headers.
        
        df = pd.DataFrame([input_data])
        
        # 2. Rename keys if they match the "old" names
        df = df.rename(columns=MAPPING)
        
        # 3. Ensure all expected columns exist, fill missing with 0
        for col in ORDERED_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
                
        # 4. Reorder columns to match training order
        X = df[ORDERED_FEATURES].copy()
        
        # 5. Clean / Coerce to Numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').astype('float32')
            
        X.replace([np.inf, -np.inf], 0, inplace=True)
        X.fillna(0, inplace=True)
        
        # 6. Scale
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
