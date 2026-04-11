
import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

class GuardianDataLoader:
    def __init__(self, seq_len=10):
        self.seq_len = seq_len
        self.scaler = MinMaxScaler()
        # Columns to drop as per requirement to prevent overfitting to identity
        self.drop_cols = [
            'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 
            'Destination Port', 'Protocol', 'Timestamp', 'SimillarHTTP'
        ]
        # Common column mapping to standardize names if needed
        # In CIC-IDS, usually just stripping whitespace is enough
        
    def _load_csvs(self, directory, limit=None, max_files=None):
        """Loads all CSV files from a directory."""
        all_files = glob.glob(os.path.join(directory, "*.csv"))
        df_list = []
        
        for i, filename in enumerate(all_files):
            if max_files is not None and i >= max_files:
                break
                
            print(f"Loading {filename}...")
            try:
                # Some files might have encoding issues or mixed types, handling robustly
                if limit:
                    df = pd.read_csv(filename, encoding='cp1252', low_memory=False, nrows=limit)
                else:
                    df = pd.read_csv(filename, encoding='cp1252', low_memory=False)
                df_list.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
            
        if not df_list:
            print(f"No CSV files found in {directory}")
            return pd.DataFrame()
            
        return pd.concat(df_list, ignore_index=True)

    def standardize_columns(self, df):
        """Strips whitespace from column names."""
        df.columns = df.columns.str.strip()
        return df

    def clean_data(self, df, mode='train_autoencoder', fill_stats=None):
        """
        Cleans data, drops variables, and handles labels.
        mode:
         - 'train_autoencoder': Returns only BENIGN data (dropped labels).
         - 'train_classifier': Returns all data with encoded labels (X, y).
        fill_stats: dict of {col: max_value} to replace Inf/NaN with. 
                    If None, uses local df.max().
        """
        # Standardize columns first
        df = self.standardize_columns(df)
        
        # Define Label Mapping
        # CIC-IDS-2017 specific mapping (can be expanded)
        label_map = {
            'BENIGN': 0,
            'DDoS': 1,
            'PortScan': 2,
            'Web Attack  Brute Force': 3,
            'Web Attack  XSS': 3,
            'Web Attack  Sql Injection': 3,
            'Bot': 4
        }
        
        # Filter logic
        if 'Label' in df.columns:
            # Map labels
            # Handle slight variations/encodings
            df['Label'] = df['Label'].astype(str).str.strip()
            
            def get_label_int(label):
                # Normalize key
                label_upper = label.upper()
                
                # Direct match
                if label in label_map:
                    return label_map[label]
                if label_upper in label_map:
                    return label_map[label_upper]
                    
                # 2018 / Robust Mappings
                if 'BENIGN' in label_upper:
                    return 0
                # Exclude 2018-only attack types not present in 2017 training data
                if 'LOIC' in label_upper or 'HOIC' in label_upper:
                    return -1
                if 'DOS' in label_upper or 'DDOS' in label_upper:
                    return 1 # Map all DoS/DDoS to 1
                if 'PORT' in label_upper or 'SSH' in label_upper or 'FTP' in label_upper:
                    return 2 # Map PortScan/Auth Brute to 2
                if 'WEB' in label_upper or 'XSS' in label_upper or 'SQL' in label_upper:
                    return 3 # Web Attacks
                if 'BOT' in label_upper:
                    return 4
                
                return -1
            
            # Create a numeric label column
            df['Label_Int'] = df['Label'].apply(get_label_int)
            
            if mode == 'train_autoencoder':
                # Filter for BENIGN only
                df = df[df['Label_Int'] == 0]
                labels = None # We don't use labels for AE training input (reconstruction)
            elif mode == 'train_classifier':
                # Filter out unknown labels if necessary, or map them?
                # For this demo, we keep only known classes
                df = df[df['Label_Int'] != -1]
                labels = df['Label_Int'].values
        else:
            print("Warning: 'Label' column not found.")
            labels = None
            
        # Drop identity columns
        cols_to_drop = [c for c in self.drop_cols if c in df.columns]
        df = df.drop(columns=cols_to_drop)
        
        # Drop Label columns
        if 'Label' in df.columns:
            df = df.drop(columns=['Label'])
        if 'Label_Int' in df.columns:
            df = df.drop(columns=['Label_Int'])

        # Handle Infinite and NaN
        # First ensure numeric types (coercing errors to NaN)
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Replace Inf with NaN temporarily
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill NaNs with the Maximum value of that column (to preserve "High Rate" signal)
        # instead of 0 (which looks like "Idle").
        # If fill_stats provided (from Training Scaler), use that.
        # Otherwise use local max.
        if fill_stats:
            # fillna with a dict {col: value}
            # Only for columns present in fill_stats and df
            df.fillna(value=fill_stats, inplace=True)
            # Cleanup any remaining
            df.fillna(0, inplace=True)
        else:
            df.fillna(df.max(), inplace=True)
            df.fillna(0, inplace=True) # Catch-all for columns that were all NaN
        
        return df, labels

    def fit_transform(self, df):
        """Fits the scaler and transforms the data."""
        # Ensure all data is numeric
        df = df.select_dtypes(include=[np.number])
        x_scaled = self.scaler.fit_transform(df)
        return x_scaled

    def transform(self, df):
        """Transforms data using the fitted scaler."""
        # Align columns with scaler if possible
        if hasattr(self.scaler, 'feature_names_in_'):
            df = df.reindex(columns=self.scaler.feature_names_in_, fill_value=0)
            
        df = df.select_dtypes(include=[np.number])
        return self.scaler.transform(df)
    
    def save_scaler(self, path='scaler.save'):
        joblib.dump(self.scaler, path)
        
    def load_scaler(self, path='scaler.save'):
        self.scaler = joblib.load(path)

    def create_sequences(self, data, labels=None):
        """
        Transforms 2D array (Samples, Features) into 3D array (Samples, Seq_Len, Features).
        If labels provided, returns valid label for each sequence (usually the label of the last step).
        """
        if len(data) < self.seq_len:
            return np.array([]), np.array([])
            
        n_samples, n_features = data.shape
        window_shape = (n_samples - self.seq_len + 1, self.seq_len, n_features)
        window_strides = (data.strides[0], data.strides[0], data.strides[1])
        
        sequences = np.lib.stride_tricks.as_strided(data, shape=window_shape, strides=window_strides)
        
        if labels is not None:
            # Labels: We assume the label of the sequence is the label of the last packet/flow in the window
            # Or we could take mode. Last is common for real-time.
            # labels shape: (n_samples,)
            # We want (n_samples - seq_len + 1,)
            seq_labels = labels[self.seq_len - 1:]
            return sequences, seq_labels
        
        return sequences, None

# Helper function to serve as the main entry point for data tasks
def process_pipeline(primary_path=None, secondary_path=None, scaler_path=None, mode='train_autoencoder', dataset_type='ids2017', limit=None, max_files=None):
    
    # Resolve default paths relative to this script (engine/data_loader.py -> ../)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if primary_path is None:
        primary_path = os.path.join(base_dir, 'data', 'train', 'ids-2017')
        
    if scaler_path is None:
        scaler_path = os.path.join(base_dir, 'checkpoints', 'guardian_scaler.pkl')

    loader = GuardianDataLoader(seq_len=10)
    
    # Pre-load Scaler statistics if we are in inference/eval mode
    # This ensures we fill 'Infinity' with the Training Data's Max, not the Test Data's local max.
    fill_stats = None
    if mode != 'train_autoencoder' and os.path.exists(scaler_path):
        try:
             # We rely on the fact that loader.load_scaler puts it in self.scaler
             s = joblib.load(scaler_path)
             if hasattr(s, 'feature_names_in_') and hasattr(s, 'data_max_'):
                 fill_stats = dict(zip(s.feature_names_in_, s.data_max_))
        except Exception:
            pass
    
    # 1. Ingestion
    print(f"Loading Primary Dataset from {primary_path}...")
    df1 = loader._load_csvs(primary_path, limit=limit, max_files=max_files)
    
    # 2018 Support: Column Mapping
    if dataset_type == 'ids2018':
        print("Applying CIC-IDS-2018 Column Mapping...")
        # Map 2018 cols to 2017 cols
        rename_map = {
            'Dst Port': ' Destination Port',
            'Tot Fwd Pkts': ' Total Fwd Packets',
            'Tot Bwd Pkts': ' Total Backward Packets',
            'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
            'TotLen Bwd Pkts': ' Total Length of Bwd Packets',
            'Fwd Pkt Len Max': ' Fwd Packet Length Max',
            'Fwd Pkt Len Min': ' Fwd Packet Length Min',
            'Fwd Pkt Len Mean': ' Fwd Packet Length Mean',
            'Fwd Pkt Len Std': ' Fwd Packet Length Std',
            'Bwd Pkt Len Max': 'Bwd Packet Length Max',
            'Bwd Pkt Len Min': ' Bwd Packet Length Min',
            'Bwd Pkt Len Mean': ' Bwd Packet Length Mean',
            'Bwd Pkt Len Std': ' Bwd Packet Length Std',
            'Flow Byts/s': 'Flow Bytes/s',
            'Flow Pkts/s': ' Flow Packets/s',
            'Flow IAT Mean': ' Flow IAT Mean',
            'Flow IAT Std': ' Flow IAT Std',
            'Flow IAT Max': ' Flow IAT Max',
            'Flow IAT Min': ' Flow IAT Min',
            'Fwd IAT Tot': 'Fwd IAT Total',
            'Fwd IAT Mean': ' Fwd IAT Mean',
            'Fwd IAT Std': ' Fwd IAT Std',
            'Fwd IAT Max': ' Fwd IAT Max',
            'Fwd IAT Min': ' Fwd IAT Min',
            'Bwd IAT Tot': 'Bwd IAT Total',
            'Bwd IAT Mean': ' Bwd IAT Mean',
            'Bwd IAT Std': ' Bwd IAT Std',
            'Bwd IAT Max': ' Bwd IAT Max',
            'Bwd IAT Min': ' Bwd IAT Min',
            'Fwd PSH Flags': 'Fwd PSH Flags',
            'Bwd PSH Flags': ' Bwd PSH Flags',
            'Fwd URG Flags': ' Fwd URG Flags',
            'Bwd URG Flags': ' Bwd URG Flags',
            'Fwd Header Len': ' Fwd Header Length',
            'Bwd Header Len': ' Bwd Header Length',
            'Fwd Pkts/s': 'Fwd Packets/s',
            'Bwd Pkts/s': ' Bwd Packets/s',
            'Pkt Len Min': ' Min Packet Length',
            'Pkt Len Max': ' Max Packet Length',
            'Pkt Len Mean': ' Packet Length Mean',
            'Pkt Len Std': ' Packet Length Std',
            'Pkt Len Var': ' Packet Length Variance',
            'FIN Flag Cnt': 'FIN Flag Count',
            'SYN Flag Cnt': ' SYN Flag Count',
            'RST Flag Cnt': ' RST Flag Count',
            'PSH Flag Cnt': ' PSH Flag Count',
            'ACK Flag Cnt': ' ACK Flag Count',
            'URG Flag Cnt': ' URG Flag Count',
            'CWE Flag Count': ' CWE Flag Count',
            'ECE Flag Cnt': ' ECE Flag Count',
            'Down/Up Ratio': ' Down/Up Ratio',
            'Pkt Size Avg': ' Average Packet Size',
            'Fwd Seg Size Avg': ' Avg Fwd Segment Size',
            'Bwd Seg Size Avg': ' Avg Bwd Segment Size',
            'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
            'Fwd Pkts/b Avg': ' Fwd Avg Packets/Bulk',
            'Fwd Blk Rate Avg': ' Fwd Avg Bulk Rate',
            'Bwd Byts/b Avg': ' Bwd Avg Bytes/Bulk',
            'Bwd Pkts/b Avg': ' Bwd Avg Packets/Bulk',
            'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
            'Subflow Fwd Pkts': 'Subflow Fwd Packets',
            'Subflow Fwd Byts': ' Subflow Fwd Bytes',
            'Subflow Bwd Pkts': ' Subflow Bwd Packets',
            'Subflow Bwd Byts': ' Subflow Bwd Bytes',
            'Init Fwd Win Byts': 'Init_Win_bytes_forward',
            'Init Bwd Win Byts': ' Init_Win_bytes_backward',
            'Fwd Act Data Pkts': ' act_data_pkt_fwd',
            'Fwd Seg Size Min': ' min_seg_size_forward',
            'Active Mean': 'Active Mean',
            'Active Std': ' Active Std',
            'Active Max': ' Active Max',
            'Active Min': ' Active Min',
            'Idle Mean': 'Idle Mean',
            'Idle Std': ' Idle Std',
            'Idle Max': ' Idle Max',
            'Idle Min': ' Idle Min',
            'Label': ' Label',
            # Identity Columns to map so they get dropped correctly
            'Dst IP': ' Destination IP',
            'Src IP': ' Source IP',
            'Src Port': ' Source Port',
            'Flow ID': 'Flow ID',
            'Protocol': 'Protocol',
            'Timestamp': 'Timestamp'
        }
        df1 = df1.rename(columns=rename_map)
        
        # Handle duplicate feature Fwd Header Length.1 if missing
        # CIC-IDS-2017 often has this duplicate. If the scaler expects it, we must provide it.
        if ' Fwd Header Length' in df1.columns and ' Fwd Header Length.1' not in df1.columns:
            print("Adding missing duplicate feature: Fwd Header Length.1")
            df1[' Fwd Header Length.1'] = df1[' Fwd Header Length']

        # Verify columns vs 2017 expected
        # Ideally we'd filter strictly, but let clean_data handle dropping.
        # We just ensured names match what clean_data expects to DROP or KEEP.
    
    # ... (Secondary loading omitted for brevity, assuming similar logic or merged)
    # For robust demo, let's keep it simple with df1 if secondary not critical for syntax
    if secondary_path:
        print(f"Loading Secondary Dataset from {secondary_path}...")
        df2 = loader._load_csvs(secondary_path)
        full_df = pd.concat([df1, df2], axis=0, ignore_index=True)
    else:
        full_df = df1

    if full_df.empty:
        # Return empty arrays to handle gracefully
        return np.array([]), np.array([]), loader # type: ignore

    # 2. Cleaning
    print(f"Cleaning data for mode: {mode}...")
    cleaned_df, labels = loader.clean_data(full_df, mode=mode, fill_stats=fill_stats)
    
    if cleaned_df.empty:
        print("Error: Cleaned DataFrame is empty! Check label mapping or input data.")
        return np.array([]), np.array([]), loader
    
    # 3. Normalization
    print("Normalizing data...")
    if mode == 'train_autoencoder':
       # Fit new scaler
       print("Fitting new scaler...")
       data_scaled = loader.fit_transform(cleaned_df)
       print(f"Saving scaler to {scaler_path}...")
       loader.save_scaler(scaler_path)
    else:
       # Load existing scaler
       # For evaluation (test 2018), we MUST use the 2017 scaler to match training distribution
       if os.path.exists(scaler_path):
           print(f"Loading scaler from {scaler_path}...")
           loader.load_scaler(scaler_path)
           data_scaled = loader.transform(cleaned_df)
       else:
           print("Warning: Scaler not found. Fitting on current data (this might be wrong for inference).")
           data_scaled = loader.fit_transform(cleaned_df)
    
    # 4. Sliding Window
    print("Creating Sequences...")
    sequences, seq_labels = loader.create_sequences(data_scaled, labels)
    
    if seq_labels is None:
        seq_labels = np.array([])

    print(f"Data Processing Complete. Sequence Shape: {sequences.shape}")
    return sequences, seq_labels, loader
