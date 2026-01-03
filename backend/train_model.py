
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time
import os
import gc
import argparse

# --- Configuration ---
# Current Dir: backend/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Datasets are at backend/datasets/raw/...
DATASET_DIR_2017 = os.path.join(BASE_DIR, "datasets", "raw", "CIC-IDS 2017")
DATASET_DIR_2018 = os.path.join(BASE_DIR, "datasets", "raw", "CIC-IDS 2018")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "base_rf_2017.pkl")
SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "scaler_base.pkl")

# File Paths
TRAIN_FILES_2017 = [
    os.path.join(DATASET_DIR_2017, "Monday-WorkingHours.pcap_ISCX.csv"),
    os.path.join(DATASET_DIR_2017, "Wednesday-workingHours.pcap_ISCX.csv"),
    os.path.join(DATASET_DIR_2017, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    os.path.join(DATASET_DIR_2017, "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
]

# Default 2018 file
DEFAULT_FILE_2018 = os.path.join(DATASET_DIR_2018, "02-14-2018.csv") # Updated to 03-02 based on user usage
# Adaptation Settings
MODULO_FACTOR = 20 # 5% Calibration
# Feature Mapping
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

def load_process_df(file_path, mapping, skiprows_func=None):
    print(f"    Loading: {os.path.basename(file_path)}...", flush=True)
    try:
        df = pd.read_csv(file_path, skiprows=skiprows_func, encoding='latin1')
    except UnicodeDecodeError:
        # Fallback if latin1 fails (highly unlikely for these datasets)
        df = pd.read_csv(file_path, skiprows=skiprows_func, encoding='cp1252')
    except FileNotFoundError:
        print(f"    [ERROR] File not found: {file_path}", flush=True)
        return None
        
    df.columns = df.columns.str.strip()
    df = df.rename(columns=mapping)
    
    # Filter Columns
    keep_cols = list(mapping.values()) + ['Label']
    valid_cols = [c for c in keep_cols if c in df.columns]
    df = df[valid_cols]

    # --- FIX: Remove Repeated Headers ---
    if 'Destination Port' in df.columns:
        df = df[pd.to_numeric(df['Destination Port'], errors='coerce').notnull()]

    for col in valid_cols:
        if col == 'Label': continue
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        
    # FIX: Infinity Handling
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def process_chunk(df, mapping, training_columns, scaler):
    df.columns = df.columns.str.strip()
    df = df.rename(columns=mapping)

    # --- FIX: Remove Repeated Headers ---
    if 'Destination Port' in df.columns:
        df = df[pd.to_numeric(df['Destination Port'], errors='coerce').notnull()]
    
    if 'Label' not in df.columns: return None, None
    y_raw = df['Label']
    y_true = np.where(y_raw.astype(str).str.lower() == 'benign', 0, 1)
    
    current_cols = set(df.columns)
    missing_cols = set(training_columns) - current_cols
    for c in missing_cols: df[c] = 0
    
    X = df[training_columns].copy()
    
    # Force Numeric & Clean
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').astype('float32')

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    
    if len(X) == 0:
        return None, None
        
    X_scaled = scaler.transform(X)
    
    return X_scaled, y_true

def train_base_model():
    print("="*60)
    print("MODE: TRAIN BASE MODEL (2017)")
    print("="*60)

    # 1. Load 2017 Data
    print("\n[1] Loading 2017 Base Data...")
    list_dfs = []
    for f in TRAIN_FILES_2017:
        df = load_process_df(f, MAPPING)
        if df is not None: list_dfs.append(df)
    
    if not list_dfs:
        print("Error: No 2017 data found.")
        return

    df_2017 = pd.concat(list_dfs, ignore_index=True)
    y_train = np.where(df_2017['Label'].astype(str).str.lower() == 'benign', 0, 1)
    X_train = df_2017.drop(columns=['Label'])
    
    print(f"    -> Base Samples: {len(X_train)}")
    del list_dfs, df_2017
    gc.collect()

    # 2. Scale & Train
    print("\n[2] Training Random Forest...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Enable warm_start for future adaptation
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42, warm_start=True)
    rf.fit(X_train_scaled, y_train)
    
    # 3. Save
    print("\n[3] Saving Base Model...")
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"    -> Model saved to: {MODEL_PATH}")
    print(f"    -> Scaler saved to: {SCALER_PATH}")
    
    # SAVE AS rf_comprehensive.pkl TOO for direct compatibility checking if needed
    # (Though API uses base_rf_2017.pkl according to updated settings)
    model_path_comp = os.path.join(SAVED_MODELS_DIR, "rf_comprehensive.pkl")
    scaler_path_comp = os.path.join(SAVED_MODELS_DIR, "scaler_comprehensive.pkl")
    joblib.dump(rf, model_path_comp)
    joblib.dump(scaler, scaler_path_comp)


def adapt_and_test(target_file):
    print("="*60)
    print("MODE: ADAPT & TEST (2018)")
    print(f"Target File: {os.path.basename(target_file)}")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Base model not found. Run --mode train_base first.")
        return

    # 1. Load Base Model
    print("\n[1] Loading Base Model...")
    rf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # 2. Load Calibration Data (5% of 2018)
    print(f"\n[2] Loading Calibration Data (Every {MODULO_FACTOR}th row)...")
    calib_skip_lambda = lambda x: x > 0 and x % MODULO_FACTOR != 0
    df_calib = load_process_df(target_file, MAPPING, skiprows_func=calib_skip_lambda)
    
    if df_calib is None: return

    y_calib = np.where(df_calib['Label'].astype(str).str.lower() == 'benign', 0, 1)
    
    # Align Columns
    expected_cols = [c for c in MAPPING.values() if c in df_calib.columns] 
    
    X_calib = df_calib.drop(columns=['Label'])
    X_calib_scaled = scaler.transform(X_calib)
    
    print(f"    -> Calibration Samples: {len(X_calib)}")
    print("    -> Adapting Model (Warm Start)...")
    
    # Increase Trees and Retrain
    # Increase Trees and Retrain
    if len(np.unique(y_calib)) < 2:
        print(f"    -> [SKIP] Calibration data has only 1 class ({np.unique(y_calib)}). Skipping adaptation to prevent errors.")
    else:
        current_trees = rf.n_estimators
        new_trees = current_trees + 50
        print(f"    -> Increasing Trees: {current_trees} -> {new_trees}")
        
        rf.n_estimators = new_trees
        rf.fit(X_calib_scaled, y_calib)
    
    # 3. Test
    print(f"\n[3] Testing on Remaining Data...")
    training_cols = list(X_calib.columns) 
    
    del X_calib, X_calib_scaled, y_calib
    gc.collect()

    y_true_all = []
    y_pred_all = []
    
    chunk_size = 200000
    try:
        reader = pd.read_csv(target_file, chunksize=chunk_size, on_bad_lines='skip', encoding='latin1')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    global_row_counter = 0     
    chunk_idx = 0
    
    for chunk in reader:
        chunk_idx += 1
        print(f"    Processing Chunk {chunk_idx}...", end='\r', flush=True)
        
        chunk_len = len(chunk)
        indices = np.arange(global_row_counter, global_row_counter + chunk_len)
        global_row_counter += chunk_len
        
        mask = (indices % MODULO_FACTOR != 0)
        chunk_test = chunk.iloc[mask]
        
        if len(chunk_test) == 0: continue
        
        X_chunk_scaled, y_chunk = process_chunk(chunk_test, MAPPING, training_cols, scaler)
        if X_chunk_scaled is None: continue
        
        y_pred_chunk = rf.predict(X_chunk_scaled)
        
        y_true_all.extend(y_chunk)
        y_pred_all.extend(y_pred_chunk)
        
    print(f"\n    Processed all chunks.")
    print("="*60)
    print("RESULTS")
    print("="*60)
    if len(y_true_all) > 0:
        print(classification_report(y_true_all, y_pred_all, target_names=['Benign', 'Attack'], digits=5))
        
        conf_mat = confusion_matrix(y_true_all, y_pred_all)
        tn, fp, fn, tp = conf_mat.ravel()
        print("Confusion Matrix:")
        print(f"TP (Caught): {tp}")
        print(f"FN (Missed): {fn}")
        print(f"FP (False Alarm): {fp}")
        print(f"TN (Good): {tn}")
        print(f"Total Benign (Actual): {tn + fp}")
        print(f"Model Predicted Benign: {tn + fn}")
        print(f"Total Attack (Actual): {tp + fn}")
        print(f"Model Predicted Attack: {tp + fp}")
    else:
        print("No test data.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train_base", "adapt"], required=True, help="Usage: --mode train_base OR --mode adapt")
    parser.add_argument("--file", default=DEFAULT_FILE_2018, help="Path to 2018 CSV (only for adapt mode)")
    args = parser.parse_args()
    
    if args.mode == "train_base":
        train_base_model()
    elif args.mode == "adapt":
        adapt_and_test(args.file)

if __name__ == "__main__":
    main()
