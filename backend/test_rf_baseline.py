"""
Test the base Random Forest model (trained on 2017) against 3 random 2018 dataset files.
TWO MODES: Baseline (no adaptation) vs Adapted (5% retraining).
Simulates the model "learning the current network" for a few days.
"""
import os
import random
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.metrics import confusion_matrix

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "base_rf_2017.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "saved_models", "scaler_base.pkl")
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "raw", "CIC-IDS 2018")

# Feature mapping (2018 column names -> 2017 column names)
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

ORDERED_FEATURES = list(MAPPING.values())


def adapt_model(model, scaler, file_path):
    """
    5% Adaptation: Read every 20th row of the file and retrain the model.
    Simulates the model observing the current network for a few days.
    """
    print(f"   [Adaptation] Reading 5% calibration data...")
    
    calib_skip = lambda x: x > 0 and x % 20 != 0
    df = pd.read_csv(file_path, skiprows=calib_skip, low_memory=False)
    df.columns = df.columns.str.strip()
    
    if 'Label' not in df.columns:
        print("   [Adaptation] No Label column. Skipping.")
        return model
    
    df = df.rename(columns=MAPPING)
    y = np.where(df['Label'].astype(str).str.lower() == 'benign', 0, 1)
    
    for col in ORDERED_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    
    X = df[ORDERED_FEATURES].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').astype('float32')
    X.replace([np.inf, -np.inf], 0, inplace=True)
    X.fillna(0, inplace=True)
    
    X_scaled = scaler.transform(X)
    
    n_attack = np.sum(y == 1)
    n_benign = np.sum(y == 0)
    unique = np.unique(y)
    print(f"   [Adaptation] Calibration: {len(y):,} rows | Benign: {n_benign:,} | Attacks: {n_attack:,}")
    
    if len(unique) < 2:
        print(f"   [Adaptation] Only one class ({unique}). Skipping retrain.")
        return model
    
    old_trees = model.n_estimators
    model.n_estimators += 50
    model.fit(X_scaled, y)
    print(f"   [Adaptation] Retrained! Trees: {old_trees} -> {model.n_estimators}")
    
    return model


def evaluate_file(model, scaler, file_path):
    """Evaluate the model on a full file. Returns metrics dict."""
    all_y_true = []
    all_y_pred = []
    attack_types = {}
    rows = 0
    
    for chunk in pd.read_csv(file_path, chunksize=50000, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        if 'Label' not in chunk.columns:
            continue
        
        labels_raw = chunk['Label'].astype(str).str.strip()
        for label in labels_raw:
            attack_types[label] = attack_types.get(label, 0) + 1
        
        y_true = np.where(labels_raw.str.lower() == 'benign', 0, 1)
        
        chunk = chunk.rename(columns=MAPPING)
        for col in ORDERED_FEATURES:
            if col not in chunk.columns:
                chunk[col] = 0.0
        
        X = chunk[ORDERED_FEATURES].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').astype('float32')
        X.replace([np.inf, -np.inf], 0, inplace=True)
        X.fillna(0, inplace=True)
        
        y_pred = model.predict(scaler.transform(X))
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        rows += len(chunk)
        print(f"   Processed {rows:,} rows...", end='\r')
    
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total_b = tn + fp
    total_a = tp + fn
    
    recall = tp / total_a * 100 if total_a > 0 else 0
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) * 100
    tnr = tn / total_b * 100 if total_b > 0 else 0
    
    return {
        'rows': rows, 'attack_types': attack_types,
        'benign': total_b, 'attacks': total_a,
        'accuracy': acc, 'recall': recall, 'precision': prec, 'f1': f1, 'tnr': tnr,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def print_results(filename, result, mode_label):
    """Pretty-print results for one file."""
    print(f"\n   --- {mode_label} ---")
    
    # Attack distribution
    print(f"   Labels:")
    for label, count in sorted(result['attack_types'].items(), key=lambda x: -x[1]):
        print(f"     {label:35s} {count:>10,}")
    
    print(f"\n   {'METRIC':<25} {'VALUE':>10}")
    print(f"   {'-'*40}")
    print(f"   {'Total Rows':<25} {result['rows']:>10,}")
    print(f"   {'Total Benign':<25} {result['benign']:>10,}")
    print(f"   {'Total Attacks':<25} {result['attacks']:>10,}")
    print(f"   {'-'*40}")
    print(f"   {'Overall Accuracy':<25} {result['accuracy']:>9.2f}%")
    print(f"   {'Benign Accuracy (TNR)':<25} {result['tnr']:>9.2f}%")
    print(f"   {'Attack Recall (TPR)':<25} {result['recall']:>9.2f}%")
    print(f"   {'Attack Precision':<25} {result['precision']:>9.2f}%")
    print(f"   {'F1-Score':<25} {result['f1']:>9.2f}%")
    print(f"   {'-'*40}")
    print(f"   {'TP':>5} {result['tp']:>10,}  |  {'FP':>5} {result['fp']:>10,}")
    print(f"   {'FN':>5} {result['fn']:>10,}  |  {'TN':>5} {result['tn']:>10,}")


def main():
    print("=" * 70)
    print("  GUARDIAN AI ENGINE - RF TEST: BASELINE vs 5% ADAPTATION")
    print("  Model: base_rf_2017.pkl | Test: CIC-IDS 2018")
    print("=" * 70)
    
    # Pick 3 random files (exclude the 4GB one for speed)
    all_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
    reasonable = [f for f in all_files if f != '02-20-2018.csv']
    
    random.seed(42)
    selected = random.sample(reasonable, min(3, len(reasonable)))
    print(f"\nSelected files: {selected}\n")
    
    baseline_results = []
    adapted_results = []
    
    for fname in selected:
        fpath = os.path.join(DATASET_DIR, fname)
        print(f"\n{'='*70}")
        print(f"  FILE: {fname}")
        print(f"{'='*70}")
        
        # --- TEST 1: BASELINE (no adaptation) ---
        print(f"\n[1/2] BASELINE - No adaptation")
        model_b, scaler_b = joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
        print(f"   Model loaded: {model_b.n_estimators} trees (fresh)")
        
        t0 = time.time()
        res_b = evaluate_file(model_b, scaler_b, fpath)
        res_b['time'] = time.time() - t0
        res_b['file'] = fname
        print_results(fname, res_b, "BASELINE (No Adaptation)")
        baseline_results.append(res_b)
        
        # --- TEST 2: ADAPTED (5% retrain) ---
        print(f"\n\n[2/2] ADAPTED - 5% calibration retrain")
        model_a, scaler_a = joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)
        model_a = adapt_model(model_a, scaler_a, fpath)
        
        t0 = time.time()
        res_a = evaluate_file(model_a, scaler_a, fpath)
        res_a['time'] = time.time() - t0
        res_a['file'] = fname
        print_results(fname, res_a, "ADAPTED (5% Retrain)")
        adapted_results.append(res_a)
        
        # Per-file comparison
        print(f"\n   >>> IMPROVEMENT: Recall {res_b['recall']:.2f}% -> {res_a['recall']:.2f}%  |  F1 {res_b['f1']:.2f}% -> {res_a['f1']:.2f}%")
    
    # ========================
    #   FINAL SUMMARY TABLE
    # ========================
    print(f"\n\n{'='*70}")
    print(f"  FINAL COMPARISON: BASELINE vs 5% ADAPTED")
    print(f"{'='*70}")
    
    header = f"  {'File':<18} | {'Mode':<10} | {'Accuracy':>8} | {'Recall':>8} | {'Precision':>8} | {'F1':>8}"
    print(header)
    print(f"  {'-'*75}")
    
    for b, a in zip(baseline_results, adapted_results):
        print(f"  {b['file']:<18} | {'Baseline':<10} | {b['accuracy']:>7.2f}% | {b['recall']:>7.2f}% | {b['precision']:>7.2f}% | {b['f1']:>7.2f}%")
        print(f"  {'':18} | {'Adapted':<10} | {a['accuracy']:>7.2f}% | {a['recall']:>7.2f}% | {a['precision']:>7.2f}% | {a['f1']:>7.2f}%")
        delta_r = a['recall'] - b['recall']
        delta_f = a['f1'] - b['f1']
        print(f"  {'':18} | {'Delta':<10} |          | {delta_r:>+7.2f}% | {' ':>8} | {delta_f:>+7.2f}%")
        print(f"  {'-'*75}")
    
    # Averages
    avg_b_r = np.mean([r['recall'] for r in baseline_results])
    avg_a_r = np.mean([r['recall'] for r in adapted_results])
    avg_b_f = np.mean([r['f1'] for r in baseline_results])
    avg_a_f = np.mean([r['f1'] for r in adapted_results])
    
    print(f"\n  AVERAGE RECALL:    Baseline {avg_b_r:.2f}%  ->  Adapted {avg_a_r:.2f}%  (delta: {avg_a_r - avg_b_r:+.2f}%)")
    print(f"  AVERAGE F1-SCORE:  Baseline {avg_b_f:.2f}%  ->  Adapted {avg_a_f:.2f}%  (delta: {avg_a_f - avg_b_f:+.2f}%)")


if __name__ == "__main__":
    main()
