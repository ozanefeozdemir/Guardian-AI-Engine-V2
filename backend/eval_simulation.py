"""Full-dataset confusion matrix for the simulation pipeline.

Replays the entire NF-UNSW-NB15-v3 CSV through the SAME decision path as
run_simulation (FlowGuardProvider preprocess + argmax, Label/Attack dropped from
features), but batched for speed. Ground truth comes from the Label/Attack
columns, which the model never sees.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np
import pandas as pd
import torch
from collections import Counter
from model_provider import get_model_provider

DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "canavar-model", "data", "raw", "NF-UNSW-NB15-v3.csv",
)
CHUNK = 50000
BATCH = 8192


def main():
    p = get_model_provider("flowguard")
    p.load()
    p._ensure_canavar_on_path()
    from src.data.preprocess import (
        _bucket_ports, _handle_inf_and_nan, _log_transform,
        _zscore_normalize, _IDENTITY_COLUMNS, _TIMESTAMP_COLUMNS,
    )
    fc = p.stats.feature_names

    def build_batch(df):
        df = df.copy()
        df.columns = df.columns.str.strip()
        if 'L4_SRC_PORT' in df.columns:
            df = pd.concat([df, _bucket_ports(df['L4_SRC_PORT'], 'SRC_PORT')], axis=1)
        if 'L4_DST_PORT' in df.columns:
            df = pd.concat([df, _bucket_ports(df['L4_DST_PORT'], 'DST_PORT')], axis=1)
        df = df.drop(columns=[c for c in _IDENTITY_COLUMNS + _TIMESTAMP_COLUMNS if c in df.columns])
        df = _handle_inf_and_nan(df)
        for c in fc:
            if c not in df.columns:
                df[c] = 0.0
        _log_transform(df, p.stats.log_transform_columns)
        df, _, _ = _zscore_normalize(df, fc, means=p.stats.feature_means, stds=p.stats.feature_stds)
        return df[fc].values.astype(np.float32)

    # Confusion matrix counts: rows=actual, cols=pred  [[TN, FP], [FN, TP]]
    TN = FP = FN = TP = 0
    per_type = {}      # attack_type -> [caught, missed]
    total = 0
    t0 = time.time()

    for chunk in pd.read_csv(DATASET, chunksize=CHUNK):
        chunk.columns = chunk.columns.str.strip()
        # Ground truth (model never sees these)
        if 'Attack' in chunk.columns:
            attack_str = chunk['Attack'].astype(str).fillna('Benign')
            y_true = (attack_str.str.lower() != 'benign').to_numpy().astype(int)
        else:
            y_true = chunk['Label'].to_numpy().astype(int)
            attack_str = pd.Series(np.where(y_true == 1, 'Attack', 'Benign'), index=chunk.index)

        feat = chunk.drop(columns=[c for c in ('Label', 'Attack') if c in chunk.columns])
        X = build_batch(feat)

        preds = np.empty(len(X), dtype=np.int64)
        with torch.no_grad():
            for i in range(0, len(X), BATCH):
                xb = torch.from_numpy(X[i:i+BATCH])
                preds[i:i+BATCH] = p.model(xb).argmax(dim=1).numpy()

        TP += int(((y_true == 1) & (preds == 1)).sum())
        TN += int(((y_true == 0) & (preds == 0)).sum())
        FP += int(((y_true == 0) & (preds == 1)).sum())
        FN += int(((y_true == 1) & (preds == 0)).sum())

        # per attack-type recall
        atk_mask = y_true == 1
        for t, hit in zip(attack_str[atk_mask], preds[atk_mask] == 1):
            d = per_type.setdefault(t, [0, 0])
            d[0 if hit else 1] += 1

        total += len(chunk)
        print(f"[eval] {total:>9,} satir | TP={TP} TN={TN} FP={FP} FN={FN} "
              f"| {total/(time.time()-t0):,.0f} satir/s", flush=True)

    # --- Report ---
    acc = (TP + TN) / total
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    spec = TN / (TN + FP) if (TN + FP) else 0.0

    lines = []
    lines.append("=" * 60)
    lines.append("CONFUSION MATRIX — NF-UNSW-NB15-v3 (tum dataset)")
    lines.append(f"Toplam akis: {total:,}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("                   Pred:Benign     Pred:Attack")
    lines.append(f"  Actual:Benign  {TN:>14,}  {FP:>14,}")
    lines.append(f"  Actual:Attack  {FN:>14,}  {TP:>14,}")
    lines.append("")
    lines.append(f"  Accuracy   : {acc:.4%}")
    lines.append(f"  Precision  : {prec:.4%}")
    lines.append(f"  Recall/TPR : {rec:.4%}")
    lines.append(f"  Specificity: {spec:.4%}")
    lines.append(f"  F1-score   : {f1:.4%}")
    lines.append(f"  FP (false alarm): {FP:,}   FN (missed attack): {FN:,}")
    lines.append("")
    lines.append("Saldiri turune gore yakalama (recall):")
    for t in sorted(per_type, key=lambda k: -(per_type[k][0] + per_type[k][1])):
        caught, missed = per_type[t]
        tot = caught + missed
        lines.append(f"  {t:<18} {caught:>8,}/{tot:<8,}  ({caught/tot:.2%})")
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(os.path.dirname(__file__), "..", "confusion_matrix.txt"), "w") as f:
        f.write(report + "\n")
    print(f"\n[eval] Toplam sure: {(time.time()-t0)/60:.1f} dk")


if __name__ == "__main__":
    main()
