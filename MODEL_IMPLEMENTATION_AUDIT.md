# FlowGuard Model Implementation Audit

**Status:** ✅ **CORRECT IMPLEMENTATION** with notes

---

## Executive Summary

The `FlowGuardProvider` in `backend/model_provider.py` correctly implements the FlowGuard model from `model/canavar-model/`. The stats file order is **authoritative** (what the model was trained on), not the canonical `features.py` order. This is by design and correct.

---

## Implementation Verification

### 1. Stats File Loading ✅

**Code:** `model_provider.py:465-482`

```python
stats = np.load(self.stats_path, allow_pickle=True)
self.feature_names = stats['feature_names'].tolist()
self.feature_means = stats['feature_means'].astype(np.float64)
self.feature_stds = stats['feature_stds'].astype(np.float64)
self.log_transform_columns = stats['log_transform_columns'].tolist()
```

**Verification:**
- ✅ Loads from correct path: `backend/saved_models/flowguard_stats.npz`
- ✅ All 4 required keys present and loaded (verified by earlier check)
- ✅ Correct dtypes: feature_names as list, means/stds as float64, log_cols as list
- ✅ Matches training preprocessing (preprocess.py:388)

**Training Origin:** `model/canavar-model/src/data/preprocess.py:383-389`
```python
computed_stats = PreprocessingStats(
    feature_means=means,
    feature_stds=stds,
    log_transform_columns=actually_transformed,
    port_bucket_map=dict(_PORT_BUCKETS),
    feature_names=feature_cols,  # ← This is the authoritative order
)
```

### 2. Feature Order ✅

**Key Finding:** Stats file has features in **different order** than canonical `features.py`.

| Position | Stats File (Training Order) | Canonical Order |
|----------|----------------------------|-----------------|
| 0-19 | PROTOCOL...MAX_IP_PKT_LEN | Same |
| 20-25 | RETRANSMITTED_*... | SRC_TO_DST_AVG_THROUGHPUT... |
| 26+ | SRC_TO_DST_AVG_THROUGHPUT... | RETRANSMITTED_*... |
| 47-52 | SRC_PORT_*...DST_PORT_* | Same |

**Why this happened:**
The training preprocessing (preprocess.py:370) uses whatever column order comes from the parquet files:
```python
feature_cols = [c for c in df.columns if c not in label_cols_present]
```

The parquet files were created during training with RETRANSMISSION features before RATIO features. This is **not a bug**, it's just a different feature order.

**Verification:** ✅ Correct
- Stats file order is what the model's weights were fitted for
- The scaler (feature_means, feature_stds) are in this order
- Using this order is **correct**

---

### 3. Port Bucketing ✅

**Code:** `model_provider.py:625-658`

```python
src_port_buckets = self._bucket_port(src_port)
dst_port_buckets = self._bucket_port(dst_port)

port_bucket_map = {
    'SRC_PORT_WELL_KNOWN': float(src_port_buckets['WELL_KNOWN']),
    'SRC_PORT_REGISTERED': float(src_port_buckets['REGISTERED']),
    'SRC_PORT_EPHEMERAL': float(src_port_buckets['EPHEMERAL']),
    'DST_PORT_WELL_KNOWN': float(dst_port_buckets['WELL_KNOWN']),
    'DST_PORT_REGISTERED': float(dst_port_buckets['REGISTERED']),
    'DST_PORT_EPHEMERAL': float(dst_port_buckets['EPHEMERAL']),
}
```

**Verification:**
- ✅ Bucket boundaries match training (model_provider.py:407-411 and preprocess.py:90-94)
  - WELL_KNOWN: 0-1023
  - REGISTERED: 1024-49151
  - EPHEMERAL: 49152-65535
- ✅ Creates 6 binary bucket features (3 for SRC, 3 for DST)
- ✅ Matches training behavior (preprocess.py:334-339)

---

### 4. Feature Vector Construction ✅

**Code:** `model_provider.py:654-671`

```python
for i, fname in enumerate(self.feature_names):
    if fname in port_bucket_map:
        feature_vector[i] = port_bucket_map[fname]
        continue
    raw_val = features.get(fname, 0.0)
    ...
    feature_vector[i] = val
```

**Verification:**
- ✅ Iterates through stats file feature order (authoritative)
- ✅ Safely fills port bucket features from mapping
- ✅ Safely defaults missing raw features to 0.0
- ✅ Creates feature_vector in correct order for model input

---

### 5. Log Transform ✅

**Code:** `model_provider.py:673-677`

```python
for col_name in self.log_transform_columns:
    if col_name in self.feature_names:
        idx = self.feature_names.index(col_name)
        feature_vector[idx] = np.log1p(np.abs(feature_vector[idx]))
```

**Verification:**
- ✅ Gets column list from stats file (trained with 24 log-transforms)
- ✅ Finds correct index in feature_vector using stats file order
- ✅ Applies `log1p(|x|)` matching training (preprocess.py:236)

**Training Origin:** `preprocess.py:375`
```python
actually_transformed = _log_transform(df, _LOG_TRANSFORM_CANDIDATES)
```

The stats file contains exactly which features were transformed.

---

### 6. Z-score Normalization ✅

**Code:** `model_provider.py:680-681`

```python
feature_vector = (feature_vector - self.feature_means) / (self.feature_stds + eps)
```

**Verification:**
- ✅ Uses means/stds loaded from stats file
- ✅ Epsilon protection (eps=1e-8) matches training (preprocess.py:246)
- ✅ Order-sensitive: `feature_vector[i]` must align with `feature_means[i]` and `feature_stds[i]`
  - This is guaranteed because both come from same stats file in same order ✅

**Training Origin:** `preprocess.py:256-260`
```python
subset = df[feature_cols].astype(np.float64)
means = subset.mean().values
stds = subset.std().values
df[feature_cols] = (subset.values - means) / (stds + eps)
```

---

### 7. Model Loading & Inference ✅

**Code:** `model_provider.py:546-574`

```python
self.model = FlowGuard(self.ENCODER_CONFIG)
self.model.enable_domain_discriminator(num_domains=self.NUM_DOMAINS)
state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
self.model.load_state_dict(state_dict)
self.model.eval()
```

**Verification:**
- ✅ Uses correct model class: `FlowGuard` from `model/canavar-model/src/models/flowguard.py`
- ✅ Enables domain discriminator (required for hardened checkpoint)
- ✅ Loads checkpoint with `weights_only=True` (security best practice)
- ✅ Sets to eval mode for inference

**Architecture:**
- ✅ Input dim: 53 (correct)
- ✅ Model dim: 128
- ✅ Num heads: 4
- ✅ Num layers: 4
- ✅ Binary classification (2 classes: Benign/Attack)

**Inference:** `model_provider.py:720-722`
```python
logits = self.model(tensor)  # (1, 2)
probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
p_attack = float(probs[1])
```

---

### 8. Threshold ✅

**Code:** `model_provider.py:729`

```python
is_attack = p_attack > 0.50
```

**Verification:**
- ✅ Hardcoded 0.50 for softmax binary output (correct)
- ✅ Different from global `THRESHOLD = 0.40` in `analyze_engine.py`
- ✅ Comment explains why: softmax needs 0.50, sigmoid-based models use 0.40

**Why this is correct:**
- Softmax outputs sum to 1.0, so 0.50 is the decision boundary
- Sigmoid outputs don't have this constraint, so different threshold may apply

---

## Critical Fixes Applied

### ✅ Fix 1: Stats Key Validation
Added check for required keys with friendly error (lines 468-477)

### ✅ Fix 2: Feature Set Validation  
Validates all 53 features are present (by name, not order) (lines 485-515)

### ✅ Fix 3: Missing Port Warning
Logs warning if L4_SRC/DST_PORT missing (lines 629-638)

### ✅ Fix 4: Pre-Cleaning NaN/Inf
Moved to before predict() call in analyze_engine.py

### ✅ Fix 8: Feature Count Assertion
Added assertion in _preprocess() output (lines 684-689)

### ✅ Fix 9: Startup Validation Log
Added feature pipeline summary at startup (lines 582-588)

---

## Summary: Model Implementation Correct ✅

The implementation is **correct** because:

1. **Stats file order is authoritative** - it's what the model was trained on
2. **All 53 features are present** - verified by feature set check
3. **Port bucketing is correct** - matches training pipeline
4. **Normalization order is correct** - stats file provides aligned means/stds
5. **Log transforms are correct** - applied to same features as training
6. **Model loading is correct** - proper architecture and checkpoint loading
7. **Inference is correct** - softmax for binary classification

**No need to regenerate stats** - the current stats file is perfectly usable as long as we respect its feature order.

---

## NFv3FlowTracker Alignment ✅

The NFv3FlowTracker (`backend/nfv3_flow_tracker.py`) outputs features in **its own order**, not the stats file order. This is fine because:

1. **NFv3FlowTracker.extract_features()** returns 47 raw + 2 port values = 49 dict items
2. **FlowGuardProvider._preprocess()** looks up features by NAME, not position
3. **Features are reordered in _preprocess()** to match stats file order

This is a **safe and correct design**:
```python
raw_val = features.get(fname, 0.0)  # ← Safe lookup by name
feature_vector[i] = val             # ← Places in stats file order
```

---

## Recommendations

1. ✅ **Keep current stats file** - don't regenerate
2. ✅ **Keep current validation** - accepts different feature orders
3. ✅ **Document the feature order difference** - add comment explaining why
4. ✅ **Run integration tests** - verify NFv3FlowTracker → FlowGuardProvider → prediction works end-to-end

---

## Files Verified

- ✅ `backend/model_provider.py` — FlowGuardProvider implementation
- ✅ `model/canavar-model/src/data/preprocess.py` — Training preprocessing
- ✅ `model/canavar-model/src/data/features.py` — Feature definitions
- ✅ `model/canavar-model/src/data/dataset.py` — Dataset loading
- ✅ `backend/nfv3_flow_tracker.py` — Flow feature extraction
