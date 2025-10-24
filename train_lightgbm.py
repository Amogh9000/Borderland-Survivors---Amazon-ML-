# train_lightgbm.py
"""
LightGBM training script with:
- fallback from precomputed embeddings -> TF-IDF (if embeddings absent)
- numeric feature extraction (pack count, total_qty, digit_count, word_count)
- 5-fold CV on log(price)
- prints RMSE(log), RMSE(price), MAE(price), SMAPE per fold and CV
- saves OOF preds and models

Expect to find:
- train.csv (must contain: sample_id, catalog_content, price)
- test.csv (must contain: sample_id, catalog_content)   <-- optional but recommended
Optional (preferred):
- text_embeddings_train.npy (shape: [n_train, D])
- text_embeddings_test.npy  (shape: [n_test, D])
- small_features_train.csv  (contains sample_id + numeric features)
- small_features_test.csv
If embeddings/files not present, the script computes TF-IDF features automatically.
"""

import os
import sys
import math
import time
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import re

warnings.filterwarnings("ignore")
RND = 42
np.random.seed(RND)

# ---------- Config ----------
TRAIN_CSV = "Text processing\\train.csv"
TEST_CSV = "Text processing\\test.csv"
EMB_TRAIN_NPY = "Text processing\\gte_train_embeddings.npy"
EMB_TEST_NPY = "Text processing\\gte_test_embeddings.npy"
SMALL_FEAT_TRAIN = "small_features_train.csv"
SMALL_FEAT_TEST = "small_features_test.csv"

OUT_DIR = "models_out"
os.makedirs(OUT_DIR, exist_ok=True)

# LightGBM params (optimized for better performance)
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,  # Reduced for better generalization
    "num_leaves": 255,  # Increased for more complexity
    "max_depth": 15,  # Increased depth
    "min_data_in_leaf": 20,  # Reduced for more splits
    "feature_fraction": 0.8,  # Increased feature usage
    "bagging_fraction": 0.8,  # Increased bagging
    "bagging_freq": 1,  # More frequent bagging
    "lambda_l1": 0.1,  # Reduced L1 regularization
    "lambda_l2": 0.5,  # Reduced L2 regularization
    "min_gain_to_split": 0.005,  # Reduced for more splits
    "verbosity": -1,
    "seed": RND,
    "nthread": 8,
    "force_col_wise": True,  # Better for many features
}
N_SPLITS = 5
NUM_BOOST_ROUND = 8000  # Increased rounds
EARLY_STOPPING = 300  # Increased early stopping patience

# ---------- Utility functions ----------
def smape(y_true, y_pred):
    # avoid zero division: use denominator average of abs values
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom == 0
    denom[mask] = 1.0  # when both are zero, contribution zero
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

def safe_exp(x):
    # Clip to avoid overflow
    return np.exp(np.clip(x, -50, 50))

def extract_small_features(df, text_col="catalog_content"):
    """
    Extract enhanced numeric features:
      - pack_count (IPQ): integer extracted from patterns like 'pack of 4', 'x4', '4 pack', 'pack 4'
      - total_qty_norm: try to compute total quantity in base unit if 'g'/'ml' present else pack_count
      - digit_count: number of digit tokens present
      - word_count: number of words
      - price_indicators: count of price-related keywords
      - brand_indicators: count of brand-related keywords
      - quality_indicators: count of quality-related keywords
      - special_chars: count of special characters
      - uppercase_ratio: ratio of uppercase letters
      - number_ratio: ratio of numeric characters
    """
    pack_counts = []
    total_qty = []
    digit_counts = []
    word_counts = []
    price_indicators = []
    brand_indicators = []
    quality_indicators = []
    special_chars = []
    uppercase_ratios = []
    number_ratios = []

    # Define keyword lists
    price_keywords = ['price', 'cost', 'value', 'worth', 'expensive', 'cheap', 'budget', 'premium', 'sale', 'discount']
    brand_keywords = ['brand', 'branded', 'original', 'authentic', 'genuine', 'official', 'licensed']
    quality_keywords = ['premium', 'quality', 'high-quality', 'best', 'top', 'superior', 'excellent', 'premium']

    for text in df[text_col].fillna("").astype(str):
        t = text.lower()

        # pack_count heuristics
        pack = 1
        # typical patterns
        m = re.search(r"pack(?:\s*of)?\s*(\d+)", t)
        if not m:
            m = re.search(r"(\d+)\s*pack\b", t)
        if not m:
            m = re.search(r"\bx\s*(\d+)\b", t)  # e.g., "2 x 200g"
        if m:
            try:
                pack = int(m.group(1))
            except:
                pack = 1

        # quantity heuristics: find numbers followed by g/ml/kg/l/pcs/pc
        total = None
        qty_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(g|kg|ml|l|pcs?|pc|count|cm|mm)", t)
        if qty_matches:
            # compute approximate total in grams or ml when possible
            total_approx = 0.0
            for val, unit in qty_matches:
                try:
                    v = float(val)
                except:
                    continue
                unit = unit.strip()
                if unit == "kg":
                    v = v * 1000.0
                if unit in ("g", "kg"):
                    total_approx += v
                elif unit in ("ml", "l"):
                    # treat ml/l similar but don't mix with g - just sum
                    if unit == "l":
                        v = v * 1000.0
                    total_approx += v
                elif unit in ("pc", "pcs", "count"):
                    total_approx += v * 1.0
                else:
                    total_approx += v
            total = total_approx * pack
        else:
            # fallback: if just numbers present, try multiply the largest number by pack
            numbers = re.findall(r"(\d+(?:\.\d+)?)", t)
            if numbers:
                try:
                    largest = max([float(x) for x in numbers])
                    total = largest * pack
                except:
                    total = float(pack)

        if total is None:
            total = float(pack)

        digits = len(re.findall(r"\d", t))
        words = len(re.findall(r"\w+", t))
        
        # Enhanced features
        price_count = sum(1 for keyword in price_keywords if keyword in t)
        brand_count = sum(1 for keyword in brand_keywords if keyword in t)
        quality_count = sum(1 for keyword in quality_keywords if keyword in t)
        special_char_count = len(re.findall(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>\/?]", text))
        
        # Calculate ratios
        total_chars = len(text)
        uppercase_ratio = len(re.findall(r"[A-Z]", text)) / max(total_chars, 1)
        number_ratio = digits / max(total_chars, 1)

        pack_counts.append(pack)
        total_qty.append(total)
        digit_counts.append(digits)
        word_counts.append(words)
        price_indicators.append(price_count)
        brand_indicators.append(brand_count)
        quality_indicators.append(quality_count)
        special_chars.append(special_char_count)
        uppercase_ratios.append(uppercase_ratio)
        number_ratios.append(number_ratio)

    out = pd.DataFrame({
        "pack_count": pack_counts,
        "total_qty_norm": total_qty,
        "digit_count": digit_counts,
        "word_count": word_counts,
        "price_indicators": price_indicators,
        "brand_indicators": brand_indicators,
        "quality_indicators": quality_indicators,
        "special_chars": special_chars,
        "uppercase_ratio": uppercase_ratios,
        "number_ratio": number_ratios
    })
    return out

# ---------- Load data ----------
if not Path(TRAIN_CSV).exists():
    print(f"ERROR: {TRAIN_CSV} not found in working directory. Place train.csv here and retry.")
    sys.exit(1)

train_df = pd.read_csv(TRAIN_CSV)
print(f"Loaded train: {train_df.shape}")
has_price = "price" in train_df.columns
if not has_price:
    raise ValueError("train.csv must include a 'price' column for model training.")

# load test if present
test_df = pd.read_csv(TEST_CSV) if Path(TEST_CSV).exists() else None
if test_df is not None:
    print(f"Loaded test: {test_df.shape}")

# ---------- Build features ----------
# 1) Try to load precomputed embeddings and small numeric features
use_embeddings = False
X_train_emb = None
X_test_emb = None
small_train = None
small_test = None

if Path(EMB_TRAIN_NPY).exists():
    try:
        X_train_emb = np.load(EMB_TRAIN_NPY)
        print(f"Loaded embeddings train: {X_train_emb.shape}")
        if test_df is not None and Path(EMB_TEST_NPY).exists():
            X_test_emb = np.load(EMB_TEST_NPY)
            print(f"Loaded embeddings test: {X_test_emb.shape}")
        use_embeddings = True
    except Exception as e:
        print("Failed to load embeddings. Falling back to TF-IDF. Error:", e)
        use_embeddings = False

if Path(SMALL_FEAT_TRAIN).exists():
    small_train = pd.read_csv(SMALL_FEAT_TRAIN)
    print("Loaded small numeric features (train):", small_train.shape)
if test_df is not None and Path(SMALL_FEAT_TEST).exists():
    small_test = pd.read_csv(SMALL_FEAT_TEST)
    print("Loaded small numeric features (test):", small_test.shape)

# 2) If embeddings not present -> create TF-IDF on catalog_content (fallback)
tfidf = None
X_train_tfidf = None
X_test_tfidf = None
tfidf_dim = 5000  # adjustable

if not use_embeddings:
    print("Embeddings not found — computing TF-IDF fallback features.")
    corpus_train = train_df["catalog_content"].fillna("").astype(str).tolist()
    corpus_test = test_df["catalog_content"].fillna("").astype(str).tolist() if test_df is not None else None

    tfidf = TfidfVectorizer(max_features=tfidf_dim, ngram_range=(1,2), min_df=3, analyzer="word")
    X_train_tfidf = tfidf.fit_transform(corpus_train)
    print("TF-IDF train shape:", X_train_tfidf.shape)
    if test_df is not None:
        X_test_tfidf = tfidf.transform(corpus_test)
        print("TF-IDF test shape:", X_test_tfidf.shape)

# 3) If small numeric features not provided, extract heuristics from text
if small_train is None:
    print("Small numeric features not provided — extracting heuristics from text (train).")
    small_train = extract_small_features(train_df, text_col="catalog_content")
if test_df is not None and small_test is None:
    print("Small numeric features not provided — extracting heuristics from text (test).")
    small_test = extract_small_features(test_df, text_col="catalog_content")

# 4) Combine features: embeddings (or tfidf) + small numeric features
def stack_features(emb, tfidf_sparse, small_df):
    """
    Return dense numpy array. If tfidf_sparse present, convert to dense in chunks (careful).
    """
    if emb is not None:
        X = emb
    else:
        # convert sparse TF-IDF to dense (may be big) — if huge, consider using sparse LightGBM input
        X = tfidf_sparse.toarray()
    # scale small numeric features
    scaler = StandardScaler()
    small_scaled = scaler.fit_transform(small_df.values)
    # save scaler for later application to test
    return np.hstack([X, small_scaled]), scaler

# Stack train
if use_embeddings:
    X_train_raw = np.hstack([X_train_emb, small_train.values])
    train_scaler = None  # embeddings already in their own space; small features not scaled
    # scale small part only
    # We'll scale the last 4 columns to zero mean unit var
    small_part = X_train_emb.shape[1]
    scaler = StandardScaler()
    X_train_raw[:, small_part:] = scaler.fit_transform(X_train_raw[:, small_part:])
    train_scaler = scaler
else:
    X_train_raw, train_small_scaler = stack_features(None, X_train_tfidf, small_train)

X_test_raw = None
if test_df is not None:
    if use_embeddings:
        if X_test_emb is None:
            raise ValueError("Embeddings for test not found but train embeddings provided. Please provide test embeddings.")
        X_test_raw = np.hstack([X_test_emb, small_test.values])
        # apply same scaler to last columns
        X_test_raw[:, X_train_emb.shape[1]:] = train_scaler.transform(X_test_raw[:, X_train_emb.shape[1]:])
    else:
        # transform small_test using train_small_scaler
        X_test_dense = X_test_tfidf.toarray()
        small_scaled = train_small_scaler.transform(small_test.values)
        X_test_raw = np.hstack([X_test_dense, small_scaled])

print("Final feature shapes -> train:", X_train_raw.shape, " test:", X_test_raw.shape if X_test_raw is not None else None)

# ---------- Prepare target ----------
y = train_df["price"].values.astype(float)
# remove non-positive prices
mask_pos = y > 0
if not np.all(mask_pos):
    print(f"Warning: {np.sum(~mask_pos)} rows with non-positive price found and will be removed.")
    X_train_raw = X_train_raw[mask_pos]
    y = y[mask_pos]
    train_df = train_df[mask_pos].reset_index(drop=True)

# Additional preprocessing: remove extreme outliers
price_q1, price_q3 = np.percentile(y, [25, 75])
iqr = price_q3 - price_q1
lower_bound = price_q1 - 3 * iqr
upper_bound = price_q3 + 3 * iqr
outlier_mask = (y >= lower_bound) & (y <= upper_bound)

if not np.all(outlier_mask):
    print(f"Warning: {np.sum(~outlier_mask)} extreme outliers found and will be removed.")
    X_train_raw = X_train_raw[outlier_mask]
    y = y[outlier_mask]
    train_df = train_df[outlier_mask].reset_index(drop=True)

y_log = np.log(y)

# ---------- CV training ----------
# Use StratifiedKFold for better distribution of price ranges
from sklearn.model_selection import StratifiedKFold

# Create price bins for stratification
price_bins = pd.cut(y, bins=10, labels=False)
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RND)

oof_pred_log = np.zeros(len(y_log))
oof_pred_price = np.zeros(len(y_log))
test_preds_log = np.zeros(X_test_raw.shape[0]) if X_test_raw is not None else None

fold = 0
metrics = []

for train_idx, valid_idx in kf.split(X_train_raw, price_bins):
    fold += 1
    print(f"\n--- Fold {fold}/{N_SPLITS} ---")
    X_tr, X_val = X_train_raw[train_idx], X_train_raw[valid_idx]
    y_tr, y_val = y_log[train_idx], y_log[valid_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    # Train multiple models with different random seeds for ensemble
    models = []
    for seed_offset in range(3):  # Train 3 models per fold
        params = lgb_params.copy()
        params['seed'] = RND + seed_offset
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(200)]
        )
        models.append(model)
    
    # Use the first model as primary (for compatibility)
    model = models[0]

    # save model
    model_path = os.path.join(OUT_DIR, f"lgbm_fold{fold}.txt")
    model.save_model(model_path)
    print("Saved model to", model_path)

    # Ensemble prediction
    pred_val_log = np.mean([m.predict(X_val, num_iteration=m.best_iteration) for m in models], axis=0)
    pred_tr_log = np.mean([m.predict(X_tr, num_iteration=m.best_iteration) for m in models], axis=0)

    oof_pred_log[valid_idx] = pred_val_log
    oof_pred_price[valid_idx] = safe_exp(pred_val_log)

    # Compute metrics for this fold
    rmse_log = math.sqrt(mean_squared_error(y_val, pred_val_log))
    rmse_price = math.sqrt(mean_squared_error(np.exp(y_val), safe_exp(pred_val_log)))
    mae_price = mean_absolute_error(np.exp(y_val), safe_exp(pred_val_log))
    sm = smape(np.exp(y_val), safe_exp(pred_val_log))

    metrics.append({"fold": fold, "rmse_log": rmse_log, "rmse_price": rmse_price, "mae_price": mae_price, "smape": sm})
    print(f"Fold {fold} metrics -> RMSE(log): {rmse_log:.6f} | RMSE(price): {rmse_price:.4f} | MAE(price): {mae_price:.4f} | SMAPE: {sm:.2f}%")

    # test predictions (ensemble)
    if X_test_raw is not None:
        test_preds_log += np.mean([m.predict(X_test_raw, num_iteration=m.best_iteration) for m in models], axis=0) / N_SPLITS

# ---------- CV summary ----------
df_metrics = pd.DataFrame(metrics)
print("\n=== CV Summary ===")
print(df_metrics)
print("CV mean RMSE(log):", df_metrics["rmse_log"].mean())
print("CV mean RMSE(price):", df_metrics["rmse_price"].mean())
print("CV mean MAE(price):", df_metrics["mae_price"].mean())
print("CV mean SMAPE (%):", df_metrics["smape"].mean())

# save OOF preds
train_df["pred_log_oof"] = oof_pred_log
train_df["pred_price_oof"] = oof_pred_price
oof_path = os.path.join(OUT_DIR, "oof_predictions_train.csv")
train_df[["sample_id", "price", "pred_log_oof", "pred_price_oof"]].to_csv(oof_path, index=False)
print("Saved OOF predictions to", oof_path)

# save test predictions
if X_test_raw is not None:
    test_df["pred_log"] = test_preds_log
    test_df["pred_price"] = safe_exp(test_preds_log)
    sub_path = os.path.join(OUT_DIR, "submission.csv")
    test_df[["sample_id", "pred_price"]].rename(columns={"pred_price": "price"}).to_csv(sub_path, index=False)
    print("Saved submission to", sub_path)

print("Done.")
