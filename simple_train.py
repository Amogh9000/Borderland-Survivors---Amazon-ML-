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

print("Loading data...")
sys.stdout.flush()
train_df = pd.read_csv(TRAIN_CSV)
print(f"Loaded train: {train_df.shape}")
sys.stdout.flush()

test_df = pd.read_csv(TEST_CSV)
print(f"Loaded test: {test_df.shape}")
sys.stdout.flush()

# Load embeddings
X_train_emb = np.load(EMB_TRAIN_NPY)
X_test_emb = np.load(EMB_TEST_NPY)
print(f"Loaded embeddings train: {X_train_emb.shape}")
print(f"Loaded embeddings test: {X_test_emb.shape}")

# Extract small features
def extract_small_features(df, text_col="catalog_content"):
    pack_counts = []
    total_qty = []
    digit_counts = []
    word_counts = []

    for text in df[text_col].fillna("").astype(str):
        t = text.lower()
        
        # pack_count heuristics
        pack = 1
        m = re.search(r"pack(?:\s*of)?\s*(\d+)", t)
        if not m:
            m = re.search(r"(\d+)\s*pack\b", t)
        if not m:
            m = re.search(r"\bx\s*(\d+)\b", t)
        if m:
            try:
                pack = int(m.group(1))
            except:
                pack = 1

        # quantity heuristics
        total = None
        qty_matches = re.findall(r"(\d+(?:\.\d+)?)\s*(g|kg|ml|l|pcs?|pc|count|cm|mm)", t)
        if qty_matches:
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
                    if unit == "l":
                        v = v * 1000.0
                    total_approx += v
                elif unit in ("pc", "pcs", "count"):
                    total_approx += v * 1.0
                else:
                    total_approx += v
            total = total_approx * pack
        else:
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

        pack_counts.append(pack)
        total_qty.append(total)
        digit_counts.append(digits)
        word_counts.append(words)

    out = pd.DataFrame({
        "pack_count": pack_counts,
        "total_qty_norm": total_qty,
        "digit_count": digit_counts,
        "word_count": word_counts
    })
    return out

print("Extracting small features...")
small_train = extract_small_features(train_df)
small_test = extract_small_features(test_df)

# Combine features
X_train_raw = np.hstack([X_train_emb, small_train.values])
X_test_raw = np.hstack([X_test_emb, small_test.values])

# Scale small features
scaler = StandardScaler()
small_part = X_train_emb.shape[1]
X_train_raw[:, small_part:] = scaler.fit_transform(X_train_raw[:, small_part:])
X_test_raw[:, small_part:] = scaler.transform(X_test_raw[:, small_part:])

print(f"Final feature shapes -> train: {X_train_raw.shape}, test: {X_test_raw.shape}")

# Prepare target
y = train_df["price"].values.astype(float)
mask_pos = y > 0
if not np.all(mask_pos):
    print(f"Warning: {np.sum(~mask_pos)} rows with non-positive price found and will be removed.")
    X_train_raw = X_train_raw[mask_pos]
    y = y[mask_pos]
    train_df = train_df[mask_pos].reset_index(drop=True)

y_log = np.log(y)

# SMAPE function
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom == 0
    denom[mask] = 1.0
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

# LightGBM params
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 128,
    "max_depth": 12,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 1.0,
    "min_gain_to_split": 0.01,
    "verbosity": -1,
    "seed": RND,
    "nthread": 8,
}

# CV training
kf = KFold(n_splits=5, shuffle=True, random_state=RND)
oof_pred_log = np.zeros(len(y_log))
oof_pred_price = np.zeros(len(y_log))
test_preds_log = np.zeros(X_test_raw.shape[0])

fold = 0
metrics = []

print("Starting cross-validation...")
for train_idx, valid_idx in kf.split(X_train_raw):
    fold += 1
    print(f"\n--- Fold {fold}/5 ---")
    X_tr, X_val = X_train_raw[train_idx], X_train_raw[valid_idx]
    y_tr, y_val = y_log[train_idx], y_log[valid_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=1000,  # Reduced for faster execution
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )

    # Predict
    pred_val_log = model.predict(X_val, num_iteration=model.best_iteration)
    pred_tr_log = model.predict(X_tr, num_iteration=model.best_iteration)

    oof_pred_log[valid_idx] = pred_val_log
    oof_pred_price[valid_idx] = np.exp(pred_val_log)

    # Compute metrics for this fold
    rmse_log = math.sqrt(mean_squared_error(y_val, pred_val_log))
    rmse_price = math.sqrt(mean_squared_error(np.exp(y_val), np.exp(pred_val_log)))
    mae_price = mean_absolute_error(np.exp(y_val), np.exp(pred_val_log))
    sm = smape(np.exp(y_val), np.exp(pred_val_log))

    metrics.append({"fold": fold, "rmse_log": rmse_log, "rmse_price": rmse_price, "mae_price": mae_price, "smape": sm})
    print(f"Fold {fold} metrics -> RMSE(log): {rmse_log:.6f} | RMSE(price): {rmse_price:.4f} | MAE(price): {mae_price:.4f} | SMAPE: {sm:.2f}%")

    # Test predictions
    test_preds_log += model.predict(X_test_raw, num_iteration=model.best_iteration) / 5

# CV summary
df_metrics = pd.DataFrame(metrics)
print("\n=== CV Summary ===")
print(df_metrics)
print("CV mean RMSE(log):", df_metrics["rmse_log"].mean())
print("CV mean RMSE(price):", df_metrics["rmse_price"].mean())
print("CV mean MAE(price):", df_metrics["mae_price"].mean())
print("CV mean SMAPE (%):", df_metrics["smape"].mean())

print(f"\nFINAL SMAPE VALUE: {df_metrics['smape'].mean():.4f}%")
