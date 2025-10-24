# 0) Imports
import re
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1) Load base data
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

# 2) Load precomputed GTE embeddings
X_train_text = np.load("gte_train_embeddings.npy", mmap_mode="r")
X_test_text  = np.load("gte_test_embeddings.npy",  mmap_mode="r")

# 3) Recompute engineered text features (or load CSVs if you saved them)
def preprocess_series(s: pd.Series) -> pd.Series:
    s = s.str.replace(r"^Item Name:\s*", "", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s.str.lower()

train_texts = preprocess_series(train["catalog_content"])
test_texts  = preprocess_series(test["catalog_content"])

def extract_ipq(s: pd.Series) -> pd.Series:
    pat = re.compile(r"(?:pack of|x)\s*(\d+)|(\d+)\s*(?:pack|count|pcs|pieces|ct)", flags=re.I)
    def f(t):
        m = pat.search(t)
        if not m: return 1
        for g in m.groups():
            if g and g.isdigit(): return int(g)
        return 1
    return s.apply(f).astype("int32")

def extract_units_total_ml_g_count(s: pd.Series) -> pd.Series:
    vol_ml = {"floz":29.5735, "oz":29.5735, "ml":1.0, "l":1000.0}
    mass_g = {"g":1.0, "kg":1000.0, "lb":453.592}
    cnt = {"count":1.0, "ct":1.0}
    rx = re.compile(r"(\d+(?:\.\d+)?)\s*(fl\s*oz|oz|ml|l|g|kg|lb|count|ct)", flags=re.I)
    def f(t):
        total = 0.0
        for val, unit in rx.findall(t):
            u = unit.lower().replace(" ", "")
            v = float(val)
            if u in vol_ml: total += v * vol_ml[u]
            elif u in mass_g: total += v * mass_g[u]
            elif u in cnt: total += v
        return total
    return s.apply(f).astype("float32")

train_feats = pd.DataFrame({
    "ipq": extract_ipq(train_texts),
    "qty_total": extract_units_total_ml_g_count(train_texts),
    "num_digits": train_texts.str.count(r"\d").astype("int16"),
    "len_tokens": train_texts.str.split().str.len().astype("int16"),
})
test_feats = pd.DataFrame({
    "ipq": extract_ipq(test_texts),
    "qty_total": extract_units_total_ml_g_count(test_texts),
    "num_digits": test_texts.str.count(r"\d").astype("int16"),
    "len_tokens": test_texts.str.split().str.len().astype("int16"),
})

# 4) Assemble matrices
num_cols = ["ipq","qty_total","num_digits","len_tokens"]
X_train_num = train_feats[num_cols].to_numpy()
X_test_num  = test_feats[num_cols].to_numpy()

X_train = np.hstack([X_train_text, X_train_num])
X_test  = np.hstack([X_test_text,  X_test_num])

y = np.log1p(train["price"].values)

# 5) Fast CV + training
N_FOLDS, N_EST, LR = 5, 1000, 0.05  # quick run; later use (5, 1000, 0.05)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof = np.zeros(len(train))
pred = np.zeros(len(test))

for tr, va in kf.split(X_train):
    model = LGBMRegressor(
        n_estimators=N_EST,
        learning_rate=LR,
        num_leaves=96,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.2,
        random_state=42,
        metric="rmse",
        force_col_wise=True
    )
    model.fit(
        X_train[tr], y[tr],
        eval_set=[(X_train[va], y[va])],
        callbacks=[early_stopping(100), log_evaluation(0)]
    )
    oof[va] = model.predict(X_train[va])
    pred += model.predict(X_test) / N_FOLDS

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = numerator / denominator
    diff[denominator == 0] = 0  # Handle division by zero
    return 100 * np.mean(diff)  # Return as percentage

# 6) Metrics and submission
rmse_log = np.sqrt(mean_squared_error(y, oof))
oof_price = np.expm1(oof)
true_price = train["price"].values
rmse_price = np.sqrt(mean_squared_error(true_price, oof_price))
mae_price  = mean_absolute_error(true_price, oof_price)
smape_score = smape(true_price, oof_price)
print(f"CV RMSE (log): {rmse_log:.6f} | RMSE(price): {rmse_price:.2f} | MAE(price): {mae_price:.2f} | SMAPE: {smape_score:.2f}%")

submission = pd.DataFrame({"sample_id": test["sample_id"], "price": np.expm1(pred)})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv", submission.shape)

