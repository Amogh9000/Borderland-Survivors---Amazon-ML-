import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom == 0
    denom[mask] = 1.0
    return 100.0 * np.mean(np.abs(y_pred - y_true) / denom)

print("Loading data...")
train_df = pd.read_csv("Text processing\\train.csv")
X_train_emb = np.load("Text processing\\gte_train_embeddings.npy")

print(f"Data shapes: train={train_df.shape}, embeddings={X_train_emb.shape}")

# Use only first 10000 samples for quick calculation
X = X_train_emb[:10000]
y = train_df["price"].values[:10000]

# Remove non-positive prices
mask = y > 0
X = X[mask]
y = y[mask]
y_log = np.log(y)

print(f"After filtering: X={X.shape}, y={y.shape}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Train simple model
dtrain = lgb.Dataset(X_train, label=y_train)
dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 20,
    "verbosity": -1,
    "seed": 42,
}

print("Training model...")
model = lgb.train(
    params,
    dtrain,
    num_boost_round=100,
    valid_sets=[dtrain, dval],
    valid_names=["train", "valid"],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
)

# Predict
pred_log = model.predict(X_val, num_iteration=model.best_iteration)
pred_price = np.exp(pred_log)
y_val_price = np.exp(y_val)

# Calculate metrics
rmse_log = math.sqrt(mean_squared_error(y_val, pred_log))
rmse_price = math.sqrt(mean_squared_error(y_val_price, pred_price))
mae_price = mean_absolute_error(y_val_price, pred_price)
smape_val = smape(y_val_price, pred_price)

print(f"\nResults:")
print(f"RMSE(log): {rmse_log:.6f}")
print(f"RMSE(price): {rmse_price:.4f}")
print(f"MAE(price): {mae_price:.4f}")
print(f"SMAPE: {smape_val:.4f}%")

print(f"\nFINAL SMAPE VALUE: {smape_val:.4f}%")
