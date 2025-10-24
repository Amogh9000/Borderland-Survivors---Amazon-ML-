import os
import sys
import pandas as pd
import numpy as np

print("Starting debug script...")
sys.stdout.flush()

# Check if files exist
TRAIN_CSV = "Text processing\\train.csv"
TEST_CSV = "Text processing\\test.csv"
EMB_TRAIN_NPY = "Text processing\\gte_train_embeddings.npy"
EMB_TEST_NPY = "Text processing\\gte_test_embeddings.npy"

print(f"Train CSV exists: {os.path.exists(TRAIN_CSV)}")
print(f"Test CSV exists: {os.path.exists(TEST_CSV)}")
print(f"Train embeddings exist: {os.path.exists(EMB_TRAIN_NPY)}")
print(f"Test embeddings exist: {os.path.exists(EMB_TEST_NPY)}")

try:
    print("Loading train data...")
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Loaded train: {train_df.shape}")
    
    print("Loading test data...")
    test_df = pd.read_csv(TEST_CSV)
    print(f"Loaded test: {test_df.shape}")
    
    print("Loading train embeddings...")
    train_emb = np.load(EMB_TRAIN_NPY)
    print(f"Loaded train embeddings: {train_emb.shape}")
    
    print("Loading test embeddings...")
    test_emb = np.load(EMB_TEST_NPY)
    print(f"Loaded test embeddings: {test_emb.shape}")
    
    print("All data loaded successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("Debug script completed.")
