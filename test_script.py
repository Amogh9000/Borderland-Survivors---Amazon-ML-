import pandas as pd
import numpy as np
import os

print("Testing file loading...")

# Test loading the datasets
train_df = pd.read_csv("Text processing\\train.csv")
print(f"Loaded train: {train_df.shape}")

test_df = pd.read_csv("Text processing\\test.csv")
print(f"Loaded test: {test_df.shape}")

# Test loading embeddings
train_emb = np.load("Text processing\\gte_train_embeddings.npy")
print(f"Loaded train embeddings: {train_emb.shape}")

test_emb = np.load("Text processing\\gte_test_embeddings.npy")
print(f"Loaded test embeddings: {test_emb.shape}")

print("All files loaded successfully!")

