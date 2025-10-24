# Amazon ML Challenge 2025

## Smart Product Pricing Challenge

A machine learning solution for predicting optimal product prices in e-commerce using advanced NLP techniques and gradient boosting models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Solution Approach](#solution-approach)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Collaborators](#collaborators)

---

## 🎯 Overview

This project tackles the challenge of determining optimal price points for e-commerce products by analyzing product details and predicting accurate prices. The solution leverages state-of-the-art text embeddings (GTE - General Text Embeddings) combined with engineered features and LightGBM regression models.

---

## 📝 Problem Statement

In e-commerce, determining the optimal price point for products is crucial for marketplace success and customer satisfaction. This challenge requires developing an ML solution that:

- Analyzes product details holistically (title, description, quantity)
- Predicts optimal product pricing
- Handles complex relationships between product attributes and pricing
- Considers factors like brand, specifications, and product quantity

---

## 📊 Dataset

### Data Description

The dataset consists of the following columns:

1. **sample_id**: Unique identifier for each product sample
2. **catalog_content**: Text field containing title, product description, and Item Pack Quantity (IPQ) concatenated
3. **image_link**: Public URL for product image download
4. **price**: Target variable (available only in training data)

### Dataset Details

- **Training Dataset**: 75,000 products with complete details and prices
- **Test Dataset**: 75,000 products for final evaluation

---

## 🚀 Solution Approach

### 1. **Text Embeddings**
- Utilized **GTE (General Text Embeddings)** for semantic text representation
- Pre-computed embeddings for both training and test datasets
- Embedding dimension: 1024 features per sample

### 2. **Feature Engineering**

Extracted multiple engineered features from catalog content:

- **Pack Count (IPQ)**: Extracted from patterns like "pack of 4", "x4", "4 pack"
- **Total Quantity**: Normalized quantity in base units (g/ml/count)
- **Digit Count**: Number of numeric tokens in text
- **Word Count**: Total number of words
- **Price Indicators**: Count of price-related keywords
- **Brand Indicators**: Count of brand-related keywords
- **Quality Indicators**: Count of quality-related keywords
- **Special Characters**: Count of special characters
- **Uppercase Ratio**: Ratio of uppercase letters
- **Number Ratio**: Ratio of numeric characters

### 3. **Model Architecture**

- **Algorithm**: LightGBM (Gradient Boosting Decision Trees)
- **Target Transformation**: Log transformation of prices
- **Cross-Validation**: 5-fold Stratified K-Fold CV
- **Ensemble**: Multiple models with different random seeds per fold

### 4. **Model Configuration**

```python
lgb_params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "num_leaves": 255,
    "max_depth": 15,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.5,
    "num_boost_round": 8000,
    "early_stopping_rounds": 300
}
```

### 5. **Data Preprocessing**

- Removal of non-positive prices
- Outlier detection and removal using IQR method (3x IQR bounds)
- Feature scaling using StandardScaler for numeric features
- TF-IDF fallback when embeddings are unavailable

---

## 📁 Project Structure

```
Amazon ML Challenge/
│
├── Text processing/
│   ├── train.csv                    # Training dataset
│   ├── test.csv                     # Test dataset
│   ├── gte_train_embeddings.npy     # Pre-computed GTE embeddings (train)
│   ├── gte_test_embeddings.npy      # Pre-computed GTE embeddings (test)
│   ├── model.py                     # Simplified model script
│   └── submission.csv               # Generated predictions
│
├── models_out/
│   ├── lgbm_fold1.txt              # Trained model (fold 1)
│   ├── lgbm_fold2.txt              # Trained model (fold 2)
│   ├── lgbm_fold3.txt              # Trained model (fold 3)
│   ├── lgbm_fold4.txt              # Trained model (fold 4)
│   ├── lgbm_fold5.txt              # Trained model (fold 5)
│   ├── oof_predictions_train.csv   # Out-of-fold predictions
│   └── submission.csv              # Final test predictions
│
├── student_resource/
│   ├── README.md                    # Challenge documentation
│   ├── Documentation_template.md    # Documentation template
│   ├── sample_code.py              # Sample code
│   ├── dataset/                    # Sample datasets
│   └── src/                        # Utility functions
│
├── train_lightgbm.py               # Main training script (advanced)
├── simple_train.py                 # Simplified training script
├── quick_smape.py                  # Quick SMAPE evaluation script
├── debug_script.py                 # Debugging utilities
├── test_script.py                  # Testing utilities
└── README.md                       # This file
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Amazon ML Challenge"
```

2. **Install required packages**
```bash
pip install numpy pandas scikit-learn lightgbm joblib
```

3. **Optional: For text embedding generation**
```bash
pip install sentence-transformers torch
```

---

## 💻 Usage

### Training the Model

#### Option 1: Advanced Training (Recommended)
```bash
python train_lightgbm.py
```

Features:
- Uses pre-computed GTE embeddings
- Falls back to TF-IDF if embeddings unavailable
- Advanced feature engineering
- Ensemble modeling with multiple seeds
- Comprehensive metrics tracking

#### Option 2: Simple Training
```bash
python simple_train.py
```

Features:
- Faster execution
- Simplified feature set
- Basic cross-validation

### Evaluating Results

```bash
python quick_smape.py
```

### Output Files

- **models_out/submission.csv**: Final predictions for test set
- **models_out/oof_predictions_train.csv**: Out-of-fold predictions for training set
- **models_out/lgbm_fold*.txt**: Saved LightGBM models for each fold

---

## 📈 Model Performance

### Evaluation Metric

**SMAPE (Symmetric Mean Absolute Percentage Error)**

```
SMAPE = (1/n) * Σ |predicted_price - actual_price| / ((|actual_price| + |predicted_price|)/2) * 100%
```

- **Range**: 0% to 200%
- **Lower is better**

### Cross-Validation Results

The model is evaluated using 5-fold cross-validation with the following metrics:

- **RMSE (log)**: Root Mean Squared Error on log-transformed prices
- **RMSE (price)**: Root Mean Squared Error on actual prices
- **MAE (price)**: Mean Absolute Error on actual prices
- **SMAPE**: Symmetric Mean Absolute Percentage Error

*Note: Specific performance metrics are printed during training execution.*

---

## ✨ Key Features

### 1. **Robust Feature Engineering**
- Automated extraction of pack quantities and product units
- Multi-pattern regex matching for quantity detection
- Normalization of different unit systems (g, kg, ml, l, etc.)

### 2. **Advanced Text Processing**
- State-of-the-art GTE embeddings for semantic understanding
- TF-IDF fallback mechanism for reliability
- Comprehensive text statistics extraction

### 3. **Model Optimization**
- Stratified K-Fold for balanced price distribution
- Ensemble approach with multiple random seeds
- Early stopping to prevent overfitting
- Hyperparameter tuning for optimal performance

### 4. **Production-Ready Code**
- Modular and maintainable structure
- Comprehensive error handling
- Detailed logging and progress tracking
- Scalable architecture

---

## 🔧 Technologies Used

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities and preprocessing
- **LightGBM**: Gradient boosting framework
- **Sentence Transformers**: Text embedding generation (GTE models)
- **Regular Expressions**: Pattern matching for feature extraction

---

## 🏆 Results

The solution successfully:

✅ Processes 75,000 training samples and generates predictions for 75,000 test samples  
✅ Implements advanced feature engineering from product descriptions  
✅ Utilizes state-of-the-art text embeddings for semantic understanding  
✅ Achieves robust cross-validation performance with ensemble modeling  
✅ Generates submission-ready predictions in the required format  
✅ Handles edge cases (outliers, missing values, non-positive prices)  

---

## 👥 Collaborators

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/amoggh">
        <img src="https://github.com/amoggh.png" width="100px;" alt="Amoggh Bharadwaj"/>
        <br />
        <sub><b>Amoggh Bharadwaj</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Saip2231">
        <img src="https://github.com/Saip2231.png" width="100px;" alt="Sai Prashanth"/>
        <br />
        <sub><b>Sai Prashanth</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Shaq16">
        <img src="https://github.com/Shaq16.png" width="100px;" alt="Shakthi Jaswanth"/>
        <br />
        <sub><b>Shakthi Jaswanth</b></sub>
      </a>
    </td>
  </tr>
</table>

**Team: Borderland Survivors**


---

## 📄 License

This project is developed for the Amazon ML Challenge 2025. The solution uses MIT/Apache 2.0 licensed models as per challenge requirements.

---

## 🙏 Acknowledgments

- Amazon ML Challenge 2025 organizers for providing the dataset and problem statement
- The open-source community for the excellent ML libraries and tools
- GTE (General Text Embeddings) model developers for state-of-the-art text representations

---
