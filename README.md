# Credit Card Default Prediction

A comprehensive machine learning project that implements and compares multiple algorithms for predicting credit card default payments using the UCI Default of Credit Card Clients dataset.

## Project Overview

This project demonstrates expertise in machine learning by implementing 7 different algorithms to predict credit card defaults. The analysis includes comprehensive model evaluation, hyperparameter tuning, cost-sensitive analysis, and feature importance assessment.

## Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- **Size**: 30,000 instances, 23 features
- **Target**: Binary classification (default payment: Yes=1, No=0)
- **Class Distribution**: ~22% default cases (imbalanced dataset)
- **Features**: Demographics, credit limit, payment history, billing statements

## Models Implemented

1. **Logistic Regression** (with SMOTE)
2. **K-Nearest Neighbors** (with SMOTE)
3. **Decision Trees** (hyperparameter tuned)
4. **Gaussian Naive Bayes** (hyperparameter tuned)
5. **Linear Discriminant Analysis** (hyperparameter tuned)
6. **Quadratic Discriminant Analysis** (hyperparameter tuned)
7. **Multi-Layer Perceptron** (neural network, hyperparameter tuned)

## Key Results

| Model | ROC AUC | F1 Score | Min Cost | Optimal Threshold |
|-------|---------|----------|----------|-------------------|
| **MLP (Tuned)** | **0.87** | **0.75** | 3,505 | 0.20 |
| **Decision Tree (Tuned)** | **0.85** | **0.72** | **3,474** | 0.39 |
| **QDA (Tuned)** | **0.83** | **0.70** | 3,690 | 0.52 |
| **Gaussian NB (Tuned)** | **0.81** | **0.68** | 3,800 | 0.45 |
| **LDA (Tuned)** | **0.82** | **0.69** | 3,884 | 0.22 |
| **Logistic Regression (SMOTE)** | **0.84** | **0.71** | 3,884 | 0.54 |
| **KNN (SMOTE)** | **0.82** | **0.69** | 3,931 | 0.28 |

*Cost analysis uses FP=1, FN=5 weights*

## Key Findings

- **Best Overall Performance**: Multi-Layer Perceptron (AUC: 0.87)
- **Best Cost Optimization**: Decision Tree (Min Cost: 3,474)
- **Most Important Features**: Recent payment history (PAY_0, PAY_2, PAY_3)
- **Class Imbalance**: SMOTE improved linear models but had limited effect on tree-based methods
- **Business Impact**: Cost-optimal thresholds varied significantly across models (0.20 to 0.54)

## Project Structure

```
CreditCardDefaultPrediction/
├── notebooks/
│   └── credit_card_default_analysis.ipynb  # Main analysis notebook
├── results/
│   └── figures/                            # Generated visualizations
├── docs/
│   └── Credit_Card_Default_Prediction.pdf
│      
├── data/                                   # Dataset (downloaded automatically)
└── README.md
```

## Technologies Used

- **Python 3.9+**
- **scikit-learn**: Machine learning algorithms and evaluation
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib/seaborn**: Data visualization
- **imbalanced-learn**: SMOTE for class balancing


## Installation and Usage

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CreditCardDefaultPrediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**:
   ```bash
   jupyter notebook notebooks/credit_card_default_analysis.ipynb
   ```


## Evaluation Metrics

- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Precision-Recall Curves**: Important for imbalanced datasets
- **F1-Score**: Harmonic mean of precision and recall
- **Cost-Sensitive Analysis**: Business-relevant cost optimization (FP=1, FN=5)
- **Confusion Matrices**: Detailed prediction breakdown
- **Feature Importance**: Interpretability analysis

## Methodology

1. **Data Preprocessing**: Standardization, categorical encoding, train-test split
2. **Class Imbalance Handling**: SMOTE resampling for linear models
3. **Hyperparameter Tuning**: Grid search with 5-fold cross-validation
4. **Model Evaluation**: Multiple metrics with cost-sensitive analysis
5. **Feature Analysis**: Importance ranking and interpretability

## Business Context

Credit card default prediction is crucial for financial institutions to assess credit risk. The cost of false negatives (missing potential defaulters) is typically higher than false positives (rejecting good customers), making this a cost-sensitive classification problem.
