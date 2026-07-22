# 🔍 Fraud-XAI: Explainable AI & Machine Learning Framework for Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive, research-grade Machine Learning & Deep Learning framework for credit card fraud detection with **Explainable AI (XAI / SHAP)**, **Cost-Sensitive Business Utility Evaluation**, and an **Interactive Streamlit Performance Dashboard**.

---

## 📌 Key Highlights & Features

- **75 CV Experiments**: 5-Fold Stratified Cross-Validation evaluating **5 Model Families** (XGBoost, Random Forest, CatBoost, Logistic Regression, PyTorch ANN) across **3 Imbalance Handling Conditions** (Class-weighting, SMOTE, SMOTE-ENN).
- **Explainable AI (XAI / SHAP)**: Complete suite of 5 popular SHAP plot types:
  - 🐝 **Beeswarm Dot Plot**: Feature directionality & risk contribution with Pink-Blue gradient.
  - 📊 **Global Summary Bar Plot**: Feature importance ranking ($\text{mean}(\lvert\text{SHAP}\rvert)$).
  - 💧 **Waterfall Plot**: Local step-by-step explanation for individual fraud transactions.
  - 📈 **Dependence Plot**: Non-linear feature interaction scatter analysis (e.g. `V14`, `Amount`).
  - 🛤️ **Decision Plot**: Multi-transaction decision path visualization.
- **Cost-Sensitive Business Financial Impact**: Evaluates net financial savings ($C_{FN} = \text{fraud amount}$, $C_{FP} = \$10$) and Precision@Top-K alerts (Top-100, Top-200, Top-500) for banking operations.
- **Interactive Streamlit Dashboard**: 5-panel interactive UI for performance comparison, heatmaps, XAI plots, and financial impact tables.
- **Automated Data Profiling**: Standalone HTML report generation via `ydata-profiling`.

---

## 📊 Summary Results (5-Fold Stratified CV)

| Model | Imbalance Condition | **PR-AUC (Mean ± Std)** | ROC-AUC (Mean ± Std) | F1-Score (Mean ± Std) | Recall (Mean ± Std) |
|---|---|:---:|:---:|:---:|:---:|
| **XGBoost** | **Class-weighting** | 🥇 **0.8581 ± 0.0286** | 0.9822 ± 0.0098 | **0.8811 ± 0.0282** | 0.8211 ± 0.0274 |
| **XGBoost** | **SMOTE** | 🥈 **0.8578 ± 0.0284** | 0.9798 ± 0.0081 | 0.8742 ± 0.0240 | 0.8069 ± 0.0358 |
| **Random Forest** | **SMOTE** | 🥉 **0.8495 ± 0.0295** | 0.9819 ± 0.0095 | 0.8534 ± 0.0312 | 0.7906 ± 0.0525 |
| **Random Forest** | **SMOTE-ENN** | **0.8431 ± 0.0338** | **0.9834 ± 0.0074** | 0.8531 ± 0.0374 | 0.7987 ± 0.0486 |
| **ANN (PyTorch)**| **SMOTE-ENN** | **0.8267 ± 0.0427** | 0.9710 ± 0.0070 | 0.8443 ± 0.0229 | 0.8090 ± 0.0192 |
| **CatBoost** | **SMOTE** | **0.8369 ± 0.0449** | 0.9769 ± 0.0058 | 0.8636 ± 0.0296 | 0.8130 ± 0.0176 |
| **Logistic Reg.** | **Class-weighting** | **0.7258 ± 0.0328** | 0.9783 ± 0.0064 | 0.7825 ± 0.0235 | 0.7665 ± 0.0403 |

---

## 📁 Repository Structure

```text
Fraud-XAI-simple/
├── data/
│   └── raw/creditcard.csv           # Raw dataset (ULB Credit Card Fraud)
├── src/
│   ├── preprocess.py                # Data loading, Stratified 80/20, RobustScaler
│   ├── imbalance.py                 # SMOTE, SMOTE-ENN, Class-weighting handlers
│   ├── models.py                    # Scikit-learn, XGBoost, CatBoost constructors
│   ├── models_dl.py                 # PyTorch FraudANN architecture & training loop
│   ├── cv.py                        # Unified StratifiedKFold index generator
│   ├── evaluation.py                # F1-optimal threshold, paired Wilcoxon, Cost-sensitive metrics
│   ├── explain.py                   # SHAP explainer & 5 SHAP plot visualizers
│   ├── visualize.py                 # Plotly & Matplotlib comparative plots
│   └── data_profiling.py            # Automated ydata-profiling HTML report generator
├── outputs/
│   ├── metrics/                     # fold_results.csv, summary.csv, paired_tests.csv
│   ├── figures/                     # Generated PNG charts & SHAP plots
│   ├── shap_values/                 # Cached SHAP values (.npy files)
│   └── profiling/                   # data_profile.html report
├── dashboard/
│   └── app.py                       # Interactive 5-panel Streamlit Dashboard
├── notebooks/
│   └── 01_eda.ipynb                 # Exploratory data analysis notebook
├── configs/
│   └── default.yaml                 # Pipeline hyperparameter configuration
├── run_experiment.py                # Executive pipeline launcher with auto-checkpointing
├── requirements.txt                 # Python dependency specification
├── .gitignore                       # Clean Git commit filters
└── README.md                        # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
Clone the repository and set up a Python virtual environment:
```bash
git clone <your-repo-url>
cd Fraud-XAI-simple

# Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Data Download
Download the ULB Credit Card Fraud dataset from Kaggle and place `creditcard.csv` in `data/raw/`:
```bash
# Via Kaggle CLI (or download manually from Kaggle)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

---

## 💻 Usage Instructions

### A. Launch Streamlit Performance & XAI Dashboard
```bash
streamlit run dashboard/app.py
```
> Opens interactive dashboard at `http://localhost:8501` featuring performance metrics, SHAP Beeswarm/Waterfall/Dependence plots, and Business Financial Impact analysis.

### B. Run Full 75-Fold Experiment Pipeline
```bash
python run_experiment.py
```
Options:
- `--conditions SMOTE-ENN` : Run specific imbalance conditions.
- `--no-ann` : Skip PyTorch ANN branch (ML models only).
- `--no-shap` : Skip SHAP value calculation for faster runs.

### C. Generate Full Data Profiling Report
```bash
python src/data_profiling.py
```
> Generates comprehensive HTML report at `outputs/profiling/data_profile.html`.

---

## 📄 License & Acknowledgments
Distributed under the MIT License. Dataset courtesy of Machine Learning Group (ULB) on Kaggle.
