# Antigravity — Explainable ML for Credit Card Fraud Detection

Nghiên cứu tác động của các kỹ thuật xử lý mất cân bằng dữ liệu lên hiệu năng phân loại giao dịch gian lận thẻ tín dụng.

**Dataset:** ULB Credit Card Fraud Detection (Kaggle)

## Research Questions

| # | Question |
|---|---|
| RQ1 | Kỹ thuật resampling nào tối ưu hóa hiệu năng phân loại (PR-AUC, F1, Recall) cho từng thuật toán máy học? |
| RQ2 | Có thể xây dựng một dashboard proof-of-concept để theo dõi hiệu năng phân loại trực quan? |

## Experiment Matrix (12 configurations)

|  | C0 (Class-weight) | C1 (SMOTE) | C2 (SMOTE-ENN) |
|---|:---:|:---:|:---:|
| Random Forest | ✓ | ✓ | ✓ |
| XGBoost | ✓ | ✓ | ✓ |
| CatBoost | ✓ | ✓ | ✓ |
| Logistic Regression | ✓ | ✓ | ✓ |

## Project Structure

```
antigravity/
├── data/raw/creditcard.csv         # Raw data (not committed)
├── notebooks/
│   └── 01_eda.ipynb                # EDA & distributions
├── src/
│   ├── preprocess.py               # Load, split, RobustScaler
│   ├── imbalance.py                # SMOTE / SMOTE-ENN / class-weight
│   ├── models.py                   # Model definitions + hyperparams
│   ├── evaluate.py                 # PR-AUC, F1, Recall, ROC-AUC
│   └── visualize.py                # Performance plots & comparisons
│   └── tuning/
│       └── optuna_tuner.py         # Optuna hyperparameter tuning
├── outputs/
│   ├── models/                     # Trained model binaries
│   ├── results/                    # JSON/CSV per configuration
│   └── figures/                    # PNG plots
├── dashboard/
│   └── app.py                      # Streamlit proof-of-concept (RQ2)
├── configs/default.yaml            # Experiment configuration
├── run_experiments.py              # Run all 12 configurations
├── tune_hyperparams.py             # Optuna tuning CLI
├── requirements.txt
└── .gitignore
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset:**
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
   ```

## Usage

### Run Full Experiment Matrix
```bash
python run_experiments.py
```



### Run specific models
```bash
python run_experiments.py --models RF CatBoost
```

### Hyperparameter Tuning (Optuna)
```bash
python tune_hyperparams.py --model CatBoostClassifier --trials 50
python tune_hyperparams.py --model XGBClassifier --trials 30
python tune_hyperparams.py --model LogisticRegression --trials 20
```

### Dashboard (RQ3)
```bash
streamlit run dashboard/app.py
```

## Notebooks

- `notebooks/01_eda.ipynb` — Exploratory Data Analysis & class distribution visualization
