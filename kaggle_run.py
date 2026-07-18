# # Fraud Detection: ML vs DL Experiment Pipeline
# 
# **Kaggle Notebook** — chạy toàn bộ pipeline trên GPU P100/T4
#
# Steps:
# 1. Clone repository
# 2. Install dependencies  
# 3. Run data profiling (optional, nặng)
# 4. Run full experiment (ML + ANN, 75 runs, Optuna per fold)
# 5. Download results

# ─────────────────────────────────────────────────────────────────────
# CELL 1 — Setup environment
# ─────────────────────────────────────────────────────────────────────
import subprocess, os, sys

# Clone repo (thay URL nếu đổi tên)
REPO_URL = "https://github.com/gnUrt1106/Fraud-Detection.git"
REPO_DIR = "Fraud-Detection"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL], check=True)
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, ".")
print("Working dir:", os.getcwd())

# ─────────────────────────────────────────────────────────────────────
# CELL 2 — Install dependencies
# ─────────────────────────────────────────────────────────────────────
# Kaggle đã có: numpy, pandas, sklearn, xgboost, catboost, optuna, shap
# Chỉ cần cài thêm: imbalanced-learn, ydata-profiling
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "imbalanced-learn>=0.12",
    "ydata-profiling>=4.6",
    "scipy>=1.12",
], check=True)
print("Dependencies installed.")

# ─────────────────────────────────────────────────────────────────────
# CELL 3 — Symlink dataset (Kaggle dataset phải được add vào notebook)
# ─────────────────────────────────────────────────────────────────────
# Kaggle dataset: mlg-ulb/creditcardfraud
# Sau khi add dataset vào notebook, file sẽ có ở:
#   /kaggle/input/creditcardfraud/creditcard.csv

KAGGLE_DATA = "/kaggle/input/creditcardfraud/creditcard.csv"
LOCAL_DATA  = "data/raw/creditcard.csv"

os.makedirs("data/raw", exist_ok=True)
if os.path.exists(KAGGLE_DATA) and not os.path.exists(LOCAL_DATA):
    os.symlink(KAGGLE_DATA, LOCAL_DATA)
    print(f"Symlinked dataset: {KAGGLE_DATA} → {LOCAL_DATA}")
elif os.path.exists(LOCAL_DATA):
    print(f"Dataset already exists at {LOCAL_DATA}")
else:
    print("⚠️ Dataset not found! Add 'Credit Card Fraud Detection' dataset to this notebook.")

# ─────────────────────────────────────────────────────────────────────
# CELL 4 — (Optional) Data Profiling
# Bỏ comment nếu muốn tạo HTML report — mất ~5-10 phút
# ─────────────────────────────────────────────────────────────────────
# subprocess.run([
#     sys.executable, "src/data_profiling.py",
#     "--minimal",   # bỏ --minimal nếu muốn report đầy đủ hơn
# ], check=True)
# print("Profiling report saved to outputs/profiling/data_profile.html")

# ─────────────────────────────────────────────────────────────────────
# CELL 5 — Kiểm tra GPU
# ─────────────────────────────────────────────────────────────────────
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────
# CELL 6 — Quick sanity check (2 folds, 3 trials, LR only, no SHAP)
# Uncomment để test pipeline trước khi chạy full
# ─────────────────────────────────────────────────────────────────────
# subprocess.run([
#     sys.executable, "run_experiment.py",
#     "--models", "LR",
#     "--conditions", "Class-weighting",
#     "--folds", "2",
#     "--trials", "3",
#     "--no-shap",
#     "--no-ann",
# ], check=True)
# print("Sanity check passed!")

# ─────────────────────────────────────────────────────────────────────
# CELL 7 — FULL EXPERIMENT RUN (ML only, no ANN)
# Chạy session GPU đầu tiên — ML × 3 conditions × 5 folds × 30 trials
# Ước tính: ~2-3 giờ trên P100
# ─────────────────────────────────────────────────────────────────────
import time
t0 = time.time()

subprocess.run([
    sys.executable, "run_experiment.py",
    "--no-ann",        # bỏ cờ này nếu muốn chạy cả ANN
    "--folds",   "5",
    "--trials",  "30",
    "--patience", "10",
    "--seed",    "42",
    # "--no-shap",   # bỏ comment nếu muốn bỏ qua SHAP để chạy nhanh hơn
], check=True)

elapsed = time.time() - t0
print(f"\n✅ ML experiment done in {elapsed/3600:.2f} hours")

# ─────────────────────────────────────────────────────────────────────
# CELL 8 — ANN EXPERIMENT (chạy session riêng hoặc tiếp session trên)
# Nếu quota GPU còn, uncomment và chạy tiếp — chỉ chạy ANN branch
# Ước tính: ~1-1.5 giờ trên P100
# ─────────────────────────────────────────────────────────────────────
# subprocess.run([
#     sys.executable, "run_experiment.py",
#     "--models",      # bỏ cờ --models để không chạy ML lại
#     "--no-ml",       # TODO: thêm flag này vào run_experiment.py nếu cần
#     "--folds",   "5",
#     "--trials",  "30",
#     "--seed",    "42",
# ], check=True)

# ─────────────────────────────────────────────────────────────────────
# CELL 9 — Xem kết quả
# ─────────────────────────────────────────────────────────────────────
import pandas as pd

summary = pd.read_csv("outputs/metrics/summary.csv")
print("\n=== Summary (mean PR-AUC per model × condition) ===")
pivot = summary.pivot_table(
    index="model", columns="condition",
    values="PR-AUC_mean", aggfunc="first",
)
print(pivot.round(4).to_string())

print("\n=== Paired Wilcoxon Test Results ===")
paired = pd.read_csv("outputs/metrics/paired_tests.csv")
sig = paired[paired["significant"] == True]
print(f"Significant pairs: {len(sig)}/{len(paired)}")
print(sig[["model_a", "model_b", "condition", "mean_a", "mean_b", "p_value", "winner"]].to_string())

# ─────────────────────────────────────────────────────────────────────
# CELL 10 — Zip và save outputs (Kaggle Output)
# ─────────────────────────────────────────────────────────────────────
import shutil, os

os.makedirs("/kaggle/working", exist_ok=True)
shutil.copytree("outputs/metrics",     "/kaggle/working/metrics",     dirs_exist_ok=True)
shutil.copytree("outputs/shap_values", "/kaggle/working/shap_values", dirs_exist_ok=True)
print("Results copied to /kaggle/working/ — download from Kaggle Output panel.")
