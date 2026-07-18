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
# CELL 7 — FULL EXPERIMENT RUN (ML + ANN)
# Kaggle P100/T4 GPU — ước tính tổng ~2-3 giờ với settings này
#
# Cơ chế dừng sớm Optuna (whichever comes first):
#   --trials 20      : tối đa 20 trials mỗi model mỗi fold
#   --patience 7     : dừng ML tune nếu 7 trial liên tiếp không cải thiện
#   --ann-patience 5 : dừng ANN tune nếu 5 trial liên tiếp không cải thiện
#   --timeout 300    : hard cap 5 phút mỗi Optuna study (backup)
#   ANN còn có MedianPruner: cắt sớm từng trial tệ hơn median các trial khác
# ─────────────────────────────────────────────────────────────────────
import time
t0 = time.time()

subprocess.run([
    sys.executable, "run_experiment.py",
    "--folds",        "5",
    "--trials",       "20",     # max 20 trials per model per fold
    "--patience",     "7",      # ML early-stop: 7 trials no improve → stop
    "--ann-patience", "5",      # ANN early-stop: 5 trials no improve → stop
    "--timeout",      "300",    # hard cap: 5 phút/study (backup safety net)
    "--seed",         "42",
    # "--no-ann",              # bỏ comment để chỉ chạy ML
    # "--no-shap",             # bỏ comment để bỏ qua SHAP và chạy nhanh hơn
], check=True)

elapsed = time.time() - t0
print(f"\n✅ Experiment done in {elapsed/3600:.2f} hours")

# ─────────────────────────────────────────────────────────────────────
# CELL 8 — Nếu muốn chạy tách biệt: chỉ ML, không ANN
# ─────────────────────────────────────────────────────────────────────
# subprocess.run([
#     sys.executable, "run_experiment.py",
#     "--no-ann",
#     "--folds",    "5",
#     "--trials",   "20",
#     "--patience", "7",
#     "--timeout",  "300",
#     "--seed",     "42",
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
