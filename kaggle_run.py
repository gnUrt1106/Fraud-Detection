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
import os
import shutil

REPO_DIR = "/kaggle/working/Fraud-Detection"
LOCAL_DATA = f"{REPO_DIR}/data/raw/creditcard.csv"

os.makedirs(f"{REPO_DIR}/data/raw", exist_ok=True)

# Tự động tìm đường dẫn file dataset
kaggle_data_path = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "creditcard.csv" in files:
        kaggle_data_path = os.path.join(root, "creditcard.csv")
        break

if not kaggle_data_path:
    raise FileNotFoundError("⚠️ Dataset not found! Add 'Credit Card Fraud Detection' dataset to this notebook.")

print(f"Found dataset at: {kaggle_data_path}")

# Xóa symlink/file cũ nếu có
if os.path.lexists(LOCAL_DATA):
    os.remove(LOCAL_DATA)

print("Copying dataset to local folder (avoiding symlink issues)...")
shutil.copy2(kaggle_data_path, LOCAL_DATA)
print(f"✅ Ready: {LOCAL_DATA}")
print(f"   Accessible: {os.path.exists(LOCAL_DATA)}")

# ─────────────────────────────────────────────────────────────────────
# CELL 4 — (Optional) Data Profiling
# Bỏ comment nếu muốn tạo HTML report — mất ~5-10 phút
# ─────────────────────────────────────────────────────────────────────
subprocess.run([
sys.executable, "src/data_profiling.py",
print("Profiling report saved to outputs/profiling/data_profile.html")

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
subprocess.run([
    sys.executable, "run_experiment.py",
    "--models", "LR",
    "--conditions", "Class-weighting",
    "--folds", "2",
    "--trials", "3",
    "--no-shap",
    "--no-ann",
], check=True)
print("Sanity check passed!")

# ─────────────────────────────────────────────────────────────────────
# CELL 7 — FULL EXPERIMENT RUN (4 ML + ANN, GPU)
# Chạy toàn bộ 5 model: LR, RF, XGB, CatBoost, ANN × 3 conditions × 5 folds
# 
# Optuna không còn nested CV — dùng outer-fold val set trực tiếp (5x nhanh hơn).
# Checkpoint sau mỗi fold: nếu timeout, chạy lại sẽ tiếp tục từ chỗ dừng.
#
# Cơ chế dừng sớm Optuna:
#   --trials 20      : tối đa 20 trials
#   --patience 7     : dừng ML tune nếu 7 trial liên tiếp không cải thiện
#   --ann-patience 5 : dừng ANN tune nếu 5 trial liên tiếp không cải thiện
#   --timeout 300    : hard cap 5 phút mỗi Optuna study (backup)
# ─────────────────────────────────────────────────────────────────────
# CELL 3b — (Optional) Load existing outputs / checkpoints if uploaded as Dataset
# ─────────────────────────────────────────────────────────────────────
for root, dirs, files in os.walk("/kaggle/input"):
    if "fold_results.csv" in files:
        checkpoint_dir = root
        print(f"📦 Found uploaded checkpoint at: {checkpoint_dir}")
        target_outputs = f"{REPO_DIR}/outputs/metrics"
        os.makedirs(target_outputs, exist_ok=True)
        shutil.copy2(os.path.join(checkpoint_dir, "fold_results.csv"), os.path.join(target_outputs, "fold_results.csv"))
        print("✅ Checkpoint fold_results.csv restored into Fraud-Detection/outputs/metrics/")
        break

# ─────────────────────────────────────────────────────────────────────
# CELL 7 — FULL EXPERIMENT RUN (ML + ANN, GPU, Parallelized)
# ─────────────────────────────────────────────────────────────────────
import time
import subprocess
import sys

t0 = time.time()
REPO_DIR = "/kaggle/working/Fraud-Detection"

# Chạy toàn bộ 5 model: LR, RF, XGB, CatBoost, ANN × 3 conditions × 5 folds
# Tự động n_jobs=-1 cho SMOTE/SMOTE-ENN & Skip các fold đã làm trong checkpoint.
subprocess.run([
    sys.executable, "run_experiment.py",
    "--models",       "LR", "RF", "XGB", "CatBoost",
    "--folds",        "5",
    "--trials",       "15",
    "--patience",     "5",
    "--ann-patience", "5",
    "--timeout",      "300",
    "--seed",         "42",
], cwd=REPO_DIR, check=True)

elapsed = time.time() - t0
print(f"\n✅ Experiment done in {elapsed/3600:.2f} hours")

# ─────────────────────────────────────────────────────────────────────
# CELL 9 — Xem kết quả
# ─────────────────────────────────────────────────────────────────────
import pandas as pd

metrics_summary_path = os.path.join(REPO_DIR, "outputs", "metrics", "summary.csv")
if os.path.exists(metrics_summary_path):
    summary = pd.read_csv(metrics_summary_path)
    print("\n=== Summary (mean PR-AUC per model × condition) ===")
    if "PR-AUC_mean" in summary.columns:
        pivot = summary.pivot_table(
            index="model", columns="condition",
            values="PR-AUC_mean", aggfunc="first",
        )
        print(pivot.round(4).to_string())
    else:
        print(summary.to_string())

paired_path = os.path.join(REPO_DIR, "outputs", "metrics", "paired_tests.csv")
if os.path.exists(paired_path):
    print("\n=== Paired Wilcoxon Test Results ===")
    paired = pd.read_csv(paired_path)
    sig = paired[paired.get("significant", False) == True]
    print(f"Significant pairs: {len(sig)}/{len(paired)}")

# ─────────────────────────────────────────────────────────────────────
# CELL 10 — Zip toàn bộ outputs chuẩn bị download (1-Click Download)
# ─────────────────────────────────────────────────────────────────────
import shutil, os

os.makedirs("/kaggle/working", exist_ok=True)
shutil.make_archive("/kaggle/working/outputs", 'zip', f"{REPO_DIR}/outputs")
print("🎉 All outputs zipped to /kaggle/working/outputs.zip — Click Download on Kaggle Output Panel!")
