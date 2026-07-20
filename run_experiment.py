#!/usr/bin/env python3
"""
run_experiment.py — Orchestration chính của toàn bộ pipeline.

Pipeline:
    Set global seed
    → Load & preprocess data
    → Sinh fold indices (dùng chung cho cả ML và DL)
    → Loop: fold × condition × model
        → Fit scaler trên train fold (không leakage)
        → Apply imbalance handling trên train fold
        → Optuna tune (per fold)
        → Train final model với best params
        → Evaluate (F1-optimal threshold)
        → SHAP
    → Paired Wilcoxon test giữa các model
    → Lưu summary.csv, fold_results.csv, paired_tests.csv

Tổng: (4 ML + 1 ANN) × 3 conditions × 5 folds = 75 runs.

Usage:
    python run_experiment.py                          # Full run
    python run_experiment.py --no-shap                # Bỏ qua SHAP
    python run_experiment.py --models RF LR --folds 2 --trials 5  # Quick test
    python run_experiment.py --conditions Class-weighting --trials 3
"""

import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import optuna

# Suppress Optuna log spam
optuna.logging.set_verbosity(optuna.logging.WARNING)

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocess import load_and_split, scale_features
from src.imbalance import apply_condition, CONDITIONS
from src.models import get_model, MODEL_NAMES
from src.cv import get_fold_indices
from src.evaluation import evaluate_fold, summarize_folds, paired_wilcoxon_test, save_summary
from src.tuning.optuna_tuner import objective as ml_optuna_objective, EarlyStoppingCallback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────
OUTPUTS = {
    "metrics":     "outputs/metrics",
    "shap_values": "outputs/shap_values",
    "models":      "outputs/models",
    "figures":     "outputs/figures",
}
for d in OUTPUTS.values():
    os.makedirs(d, exist_ok=True)


# ── Checkpoint helpers ──────────────────────────────────────────────────

def _save_checkpoint(fold_records: list[dict], output_dir: str) -> None:
    """Ghi fold_results.csv ngay sau mỗi fold — tránh mất data khi crash/timeout."""
    pd.DataFrame(fold_records).to_csv(
        f"{output_dir}/fold_results.csv", index=False
    )


def _load_checkpoint(output_dir: str) -> tuple[list[dict], set]:
    """Load fold_results.csv (nếu có) để resume từ chỗ đã chạy.

    Returns:
        fold_records : list of metric dicts đã chạy.
        done_keys   : set of (model, condition, fold) đã hoàn thành.
    """
    checkpoint_path = f"{output_dir}/fold_results.csv"
    if os.path.exists(checkpoint_path):
        df = pd.read_csv(checkpoint_path)
        records = df.to_dict("records")
        done_keys = {
            (r["model"], r["condition"], int(r["fold"])) for r in records
        }
        logger.info("🔄 Checkpoint loaded: %d fold(s) already done.", len(records))
        return records, done_keys
    return [], set()


# ── Global seed ───────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    """Cố định seed cho numpy, random, sklearn, torch (nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Reproducibility trên MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass
    logger.info("Global seed set to %d", seed)


# ── Model name mapping for Optuna tuner ─────────────
_ML_MODEL_NAME_MAP = {
    "RF": "RandomForestClassifier",
    "XGB": "XGBClassifier",
    "CatBoost": "CatBoostClassifier",
    "LR": "LogisticRegression",
}


# ── Optuna tune helpers ──────────────────────────

def tune_ml_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    patience: int,
    seed: int,
    timeout: Optional[int] = None,
) -> dict:
    """
    Chạy Optuna tune cho 1 ML model trên dữ liệu train của fold hiện tại.

    Optuna dùng outer-fold val set (X_val/y_val) — không nested CV bên trong.

    Dừng sớm theo 3 điều kiện (whichever comes first):
    - n_trials đạt giới hạn
    - EarlyStoppingCallback: 'patience' trials liên tiếp không cải thiện
    - timeout: số giây tối đa cho cả study (None = không giới hạn)

    Returns:
        best_params (dict) để truyền vào get_model().
    """
    optuna_name = _ML_MODEL_NAME_MAP[model_name]

    study = optuna.create_study(direction="maximize")
    callbacks = [EarlyStoppingCallback(patience=patience)]

    study.optimize(
        lambda trial: ml_optuna_objective(
            trial, X_train, y_train, optuna_name, X_val=X_val, y_val=y_val
        ),
        n_trials=n_trials,
        timeout=timeout,      # dừng sau X giây bất kể còn trial nào
        n_jobs=1,
        callbacks=callbacks,
    )

    best = study.best_trial
    params = dict(best.params)

    # ── Fix LR solver ──────────────────────────────────────────────────
    if model_name == "LR":
        params.setdefault("solver", "saga")
        params.setdefault("max_iter", 5000)

    logger.info(
        "  [Optuna-ML] %s | %d trials run | best PR-AUC=%.4f | params=%s",
        model_name, len(study.trials), best.value, params,
    )
    return params



def tune_ann_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    condition: str,
    n_trials: int,
    input_dim: int,
    seed: int,
    patience: int = 5,
    timeout: Optional[int] = None,
) -> dict:
    """
    Chạy Optuna tune cho ANN trên train/val của fold hiện tại.

    Dừng sớm theo 3 điều kiện:
    - n_trials đạt giới hạn
    - EarlyStoppingCallback: 'patience' trials liên tiếp không cải thiện
    - timeout: số giây tối đa cho cả study

    Ngoài ra dùng MedianPruner để cắt bỏ từng trial ANN ngay trong quá
    trình training nếu val PR-AUC tệ hơn median các trial đã hoàn thành.

    Returns:
        best_params (dict) để khởi tạo FraudANN.
    """
    from src.models_dl import optuna_objective_ann

    use_pos_weight = (condition == "Class-weighting")

    # MedianPruner: sau n_startup_trials đầu tiên (warmup, không prune),
    # cắt trial nếu val PR-AUC ở epoch hiện tại thấp hơn median
    # của các trial đã hoàn thành ở cùng epoch.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,   # 5 trial đầu không prune (cần warm-up)
        n_warmup_steps=5,     # 5 epoch đầu mỗi trial không prune
        interval_steps=3,     # kiểm tra mỗi 3 epoch
    )

    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
    )
    callbacks = [EarlyStoppingCallback(patience=patience)]

    study.optimize(
        lambda trial: optuna_objective_ann(
            trial,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            input_dim=input_dim,
            use_pos_weight=use_pos_weight,
            seed=seed,
        ),
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
        callbacks=callbacks,
    )

    best = study.best_trial
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    logger.info(
        "  [Optuna-ANN] %d trials (%d pruned) | best PR-AUC=%.4f | params=%s",
        len(study.trials), n_pruned, best.value, best.params,
    )
    return best.params


# ── Train & evaluate 1 fold ───────────────────────────────────────────

def run_ml_fold(
    model_name: str,
    condition: str,
    fold: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    n_trials: int,
    patience: int,
    seed: int,
    timeout: Optional[int] = None,
    run_shap: bool = True,
) -> dict:
    """
    Tune → train → evaluate → SHAP cho 1 (ML model, condition, fold).

    Returns:
        metrics dict từ evaluate_fold().
    """
    logger.info("── [ML] %s | %s | fold %d ──", model_name, condition, fold)

    # 1. Imbalance handling (chỉ trên train fold)
    X_tr_bal, y_tr_bal = apply_condition(X_train, y_train, condition=condition)

    # 2. Optuna tune trên dữ liệu đã balance
    # Truyền X_val/y_val từ outer fold — không nested CV bên trong
    best_params = tune_ml_model(
        model_name, X_tr_bal, y_tr_bal,
        X_val=X_val, y_val=y_val,
        n_trials=n_trials, patience=patience, seed=seed, timeout=timeout,
    )

    # 3. Train final model với best params
    import joblib
    model = get_model(model_name, condition=condition, y_train=y_tr_bal, custom_params=best_params)

    if model_name == "CatBoost":
        model.fit(X_tr_bal, y_tr_bal, eval_set=(X_val, y_val), verbose=0)
    elif model_name == "XGB":
        model.fit(X_tr_bal, y_tr_bal, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_tr_bal, y_tr_bal)

    model_path = f"{OUTPUTS['models']}/{model_name}_{condition}_fold{fold}.pkl"
    joblib.dump(model, model_path)

    # 4. Predict
    y_prob = model.predict_proba(X_val)
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]

    # 5. Evaluate
    metrics = evaluate_fold(y_val, y_prob, fold, model_name, condition)

    # 6. SHAP
    if run_shap:
        try:
            from src.explain import compute_shap_ml
            shap_path = f"{OUTPUTS['shap_values']}/{model_name}_{condition}_fold{fold}.npy"
            compute_shap_ml(
                model, model_name, X_val, feature_names, save_path=shap_path, seed=seed,
            )
        except Exception as e:
            logger.warning("SHAP failed for %s fold %d: %s", model_name, fold, e)

    return metrics


def run_ann_fold(
    condition: str,
    fold: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    n_trials: int,
    patience: int,
    seed: int,
    timeout: Optional[int] = None,
    run_shap: bool = True,
) -> dict:
    """
    Tune → train → evaluate → SHAP cho ANN trên 1 (condition, fold).

    Returns:
        metrics dict từ evaluate_fold().
    """
    from src.models_dl import FraudANN, train_ann

    logger.info("── [DL] ANN | %s | fold %d ──", condition, fold)
    input_dim = X_train.shape[1]

    # 1. Imbalance handling (chỉ trên train fold)
    X_tr_bal, y_tr_bal = apply_condition(X_train, y_train, condition=condition)

    # 2. Optuna tune ANN
    best_params = tune_ann_model(
        X_tr_bal, y_tr_bal, X_val, y_val,
        condition=condition,
        n_trials=n_trials,
        patience=patience,
        input_dim=input_dim,
        seed=seed,
        timeout=timeout,
    )

    # 3. Rebuild model với best params và train lại
    n_layers = best_params["n_layers"]
    units_l0 = best_params["units_l0"]
    dropout  = best_params["dropout"]
    lr       = best_params["learning_rate"]
    batch_sz = best_params["batch_size"]

    units        = [max(16, units_l0 // (2 ** i)) for i in range(n_layers)]
    dropout_rates = [dropout] * n_layers

    model = FraudANN(input_dim=input_dim, n_layers=n_layers, units=units, dropout_rates=dropout_rates)

    use_pos_weight = (condition == "Class-weighting")
    pos_weight = None
    if use_pos_weight:
        n_pos = y_tr_bal.sum()
        n_neg = len(y_tr_bal) - n_pos
        pos_weight = float(n_neg / max(n_pos, 1))

    train_result = train_ann(
        model=model,
        X_train=X_tr_bal, y_train=y_tr_bal,
        X_val=X_val, y_val=y_val,
        learning_rate=lr, batch_size=batch_sz,
        pos_weight=pos_weight, seed=seed,
    )
    logger.info(
        "  ANN trained %d epochs, best val PR-AUC=%.4f",
        train_result["epochs_trained"], train_result["best_val_prauc"],
    )

    # 4. Save model
    import torch
    torch.save(
        model.state_dict(),
        f"{OUTPUTS['models']}/ANN_{condition}_fold{fold}.pt",
    )

    # 5. Predict & evaluate
    y_prob = model.predict_proba(X_val)
    metrics = evaluate_fold(y_val, y_prob, fold, "ANN", condition)

    # 6. SHAP (approximate, DeepExplainer)
    if run_shap:
        try:
            from src.explain import compute_shap_dl
            shap_path = f"{OUTPUTS['shap_values']}/ANN_{condition}_fold{fold}.npy"
            compute_shap_dl(
                model, X_tr_bal, X_val, feature_names, save_path=shap_path, seed=seed,
            )
        except Exception as e:
            logger.warning("SHAP (ANN) failed fold %d: %s", fold, e)

    return metrics


# ── Main ──────────────────────────────────────────────────────────────

def main(
    models: list[str],
    conditions: list[str],
    n_folds: int,
    n_trials: int,
    patience: int,
    ann_patience: int,
    timeout: Optional[int],
    run_shap: bool,
    seed: int,
    run_ann: bool,
):
    set_global_seed(seed)
    logger.info("=" * 65)
    logger.info(
        "Experiment: %d models × %d conditions × %d folds = %d ML runs%s",
        len(models), len(conditions), n_folds,
        len(models) * len(conditions) * n_folds,
        f" + {len(conditions) * n_folds} ANN runs" if run_ann else "",
    )
    logger.info("=" * 65)

    # ── Load data ──
    logger.info("Loading data...")
    df = pd.read_csv("data/raw/creditcard.csv")
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    feature_names = list(df.drop(columns=["Class"]).columns)

    # ── Sinh fold indices — DÙNG CHUNG cho cả ML lẫn DL ──
    folds = get_fold_indices(y, n_splits=n_folds, random_state=seed)

    # ── Load checkpoint: resume từ chỗ dừng nếu có ──
    fold_records, done_keys = _load_checkpoint(OUTPUTS["metrics"])

    summary_records = []
    paired_records  = []

    # Rebuild prauc_store từ checkpoint để paired-test đúng sau resume
    prauc_store: dict[tuple, list[float]] = {}
    for r in fold_records:
        prauc_store.setdefault((r["model"], r["condition"]), []).append(r["PR-AUC"])

    all_model_keys = list(models)
    if run_ann:
        all_model_keys = list(models) + ["ANN"]

    for condition in conditions:
        logger.info("\n%s\nCondition: %s\n%s", "=" * 65, condition, "=" * 65)

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info("\n── Fold %d/%d ──", fold_idx, n_folds - 1)

            X_train_raw, X_val_raw = X[train_idx], X[val_idx]
            y_train, y_val         = y[train_idx], y[val_idx]

            # Scale: fit trên train fold, transform cả 2 — không leakage
            from sklearn.preprocessing import RobustScaler
            import pandas as _pd

            _feat_df  = _pd.DataFrame(X_train_raw, columns=feature_names)
            _feat_val = _pd.DataFrame(X_val_raw,   columns=feature_names)

            scaler = RobustScaler()
            scale_cols = ["Amount", "Time"]
            _feat_df[scale_cols]  = scaler.fit_transform(_feat_df[scale_cols])
            _feat_val[scale_cols] = scaler.transform(_feat_val[scale_cols])

            X_train = _feat_df.values.astype(np.float32)
            X_val   = _feat_val.values.astype(np.float32)

            # ── ML models ──
            for model_name in models:
                ck_key = (model_name, condition, fold_idx)
                if ck_key in done_keys:
                    logger.info(
                        "⏭  Skip (checkpoint): %s | %s | fold %d",
                        model_name, condition, fold_idx,
                    )
                    continue

                t0 = time.time()
                metrics = run_ml_fold(
                    model_name=model_name,
                    condition=condition,
                    fold=fold_idx,
                    X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val,
                    feature_names=feature_names,
                    n_trials=n_trials,
                    patience=patience,
                    seed=seed,
                    timeout=timeout,
                    run_shap=run_shap,
                )
                fold_records.append(metrics)
                done_keys.add(ck_key)
                prauc_store.setdefault((model_name, condition), []).append(metrics["PR-AUC"])
                _save_checkpoint(fold_records, OUTPUTS["metrics"])  # checkpoint ngay sau mỗi fold
                logger.info("  Done in %.1fs", time.time() - t0)

            # ── ANN ──
            if run_ann:
                ck_key = ("ANN", condition, fold_idx)
                if ck_key in done_keys:
                    logger.info(
                        "⏭  Skip (checkpoint): ANN | %s | fold %d",
                        condition, fold_idx,
                    )
                else:
                    t0 = time.time()
                    metrics = run_ann_fold(
                        condition=condition,
                        fold=fold_idx,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        feature_names=feature_names,
                        n_trials=n_trials,
                        patience=ann_patience,
                        seed=seed,
                        timeout=timeout,
                        run_shap=run_shap,
                    )
                    fold_records.append(metrics)
                    done_keys.add(ck_key)
                    prauc_store.setdefault(("ANN", condition), []).append(metrics["PR-AUC"])
                    _save_checkpoint(fold_records, OUTPUTS["metrics"])  # checkpoint ngay sau mỗi fold
                    logger.info("  Done in %.1fs", time.time() - t0)

        # ── Summary per (model, condition) ──
        for mkey in all_model_keys:
            fold_res = [r for r in fold_records if r["model"] == mkey and r["condition"] == condition]
            if fold_res:
                summary_records.append(summarize_folds(fold_res, mkey, condition))

        # ── Paired Wilcoxon: mọi cặp model trong cùng condition ──
        for i, ma in enumerate(all_model_keys):
            for mb in all_model_keys[i + 1:]:
                scores_a = prauc_store.get((ma, condition), [])
                scores_b = prauc_store.get((mb, condition), [])
                if len(scores_a) == len(scores_b) == n_folds:
                    paired_records.append(
                        paired_wilcoxon_test(
                            scores_a, scores_b,
                            model_a=ma, model_b=mb,
                            condition=condition,
                        )
                    )

    # ── Save all results ──
    save_summary(fold_records, summary_records, paired_records, output_dir=OUTPUTS["metrics"])

    logger.info("\n🎉 Experiment complete!")
    logger.info("  fold_results.csv  → %s", OUTPUTS["metrics"])
    logger.info("  summary.csv       → %s", OUTPUTS["metrics"])
    logger.info("  paired_tests.csv  → %s", OUTPUTS["metrics"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection: Full Experiment Pipeline (ML + DL)")
    parser.add_argument("--models",       nargs="+", default=MODEL_NAMES,
                        help="ML models to run (default: all)")
    parser.add_argument("--conditions",   nargs="+", default=list(CONDITIONS.keys()),
                        help="Imbalance conditions")
    parser.add_argument("--folds",        type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--trials",       type=int, default=20,
                        help="Max Optuna trials per model per fold (default: 20)")
    parser.add_argument("--patience",     type=int, default=7,
                        help="ML Optuna early-stop patience — stop after N trials no improve (default: 7)")
    parser.add_argument("--ann-patience", type=int, default=5,
                        help="ANN Optuna early-stop patience (default: 5, lower because pruner helps)")
    parser.add_argument("--timeout",      type=int, default=None,
                        help="Max seconds per Optuna study (e.g. 300 = 5 min). None = unlimited.")
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--no-shap",      action="store_true",
                        help="Skip SHAP computation (faster)")
    parser.add_argument("--no-ann",       action="store_true",
                        help="Skip ANN branch (ML only)")
    args = parser.parse_args()

    main(
        models=args.models,
        conditions=args.conditions,
        n_folds=args.folds,
        n_trials=args.trials,
        patience=args.patience,
        ann_patience=args.ann_patience,
        timeout=args.timeout,
        run_shap=not args.no_shap,
        seed=args.seed,
        run_ann=not args.no_ann,
    )
