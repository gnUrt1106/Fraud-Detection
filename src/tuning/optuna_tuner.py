"""
Optuna hyperparameter tuning for all experiment models.

Supports: RF, XGBoost, CatBoost, Logistic Regression.
Objective: maximize PR-AUC via 5-fold Stratified CV.
"""

import logging

import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Suppress Optuna info spam
optuna.logging.set_verbosity(optuna.logging.WARNING)


class EarlyStoppingCallback:
    """Callback to stop Optuna study early if no improvement is made."""
    def __init__(self, patience: int):
        self.patience = patience
        self.best_value = None
        self.no_improvement_trials = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        if study.best_value is None:
            return
        
        if self.best_value is None or study.best_value > self.best_value:
            self.best_value = study.best_value
            self.no_improvement_trials = 0
        else:
            self.no_improvement_trials += 1
            if self.no_improvement_trials >= self.patience:
                logger.info(
                    "Early stopping study: No improvement for %d trials. Best PR-AUC: %.4f",
                    self.patience, self.best_value
                )
                study.stop()


def objective(trial, X, y, model_name):
    """Single Optuna trial — 5-fold CV, returns mean PR-AUC."""
    pr_auc_scores = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Detect GPU availability safely
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    for train_idx, val_idx in cv.split(X, y):
        if hasattr(X, "iloc"):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

        if model_name == "XGBClassifier":
            param = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            
            if use_gpu:
                param["device"] = "cuda"
                param["tree_method"] = "hist"
            else:
                param["tree_method"] = "hist"
                param["n_jobs"] = -1

            neg = np.sum(np.asarray(y_train) == 0)
            pos = np.sum(np.asarray(y_train) == 1)
            if pos > 0:
                param["scale_pos_weight"] = trial.suggest_float(
                    "scale_pos_weight", 1.0, neg / pos,
                )
            model = XGBClassifier(**param, random_state=42, eval_metric="aucpr", early_stopping_rounds=25)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            y_prob = model.predict_proba(X_val)[:, 1]

        elif model_name == "RandomForestClassifier":
            param = {
                "n_jobs": -1,
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 25),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", "balanced_subsample", None],
                ),
            }
            model = RandomForestClassifier(**param, random_state=42)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]

        elif model_name == "CatBoostClassifier":
            param = {
                "iterations": trial.suggest_int("iterations", 200, 800),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
                "auto_class_weights": trial.suggest_categorical(
                    "auto_class_weights", ["Balanced", "SqrtBalanced", None],
                ),
                "verbose": 0,
            }
            
            if use_gpu:
                param["task_type"] = "GPU"

            model = CatBoostClassifier(**param, random_seed=42, early_stopping_rounds=25)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                verbose=0
            )
            y_prob = model.predict_proba(X_val)[:, 1]

        elif model_name == "LogisticRegression":
            # l1_ratio: 0.0 = L2, 1.0 = L1 (replaces deprecated penalty=)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
            param = {
                "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
                "l1_ratio": l1_ratio,
                "solver": "saga",   # saga supports full elastic-net / l1 / l2
                "max_iter": 5000,   # saga converges slowly on 284k rows
                "tol": 1e-2,
                "class_weight": trial.suggest_categorical(
                    "class_weight", ["balanced", None],
                ),
            }
            model = LogisticRegression(**param, random_state=42)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_val)[:, 1]

        else:
            raise ValueError(f"Tuning not supported for {model_name}")

        score = average_precision_score(y_val, y_prob)
        pr_auc_scores.append(score)

    return np.mean(pr_auc_scores)


def run_optimization(X, y, model_name="XGBClassifier", n_trials=20, patience=10):
    """
    Run Optuna study to find optimal hyperparameters.

    Returns:
        dict of best parameters.
    """
    study = optuna.create_study(
        direction="maximize",
        study_name=f"{model_name}_PR_AUC",
    )

    # Detect GPU availability safely for logging
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    logger.info(
        "Starting Optuna tuning for %s (%d trials, patience=%d, GPU=%s)...", 
        model_name, n_trials, patience, "Yes" if use_gpu else "No"
    )
    
    callbacks = [EarlyStoppingCallback(patience=patience)]
    
    def logging_callback(study, trial):
        val_str = f"{trial.value:.4f}" if trial.value is not None else "Pruned/Failed"
        best_val_str = f"{study.best_value:.4f}" if study.best_value is not None else "None"
        logger.info(
            "Trial %d done | PR-AUC: %s | Best: %s",
            trial.number, val_str, best_val_str
        )
    
    callbacks.append(logging_callback)
    
    study.optimize(
        lambda trial: objective(trial, X, y, model_name),
        n_trials=n_trials,
        n_jobs=1,
        callbacks=callbacks,
    )

    logger.info("Finished %d trials", len(study.trials))
    best = study.best_trial
    logger.info("Best PR-AUC: %.4f", best.value)
    logger.info("Best params:")
    for key, value in best.params.items():
        logger.info("  %s: %s", key, value)

    return best.params
