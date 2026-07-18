"""
Model definitions for experiments.
Phase 4 of the pipeline.

Lineup:
    - Random Forest          — Primary (high performance ensemble)
    - XGBoost                — Primary (best PR-AUC)
    - CatBoost               — Primary (gradient boosting)
    - Logistic Regression    — Baseline linear (cross-family comparison)

Hyperparameters are fixed to ensure reproducibility.
Class-weight params are ONLY used for condition C0 (no resampling).
"""

import logging
from typing import Optional

import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ── Default hyperparameters ─────────────────

_BASE_PARAMS = {
    "RF": dict(
        n_estimators=185,
        max_depth=17,
        min_samples_split=18,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
    "XGB": dict(
        n_estimators=244,
        learning_rate=0.085084,
        max_depth=8,
        subsample=0.676336,
        colsample_bytree=0.707718,
        min_child_weight=1,
        scale_pos_weight=27.153399,
        random_state=42,
        eval_metric="aucpr",
    ),
    "CatBoost": dict(
        iterations=535,
        depth=5,
        learning_rate=0.037576,
        l2_leaf_reg=0.321827,
        random_seed=42,
        verbose=0,
    ),
    "LR": dict(
        C=0.000382,
        penalty="l2",
        max_iter=1000,
        random_state=42,
    ),
}


# ── Class-weight additions for C0 ───────────────────────────────────

def _add_class_weights(name: str, params: dict, y_train=None) -> dict:
    """Inject class-weight params for C0 baseline condition."""
    p = params.copy()
    if name == "RF":
        p["class_weight"] = "balanced"
    elif name == "XGB":
        # Keep tuned scale_pos_weight if present, otherwise calculate based on y_train
        if "scale_pos_weight" not in p:
            if y_train is not None:
                neg = np.sum(y_train == 0)
                pos = np.sum(y_train == 1)
                p["scale_pos_weight"] = neg / max(pos, 1)
                logger.info("XGB scale_pos_weight = %.2f", p["scale_pos_weight"])
            else:
                p["scale_pos_weight"] = 577.0  # approximate ULB ratio
        else:
            logger.info("XGB keeping tuned scale_pos_weight = %.6f", p["scale_pos_weight"])
    elif name == "CatBoost":
        p["auto_class_weights"] = "Balanced"
    elif name == "LR":
        p["class_weight"] = "balanced"
    return p


# ── Factory ──────────────────────────────────────────────────────────

_CONSTRUCTORS = {
    "RF": RandomForestClassifier,
    "XGB": XGBClassifier,
    "CatBoost": CatBoostClassifier,
    "LR": LogisticRegression,
}


def get_model(
    name: str,
    condition: str = "C0",
    y_train=None,
    custom_params: Optional[dict] = None,
):
    """
    Build a model instance with the appropriate hyperparameters.

    Args:
        name: One of 'RF', 'XGB', 'CatBoost', 'LR'.
        condition: 'C0' (class-weight), 'C1' (SMOTE), 'C2' (SMOTE-ENN).
        y_train: Training labels — needed for C0 to compute scale_pos_weight.
        custom_params: Override default hyperparameters (e.g., from Optuna).

    Returns:
        Instantiated sklearn/xgb/catboost model.
    """
    if name not in _CONSTRUCTORS:
        raise ValueError(
            f"Unknown model '{name}'. Choose from {list(_CONSTRUCTORS.keys())}"
        )

    params = (custom_params or _BASE_PARAMS[name]).copy()

    # Class-weighting → inject class-weight params; SMOTE/SMOTE-ENN → data already balanced
    if condition.lower() == "class-weighting":
        params = _add_class_weights(name, params, y_train)
        logger.info("%s [Class-weighting] — class-weighting enabled", name)
    else:
        logger.info("%s [%s] — no class-weighting (data resampled)", name, condition)

    model = _CONSTRUCTORS[name](**params)
    logger.info("Created %s with params: %s", name, params)
    return model


def get_model_type(name: str) -> str:
    """Return model type classification: 'tree' or 'linear'."""
    if name in ("RF", "XGB", "CatBoost"):
        return "tree"
    elif name == "LR":
        return "linear"
    else:
        raise ValueError(f"Unknown model '{name}'")


MODEL_NAMES = list(_CONSTRUCTORS.keys())
