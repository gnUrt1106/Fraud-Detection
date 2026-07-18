"""
SHAP Explainability utilities.

Hỗ trợ 3 loại explainer tương ứng với 2 nhánh model:
- TreeExplainer   → XGBoost, CatBoost, RandomForest  (exact Shapley)
- LinearExplainer → LogisticRegression                 (exact Shapley)
- DeepExplainer   → ANN (PyTorch)                      (approximate)

QUAN TRỌNG — Giới hạn so sánh:
  SHAP values từ TreeExplainer/LinearExplainer là exact Shapley values.
  SHAP values từ DeepExplainer là approximate (gradient-based).
  Không so sánh giá trị SHAP tuyệt đối giữa nhánh cây và nhánh ANN
  như thể chúng cùng thang đo — chỉ so sánh ranking feature importance
  trong nội bộ từng nhánh.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import shap

logger = logging.getLogger(__name__)


# ── Constant ──────────────────────────────────────────────────────────

# Số background samples cho DeepExplainer / GradientExplainer
# Đủ nhỏ để chạy nhanh, đủ lớn để giảm variance approximation.
N_BACKGROUND = 100


# ── Public API ────────────────────────────────────────────────────────

def compute_shap_ml(
    model,
    model_name: str,
    X_val: np.ndarray,
    feature_names: list[str],
    save_path: Optional[str] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Tính SHAP values cho model ML (cây hoặc tuyến tính).

    Args:
        model:        Fitted sklearn-compatible model.
        model_name:   Tên model ('RF', 'XGB', 'CatBoost', 'LR').
        X_val:        Validation features (numpy array).
        feature_names: Danh sách tên features.
        save_path:    Nếu đặt, lưu array .npy vào path này.
        seed:         Seed cho LinearExplainer (không dùng đến TreeExplainer).

    Returns:
        shap_values: numpy array shape (n_samples, n_features).
    """
    logger.info("Computing SHAP for %s (%d samples)...", model_name, len(X_val))

    if model_name in ("RF", "XGB", "CatBoost"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        # TreeExplainer binary classification trả về list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    elif model_name == "LR":
        explainer = shap.LinearExplainer(
            model,
            X_val,
            feature_perturbation="interventional",
        )
        shap_values = explainer.shap_values(X_val)

    else:
        raise ValueError(
            f"Unknown ML model '{model_name}'. "
            "Dùng compute_shap_dl() cho ANN."
        )

    logger.info(
        "SHAP done for %s — mean |SHAP|: %s",
        model_name,
        dict(zip(feature_names, np.abs(shap_values).mean(axis=0).round(4))),
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, shap_values)
        logger.info("SHAP values saved to %s", save_path)

    return shap_values


def compute_shap_dl(
    model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    feature_names: list[str],
    save_path: Optional[str] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Tính SHAP values (approximate) cho ANN (PyTorch).

    Dùng shap.DeepExplainer với background là N_BACKGROUND sample
    từ tập train, chọn bằng seed cố định để reproducible.

    ⚠ Approximate — không so sánh tuyệt đối với TreeExplainer values.

    Args:
        model:        Fitted PyTorch model (eval mode).
        X_train:      Training features dùng để lấy background samples.
        X_val:        Validation features để tính SHAP.
        feature_names: Danh sách tên features.
        save_path:    Nếu đặt, lưu array .npy vào path này.
        seed:         Seed để chọn background samples.

    Returns:
        shap_values: numpy array shape (n_samples, n_features).
    """
    import torch

    logger.info(
        "Computing SHAP (DeepExplainer, approximate) for ANN (%d val samples)...",
        len(X_val),
    )

    # Background samples: cố định seed để reproducible
    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_train), size=min(N_BACKGROUND, len(X_train)), replace=False)
    background = torch.tensor(X_train[bg_idx], dtype=torch.float32)

    model.eval()
    explainer = shap.DeepExplainer(model, background)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    shap_values = explainer.shap_values(X_val_tensor)

    # DeepExplainer có thể trả về list; lấy class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if isinstance(shap_values, np.ndarray) is False:
        shap_values = np.array(shap_values)

    logger.info(
        "SHAP (ANN) done — mean |SHAP|: %s",
        dict(zip(feature_names, np.abs(shap_values).mean(axis=0).round(4))),
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, shap_values)
        logger.info("SHAP values (ANN) saved to %s", save_path)

    return shap_values


def mean_abs_shap(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Tính mean |SHAP| per feature — dùng để tạo summary bar chart.

    Returns:
        Dict {feature_name: mean_abs_shap}, sorted descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    result = dict(zip(feature_names, mean_abs))
    return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
