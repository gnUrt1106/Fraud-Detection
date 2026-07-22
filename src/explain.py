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
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

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
    eval_module = getattr(model, "network", model)
    explainer = shap.DeepExplainer(eval_module, background)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    shap_values = explainer.shap_values(X_val_tensor)

    # DeepExplainer có thể trả về list hoặc array 3D
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if not isinstance(shap_values, np.ndarray):
        shap_values = np.array(shap_values)

    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)

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
    vals = np.abs(shap_values).mean(axis=0)
    sorted_pairs = sorted(zip(feature_names, vals), key=lambda x: x[1], reverse=True)
    return dict(sorted_pairs)


# ── Popular SHAP Visualizations ──────────────────────────────────────

import matplotlib.pyplot as plt

def _ensure_2d_shap(shap_values: np.ndarray) -> np.ndarray:
    """Ensure SHAP values array is 2D (n_samples, n_features) for binary classification class 1."""
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # If shape is (N, M, 2), take class 1 (index 1)
        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]
        elif shap_values.shape[-1] == 1:
            shap_values = shap_values.squeeze(-1)
    return shap_values


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_val: np.ndarray,
    feature_names: list[str],
    save_path: Optional[str] = None,
    max_display: int = 15,
) -> plt.Figure:
    """1. SHAP Beeswarm Plot (Dot summary plot showing directionality)."""
    shap_values = _ensure_2d_shap(shap_values)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap_obj = shap.Explanation(
        values=shap_values,
        data=X_val,
        feature_names=feature_names,
    )
    shap.plots.beeswarm(shap_obj, max_display=max_display, plot_size=(8, 5), show=False)
    plt.title("SHAP Beeswarm Plot — Feature Impact & Values", fontsize=12, pad=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved SHAP Beeswarm plot to %s", save_path)
    return fig


def plot_shap_summary_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    save_path: Optional[str] = None,
    max_display: int = 15,
) -> plt.Figure:
    """2. SHAP Global Feature Importance Bar Chart."""
    shap_values = _ensure_2d_shap(shap_values)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    shap_obj = shap.Explanation(
        values=shap_values,
        feature_names=feature_names,
    )
    shap.plots.bar(shap_obj, max_display=max_display, show=False)
    plt.title("SHAP Global Feature Importance", fontsize=12, pad=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved SHAP Bar plot to %s", save_path)
    return fig


def plot_shap_waterfall(
    shap_values: np.ndarray,
    X_val: np.ndarray,
    feature_names: list[str],
    sample_idx: int = 0,
    save_path: Optional[str] = None,
    base_value: float = 0.0,
) -> plt.Figure:
    """3. SHAP Waterfall Plot for single transaction explanation."""
    shap_values = _ensure_2d_shap(shap_values)
    fig, ax = plt.subplots(figsize=(8, 5))
    exp = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=base_value,
        data=X_val[sample_idx],
        feature_names=feature_names,
    )
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.title(f"SHAP Waterfall — Local Explanation (Sample #{sample_idx})", fontsize=12, pad=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved SHAP Waterfall plot to %s", save_path)
    return fig


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_val: np.ndarray,
    feature_names: list[str],
    feature_name: str = "V14",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """4. SHAP Dependence Scatter Plot."""
    shap_values = _ensure_2d_shap(shap_values)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    feat_idx = feature_names.index(feature_name) if feature_name in feature_names else 0
    shap.dependence_plot(
        feat_idx,
        shap_values,
        X_val,
        feature_names=feature_names,
        ax=ax,
        show=False,
    )
    plt.title(f"SHAP Dependence Plot — {feature_name}", fontsize=12, pad=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved SHAP Dependence plot to %s", save_path)
    return fig


def plot_shap_decision(
    shap_values: np.ndarray,
    X_val: np.ndarray,
    feature_names: list[str],
    save_path: Optional[str] = None,
    base_value: float = 0.0,
    n_samples: int = 20,
) -> plt.Figure:
    """5. SHAP Decision Plot for multi-instance cumulative path analysis."""
    shap_values = _ensure_2d_shap(shap_values)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.decision_plot(
        base_value,
        shap_values[:n_samples],
        X_val[:n_samples],
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"SHAP Decision Plot — Path Analysis (Top {n_samples} Samples)", fontsize=12, pad=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info("Saved SHAP Decision plot to %s", save_path)
    return fig
