"""
Model evaluation utilities.
Phase 5 of the pipeline.

Primary metrics: PR-AUC, Recall
Secondary: F1, ROC-AUC (reference only — inflated on imbalanced data)

Includes threshold optimization via precision-recall curve,
confusion matrix plotting, and experiment result logging.
"""

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ── Beautiful global theme ───────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})


# ── Threshold Optimization ───────────────────────────────────────────

def find_optimal_threshold(y_true, y_prob) -> float:
    """Find the threshold that maximizes F1-Score on the PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    idx = np.argmax(f1_scores)
    threshold = thresholds[idx] if idx < len(thresholds) else 0.5
    logger.info(
        "Optimal threshold (max F1): %.4f → F1=%.4f", threshold, f1_scores[idx]
    )
    return threshold


# ── Core Evaluation ──────────────────────────────────────────────────

def evaluate_model(y_true, y_prob, threshold: float = None, save_dir: str = None):
    """
    Compute performance metrics.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities (class 1).
        threshold: Decision threshold. If None, uses optimal F1 threshold.
        save_dir: If provided, save plots and JSON to this directory.

    Returns:
        dict with PR-AUC, ROC-AUC, F1, Recall, Precision, threshold, CM.
    """
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_prob)

    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    metrics = {
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "F1": float(f1_score(y_true, y_pred)),
        "Recall": float(recall_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred)),
        "Threshold": float(threshold),
    }

    logger.info("── Evaluation Metrics ──")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    cm = confusion_matrix(y_true, y_pred)
    metrics["CM"] = cm.tolist()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _plot_confusion_matrix(y_true, y_pred, save_dir)
        _plot_pr_roc_curves(y_true, y_prob, metrics, save_dir)
        _plot_metrics_table(metrics, save_dir)

    return metrics


# ── Plotting Utilities ───────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
        annot_kws={"size": 16, "weight": "bold"},
        linewidths=1, linecolor="white",
    )
    plt.xlabel("Predicted Label", fontweight="bold", labelpad=10)
    plt.ylabel("Actual Label", fontweight="bold", labelpad=10)
    plt.title("Confusion Matrix", pad=15, fontweight="bold")
    plt.xticks([0.5, 1.5], ["Normal", "Fraud"])
    plt.yticks([0.5, 1.5], ["Normal", "Fraud"], va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved to %s", save_dir)


def _plot_pr_roc_curves(y_true, y_prob, metrics, save_dir):
    """PR and ROC curves side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # PR Curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = metrics["PR-AUC"]
    axes[0].plot(rec, prec, color="#1f77b4", lw=2,
                 label=f"PR Curve (AUC = {pr_auc:.3f})")
    axes[0].fill_between(rec, prec, alpha=0.1, color="#1f77b4")
    axes[0].set_xlabel("Recall", fontweight="bold")
    axes[0].set_ylabel("Precision", fontweight="bold")
    axes[0].set_title("Precision-Recall Curve", fontweight="bold")
    axes[0].legend(loc="lower left")
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = metrics["ROC-AUC"]
    axes[1].plot(fpr, tpr, color="#ff7f0e", lw=2,
                 label=f"ROC Curve (AUC = {roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="#ff7f0e")
    axes[1].set_xlabel("False Positive Rate", fontweight="bold")
    axes[1].set_ylabel("True Positive Rate", fontweight="bold")
    axes[1].set_title("ROC Curve", fontweight="bold")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curves.png"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("PR & ROC curves saved to %s", save_dir)


def _plot_metrics_table(metrics, save_dir):
    """Render metrics as a styled table image."""
    display_metrics = {k: v for k, v in metrics.items() if k not in ("CM", "Threshold")}
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("tight")
    ax.axis("off")

    data = [[k, f"{v:.4f}"] for k, v in display_metrics.items()]
    df = pd.DataFrame(data, columns=["Metric", "Score"])

    table = ax.table(
        cellText=df.values, colLabels=df.columns,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#4A4A4A")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F2F2")

    plt.title("Evaluation Metrics Summary", fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "metrics_table.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importances, feature_names, save_dir, top_n=20):
    """Plot top-N feature importances (bar chart)."""
    if importances is None or feature_names is None:
        logger.warning("No feature importances provided — skipping.")
        return

    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances", fontweight="bold")
    sns.barplot(x=list(range(len(indices))), y=importances[indices], palette="viridis")

    if isinstance(feature_names, (list, np.ndarray)):
        sorted_names = [feature_names[i] for i in indices]
    elif hasattr(feature_names, "columns"):
        sorted_names = feature_names.columns[indices]
    else:
        sorted_names = [str(i) for i in indices]

    plt.xticks(range(len(indices)), sorted_names, rotation=45, ha="right", fontweight="bold")
    plt.ylabel("Importance Score", fontweight="bold")
    plt.xlabel("Features", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Feature importance plot saved to %s", save_dir)


# ── Result Logging ───────────────────────────────────────────────────

def save_result_json(
    model_name: str,
    condition: str,
    metrics: dict,
    save_dir: str = "outputs/results",
):
    """Persist a single experiment result as JSON."""
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{model_name}_{condition}.json"
    path = os.path.join(save_dir, fname)

    result = {
        "model": model_name,
        "condition": condition,
        **{k: v for k, v in metrics.items() if k != "CM"},
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Result saved to %s", path)


