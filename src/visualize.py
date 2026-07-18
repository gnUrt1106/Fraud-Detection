"""
Visualization utilities.
Phase 6 of the pipeline — analysis & reporting.

Plots:
    - Performance comparison bar chart
    - Results heatmap (model × condition)
"""

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Theme ────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.titlesize": 16,
})

# Color palette for models
MODEL_COLORS = {
    "RF": "#2ecc71",
    "XGB": "#e74c3c",
    "CatBoost": "#3498db",
    "LR": "#9b59b6",
}

CONDITION_MARKERS = {
    "Class-weighting": "o",
    "SMOTE": "s",
    "SMOTE-ENN": "D",
}


# ── Performance Comparison Bar Chart ─────────────────────────────────

def plot_performance_comparison(
    summary_df: pd.DataFrame,
    save_path: str = "outputs/figures/performance_comparison.png",
):
    """
    Grouped bar chart: PR-AUC / F1 / Recall for each configuration.
    """
    df_melted = summary_df.melt(
        id_vars=["Model", "Condition"],
        value_vars=["PR-AUC", "F1", "Recall"],
        var_name="Metric", value_name="Score",
    )
    df_melted["Config"] = df_melted["Model"] + " (" + df_melted["Condition"] + ")"

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(
        data=df_melted, x="Score", y="Config", hue="Metric",
        palette="Set2", ax=ax,
    )
    ax.set_title("Performance Comparison (sorted by PR-AUC)", fontweight="bold", pad=15)
    ax.set_xlabel("Score", fontweight="bold")
    ax.set_ylabel("Configuration", fontweight="bold")
    ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Performance comparison saved to %s", save_path)


# ── Results Heatmap ──────────────────────────────────────────────────

def plot_results_heatmap(
    summary_df: pd.DataFrame,
    metric: str = "PR-AUC",
    save_path: str = "outputs/figures/prauc_heatmap.png",
):
    """
    Heatmap of a metric across Model × Condition.
    """
    pivot = summary_df.pivot_table(
        index="Model", columns="Condition", values=metric, aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="YlGnBu",
        linewidths=0.5, ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(f"{metric} by Model × Condition", fontweight="bold", pad=15)
    ax.set_ylabel("Model", fontweight="bold")
    ax.set_xlabel("Condition", fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("%s heatmap saved to %s", metric, save_path)


# ── Summary table loader ────────────────────────────────────────────

def load_all_results(results_dir: str = "outputs/results") -> pd.DataFrame:
    """Load all JSON result files into a summary DataFrame."""
    rows = []
    if not os.path.exists(results_dir):
        logger.warning("Results directory %s does not exist.", results_dir)
        return pd.DataFrame()

    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                rows.append(json.load(f))

    return pd.DataFrame(rows)
