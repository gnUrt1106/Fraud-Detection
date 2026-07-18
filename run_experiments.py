#!/usr/bin/env python3
"""
Run the full experiment matrix.

4 models × 3 conditions = 12 configurations.
Pipeline per configuration:
    Preprocess → Imbalance → Train → Evaluate

Usage:
    python run_experiments.py
    python run_experiments.py --models RF XGB  # Subset of models
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np

# Ensure importability
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocess import load_and_split, scale_features
from src.imbalance import apply_condition, CONDITIONS
from src.models import get_model, MODEL_NAMES
from src.evaluate import evaluate_model, save_result_json, plot_feature_importance
from src.visualize import (
    plot_performance_comparison,
    plot_results_heatmap,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Directories ──────────────────────────────────────────────────────

OUTPUTS_DIR = "outputs"
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")

for d in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)


def run_single_config(
    model_name: str,
    condition: str,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
):
    """Run one (model, condition) configuration through the full pipeline."""
    config_label = f"{model_name}_{condition}"
    logger.info("=" * 60)
    logger.info("CONFIG: %s | %s", model_name, condition)
    logger.info("=" * 60)

    # ── Phase 3: Imbalance handling ──
    X_train_cond, y_train_cond = apply_condition(
        X_train_scaled, y_train, condition=condition,
    )

    # ── Phase 4: Train ──
    model = get_model(
        model_name, condition=condition, y_train=y_train,
    )

    if model_name == "CatBoost":
        model.fit(
            X_train_cond, y_train_cond,
            eval_set=(X_test_scaled, y_test),
            verbose=0,
        )
    elif model_name == "XGB":
        model.fit(
            X_train_cond, y_train_cond,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )
    else:
        model.fit(X_train_cond, y_train_cond)

    # Save trained model
    model_path = os.path.join(MODELS_DIR, f"{config_label}.pkl")
    joblib.dump(model, model_path)
    logger.info("Model saved to %s", model_path)

    # ── Phase 5: Evaluate performance ──
    y_prob = model.predict_proba(X_test_scaled)
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]

    perf_metrics = evaluate_model(
        y_test, y_prob,
        save_dir=os.path.join(FIGURES_DIR, config_label),
    )

    # Feature importance (tree-based models)
    if hasattr(model, "feature_importances_"):
        feature_names = list(X_train_scaled.columns) if hasattr(X_train_scaled, "columns") else None
        if feature_names:
            plot_feature_importance(
                model.feature_importances_, feature_names,
                save_dir=os.path.join(FIGURES_DIR, config_label),
            )

    # Save combined result
    save_result_json(model_name, condition, perf_metrics, save_dir=RESULTS_DIR)

    return perf_metrics


def main(models=None, skip_viz=False):
    """Run the full experiment matrix."""
    models = models or MODEL_NAMES
    conditions = list(CONDITIONS.keys())

    total = len(models) * len(conditions)
    logger.info(
        "Starting experiment matrix: %d models × %d conditions = %d configs",
        len(models), len(conditions), total,
    )

    # ── Phase 1–2: Preprocess (shared across all configs) ──
    X_train, X_test, y_train, y_test = load_and_split()
    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test,
        save_path=os.path.join(OUTPUTS_DIR, "scaler.pkl"),
    )

    all_perf = {}
    current = 0

    for model_name in models:
        for condition in conditions:
            current += 1
            logger.info(
                "\n>>> Run %d/%d: %s + %s <<<", current, total, model_name, condition,
            )
            try:
                perf = run_single_config(
                    model_name, condition,
                    X_train_scaled, X_test_scaled,
                    y_train, y_test,
                )
                all_perf[(model_name, condition)] = perf
            except Exception as e:
                logger.error("FAILED %s + %s: %s", model_name, condition, e)
                import traceback
                traceback.print_exc()

    # ── Phase 6: Visualization ──
    if not skip_viz:
        logger.info("\n" + "=" * 60)
        logger.info("Generating summary visualizations...")
        logger.info("=" * 60)

        try:
            import pandas as pd

            # Build summary DataFrame
            rows = []
            for (m, c), perf in all_perf.items():
                row = {
                    "Model": m, "Condition": c,
                    "PR-AUC": perf.get("PR-AUC", 0),
                    "F1": perf.get("F1", 0),
                    "Recall": perf.get("Recall", 0),
                    "ROC-AUC": perf.get("ROC-AUC", 0),
                }
                rows.append(row)

            summary_df = pd.DataFrame(rows)
            summary_df.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
            logger.info("Summary CSV saved.")

            plot_performance_comparison(summary_df)
            plot_results_heatmap(summary_df, metric="PR-AUC",
                                 save_path="outputs/figures/prauc_heatmap.png")

        except Exception as e:
            logger.error("Visualization failed: %s", e)
            import traceback
            traceback.print_exc()

    logger.info("\n🎉 All experiments complete! Results in: %s", RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment matrix")
    parser.add_argument(
        "--models", nargs="+", default=None,
        choices=MODEL_NAMES,
        help=f"Models to evaluate (default: all {MODEL_NAMES})",
    )
    parser.add_argument(
        "--skip-viz", action="store_true",
        help="Skip summary visualization generation",
    )
    args = parser.parse_args()
    main(models=args.models, skip_viz=args.skip_viz)
