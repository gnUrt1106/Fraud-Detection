#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning CLI.

Supports: XGBClassifier, RandomForestClassifier, CatBoostClassifier, LogisticRegression.

Usage:
    python tune_hyperparams.py --model XGBClassifier --trials 50
    python tune_hyperparams.py --model CatBoostClassifier --trials 30
    python tune_hyperparams.py --model LogisticRegression --trials 20
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is importable
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocess import load_and_split, scale_features
from src.tuning.optuna_tuner import run_optimization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_MODELS = [
    "RandomForestClassifier",
    "XGBClassifier",
    "CatBoostClassifier",
    "LogisticRegression",
]

GPU_MODELS = ["XGBClassifier", "CatBoostClassifier"]
CPU_MODELS = ["RandomForestClassifier", "LogisticRegression"]


def main(n_trials: int, model_name: str, data_path: str, patience: int):
    if model_name == "all":
        models_to_tune = SUPPORTED_MODELS
    elif model_name == "gpu":
        models_to_tune = GPU_MODELS
    elif model_name == "cpu":
        models_to_tune = CPU_MODELS
    else:
        models_to_tune = [model_name]
    
    logger.info("Loading data for hyperparameter tuning...")
    try:
        X_train, X_test, y_train, y_test = load_and_split(data_path=data_path)
    except Exception as e:
        logger.error("Failed to load data: %s", str(e))
        return

    X_train_scaled, _ = scale_features(X_train, X_test)
    
    all_best_params = {}

    for name in models_to_tune:
        logger.info("\n" + "=" * 50)
        logger.info("🚀 Starting Optuna tuning on %s...", name)
        logger.info("=" * 50)
        
        try:
            best_params = run_optimization(
                X_train_scaled, y_train,
                model_name=name,
                n_trials=n_trials,
                patience=patience,
            )
            all_best_params[name] = best_params
        except Exception as e:
            logger.error("Error tuning %s: %s", name, str(e))

    logger.info("\n" + "=" * 50)
    logger.info("🎉 ALL OPTIMIZATION TRIALS COMPLETE 🎉")
    logger.info("=" * 50)
    
    for name, params in all_best_params.items():
        logger.info("Best parameters for %s:", name)
        for key, value in params.items():
            if isinstance(value, float):
                logger.info("    %s: %.6f", key, value)
            else:
                logger.info("    %s: %s", key, value)
        logger.info("-" * 30)
    
    logger.info("=" * 50)
    logger.info(
        "TIP: Update configs/default.yaml or src/models.py with these best parameters."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Optuna Hyperparameter Tuning for experiment models",
    )
    parser.add_argument(
        "--model", type=str, default="XGBClassifier",
        choices=SUPPORTED_MODELS + ["all", "gpu", "cpu"],
        help="Model to tune: specific model, 'all', 'gpu' (XGB & CatBoost), or 'cpu' (RF & LR)",
    )
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Number of Optuna trials per model",
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience for Optuna (number of trials without improvement)",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/raw/creditcard.csv",
        help="Path to creditcard.csv dataset",
    )
    args = parser.parse_args()
    main(args.trials, args.model, args.data_path, args.patience)
