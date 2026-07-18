"""
Preprocessing utilities for credit card fraud detection.
Phase 1–2 of the experiment pipeline.

- Load creditcard.csv
- Stratified 80/20 split (random_state=42)
- RobustScaler on Amount & Time only (V1–V28 already PCA-normalized)
- Persist scaler via joblib for reproducibility
"""

import os
import logging
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


# ── Data Loading & Splitting ─────────────────────────────────────────

def load_and_split(
    data_path: str = "data/raw/creditcard.csv",
    target_column: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Load the credit card dataset and perform a stratified train/test split.

    Returns:
        (X_train, X_test, y_train, y_test) as numpy arrays / pandas objects.
    """
    # Auto-detect Kaggle environment or fallback paths if data_path does not exist
    if not os.path.exists(data_path):
        found = False
        
        # 1. Try recursive search inside /kaggle/input first (most robust on Kaggle)
        kaggle_input_dir = "/kaggle/input"
        if os.path.exists(kaggle_input_dir):
            for root, dirs, files in os.walk(kaggle_input_dir):
                for f in files:
                    if f.lower() == "creditcard.csv":
                        data_path = os.path.join(root, f)
                        logger.info("Auto-detected dataset via recursive search at: %s", data_path)
                        found = True
                        break
                if found:
                    break
                    
        # 2. Try static fallback paths if not found recursively
        if not found:
            kaggle_paths = [
                "/kaggle/input/creditcard/creditcard.csv",
                "/kaggle/input/creditcardfraud/creditcard.csv",
                "/kaggle/input/creditcard-fraud-detection/creditcard.csv",
                "../input/creditcard/creditcard.csv",
                "../input/creditcardfraud/creditcard.csv"
            ]
            for kp in kaggle_paths:
                if os.path.exists(kp):
                    logger.info("Auto-detected dataset at: %s", kp)
                    data_path = kp
                    found = True
                    break
                    
        if not found:
            raise FileNotFoundError(
                f"Dataset not found at '{data_path}'. "
                "Download from Kaggle (mlg-ulb/creditcardfraud) or place it in data/raw/creditcard.csv"
            )

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows × %d columns", *df.shape)

    # Basic sanity checks
    assert target_column in df.columns, f"Target column '{target_column}' not found"
    logger.info(
        "Class distribution:\n%s",
        df[target_column].value_counts().to_string(),
    )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


# ── Feature Scaling ──────────────────────────────────────────────────

_SCALE_COLS = ["Amount", "Time"]


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scale_cols: Optional[list] = None,
    save_path: Optional[str] = None,
) -> tuple:
    """
    Apply RobustScaler to *only* Amount & Time.
    V1–V28 are kept as-is (already PCA-normalized).

    Args:
        X_train: Training features.
        X_test:  Test features.
        scale_cols: Columns to scale (default: Amount, Time).
        save_path:  If provided, persist the fitted scaler via joblib.

    Returns:
        (X_train_scaled, X_test_scaled) as DataFrames.
    """
    cols = scale_cols or _SCALE_COLS
    cols = [c for c in cols if c in X_train.columns]

    if not cols:
        logger.warning("No columns to scale — returning data unchanged.")
        return X_train.copy(), X_test.copy()

    scaler = RobustScaler()

    X_train_out = X_train.copy()
    X_test_out = X_test.copy()

    X_train_out[cols] = scaler.fit_transform(X_train[cols])
    X_test_out[cols] = scaler.transform(X_test[cols])

    logger.info("Scaled columns %s with RobustScaler (fit on train only)", cols)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump(scaler, save_path)
        logger.info("Scaler saved to %s", save_path)

    return X_train_out, X_test_out


def load_scaler(path: str) -> RobustScaler:
    """Load a previously saved scaler."""
    scaler = joblib.load(path)
    logger.info("Scaler loaded from %s", path)
    return scaler
