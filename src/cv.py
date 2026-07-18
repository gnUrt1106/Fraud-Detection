"""
Cross-validation utilities.

Cung cấp một bộ StratifiedKFold fold indices dùng chung cho cả nhánh ML và DL,
đảm bảo so sánh paired giữa 2 nhánh là công bằng.

Design decisions:
- Seed cố định một chỗ, truyền vào từ run_experiment.py — không set riêng lẻ ở đây.
- Fold indices được sinh ra một lần và truyền đi — không sinh lại nhiều lần.
- Stratified để giữ tỷ lệ fraud/non-fraud trong mỗi fold.
"""

import logging
from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def get_fold_indices(
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Sinh ra danh sách (train_idx, val_idx) cho StratifiedKFold.

    Đây là fold indices DÙNG CHUNG cho cả nhánh ML lẫn DL.
    Gọi hàm này một lần duy nhất ở đầu run_experiment.py,
    truyền kết quả xuống cho mọi model.

    Args:
        y: Nhãn (0/1), dùng để stratify.
        n_splits: Số fold (mặc định 5).
        random_state: Seed cố định — phải khớp với global seed.

    Returns:
        List of (train_indices, val_indices) tuples, length = n_splits.
    """
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds = list(skf.split(np.zeros(len(y)), y))
    logger.info(
        "Generated %d StratifiedKFold folds (seed=%d, fraud_rate=%.4f%%)",
        n_splits,
        random_state,
        100.0 * y.sum() / len(y),
    )
    for i, (train_idx, val_idx) in enumerate(folds):
        n_fraud_train = y[train_idx].sum()
        n_fraud_val = y[val_idx].sum()
        logger.debug(
            "  Fold %d: train=%d (fraud=%d), val=%d (fraud=%d)",
            i,
            len(train_idx),
            n_fraud_train,
            len(val_idx),
            n_fraud_val,
        )
    return folds
