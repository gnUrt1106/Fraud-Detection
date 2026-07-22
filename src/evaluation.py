"""
Evaluation utilities — chuẩn hoá giữa nhánh ML và DL.

Quy tắc bắt buộc áp dụng thống nhất cho tất cả 5 model:
1. Threshold: tối đa hoá F1 trên validation fold (không dùng 0.5 cứng).
2. Report mean ± std qua 5 fold cho mỗi metric.
3. Paired Wilcoxon signed-rank test để kết luận có ý nghĩa thống kê.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


# ── Threshold selection ───────────────────────────────────────────────

def find_f1_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """
    Tìm threshold tối đa hoá F1 trên tập validation.

    Áp dụng nhất quán cho tất cả model — không để mỗi model tự chọn
    threshold có lợi cho mình.

    Args:
        y_true: Nhãn thật (0/1).
        y_prob: Xác suất dự đoán (class 1).

    Returns:
        (best_threshold, best_f1): float, float.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds có length = len(precisions) - 1
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
        0.0,
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ── Per-fold metrics ──────────────────────────────────────────────────

def evaluate_fold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fold: int,
    model_name: str,
    condition: str,
) -> dict:
    """
    Tính tất cả metrics cho một fold.

    Args:
        y_true:     Nhãn thật.
        y_prob:     Xác suất dự đoán (class 1).
        fold:       Số thứ tự fold (0-indexed).
        model_name: Tên model ('RF', 'XGB', ..., 'ANN').
        condition:  Imbalance condition ('Class-weighting', 'SMOTE', 'SMOTE-ENN').

    Returns:
        Dict chứa đầy đủ metrics của fold này.
    """
    threshold, f1 = find_f1_optimal_threshold(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)

    result = {
        "model":     model_name,
        "condition": condition,
        "fold":      fold,
        "PR-AUC":    round(pr_auc, 6),
        "ROC-AUC":   round(roc_auc, 6),
        "F1":        round(f1, 6),
        "Recall":    round(recall, 6),
        "Precision": round(precision, 6),
        "threshold": round(threshold, 6),
    }

    logger.info(
        "[Fold %d] %s | %s → PR-AUC=%.4f F1=%.4f Recall=%.4f Prec=%.4f thr=%.4f",
        fold, model_name, condition,
        pr_auc, f1, recall, precision, threshold,
    )
    return result


# ── Cross-fold summary ────────────────────────────────────────────────

def summarize_folds(
    fold_results: list[dict],
    model_name: str,
    condition: str,
) -> dict:
    """
    Tính mean ± std qua 5 fold cho một (model, condition).

    Args:
        fold_results: List of dict từ evaluate_fold().
        model_name:   Tên model.
        condition:    Imbalance condition.

    Returns:
        Dict summary với keys: model, condition, PR-AUC_mean, PR-AUC_std, ...
    """
    metrics = ["PR-AUC", "ROC-AUC", "F1", "Recall", "Precision"]
    summary = {"model": model_name, "condition": condition}

    for m in metrics:
        values = [r[m] for r in fold_results]
        summary[f"{m}_mean"] = round(float(np.mean(values)), 5)
        summary[f"{m}_std"]  = round(float(np.std(values, ddof=1)), 5)

    return summary


# ── Paired statistical test ───────────────────────────────────────────

def paired_wilcoxon_test(
    fold_scores_a: list[float],
    fold_scores_b: list[float],
    model_a: str,
    model_b: str,
    condition: str,
    metric: str = "PR-AUC",
    alpha: float = 0.05,
) -> dict:
    """
    Wilcoxon signed-rank test so sánh 2 model trên cùng fold indices.

    Dùng được vì cả 2 model dùng chung fold indices (đảm bảo paired).

    Args:
        fold_scores_a/b: List PR-AUC (hoặc metric khác) per fold.
        model_a/b:       Tên 2 model.
        condition:       Imbalance condition.
        metric:          Metric so sánh.
        alpha:           Ngưỡng p-value để kết luận significant.

    Returns:
        Dict với statistic, p_value, significant, winner.
    """
    stat, p_value = stats.wilcoxon(fold_scores_a, fold_scores_b, alternative="two-sided")
    mean_a = np.mean(fold_scores_a)
    mean_b = np.mean(fold_scores_b)
    winner = model_a if mean_a >= mean_b else model_b

    result = {
        "model_a":    model_a,
        "model_b":    model_b,
        "condition":  condition,
        "metric":     metric,
        "mean_a":     round(float(mean_a), 5),
        "mean_b":     round(float(mean_b), 5),
        "statistic":  round(float(stat), 4),
        "p_value":    round(float(p_value), 6),
        "significant": bool(p_value < alpha),
        "winner":      winner if p_value < alpha else "no significant difference",
    }

    logger.info(
        "Wilcoxon [%s vs %s | %s | %s]: p=%.4f %s | %s vs %s → %s",
        model_a, model_b, condition, metric,
        p_value,
        "✓ SIGNIFICANT" if result["significant"] else "✗ not significant",
        f"{mean_a:.4f}", f"{mean_b:.4f}",
        result["winner"],
    )
    return result


# ── Save utilities ────────────────────────────────────────────────────

def save_summary(
    fold_records: list[dict],
    summary_records: list[dict],
    paired_records: list[dict],
    output_dir: str = "outputs/metrics",
) -> None:
    """
    Lưu 3 file CSV:
    - fold_results.csv    : metrics từng fold
    - summary.csv         : mean ± std per (model, condition)
    - paired_tests.csv    : kết quả Wilcoxon test
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(fold_records).to_csv(
        f"{output_dir}/fold_results.csv", index=False
    )
    pd.DataFrame(summary_records).to_csv(
        f"{output_dir}/summary.csv", index=False
    )
    pd.DataFrame(paired_records).to_csv(
        f"{output_dir}/paired_tests.csv", index=False
    )
    logger.info("Saved metrics to %s/", output_dir)


# ── Business & Practical Utility Evaluation ─────────────────────────

def cohens_d(x: list[float], y: list[float]) -> float:
    """Compute Cohen's d effect size between two score distributions."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    if dof <= 0:
        return 0.0
    var_x = np.var(x, ddof=1) if nx > 1 else 0.0
    var_y = np.var(y, ddof=1) if ny > 1 else 0.0
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / dof)
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def compute_business_utility(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: Optional[np.ndarray] = None,
    fp_cost: float = 10.0,
) -> dict:
    """
    Compute business financial impact & Precision@Top-K metrics.

    Args:
        y_true: Ground truth labels (0/1).
        y_prob: Predicted probabilities.
        amounts: Transaction monetary amounts (optional).
        fp_cost: Operational cost per false alarm ($10 default).

    Returns:
        Dict with Precision@100, Precision@200, Precision@500, Net_Savings.
    """
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true[sorted_indices]

    precisions_k = {}
    for k in [100, 200, 500]:
        k_eff = min(k, len(y_true_sorted))
        top_k_tp = np.sum(y_true_sorted[:k_eff])
        precisions_k[f"Precision@Top{k}"] = round(float(top_k_tp / k_eff), 4)

    # Net Financial Savings calculation (if amounts provided)
    net_savings = 0.0
    if amounts is not None:
        amounts_sorted = amounts[sorted_indices]
        # Optimal F1 threshold prediction
        thresh, _ = find_f1_optimal_threshold(y_true, y_prob)
        y_pred = (y_prob >= thresh).astype(int)
        tp_mask = (y_pred == 1) & (y_true == 1)
        fp_mask = (y_pred == 1) & (y_true == 0)
        saved_fraud_val = np.sum(amounts[tp_mask])
        false_alarm_cost = np.sum(fp_mask) * fp_cost
        net_savings = float(saved_fraud_val - false_alarm_cost)

    res = {**precisions_k, "Net_Savings_USD": round(net_savings, 2)}
    return res
