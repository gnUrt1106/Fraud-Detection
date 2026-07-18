#!/usr/bin/env python3
"""
Data profiling — Bước mô tả dữ liệu (EDA).

Chạy MỘT LẦN DUY NHẤT trên toàn bộ creditcard.csv gốc,
TRƯỚC KHI chia fold — không dùng để train hay tune, không có leakage.

Output: outputs/profiling/data_profile.html

Usage:
    python src/data_profiling.py
    python src/data_profiling.py --minimal          # bỏ qua correlations nặng
    python src/data_profiling.py --data-path data/raw/creditcard.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_profiling(
    data_path: str = "data/raw/creditcard.csv",
    output_path: str = "outputs/profiling/data_profile.html",
    minimal: bool = False,
) -> None:
    """
    Chạy ydata-profiling trên creditcard.csv và lưu HTML report.

    Report bao gồm (mặc định ydata-profiling đã tự làm):
    - Phân phối, missing values, cardinality của V1-V28, Amount, Time.
    - Tỉ lệ mất cân bằng Class (492/284807 ≈ 0.17%) — căn cứ chọn imbalance handling.
    - Ma trận tương quan — V1-V28 đã qua PCA nên tương quan phải gần 0;
      tương quan cao bất thường là dấu hiệu cần kiểm tra lại dữ liệu.
    - Phân phối Amount tách theo Class (fraud vs non-fraud) —
      lý do cần scale Amount trước khi vào ANN.

    Args:
        data_path:   Đường dẫn đến creditcard.csv.
        output_path: Đường dẫn lưu HTML report.
        minimal:     Nếu True, bỏ bớt pairwise correlations và tính toán nặng
                     (khuyến nghị khi chạy với ~284k rows trên máy local yếu).
    """
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        logger.error(
            "Thư viện 'ydata-profiling' chưa được cài đặt.\n"
            "Chạy: pip install ydata-profiling>=4.6"
        )
        sys.exit(1)

    logger.info("Loading data from %s ...", data_path)
    df = pd.read_csv(data_path)
    logger.info(
        "Loaded %d rows × %d columns | Fraud rate: %.4f%%",
        len(df),
        df.shape[1],
        100.0 * df["Class"].sum() / len(df),
    )

    logger.info(
        "Generating ProfileReport (minimal=%s)... "
        "Quá trình này có thể mất vài phút với ~284k rows.",
        minimal,
    )

    profile = ProfileReport(
        df,
        title="ULB Credit Card Fraud — Data Profiling",
        explorative=True,
        minimal=minimal,
        # Ẩn các cảnh báo không cần thiết
        config_file=None,
        lazy=False,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    profile.to_file(output_path)
    logger.info("Report saved to %s", output_path)
    logger.info(
        "\nTóm tắt nhanh:\n"
        "  Số dòng:   %d\n"
        "  Fraud:     %d (%.4f%%)\n"
        "  Non-fraud: %d\n"
        "  Missing:   %d",
        len(df),
        int(df["Class"].sum()),
        100.0 * df["Class"].sum() / len(df),
        int((df["Class"] == 0).sum()),
        df.isnull().sum().sum(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tạo ydata-profiling HTML report cho creditcard.csv",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/creditcard.csv",
        help="Đường dẫn đến creditcard.csv",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/profiling/data_profile.html",
        help="Đường dẫn lưu HTML report",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Bỏ bớt pairwise correlations để chạy nhanh hơn",
    )
    args = parser.parse_args()
    run_profiling(
        data_path=args.data_path,
        output_path=args.output_path,
        minimal=args.minimal,
    )
