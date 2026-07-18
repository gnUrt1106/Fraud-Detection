#!/usr/bin/env python3
"""
Tái tạo toàn bộ visualizations từ outputs/results/ đã có.

Chạy sau run_experiments.py hoặc bất cứ lúc nào muốn vẽ lại:
    python generate_visuals.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.visualize import (
    plot_performance_comparison,
    plot_results_heatmap,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"


def load_results():
    """Load tất cả JSON results thành summary DataFrame."""
    perf_rows = []

    if not os.path.exists(RESULTS_DIR):
        logger.error("Không tìm thấy %s — chạy run_experiments.py trước!", RESULTS_DIR)
        sys.exit(1)

    for fname in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, fname)

        if fname == "summary.csv" or fname.endswith(".gitkeep"):
            continue

        if fname.endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            # Parse model + condition từ filename (e.g. RF_C0.json)
            stem = fname.replace(".json", "")
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                data["Model"] = parts[0]
                data["Condition"] = parts[1]
            perf_rows.append(data)

    if not perf_rows:
        logger.error("Không có kết quả nào trong %s!", RESULTS_DIR)
        sys.exit(1)

    df = pd.DataFrame(perf_rows)
    logger.info("Loaded %d experiment results", len(df))

    return df


def main():
    logger.info("=" * 60)
    logger.info("Generating all visualizations from outputs/results/")
    logger.info("=" * 60)

    df = load_results()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── 1. Performance comparison ─────────────────────────────────────
    logger.info("Plotting performance comparison...")
    plot_performance_comparison(
        df, save_path=f"{FIGURES_DIR}/performance_comparison.png"
    )

    # ── 2. PR-AUC heatmap ─────────────────────────────────────────────
    logger.info("Plotting PR-AUC heatmap...")
    if "PR-AUC" in df.columns:
        plot_results_heatmap(
            df, metric="PR-AUC",
            save_path=f"{FIGURES_DIR}/prauc_heatmap.png"
        )



    # ── 4. Save updated summary CSV ───────────────────────────────────
    csv_path = f"{RESULTS_DIR}/summary.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Summary CSV updated: %s", csv_path)

    logger.info("=" * 60)
    logger.info("✅ Tất cả visualizations đã được tạo trong: %s", FIGURES_DIR)
    logger.info("=" * 60)
    logger.info("Mở dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
