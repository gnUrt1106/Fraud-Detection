"""
Fraud Detection Dashboard — Performance Monitoring

Interactive Streamlit app with 3 panels:
    1. Performance — PR-AUC / F1 / Recall bar chart (Plotly)
    2. Analysis — Performance heatmap & scatter comparison
    3. Model Explorer — Per-configuration result details

Usage:
    streamlit run dashboard/app.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is importable
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── Config ───────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(project_root, "outputs", "results")

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom color palette mapping for Models and Conditions
MODEL_COLORS = {
    "Random Forest": "#2ecc71",
    "XGBoost": "#e74c3c",
    "CatBoost": "#3498db",
    "Logistic Regression": "#9b59b6",
}
CONDITION_MARKERS = {
    "Class-weighting": "circle",
    "SMOTE": "square",
    "SMOTE-ENN": "diamond",
}

# ── Data Loading ─────────────────────────────────────────────────────

@st.cache_data
def load_summary():
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    return pd.DataFrame()

# ── Main App ─────────────────────────────────────────────────────────

def main():
    # Inject Custom Premium CSS
    st.markdown("""
    <style>
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Reduce top padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Premium Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: 600;
            border: 1px solid #e9ecef;
            transition: all 0.2s ease-in-out;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e2e8f0;
            border-color: #cbd5e1;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2c3e50 !important;
            color: white !important;
            border: none;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
        
        /* Premium Metrics */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.1);
        }
        
        /* Headers styling */
        h1, h2, h3 {
            color: #1e293b;
            font-family: 'Inter', sans-serif;
            letter-spacing: -0.5px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>🔍 Fraud Detection: Performance Dashboard</h1>", unsafe_allow_html=True)

    raw_df = load_summary()

    if raw_df.empty:
        st.error(
            "No experiment results found. "
            "Please run `python run_experiments.py` first."
        )
        return

    # Map model abbreviations to full names
    MODEL_NAME_MAP = {
        "RF": "Random Forest",
        "XGB": "XGBoost",
        "LR": "Logistic Regression",
        "CatBoost": "CatBoost"
    }
    
    df = raw_df.copy()
    df["Model"] = df["Model"].map(lambda x: MODEL_NAME_MAP.get(x, x))
    df["Config"] = df["Model"] + " (" + df["Condition"] + ")"

    # Create Tabs for layout
    tab1, tab2, tab3 = st.tabs([
        "📊 Performance", 
        "🔬 Analysis (Heatmap & Scatter)", 
        "📋 Results Table"
    ])

    # ── Panel 1: Performance ─────────────────────────────────────────
    with tab1:
        st.markdown("<h3 style='margin-top: 0;'>📊 Performance Metrics Comparison</h3>", unsafe_allow_html=True)
        st.markdown("Đánh giá độ chính xác của các mô hình trên tập kiểm thử (Test Set).")
        st.write("")  # Spacer
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            metric = st.selectbox("Select Metric", ["PR-AUC", "F1", "Recall", "ROC-AUC"], index=0)
        with col2:
            sort_by = st.selectbox("Sort Output By", ["PR-AUC", "F1", "Recall"], index=0)

        df_sorted = df.sort_values(sort_by, ascending=False) if sort_by in df.columns else df

        # Summary metrics row
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            best = df.loc[df["PR-AUC"].idxmax()]
            st.metric("Best PR-AUC", f"{best['PR-AUC']:.3f}", delta=f"{best['Model']} ({best['Condition']})")
        with col_b:
            best_f1 = df.loc[df["F1"].idxmax()]
            st.metric("Best F1", f"{best_f1['F1']:.3f}", delta=f"{best_f1['Model']} ({best_f1['Condition']})")
        with col_c:
            best_recall = df.loc[df["Recall"].idxmax()]
            st.metric("Best Recall", f"{best_recall['Recall']:.3f}", delta=f"{best_recall['Model']} ({best_recall['Condition']})")

        st.write("")

        # Plotly Bar Chart
        if metric in df.columns:
            with st.container(border=True):
                fig = px.bar(
                    df_sorted, 
                    x="Config", 
                    y=metric,
                    color="Model",
                    color_discrete_map=MODEL_COLORS,
                    title=f"{metric} Comparison by Configuration",
                    labels={"Config": "Configuration", metric: metric},
                    text_auto='.3f'
                )
                fig.update_layout(xaxis_tickangle=-45, height=500, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

    # ── Panel 2: Analysis ────────────────────────────────────────────
    with tab2:
        st.markdown("<h3 style='margin-top: 0;'>🔬 Detailed Analysis</h3>", unsafe_allow_html=True)
        st.write("")

        # Heatmap
        st.markdown("#### Performance Heatmap")
        heatmap_metric = st.selectbox("Heatmap Metric", ["PR-AUC", "F1", "Recall", "ROC-AUC"], index=0, key="heatmap_metric")
        with st.container(border=True):
            pivot_df = df.pivot(index="Model", columns="Condition", values=heatmap_metric)
            fig_heat = px.imshow(
                pivot_df,
                text_auto=".3f",
                color_continuous_scale="YlGnBu",
                title=f"{heatmap_metric} by Model × Condition",
                labels=dict(x="Condition", y="Model", color=heatmap_metric)
            )
            fig_heat.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")

        # Scatter: PR-AUC vs F1
        st.subheader("PR-AUC vs F1 Trade-off")
        with st.container(border=True):
            fig_scatter = px.scatter(
                df,
                x="PR-AUC",
                y="F1",
                color="Model",
                symbol="Condition",
                color_discrete_map=MODEL_COLORS,
                symbol_map=CONDITION_MARKERS,
                size_max=15,
                hover_name="Config",
                hover_data={"PR-AUC": ':.3f', "F1": ':.3f', "Recall": ':.3f'}
            )
            fig_scatter.update_traces(marker=dict(size=14, line=dict(width=1.5, color='#ffffff')))
            fig_scatter.update_layout(height=500, xaxis_title="PR-AUC", yaxis_title="F1 Score", margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Panel 3: Results Table ────────────────────────────────────────
    with tab3:
        st.markdown("<h3 style='margin-top: 0;'>📋 Full Results Table</h3>", unsafe_allow_html=True)
        st.write("")

        sort_col = st.selectbox("Sort by", ["PR-AUC", "F1", "Recall", "ROC-AUC"], index=0, key="table_sort")
        df_table = df.sort_values(sort_col, ascending=False)

        # Display metrics columns only
        display_cols = [c for c in ["Model", "Condition", "PR-AUC", "ROC-AUC", "F1", "Recall"] if c in df_table.columns]
        st.dataframe(
            df_table[display_cols].style.format({
                "PR-AUC": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "F1": "{:.4f}",
                "Recall": "{:.4f}",
            }),
            use_container_width=True
        )

        # Download button
        csv = df_table[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name="experiment_results.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
