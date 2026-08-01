import streamlit as st
import pandas as pd
from pathlib import Path
from dashboard.utils import REPORTS_DIR, load_model_ranking_df, read_markdown_file
from dashboard.components.tables import render_styled_dataframe
from dashboard.components.charts import render_publication_figure

def render_evaluation_page():
    """Renders Forecast Model Evaluation & Statistical Analysis page (Module 3.5)."""
    st.title("⚖️ Module 3.5: Forecast Model Evaluation & Statistical Analysis")
    st.markdown("Compares 6 forecasting models (Random Forest, SVR, XGBoost, MLP, LSTM, LightGBM Quantile) across accuracy, runtime efficiency, and statistical significance tests.")

    st.markdown("---")

    # 1. Interactive Model Ranking & Comparison Matrix
    st.subheader("📊 Model Performance Comparison Matrix")
    ranking_df = load_model_ranking_df()
    render_styled_dataframe(ranking_df, title="Multi-Metric Model Evaluation Table", csv_filename="model_ranking.csv")

    st.markdown("---")

    # 2. IEEE Publication Metric Comparison Figures
    st.subheader("🖼️ Publication Metric Comparison Figures")

    plots_dir = REPORTS_DIR / "model_plots"

    metric_fig_choice = st.selectbox(
        "Select Metric Comparison Chart to Display",
        options=[
            "RMSE Comparison (rmse_comparison.png)",
            "MAE Comparison (mae_comparison.png)",
            "WMAPE Comparison (wmape_comparison.png)",
            "R² Score Comparison (r2_comparison.png)",
            "Training Time Comparison (training_time_comparison.png)",
            "Inference Time Comparison (inference_time_comparison.png)"
        ]
    )

    if "RMSE" in metric_fig_choice:
        f_name = "rmse_comparison.png"
    elif "MAE" in metric_fig_choice:
        f_name = "mae_comparison.png"
    elif "WMAPE" in metric_fig_choice:
        f_name = "wmape_comparison.png"
    elif "R²" in metric_fig_choice:
        f_name = "r2_comparison.png"
    elif "Training" in metric_fig_choice:
        f_name = "training_time_comparison.png"
    else:
        f_name = "inference_time_comparison.png"

    render_publication_figure(plots_dir / f_name, caption=f"Figure: {metric_fig_choice}")

    st.markdown("---")

    # 3. Statistical Analysis Reports & Significance Tests
    st.subheader("🔬 Statistical Significance Tests (Wilcoxon & Friedman)")

    stat_tab1, stat_tab2 = st.tabs(["📑 Statistical Report (statistical_evaluation.md)", "📋 Pairwise Wilcoxon Test CSV"])

    with stat_tab1:
        md_text = read_markdown_file(REPORTS_DIR / "statistical_evaluation.md")
        st.markdown(md_text, unsafe_allow_html=True)

    with stat_tab2:
        wil_csv = REPORTS_DIR / "wilcoxon_results.csv"
        if wil_csv.exists():
            wil_df = pd.read_csv(wil_csv)
            render_styled_dataframe(wil_df, title="Wilcoxon Signed-Rank Test Results", csv_filename="wilcoxon_results.csv")
        else:
            st.info("Wilcoxon results CSV file not found.")
