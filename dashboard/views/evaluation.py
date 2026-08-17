import streamlit as st
import pandas as pd
from pathlib import Path
from dashboard.utils import REPORTS_DIR, load_model_ranking_df
from dashboard.components.tables import render_styled_dataframe
from dashboard.components.charts import render_publication_figure, render_interactive_metrics_chart

def render_evaluation_page():
    """Renders Forecast Model Evaluation & Statistical Analysis page."""
    st.title("⚖️ Model Performance & Statistical Evaluation")

    # 1. Model Leaderboard
    st.subheader("📊 Model Performance Leaderboard")
    ranking_df = load_model_ranking_df()
    render_styled_dataframe(ranking_df, title="Model Evaluation Ranking Matrix", csv_filename="model_ranking.csv")

    st.markdown("---")

    # 2. Interactive Metric Bar Comparisons
    st.subheader("📈 Out-of-Sample Metric Comparisons")
    render_interactive_metrics_chart(ranking_df)

    st.markdown("---")

    # 3. Model Comparison Figures
    st.subheader("🖼️ Comparative Metric Figures")

    plots_dir = REPORTS_DIR / "model_plots"

    metric_fig_choice = st.selectbox(
        "Select Metric Chart",
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

    # 4. Statistical Significance Tests
    st.subheader("🔬 Pairwise Statistical Significance (Wilcoxon Test)")
    wil_csv = REPORTS_DIR / "wilcoxon_results.csv"
    if wil_csv.exists():
        wil_df = pd.read_csv(wil_csv)
        render_styled_dataframe(wil_df, title="Wilcoxon Signed-Rank Test Results", csv_filename="wilcoxon_results.csv")
    else:
        st.info("Wilcoxon results CSV file not found.")
