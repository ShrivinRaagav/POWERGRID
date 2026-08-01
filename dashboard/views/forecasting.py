import streamlit as st
from pathlib import Path
from dashboard.utils import REPORTS_DIR, load_best_model_info, load_procurement_recommendations_df
from dashboard.components.charts import render_publication_figure, render_kpi_card
from dashboard.components.tables import render_styled_dataframe

def render_forecasting_page():
    """Renders Demand Forecasting & Prediction Intervals page."""
    st.title("📈 Module 3: Material Demand Forecasting Engine")
    st.markdown("Displays forecasted material demand, actual vs. predicted time-series curves, and probabilistic quantile bands (P10, P50, P90).")

    st.markdown("---")

    # Best Model Information Banner
    best_info = load_best_model_info()
    best_model_name = str(best_info.get("best_model", "xgboost")).upper()

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Selected Best Model Engine**: `{best_model_name}`")
    with col2:
        st.success("**Selection Criterion**: Lowest RMSE across independent test set")

    st.markdown("---")

    # 1. Publication Visualizations: Actual vs. Predicted & Probabilistic Forecasts
    st.subheader("🖼️ Publication-Quality Forecast Plots")

    plot_tab1, plot_tab2 = st.tabs(["📈 Best Model Actual vs. Predicted", "📊 Probabilistic Quantile Forecast (P10, P50, P90)"])

    plots_dir = REPORTS_DIR / "model_plots"

    with plot_tab1:
        fig_path1 = plots_dir / "best_model_actual_vs_predicted.png"
        if not fig_path1.exists():
            fig_path1 = plots_dir / "actual_vs_predicted_best_model.png"
        render_publication_figure(fig_path1, caption="Figure 1: Best Model Actual vs. Predicted Demand Time-Series")

    with plot_tab2:
        fig_path2 = plots_dir / "lightgbm_quantile_probabilistic_forecast.png"
        render_publication_figure(fig_path2, caption="Figure 2: LightGBM Quantile Probabilistic Forecast (P10 lower bound, P50 median, P90 upper bound)")

    st.markdown("---")

    # 2. Material Demand Forecast Table
    st.subheader("📋 Itemized Material Demand Forecasts")
    rec_df = load_procurement_recommendations_df()

    if not rec_df.empty:
        display_df = rec_df[["Material_Type", "Forecasted_Demand", "Current_Inventory", "Lead_Time_Weeks"]].copy()
        render_styled_dataframe(display_df, title="Material Category Demand Summary", csv_filename="material_demand_forecasts.csv")
    else:
        st.info("No forecast recommendations available in reports.")
