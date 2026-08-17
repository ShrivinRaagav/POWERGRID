import streamlit as st
import pandas as pd
from pathlib import Path
from dashboard.utils import REPORTS_DIR, load_shap_importance_df
from dashboard.components.tables import render_styled_dataframe
from dashboard.components.charts import render_publication_figure

def render_explainability_page():
    """Renders SHAP Explainability & Feature Attributions page."""
    st.title("🔍 Model Explainability & Feature Attributions")

    # ---------------------------------------------------------
    # SECTION 1: Global Feature Importance
    # ---------------------------------------------------------
    st.subheader("1️⃣ Global Feature Attributions")

    shap_plots_dir = REPORTS_DIR / "shap_plots"

    glob_tab1, glob_tab2, glob_tab3 = st.tabs([
        "🐝 SHAP Summary Plot (Beeswarm)",
        "🏆 Feature Importance Table",
        "📊 Global Importance Bar Plot"
    ])

    with glob_tab1:
        render_publication_figure(
            shap_plots_dir / "shap_summary.png", 
            caption="Global SHAP Beeswarm Summary Plot"
        )

    with glob_tab2:
        shap_df = load_shap_importance_df()
        if not shap_df.empty:
            render_styled_dataframe(shap_df.head(15), title="Top 15 Feature Drivers (|mean SHAP|)", csv_filename="shap_feature_importance.csv")
        else:
            st.info("SHAP feature importance CSV not found.")

    with glob_tab3:
        render_publication_figure(
            shap_plots_dir / "shap_bar.png", 
            caption="Global Mean Absolute SHAP Feature Attributions"
        )

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION 2: Local Prediction Waterfall Breakdown
    # ---------------------------------------------------------
    st.subheader("2️⃣ Local Prediction Breakdown (Waterfall Analysis)")

    waterfall_choice = st.selectbox(
        "Select Prediction Case Study",
        options=[
            "Highest Demand Sample (Peak Demand)",
            "Median Demand Sample (Standard Operation)",
            "Lowest Demand Sample (Baseline Inventory)"
        ]
    )

    if "Highest" in waterfall_choice:
        wf_file = "shap_waterfall_highest.png"
    elif "Median" in waterfall_choice:
        wf_file = "shap_waterfall_median.png"
    else:
        wf_file = "shap_waterfall_lowest.png"

    render_publication_figure(shap_plots_dir / wf_file, caption=f"SHAP Waterfall: {waterfall_choice}")

    st.markdown("---")

    # ---------------------------------------------------------
    # SECTION 3: Material Uncertainty Risk Matrix
    # ---------------------------------------------------------
    st.subheader("3️⃣ Material Uncertainty Risk Matrix & Coverage")

    mat_analysis_path = REPORTS_DIR / "material_uncertainty_analysis.csv"
    mat_analysis_df = pd.read_csv(mat_analysis_path) if mat_analysis_path.exists() else pd.DataFrame()

    uc_col1, uc_col2, uc_col3 = st.columns(3)
    with uc_col1:
        st.metric("Overall PICP Coverage", "89.65%")
    with uc_col2:
        st.metric("Avg Interval Width", "286.46 Units")
    with uc_col3:
        st.metric("P90 Violation Rate", "6.32%")

    uc_tab1, uc_tab2, uc_tab3 = st.tabs([
        "📊 Material Risk Matrix",
        "📈 Coverage Timeline Plot",
        "🔬 Quantile Strategy Experiment"
    ])

    with uc_tab1:
        if not mat_analysis_df.empty:
            risk_summary = mat_analysis_df.groupby("Material").agg({
                "Total Samples": "sum",
                "PICP": "mean",
                "Average_Interval_Width": "mean",
                "P90_Violations": "sum",
                "Demand_Std": "mean"
            }).reset_index()

            risk_summary["P90_Violation_Pct"] = (risk_summary["P90_Violations"] / risk_summary["Total Samples"]) * 100.0
            
            def get_risk_level(row):
                if row["P90_Violation_Pct"] > 10.0 or row["PICP"] < 80.0:
                    return "🔴 High Volatility (Demand Spikes)"
                elif row["PICP"] > 95.0 and row["Average_Interval_Width"] > 1.5 * row["Demand_Std"]:
                    return "🟡 Wide Buffer"
                else:
                    return "🟢 Well-Calibrated"

            risk_summary["Status"] = risk_summary.apply(get_risk_level, axis=1)

            display_cols = ["Material", "PICP", "Average_Interval_Width", "P90_Violations", "Status"]
            risk_matrix_df = risk_summary[display_cols].rename(columns={"Average_Interval_Width": "Interval Width"})
            render_styled_dataframe(risk_matrix_df, title="Material Calibration Matrix", csv_filename="material_risk_matrix.csv")
        else:
            st.info("Material uncertainty analysis CSV not found.")

    with uc_tab2:
        cov_plot_path = REPORTS_DIR / "important_plots" / "prediction_interval_coverage.png"
        render_publication_figure(cov_plot_path, caption="LightGBM Quantile Prediction Interval Coverage")

    with uc_tab3:
        strat_path = REPORTS_DIR / "quantile_strategy_comparison.csv"
        if strat_path.exists():
            strat_df = pd.read_csv(strat_path)
            render_styled_dataframe(strat_df, title="Quantile Strategy Comparison", csv_filename="quantile_strategy_comparison.csv")
        else:
            st.info("Quantile strategy comparison CSV not found.")
