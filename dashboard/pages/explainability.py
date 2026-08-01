import streamlit as st
import pandas as pd
from pathlib import Path
from dashboard.utils import REPORTS_DIR, load_shap_importance_df, read_markdown_file
from dashboard.components.tables import render_styled_dataframe
from dashboard.components.charts import render_publication_figure

def render_explainability_page():
    """Renders SHAP Explainability & XAI attributions page (Module 4)."""
    st.title("🔍 Module 4: SHAP Explainability & XAI Transparency")
    st.markdown("Interprets global feature attributions and local prediction equations using game-theoretic Shapley values.")

    st.markdown("---")

    # 1. Global Feature Importance Table
    st.subheader("🏆 Ranked Global Feature Importance (|mean SHAP|)")
    shap_df = load_shap_importance_df()
    if not shap_df.empty:
        render_styled_dataframe(shap_df.head(15), title="Top 15 Predictive Feature Drivers", csv_filename="shap_feature_importance.csv")
    else:
        st.info("SHAP feature importance CSV not found.")

    st.markdown("---")

    # 2. Publication SHAP Visualizations
    st.subheader("🖼️ Publication SHAP Figures")

    shap_plots_dir = REPORTS_DIR / "shap_plots"

    vis_tab1, vis_tab2, vis_tab3, vis_tab4 = st.tabs([
        "🐝 Beeswarm Summary Plot",
        "📊 Importance Bar Plot",
        "🌊 Local Waterfall Case Studies",
        "📈 Dependence Plots"
    ])

    with vis_tab1:
        render_publication_figure(shap_plots_dir / "shap_summary.png", caption="Figure 1: SHAP Beeswarm Summary Plot (Feature Value Impact on Forecast)")

    with vis_tab2:
        render_publication_figure(shap_plots_dir / "shap_bar.png", caption="Figure 2: Global Mean Absolute SHAP Value Importance Bar Chart")

    with vis_tab3:
        waterfall_choice = st.selectbox(
            "Select Demand Scenario Case Study",
            options=["Highest Demand Sample", "Median Demand Sample", "Lowest Demand Sample"]
        )
        if "Highest" in waterfall_choice:
            wf_file = "shap_waterfall_highest.png"
        elif "Median" in waterfall_choice:
            wf_file = "shap_waterfall_median.png"
        else:
            wf_file = "shap_waterfall_lowest.png"

        render_publication_figure(shap_plots_dir / wf_file, caption=f"Local Waterfall Explanation: {waterfall_choice}")

    with vis_tab4:
        dep_choice = st.selectbox(
            "Select Feature Dependence Plot",
            options=[
                "Feature 1: EMD_IMF_6 (dependence_feature_1.png)",
                "Feature 2: EMD_IMF_5 (dependence_feature_2.png)",
                "Feature 3: Classical_Residual (dependence_feature_3.png)",
                "Feature 4: Classical_Seasonal (dependence_feature_4.png)",
                "Feature 5: EMD_IMF_4 (dependence_feature_5.png)"
            ]
        )
        idx_str = dep_choice.split(":")[0].split(" ")[1]
        dep_file = f"dependence_feature_{idx_str}.png"
        render_publication_figure(shap_plots_dir / dep_file, caption=f"Dependence Plot: {dep_choice}")

    st.markdown("---")

    # 3. Markdown XAI Reports
    st.subheader("📑 Markdown Explainability Reports")

    report_tab1, report_tab2 = st.tabs(["📄 Comprehensive SHAP Report", "📄 Local Explanations Report"])

    with report_tab1:
        md_text1 = read_markdown_file(REPORTS_DIR / "shap_report.md")
        st.markdown(md_text1, unsafe_allow_html=True)

    with report_tab2:
        md_text2 = read_markdown_file(REPORTS_DIR / "local_explanations.md")
        st.markdown(md_text2, unsafe_allow_html=True)
