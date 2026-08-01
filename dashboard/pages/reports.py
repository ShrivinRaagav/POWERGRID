import streamlit as st
import pandas as pd
from pathlib import Path
from dashboard.utils import REPORTS_DIR

def render_reports_page():
    """Renders Reports & Download Center page."""
    st.title("📥 Reports & Asset Download Center")
    st.markdown("Download all generated CSV data tables, Markdown research reports, JSON metadata, and 300 DPI publication figures.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    # 1. CSV Data Reports
    with col1:
        st.subheader("📊 CSV Data Tables")

        csv_files = [
            ("Model Performance Ranking", "model_ranking.csv"),
            ("Procurement Recommendations", "procurement_recommendations.csv"),
            ("SHAP Feature Importance Table", "shap_feature_importance.csv"),
            ("Pareto Optimal Front Solutions", "pareto_front.csv"),
            ("Optimization Summary Metrics", "optimization_results.csv"),
            ("Pairwise Wilcoxon Test Results", "wilcoxon_results.csv"),
            ("Friedman Test Summary", "friedman_results.csv")
        ]

        for label, fname in csv_files:
            f_path = REPORTS_DIR / fname
            if f_path.exists():
                with open(f_path, "rb") as f:
                    st.download_button(
                        label=f"📥 Download {label} ({fname})",
                        data=f,
                        file_name=fname,
                        mime="text/csv",
                        key=f"dl_csv_{fname}"
                    )
            else:
                st.caption(f"⚠️ {fname} not found")

    # 2. Markdown Research Reports
    with col2:
        st.subheader("📑 Markdown Research Reports")

        md_files = [
            ("SHAP Explainability Report", "shap_report.md"),
            ("Supply Chain Optimization Report", "optimization_report.md"),
            ("Statistical Evaluation Report", "statistical_evaluation.md"),
            ("Data Quality Report", "data_quality_report.md"),
            ("Local Explanations Report", "local_explanations.md")
        ]

        for label, fname in md_files:
            f_path = REPORTS_DIR / fname
            if f_path.exists():
                with open(f_path, "rb") as f:
                    st.download_button(
                        label=f"📥 Download {label} ({fname})",
                        data=f,
                        file_name=fname,
                        mime="text/markdown",
                        key=f"dl_md_{fname}"
                    )
            else:
                st.caption(f"⚠️ {fname} not found")

    st.markdown("---")

    # 3. JSON Metadata & Publication Figures
    st.subheader("🖼️ 300 DPI Publication PNG Figures")

    fig_dirs = [
        ("Forecast & Evaluation Figures", REPORTS_DIR / "model_plots"),
        ("SHAP Explainability Figures", REPORTS_DIR / "shap_plots"),
        ("Supply Chain Optimization Figures", REPORTS_DIR / "optimization_plots")
    ]

    for category, p_dir in fig_dirs:
        if p_dir.exists():
            st.markdown(f"#### {category}")
            png_files = sorted(list(p_dir.glob("*.png")))
            f_cols = st.columns(3)
            for i, png_file in enumerate(png_files):
                with f_cols[i % 3]:
                    with open(png_file, "rb") as f:
                        st.download_button(
                            label=f"📷 {png_file.name}",
                            data=f,
                            file_name=png_file.name,
                            mime="image/png",
                            key=f"dl_png_{png_file.name}"
                        )
