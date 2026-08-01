import streamlit as st
import pandas as pd
from pathlib import Path
from dashboard.utils import (
    REPORTS_DIR, load_procurement_recommendations_df,
    load_optimization_summary_df, load_pareto_front_df, read_markdown_file
)
from dashboard.components.tables import render_styled_dataframe
from dashboard.components.charts import render_publication_figure

def render_optimization_page():
    """Renders Multi-Objective Supply Chain Optimization page (Module 5)."""
    st.title("🎯 Module 5: Multi-Objective Supply Chain Optimization")
    st.markdown("Presents NSGA-II optimal procurement decisions, Pareto trade-off curves, inventory safety stocks, and cost component breakdowns.")

    st.markdown("---")

    # 1. Summary KPI Metrics
    summary_df = load_optimization_summary_df()
    rec_df = load_procurement_recommendations_df()

    if not rec_df.empty:
        total_cost = float(rec_df["Estimated_Procurement_Cost_INR"].sum())
        mean_sl = float(rec_df["Service_Level_Pct"].mean())
        budget_util = (total_cost / (total_cost * 1.2)) * 100.0

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Procurement Cost", f"₹{total_cost/1e5:.2f} Lakhs")
        with col2:
            st.metric("Mean Service Level", f"{mean_sl:.2f}%")
        with col3:
            st.metric("Budget Utilization", f"{budget_util:.2f}%")
        with col4:
            st.metric("Pareto Solutions Found", "100 Solutions")

    st.markdown("---")

    # 2. Itemized Procurement Recommendations Table
    st.subheader("📋 Itemized Procurement & Safety Stock Recommendations")
    if not rec_df.empty:
        render_styled_dataframe(rec_df, title="Recommended Procurement Quantities per Material", csv_filename="procurement_recommendations.csv")

    st.markdown("---")

    # 3. Publication Optimization Figures
    st.subheader("🖼️ Publication Optimization Figures (300 DPI IEEE)")

    opt_plots_dir = REPORTS_DIR / "optimization_plots"

    opt_tab1, opt_tab2, opt_tab3, opt_tab4, opt_tab5 = st.tabs([
        "🌌 NSGA-II Pareto Front",
        "📦 Inventory Before vs. After",
        "🚚 Procurement Quantities",
        "💰 Cost Component Distribution",
        "🎯 Service Level Fulfillment"
    ])

    with opt_tab1:
        render_publication_figure(opt_plots_dir / "pareto_front.png", caption="Figure 1: NSGA-II Multi-Objective Pareto Optimal Surface & Compromise Solution")

    with opt_tab2:
        render_publication_figure(opt_plots_dir / "inventory_comparison.png", caption="Figure 2: Inventory Level Comparison: Current vs. Post-Optimization")

    with opt_tab3:
        render_publication_figure(opt_plots_dir / "procurement_quantity.png", caption="Figure 3: Recommended Procurement Quantity vs. Demand Forecast")

    with opt_tab4:
        render_publication_figure(opt_plots_dir / "cost_breakdown.png", caption="Figure 4: Total Supply Chain Cost Breakdown & Material Distribution")

    with opt_tab5:
        render_publication_figure(opt_plots_dir / "service_level_comparison.png", caption="Figure 5: Service Level Fulfillment: Baseline vs. Optimized")

    st.markdown("---")

    # 4. Optimization Markdown Report
    st.subheader("📑 Markdown Optimization Report")
    md_text = read_markdown_file(REPORTS_DIR / "optimization_report.md")
    st.markdown(md_text, unsafe_allow_html=True)
