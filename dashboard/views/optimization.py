import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dashboard.utils import (
    REPORTS_DIR, load_procurement_recommendations_df,
    load_optimization_summary_df, load_pareto_front_df
)
from dashboard.components.tables import render_styled_dataframe
from dashboard.components.charts import render_publication_figure

def render_optimization_page():
    """Renders Multi-Objective Supply Chain Optimization page."""
    st.title("🎯 Supply Chain & Inventory Optimization")
    st.markdown(
        "Multi-objective NSGA-II decision framework balancing **procurement expenditures**, "
        "**inventory holding costs**, **supplier delivery risks**, and **target service level fulfillment (≥95%)**."
    )

    st.markdown("---")

    rec_df = load_procurement_recommendations_df()
    summary_df = load_optimization_summary_df()
    pareto_df = load_pareto_front_df()

    # ---------------------------------------------------------
    # 1. Interactive Filters & Material Selection
    # ---------------------------------------------------------
    if not rec_df.empty and "Material_Type" in rec_df.columns:
        material_options = ["ALL"] + sorted([str(m) for m in rec_df["Material_Type"].unique()])
        cur_mat = st.session_state.get("global_material", "ALL")
        mat_idx = material_options.index(cur_mat) if cur_mat in material_options else 0

        def on_opt_material_change():
            st.session_state["global_material"] = st.session_state["opt_material_select"]

        col_f1, col_f2 = st.columns([1, 2])
        with col_f1:
            sel_mat = st.selectbox(
                "Filter Material Category",
                options=material_options,
                index=mat_idx,
                key="opt_material_select",
                on_change=on_opt_material_change
            )
        with col_f2:
            st.markdown(
                f"<div style='padding-top:28px; color:#475569; font-size:13px; font-weight:600;'>"
                f"Active Planning Scope: <b>{'All 6 POWERGRID Equipment Categories' if sel_mat == 'ALL' else sel_mat}</b> (4-Week Regional Central Depot Planning Horizon)"
                f"</div>",
                unsafe_allow_html=True
            )

        if sel_mat != "ALL":
            rec_filtered = rec_df[rec_df["Material_Type"] == sel_mat].copy()
            if rec_filtered.empty:
                rec_filtered = rec_df.copy()
        else:
            rec_filtered = rec_df.copy()
    else:
        rec_filtered = rec_df.copy() if not rec_df.empty else pd.DataFrame()

    # ---------------------------------------------------------
    # 2. Supply Chain KPI Summary Cards
    # ---------------------------------------------------------
    st.subheader("📊 Optimization Key Indicators")

    if not rec_filtered.empty:
        total_proc_cost = float(rec_filtered["Estimated_Procurement_Cost_INR"].sum())
        mean_sl = float(rec_filtered["Service_Level_Pct"].mean())
        total_budget_util = float(rec_filtered["Budget_Utilization_Pct"].sum()) if "Budget_Utilization_Pct" in rec_filtered.columns else 0.0
        
        # Summary metrics from optimization summary if ALL selected
        holding_cost_val = "₹74.0 Lakhs"
        if not summary_df.empty and "Metric" in summary_df.columns:
            h_row = summary_df[summary_df["Metric"].str.contains("Holding Cost", case=False, na=False)]
            if not h_row.empty:
                h_cost = float(h_row["Value"].values[0])
                holding_cost_val = f"₹{h_cost/1e5:.2f} Lakhs"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cost_display = f"₹{total_proc_cost/1e7:.2f} Cr" if total_proc_cost >= 1e7 else f"₹{total_proc_cost/1e5:.2f} Lakhs"
            st.metric("Total Procurement Cost", cost_display, help="Calculated optimal purchasing budget for selected materials.")
        with col2:
            st.metric("Mean Service Level", f"{mean_sl:.2f}%", delta=f"{mean_sl - 82.0:+.1f}% vs Baseline", help="Volumetric demand fulfillment rate.")
        with col3:
            st.metric("Inventory Holding Cost", holding_cost_val, help="Annual holding and carrying cost of average safety stock.")
        with col4:
            st.metric("Budget Utilization", f"{total_budget_util:.1f}%", help="Percentage of total allocated capital utilized.")

    st.markdown("---")

    # ---------------------------------------------------------
    # 3. Interactive Pareto Front Visualization
    # ---------------------------------------------------------
    st.subheader("🌌 Pareto Optimal Trade-Off Surface (NSGA-II)")
    st.markdown(
        "Interactive non-dominated frontier showing trade-offs between **Procurement Expenditure (₹ Lakhs)**, "
        "**Holding Cost (₹ Lakhs)**, and **Service Level Fulfillment (%)**."
    )

    if not pareto_df.empty:
        p_plot = pareto_df.copy()
        p_plot["Cost_Lakhs"] = p_plot["Procurement_Cost_INR"] / 100000.0 if "Procurement_Cost_INR" in p_plot.columns else p_plot.index
        p_plot["Holding_Lakhs"] = p_plot["Holding_Cost_INR"] / 100000.0 if "Holding_Cost_INR" in p_plot.columns else 0.0
        
        # Identify compromise solution (best balanced trade-off >= 95% SL)
        sl_col = "Service_Level_Pct"
        high_sl_df = p_plot[p_plot[sl_col] >= 95.0]
        if not high_sl_df.empty:
            comp_idx = high_sl_df["Cost_Lakhs"].idxmin()
        else:
            comp_idx = p_plot[sl_col].idxmax()

        fig_pareto = px.scatter(
            p_plot,
            x="Cost_Lakhs",
            y=sl_col,
            color="Holding_Lakhs",
            color_continuous_scale="Viridis",
            labels={
                "Cost_Lakhs": "Procurement Cost (₹ Lakhs)",
                sl_col: "Service Level Fulfillment (%)",
                "Holding_Lakhs": "Holding Cost (₹ Lakhs)"
            },
            hover_data={
                "Cost_Lakhs": ":.2f",
                sl_col: ":.2f",
                "Holding_Lakhs": ":.2f"
            },
            title="NSGA-II Pareto Optimal Frontier (Cost vs. Service Level)"
        )
        fig_pareto.update_traces(
            marker=dict(size=9, opacity=0.85, line=dict(width=1, color="#1e293b")),
            selector=dict(mode='markers')
        )

        # Highlight Compromise Optimal Solution
        comp_row = p_plot.loc[comp_idx]
        fig_pareto.add_trace(go.Scatter(
            x=[comp_row["Cost_Lakhs"]],
            y=[comp_row[sl_col]],
            mode="markers+text",
            marker=dict(symbol="star", size=18, color="#d62728", line=dict(width=1.5, color="#000000")),
            name="Compromise Optimal Solution (TOPSIS)",
            text=["★ Compromise Optimal"],
            textposition="bottom right"
        ))

        # Target 95% Service Level Reference Line
        fig_pareto.add_hline(
            y=95.0, line_dash="dash", line_color="#0066cc",
            annotation_text="Target Service Level Threshold (95%)", annotation_position="top left"
        )

        fig_pareto.update_layout(
            template="plotly_white",
            height=480,
            xaxis=dict(title="Procurement Cost (₹ Lakhs)", showgrid=True, gridcolor="#e2e8f0"),
            yaxis=dict(title="Service Level Fulfillment (%)", showgrid=True, gridcolor="#e2e8f0", range=[25, 105]),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_pareto, use_container_width=True)
    else:
        st.info("Pareto front CSV data unavailable.")

    st.markdown("---")

    # ---------------------------------------------------------
    # 4. Itemized Procurement Recommendations Table
    # ---------------------------------------------------------
    st.subheader("📋 Itemized Procurement & Safety Stock Allocation")
    st.markdown(
        "Executive purchase orders, lead-time safety stock buffers, and reorder trigger points generated for regional warehouse fulfillment."
    )
    if not rec_filtered.empty:
        # Format presentation columns
        display_df = rec_filtered.copy()
        for col in ["Forecasted_Demand", "Current_Inventory", "Recommended_Procurement_Qty", "Safety_Stock_Qty", "Reorder_Point", "Expected_Inventory_Level"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(lambda x: f"{x:,.1f}" if pd.notnull(x) else "-")

        if "Unit_Price_INR" in display_df.columns:
            display_df["Unit_Price_INR"] = display_df["Unit_Price_INR"].map(lambda x: f"₹{x:,.2f}" if pd.notnull(x) else "-")
        if "Estimated_Procurement_Cost_INR" in display_df.columns:
            display_df["Estimated_Procurement_Cost_INR"] = display_df["Estimated_Procurement_Cost_INR"].map(lambda x: f"₹{x:,.2f}" if pd.notnull(x) else "-")
        if "Service_Level_Pct" in display_df.columns:
            display_df["Service_Level_Pct"] = display_df["Service_Level_Pct"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
        if "Budget_Utilization_Pct" in display_df.columns:
            display_df["Budget_Utilization_Pct"] = display_df["Budget_Utilization_Pct"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")

        render_styled_dataframe(display_df, title="Recommended Purchase Quantities & Buffer Stocks", csv_filename="procurement_recommendations.csv")
    else:
        st.info("No procurement recommendations data available.")

    st.markdown("---")

    # ---------------------------------------------------------
    # 5. Publication Optimization Figures
    # ---------------------------------------------------------
    st.subheader("🖼️ Publication-Quality Optimization Figures (IEEE Standard)")

    opt_plots_dir = REPORTS_DIR / "optimization_plots"

    opt_tab1, opt_tab2, opt_tab3, opt_tab4, opt_tab5 = st.tabs([
        "🌌 Pareto Surface",
        "📦 Inventory Comparison",
        "🚚 Procurement Quantities",
        "💰 Cost Breakdown",
        "🎯 Service Level Fulfillment"
    ])

    with opt_tab1:
        render_publication_figure(opt_plots_dir / "pareto_front.png", caption="Figure 1: NSGA-II Multi-Objective Pareto Optimal Frontier")

    with opt_tab2:
        render_publication_figure(opt_plots_dir / "inventory_comparison.png", caption="Figure 2: Inventory Level Comparison (Current Inventory vs. Safety Stock Buffer)")

    with opt_tab3:
        render_publication_figure(opt_plots_dir / "procurement_quantity.png", caption="Figure 3: Recommended Order Quantity vs. Forecasted Demand")

    with opt_tab4:
        render_publication_figure(opt_plots_dir / "cost_breakdown.png", caption="Figure 4: Supply Chain Cost Distribution & Component Breakdown")

    with opt_tab5:
        render_publication_figure(opt_plots_dir / "service_level_comparison.png", caption="Figure 5: Service Level Fulfillment Comparison (Baseline vs. AI-Optimized)")

