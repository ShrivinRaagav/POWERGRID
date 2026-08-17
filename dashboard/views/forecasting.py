import streamlit as st
import pandas as pd
from pathlib import Path
from dashboard.utils import (
    REPORTS_DIR, load_best_model_info, load_procurement_recommendations_df,
    load_latest_predictions_df
)
from dashboard.components.charts import (
    render_publication_figure, render_kpi_card,
    render_interactive_forecast_chart, render_decision_recommendation_card
)
from dashboard.components.tables import render_styled_dataframe

def render_forecasting_page():
    """Renders Material Demand Forecasting Engine page."""
    st.title("📈 Material Demand Forecasting Engine")
    st.markdown("Displays interactive demand forecasts, quantile confidence bands (P10, P50, P90), and AI executive procurement directives.")

    st.markdown("---")

    preds_df = load_latest_predictions_df()

    # ---------------------------------------------------------
    # SECTION 1: Forecast Summary & Control Cards
    # ---------------------------------------------------------
    st.subheader("1️⃣ Forecast Summary & Interactive Filters")

    if not preds_df.empty:
        regions = ["ALL"] + sorted([str(r) for r in preds_df["Region_Name"].unique()]) if "Region_Name" in preds_df.columns else ["ALL"]
        materials = ["ALL"] + sorted([str(m) for m in preds_df["Material_Name"].unique()]) if "Material_Name" in preds_df.columns else ["ALL"]

        reg_default = st.session_state.get("global_region", "ALL")
        mat_default = st.session_state.get("global_material", "ALL")

        reg_i = regions.index(reg_default) if reg_default in regions else 0
        mat_i = materials.index(mat_default) if mat_default in materials else 0

        def on_fc_region_change():
            st.session_state["global_region"] = st.session_state["fc_region_select"]

        def on_fc_material_change():
            st.session_state["global_material"] = st.session_state["fc_material_select"]

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            sel_region = st.selectbox(
                "Select Operating Region",
                options=regions,
                index=reg_i,
                key="fc_region_select",
                on_change=on_fc_region_change
            )
        with filter_col2:
            sel_material = st.selectbox(
                "Select Material Category",
                options=materials,
                index=mat_i,
                key="fc_material_select",
                on_change=on_fc_material_change
            )

        # Filter dataset
        df_filtered = preds_df.copy()
        if sel_region != "ALL" and "Region_Name" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Region_Name"] == sel_region]
        if sel_material != "ALL" and "Material_Name" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["Material_Name"] == sel_material]

        pred_col = "Forecast_Prediction" if "Forecast_Prediction" in df_filtered.columns else "P50"
        p10_col = "Forecast_Prediction_P10" if "Forecast_Prediction_P10" in df_filtered.columns else pred_col
        p90_col = "Forecast_Prediction_P90" if "Forecast_Prediction_P90" in df_filtered.columns else pred_col

        tot_pred = float(df_filtered[pred_col].sum()) if pred_col in df_filtered.columns else 0.0
        avg_p10 = float(df_filtered[p10_col].mean()) if p10_col in df_filtered.columns else 0.0
        avg_p90 = float(df_filtered[p90_col].mean()) if p90_col in df_filtered.columns else 0.0

        # Summary Metric Cards
        card1, card2, card3 = st.columns(3)
        with card1:
            st.metric("Forecast Horizon", "312 Weeks (6 Years)", help="Weekly timeline observations 2020-2025")
        with card2:
            st.metric("Predicted Total Demand", f"{tot_pred:,.0f} Units", help="Sum of P50 predicted demand across selected horizon")
        with card3:
            st.metric("80% Confidence Interval", f"{avg_p10:,.0f} - {avg_p90:,.0f} Units", help="Average weekly P10 (Lower) to P90 (Upper) interval")

        st.markdown("---")

        # Executive Directive Card
        rec_df = load_procurement_recommendations_df()
        current_inv = 500.0
        lead_time_w = 4.0
        safety_s = float(df_filtered[pred_col].std() * 0.5) if (pred_col in df_filtered.columns and len(df_filtered) > 1) else 80.0
        avg_forecast = float(df_filtered[pred_col].mean()) if pred_col in df_filtered.columns else 450.0

        if not rec_df.empty and sel_material != "ALL":
            for _, rrow in rec_df.iterrows():
                m_type = str(rrow.get("Material_Type", "")).lower()
                s_mat = sel_material.lower()
                if s_mat in m_type or m_type in s_mat:
                    current_inv = float(rrow.get("Current_Inventory", 500.0))
                    lead_time_w = float(rrow.get("Lead_Time_Weeks", 4.0))
                    if "Safety_Stock_Qty" in rrow:
                        safety_s = float(rrow.get("Safety_Stock_Qty", safety_s))
                    break

        render_decision_recommendation_card(
            material=sel_material if sel_material != "ALL" else "All Materials",
            region=sel_region if sel_region != "ALL" else "All Regions",
            forecast_demand=avg_forecast,
            p90_demand=avg_p90,
            current_inventory=current_inv,
            lead_time_weeks=lead_time_w,
            safety_stock=safety_s,
            unit_cost_inr=1250.0
        )

        st.markdown("---")

        # ---------------------------------------------------------
        # SECTION 2: Actual vs Predicted Demand
        # ---------------------------------------------------------
        st.subheader("2️⃣ Actual vs. Predicted Demand Time-Series (P50 Point Forecast)")
        
        # Point Forecast Line Chart
        chart_title1 = f"Actual Demand vs. Predicted Demand P50 ({sel_material} - Region: {sel_region})"
        chart_key_str1 = f"plotly_actual_vs_pred_{sel_region}_{sel_material}"
        render_interactive_forecast_chart(df_filtered, title=chart_title1, show_quantile_bands=False, chart_key=chart_key_str1)

        st.markdown("---")

        # ---------------------------------------------------------
        # SECTION 3: Probabilistic Forecast (P10, P50, P90)
        # ---------------------------------------------------------
        st.subheader("3️⃣ Probabilistic Quantile Forecast (P10 - P50 - P90 Confidence Band)")

        # Full Probabilistic Chart
        chart_title2 = f"Probabilistic Forecast Bounds P10-P50-P90 ({sel_material} - Region: {sel_region})"
        chart_key_str2 = f"plotly_probabilistic_{sel_region}_{sel_material}"
        render_interactive_forecast_chart(df_filtered, title=chart_title2, show_quantile_bands=True, chart_key=chart_key_str2)
