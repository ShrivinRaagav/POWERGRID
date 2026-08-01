import streamlit as st
from typing import Dict, Any, List
from dashboard.utils import load_best_model_info, load_processed_dataset

def render_sidebar() -> Dict[str, Any]:
    """
    Renders standard navigation sidebar with POWERGRID branding, global filters, and best model status.
    """
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="color: #003366; margin-bottom: 0;">⚡ POWERGRID</h2>
            <p style="font-size: 13px; color: #555; font-weight: bold; margin-top: 2px;">
                Demand Forecasting & XAI Supply Chain System
            </p>
        </div>
        <hr style="margin: 5px 0 15px 0; border-top: 1px solid #ccc;"/>
        """,
        unsafe_allow_html=True
    )

    # Page Navigation
    page = st.sidebar.radio(
        "Navigation Menu",
        options=[
            "🏠 Executive Home",
            "📊 Data Quality & Pipeline",
            "📈 Material Demand Forecasting",
            "⚖️ Forecast Model Evaluation",
            "🔍 SHAP Explainability (XAI)",
            "🎯 Multi-Objective Optimization",
            "📥 Reports & Download Center"
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Global Data Filters")

    # Load dataset for filter options
    df_proc = load_processed_dataset()

    materials = ["All Materials"]
    regions = ["All Regions"]

    if not df_proc.empty:
        if "Material_Type" in df_proc.columns:
            materials += sorted(list(df_proc["Material_Type"].dropna().unique()))
        if "Region" in df_proc.columns:
            regions += sorted(list(df_proc["Region"].dropna().unique()))

    if len(materials) == 1:
        materials += ["Transformer_Oil", "Conductor_ACSR", "Insulator_Porcelain", "Steel_Structure", "Control_Cable"]
    if len(regions) == 1:
        regions += ["NR", "WR", "SR", "ER", "NER"]

    selected_material = st.sidebar.selectbox("Material Category", options=materials, index=0)
    selected_region = st.sidebar.selectbox("POWERGRID Regional Grid", options=regions, index=0)

    # Best Model Status Card
    st.sidebar.markdown("---")
    best_info = load_best_model_info()
    best_model_name = str(best_info.get("best_model", "xgboost")).upper()

    st.sidebar.markdown(
        f"""
        <div style="background-color: #f0f4f8; padding: 12px; border-radius: 8px; border-left: 4px solid #003366;">
            <p style="margin:0; font-size:12px; color:#555;">ACTIVE BEST MODEL</p>
            <h4 style="margin:4px 0 2px 0; color:#003366;">{best_model_name}</h4>
            <p style="margin:0; font-size:11px; color:#777;">Selected via Lowest RMSE</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    return {
        "page": page,
        "material": selected_material,
        "region": selected_region
    }
