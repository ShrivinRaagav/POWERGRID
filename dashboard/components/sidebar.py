import streamlit as st
from typing import Dict, Any, List
from dashboard.utils import load_best_model_info, load_processed_dataset

def render_sidebar() -> Dict[str, Any]:
    """
    Renders standard executive navigation sidebar with POWERGRID branding, 
    sleek page navigation menu, and active model status.
    """
    st.sidebar.markdown(
        """
        <div style="background-color: #ffffff; padding: 16px; border-radius: 8px; border: 1px solid #cbd5e1; text-align: center; margin-bottom: 15px;">
            <div style="font-size: 26px; font-weight: 800; color: #003366; letter-spacing: 0.5px;">⚡ POWERGRID</div>
            <div style="font-size: 11px; color: #475569; font-weight: 700; text-transform: uppercase; margin-top: 4px; letter-spacing: 0.5px;">
                Decision Support System
            </div>
            <div style="font-size: 11px; color: #0066cc; font-weight: 600; margin-top: 2px;">
                Material Demand & XAI Supply Chain
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown("<p style='font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>NAVIGATION MENU</p>", unsafe_allow_html=True)

    # Page Navigation Menu
    page = st.sidebar.radio(
        "Select Page",
        options=[
            "🏠 Executive Dashboard",
            "📊 Data Quality",
            "📈 Forecasting",
            "⚖️ Model Evaluation",
            "🔍 Explainability",
            "🎯 Optimization"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("<p style='font-size: 12px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;'>SYSTEM METADATA</p>", unsafe_allow_html=True)

    # Best Model Status Card
    best_info = load_best_model_info()
    best_model_name = str(best_info.get("best_model", "lightgbm_quantile")).upper()

    st.sidebar.markdown(
        f"""
        <div style="background-color: #ffffff; padding: 12px 14px; border-radius: 6px; border-left: 4px solid #003366; border: 1px solid #cbd5e1; margin-bottom: 10px;">
            <p style="margin:0; font-size:11px; color:#64748b; font-weight: 600; text-transform: uppercase;">ACTIVE BEST MODEL</p>
            <h4 style="margin:4px 0 2px 0; color:#003366; font-size: 14px; font-weight: 700;">{best_model_name}</h4>
            <p style="margin:0; font-size:11px; color:#475569;">Lowest Out-of-Sample Test RMSE</p>
        </div>
        
        <div style="background-color: #ffffff; padding: 12px 14px; border-radius: 6px; border-left: 4px solid #0066cc; border: 1px solid #cbd5e1;">
            <p style="margin:0; font-size:11px; color:#64748b; font-weight: 600; text-transform: uppercase;">DATASET OVERVIEW</p>
            <p style="margin:4px 0 0 0; font-size:12px; color:#003366; font-weight: 600;">• 9,360 Timeline Samples</p>
            <p style="margin:2px 0 0 0; font-size:12px; color:#003366; font-weight: 600;">• 6 Material Categories</p>
            <p style="margin:2px 0 0 0; font-size:12px; color:#003366; font-weight: 600;">• 5 Regional Grids (NR, SR, WR, ER, NER)</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    return {
        "page": page
    }
