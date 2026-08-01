import streamlit as st
from pathlib import Path

# Page Config (Must be very first Streamlit call)
st.set_page_config(
    page_title="POWERGRID Decision Support System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom IEEE Executive Theme CSS with High Contrast Token Guarantees
st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4 {
            color: #003366 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 700;
        }
        div[data-testid="stMetric"] {
            background-color: #f0f4f8 !important;
            border-radius: 10px !important;
            padding: 14px 18px !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div[data-testid="stMetricValue"],
        div[data-testid="stMetric"] div[data-testid="stMetricDelta"],
        div[data-testid="stMetric"] span {
            color: #003366 !important;
            font-weight: 700 !important;
        }
        .stButton>button {
            background-color: #003366 !important;
            color: #ffffff !important;
            border-radius: 6px !important;
            font-weight: bold !important;
            border: none !important;
        }
        .stButton>button:hover {
            background-color: #002244 !important;
            color: #ffffff !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

from dashboard.components.sidebar import render_sidebar
from dashboard.views.home import render_home_page
from dashboard.views.data_quality import render_data_quality_page
from dashboard.views.forecasting import render_forecasting_page
from dashboard.views.evaluation import render_evaluation_page
from dashboard.views.explainability import render_explainability_page
from dashboard.views.optimization import render_optimization_page
from dashboard.views.reports import render_reports_page

def main():
    """Master Streamlit Dashboard Application Controller."""
    state = render_sidebar()
    selected_page = state["page"]

    if "Home" in selected_page:
        render_home_page()
    elif "Data Quality" in selected_page:
        render_data_quality_page()
    elif "Forecasting" in selected_page:
        render_forecasting_page()
    elif "Evaluation" in selected_page:
        render_evaluation_page()
    elif "Explainability" in selected_page:
        render_explainability_page()
    elif "Optimization" in selected_page:
        render_optimization_page()
    elif "Reports" in selected_page:
        render_reports_page()
    else:
        render_home_page()

if __name__ == "__main__":
    main()
