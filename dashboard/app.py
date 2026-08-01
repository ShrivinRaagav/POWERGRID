import streamlit as st
from pathlib import Path

# Page Config (Must be very first Streamlit call)
st.set_page_config(
    page_title="POWERGRID Decision Support System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom IEEE Executive Theme CSS
st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            color: #003366;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stMetric {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #e9ecef;
        }
        .stButton>button {
            border-radius: 6px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

from dashboard.components.sidebar import render_sidebar
from dashboard.pages.home import render_home_page
from dashboard.pages.data_quality import render_data_quality_page
from dashboard.pages.forecasting import render_forecasting_page
from dashboard.pages.evaluation import render_evaluation_page
from dashboard.pages.explainability import render_explainability_page
from dashboard.pages.optimization import render_optimization_page
from dashboard.pages.reports import render_reports_page

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
