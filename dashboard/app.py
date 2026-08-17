import streamlit as st
from pathlib import Path

# Page Config (Must be very first Streamlit call)
st.set_page_config(
    page_title="POWERGRID Decision Support System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Corporate IEEE Theme CSS
st.markdown(
    """
    <style>
        /* Base page layout */
        .main .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 95%;
        }
        
        /* Typography */
        .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
            color: #003366;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        /* Sidebar Base Styling */
        section[data-testid="stSidebar"] {
            background-color: #f8fafc !important;
            border-right: 1px solid #cbd5e1 !important;
        }
        
        /* Uniform & Equal-Sized Sidebar Navigation Radio Boxes */
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            display: flex !important;
            flex-direction: column !important;
            width: 100% !important;
            gap: 8px !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            display: flex !important;
            align-items: center !important;
            width: 100% !important;
            min-height: 44px !important;
            max-height: 44px !important;
            box-sizing: border-box !important;
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 6px !important;
            padding: 0 14px !important;
            margin: 0 !important;
            cursor: pointer !important;
            transition: all 0.2s ease-in-out !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {
            width: 100% !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
            color: #1e293b !important;
            font-weight: 600 !important;
            font-size: 13.5px !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1.2 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background-color: #e6f0fa !important;
            border-color: #0066cc !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover p {
            color: #003366 !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background-color: #003366 !important;
            border-color: #003366 !important;
            box-shadow: 0 2px 5px rgba(0, 51, 102, 0.2) !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] p,
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
            color: #ffffff !important;
            font-weight: 700 !important;
        }
        
        /* Metric card styling */
        div[data-testid="stMetric"] {
            background-color: #f8fafc !important;
            border-radius: 8px !important;
            padding: 14px 18px !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }
        div[data-testid="stMetric"] label {
            color: #475569 !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            color: #003366 !important;
            font-size: 24px !important;
            font-weight: 700 !important;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #003366 !important;
            color: #ffffff !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
        }
        .stButton>button:hover {
            background-color: #002244 !important;
            color: #ffffff !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e6f0fa !important;
            color: #003366 !important;
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

def main():
    """Master Streamlit Dashboard Application Controller."""
    # Executive Header Banner (High Contrast Guarantee)
    st.markdown(
        """
        <div style="background-color: #003366; padding: 20px 26px; border-radius: 8px; margin-bottom: 22px; color: #ffffff;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="color: #ffffff !important; font-size: 22px; font-weight: 800; margin: 0; font-family: 'Segoe UI', sans-serif; letter-spacing: 0.3px;">
                        AI-Based Material Demand Forecasting and Supply Chain Optimization
                    </div>
                    <div style="color: #cbd5e1 !important; margin: 5px 0 0 0; font-size: 14px; font-weight: 600; letter-spacing: 0.2px;">
                        POWERGRID Decision Support System & Executive Analytics Portal
                    </div>
                </div>
                <div style="text-align: right; background-color: rgba(255,255,255,0.12); padding: 8px 16px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.25);">
                    <span style="font-size: 11px; color: #cbd5e1; display: block; text-transform: uppercase; font-weight: 700;">Status</span>
                    <span style="font-size: 13px; color: #38bdf8; font-weight: 800;">● Operational</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    state = render_sidebar()
    selected_page = state["page"]

    if "Executive Dashboard" in selected_page or "Home" in selected_page:
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
    else:
        render_home_page()

if __name__ == "__main__":
    main()
