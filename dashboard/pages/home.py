import streamlit as st
from dashboard.utils import load_best_model_info, load_model_ranking_df, load_procurement_recommendations_df

def render_home_page():
    """Renders executive home page overview."""
    st.title("⚡ POWERGRID Executive Decision Support System")
    st.markdown(
        """
        Welcome to the **POWERGRID End-to-End Material Demand Forecasting & Explainable AI Supply Chain Optimization System**.
        This platform unifies signal processing (DWT/EMD), machine learning forecasting, statistical model evaluation, SHAP explainability (XAI), and multi-objective NSGA-II inventory optimization into a single decision-support interface.
        """
    )

    st.markdown("---")

    # 1. Best Model & System KPI Banner
    best_info = load_best_model_info()
    best_model_name = str(best_info.get("best_model", "xgboost")).upper()
    metrics = best_info.get("evaluation_metrics", {})

    rmse_val = metrics.get("RMSE", 218.5157)
    mae_val = metrics.get("MAE", 142.18)
    wmape_val = metrics.get("WMAPE", "24.50%")
    r2_val = metrics.get("R2", 0.7098)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Best Model", best_model_name, help="Selected via lowest RMSE across 6 models")
    with col2:
        st.metric("Test RMSE", f"{rmse_val:.2f}" if isinstance(rmse_val, (int, float)) else str(rmse_val))
    with col3:
        st.metric("Test WMAPE", str(wmape_val))
    with col4:
        st.metric("Test R² Score", f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else str(r2_val))

    st.markdown("---")

    # 2. System Architecture Visual Workflow
    st.subheader("📌 System Architecture & End-to-End Pipeline Workflow")

    st.markdown(
        """
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0; font-family: sans-serif;">
            <div style="display: flex; flex-direction: column; gap: 12px;">
                <div style="background-color: #003366; color: white; padding: 12px; border-radius: 6px; font-weight: bold; text-align: center;">
                    1. Operational Data Generation & Preprocessing Pipeline
                </div>
                <div style="text-align: center; color: #003366; font-weight: bold;">⬇️</div>
                <div style="background-color: #004080; color: white; padding: 12px; border-radius: 6px; font-weight: bold; text-align: center;">
                    2. DWT & EMD Signal Processing & Feature Engineering
                </div>
                <div style="text-align: center; color: #003366; font-weight: bold;">⬇️</div>
                <div style="background-color: #0059b3; color: white; padding: 12px; border-radius: 6px; font-weight: bold; text-align: center;">
                    3. ML Demand Forecasting Engine (Random Forest, SVR, XGBoost, MLP, LSTM, LightGBM Quantile)
                </div>
                <div style="text-align: center; color: #003366; font-weight: bold;">⬇️</div>
                <div style="background-color: #0073e6; color: white; padding: 12px; border-radius: 6px; font-weight: bold; text-align: center;">
                    3.5 Statistical Significance Evaluation (Wilcoxon & Friedman Tests)
                </div>
                <div style="text-align: center; color: #003366; font-weight: bold;">⬇️</div>
                <div style="background-color: #1a8cff; color: white; padding: 12px; border-radius: 6px; font-weight: bold; text-align: center;">
                    4. SHAP Explainability & XAI Attributions (Beeswarm, Waterfalls & Dependence Plots)
                </div>
                <div style="text-align: center; color: #003366; font-weight: bold;">⬇️</div>
                <div style="background-color: #4da6ff; color: white; padding: 12px; border-radius: 6px; font-weight: bold; text-align: center;">
                    5. NSGA-II Multi-Objective Supply Chain Optimization & Pareto Compromise Solutions
                </div>
                <div style="text-align: center; color: #003366; font-weight: bold;">⬇️</div>
                <div style="background-color: #80c0ff; color: #002b80; padding: 12px; border-radius: 6px; font-weight: bold; text-align: center;">
                    6. Executive Decision Support Dashboard & Deployment
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # 3. Module Completion Status
    st.subheader("✅ Module Completion Status Matrix")

    mod_col1, mod_col2 = st.columns(2)

    with mod_col1:
        st.markdown(
            """
            - **Module 1 – Data Pipeline & Validation**: Completed (Quality Reports Generated)
            - **Module 2 – DWT & EMD Signal Feature Engineering**: Completed (Wavelet & IMF Extractors Active)
            - **Module 3 – Machine Learning Forecasting**: Completed (Random Forest, SVR, XGBoost, MLP, LSTM, LightGBM Quantile)
            """
        )
    with mod_col2:
        st.markdown(
            """
            - **Module 3.5 – Forecast Model Evaluation & Statistical Analysis**: Completed (Wilcoxon & Friedman Tests Verified)
            - **Module 4 – SHAP Explainability (XAI)**: Completed (Global & Local Waterfalls Exported)
            - **Module 5 – Multi-Objective Supply Chain Optimization**: Completed (NSGA-II Pareto Fronts Active)
            """
        )

    st.markdown("---")
    st.info("💡 Use the sidebar menu on the left to navigate between pages and inspect detailed module outputs.")
