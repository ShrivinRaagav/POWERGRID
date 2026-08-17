import streamlit as st
from dashboard.utils import load_processed_dataset
from dashboard.components.tables import render_styled_dataframe

def render_data_quality_page():
    """Renders Data Quality, Preprocessing, and Sample Preview page."""
    st.title("📊 Data Quality & Preprocessed Dataset Preview")

    # 1. Dataset Preview & Summary
    df_proc = load_processed_dataset()
    if not df_proc.empty:
        g_reg = st.session_state.get("global_region", "ALL")
        g_mat = st.session_state.get("global_material", "ALL")
        
        df_preview = df_proc.copy()
        if g_reg != "ALL" and "Region" in df_preview.columns:
            df_preview = df_preview[df_preview["Region"].astype(str).str.lower() == g_reg.lower()]
        if g_mat != "ALL" and "Material_Type" in df_preview.columns:
            df_preview = df_preview[df_preview["Material_Type"].astype(str).str.lower() == g_mat.lower()]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Processed Records", f"{len(df_proc):,}")
        with col2:
            st.metric("Engineered Features", f"{len(df_proc.columns):,}")
        with col3:
            st.metric("Filtered Records", f"{len(df_preview):,}")

        st.markdown("---")
        st.subheader(f"📋 Dataset Sample Preview ({g_mat} | Region: {g_reg})")
        render_styled_dataframe(df_preview.head(20), csv_filename="processed_dataset_sample.csv")
    else:
        st.info("Processed dataset not found.")
