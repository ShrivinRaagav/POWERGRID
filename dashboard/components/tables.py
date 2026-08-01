import streamlit as st
import pandas as pd
from typing import Optional

def render_styled_dataframe(
    df: pd.DataFrame,
    title: Optional[str] = None,
    csv_filename: Optional[str] = None
):
    """
    Renders styled Streamlit dataframe with optional title and CSV download button.
    """
    if title:
        st.markdown(f"#### {title}")

    if not df.empty:
        st.dataframe(df, hide_index=True)
        if csv_filename:
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {csv_filename}",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
                key=f"dl_{csv_filename}"
            )
    else:
        st.info("No data available to display in table.")
