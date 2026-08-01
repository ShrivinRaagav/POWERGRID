import streamlit as st
from pathlib import Path
from PIL import Image
from typing import Optional

from dashboard.utils import load_image

def render_publication_figure(
    filepath: Path,
    caption: str
):
    """
    Renders 300 DPI IEEE publication figure with clean white background wrapper.
    """
    img = load_image(filepath)
    if img is not None:
        st.image(img, caption=caption, use_column_width="auto")
    else:
        st.info(f"Figure image not found at `{filepath.name}`")

def render_kpi_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    help_text: Optional[str] = None
):
    """Renders styled metric card."""
    st.metric(label=label, value=value, delta=delta, help=help_text)
