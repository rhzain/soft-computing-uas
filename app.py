from __future__ import annotations

import streamlit as st

from dashboard.config import APP_TITLE, NAVIGATION_ITEMS
from dashboard.pages import (
    anomaly_simulation,
    dataset_explorer,
    model_visualization,
)


PAGES = {
    "Project & Model": model_visualization.render,
    "Dataset Explorer": dataset_explorer.render,
    "Anomaly Simulation": anomaly_simulation.render,
}


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            .metric-card {
                padding: 1rem;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                background: #ffffff;
            }
            .section-note {
                color: #4b5563;
                font-size: 0.95rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.title(APP_TITLE)
    st.sidebar.caption("Hydraulic pump leakage classification")
    return st.sidebar.radio(
        "Navigation",
        NAVIGATION_ITEMS,
        index=0,
    )


def main() -> None:
    configure_page()
    selected_page = render_sidebar()
    PAGES[selected_page]()


if __name__ == "__main__":
    main()
