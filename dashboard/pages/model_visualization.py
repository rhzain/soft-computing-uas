from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.config import (
    CV_RESULTS,
    FINAL_EVALUATION,
    MODEL_PACKAGE_NAME,
    MODEL_PACKAGE_PATH,
    RA_GRID_RESULTS,
    VISUAL_ASSETS,
)


def render() -> None:
    st.title("Project & Model")
    st.caption(
        "Wavelet-ANFIS analysis for hydraulic pump leakage classification."
    )

    render_model_metrics()
    render_project_context()

    (
        tab_pipeline,
        tab_architecture,
        tab_tuning,
        tab_evaluation,
        tab_membership,
        tab_package,
    ) = st.tabs(
        [
            "Pipeline",
            "Architecture",
            "Tuning r_a",
            "Evaluation",
            "Membership Function",
            "Model Package",
        ]
    )

    with tab_pipeline:
        render_pipeline()

    with tab_architecture:
        render_architecture()

    with tab_tuning:
        render_tuning()

    with tab_evaluation:
        render_evaluation()

    with tab_membership:
        render_membership_functions()

    with tab_package:
        render_model_package()


def render_model_metrics() -> None:
    metric_columns = st.columns(5)
    metric_columns[0].metric("Model", "WANFIS")
    metric_columns[1].metric("Input Features", "15")
    metric_columns[2].metric("Fuzzy Rules", "4")
    metric_columns[3].metric("Best r_a", "0.6")
    metric_columns[4].metric("Classes", "3")


def render_project_context() -> None:
    st.markdown(
        """
        This project combines wavelet-based feature extraction, Subtractive
        Clustering, and a PyTorch-based ANFIS classifier to identify pump
        leakage conditions from hydraulic-system sensor data.
        """
    )

    left_column, right_column = st.columns([1, 1])

    with left_column:
        final_table = pd.DataFrame(
            {
                "Metrik Final": list(FINAL_EVALUATION.keys()),
                "Nilai": list(FINAL_EVALUATION.values()),
            }
        )
        st.markdown("**Final Evaluation**")
        st.dataframe(final_table, hide_index=True, use_container_width=True)

    with right_column:
        cv_table = pd.DataFrame(
            {
                "Metrik CV": list(CV_RESULTS.keys()),
                "Nilai": list(CV_RESULTS.values()),
            }
        )
        st.markdown("**Cross-Validation Summary**")
        st.dataframe(cv_table, hide_index=True, use_container_width=True)


def render_pipeline() -> None:
    st.subheader("Model Pipeline")

    pipeline_steps = pd.DataFrame(
        [
            {
                "No": 1,
                "Stage": "Sensor data ingestion",
                "Output": "PS1, PS2, PS3, TS1, and TS2 matrices",
            },
            {
                "No": 2,
                "Stage": "Wavelet feature extraction",
                "Output": "15 features: mean_cA, std_cA, energy_cD",
            },
            {
                "No": 3,
                "Stage": "Feature standardization",
                "Output": "Z-score scaled features",
            },
            {
                "No": 4,
                "Stage": "Subtractive Clustering",
                "Output": "Initial fuzzy rule centers",
            },
            {
                "No": 5,
                "Stage": "ANFIS training",
                "Output": "Three-class pump_leak classifier",
            },
            {
                "No": 6,
                "Stage": "Evaluation",
                "Output": "Accuracy, weighted F1-score, confusion matrix",
            },
        ]
    )

    st.dataframe(pipeline_steps, hide_index=True, use_container_width=True)
    st.code(
        "Sensor signals -> Wavelet features -> Scaling -> "
        "Subtractive Clustering -> ANFIS -> Prediction",
        language="text",
    )


def render_architecture() -> None:
    st.subheader("WANFIS Architecture")

    left_column, right_column = st.columns([1.1, 1])

    with left_column:
        st.markdown(
            """
            The model architecture is designed for compact, interpretable
            classification:

            1. Sensor signals are transformed using Discrete Wavelet Transform.
            2. Each sensor contributes `mean_cA`, `std_cA`, and `energy_cD`.
            3. Subtractive Clustering initializes fuzzy rule centers.
            4. ANFIS optimizes Gaussian membership functions and consequent parameters.
            5. The final output is a three-class leakage prediction.
            """
        )

    with right_column:
        architecture_table = pd.DataFrame(
            [
                {"Layer": "L1", "Function": "Gaussian membership function"},
                {"Layer": "L2", "Function": "Rule firing strength"},
                {"Layer": "L3", "Function": "Normalized firing strength"},
                {"Layer": "L4", "Function": "Linear Takagi-Sugeno consequent"},
                {"Layer": "L5", "Function": "Three-class logits"},
            ]
        )
        st.dataframe(architecture_table, hide_index=True, use_container_width=True)

    render_method_positioning()

    st.code(
        "Sensor signals -> Wavelet features -> StandardScaler -> "
        "Subtractive Clustering -> ANFIS -> pump_leak class",
        language="text",
    )


def render_method_positioning() -> None:
    st.markdown("**Method Positioning**")

    comparison_table = pd.DataFrame(
        [
            {
                "Aspect": "Feature representation",
                "WANFIS Project": "Wavelet-derived compact features",
                "Purpose": "Reduce high-dimensional time-series signals",
            },
            {
                "Aspect": "Rule initialization",
                "WANFIS Project": "Subtractive Clustering",
                "Purpose": "Build data-driven fuzzy rule centers",
            },
            {
                "Aspect": "Learning mechanism",
                "WANFIS Project": "Gradient-based ANFIS training",
                "Purpose": "Optimize membership and consequent parameters",
            },
            {
                "Aspect": "Interpretability",
                "WANFIS Project": "Four fuzzy rules",
                "Purpose": "Keep the classifier compact and explainable",
            },
        ]
    )
    st.dataframe(comparison_table, hide_index=True, use_container_width=True)


def render_tuning() -> None:
    st.subheader("Tuning Radius Subtractive Clustering")

    st.markdown(
        "`r_a` controls cluster granularity. Lower values generally produce more "
        "rules, while higher values produce a more compact rule base."
    )

    results = pd.DataFrame(RA_GRID_RESULTS)
    st.dataframe(results, hide_index=True, use_container_width=True)

    chart_data = results.sort_values("r_a").set_index("r_a")[
        ["Mean Accuracy", "Mean F1"]
    ]
    st.line_chart(chart_data)

    render_visual_asset(
        title="r_a Experiment",
        asset_key="r_a Experiment",
        caption="Cross-validation comparison across candidate cluster radii.",
    )


def render_evaluation() -> None:
    st.subheader("Final Model Evaluation")

    metrics = pd.DataFrame(
        {
            "Metrik": list(FINAL_EVALUATION.keys()),
            "Nilai": list(FINAL_EVALUATION.values()),
        }
    )

    left_column, right_column = st.columns([1, 1.2])

    with left_column:
        st.dataframe(metrics, hide_index=True, use_container_width=True)

        st.markdown(
            """
            Class 1 has the lowest per-class accuracy. Weak leakage is typically
            more difficult to separate because its signal profile can overlap
            with normal and severe leakage conditions.
            """
        )

    with right_column:
        render_visual_asset(
            title="Evaluation Results",
            asset_key="Evaluation Results",
            caption="Confusion matrix and final evaluation summary.",
        )

    render_visual_asset(
        title="Training Curves",
        asset_key="Training Curves",
        caption="Training dynamics for loss and accuracy.",
    )


def render_membership_functions() -> None:
    st.subheader("Membership Function")

    st.markdown(
        """
        Gaussian membership functions are initialized from Subtractive
        Clustering centers. During training, `mean` and `sigma` are adjusted
        through gradient-based optimization.
        """
    )

    render_visual_asset(
        title="Membership Functions",
        asset_key="Membership Functions",
        caption="Membership functions before and after ANFIS training.",
    )


def render_model_package() -> None:
    st.subheader("Model Package")

    if MODEL_PACKAGE_PATH.exists():
        size_kb = MODEL_PACKAGE_PATH.stat().st_size / 1024

        package_table = pd.DataFrame(
            [
                {"Property": "Package", "Value": MODEL_PACKAGE_NAME},
                {"Property": "Size", "Value": f"{size_kb:.2f} KB"},
                {"Property": "Format", "Value": "PyTorch checkpoint (.pth)"},
                {"Property": "Repository status", "Value": "Available"},
            ]
        )
        st.dataframe(package_table, hide_index=True, use_container_width=True)

        st.markdown(
            """
            The checkpoint stores ANFIS weights and metadata required for
            consistent inference, including `best_ra`, feature names, scaler,
            class labels, and Subtractive Clustering centers.
            """
        )
    else:
        st.warning("Model package is not available in the repository.")


def render_visual_asset(title: str, asset_key: str, caption: str) -> None:
    image_path = VISUAL_ASSETS[asset_key]

    if image_path.exists():
        st.markdown(f"**{title}**")
        st.image(str(image_path), use_container_width=True)
        st.caption(caption)
    else:
        st.warning(f"Visual asset is not available: `{image_path.name}`")
