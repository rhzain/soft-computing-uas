from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.config import SENSOR_LIST
from dashboard.data import get_dataset_summary, load_profile, load_sensor_data


def render() -> None:
    summary = get_dataset_summary()

    st.title("Dataset Explorer")
    st.caption(
        "Sensor data profile, target distribution, and cycle-level signal preview."
    )

    render_dataset_metrics(summary)

    tab_summary, tab_distribution, tab_sensor, tab_profile = st.tabs(
        ["Dataset Summary", "Target Distribution", "Sensor Preview", "Profile Data"]
    )

    with tab_summary:
        render_dataset_summary(summary)

    with tab_distribution:
        render_target_distribution(summary.class_distribution)

    with tab_sensor:
        render_sensor_preview(summary.sensor_shapes)

    with tab_profile:
        render_profile_preview()


def render_dataset_metrics(summary) -> None:
    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Operating Cycles",
        f"{summary.cycle_count:,}".replace(",", "."),
    )
    metric_columns[1].metric("Selected Sensors", len(SENSOR_LIST))
    metric_columns[2].metric("Target", "pump_leak")
    metric_columns[3].metric("Classes", len(summary.class_distribution))
    metric_columns[4].metric("Wavelet Features", "15")


def render_dataset_summary(summary) -> None:
    st.subheader("Dataset Context")

    st.markdown(
        """
        Dataset yang digunakan adalah **Condition Monitoring of Hydraulic
        Systems**. Setiap baris merepresentasikan satu cycle operasi sistem
        hidrolik. Analisis berfokus pada klasifikasi kondisi `pump_leak`
        berdasarkan sensor tekanan dan temperatur.
        """
    )

    left_column, right_column = st.columns([1, 1])

    with left_column:
        class_table = pd.DataFrame(
            {
                "Class": summary.class_distribution["class"],
            }
        )
        st.markdown("**Target Classes**")
        st.dataframe(class_table, hide_index=True, use_container_width=True)

    with right_column:
        sensor_table = pd.DataFrame(
            {
                "Sensor": SENSOR_LIST,
                "Operating Cycles": [
                    summary.sensor_shapes[sensor][0] for sensor in SENSOR_LIST
                ],
                "Time Steps": [
                    summary.sensor_shapes[sensor][1] for sensor in SENSOR_LIST
                ],
            }
        )
        st.markdown("**Selected Sensors**")
        st.dataframe(sensor_table, hide_index=True, use_container_width=True)

    render_feature_summary()
    render_stable_flag_distribution()


def render_feature_summary() -> None:
    st.subheader("Wavelet Features")

    feature_table = pd.DataFrame(
        [
            {
                "Fitur": "mean_cA",
                "Sumber": "Approximation coefficient",
                "Makna": "Average low-frequency signal component",
            },
            {
                "Fitur": "std_cA",
                "Sumber": "Approximation coefficient",
                "Makna": "Variation of low-frequency signal component",
            },
            {
                "Fitur": "energy_cD",
                "Sumber": "Detail coefficient",
                "Makna": "Energy of high-frequency signal variation",
            },
        ]
    )

    st.dataframe(feature_table, hide_index=True, use_container_width=True)
    st.info(
        "Each selected sensor contributes three wavelet features, producing 15 "
        "model inputs in total."
    )


def render_stable_flag_distribution() -> None:
    profile = load_profile()
    stable_counts = profile["stable_flag"].value_counts().sort_index()

    st.subheader("Stable Flag Distribution")
    st.bar_chart(stable_counts)
    st.caption(
        "`stable_flag = 0` indicates stable operation, while `stable_flag = 1` "
        "indicates that steady-state conditions may not have been reached."
    )


def render_target_distribution(class_distribution: pd.DataFrame) -> None:
    st.subheader("Target Distribution")

    left_column, right_column = st.columns([1.1, 1])

    with left_column:
        chart_data = class_distribution.set_index("class")["count"]
        st.bar_chart(chart_data)
        st.caption("Sample count for each `pump_leak` class.")

    with right_column:
        table = class_distribution.copy()
        table = table[["class", "count", "percentage"]]
        table.columns = ["Class", "Count", "Percentage (%)"]
        st.dataframe(table, hide_index=True, use_container_width=True)

    st.markdown(
        """
        The normal class is the largest group, while weak and severe leakage
        have balanced sample counts. Weighted F1-score is used alongside
        accuracy to account for this distribution.
        """
    )


def render_sensor_preview(sensor_shapes: dict[str, tuple[int, int]]) -> None:
    st.subheader("Sensor Signal Preview")

    selected_sensor = st.selectbox("Sensor", SENSOR_LIST)
    sensor_data = load_sensor_data(selected_sensor)
    cycle_count, time_steps = sensor_shapes[selected_sensor]

    control_columns = st.columns([1, 1, 1])
    cycle_index = control_columns[0].slider(
        "Cycle",
        min_value=0,
        max_value=cycle_count - 1,
        value=0,
    )
    max_points = control_columns[1].selectbox(
        "Maximum chart points",
        options=[250, 500, 1000, 2000, time_steps],
        index=2,
    )
    show_table = control_columns[2].checkbox("Show sampled values", value=False)

    step = max(1, time_steps // int(max_points))
    series = sensor_data.iloc[cycle_index, ::step]
    chart_data = pd.DataFrame(
        {
            "time_step": series.index.astype(int),
            "value": series.values.astype(float),
        }
    )

    st.line_chart(chart_data, x="time_step", y="value")

    info_columns = st.columns(3)
    info_columns[0].metric("Sensor", selected_sensor)
    info_columns[1].metric("Time Steps", f"{time_steps:,}".replace(",", "."))
    info_columns[2].metric("Downsample Step", step)

    if show_table:
        st.dataframe(chart_data, hide_index=True, use_container_width=True)

    st.caption(
        "The chart displays one operating cycle. Pressure sensors contain 6000 "
        "time steps, while temperature sensors contain 60 time steps."
    )


def render_profile_preview() -> None:
    st.subheader("Profile Data")

    profile = load_profile()
    selected_columns = st.multiselect(
        "Displayed columns",
        options=list(profile.columns),
        default=list(profile.columns),
    )
    row_count = st.slider("Rows", min_value=5, max_value=100, value=20)

    if selected_columns:
        st.dataframe(
            profile[selected_columns].head(row_count),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.warning("Select at least one column.")

    with st.expander("Profile column definitions"):
        st.markdown(
            """
            - `cooler_cond`: cooler condition percentage.
            - `valve_cond`: valve condition percentage.
            - `pump_leak`: pump leakage classification target.
            - `accumulator_cond`: hydraulic accumulator pressure.
            - `stable_flag`: stability indicator for operating conditions.
            """
        )
