from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.config import CLASS_LABELS, SENSOR_LIST
from dashboard.data import get_dataset_summary, load_profile, load_sensor_data


def render() -> None:
    summary = get_dataset_summary()

    st.title("Dataset Explorer")
    st.caption(
        "Eksplorasi sederhana untuk data sensor mentah dan target kondisi "
        "`pump_leak`."
    )

    render_dataset_metrics(summary)

    tab_distribution, tab_sensor, tab_profile = st.tabs(
        ["Distribusi Target", "Preview Sensor", "Profile Data"]
    )

    with tab_distribution:
        render_target_distribution(summary.class_distribution)

    with tab_sensor:
        render_sensor_preview(summary.sensor_shapes)

    with tab_profile:
        render_profile_preview()


def render_dataset_metrics(summary) -> None:
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Jumlah Siklus",
        f"{summary.cycle_count:,}".replace(",", "."),
    )
    metric_columns[1].metric("Sensor Dipakai", len(SENSOR_LIST))
    metric_columns[2].metric("Target", "pump_leak")
    metric_columns[3].metric("Jumlah Kelas", len(CLASS_LABELS))


def render_target_distribution(class_distribution: pd.DataFrame) -> None:
    st.subheader("Distribusi Target")

    left_column, right_column = st.columns([1.1, 1])

    with left_column:
        chart_data = class_distribution.set_index("class")["count"]
        st.bar_chart(chart_data)
        st.caption("Jumlah data untuk setiap kelas `pump_leak`.")

    with right_column:
        table = class_distribution.copy()
        table["label"] = table["class"].map(CLASS_LABELS)
        table = table[["class", "label", "count", "percentage"]]
        table.columns = ["Kelas", "Label", "Jumlah", "Persentase (%)"]
        st.dataframe(table, hide_index=True, use_container_width=True)

    st.markdown(
        """
        Kelas normal memiliki jumlah data paling besar, sedangkan kelas
        kebocoran lemah dan kebocoran parah lebih kecil tetapi seimbang.
        Ini alasan evaluasi model memakai weighted F1-score selain accuracy.
        """
    )


def render_sensor_preview(sensor_shapes: dict[str, tuple[int, int]]) -> None:
    st.subheader("Preview Sensor Mentah")

    selected_sensor = st.selectbox("Pilih sensor", SENSOR_LIST)
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
        "Maksimum titik grafik",
        options=[250, 500, 1000, 2000, time_steps],
        index=2,
    )
    show_table = control_columns[2].checkbox("Tampilkan nilai mentah", value=False)

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
        "Grafik ini menampilkan satu cycle operasi. Sensor tekanan memiliki "
        "6000 time steps, sedangkan sensor temperatur memiliki 60 time steps."
    )


def render_profile_preview() -> None:
    st.subheader("Profile Data")

    profile = load_profile()
    selected_columns = st.multiselect(
        "Kolom yang ditampilkan",
        options=list(profile.columns),
        default=list(profile.columns),
    )
    row_count = st.slider("Jumlah baris", min_value=5, max_value=100, value=20)

    if selected_columns:
        st.dataframe(
            profile[selected_columns].head(row_count),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.warning("Pilih minimal satu kolom untuk ditampilkan.")

    with st.expander("Keterangan kolom profile"):
        st.markdown(
            """
            - `cooler_cond`: kondisi cooler dalam persen.
            - `valve_cond`: kondisi valve dalam persen.
            - `pump_leak`: target klasifikasi kebocoran pompa.
            - `accumulator_cond`: tekanan accumulator.
            - `stable_flag`: penanda kondisi stabil atau belum stabil.
            """
        )
