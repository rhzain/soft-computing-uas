from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.config import (
    CLASS_LABELS,
    CV_RESULTS,
    PROJECT_METRICS,
    SENSOR_LIST,
)
from dashboard.data import get_dataset_summary


def render() -> None:
    summary = get_dataset_summary()

    st.title("Overview")
    st.caption(
        "Ringkasan proyek WANFIS untuk klasifikasi kondisi kebocoran pompa "
        "pada hydraulic system."
    )

    render_key_metrics(summary.cycle_count)
    render_project_summary()
    render_dataset_snapshot(summary.class_distribution, summary.sensor_shapes)
    render_pipeline()
    render_model_summary()


def render_key_metrics(cycle_count: int) -> None:
    metric_columns = st.columns(5)
    metric_columns[0].metric("Jumlah Siklus", f"{cycle_count:,}".replace(",", "."))

    for column, (label, value) in zip(metric_columns[1:], PROJECT_METRICS.items()):
        column.metric(label, value)


def render_project_summary() -> None:
    st.subheader("Apa Yang Dikerjakan")
    st.write(
        "Dashboard ini merangkum proyek klasifikasi `pump_leak` menggunakan "
        "Wavelet-ANFIS. Data sensor sistem hidrolik diringkas menjadi fitur "
        "wavelet, kemudian diklasifikasikan ke 3 kondisi: normal, kebocoran "
        "lemah, dan kebocoran parah."
    )

    class_table = pd.DataFrame(
        {
            "Kelas": list(CLASS_LABELS.keys()),
            "Makna": list(CLASS_LABELS.values()),
        }
    )
    st.table(class_table)


def render_dataset_snapshot(
    class_distribution: pd.DataFrame,
    sensor_shapes: dict[str, tuple[int, int]],
) -> None:
    st.subheader("Snapshot Dataset")

    left_column, right_column = st.columns([1.15, 1])

    with left_column:
        chart_data = class_distribution.set_index("class")["count"]
        st.bar_chart(chart_data)
        st.caption("Distribusi kelas target `pump_leak`.")

    with right_column:
        distribution_table = class_distribution.copy()
        distribution_table["label"] = distribution_table["class"].map(CLASS_LABELS)
        distribution_table = distribution_table[
            ["class", "label", "count", "percentage"]
        ]
        distribution_table.columns = ["Kelas", "Label", "Jumlah", "Persentase (%)"]
        st.dataframe(distribution_table, hide_index=True, use_container_width=True)

    sensor_table = pd.DataFrame(
        {
            "Sensor": SENSOR_LIST,
            "Jumlah Siklus": [sensor_shapes[sensor][0] for sensor in SENSOR_LIST],
            "Time Steps": [sensor_shapes[sensor][1] for sensor in SENSOR_LIST],
        }
    )
    st.dataframe(sensor_table, hide_index=True, use_container_width=True)


def render_pipeline() -> None:
    st.subheader("Pipeline")

    pipeline_steps = [
        "Load data sensor PS1, PS2, PS3, TS1, TS2",
        "Ekstraksi fitur wavelet DB4: mean_cA, std_cA, energy_cD",
        "Standardisasi fitur dan label encoding target",
        "Subtractive Clustering untuk membentuk rule fuzzy awal",
        "Training ANFIS berbasis PyTorch",
        "Evaluasi dengan accuracy, weighted F1-score, dan confusion matrix",
    ]

    for index, step in enumerate(pipeline_steps, start=1):
        st.markdown(f"**{index}.** {step}")


def render_model_summary() -> None:
    st.subheader("Ringkasan Model")

    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown(
            """
            - Model: **Wavelet-ANFIS**
            - Input: **15 fitur wavelet**
            - Membership function: **Gaussian**
            - Consequent: **linear Takagi-Sugeno**
            - Optimizer: **Adam**
            """
        )

    with right_column:
        cv_table = pd.DataFrame(
            {
                "Metrik CV": list(CV_RESULTS.keys()),
                "Nilai": list(CV_RESULTS.values()),
            }
        )
        st.dataframe(cv_table, hide_index=True, use_container_width=True)

    st.info(
        "Catatan: halaman Dataset Explorer, Visualization Model, dan Anomaly "
        "Simulation sudah disiapkan di navigasi, tetapi implementasi detailnya "
        "bisa ditambahkan bertahap setelah halaman Overview ini stabil."
    )
