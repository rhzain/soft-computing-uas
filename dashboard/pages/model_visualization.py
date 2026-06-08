from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.config import (
    FINAL_EVALUATION,
    IMAGE_ARTIFACTS,
    MODEL_ARTIFACT_PATH,
    RA_GRID_RESULTS,
)


def render() -> None:
    st.title("Visualization Model")
    st.caption(
        "Visualisasi sederhana untuk melihat konfigurasi model, hasil tuning, "
        "hasil evaluasi, dan fungsi keanggotaan fuzzy."
    )

    render_model_metrics()

    tab_architecture, tab_tuning, tab_evaluation, tab_membership, tab_artifact = st.tabs(
        [
            "Arsitektur",
            "Tuning r_a",
            "Evaluasi",
            "Membership Function",
            "Artifact",
        ]
    )

    with tab_architecture:
        render_architecture()

    with tab_tuning:
        render_tuning()

    with tab_evaluation:
        render_evaluation()

    with tab_membership:
        render_membership_functions()

    with tab_artifact:
        render_artifact()


def render_model_metrics() -> None:
    metric_columns = st.columns(5)
    metric_columns[0].metric("Model", "WANFIS")
    metric_columns[1].metric("Input Features", "15")
    metric_columns[2].metric("Fuzzy Rules", "4")
    metric_columns[3].metric("Best r_a", "0.6")
    metric_columns[4].metric("Classes", "3")


def render_architecture() -> None:
    st.subheader("Arsitektur WANFIS")

    left_column, right_column = st.columns([1.1, 1])

    with left_column:
        st.markdown(
            """
            Model menggunakan pendekatan **Wavelet-ANFIS**:

            1. Data sensor diproses dengan Discrete Wavelet Transform.
            2. Setiap sensor menghasilkan `mean_cA`, `std_cA`, dan `energy_cD`.
            3. Total input model adalah 15 fitur.
            4. Subtractive Clustering membentuk pusat rule fuzzy.
            5. ANFIS dilatih dengan PyTorch untuk klasifikasi 3 kelas.
            """
        )

    with right_column:
        architecture_table = pd.DataFrame(
            [
                {"Layer": "L1", "Fungsi": "Gaussian membership function"},
                {"Layer": "L2", "Fungsi": "Firing strength tiap rule"},
                {"Layer": "L3", "Fungsi": "Normalisasi firing strength"},
                {"Layer": "L4", "Fungsi": "Consequent Takagi-Sugeno linear"},
                {"Layer": "L5", "Fungsi": "Output logits untuk 3 kelas"},
            ]
        )
        st.dataframe(architecture_table, hide_index=True, use_container_width=True)

    st.code(
        "Sensor signals -> Wavelet features -> StandardScaler -> "
        "Subtractive Clustering -> ANFIS -> pump_leak class",
        language="text",
    )


def render_tuning() -> None:
    st.subheader("Tuning Radius Subtractive Clustering")

    st.markdown(
        "`r_a` mengontrol radius cluster. Nilai kecil cenderung menghasilkan "
        "lebih banyak rule, sedangkan nilai besar membuat rule lebih sedikit."
    )

    results = pd.DataFrame(RA_GRID_RESULTS)
    st.dataframe(results, hide_index=True, use_container_width=True)

    chart_data = results.set_index("r_a")[["Mean Accuracy", "Mean F1"]]
    st.line_chart(chart_data)

    render_image_artifact(
        title="Visualisasi Eksperimen r_a",
        artifact_key="r_a Experiment",
        caption="Grafik hasil eksperimen nilai radius cluster dari notebook.",
    )


def render_evaluation() -> None:
    st.subheader("Evaluasi Model Final")

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
            Kelas 1 memiliki akurasi paling rendah dibanding kelas lain.
            Ini masuk akal karena kebocoran lemah biasanya lebih sulit
            dibedakan dari kondisi normal atau kebocoran parah.
            """
        )

    with right_column:
        render_image_artifact(
            title="Evaluation Results",
            artifact_key="Evaluation Results",
            caption="Confusion matrix dan ringkasan evaluasi dari notebook.",
        )

    render_image_artifact(
        title="Training Curves",
        artifact_key="Training Curves",
        caption="Kurva training yang menunjukkan dinamika loss dan akurasi.",
    )


def render_membership_functions() -> None:
    st.subheader("Membership Function")

    st.markdown(
        """
        Fungsi keanggotaan Gaussian diinisialisasi dari pusat cluster hasil
        Subtractive Clustering. Setelah training, parameter `mean` dan `sigma`
        dapat bergeser mengikuti optimasi gradient-based.
        """
    )

    render_image_artifact(
        title="Membership Functions",
        artifact_key="Membership Functions",
        caption="Perbandingan membership function sebelum dan sesudah training.",
    )


def render_artifact() -> None:
    st.subheader("Model Artifact")

    if MODEL_ARTIFACT_PATH.exists():
        size_kb = MODEL_ARTIFACT_PATH.stat().st_size / 1024
        st.success("Model artifact tersedia.")

        artifact_table = pd.DataFrame(
            [
                {"Properti": "Path", "Nilai": str(MODEL_ARTIFACT_PATH)},
                {"Properti": "Ukuran", "Nilai": f"{size_kb:.2f} KB"},
                {"Properti": "Format", "Nilai": "PyTorch checkpoint (.pth)"},
            ]
        )
        st.dataframe(artifact_table, hide_index=True, use_container_width=True)

        st.markdown(
            """
            Checkpoint model menyimpan bobot ANFIS serta metadata penting seperti
            `best_ra`, jumlah input, jumlah rule, nama fitur, scaler, dan pusat
            cluster. Metadata ini diperlukan agar inference memakai preprocessing
            yang sama dengan training.
            """
        )
    else:
        st.warning("Model artifact belum ditemukan di folder `models`.")


def render_image_artifact(title: str, artifact_key: str, caption: str) -> None:
    image_path = IMAGE_ARTIFACTS[artifact_key]

    if image_path.exists():
        st.markdown(f"**{title}**")
        st.image(str(image_path), use_container_width=True)
        st.caption(caption)
    else:
        st.warning(f"File gambar belum ditemukan: `{image_path.name}`")
