from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.config import CLASS_LABELS, SENSOR_LIST
from dashboard.data import load_profile, load_sensor_data


ANOMALY_TYPES = [
    "Noise",
    "Spike",
    "Drift",
    "Offset",
    "Scale",
]


def render() -> None:
    st.title("Anomaly Simulation")
    st.caption(
        "Simulasi sederhana untuk melihat dampak anomali buatan pada sinyal "
        "sensor mentah."
    )

    sensor, cycle_index, max_points = render_input_controls()
    original_signal = load_sensor_data(sensor).iloc[cycle_index].to_numpy(dtype=float)
    modified_signal, anomaly_config = render_anomaly_controls(original_signal)

    render_cycle_context(cycle_index)
    render_signal_comparison(original_signal, modified_signal, max_points)
    render_impact_summary(original_signal, modified_signal, anomaly_config)
    render_notes()


def render_input_controls() -> tuple[str, int, int]:
    st.subheader("Pilih Data Sinyal")

    control_columns = st.columns([1, 1, 1])
    sensor = control_columns[0].selectbox("Sensor", SENSOR_LIST)

    sensor_data = load_sensor_data(sensor)
    cycle_index = control_columns[1].slider(
        "Cycle",
        min_value=0,
        max_value=len(sensor_data) - 1,
        value=0,
    )

    time_steps = sensor_data.shape[1]
    point_options = [250, 500, 1000, 2000, time_steps]
    point_options = sorted(set(option for option in point_options if option <= time_steps))
    max_points = control_columns[2].selectbox(
        "Maksimum titik grafik",
        options=point_options,
        index=min(2, len(point_options) - 1),
    )

    return sensor, cycle_index, int(max_points)


def render_anomaly_controls(original_signal: np.ndarray) -> tuple[np.ndarray, dict]:
    st.subheader("Konfigurasi Anomali")

    anomaly_type = st.selectbox("Jenis anomali", ANOMALY_TYPES)
    signal_std = float(np.std(original_signal)) or 1.0

    if anomaly_type == "Noise":
        intensity = st.slider("Intensitas noise (% dari standar deviasi)", 0, 100, 20)
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
        modified_signal = add_noise(original_signal, signal_std, intensity, int(seed))
        config = {
            "Jenis": anomaly_type,
            "Intensitas": f"{intensity}% std",
            "Seed": seed,
        }

    elif anomaly_type == "Spike":
        col_a, col_b = st.columns(2)
        position = col_a.slider("Posisi spike (% time)", 0, 100, 50)
        magnitude = col_b.slider("Magnitudo spike (x standar deviasi)", 1.0, 10.0, 4.0)
        width = st.slider("Lebar spike (% panjang sinyal)", 1, 20, 4)
        modified_signal = add_spike(original_signal, signal_std, position, magnitude, width)
        config = {
            "Jenis": anomaly_type,
            "Posisi": f"{position}% time",
            "Magnitudo": f"{magnitude:.1f}x std",
            "Lebar": f"{width}% panjang sinyal",
        }

    elif anomaly_type == "Drift":
        col_a, col_b = st.columns(2)
        direction = col_a.selectbox("Arah drift", ["Naik", "Turun"])
        magnitude = col_b.slider("Magnitudo akhir (x standar deviasi)", 0.1, 5.0, 1.5)
        modified_signal = add_drift(original_signal, signal_std, direction, magnitude)
        config = {
            "Jenis": anomaly_type,
            "Arah": direction,
            "Magnitudo akhir": f"{magnitude:.1f}x std",
        }

    elif anomaly_type == "Offset":
        magnitude = st.slider("Offset (x standar deviasi)", -5.0, 5.0, 1.0)
        modified_signal = add_offset(original_signal, signal_std, magnitude)
        config = {
            "Jenis": anomaly_type,
            "Offset": f"{magnitude:.1f}x std",
        }

    else:
        factor = st.slider("Faktor skala", 0.1, 3.0, 1.2)
        modified_signal = scale_signal(original_signal, factor)
        config = {
            "Jenis": anomaly_type,
            "Faktor skala": f"{factor:.2f}x",
        }

    return modified_signal, config


def add_noise(
    signal: np.ndarray,
    signal_std: float,
    intensity_percent: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise_scale = signal_std * (intensity_percent / 100)
    return signal + rng.normal(loc=0.0, scale=noise_scale, size=len(signal))


def add_spike(
    signal: np.ndarray,
    signal_std: float,
    position_percent: int,
    magnitude: float,
    width_percent: int,
) -> np.ndarray:
    modified = signal.copy()
    positions = np.arange(len(signal))
    center = int((position_percent / 100) * (len(signal) - 1))
    width = max(1, int((width_percent / 100) * len(signal)))
    spike = np.exp(-0.5 * ((positions - center) / width) ** 2)
    modified += spike * signal_std * magnitude
    return modified


def add_drift(
    signal: np.ndarray,
    signal_std: float,
    direction: str,
    magnitude: float,
) -> np.ndarray:
    sign = 1 if direction == "Naik" else -1
    drift = np.linspace(0, sign * signal_std * magnitude, len(signal))
    return signal + drift


def add_offset(signal: np.ndarray, signal_std: float, magnitude: float) -> np.ndarray:
    return signal + (signal_std * magnitude)


def scale_signal(signal: np.ndarray, factor: float) -> np.ndarray:
    return signal * factor


def render_cycle_context(cycle_index: int) -> None:
    profile = load_profile()
    row = profile.iloc[cycle_index]
    pump_class = int(row["pump_leak"])

    context_columns = st.columns(4)
    context_columns[0].metric("Cycle", cycle_index)
    context_columns[1].metric("Kelas Aktual", pump_class)
    context_columns[2].metric("Label", CLASS_LABELS[pump_class])
    context_columns[3].metric("Stable Flag", int(row["stable_flag"]))


def render_signal_comparison(
    original_signal: np.ndarray,
    modified_signal: np.ndarray,
    max_points: int,
) -> None:
    st.subheader("Perbandingan Sinyal")

    chart_data = build_chart_data(original_signal, modified_signal, max_points)
    st.line_chart(chart_data, x="time_step", y=["Original", "Simulated"])

    with st.expander("Lihat data grafik"):
        st.dataframe(chart_data, hide_index=True, use_container_width=True)


def build_chart_data(
    original_signal: np.ndarray,
    modified_signal: np.ndarray,
    max_points: int,
) -> pd.DataFrame:
    step = max(1, len(original_signal) // max_points)
    time_steps = np.arange(len(original_signal))[::step]

    return pd.DataFrame(
        {
            "time_step": time_steps,
            "Original": original_signal[::step],
            "Simulated": modified_signal[::step],
        }
    )


def render_impact_summary(
    original_signal: np.ndarray,
    modified_signal: np.ndarray,
    anomaly_config: dict,
) -> None:
    st.subheader("Ringkasan Dampak")

    delta = modified_signal - original_signal
    anomaly_score = calculate_anomaly_score(original_signal, delta)
    energy_delta = calculate_energy_delta(original_signal, modified_signal)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Mean |Delta|", f"{np.mean(np.abs(delta)):.4f}")
    metric_columns[1].metric("Max |Delta|", f"{np.max(np.abs(delta)):.4f}")
    metric_columns[2].metric("Energy Delta", f"{energy_delta:.2f}%")
    metric_columns[3].metric("Anomaly Score", f"{anomaly_score:.3f}")

    left_column, right_column = st.columns([1.2, 1])

    with left_column:
        st.dataframe(
            build_statistics_table(original_signal, modified_signal),
            hide_index=True,
            use_container_width=True,
        )

    with right_column:
        config_table = pd.DataFrame(
            {
                "Parameter": list(anomaly_config.keys()),
                "Nilai": list(anomaly_config.values()),
            }
        )
        st.dataframe(config_table, hide_index=True, use_container_width=True)


def calculate_anomaly_score(original_signal: np.ndarray, delta: np.ndarray) -> float:
    original_std = float(np.std(original_signal)) or 1.0
    return float(np.mean(np.abs(delta)) / original_std)


def calculate_energy_delta(
    original_signal: np.ndarray,
    modified_signal: np.ndarray,
) -> float:
    original_energy = float(np.sum(original_signal**2)) or 1.0
    modified_energy = float(np.sum(modified_signal**2))
    return ((modified_energy - original_energy) / original_energy) * 100


def build_statistics_table(
    original_signal: np.ndarray,
    modified_signal: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for label, signal in [
        ("Original", original_signal),
        ("Simulated", modified_signal),
    ]:
        rows.append(
            {
                "Sinyal": label,
                "Mean": np.mean(signal),
                "Std": np.std(signal),
                "Min": np.min(signal),
                "Max": np.max(signal),
                "Energy": np.sum(signal**2),
            }
        )

    table = pd.DataFrame(rows)
    numeric_columns = ["Mean", "Std", "Min", "Max", "Energy"]
    table[numeric_columns] = table[numeric_columns].round(4)
    return table


def render_notes() -> None:
    st.info(
        "Simulasi ini masih berfokus pada perubahan sinyal sensor. Prediksi "
        "ulang menggunakan model WANFIS bisa ditambahkan setelah pipeline "
        "inference model disiapkan."
    )

    with st.expander("Cara membaca simulasi"):
        st.markdown(
            """
            - **Noise** menambahkan gangguan acak pada seluruh sinyal.
            - **Spike** menambahkan lonjakan lokal pada titik waktu tertentu.
            - **Drift** membuat sinyal naik atau turun perlahan sepanjang cycle.
            - **Offset** menggeser seluruh sinyal dengan nilai konstan.
            - **Scale** mengalikan seluruh sinyal dengan faktor tertentu.

            `Anomaly Score` dihitung sederhana dari rata-rata perubahan absolut
            dibanding standar deviasi sinyal asli. Nilai ini bukan skor model,
            tetapi indikator cepat seberapa besar perubahan buatan terhadap sinyal.
            """
        )
