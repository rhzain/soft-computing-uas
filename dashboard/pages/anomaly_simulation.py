from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.config import SENSOR_LIST
from dashboard.data import load_profile, load_sensor_data
from dashboard.inference import DetectionResult, predict_cycle


ANOMALY_TYPES = [
    "Noise",
    "Spike",
    "Drift",
    "Offset",
    "Scale",
]

SCENARIOS = [
    {"Name": "Normal", "Class": 0, "Default Cycle": 10},
    {"Name": "Weak Leakage", "Class": 1, "Default Cycle": 251},
    {"Name": "Severe Leakage", "Class": 2, "Default Cycle": 210},
]

ALARM_STYLES = {
    0: {
        "Status": "SYSTEM HEALTHY",
        "Background": "#ecfdf5",
        "Border": "#10b981",
        "Text": "#065f46",
    },
    1: {
        "Status": "WARNING: WEAK LEAKAGE DETECTED",
        "Background": "#fffbeb",
        "Border": "#f59e0b",
        "Text": "#92400e",
    },
    2: {
        "Status": "CRITICAL: SEVERE LEAKAGE DETECTED",
        "Background": "#fef2f2",
        "Border": "#ef4444",
        "Text": "#991b1b",
    },
}

CHART_MAX_POINTS = 6000


def render() -> None:
    st.title("WANFIS Detection Simulator")
    st.caption("Hydraulic pump leakage classification from wavelet sensor features.")

    base_signals, base_context = render_base_input_section()
    if not base_signals:
        return

    simulation_controls = render_simulation_controls()
    results_table, generated_signals = generate_detection_sequence(
        base_signals,
        simulation_controls,
    )
    if results_table.empty:
        return

    final_result = generated_signals[-1]["result"]
    render_detection_outputs(
        final_result,
        results_table,
        base_context,
        base_signals,
        generated_signals[-1]["signals"],
        simulation_controls,
    )


def render_base_input_section() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    st.subheader("Input Data")

    input_source = st.radio(
        "Input source",
        options=["Dataset cycle", "Upload CSV"],
        horizontal=True,
    )

    if input_source == "Dataset cycle":
        return render_dataset_cycle_input()
    return render_upload_input()


def render_dataset_cycle_input() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    input_columns = st.columns([1, 1])
    scenario_name = input_columns[0].selectbox(
        "Scenario",
        options=[scenario["Name"] for scenario in SCENARIOS],
    )
    scenario = next(item for item in SCENARIOS if item["Name"] == scenario_name)
    cycle_options = get_cycle_options(int(scenario["Class"]))
    cycle_index = input_columns[1].selectbox(
        "Cycle",
        options=cycle_options,
        index=get_default_cycle_index(scenario, cycle_options),
        format_func=lambda value: f"Cycle {value}",
    )

    profile = load_profile()
    actual_class = int(profile.iloc[int(cycle_index)]["pump_leak"])
    context = {
        "Source": "Dataset cycle",
        "Cycle": int(cycle_index),
        "Actual Class": actual_class,
    }
    return load_cycle_signals(int(cycle_index)), context


def render_upload_input() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    template_csv = build_upload_template_csv()
    st.download_button(
        "Download CSV Template",
        data=template_csv,
        file_name="wanfis_cycle_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload one raw cycle CSV",
        type=["csv"],
    )
    if uploaded_file is None:
        st.info("Upload a CSV with columns PS1, PS2, PS3, TS1, and TS2.")
        return {}, {}

    try:
        sensor_signals = parse_uploaded_cycle(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        return {}, {}

    context = {
        "Source": "Uploaded CSV",
        "Cycle": "-",
        "Actual Class": "-",
    }
    return sensor_signals, context


def render_simulation_controls() -> dict[str, object]:
    st.subheader("Synthetic What-if Controls")

    control_columns = st.columns([1, 1])
    generated_cycles = control_columns[0].slider("Generated cycles", 1, 80, 1)
    apply_anomaly = control_columns[1].checkbox("Apply synthetic anomaly", value=False)

    anomaly_scope = "Selected sensor"
    anomaly_controls: dict[str, object] = {}
    anomaly_config: dict[str, object] = {"Mode": "Original baseline"}

    if apply_anomaly:
        anomaly_scope = st.radio(
            "Perturbation target",
            options=["Selected sensor", "All sensors"],
            horizontal=True,
        )
        target_sensor = SENSOR_LIST[0]
        if anomaly_scope == "Selected sensor":
            target_sensor = st.selectbox("Perturbation sensor", SENSOR_LIST)
        anomaly_controls, anomaly_config = render_anomaly_controls()
    else:
        target_sensor = SENSOR_LIST[0]

    return {
        "generated_cycles": int(generated_cycles),
        "apply_anomaly": apply_anomaly,
        "anomaly_scope": anomaly_scope,
        "target_sensor": target_sensor,
        "anomaly_controls": anomaly_controls,
        "anomaly_config": anomaly_config,
    }


def render_anomaly_controls() -> tuple[dict[str, object], dict[str, object]]:
    control_columns = st.columns([1, 1, 1])
    anomaly_type = control_columns[0].selectbox("Anomaly type", ANOMALY_TYPES)
    controls: dict[str, object] = {"Type": anomaly_type}
    config: dict[str, object] = {"Mode": "Synthetic perturbation", "Type": anomaly_type}

    if anomaly_type == "Noise":
        intensity = control_columns[1].slider(
            "Final noise intensity (% std)",
            min_value=0,
            max_value=100,
            value=20,
        )
        seed = control_columns[2].number_input(
            "Random seed",
            min_value=0,
            max_value=9999,
            value=42,
        )
        controls.update({"Intensity": intensity, "Seed": int(seed)})
        config.update({"Final intensity": f"{intensity}% std", "Seed": int(seed)})

    elif anomaly_type == "Spike":
        position = control_columns[1].slider("Spike position (% time)", 0, 100, 50)
        magnitude = control_columns[2].slider("Final spike magnitude (x std)", 1.0, 10.0, 4.0)
        width = st.slider("Spike width (% length)", 1, 20, 4)
        controls.update(
            {
                "Position": position,
                "Magnitude": magnitude,
                "Width": width,
            }
        )
        config.update(
            {
                "Position": f"{position}% time",
                "Final magnitude": f"{magnitude:.1f}x std",
                "Width": f"{width}% length",
            }
        )

    elif anomaly_type == "Drift":
        direction = control_columns[1].selectbox("Drift direction", ["Increase", "Decrease"])
        magnitude = control_columns[2].slider("Final magnitude (x std)", 0.1, 5.0, 1.5)
        controls.update({"Direction": direction, "Magnitude": magnitude})
        config.update(
            {
                "Direction": direction,
                "Final magnitude": f"{magnitude:.1f}x std",
            }
        )

    elif anomaly_type == "Offset":
        magnitude = control_columns[1].slider("Final offset (x std)", -5.0, 5.0, -1.0)
        controls.update({"Magnitude": magnitude})
        config.update({"Final offset": f"{magnitude:.1f}x std"})

    else:
        factor = control_columns[1].slider("Final scale factor", 0.1, 3.0, 1.2)
        controls.update({"Factor": factor})
        config.update({"Final scale factor": f"{factor:.2f}x"})

    return controls, config


def generate_detection_sequence(
    base_signals: dict[str, np.ndarray],
    controls: dict[str, object],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    generated_cycles = int(controls["generated_cycles"])
    rows = []
    generated_signals = []

    progress = None
    if generated_cycles > 1:
        progress = st.progress(0, text="Running DWT and WANFIS inference...")

    for cycle_offset in range(generated_cycles):
        severity = resolve_cycle_severity(cycle_offset, generated_cycles, controls)
        model_input_signals = build_model_input_signals(base_signals, controls, severity)
        result = run_detection_safely(model_input_signals)
        if result is None:
            if progress is not None:
                progress.empty()
            return pd.DataFrame(), []

        rows.append(build_result_row(cycle_offset, severity, result))
        generated_signals.append({"signals": model_input_signals, "result": result})

        if progress is not None:
            progress.progress(
                (cycle_offset + 1) / generated_cycles,
                text=f"Running DWT and WANFIS inference... {cycle_offset + 1}/{generated_cycles}",
            )

    if progress is not None:
        progress.empty()

    return pd.DataFrame(rows), generated_signals


def resolve_cycle_severity(
    cycle_offset: int,
    generated_cycles: int,
    controls: dict[str, object],
) -> float:
    if not controls["apply_anomaly"]:
        return 0.0
    if generated_cycles == 1:
        return 1.0
    return cycle_offset / (generated_cycles - 1)


def build_model_input_signals(
    base_signals: dict[str, np.ndarray],
    controls: dict[str, object],
    severity: float,
) -> dict[str, np.ndarray]:
    model_input_signals = {
        sensor: signal.copy()
        for sensor, signal in base_signals.items()
    }
    if not controls["apply_anomaly"]:
        return model_input_signals

    target_sensors = (
        SENSOR_LIST
        if controls["anomaly_scope"] == "All sensors"
        else [str(controls["target_sensor"])]
    )

    for sensor_position, sensor in enumerate(target_sensors):
        model_input_signals[sensor] = apply_anomaly(
            base_signals[sensor],
            controls["anomaly_controls"],
            severity,
            sensor_position,
        )

    return model_input_signals


def apply_anomaly(
    signal: np.ndarray,
    controls: dict[str, object],
    severity: float,
    sensor_position: int,
) -> np.ndarray:
    signal_std = float(np.std(signal)) or 1.0
    anomaly_type = str(controls["Type"])

    if anomaly_type == "Noise":
        return add_noise(
            signal,
            signal_std,
            int(float(controls["Intensity"]) * severity),
            int(controls["Seed"]) + sensor_position,
        )
    if anomaly_type == "Spike":
        return add_spike(
            signal,
            signal_std,
            int(controls["Position"]),
            float(controls["Magnitude"]) * severity,
            int(controls["Width"]),
        )
    if anomaly_type == "Drift":
        return add_drift(
            signal,
            signal_std,
            str(controls["Direction"]),
            float(controls["Magnitude"]) * severity,
        )
    if anomaly_type == "Offset":
        return add_offset(signal, signal_std, float(controls["Magnitude"]) * severity)
    return scale_signal(signal, 1.0 + ((float(controls["Factor"]) - 1.0) * severity))


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
    sign = 1 if direction == "Increase" else -1
    drift = np.linspace(0, sign * signal_std * magnitude, len(signal))
    return signal + drift


def add_offset(signal: np.ndarray, signal_std: float, magnitude: float) -> np.ndarray:
    return signal + (signal_std * magnitude)


def scale_signal(signal: np.ndarray, factor: float) -> np.ndarray:
    return signal * factor


def run_detection_safely(
    sensor_signals: dict[str, np.ndarray],
) -> DetectionResult | None:
    try:
        return predict_cycle(sensor_signals)
    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Unable to run WANFIS inference: {error}")
    return None


def build_result_row(
    cycle_offset: int,
    severity: float,
    result: DetectionResult,
) -> dict[str, object]:
    probabilities = result.probabilities.set_index("Class")["Probability"]
    return {
        "Generated Cycle": cycle_offset + 1,
        "Severity": severity,
        "Predicted Class": result.predicted_class,
        "Status": ALARM_STYLES[result.predicted_class]["Status"],
        "Confidence": result.confidence,
        "P Normal": float(probabilities.loc[0]),
        "P Weak Leakage": float(probabilities.loc[1]),
        "P Severe Leakage": float(probabilities.loc[2]),
    }


def render_detection_outputs(
    final_result: DetectionResult,
    results_table: pd.DataFrame,
    base_context: dict[str, object],
    base_signals: dict[str, np.ndarray],
    final_signals: dict[str, np.ndarray],
    controls: dict[str, object],
) -> None:
    render_alarm_panel(final_result)
    render_input_context(base_context, final_result)

    if len(results_table) > 1:
        render_multi_cycle_summary(results_table)
        render_probability_trend(results_table)
    else:
        render_probability_chart(final_result)

    render_signal_section(base_signals, final_signals)
    render_feature_tables(final_result, controls)
    render_results_table(results_table)


def render_alarm_panel(result: DetectionResult) -> None:
    st.subheader("Detection Result")
    style = ALARM_STYLES.get(result.predicted_class, ALARM_STYLES[0])
    left_column, right_column = st.columns([1.4, 1])

    with left_column:
        st.markdown(
            f"""
            <div style="
                padding: 1rem;
                border: 2px solid {style['Border']};
                border-radius: 8px;
                background: {style['Background']};
                color: {style['Text']};
            ">
                <div style="font-size: 0.85rem; font-weight: 700;">
                    PREDICTED CLASS {result.predicted_class}
                </div>
                <div style="font-size: 1.45rem; font-weight: 800; margin-top: 0.25rem;">
                    {style['Status']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_column:
        st.metric("Confidence", f"{result.confidence * 100:.2f}%")
        st.metric("Softmax Winner", f"Class {result.predicted_class}")


def render_input_context(
    context: dict[str, object],
    result: DetectionResult,
) -> None:
    actual_class = context["Actual Class"]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Source", str(context["Source"]))
    metric_columns[1].metric("Cycle", str(context["Cycle"]))
    metric_columns[2].metric("Actual Class", str(actual_class))
    metric_columns[3].metric(
        "Prediction Match",
        "-"
        if actual_class == "-"
        else "Yes" if int(actual_class) == result.predicted_class else "No",
    )


def render_probability_chart(result: DetectionResult) -> None:
    st.subheader("Class Probability")
    chart_data = result.probabilities.copy()
    chart_data["Probability (%)"] = chart_data["Probability"] * 100
    chart_data["Class Label"] = chart_data["Class"].map(lambda value: f"Class {value}")
    st.bar_chart(chart_data, x="Class Label", y="Probability (%)")

    with st.expander("View probability values"):
        probability_table = chart_data[
            ["Class", "Probability", "Probability (%)"]
        ].copy()
        probability_table[["Probability", "Probability (%)"]] = probability_table[
            ["Probability", "Probability (%)"]
        ].round(4)
        st.dataframe(probability_table, hide_index=True, use_container_width=True)


def render_multi_cycle_summary(results_table: pd.DataFrame) -> None:
    st.subheader("Anomaly Status Over Generated Cycles")
    first_warning = results_table[results_table["Predicted Class"] > 0]
    metric_columns = st.columns(4)
    metric_columns[0].metric("Generated Cycles", len(results_table))
    metric_columns[1].metric("Final Class", int(results_table.iloc[-1]["Predicted Class"]))
    metric_columns[2].metric(
        "Final Confidence",
        f"{results_table.iloc[-1]['Confidence'] * 100:.2f}%",
    )
    metric_columns[3].metric(
        "First Non-normal",
        "-"
        if first_warning.empty
        else int(first_warning.iloc[0]["Generated Cycle"]),
    )
    st.line_chart(results_table, x="Generated Cycle", y="Predicted Class")


def render_probability_trend(results_table: pd.DataFrame) -> None:
    st.subheader("Softmax Probability Trend")
    st.line_chart(
        results_table,
        x="Generated Cycle",
        y=["P Normal", "P Weak Leakage", "P Severe Leakage"],
    )


def render_signal_section(
    base_signals: dict[str, np.ndarray],
    final_signals: dict[str, np.ndarray],
) -> None:
    st.subheader("Signal Evidence")
    signal_columns = st.columns([1, 3])
    sensor = signal_columns[0].selectbox(
        "Signal view",
        SENSOR_LIST,
        key="signal_view_sensor",
    )
    chart_data = build_chart_data(
        base_signals[sensor],
        final_signals[sensor],
        CHART_MAX_POINTS,
    )
    signal_columns[1].line_chart(chart_data, x="time_step", y=["Original", "Model Input"])

    with st.expander("View signal data"):
        st.dataframe(chart_data, hide_index=True, use_container_width=True)


def render_feature_tables(
    result: DetectionResult,
    controls: dict[str, object],
) -> None:
    st.subheader("Wavelet Feature Evidence")
    left_column, right_column = st.columns([1.5, 1])

    with left_column:
        feature_table = build_feature_table(result)
        st.dataframe(feature_table, hide_index=True, use_container_width=True)

    with right_column:
        anomaly_config = dict(controls["anomaly_config"])
        anomaly_config["Generated cycles"] = controls["generated_cycles"]
        anomaly_config["Target"] = controls["anomaly_scope"]
        if controls["anomaly_scope"] == "Selected sensor":
            anomaly_config["Perturbation sensor"] = controls["target_sensor"]
        config_table = pd.DataFrame(
            {
                "Parameter": list(anomaly_config.keys()),
                "Value": [str(value) for value in anomaly_config.values()],
            }
        )
        st.dataframe(config_table, hide_index=True, use_container_width=True)


def render_results_table(results_table: pd.DataFrame) -> None:
    if len(results_table) == 1:
        return

    with st.expander("View generated cycle classifications", expanded=True):
        display_table = results_table.copy()
        display_table["Severity"] = display_table["Severity"].round(3)
        probability_columns = [
            "Confidence",
            "P Normal",
            "P Weak Leakage",
            "P Severe Leakage",
        ]
        display_table[probability_columns] = display_table[probability_columns].round(4)
        st.dataframe(display_table, hide_index=True, use_container_width=True)


def build_feature_table(result: DetectionResult) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Feature": result.raw_features.columns,
            "Raw value": result.raw_features.iloc[0].to_numpy(),
            "Scaled value": result.scaled_features.iloc[0].to_numpy(),
        }
    )
    table[["Raw value", "Scaled value"]] = table[
        ["Raw value", "Scaled value"]
    ].round(4)
    return table


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
            "Model Input": modified_signal[::step],
        }
    )


def build_upload_template_csv() -> str:
    template_signals = load_cycle_signals(10)
    max_length = max(len(signal) for signal in template_signals.values())
    template = pd.DataFrame(index=range(max_length))

    for sensor in SENSOR_LIST:
        values = pd.Series(template_signals[sensor], dtype=float)
        template[sensor] = values.reindex(range(max_length))

    return template.to_csv(index=False)


def parse_uploaded_cycle(uploaded_file) -> dict[str, np.ndarray]:
    uploaded_text = uploaded_file.getvalue().decode("utf-8")
    uploaded_data = pd.read_csv(StringIO(uploaded_text))
    missing_columns = [
        sensor for sensor in SENSOR_LIST if sensor not in uploaded_data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing CSV columns: {', '.join(missing_columns)}")

    sensor_signals: dict[str, np.ndarray] = {}
    for sensor in SENSOR_LIST:
        values = pd.to_numeric(uploaded_data[sensor], errors="coerce").dropna()
        if len(values) < 2:
            raise ValueError(f"Column {sensor} must contain at least two numeric values.")
        sensor_signals[sensor] = values.to_numpy(dtype=float)

    return sensor_signals


def get_cycle_options(class_id: int) -> list[int]:
    profile = load_profile()
    cycle_options = profile.index[profile["pump_leak"] == class_id].tolist()
    if not cycle_options:
        raise ValueError(f"No cycles found for class {class_id}")
    return [int(cycle_index) for cycle_index in cycle_options]


def get_default_cycle_index(scenario: dict, cycle_options: list[int]) -> int:
    default_cycle = int(scenario["Default Cycle"])
    return cycle_options.index(default_cycle) if default_cycle in cycle_options else 0


def load_cycle_signals(cycle_index: int) -> dict[str, np.ndarray]:
    return {
        sensor: load_sensor_data(sensor).iloc[cycle_index].to_numpy(dtype=float)
        for sensor in SENSOR_LIST
    }
