from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from dashboard.config import DATA_DIR, SENSOR_LIST


PROFILE_COLUMNS = [
    "cooler_cond",
    "valve_cond",
    "pump_leak",
    "accumulator_cond",
    "stable_flag",
]


@dataclass(frozen=True)
class DatasetSummary:
    cycle_count: int
    sensor_shapes: dict[str, tuple[int, int]]
    class_distribution: pd.DataFrame


@st.cache_data(show_spinner=False)
def load_profile() -> pd.DataFrame:
    profile_path = DATA_DIR / "profile.txt"
    profile = pd.read_csv(profile_path, sep="\t", header=None)
    profile.columns = PROFILE_COLUMNS
    return profile


@st.cache_data(show_spinner=False)
def load_sensor_shapes() -> dict[str, tuple[int, int]]:
    shapes: dict[str, tuple[int, int]] = {}

    for sensor in SENSOR_LIST:
        sensor_path = DATA_DIR / f"{sensor}.txt"
        sensor_data = pd.read_csv(sensor_path, sep="\t", header=None)
        shapes[sensor] = sensor_data.shape

    return shapes


@st.cache_data(show_spinner=False)
def load_sensor_data(sensor: str) -> pd.DataFrame:
    if sensor not in SENSOR_LIST:
        raise ValueError(f"Unknown sensor: {sensor}")

    sensor_path = DATA_DIR / f"{sensor}.txt"
    return pd.read_csv(sensor_path, sep="\t", header=None)


@st.cache_data(show_spinner=False)
def get_dataset_summary() -> DatasetSummary:
    profile = load_profile()
    sensor_shapes = load_sensor_shapes()

    class_distribution = (
        profile["pump_leak"]
        .value_counts()
        .sort_index()
        .rename_axis("class")
        .reset_index(name="count")
    )
    class_distribution["percentage"] = (
        class_distribution["count"] / len(profile) * 100
    ).round(1)

    return DatasetSummary(
        cycle_count=len(profile),
        sensor_shapes=sensor_shapes,
        class_distribution=class_distribution,
    )
