from __future__ import annotations

from pathlib import Path


APP_TITLE = "WANFIS Hydraulic Dashboard"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

SENSOR_LIST = ["PS1", "PS2", "PS3", "TS1", "TS2"]
TARGET_COLUMN = "pump_leak"

CLASS_LABELS = {
    0: "Normal / tidak ada kebocoran",
    1: "Kebocoran lemah",
    2: "Kebocoran parah",
}

NAVIGATION_ITEMS = [
    "Overview",
    "Dataset Explorer",
    "Visualization Model",
    "Anomaly Simulation",
]

PROJECT_METRICS = {
    "Accuracy": "94.33%",
    "Weighted F1": "94.36%",
    "Best r_a": "0.6",
    "Fuzzy Rules": "4",
}

CV_RESULTS = {
    "Mean Accuracy": "96.37%",
    "Mean Weighted F1": "96.40%",
    "Mean Rules": "4.0",
}
