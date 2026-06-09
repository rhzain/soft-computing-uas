from __future__ import annotations

from pathlib import Path


APP_TITLE = "WANFIS Hydraulic System"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_ARTIFACT_PATH = MODEL_DIR / "anfis_subtractive_best_model.pth"

IMAGE_ARTIFACTS = {
    "Training Curves": BASE_DIR / "training_curves.png",
    "Evaluation Results": BASE_DIR / "evaluation_results.png",
    "Membership Functions": BASE_DIR / "membership_functions.png",
    "r_a Experiment": BASE_DIR / "ra_experiment.png",
}

SENSOR_LIST = ["PS1", "PS2", "PS3", "TS1", "TS2"]
TARGET_COLUMN = "pump_leak"

CLASS_LABELS = {
    0: "Normal / tidak ada kebocoran",
    1: "Kebocoran lemah",
    2: "Kebocoran parah",
}

NAVIGATION_ITEMS = [
    "Anomaly Simulation",
    "Dataset Explorer",
    "Visualization Model",
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

FINAL_EVALUATION = {
    "Accuracy": "94.3311%",
    "Weighted F1-score": "94.3623%",
    "Class 0 Accuracy": "97.9508%",
    "Class 1 Accuracy": "86.8687%",
    "Class 2 Accuracy": "92.8571%",
}

RA_GRID_RESULTS = [
    {
        "r_a": 0.6,
        "Mean Accuracy": 0.963719,
        "Std Accuracy": 0.003794,
        "Mean F1": 0.963985,
        "Std F1": 0.003763,
        "Mean Rules": 4.0,
    },
    {
        "r_a": 0.4,
        "Mean Accuracy": 0.960091,
        "Std Accuracy": 0.009998,
        "Mean F1": 0.960491,
        "Std F1": 0.009705,
        "Mean Rules": 4.2,
    },
    {
        "r_a": 0.7,
        "Mean Accuracy": 0.956463,
        "Std Accuracy": 0.004397,
        "Mean F1": 0.956791,
        "Std F1": 0.004447,
        "Mean Rules": 4.0,
    },
    {
        "r_a": 0.3,
        "Mean Accuracy": 0.943764,
        "Std Accuracy": 0.018332,
        "Mean F1": 0.944252,
        "Std F1": 0.018215,
        "Mean Rules": 5.6,
    },
    {
        "r_a": 0.5,
        "Mean Accuracy": 0.941497,
        "Std Accuracy": 0.022512,
        "Mean F1": 0.941723,
        "Std F1": 0.022549,
        "Mean Rules": 4.0,
    },
]
