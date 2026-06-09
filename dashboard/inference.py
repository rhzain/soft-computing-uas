from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import streamlit as st
import torch
from torch import nn

from dashboard.config import MODEL_PACKAGE_PATH, SENSOR_LIST


class ANFIS(nn.Module):
    def __init__(self, n_inputs: int, n_rules: int, n_classes: int):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.n_classes = n_classes

        self.mf_mean = nn.Parameter(torch.randn(n_rules, n_inputs))
        self.mf_sigma = nn.Parameter(torch.ones(n_rules, n_inputs) * 0.5)
        self.consequent = nn.Linear(n_rules * (n_inputs + 1), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_exp = x.unsqueeze(1).expand(batch_size, self.n_rules, self.n_inputs)
        mean_exp = self.mf_mean.unsqueeze(0)
        sigma_exp = torch.abs(self.mf_sigma).unsqueeze(0) + 1e-6

        gauss = torch.exp(-((x_exp - mean_exp) / sigma_exp) ** 2)
        firing_strength = gauss.prod(dim=2)
        normalized_strength = firing_strength / (
            firing_strength.sum(dim=1, keepdim=True) + 1e-10
        )

        strength_exp = normalized_strength.unsqueeze(2).expand(
            batch_size,
            self.n_rules,
            self.n_inputs,
        )
        weighted_features = (strength_exp * x_exp).reshape(batch_size, -1)
        consequent_features = torch.cat([weighted_features, normalized_strength], dim=1)
        return self.consequent(consequent_features)


@dataclass(frozen=True)
class DetectionResult:
    predicted_class: int
    confidence: float
    probabilities: pd.DataFrame
    raw_features: pd.DataFrame
    scaled_features: pd.DataFrame
    logits: np.ndarray


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model package is not available: {path}")

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@st.cache_resource(show_spinner=False)
def load_wanfis_model() -> tuple[ANFIS, dict]:
    checkpoint = _load_checkpoint(MODEL_PACKAGE_PATH)
    model = ANFIS(
        n_inputs=int(checkpoint["n_inputs"]),
        n_rules=int(checkpoint["n_rules"]),
        n_classes=int(checkpoint["n_classes"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def extract_wavelet_features(sensor_signals: dict[str, np.ndarray]) -> pd.DataFrame:
    missing_sensors = [sensor for sensor in SENSOR_LIST if sensor not in sensor_signals]
    if missing_sensors:
        raise ValueError(f"Missing sensor signals: {', '.join(missing_sensors)}")

    feature_dict: dict[str, np.ndarray] = {}
    for sensor in SENSOR_LIST:
        sensor_matrix = np.asarray(sensor_signals[sensor], dtype=float).reshape(1, -1)
        c_a, c_d = pywt.dwt(sensor_matrix, wavelet="db4", axis=-1)
        feature_dict[f"{sensor}_mean_cA"] = np.mean(c_a, axis=-1)
        feature_dict[f"{sensor}_std_cA"] = np.std(c_a, axis=-1)
        feature_dict[f"{sensor}_energy_cD"] = np.sum(c_d**2, axis=-1)

    return pd.DataFrame(feature_dict)


def predict_cycle(sensor_signals: dict[str, np.ndarray]) -> DetectionResult:
    model, checkpoint = load_wanfis_model()
    feature_columns = list(checkpoint["feature_columns"])
    raw_features = extract_wavelet_features(sensor_signals)[feature_columns]

    scaler = checkpoint["scaler"]
    scaled_values = scaler.transform(raw_features.to_numpy())
    scaled_features = pd.DataFrame(scaled_values, columns=feature_columns)

    input_tensor = torch.tensor(scaled_values, dtype=torch.float32)
    with torch.no_grad():
        logits_tensor = model(input_tensor)
        probabilities_array = torch.softmax(logits_tensor, dim=1).numpy()[0]

    predicted_class = int(np.argmax(probabilities_array))
    confidence = float(probabilities_array[predicted_class])
    probabilities = _build_probability_table(checkpoint, probabilities_array)

    return DetectionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        raw_features=raw_features,
        scaled_features=scaled_features,
        logits=logits_tensor.numpy()[0],
    )

def _build_probability_table(
    checkpoint: dict,
    probabilities_array: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for class_index, probability in enumerate(probabilities_array):
        rows.append(
            {
                "Class": class_index,
                "Probability": float(probability),
            }
        )
    return pd.DataFrame(rows)
