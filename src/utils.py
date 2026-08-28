"""Small helpers shared by command-line and task modules."""

import json
from pathlib import Path

import numpy as np


def ensure_dir(path: Path) -> None:
    """Create a directory and its parents when they do not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    """Write a dictionary as an indented UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def compute_rmse_from_mse(mse: float | None) -> float | None:
    """Convert a mean-squared error to RMSE while preserving missing values."""
    if mse is None:
        return None
    mse_value = float(mse)
    if np.isnan(mse_value):
        return None
    return float(np.sqrt(max(mse_value, 0.0)))


def parse_hidden_units(units_text: str) -> tuple[int, ...]:
    """Parse a comma-separated hidden-layer size list."""
    parts = [p.strip() for p in units_text.split(",") if p.strip()]
    if not parts:
        raise ValueError("classifier-hidden-units bos olamaz. Ornek: 128,64")
    units = tuple(int(p) for p in parts)
    if any(u <= 0 for u in units):
        raise ValueError("classifier-hidden-units pozitif tam sayilar olmali.")
    return units


def parse_dropout_rates(dropout_text: str | None, layer_count: int) -> tuple[float, ...] | None:
    """Parse optional comma-separated dropout rates."""
    if dropout_text is None:
        return None
    text = dropout_text.strip()
    if text == "":
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    dropouts = tuple(float(p) for p in parts)
    if len(dropouts) != layer_count:
        raise ValueError("classifier-dropout-rates uzunlugu, hidden katman sayisi ile ayni olmali.")
    if any((d < 0.0 or d >= 1.0) for d in dropouts):
        raise ValueError("dropout oranlari [0.0, 1.0) araliginda olmali.")
    return dropouts
