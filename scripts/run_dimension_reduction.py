"""Small compatibility helpers shared by the leakage-free CV script."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src.config import TARGET_COLUMN
from src.data_loader import load_data
from src.classification import train_and_evaluate_direct_classifier
from src.preprocessing import (
    encode_target,
    handle_pid_unrealistic_zeros,
    keep_numeric_features_only,
)


class FiniteLossGuard(tf.keras.callbacks.Callback):
    """Stop training immediately when a loss becomes NaN or infinite."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if any(value is not None and not np.isfinite(value) for value in logs.values()):
            self.model.stop_training = True
            raise FloatingPointError("Egitim kaybi NaN/Inf oldu.")


@dataclass(frozen=True)
class RawClassificationDataset:
    dataset_name: str
    X_raw: pd.DataFrame
    y_raw: pd.Series
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class PreparedSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]


def format_percentage(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "_")


def parse_percentages(text: str) -> list[float]:
    if str(text).strip().lower() == "all":
        return [float(value) for value in range(10, 101, 10)]
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values or any(value <= 0 or value > 100 for value in values):
        raise ValueError("percentages 0 ile 100 arasinda olmali.")
    return list(dict.fromkeys(values))


def parse_random_state(text: str) -> int | None:
    return None if str(text).strip().lower() in {"none", "null"} else int(text)


def load_raw_classification_dataset(
    dataset_name: str,
    target_column: str = TARGET_COLUMN,
    id_column: str | None = "ID",
    min_feature_count: int = 1,
) -> RawClassificationDataset:
    df = encode_target(
        load_data(dataset_name, folder="raw", target_column=target_column), target_column
    )
    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])
    X, y = df.drop(columns=[target_column]), df[target_column]
    X = keep_numeric_features_only(handle_pid_unrealistic_zeros(X))
    if X.shape[1] < min_feature_count:
        raise ValueError("Yeterli sayisal feature bulunamadi.")
    return RawClassificationDataset(dataset_name, X, y, tuple(X.columns))


def prepare_indexed_split(
    raw: RawClassificationDataset,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    **_metadata,
) -> PreparedSplit:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(raw.X_raw.iloc[train_indices]).astype(np.float32)
    X_test = scaler.transform(raw.X_raw.iloc[test_indices]).astype(np.float32)
    y = raw.y_raw.to_numpy(dtype=np.int32)
    return PreparedSplit(X_train, X_test, y[train_indices], y[test_indices], raw.feature_names)


def repeated_metric_statistics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci_95_lower": mean - margin,
        "ci_95_upper": mean + margin,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def train_classifier(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
    validation_size: float = 0.1,
    **_unused_options,
):
    """Evaluate the canonical direct classifier.

    The CV script still forwards its historical tuning options; they are
    accepted for compatibility while the production classifier reads its
    fixed settings from :mod:`src.config`.
    """
    accuracy, y_pred, y_score = train_and_evaluate_direct_classifier(
        X_train, X_test, y_train, y_test, random_state=random_state
    )
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "metric_average": "binary",
    }
    return metrics, None, len(y_train), max(1, int(len(y_train) * validation_size))
