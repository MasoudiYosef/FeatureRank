from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.runtime import set_reproducible


DEFAULT_DECODING_DIM = 128
DEFAULT_FACTORIZATION_RANK = 128
DEFAULT_MAX_DENSE_WEIGHTS = 5_000_000


@dataclass(frozen=True)
class DocumentAutoencoderSettings:
    """Architecture and training options for the document-aligned model."""

    encoding_dim: int
    decoding_dim: int = DEFAULT_DECODING_DIM
    encoding_activation: str = "relu"
    decoding_activation: str = "relu"
    learning_rate: float = 0.0001
    encoding_implementation: str = "auto"
    factorization_rank: int = DEFAULT_FACTORIZATION_RANK
    max_dense_weights: int = DEFAULT_MAX_DENSE_WEIGHTS
    epochs: int = 50
    batch_size: int = 8
    validation_split: float = 0.1
    verbose: int = 1


def set_seed(seed: int) -> None:
    set_reproducible(seed)
    random.seed(seed)
    np.random.seed(seed)


def parse_retained_percentages(text: str) -> list[int]:
    text = str(text).strip().lower()
    if text in {"all", "grid", "range"}:
        return list(range(10, 101, 10))

    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(float(part))
        if value < 10 or value > 100:
            raise ValueError("retained-percent degeri 10 ile 100 arasinda olmali.")
        if value not in values:
            values.append(value)

    if not values:
        raise ValueError("retained-percent bos olamaz.")
    return values


def compute_encoding_dim(original_feature_count: int, retained_percent: int) -> int:
    if retained_percent == 100:
        return int(original_feature_count)
    return max(1, int(round(original_feature_count * retained_percent / 100.0)))


class FactorizedEncoding(tf.keras.layers.Layer):
    """
    Bellek dostu TEK conceptual encoding layer.

    Matematiksel olarak:
        W = A @ B
        output = activation(X @ W + b)

    Keras modelinde bu hala tek bir "encoding_layer" katmanidir;
    arada ikinci bir aktivasyon/hidden representation uretilmez.
    Bu nedenle modelin conceptual katmanlari:
        Input -> Encoding -> Decoding -> Output
    olarak kalir.

    Strict full Dense istenirse --encoding-implementation dense kullanilabilir.
    """

    def __init__(
        self,
        units: int,
        rank: int = DEFAULT_FACTORIZATION_RANK,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = int(units)
        self.rank = int(rank)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        actual_rank = max(1, min(self.rank, input_dim, self.units))
        self.actual_rank = actual_rank

        self.left_kernel = self.add_weight(
            name="left_kernel",
            shape=(input_dim, actual_rank),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.right_kernel = self.add_weight(
            name="right_kernel",
            shape=(actual_rank, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        x = tf.linalg.matmul(inputs, self.left_kernel)
        x = tf.linalg.matmul(x, self.right_kernel)
        x = tf.nn.bias_add(x, self.bias)
        if self.activation is not None:
            x = self.activation(x)
        return x


def estimate_dense_memory_gb(
    input_dim: int,
    encoding_dim: int,
    bytes_per_parameter: int = 4,
    training_multiplier: float = 4.0,
) -> float:
    params = int(input_dim) * int(encoding_dim)
    return float(params * bytes_per_parameter * training_multiplier / (1024**3))


def choose_encoding_implementation(
    input_dim: int,
    encoding_dim: int,
    requested: str,
    max_dense_weights: int,
) -> str:
    requested = requested.strip().lower()
    if requested not in {"auto", "dense", "factorized"}:
        raise ValueError("encoding-implementation: auto, dense veya factorized olmali.")

    if requested != "auto":
        return requested

    dense_weight_count = int(input_dim) * int(encoding_dim)
    return "factorized" if dense_weight_count > int(max_dense_weights) else "dense"


def build_document_aligned_autoencoder(
    input_dim: int,
    settings: DocumentAutoencoderSettings,
):
    """
    Dokumandaki 4-layer conceptual mimari:

        Input -> Encoding -> Decoding -> Output

    Reduced representation = encoding_layer output.
    """

    if input_dim <= 0 or settings.encoding_dim <= 0 or settings.decoding_dim <= 0:
        raise ValueError("input_dim, encoding_dim ve decoding_dim pozitif olmali.")

    inputs = tf.keras.layers.Input(
        shape=(input_dim,),
        dtype="float32",
        name="input_layer",
    )

    # Tum reduction oranlari AYNI pipeline'i kullanir.
    # 0% reduction icin de encoding_dim == input_dim olur; ancak encoding
    # katmani atlanmaz ve identity ozel-durumu kullanilmaz.
    implementation = choose_encoding_implementation(
        input_dim=input_dim,
        encoding_dim=settings.encoding_dim,
        requested=settings.encoding_implementation,
        max_dense_weights=settings.max_dense_weights,
    )

    if implementation == "dense":
        encoded = tf.keras.layers.Dense(
            settings.encoding_dim,
            activation=settings.encoding_activation,
            name="encoding_layer",
        )(inputs)
    else:
        encoded = FactorizedEncoding(
            settings.encoding_dim,
            rank=settings.factorization_rank,
            activation=settings.encoding_activation,
            name="encoding_layer",
        )(inputs)

    decoded = tf.keras.layers.Dense(
        settings.decoding_dim,
        activation=settings.decoding_activation,
        name="decoding_layer",
    )(encoded)

    outputs = tf.keras.layers.Dense(
        input_dim,
        activation="linear",
        name="output_layer",
    )(decoded)

    autoencoder = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="document_aligned_feature_rank_autoencoder",
    )
    encoder = tf.keras.Model(
        inputs=inputs,
        outputs=encoded,
        name="document_aligned_encoder",
    )

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=settings.learning_rate,
            clipnorm=1.0,
        ),
        loss="mse",
        jit_compile=False,
    )

    return autoencoder, encoder, implementation


def fit_document_aligned_autoencoder(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    settings: DocumentAutoencoderSettings,
    seed: int,
):
    set_seed(seed)

    X_train_sub, X_val, _, _ = train_test_split(
        X_train_scaled,
        y_train,
        test_size=settings.validation_split,
        random_state=seed,
        shuffle=True,
        stratify=y_train,
    )

    autoencoder, encoder, implementation = build_document_aligned_autoencoder(
        input_dim=X_train_scaled.shape[1], settings=settings
    )

    history = autoencoder.fit(
        X_train_sub.astype(np.float32),
        X_train_sub.astype(np.float32),
        validation_data=(X_val.astype(np.float32), X_val.astype(np.float32)),
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        shuffle=False,
        verbose=settings.verbose,
    )

    X_test_scaled = X_test_scaled.astype(np.float32)
    reconstruction_mse = float(
        autoencoder.evaluate(
            X_test_scaled,
            X_test_scaled,
            verbose=0,
        )
    )
    reconstruction_rmse = float(np.sqrt(reconstruction_mse))

    X_train_encoded = encoder.predict(
        X_train_scaled.astype(np.float32),
        verbose=0,
    ).astype(np.float32)
    X_test_encoded = encoder.predict(
        X_test_scaled.astype(np.float32),
        verbose=0,
    ).astype(np.float32)

    if X_train_encoded.shape[1] != settings.encoding_dim:
        raise RuntimeError(
            f"Encoding output dimension mismatch: expected={settings.encoding_dim}, "
            f"actual={X_train_encoded.shape[1]}"
        )

    return {
        "autoencoder": autoencoder,
        "encoder": encoder,
        "history": history,
        "encoding_implementation": implementation,
        "reconstruction_mse": reconstruction_mse,
        "reconstruction_rmse": reconstruction_rmse,
        "X_train_encoded": X_train_encoded,
        "X_test_encoded": X_test_encoded,
    }


def save_headerless_matrix(path: Path, X: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X).to_csv(path, index=False, header=False)


def save_headerless_labels(path: Path, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(y).to_csv(path, index=False, header=False)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
