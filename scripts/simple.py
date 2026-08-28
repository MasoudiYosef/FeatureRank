"""Legacy, standalone autoencoder example kept for historical comparisons.

The production entry points are ``run_autoencoder.py`` and
``feature_ranking.py``.  This script intentionally retains its original
architecture and fixed training settings.
"""

from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def read_data(data_path: str | Path, label_path: str | Path) -> tuple[list[list[float]], list[int]]:
    with open(data_path, "r", encoding="utf-8") as data_file:
        lines = data_file.readlines()
    data = [[float(value) for value in line.strip().split(",")] for line in lines]
    with open(label_path, "r", encoding="utf-8") as label_file:
        lines = label_file.readlines()
    labels = [int(line.strip()) for line in lines]
    return data, labels


def filter_data(data_path: str | Path, selected_columns: Iterable[int]) -> list[list[float]]:
    with open(data_path, "r", encoding="utf-8") as data_file:
        lines = data_file.readlines()
    selected_columns = set(selected_columns)
    return [
        [
            float(value)
            for index, value in enumerate(line.strip().split(","))
            if index in selected_columns
        ]
        for line in lines
    ]


def autoencoder_accuracy(X, y) -> float:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_dim = X_train.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation="sigmoid")(input_layer)
    encoded = Dense(8, activation="sigmoid")(encoded)
    decoded = Dense(16, activation="sigmoid")(encoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    autoencoder.fit(
        X_train,
        X_train,
        epochs=50,
        batch_size=16,
        shuffle=True,
        validation_data=(X_test, X_test),
        verbose=0,
    )
    encoder = Model(inputs=input_layer, outputs=encoded)
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)
    classifier_input = Input(shape=(X_train_encoded.shape[1],))
    x = Dense(64, activation="sigmoid")(classifier_input)
    output = Dense(1, activation="sigmoid")(x)
    classifier = Model(inputs=classifier_input, outputs=output)
    classifier.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    classifier.fit(
        X_train_encoded,
        y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.1,
        verbose=0,
    )
    y_pred_prob = classifier.predict(X_test_encoded, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Backward-compatible names used by the original example.
ReadData = read_data
FilterData = filter_data
AutoEncoder = autoencoder_accuracy


def main() -> None:
    """Run the original two-pass example against pid_data.csv."""
    data_path = Path("pid_data.csv")
    label_path = Path("pid_label.csv")
    data, labels = read_data(data_path, label_path)
    X = np.asarray(data)
    y = np.asarray(labels)

    # Keep the original two training passes for historical comparisons.
    print(autoencoder_accuracy(X, y))
    print(autoencoder_accuracy(X, y))


if __name__ == "__main__":
    main()
