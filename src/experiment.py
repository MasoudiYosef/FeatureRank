"""Shared and legacy-compatible training helpers.

The public task flows live in :mod:`src.classification`, :mod:`src.regression`,
and :mod:`src.clustering`.  This module remains as a stable compatibility layer
for helpers still reused by those flows and older scripts.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    AUTOENCODER_EARLY_STOPPING_MIN_DELTA as DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
    AUTOENCODER_EARLY_STOPPING_PATIENCE as DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE,
    AUTOENCODER_EPOCHS,
    BATCH_SIZE,
    CLASSIFIER_CLASS_WEIGHT,
    CLASSIFIER_EARLY_STOPPING_MIN_DELTA as DEFAULT_EARLY_STOPPING_MIN_DELTA,
    CLASSIFIER_EARLY_STOPPING_MONITOR as DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR,
    CLASSIFIER_EARLY_STOPPING_PATIENCE as DEFAULT_EARLY_STOPPING_PATIENCE,
    CLASSIFIER_EPOCHS as DEFAULT_CLASSIFIER_EPOCHS,
    CLASSIFIER_MODEL as DEFAULT_CLASSIFIER_MODEL,
    CLASSIFIER_SAMPLING,
    CLASSIFIER_VALIDATION_SPLIT,
    THRESHOLD,
)
from src.models import build_sigmoid_autoencoder, build_latent_classifier
from src.utils import ensure_dir


def save_training_history(
    history: tf.keras.callbacks.History,
    output_dir: Path,
    file_prefix: str,
    plot_metrics: tuple[str, ...],
) -> None:
    """Save training metrics as CSV and one plot per requested metric."""
    ensure_dir(output_dir)
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))

    csv_path = output_dir / f"{file_prefix}_history.csv"
    history_df.to_csv(csv_path, index=False)
    print(f"[OK] Training history CSV: {csv_path}")

    for metric in plot_metrics:
        if metric not in history_df.columns:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(history_df["epoch"], history_df[metric], label=metric)
        val_metric = f"val_{metric}"
        if val_metric in history_df.columns:
            plt.plot(history_df["epoch"], history_df[val_metric], label=val_metric)
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{file_prefix} {metric}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = output_dir / f"{file_prefix}_{metric}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[OK] Training plot: {plot_path}")


def train_autoencoder_model(
    X_train_sub: np.ndarray,
    X_val: np.ndarray,
    X_eval: np.ndarray,
    encoding_dim: int,
    autoencoder_epochs: int | None = None,
    history_output_dir: Path | None = None,
    history_prefix: str | None = None,
    shuffle_training: bool = True,
) -> tuple[float, tf.keras.Model, tf.keras.Model]:
    """Train the shared sigmoid autoencoder using project-wide defaults."""
    if autoencoder_epochs is None:
        autoencoder_epochs = AUTOENCODER_EPOCHS
    early_stopping_patience = DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE or None
    early_stopping_min_delta = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA
    autoencoder, encoder = build_sigmoid_autoencoder(
        input_dim=X_train_sub.shape[1],
        encoding_dim=encoding_dim,
        activation="sigmoid",
    )

    # Ensure consistent dtypes
    X_train_sub = X_train_sub.astype(np.float32)
    X_val = X_val.astype(np.float32)

    callbacks: list[tf.keras.callbacks.Callback] = []
    if early_stopping_patience is not None and early_stopping_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                restore_best_weights=True,
                mode="min",
                verbose=1,
            )
        )

    autoencoder_history = autoencoder.fit(
        X_train_sub,
        X_train_sub,
        validation_data=(X_val, X_val),
        epochs=autoencoder_epochs,
        batch_size=BATCH_SIZE,
        shuffle=shuffle_training,
        callbacks=callbacks,
        verbose=1,
    )
    if history_output_dir is not None and history_prefix is not None:
        save_training_history(
            history=autoencoder_history,
            output_dir=history_output_dir,
            file_prefix=f"{history_prefix}_autoencoder",
            plot_metrics=("loss",),
        )

    X_eval = X_eval.astype(np.float32)
    eval_mse = float(autoencoder.evaluate(X_eval, X_eval, verbose=0))
    return eval_mse, autoencoder, encoder


def train_and_evaluate_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    encoding_dim: int,
    random_state: int | None,
    history_output_dir: Path | None = None,
    history_prefix: str | None = None,
    return_train_predictions: bool = False,
) -> tuple[float, float, tf.keras.Model, tf.keras.Model, np.ndarray, np.ndarray, np.ndarray]:
    """Train the autoencoder, classify its latent features, and return predictions."""
    # These helpers belong to the classification flow; import lazily to keep
    # this legacy-compatible shared module free of an import cycle.
    from src.classification import (
        build_sklearn_classifier,
        compute_binary_class_weight,
        predict_binary_scores,
        undersample_binary_training_data,
    )

    classifier_model = DEFAULT_CLASSIFIER_MODEL
    classifier_epochs = DEFAULT_CLASSIFIER_EPOCHS
    classifier_early_stopping_patience = DEFAULT_EARLY_STOPPING_PATIENCE or None
    classifier_early_stopping_monitor = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR
    classifier_early_stopping_min_delta = DEFAULT_EARLY_STOPPING_MIN_DELTA
    classifier_class_weight = CLASSIFIER_CLASS_WEIGHT
    classifier_sampling = CLASSIFIER_SAMPLING
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=CLASSIFIER_VALIDATION_SPLIT,
        random_state=random_state,
        shuffle=True,
        stratify=y_train,
    )
    X_train_sub, y_train_sub = undersample_binary_training_data(
        X_train=X_train_sub,
        y_train=y_train_sub,
        random_state=random_state,
        mode=classifier_sampling,
    )

    test_mse, autoencoder, encoder = train_autoencoder_model(
        X_train_sub=X_train_sub,
        X_val=X_val,
        X_eval=X_test,
        encoding_dim=encoding_dim,
        history_output_dir=history_output_dir,
        history_prefix=history_prefix,
        shuffle_training=random_state is None,
    )

    X_train_encoded = encoder.predict(X_train_sub, verbose=0).astype(np.float32)
    X_val_encoded = encoder.predict(X_val, verbose=0).astype(np.float32)
    X_test_encoded = encoder.predict(X_test, verbose=0).astype(np.float32)

    # Get encoded dimension for validation
    encoder_output_dim = X_train_encoded.shape[1]

    # Verify input/output shapes match model expectations
    if X_train_encoded.shape[1] != encoder_output_dim:
        raise ValueError(
            f"Encoder output dim {X_train_encoded.shape[1]} != expected dim {encoder_output_dim}"
        )

    classifier_model = str(classifier_model).strip().lower()
    class_weight = compute_binary_class_weight(y_train_sub, classifier_class_weight)
    if classifier_model == "neural":
        classifier = build_latent_classifier(
            input_dim=encoder_output_dim,
        )
        y_train_fit = y_train_sub.astype(np.float32)
        y_val_fit = y_val.astype(np.float32)

        callbacks: list[tf.keras.callbacks.Callback] = []
        if (
            classifier_early_stopping_patience is not None
            and classifier_early_stopping_patience > 0
        ):
            if classifier_early_stopping_monitor not in {"val_loss", "val_accuracy"}:
                raise ValueError(
                    "classifier early stopping monitor 'val_loss' veya 'val_accuracy' olmali."
                )
            monitor_mode = "max" if classifier_early_stopping_monitor == "val_accuracy" else "min"
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=classifier_early_stopping_monitor,
                    patience=classifier_early_stopping_patience,
                    min_delta=classifier_early_stopping_min_delta,
                    restore_best_weights=True,
                    mode=monitor_mode,
                    verbose=1,
                )
            )

        classifier_history = classifier.fit(
            X_train_encoded,
            y_train_fit,
            epochs=classifier_epochs,
            batch_size=BATCH_SIZE,
            validation_data=(X_val_encoded, y_val_fit),
            class_weight=class_weight,
            shuffle=random_state is None,
            callbacks=callbacks,
            verbose=1,
        )
        if history_output_dir is not None and history_prefix is not None:
            save_training_history(
                history=classifier_history,
                output_dir=history_output_dir,
                file_prefix=f"{history_prefix}_classifier",
                plot_metrics=("accuracy", "loss"),
            )
        y_pred_prob = classifier.predict(X_test_encoded, verbose=0)
        if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 1:
            y_pred_prob = y_pred_prob.ravel()
        print("classifier output shape:", classifier.output_shape)
    else:
        classifier = build_sklearn_classifier(
            classifier_model=classifier_model,
            random_state=random_state,
            class_weight=class_weight,
        )
        classifier.fit(X_train_encoded, y_train_sub.astype(int))
        y_pred_prob = predict_binary_scores(classifier, X_test_encoded)
        print(f"classifier model: {classifier_model}")

    # Handle both single-output (sigmoid) and multi-output predictions
    y_pred = (y_pred_prob > THRESHOLD).astype(int).ravel()

    if len(y_pred) != len(y_test):
        raise ValueError(f"Prediction length {len(y_pred)} != y_test length {len(y_test)}")

    test_accuracy = float(accuracy_score(y_test.astype(int), y_pred))
    if return_train_predictions:
        X_train_eval_encoded = encoder.predict(X_train.astype(np.float32), verbose=0).astype(
            np.float32
        )
        if classifier_model == "neural":
            y_train_score = classifier.predict(X_train_eval_encoded, verbose=0)
            if y_train_score.ndim == 2 and y_train_score.shape[1] == 1:
                y_train_score = y_train_score.ravel()
        else:
            y_train_score = predict_binary_scores(classifier, X_train_eval_encoded)
        y_train_pred = (np.asarray(y_train_score).ravel() > THRESHOLD).astype(int)
        return (
            test_mse,
            test_accuracy,
            autoencoder,
            encoder,
            X_train_sub,
            y_pred,
            y_pred_prob.ravel(),
            y_train_pred.ravel(),
            np.asarray(y_train_score, dtype=np.float32).ravel(),
        )
    return test_mse, test_accuracy, autoencoder, encoder, X_train_sub, y_pred, y_pred_prob.ravel()


def train_and_evaluate_direct_classifier(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int | None,
    history_output_dir: Path | None = None,
    history_prefix: str | None = None,
    **_fixed_options,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train the configured classifier directly on already reduced features."""
    from src.classification import (
        build_sklearn_classifier,
        compute_binary_class_weight,
        predict_binary_scores,
        undersample_binary_training_data,
    )

    classifier_model = DEFAULT_CLASSIFIER_MODEL
    classifier_epochs = DEFAULT_CLASSIFIER_EPOCHS
    classifier_early_stopping_patience = DEFAULT_EARLY_STOPPING_PATIENCE or None
    classifier_early_stopping_monitor = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR
    classifier_early_stopping_min_delta = DEFAULT_EARLY_STOPPING_MIN_DELTA
    classifier_class_weight = CLASSIFIER_CLASS_WEIGHT
    classifier_sampling = CLASSIFIER_SAMPLING
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=CLASSIFIER_VALIDATION_SPLIT,
        random_state=random_state,
        shuffle=True,
        stratify=y_train,
    )
    X_train_sub, y_train_sub = undersample_binary_training_data(
        X_train=X_train_sub,
        y_train=y_train_sub,
        random_state=random_state,
        mode=classifier_sampling,
    )

    classifier_model = str(classifier_model).strip().lower()
    class_weight = compute_binary_class_weight(y_train_sub, classifier_class_weight)
    if classifier_model == "neural":
        classifier = build_latent_classifier(
            input_dim=X_train.shape[1],
        )
        callbacks: list[tf.keras.callbacks.Callback] = []
        if (
            classifier_early_stopping_patience is not None
            and classifier_early_stopping_patience > 0
        ):
            if classifier_early_stopping_monitor not in {"val_loss", "val_accuracy"}:
                raise ValueError(
                    "classifier early stopping monitor 'val_loss' veya 'val_accuracy' olmali."
                )
            monitor_mode = "max" if classifier_early_stopping_monitor == "val_accuracy" else "min"
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=classifier_early_stopping_monitor,
                    patience=classifier_early_stopping_patience,
                    min_delta=classifier_early_stopping_min_delta,
                    restore_best_weights=True,
                    mode=monitor_mode,
                    verbose=1,
                )
            )
        classifier_history = classifier.fit(
            X_train_sub.astype(np.float32),
            y_train_sub.astype(np.float32),
            epochs=classifier_epochs,
            batch_size=BATCH_SIZE,
            validation_data=(X_val.astype(np.float32), y_val.astype(np.float32)),
            class_weight=class_weight,
            shuffle=random_state is None,
            callbacks=callbacks,
            verbose=1,
        )
        if history_output_dir is not None and history_prefix is not None:
            save_training_history(
                history=classifier_history,
                output_dir=history_output_dir,
                file_prefix=f"{history_prefix}_direct_classifier",
                plot_metrics=("accuracy", "loss"),
            )
        y_score = classifier.predict(X_test.astype(np.float32), verbose=0)
        if y_score.ndim == 2 and y_score.shape[1] == 1:
            y_score = y_score.ravel()
        print("direct classifier output shape:", classifier.output_shape)
    else:
        classifier = build_sklearn_classifier(
            classifier_model=classifier_model,
            random_state=random_state,
            class_weight=class_weight,
        )
        classifier.fit(X_train_sub, y_train_sub.astype(int))
        y_score = predict_binary_scores(classifier, X_test)
        print(f"direct classifier model: {classifier_model}")

    y_score = np.asarray(y_score, dtype=np.float32).ravel()
    y_pred = (y_score > THRESHOLD).astype(int)
    if len(y_pred) != len(y_test):
        raise ValueError(f"Prediction length {len(y_pred)} != y_test length {len(y_test)}")
    test_accuracy = float(accuracy_score(y_test.astype(int), y_pred))
    return test_accuracy, y_pred, y_score


def train_and_evaluate_direct_multiclass_classifier(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int | None,
    history_output_dir: Path | None = None,
    history_prefix: str | None = None,
    **_fixed_options,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train the configured multiclass classifier on already reduced features."""
    from src.classification import build_sklearn_classifier

    classifier_model = DEFAULT_CLASSIFIER_MODEL
    classifier_epochs = DEFAULT_CLASSIFIER_EPOCHS
    classifier_early_stopping_patience = DEFAULT_EARLY_STOPPING_PATIENCE or None
    classifier_early_stopping_monitor = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR
    classifier_early_stopping_min_delta = DEFAULT_EARLY_STOPPING_MIN_DELTA
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=CLASSIFIER_VALIDATION_SPLIT,
        random_state=random_state,
        shuffle=True,
        stratify=y_train,
    )
    num_classes = int(len(np.unique(y_train)))
    classifier_model = str(classifier_model).strip().lower()
    if classifier_model == "neural":
        classifier = build_latent_classifier(
            input_dim=X_train.shape[1],
            num_classes=num_classes,
        )
        callbacks: list[tf.keras.callbacks.Callback] = []
        if (
            classifier_early_stopping_patience is not None
            and classifier_early_stopping_patience > 0
        ):
            monitor_mode = "max" if classifier_early_stopping_monitor == "val_accuracy" else "min"
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=classifier_early_stopping_monitor,
                    patience=classifier_early_stopping_patience,
                    min_delta=classifier_early_stopping_min_delta,
                    restore_best_weights=True,
                    mode=monitor_mode,
                    verbose=1,
                )
            )
        history = classifier.fit(
            X_train_sub.astype(np.float32),
            y_train_sub.astype(np.int32),
            epochs=classifier_epochs,
            batch_size=BATCH_SIZE,
            validation_data=(X_val.astype(np.float32), y_val.astype(np.int32)),
            shuffle=random_state is None,
            callbacks=callbacks,
            verbose=1,
        )
        if history_output_dir is not None and history_prefix is not None:
            save_training_history(
                history=history,
                output_dir=history_output_dir,
                file_prefix=f"{history_prefix}_direct_classifier",
                plot_metrics=("accuracy", "loss"),
            )
        y_score = np.asarray(
            classifier.predict(X_test.astype(np.float32), verbose=0), dtype=np.float32
        )
        y_pred = np.argmax(y_score, axis=1).astype(int)
    else:
        classifier = build_sklearn_classifier(
            classifier_model=classifier_model,
            random_state=random_state,
            class_weight=None,
        )
        classifier.fit(X_train_sub, y_train_sub.astype(int))
        y_pred = np.asarray(classifier.predict(X_test), dtype=int).ravel()
        if hasattr(classifier, "predict_proba"):
            y_score = np.asarray(classifier.predict_proba(X_test), dtype=np.float32)
        else:
            y_score = np.empty((len(y_pred), 0), dtype=np.float32)

    return float(accuracy_score(y_test.astype(int), y_pred)), y_pred, y_score
