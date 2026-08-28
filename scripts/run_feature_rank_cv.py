"""Leakage-free repeated cross-validation for the FeatureRank workflow.

For every outer fold this script fits preprocessing and a fresh ranking
Autoencoder on training samples only. Feature scores come from the first
encoder layer. By default, the selected original features are passed directly
to the classifier so FeatureRank remains a feature-selection experiment. The
legacy second-Autoencoder path is available only as an explicit option. No
feature list is shared between folds.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import build_sigmoid_autoencoder

from run_dimension_reduction import (
    FiniteLossGuard,
    format_percentage,
    load_raw_classification_dataset,
    parse_percentages,
    parse_random_state,
    prepare_indexed_split,
    repeated_metric_statistics,
    train_classifier,
)
from src.runtime import configure_tensorflow_device, set_reproducible
from src.utils import parse_hidden_units


DEFAULT_ENCODING_DIM = 8
DEFAULT_AUTOENCODER_EPOCHS = 50
DEFAULT_CLASSIFIER_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_VALIDATION_SIZE = 0.1
DEFAULT_LEARNING_RATE = 0.001


@dataclass(frozen=True)
class TrainingSettings:
    """Model settings shared by ranking and selected-feature evaluation."""

    encoding_dim: int
    autoencoder_epochs: int
    classifier_model: str
    classifier_hidden_units: tuple[int, ...]
    classifier_learning_rate: float
    classifier_epochs: int
    validation_size: float
    batch_size: int
    autoencoder_early_stopping_patience: int
    classifier_early_stopping_patience: int
    classifier_input: str
    verbose: int


@dataclass(frozen=True)
class FeatureRankFoldResult:
    dataset: str
    percentage: float
    selected_feature_count: int
    original_feature_count: int
    run_number: int
    run_seed: int
    cv_repeat: int
    cv_fold: int
    split_identifier: str
    train_samples: int
    test_samples: int
    ranking_autoencoder_fit_samples: int
    ranking_autoencoder_validation_samples: int
    selected_autoencoder_fit_samples: int
    selected_autoencoder_validation_samples: int
    classifier_fit_samples: int
    classifier_validation_samples: int
    classifier_test_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    metric_average: str
    classifier_input: str
    ranking_autoencoder_epochs_completed: int
    selected_autoencoder_epochs_completed: int
    classifier_epochs_completed: int | None
    selected_features_path: str
    metrics_path: str
    elapsed_seconds: float


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4, ensure_ascii=False)


def split_inner_train_validation(
    X: np.ndarray,
    y: np.ndarray,
    validation_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y), dtype=np.int64)
    fit_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )
    return X[fit_indices], X[validation_indices], y[fit_indices], y[validation_indices]


def train_feature_rank_autoencoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    settings: TrainingSettings,
    random_state: int,
) -> tuple[tf.keras.Model, tf.keras.Model, np.ndarray, int, int, int]:
    """Fit one project-compatible Autoencoder without touching outer test."""
    X_fit, X_validation, _, _ = split_inner_train_validation(
        X_train,
        y_train,
        validation_size=settings.validation_size,
        random_state=random_state,
    )
    tf.keras.backend.clear_session()
    gc.collect()
    set_reproducible(random_state)
    autoencoder, encoder = build_sigmoid_autoencoder(
        input_dim=int(X_train.shape[1]),
        encoding_dim=int(settings.encoding_dim),
        activation="sigmoid",
    )
    callbacks: list[tf.keras.callbacks.Callback] = [FiniteLossGuard()]
    if settings.autoencoder_early_stopping_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=settings.autoencoder_early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
    history = autoencoder.fit(
        X_fit.astype(np.float32),
        X_fit.astype(np.float32),
        validation_data=(X_validation.astype(np.float32), X_validation.astype(np.float32)),
        epochs=settings.autoencoder_epochs,
        batch_size=settings.batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=settings.verbose,
    )
    losses = np.asarray(history.history.get("loss", []), dtype=np.float64)
    if losses.size == 0 or not np.isfinite(losses).all():
        raise FloatingPointError("FeatureRank Autoencoder loss degerleri gecersiz.")
    return autoencoder, encoder, X_fit, len(X_fit), len(X_validation), int(len(losses))


def calculate_feature_ranking(
    autoencoder: tf.keras.Model,
    X_ranking_fit: np.ndarray,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """Calculate the same contribution score as run_autoencoder without a 3-D tensor."""
    weights = np.asarray(autoencoder.get_layer("enc_dense_1").get_weights()[0], dtype=np.float64)
    if weights.shape[0] != X_ranking_fit.shape[1] or len(feature_names) != weights.shape[0]:
        raise ValueError("FeatureRank agirlik ve feature boyutlari eslesmiyor.")
    mean_absolute_input = np.mean(np.abs(X_ranking_fit.astype(np.float64)), axis=0)
    max_absolute_weight = np.max(np.abs(weights), axis=1)
    scores = mean_absolute_input * max_absolute_weight
    ranking = pd.DataFrame(
        {
            "feature_index": np.arange(len(feature_names), dtype=np.int64),
            "feature": [f"F{index + 1}" for index in range(len(feature_names))],
            "feature_name": list(feature_names),
            "max_abs_weight_score": scores,
        }
    )
    return ranking.sort_values("max_abs_weight_score", ascending=False).reset_index(drop=True)


def evaluate_selected_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_indices: np.ndarray,
    settings: TrainingSettings,
    random_state: int,
) -> tuple[dict[str, float | str], dict[str, int | None]]:
    selected_X_train = X_train[:, selected_indices].astype(np.float32)
    selected_X_test = X_test[:, selected_indices].astype(np.float32)
    classifier_input = str(settings.classifier_input).strip().lower()

    if classifier_input == "selected_features":
        set_reproducible(random_state + 1)
        metrics, classifier_epochs_completed, classifier_fit_count, classifier_validation_count = (
            train_classifier(
                X_train=selected_X_train,
                X_test=selected_X_test,
                y_train=y_train,
                y_test=y_test,
                classifier_model=settings.classifier_model,
                hidden_units=settings.classifier_hidden_units,
                learning_rate=settings.classifier_learning_rate,
                epochs=settings.classifier_epochs,
                batch_size=settings.batch_size,
                early_stopping_patience=settings.classifier_early_stopping_patience,
                validation_size=settings.validation_size,
                random_state=random_state + 1,
                verbose=settings.verbose,
            )
        )
        counts: dict[str, int | None] = {
            "selected_encoding_dim": int(selected_X_train.shape[1]),
            "selected_autoencoder_fit_samples": 0,
            "selected_autoencoder_validation_samples": 0,
            "selected_autoencoder_epochs_completed": 0,
            "classifier_fit_samples": classifier_fit_count,
            "classifier_validation_samples": classifier_validation_count,
            "classifier_epochs_completed": classifier_epochs_completed,
        }
        return metrics, counts

    if classifier_input != "second_autoencoder":
        raise ValueError(f"Desteklenmeyen classifier input modu: {classifier_input}")

    selected_encoding_dim = min(int(settings.encoding_dim), int(selected_X_train.shape[1]))

    autoencoder, encoder, _, ae_fit_count, ae_validation_count, ae_epochs_completed = (
        train_feature_rank_autoencoder(
            X_train=selected_X_train,
            y_train=y_train,
            encoding_dim=selected_encoding_dim,
            settings=replace(settings, encoding_dim=selected_encoding_dim),
            random_state=random_state,
        )
    )
    encoded_X_train = np.asarray(
        encoder.predict(selected_X_train, batch_size=settings.batch_size, verbose=0),
        dtype=np.float32,
    )
    encoded_X_test = np.asarray(
        encoder.predict(selected_X_test, batch_size=settings.batch_size, verbose=0),
        dtype=np.float32,
    )
    if encoded_X_train.shape != (len(X_train), selected_encoding_dim):
        raise RuntimeError(f"Encoded train shape hatali: {encoded_X_train.shape}")
    if encoded_X_test.shape != (len(X_test), selected_encoding_dim):
        raise RuntimeError(f"Encoded test shape hatali: {encoded_X_test.shape}")

    del autoencoder, encoder
    tf.keras.backend.clear_session()
    gc.collect()
    set_reproducible(random_state + 1)
    metrics, classifier_epochs_completed, classifier_fit_count, classifier_validation_count = (
        train_classifier(
            X_train=encoded_X_train,
            X_test=encoded_X_test,
            y_train=y_train,
            y_test=y_test,
            classifier_model=settings.classifier_model,
            hidden_units=settings.classifier_hidden_units,
            learning_rate=settings.classifier_learning_rate,
            epochs=settings.classifier_epochs,
            batch_size=settings.batch_size,
            early_stopping_patience=settings.classifier_early_stopping_patience,
            validation_size=settings.validation_size,
            random_state=random_state + 1,
            verbose=settings.verbose,
        )
    )
    counts: dict[str, int | None] = {
        "selected_encoding_dim": selected_encoding_dim,
        "selected_autoencoder_fit_samples": ae_fit_count,
        "selected_autoencoder_validation_samples": ae_validation_count,
        "selected_autoencoder_epochs_completed": ae_epochs_completed,
        "classifier_fit_samples": classifier_fit_count,
        "classifier_validation_samples": classifier_validation_count,
        "classifier_epochs_completed": classifier_epochs_completed,
    }
    return metrics, counts


def save_metric_text(results: list[FeatureRankFoldResult], output_path: Path) -> None:
    lines = [
        f"Dataset: {results[0].dataset}",
        f"Feature Percentage: {results[0].percentage:g}%",
        f"Evaluation Count: {len(results)}",
        "Evaluation Scheme: repeated_stratified_kfold",
        "",
    ]
    for label, field in (
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
    ):
        values = [float(getattr(result, field)) for result in results]
        stats = repeated_metric_statistics(values)
        lines.extend(
            [
                f"{label} Values:",
                str(values),
                f"Mean {label}: {stats['mean']:.8f}",
                f"Standard Deviation: {stats['std']:.8f}",
                f"95% Confidence Interval: [{stats['ci_95_lower']:.8f}, {stats['ci_95_upper']:.8f}]",
                f"Minimum: {stats['minimum']:.8f}",
                f"Maximum: {stats['maximum']:.8f}",
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def percentage_set_identifier(percentages: list[float]) -> str:
    if percentages == [float(value) for value in range(10, 101, 10)]:
        return "all_percentages"
    return "percentages_" + "_".join(format_percentage(value) for value in percentages)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-free repeated CV for FeatureRank")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--percentages", "--feature-percent", dest="percentages", default="all")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--id-column", default="ID")
    parser.add_argument("--encoding-dim", type=int, default=DEFAULT_ENCODING_DIM)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=2)
    parser.add_argument("--random-state", default="42")
    parser.add_argument("--autoencoder-epochs", type=int, default=DEFAULT_AUTOENCODER_EPOCHS)
    parser.add_argument("--classifier-epochs", type=int, default=DEFAULT_CLASSIFIER_EPOCHS)
    parser.add_argument(
        "--classifier-model", choices=["neural", "logistic", "random_forest"], default="neural"
    )
    parser.add_argument("--classifier-hidden-units", default="32,16")
    parser.add_argument("--classifier-learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--classifier-input",
        choices=["selected_features", "second_autoencoder"],
        default="selected_features",
        help=(
            "selected_features: FeatureRank ile secilen orijinal feature'lari classifier'a dogrudan verir; "
            "second_autoencoder: eski akistaki gibi once encoding-dim boyutuna indirir."
        ),
    )
    parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--autoencoder-early-stopping-patience", type=int, default=0)
    parser.add_argument("--classifier-early-stopping-patience", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "feature_rank_cv"
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    for name in (
        "encoding_dim",
        "cv_folds",
        "cv_repeats",
        "autoencoder_epochs",
        "classifier_epochs",
        "batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} pozitif olmali.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds en az 2 olmali.")
    if not 0.0 < args.validation_size < 1.0:
        raise ValueError("--validation-size 0 ile 1 arasinda olmali.")
    for name in ("autoencoder_early_stopping_patience", "classifier_early_stopping_patience"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} negatif olamaz.")


def build_training_settings(args: argparse.Namespace) -> TrainingSettings:
    """Read model-related CLI values once for all folds and percentages."""
    return TrainingSettings(
        encoding_dim=args.encoding_dim,
        autoencoder_epochs=args.autoencoder_epochs,
        classifier_model=args.classifier_model,
        classifier_hidden_units=parse_hidden_units(args.classifier_hidden_units),
        classifier_learning_rate=args.classifier_learning_rate,
        classifier_epochs=args.classifier_epochs,
        validation_size=args.validation_size,
        batch_size=args.batch_size,
        autoencoder_early_stopping_patience=args.autoencoder_early_stopping_patience,
        classifier_early_stopping_patience=args.classifier_early_stopping_patience,
        classifier_input=args.classifier_input,
        verbose=args.verbose,
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_arguments(args)
    percentages = parse_percentages(args.percentages)
    random_state = parse_random_state(args.random_state)
    settings = build_training_settings(args)
    configure_tensorflow_device(args.device)
    set_reproducible(random_state)

    raw = load_raw_classification_dataset(
        dataset_name=args.dataset_name,
        target_column=args.target_column,
        id_column=args.id_column,
        min_feature_count=1,
    )
    class_counts = raw.y_raw.value_counts(dropna=False)
    if int(class_counts.min()) < args.cv_folds:
        raise ValueError(
            f"{args.cv_folds}-fold icin her sinifta en az {args.cv_folds} ornek olmali: "
            f"{class_counts.to_dict()}"
        )
    output_dir = (
        args.output_dir.resolve()
        / Path(raw.dataset_name).stem
        / f"cross_validation_{args.cv_folds}x{args.cv_repeats}"
        / percentage_set_identifier(percentages)
        / f"classifier_input_{args.classifier_input}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = RepeatedStratifiedKFold(
        n_splits=args.cv_folds,
        n_repeats=args.cv_repeats,
        random_state=random_state,
    )
    total_evaluations = args.cv_folds * args.cv_repeats
    results: list[FeatureRankFoldResult] = []

    print(f"[INFO] Dataset: {raw.dataset_name}")
    print(f"[INFO] Percentages: {percentages}")
    print(f"[INFO] CV: {args.cv_folds} fold x {args.cv_repeats} repeat")
    print(
        "[INFO] Her fold icin FeatureRank agirliklari yalnizca train verisinden yeniden uretilecek."
    )

    for evaluation_index, (train_indices, test_indices) in enumerate(
        splitter.split(raw.X_raw, raw.y_raw.to_numpy())
    ):
        run_number = evaluation_index + 1
        cv_repeat = evaluation_index // args.cv_folds + 1
        cv_fold = evaluation_index % args.cv_folds + 1
        run_seed = random_state + evaluation_index
        split_identifier = f"repeat_{cv_repeat:03d}_fold_{cv_fold:02d}"
        prepared = prepare_indexed_split(
            raw=raw,
            train_indices=train_indices,
            test_indices=test_indices,
            min_feature_count=1,
            split_random_state=random_state,
            split_identifier=split_identifier,
            cv_repeat=cv_repeat,
            cv_fold=cv_fold,
        )
        print(
            f"\n[INFO] Fold {run_number}/{total_evaluations}: {split_identifier}, "
            f"train={prepared.X_train.shape}, test={prepared.X_test.shape}"
        )

        (
            ranking_autoencoder,
            _,
            ranking_X_fit,
            ranking_fit_count,
            ranking_validation_count,
            ranking_epochs,
        ) = train_feature_rank_autoencoder(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            settings=settings,
            random_state=run_seed,
        )
        ranking = calculate_feature_ranking(
            autoencoder=ranking_autoencoder,
            X_ranking_fit=ranking_X_fit,
            feature_names=prepared.feature_names,
        )
        fold_dir = output_dir / "folds" / f"repeat_{cv_repeat:03d}" / f"fold_{cv_fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ranking_path = fold_dir / "feature_ranking.csv"
        ranking.to_csv(ranking_path, index=False)
        del ranking_autoencoder
        tf.keras.backend.clear_session()
        gc.collect()

        for percentage in percentages:
            started_at = time.perf_counter()
            percentage_tag = format_percentage(percentage)
            selected_count = max(1, int(math.ceil(len(ranking) * percentage / 100.0)))
            selected = ranking.head(selected_count).copy()
            selected_path = fold_dir / f"top_{percentage_tag}_selected_features.csv"
            selected.to_csv(selected_path, index=False)
            selected_indices = selected["feature_index"].to_numpy(dtype=np.int64)

            metrics, counts = evaluate_selected_features(
                X_train=prepared.X_train,
                X_test=prepared.X_test,
                y_train=prepared.y_train,
                y_test=prepared.y_test,
                selected_indices=selected_indices,
                settings=settings,
                random_state=run_seed,
            )
            metrics_path = fold_dir / f"top_{percentage_tag}_metrics.json"
            payload = {
                "dataset": raw.dataset_name,
                "method": "FeatureRank",
                "evaluation_scheme": "repeated_stratified_kfold",
                "percentage": percentage,
                "selected_feature_count": selected_count,
                "original_feature_count": len(ranking),
                "run_number": run_number,
                "run_seed": run_seed,
                "cv_repeat": cv_repeat,
                "cv_fold": cv_fold,
                "split_identifier": split_identifier,
                "original_X_train_shape": list(prepared.X_train.shape),
                "original_X_test_shape": list(prepared.X_test.shape),
                "selected_X_train_shape": [len(prepared.X_train), selected_count],
                "selected_X_test_shape": [len(prepared.X_test), selected_count],
                "ranking_fit_scope": "cv_fold_train_only",
                "scaler_fit_scope": "cv_fold_train_only",
                "selected_autoencoder_fit_scope": (
                    "not_used"
                    if args.classifier_input == "selected_features"
                    else "selected_cv_fold_train_only"
                ),
                "classifier_fit_scope": (
                    "selected_cv_fold_train_only"
                    if args.classifier_input == "selected_features"
                    else "encoded_selected_cv_fold_train_only"
                ),
                "classifier_input": args.classifier_input,
                "test_seen_during_feature_ranking": False,
                "test_seen_during_model_fit": False,
                "ranking_path": str(ranking_path),
                "selected_features_path": str(selected_path),
                "ranking_autoencoder_fit_samples": ranking_fit_count,
                "ranking_autoencoder_validation_samples": ranking_validation_count,
                "ranking_autoencoder_epochs_completed": ranking_epochs,
                **counts,
                **metrics,
            }
            save_json(payload, metrics_path)
            result = FeatureRankFoldResult(
                dataset=raw.dataset_name,
                percentage=float(percentage),
                selected_feature_count=selected_count,
                original_feature_count=len(ranking),
                run_number=run_number,
                run_seed=run_seed,
                cv_repeat=cv_repeat,
                cv_fold=cv_fold,
                split_identifier=split_identifier,
                train_samples=len(prepared.X_train),
                test_samples=len(prepared.X_test),
                ranking_autoencoder_fit_samples=ranking_fit_count,
                ranking_autoencoder_validation_samples=ranking_validation_count,
                selected_autoencoder_fit_samples=int(counts["selected_autoencoder_fit_samples"]),
                selected_autoencoder_validation_samples=int(
                    counts["selected_autoencoder_validation_samples"]
                ),
                classifier_fit_samples=int(counts["classifier_fit_samples"]),
                classifier_validation_samples=int(counts["classifier_validation_samples"]),
                classifier_test_samples=len(prepared.X_test),
                accuracy=float(metrics["accuracy"]),
                precision=float(metrics["precision"]),
                recall=float(metrics["recall"]),
                f1=float(metrics["f1"]),
                metric_average=str(metrics["metric_average"]),
                classifier_input=args.classifier_input,
                ranking_autoencoder_epochs_completed=ranking_epochs,
                selected_autoencoder_epochs_completed=int(
                    counts["selected_autoencoder_epochs_completed"]
                ),
                classifier_epochs_completed=(
                    int(counts["classifier_epochs_completed"])
                    if counts["classifier_epochs_completed"] is not None
                    else None
                ),
                selected_features_path=str(selected_path),
                metrics_path=str(metrics_path),
                elapsed_seconds=time.perf_counter() - started_at,
            )
            results.append(result)
            print(
                f"[OK] Top %{percentage:g}: {selected_count} feature, "
                f"accuracy={result.accuracy:.6f}"
            )

    results_path = output_dir / "feature_rank_cv_results.csv"
    pd.DataFrame([asdict(result) for result in results]).to_csv(results_path, index=False)
    summary_rows: list[dict[str, Any]] = []
    text_paths: dict[str, str] = {}
    for percentage in percentages:
        percentage_results = [result for result in results if result.percentage == percentage]
        percentage_tag = format_percentage(percentage)
        text_path = output_dir / "metrics" / f"top_{percentage_tag}_cv_metrics.txt"
        save_metric_text(percentage_results, text_path)
        text_paths[percentage_tag] = str(text_path)
        row: dict[str, Any] = {
            "dataset": raw.dataset_name,
            "method": "FeatureRank",
            "evaluation_scheme": "repeated_stratified_kfold",
            "percentage": percentage,
            "original_feature_count": percentage_results[0].original_feature_count,
            "selected_feature_count": percentage_results[0].selected_feature_count,
            "evaluation_count": len(percentage_results),
            "cv_folds": args.cv_folds,
            "cv_repeats": args.cv_repeats,
        }
        for metric_name in ("accuracy", "precision", "recall", "f1"):
            values = [float(getattr(result, metric_name)) for result in percentage_results]
            stats = repeated_metric_statistics(values)
            row.update(
                {
                    f"{metric_name}_mean": stats["mean"],
                    f"{metric_name}_std": stats["std"],
                    f"{metric_name}_ci_95_lower": max(0.0, float(stats["ci_95_lower"])),
                    f"{metric_name}_ci_95_upper": min(1.0, float(stats["ci_95_upper"])),
                    f"{metric_name}_min": stats["minimum"],
                    f"{metric_name}_max": stats["maximum"],
                }
            )
        summary_rows.append(row)

    summary_path = output_dir / "feature_rank_cv_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    run_summary_path = output_dir / "feature_rank_cv_run_summary.json"
    save_json(
        {
            "dataset": raw.dataset_name,
            "method": "FeatureRank",
            "percentages": percentages,
            "cv_folds": args.cv_folds,
            "cv_repeats": args.cv_repeats,
            "evaluation_count_per_percentage": total_evaluations,
            "base_random_state": random_state,
            "same_splits_across_percentages": True,
            "new_feature_ranking_per_fold": True,
            "feature_ranking_fit_scope": "cv_fold_train_only",
            "selected_model_fit_scope": "cv_fold_train_only",
            "classifier_input": args.classifier_input,
            "test_seen_during_fit": False,
            "summary_path": str(summary_path),
            "results_path": str(results_path),
            "metric_text_paths": text_paths,
            "summary": summary_rows,
        },
        run_summary_path,
    )
    print(f"\n[OK] Fold results: {results_path}")
    print(f"[OK] CV summary: {summary_path}")
    print(f"[OK] Run summary: {run_summary_path}")


if __name__ == "__main__":
    main()
