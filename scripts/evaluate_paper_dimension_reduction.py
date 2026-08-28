"""Evaluate exported encoding representations with the project's classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paper_dimension_reduction_common import (
    PROJECT_ROOT,
    load_json,
    save_json,
)
from src.classification import train_and_evaluate_direct_classifier
from src.runtime import configure_tensorflow_device, set_reproducible


def load_matrix(path: Path) -> np.ndarray:
    """Load one headerless encoded feature matrix."""
    return pd.read_csv(
        path,
        header=None,
        dtype=np.float32,
    ).to_numpy(dtype=np.float32)


def load_labels(path: Path) -> np.ndarray:
    """Load one headerless label vector."""
    return (
        pd.read_csv(
            path,
            header=None,
        )
        .iloc[:, 0]
        .to_numpy(dtype=np.int32)
    )


def parse_hidden_units(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("--classifier-hidden-units bos olamaz.")
    return values


def parse_dropout_rates(
    text: str,
    hidden_count: int,
) -> tuple[float, ...] | None:
    if str(text).strip().lower() in {"", "none", "null", "-"}:
        return None

    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != hidden_count:
        raise ValueError("classifier-dropout-rates uzunlugu hidden layer sayisiyla ayni olmali.")
    return values


def direct_binary_accuracy(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    history_output_dir: Path | None = None,
    history_prefix: str | None = None,
) -> float:
    """Train the canonical binary classifier and return test accuracy."""
    set_reproducible(seed)

    accuracy, _, _ = train_and_evaluate_direct_classifier(
        X_train=X_train.astype(np.float32),
        X_test=X_test.astype(np.float32),
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        random_state=seed,
        history_output_dir=history_output_dir,
        history_prefix=history_prefix,
    )
    return float(accuracy)


def evaluate_classification(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    history_root: Path | None = None,
):
    """Evaluate binary data or the existing one-vs-rest multiclass flow."""
    class_labels = sorted(np.unique(np.concatenate([y_train, y_test])).tolist())

    if len(class_labels) == 2:
        accuracy = direct_binary_accuracy(
            X_train,
            X_test,
            y_train,
            y_test,
            seed,
            history_root,
            "classifier" if history_root is not None else None,
        )
        return accuracy, None

    # Mevcut FeatureRank projesindeki multiclass mantigini koru:
    # selected class -> 0, rest -> 1
    class_counts = {
        int(label): int(np.sum(np.concatenate([y_train, y_test]) == label))
        for label in class_labels
    }

    class_rows = []
    for class_index, class_label in enumerate(class_labels):
        y_train_binary = (y_train != class_label).astype(np.int32)
        y_test_binary = (y_test != class_label).astype(np.int32)

        class_history_dir = None
        if history_root is not None:
            class_history_dir = history_root / f"class_{class_label}"
            class_history_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        class_accuracy = direct_binary_accuracy(
            X_train,
            X_test,
            y_train_binary,
            y_test_binary,
            seed + class_index,
            class_history_dir,
            f"class_{class_label}" if class_history_dir is not None else None,
        )

        class_rows.append(
            {
                "class_label": int(class_label),
                "class_count": class_counts[int(class_label)],
                "accuracy": float(class_accuracy),
            }
        )

    total_weight = float(sum(row["class_count"] for row in class_rows))
    weighted_accuracy = float(
        sum(row["accuracy"] * row["class_count"] for row in class_rows) / total_weight
    )

    return weighted_accuracy, class_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export edilmis Encoding layer representationlarini "
            "IKINCI AUTOENCODER KULLANMADAN dogrudan classifier ile degerlendirir."
        )
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Ornek: arcene_data veya arcene_data.csv",
    )
    parser.add_argument(
        "--classifier-model",
        choices=[
            "neural",
            "logistic",
            "random_forest",
        ],
        default="neural",
    )
    parser.add_argument(
        "--classifier-epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--classifier-hidden-units",
        default="32,16",
    )
    parser.add_argument(
        "--classifier-dropout-rates",
        default="none",
    )
    parser.add_argument(
        "--classifier-learning-rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--classifier-early-stopping-patience",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--classifier-early-stopping-monitor",
        choices=["val_accuracy", "val_loss"],
        default="val_accuracy",
    )
    parser.add_argument(
        "--classifier-early-stopping-min-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--classifier-class-weight",
        choices=["none", "balanced"],
        default="none",
    )
    parser.add_argument(
        "--classifier-sampling",
        choices=["none", "undersample"],
        default="none",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--save-training-plots",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.classifier_hidden_units_parsed = parse_hidden_units(args.classifier_hidden_units)
    args.classifier_dropout_rates_parsed = parse_dropout_rates(
        args.classifier_dropout_rates,
        len(args.classifier_hidden_units_parsed),
    )

    configure_tensorflow_device(args.device)

    dataset_folder = Path(args.dataset_name).stem

    candidate_folders = [dataset_folder]
    if dataset_folder.endswith("_data"):
        candidate_folders.append(dataset_folder[:-5])
    else:
        candidate_folders.append(f"{dataset_folder}_data")

    resolved_folder = None
    output_root = None
    for candidate in candidate_folders:
        candidate_root = PROJECT_ROOT / "outputs" / "paper_dimension_reduction" / candidate
        if candidate_root.exists():
            resolved_folder = candidate
            output_root = candidate_root
            break

    if output_root is None or resolved_folder is None:
        raise FileNotFoundError(
            "paper_dimension_reduction output klasoru bulunamadi. "
            "Once generate_paper_dimension_reduction.py calistir."
        )

    metadata_files = sorted(output_root.glob("reduction_*_retained_*/run_*/metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"metadata.json bulunamadi: {output_root}")

    print(f"[INFO] Dataset: {resolved_folder}")
    print(f"[INFO] Experiment count: {len(metadata_files)}")
    print("[INFO] Second Autoencoder: YOK")
    print("[INFO] Flow: Encoding output -> DIRECT CLASSIFIER -> Accuracy")

    rows = []

    for exp_idx, metadata_path in enumerate(
        metadata_files,
        start=1,
    ):
        metadata = load_json(metadata_path)

        X_train = load_matrix(Path(metadata["train_data_path"]))
        y_train = load_labels(Path(metadata["train_label_path"]))
        X_test = load_matrix(Path(metadata["test_data_path"]))
        y_test = load_labels(Path(metadata["test_label_path"]))

        history_root = None
        if args.save_training_plots:
            history_root = metadata_path.parent / "classifier_history"
            history_root.mkdir(
                parents=True,
                exist_ok=True,
            )

        accuracy, class_rows = evaluate_classification(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            seed=int(metadata["seed"]),
            history_root=history_root,
        )

        row = {
            "dataset": resolved_folder,
            "run": int(metadata["run"]),
            "seed": int(metadata["seed"]),
            "reduction_percent": int(metadata["reduction_percent"]),
            "retained_percent": int(metadata["retained_percent"]),
            "original_feature_count": int(metadata["original_feature_count"]),
            "encoding_layer_neurons": int(metadata["encoding_layer_neurons"]),
            "accuracy": float(accuracy),
            "rmse": float(metadata["reconstruction_rmse"]),
            "rmse_source": "dimension-reduction autoencoder test reconstruction",
            "second_autoencoder_used": False,
            "classifier_model": args.classifier_model,
        }

        rows.append(row)

        save_json(
            metadata_path.parent / "evaluation.json",
            {
                **row,
                "class_details": class_rows,
            },
        )

        print(
            f"[{exp_idx}/{len(metadata_files)}] "
            f"Reduction={row['reduction_percent']}% | "
            f"Run={row['run']} | "
            f"Accuracy={row['accuracy']:.6f} | "
            f"RMSE={row['rmse']:.6f}"
        )

    all_runs_df = pd.DataFrame(rows)

    all_runs_path = output_root / "dimension_reduction_all_runs.csv"
    all_runs_df.to_csv(
        all_runs_path,
        index=False,
    )

    summary_df = (
        all_runs_df.groupby(
            [
                "dataset",
                "reduction_percent",
                "retained_percent",
                "original_feature_count",
                "encoding_layer_neurons",
            ],
            as_index=False,
        )
        .agg(
            runs=("run", "count"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
        )
        .sort_values(
            "reduction_percent",
            ascending=False,
        )
    )

    summary_df[["std_accuracy", "std_rmse"]] = summary_df[["std_accuracy", "std_rmse"]].fillna(0.0)

    results_path = output_root / "dimension_reduction_results.csv"
    summary_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print(summary_df.to_string(index=False))
    print(f"\n[OK] All runs: {all_runs_path}")
    print(f"[OK] Results : {results_path}")


if __name__ == "__main__":
    main()
