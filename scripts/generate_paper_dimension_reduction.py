from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from paper_dimension_reduction_common import (
    DEFAULT_DECODING_DIM,
    DEFAULT_FACTORIZATION_RANK,
    DEFAULT_MAX_DENSE_WEIGHTS,
    PROJECT_ROOT,
    compute_encoding_dim,
    estimate_dense_memory_gb,
    fit_document_aligned_autoencoder,
    normalize_id_column,
    parse_retained_percentages,
    ra,
    save_headerless_labels,
    save_headerless_matrix,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "DimensionReduction dokumanina gore: "
            "Input -> Encoding -> Decoding -> Output. "
            "Reduced representation = Encoding layer output."
        )
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--id-column", default="none")
    parser.add_argument("--retained-percent", default="all")
    parser.add_argument("--repeat-runs", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=42)

    parser.add_argument("--autoencoder-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-split", type=float, default=0.1)

    parser.add_argument("--decoding-dim", type=int, default=DEFAULT_DECODING_DIM)
    parser.add_argument("--encoding-activation", default="relu")
    parser.add_argument("--decoding-activation", default="relu")
    parser.add_argument("--learning-rate", type=float, default=0.0001)

    parser.add_argument(
        "--encoding-implementation",
        choices=["auto", "dense", "factorized"],
        default="auto",
    )
    parser.add_argument(
        "--factorization-rank",
        type=int,
        default=DEFAULT_FACTORIZATION_RANK,
    )
    parser.add_argument(
        "--max-dense-weights",
        type=int,
        default=DEFAULT_MAX_DENSE_WEIGHTS,
    )

    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.repeat_runs <= 0:
        raise ValueError("--repeat-runs pozitif olmali.")
    if not 0.0 < args.validation_split < 1.0:
        raise ValueError("--validation-split 0 ile 1 arasinda olmali.")

    ra.configure_tensorflow_device(args.device)

    dataset_filename = ra.convert_txt_dataset_to_csv(args.dataset_name)
    dataset_folder = Path(dataset_filename).stem
    id_column = normalize_id_column(args.id_column)

    print(f"[INFO] Dataset yukleniyor: {dataset_filename}")
    df = ra.load_data(
        dataset_filename,
        folder="raw",
        target_column=args.target_column,
    )

    # Sadece >1000 input feature datasetler.
    prepared_for_count = ra.preprocess_data(
        df,
        target_column=args.target_column,
        id_column=id_column,
        random_state=args.base_seed,
        scale_features=False,
    )
    original_feature_count = int(prepared_for_count["X_train"].shape[1])

    # Feature-count threshold intentionally disabled.
    # The original document applies the experiment to datasets with >1000 features,
    # but this implementation is allowed to run on any positive feature count.
    # The dimensionality-reduction logic remains identical.
    if original_feature_count <= 0:
        raise ValueError(
            f"Dataset en az 1 feature icermeli. "
            f"Bu dataset: {original_feature_count} feature."
        )

    if original_feature_count <= 1000:
        print(
            f"[WARN] Dataset {original_feature_count} feature iceriyor. "
            "Dokumandaki deney kapsaminda >1000 feature kullanilmisti; "
            "ancak bu script kullanici istegiyle her feature sayisinda calisacak sekilde ayarlandi."
        )

    retained_values = parse_retained_percentages(args.retained_percent)

    data_root = PROJECT_ROOT / "data" / "paper_dimension_reduction" / dataset_folder
    output_root = PROJECT_ROOT / "outputs" / "paper_dimension_reduction" / dataset_folder
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Original feature count: {original_feature_count}")
    print("[INFO] Mimari: Input -> Encoding -> Decoding -> Output")
    print("[INFO] Reduced representation: ENCODING LAYER OUTPUT")
    print("[INFO] Feature ranking/selection: YOK")
    print("[INFO] Ikinci Autoencoder: YOK")
    print(f"[INFO] Independent runs: {args.repeat_runs}")

    manifest_rows = []

    for run_idx in range(1, args.repeat_runs + 1):
        seed = args.base_seed + run_idx - 1

        # Her independent run kendi outer split'ini kullanir.
        processed = ra.preprocess_data(
            df,
            target_column=args.target_column,
            id_column=id_column,
            random_state=seed,
            scale_features=False,
        )

        X_train_raw = processed["X_train"]
        X_test_raw = processed["X_test"]
        y_train = processed["y_train"].to_numpy().astype(np.int32)
        y_test = processed["y_test"].to_numpy().astype(np.int32)

        X_train_scaled, X_test_scaled, _ = ra.scale_data(
            X_train_raw,
            X_test_raw,
        )

        for retained_percent in retained_values:
            reduction_percent = 100 - retained_percent
            encoding_dim = compute_encoding_dim(
                original_feature_count,
                retained_percent,
            )

            run_data_dir = (
                data_root
                / f"reduction_{reduction_percent:02d}_retained_{retained_percent:03d}"
                / f"run_{run_idx:03d}"
            )
            run_output_dir = (
                output_root
                / f"reduction_{reduction_percent:02d}_retained_{retained_percent:03d}"
                / f"run_{run_idx:03d}"
            )
            run_data_dir.mkdir(parents=True, exist_ok=True)
            run_output_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 80)
            print(
                f"[RUN {run_idx:03d}/{args.repeat_runs:03d}] "
                f"Reduction={reduction_percent}% | "
                f"Retained={retained_percent}% | "
                f"Encoding neurons={encoding_dim}"
            )

            dense_memory_gb = estimate_dense_memory_gb(
                original_feature_count,
                encoding_dim,
            )
            print(
                f"[INFO] Full Dense encoding icin kaba training-memory tahmini "
                f"(encoding matrisi bazli) ~{dense_memory_gb:.2f} GB"
            )

            if (
                args.encoding_implementation == "dense"
                and original_feature_count * encoding_dim > args.max_dense_weights
            ):
                raise MemoryError(
                    "Dense encoding katmani cok buyuk. "
                    "--encoding-implementation auto veya factorized kullanin. "
                    f"input_dim={original_feature_count}, encoding_dim={encoding_dim}, "
                    f"weights={original_feature_count * encoding_dim:,}"
                )

            result = fit_document_aligned_autoencoder(
                X_train_scaled=X_train_scaled,
                X_test_scaled=X_test_scaled,
                y_train=y_train,
                encoding_dim=encoding_dim,
                seed=seed,
                epochs=args.autoencoder_epochs,
                batch_size=args.batch_size,
                validation_split=args.validation_split,
                decoding_dim=args.decoding_dim,
                encoding_activation=args.encoding_activation,
                decoding_activation=args.decoding_activation,
                learning_rate=args.learning_rate,
                encoding_implementation=args.encoding_implementation,
                factorization_rank=args.factorization_rank,
                max_dense_weights=args.max_dense_weights,
                verbose=0 if args.quiet else 1,
            )

            train_data_path = run_data_dir / "train_data.csv"
            train_label_path = run_data_dir / "train_label.csv"
            test_data_path = run_data_dir / "test_data.csv"
            test_label_path = run_data_dir / "test_label.csv"

            save_headerless_matrix(
                train_data_path,
                result["X_train_encoded"],
            )
            save_headerless_labels(
                train_label_path,
                y_train,
            )
            save_headerless_matrix(
                test_data_path,
                result["X_test_encoded"],
            )
            save_headerless_labels(
                test_label_path,
                y_test,
            )

            history_df = pd.DataFrame(result["history"].history)
            history_df.insert(
                0,
                "epoch",
                np.arange(1, len(history_df) + 1),
            )
            history_path = run_output_dir / "autoencoder_history.csv"
            history_df.to_csv(history_path, index=False)

            metadata = {
                "dataset": dataset_folder,
                "source_dataset": dataset_filename,
                "run": run_idx,
                "seed": seed,
                "original_feature_count": original_feature_count,
                "retained_percent": retained_percent,
                "reduction_percent": reduction_percent,
                "encoding_layer_neurons": encoding_dim,
                "architecture": "Input -> Encoding -> Decoding -> Output",
                "reduced_representation_source": "encoding_layer output",
                "feature_ranking_used": False,
                "feature_selection_used": False,
                "second_autoencoder_used": False,
                "scaler_fit_scope": "outer_train_only",
                "autoencoder_fit_scope": "outer_train_only",
                "test_seen_during_fit": False,
                "encoding_implementation": result["encoding_implementation"],
                "document_four_layer_conceptual_architecture": True,
                "literal_dense_encoding": result["encoding_implementation"] == "dense",
                "memory_safe_factorization_used": result["encoding_implementation"] == "factorized",
                "zero_percent_uses_same_autoencoder_pipeline": True,
                "decoding_dim": args.decoding_dim,
                "autoencoder_epochs": args.autoencoder_epochs,
                "reconstruction_mse": result["reconstruction_mse"],
                "reconstruction_rmse": result["reconstruction_rmse"],
                "rmse_definition_note": (
                    "Operational implementation: sqrt(autoencoder test reconstruction MSE). "
                    "The provided document reports RMSE but does not define its exact formula."
                ),
                "train_data_path": str(train_data_path),
                "train_label_path": str(train_label_path),
                "test_data_path": str(test_data_path),
                "test_label_path": str(test_label_path),
                "history_path": str(history_path),
            }

            save_json(
                run_output_dir / "metadata.json",
                metadata,
            )
            manifest_rows.append(metadata)

            if result["encoding_implementation"] == "factorized":
                print(
                    "[INFO] Memory-safe factorized encoding kullanildi. "
                    "Reduced representation yine tek encoding_layer output'udur."
                )

            print(
                f"[OK] Encoded train/test saved. "
                f"RMSE={result['reconstruction_rmse']:.6f}"
            )

    manifest_path = output_root / "generation_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(
        manifest_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print(f"[OK] Generation tamamlandi: {manifest_path}")
    print(
        "[NEXT] python scripts/evaluate_paper_dimension_reduction.py "
        f"--dataset-name {dataset_folder}"
    )


if __name__ == "__main__":
    main()
