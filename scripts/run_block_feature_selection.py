from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".matplotlib_cache").resolve()))
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "scripts"))

from src.autoencoder_feature_selection import validate_feature_percent
from src.data_loader import load_data
from src.preprocessing import preprocess_data, scale_data
from src.utils import ensure_dir, save_json

import run_autoencoder as ra


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str
    dataset_folder: str
    target_column: str
    id_column: str | None
    block_size: int
    feature_percent: float | None
    top_k: int | None
    encoding_dim: int
    random_state: int | None
    autoencoder_epochs: int
    classifier_epochs: int
    classifier_hidden_units: tuple[int, ...]
    classifier_dropout_rates: tuple[float, ...] | None
    classifier_learning_rate: float
    classifier_model: str
    classifier_class_weight: str
    classifier_sampling: str
    final_model_mode: str
    output_dir: Path


def parse_random_state(text: str | None) -> int | None:
    if text is None:
        return 42
    value = str(text).strip().lower()
    if value in {"none", "null", ""}:
        return None
    return int(value)


def parse_hidden_units(text: str) -> tuple[int, ...]:
    units = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not units:
        raise ValueError("--classifier-hidden-units bos olamaz.")
    return units


def parse_dropout_rates(text: str | None, layer_count: int) -> tuple[float, ...] | None:
    if text is None or str(text).strip() == "":
        return None
    rates = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if len(rates) != layer_count:
        raise ValueError("--classifier-dropout-rates uzunlugu hidden layer sayisiyla ayni olmali.")
    return rates


def select_count(total_features: int, feature_percent: float | None, top_k: int | None) -> int:
    if top_k is not None:
        return min(max(int(top_k), 1), total_features)
    if feature_percent is None:
        raise ValueError("--feature-percent veya --top-k verilmelidir.")
    validate_feature_percent(float(feature_percent))
    return min(max(math.ceil(total_features * (float(feature_percent) / 100.0)), 1), total_features)


def make_blocks(feature_names: list[str], block_size: int) -> list[tuple[int, int, list[str]]]:
    if block_size <= 0:
        raise ValueError("--block-size pozitif olmali.")
    blocks: list[tuple[int, int, list[str]]] = []
    for start in range(0, len(feature_names), block_size):
        end = min(start + block_size, len(feature_names))
        blocks.append((start, end, feature_names[start:end]))
    return blocks


def feature_scores_from_autoencoder(
    autoencoder,
    X_train_sub: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    weights = autoencoder.get_layer("enc_dense_1").get_weights()[0]
    if weights.shape[0] != X_train_sub.shape[1]:
        raise ValueError(
            f"Agirlik feature sayisi ({weights.shape[0]}) ile X_train_sub ({X_train_sub.shape[1]}) eslesmiyor."
        )
    contributions = np.abs(X_train_sub[:, :, np.newaxis] * weights[np.newaxis, :, :])
    weighted = np.mean(contributions, axis=0)
    scores = np.max(np.abs(weighted), axis=1)
    return (
        pd.DataFrame(
            {
                "feature_name": feature_names,
                "max_abs_weight_score": scores,
            }
        )
        .sort_values("max_abs_weight_score", ascending=False)
        .reset_index(drop=True)
    )


def run_featurerank_selection(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    config: ExperimentConfig,
    selected_count: int,
) -> pd.DataFrame:
    ra.set_reproducible(config.random_state)
    X_train_sub, X_val, _y_train_sub, _ = ra.train_test_split(
        X_train,
        y_train,
        test_size=ra.CLASSIFIER_VALIDATION_SPLIT,
        random_state=config.random_state,
        shuffle=True,
        stratify=y_train,
    )
    _mse, autoencoder, _encoder = ra.train_autoencoder_model(
        X_train_sub=X_train_sub.astype(np.float32),
        X_val=X_val.astype(np.float32),
        X_eval=X_test.astype(np.float32),
        encoding_dim=config.encoding_dim,
        autoencoder_epochs=config.autoencoder_epochs,
        early_stopping_patience=0,
        early_stopping_min_delta=0.0,
        shuffle_training=config.random_state is None,
    )
    ranked = feature_scores_from_autoencoder(autoencoder, X_train_sub, feature_names)
    return ranked.head(selected_count).copy()


def evaluate_classifier(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: ExperimentConfig,
) -> dict:
    class_labels = np.unique(np.concatenate([y_train, y_test]))
    ra.set_reproducible(config.random_state)

    if config.final_model_mode == "autoencoder_pipeline":
        if len(class_labels) != 2:
            raise ValueError("--final-model-mode autoencoder_pipeline su an sadece binary sinif icin destekleniyor.")
        _mse, accuracy, _ae, _enc, _train_sub, y_pred, _score = ra.train_and_evaluate_pipeline(
            X_train=X_train.astype(np.float32),
            X_test=X_test.astype(np.float32),
            y_train=y_train.astype(np.int32),
            y_test=y_test.astype(np.int32),
            encoding_dim=config.encoding_dim,
            random_state=config.random_state,
            classifier_epochs=config.classifier_epochs,
            classifier_hidden_units=config.classifier_hidden_units,
            classifier_dropout_rates=config.classifier_dropout_rates,
            classifier_learning_rate=config.classifier_learning_rate,
            classifier_model=config.classifier_model,
            classifier_early_stopping_patience=0,
            autoencoder_early_stopping_patience=0,
            classifier_class_weight=config.classifier_class_weight,
            classifier_sampling=config.classifier_sampling,
        )
    elif len(class_labels) == 2:
        accuracy, y_pred, _score = ra.train_and_evaluate_direct_classifier(
            X_train=X_train.astype(np.float32),
            X_test=X_test.astype(np.float32),
            y_train=y_train.astype(np.int32),
            y_test=y_test.astype(np.int32),
            random_state=config.random_state,
            classifier_epochs=config.classifier_epochs,
            classifier_hidden_units=config.classifier_hidden_units,
            classifier_dropout_rates=config.classifier_dropout_rates,
            classifier_learning_rate=config.classifier_learning_rate,
            classifier_model=config.classifier_model,
            classifier_early_stopping_patience=0,
            classifier_class_weight=config.classifier_class_weight,
            classifier_sampling=config.classifier_sampling,
        )
    else:
        accuracy, y_pred, _score = ra.train_and_evaluate_direct_multiclass_classifier(
            X_train=X_train.astype(np.float32),
            X_test=X_test.astype(np.float32),
            y_train=y_train.astype(np.int32),
            y_test=y_test.astype(np.int32),
            random_state=config.random_state,
            classifier_epochs=config.classifier_epochs,
            classifier_hidden_units=config.classifier_hidden_units,
            classifier_dropout_rates=config.classifier_dropout_rates,
            classifier_learning_rate=config.classifier_learning_rate,
            classifier_model=config.classifier_model,
            classifier_early_stopping_patience=0,
        )

    average = "binary" if len(class_labels) == 2 else "weighted"
    return {
        "accuracy": float(accuracy),
        "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
    }


def evaluate_feature_subset(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    config: ExperimentConfig,
) -> dict:
    X_train_subset_raw = X_train_raw[feature_names]
    X_test_subset_raw = X_test_raw[feature_names]
    X_train_subset, X_test_subset, _ = scale_data(X_train_subset_raw, X_test_subset_raw)
    return evaluate_classifier(X_train_subset, X_test_subset, y_train, y_test, config)

def run_experiment(config: ExperimentConfig) -> None:
    start_time = time.perf_counter()
    ensure_dir(config.output_dir)

    df = load_data(config.dataset_name, folder="raw", target_column=config.target_column)
    processed = preprocess_data(
        df,
        target_column=config.target_column,
        id_column=config.id_column,
        random_state=config.random_state,
        scale_features=False,
    )
    X_train_raw: pd.DataFrame = processed["X_train"]
    X_test_raw: pd.DataFrame = processed["X_test"]
    y_train = processed["y_train"].to_numpy().astype(np.int32)
    y_test = processed["y_test"].to_numpy().astype(np.int32)

    X_train_scaled, X_test_scaled, _ = scale_data(X_train_raw, X_test_raw)
    feature_names = X_train_raw.columns.tolist()
    total_features = len(feature_names)
    selected_global_count = select_count(total_features, config.feature_percent, config.top_k)

    print(f"[INFO] Dataset: {config.dataset_name}")
    print(f"[INFO] Train shape: {X_train_raw.shape}, Test shape: {X_test_raw.shape}")
    print(f"[INFO] Global selected feature count: {selected_global_count}")

    all_feature_metrics = evaluate_classifier(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        config,
    )

    global_selected = run_featurerank_selection(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        feature_names=feature_names,
        config=config,
        selected_count=selected_global_count,
    )
    global_selected["rank"] = np.arange(1, len(global_selected) + 1)
    global_selected.to_csv(config.output_dir / "global_selected_features.csv", index=False)
    global_features = global_selected["feature_name"].tolist()
    global_metrics = evaluate_feature_subset(
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
        global_features,
        config,
    )

    block_rows: list[dict] = []
    block_selected_rows: list[pd.DataFrame] = []
    for block_index, (start, end, block_feature_names) in enumerate(make_blocks(feature_names, config.block_size), start=1):
        block_selected_count = select_count(len(block_feature_names), config.feature_percent, config.top_k)
        X_train_block = X_train_scaled[:, start:end]
        X_test_block = X_test_scaled[:, start:end]
        selected = run_featurerank_selection(
            X_train=X_train_block,
            X_test=X_test_block,
            y_train=y_train,
            feature_names=block_feature_names,
            config=config,
            selected_count=block_selected_count,
        )
        selected.insert(0, "block", block_index)
        selected.insert(1, "block_feature_start", start + 1)
        selected.insert(2, "block_feature_end", end)
        selected.insert(3, "rank_in_block", np.arange(1, len(selected) + 1))
        block_selected_rows.append(selected)
        block_rows.append(
            {
                "block": block_index,
                "feature_range": f"{start + 1}-{end}",
                "initial_feature_count": len(block_feature_names),
                "selected_feature_count": len(selected),
                "selected_features": ", ".join(selected["feature_name"].tolist()),
            }
        )
        print(f"[OK] Block {block_index}: {len(block_feature_names)} -> {len(selected)} feature")

    block_selected = pd.concat(block_selected_rows, ignore_index=True)
    block_selected.to_csv(config.output_dir / "block_selected_features.csv", index=False)
    pd.DataFrame(block_rows).to_csv(config.output_dir / "block_feature_counts.csv", index=False)

    block_features = list(dict.fromkeys(block_selected["feature_name"].tolist()))
    block_metrics = evaluate_feature_subset(
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
        block_features,
        config,
    )

    global_set = set(global_features)
    block_set = set(block_features)
    common = sorted(global_set & block_set)
    only_global = sorted(global_set - block_set)
    only_block = sorted(block_set - global_set)
    union_count = len(global_set | block_set)
    jaccard = float(len(common) / union_count) if union_count else 0.0

    comparison_rows = [
        {"group": "common_features", "count": len(common), "features": ", ".join(common)},
        {"group": "only_global", "count": len(only_global), "features": ", ".join(only_global)},
        {"group": "only_block_based", "count": len(only_block), "features": ", ".join(only_block)},
        {"group": "jaccard_similarity", "count": jaccard, "features": ""},
    ]
    pd.DataFrame(comparison_rows).to_csv(config.output_dir / "feature_comparison.csv", index=False)

    performance_rows = [
        {
            "method": "All Features Model",
            "initial_feature_count": total_features,
            "selected_feature_count": total_features,
            **all_feature_metrics,
        },
        {
            "method": "Global Feature Selection",
            "initial_feature_count": total_features,
            "selected_feature_count": len(global_features),
            **global_metrics,
        },
        {
            "method": "Block-Based Feature Selection",
            "initial_feature_count": f"{len(block_rows)} x {config.block_size}",
            "selected_feature_count": len(block_features),
            **block_metrics,
        },
    ]
    performance_df = pd.DataFrame(performance_rows)
    performance_df.to_csv(config.output_dir / "model_performance_comparison.csv", index=False)

    summary = {
        "dataset_name": config.dataset_name,
        "random_state": config.random_state,
        "block_size": config.block_size,
        "feature_percent": config.feature_percent,
        "top_k": config.top_k,
        "total_features": total_features,
        "block_count": len(block_rows),
        "global_selected_feature_count": len(global_features),
        "block_based_selected_feature_count": len(block_features),
        "common_feature_count": len(common),
        "only_global_feature_count": len(only_global),
        "only_block_based_feature_count": len(only_block),
        "jaccard_similarity": jaccard,
        "all_features_metrics": all_feature_metrics,
        "global_feature_selection_metrics": global_metrics,
        "block_based_feature_selection_metrics": block_metrics,
        "elapsed_seconds": time.perf_counter() - start_time,
        "output_dir": str(config.output_dir),
    }
    save_json(summary, config.output_dir / "summary.json")

    print("\n[OK] Block feature selection experiment tamamlandi.")
    print(performance_df.to_string(index=False))
    print(f"[OK] Jaccard Similarity: {jaccard:.6f}")
    print(f"[OK] Output dir: {config.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Global FeatureRank ile block-based FeatureRank secimini ayni split uzerinde karsilastirir."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--target-column", default="diagnosis")
    parser.add_argument("--id-column", default="none")
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--feature-percent", type=float, default=20.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--encoding-dim", type=int, default=8)
    parser.add_argument("--random-state", default="42")
    parser.add_argument("--autoencoder-epochs", type=int, default=50)
    parser.add_argument("--classifier-model", choices=["neural", "logistic", "svm", "random_forest"], default="neural")
    parser.add_argument("--classifier-epochs", type=int, default=50)
    parser.add_argument("--classifier-hidden-units", default="32,16")
    parser.add_argument("--classifier-dropout-rates", default=None)
    parser.add_argument("--classifier-learning-rate", type=float, default=0.001)
    parser.add_argument("--classifier-class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--classifier-sampling", choices=["none", "undersample"], default="none")
    parser.add_argument("--final-model-mode", choices=["direct", "autoencoder_pipeline"], default="direct")
    parser.add_argument("--output-root", default="outputs/block_feature_selection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hidden_units = parse_hidden_units(args.classifier_hidden_units)
    dropout_rates = parse_dropout_rates(args.classifier_dropout_rates, len(hidden_units))
    dataset_folder = Path(args.dataset_name).stem
    id_column = None if str(args.id_column).strip().lower() in {"none", "null", ""} else args.id_column
    selection_label = f"top_k_{args.top_k}" if args.top_k is not None else f"top_{ra.format_feature_percent_tag(args.feature_percent)}"
    output_dir = PROJECT_ROOT / args.output_root / dataset_folder / f"{selection_label}_block_{args.block_size}"

    config = ExperimentConfig(
        dataset_name=args.dataset_name,
        dataset_folder=dataset_folder,
        target_column=args.target_column,
        id_column=id_column,
        block_size=args.block_size,
        feature_percent=args.feature_percent,
        top_k=args.top_k,
        encoding_dim=args.encoding_dim,
        random_state=parse_random_state(args.random_state),
        autoencoder_epochs=args.autoencoder_epochs,
        classifier_epochs=args.classifier_epochs,
        classifier_hidden_units=hidden_units,
        classifier_dropout_rates=dropout_rates,
        classifier_learning_rate=args.classifier_learning_rate,
        classifier_model=args.classifier_model,
        classifier_class_weight=args.classifier_class_weight,
        classifier_sampling=args.classifier_sampling,
        final_model_mode=args.final_model_mode,
        output_dir=output_dir,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
