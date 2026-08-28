"""Readable orchestration for FeatureRank experiments."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from src.autoencoder_feature_selection import validate_feature_percent
from src.classification import run_classification
from src.clustering import run_clustering
from src.config import DEVICE, ExperimentConfig
from src.data_loader import convert_txt_dataset_to_csv, load_data
from src.runtime import configure_tensorflow_device, set_reproducible
from src.output_paths import normalize_id_column
from src.preprocessing import is_probable_regression_target
from src.reporting import collect_run_row, save_repeated_run_tables


TASK_RUNNERS = {
    "classification": run_classification,
    "clustering": run_clustering,
}


def _prepare_config(config: ExperimentConfig) -> ExperimentConfig:
    task = config.task.lower().strip()
    if task not in {"classification", "regression", "clustering"}:
        raise ValueError("task 'classification', 'regression' veya 'clustering' olmali.")

    return replace(
        config,
        dataset_name=convert_txt_dataset_to_csv(config.dataset_name),
        task=task,
        feature_percent=validate_feature_percent(config.feature_percent),
        id_column=normalize_id_column(config.id_column),
    )


def run_experiment(config: ExperimentConfig) -> tuple[float, float]:
    """Load the dataset and run its configured task."""
    prepared = _prepare_config(config)
    configure_tensorflow_device(DEVICE)
    set_reproducible(prepared.random_state)
    seed_text = "rastgele" if prepared.random_state is None else "sabit"
    print(f"[INFO] random_state: {prepared.random_state} ({seed_text})")

    print(f"[INFO] Veri yukleniyor: {prepared.dataset_name}")
    df = load_data(
        prepared.dataset_name,
        folder="raw",
        target_column=prepared.target_column,
    )

    if prepared.task == "classification" and is_probable_regression_target(
        df[prepared.target_column]
    ):
        raise ValueError(
            "Target kolonu surekli sayisal gorunuyor. Bu veri setini "
            "--task regression ile calistirin."
        )

    if prepared.task == "regression":
        from src.regression import run_regression

        return run_regression(df, prepared)
    return TASK_RUNNERS[prepared.task](df, prepared)


def run_repeated_experiments(
    config: ExperimentConfig,
    repeat_runs: int,
    metric_list_path: Path,
    metric_name: str,
) -> tuple[list[float], float]:
    """Run sequential seeds and save the returned filtered-task metric."""
    if repeat_runs < 1:
        raise ValueError("repeat-runs en az 1 olmali.")

    metric_values: list[float] = []
    durations: list[float] = []
    run_rows: list[dict] = []
    for run_index in range(repeat_runs):
        seed = config.random_state
        if seed is not None and repeat_runs > 1:
            seed += run_index
        print(f"\n[INFO] Calisma {run_index + 1}/{repeat_runs} basladi.")
        started_at = time.perf_counter()
        _, filtered_metric = run_experiment(replace(config, random_state=seed))
        duration = time.perf_counter() - started_at
        durations.append(duration)
        metric_values.append(float(filtered_metric))
        run_rows.append(collect_run_row(config, run_index + 1, duration))

    lower_is_better = metric_name == "Cluster_RMSE"
    sorted_values = sorted(metric_values, reverse=not lower_is_better)
    metric_list_path.parent.mkdir(parents=True, exist_ok=True)
    metric_list_path.write_text(
        f"{metric_values}\n"
        f"Sirali {metric_name}: {sorted_values}\n"
        f"Run sureleri saniye: {[round(value, 3) for value in durations]}",
        encoding="utf-8",
    )
    save_repeated_run_tables(config, run_rows)
    return metric_values, sum(metric_values) / len(metric_values)
