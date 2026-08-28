"""Build the repeated-run tables consumed by the figure scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.output_paths import format_feature_percent_tag, task_output_dir
from src.utils import save_json


def _metrics_path(config: ExperimentConfig) -> Path:
    """Find the metric file produced by the current task run."""
    folder = Path(config.dataset_name).stem
    tag = format_feature_percent_tag(config.feature_percent)
    filename = (
        f"top_{tag}_cluster_metrics.json"
        if config.task == "clustering"
        else f"top_{tag}_test_metrics.json"
    )
    return task_output_dir(config.task, folder) / "metrics" / filename


def collect_run_row(
    config: ExperimentConfig,
    run_number: int,
    duration: float,
) -> dict:
    """Read the just-written task metrics into one stable run-table row."""
    metrics = json.loads(_metrics_path(config).read_text(encoding="utf-8"))
    selected_feature_count = metrics.get("selected_feature_count")
    # Older multiclass metric files did not copy this value to the macro row.
    # Recover it from the first completed one-vs-rest class when possible.
    if selected_feature_count is None and config.task == "classification":
        class_metrics_path = metrics.get("class_metrics_path")
        if class_metrics_path:
            class_table_path = Path(class_metrics_path)
            if class_table_path.exists():
                class_table = pd.read_csv(class_table_path)
                for metric_path in class_table.get("metrics_path", []):
                    candidate = Path(str(metric_path))
                    if candidate.exists():
                        class_metrics = json.loads(candidate.read_text(encoding="utf-8"))
                        selected_feature_count = class_metrics.get("selected_feature_count")
                        if selected_feature_count is not None:
                            break
    common = {
        "run": run_number,
        "feature_percent": config.feature_percent,
        "selected_feature_count": selected_feature_count,
    }
    if config.task == "classification":
        return {
            **common,
            "test_accuracy": metrics.get("test_accuracy"),
            "test_precision": metrics.get("test_precision"),
            "test_recall": metrics.get("test_recall"),
            "test_f1": metrics.get("test_f1"),
            "average_precision": metrics.get("average_precision"),
            "roc_auc": metrics.get("roc_auc"),
            "classifier_model": metrics.get("classifier_model"),
            "classifier_class_weight": metrics.get("classifier_class_weight"),
            "classifier_sampling": metrics.get("classifier_sampling"),
            "split_seed": metrics.get("split_seed", config.random_state),
            "method": metrics.get("method", "FeatureRank"),
            "encoded_dataset": metrics.get("encoded_dataset", False),
            "encoded_source_feature_percent": metrics.get("encoded_source_feature_percent"),
            "encoded_active_feature_percent": metrics.get("encoded_active_feature_percent"),
            "elapsed_seconds": duration,
        }
    if config.task == "regression":
        return {
            **common,
            "regression_mse": metrics.get("regression_mse"),
            "regression_rmse": metrics.get("regression_rmse"),
            "regression_mae": metrics.get("regression_mae"),
            "regression_r2": metrics.get("regression_r2"),
            "pearson_r": metrics.get("pearson_r"),
            "correlation": metrics.get("correlation", metrics.get("pearson_r")),
            "cosine_similarity": metrics.get("cosine_similarity"),
            "regression_model": metrics.get("regression_model"),
            "kmeans_regression_clusters": metrics.get("kmeans_regression_clusters"),
            "kmeans_regression_effective_clusters": metrics.get(
                "kmeans_regression_effective_clusters"
            ),
            "kmeans_regression_n_init": metrics.get("kmeans_regression_n_init"),
            "elapsed_seconds": duration,
        }
    return {
        "run": run_number,
        "method": "FeatureRank",
        "feature_percent": config.feature_percent,
        "selected_feature_count": metrics.get("selected_feature_count"),
        "best_k": metrics.get("best_k"),
        "fixed_cluster_k": metrics.get("fixed_cluster_k"),
        "silhouette_score": metrics.get("silhouette_score"),
        "cluster_rmse": metrics.get("cluster_rmse"),
        "inertia": metrics.get("inertia"),
        "elapsed_seconds": duration,
    }


def _mean_std(values: pd.Series) -> tuple[float, float]:
    """Return mean and sample standard deviation for numeric values."""
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    mean = float(np.mean(clean)) if len(clean) else float("nan")
    std = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    return mean, std


def save_repeated_run_tables(config: ExperimentConfig, rows: list[dict]) -> None:
    """Save the CSV and compact summary JSON expected by plotting scripts."""
    if not rows:
        return
    folder = Path(config.dataset_name).stem
    tag = format_feature_percent_tag(config.feature_percent)
    metrics_dir = task_output_dir(config.task, folder) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    runs_path = metrics_dir / f"top_{tag}_{config.task}_runs.csv"
    table = pd.DataFrame(rows)
    table.to_csv(runs_path, index=False)

    selected_counts = table["selected_feature_count"].dropna()
    if selected_counts.empty:
        raise ValueError(
            "Task metrics do not contain selected_feature_count; "
            "the experiment likely stopped before feature selection completed."
        )
    nof = int(selected_counts.iloc[-1])
    if config.task == "classification":
        accuracy, accuracy_std = _mean_std(table["test_accuracy"])
        precision, precision_std = _mean_std(table["test_precision"])
        recall, recall_std = _mean_std(table["test_recall"])
        f1, f1_std = _mean_std(table["test_f1"])
        summary = {
            "Dataset": folder,
            "Method": "FeatureRank",
            "Feature_Percent": config.feature_percent,
            "NOF": nof,
            "Average_Accuracy": accuracy,
            "Accuracy_STD": accuracy_std,
            "Average_Precision": precision,
            "Precision_STD": precision_std,
            "Average_Recall": recall,
            "Recall_STD": recall_std,
            "Average_F1": f1,
            "F1_STD": f1_std,
            "Accuracy_pm_STD": f"{accuracy:.6f} ± {accuracy_std:.6f}",
            "Precision_pm_STD": f"{precision:.6f} ± {precision_std:.6f}",
            "Recall_pm_STD": f"{recall:.6f} ± {recall_std:.6f}",
            "F1_pm_STD": f"{f1:.6f} ± {f1_std:.6f}",
        }
    elif config.task == "regression":
        rmse, rmse_std = _mean_std(table["regression_rmse"])
        corr, corr_std = _mean_std(table["correlation"])
        elapsed, _ = _mean_std(table["elapsed_seconds"])
        summary = {
            "DS": folder.removesuffix("_data").upper(),
            "AL": "FeatureRank",
            "NOF": nof,
            "ET": elapsed,
            "ER": rmse,
            "ER_STD": rmse_std,
            "ER_CI_1": rmse - rmse_std,
            "ER_CI_2": rmse + rmse_std,
            "CORR": corr,
            "CORR_STD": corr_std,
            "CORR_CI_1": corr - corr_std,
            "CORR_CI_2": corr + corr_std,
            "REGRESSION_MODEL": table["regression_model"].iloc[-1],
            "KMEANS_REGRESSION_CLUSTERS": table["kmeans_regression_clusters"].iloc[-1],
            "KMEANS_REGRESSION_N_INIT": table["kmeans_regression_n_init"].iloc[-1],
        }
    else:
        silhouette, silhouette_std = _mean_std(table["silhouette_score"])
        best_k_values = pd.to_numeric(table["best_k"], errors="coerce").dropna()
        best_k_mode = int(best_k_values.mode().iloc[0]) if not best_k_values.empty else None
        summary = {
            "Dataset": folder,
            "Method": "FeatureRank",
            "Feature_Percent": config.feature_percent,
            "NOF": nof,
            "Best_K_Mode": best_k_mode,
            "Average_Silhouette": silhouette,
            "Silhouette_STD": silhouette_std,
            "Silhouette_CI_1": silhouette - silhouette_std,
            "Silhouette_CI_2": silhouette + silhouette_std,
            "Average_Silhouette_pm_STD": f"{silhouette:.6f} ± {silhouette_std:.6f}",
        }

    summary.update(
        {
            "repeat_runs": len(rows),
            "feature_percent": config.feature_percent,
            "runs_csv_path": str(runs_path),
        }
    )
    summary_path = metrics_dir / f"top_{tag}_{config.task}_table_summary.json"
    save_json(summary, summary_path)
    pd.DataFrame([summary]).to_csv(summary_path.with_suffix(".csv"), index=False)
