"""Single source for output directories and stable artifact names."""

import re
from pathlib import Path


OUTPUT_ROOT = Path("outputs")
CLASSIFICATION_OUTPUT_ROOT = OUTPUT_ROOT / "Classification"
REGRESSION_OUTPUT_ROOT = OUTPUT_ROOT / "Regression"
CLUSTERING_OUTPUT_ROOT = OUTPUT_ROOT / "Clustering"


def classification_output_dir(dataset_folder: str | Path) -> Path:
    return CLASSIFICATION_OUTPUT_ROOT / Path(dataset_folder)


def regression_output_dir(dataset_folder: str | Path) -> Path:
    return REGRESSION_OUTPUT_ROOT / Path(dataset_folder)


def clustering_output_dir(dataset_folder: str | Path) -> Path:
    return CLUSTERING_OUTPUT_ROOT / Path(dataset_folder)


def task_output_dir(task: str, dataset_folder: str | Path) -> Path:
    paths = {
        "classification": classification_output_dir,
        "regression": regression_output_dir,
        "clustering": clustering_output_dir,
    }
    try:
        return paths[task.lower().strip()](dataset_folder)
    except KeyError as exc:
        raise ValueError(f"Bilinmeyen task output tipi: {task}") from exc


def normalize_id_column(id_column: str | None) -> str | None:
    if id_column and id_column.lower() in {"none", "null", "-", ""}:
        return None
    return id_column


def format_feature_percent_tag(feature_percent: float) -> str:
    if float(feature_percent).is_integer():
        return str(int(feature_percent))
    return str(feature_percent).replace(".", "_")


def is_encoded_dataset_folder(dataset_folder: str | None) -> bool:
    """Return whether a folder name represents encoded/reduced features."""
    if dataset_folder is None:
        return False
    folder_name = str(dataset_folder).lower()
    return (
        "_encoded_dim_" in folder_name
        or "_encoded_" in folder_name
        or "_dimension_reduction_" in folder_name
    )


def get_encoded_source_feature_percent(dataset_folder: str) -> float | None:
    match = re.search(r"_top_(\d+(?:_\d+)?)_encoded_dim_", str(dataset_folder).lower())
    if not match:
        return None
    return float(match.group(1).replace("_", "."))


def format_encoded_output_percent(
    feature_percent: float, dataset_folder: str | None = None
) -> float:
    """Keep the active CLI percentage in encoded-dataset output names."""
    return feature_percent


def format_metric_output_prefix(feature_percent: float, dataset_folder: str | None = None) -> str:
    tag = format_feature_percent_tag(format_encoded_output_percent(feature_percent, dataset_folder))
    return f"top_{tag}_encoder" if is_encoded_dataset_folder(dataset_folder) else f"top_{tag}"


def format_test_metrics_filename(feature_percent: float, dataset_folder: str | None = None) -> str:
    tag = format_feature_percent_tag(format_encoded_output_percent(feature_percent, dataset_folder))
    suffix = (
        "_test_encoder_metrics.json"
        if is_encoded_dataset_folder(dataset_folder)
        else "_test_metrics.json"
    )
    return f"top_{tag}{suffix}"


def format_feature_output_label(feature_percent: float, dataset_folder: str | None = None) -> str:
    percent_tag = format_feature_percent_tag(
        format_encoded_output_percent(feature_percent, dataset_folder)
    ).replace("_", ".")
    return (
        f"Top %{percent_tag} encoder"
        if is_encoded_dataset_folder(dataset_folder)
        else f"Top %{percent_tag}"
    )


def add_encoded_metric_metadata(
    metrics_data: dict, feature_percent: float, dataset_folder: str
) -> None:
    if not is_encoded_dataset_folder(dataset_folder):
        return
    source_percent = get_encoded_source_feature_percent(dataset_folder)
    metrics_data["encoded_dataset"] = True
    if source_percent is not None:
        metrics_data["encoded_source_feature_percent"] = source_percent
    metrics_data["encoded_active_feature_percent"] = feature_percent


def format_encoded_dataset_stem(
    dataset_folder: str, feature_percent: float, encoding_dim: int
) -> str:
    tag = format_feature_percent_tag(feature_percent)
    return f"{dataset_folder}_dimension_reduction_{tag}_data"


def metric_runs_path(task: str, dataset_folder: str | Path, feature_percent: float) -> Path:
    """Return the stable text file used for repeated-run metric values."""
    suffixes = {
        "classification": "accuracy_runs.txt",
        "regression": "pearson_r_runs.txt",
        "clustering": "cluster_rmse_runs.txt",
    }
    task_name = task.lower().strip()
    if task_name not in suffixes:
        raise ValueError(f"Bilinmeyen task metric tipi: {task}")
    tag = format_feature_percent_tag(feature_percent)
    return (
        task_output_dir(task_name, dataset_folder) / "metrics" / f"top_{tag}_{suffixes[task_name]}"
    )
