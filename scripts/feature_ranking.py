"""Command-line entry point for the canonical FeatureRank workflow."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.autoencoder_feature_selection import validate_feature_percent
from src.config import ID_COLUMN, TARGET_COLUMN, ExperimentConfig, RANDOM_STATE
from src.divide_combine import DEFAULT_BLOCK_COUNT
from src.output_paths import metric_runs_path


def parse_feature_percent_values(value: str) -> list[float]:
    """Accept one percentage, a comma-separated list, or ``all``."""
    text = value.strip().lower()
    if text == "all":
        return [float(percent) for percent in range(10, 101, 10)]
    percentages = [
        validate_feature_percent(float(part.strip())) for part in text.split(",") if part.strip()
    ]
    if not percentages:
        raise ValueError("feature-percent bos olamaz.")
    return list(dict.fromkeys(percentages))


def parse_random_state(value: str) -> int | None:
    """Turn the CLI seed into an integer, or disable seeding with ``none``."""
    return None if value.strip().lower() in {"none", "null"} else int(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FeatureRank autoencoder ile feature ranking ve task evaluation"
    )
    parser.add_argument("--dataset-name", default="breast_cancer_data.csv")
    parser.add_argument(
        "--task",
        choices=("classification", "regression", "clustering"),
        default="classification",
    )
    parser.add_argument("--feature-percent", default="20")
    parser.add_argument("--random-state", default=str(RANDOM_STATE))
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--id-column", default=ID_COLUMN)
    parser.add_argument("--encoding-dim", type=int, default=8)
    parser.add_argument("--cluster-k", type=int)
    parser.add_argument("--save-details", action="store_true")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--global",
        dest="mode",
        action="store_const",
        const="global",
        help="Normal, dataset geneli FeatureRank akisini calistirir (varsayilan).",
    )
    mode_group.add_argument(
        "--dc",
        dest="mode",
        action="store_const",
        const="dc",
        help="Divide & Combine: split, blok ranking, mapping, combine ve final egitim.",
    )
    parser.set_defaults(mode="global")
    parser.add_argument(
        "--block-count",
        type=int,
        default=DEFAULT_BLOCK_COUNT,
        help=f"DC modunda feature blok sayisi (varsayilan: {DEFAULT_BLOCK_COUNT}).",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace, feature_percent: float) -> ExperimentConfig:
    """Create one immutable experiment configuration for a percentage."""
    id_column = args.id_column
    if id_column.lower() in {"none", "null", "-", ""}:
        id_column = None
    return ExperimentConfig(
        dataset_name=args.dataset_name,
        task=args.task,
        feature_percent=feature_percent,
        random_state=parse_random_state(args.random_state),
        encoding_dim=args.encoding_dim,
        target_column=args.target_column,
        id_column=id_column,
        cluster_k=args.cluster_k,
        save_details=args.save_details,
    )


def metric_name_for_task(task: str) -> str:
    """Return the metric printed for the selected task."""
    return {
        "classification": "Accuracy",
        "regression": "Pearson_r",
        "clustering": "Cluster_RMSE",
    }[task]


def print_summary(
    config: ExperimentConfig,
    metric_name: str,
    metric_values: list[float],
    average_metric: float,
) -> None:
    print(f"Dataset: {Path(config.dataset_name).stem}")
    print(f"Task: {config.task}")
    print(f"Feature percent: {config.feature_percent:g}%")
    print(f"{metric_name} runs: {metric_values}")
    print(f"Average {metric_name}: {average_metric:.6f}")


def run_workflow(
    config: ExperimentConfig,
    mode: str,
    repeat_runs: int,
    metric_name: str,
    metric_path: Path,
    block_count: int,
) -> tuple[list[float], float]:
    """Run either the normal Global workflow or the Divide & Combine workflow."""
    if mode == "dc":
        from src.divide_combine import run_repeated_divide_combine

        return run_repeated_divide_combine(
            config,
            repeat_runs=repeat_runs,
            block_count=block_count,
            metric_path=metric_path,
            metric_name=metric_name,
        )

    from src.workflow import run_repeated_experiments

    return run_repeated_experiments(
        config,
        repeat_runs=repeat_runs,
        metric_list_path=metric_path,
        metric_name=metric_name,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_config = build_config(args, feature_percent=20.0)
    for feature_percent in parse_feature_percent_values(args.feature_percent):
        config = replace(base_config, feature_percent=feature_percent)
        metric_name = metric_name_for_task(config.task)
        metric_path = metric_runs_path(
            config.task,
            Path(config.dataset_name).stem,
            config.feature_percent,
        )
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_values, average_metric = run_workflow(
            config=config,
            mode=args.mode,
            repeat_runs=args.repeat_runs,
            metric_name=metric_name,
            metric_path=metric_path,
            block_count=args.block_count,
        )
        print_summary(config, metric_name, metric_values, average_metric)


if __name__ == "__main__":
    main()
