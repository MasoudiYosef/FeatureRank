from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# DATA ENTRY AREA
# ---------------------------------------------------------------------------
# Edit these lists for the datasets that should appear in Panel A and Panel B.
#
# dataset_name:
#   Folder name under outputs/clustering or outputs/autoencoder.
# label:
#   Display name on the x-axis.
# feature_ratio:
#   Feature percentage used for the related result file.
# true_class_count:
#   Panel A only. Ground-truth number of classes. This is shown as the green check mark.
# optimum_k_values:
#   Panel A only. Optional manual optimum k values. Leave as None to read the best k
#   automatically from outputs/clustering/<dataset>/top_<feature_ratio>_cluster_scores.csv.
# metrics_override:
#   Panel B only. Optional manual classification metrics. Leave empty to read from
#   outputs/autoencoder/<dataset>/metrics/top_<feature_ratio>_test_metrics.json.
PANEL_A_DATASET_CONFIGS = [
	{
		"dataset_name": "cortex_nuclear_data",
		"feature_ratio": 40,
		"label": "MiceProtein",
		"true_class_count": 8,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "arcene_data",
		"label": "Arcene",
		"feature_ratio": 60,
		"true_class_count": 2,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "basehock_data",
		"label": "Basehock",
		"feature_ratio": 20,
		"true_class_count": 2,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "breast_cancer_data",
		"label": "BreastCancer",
		"feature_ratio": 60,
		"true_class_count": 2,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "codon_usage_data",
		"label": "CodonUsage",
		"feature_ratio": 60,
		"true_class_count": 11,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "carcinom_data",
		"feature_ratio": 90,
		"label": "Carcinom",
		"true_class_count": 10,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "chd2_data",
		"feature_ratio": 80,
		"label": "CHD2",
		"true_class_count": 2,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "chd5_data",
		"feature_ratio": 90,
		"label": "CHD5",
		"true_class_count": 5,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "gen_expression_data",
		"feature_ratio": 10,
		"label": "GenXpert",
		"true_class_count": 5,
		"optimum_k_values": None,
	},	
	{
		"dataset_name": "arrhythmia_data",
		"feature_ratio": 30,
		"label": "Arrythmia",
		"true_class_count": 13,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "shd_data",
		"feature_ratio": 80,
		"label": "SHD",
		"true_class_count": 2,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "usps_data",
		"feature_ratio": 80,
		"label": "USPS",
		"true_class_count": 2,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "pid_data",
		"feature_ratio": 80,
		"label": "PID",
		"true_class_count": 2,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "pd_data",
		"feature_ratio": 50,
		"label": "ParkinsonSpeech",
		"true_class_count": 2,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "heart_disease_data",
		"feature_ratio": 100,
		"label": "HeartDisease",
		"true_class_count": 2,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "ckd_data",
		"label": "ChronicKidney",
		"feature_ratio": 40,
		"true_class_count": 2,
		"optimum_k_values": None,
	},
]

PANEL_B_DATASET_CONFIGS = [
	{
		"dataset_name": "carcinom_data",
		"feature_ratio": 90,
		"label": "Carcinom",
		"true_class_count": 10,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "cortex_nuclear_data",
		"feature_ratio": 40,
		"label": "MiceProtein",
		"true_class_count": 8,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "gen_expression_data",
		"feature_ratio": 10,
		"label": "GenXpert",
		"true_class_count": 5,
		"optimum_k_values": None,
	},	
	{
		"dataset_name": "codon_usage_data",
		"label": "CodonUsage",
		"feature_ratio": 60,
		"true_class_count": 11,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "arrhythmia_data",
		"feature_ratio": 30,
		"label": "Arrythmia",
		"true_class_count": 13,
		"optimum_k_values": None,
	},
	{
		"dataset_name": "chd5_data",
		"feature_ratio": 90,
		"label": "CHD5",
		"true_class_count": 5,
		"optimum_k_values": None,
	},
]


AUTO_OPTIMUM_K_COUNT = 3
K_MIN = 1
K_MAX = 15
OUTPUT_DIR = Path("outputs") / "FIGURES"
OUTPUT_BASENAME = "cluster_classification_summary_figure"

AXIS_LABEL_FONT_SIZE = 25
TICK_FONT_SIZE = 17
DATASET_TICK_FONT_SIZE = 22
LEGEND_FONT_SIZE = 18
BAR_LABEL_FONT_SIZE = 11
CHECK_FONT_SIZE = 29
PANEL_LABEL_FONT_SIZE = 35


@dataclass
class DatasetConfig:
	dataset_name: str
	label: str
	feature_ratio: str
	true_class_count: int | None = None
	optimum_k_values: list[int] | None = None
	metrics_override: dict[str, float] | None = None


def normalize_feature_ratio(feature_ratio: str | int | float) -> str:
	text = str(feature_ratio).strip().replace("%", "")
	if text.startswith("top_"):
		text = text[4:]
	try:
		value = float(text)
	except ValueError:
		return text.replace(".", "_")
	if value.is_integer():
		return str(int(value))
	return str(value).replace(".", "_")


def load_configs(raw_configs: list[dict]) -> list[DatasetConfig]:
	configs: list[DatasetConfig] = []
	for item in raw_configs:
		optimum_k_values = item.get("optimum_k_values")
		if optimum_k_values is not None:
			optimum_k_values = [int(k) for k in optimum_k_values]
		metrics_override = item.get("metrics_override", {}) or {}
		configs.append(
			DatasetConfig(
				dataset_name=str(item["dataset_name"]).strip(),
				label=str(item.get("label") or item["dataset_name"]).strip(),
				feature_ratio=normalize_feature_ratio(item["feature_ratio"]),
				true_class_count=int(item["true_class_count"]) if item.get("true_class_count") is not None else None,
				optimum_k_values=optimum_k_values,
				metrics_override={k: float(v) for k, v in metrics_override.items()},
			)
		)
	return configs


def cluster_scores_path(config: DatasetConfig, clustering_base_dir: Path) -> Path:
	return clustering_base_dir / config.dataset_name / f"top_{config.feature_ratio}_cluster_scores.csv"


def classification_metrics_path(config: DatasetConfig, autoencoder_base_dir: Path) -> Path:
	return autoencoder_base_dir / config.dataset_name / "metrics" / f"top_{config.feature_ratio}_test_metrics.json"


def read_optimum_k_values(
	config: DatasetConfig,
	clustering_base_dir: Path,
	auto_optimum_count: int = AUTO_OPTIMUM_K_COUNT,
) -> list[int]:
	if config.optimum_k_values:
		return [k for k in config.optimum_k_values if K_MIN <= k <= K_MAX]

	path = cluster_scores_path(config, clustering_base_dir)
	if not path.exists():
		print(f"[WARN] Cluster score file not found: {path}. Optimum k skipped for {config.label}.")
		return []

	scores_df = pd.read_csv(path)
	required_columns = {"k", "silhouette_score"}
	if not required_columns.issubset(scores_df.columns):
		print(f"[WARN] Cluster score file missing columns: {path}. Required: {sorted(required_columns)}")
		return []

	valid_scores = scores_df.dropna(subset=["silhouette_score"]).copy()
	valid_scores = valid_scores[(valid_scores["k"] >= K_MIN) & (valid_scores["k"] <= K_MAX)]
	if valid_scores.empty:
		print(f"[WARN] No valid silhouette scores in k={K_MIN}-{K_MAX}: {path}")
		return []

	best_rows = valid_scores.sort_values("silhouette_score", ascending=False).head(auto_optimum_count)
	return [int(k) for k in best_rows["k"].tolist()]


def read_classification_metrics(config: DatasetConfig, autoencoder_base_dir: Path) -> dict[str, float]:
	if config.metrics_override:
		return normalize_metric_keys(config.metrics_override)

	path = classification_metrics_path(config, autoencoder_base_dir)
	if not path.exists():
		print(f"[WARN] Classification metric file not found: {path}. Using NaN values for {config.label}.")
		return empty_metrics()

	with path.open("r", encoding="utf-8") as f:
		metrics = json.load(f)
	return normalize_metric_keys(metrics)


def normalize_metric_keys(metrics: dict) -> dict[str, float]:
	key_map = {
		"accuracy": ["accuracy", "test_accuracy", "macro_test_accuracy"],
		"precision": ["precision", "test_precision", "macro_test_precision"],
		"recall": ["recall", "test_recall", "macro_test_recall"],
		"f1": ["f1", "f1_score", "test_f1", "macro_test_f1"],
	}
	normalized = {}
	for metric_name, candidate_keys in key_map.items():
		normalized[metric_name] = first_numeric_value(metrics, candidate_keys)
	return normalized


def first_numeric_value(metrics: dict, keys: Iterable[str]) -> float:
	for key in keys:
		if key in metrics and metrics[key] is not None:
			try:
				return float(metrics[key])
			except (TypeError, ValueError):
				continue
	return float("nan")


def empty_metrics() -> dict[str, float]:
	return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}


def plot_panel_a(ax, configs: list[DatasetConfig], optimum_k_map: dict[str, list[int]]) -> None:
	x_positions = np.arange(len(configs))
	k_values = np.arange(K_MIN, K_MAX + 1)
	row_positions = np.arange(len(k_values))

	table_background = np.ones((len(k_values), len(configs)))
	ax.imshow(table_background, cmap="Greys", vmin=0, vmax=1, aspect="auto", alpha=0.0, origin="lower")
	ax.set_xticks(x_positions)
	ax.set_xticklabels(
		[config.label for config in configs],
		rotation=45,
		ha="right",
		rotation_mode="anchor",
		fontsize=DATASET_TICK_FONT_SIZE,
	)
	ax.set_yticks(row_positions)
	ax.set_yticklabels([str(k) for k in k_values], fontsize=TICK_FONT_SIZE)
	ax.set_xticks(np.arange(-0.5, len(configs), 1), minor=True)
	ax.set_yticks(np.arange(-0.5, len(k_values), 1), minor=True)
	ax.set_ylabel("Number of clusters (k)", fontsize=AXIS_LABEL_FONT_SIZE)
	#ax.set_xlabel("Datasets", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=-35)
	ax.set_axisbelow(True)
	ax.grid(which="minor", axis="both", color="#CFCFCF", linewidth=0.8)
	ax.grid(which="major", visible=False)
	ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
	ax.tick_params(which="minor", bottom=False, left=False)
	for spine in ax.spines.values():
		spine.set_linewidth(1.0)

	for x_idx, config in enumerate(configs):
		predicted_values = set(optimum_k_map.get(config.dataset_name, []))
		true_value = config.true_class_count
		for k in predicted_values:
			if K_MIN <= k <= K_MAX:
				y_idx = k - K_MIN
				x_offset = -0.08 if k == true_value else 0.0
				ax.text(
					x_idx + x_offset,
					y_idx,
					"✓",
					ha="center",
					va="center",
					color="black",
					fontsize=CHECK_FONT_SIZE,
					fontweight="bold",
				)
		if true_value is not None and K_MIN <= true_value <= K_MAX:
			y_idx = true_value - K_MIN
			x_offset = 0.08 if true_value in predicted_values else 0.0
			ax.text(
				x_idx + x_offset,
				y_idx,
				"✓",
				ha="center",
				va="center",
				color="#188038",
				fontsize=CHECK_FONT_SIZE,
				fontweight="bold",
			)

	legend_handles = [
		Line2D([0], [0], marker=r"$\checkmark$", color="black", linestyle="None", markersize=20, label="Predicted optimum k"),
		Line2D([0], [0], marker=r"$\checkmark$", color="#188038", linestyle="None", markersize=20, label="True class count"),
	]
	ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=LEGEND_FONT_SIZE)
	panel_label(ax, "A")


def plot_panel_b(ax, configs: list[DatasetConfig], metrics_map: dict[str, dict[str, float]]) -> None:
	metric_names = ["accuracy", "precision", "recall", "f1"]
	metric_labels = ["Accuracy", "Precision", "Recall", "F1-score"]
	colors = ["#2B6CB0", "#DD6B20", "#2F855A", "#805AD5"]
	metric_styles = {
		metric_name: {"label": metric_label, "color": color}
		for metric_name, metric_label, color in zip(metric_names, metric_labels, colors)
	}

	x = np.arange(len(configs))
	width = 0.18
	offsets = (np.arange(len(metric_names)) - (len(metric_names) - 1) / 2.0) * width
	all_values = [
		metrics_map[config.dataset_name].get(metric_name, float("nan"))
		for config in configs
		for metric_name in metric_names
	]
	valid_values = [float(value) for value in all_values if value is not None and not np.isnan(float(value))]
	y_axis_min = 0.70
	y_axis_max = 1.035

	for x_idx, config in enumerate(configs):
		dataset_metrics = [
			(metric_name, metrics_map[config.dataset_name].get(metric_name, float("nan")))
			for metric_name in metric_names
		]
		sorted_dataset_metrics = sorted(
			dataset_metrics,
			key=lambda item: float(item[1]) if item[1] is not None and not np.isnan(float(item[1])) else -np.inf,
			reverse=True,
		)
		for metric_idx, (metric_name, value) in enumerate(sorted_dataset_metrics):
			style = metric_styles[metric_name]
			bars = ax.bar(
				x[x_idx] + offsets[metric_idx],
				value,
				width=width,
				color=style["color"],
				edgecolor="black",
				linewidth=0.4,
			)
			add_bar_labels(ax, bars, y_axis_min=y_axis_min, y_axis_max=y_axis_max)

	ax.set_xticks(x)
	ax.set_xticklabels(
		[config.label for config in configs],
		rotation=30,
		ha="right",
		rotation_mode="anchor",
		fontsize=DATASET_TICK_FONT_SIZE,
	)
	ax.set_ylabel("Performance", fontsize=AXIS_LABEL_FONT_SIZE)
	ax.set_ylim(y_axis_min, y_axis_max)
	#ax.set_yticks(np.arange(0.75, 1.01, 0.05))
	ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
	ax.tick_params(axis="y", labelleft=False)
	ax.grid(axis="y", color="#D0D0D0", linewidth=0.8, alpha=0.75)
	ax.set_axisbelow(True)
	legend_handles = [
		plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.4, label=metric_label)
		for metric_label, color in zip(metric_labels, colors)
	]
	ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.0, 1.17), ncols=4, frameon=True, fontsize=LEGEND_FONT_SIZE)
	panel_label(ax, "B")


def add_bar_labels(ax, bars, y_axis_min: float, y_axis_max: float) -> None:
	for bar in bars:
		height = bar.get_height()
		if np.isnan(height):
			continue
		label_offset = max((y_axis_max - y_axis_min) * 0.015, 0.003)
		ax.text(
			bar.get_x() + bar.get_width() / 2.0,
			min(height + label_offset, y_axis_max - label_offset),
			f"{height:.3f}",
			ha="center",
			va="bottom",
			fontsize=BAR_LABEL_FONT_SIZE,
			rotation=90,
		)


def panel_label(ax, label: str) -> None:
	ax.text(
		-0.07,
		1.03,
		label,
		transform=ax.transAxes,
		ha="left",
		va="bottom",
		fontsize=PANEL_LABEL_FONT_SIZE,
		fontweight="bold",
	)


def style_axes(fig) -> None:
	for ax in fig.axes:
		ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
		ax.xaxis.label.set_size(AXIS_LABEL_FONT_SIZE)
		ax.yaxis.label.set_size(AXIS_LABEL_FONT_SIZE)


def create_figure(
	panel_a_configs: list[DatasetConfig],
	panel_b_configs: list[DatasetConfig],
	clustering_base_dir: Path,
	autoencoder_base_dir: Path,
	output_dir: Path,
	output_basename: str,
	show: bool,
) -> tuple[Path, Path]:
	optimum_k_map = {
		config.dataset_name: read_optimum_k_values(config, clustering_base_dir)
		for config in panel_a_configs
	}
	metrics_map = {
		config.dataset_name: read_classification_metrics(config, autoencoder_base_dir)
		for config in panel_b_configs
	}

	max_dataset_count = max(len(panel_a_configs), len(panel_b_configs), 1)
	fig, axes = plt.subplots(
		2,
		1,
		figsize=(max(12, max_dataset_count * 1.05), 13.5),
		gridspec_kw={"height_ratios": [1.55, 1.0], "hspace": 0.58},
	)
	plot_panel_a(axes[0], panel_a_configs, optimum_k_map)
	plot_panel_b(axes[1], panel_b_configs, metrics_map)
	style_axes(fig)
	for ax in axes:
		ax.yaxis.set_label_coords(-0.025, 0.5)
	fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.095)

	output_dir.mkdir(parents=True, exist_ok=True)
	png_path = output_dir / f"{output_basename}.jpeg"
	pdf_path = output_dir / f"{output_basename}.pdf"
	fig.savefig(png_path, dpi=300, bbox_inches="tight")
	fig.savefig(pdf_path, bbox_inches="tight")
	print(f"[OK] Figure saved: {png_path}")
	print(f"[OK] Figure saved: {pdf_path}")
	if show:
		plt.show()
	plt.close(fig)
	return png_path, pdf_path


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Create clustering/classification summary figure.")
	parser.add_argument("--clustering-base-dir", type=Path, default=Path("outputs") / "clustering")
	parser.add_argument("--autoencoder-base-dir", type=Path, default=Path("outputs") / "autoencoder")
	parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
	parser.add_argument("--output-basename", type=str, default=OUTPUT_BASENAME)
	parser.add_argument("--no-show", action="store_true", help="Do not show the figure after saving.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	panel_a_configs = load_configs(PANEL_A_DATASET_CONFIGS)
	panel_b_configs = load_configs(PANEL_B_DATASET_CONFIGS)
	create_figure(
		panel_a_configs=panel_a_configs,
		panel_b_configs=panel_b_configs,
		clustering_base_dir=args.clustering_base_dir,
		autoencoder_base_dir=args.autoencoder_base_dir,
		output_dir=args.output_dir,
		output_basename=args.output_basename,
		show=not args.no_show,
	)


if __name__ == "__main__":
	main()
