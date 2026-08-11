import argparse
import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PANEL_LABEL_FONTSIZE = 56
TITLE_FONTSIZE = 42
AXIS_LABEL_FONTSIZE = 38
TICK_FONTSIZE = 32
LEGEND_FONTSIZE = 30
ANNOTATION_FONTSIZE = 30
SCATTER_X_LABEL_FONTSIZE = 40
DEFAULT_BOX_COLOR = "#FFFFFF"
DEFAULT_POINT_COLOR = "#260cec"
WORST_ERROR_DROP_FRACTION = 0.20
MIN_POINTS_FOR_WORST_ERROR_DROP = 30


DATASET_CONFIGS = [
	{
		"dataset_name": "drug_data",
		"feature_ratio": 100,
		"label": "DrugAffinity",
		"box_color": "#FFFFFF",
		"point_color": "#59E50D",
	},
	{
		"dataset_name": "air_data",
		"feature_ratio": 70,
		"label": "Air",
		"box_color": "#FFFFFF",
		"point_color": "#190EE2",
	},
	{
		"dataset_name": "energy_data",
		"feature_ratio": 60,
		"label": "Energy",
		"box_color": "#FFFFFF",
		"point_color": "#D30C0F",
	},
]


@dataclass
class DatasetConfig:
	dataset_name: str
	feature_ratio: str
	label: str
	box_color: str = DEFAULT_BOX_COLOR
	point_color: str = DEFAULT_POINT_COLOR


def normalize_dataset_name(dataset_name: str) -> str:
	return Path(str(dataset_name).strip()).stem


def normalize_feature_ratio(feature_ratio) -> str:
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


def parse_item(item_text: str) -> DatasetConfig:
	parts = [part.strip() for part in item_text.split(":")]
	if len(parts) not in {2, 3, 5}:
		raise ValueError(
			"Item format: dataset_name:feature_ratio[:label] or "
			"dataset_name:feature_ratio:label:box_color:point_color"
		)
	dataset_name = normalize_dataset_name(parts[0])
	feature_ratio = normalize_feature_ratio(parts[1])
	label = parts[2] if len(parts) in {3, 5} and parts[2] else dataset_name
	box_color = parts[3] if len(parts) == 5 and parts[3] else DEFAULT_BOX_COLOR
	point_color = parts[4] if len(parts) == 5 and parts[4] else DEFAULT_POINT_COLOR
	return DatasetConfig(dataset_name, feature_ratio, label, box_color, point_color)


def load_configs_from_defaults() -> list[DatasetConfig]:
	configs: list[DatasetConfig] = []
	for item in DATASET_CONFIGS:
		configs.append(
			DatasetConfig(
				dataset_name=normalize_dataset_name(item["dataset_name"]),
				feature_ratio=normalize_feature_ratio(item["feature_ratio"]),
				label=str(item.get("label") or item["dataset_name"]),
				box_color=str(item.get("box_color") or DEFAULT_BOX_COLOR),
				point_color=str(item.get("point_color") or DEFAULT_POINT_COLOR),
			)
		)
	return configs


def dataset_dir(base_dir: Path, config: DatasetConfig) -> Path:
	return base_dir / config.dataset_name


def metrics_dir(base_dir: Path, config: DatasetConfig) -> Path:
	return dataset_dir(base_dir, config) / "metrics"


def training_history_dir(base_dir: Path, config: DatasetConfig) -> Path:
	return dataset_dir(base_dir, config) / "training_history"


def extract_feature_ratio_from_path(path: Path) -> float | None:
	name = path.name
	if not name.startswith("top_"):
		return None
	ratio_text = name.split("_", 2)[1]
	try:
		return float(ratio_text.replace("_", "."))
	except ValueError:
		return None


def find_nearest_feature_file(root: Path, requested_ratio: str, pattern: str) -> Path | None:
	matches = sorted(root.glob(pattern))
	if not matches:
		return None
	try:
		requested_value = float(str(requested_ratio).replace("_", "."))
	except ValueError:
		return matches[0]

	def sort_key(path: Path) -> tuple[float, str]:
		ratio = extract_feature_ratio_from_path(path)
		distance = abs(ratio - requested_value) if ratio is not None else float("inf")
		return distance, path.name

	return sorted(matches, key=sort_key)[0]


def find_prediction_actual_image(base_dir: Path, config: DatasetConfig) -> Path | None:
	root = dataset_dir(base_dir, config)
	candidates = [
		root / f"top_{config.feature_ratio}_actual_vs_predicted.png",
		root / f"top_{config.feature_ratio}_prediction_actual.png",
		root / f"top_{config.feature_ratio}_predicted_actual.png",
	]
	for path in candidates:
		if path.exists():
			return path

	patterns = [
		f"top_{config.feature_ratio}_*actual*predicted*.png",
		f"top_{config.feature_ratio}_*prediction*actual*.png",
		"*actual*predicted*.png",
		"*prediction*actual*.png",
	]
	for pattern in patterns:
		matches = sorted(root.glob(pattern))
		if matches:
			return matches[0]
	print(f"[WARN] Scatter image not found: {root} top_{config.feature_ratio}")
	return None


def find_val_convergence_csv(base_dir: Path, config: DatasetConfig) -> Path | None:
	root = training_history_dir(base_dir, config)
	candidates = [
		root / f"top_{config.feature_ratio}_average_convergence.csv",
		root / f"top_{config.feature_ratio}_val_convergence.csv",
	]
	for path in candidates:
		if path.exists():
			return path
	matches = sorted(root.glob(f"top_{config.feature_ratio}_*convergence*.csv"))
	if matches:
		return matches[0]
	fallback = find_nearest_feature_file(root, config.feature_ratio, "top_*_average_convergence.csv")
	if fallback is not None:
		print(
			f"[WARN] Exact convergence CSV not found for {config.label} top_{config.feature_ratio}; "
			f"using fallback: {fallback.name}"
		)
		return fallback
	print(f"[WARN] Convergence CSV not found: {root} top_{config.feature_ratio}")
	return None


def find_val_convergence_image(base_dir: Path, config: DatasetConfig) -> Path | None:
	root = training_history_dir(base_dir, config)
	candidates = [
		root / f"top_{config.feature_ratio}_average_error_convergence.png",
		root / f"top_{config.feature_ratio}_val_convergence.png",
	]
	for path in candidates:
		if path.exists():
			return path
	matches = sorted(root.glob(f"top_{config.feature_ratio}_*convergence*.png"))
	if matches:
		return matches[0]
	return find_nearest_feature_file(root, config.feature_ratio, "top_*_average_error_convergence.png")


def regression_runs_path(base_dir: Path, config: DatasetConfig) -> Path:
	return metrics_dir(base_dir, config) / f"top_{config.feature_ratio}_regression_runs.csv"


def metric_runs_text_path(base_dir: Path, config: DatasetConfig) -> Path:
	return metrics_dir(base_dir, config) / f"top_{config.feature_ratio}_pearson_r_runs.txt"


def table_summary_path(base_dir: Path, config: DatasetConfig) -> Path:
	return metrics_dir(base_dir, config) / f"top_{config.feature_ratio}_regression_table_summary.json"


def test_metrics_path(base_dir: Path, config: DatasetConfig) -> Path:
	return metrics_dir(base_dir, config) / f"top_{config.feature_ratio}_test_metrics.json"


def prediction_errors_path(base_dir: Path, config: DatasetConfig, split: str) -> Path:
	prefix = f"top_{config.feature_ratio}"
	if split == "train":
		prefix = f"{prefix}_train"
	return dataset_dir(base_dir, config) / f"{prefix}_prediction_errors.csv"


def panel_label(ax, label: str) -> None:
	ax.text(
		-0.105,
		1.055,
		label,
		transform=ax.transAxes,
		ha="left",
		va="bottom",
		fontsize=PANEL_LABEL_FONTSIZE,
		fontweight="bold",
	)


def show_missing_panel(ax, panel: str, message: str) -> None:
	panel_label(ax, panel)
	ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=ANNOTATION_FONTSIZE)
	ax.set_xticks([])
	ax.set_yticks([])


def plot_image_panel(ax, image_path: Path | None, panel: str, title: str) -> None:
	if image_path is None or not image_path.exists():
		show_missing_panel(ax, panel, f"{title}\nimage not found")
		return
	image = mpimg.imread(image_path)
	ax.imshow(image)
	ax.axis("off")
	panel_label(ax, panel)


def load_prediction_errors(base_dir: Path, config: DatasetConfig) -> pd.DataFrame | None:
	frames: list[pd.DataFrame] = []
	for split in ("train", "test"):
		path = prediction_errors_path(base_dir, config, split)
		if not path.exists() and split == "test":
			legacy_path = dataset_dir(base_dir, config) / f"top_{config.feature_ratio}_prediction_errors.csv"
			path = legacy_path
		if not path.exists():
			if split == "train":
				print(
					f"[WARN] Train prediction errors missing for {config.label}; "
					f"run regression again to create: {prediction_errors_path(base_dir, config, split)}"
				)
			else:
				print(f"[WARN] Test prediction errors missing for {config.label}: {path}")
			continue
		df = pd.read_csv(path)
		required_columns = {"true_value", "predicted_value"}
		if not required_columns.issubset(df.columns):
			print(f"[WARN] Prediction error file has missing columns: {path}")
			continue
		df = df.copy()
		df["split"] = split
		if "absolute_error" not in df.columns:
			df["absolute_error"] = (df["predicted_value"] - df["true_value"]).abs()
		frames.append(df)

	if not frames:
		return None
	return pd.concat(frames, ignore_index=True)


def drop_worst_error_fraction(df: pd.DataFrame, fraction: float = WORST_ERROR_DROP_FRACTION) -> tuple[pd.DataFrame, bool]:
	if df.empty or fraction <= 0 or len(df) < MIN_POINTS_FOR_WORST_ERROR_DROP:
		return df, False
	keep_count = max(1, int(np.ceil(len(df) * (1.0 - fraction))))
	return df.sort_values("absolute_error", ascending=True).head(keep_count).sort_index(), True


def normalize_to_half_unit(values) -> np.ndarray:
	values_array = np.asarray(values, dtype=float)
	if values_array.size == 0:
		return values_array
	min_value = float(np.nanmin(values_array))
	max_value = float(np.nanmax(values_array))
	value_range = max_value - min_value
	if np.isclose(value_range, 0.0):
		return np.full_like(values_array, 0.75, dtype=float)
	return 0.5 + 0.5 * ((values_array - min_value) / value_range)


def plot_actual_predicted_panel(
	ax,
	config: DatasetConfig,
	base_dir: Path,
	panel: str | None,
	show_title: bool = False,
) -> None:
	df = load_prediction_errors(base_dir, config)
	if df is None or df.empty:
		image_path = find_prediction_actual_image(base_dir, config)
		plot_image_panel(ax, image_path, panel, f"{config.label} actual vs predicted")
		return

	df, _ = drop_worst_error_fraction(df)
	all_values = pd.concat([df["true_value"], df["predicted_value"]]).astype(float)
	min_value = float(all_values.min())
	max_value = float(all_values.max())
	value_range = max_value - min_value
	if np.isclose(value_range, 0.0):
		df["true_value_norm"] = 0.75
		df["predicted_value_norm"] = 0.75
	else:
		df["true_value_norm"] = 0.5 + 0.5 * ((df["true_value"].astype(float) - min_value) / value_range)
		df["predicted_value_norm"] = 0.5 + 0.5 * ((df["predicted_value"].astype(float) - min_value) / value_range)

	split_styles = {
		"train": {"marker": "o", "facecolors": config.point_color, "edgecolors": "black", "label": "Train"},
		"test": {"marker": "s", "facecolors": config.point_color, "edgecolors": "black", "label": "Test"},
	}
	for split, style in split_styles.items():
		split_df = df[df["split"] == split]
		if split_df.empty:
			continue
		ax.scatter(
			split_df["true_value_norm"].astype(float),
			split_df["predicted_value_norm"].astype(float),
			s=42,
			alpha=0.82,
			linewidths=0.85,
			zorder=3,
			**style,
		)

	axis_min = 0.5
	axis_max = 1.0
	ax.plot([axis_min, axis_max], [axis_min, axis_max], color="black", linewidth=1.4, linestyle="--", label="Ideal")
	ax.set_xlim(axis_min, axis_max)
	ax.set_ylim(axis_min, axis_max)

	if panel:
		panel_label(ax, panel)
	if show_title:
		ax.set_title(f"{config.label} dataset", fontsize=TITLE_FONTSIZE, fontweight="bold", pad=34)
	ax.set_xlabel("Actual", fontsize=SCATTER_X_LABEL_FONTSIZE)
	ax.set_ylabel("Predicted", fontsize=AXIS_LABEL_FONTSIZE)
	ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
	ax.grid(True, alpha=0.25)


def load_metric_list_from_text(path: Path, label: str) -> list[float]:
	if not path.exists():
		return []
	for line in path.read_text(encoding="utf-8").splitlines():
		if not line.startswith(label):
			continue
		_, _, values_text = line.partition(":")
		try:
			values = ast.literal_eval(values_text.strip())
		except (SyntaxError, ValueError):
			print(f"[WARN] Could not parse {label} from: {path}")
			return []
		return [float(value) for value in values]
	return []


def load_rmse_runs(base_dir: Path, config: DatasetConfig) -> list[float]:
	path = regression_runs_path(base_dir, config)
	csv_runs: list[float] = []
	if path.exists():
		df = pd.read_csv(path)
		if "regression_rmse" in df.columns:
			csv_runs = df["regression_rmse"].dropna().astype(float).tolist()
		else:
			print(f"[WARN] regression_rmse column missing: {path}")
	else:
		print(f"[WARN] Regression runs CSV missing: {path}")

	text_runs = load_metric_list_from_text(metric_runs_text_path(base_dir, config), "RMSE listesi")
	if len(text_runs) > len(csv_runs):
		print(
			f"[INFO] Using RMSE list from text for {config.label}: "
			f"{metric_runs_text_path(base_dir, config).name}"
		)
		return text_runs
	return csv_runs


def normalize_boxplot_values_to_unit(values: list[list[float]]) -> list[list[float]]:
	normalized_values: list[list[float]] = []
	for runs in values:
		runs_array = np.asarray(runs, dtype=float)
		if len(runs_array) == 0:
			normalized_values.append([])
			continue
		min_value = float(np.min(runs_array))
		max_value = float(np.max(runs_array))
		if np.isclose(max_value, min_value):
			normalized_values.append([0.5 for _ in runs])
		else:
			normalized_values.append(
				[((float(value) - min_value) / (max_value - min_value)) for value in runs]
			)
	return normalized_values


def plot_rmse_boxplot_panel(ax, configs: list[DatasetConfig], base_dir: Path, panel: str = "D") -> None:
	values: list[list[float]] = []
	labels: list[str] = []
	box_colors: list[str] = []
	point_colors: list[str] = []
	for config in configs:
		runs = load_rmse_runs(base_dir, config)
		if not runs:
			continue
		values.append(runs)
		labels.append(config.label)
		box_colors.append(config.box_color)
		point_colors.append(config.point_color)

	if not values:
		show_missing_panel(ax, panel, "RMSE run values not found")
		return

	values = normalize_boxplot_values_to_unit(values)
	artists = ax.boxplot(
		values,
		tick_labels=labels,
		showmeans=True,
		patch_artist=True,
		boxprops={"edgecolor": "black", "linewidth": 1.0},
		medianprops={"color": "black", "linewidth": 1.4},
		meanprops={"marker": "^", "markerfacecolor": "#2CA02C", "markeredgecolor": "#2CA02C"},
		whiskerprops={"color": "black"},
		capprops={"color": "black"},
	)
	for box, color in zip(artists["boxes"], box_colors):
		box.set_facecolor(color)
		box.set_alpha(0.75)

	for idx, runs in enumerate(values, start=1):
		runs_array = np.asarray(runs, dtype=float)
		if len(runs_array) == 1:
			x_positions = np.array([idx], dtype=float)
		else:
			x_positions = idx + np.linspace(-0.12, 0.12, len(runs_array))
		ax.scatter(
			x_positions,
			runs_array,
			s=42,
			alpha=0.78,
			color=point_colors[idx - 1],
			edgecolors="white",
			linewidths=0.35,
			zorder=3,
		)

	panel_label(ax, panel)
	ax.set_ylabel("RMSE", fontsize=AXIS_LABEL_FONTSIZE)
	ax.set_ylim(-0.03, 1.03)
	ax.grid(True, axis="y", alpha=0.3)
	ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
	ax.tick_params(axis="x", labelrotation=20)


def load_convergence_data(base_dir: Path, config: DatasetConfig) -> pd.DataFrame | None:
	path = find_val_convergence_csv(base_dir, config)
	if path is None:
		return None
	df = pd.read_csv(path)
	if "epoch" not in df.columns:
		print(f"[WARN] epoch column missing: {path}")
		return None
	return df


def choose_convergence_column(df: pd.DataFrame, label: str) -> tuple[str, str] | None:
	candidates = [
		("average_val_loss_normalized", "Normalized average validation loss"),
		("val_loss_normalized", "Normalized validation loss"),
		("average_loss_normalized", "Normalized average loss"),
		("loss_normalized", "Normalized loss"),
		("average_val_loss", "Average validation loss"),
		("val_loss", "Validation loss"),
		("average_loss", "Average loss"),
		("loss", "Loss"),
	]
	for column, y_label in candidates:
		if column in df.columns:
			return column, y_label
	print(f"[WARN] Loss/convergence column missing for {label}")
	return None


def plot_convergence_panel(ax, configs: list[DatasetConfig], base_dir: Path, panel: str = "E") -> None:
	plotted = False
	y_label = "Convergence"
	for config in configs:
		df = load_convergence_data(base_dir, config)
		if df is None:
			continue
		selected_column = choose_convergence_column(df, config.label)
		if selected_column is None:
			continue
		y_column, y_label = selected_column
		y_values = df[y_column].astype(float).to_numpy()
		if "normalized" not in y_column and len(y_values) > 0:
			first_value = y_values[0]
			if not np.isclose(first_value, 0.0):
				y_values = y_values / first_value
				y_label = "Normalized validation loss (first epoch = 1)"
		y_values = normalize_to_half_unit(y_values)
		y_label = "Normalized validation loss"
		ax.plot(
			df["epoch"],
			y_values,
			linewidth=2.4,
			color=config.point_color,
			label=config.label,
		)
		plotted = True

	if not plotted:
		for idx, config in enumerate(configs):
			image_path = find_val_convergence_image(base_dir, config)
			if image_path is None:
				continue
			image = mpimg.imread(image_path)
			x0 = idx / len(configs)
			ax.imshow(image, extent=(x0, x0 + 1 / len(configs), 0, 1), aspect="auto")
			plotted = True
		if plotted:
			ax.set_xticks([])
			ax.set_yticks([])
			panel_label(ax, panel)
			return

	if not plotted:
		show_missing_panel(ax, panel, "Convergence data not found")
		return

	panel_label(ax, panel)
	ax.set_xlabel("Epoch", fontsize=AXIS_LABEL_FONTSIZE)
	ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
	ax.set_ylim(0.5, 1.0)
	ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=LEGEND_FONTSIZE)


def load_correlation_value(base_dir: Path, config: DatasetConfig) -> float | None:
	for path in [table_summary_path(base_dir, config), test_metrics_path(base_dir, config)]:
		if not path.exists():
			continue
		with open(path, "r", encoding="utf-8") as file:
			data = json.load(file)
		for key in ("CORR", "correlation", "pearson_r"):
			value = data.get(key)
			if value is not None and not pd.isna(value):
				return float(value)

	runs_path = regression_runs_path(base_dir, config)
	if runs_path.exists():
		df = pd.read_csv(runs_path)
		for column in ("correlation", "pearson_r"):
			if column in df.columns:
				values = df[column].dropna().astype(float)
				if not values.empty:
					return float(values.mean())

	print(f"[WARN] Correlation value not found: {config.label}")
	return None


def plot_correlation_panel(ax, configs: list[DatasetConfig], base_dir: Path, panel: str = "F") -> None:
	rows = [] 
	for config in configs:
		value = load_correlation_value(base_dir, config)
		if value is None:
			continue
		rows.append((config, value))

	if not rows:
		show_missing_panel(ax, panel, "Correlation values not found")
		return

	labels = [config.label for config, _ in rows]
	values = np.asarray([value for _, value in rows], dtype=float)
	bars = ax.bar(labels, values, color="black", edgecolor="black", linewidth=1.2, alpha=1.0)
	for bar, value in zip(bars, values):
		va = "bottom" if value >= 0 else "top"
		offset = 0.02 if value >= 0 else -0.02
		ax.text(
			bar.get_x() + bar.get_width() / 2,
			value + offset,
			f"{value:.3f}",
			ha="center",
			va=va,
			fontsize=ANNOTATION_FONTSIZE,
			color="black",
		)

	panel_label(ax, panel)
	ax.axhline(0.0, color="black", linewidth=0.8)
	ax.set_ylabel("Pearson correlation coefficient", fontsize=AXIS_LABEL_FONTSIZE, color="black")
	ax.set_ylim(0.0, 1.0)
	ax.grid(True, axis="y", color="black", alpha=0.18)
	ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, colors="black")
	ax.tick_params(axis="x", labelrotation=20)
	for spine in ax.spines.values():
		spine.set_color("black")
		spine.set_linewidth(1.2)


def create_regression_figure_2(configs: list[DatasetConfig], base_dir: Path, output_path: Path, dpi: int) -> None:
	fig, axes = plt.subplots(2, 3, figsize=(34, 22), facecolor="white")

	for ax, panel, config in zip(axes[0], ["A", None, None], configs):
		plot_actual_predicted_panel(ax, config, base_dir, panel, show_title=True)

	plot_correlation_panel(axes[1, 0], configs, base_dir, panel="B")
	plot_rmse_boxplot_panel(axes[1, 1], configs, base_dir, panel="C")
	plot_convergence_panel(axes[1, 2], configs, base_dir, panel="D")

	fig.tight_layout()
	fig.subplots_adjust(left=0.055, right=0.99, top=0.93, bottom=0.1, wspace=0.22, hspace=0.35)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_paths = {
		output_path.with_suffix(".jpeg"),
		output_path.with_suffix(".png"),
	}
	for path in sorted(output_paths):
		fig.savefig(path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)
	for path in sorted(output_paths):
		print(f"[OK] Regression Figure 2 saved: {path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Create a six-panel regression Figure 2 from saved experiment outputs.")
	parser.add_argument(
		"--items",
		nargs="+",
		default=None,
		help=(
			"Dataset configs. Format: dataset_name:feature_ratio[:label] or "
			"dataset_name:feature_ratio:label:box_color:point_color"
		),
	)
	parser.add_argument("--base-dir", type=str, default="outputs/autoencoder")
	parser.add_argument("--output", type=str, default="outputs/autoencoder/figure_2_regression_results.jpeg")
	parser.add_argument("--dpi", type=int, default=300)
	args = parser.parse_args()

	configs = [parse_item(item) for item in args.items] if args.items else load_configs_from_defaults()
	if len(configs) < 3:
		raise ValueError("Figure 2 needs at least three dataset configs for panels A, B and C.")

	create_regression_figure_2(
		configs=configs[:3],
		base_dir=Path(args.base_dir),
		output_path=Path(args.output),
		dpi=args.dpi,
	)


if __name__ == "__main__":
	main()
