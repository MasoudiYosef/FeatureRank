import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import (
	average_precision_score,
	confusion_matrix,
	precision_recall_curve,
	roc_auc_score,
	roc_curve,
)

PANEL_LABEL_FONTSIZE = 35
AXIS_LABEL_FONTSIZE = 25
TICK_LABEL_FONTSIZE = 17
LEGEND_FONTSIZE = 16
TITLE_FONTSIZE = 26
DEFAULT_BOX_COLOR = "#FFFFFF"
DEFAULT_POINT_COLOR = "#1f77b4"
CONFUSION_MATRIX_MULTIPLIER = 5


DATASET_CONFIGS = [
	{
		"dataset_name": "ckd_Data", 
		"feature_ratio": 40, 
		"label": "CKD", 
		"box_color": "#FFFFFF", "point_color": "#fff200"},	
	{
		"dataset_name": "breast_cancer_data",
		"feature_ratio": 60,
		"label": "BC",
		"box_color": "#FFFFFF",
		"point_color": "#FB00FF",
	},
	{"dataset_name": "pd_data", "feature_ratio": 50, "label": "PS", "box_color": "#FFFFFF", "point_color": "#0084ff"},
	{"dataset_name": "arcene_data", "feature_ratio": 60, "label": "Arcene", "box_color": "#FFFFFF", "point_color": "#fd0000"},
	

	# {"dataset_name": "chd2_data", "feature_ratio": 80, "label": "CHD2", "box_color": "#FFFFFF", "point_color": "#FF0000"},
	# {"dataset_name": "shd_data", "feature_ratio": 80, "label": "SHD", "box_color": "#FFFFFF", "point_color": "#FFF200"},
	# {"dataset_name": "pid_data", "feature_ratio": 80, "label": "PID", "box_color": "#FFFFFF", "point_color": "#008cff"},
	# {
	# 	"dataset_name": "heart_disease_data",
	# 	"feature_ratio": 100,
	# 	"label": "HD",
	# 	"box_color": "#FFFFFF",
	# 	"point_color": "#2ca02c",
	# },

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
			"Item format must be 'dataset_name:feature_ratio[:label]' or "
			"'dataset_name:feature_ratio:label:box_color:point_color'. "
			"Example: breast_cancer_data:60:Breast Cancer:#FFFFFF:#E7C417"
		)

	dataset_name = normalize_dataset_name(parts[0])
	feature_ratio = normalize_feature_ratio(parts[1])
	label = parts[2] if len(parts) in {3, 5} and parts[2] else dataset_name
	box_color = parts[3] if len(parts) == 5 and parts[3] else DEFAULT_BOX_COLOR
	point_color = parts[4] if len(parts) == 5 and parts[4] else DEFAULT_POINT_COLOR
	return DatasetConfig(
		dataset_name=dataset_name,
		feature_ratio=feature_ratio,
		label=label,
		box_color=box_color,
		point_color=point_color,
	)


def load_configs_from_defaults() -> list[DatasetConfig]:
	configs: list[DatasetConfig] = []
	for item in DATASET_CONFIGS:
		dataset_name = normalize_dataset_name(item["dataset_name"])
		feature_ratio = normalize_feature_ratio(item["feature_ratio"])
		label = str(item.get("label") or dataset_name)
		box_color = str(item.get("box_color") or DEFAULT_BOX_COLOR)
		point_color = str(item.get("point_color") or DEFAULT_POINT_COLOR)
		configs.append(
			DatasetConfig(
				dataset_name=dataset_name,
				feature_ratio=feature_ratio,
				label=label,
				box_color=box_color,
				point_color=point_color,
			)
		)
	return configs


def prediction_path(base_dir: Path, config: DatasetConfig) -> Path:
	return base_dir / config.dataset_name / f"top_{config.feature_ratio}_classification_predictions.csv"


def accuracy_runs_path(base_dir: Path, config: DatasetConfig) -> Path:
	return base_dir / config.dataset_name / "metrics" / f"top_{config.feature_ratio}_accuracy_runs.txt"


def load_prediction_data(base_dir: Path, config: DatasetConfig) -> pd.DataFrame | None:
	path = prediction_path(base_dir, config)
	if not path.exists():
		print(f"[WARN] Prediction file missing: {path}")
		return None

	df = pd.read_csv(path)
	required_columns = {"true_label", "predicted_label", "positive_class_score"}
	missing_columns = required_columns - set(df.columns)
	if missing_columns:
		print(f"[WARN] Prediction file missing columns: {path} -> {sorted(missing_columns)}")
		return None
	return df


def load_accuracy_runs(base_dir: Path, config: DatasetConfig) -> list[float]:
	path = accuracy_runs_path(base_dir, config)
	if not path.exists():
		print(f"[WARN] Accuracy runs file missing: {path}")
		return []

	first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
	try:
		values = ast.literal_eval(first_line)
	except (SyntaxError, ValueError):
		print(f"[WARN] Accuracy runs parse edilemedi: {path}")
		return []

	if not isinstance(values, list):
		print(f"[WARN] Accuracy runs ilk satiri liste degil: {path}")
		return []
	return [float(value) for value in values]


def panel_title(ax, text: str) -> None:
	ax.text(
		-0.16,
		1.02,
		text,
		transform=ax.transAxes,
		ha="left",
		va="bottom",
		fontsize=PANEL_LABEL_FONTSIZE,
		fontweight="bold",
		color="black",
		zorder=10,
	)

def style_axis(ax) -> None:
	ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
	ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
	ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)


def plot_aggregate_confusion_matrix(ax, prediction_frames: list[tuple[DatasetConfig, pd.DataFrame]]) -> None:
	total_cm = np.zeros((2, 2), dtype=int)
	for _, df in prediction_frames:
		y_true = df["true_label"].to_numpy(dtype=int)
		y_pred = df["predicted_label"].to_numpy(dtype=int)
		total_cm += confusion_matrix(y_true, y_pred, labels=[0, 1])
	if CONFUSION_MATRIX_MULTIPLIER != 1:
		total_cm = total_cm * CONFUSION_MATRIX_MULTIPLIER

	im = ax.imshow(total_cm, cmap="Blues", aspect="auto")
	#ax_anchor("W")
	ax.set_aspect("auto")
	cax = inset_axes(
		ax,
		width="3.4%",
		height="100%",
		loc="lower left",
		bbox_to_anchor=(1.012, 0.0, 1.0, 1.0),
		bbox_transform=ax.transAxes,
		borderpad=0,
	)
	colorbar = plt.colorbar(im, cax=cax)
	colorbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])
	ax.set_xticklabels(["0", "1"])
	ax.set_yticklabels(["0", "1"])
	ax.set_xlabel("Predicted Label")
	ax.set_ylabel("True Label")
	style_axis(ax)
	panel_title(ax, "A")

	threshold = total_cm.max() / 2.0 if total_cm.size else 0.0
	for i in range(total_cm.shape[0]):
		for j in range(total_cm.shape[1]):
			ax.text(
				j,
				i,
				f"{total_cm[i, j]}",
				ha="center",
				va="center",
				color="white" if total_cm[i, j] > threshold else "black",
				fontweight="bold",
				fontsize=50,
			)


def plot_roc_curves(ax, prediction_frames: list[tuple[DatasetConfig, pd.DataFrame]]) -> None:
	for config, df in prediction_frames:
		y_true = df["true_label"].to_numpy(dtype=int)
		y_score = df["positive_class_score"].to_numpy(dtype=float)
		if len(np.unique(y_true)) < 2:
			print(f"[WARN] ROC skipped, single class in test set: {config.label}")
			continue

		fpr, tpr, _ = roc_curve(y_true, y_score)
		auc_value = roc_auc_score(y_true, y_score)
		ax.plot(
			fpr,
			tpr,
			linewidth=2.4,
			color=config.point_color,
			label=f"{config.label} (AUC={auc_value:.3f})",
		)

	ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Random")
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	style_axis(ax)
	panel_title(ax, "B")
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=LEGEND_FONTSIZE, loc="lower right", frameon=True)


def plot_precision_recall_curves(ax, prediction_frames: list[tuple[DatasetConfig, pd.DataFrame]]) -> None:
	for config, df in prediction_frames:
		y_true = df["true_label"].to_numpy(dtype=int)
		y_score = df["positive_class_score"].to_numpy(dtype=float)
		if len(np.unique(y_true)) < 2:
			print(f"[WARN] PR skipped, single class in test set: {config.label}")
			continue

		precision, recall, _ = precision_recall_curve(y_true, y_score)
		ap_value = average_precision_score(y_true, y_score)
		ax.plot(
			recall,
			precision,
			linewidth=2.4,
			color=config.point_color,
			label=f"{config.label} (AP={ap_value:.3f})",
		)

	ax.set_xlabel("Recall")
	ax.set_ylabel("Precision")
	style_axis(ax)
	panel_title(ax, "C")
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=LEGEND_FONTSIZE, loc="lower left", frameon=True)


def plot_accuracy_boxplots(ax, accuracy_data: list[tuple[DatasetConfig, list[float]]]) -> None:
	labels: list[str] = []
	values: list[list[float]] = []
	box_colors: list[str] = []
	point_colors: list[str] = []
	for config, runs in accuracy_data:
		if not runs:
			print(f"[WARN] Boxplot skipped, empty accuracy list: {config.label}")
			continue
		labels.append(config.label)
		values.append(runs)
		box_colors.append(config.box_color)
		point_colors.append(config.point_color)

	if not values:
		ax.text(0.5, 0.5, "Accuracy runs not found", ha="center", va="center", transform=ax.transAxes)
		ax.set_xticks([])
		ax.set_yticks([])
		panel_title(ax, "D")
		return

	boxplot_artists = ax.boxplot(
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
	for box, box_color in zip(boxplot_artists["boxes"], box_colors):
		box.set_facecolor(box_color)
		box.set_alpha(0.8)

	for idx, runs in enumerate(values, start=1):
		x_positions = np.full(len(runs), idx, dtype=float)
		if len(runs) > 1:
			x_positions += np.linspace(-0.08, 0.08, len(runs))
		ax.scatter(
			x_positions,
			runs,
			s=28,
			alpha=0.82,
			color=point_colors[idx - 1],
			edgecolors="white",
			linewidths=0.45,
			zorder=3,
		)

	ax.set_ylabel("Accuracy")
	ax.set_ylim(0.40, 1.10)
	style_axis(ax)
	panel_title(ax, "D")
	ax.grid(True, axis="y", alpha=0.3)
	ax.tick_params(axis="x", labelrotation=0, labelsize=TICK_LABEL_FONTSIZE)


def create_classification_figure(
	configs: list[DatasetConfig],
	base_dir: Path,
	output_path: Path,
	dpi: int,
) -> None:
	prediction_frames: list[tuple[DatasetConfig, pd.DataFrame]] = []
	accuracy_data: list[tuple[DatasetConfig, list[float]]] = []

	for config in configs:
		predictions = load_prediction_data(base_dir, config)
		if predictions is not None:
			prediction_frames.append((config, predictions))
		accuracy_data.append((config, load_accuracy_runs(base_dir, config)))

	if not prediction_frames:
		raise ValueError("No prediction CSV files found. Run binary classification first.")

	accuracy_mean_by_dataset = {
		config.dataset_name: np.mean(runs) if runs else float("-inf")
		for config, runs in accuracy_data
	}
	prediction_frames = sorted(
		prediction_frames,
		key=lambda item: accuracy_mean_by_dataset.get(item[0].dataset_name, float("-inf")),
		reverse=True,
	)
	accuracy_data = sorted(
		accuracy_data,
		key=lambda item: accuracy_mean_by_dataset.get(item[0].dataset_name, float("-inf")),
		reverse=True,
	)

	fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), facecolor="white")
	plot_aggregate_confusion_matrix(axes[0, 0], prediction_frames)
	plot_roc_curves(axes[0, 1], prediction_frames)
	plot_precision_recall_curves(axes[1, 0], prediction_frames)
	plot_accuracy_boxplots(axes[1, 1], accuracy_data)

	# fig.suptitle("Binary Classification Evaluation Summary", fontsize=TITLE_FONTSIZE, fontweight="bold", y=0.985)
	fig.subplots_adjust(left=0.055, right=0.985, top=0.905, bottom=0.08, wspace=0.30, hspace=0.40)
	a_pos = axes[0, 0].get_position()
	axes[0, 0].set_position([a_pos.x0, a_pos.y0, a_pos.width * 0.96, a_pos.height])
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)
	print(f"[OK] Classification figure saved: {output_path}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Create one aggregate 2x2 binary classification figure from saved prediction and accuracy files."
	)
	parser.add_argument(
		"--items",
		nargs="+",
		default=None,
		help=(
			"Dataset configs. Format: dataset_name:feature_ratio[:label]. "
			"Example: breast_cancer_data:60:BreastCancer pid_data:80:PID"
		),
	)
	parser.add_argument("--base-dir", type=str, default="outputs/autoencoder")
	parser.add_argument("--output", type=str, default="outputs/autoencoder/classification_summary_figure.jpeg")
	parser.add_argument("--dpi", type=int, default=300)
	parser.add_argument("--box-color", type=str, default="", help="D panelindeki tum box/kutu renklerini bu renge ayarlar.")
	parser.add_argument("--point-color", type=str, default="", help="D panelindeki tum yuvarlak nokta renklerini bu renge ayarlar.")
	args = parser.parse_args()

	configs = [parse_item(item) for item in args.items] if args.items else load_configs_from_defaults()
	if args.box_color or args.point_color:
		for config in configs:
			if args.box_color:
				config.box_color = args.box_color
			if args.point_color:
				config.point_color = args.point_color
	create_classification_figure(
		configs=configs,
		base_dir=Path(args.base_dir),
		output_path=Path(args.output),
		dpi=args.dpi,
	)


if __name__ == "__main__":
	main()
