import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


DEFAULT_DATASET_NAME = "carcinom_data"
DEFAULT_TOP_PERCENTAGE = 100
DEFAULT_K_VALUES = [2,11,12]
DEFAULT_PANEL_A_K = 2
DEFAULT_BASE_DIR = Path("outputs") / "clustering"
DEFAULT_OUTPUT_DIR = Path("outputs") / "FIGURES"
SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
PANEL_LETTER_FONTSIZE = 54
SUBPLOT_TITLE_FONTSIZE = 32
AXIS_LABEL_FONTSIZE = 45
TICK_LABEL_FONTSIZE = 36
SCALE_NOTE_FONTSIZE = 33
PANEL_LETTER_X = 0.02
PANEL_LETTER_Y_OFFSET = -0.005
PANEL_A_ALIGN_LEFT = 0.081
PANEL_A_ALIGN_RIGHT = 0.951
PANEL_A_WIDTH_SCALE = 0.915
PCA_CLUSTER_CMAP = ListedColormap(
	[
		"#2b8cbe",
		"#36b37e",
		"#ef4444",
		"#9e9e9e",
		"#f472b6",
		"#ffd92f",
		"#7b61ff",
		"#8dd3c7",
		"#fb8072",
		"#80b1d3",
		"#b3de69",
		"#bc80bd",
		"#ccebc5",
		"#ffed6f",
		"#8c564b",
	]
)


def build_cluster_colormap(cluster_count: int) -> ListedColormap:
	if cluster_count <= PCA_CLUSTER_CMAP.N:
		return ListedColormap(PCA_CLUSTER_CMAP.colors[:cluster_count], name="feature_rank_cluster_dynamic")
	return ListedColormap(plt.get_cmap("tab20", cluster_count)(np.arange(cluster_count)), name="feature_rank_cluster_dynamic")


@dataclass(frozen=True)
class ClusterFigureConfig:
	dataset_name: str = DEFAULT_DATASET_NAME
	top_percentage: int = DEFAULT_TOP_PERCENTAGE
	k_values: tuple[int, ...] = tuple(DEFAULT_K_VALUES)
	panel_a_k: int = DEFAULT_PANEL_A_K
	use_pca_csv: bool = False
	base_dir: Path = DEFAULT_BASE_DIR
	output_dir: Path = DEFAULT_OUTPUT_DIR
	output_prefix: str | None = None
	dpi: int = 300


def normalize_dataset_name(dataset_name: str) -> str:
	return Path(str(dataset_name).strip()).stem


def format_top_percentage(top_percentage: int | float | str) -> str:
	value = float(str(top_percentage).strip().replace("%", ""))
	if value.is_integer():
		return str(int(value))
	return str(value).replace(".", "_")


def dataset_title(dataset_name: str) -> str:
	return normalize_dataset_name(dataset_name).replace("_", " ").title()


def find_existing_image(path_without_suffix: Path) -> Path | None:
	for suffix in SUPPORTED_IMAGE_SUFFIXES:
		path = path_without_suffix.with_suffix(suffix)
		if path.exists():
			return path
	return None


def cluster_output_dir(config: ClusterFigureConfig) -> Path:
	return config.base_dir / normalize_dataset_name(config.dataset_name)


def build_source_path(config: ClusterFigureConfig, k_value: int, plot_kind: str) -> Path | None:
	top_tag = format_top_percentage(config.top_percentage)
	stem = cluster_output_dir(config) / f"k_{k_value}_top_{top_tag}_{plot_kind}"
	image_path = find_existing_image(stem)
	if image_path is None:
		print(f"[WARN] Image not found: {stem}.*")
	return image_path


def build_pca_data_path(config: ClusterFigureConfig, k_value: int) -> Path | None:
	top_tag = format_top_percentage(config.top_percentage)
	path = cluster_output_dir(config) / f"k_{k_value}_top_{top_tag}_clusters_pca_2d.csv"
	if path.exists():
		return path
	print(f"[WARN] PCA data not found: {path}")
	return None


def cluster_scores_path(config: ClusterFigureConfig) -> Path:
	top_tag = format_top_percentage(config.top_percentage)
	return cluster_output_dir(config) / f"top_{top_tag}_cluster_scores.csv"


def trim_white_border(image: np.ndarray, padding: int = 6) -> np.ndarray:
	rgb_image = image[..., :3] if image.ndim == 3 else image
	non_white_mask = np.any(rgb_image < 0.985, axis=-1) if rgb_image.ndim == 3 else rgb_image < 0.985
	rows = np.where(non_white_mask.any(axis=1))[0]
	cols = np.where(non_white_mask.any(axis=0))[0]
	if rows.size == 0 or cols.size == 0:
		return image

	row_start = max(int(rows[0]) - padding, 0)
	row_end = min(int(rows[-1]) + padding + 1, image.shape[0])
	col_start = max(int(cols[0]) - padding, 0)
	col_end = min(int(cols[-1]) + padding + 1, image.shape[1])
	return image[row_start:row_end, col_start:col_end]


def draw_image_or_placeholder(
	ax,
	image_path: Path | None,
	title: str,
	image_aspect: str = "equal",
	trim_border: bool = False,
) -> None:
	if title:
		ax.set_title(title, fontsize=SUBPLOT_TITLE_FONTSIZE, fontweight="bold", pad=8)
	ax.set_xticks([])
	ax.set_yticks([])
	for spine in ax.spines.values():
		spine.set_visible(False)

	if image_path is None:
		ax.set_facecolor("white")
		ax.text(
			0.5,
			0.5,
			"Image not found",
			ha="center",
			va="center",
			transform=ax.transAxes,
			fontsize=22,
			color="crimson",
			fontweight="bold",
		)
		return

	image = mpimg.imread(image_path)
	if trim_border:
		image = trim_white_border(image)
	ax.imshow(image, aspect=image_aspect, interpolation="lanczos")


def draw_panel_b_pca(
	ax,
	pca_data_path: Path | None,
	image_path: Path | None = None,
	show_ylabel: bool = True,
):
	if pca_data_path is None or not pca_data_path.exists():
		draw_image_or_placeholder(ax, image_path, "", image_aspect="equal", trim_border=True)
		return None, 0

	pca_df = pd.read_csv(pca_data_path)
	required_columns = {"pc1", "pc2", "cluster"}
	if not required_columns.issubset(pca_df.columns):
		print(f"[WARN] PCA data missing columns: {pca_data_path}")
		draw_image_or_placeholder(ax, image_path, "", image_aspect="equal", trim_border=True)
		return None, 0

	cluster_ids = pca_df["cluster"].astype(int).to_numpy()
	cluster_count = int(cluster_ids.max()) + 1 if cluster_ids.size else 1
	pc1_variance = float(pca_df["pc1_variance"].iloc[0]) if "pc1_variance" in pca_df.columns else 16.4
	pc2_variance = float(pca_df["pc2_variance"].iloc[0]) if "pc2_variance" in pca_df.columns else 8.0
	cluster_cmap = build_cluster_colormap(cluster_count)
	cluster_norm = BoundaryNorm(np.arange(-0.5, cluster_count + 0.5, 1), cluster_cmap.N)
	scatter = ax.scatter(
		pca_df["pc1"],
		pca_df["pc2"],
		c=cluster_ids,
		cmap=cluster_cmap,
		norm=cluster_norm,
		s=34,
		alpha=0.86,
		edgecolors="none",
	)
	ax.set_xlabel(f"PC1 ({pc1_variance:.1f}% variance)", fontsize=38, labelpad=8)
	if show_ylabel:
		ax.set_ylabel(f"PC2 ({pc2_variance:.1f}% variance)", fontsize=38, labelpad=8)
	else:
		ax.tick_params(axis="y", labelleft=False)
	ax.tick_params(axis="both", labelsize=30)
	ax.grid(True, alpha=0.22)
	return scatter, cluster_count


def add_panel_b_k_labels(fig, panel_b_axes, k_values: tuple[int, ...]) -> None:
	for ax, k_value in zip(panel_b_axes, k_values):
		position = ax.get_position()
		fig.text(
			(position.x0 + position.x1) / 2.0,
			position.y1 + 0.012,
			f"K={k_value}",
			ha="center",
			va="bottom",
			fontsize=36,
			fontweight="normal",
		)


def draw_panel_a_elbow_silhouette(ax, config: ClusterFigureConfig) -> None:
	path = cluster_scores_path(config)
	if not path.exists():
		print(f"[WARN] Cluster scores CSV not found: {path}")
		panel_a_path = build_source_path(config, config.panel_a_k, "elbow_silhouette")
		draw_image_or_placeholder(ax, panel_a_path, "", image_aspect="equal")
		return

	scores_df = pd.read_csv(path)
	required_columns = {"k", "inertia", "silhouette_score"}
	if not required_columns.issubset(scores_df.columns):
		print(f"[WARN] Cluster scores CSV missing columns: {path}")
		panel_a_path = build_source_path(config, config.panel_a_k, "elbow_silhouette")
		draw_image_or_placeholder(ax, panel_a_path, "", image_aspect="equal")
		return

	scores_df = scores_df.dropna(subset=["k", "inertia", "silhouette_score"])
	if scores_df.empty:
		draw_image_or_placeholder(ax, None, "")
		return

	ax_inertia = ax
	min_k_for_axis = int(scores_df["k"].min())
	max_k_for_axis = max(16, int(scores_df["k"].max()))
	k_ticks = np.arange(min_k_for_axis, max_k_for_axis + 1, 1)
	inertia_values = scores_df["inertia"] / 10000.0
	inertia_line = ax_inertia.plot(
		scores_df["k"],
		inertia_values,
		color="#1f77b4",
		marker="o",
		linewidth=3.1,
		markersize=9,
		label="WCSS",
	)
	ax_inertia.set_xlabel("Number of clusters (k)", fontsize=AXIS_LABEL_FONTSIZE)
	ax_inertia.set_ylabel("WCSS", color="#1f77b4", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
	ax_inertia.set_xlim(min_k_for_axis - 0.5, max_k_for_axis + 0.5)
	ax_inertia.set_xticks(k_ticks)
	ax_inertia.yaxis.set_label_coords(-0.060, 0.5)
	ax_inertia.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
	ax_inertia.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE, labelcolor="#1f77b4")
	ax_inertia.grid(True, alpha=0.28)
	ax_inertia.text(
		0.0,
		1.025,
		r"$\times 10^4$",
		transform=ax_inertia.transAxes,
		ha="left",
		va="bottom",
		fontsize=SCALE_NOTE_FONTSIZE,
		color="#1f77b4",
	)

	ax_silhouette = ax_inertia.twinx()
	silhouette_line = ax_silhouette.plot(
		scores_df["k"],
		scores_df["silhouette_score"] * 100.0,
		color="#ff7f0e",
		marker="s",
		linewidth=3.2,
		markersize=9,
		label="Silhouette",
	)
	ax_silhouette.set_ylabel("Silhouette score", color="#ff7f0e", fontsize=AXIS_LABEL_FONTSIZE)
	ax_silhouette.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE, labelcolor="#ff7f0e")
	ax_silhouette.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
	ax_silhouette.text(
		1.0,
		1.025,
		r"$\times 10^{-2}$",
		transform=ax_silhouette.transAxes,
		ha="right",
		va="bottom",
		fontsize=SCALE_NOTE_FONTSIZE,
		color="#ff7f0e",
	)


def add_panel_header(fig, row_axes, panel_letter: str, panel_title: str) -> None:
	top = max(ax.get_position().y1 for ax in row_axes)
	fig.text(
		PANEL_LETTER_X,
		top + PANEL_LETTER_Y_OFFSET,
		f"{panel_letter}",
		ha="left",
		va="bottom",
		fontsize=PANEL_LETTER_FONTSIZE,
		fontweight="bold",
	)
	if panel_title:
		fig.text(
			0.5,
			top + PANEL_LETTER_Y_OFFSET,
			panel_title,
			ha="center",
			va="bottom",
			fontsize=36,
			fontweight="bold",
		)


def align_panel_a_with_pca_row(panel_a_ax, panel_b_axes) -> None:
	b_left = min(ax.get_position().x0 for ax in panel_b_axes)
	b_right = max(ax.get_position().x1 for ax in panel_b_axes)
	b_center = (b_left + b_right) / 2.0
	panel_a_width = (b_right - b_left) * PANEL_A_WIDTH_SCALE
	panel_a_ax.set_position(
		[
			b_center - panel_a_width / 2.0,
			0.53,
			panel_a_width,
			0.35,
		]
	)


def set_panel_b_axes_layout(panel_b_axes) -> None:
	axis_count = len(panel_b_axes)
	if axis_count == 0:
		return

	if axis_count == 2:
		width = 0.3
		height = 0.43
		gap = 0.115
		bottom = 0.035
		total_width = axis_count * width + (axis_count - 1) * gap
		left = (1.0 - total_width) / 2.0
	elif axis_count == 3:
		width = 0.27
		height = 0.38
		gap = 0.035
		bottom = 0.035
		total_width = axis_count * width + (axis_count - 1) * gap
		left = (1.0 - total_width) / 2.0
	else:
		width = min(0.88 / axis_count, 0.28)
		height = 0.32
		gap = max((0.88 - axis_count * width) / max(axis_count - 1, 1), 0.025)
		bottom = 0.055
		total_width = axis_count * width + (axis_count - 1) * gap
		left = (1.0 - total_width) / 2.0
	for index, ax in enumerate(panel_b_axes):
		ax.set_position([left + index * (width + gap), bottom, width, height])


def output_prefix(config: ClusterFigureConfig) -> str:
	if config.output_prefix:
		return config.output_prefix
	return f"Figure_1_{dataset_title(config.dataset_name).replace(' ', '_')}_top_{format_top_percentage(config.top_percentage)}"


def save_figure(fig, config: ClusterFigureConfig) -> list[Path]:
	config.output_dir.mkdir(parents=True, exist_ok=True)
	prefix = output_prefix(config)
	output_paths = [
		config.output_dir / f"{prefix}.jpeg",
	]
	for path in output_paths:
		fig.savefig(path, dpi=config.dpi, bbox_inches="tight", facecolor="white")
		print(f"[OK] Cluster figure saved: {path}")
	return output_paths


def create_cluster_figure(config: ClusterFigureConfig) -> list[Path]:
	fig = plt.figure(figsize=(28, 16.5), facecolor="white")
	grid = fig.add_gridspec(2, len(config.k_values), height_ratios=[1.28, 1.0])
	panel_a_ax = fig.add_subplot(grid[0, :])
	panel_b_axes = [fig.add_subplot(grid[1, col_idx]) for col_idx in range(len(config.k_values))]

	draw_panel_a_elbow_silhouette(panel_a_ax, config)
	pca_colorbars = []
	for col_idx, k_value in enumerate(config.k_values):
		pca_path = build_source_path(config, k_value, "clusters_pca_2d")
		pca_data_path = build_pca_data_path(config, k_value) if config.use_pca_csv else None
		scatter, cluster_count = draw_panel_b_pca(
			panel_b_axes[col_idx],
			pca_data_path,
			pca_path,
			show_ylabel=True,
		)
		if scatter is not None:
			pca_colorbars.append((panel_b_axes[col_idx], scatter, cluster_count))

	fig.subplots_adjust(left=0.05, right=0.97, top=0.92, bottom=0.055, wspace=0.04, hspace=0.07)
	set_panel_b_axes_layout(panel_b_axes)
	align_panel_a_with_pca_row(panel_a_ax, panel_b_axes)
	add_panel_b_k_labels(fig, panel_b_axes, config.k_values)
	for colorbar_index, (ax, scatter, cluster_count) in enumerate(pca_colorbars):
		position = ax.get_position()
		colorbar_ax = fig.add_axes([position.x1 + 0.008, position.y0, 0.011, position.height])
		colorbar = fig.colorbar(scatter, cax=colorbar_ax, ticks=np.arange(cluster_count))
		colorbar.ax.tick_params(labelsize=26)
		if colorbar_index == len(pca_colorbars) - 1:
			colorbar.set_label("Cluster", fontsize=32)
	add_panel_header(fig, [panel_a_ax], "A", "")
	add_panel_header(fig, panel_b_axes, "B", "")
	return save_figure(fig, config)


def parse_args() -> ClusterFigureConfig:
	parser = argparse.ArgumentParser(description="Create publication-ready cluster Figure 1 from saved clustering plots.")
	parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Dataset folder name under outputs/clustering")
	parser.add_argument("--top-percentage", type=int, default=DEFAULT_TOP_PERCENTAGE, help="Feature selection percentage, e.g. 20, 40, 60")
	parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES, help="K values to display, e.g. 2 8 10")
	parser.add_argument("--panel-a-k", type=int, default=DEFAULT_PANEL_A_K, help="K value used for the single Panel A elbow/silhouette image")
	parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR, help="Base clustering output directory")
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for final figure outputs")
	parser.add_argument("--output-prefix", default=None, help="Output filename without extension")
	parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
	parser.add_argument("--use-pca-csv", action="store_true", help="Redraw PCA panels from saved CSV data instead of embedding saved PNG images")
	args = parser.parse_args()

	return ClusterFigureConfig(
		dataset_name=normalize_dataset_name(args.dataset_name),
		top_percentage=args.top_percentage,
		k_values=tuple(args.k_values),
		panel_a_k=args.panel_a_k,
		use_pca_csv=args.use_pca_csv,
		base_dir=args.base_dir,
		output_dir=args.output_dir,
		output_prefix=args.output_prefix,
		dpi=args.dpi,
	)


def main() -> None:
	config = parse_args()
	create_cluster_figure(config)


if __name__ == "__main__":
	main()
