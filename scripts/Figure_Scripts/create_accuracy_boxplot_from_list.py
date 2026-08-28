import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Accuracy degerlerini buraya gir.
# Ornek:
# ACCURACY_VALUES = [0.94, 0.96, 0.95, 0.98, 1.00]
ACCURACY_VALUES = [
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
    0.9051724137931034,
]

OUTPUT_DIR = Path("outputs") / "FIGURES"
OUTPUT_NAME = "custom_accuracy_boxplot"
TITLE = "Accuracy Boxplot"
Y_LABEL = "Accuracy"
X_LABEL = "Accuracy"

BOX_COLOR = "#FFFFFF"
POINT_COLOR = "#1f77b4"
MEDIAN_COLOR = "#ff7f0e"
MEAN_COLOR = "#2ca02c"
EDGE_COLOR = "#000000"

Y_MIN = 0.0
Y_MAX = 1.02
RANDOM_SEED = 42


def parse_values(values_text: str | None) -> list[float]:
    if not values_text:
        return ACCURACY_VALUES
    return [float(value.strip()) for value in values_text.split(",") if value.strip()]


def create_accuracy_boxplot(
    values: list[float],
    output_dir: Path,
    output_name: str,
    title: str,
    box_color: str,
    point_color: str,
    median_color: str,
    mean_color: str,
    y_min: float,
    y_max: float,
) -> list[Path]:
    if not values:
        raise ValueError(
            "Accuracy listesi bos. ACCURACY_VALUES icine deger gir veya --values kullan."
        )

    values_array = np.asarray(values, dtype=float)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.boxplot(
        values_array,
        tick_labels=[X_LABEL],
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "^",
            "markerfacecolor": mean_color,
            "markeredgecolor": mean_color,
            "markersize": 8,
        },
        boxprops={"facecolor": box_color, "edgecolor": EDGE_COLOR, "linewidth": 1.6},
        medianprops={"color": median_color, "linewidth": 1.8},
        whiskerprops={"color": EDGE_COLOR, "linewidth": 1.5},
        capprops={"color": EDGE_COLOR, "linewidth": 1.5},
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": EDGE_COLOR,
            "markersize": 6,
        },
    )

    rng = np.random.default_rng(RANDOM_SEED)
    x_positions = 1 + rng.uniform(-0.08, 0.08, size=len(values_array))
    ax.scatter(
        x_positions,
        values_array,
        s=36,
        color=point_color,
        alpha=0.78,
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
    )

    ax.set_title(title, fontsize=20, pad=12)
    ax.set_ylabel(Y_LABEL, fontsize=16)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(axis="y", alpha=0.3)

    mean_value = float(np.mean(values_array))
    std_value = float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0
    ax.text(
        0.02,
        0.98,
        f"mean={mean_value:.4f}\nstd={std_value:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )

    fig.tight_layout()
    output_paths = [
        output_dir / f"{output_name}.png",
        output_dir / f"{output_name}.pdf",
    ]
    for path in output_paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[OK] Boxplot saved: {path}")
    plt.close(fig)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an accuracy boxplot from a manually provided accuracy list."
    )
    parser.add_argument(
        "--values",
        default=None,
        help="Virgulle ayrilmis accuracy listesi. Ornek: 0.94,0.95,1.0",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-name", default=OUTPUT_NAME)
    parser.add_argument("--title", default=TITLE)
    parser.add_argument("--box-color", default=BOX_COLOR)
    parser.add_argument("--point-color", default=POINT_COLOR)
    parser.add_argument("--median-color", default=MEDIAN_COLOR)
    parser.add_argument("--mean-color", default=MEAN_COLOR)
    parser.add_argument("--y-min", type=float, default=Y_MIN)
    parser.add_argument("--y-max", type=float, default=Y_MAX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = parse_values(args.values)
    create_accuracy_boxplot(
        values=values,
        output_dir=args.output_dir,
        output_name=args.output_name,
        title=args.title,
        box_color=args.box_color,
        point_color=args.point_color,
        median_color=args.median_color,
        mean_color=args.mean_color,
        y_min=args.y_min,
        y_max=args.y_max,
    )


if __name__ == "__main__":
    main()
