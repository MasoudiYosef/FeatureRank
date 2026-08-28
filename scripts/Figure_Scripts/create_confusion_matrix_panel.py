import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def parse_item(item_text: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in item_text.split(":") if part.strip()]
    if len(parts) not in {2, 3}:
        raise ValueError(
            "Her item 'dataset_folder:percent[:label]' formatinda olmali. Ornek: pid_data:80:PID"
        )

    dataset_folder = Path(parts[0]).stem
    feature_percent = parts[1].replace(".", "_")
    label = parts[2] if len(parts) == 3 else dataset_folder
    return dataset_folder, feature_percent, label


def load_predictions(dataset_folder: str, feature_percent: str) -> pd.DataFrame:
    prediction_path = (
        Path("outputs")
        / "Classification"
        / dataset_folder
        / f"top_{feature_percent}_classification_predictions.csv"
    )
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction dosyasi bulunamadi: {prediction_path}\n"
            "Once ilgili binary classification komutunu tekrar calistir."
        )

    predictions_df = pd.read_csv(prediction_path)
    required_columns = {"true_label", "predicted_label"}
    missing_columns = required_columns - set(predictions_df.columns)
    if missing_columns:
        raise ValueError(
            f"Prediction dosyasinda eksik kolon var: {prediction_path}, {sorted(missing_columns)}"
        )
    return predictions_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binary classification confusion matrix paneli olusturur."
    )
    parser.add_argument(
        "--items",
        nargs="+",
        required=True,
        help="Dataset listesi. Format: dataset_folder:percent[:label]. Ornek: pid_data:80:PID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/Classification/classification_confusion_matrix_panel.png",
        help="Panel PNG cikti yolu.",
    )
    args = parser.parse_args()

    items = [parse_item(item_text) for item_text in args.items]
    panel_count = len(items)
    cols = math.ceil(math.sqrt(panel_count))
    rows = math.ceil(panel_count / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
    axes_array = np.asarray(axes).reshape(-1)

    for ax, (dataset_folder, feature_percent, label) in zip(axes_array, items):
        predictions_df = load_predictions(dataset_folder, feature_percent)
        y_true = predictions_df["true_label"].to_numpy(dtype=int)
        y_pred = predictions_df["predicted_label"].to_numpy(dtype=int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{label} (top {feature_percent.replace('_', '.')}%)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0", "1"])
        ax.set_yticklabels(["0", "1"])

        threshold = cm.max() / 2.0 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                )

    for ax in axes_array[panel_count:]:
        ax.axis("off")

    fig.colorbar(im, ax=axes_array[:panel_count], fraction=0.025, pad=0.02)
    fig.suptitle("Binary Classification Confusion Matrix Panel", fontsize=14)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Confusion matrix panel: {output_path}")


if __name__ == "__main__":
    main()
