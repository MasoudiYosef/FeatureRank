import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


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
    required_columns = {"true_label", "positive_class_score"}
    missing_columns = required_columns - set(predictions_df.columns)
    if missing_columns:
        raise ValueError(
            f"Prediction dosyasinda eksik kolon var: {prediction_path}, {sorted(missing_columns)}"
        )
    return predictions_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Binary classification precision-recall paneli olusturur."
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
        default="outputs/Classification/classification_precision_recall_panel.png",
        help="Panel PNG cikti yolu.",
    )
    args = parser.parse_args()

    plt.figure(figsize=(7.5, 6))
    for item_text in args.items:
        dataset_folder, feature_percent, label = parse_item(item_text)
        predictions_df = load_predictions(dataset_folder, feature_percent)
        y_true = predictions_df["true_label"].to_numpy(dtype=int)
        y_score = predictions_df["positive_class_score"].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2:
            print(f"[WARN] {label} atlandi: test setinde tek sinif var.")
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        average_precision = average_precision_score(y_true, y_score)
        plt.plot(
            recall,
            precision,
            linewidth=1.6,
            label=f"{label} (AP={average_precision:.3f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[OK] Precision-recall panel: {output_path}")


if __name__ == "__main__":
    main()
