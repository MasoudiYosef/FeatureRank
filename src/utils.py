""""" Info: Bu dosyalar, projenin farklı aşamalarında kullanılan temel bileşenleri içerir. Her dosya, belirli bir görevi yerine getirmek için tasarlanmıştır ve projenin genel yapısını oluşturur."""

from pathlib import Path
import json


def ensure_dir(path: Path) -> None:
    """
    Verilen klasör yoksa oluşturur.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    """
    Dictionary verisini JSON olarak kaydeder.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def compute_multiclass_macro_accuracy(
    dataset_folder: str,
    class_labels: list,
    metric_filename: str = "ORG_test_metrics.json",
) -> float:
    """
    One-vs-rest klasorlerinden accuracy okuyup macro-average hesaplar.
    Klasor yapisi: outputs/autoencoder/{dataset_folder}/{class_label}_{dataset_folder}/metrics/{metric_filename}
    """
    accuracy_list: list[float] = []

    for class_label in class_labels:
        binary_dataset_folder = f"{class_label}_{dataset_folder}"
        metrics_path = (
            Path("outputs")
            / "autoencoder"
            / dataset_folder
            / binary_dataset_folder
            / "metrics"
            / metric_filename
        )
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metric dosyasi bulunamadi: {metrics_path}")

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        accuracy_list.append(float(metrics["test_accuracy"]))

    if not accuracy_list:
        raise ValueError("Macro-average icin en az bir accuracy degeri olmali.")

    return sum(accuracy_list) / len(accuracy_list)