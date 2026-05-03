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
    One-vs-rest klasorlerinden accuracy okuyup weighted-average hesaplar.
    Klasor yapisi: outputs/autoencoder/{dataset_folder}/{class_label}_{dataset_folder}/metrics/{metric_filename}
    
    Hesaplama: weighted_accuracy = sum(accuracy_i * class_count_i) / sum(class_count_i)
    """
    accuracy_list: list[float] = []
    class_counts_list: list[int] = []

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
        
        accuracy = float(metrics["test_accuracy"])
        accuracy_list.append(accuracy)
        
        # Multiclass için class_counts var ise, weighted average yap
        if "class_counts" in metrics:
            # JSON'da key string olabilir, class_label string veya int olabilir
            count_dict = metrics["class_counts"]
            # Önce string key ile dene
            count = count_dict.get(str(class_label))
            # Eğer class_label integer ise int key ile de dene
            if count is None and isinstance(class_label, int):
                count = count_dict.get(class_label)
            if count is not None:
                class_counts_list.append(int(count))
            else:
                class_counts_list.append(1)
        else:
            # Backward compatibility: binary model veya eski format
            class_counts_list.append(1)

    if not accuracy_list:
        raise ValueError("Macro-average icin en az bir accuracy degeri olmali.")

    # Weighted average: sum(accuracy * count) / sum(count)
    total_samples = sum(class_counts_list)
    
    # Eğer hiç class_counts bulunamadıysa (eski metric dosyaları), macro average yap
    if total_samples == len(accuracy_list):
        # Fallback: tüm class'lar eşit ağırlık (basit average)
        return sum(accuracy_list) / len(accuracy_list)
    
    if total_samples == 0:
        raise ValueError(
            f"Class counts toplami sifir olamaz. Veri problemi var.\n"
            f"  - dataset_folder: {dataset_folder}\n"
            f"  - class_labels: {class_labels}\n"
            f"  - accuracy_list: {accuracy_list}\n"
            f"  - class_counts_list: {class_counts_list}"
        )
    
    total_correct = sum(acc * cnt for acc, cnt in zip(accuracy_list, class_counts_list))
    return total_correct / total_samples