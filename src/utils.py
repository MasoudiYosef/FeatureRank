from pathlib import Path
import json
import numpy as np

# Otomatik olarak tüm parent klasörleri yaratır
def ensure_dir(path: Path) -> None: 
    """
    Verilen klasör yoksa oluşturur.
    """
    path.mkdir(parents=True, exist_ok=True)

#Python dictionary'sini JSON dosyası olarak kaydeder
def save_json(data: dict, path: Path) -> None:
    """
    Dictionary verisini JSON olarak kaydeder.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

#Multiclass weighted average hesaplar
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


def normalize_id_column(id_column: str | None) -> str | None:
    """
    ID column string'ini işle ("none" → None)
    """
    if id_column and id_column.lower() in {"none", "null", "-", ""}:
        return None
    return id_column


def format_feature_percent_tag(feature_percent: float) -> str:
    """
    % sayısını tag'a çevir (50.0 → "50", 50.5 → "50_5")
    """
    if float(feature_percent).is_integer():
        return str(int(feature_percent))
    return str(feature_percent).replace(".", "_")


def parse_hidden_units(units_text: str) -> tuple[int, ...]:
    """
    String'den hidden unit list'i parse et ("32,16" → (32, 16))
    """
    parts = [p.strip() for p in units_text.split(",") if p.strip()]
    if not parts:
        raise ValueError("classifier-hidden-units bos olamaz. Ornek: 128,64")
    units = tuple(int(p) for p in parts)
    if any(u <= 0 for u in units):
        raise ValueError("classifier-hidden-units pozitif tam sayilar olmali.")
    return units


def parse_dropout_rates(dropout_text: str | None, layer_count: int) -> tuple[float, ...] | None:
    """
    Dropout oranlarını string'den parse et ("0.2,0.3" → (0.2, 0.3))
    """
    if dropout_text is None:
        return None
    text = dropout_text.strip()
    if text == "":
        return None
    parts = [p.strip() for p in text.split(",") if p.strip()]
    dropouts = tuple(float(p) for p in parts)
    if len(dropouts) != layer_count:
        raise ValueError("classifier-dropout-rates uzunlugu, hidden katman sayisi ile ayni olmali.")
    if any((d < 0.0 or d >= 1.0) for d in dropouts):
        raise ValueError("dropout oranlari [0.0, 1.0) araliginda olmali.")
    return dropouts


def unpack_processed_arrays(processed: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocessed dict'ten X_train, X_test, y_train, y_test çıkart
    """
    X_train = processed["X_train_scaled"]
    X_test = processed["X_test_scaled"]
    y_train = processed["y_train"].to_numpy().astype(np.int32)
    y_test = processed["y_test"].to_numpy().astype(np.int32)
    return X_train, X_test, y_train, y_test