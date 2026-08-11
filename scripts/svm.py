from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. AYARLAR
# ============================================================

DATA_FILENAME = "dorothea_data.csv"
LABEL_FILENAME = "dorothea_label.csv"

OUTPUT_FILENAME = "svm_all_predicted_labels.csv"

# Dorothea veri setindeki toplam özellik sayısı
N_FEATURES = 100000

# Verinin yüzde 20'si test için kullanılacak
TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# 2. PROJE VE DOSYA YOLLARI
# ============================================================

# Script:
# Feature_Ranking_Project/scripts/svm.py
#
# Proje kökü:
# Feature_Ranking_Project/

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent


def find_file(filename: str) -> Path:
    """
    Dosyayı önce proje kökünde, ardından tüm alt klasörlerde arar.
    """

    direct_path = PROJECT_ROOT / filename

    if direct_path.exists():
        return direct_path

    matches = list(PROJECT_ROOT.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"\n'{filename}' dosyası bulunamadı.\n"
            f"Aranan proje klasörü:\n{PROJECT_ROOT}\n\n"
            f"Dosyayı proje klasörüne veya bir alt klasöre koyun.\n\n"
            f"Örnek:\n"
            f"{PROJECT_ROOT / 'data' / filename}"
        )

    if len(matches) > 1:
        print(
            f"\nUyarı: '{filename}' adına sahip birden fazla dosya bulundu."
        )
        print("Kullanılacak dosya:", matches[0])

    return matches[0]


DATA_FILE = find_file(DATA_FILENAME)
LABEL_FILE = find_file(LABEL_FILENAME)
OUTPUT_FILE = PROJECT_ROOT / OUTPUT_FILENAME

print("=" * 60)
print("DOSYA YOLLARI")
print("=" * 60)
print("Data dosyası :", DATA_FILE)
print("Label dosyası:", LABEL_FILE)
print("Çıktı dosyası:", OUTPUT_FILE)


# ============================================================
# 3. DOROTHEA SPARSE DATA OKUYUCU
# ============================================================

def read_dorothea_sparse_data(
    file_path: Path,
    n_features: int
) -> csr_matrix:
    """
    Dorothea sparse veri formatını CSR sparse matrise dönüştürür.

    Dosyanın her satırında değeri 1 olan özelliklerin indeksleri
    bulunur.

    Örnek satır:
    12 45 891 2456

    Bu örnekte 12, 45, 891 ve 2456 numaralı özellikler 1,
    diğer özellikler 0 kabul edilir.

    Dorothea özellik indeksleri 1'den başlar.
    Python indeksleri 0'dan başladığı için her indeksten 1 çıkarılır.
    """

    row_indices = []
    column_indices = []
    values = []

    number_of_rows = 0
    max_feature_index = 0

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
        errors="ignore"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            # Boş satır varsa boş bir örnek olarak korunur.
            if not line:
                number_of_rows += 1
                continue

            # Farklı ayraç ihtimallerini boşluğa dönüştürür.
            cleaned_line = (
                line
                .replace(",", " ")
                .replace(";", " ")
                .replace("\t", " ")
            )

            tokens = cleaned_line.split()

            for token in tokens:

                try:
                    feature_index = int(float(token))

                except ValueError as error:
                    raise ValueError(
                        f"\nGeçersiz özellik indeksi bulundu.\n"
                        f"Dosya satırı: {line_number}\n"
                        f"Geçersiz değer: {token}"
                    ) from error

                if feature_index <= 0:
                    raise ValueError(
                        f"\nÖzellik indekslerinin 1 veya daha büyük "
                        f"olması gerekiyor.\n"
                        f"Dosya satırı: {line_number}\n"
                        f"Bulunan indeks: {feature_index}"
                    )

                if feature_index > n_features:
                    raise ValueError(
                        f"\nDosyada N_FEATURES değerinden büyük bir "
                        f"özellik indeksi bulundu.\n"
                        f"Dosya satırı: {line_number}\n"
                        f"Bulunan indeks: {feature_index}\n"
                        f"N_FEATURES: {n_features}\n\n"
                        f"N_FEATURES değerini en az "
                        f"{feature_index} yapın."
                    )

                max_feature_index = max(
                    max_feature_index,
                    feature_index
                )

                # 1 tabanlı indeksi 0 tabanlı indekse dönüştürür.
                zero_based_index = feature_index - 1

                row_indices.append(number_of_rows)
                column_indices.append(zero_based_index)
                values.append(1.0)

            number_of_rows += 1

    if number_of_rows == 0:
        raise ValueError("Data dosyası boş.")

    X_sparse = csr_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (
                np.asarray(row_indices, dtype=np.int32),
                np.asarray(column_indices, dtype=np.int32)
            )
        ),
        shape=(number_of_rows, n_features),
        dtype=np.float32
    )

    # Aynı özellik bir satırda birden fazla yazılmışsa değeri 1 yapar.
    X_sparse.sum_duplicates()
    X_sparse.data[:] = 1.0
    X_sparse.eliminate_zeros()
    X_sparse.sort_indices()

    print("\n" + "=" * 60)
    print("DATA BİLGİLERİ")
    print("=" * 60)
    print("Örnek sayısı                  :", number_of_rows)
    print("Toplam özellik sayısı         :", n_features)
    print("En büyük özellik indeksi      :", max_feature_index)
    print("Sıfır olmayan özellik sayısı  :", X_sparse.nnz)
    print("Sparse matris boyutu          :", X_sparse.shape)

    return X_sparse


# ============================================================
# 4. LABEL DOSYASI OKUYUCU
# ============================================================

def read_labels(file_path: Path) -> np.ndarray:
    """
    Label dosyasındaki tüm değerleri okur.

    Desteklenen örnekler:

    Her satırda bir label:
    -1
    1
    -1

    Aynı satırda boşlukla ayrılmış label:
    -1 1 -1 1
    """

    labels = []

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
        errors="ignore"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            cleaned_line = (
                line
                .replace(",", " ")
                .replace(";", " ")
                .replace("\t", " ")
            )

            tokens = cleaned_line.split()

            for token in tokens:

                try:
                    label_value = float(token)

                except ValueError as error:
                    raise ValueError(
                        f"\nGeçersiz label değeri bulundu.\n"
                        f"Dosya satırı: {line_number}\n"
                        f"Geçersiz değer: {token}"
                    ) from error

                # 1.0 veya -1.0 değerlerini integer olarak tutar.
                if label_value.is_integer():
                    label_value = int(label_value)

                labels.append(label_value)

    if not labels:
        raise ValueError("Label dosyası boş.")

    return np.asarray(labels)


# ============================================================
# 5. DATA VE LABEL YÜKLEME
# ============================================================

print("\nData yükleniyor...")

X = read_dorothea_sparse_data(
    file_path=DATA_FILE,
    n_features=N_FEATURES
)

print("\nLabel yükleniyor...")

y = read_labels(LABEL_FILE)


# ============================================================
# 6. DATA VE LABEL KONTROLLERİ
# ============================================================

if X.shape[0] != len(y):
    raise ValueError(
        "\nData ve label örnek sayıları eşit değil.\n"
        f"Data örnek sayısı : {X.shape[0]}\n"
        f"Label örnek sayısı: {len(y)}"
    )

unique_labels, label_counts = np.unique(
    y,
    return_counts=True
)

if len(unique_labels) < 2:
    raise ValueError(
        "SVM eğitimi için en az iki farklı label gereklidir."
    )

print("\n" + "=" * 60)
print("LABEL BİLGİLERİ")
print("=" * 60)
print("Toplam label sayısı:", len(y))
print("Farklı sınıf sayısı:", len(unique_labels))

print("\nSınıf dağılımı:")

for label, count in zip(unique_labels, label_counts):
    print(f"Label {label}: {count} örnek")


# ============================================================
# 7. ORİJİNAL SATIR İNDEKSLERİ
# ============================================================

# Tahminleri sonradan orijinal veri sırasına yerleştirmek için
# her örneğe bir satır indeksi atanır.
original_indices = np.arange(X.shape[0])


# ============================================================
# 8. EĞİTİM VE TEST AYRIMI
# ============================================================

(
    X_train,
    X_test,
    y_train,
    y_test,
    train_indices,
    test_indices
) = train_test_split(
    X,
    y,
    original_indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n" + "=" * 60)
print("EĞİTİM-TEST AYRIMI")
print("=" * 60)
print("Toplam örnek sayısı :", X.shape[0])
print("Eğitim örnek sayısı:", X_train.shape[0])
print("Test örnek sayısı   :", X_test.shape[0])


# ============================================================
# 9. LINEAR SVM MODELİ
# ============================================================

# 1151 örnek ve 100.000 özellik gibi yüksek boyutlu sparse
# veriler için RBF SVC yerine LinearSVC kullanılır.

svm_model = LinearSVC(
    C=1.0,
    class_weight=None,
    dual="auto",
    max_iter=20000,
    random_state=RANDOM_STATE
)


# ============================================================
# 10. MODELİ EĞİTME
# ============================================================

print("\nSVM modeli eğitiliyor...")

svm_model.fit(
    X_train,
    y_train
)

print("SVM modeli başarıyla eğitildi.")


# ============================================================
# 11. EĞİTİM VE TEST TAHMİNLERİ
# ============================================================

# Eğitim kümesindeki tüm örnekler için tahmin
predicted_train = svm_model.predict(X_train)

# Test kümesindeki tüm örnekler için tahmin
predicted_test = svm_model.predict(X_test)

print("\n" + "=" * 60)
print("TAHMİN SAYILARI")
print("=" * 60)
print("Eğitim tahmin sayısı:", len(predicted_train))
print("Test tahmin sayısı   :", len(predicted_test))
print(
    "Tahminlerin toplamı :",
    len(predicted_train) + len(predicted_test)
)


# ============================================================
# 12. TÜM TAHMİNLERİ ORİJİNAL SIRAYA YERLEŞTİRME
# ============================================================

# Toplam örnek sayısı kadar boş bir dizi oluşturulur.
all_predictions = np.empty(
    X.shape[0],
    dtype=predicted_train.dtype
)

# Eğitim örneklerinin tahminleri orijinal satırlarına yerleştirilir.
all_predictions[train_indices] = predicted_train

# Test örneklerinin tahminleri orijinal satırlarına yerleştirilir.
all_predictions[test_indices] = predicted_test


# ============================================================
# 13. TAHMİN SAYISI KONTROLÜ
# ============================================================

expected_prediction_count = X.shape[0]
actual_prediction_count = len(all_predictions)

if actual_prediction_count != expected_prediction_count:
    raise ValueError(
        "\nToplam tahmin sayısı hatalı.\n"
        f"Beklenen tahmin sayısı: {expected_prediction_count}\n"
        f"Bulunan tahmin sayısı: {actual_prediction_count}"
    )

# Tüm satırlara gerçekten değer atanıp atanmadığını kontrol eder.
covered_indices = np.concatenate(
    [train_indices, test_indices]
)

if len(np.unique(covered_indices)) != X.shape[0]:
    raise ValueError(
        "Eğitim ve test indeksleri tüm örnekleri kapsamıyor."
    )

print("\nToplam tahmin sayısı:", actual_prediction_count)


# ============================================================
# 14. TÜM TAHMİNLERİ TEK CSV DOSYASINA KAYDETME
# ============================================================

# CSV dosyasında yalnızca predicted_label sütunu bulunur.
all_predictions_dataframe = pd.DataFrame({
    "predicted_label": all_predictions
})

all_predictions_dataframe.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("CSV DOSYASI OLUŞTURULDU")
print("=" * 60)
print("Dosya:", OUTPUT_FILE)
print("Satır sayısı:", len(all_predictions_dataframe))
print("Sütun adı: predicted_label")


# ============================================================
# 15. EĞİTİM PERFORMANSI
# ============================================================

train_accuracy = accuracy_score(
    y_train,
    predicted_train
)

print("\n" + "=" * 60)
print("EĞİTİM SONUÇLARI")
print("=" * 60)

print(f"Eğitim Accuracy: {train_accuracy:.4f}")

print("\nEğitim Confusion Matrix:")
print(
    confusion_matrix(
        y_train,
        predicted_train
    )
)

print("\nEğitim Classification Report:")
print(
    classification_report(
        y_train,
        predicted_train,
        zero_division=0
    )
)


# ============================================================
# 16. TEST PERFORMANSI
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    predicted_test
)

print("\n" + "=" * 60)
print("TEST SONUÇLARI")
print("=" * 60)

print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nTest Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predicted_test
    )
)

print("\nTest Classification Report:")
print(
    classification_report(
        y_test,
        predicted_test,
        zero_division=0
    )
)


# ============================================================
# 17. SON KONTROL
# ============================================================

saved_predictions = pd.read_csv(OUTPUT_FILE)

if len(saved_predictions) != X.shape[0]:
    raise ValueError(
        "\nKaydedilen CSV dosyasındaki tahmin sayısı hatalı.\n"
        f"Beklenen: {X.shape[0]}\n"
        f"Kaydedilen: {len(saved_predictions)}"
    )

print("\n" + "=" * 60)
print("İŞLEM TAMAMLANDI")
print("=" * 60)
print(
    f"Toplam {len(saved_predictions)} adet tahmin "
    f"tek CSV dosyasına kaydedildi."
)
print("Çıktı dosyası:")
print(OUTPUT_FILE)