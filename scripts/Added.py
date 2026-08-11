from pathlib import Path
import csv


# ============================================================
# AYARLAR
# ============================================================

MAIN_FILE_NAME = "dorothea_data.csv"
PREDICTION_FILE_NAME = "svm_all_predicted_labels.csv"
OUTPUT_FILE_NAME = "dorothea_data_with_prediction.csv"

# Ana dosyadaki değerler boşlukla ayrıldığı için:
OUTPUT_SEPARATOR = " "


# ============================================================
# PROJE KLASÖRÜ
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent


# ============================================================
# DOSYA BULMA
# ============================================================

def find_file(filename: str) -> Path:
    direct_path = PROJECT_ROOT / filename

    if direct_path.exists():
        return direct_path

    matches = list(PROJECT_ROOT.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"'{filename}' dosyası bulunamadı.\n"
            f"Aranan klasör: {PROJECT_ROOT}"
        )

    if len(matches) > 1:
        print(f"Uyarı: Birden fazla '{filename}' bulundu.")
        print("Kullanılan dosya:", matches[0])

    return matches[0]


MAIN_FILE = find_file(MAIN_FILE_NAME)
PREDICTION_FILE = find_file(PREDICTION_FILE_NAME)
OUTPUT_FILE = PROJECT_ROOT / OUTPUT_FILE_NAME


# ============================================================
# TAHMİNLERİ OKUMA
# ============================================================

def read_predictions(file_path: Path) -> list[str]:
    """
    Tahmin dosyasında header olmadığını varsayar.

    Örnek:
    -1
    1
    -1
    """

    predictions = []

    with file_path.open(
        mode="r",
        encoding="utf-8-sig",
        errors="ignore",
        newline=""
    ) as file:

        reader = csv.reader(file)

        for row_number, row in enumerate(reader, start=1):

            if not row:
                continue

            value = row[0].strip()

            if not value:
                continue

            try:
                numeric_value = float(value)
            except ValueError as error:
                raise ValueError(
                    f"Tahmin dosyasında sayısal olmayan değer bulundu.\n"
                    f"Satır: {row_number}\n"
                    f"Değer: {value}\n\n"
                    "Tahmin dosyasında header bulunmamalıdır."
                ) from error

            if numeric_value.is_integer():
                value = str(int(numeric_value))
            else:
                value = str(numeric_value)

            predictions.append(value)

    if not predictions:
        raise ValueError("Tahmin dosyasında değer bulunamadı.")

    return predictions


predictions = read_predictions(PREDICTION_FILE)


# ============================================================
# ANA DOSYAYI OKUMA
# ============================================================

with MAIN_FILE.open(
    mode="r",
    encoding="utf-8-sig",
    errors="ignore"
) as file:
    main_lines = file.readlines()


print("Ana dosya satır sayısı:", len(main_lines))
print("Tahmin sayısı:", len(predictions))


# ============================================================
# SATIR SAYISI KONTROLÜ
# ============================================================

if len(main_lines) != len(predictions):
    raise ValueError(
        "\nAna dosya ile tahmin dosyasının satır sayıları eşit değil.\n"
        f"Ana dosya satır sayısı: {len(main_lines)}\n"
        f"Tahmin sayısı         : {len(predictions)}"
    )


# ============================================================
# TAHMİNLERİ YENİ SÜTUN OLARAK EKLEME
# ============================================================

with OUTPUT_FILE.open(
    mode="w",
    encoding="utf-8",
    newline=""
) as output_file:

    for line, prediction in zip(main_lines, predictions):

        original_line = line.rstrip("\r\n").rstrip()

        if original_line:
            output_file.write(
                original_line
                + OUTPUT_SEPARATOR
                + prediction
                + "\n"
            )
        else:
            output_file.write(prediction + "\n")


# ============================================================
# SONUÇ
# ============================================================

print("\nİşlem tamamlandı.")
print("Eklenen tahmin sayısı:", len(predictions))
print("Yeni dosya:", OUTPUT_FILE)