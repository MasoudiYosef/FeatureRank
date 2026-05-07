""""" Info: Data loader dosyası, veri setini yüklemek ve temel veri kontrolü yapmak için kullanılır.
"""

from pathlib import Path
from io import StringIO
import re

import pandas as pd
from pandas.errors import ParserError
from src.config import get_data


def _read_txt_table(path: Path) -> pd.DataFrame:
    for sep, kwargs in [
        ("\t", {"header": None}),
        (r"\s+", {"header": None, "engine": "python"}),
        (",", {"header": None}),
    ]:
        try:
            df = pd.read_csv(path, sep=sep, **kwargs)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    raise ValueError(f"TXT dosyasi okunamadi: {path}")


def convert_txt_dataset_to_csv(dataset_name: str) -> str:
    if not dataset_name.lower().endswith(".txt"):
        return dataset_name if dataset_name.lower().endswith(".csv") else f"{dataset_name}.csv"

    raw_dir = Path("data") / "raw"
    data_txt = raw_dir / dataset_name
    if not data_txt.exists():
        raise FileNotFoundError(f"Data txt dosyasi bulunamadi: {data_txt}")

    if "_data" not in data_txt.stem:
        raise ValueError("TXT data dosyasi adinda '_data' olmasi gerekiyor. Ornek: breast_cancer_data2.txt")

    label_txt_name = data_txt.name.replace("_data", "_label", 1)
    label_txt = raw_dir / label_txt_name
    if not label_txt.exists():
        raise FileNotFoundError(f"Label txt dosyasi bulunamadi: {label_txt}")

    data_df = _read_txt_table(data_txt)
    label_df = pd.read_csv(label_txt, header=None)

    data_stem = data_txt.stem if data_txt.stem.endswith("_data") else f"{data_txt.stem}_data"
    data_csv = raw_dir / f"{data_stem}.csv"
    label_csv = raw_dir / data_csv.name.replace("_data.csv", "_label.csv")

    data_df.to_csv(data_csv, index=False, header=False)
    label_df.to_csv(label_csv, index=False, header=False)

    print(f"[INFO] TXT -> CSV donusturuldu: {data_csv.name}, {label_csv.name}")
    return data_csv.name


def _normalize_csv_to_comma(path, df: pd.DataFrame) -> None:
    # Ham dosyayı tek standarda çeker: ',' delimiter ve '.' decimal.
    df.to_csv(path, index=False)


def _are_all_columns_numeric_like(columns) -> bool:
    for col in columns:
        text = str(col).strip().replace(",", ".")
        try:
            float(text)
        except ValueError:
            return False
    return True


def _build_label_filename(dataset_name: str) -> str:
    if not dataset_name.endswith("_data.csv"):
        raise ValueError(
            "Raw dataset adı 'dataset_name_data.csv' formatında olmalı. "
            f"Gelen: {dataset_name}"
        )
    return dataset_name.replace("_data.csv", "_label.csv")


def _read_csv_with_scientific_comma_fix(path: Path, is_feature_file: bool = False) -> pd.DataFrame:
    """
    Bazi ham CSV dosyalarinda bilimsel gosterimde ondalik virgul kullaniliyor
    (ornek: 5,00E-04). Bu durum delimiter olan ',' ile cakisip parser hatasi
    uretebiliyor. Yalnizca bu paterni 5.00E-04 formatina cevirip tekrar parse eder.
    """
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    fixed_text = re.sub(r"(?<=\d),(?=\d+E[+-]?\d+)", ".", raw_text, flags=re.IGNORECASE)

    df = pd.read_csv(StringIO(fixed_text), header=None)
    if is_feature_file:
        df.columns = [f"feature_{i+1}" for i in range(df.shape[1])]
    return df


def _read_csv_flexible(path, is_feature_file: bool = False) -> pd.DataFrame:
    try:
        if is_feature_file:
            df = pd.read_csv(path, header=None)
            df.columns = [f"feature_{i+1}" for i in range(df.shape[1])]
            return df

        return pd.read_csv(path, header=None)
    except ParserError:
        # 1) Once bilimsel gosterimdeki ondalik virgul sorununu duzeltmeyi dene.
        try:
            return _read_csv_with_scientific_comma_fix(path, is_feature_file=is_feature_file)
        except Exception:
            pass

        # 2) Fallback: bazi dosyalar ';' ayrac ve ',' ondalik ile geliyor.
        if is_feature_file:
            df = pd.read_csv(path, sep=';', decimal=',', header=None)
            df.columns = [f"feature_{i+1}" for i in range(df.shape[1])]
        else:
            df = pd.read_csv(path, sep=';', decimal=',', header=None)

        _normalize_csv_to_comma(path, df)
        return df


def load_data(
    dataset_name: str = "breast_cancer_data.csv",
    model_name: str = "",
    dataset_name_folder: str = "",
    folder: str = "raw",
    target_column: str = "diagnosis",
) -> pd.DataFrame:
    if folder != "raw":
        return pd.read_csv(get_data(dataset_name, model_name=model_name, dataset_name_folder=dataset_name_folder, folder=folder))

    data_path = get_data(dataset_name, folder="raw")
    label_name = _build_label_filename(dataset_name)
    label_path = get_data(label_name, folder="raw")

    df_data = _read_csv_flexible(data_path, is_feature_file=True)
    df_label = _read_csv_flexible(label_path, is_feature_file=False)

    if df_label.shape[1] != 1:
        raise ValueError(
            f"Label dosyasında tek kolon olmalı. Dosya: {label_name}, kolon sayısı: {df_label.shape[1]}"
        )

    if len(df_data) != len(df_label):
        raise ValueError(
            f"Data ve label satır sayısı eşleşmiyor. data={len(df_data)}, label={len(df_label)}"
        )

    label_col = df_label.columns[0]
    df_label = df_label.rename(columns={label_col: target_column})

    return pd.concat([df_data, df_label], axis=1)


def basic_info(df: pd.DataFrame) -> None:
    print("\n--- İlk 5 Satır ---")
    print(df.head())

    print("\n--- Shape ---")
    print(df.shape)

    print("\n--- Sütunlar ---")
    print(df.columns.tolist())

    print("\n--- Eksik Veri Sayıları ---")
    print(df.isnull().sum())

    print("\n--- Veri Tipleri ---")
    print(df.dtypes)