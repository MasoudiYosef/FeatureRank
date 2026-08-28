"""Prepare tabular data for the classification, regression, and clustering workflows.

The module keeps the established ID handling, target encoding, train/test split,
and feature scaling behavior in one place.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import ID_COLUMN, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


def _prepare_target(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
) -> tuple[pd.DataFrame, bool]:
    """Encode the target and return whether the split should be stratified."""
    task_name = task_type.lower().strip()
    if task_name == "regression":
        return encode_regression_target(df, target_column), False
    if task_name == "classification":
        return encode_target(df, target_column), True
    raise ValueError("task_type 'classification' veya 'regression' olmali.")


def _validate_scaled_data(X_train: np.ndarray, X_test: np.ndarray) -> None:
    """Fail early when scaling produced an invalid numeric matrix."""
    for name, values in (("X_train_scaled", X_train), ("X_test_scaled", X_test)):
        if np.isnan(values).any() or np.isinf(values).any():
            raise ValueError(f"{name} contains NaN/Inf. Shape: {values.shape}")


def is_probable_regression_target(y: pd.Series) -> bool:
    """Return True when a numeric target looks continuous rather than categorical."""
    values = y.dropna()
    if values.empty or not pd.api.types.is_numeric_dtype(values):
        return False
    unique_count = int(values.nunique())
    if unique_count <= 20:
        return False

    numeric = values.astype(float).to_numpy()
    has_decimals = np.any(~np.isclose(numeric, np.round(numeric)))
    unique_ratio = unique_count / len(values)
    singleton_ratio = float((values.value_counts() == 1).mean())
    return bool(has_decimals and (unique_ratio > 0.1 or singleton_ratio > 0.5))


def drop_id_column(df: pd.DataFrame, id_column: str | None = ID_COLUMN) -> pd.DataFrame:
    """
    Eğer ID sütunu varsa veri setinden kaldırır.
    """
    if id_column and id_column in df.columns:
        df = df.drop(columns=[id_column])
    return df


# df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})
def encode_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """
    diagnosis sütununu sayısal hale getirir.
    M -> 1
    B -> 0
    """
    df = df.copy()
    if target_column not in df.columns:
        raise ValueError(f"Target kolonu bulunamadı: {target_column}")

    y = df[target_column]

    if y.dtype == bool:
        df[target_column] = y.astype(int)
        return df

    if pd.api.types.is_numeric_dtype(y):
        y_int = y.astype(int)
        unique_labels = sorted(pd.Series(y_int).dropna().unique().tolist())
        if len(unique_labels) == 2 and set(unique_labels) != {0, 1}:
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            df[target_column] = y_int.map(label_map).astype(int)
            return df
        df[target_column] = y_int
        return df

    y_str = y.astype(str).str.strip()
    unique_labels = sorted(pd.Series(y_str).dropna().unique().tolist())

    if set(unique_labels) == {"B", "M"}:
        df[target_column] = y_str.map({"M": 1, "B": 0})
        return df

    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    df[target_column] = y_str.map(label_map).astype(int)
    return df


def encode_regression_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """
    Regression hedefini sınıf etiketine çevirmeden sayısal float olarak hazırlar.
    """
    df = df.copy()
    if target_column not in df.columns:
        raise ValueError(f"Target kolonu bulunamadı: {target_column}")

    y = df[target_column]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.strip().str.replace(",", ".", regex=False)

    y_numeric = pd.to_numeric(y, errors="coerce")
    if y_numeric.isna().any():
        bad_count = int(y_numeric.isna().sum())
        raise ValueError(
            f"Regression hedef kolonunda sayısala çevrilemeyen {bad_count} değer var: {target_column}"
        )

    df[target_column] = y_numeric.astype(np.float32)
    return df


def split_features_target(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Girdi özelliklerini (X) ve hedef değişkeni (y) ayırır.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target kolonu bulunamadı: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def handle_pid_unrealistic_zeros(X: pd.DataFrame) -> pd.DataFrame:
    """
    PID (Pima Indians Diabetes) veri seti icin gercek disi 0 degerleri NaN yapar.
    feature_2..feature_6 sirasiyla glucose, blood_pressure, skin_thickness,
    insulin ve BMI'ye karsilik gelir; bu alanlarda 0 fizyolojik olarak gecersizdir.
    """
    X_fixed = X.copy()

    pid_like_columns = [f"feature_{i}" for i in range(1, 9)]
    if list(X_fixed.columns) != pid_like_columns:
        return X_fixed

    zero_as_missing_cols = [
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
        "feature_6",
    ]
    X_fixed[zero_as_missing_cols] = X_fixed[zero_as_missing_cols].replace(0, np.nan)
    return X_fixed


def sanitize_mixed_type_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    String formatta gelen sayisal degerleri temizleyip sayisala cevirir.
    - Virgullu ondaliklari noktaya cevirir (1,23 -> 1.23)
    - Bos/metin bozugu degerleri NaN yapar
    - Tamami NaN olan kolonlari atar
    - Kalan NaN degerleri kolon medyani ile doldurur
    """
    X_clean = X.copy()

    if all(pd.api.types.is_numeric_dtype(X_clean[col]) for col in X_clean.columns):
        if X_clean.shape[1] == 0:
            raise ValueError("Sayısal feature kolonu bulunamadı. Data dosyasını kontrol edin.")
        if X_clean.isna().any().any():
            X_clean = X_clean.fillna(X_clean.median(numeric_only=True)).fillna(0.0)
        return X_clean

    for col in X_clean.columns:
        series = X_clean[col]
        if pd.api.types.is_numeric_dtype(series):
            continue

        s = series.astype(str).str.strip()
        s = s.replace(
            {
                "": np.nan,
                "nan": np.nan,
                "None": np.nan,
                "NA": np.nan,
                "N/A": np.nan,
                "?": np.nan,
            }
        )
        s = s.str.replace(",", ".", regex=False)
        X_clean[col] = pd.to_numeric(s, errors="coerce")

    # Sayisal gorunen tiplerde de kalan metin olabilir; son bir coercion uygula.
    X_clean = X_clean.apply(pd.to_numeric, errors="coerce")

    all_nan_cols = [col for col in X_clean.columns if X_clean[col].isna().all()]
    if all_nan_cols:
        X_clean = X_clean.drop(columns=all_nan_cols)

    if X_clean.shape[1] == 0:
        raise ValueError("Sayısal feature kolonu bulunamadı. Data dosyasını kontrol edin.")

    if X_clean.isna().any().any():
        X_clean = X_clean.fillna(X_clean.median(numeric_only=True))

    # Nadir durumda median da NaN kalirsa 0 ile tamamla.
    if X_clean.isna().any().any():
        X_clean = X_clean.fillna(0.0)

    return X_clean


def keep_numeric_features_only(X: pd.DataFrame) -> pd.DataFrame:
    """
    Sayısal olmayan feature kolonlarını otomatik olarak çıkarır.
    Örn: sample_id gibi string kolonlar.
    """
    return sanitize_mixed_type_features(X)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int | None = RANDOM_STATE,
    stratify: bool = True,
):
    """
    Veriyi train ve test olarak böler.
    stratify=y kullanarak sınıf dağılımını korur.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=(
            y if stratify else None
        ),  # Classification'da sınıf dağılımını korur; regression'da kapalıdır.
    )
    return X_train, X_test, y_train, y_test


def scale_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    StandardScaler ile veriyi ölçekler.
    Sadece X_train üzerinde fit edilir; X_test aynı scaler ile dönüştürülür.
    Çıkış dtype: float32 (TensorFlow uyumluluğu için)
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_column: str | None = ID_COLUMN,
    random_state: int | None = RANDOM_STATE,
    scale_features: bool = True,
    task_type: str = "classification",
):
    """
    Tüm preprocessing adımlarını sırasıyla uygular.
    Çıkış: float32 arrays (TensorFlow uyumluluğu)
    """
    df = drop_id_column(df, id_column=id_column)
    df, use_stratify = _prepare_target(df, target_column, task_type)

    X, y = split_features_target(df, target_column=target_column)
    X = handle_pid_unrealistic_zeros(X)
    X = keep_numeric_features_only(X)
    X_train, X_test, y_train, y_test = split_data(
        X, y, random_state=random_state, stratify=use_stratify
    )
    if not scale_features:
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "X_train_scaled": None,
            "X_test_scaled": None,
            "scaler": None,
        }

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    _validate_scaled_data(X_train_scaled, X_test_scaled)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled.astype(np.float32),
        "X_test_scaled": X_test_scaled.astype(np.float32),
        "scaler": scaler,
    }
