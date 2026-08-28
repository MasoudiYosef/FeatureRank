"""Project defaults and the small configuration object shared by workflows."""

from dataclasses import dataclass
from pathlib import Path


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Dataset defaults
TARGET_COLUMN = "target"
ID_COLUMN = "ID"
TEST_SIZE = 0.1
RANDOM_STATE = 42

# Autoencoder and classifier defaults
AUTOENCODER_EPOCHS = 50
BATCH_SIZE = 16
CLASSIFIER_VALIDATION_SPLIT = 0.1
THRESHOLD = 0.5

CLASSIFIER_EPOCHS = 50
CLASSIFIER_HIDDEN_UNITS = (32, 16)
CLASSIFIER_DROPOUT_RATES = None
CLASSIFIER_LEARNING_RATE = 0.001
CLASSIFIER_MODEL = "neural"
CLASSIFIER_CLASS_WEIGHT = "none"
CLASSIFIER_SAMPLING = "none"
CLASSIFIER_EARLY_STOPPING_PATIENCE = 0
CLASSIFIER_EARLY_STOPPING_MONITOR = "val_accuracy"
CLASSIFIER_EARLY_STOPPING_MIN_DELTA = 0.0

AUTOENCODER_EARLY_STOPPING_PATIENCE = 0
AUTOENCODER_EARLY_STOPPING_MIN_DELTA = 0.001

FEATURE_CHUNK_SIZE = 10000
CHUNK_FEATURE_THRESHOLD = 50000
ENABLE_FEATURE_CHUNKING = True

# Clustering and regression defaults
CLUSTER_MIN_K = 2
CLUSTER_MAX_K = 16

REGRESSION_MODEL = "neural"
KMEANS_REGRESSION_CLUSTERS = 5
KMEANS_REGRESSION_N_INIT = 10
ACTUAL_PREDICTED_TOP_N = None
DEVICE = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters for one FeatureRank experiment run."""

    dataset_name: str = "breast_cancer_data.csv"
    task: str = "classification"
    feature_percent: float = 20.0
    random_state: int | None = RANDOM_STATE
    encoding_dim: int = 8
    target_column: str = "target"
    id_column: str | None = "ID"
    cluster_k: int | None = None
    save_details: bool = False


def get_data(
    dataset_name: str = "breast_cancer_data.csv",
    model_name: str = "",
    dataset_name_folder: str = "",
    folder: str = "raw",
) -> Path:
    """Resolve a raw or legacy filtered dataset path."""
    if folder == "raw":
        return RAW_DATA_DIR / dataset_name
    if folder == "filtered_datasets":
        if not model_name or not dataset_name_folder:
            raise ValueError("filtered_datasets icin model_name ve dataset_name_folder zorunludur.")
        return (
            BASE_DIR
            / "data"
            / "filtered_datasets"
            / model_name
            / dataset_name_folder
            / "reports"
            / dataset_name
        )
    raise ValueError(f"Gecersiz folder: {folder}. 'raw' veya 'filtered_datasets' olmali.")
