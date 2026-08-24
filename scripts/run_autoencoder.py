import os
import sys
import argparse
import random
import json
import time
import re
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FormatStrFormatter
from sklearn.cluster import KMeans
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	confusion_matrix,
	f1_score,
	mean_absolute_error,
	mean_squared_error,
	precision_recall_curve,
	precision_score,
	r2_score,
	recall_score,
	roc_auc_score,
	roc_curve,
	silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR

# Proje kokunu import path'ine ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RANDOM_STATE
from src.data_loader import convert_txt_dataset_to_csv, load_data
from src.models import build_sigmoid_autoencoder, build_latent_classifier, build_latent_regressor
from src.preprocessing import (
	drop_id_column,
	encode_target,
	handle_pid_unrealistic_zeros,
	keep_numeric_features_only,
	preprocess_data,
	scale_data,
	split_features_target,
)
from src.autoencoder_feature_selection import (
	save_top_percent_features_by_abs_max_weight,
	save_filtered_dataset_from_selected_features,
	validate_feature_percent,
)
from src.utils import ensure_dir, save_json, compute_multiclass_macro_accuracy

AUTOENCODER_EPOCHS = 50
BATCH_SIZE = 16
CLASSIFIER_VALIDATION_SPLIT = 0.1
THRESHOLD = 0.5
DEFAULT_CLASSIFIER_EPOCHS = 50
DEFAULT_CLASSIFIER_HIDDEN_UNITS = (32, 16)
DEFAULT_FEATURE_CHUNK_SIZE = 10000
DEFAULT_CHUNK_FEATURE_THRESHOLD = 50000
DEFAULT_CLUSTER_MIN_K = 2
DEFAULT_CLUSTER_MAX_K = 16
DEFAULT_EARLY_STOPPING_PATIENCE = 0
DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE = 0
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.0
DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA = 0.001
DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR = "val_accuracy"
DEFAULT_CLASSIFIER_MODEL = "neural"
DEFAULT_REGRESSION_MODEL = "neural"
DEFAULT_SVR_KERNEL = "rbf"
DEFAULT_SVR_C = 1.0
DEFAULT_SVR_EPSILON = 0.1
DEFAULT_SVR_GAMMA = "scale"
DEFAULT_KMEANS_REGRESSION_CLUSTERS = 5
DEFAULT_KMEANS_REGRESSION_N_INIT = 10
DEFAULT_DIMENSION_REDUCTION_MIN_HIDDEN_DIM = 128
DEFAULT_DIMENSION_REDUCTION_MAX_HIDDEN_DIM = 2048
PCA_CLUSTER_CMAP = ListedColormap(
	[
		"#0072B2",  # blue
		"#009E73",  # green
		"#FED000",  # yellow
		"#E41A1C",  # red
		"#8A8A8A",  # gray
		"#A6761D",  # brown
		"#F781BF",  # pink
	],
	name="feature_rank_cluster",
)


OUTPUT_ROOT = Path("outputs")
CLASSIFICATION_OUTPUT_ROOT = OUTPUT_ROOT / "Classification"
REGRESSION_OUTPUT_ROOT = OUTPUT_ROOT / "Regression"
CLUSTERING_OUTPUT_ROOT = OUTPUT_ROOT / "Clustering"


def classification_output_dir(dataset_folder: str | Path) -> Path:
	return CLASSIFICATION_OUTPUT_ROOT / Path(dataset_folder)


def regression_output_dir(dataset_folder: str | Path) -> Path:
	return REGRESSION_OUTPUT_ROOT / Path(dataset_folder)


def clustering_output_dir(dataset_folder: str | Path) -> Path:
	return CLUSTERING_OUTPUT_ROOT / Path(dataset_folder)


def task_output_dir(task: str, dataset_folder: str | Path) -> Path:
	task_key = task.lower().strip()
	if task_key == "classification":
		return classification_output_dir(dataset_folder)
	if task_key == "regression":
		return regression_output_dir(dataset_folder)
	if task_key == "clustering":
		return clustering_output_dir(dataset_folder)
	raise ValueError(f"Bilinmeyen task output tipi: {task}")


@dataclass(frozen=True)
class ExperimentConfig:
	"""Runtime parameters shared by a single experiment run."""

	dataset_name: str = "breast_cancer_data.csv"
	validation_dataset_name: str | None = None
	target_column: str = "target"
	id_column: str | None = "ID"
	task: str = "classification"
	encoding_dim: int = 8
	feature_percent: float = 50.0
	random_state: int | None = RANDOM_STATE
	classifier_epochs: int = DEFAULT_CLASSIFIER_EPOCHS
	classifier_hidden_units: tuple[int, ...] = DEFAULT_CLASSIFIER_HIDDEN_UNITS
	classifier_dropout_rates: tuple[float, ...] | None = None
	classifier_learning_rate: float = 0.001
	classifier_model: str = DEFAULT_CLASSIFIER_MODEL
	regression_model: str = DEFAULT_REGRESSION_MODEL
	svr_kernel: str = DEFAULT_SVR_KERNEL
	svr_c: float = DEFAULT_SVR_C
	svr_epsilon: float = DEFAULT_SVR_EPSILON
	svr_gamma: str | float = DEFAULT_SVR_GAMMA
	kmeans_regression_clusters: int = DEFAULT_KMEANS_REGRESSION_CLUSTERS
	kmeans_regression_n_init: int = DEFAULT_KMEANS_REGRESSION_N_INIT
	device: str = "auto"
	feature_chunk_size: int = DEFAULT_FEATURE_CHUNK_SIZE
	chunk_feature_threshold: int = DEFAULT_CHUNK_FEATURE_THRESHOLD
	enable_feature_chunking: bool = True
	classifier_early_stopping_patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE
	autoencoder_early_stopping_patience: int | None = DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE
	classifier_early_stopping_monitor: str = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR
	classifier_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
	autoencoder_early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA
	classifier_class_weight: str = "none"
	classifier_sampling: str = "none"
	cluster_k: int | None = None
	cluster_min_k: int = DEFAULT_CLUSTER_MIN_K
	cluster_max_k: int = DEFAULT_CLUSTER_MAX_K
	save_training_plots: bool = False
	actual_predicted_top_n: int | None = None
	save_encoded_dataset: bool = False
	evaluate_dimension_reduction: bool = False


def compute_dimension_reduction_hidden_dim(input_dim: int, encoding_dim: int) -> int:
	if input_dim <= 0:
		raise ValueError("Dimension reduction input_dim pozitif olmali.")
	if encoding_dim <= 0:
		raise ValueError("Dimension reduction encoding_dim pozitif olmali.")

	# Buyuk veri setlerinde input_dim kadar genis gizli katman kurmak
	# bellek ve sure maliyetini gereksiz sekilde patlatıyor. Reduction
	# akisi icin gizli katmani simetrik tutuyoruz ama makul bir ust sinir
	# ile daraltiyoruz.
	candidate_hidden_dim = max(
		int(encoding_dim * 4),
		int(np.sqrt(float(input_dim) * float(encoding_dim))),
		DEFAULT_DIMENSION_REDUCTION_MIN_HIDDEN_DIM,
	)
	return min(input_dim, candidate_hidden_dim, DEFAULT_DIMENSION_REDUCTION_MAX_HIDDEN_DIM)


def build_dimension_reduction_autoencoder(input_dim: int, encoding_dim: int, activation: str = "relu"):
	if encoding_dim <= 0:
		raise ValueError("Dimension reduction encoding_dim pozitif olmali.")
	hidden_dim = compute_dimension_reduction_hidden_dim(input_dim, encoding_dim)
	input_layer = tf.keras.layers.Input(shape=(input_dim,), dtype="float32", name="input_layer")
	encoder_hidden = tf.keras.layers.Dense(hidden_dim, activation=activation, name="encoder_hidden")(input_layer)
	latent = tf.keras.layers.Dense(encoding_dim, activation="linear", name="latent")(encoder_hidden)
	decoded_input = tf.keras.layers.Dense(hidden_dim, activation=activation, name="decoder_hidden")(latent)
	decoded = tf.keras.layers.Dense(input_dim, activation="linear", name="reconstruction")(decoded_input)
	autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoded, name="dimension_reduction_autoencoder")
	encoder = tf.keras.Model(inputs=input_layer, outputs=latent, name="dimension_reduction_encoder")
	autoencoder.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
		loss="mse",
		jit_compile=False,
	)
	return autoencoder, encoder


def build_cluster_colormap(cluster_count: int) -> ListedColormap:
	if cluster_count <= PCA_CLUSTER_CMAP.N:
		return ListedColormap(PCA_CLUSTER_CMAP.colors[:cluster_count], name="feature_rank_cluster_dynamic")
	return ListedColormap(plt.get_cmap("tab20", cluster_count)(np.arange(cluster_count)), name="feature_rank_cluster_dynamic")


def set_reproducible(seed: int | None) -> None:
	tf.keras.backend.clear_session()
	if seed is None:
		return
	random.seed(seed)
	np.random.seed(seed)
	tf.keras.utils.set_random_seed(seed)
	try:
		tf.config.experimental.enable_op_determinism()
	except Exception:
		pass


def configure_tensorflow_device(device: str = "auto") -> None:
	device = device.lower().strip()
	if device not in {"auto", "gpu", "cpu"}:
		raise ValueError("device parametresi 'auto', 'gpu' veya 'cpu' olmalidir.")

	available_gpus = tf.config.list_physical_devices("GPU")
	if device == "gpu":
		if not available_gpus:
			raise RuntimeError(
				"GPU bulunamadi. GPU ile calistirmak icin uygun CUDA/cuDNN ve GPU destekli TensorFlow yuklu olmalidir."
			)
		print(f"[INFO] GPU algilandi: {[gpu.name for gpu in available_gpus]}")
		try:
			for gpu in available_gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			tf.config.set_visible_devices(available_gpus, "GPU")
		except Exception as exc:
			print(f"[WARN] GPU ayarlari yapilamadi: {exc}")
	elif device == "cpu":
		try:
			tf.config.set_visible_devices([], "GPU")
			print("[INFO] GPU devre disi birakildi, CPU uzerinden calisiyor.")
		except Exception as exc:
			print(f"[WARN] GPU devre disi birakilamadi: {exc}")
	else:
		if available_gpus:
			print(f"[INFO] GPU mevcut, GPU uzerinden calisacak: {[gpu.name for gpu in available_gpus]}")
			try:
				for gpu in available_gpus:
					tf.config.experimental.set_memory_growth(gpu, True)
				tf.config.set_visible_devices(available_gpus, "GPU")
			except Exception as exc:
				print(f"[WARN] GPU ayarlari yapilamadi: {exc}")
		else:
			print("[INFO] GPU bulunamadi, CPU uzerinden calisiyor.")


def save_feature_weighted_lists(autoencoder, X_train_sub: np.ndarray, feature_names: list[str], output_path: Path) -> None:
	"""
	Her feature icin bagli oldugu nöronlara sample-bazli katkı listesi uretir:
	contribution_list_i[j] = mean_s( abs(x_s,i * w_i,j) )
	"""
	weights = autoencoder.get_layer("enc_dense_1").get_weights()[0]  # (n_features, n_neurons)
	if X_train_sub.ndim != 2:
		raise ValueError(f"X_train_sub 2 boyutlu olmali, gelen shape: {X_train_sub.shape}")

	if weights.shape[0] != X_train_sub.shape[1]:
		raise ValueError(
			f"X_train feature boyutu ({X_train_sub.shape[1]}) ile agirlik satir sayisi ({weights.shape[0]}) eslesmiyor."
		)

	# (n_samples, n_features, 1) * (1, n_features, n_neurons)
	# -> (n_samples, n_features, n_neurons)
	contributions = np.abs(X_train_sub[:, :, np.newaxis] * weights[np.newaxis, :, :])
	weighted = np.mean(contributions, axis=0)  # (n_features, n_neurons)

	df = pd.DataFrame(
		{
			"feature": [f"F{i+1}" for i in range(weighted.shape[0])],
			"weight_list": [weighted[i].tolist() for i in range(weighted.shape[0])],
		}
	)
	df.to_csv(output_path, index=False)


def normalize_id_column(id_column: str | None) -> str | None:
	if id_column and id_column.lower() in {"none", "null", "-", ""}:
		return None
	return id_column


def format_feature_percent_tag(feature_percent: float) -> str:
	if float(feature_percent).is_integer():
		return str(int(feature_percent))
	return str(feature_percent).replace(".", "_")


def is_encoded_dataset_folder(dataset_folder: str) -> bool:
	folder_name = str(dataset_folder).lower()
	return "_encoded_dim_" in folder_name or "_encoded_" in folder_name or "_dimension_reduction_" in folder_name


def get_encoded_source_feature_percent(dataset_folder: str) -> float | None:
	match = re.search(r"_top_(\d+(?:_\d+)?)_encoded_dim_", str(dataset_folder).lower())
	if not match:
		return None
	return float(match.group(1).replace("_", "."))


def format_encoded_output_percent(feature_percent: float, dataset_folder: str) -> float:
	# Encoded dataset adindaki top_X, latent CSV'nin hangi orijinal feature yuzdesinden
	# uretildigini anlatir. Deney ciktilari ise komutta verilen aktif yuzdeyi kullanmalidir.
	return feature_percent


def format_metric_output_prefix(feature_percent: float, dataset_folder: str) -> str:
	if is_encoded_dataset_folder(dataset_folder):
		feature_percent = format_encoded_output_percent(feature_percent, dataset_folder)
		feature_percent_tag = format_feature_percent_tag(feature_percent)
		return f"top_{feature_percent_tag}_encoder"
	feature_percent_tag = format_feature_percent_tag(feature_percent)
	return f"top_{feature_percent_tag}"


def format_test_metrics_filename(feature_percent: float, dataset_folder: str) -> str:
	if is_encoded_dataset_folder(dataset_folder):
		feature_percent = format_encoded_output_percent(feature_percent, dataset_folder)
		feature_percent_tag = format_feature_percent_tag(feature_percent)
		return f"top_{feature_percent_tag}_test_encoder_metrics.json"
	feature_percent_tag = format_feature_percent_tag(feature_percent)
	return f"top_{feature_percent_tag}_test_metrics.json"


def format_feature_output_label(feature_percent: float, dataset_folder: str) -> str:
	output_feature_percent = format_encoded_output_percent(feature_percent, dataset_folder)
	output_feature_percent_tag = format_feature_percent_tag(output_feature_percent).replace("_", ".")
	if is_encoded_dataset_folder(dataset_folder):
		return f"Top %{output_feature_percent_tag} encoder"
	return f"Top %{output_feature_percent_tag}"


def add_encoded_metric_metadata(metrics_data: dict, feature_percent: float, dataset_folder: str) -> None:
	if not is_encoded_dataset_folder(dataset_folder):
		return
	source_percent = get_encoded_source_feature_percent(dataset_folder)
	metrics_data["encoded_dataset"] = True
	if source_percent is not None:
		metrics_data["encoded_source_feature_percent"] = source_percent
	metrics_data["encoded_active_feature_percent"] = feature_percent


def format_encoded_dataset_stem(dataset_folder: str, feature_percent: float, encoding_dim: int) -> str:
	feature_percent_tag = format_feature_percent_tag(feature_percent)
	return f"{dataset_folder}_dimension_reduction_{feature_percent_tag}_data"


def parse_feature_percent_values(feature_percent_text: str | float | int) -> list[float]:
	text = str(feature_percent_text).strip().lower()
	if text in {"all", "grid", "range"}:
		return [float(value) for value in range(10, 101, 10)]

	parts = [part.strip() for part in text.split(",") if part.strip()]
	if not parts:
		raise ValueError("feature-percent bos olamaz. Ornek: 20, 10,20,30 veya all")

	values: list[float] = []
	for part in parts:
		value = validate_feature_percent(float(part))
		if value not in values:
			values.append(value)
	return values


def parse_hidden_units(units_text: str) -> tuple[int, ...]:
	parts = [p.strip() for p in units_text.split(",") if p.strip()]
	if not parts:
		raise ValueError("classifier-hidden-units bos olamaz. Ornek: 128,64")
	units = tuple(int(p) for p in parts)
	if any(u <= 0 for u in units):
		raise ValueError("classifier-hidden-units pozitif tam sayilar olmali.")
	return units


def parse_dropout_rates(dropout_text: str | None, layer_count: int) -> tuple[float, ...] | None:
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


def parse_random_state(random_state_text: str | None) -> int | None:
	if random_state_text is None:
		return RANDOM_STATE
	text = random_state_text.strip().lower()
	if text in {"none", "null", ""}:
		return None
	return int(random_state_text)


def parse_svr_gamma(gamma_text: str) -> str | float:
	text = str(gamma_text).strip().lower()
	if text in {"scale", "auto"}:
		return text
	return float(gamma_text)


def compute_rmse_from_mse(mse: float | None) -> float | None:
	if mse is None:
		return None
	mse_value = float(mse)
	if np.isnan(mse_value):
		return None
	return float(np.sqrt(max(mse_value, 0.0)))


def add_test_rmse_metric(metrics_data: dict, mse_key: str = "test_mse", rmse_key: str = "test_rmse") -> None:
	if mse_key in metrics_data:
		metrics_data[rmse_key] = compute_rmse_from_mse(metrics_data.get(mse_key))


def compute_binary_class_weight(y_train: np.ndarray, mode: str = "none") -> dict[int, float] | None:
	mode = str(mode).strip().lower()
	if mode in {"none", "", "off", "false"}:
		return None
	if mode != "balanced":
		raise ValueError("classifier-class-weight 'none' veya 'balanced' olmali.")

	classes, counts = np.unique(y_train.astype(int), return_counts=True)
	if not set(classes).issubset({0, 1}) or len(classes) < 2:
		print("[WARN] class_weight balanced atlandi: y_train icinde iki binary sinif yok.")
		return None

	total = float(np.sum(counts))
	class_count = float(len(classes))
	class_weight = {
		int(class_label): total / (class_count * float(count))
		for class_label, count in zip(classes, counts)
	}
	print(f"[INFO] Class weight kullaniliyor: {class_weight}")
	return class_weight


def undersample_binary_training_data(
	X_train: np.ndarray,
	y_train: np.ndarray,
	random_state: int | None,
	mode: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
	mode = str(mode).strip().lower()
	if mode in {"none", "", "off", "false"}:
		return X_train, y_train
	if mode != "undersample":
		raise ValueError("classifier-sampling 'none' veya 'undersample' olmali.")

	y_train_int = y_train.astype(int)
	classes, counts = np.unique(y_train_int, return_counts=True)
	if not set(classes).issubset({0, 1}) or len(classes) < 2:
		print("[WARN] undersample atlandi: y_train icinde iki binary sinif yok.")
		return X_train, y_train

	target_count = int(np.min(counts))
	rng = np.random.default_rng(random_state)
	selected_indices: list[np.ndarray] = []
	before_counts = {int(class_label): int(count) for class_label, count in zip(classes, counts)}
	for class_label in classes:
		class_indices = np.where(y_train_int == int(class_label))[0]
		if len(class_indices) > target_count:
			class_indices = rng.choice(class_indices, size=target_count, replace=False)
		selected_indices.append(class_indices)

	balanced_indices = np.concatenate(selected_indices)
	rng.shuffle(balanced_indices)
	balanced_y = y_train[balanced_indices]
	after_classes, after_counts = np.unique(balanced_y.astype(int), return_counts=True)
	after_counts_dict = {int(class_label): int(count) for class_label, count in zip(after_classes, after_counts)}
	print(f"[INFO] Undersampling uygulandi. Once: {before_counts}, sonra: {after_counts_dict}")
	return X_train[balanced_indices], balanced_y


def build_sklearn_classifier(
	classifier_model: str,
	random_state: int | None,
	class_weight: dict[int, float] | None,
):
	classifier_model = str(classifier_model).strip().lower()
	if classifier_model == "logistic":
		return LogisticRegression(
			max_iter=5000,
			random_state=random_state,
			class_weight=class_weight,
		)
	if classifier_model == "svm":
		return SVC(
			kernel="rbf",
			probability=True,
			random_state=random_state,
			class_weight=class_weight,
		)
	if classifier_model == "random_forest":
		return RandomForestClassifier(
			n_estimators=300,
			random_state=random_state,
			class_weight=class_weight,
			n_jobs=-1,
		)
	raise ValueError("classifier-model 'neural', 'logistic', 'svm' veya 'random_forest' olmali.")


def predict_binary_scores(classifier, X_test_encoded: np.ndarray) -> np.ndarray:
	if hasattr(classifier, "predict_proba"):
		y_score = classifier.predict_proba(X_test_encoded)[:, 1]
	elif hasattr(classifier, "decision_function"):
		decision_scores = classifier.decision_function(X_test_encoded)
		y_score = 1.0 / (1.0 + np.exp(-decision_scores))
	else:
		y_score = classifier.predict(X_test_encoded)
	return np.asarray(y_score, dtype=np.float32).ravel()


def save_training_history(
	history: tf.keras.callbacks.History,
	output_dir: Path,
	file_prefix: str,
	plot_metrics: tuple[str, ...],
) -> None:
	ensure_dir(output_dir)
	history_df = pd.DataFrame(history.history)
	history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))

	csv_path = output_dir / f"{file_prefix}_history.csv"
	history_df.to_csv(csv_path, index=False)
	print(f"[OK] Training history CSV: {csv_path}")

	for metric in plot_metrics:
		if metric not in history_df.columns:
			continue

		plt.figure(figsize=(8, 5))
		plt.plot(history_df["epoch"], history_df[metric], label=metric)
		val_metric = f"val_{metric}"
		if val_metric in history_df.columns:
			plt.plot(history_df["epoch"], history_df[val_metric], label=val_metric)
		plt.xlabel("Epoch")
		plt.ylabel(metric)
		plt.title(f"{file_prefix} {metric}")
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.tight_layout()

		plot_path = output_dir / f"{file_prefix}_{metric}.png"
		plt.savefig(plot_path, dpi=150)
		plt.close()
		print(f"[OK] Training plot: {plot_path}")


def save_average_convergence(
	history_frames: list[pd.DataFrame],
	output_dir: Path,
	file_prefix: str,
) -> None:
	if not history_frames:
		return

	ensure_dir(output_dir)
	accuracy_series: list[pd.Series] = []
	val_accuracy_series: list[pd.Series] = []
	loss_series: list[pd.Series] = []
	val_loss_series: list[pd.Series] = []

	for history_df in history_frames:
		if "epoch" not in history_df.columns:
			continue
		indexed_history = history_df.set_index("epoch")
		if "accuracy" in indexed_history.columns:
			accuracy_series.append(indexed_history["accuracy"])
		if "val_accuracy" in indexed_history.columns:
			val_accuracy_series.append(indexed_history["val_accuracy"])
		if "loss" in indexed_history.columns:
			loss_series.append(indexed_history["loss"])
		if "val_loss" in indexed_history.columns:
			val_loss_series.append(indexed_history["val_loss"])

	if not accuracy_series and not loss_series:
		return

	all_epoch_indexes = [series.index for series in accuracy_series + loss_series + val_accuracy_series + val_loss_series]
	average_df = pd.DataFrame({"epoch": sorted(set().union(*all_epoch_indexes))})
	average_df = average_df.set_index("epoch")
	if accuracy_series:
		average_df["average_accuracy"] = pd.concat(accuracy_series, axis=1).mean(axis=1)
	if val_accuracy_series:
		average_df["average_val_accuracy"] = pd.concat(val_accuracy_series, axis=1).mean(axis=1)
	if loss_series:
		average_df["average_loss"] = pd.concat(loss_series, axis=1).mean(axis=1)
	if val_loss_series:
		average_df["average_val_loss"] = pd.concat(val_loss_series, axis=1).mean(axis=1)
	average_df = average_df.reset_index()

	loss_column = "average_val_loss" if "average_val_loss" in average_df.columns else "average_loss"
	normalized_loss_column = None
	if loss_column in average_df.columns:
		first_loss_value = float(average_df[loss_column].dropna().iloc[0])
		if np.isclose(first_loss_value, 0.0):
			normalized_loss_column = loss_column
		else:
			normalized_loss_column = f"{loss_column}_normalized"
			average_df[normalized_loss_column] = average_df[loss_column] / first_loss_value

	csv_path = output_dir / f"{file_prefix}_average_convergence.csv"
	average_df.to_csv(csv_path, index=False)
	print(f"[OK] Average convergence CSV: {csv_path}")

	if "average_accuracy" in average_df.columns:
		plt.figure(figsize=(8, 5))
		plt.plot(average_df["epoch"], average_df["average_accuracy"], label="average_accuracy")
		if "average_val_accuracy" in average_df.columns:
			plt.plot(average_df["epoch"], average_df["average_val_accuracy"], label="average_val_accuracy")
		plt.xlabel("Epoch")
		plt.ylabel("Average Accuracy")
		plt.title(f"{file_prefix} average accuracy convergence")
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		accuracy_plot_path = output_dir / f"{file_prefix}_average_accuracy_convergence.png"
		plt.savefig(accuracy_plot_path, dpi=150)
		plt.close()
		print(f"[OK] Average accuracy convergence plot: {accuracy_plot_path}")

	if loss_column in average_df.columns:
		plot_loss_column = normalized_loss_column or loss_column
		plt.figure(figsize=(8, 5))
		plt.plot(average_df["epoch"], average_df[plot_loss_column], label=plot_loss_column)
		plt.xlabel("Epoch")
		if normalized_loss_column is not None:
			plt.ylabel("Normalized Average Validation Loss" if loss_column == "average_val_loss" else "Normalized Average Loss")
			plt.ylim(bottom=0.0)
			plt.title(f"{file_prefix} {loss_column} convergence")
		else:
			plt.ylabel("Average Validation Loss" if loss_column == "average_val_loss" else "Average Loss")
			plt.title(f"{file_prefix} {loss_column} convergence")
		plt.legend()
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		loss_plot_path = output_dir / f"{file_prefix}_average_error_convergence.png"
		plt.savefig(loss_plot_path, dpi=150)
		plt.close()
		print(f"[OK] Average error convergence plot: {loss_plot_path}")


def save_metric_boxplot(
	metric_values: list[float],
	output_dir: Path,
	file_prefix: str,
	metric_name: str,
	normalize_to_unit: bool = False,
) -> None:
	if not metric_values:
		return

	ensure_dir(output_dir)
	metric_label = metric_name.lower()
	values = np.asarray(metric_values, dtype=float)
	y_label = metric_name
	title_metric_label = metric_label
	if normalize_to_unit:
		min_value = float(np.min(values))
		max_value = float(np.max(values))
		if np.isclose(max_value, min_value):
			values = np.full_like(values, 0.5, dtype=float)
		else:
			values = (values - min_value) / (max_value - min_value)
		y_label = f"Normalized {metric_name} (0-1)"
		title_metric_label = f"normalized {metric_label}"

	plt.figure(figsize=(6, 5))
	plt.boxplot(values, tick_labels=[metric_name], showmeans=True)
	if len(values) == 1:
		x_positions = np.array([1.0])
	else:
		x_positions = 1.0 + np.linspace(-0.12, 0.12, len(values))
	plt.scatter(
		x_positions,
		values,
		s=24,
		alpha=0.75,
		color="#1f77b4",
		edgecolors="white",
		linewidths=0.4,
		zorder=3,
	)
	plt.ylabel(y_label)
	if metric_name.lower() == "accuracy":
		plt.ylim(0.65, 1.02)
	elif normalize_to_unit:
		plt.ylim(-0.03, 1.03)
	plt.title(f"{file_prefix} {title_metric_label} boxplot")
	plt.grid(True, axis="y", alpha=0.3)
	plt.gca().ticklabel_format(axis="y", style="plain", useOffset=False)
	plt.tight_layout()

	plot_path = output_dir / f"{file_prefix}_{metric_label}_boxplot.png"
	plt.savefig(plot_path, dpi=150)
	plt.close()
	print(f"[OK] {metric_name} boxplot: {plot_path}")


def save_repeated_metric_distribution_plot(
	metric_values: list[float],
	output_dir: Path,
	file_prefix: str,
	metric_name: str,
) -> None:
	if not metric_values:
		return

	ensure_dir(output_dir)
	runs = np.arange(1, len(metric_values) + 1)
	values = np.asarray(metric_values, dtype=float)
	mean_value = float(np.mean(values))
	std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
	metric_label = metric_name.lower()

	plt.figure(figsize=(9, 5))
	plt.plot(runs, values, marker="o", linestyle="-", linewidth=1.2, markersize=4, label=metric_name)
	plt.axhline(mean_value, color="#d62728", linestyle="--", linewidth=1.4, label=f"mean={mean_value:.4f}")
	if len(values) > 1:
		plt.fill_between(
			runs,
			mean_value - std_value,
			mean_value + std_value,
			color="#d62728",
			alpha=0.12,
			label=f"mean ± std ({std_value:.4f})",
		)
	plt.xlabel("Run")
	plt.ylabel(metric_name)
	plt.title(f"{file_prefix} repeated {metric_label} distribution")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()

	plot_path = output_dir / f"{file_prefix}_{metric_label}_repeated_distribution.png"
	plt.savefig(plot_path, dpi=150)
	plt.close()
	print(f"[OK] Repeated {metric_name} distribution plot: {plot_path}")

def save_regression_actual_vs_predicted_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    file_prefix: str,
    top_n: int | None = None,
) -> None:
    if len(y_true) == 0 or len(y_pred) == 0:
        return

    ensure_dir(output_dir)

    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Regression true/pred length mismatch: {len(y_true)} != {len(y_pred)}")

    if top_n is not None:
        if top_n <= 0:
            raise ValueError("actual_vs_predicted top_n pozitif olmali.")
        top_n = min(top_n, len(y_true))
        y_true_plot = y_true[:top_n]
        y_pred_plot = y_pred[:top_n]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    # ---------------------------------------------------------
    # NORMALIZATION
    # Actual ve predicted aynı actual min-max ölçeğine göre normalize edilir.
    # Böylece x=actual, y=predicted ilişkisi bozulmaz.
    # ---------------------------------------------------------
    actual_min = float(np.min(y_true_plot))
    actual_max = float(np.max(y_true_plot))
    actual_range = actual_max - actual_min

    if np.isclose(actual_range, 0.0):
        print(f"[WARN] Normalization skipped because actual range is zero: {file_prefix}")
        y_true_norm = y_true_plot
        y_pred_norm = y_pred_plot
    else:
        y_true_norm = (y_true_plot - actual_min) / actual_range
        y_pred_norm = (y_pred_plot - actual_min) / actual_range

    mse_value = float(mean_squared_error(y_true_norm, y_pred_norm))

    rank_true = pd.Series(y_true_norm).rank().to_numpy()
    rank_pred = pd.Series(y_pred_norm).rank().to_numpy()

    if np.isclose(np.std(rank_true), 0.0) or np.isclose(np.std(rank_pred), 0.0):
        spearman_corr = float("nan")
    else:
        spearman_corr = float(np.corrcoef(rank_true, rank_pred)[0, 1])

    # n_points = len(y_true_norm)

    # if n_points < 200:
    #     point_alpha = 0.70
    #     point_size = 35
    # elif n_points < 1000:
    #     point_alpha = 0.45
    #     point_size = 24
    # else:
    #     point_alpha = 0.20
    #     point_size = 16

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("white")

    ax.scatter(
        y_true_norm,
        y_pred_norm,
        color="black",
        alpha=1.0,
        s=35,
        edgecolors="none",
        zorder=3,
        label="Samples",
    )

    # Normalize edilmiş grafikte ideal çizgi 0-1 arasıdır
    ax.plot(
        [0, 1],
        [0, 1],
        color="black",
        linestyle="-",
        linewidth=1.5,
        zorder=2,
        label="Ideal line",
    )

    ax.set_xlabel("Real Value", fontsize=12)
    ax.set_ylabel("Predicted Value", fontsize=12)
    ax.set_title("Scatter Plot Across All Datasets", fontsize=14)

    ax.text(
        0.05,
        0.95,
        f"Spearman Correlation Coefficient: {spearman_corr:.4f}\nMSE: {mse_value:.5f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )

    ax.set_xlim(-0.03, 1.03)

    # Tahminler 1'in biraz üstüne veya 0'ın altına taşarsa görünür kalsın
    y_min = min(-0.03, float(np.min(y_pred_norm)) - 0.03)
    y_max = max(1.03, float(np.max(y_pred_norm)) + 0.03)
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.25)

    plt.tight_layout()

    plot_path = output_dir / f"{file_prefix}_actual_vs_predicted.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] Normalized actual vs predicted plot: {plot_path}")

def save_regression_prediction_errors(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	sample_index,
	output_dir: Path,
	file_prefix: str,
) -> Path | None:
	if len(y_true) == 0 or len(y_pred) == 0:
		return None

	ensure_dir(output_dir)
	y_true = np.asarray(y_true, dtype=float).ravel()
	y_pred = np.asarray(y_pred, dtype=float).ravel()
	if len(y_true) != len(y_pred):
		raise ValueError(f"Regression true/pred length mismatch: {len(y_true)} != {len(y_pred)}")

	sample_index_values = list(sample_index)
	if len(sample_index_values) != len(y_true):
		sample_index_values = list(range(len(y_true)))

	error = y_pred - y_true
	predictions_df = pd.DataFrame(
		{
			"sample_index": sample_index_values,
			"true_value": y_true,
			"predicted_value": y_pred,
			"error_pred_minus_true": error,
			"absolute_error": np.abs(error),
			"squared_error": error ** 2,
		}
	)
	csv_path = output_dir / f"{file_prefix}_prediction_errors.csv"
	predictions_df.to_csv(csv_path, index=False)
	print(f"[OK] Regression prediction errors CSV: {csv_path}")
	return csv_path


def save_classification_predictions(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	y_score: np.ndarray,
	output_dir: Path,
	file_prefix: str,
) -> Path | None:
	if len(y_true) == 0 or len(y_pred) == 0 or len(y_score) == 0:
		return None

	ensure_dir(output_dir)
	y_true = np.asarray(y_true, dtype=int).ravel()
	y_pred = np.asarray(y_pred, dtype=int).ravel()
	y_score = np.asarray(y_score, dtype=float).ravel()
	if len(y_true) != len(y_pred) or len(y_true) != len(y_score):
		raise ValueError(
			f"Classification prediction length mismatch: "
			f"y_true={len(y_true)}, y_pred={len(y_pred)}, y_score={len(y_score)}"
		)

	predictions_df = pd.DataFrame(
		{
			"sample_index": list(range(len(y_true))),
			"true_label": y_true,
			"predicted_label": y_pred,
			"positive_class_score": y_score,
		}
	)
	csv_path = output_dir / f"{file_prefix}_classification_predictions.csv"
	predictions_df.to_csv(csv_path, index=False)
	print(f"[OK] Classification predictions CSV: {csv_path}")
	return csv_path


def save_classification_confusion_matrix_plot(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	output_dir: Path,
	file_prefix: str,
) -> Path | None:
	if len(y_true) == 0 or len(y_pred) == 0:
		return None

	ensure_dir(output_dir)
	y_true = np.asarray(y_true, dtype=int).ravel()
	y_pred = np.asarray(y_pred, dtype=int).ravel()
	cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

	fig, ax = plt.subplots(figsize=(5, 4.5))
	im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
	fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	ax.set(
		xticks=np.arange(2),
		yticks=np.arange(2),
		xticklabels=["0", "1"],
		yticklabels=["0", "1"],
		xlabel="Predicted label",
		ylabel="True label",
		title=f"{file_prefix} confusion matrix",
	)

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

	fig.tight_layout()
	plot_path = output_dir / f"{file_prefix}_confusion_matrix.png"
	fig.savefig(plot_path, dpi=150)
	plt.close(fig)
	print(f"[OK] Confusion matrix plot: {plot_path}")
	return plot_path


def compute_binary_classification_metrics(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	y_score: np.ndarray,
) -> dict:
	y_true = np.asarray(y_true, dtype=int).ravel()
	y_pred = np.asarray(y_pred, dtype=int).ravel()
	y_score = np.asarray(y_score, dtype=float).ravel()
	if len(y_true) != len(y_pred) or len(y_true) != len(y_score):
		raise ValueError(
			f"Classification metric length mismatch: "
			f"y_true={len(y_true)}, y_pred={len(y_pred)}, y_score={len(y_score)}"
		)

	metrics = {
		"test_accuracy": float(accuracy_score(y_true, y_pred)),
		"test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
	}
	if len(np.unique(y_true)) < 2:
		metrics["average_precision"] = None
		metrics["roc_auc"] = None
	else:
		metrics["average_precision"] = float(average_precision_score(y_true, y_score))
		metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
	return metrics


def rename_classification_metric_prefix(metrics: dict, prefix: str) -> dict:
	renamed = {}
	for key, value in metrics.items():
		if key.startswith("test_"):
			renamed[f"{prefix}_{key.removeprefix('test_')}"] = value
		else:
			renamed[f"{prefix}_{key}"] = value
	return renamed


def compute_multiclass_one_vs_rest_metric_summary(
	dataset_folder: str,
	class_labels: list,
	metric_filename: str,
) -> dict:
	metric_rows: list[dict] = []
	for class_label in class_labels:
		binary_dataset_folder = f"{class_label}_{dataset_folder}"
		metrics_path = (
			classification_output_dir(dataset_folder)
			/ binary_dataset_folder
			/ "metrics"
			/ metric_filename
		)
		if not metrics_path.exists():
			raise FileNotFoundError(f"Metric dosyasi bulunamadi: {metrics_path}")

		with open(metrics_path, "r", encoding="utf-8") as f:
			metrics = json.load(f)

		class_counts = metrics.get("class_counts", {})
		class_count = class_counts.get(str(class_label), class_counts.get(class_label, 1))
		metric_mse = metrics.get("test_mse")
		metric_rmse = metrics.get("test_rmse", compute_rmse_from_mse(metric_mse))
		row = {
			"class_label": class_label,
			"accuracy": float(metrics["test_accuracy"]),
			"precision": float(metrics["test_precision"]) if metrics.get("test_precision") is not None else None,
			"recall": float(metrics["test_recall"]) if metrics.get("test_recall") is not None else None,
			"f1": float(metrics["test_f1"]) if metrics.get("test_f1") is not None else None,
			"mse": float(metric_mse) if metric_mse is not None else None,
			"rmse": float(metric_rmse) if metric_rmse is not None else None,
			"class_count": int(class_count) if class_count is not None else 1,
			"metrics_path": str(metrics_path),
		}

		predictions_path = metrics.get("classification_predictions_path")
		if predictions_path:
			row["classification_predictions_path"] = predictions_path

		metric_rows.append(row)

	def weighted_average(metric_name: str) -> float | None:
		valid_rows = [row for row in metric_rows if row.get(metric_name) is not None]
		if not valid_rows:
			return None
		total_weight = sum(row["class_count"] for row in valid_rows)
		if total_weight == 0:
			return None
		return float(sum(row[metric_name] * row["class_count"] for row in valid_rows) / total_weight)

	return {
		"test_accuracy": weighted_average("accuracy"),
		"test_precision": weighted_average("precision"),
		"test_recall": weighted_average("recall"),
		"test_f1": weighted_average("f1"),
		"test_mse": weighted_average("mse"),
		"test_rmse": weighted_average("rmse"),
		"class_metric_rows": metric_rows,
	}


def save_classification_precision_recall_plot(
	y_true: np.ndarray,
	y_score: np.ndarray,
	output_dir: Path,
	file_prefix: str,
) -> Path | None:
	if len(y_true) == 0 or len(y_score) == 0:
		return None

	ensure_dir(output_dir)
	y_true = np.asarray(y_true, dtype=int).ravel()
	y_score = np.asarray(y_score, dtype=float).ravel()
	if len(y_true) != len(y_score):
		raise ValueError(f"PR curve length mismatch: y_true={len(y_true)}, y_score={len(y_score)}")
	if len(np.unique(y_true)) < 2:
		print(f"[WARN] PR curve cizilemedi, test setinde tek sinif var: {file_prefix}")
		return None

	precision, recall, _ = precision_recall_curve(y_true, y_score)
	average_precision = float(average_precision_score(y_true, y_score))

	plt.figure(figsize=(6, 5))
	plt.plot(recall, precision, linewidth=1.8, label=f"AP={average_precision:.4f}")
	plt.xlabel("Recall")
	plt.ylabel("Precision")
	plt.title(f"{file_prefix} precision-recall curve")
	plt.grid(True, alpha=0.3)
	plt.legend(loc="best")
	plt.tight_layout()

	plot_path = output_dir / f"{file_prefix}_precision_recall_curve.png"
	plt.savefig(plot_path, dpi=150)
	plt.close()
	print(f"[OK] Precision-recall plot: {plot_path}")
	return plot_path


def save_classification_roc_plot(
	y_true: np.ndarray,
	y_score: np.ndarray,
	output_dir: Path,
	file_prefix: str,
) -> Path | None:
	if len(y_true) == 0 or len(y_score) == 0:
		return None

	ensure_dir(output_dir)
	y_true = np.asarray(y_true, dtype=int).ravel()
	y_score = np.asarray(y_score, dtype=float).ravel()
	if len(y_true) != len(y_score):
		raise ValueError(f"ROC curve length mismatch: y_true={len(y_true)}, y_score={len(y_score)}")
	if len(np.unique(y_true)) < 2:
		print(f"[WARN] ROC curve cizilemedi, test setinde tek sinif var: {file_prefix}")
		return None

	fpr, tpr, _ = roc_curve(y_true, y_score)
	roc_auc = float(roc_auc_score(y_true, y_score))

	plt.figure(figsize=(6, 5))
	plt.plot(fpr, tpr, linewidth=1.8, label=f"AUC={roc_auc:.4f}")
	plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="Random")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title(f"{file_prefix} ROC curve")
	plt.grid(True, alpha=0.3)
	plt.legend(loc="best")
	plt.tight_layout()

	plot_path = output_dir / f"{file_prefix}_roc_curve.png"
	plt.savefig(plot_path, dpi=150)
	plt.close()
	print(f"[OK] ROC plot: {plot_path}")
	return plot_path


def extract_final_history_metric_values(history_frames: list[pd.DataFrame], metric_name: str) -> list[float]:
	values: list[float] = []
	for history_df in history_frames:
		if metric_name not in history_df.columns or history_df.empty:
			continue
		metric_values = history_df[metric_name].dropna()
		if metric_values.empty:
			continue
		values.append(float(metric_values.iloc[-1]))
	return values


def collect_repeated_run_history(
	dataset_folder: str,
	feature_percent: float,
	run_idx: int,
	task: str = "classification",
) -> pd.DataFrame | None:
	feature_percent_tag = format_feature_percent_tag(feature_percent)
	history_dir = task_output_dir(task, dataset_folder) / "training_history"
	candidates = [
		history_dir / f"top_{feature_percent_tag}_classifier_history.csv",
		history_dir / f"top_{feature_percent_tag}_regressor_history.csv",
		history_dir / f"chunked_top_{feature_percent_tag}_final_classifier_history.csv",
	]
	for history_path in candidates:
		if not history_path.exists():
			continue
		history_df = pd.read_csv(history_path)
		run_history_path = history_dir / f"run_{run_idx:03d}_{history_path.name}"
		history_df.to_csv(run_history_path, index=False)
		print(f"[OK] Run epoch history kaydedildi: {run_history_path}")
		return history_df

	print(f"[WARN] Run {run_idx} icin classifier history bulunamadi: {history_dir}")
	return None


def collect_repeated_multiclass_run_history(
	dataset_folder: str,
	feature_percent: float,
	run_idx: int,
) -> pd.DataFrame | None:
	metrics_path = (
		classification_output_dir(dataset_folder)
		/ "metrics"
		/ format_test_metrics_filename(feature_percent, dataset_folder)
	)
	if not metrics_path.exists():
		print(f"[WARN] Multiclass metrics dosyasi bulunamadi: {metrics_path}")
		return None

	with open(metrics_path, "r", encoding="utf-8") as f:
		metrics = json.load(f)

	if not metrics.get("macro_average"):
		return None

	class_labels = metrics.get("class_labels", [])
	if not class_labels:
		print(f"[WARN] Multiclass class_labels bos: {metrics_path}")
		return None

	class_history_frames: list[pd.DataFrame] = []
	for class_label in class_labels:
		binary_dataset_folder = f"{class_label}_{dataset_folder}"
		history_dir = classification_output_dir(dataset_folder) / binary_dataset_folder / "training_history"
		candidates = [
			history_dir / f"top_{feature_percent_tag}_classifier_history.csv",
			history_dir / f"chunked_top_{feature_percent_tag}_final_classifier_history.csv",
		]
		for history_path in candidates:
			if not history_path.exists():
				continue
			history_df = pd.read_csv(history_path)
			run_history_path = history_dir / f"run_{run_idx:03d}_{history_path.name}"
			history_df.to_csv(run_history_path, index=False)
			class_history_frames.append(history_df)
			break

	if not class_history_frames:
		print(f"[WARN] Run {run_idx} icin multiclass classifier history bulunamadi: {dataset_folder}")
		return None

	combined_history = pd.concat(class_history_frames, ignore_index=True)
	metric_columns = [col for col in ["accuracy", "val_accuracy", "loss", "val_loss"] if col in combined_history.columns]
	if "epoch" not in combined_history.columns or not metric_columns:
		return None

	macro_history = combined_history.groupby("epoch", as_index=False)[metric_columns].mean()
	history_dir = classification_output_dir(dataset_folder) / "training_history"
	ensure_dir(history_dir)
	run_macro_path = history_dir / f"run_{run_idx:03d}_top_{feature_percent_tag}_macro_classifier_history.csv"
	macro_history.to_csv(run_macro_path, index=False)
	print(f"[OK] Multiclass run macro epoch history kaydedildi: {run_macro_path}")
	return macro_history


def is_probable_regression_target(y: pd.Series) -> bool:
	"""
	Sürekli sayısal hedefleri classification class listesi gibi çalıştırmayı engeller.
	0/1 veya az sayıda tekrarlı integer label multiclass olarak kalır.
	"""
	y_nonnull = y.dropna()
	if y_nonnull.empty or not pd.api.types.is_numeric_dtype(y_nonnull):
		return False

	unique_count = int(y_nonnull.nunique())
	if unique_count <= 20:
		return False

	values = y_nonnull.astype(float).to_numpy()
	has_decimal_values = np.any(~np.isclose(values, np.round(values)))
	unique_ratio = unique_count / len(y_nonnull)
	singleton_ratio = float((y_nonnull.value_counts() == 1).mean())
	return bool(has_decimal_values and (unique_ratio > 0.1 or singleton_ratio > 0.5))


def unpack_processed_arrays(processed: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Unpack processed data and ensure consistent float32 dtype for TensorFlow."""
	X_train = processed["X_train_scaled"].astype(np.float32)
	X_test = processed["X_test_scaled"].astype(np.float32)
	y_train = processed["y_train"].to_numpy().astype(np.int32)
	y_test = processed["y_test"].to_numpy().astype(np.int32)
	
	# Validate data
	if np.isnan(X_train).any() or np.isinf(X_train).any():
		raise ValueError(f"X_train contains NaN/Inf values. X_train shape: {X_train.shape}")
	if np.isnan(X_test).any() or np.isinf(X_test).any():
		raise ValueError(f"X_test contains NaN/Inf values. X_test shape: {X_test.shape}")
	
	return X_train, X_test, y_train, y_test


def train_autoencoder_model(
	X_train_sub: np.ndarray,
	X_val: np.ndarray,
	X_eval: np.ndarray,
	encoding_dim: int,
	autoencoder_epochs: int = AUTOENCODER_EPOCHS,
	early_stopping_patience: int | None = None,
	early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	history_output_dir: Path | None = None,
	history_prefix: str | None = None,
	shuffle_training: bool = True,
	autoencoder_builder=build_sigmoid_autoencoder,
	autoencoder_activation: str = "sigmoid",
) -> tuple[float, tf.keras.Model, tf.keras.Model]:
	autoencoder, encoder = autoencoder_builder(
		input_dim=X_train_sub.shape[1],
		encoding_dim=encoding_dim,
		activation=autoencoder_activation,
	)
	
	# Ensure consistent dtypes
	X_train_sub = X_train_sub.astype(np.float32)
	X_val = X_val.astype(np.float32)

	callbacks: list[tf.keras.callbacks.Callback] = []
	if early_stopping_patience is not None and early_stopping_patience > 0:
		callbacks.append(
			tf.keras.callbacks.EarlyStopping(
				monitor="val_loss",
				patience=early_stopping_patience,
				min_delta=early_stopping_min_delta,
				restore_best_weights=True,
				mode="min",
				verbose=1,
			)
		)
	
	autoencoder_history = autoencoder.fit(
		X_train_sub,
		X_train_sub,
		validation_data=(X_val, X_val),
		epochs=autoencoder_epochs,
		batch_size=BATCH_SIZE,
		shuffle=shuffle_training,
		callbacks=callbacks,
		verbose=1,
	)
	if history_output_dir is not None and history_prefix is not None:
		save_training_history(
			history=autoencoder_history,
			output_dir=history_output_dir,
			file_prefix=f"{history_prefix}_autoencoder",
			plot_metrics=("loss",),
		)

	X_eval = X_eval.astype(np.float32)
	eval_mse = float(autoencoder.evaluate(X_eval, X_eval, verbose=0))
	return eval_mse, autoencoder, encoder


def train_encoder_from_raw_features(
	X_train_raw: pd.DataFrame,
	X_test_raw: pd.DataFrame,
	encoding_dim: int,
	random_state: int | None,
	autoencoder_early_stopping_patience: int | None,
	autoencoder_early_stopping_min_delta: float,
	y_train: np.ndarray | None = None,
	autoencoder_epochs: int = AUTOENCODER_EPOCHS,
	history_output_dir: Path | None = None,
	history_prefix: str | None = None,
	autoencoder_builder=build_sigmoid_autoencoder,
	autoencoder_activation: str = "sigmoid",
) -> tuple[float, tf.keras.Model, tf.keras.Model, StandardScaler, np.ndarray]:
	X_train, X_test, scaler = scale_data(X_train_raw, X_test_raw)
	X_train = X_train.astype(np.float32)
	X_test = X_test.astype(np.float32)
	if y_train is not None:
		X_train_sub, X_val, _, _ = train_test_split(
			X_train,
			y_train,
			test_size=CLASSIFIER_VALIDATION_SPLIT,
			random_state=random_state,
			shuffle=True,
			stratify=y_train,
		)
	else:
		X_train_sub, X_val = train_test_split(
			X_train,
			test_size=CLASSIFIER_VALIDATION_SPLIT,
			random_state=random_state,
			shuffle=True,
		)
	eval_mse, autoencoder, encoder = train_autoencoder_model(
		X_train_sub=X_train_sub,
		X_val=X_val,
		X_eval=X_test,
		encoding_dim=encoding_dim,
		autoencoder_epochs=autoencoder_epochs,
		early_stopping_patience=autoencoder_early_stopping_patience,
		early_stopping_min_delta=autoencoder_early_stopping_min_delta,
		history_output_dir=history_output_dir,
		history_prefix=history_prefix,
		shuffle_training=random_state is None,
		autoencoder_builder=autoencoder_builder,
		autoencoder_activation=autoencoder_activation,
	)
	return eval_mse, autoencoder, encoder, scaler, X_train_sub


def train_feature_ranking_autoencoder_for_selection(
	X_train: np.ndarray,
	X_test: np.ndarray,
	y_train: np.ndarray,
	encoding_dim: int,
	random_state: int | None,
	autoencoder_early_stopping_patience: int | None,
	autoencoder_early_stopping_min_delta: float,
	history_output_dir: Path | None = None,
	history_prefix: str | None = None,
) -> tuple[float, tf.keras.Model, np.ndarray]:
	"""
	FeatureRank icin autoencoder sadece feature importance/weight uretmek amaciyla egitilir.
	Final classifier dogrudan secilen feature'lar uzerinde calisir.
	"""
	X_train_sub, X_val, y_train_sub, _ = train_test_split(
		X_train,
		y_train,
		test_size=CLASSIFIER_VALIDATION_SPLIT,
		random_state=random_state,
		shuffle=True,
		stratify=y_train,
	)
	eval_mse, autoencoder, _ = train_autoencoder_model(
		X_train_sub=X_train_sub.astype(np.float32),
		X_val=X_val.astype(np.float32),
		X_eval=X_test.astype(np.float32),
		encoding_dim=encoding_dim,
		early_stopping_patience=autoencoder_early_stopping_patience,
		early_stopping_min_delta=autoencoder_early_stopping_min_delta,
		history_output_dir=history_output_dir,
		history_prefix=history_prefix,
		shuffle_training=random_state is None,
	)
	return eval_mse, autoencoder, X_train_sub


def save_encoded_dataset_from_encoder(
	encoder: tf.keras.Model,
	scaler: StandardScaler,
	X_all_raw: pd.DataFrame,
	y_all: pd.Series,
	target_column: str,
	output_path: Path,
	labeled_output_path: Path | None = None,
) -> Path:
	ensure_dir(output_path.parent)
	if not output_path.name.endswith("_data.csv"):
		raise ValueError(f"Encoded data dosyasi '_data.csv' ile bitmeli. Gelen: {output_path.name}")
	if hasattr(scaler, "feature_names_in_"):
		expected_features = list(scaler.feature_names_in_)
		missing_features = [feature for feature in expected_features if feature not in X_all_raw.columns]
		if missing_features:
			raise ValueError(f"Encoded dataset icin eksik feature var: {missing_features}")
		X_all_raw = X_all_raw[expected_features]
	X_all_scaled = scaler.transform(X_all_raw).astype(np.float32)
	X_all_encoded = encoder.predict(X_all_scaled, verbose=0).astype(np.float32)
	encoded_columns = [f"encoded_{idx + 1}" for idx in range(X_all_encoded.shape[1])]
	encoded_df = pd.DataFrame(X_all_encoded, columns=encoded_columns)
	encoded_df.to_csv(output_path, index=False, header=False)
	label_path = output_path.with_name(output_path.name.replace("_data.csv", "_label.csv"))
	pd.Series(y_all).reset_index(drop=True).to_frame().to_csv(label_path, index=False, header=False)
	print(f"[OK] Encoded label CSV kaydedildi: {label_path}")
	if labeled_output_path is not None:
		ensure_dir(labeled_output_path.parent)
		labeled_df = encoded_df.copy()
		labeled_df[target_column] = pd.Series(y_all).reset_index(drop=True)
		labeled_df.to_csv(labeled_output_path, index=False)
		print(f"[OK] Label eklenmis encoded CSV kaydedildi: {labeled_output_path}")
	return output_path


def save_dimension_reduced_classification_dataset(
	df: pd.DataFrame,
	dataset_folder: str,
	target_column: str,
	id_column: str | None,
	feature_percent: float,
	random_state: int | None,
	autoencoder_early_stopping_patience: int | None,
	autoencoder_early_stopping_min_delta: float,
	save_training_plots: bool = False,
) -> Path:
	feature_percent = validate_feature_percent(feature_percent)
	processed = preprocess_data(
		df,
		target_column=target_column,
		id_column=id_column,
		random_state=random_state,
		scale_features=False,
	)
	X_train_raw = processed["X_train"]
	X_test_raw = processed["X_test"]
	feature_names = X_train_raw.columns.tolist()
	original_feature_count = len(feature_names)
	if original_feature_count <= 0:
		raise ValueError("Dimension reduction icin en az bir feature olmali.")

	feature_percent_tag = format_feature_percent_tag(feature_percent)
	reduction_ratio = feature_percent / 100.0
	latent_dimension = max(1, int(original_feature_count * reduction_ratio))
	hidden_dimension = compute_dimension_reduction_hidden_dim(original_feature_count, latent_dimension)
	dimension_reduction_epochs = AUTOENCODER_EPOCHS
	dimension_reduction_hidden_activation = "relu"

	output_dir = classification_output_dir(dataset_folder)
	encoded_output_dir = output_dir / "dimension_reduction"
	history_dir = output_dir / "training_history"
	ensure_dir(output_dir)
	ensure_dir(encoded_output_dir)
	if save_training_plots:
		ensure_dir(history_dir)

	print(f"\n[INFO] Dataset: {dataset_folder}")
	print(f"[INFO] Original Feature Count: {original_feature_count}")
	print(f"[INFO] Reduction Ratio: {feature_percent:g}%")
	print(f"[INFO] Encoder Output Dimension: {latent_dimension}")
	print(f"[INFO] Symmetric Hidden Dimension: {hidden_dimension}")
	print(f"[INFO] Dimension Reduction Epochs: {dimension_reduction_epochs}")
	print("[INFO] Training Autoencoder...")

	reconstruction_mse, _, encoder, scaler, _ = train_encoder_from_raw_features(
		X_train_raw=X_train_raw,
		X_test_raw=X_test_raw,
		encoding_dim=latent_dimension,
		random_state=random_state,
		y_train=processed["y_train"].to_numpy().astype(np.int32),
		autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
		autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
		autoencoder_epochs=dimension_reduction_epochs,
		history_output_dir=history_dir if save_training_plots else None,
		history_prefix=f"dimension_reduction_{feature_percent_tag}" if save_training_plots else None,
		autoencoder_builder=build_dimension_reduction_autoencoder,
		autoencoder_activation=dimension_reduction_hidden_activation,
	)
	reconstruction_rmse = float(np.sqrt(reconstruction_mse))

	prepared_df = drop_id_column(df.copy(), id_column=id_column)
	prepared_df = encode_target(prepared_df, target_column=target_column)
	X_all_raw, y_all = split_features_target(prepared_df, target_column=target_column)
	X_all_raw = handle_pid_unrealistic_zeros(X_all_raw)
	X_all_raw = keep_numeric_features_only(X_all_raw)
	X_all_raw = X_all_raw[feature_names]

	encoded_filename = f"{format_encoded_dataset_stem(dataset_folder, feature_percent, latent_dimension)}.csv"
	raw_encoded_path = Path("data") / "raw" / encoded_filename
	output_encoded_path = encoded_output_dir / encoded_filename
	labeled_output_path = encoded_output_dir / encoded_filename.replace("_data.csv", "_with_label.csv")
	save_encoded_dataset_from_encoder(
		encoder=encoder,
		scaler=scaler,
		X_all_raw=X_all_raw,
		y_all=y_all,
		target_column=target_column,
		output_path=raw_encoded_path,
	)
	save_encoded_dataset_from_encoder(
		encoder=encoder,
		scaler=scaler,
		X_all_raw=X_all_raw,
		y_all=y_all,
		target_column=target_column,
		output_path=output_encoded_path,
		labeled_output_path=labeled_output_path,
	)
	metadata_path = encoded_output_dir / f"{Path(encoded_filename).stem}_metadata.json"
	save_json(
		{
			"dataset": dataset_folder,
			"task": "dimension_reduction",
			"reduction_percent": feature_percent,
			"reduction_ratio": reduction_ratio,
			"original_feature_count": original_feature_count,
			"latent_dimension": latent_dimension,
			"symmetric_hidden_dimension": hidden_dimension,
			"hidden_activation": dimension_reduction_hidden_activation,
			"latent_activation": "linear",
			"decoder_output_activation": "linear",
			"optimizer": "Adam",
			"optimizer_learning_rate": 0.0001,
			"optimizer_clipnorm": 1.0,
			"autoencoder_epochs": dimension_reduction_epochs,
			"encoded_dataset_shape": [int(len(X_all_raw)), int(latent_dimension)],
			"autoencoder_architecture": "input_to_encoder_hidden_to_latent_to_decoder_hidden_to_linear_reconstruction",
			"feature_ranking_used": False,
			"dimension_reduction_mse": reconstruction_mse,
			"dimension_reduction_rmse": reconstruction_rmse,
			"autoencoder_reconstruction_mse": reconstruction_mse,
			"autoencoder_reconstruction_rmse": reconstruction_rmse,
			"raw_encoded_dataset_path": str(raw_encoded_path),
			"raw_label_path": str(raw_encoded_path.with_name(raw_encoded_path.name.replace("_data.csv", "_label.csv"))),
			"output_encoded_dataset_path": str(output_encoded_path),
			"output_labeled_dataset_path": str(labeled_output_path),
		},
		metadata_path,
	)
	print(f"[INFO] Encoded Dataset Shape: ({len(X_all_raw)}, {latent_dimension})")
	print(f"[OK] Dimension reduction CSV kaydedildi: {raw_encoded_path}")
	print(f"[OK] Dimension reduction kopyasi: {output_encoded_path}")
	print(f"[OK] Dimension reduction metadata: {metadata_path}")
	return raw_encoded_path


def train_and_evaluate_pipeline(
	X_train: np.ndarray,
	X_test: np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
	encoding_dim: int,
	random_state: int | None,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
	classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
	classifier_early_stopping_patience: int | None = None,
	autoencoder_early_stopping_patience: int | None = None,
	classifier_early_stopping_monitor: str = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR,
	classifier_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	autoencoder_early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	classifier_class_weight: str = "none",
	classifier_sampling: str = "none",
	history_output_dir: Path | None = None,
	history_prefix: str | None = None,
	return_train_predictions: bool = False,
) -> tuple[float, float, tf.keras.Model, tf.keras.Model, np.ndarray, np.ndarray, np.ndarray]:
	X_train_sub, X_val, y_train_sub, y_val = train_test_split(
		X_train,
		y_train,
		test_size=CLASSIFIER_VALIDATION_SPLIT,
		random_state=random_state,
		shuffle=True,
		stratify=y_train,
	)
	X_train_sub, y_train_sub = undersample_binary_training_data(
		X_train=X_train_sub,
		y_train=y_train_sub,
		random_state=random_state,
		mode=classifier_sampling,
	)

	test_mse, autoencoder, encoder = train_autoencoder_model(
		X_train_sub=X_train_sub,
		X_val=X_val,
		X_eval=X_test,
		encoding_dim=encoding_dim,
		early_stopping_patience=autoencoder_early_stopping_patience,
		early_stopping_min_delta=autoencoder_early_stopping_min_delta,
		history_output_dir=history_output_dir,
		history_prefix=history_prefix,
		shuffle_training=random_state is None,
	)

	X_train_encoded = encoder.predict(X_train_sub, verbose=0).astype(np.float32)
	X_val_encoded = encoder.predict(X_val, verbose=0).astype(np.float32)
	X_test_encoded = encoder.predict(X_test, verbose=0).astype(np.float32)

	# Get encoded dimension for validation
	encoder_output_dim = X_train_encoded.shape[1]

	# Verify input/output shapes match model expectations
	if X_train_encoded.shape[1] != encoder_output_dim:
		raise ValueError(
			f"Encoder output dim {X_train_encoded.shape[1]} != expected dim {encoder_output_dim}"
		)

	classifier_model = str(classifier_model).strip().lower()
	class_weight = compute_binary_class_weight(y_train_sub, classifier_class_weight)
	if classifier_model == "neural":
		classifier = build_latent_classifier(
			input_dim=encoder_output_dim,
			hidden_units=classifier_hidden_units,
			dropout_rates=classifier_dropout_rates,
			learning_rate=classifier_learning_rate,
		)
		y_train_fit = y_train_sub.astype(np.float32)
		y_val_fit = y_val.astype(np.float32)

		callbacks: list[tf.keras.callbacks.Callback] = []
		if classifier_early_stopping_patience is not None and classifier_early_stopping_patience > 0:
			if classifier_early_stopping_monitor not in {"val_loss", "val_accuracy"}:
				raise ValueError("classifier early stopping monitor 'val_loss' veya 'val_accuracy' olmali.")
			monitor_mode = "max" if classifier_early_stopping_monitor == "val_accuracy" else "min"
			callbacks.append(
				tf.keras.callbacks.EarlyStopping(
					monitor=classifier_early_stopping_monitor,
					patience=classifier_early_stopping_patience,
					min_delta=classifier_early_stopping_min_delta,
					restore_best_weights=True,
					mode=monitor_mode,
					verbose=1,
				)
			)

		classifier_history = classifier.fit(
			X_train_encoded,
			y_train_fit,
			epochs=classifier_epochs,
			batch_size=BATCH_SIZE,
			validation_data=(X_val_encoded, y_val_fit),
			class_weight=class_weight,
			shuffle=random_state is None,
			callbacks=callbacks,
			verbose=1,
		)
		if history_output_dir is not None and history_prefix is not None:
			save_training_history(
				history=classifier_history,
				output_dir=history_output_dir,
				file_prefix=f"{history_prefix}_classifier",
				plot_metrics=("accuracy", "loss"),
			)
		y_pred_prob = classifier.predict(X_test_encoded, verbose=0)
		if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 1:
			y_pred_prob = y_pred_prob.ravel()
		print("classifier output shape:", classifier.output_shape)
	else:
		classifier = build_sklearn_classifier(
			classifier_model=classifier_model,
			random_state=random_state,
			class_weight=class_weight,
		)
		classifier.fit(X_train_encoded, y_train_sub.astype(int))
		y_pred_prob = predict_binary_scores(classifier, X_test_encoded)
		print(f"classifier model: {classifier_model}")

	# Handle both single-output (sigmoid) and multi-output predictions
	y_pred = (y_pred_prob > THRESHOLD).astype(int).ravel()
	
	if len(y_pred) != len(y_test):
		raise ValueError(
			f"Prediction length {len(y_pred)} != y_test length {len(y_test)}"
		)
		
	test_accuracy = float(accuracy_score(y_test.astype(int), y_pred))
	if return_train_predictions:
		X_train_eval_encoded = encoder.predict(X_train.astype(np.float32), verbose=0).astype(np.float32)
		if classifier_model == "neural":
			y_train_score = classifier.predict(X_train_eval_encoded, verbose=0)
			if y_train_score.ndim == 2 and y_train_score.shape[1] == 1:
				y_train_score = y_train_score.ravel()
		else:
			y_train_score = predict_binary_scores(classifier, X_train_eval_encoded)
		y_train_pred = (np.asarray(y_train_score).ravel() > THRESHOLD).astype(int)
		return (
			test_mse,
			test_accuracy,
			autoencoder,
			encoder,
			X_train_sub,
			y_pred,
			y_pred_prob.ravel(),
			y_train_pred.ravel(),
			np.asarray(y_train_score, dtype=np.float32).ravel(),
		)
	return test_mse, test_accuracy, autoencoder, encoder, X_train_sub, y_pred, y_pred_prob.ravel()


def train_and_evaluate_direct_classifier(
	X_train: np.ndarray,
	X_test: np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
	random_state: int | None,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
	classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
	classifier_early_stopping_patience: int | None = None,
	classifier_early_stopping_monitor: str = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR,
	classifier_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	classifier_class_weight: str = "none",
	classifier_sampling: str = "none",
	history_output_dir: Path | None = None,
	history_prefix: str | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
	X_train_sub, X_val, y_train_sub, y_val = train_test_split(
		X_train,
		y_train,
		test_size=CLASSIFIER_VALIDATION_SPLIT,
		random_state=random_state,
		shuffle=True,
		stratify=y_train,
	)
	X_train_sub, y_train_sub = undersample_binary_training_data(
		X_train=X_train_sub,
		y_train=y_train_sub,
		random_state=random_state,
		mode=classifier_sampling,
	)

	classifier_model = str(classifier_model).strip().lower()
	class_weight = compute_binary_class_weight(y_train_sub, classifier_class_weight)
	if classifier_model == "neural":
		classifier = build_latent_classifier(
			input_dim=X_train.shape[1],
			hidden_units=classifier_hidden_units,
			dropout_rates=classifier_dropout_rates,
			learning_rate=classifier_learning_rate,
		)
		callbacks: list[tf.keras.callbacks.Callback] = []
		if classifier_early_stopping_patience is not None and classifier_early_stopping_patience > 0:
			if classifier_early_stopping_monitor not in {"val_loss", "val_accuracy"}:
				raise ValueError("classifier early stopping monitor 'val_loss' veya 'val_accuracy' olmali.")
			monitor_mode = "max" if classifier_early_stopping_monitor == "val_accuracy" else "min"
			callbacks.append(
				tf.keras.callbacks.EarlyStopping(
					monitor=classifier_early_stopping_monitor,
					patience=classifier_early_stopping_patience,
					min_delta=classifier_early_stopping_min_delta,
					restore_best_weights=True,
					mode=monitor_mode,
					verbose=1,
				)
			)
		classifier_history = classifier.fit(
			X_train_sub.astype(np.float32),
			y_train_sub.astype(np.float32),
			epochs=classifier_epochs,
			batch_size=BATCH_SIZE,
			validation_data=(X_val.astype(np.float32), y_val.astype(np.float32)),
			class_weight=class_weight,
			shuffle=random_state is None,
			callbacks=callbacks,
			verbose=1,
		)
		if history_output_dir is not None and history_prefix is not None:
			save_training_history(
				history=classifier_history,
				output_dir=history_output_dir,
				file_prefix=f"{history_prefix}_direct_classifier",
				plot_metrics=("accuracy", "loss"),
			)
		y_score = classifier.predict(X_test.astype(np.float32), verbose=0)
		if y_score.ndim == 2 and y_score.shape[1] == 1:
			y_score = y_score.ravel()
		print("direct classifier output shape:", classifier.output_shape)
	else:
		classifier = build_sklearn_classifier(
			classifier_model=classifier_model,
			random_state=random_state,
			class_weight=class_weight,
		)
		classifier.fit(X_train_sub, y_train_sub.astype(int))
		y_score = predict_binary_scores(classifier, X_test)
		print(f"direct classifier model: {classifier_model}")

	y_score = np.asarray(y_score, dtype=np.float32).ravel()
	y_pred = (y_score > THRESHOLD).astype(int)
	if len(y_pred) != len(y_test):
		raise ValueError(f"Prediction length {len(y_pred)} != y_test length {len(y_test)}")
	test_accuracy = float(accuracy_score(y_test.astype(int), y_pred))
	return test_accuracy, y_pred, y_score


def train_and_evaluate_direct_multiclass_classifier(
	X_train: np.ndarray,
	X_test: np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
	random_state: int | None,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
	classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
	classifier_early_stopping_patience: int | None = None,
	classifier_early_stopping_monitor: str = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR,
	classifier_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	history_output_dir: Path | None = None,
	history_prefix: str | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
	X_train_sub, X_val, y_train_sub, y_val = train_test_split(
		X_train,
		y_train,
		test_size=CLASSIFIER_VALIDATION_SPLIT,
		random_state=random_state,
		shuffle=True,
		stratify=y_train,
	)
	num_classes = int(len(np.unique(y_train)))
	classifier_model = str(classifier_model).strip().lower()
	if classifier_model == "neural":
		classifier = build_latent_classifier(
			input_dim=X_train.shape[1],
			num_classes=num_classes,
			hidden_units=classifier_hidden_units,
			dropout_rates=classifier_dropout_rates,
			learning_rate=classifier_learning_rate,
		)
		callbacks: list[tf.keras.callbacks.Callback] = []
		if classifier_early_stopping_patience is not None and classifier_early_stopping_patience > 0:
			monitor_mode = "max" if classifier_early_stopping_monitor == "val_accuracy" else "min"
			callbacks.append(
				tf.keras.callbacks.EarlyStopping(
					monitor=classifier_early_stopping_monitor,
					patience=classifier_early_stopping_patience,
					min_delta=classifier_early_stopping_min_delta,
					restore_best_weights=True,
					mode=monitor_mode,
					verbose=1,
				)
			)
		history = classifier.fit(
			X_train_sub.astype(np.float32),
			y_train_sub.astype(np.int32),
			epochs=classifier_epochs,
			batch_size=BATCH_SIZE,
			validation_data=(X_val.astype(np.float32), y_val.astype(np.int32)),
			shuffle=random_state is None,
			callbacks=callbacks,
			verbose=1,
		)
		if history_output_dir is not None and history_prefix is not None:
			save_training_history(
				history=history,
				output_dir=history_output_dir,
				file_prefix=f"{history_prefix}_direct_classifier",
				plot_metrics=("accuracy", "loss"),
			)
		y_score = np.asarray(classifier.predict(X_test.astype(np.float32), verbose=0), dtype=np.float32)
		y_pred = np.argmax(y_score, axis=1).astype(int)
	else:
		classifier = build_sklearn_classifier(
			classifier_model=classifier_model,
			random_state=random_state,
			class_weight=None,
		)
		classifier.fit(X_train_sub, y_train_sub.astype(int))
		y_pred = np.asarray(classifier.predict(X_test), dtype=int).ravel()
		if hasattr(classifier, "predict_proba"):
			y_score = np.asarray(classifier.predict_proba(X_test), dtype=np.float32)
		else:
			y_score = np.empty((len(y_pred), 0), dtype=np.float32)

	return float(accuracy_score(y_test.astype(int), y_pred)), y_pred, y_score


def run_dimension_reduction_classification_experiment(
	df: pd.DataFrame,
	dataset_folder: str,
	target_column: str,
	id_column: str | None,
	encoding_dim: int,
	feature_percent: float,
	random_state: int | None,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
	classifier_model: str,
	classifier_early_stopping_patience: int | None,
	autoencoder_early_stopping_patience: int | None,
	classifier_early_stopping_monitor: str,
	classifier_early_stopping_min_delta: float,
	autoencoder_early_stopping_min_delta: float,
	classifier_class_weight: str,
	classifier_sampling: str,
	save_training_plots: bool,
	current_class_label: int | None = None,
	class_counts: dict[int, int] | None = None,
) -> tuple[float, float]:
	"""Evaluate latent features without exposing the outer test split to model fitting."""
	start_time = time.perf_counter()
	feature_percent = validate_feature_percent(feature_percent)
	processed = preprocess_data(
		df,
		target_column=target_column,
		id_column=id_column,
		random_state=random_state,
		scale_features=False,
	)
	X_train_raw = processed["X_train"]
	X_test_raw = processed["X_test"]
	y_train = processed["y_train"].to_numpy().astype(np.int32)
	y_test = processed["y_test"].to_numpy().astype(np.int32)
	original_feature_count = int(X_train_raw.shape[1])
	latent_dimension = max(1, int(original_feature_count * feature_percent / 100.0))
	feature_percent_tag = format_feature_percent_tag(feature_percent)
	file_prefix = f"top_{feature_percent_tag}_dimension_reduction"

	output_dir = classification_output_dir(dataset_folder)
	metrics_dir = output_dir / "metrics"
	history_dir = output_dir / "training_history"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)
	if save_training_plots:
		ensure_dir(history_dir)

	print("[INFO] Leakage-free dimension reduction classification basladi.")
	print(f"[INFO] Outer train/test: {len(X_train_raw)}/{len(X_test_raw)}")
	print(f"[INFO] Original/latent feature: {original_feature_count}/{latent_dimension}")
	print(f"[INFO] Autoencoder epoch (tum yuzdelerde sabit): {AUTOENCODER_EPOCHS}")

	reconstruction_mse, _, encoder, scaler, _ = train_encoder_from_raw_features(
		X_train_raw=X_train_raw,
		X_test_raw=X_test_raw,
		encoding_dim=latent_dimension,
		random_state=random_state,
		y_train=y_train,
		autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
		autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
		autoencoder_epochs=AUTOENCODER_EPOCHS,
		history_output_dir=history_dir if save_training_plots else None,
		history_prefix=file_prefix if save_training_plots else None,
		autoencoder_builder=build_dimension_reduction_autoencoder,
		autoencoder_activation="relu",
	)
	X_train_scaled = scaler.transform(X_train_raw).astype(np.float32)
	X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
	X_train_encoded = encoder.predict(X_train_scaled, verbose=0).astype(np.float32)
	X_test_encoded = encoder.predict(X_test_scaled, verbose=0).astype(np.float32)
	encoded_scaler = StandardScaler()
	X_train_encoded = encoded_scaler.fit_transform(X_train_encoded).astype(np.float32)
	X_test_encoded = encoded_scaler.transform(X_test_encoded).astype(np.float32)

	class_count = int(len(np.unique(y_train)))
	if class_count != 2:
		raise ValueError(
			"run_dimension_reduction_classification_experiment binary etiket bekliyor. "
			"Multiclass icin one-vs-rest akis kullanilmali."
		)
	(
		_classifier_test_mse,
		test_accuracy,
		_classifier_autoencoder,
		_classifier_encoder,
		_classifier_train_sub,
		y_pred,
		y_score,
	) = train_and_evaluate_pipeline(
		X_train=X_train_encoded,
		X_test=X_test_encoded,
		y_train=y_train,
		y_test=y_test,
		encoding_dim=encoding_dim,
		random_state=random_state,
		classifier_epochs=classifier_epochs,
		classifier_hidden_units=classifier_hidden_units,
		classifier_dropout_rates=classifier_dropout_rates,
		classifier_learning_rate=classifier_learning_rate,
		classifier_model=classifier_model,
		classifier_early_stopping_patience=classifier_early_stopping_patience,
		autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
		classifier_early_stopping_monitor=classifier_early_stopping_monitor,
		classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
		autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
		classifier_class_weight=classifier_class_weight,
		classifier_sampling=classifier_sampling,
		history_output_dir=history_dir if save_training_plots else None,
		history_prefix=file_prefix if save_training_plots else None,
	)
	metrics_data = compute_binary_classification_metrics(y_test, y_pred, y_score)

	predictions_path = output_dir / f"{file_prefix}_classification_predictions.csv"
	pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(predictions_path, index=False)
	metrics_data.update(
		{
			"task": "classification",
			"method": "DimensionReduction",
			"feature_percent": feature_percent,
			"original_feature_count": original_feature_count,
			"selected_feature_count": latent_dimension,
			"latent_feature_count": latent_dimension,
			"class_count": class_count,
			"split_seed": random_state,
			"outer_train_count": int(len(X_train_raw)),
			"outer_test_count": int(len(X_test_raw)),
			"scaler_fit_scope": "outer_train_only",
			"encoded_scaler_fit_scope": "encoded_outer_train_only",
			"autoencoder_fit_scope": "outer_train_only",
			"classifier_fit_scope": "encoded_outer_train_only",
			"test_seen_during_fit": False,
			"autoencoder_epochs": AUTOENCODER_EPOCHS,
			"dimension_reduction_mse": reconstruction_mse,
			"dimension_reduction_rmse": float(np.sqrt(reconstruction_mse)),
			"autoencoder_reconstruction_mse": reconstruction_mse,
			"autoencoder_reconstruction_rmse": float(np.sqrt(reconstruction_mse)),
			"classifier_model": classifier_model,
			"elapsed_seconds": time.perf_counter() - start_time,
			"classification_predictions_path": str(predictions_path),
		}
	)
	if current_class_label is not None and class_counts is not None:
		metrics_data["current_class_label"] = current_class_label
		metrics_data["class_counts"] = class_counts
		metrics_data["binary_label_counts"] = {
			"label_0": int(np.sum(y_test == 0)),
			"label_1": int(np.sum(y_test == 1)),
		}
	metrics_path = metrics_dir / f"{file_prefix}_test_metrics.json"
	save_json(metrics_data, metrics_path)
	print(f"[OK] Leakage-free dimension reduction test_accuracy: {test_accuracy:.6f}")
	print(f"[OK] Dimension reduction metrik dosyasi: {metrics_path}")
	return reconstruction_mse, test_accuracy


def run_dimension_reduction_multiclass_one_vs_rest(
	df: pd.DataFrame,
	dataset_folder: str,
	target_column: str,
	id_column: str | None,
	encoding_dim: int,
	feature_percent: float,
	random_state: int | None,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
	classifier_model: str,
	classifier_early_stopping_patience: int | None,
	autoencoder_early_stopping_patience: int | None,
	classifier_early_stopping_monitor: str,
	classifier_early_stopping_min_delta: float,
	autoencoder_early_stopping_min_delta: float,
	classifier_class_weight: str,
	classifier_sampling: str,
	save_training_plots: bool,
) -> tuple[float, float]:
	class_labels = sorted(df[target_column].dropna().unique().tolist())
	if len(class_labels) <= 2:
		raise ValueError("run_dimension_reduction_multiclass_one_vs_rest sadece 2'den fazla sinif icin kullanilmali.")

	print(f"[INFO] Dimension Reduction multi-class tespit edildi. Siniflar: {class_labels}")
	class_counts = {label: int((df[target_column] == label).sum()) for label in class_labels}
	print(f"[INFO] Dataset class sayilari: {class_counts}")

	for class_label in class_labels:
		binary_df = df.copy()
		binary_df[target_column] = (binary_df[target_column] != class_label).astype(np.int32)
		binary_dataset_folder = f"{class_label}_{dataset_folder}"
		nested_binary_folder = str(Path(dataset_folder) / binary_dataset_folder)

		print(f"\n[INFO] Dimension Reduction one-vs-rest basliyor: class={class_label}, klasor={binary_dataset_folder}")
		run_dimension_reduction_classification_experiment(
			df=binary_df,
			dataset_folder=nested_binary_folder,
			target_column=target_column,
			id_column=id_column,
			encoding_dim=encoding_dim,
			feature_percent=feature_percent,
			random_state=random_state,
			classifier_epochs=classifier_epochs,
			classifier_hidden_units=classifier_hidden_units,
			classifier_dropout_rates=classifier_dropout_rates,
			classifier_learning_rate=classifier_learning_rate,
			classifier_model=classifier_model,
			classifier_early_stopping_patience=classifier_early_stopping_patience,
			autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor=classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
			classifier_class_weight=classifier_class_weight,
			classifier_sampling=classifier_sampling,
			save_training_plots=save_training_plots,
			current_class_label=class_label,
			class_counts=class_counts,
		)

	output_feature_percent = format_encoded_output_percent(feature_percent, dataset_folder)
	output_feature_percent_tag = format_feature_percent_tag(output_feature_percent)
	output_feature_label = format_feature_output_label(feature_percent, dataset_folder)
	filtered_metric_filename = f"top_{output_feature_percent_tag}_dimension_reduction_test_metrics.json"
	filtered_summary_metrics = compute_multiclass_one_vs_rest_metric_summary(
		dataset_folder=dataset_folder,
		class_labels=class_labels,
		metric_filename=filtered_metric_filename,
	)
	macro_filtered_accuracy = float(filtered_summary_metrics["test_accuracy"])

	output_dir = classification_output_dir(dataset_folder)
	metrics_dir = output_dir / "metrics"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)

	filtered_class_metric_rows = filtered_summary_metrics.pop("class_metric_rows", [])
	filtered_class_metrics_path = metrics_dir / f"top_{output_feature_percent_tag}_dimension_reduction_multiclass_class_metrics.csv"
	if filtered_class_metric_rows:
		pd.DataFrame(filtered_class_metric_rows).to_csv(filtered_class_metrics_path, index=False)

	filtered_multiclass_metrics_data = {
		"task": "classification",
		"method": "DimensionReduction",
		"feature_percent": output_feature_percent,
		"num_classes": len(class_labels),
		"class_labels": class_labels,
		"macro_average": True,
		"class_metrics_path": str(filtered_class_metrics_path) if filtered_class_metric_rows else None,
		**filtered_summary_metrics,
	}
	add_encoded_metric_metadata(filtered_multiclass_metrics_data, feature_percent, dataset_folder)
	save_json(filtered_multiclass_metrics_data, metrics_dir / filtered_metric_filename)

	print("\n[OK] Dimension Reduction multi-class one-vs-rest tamamlandi.")
	print(f"[OK] {output_feature_label} macro test_accuracy: {macro_filtered_accuracy:.6f}")
	if filtered_summary_metrics.get("test_precision") is not None:
		print(f"[OK] {output_feature_label} macro test_precision: {float(filtered_summary_metrics['test_precision']):.6f}")
	if filtered_summary_metrics.get("test_recall") is not None:
		print(f"[OK] {output_feature_label} macro test_recall: {float(filtered_summary_metrics['test_recall']):.6f}")
	if filtered_summary_metrics.get("test_f1") is not None:
		print(f"[OK] {output_feature_label} macro test_f1: {float(filtered_summary_metrics['test_f1']):.6f}")
	print(f"[OK] Metrik dosyasi: {metrics_dir / filtered_metric_filename}")
	if filtered_class_metric_rows:
		print(f"[OK] Sinif bazli multiclass metrik CSV: {filtered_class_metrics_path}")
	return macro_filtered_accuracy, macro_filtered_accuracy


def train_and_evaluate_regression_pipeline(
	X_train: np.ndarray,
	X_test: np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
	encoding_dim: int,
	random_state: int | None,
	regressor_epochs: int,
	regressor_hidden_units: tuple[int, ...],
	regressor_dropout_rates: tuple[float, ...] | None,
	regressor_learning_rate: float,
	regression_model: str = DEFAULT_REGRESSION_MODEL,
	svr_kernel: str = DEFAULT_SVR_KERNEL,
	svr_c: float = DEFAULT_SVR_C,
	svr_epsilon: float = DEFAULT_SVR_EPSILON,
	svr_gamma: str | float = DEFAULT_SVR_GAMMA,
	kmeans_regression_clusters: int = DEFAULT_KMEANS_REGRESSION_CLUSTERS,
	kmeans_regression_n_init: int = DEFAULT_KMEANS_REGRESSION_N_INIT,
	regressor_early_stopping_patience: int | None = None,
	autoencoder_early_stopping_patience: int | None = None,
	regressor_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	autoencoder_early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	history_output_dir: Path | None = None,
	history_prefix: str | None = None,
) -> tuple[dict, object, tf.keras.Model, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	regression_model = regression_model.lower().strip()
	if regression_model not in {"neural", "svr", "kmeans"}:
		raise ValueError("regression_model 'neural', 'svr' veya 'kmeans' olmali.")

	X_train_sub, X_val, y_train_sub, y_val = train_test_split(
		X_train,
		y_train,
		test_size=CLASSIFIER_VALIDATION_SPLIT,
		random_state=random_state,
		shuffle=True,
	)
	y_scaler = StandardScaler()
	y_train_scaled = y_scaler.fit_transform(np.asarray(y_train_sub).reshape(-1, 1)).ravel().astype(np.float32)
	y_val_scaled = y_scaler.transform(np.asarray(y_val).reshape(-1, 1)).ravel().astype(np.float32)

	autoencoder_mse, autoencoder, encoder = train_autoencoder_model(
		X_train_sub=X_train_sub,
		X_val=X_val,
		X_eval=X_test,
		encoding_dim=encoding_dim,
		early_stopping_patience=autoencoder_early_stopping_patience,
		early_stopping_min_delta=autoencoder_early_stopping_min_delta,
		history_output_dir=history_output_dir,
		history_prefix=history_prefix,
	)

	X_train_encoded = encoder.predict(X_train_sub, verbose=0).astype(np.float32)
	X_val_encoded = encoder.predict(X_val, verbose=0).astype(np.float32)
	X_test_encoded = encoder.predict(X_test, verbose=0).astype(np.float32)

	if regression_model == "svr":
		regressor = SVR(
			kernel=svr_kernel,
			C=svr_c,
			epsilon=svr_epsilon,
			gamma=svr_gamma,
		)
		regressor.fit(X_train_encoded, y_train_scaled)
		y_train_pred_scaled = regressor.predict(X_train_encoded).ravel()
		y_pred_scaled = regressor.predict(X_test_encoded).ravel()
	elif regression_model == "kmeans":
		if kmeans_regression_clusters < 1:
			raise ValueError("kmeans-regression-clusters en az 1 olmali.")
		effective_cluster_count = min(kmeans_regression_clusters, X_train_encoded.shape[0])
		regressor = KMeans(
			n_clusters=effective_cluster_count,
			random_state=random_state,
			n_init=kmeans_regression_n_init,
		)
		train_cluster_labels = regressor.fit_predict(X_train_encoded)
		global_target_mean = float(np.mean(y_train_scaled))
		cluster_target_means = {}
		for cluster_id in range(effective_cluster_count):
			cluster_values = y_train_scaled[train_cluster_labels == cluster_id]
			cluster_target_means[cluster_id] = (
				float(np.mean(cluster_values)) if len(cluster_values) > 0 else global_target_mean
			)
		test_cluster_labels = regressor.predict(X_test_encoded)
		y_train_pred_scaled = np.asarray(
			[cluster_target_means[int(cluster_id)] for cluster_id in train_cluster_labels],
			dtype=np.float32,
		)
		y_pred_scaled = np.asarray(
			[cluster_target_means[int(cluster_id)] for cluster_id in test_cluster_labels],
			dtype=np.float32,
		)
	else:
		regressor = build_latent_regressor(
			input_dim=X_train_encoded.shape[1],
			hidden_units=regressor_hidden_units,
			dropout_rates=regressor_dropout_rates,
			learning_rate=regressor_learning_rate,
		)

		callbacks: list[tf.keras.callbacks.Callback] = []
		if regressor_early_stopping_patience is not None and regressor_early_stopping_patience > 0:
			callbacks.append(
				tf.keras.callbacks.EarlyStopping(
					monitor="val_loss",
					patience=regressor_early_stopping_patience,
					min_delta=regressor_early_stopping_min_delta,
					restore_best_weights=True,
					mode="min",
					verbose=1,
				)
			)

		regressor_history = regressor.fit(
			X_train_encoded,
			y_train_scaled,
			epochs=regressor_epochs,
			batch_size=BATCH_SIZE,
			validation_data=(X_val_encoded, y_val_scaled),
			shuffle=random_state is None,
			callbacks=callbacks,
			verbose=1,
		)
		if history_output_dir is not None and history_prefix is not None:
			save_training_history(
				history=regressor_history,
				output_dir=history_output_dir,
				file_prefix=f"{history_prefix}_regressor",
				plot_metrics=("mae", "loss"),
			)
		y_train_pred_scaled = regressor.predict(X_train_encoded, verbose=0).ravel()
		y_pred_scaled = regressor.predict(X_test_encoded, verbose=0).ravel()

	y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
	y_train_true = y_train_sub.astype(np.float32).ravel()
	y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
	y_true = y_test.astype(np.float32).ravel()
	regression_mse = float(mean_squared_error(y_true, y_pred))
	regression_rmse = float(np.sqrt(regression_mse))
	regression_mae = float(mean_absolute_error(y_true, y_pred))
	regression_r2 = float(r2_score(y_true, y_pred))
	if len(y_true) < 2 or np.isclose(np.std(y_true), 0.0) or np.isclose(np.std(y_pred), 0.0):
		pearson_r = float("nan")
	else:
		pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
	if np.isclose(np.linalg.norm(y_true), 0.0) or np.isclose(np.linalg.norm(y_pred), 0.0):
		cosine_sim = float("nan")
	else:
		cosine_sim = float(cosine_similarity(y_true.reshape(1, -1), y_pred.reshape(1, -1))[0, 0])
	metrics = {
		"autoencoder_reconstruction_mse": autoencoder_mse,
		"regression_mse": regression_mse,
		"regression_rmse": regression_rmse,
		"regression_mae": regression_mae,
		"regression_r2": regression_r2,
		"cosine_similarity": cosine_sim,
		"pearson_r": pearson_r,
		"correlation": pearson_r,
		"target_scaling": "standard",
		"regression_model": regression_model,
	}
	if regression_model == "svr":
		metrics.update(
			{
				"svr_kernel": svr_kernel,
				"svr_c": svr_c,
				"svr_epsilon": svr_epsilon,
				"svr_gamma": svr_gamma,
			}
		)
	elif regression_model == "kmeans":
		metrics.update(
			{
				"kmeans_regression_clusters": int(kmeans_regression_clusters),
				"kmeans_regression_effective_clusters": int(min(kmeans_regression_clusters, X_train_encoded.shape[0])),
				"kmeans_regression_n_init": int(kmeans_regression_n_init),
			}
		)
	#regression_r2 = model performansı / açıklama gücü
	#pearson_r     = gerçek-tahmin korelasyonu
	if regression_model == "neural":
		print("regressor output shape:", regressor.output_shape)
	elif regression_model == "svr":
		print(f"regressor model: SVR(kernel={svr_kernel}, C={svr_c}, epsilon={svr_epsilon}, gamma={svr_gamma})")
	else:
		print(
			"regressor model: "
			f"KMeansRegression(k={min(kmeans_regression_clusters, X_train_encoded.shape[0])}, "
			f"n_init={kmeans_regression_n_init})"
		)
	return metrics, autoencoder, encoder, X_train_sub, y_true, y_pred, y_train_true, y_train_pred


def normalize_cluster_k_range(min_k: int, max_k: int, sample_count: int) -> tuple[int, int]:
	if min_k < 2:
		raise ValueError("cluster-min-k en az 2 olmali.")
	if max_k < min_k:
		raise ValueError("cluster-max-k, cluster-min-k degerinden kucuk olamaz.")
	if sample_count < 3:
		raise ValueError("Clustering icin en az 3 satir gerekir.")
	return min_k, min(max_k, sample_count - 1)


def evaluate_kmeans_range(
	X_cluster: np.ndarray,
	min_k: int,
	max_k: int,
	random_state: int | None,
	selected_k: int | None = None,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
	min_k, max_k = normalize_cluster_k_range(min_k, max_k, X_cluster.shape[0])
	rows: list[dict] = []
	best_labels: np.ndarray | None = None
	best_row: dict | None = None
	selected_labels: np.ndarray | None = None
	selected_row: dict | None = None

	for k in range(min_k, max_k + 1):
		model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
		labels = model.fit_predict(X_cluster)
		unique_count = len(np.unique(labels))
		if unique_count < 2 or unique_count >= X_cluster.shape[0]:
			silhouette = float("nan")
		else:
			silhouette = float(silhouette_score(X_cluster, labels))

		row = {
			"k": k,
			"inertia": float(model.inertia_),
			"cluster_rmse": float(np.sqrt(model.inertia_ / X_cluster.shape[0])),
			"silhouette_score": silhouette,
		}
		rows.append(row)
		if selected_k is not None and k == selected_k:
			selected_row = row
			selected_labels = labels
		if not np.isnan(silhouette) and (best_row is None or silhouette > best_row["silhouette_score"]):
			best_row = row
			best_labels = labels

	if selected_k is not None:
		if selected_row is None or selected_labels is None:
			raise ValueError(f"Sabit cluster k={selected_k}, k araliginda bulunamadi: {min_k}-{max_k}")
		if np.isnan(selected_row["silhouette_score"]):
			raise ValueError(f"Sabit cluster k={selected_k} icin gecerli silhouette skoru hesaplanamadi.")
		return pd.DataFrame(rows), selected_row, selected_labels

	if best_row is None or best_labels is None:
		raise ValueError("Gecerli silhouette skoru hesaplanamadi. k araligini veya veri boyutunu kontrol edin.")

	return pd.DataFrame(rows), best_row, best_labels


def save_cluster_evaluation_plots(
	scores_df: pd.DataFrame,
	output_dir: Path,
	file_prefix: str,
	selected_k: int | None = None,
) -> None:
	if scores_df.empty or "k" not in scores_df.columns:
		return

	ensure_dir(output_dir)
	min_k_for_axis = int(scores_df["k"].min())
	max_k_for_axis = max(16, int(scores_df["k"].max()))
	k_ticks = np.arange(min_k_for_axis, max_k_for_axis + 1, 1)
	silhouette_df = pd.DataFrame()
	if "inertia" in scores_df.columns:
		plt.figure(figsize=(8, 5))
		plt.plot(scores_df["k"], scores_df["inertia"], marker="o")
		plt.xlabel("Number of clusters (k)", fontsize=15)
		plt.ylabel("Within-cluster sum of squares (Inertia)", fontsize=15)
		plt.xlim(min_k_for_axis - 0.5, max_k_for_axis + 0.5)
		plt.xticks(k_ticks)
		plt.tick_params(axis="both", labelsize=13)
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		elbow_path = output_dir / f"{file_prefix}_elbow.png"
		plt.savefig(elbow_path, dpi=150)
		plt.close()
		print(f"[OK] Elbow plot: {elbow_path}")

	if "silhouette_score" in scores_df.columns:
		silhouette_df = scores_df.dropna(subset=["silhouette_score"])
		if not silhouette_df.empty:
			plt.figure(figsize=(8, 5))
			plt.plot(silhouette_df["k"], silhouette_df["silhouette_score"], marker="o")
			plt.xlabel("Number of clusters (k)", fontsize=15)
			plt.ylabel("Silhouette score", fontsize=30)
			plt.xlim(min_k_for_axis - 0.5, max_k_for_axis + 0.5)
			plt.xticks(k_ticks)
			plt.tick_params(axis="both", labelsize=13)
			plt.grid(True, alpha=0.3)
			plt.tight_layout()
			silhouette_path = output_dir / f"{file_prefix}_silhouette.png"
			plt.savefig(silhouette_path, dpi=150)
			plt.close()
			print(f"[OK] Silhouette plot: {silhouette_path}")

	if "inertia" in scores_df.columns and not silhouette_df.empty:
		fig, ax_inertia = plt.subplots(figsize=(9, 5))
		ax_inertia.plot(
			scores_df["k"],
			scores_df["inertia"] / 10000.0,
			color="#1f77b4",
			marker="o",
			linewidth=2.4,
			markersize=7,
		)
		ax_inertia.set_xlabel("Number of clusters (k)", fontsize=20)
		ax_inertia.set_ylabel("WCSS", color="#1f77b4", fontsize=35)
		ax_inertia.set_xlim(min_k_for_axis - 0.5, max_k_for_axis + 0.5)
		ax_inertia.set_xticks(k_ticks)
		ax_inertia.tick_params(axis="y", labelcolor="#1f77b4")
		ax_inertia.tick_params(axis="both", labelsize=15)
		ax_inertia.grid(True, alpha=0.3)
		ax_inertia.text(
			0.0,
			1.02,
			r"$\times 10^4$",
			transform=ax_inertia.transAxes,
			ha="left",
			va="bottom",
			fontsize=15,
			color="#1f77b4",
		)

		ax_silhouette = ax_inertia.twinx()
		ax_silhouette.plot(
			silhouette_df["k"],
			silhouette_df["silhouette_score"] * 100.0,
			color="#ff7f0e",
			marker="s",
			linewidth=2.4,
			markersize=7,
		)
		ax_silhouette.set_ylabel("Silhouette score", color="#ff7f0e", fontsize=30)
		ax_silhouette.tick_params(axis="y", labelcolor="#ff7f0e")
		ax_silhouette.tick_params(axis="y", labelsize=15)
		ax_silhouette.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
		ax_silhouette.text(
			1.0,
			1.02,
			r"$\times 10^{-2}$",
			transform=ax_silhouette.transAxes,
			ha="right",
			va="bottom",
			fontsize=15,
			color="#ff7f0e",
		)
		fig.tight_layout()

		combined_path = output_dir / f"{file_prefix}_elbow_silhouette.png"
		fig.savefig(combined_path, dpi=150)
		if selected_k is not None:
			selected_combined_path = output_dir / f"k_{selected_k}_{file_prefix}_elbow_silhouette.png"
			fig.savefig(selected_combined_path, dpi=150)
			print(f"[OK] Selected-k combined elbow/silhouette plot: {selected_combined_path}")
		plt.close(fig)
		print(f"[OK] Combined elbow/silhouette plot: {combined_path}")


def save_cluster_pca_scatter(
	X_cluster: np.ndarray,
	labels: np.ndarray,
	output_dir: Path,
	file_prefix: str,
	selected_k: int | None = None,
) -> None:
	if X_cluster.shape[0] < 2 or X_cluster.shape[1] < 2:
		return

	ensure_dir(output_dir)
	pca = PCA(n_components=2, random_state=RANDOM_STATE)
	X_2d = pca.fit_transform(X_cluster)
	explained = pca.explained_variance_ratio_ * 100
	cluster_ids = np.asarray(labels, dtype=int)
	pca_df = pd.DataFrame(
		{
			"pc1": X_2d[:, 0],
			"pc2": X_2d[:, 1],
			"cluster": cluster_ids,
			"pc1_variance": explained[0],
			"pc2_variance": explained[1],
		}
	)

	plt.figure(figsize=(10, 7.5))
	cluster_count = int(cluster_ids.max()) + 1 if cluster_ids.size else 1
	cluster_cmap = build_cluster_colormap(cluster_count)
	cluster_norm = BoundaryNorm(np.arange(-0.5, cluster_count + 0.5, 1), cluster_cmap.N)
	scatter = plt.scatter(
		X_2d[:, 0],
		X_2d[:, 1],
		c=cluster_ids,
		cmap=cluster_cmap,
		norm=cluster_norm,
		s=28,
		alpha=0.8,
		edgecolors="none",
	)
	plt.xlabel(f"PC1 ({explained[0]:.1f}% variance)", fontsize=36)
	plt.ylabel(f"PC2 ({explained[1]:.1f}% variance)", fontsize=36)
	plt.tick_params(axis="both", labelsize=28)
	plt.grid(True, alpha=0.25)
	colorbar = plt.colorbar(scatter, label="Cluster", ticks=np.arange(cluster_count))
	colorbar.ax.tick_params(labelsize=26)
	colorbar.set_label("Cluster", fontsize=32)
	plt.tight_layout()

	plot_path = output_dir / f"{file_prefix}_clusters_pca_2d.png"
	plt.savefig(plot_path, dpi=300)
	pca_csv_path = output_dir / f"{file_prefix}_clusters_pca_2d.csv"
	pca_df.to_csv(pca_csv_path, index=False)
	if selected_k is not None:
		selected_plot_path = output_dir / f"k_{selected_k}_{file_prefix}_clusters_pca_2d.png"
		plt.savefig(selected_plot_path, dpi=300)
		selected_pca_csv_path = output_dir / f"k_{selected_k}_{file_prefix}_clusters_pca_2d.csv"
		pca_df.to_csv(selected_pca_csv_path, index=False)
		print(f"[OK] Selected-k Cluster PCA 2D plot: {selected_plot_path}")
	plt.close()
	print(f"[OK] Cluster PCA 2D plot: {plot_path}")


def load_selected_features_if_compatible(selected_features_path: Path, feature_names: list[str]) -> pd.DataFrame | None:
	selected_df = pd.read_csv(selected_features_path)
	if "feature_name" not in selected_df.columns:
		print(f"[WARN] Feature listesi uyumsuz, 'feature_name' kolonu yok: {selected_features_path}")
		return None

	feature_name_set = set(feature_names)
	missing_features = [name for name in selected_df["feature_name"].tolist() if name not in feature_name_set]
	if missing_features:
		preview = missing_features[:5]
		print(
			f"[WARN] Feature listesi mevcut dataset ile uyumsuz: {selected_features_path}. "
			f"Eksik feature sayisi: {len(missing_features)}. Ornek: {preview}. "
			"Bu liste atlanacak."
		)
		return None

	return selected_df


def ensure_shared_selected_features(
	processed: dict,
	dataset_folder: str,
	encoding_dim: int,
	feature_percent: float,
	random_state: int | None,
	feature_chunk_size: int,
	chunk_feature_threshold: int,
	enable_feature_chunking: bool,
) -> pd.DataFrame:
	X_train_raw = processed["X_train"]
	X_test_raw = processed["X_test"]
	y_train = processed["y_train"].to_numpy().astype(np.int32)
	feature_names = X_train_raw.columns.tolist()
	feature_percent_tag = format_feature_percent_tag(feature_percent)

	output_dir = classification_output_dir(dataset_folder)
	ensure_dir(output_dir)
	selected_features_path = output_dir / f"top_{feature_percent_tag}_max_abs_features.csv"
	if selected_features_path.exists():
		selected_df = load_selected_features_if_compatible(selected_features_path, feature_names)
		if selected_df is not None:
			print(f"[INFO] Ortak feature listesi kullaniliyor: {selected_features_path}")
			return selected_df

	chunked_selected_features_path = output_dir / f"chunked_merged_top_{feature_percent_tag}_features.csv"
	if chunked_selected_features_path.exists():
		selected_df = load_selected_features_if_compatible(chunked_selected_features_path, feature_names)
		if selected_df is not None:
			print(f"[INFO] Ortak chunked feature listesi kullaniliyor: {chunked_selected_features_path}")
			return selected_df

	if should_use_feature_chunking(
		feature_count=len(feature_names),
		chunk_feature_threshold=chunk_feature_threshold,
		feature_chunk_size=feature_chunk_size,
		enable_feature_chunking=enable_feature_chunking,
	):
		return generate_chunked_shared_selected_features(
			processed=processed,
			dataset_folder=dataset_folder,
			encoding_dim=encoding_dim,
			feature_percent=feature_percent,
			random_state=random_state,
			feature_chunk_size=feature_chunk_size,
		)

	print("[INFO] Ortak feature listesi bulunamadi. Classification ile ayni train split mantigiyla uretiliyor.")
	X_train, X_test, _ = scale_data(X_train_raw, X_test_raw)
	X_train_sub, X_val, _, _ = train_test_split(
		X_train,
		y_train,
		test_size=CLASSIFIER_VALIDATION_SPLIT,
		random_state=random_state,
		shuffle=True,
		stratify=y_train,
	)
	_, autoencoder, _ = train_autoencoder_model(
		X_train_sub=X_train_sub,
		X_val=X_val,
		X_eval=X_test,
		encoding_dim=encoding_dim,
	)

	weights_path = output_dir / "first_layer_W_list.csv"
	save_feature_weighted_lists(autoencoder, X_train_sub, feature_names, weights_path)
	return save_top_percent_features_by_abs_max_weight(
		weight_list_csv_path=weights_path,
		feature_names=feature_names,
		feature_percent=feature_percent,
		output_path=selected_features_path,
	)


def generate_chunked_shared_selected_features(
	processed: dict,
	dataset_folder: str,
	encoding_dim: int,
	feature_percent: float,
	random_state: int | None,
	feature_chunk_size: int,
) -> pd.DataFrame:
	X_train_raw = processed["X_train"]
	X_test_raw = processed["X_test"]
	y_train = processed["y_train"].to_numpy().astype(np.int32)
	feature_names = X_train_raw.columns.tolist()
	feature_chunks = split_feature_names_into_chunks(feature_names, feature_chunk_size)
	feature_percent_tag = format_feature_percent_tag(feature_percent)

	output_dir = classification_output_dir(dataset_folder)
	chunks_dir = output_dir / "chunks" / "shared_feature_ranking"
	ensure_dir(output_dir)
	ensure_dir(chunks_dir)

	print(
		f"[INFO] Ortak feature listesi chunked uretilecek: {len(feature_names)} feature, "
		f"{len(feature_chunks)} parca (chunk_size={feature_chunk_size})."
	)

	chunk_selected_frames: list[pd.DataFrame] = []
	for chunk_idx, chunk_feature_names in enumerate(feature_chunks, start=1):
		chunk_name = f"chunk_{chunk_idx:03d}"
		chunk_dir = chunks_dir / chunk_name
		ensure_dir(chunk_dir)
		print(
			f"\n[INFO] Ortak {chunk_name}/{len(feature_chunks):03d} feature ranking basliyor "
			f"(feature sayisi: {len(chunk_feature_names)})."
		)

		X_train_chunk_raw = X_train_raw[chunk_feature_names]
		X_test_chunk_raw = X_test_raw[chunk_feature_names]
		X_train_chunk, X_test_chunk, _ = scale_data(X_train_chunk_raw, X_test_chunk_raw)
		X_train_chunk = X_train_chunk.astype(np.float32)
		X_test_chunk = X_test_chunk.astype(np.float32)

		X_train_sub, X_val, _, _ = train_test_split(
			X_train_chunk,
			y_train,
			test_size=CLASSIFIER_VALIDATION_SPLIT,
			random_state=random_state,
			shuffle=True,
			stratify=y_train,
		)
		_, chunk_autoencoder, _ = train_autoencoder_model(
			X_train_sub=X_train_sub,
			X_val=X_val,
			X_eval=X_test_chunk,
			encoding_dim=encoding_dim,
		)

		chunk_weights_path = chunk_dir / "first_layer_W_list.csv"
		save_feature_weighted_lists(
			chunk_autoencoder,
			X_train_sub,
			chunk_feature_names,
			chunk_weights_path,
		)

		chunk_selected_path = chunk_dir / f"top_{feature_percent_tag}_max_abs_features.csv"
		chunk_selected_df = save_top_percent_features_by_abs_max_weight(
			weight_list_csv_path=chunk_weights_path,
			feature_names=chunk_feature_names,
			feature_percent=feature_percent,
			output_path=chunk_selected_path,
		)
		chunk_selected_df.insert(0, "chunk", chunk_name)
		chunk_selected_frames.append(chunk_selected_df)
		print(f"[OK] Ortak {chunk_name} tamamlandi. Top %{feature_percent}: {len(chunk_selected_df)} feature.")

	all_chunk_selected_df = pd.concat(chunk_selected_frames, ignore_index=True)
	all_chunk_selected_path = output_dir / f"chunked_top_{feature_percent_tag}_max_abs_features.csv"
	all_chunk_selected_df.to_csv(all_chunk_selected_path, index=False)

	merged_feature_names = list(dict.fromkeys(all_chunk_selected_df["feature_name"].tolist()))
	merged_selected_df = pd.DataFrame(
		{
			"feature": [f"F{i+1}" for i in range(len(merged_feature_names))],
			"feature_name": merged_feature_names,
			"source": "chunked_top_features",
		}
	)
	merged_selected_path = output_dir / f"chunked_merged_top_{feature_percent_tag}_features.csv"
	merged_selected_df.to_csv(merged_selected_path, index=False)

	print(f"[OK] Ortak chunked feature listesi olusturuldu: {merged_selected_path}")
	print(f"[OK] Birlesen feature sayisi: {len(merged_selected_df)}")
	return merged_selected_df


def run_clustering_experiment(
	df: pd.DataFrame,
	dataset_folder: str,
	target_column: str,
	id_column: str | None,
	encoding_dim: int,
	feature_percent: float,
	random_state: int | None,
	cluster_k: int | None,
	cluster_min_k: int,
	cluster_max_k: int,
	feature_chunk_size: int,
	chunk_feature_threshold: int,
	enable_feature_chunking: bool,
	save_training_plots: bool,
) -> tuple[float, float]:
	start_time = time.perf_counter()
	processed = preprocess_data(
		df,
		target_column=target_column,
		id_column=id_column,
		random_state=random_state,
		scale_features=False,
	)
	X_raw = pd.concat([processed["X_train"], processed["X_test"]], axis=0)
	y_all = pd.concat([processed["y_train"], processed["y_test"]], axis=0)
	feature_names = X_raw.columns.tolist()

	output_dir = clustering_output_dir(dataset_folder)
	metrics_dir = output_dir / "metrics"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)

	print(f"[INFO] Clustering modu basladi. X shape: {X_raw.shape}")
	print("[INFO] Label varsa clustering egitiminde kullanilmayacak; k degeri silhouette skoruna gore secilecek.")

	effective_min_k = cluster_min_k
	effective_max_k = cluster_max_k
	if cluster_k is not None:
		if cluster_k < 2:
			raise ValueError("--cluster-k en az 2 olmali.")
		if cluster_k >= X_raw.shape[0]:
			raise ValueError(f"--cluster-k degeri satir sayisindan kucuk olmali. Gelen k={cluster_k}, satir={X_raw.shape[0]}")
		effective_min_k = min(effective_min_k, cluster_k)
		effective_max_k = max(effective_max_k, cluster_k)
		print(
			f"[INFO] Sabit cluster k kullaniliyor: k={cluster_k}. "
			f"Elbow/silhouette grafikleri k={effective_min_k}-{effective_max_k} araliginda cizilecek."
		)
	if y_all is not None and y_all.nunique(dropna=True) > 1:
		class_count = int(y_all.nunique(dropna=True))
		print(
			f"[INFO] Label bulundu: class_count={class_count}. "
			f"KMeans k araligi korunuyor: {effective_min_k}-{effective_max_k}."
		)

	X_org_scaled, _, _ = scale_data(X_raw, X_raw)
	X_org_scaled = X_org_scaled.astype(np.float32)
	org_scores_df, org_best_row, org_best_labels = evaluate_kmeans_range(
		X_cluster=X_org_scaled,
		min_k=effective_min_k,
		max_k=effective_max_k,
		random_state=random_state,
		selected_k=cluster_k,
	)
	org_scores_path = output_dir / "ORG_cluster_scores.csv"
	org_scores_df.to_csv(org_scores_path, index=False)
	save_cluster_evaluation_plots(
		scores_df=org_scores_df,
		output_dir=output_dir,
		file_prefix="ORG",
		selected_k=cluster_k,
	)
	save_cluster_pca_scatter(
		X_cluster=X_org_scaled,
		labels=org_best_labels,
		output_dir=output_dir,
		file_prefix="ORG",
		selected_k=cluster_k,
	)
	org_assignments_df = pd.DataFrame(
		{
			"sample_index": X_raw.index.tolist(),
			"cluster": org_best_labels.astype(int),
		}
	)
	if y_all is not None:
		org_assignments_df["true_label"] = y_all.to_numpy()
	org_assignments_path = output_dir / "ORG_cluster_assignments.csv"
	org_assignments_df.to_csv(org_assignments_path, index=False)
	org_elapsed_seconds = time.perf_counter() - start_time
	org_metrics_data = {
		"task": "clustering",
		"feature_set": "ORG",
		"original_feature_count": len(feature_names),
		"selected_feature_count": len(feature_names),
		"cluster_min_k": effective_min_k,
		"cluster_max_k": effective_max_k,
		"fixed_cluster_k": cluster_k,
		"best_k": int(org_best_row["k"]),
		"silhouette_score": float(org_best_row["silhouette_score"]),
		"inertia": float(org_best_row["inertia"]),
		"cluster_rmse": float(org_best_row["cluster_rmse"]),
		"elapsed_seconds": org_elapsed_seconds,
	}
	org_metrics_path = metrics_dir / "ORG_cluster_metrics.json"
	save_json(org_metrics_data, org_metrics_path)

	selected_df = ensure_shared_selected_features(
		processed=processed,
		dataset_folder=dataset_folder,
		encoding_dim=encoding_dim,
		feature_percent=feature_percent,
		random_state=random_state,
		feature_chunk_size=feature_chunk_size,
		chunk_feature_threshold=chunk_feature_threshold,
		enable_feature_chunking=enable_feature_chunking,
	)

	feature_percent_tag = format_feature_percent_tag(feature_percent)
	selected_feature_names = selected_df["feature_name"].tolist()
	missing_features = [name for name in selected_feature_names if name not in feature_names]
	if missing_features:
		raise ValueError(f"Ortak feature listesinde veri setinde bulunmayan feature var: {missing_features}")
	X_selected_raw = X_raw[selected_feature_names]
	X_selected_scaled, _, _ = scale_data(X_selected_raw, X_selected_raw)
	X_selected_scaled = X_selected_scaled.astype(np.float32)

	scores_df, best_row, best_labels = evaluate_kmeans_range(
		X_cluster=X_selected_scaled,
		min_k=effective_min_k,
		max_k=effective_max_k,
		random_state=random_state,
		selected_k=cluster_k,
	)

	scores_path = output_dir / f"top_{feature_percent_tag}_cluster_scores.csv"
	scores_df.to_csv(scores_path, index=False)
	save_cluster_evaluation_plots(
		scores_df=scores_df,
		output_dir=output_dir,
		file_prefix=f"top_{feature_percent_tag}",
		selected_k=cluster_k,
	)
	save_cluster_pca_scatter(
		X_cluster=X_selected_scaled,
		labels=best_labels,
		output_dir=output_dir,
		file_prefix=f"top_{feature_percent_tag}",
		selected_k=cluster_k,
	)

	assignments_df = pd.DataFrame(
		{
			"sample_index": X_raw.index.tolist(),
			"cluster": best_labels.astype(int),
		}
	)
	if y_all is not None:
		assignments_df["true_label"] = y_all.to_numpy()
	assignments_path = output_dir / f"top_{feature_percent_tag}_cluster_assignments.csv"
	assignments_df.to_csv(assignments_path, index=False)
	elapsed_seconds = time.perf_counter() - start_time

	metrics_data = {
		"task": "clustering",
		"feature_set": f"top_{feature_percent_tag}",
		"feature_percent": feature_percent,
		"original_feature_count": len(feature_names),
		"selected_feature_count": len(selected_df),
		"cluster_min_k": effective_min_k,
		"cluster_max_k": effective_max_k,
		"fixed_cluster_k": cluster_k,
		"best_k": int(best_row["k"]),
		"silhouette_score": float(best_row["silhouette_score"]),
		"inertia": float(best_row["inertia"]),
		"cluster_rmse": float(best_row["cluster_rmse"]),
		"elapsed_seconds": elapsed_seconds,
	}
	metrics_path = metrics_dir / f"top_{feature_percent_tag}_cluster_metrics.json"
	save_json(metrics_data, metrics_path)

	print("\n[OK] Clustering tamamlandi.")
	print(f"[OK] ORG silhouette_score: {float(org_best_row['silhouette_score']):.6f}")
	print(f"[OK] ORG cluster_rmse: {float(org_best_row['cluster_rmse']):.6f}")
	print(f"[OK] ORG best k: {int(org_best_row['k'])}")
	print(f"[OK] ORG metrik dosyasi: {org_metrics_path}")
	print(f"[OK] Top %{feature_percent} secilen feature sayisi: {len(selected_df)}")
	print(f"[OK] En iyi k: {int(best_row['k'])}")
	print(f"[OK] En iyi silhouette_score: {float(best_row['silhouette_score']):.6f}")
	print(f"[OK] Cluster RMSE: {float(best_row['cluster_rmse']):.6f}")
	print(f"[OK] Calisma suresi: {elapsed_seconds:.2f} saniye")
	print(f"[OK] Elbow/Inertia skor CSV: {scores_path}")
	print(f"[OK] Cluster atamalari: {assignments_path}")
	return float(best_row["cluster_rmse"]), float(best_row["cluster_rmse"])


def split_feature_names_into_chunks(feature_names: list[str], chunk_size: int) -> list[list[str]]:
	if chunk_size <= 0:
		raise ValueError("feature-chunk-size pozitif tam sayi olmali.")
	return [
		feature_names[start : start + chunk_size]
		for start in range(0, len(feature_names), chunk_size)
	]


def should_use_feature_chunking(
	feature_count: int,
	chunk_feature_threshold: int,
	feature_chunk_size: int,
	enable_feature_chunking: bool,
) -> bool:
	if not enable_feature_chunking:
		return False
	if chunk_feature_threshold <= 0:
		raise ValueError("chunk-feature-threshold pozitif tam sayi olmali.")
	if feature_chunk_size <= 0:
		raise ValueError("feature-chunk-size pozitif tam sayi olmali.")
	return feature_count > chunk_feature_threshold


def preprocess_explicit_train_validation_data(
	train_df: pd.DataFrame,
	validation_df: pd.DataFrame,
	target_column: str,
	id_column: str | None,
) -> dict:
	train_df = drop_id_column(train_df, id_column=id_column)
	validation_df = drop_id_column(validation_df, id_column=id_column)
	train_df = encode_target(train_df, target_column=target_column)
	validation_df = encode_target(validation_df, target_column=target_column)

	X_train_raw, y_train = split_features_target(train_df, target_column=target_column)
	X_validation_raw, y_validation = split_features_target(validation_df, target_column=target_column)
	X_train_raw = keep_numeric_features_only(handle_pid_unrealistic_zeros(X_train_raw))
	X_validation_raw = keep_numeric_features_only(handle_pid_unrealistic_zeros(X_validation_raw))

	missing_features = [feature for feature in X_train_raw.columns if feature not in X_validation_raw.columns]
	if missing_features:
		raise ValueError(
			"Validation dosyasinda train dosyasinda bulunan feature kolonlari eksik. "
			f"Ornek eksikler: {missing_features[:10]}"
		)

	X_validation_raw = X_validation_raw[X_train_raw.columns]
	return {
		"X_train": X_train_raw,
		"X_validation": X_validation_raw,
		"y_train": y_train,
		"y_validation": y_validation,
	}


def save_binary_classification_report_outputs(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	y_score: np.ndarray,
	output_dir: Path,
	metrics_dir: Path,
	file_prefix: str,
	metric_prefix: str,
	base_metrics: dict,
) -> dict:
	predictions_path = save_classification_predictions(
		y_true=y_true,
		y_pred=y_pred,
		y_score=y_score,
		output_dir=output_dir,
		file_prefix=file_prefix,
	)
	confusion_matrix_path = save_classification_confusion_matrix_plot(
		y_true=y_true,
		y_pred=y_pred,
		output_dir=output_dir,
		file_prefix=file_prefix,
	)
	precision_recall_path = save_classification_precision_recall_plot(
		y_true=y_true,
		y_score=y_score,
		output_dir=output_dir,
		file_prefix=file_prefix,
	)
	roc_path = save_classification_roc_plot(
		y_true=y_true,
		y_score=y_score,
		output_dir=output_dir,
		file_prefix=file_prefix,
	)

	metrics_data = dict(base_metrics)
	add_test_rmse_metric(metrics_data)
	metrics_data.update(
		rename_classification_metric_prefix(
			compute_binary_classification_metrics(y_true=y_true, y_pred=y_pred, y_score=y_score),
			metric_prefix,
		)
	)
	if predictions_path is not None:
		metrics_data["classification_predictions_path"] = str(predictions_path)
	if confusion_matrix_path is not None:
		metrics_data["confusion_matrix_path"] = str(confusion_matrix_path)
	if precision_recall_path is not None:
		metrics_data["precision_recall_curve_path"] = str(precision_recall_path)
	if roc_path is not None:
		metrics_data["roc_curve_path"] = str(roc_path)

	save_json(metrics_data, metrics_dir / f"{file_prefix}_metrics.json")
	return metrics_data


def run_chunked_binary_train_validation_experiment(
	X_train_raw: pd.DataFrame,
	X_validation_raw: pd.DataFrame,
	y_train: np.ndarray,
	y_validation: np.ndarray,
	dataset_folder: str,
	validation_dataset_folder: str,
	encoding_dim: int,
	feature_percent: float,
	random_state: int | None,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
	classifier_model: str,
	classifier_early_stopping_patience: int | None,
	autoencoder_early_stopping_patience: int | None,
	classifier_early_stopping_monitor: str,
	classifier_early_stopping_min_delta: float,
	autoencoder_early_stopping_min_delta: float,
	classifier_class_weight: str,
	classifier_sampling: str,
	feature_chunk_size: int,
	save_training_plots: bool,
	start_time: float,
) -> tuple[float, float]:
	feature_names = X_train_raw.columns.tolist()
	feature_chunks = split_feature_names_into_chunks(feature_names, feature_chunk_size)
	feature_percent_tag = format_feature_percent_tag(feature_percent)

	output_dir = classification_output_dir(dataset_folder)
	metrics_dir = output_dir / "metrics"
	chunks_dir = output_dir / "chunks"
	history_dir = output_dir / "training_history"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)
	ensure_dir(chunks_dir)
	if save_training_plots:
		ensure_dir(history_dir)

	print(
		f"[INFO] Explicit train/validation icin büyük feature seti tespit edildi: "
		f"{len(feature_names)} feature. ORG full-feature egitimi atlanip "
		f"{len(feature_chunks)} parcali feature ranking yapilacak (chunk_size={feature_chunk_size})."
	)

	chunk_selected_frames: list[pd.DataFrame] = []
	chunk_summaries: list[dict] = []
	for chunk_idx, chunk_feature_names in enumerate(feature_chunks, start=1):
		chunk_name = f"chunk_{chunk_idx:03d}"
		chunk_dir = chunks_dir / chunk_name
		ensure_dir(chunk_dir)
		print(
			f"\n[INFO] {chunk_name}/{len(feature_chunks):03d} train/validation egitimi basliyor "
			f"(feature sayisi: {len(chunk_feature_names)})."
		)

		X_train_chunk_raw = X_train_raw[chunk_feature_names]
		X_validation_chunk_raw = X_validation_raw[chunk_feature_names]
		X_train_chunk, X_validation_chunk, _ = scale_data(X_train_chunk_raw, X_validation_chunk_raw)
		(
			chunk_validation_mse,
			chunk_validation_accuracy,
			chunk_autoencoder,
			_,
			chunk_train_sub,
			_,
			_,
		) = train_and_evaluate_pipeline(
			X_train_chunk.astype(np.float32),
			X_validation_chunk.astype(np.float32),
			y_train,
			y_validation,
			encoding_dim,
			random_state,
			classifier_epochs,
			classifier_hidden_units,
			classifier_dropout_rates,
			classifier_learning_rate,
			classifier_model,
			classifier_early_stopping_patience,
			autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta,
			classifier_class_weight=classifier_class_weight,
			classifier_sampling=classifier_sampling,
			history_output_dir=history_dir if save_training_plots else None,
			history_prefix=chunk_name if save_training_plots else None,
		)

		chunk_weights_path = chunk_dir / "first_layer_W_list.csv"
		save_feature_weighted_lists(
			chunk_autoencoder,
			chunk_train_sub,
			chunk_feature_names,
			chunk_weights_path,
		)
		chunk_selected_path = chunk_dir / f"top_{feature_percent_tag}_max_abs_features.csv"
		chunk_selected_df = save_top_percent_features_by_abs_max_weight(
			weight_list_csv_path=chunk_weights_path,
			feature_names=chunk_feature_names,
			feature_percent=feature_percent,
			output_path=chunk_selected_path,
		)
		chunk_selected_df.insert(0, "chunk", chunk_name)
		chunk_selected_frames.append(chunk_selected_df)
		chunk_summary = {
			"chunk": chunk_name,
			"feature_count": len(chunk_feature_names),
			"selected_feature_count": len(chunk_selected_df),
			"validation_mse": chunk_validation_mse,
			"validation_accuracy": chunk_validation_accuracy,
			"weights_path": str(chunk_weights_path),
			"selected_features_path": str(chunk_selected_path),
		}
		chunk_summaries.append(chunk_summary)
		save_json(chunk_summary, metrics_dir / f"{chunk_name}_train_validation_metrics.json")
		print(
			f"[OK] {chunk_name} tamamlandi. Top %{feature_percent}: {len(chunk_selected_df)} feature, "
			f"validation_accuracy: {chunk_validation_accuracy:.6f}"
		)

	all_chunk_selected_df = pd.concat(chunk_selected_frames, ignore_index=True)
	all_chunk_selected_path = output_dir / f"chunked_top_{feature_percent_tag}_max_abs_features.csv"
	all_chunk_selected_df.to_csv(all_chunk_selected_path, index=False)
	merged_feature_names = list(dict.fromkeys(all_chunk_selected_df["feature_name"].tolist()))
	merged_selected_df = pd.DataFrame(
		{
			"feature": [f"F{i+1}" for i in range(len(merged_feature_names))],
			"feature_name": merged_feature_names,
			"source": "chunked_top_features",
		}
	)
	merged_selected_path = output_dir / f"chunked_merged_top_{feature_percent_tag}_features.csv"
	merged_selected_df.to_csv(merged_selected_path, index=False)
	# Normal top_X dosya adini da yazalim; sonraki calismalar bu listeyi tekrar kullanabilir.
	merged_selected_df[["feature", "feature_name"]].to_csv(
		output_dir / f"top_{feature_percent_tag}_max_abs_features.csv",
		index=False,
	)

	print(
		f"\n[INFO] Chunk top feature'lari birlestirildi: {len(merged_feature_names)} feature. "
		"Final train/validation egitimi basliyor."
	)
	X_train_final_raw = X_train_raw[merged_feature_names]
	X_validation_final_raw = X_validation_raw[merged_feature_names]
	X_train_final, X_validation_final, _ = scale_data(X_train_final_raw, X_validation_final_raw)
	(
		final_validation_mse,
		final_validation_accuracy,
		_,
		_,
		_,
		final_validation_y_pred,
		final_validation_y_score,
		final_train_y_pred,
		final_train_y_score,
	) = train_and_evaluate_pipeline(
		X_train_final.astype(np.float32),
		X_validation_final.astype(np.float32),
		y_train,
		y_validation,
		encoding_dim,
		random_state,
		classifier_epochs,
		classifier_hidden_units,
		classifier_dropout_rates,
		classifier_learning_rate,
		classifier_model,
		classifier_early_stopping_patience,
		autoencoder_early_stopping_patience,
		classifier_early_stopping_monitor,
		classifier_early_stopping_min_delta,
		autoencoder_early_stopping_min_delta,
		classifier_class_weight=classifier_class_weight,
		classifier_sampling=classifier_sampling,
		history_output_dir=history_dir if save_training_plots else None,
		history_prefix=f"chunked_top_{feature_percent_tag}_final" if save_training_plots else None,
		return_train_predictions=True,
	)

	elapsed_seconds = time.perf_counter() - start_time
	common_base = {
		"task": "classification",
		"mode": "explicit_train_validation_chunked",
		"chunked_feature_selection": True,
		"validation_dataset": validation_dataset_folder,
		"feature_percent": feature_percent,
		"original_feature_count": len(feature_names),
		"feature_chunk_size": feature_chunk_size,
		"chunk_count": len(feature_chunks),
		"merged_feature_count": len(merged_feature_names),
		"selected_feature_count": len(merged_feature_names),
		"threshold": THRESHOLD,
		"classifier_model": classifier_model,
		"classifier_class_weight": classifier_class_weight,
		"classifier_sampling": classifier_sampling,
		"elapsed_seconds": elapsed_seconds,
		"all_chunk_selected_features_path": str(all_chunk_selected_path),
		"merged_selected_features_path": str(merged_selected_path),
		"chunk_summaries": chunk_summaries,
	}
	train_metrics = save_binary_classification_report_outputs(
		y_true=y_train,
		y_pred=final_train_y_pred,
		y_score=final_train_y_score,
		output_dir=output_dir,
		metrics_dir=metrics_dir,
		file_prefix=f"top_{feature_percent_tag}_train",
		metric_prefix="train",
		base_metrics={**common_base, "feature_set": f"top_{feature_percent_tag}"},
	)
	validation_metrics = save_binary_classification_report_outputs(
		y_true=y_validation,
		y_pred=final_validation_y_pred,
		y_score=final_validation_y_score,
		output_dir=output_dir,
		metrics_dir=metrics_dir,
		file_prefix=f"top_{feature_percent_tag}_validation",
		metric_prefix="validation",
		base_metrics={
			**common_base,
			"feature_set": f"top_{feature_percent_tag}",
			"validation_mse": final_validation_mse,
		},
	)
	summary = {
		"task": "classification",
		"mode": "explicit_train_validation_chunked",
		"dataset": dataset_folder,
		"validation_dataset": validation_dataset_folder,
		"feature_percent": feature_percent,
		"selected_feature_count": len(merged_feature_names),
		"ORG_full_feature_note": "ORG full-feature egitimi buyuk veri/RAM nedeniyle chunked modda atlandi.",
		f"top_{feature_percent_tag}_train": train_metrics,
		f"top_{feature_percent_tag}_validation": validation_metrics,
	}
	summary_path = metrics_dir / f"top_{feature_percent_tag}_train_validation_summary.json"
	save_json(summary, summary_path)

	print("\n[OK] Chunked explicit train/validation classification tamamlandi.")
	print(f"[OK] Top %{feature_percent} train accuracy: {train_metrics.get('train_accuracy'):.6f}")
	print(f"[OK] Top %{feature_percent} validation accuracy: {validation_metrics.get('validation_accuracy'):.6f}")
	print(f"[OK] Ozet metrik dosyasi: {summary_path}")
	return 0.0, float(validation_metrics.get("validation_accuracy", 0.0))


def run_binary_train_validation_experiment(
	train_df: pd.DataFrame,
	validation_df: pd.DataFrame,
	dataset_folder: str,
	validation_dataset_folder: str,
	target_column: str,
	id_column: str | None,
	encoding_dim: int,
	feature_percent: float,
	random_state: int | None,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
	classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
	classifier_early_stopping_patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE,
	autoencoder_early_stopping_patience: int | None = DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE,
	classifier_early_stopping_monitor: str = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR,
	classifier_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	autoencoder_early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	classifier_class_weight: str = "none",
	classifier_sampling: str = "none",
	feature_chunk_size: int = DEFAULT_FEATURE_CHUNK_SIZE,
	chunk_feature_threshold: int = DEFAULT_CHUNK_FEATURE_THRESHOLD,
	enable_feature_chunking: bool = True,
	save_training_plots: bool = False,
) -> tuple[float, float]:
	start_time = time.perf_counter()
	processed = preprocess_explicit_train_validation_data(
		train_df=train_df,
		validation_df=validation_df,
		target_column=target_column,
		id_column=id_column,
	)
	X_train_raw = processed["X_train"]
	X_validation_raw = processed["X_validation"]
	y_train = processed["y_train"].to_numpy().astype(np.int32)
	y_validation = processed["y_validation"].to_numpy().astype(np.int32)
	for set_name, y_values in {"train": y_train, "validation": y_validation}.items():
		if not set(np.unique(y_values)).issubset({0, 1}):
			raise ValueError(f"{set_name} dosyasi binary etiket bekliyor. Label degerleri sadece 0 ve 1 olmali.")

	print(f"[INFO] Explicit train/validation modu.")
	print(f"[INFO] Train X shape      : {X_train_raw.shape}")
	print(f"[INFO] Validation X shape : {X_validation_raw.shape}")
	print(f"[INFO] Train label dagilimi: {pd.Series(y_train).value_counts().to_dict()}")
	print(f"[INFO] Validation label dagilimi: {pd.Series(y_validation).value_counts().to_dict()}")

	if should_use_feature_chunking(
		feature_count=X_train_raw.shape[1],
		chunk_feature_threshold=chunk_feature_threshold,
		feature_chunk_size=feature_chunk_size,
		enable_feature_chunking=enable_feature_chunking,
	):
		return run_chunked_binary_train_validation_experiment(
			X_train_raw=X_train_raw,
			X_validation_raw=X_validation_raw,
			y_train=y_train,
			y_validation=y_validation,
			dataset_folder=dataset_folder,
			validation_dataset_folder=validation_dataset_folder,
			encoding_dim=encoding_dim,
			feature_percent=feature_percent,
			random_state=random_state,
			classifier_epochs=classifier_epochs,
			classifier_hidden_units=classifier_hidden_units,
			classifier_dropout_rates=classifier_dropout_rates,
			classifier_learning_rate=classifier_learning_rate,
			classifier_model=classifier_model,
			classifier_early_stopping_patience=classifier_early_stopping_patience,
			autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor=classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
			classifier_class_weight=classifier_class_weight,
			classifier_sampling=classifier_sampling,
			feature_chunk_size=feature_chunk_size,
			save_training_plots=save_training_plots,
			start_time=start_time,
		)

	X_train, X_validation, _ = scale_data(X_train_raw, X_validation_raw)
	X_train = X_train.astype(np.float32)
	X_validation = X_validation.astype(np.float32)

	output_dir = classification_output_dir(dataset_folder)
	metrics_dir = output_dir / "metrics"
	history_dir = output_dir / "training_history"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)
	if save_training_plots:
		ensure_dir(history_dir)

	(
		validation_mse,
		validation_accuracy,
		autoencoder,
		_,
		X_train_sub_used,
		validation_y_pred,
		validation_y_score,
		train_y_pred,
		train_y_score,
	) = train_and_evaluate_pipeline(
		X_train,
		X_validation,
		y_train,
		y_validation,
		encoding_dim,
		random_state,
		classifier_epochs,
		classifier_hidden_units,
		classifier_dropout_rates,
		classifier_learning_rate,
		classifier_model,
		classifier_early_stopping_patience,
		autoencoder_early_stopping_patience,
		classifier_early_stopping_monitor,
		classifier_early_stopping_min_delta,
		autoencoder_early_stopping_min_delta,
		classifier_class_weight=classifier_class_weight,
		classifier_sampling=classifier_sampling,
		history_output_dir=history_dir if save_training_plots else None,
		history_prefix="ORG" if save_training_plots else None,
		return_train_predictions=True,
	)

	feature_names = X_train_raw.columns.tolist()
	feature_percent_tag = format_feature_percent_tag(feature_percent)
	selected_features_path = output_dir / f"top_{feature_percent_tag}_max_abs_features.csv"
	selected_df = None
	if selected_features_path.exists():
		selected_df = load_selected_features_if_compatible(selected_features_path, feature_names)
		if selected_df is not None:
			print(f"[INFO] Mevcut feature listesi kullaniliyor: {selected_features_path}")
	if selected_df is None:
		weights_path = output_dir / "first_layer_W_list.csv"
		save_feature_weighted_lists(autoencoder, X_train_sub_used, feature_names, weights_path)
		selected_df = save_top_percent_features_by_abs_max_weight(
			weight_list_csv_path=weights_path,
			feature_names=feature_names,
			feature_percent=feature_percent,
			output_path=selected_features_path,
		)

	if len(selected_df) == len(feature_names):
		filtered_validation_mse = validation_mse
		filtered_validation_accuracy = validation_accuracy
		filtered_validation_y_pred = validation_y_pred
		filtered_validation_y_score = validation_y_score
		filtered_train_y_pred = train_y_pred
		filtered_train_y_score = train_y_score
		print("[INFO] Top %100 tum feature'lari iceriyor. ORG sonucu yeniden egitilmeden kullaniliyor.")
	else:
		selected_feature_names = selected_df["feature_name"].tolist()
		X_train_filtered_raw = X_train_raw[selected_feature_names]
		X_validation_filtered_raw = X_validation_raw[selected_feature_names]
		X_train_filtered, X_validation_filtered, _ = scale_data(X_train_filtered_raw, X_validation_filtered_raw)
		(
			filtered_validation_mse,
			filtered_validation_accuracy,
			_,
			_,
			_,
			filtered_validation_y_pred,
			filtered_validation_y_score,
			filtered_train_y_pred,
			filtered_train_y_score,
		) = train_and_evaluate_pipeline(
			X_train_filtered.astype(np.float32),
			X_validation_filtered.astype(np.float32),
			y_train,
			y_validation,
			encoding_dim,
			random_state,
			classifier_epochs,
			classifier_hidden_units,
			classifier_dropout_rates,
			classifier_learning_rate,
			classifier_model,
			classifier_early_stopping_patience,
			autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta,
			classifier_class_weight=classifier_class_weight,
			classifier_sampling=classifier_sampling,
			history_output_dir=history_dir if save_training_plots else None,
			history_prefix=f"top_{feature_percent_tag}" if save_training_plots else None,
			return_train_predictions=True,
		)

	elapsed_seconds = time.perf_counter() - start_time
	common_base = {
		"task": "classification",
		"validation_dataset": validation_dataset_folder,
		"threshold": THRESHOLD,
		"classifier_model": classifier_model,
		"classifier_class_weight": classifier_class_weight,
		"classifier_sampling": classifier_sampling,
		"elapsed_seconds": elapsed_seconds,
	}
	org_train_metrics = save_binary_classification_report_outputs(
		y_true=y_train,
		y_pred=train_y_pred,
		y_score=train_y_score,
		output_dir=output_dir,
		metrics_dir=metrics_dir,
		file_prefix="ORG_train",
		metric_prefix="train",
		base_metrics={**common_base, "feature_set": "ORG"},
	)
	org_validation_metrics = save_binary_classification_report_outputs(
		y_true=y_validation,
		y_pred=validation_y_pred,
		y_score=validation_y_score,
		output_dir=output_dir,
		metrics_dir=metrics_dir,
		file_prefix="ORG_validation",
		metric_prefix="validation",
		base_metrics={**common_base, "feature_set": "ORG", "validation_mse": validation_mse},
	)
	top_train_metrics = save_binary_classification_report_outputs(
		y_true=y_train,
		y_pred=filtered_train_y_pred,
		y_score=filtered_train_y_score,
		output_dir=output_dir,
		metrics_dir=metrics_dir,
		file_prefix=f"top_{feature_percent_tag}_train",
		metric_prefix="train",
		base_metrics={
			**common_base,
			"feature_set": f"top_{feature_percent_tag}",
			"feature_percent": feature_percent,
			"selected_feature_count": len(selected_df),
		},
	)
	top_validation_metrics = save_binary_classification_report_outputs(
		y_true=y_validation,
		y_pred=filtered_validation_y_pred,
		y_score=filtered_validation_y_score,
		output_dir=output_dir,
		metrics_dir=metrics_dir,
		file_prefix=f"top_{feature_percent_tag}_validation",
		metric_prefix="validation",
		base_metrics={
			**common_base,
			"feature_set": f"top_{feature_percent_tag}",
			"feature_percent": feature_percent,
			"selected_feature_count": len(selected_df),
			"validation_mse": filtered_validation_mse,
		},
	)
	summary = {
		"task": "classification",
		"mode": "explicit_train_validation",
		"dataset": dataset_folder,
		"validation_dataset": validation_dataset_folder,
		"feature_percent": feature_percent,
		"selected_feature_count": len(selected_df),
		"ORG_train": org_train_metrics,
		"ORG_validation": org_validation_metrics,
		f"top_{feature_percent_tag}_train": top_train_metrics,
		f"top_{feature_percent_tag}_validation": top_validation_metrics,
	}
	save_json(summary, metrics_dir / f"top_{feature_percent_tag}_train_validation_summary.json")

	print("\n[OK] Explicit train/validation classification tamamlandi.")
	print(f"[OK] ORG train accuracy: {org_train_metrics.get('train_accuracy'):.6f}")
	print(f"[OK] ORG validation accuracy: {org_validation_metrics.get('validation_accuracy'):.6f}")
	print(f"[OK] Top %{feature_percent} train accuracy: {top_train_metrics.get('train_accuracy'):.6f}")
	print(f"[OK] Top %{feature_percent} validation accuracy: {top_validation_metrics.get('validation_accuracy'):.6f}")
	print(f"[OK] Ozet metrik dosyasi: {metrics_dir / f'top_{feature_percent_tag}_train_validation_summary.json'}")
	return float(org_validation_metrics.get("validation_accuracy", 0.0)), float(
		top_validation_metrics.get("validation_accuracy", 0.0)
	)


def run_chunked_binary_experiment(
	df: pd.DataFrame,
	processed: dict,
	dataset_folder: str,
	target_column: str,
	id_column: str | None,
	encoding_dim: int,
	feature_percent: float,
	random_state: int | None,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
	classifier_model: str,
	classifier_early_stopping_patience: int | None,
	autoencoder_early_stopping_patience: int | None,
	classifier_early_stopping_monitor: str,
	classifier_early_stopping_min_delta: float,
	autoencoder_early_stopping_min_delta: float,
	classifier_class_weight: str,
	classifier_sampling: str,
	feature_chunk_size: int,
	save_training_plots: bool,
	current_class_label: int | None = None,
	class_counts: dict[int, int] | None = None,
) -> tuple[float, float]:
	X_train_raw = processed["X_train"]
	X_test_raw = processed["X_test"]
	y_train = processed["y_train"].to_numpy().astype(np.int32)
	y_test = processed["y_test"].to_numpy().astype(np.int32)
	feature_names = X_train_raw.columns.tolist()
	feature_chunks = split_feature_names_into_chunks(feature_names, feature_chunk_size)
	feature_percent_tag = format_feature_percent_tag(feature_percent)

	output_dir = classification_output_dir(dataset_folder)
	metrics_dir = output_dir / "metrics"
	chunks_dir = output_dir / "chunks"
	history_dir = output_dir / "training_history"
	filtered_data_dir = Path("data") / "filtered" / dataset_folder
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)
	ensure_dir(chunks_dir)
	if save_training_plots:
		ensure_dir(history_dir)
	ensure_dir(filtered_data_dir)

	print(
		f"[INFO] Büyük feature seti tespit edildi: {len(feature_names)} feature. "
		f"{len(feature_chunks)} parçaya bölünüyor (chunk_size={feature_chunk_size})."
	)

	chunk_selected_frames: list[pd.DataFrame] = []
	chunk_summaries: list[dict] = []

	for chunk_idx, chunk_feature_names in enumerate(feature_chunks, start=1):
		chunk_name = f"chunk_{chunk_idx:03d}"
		chunk_dir = chunks_dir / chunk_name
		ensure_dir(chunk_dir)

		print(
			f"\n[INFO] {chunk_name}/{len(feature_chunks):03d} egitimi basliyor "
			f"(feature sayisi: {len(chunk_feature_names)})."
		)

		X_train_chunk_raw = X_train_raw[chunk_feature_names]
		X_test_chunk_raw = X_test_raw[chunk_feature_names]
		X_train_chunk, X_test_chunk, _ = scale_data(X_train_chunk_raw, X_test_chunk_raw)
		X_train_chunk = X_train_chunk.astype(np.float32)
		X_test_chunk = X_test_chunk.astype(np.float32)

		(
			chunk_test_mse,
			_chunk_test_accuracy,
			chunk_autoencoder,
			_chunk_encoder,
			chunk_train_sub,
			_chunk_y_pred,
			_chunk_y_score,
		) = train_and_evaluate_pipeline(
			X_train=X_train_chunk,
			X_test=X_test_chunk,
			y_train=y_train,
			y_test=y_test,
			encoding_dim=encoding_dim,
			random_state=random_state,
			classifier_epochs=classifier_epochs,
			classifier_hidden_units=classifier_hidden_units,
			classifier_dropout_rates=classifier_dropout_rates,
			classifier_learning_rate=classifier_learning_rate,
			classifier_model=classifier_model,
			classifier_early_stopping_patience=classifier_early_stopping_patience,
			autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor=classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
			classifier_class_weight=classifier_class_weight,
			classifier_sampling=classifier_sampling,
			history_output_dir=history_dir if save_training_plots else None,
			history_prefix=chunk_name if save_training_plots else None,
		)

		chunk_weights_path = chunk_dir / "first_layer_W_list.csv"
		save_feature_weighted_lists(
			chunk_autoencoder,
			chunk_train_sub,
			chunk_feature_names,
			chunk_weights_path,
		)

		chunk_selected_path = chunk_dir / f"top_{feature_percent_tag}_max_abs_features.csv"
		chunk_selected_df = save_top_percent_features_by_abs_max_weight(
			weight_list_csv_path=chunk_weights_path,
			feature_names=chunk_feature_names,
			feature_percent=feature_percent,
			output_path=chunk_selected_path,
		)
		chunk_selected_df.insert(0, "chunk", chunk_name)
		chunk_selected_frames.append(chunk_selected_df)

		chunk_summaries.append(
			{
				"chunk": chunk_name,
				"feature_count": len(chunk_feature_names),
				"selected_feature_count": len(chunk_selected_df),
				"test_mse": chunk_test_mse,
				"test_rmse": compute_rmse_from_mse(chunk_test_mse),
				"test_accuracy": None,
				"weights_path": str(chunk_weights_path),
				"selected_features_path": str(chunk_selected_path),
			}
		)

		save_json(chunk_summaries[-1], metrics_dir / f"{chunk_name}_test_metrics.json")
		print(
			f"[OK] {chunk_name} tamamlandi. "
			f"Top %{feature_percent}: {len(chunk_selected_df)} feature."
		)

	all_chunk_selected_df = pd.concat(chunk_selected_frames, ignore_index=True)
	all_chunk_selected_path = output_dir / f"chunked_top_{feature_percent_tag}_max_abs_features.csv"
	all_chunk_selected_df.to_csv(all_chunk_selected_path, index=False)

	merged_feature_names = list(dict.fromkeys(all_chunk_selected_df["feature_name"].tolist()))
	merged_selected_df = pd.DataFrame(
		{
			"feature": [f"F{i+1}" for i in range(len(merged_feature_names))],
			"feature_name": merged_feature_names,
			"source": "chunked_top_features",
		}
	)
	merged_selected_path = output_dir / f"chunked_merged_top_{feature_percent_tag}_features.csv"
	merged_selected_df.to_csv(merged_selected_path, index=False)

	merged_dataset_path = filtered_data_dir / f"chunked_top_{feature_percent_tag}_max_abs_features_dataset.csv"
	merged_filtered_df = save_filtered_dataset_from_selected_features(
		full_df=df,
		selected_df=merged_selected_df,
		target_column=target_column,
		output_path=merged_dataset_path,
		id_column=id_column,
	)

	X_train_merged_raw = X_train_raw[merged_feature_names]
	X_test_merged_raw = X_test_raw[merged_feature_names]
	X_train_merged, X_test_merged, _ = scale_data(X_train_merged_raw, X_test_merged_raw)
	X_train_merged = X_train_merged.astype(np.float32)
	X_test_merged = X_test_merged.astype(np.float32)

	print(
		f"\n[INFO] Chunk top feature'lari birlestirildi: "
		f"{len(merged_feature_names)} feature. Final egitim basliyor."
	)
	(
		final_test_mse,
		final_test_accuracy,
		_final_autoencoder,
		_final_encoder,
		_final_train_sub,
		final_y_pred,
		final_y_score,
	) = train_and_evaluate_pipeline(
		X_train=X_train_merged,
		X_test=X_test_merged,
		y_train=y_train,
		y_test=y_test,
		encoding_dim=encoding_dim,
		random_state=random_state,
		classifier_epochs=classifier_epochs,
		classifier_hidden_units=classifier_hidden_units,
		classifier_dropout_rates=classifier_dropout_rates,
		classifier_learning_rate=classifier_learning_rate,
		classifier_model=classifier_model,
		classifier_early_stopping_patience=classifier_early_stopping_patience,
		autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
		classifier_early_stopping_monitor=classifier_early_stopping_monitor,
		classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
		autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
		classifier_class_weight=classifier_class_weight,
		classifier_sampling=classifier_sampling,
		history_output_dir=history_dir if save_training_plots else None,
		history_prefix=f"chunked_top_{feature_percent_tag}_final" if save_training_plots else None,
	)

	final_predictions_path = save_classification_predictions(
		y_true=y_test,
		y_pred=final_y_pred,
		y_score=final_y_score,
		output_dir=output_dir,
		file_prefix=f"top_{feature_percent_tag}",
	)
	final_classification_metrics = compute_binary_classification_metrics(
		y_true=y_test,
		y_pred=final_y_pred,
		y_score=final_y_score,
	)
	final_confusion_matrix_path = save_classification_confusion_matrix_plot(
		y_true=y_test,
		y_pred=final_y_pred,
		output_dir=output_dir,
		file_prefix=f"top_{feature_percent_tag}",
	)
	final_precision_recall_path = save_classification_precision_recall_plot(
		y_true=y_test,
		y_score=final_y_score,
		output_dir=output_dir,
		file_prefix=f"top_{feature_percent_tag}",
	)
	final_roc_path = save_classification_roc_plot(
		y_true=y_test,
		y_score=final_y_score,
		output_dir=output_dir,
		file_prefix=f"top_{feature_percent_tag}",
	)
	final_metrics_data = {
		"chunked_feature_selection": True,
		"feature_percent": format_encoded_output_percent(feature_percent, dataset_folder),
		"original_feature_count": len(feature_names),
		"feature_chunk_size": feature_chunk_size,
		"chunk_count": len(feature_chunks),
		"merged_feature_count": len(merged_feature_names),
		"test_mse": final_test_mse,
		"test_rmse": compute_rmse_from_mse(final_test_mse),
		"threshold": THRESHOLD,
		"classifier_model": classifier_model,
		"classifier_class_weight": classifier_class_weight,
		"classifier_sampling": classifier_sampling,
		"chunk_summaries": chunk_summaries,
		"all_chunk_selected_features_path": str(all_chunk_selected_path),
		"merged_selected_features_path": str(merged_selected_path),
		"merged_dataset_path": str(merged_dataset_path),
	}
	add_encoded_metric_metadata(final_metrics_data, feature_percent, dataset_folder)
	final_metrics_data.update(final_classification_metrics)
	if final_predictions_path is not None:
		final_metrics_data["classification_predictions_path"] = str(final_predictions_path)
	if final_confusion_matrix_path is not None:
		final_metrics_data["confusion_matrix_path"] = str(final_confusion_matrix_path)
	if final_precision_recall_path is not None:
		final_metrics_data["precision_recall_curve_path"] = str(final_precision_recall_path)
	if final_roc_path is not None:
		final_metrics_data["roc_curve_path"] = str(final_roc_path)
	if current_class_label is not None and class_counts is not None:
		final_metrics_data["current_class_label"] = current_class_label
		final_metrics_data["class_counts"] = class_counts
		final_metrics_data["binary_label_counts"] = {
			"label_0": int(np.sum(y_test == 0)),
			"label_1": int(np.sum(y_test == 1)),
		}

	final_metrics_filename = format_test_metrics_filename(feature_percent, dataset_folder)
	chunked_metrics_filename = f"chunked_{final_metrics_filename}"
	save_json(final_metrics_data, metrics_dir / chunked_metrics_filename)
	save_json(final_metrics_data, metrics_dir / final_metrics_filename)

	print("\n[OK] Chunked autoencoder akisi tamamlandi.")
	print(f"[OK] Orijinal feature sayisi: {len(feature_names)}")
	print(f"[OK] Chunk sayisi: {len(feature_chunks)}")
	print(f"[OK] Birlesen top feature sayisi: {len(merged_feature_names)}")
	print(f"[OK] Final dataset test_mse: {final_test_mse:.6f}")
	print(f"[OK] Final dataset test_accuracy: {final_test_accuracy:.6f}")
	print(f"[OK] Chunk secim CSV: {all_chunk_selected_path}")
	print(f"[OK] Birlesik feature CSV: {merged_selected_path}")
	print(f"[OK] Birlesik dataset CSV: {merged_dataset_path} (satir: {len(merged_filtered_df)})")
	print(f"[OK] Final metrik dosyasi: {metrics_dir / chunked_metrics_filename}")
	return final_test_accuracy, final_test_accuracy


def run_binary_experiment(
	df: pd.DataFrame,
	dataset_folder: str,
	target_column: str,
	id_column: str | None,
	encoding_dim: int,
	feature_percent: float,
	random_state: int | None,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
	classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
	feature_chunk_size: int = DEFAULT_FEATURE_CHUNK_SIZE,
	chunk_feature_threshold: int = DEFAULT_CHUNK_FEATURE_THRESHOLD,
	enable_feature_chunking: bool = True,
	classifier_early_stopping_patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE,
	autoencoder_early_stopping_patience: int | None = DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE,
	classifier_early_stopping_monitor: str = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR,
	classifier_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	autoencoder_early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	classifier_class_weight: str = "none",
	classifier_sampling: str = "none",
	save_training_plots: bool = False,
	current_class_label: int | None = None,
	class_counts: dict[int, int] | None = None,
) -> tuple[float, float]:
	start_time = time.perf_counter()
	processed = preprocess_data(
		df,
		target_column=target_column,
		id_column=id_column,
		random_state=random_state,
		scale_features=False,
	)
	X_train_raw = processed["X_train"]
	X_test_raw = processed["X_test"]
	y_train = processed["y_train"].to_numpy().astype(np.int32)
	y_test = processed["y_test"].to_numpy().astype(np.int32)
	if not set(np.unique(y_train)).issubset({0, 1}) or not set(np.unique(y_test)).issubset({0, 1}):
		raise ValueError("Bu script binary etiket bekliyor. Label degerleri sadece 0 ve 1 olmali.")

	print(f"[INFO] X_train shape: {X_train_raw.shape}")
	print(f"[INFO] X_test shape : {X_test_raw.shape}")

	if should_use_feature_chunking(
		feature_count=X_train_raw.shape[1],
		chunk_feature_threshold=chunk_feature_threshold,
		feature_chunk_size=feature_chunk_size,
		enable_feature_chunking=enable_feature_chunking,
	):
		return run_chunked_binary_experiment(
			df=df,
			processed=processed,
			dataset_folder=dataset_folder,
			target_column=target_column,
			id_column=id_column,
			encoding_dim=encoding_dim,
			feature_percent=feature_percent,
				random_state=random_state,
				classifier_epochs=classifier_epochs,
				classifier_hidden_units=classifier_hidden_units,
				classifier_dropout_rates=classifier_dropout_rates,
				classifier_learning_rate=classifier_learning_rate,
				classifier_model=classifier_model,
				classifier_early_stopping_patience=classifier_early_stopping_patience,
				autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
				classifier_early_stopping_monitor=classifier_early_stopping_monitor,
				classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
				autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
				classifier_class_weight=classifier_class_weight,
				classifier_sampling=classifier_sampling,
				feature_chunk_size=feature_chunk_size,
				save_training_plots=save_training_plots,
				current_class_label=current_class_label,
				class_counts=class_counts,
			)

	X_train, X_test, _ = scale_data(X_train_raw, X_test_raw)
	X_train = X_train.astype(np.float32)
	X_test = X_test.astype(np.float32)

	output_dir = classification_output_dir(dataset_folder)
	metrics_dir = output_dir / "metrics"
	history_dir = output_dir / "training_history"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)
	if save_training_plots:
		ensure_dir(history_dir)

	if is_encoded_dataset_folder(dataset_folder):
		print("[INFO] Dimension-reduction/encoded dataset tespit edildi.")
		print("[INFO] FeatureRank ve ikinci autoencoder uygulanmayacak; direkt reduced feature'lar ile classifier egitilecek.")
		print(
			"[WARN] Onceden uretilmis tek encoded CSV yeniden split ediliyor. "
			"Bu sonuc makaledeki held-out performans icin kullanilmamali; "
			"ham dataset ile --evaluate-dimension-reduction kullanin."
		)
		output_feature_prefix = format_metric_output_prefix(feature_percent, dataset_folder)
		output_feature_label = format_feature_output_label(feature_percent, dataset_folder)
		filtered_metrics_filename = format_test_metrics_filename(feature_percent, dataset_folder)

		direct_test_accuracy, direct_y_pred, direct_y_score = train_and_evaluate_direct_classifier(
			X_train=X_train,
			X_test=X_test,
			y_train=y_train,
			y_test=y_test,
			random_state=random_state,
			classifier_epochs=classifier_epochs,
			classifier_hidden_units=classifier_hidden_units,
			classifier_dropout_rates=classifier_dropout_rates,
			classifier_learning_rate=classifier_learning_rate,
			classifier_model=classifier_model,
			classifier_early_stopping_patience=classifier_early_stopping_patience,
			classifier_early_stopping_monitor=classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
			classifier_class_weight=classifier_class_weight,
			classifier_sampling=classifier_sampling,
			history_output_dir=history_dir if save_training_plots else None,
			history_prefix=output_feature_prefix if save_training_plots else None,
		)

		elapsed_seconds = time.perf_counter() - start_time
		predictions_path = save_classification_predictions(
			y_true=y_test,
			y_pred=direct_y_pred,
			y_score=direct_y_score,
			output_dir=output_dir,
			file_prefix=output_feature_prefix,
		)
		confusion_matrix_path = save_classification_confusion_matrix_plot(
			y_true=y_test,
			y_pred=direct_y_pred,
			output_dir=output_dir,
			file_prefix=output_feature_prefix,
		)
		precision_recall_path = save_classification_precision_recall_plot(
			y_true=y_test,
			y_score=direct_y_score,
			output_dir=output_dir,
			file_prefix=output_feature_prefix,
		)
		roc_path = save_classification_roc_plot(
			y_true=y_test,
			y_score=direct_y_score,
			output_dir=output_dir,
			file_prefix=output_feature_prefix,
		)
		direct_metrics_data = {
			"feature_percent": format_encoded_output_percent(feature_percent, dataset_folder),
			"selected_feature_count": int(X_train_raw.shape[1]),
			"latent_feature_count": int(X_train_raw.shape[1]),
			"dimension_reduction_dataset": True,
			"classification_input": "direct_dimension_reduced_features",
			"test_mse": None,
			"test_rmse": None,
			"threshold": THRESHOLD,
			"elapsed_seconds": elapsed_seconds,
			"classifier_model": classifier_model,
			"classifier_class_weight": classifier_class_weight,
			"classifier_sampling": classifier_sampling,
		}
		add_encoded_metric_metadata(direct_metrics_data, feature_percent, dataset_folder)
		direct_metrics_data.update(
			compute_binary_classification_metrics(
				y_true=y_test,
				y_pred=direct_y_pred,
				y_score=direct_y_score,
			)
		)
		if predictions_path is not None:
			direct_metrics_data["classification_predictions_path"] = str(predictions_path)
		if confusion_matrix_path is not None:
			direct_metrics_data["confusion_matrix_path"] = str(confusion_matrix_path)
		if precision_recall_path is not None:
			direct_metrics_data["precision_recall_curve_path"] = str(precision_recall_path)
		if roc_path is not None:
			direct_metrics_data["roc_curve_path"] = str(roc_path)
		if current_class_label is not None and class_counts is not None:
			direct_metrics_data["current_class_label"] = current_class_label
			direct_metrics_data["class_counts"] = class_counts
			direct_metrics_data["binary_label_counts"] = {
				"label_0": int(np.sum(y_test == 0)),
				"label_1": int(np.sum(y_test == 1)),
			}

		save_json(direct_metrics_data, metrics_dir / "ORG_test_metrics.json")
		save_json(direct_metrics_data, metrics_dir / filtered_metrics_filename)

		print("\n[OK] Dimension-reduction classifier egitimi tamamlandi.")
		print(f"[OK] Direkt latent feature sayisi: {X_train_raw.shape[1]}")
		print(f"[OK] {output_feature_label} dataset test_accuracy: {direct_test_accuracy:.6f}")
		print(f"[OK] Calisma suresi: {elapsed_seconds:.2f} saniye")
		print(f"[OK] {output_feature_label} metrik dosyasi: {metrics_dir / filtered_metrics_filename}")
		return 0.0, direct_test_accuracy

	(
		test_mse,
		test_accuracy,
		autoencoder,
		_org_encoder,
		X_train_sub_used,
		org_y_pred,
		org_y_score,
	) = train_and_evaluate_pipeline(
		X_train=X_train,
		X_test=X_test,
		y_train=y_train,
		y_test=y_test,
		encoding_dim=encoding_dim,
		random_state=random_state,
		classifier_epochs=classifier_epochs,
		classifier_hidden_units=classifier_hidden_units,
		classifier_dropout_rates=classifier_dropout_rates,
		classifier_learning_rate=classifier_learning_rate,
		classifier_model=classifier_model,
		classifier_early_stopping_patience=classifier_early_stopping_patience,
		autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
		classifier_early_stopping_monitor=classifier_early_stopping_monitor,
		classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
		autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
		classifier_class_weight=classifier_class_weight,
		classifier_sampling=classifier_sampling,
		history_output_dir=history_dir if save_training_plots else None,
		history_prefix="ORG" if save_training_plots else None,
	)

	feature_names = X_train_raw.columns.tolist()
	feature_percent_tag = format_feature_percent_tag(feature_percent)
	output_feature_percent = format_encoded_output_percent(feature_percent, dataset_folder)
	output_feature_prefix = format_metric_output_prefix(feature_percent, dataset_folder)
	output_feature_label = format_feature_output_label(feature_percent, dataset_folder)
	selected_features_path = output_dir / f"top_{feature_percent_tag}_max_abs_features.csv"
	selected_df = None
	if selected_features_path.exists():
		selected_df = load_selected_features_if_compatible(selected_features_path, feature_names)
		if selected_df is not None:
			print(f"[INFO] Mevcut feature listesi kullaniliyor: {selected_features_path}")
	if selected_df is None:
		weights_path = output_dir / "first_layer_W_list.csv"
		save_feature_weighted_lists(autoencoder, X_train_sub_used, feature_names, weights_path)
		selected_df = save_top_percent_features_by_abs_max_weight(
			weight_list_csv_path=weights_path,
			feature_names=feature_names,
			feature_percent=feature_percent,
			output_path=selected_features_path,
		)

	filtered_data_dir = Path("data") / "filtered" / dataset_folder
	ensure_dir(filtered_data_dir)
	filtered_dataset_path = filtered_data_dir / f"top_{feature_percent_tag}_max_abs_features_dataset.csv"
	filtered_df = save_filtered_dataset_from_selected_features(
		full_df=df,
		selected_df=selected_df,
		target_column=target_column,
		output_path=filtered_dataset_path,
		id_column=id_column,
	)

	y_train_filtered = y_train
	y_test_filtered = y_test
	if len(selected_df) == len(feature_names):
		filtered_test_mse = test_mse
		filtered_test_accuracy = test_accuracy
		filtered_y_pred = org_y_pred
		filtered_y_score = org_y_score
		print(f"[INFO] {output_feature_label} tum secili/encoded feature'lari iceriyor. ORG sonucu yeniden egitilmeden kullaniliyor.")
	else:
		selected_feature_names = selected_df["feature_name"].tolist()
		X_train_filtered_raw = X_train_raw[selected_feature_names]
		X_test_filtered_raw = X_test_raw[selected_feature_names]
		X_train_filtered, X_test_filtered, _ = scale_data(X_train_filtered_raw, X_test_filtered_raw)
		# Ensure consistent float32 dtype
		X_train_filtered = X_train_filtered.astype(np.float32)
		X_test_filtered = X_test_filtered.astype(np.float32)
		(
			filtered_test_mse,
			filtered_test_accuracy,
			_filtered_autoencoder,
			_filtered_encoder,
			_filtered_train_sub,
			filtered_y_pred,
			filtered_y_score,
		) = train_and_evaluate_pipeline(
			X_train=X_train_filtered,
			X_test=X_test_filtered,
			y_train=y_train_filtered,
			y_test=y_test_filtered,
			encoding_dim=encoding_dim,
			random_state=random_state,
			classifier_epochs=classifier_epochs,
			classifier_hidden_units=classifier_hidden_units,
			classifier_dropout_rates=classifier_dropout_rates,
			classifier_learning_rate=classifier_learning_rate,
			classifier_model=classifier_model,
			classifier_early_stopping_patience=classifier_early_stopping_patience,
			autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor=classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
			classifier_class_weight=classifier_class_weight,
			classifier_sampling=classifier_sampling,
			history_output_dir=history_dir if save_training_plots else None,
			history_prefix=output_feature_prefix if save_training_plots else None,
		)

	elapsed_seconds = time.perf_counter() - start_time
	org_predictions_path = save_classification_predictions(
		y_true=y_test,
		y_pred=org_y_pred,
		y_score=org_y_score,
		output_dir=output_dir,
		file_prefix="ORG",
	)
	org_classification_metrics = compute_binary_classification_metrics(
		y_true=y_test,
		y_pred=org_y_pred,
		y_score=org_y_score,
	)
	org_confusion_matrix_path = save_classification_confusion_matrix_plot(
		y_true=y_test,
		y_pred=org_y_pred,
		output_dir=output_dir,
		file_prefix="ORG",
	)
	org_precision_recall_path = save_classification_precision_recall_plot(
		y_true=y_test,
		y_score=org_y_score,
		output_dir=output_dir,
		file_prefix="ORG",
	)
	org_roc_path = save_classification_roc_plot(
		y_true=y_test,
		y_score=org_y_score,
		output_dir=output_dir,
		file_prefix="ORG",
	)
	filtered_predictions_path = save_classification_predictions(
		y_true=y_test_filtered,
		y_pred=filtered_y_pred,
		y_score=filtered_y_score,
		output_dir=output_dir,
		file_prefix=output_feature_prefix,
	)
	filtered_classification_metrics = compute_binary_classification_metrics(
		y_true=y_test_filtered,
		y_pred=filtered_y_pred,
		y_score=filtered_y_score,
	)
	filtered_confusion_matrix_path = save_classification_confusion_matrix_plot(
		y_true=y_test_filtered,
		y_pred=filtered_y_pred,
		output_dir=output_dir,
		file_prefix=output_feature_prefix,
	)
	filtered_precision_recall_path = save_classification_precision_recall_plot(
		y_true=y_test_filtered,
		y_score=filtered_y_score,
		output_dir=output_dir,
		file_prefix=output_feature_prefix,
	)
	filtered_roc_path = save_classification_roc_plot(
		y_true=y_test_filtered,
		y_score=filtered_y_score,
		output_dir=output_dir,
		file_prefix=output_feature_prefix,
	)
	org_metrics_data = {
		"test_mse": test_mse,
		"test_rmse": compute_rmse_from_mse(test_mse),
		"threshold": THRESHOLD,
		"elapsed_seconds": elapsed_seconds,
		"classifier_model": classifier_model,
		"classifier_class_weight": classifier_class_weight,
		"classifier_sampling": classifier_sampling,
	}
	org_metrics_data.update(org_classification_metrics)
	if org_predictions_path is not None:
		org_metrics_data["classification_predictions_path"] = str(org_predictions_path)
	if org_confusion_matrix_path is not None:
		org_metrics_data["confusion_matrix_path"] = str(org_confusion_matrix_path)
	if org_precision_recall_path is not None:
		org_metrics_data["precision_recall_curve_path"] = str(org_precision_recall_path)
	if org_roc_path is not None:
		org_metrics_data["roc_curve_path"] = str(org_roc_path)
	if current_class_label is not None and class_counts is not None:
		org_metrics_data["current_class_label"] = current_class_label
		org_metrics_data["class_counts"] = class_counts
		# Binary model'de label 0 ve 1 sayılarını ekle
		label_0_count = int(np.sum(y_test == 0))
		label_1_count = int(np.sum(y_test == 1))
		org_metrics_data["binary_label_counts"] = {
			"label_0": label_0_count,
			"label_1": label_1_count
		}

	save_json(
		org_metrics_data,
		metrics_dir / "ORG_test_metrics.json",
	)

	filtered_metrics_filename = format_test_metrics_filename(feature_percent, dataset_folder)
	filtered_metrics_data = {
		"feature_percent": output_feature_percent,
		"selected_feature_count": len(selected_df),
		"test_mse": filtered_test_mse,
		"test_rmse": compute_rmse_from_mse(filtered_test_mse),
		"threshold": THRESHOLD,
		"elapsed_seconds": elapsed_seconds,
		"classifier_model": classifier_model,
		"classifier_class_weight": classifier_class_weight,
		"classifier_sampling": classifier_sampling,
	}
	add_encoded_metric_metadata(filtered_metrics_data, feature_percent, dataset_folder)
	filtered_metrics_data.update(filtered_classification_metrics)
	if filtered_predictions_path is not None:
		filtered_metrics_data["classification_predictions_path"] = str(filtered_predictions_path)
	if filtered_confusion_matrix_path is not None:
		filtered_metrics_data["confusion_matrix_path"] = str(filtered_confusion_matrix_path)
	if filtered_precision_recall_path is not None:
		filtered_metrics_data["precision_recall_curve_path"] = str(filtered_precision_recall_path)
	if filtered_roc_path is not None:
		filtered_metrics_data["roc_curve_path"] = str(filtered_roc_path)
	if current_class_label is not None and class_counts is not None:
		filtered_metrics_data["current_class_label"] = current_class_label
		filtered_metrics_data["class_counts"] = class_counts
		# Binary model'de label 0 ve 1 sayılarını ekle (filtered dataset)
		label_0_count_filtered = int(np.sum(y_test_filtered == 0))
		label_1_count_filtered = int(np.sum(y_test_filtered == 1))
		filtered_metrics_data["binary_label_counts"] = {
			"label_0": label_0_count_filtered,
			"label_1": label_1_count_filtered
		}

	save_json(
		filtered_metrics_data,
		metrics_dir / filtered_metrics_filename,
	)

	print("\n[OK] Autoencoder egitimi tamamlandi.")
	#print(f"[OK] test_mse: {test_mse:.6f}")
	print(f"[OK] test_accuracy: {test_accuracy:.6f}")
	#print(f"[OK] Feature weighted listeleri: {weights_path}")
	print(f"[OK] {output_feature_label} secilen feature sayisi: {len(selected_df)}")
	print(f"[OK] Secilen feature CSV: {selected_features_path}")
	#print(f"[OK] Filterlenmis dataset CSV: {filtered_dataset_path} (satir: {len(filtered_df)})")
	#print(f"[OK] Top %{feature_percent} dataset test_mse: {filtered_test_mse:.6f}")
	print(f"[OK] {output_feature_label} dataset test_accuracy: {filtered_test_accuracy:.6f}")
	print(f"[OK] Calisma suresi: {elapsed_seconds:.2f} saniye")
	filtered_metrics_path = metrics_dir / filtered_metrics_filename
	print(f"[OK] {output_feature_label} metrik dosyasi: {filtered_metrics_path}")
	#print(f"[OK] Output klasoru: {output_dir}")
	#print(f"[OK] Metrik dosyasi: {metrics_dir / 'ORG_test_metrics.json'}")
	return test_accuracy, filtered_test_accuracy


def run_regression_experiment(
	df: pd.DataFrame,
	dataset_folder: str,
	target_column: str,
	id_column: str | None,
	encoding_dim: int,
	feature_percent: float,
	random_state: int | None,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
	regression_model: str = DEFAULT_REGRESSION_MODEL,
	svr_kernel: str = DEFAULT_SVR_KERNEL,
	svr_c: float = DEFAULT_SVR_C,
	svr_epsilon: float = DEFAULT_SVR_EPSILON,
	svr_gamma: str | float = DEFAULT_SVR_GAMMA,
	kmeans_regression_clusters: int = DEFAULT_KMEANS_REGRESSION_CLUSTERS,
	kmeans_regression_n_init: int = DEFAULT_KMEANS_REGRESSION_N_INIT,
	classifier_early_stopping_patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE,
	autoencoder_early_stopping_patience: int | None = DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE,
	classifier_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	autoencoder_early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	save_training_plots: bool = False,
	actual_predicted_top_n: int | None = None,
) -> tuple[float, float]:
	start_time = time.perf_counter()
	processed = preprocess_data(
		df,
		target_column=target_column,
		id_column=id_column,
		random_state=random_state,
		scale_features=False,
		task_type="regression",
	)
	X_train_raw = processed["X_train"]
	X_test_raw = processed["X_test"]
	y_train = processed["y_train"].to_numpy().astype(np.float32)
	y_test = processed["y_test"].to_numpy().astype(np.float32)
	target_normalization_info = None
	if dataset_folder.lower() == "energy_data":
		target_min = float(np.min(y_train))
		target_max = float(np.max(y_train))
		target_range = target_max - target_min
		if np.isclose(target_range, 0.0):
			print("[WARN] Energy target normalization atlandi: train label araligi 0.")
		else:
			y_train = ((y_train - target_min) / target_range).astype(np.float32)
			y_test = ((y_test - target_min) / target_range).astype(np.float32)
			target_normalization_info = {
				"method": "min_max",
				"applied_to": "target",
				"dataset": dataset_folder,
				"fit_on": "train_target",
				"target_min": target_min,
				"target_max": target_max,
			}
			print(
				"[INFO] Energy target min-max normalize edildi: "
				f"min={target_min:.6f}, max={target_max:.6f}"
			)

	print(f"[INFO] Regression modu. X_train shape: {X_train_raw.shape}")
	print(f"[INFO] X_test shape : {X_test_raw.shape}")

	X_train, X_test, _ = scale_data(X_train_raw, X_test_raw)
	X_train = X_train.astype(np.float32)
	X_test = X_test.astype(np.float32)

	output_dir = regression_output_dir(dataset_folder)
	metrics_dir = output_dir / "metrics"
	history_dir = output_dir / "training_history"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)
	if save_training_plots:
		ensure_dir(history_dir)

	org_metrics, autoencoder, _, X_train_sub_used, org_y_true, org_y_pred, org_train_y_true, org_train_y_pred = train_and_evaluate_regression_pipeline(
		X_train=X_train,
		X_test=X_test,
		y_train=y_train,
		y_test=y_test,
		encoding_dim=encoding_dim,
		random_state=random_state,
		regressor_epochs=classifier_epochs,
		regressor_hidden_units=classifier_hidden_units,
		regressor_dropout_rates=classifier_dropout_rates,
		regressor_learning_rate=classifier_learning_rate,
		regression_model=regression_model,
		svr_kernel=svr_kernel,
		svr_c=svr_c,
		svr_epsilon=svr_epsilon,
		svr_gamma=svr_gamma,
		kmeans_regression_clusters=kmeans_regression_clusters,
		kmeans_regression_n_init=kmeans_regression_n_init,
		regressor_early_stopping_patience=classifier_early_stopping_patience,
		autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
		regressor_early_stopping_min_delta=classifier_early_stopping_min_delta,
		autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
		history_output_dir=history_dir if save_training_plots else None,
		history_prefix="ORG" if save_training_plots else None,
	)
	save_regression_actual_vs_predicted_plot(
		y_true=org_y_true,
		y_pred=org_y_pred,
		output_dir=output_dir,
		file_prefix="ORG",
		top_n=actual_predicted_top_n,
	)
	org_prediction_errors_path = save_regression_prediction_errors(
		y_true=org_y_true,
		y_pred=org_y_pred,
		sample_index=X_test_raw.index,
		output_dir=output_dir,
		file_prefix="ORG",
	)
	org_train_prediction_errors_path = save_regression_prediction_errors(
		y_true=org_train_y_true,
		y_pred=org_train_y_pred,
		sample_index=range(len(org_train_y_true)),
		output_dir=output_dir,
		file_prefix="ORG_train",
	)

	feature_names = X_train_raw.columns.tolist()
	feature_percent_tag = format_feature_percent_tag(feature_percent)
	selected_features_path = output_dir / f"top_{feature_percent_tag}_max_abs_features.csv"
	selected_df = None
	if selected_features_path.exists():
		selected_df = load_selected_features_if_compatible(selected_features_path, feature_names)
		if selected_df is not None:
			print(f"[INFO] Mevcut feature listesi kullaniliyor: {selected_features_path}")
	if selected_df is None:
		weights_path = output_dir / "first_layer_W_list.csv"
		save_feature_weighted_lists(autoencoder, X_train_sub_used, feature_names, weights_path)
		selected_df = save_top_percent_features_by_abs_max_weight(
			weight_list_csv_path=weights_path,
			feature_names=feature_names,
			feature_percent=feature_percent,
			output_path=selected_features_path,
		)

	filtered_data_dir = Path("data") / "filtered" / dataset_folder
	ensure_dir(filtered_data_dir)
	filtered_dataset_path = filtered_data_dir / f"top_{feature_percent_tag}_max_abs_features_dataset.csv"
	save_filtered_dataset_from_selected_features(
		full_df=df,
		selected_df=selected_df,
		target_column=target_column,
		output_path=filtered_dataset_path,
		id_column=id_column,
	)

	if len(selected_df) == len(feature_names):
		filtered_metrics = dict(org_metrics)
		filtered_y_true = org_y_true
		filtered_y_pred = org_y_pred
		filtered_train_y_true = org_train_y_true
		filtered_train_y_pred = org_train_y_pred
		print("[INFO] Top %100 tum feature'lari iceriyor. ORG regression sonucu yeniden egitilmeden kullaniliyor.")
	else:
		selected_feature_names = selected_df["feature_name"].tolist()
		X_train_filtered_raw = X_train_raw[selected_feature_names]
		X_test_filtered_raw = X_test_raw[selected_feature_names]
		X_train_filtered, X_test_filtered, _ = scale_data(X_train_filtered_raw, X_test_filtered_raw)
		filtered_metrics, _, _, _, filtered_y_true, filtered_y_pred, filtered_train_y_true, filtered_train_y_pred = train_and_evaluate_regression_pipeline(
			X_train=X_train_filtered.astype(np.float32),
			X_test=X_test_filtered.astype(np.float32),
			y_train=y_train,
			y_test=y_test,
			encoding_dim=encoding_dim,
			random_state=random_state,
			regressor_epochs=classifier_epochs,
			regressor_hidden_units=classifier_hidden_units,
			regressor_dropout_rates=classifier_dropout_rates,
			regressor_learning_rate=classifier_learning_rate,
			regression_model=regression_model,
			svr_kernel=svr_kernel,
			svr_c=svr_c,
			svr_epsilon=svr_epsilon,
			svr_gamma=svr_gamma,
			kmeans_regression_clusters=kmeans_regression_clusters,
			kmeans_regression_n_init=kmeans_regression_n_init,
			regressor_early_stopping_patience=classifier_early_stopping_patience,
			autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
			regressor_early_stopping_min_delta=classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
			history_output_dir=history_dir if save_training_plots else None,
			history_prefix=f"top_{feature_percent_tag}" if save_training_plots else None,
		)
	save_regression_actual_vs_predicted_plot(
		y_true=filtered_y_true,
		y_pred=filtered_y_pred,
		output_dir=output_dir,
		file_prefix=f"top_{feature_percent_tag}",
		top_n=actual_predicted_top_n,
	)
	filtered_prediction_errors_path = save_regression_prediction_errors(
		y_true=filtered_y_true,
		y_pred=filtered_y_pred,
		sample_index=X_test_raw.index,
		output_dir=output_dir,
		file_prefix=f"top_{feature_percent_tag}",
	)
	filtered_train_prediction_errors_path = save_regression_prediction_errors(
		y_true=filtered_train_y_true,
		y_pred=filtered_train_y_pred,
		sample_index=range(len(filtered_train_y_true)),
		output_dir=output_dir,
		file_prefix=f"top_{feature_percent_tag}_train",
	)

	elapsed_seconds = time.perf_counter() - start_time
	org_metrics_data = {"task": "regression", **org_metrics}
	org_metrics_data["elapsed_seconds"] = elapsed_seconds
	if target_normalization_info is not None:
		org_metrics_data["target_normalization"] = target_normalization_info
	if org_prediction_errors_path is not None:
		org_metrics_data["prediction_errors_path"] = str(org_prediction_errors_path)
	if org_train_prediction_errors_path is not None:
		org_metrics_data["train_prediction_errors_path"] = str(org_train_prediction_errors_path)
	filtered_metrics_data = {
		"task": "regression",
		"feature_percent": feature_percent,
		"selected_feature_count": len(selected_df),
		"elapsed_seconds": elapsed_seconds,
		**filtered_metrics,
	}
	if target_normalization_info is not None:
		filtered_metrics_data["target_normalization"] = target_normalization_info
	if filtered_prediction_errors_path is not None:
		filtered_metrics_data["prediction_errors_path"] = str(filtered_prediction_errors_path)
	if filtered_train_prediction_errors_path is not None:
		filtered_metrics_data["train_prediction_errors_path"] = str(filtered_train_prediction_errors_path)
	save_json(org_metrics_data, metrics_dir / "ORG_test_metrics.json")
	save_json(filtered_metrics_data, metrics_dir / f"top_{feature_percent_tag}_test_metrics.json")

	print("\n[OK] Regression autoencoder egitimi tamamlandi.")
	print(f"[OK] ORG R2: {org_metrics['regression_r2']:.6f}")
	print(f"[OK] ORG RMSE: {org_metrics['regression_rmse']:.6f}")
	print(f"[OK] Top %{feature_percent} secilen feature sayisi: {len(selected_df)}")
	print(f"[OK] Top %{feature_percent} R2: {filtered_metrics['regression_r2']:.6f}")
	print(f"[OK] Top %{feature_percent} RMSE: {filtered_metrics['regression_rmse']:.6f}")
	print(f"[OK] Calisma suresi: {elapsed_seconds:.2f} saniye")
	print(f"[OK] Metrik dosyasi: {metrics_dir / f'top_{feature_percent_tag}_test_metrics.json'}")
	return org_metrics["pearson_r"], filtered_metrics["pearson_r"]


def run_multiclass_one_vs_rest(
	df: pd.DataFrame,
	dataset_folder: str,
	target_column: str,
	id_column: str | None,
	encoding_dim: int,
	feature_percent: float,
	random_state: int | None,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
	classifier_model: str,
	feature_chunk_size: int,
	classifier_early_stopping_patience: int | None,
	autoencoder_early_stopping_patience: int | None,
	classifier_early_stopping_monitor: str,
	classifier_early_stopping_min_delta: float,
	autoencoder_early_stopping_min_delta: float,
	classifier_class_weight: str,
	classifier_sampling: str,
	chunk_feature_threshold: int,
	enable_feature_chunking: bool,
	save_training_plots: bool = False,
) -> tuple[float, float]:
	class_labels = sorted(df[target_column].dropna().unique().tolist())
	if len(class_labels) <= 2:
		raise ValueError("run_multiclass_one_vs_rest sadece 2'den fazla sinif icin kullanilmali.")

	print(f"[INFO] Multi-class tespit edildi. Siniflar: {class_labels}")
	
	# Orijinal df'den multiclass label distribution hesapla (weighted average için)
	# preprocessing sonrası y_test encoded olur, bu yüzden orijinal df'den count alalım
	class_counts = {label: int((df[target_column] == label).sum()) for label in class_labels}
	print(f"[INFO] Dataset class sayilari: {class_counts}")
	
	for class_label in class_labels:
		binary_df = df.copy()
		# Istek: secili class 0, diger tum class'lar 1
		binary_df[target_column] = (binary_df[target_column] != class_label).astype(np.int32)
		binary_dataset_folder = f"{class_label}_{dataset_folder}"
		nested_binary_folder = str(Path(dataset_folder) / binary_dataset_folder)

		print(f"\n[INFO] One-vs-rest egitimi basliyor: class={class_label}, klasor={binary_dataset_folder}")
		run_binary_experiment(
			df=binary_df,
			dataset_folder=nested_binary_folder,
			target_column=target_column,
			id_column=id_column,
			encoding_dim=encoding_dim,
			feature_percent=feature_percent,
			random_state=random_state,
			classifier_epochs=classifier_epochs,
				classifier_hidden_units=classifier_hidden_units,
				classifier_dropout_rates=classifier_dropout_rates,
				classifier_learning_rate=classifier_learning_rate,
				classifier_model=classifier_model,
				classifier_early_stopping_patience=classifier_early_stopping_patience,
				autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
				classifier_early_stopping_monitor=classifier_early_stopping_monitor,
				classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
				autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
				classifier_class_weight=classifier_class_weight,
				classifier_sampling=classifier_sampling,
				feature_chunk_size=feature_chunk_size,
				chunk_feature_threshold=chunk_feature_threshold,
				enable_feature_chunking=enable_feature_chunking,
				save_training_plots=save_training_plots,
				current_class_label=class_label,
				class_counts=class_counts,
			)

	output_feature_percent = format_encoded_output_percent(feature_percent, dataset_folder)
	output_feature_percent_tag = format_feature_percent_tag(output_feature_percent)
	output_feature_label = format_feature_output_label(feature_percent, dataset_folder)
	filtered_metric_filename = format_test_metrics_filename(feature_percent, dataset_folder)
	filtered_summary_metrics = compute_multiclass_one_vs_rest_metric_summary(
		dataset_folder=dataset_folder,
		class_labels=class_labels,
		metric_filename=filtered_metric_filename,
	)
	macro_filtered_accuracy = float(filtered_summary_metrics["test_accuracy"])
	try:
		org_summary_metrics = compute_multiclass_one_vs_rest_metric_summary(
			dataset_folder=dataset_folder,
			class_labels=class_labels,
			metric_filename="ORG_test_metrics.json",
		)
		macro_org_accuracy = float(org_summary_metrics["test_accuracy"])
	except FileNotFoundError:
		org_summary_metrics = dict(filtered_summary_metrics)
		macro_org_accuracy = macro_filtered_accuracy

	output_dir = classification_output_dir(dataset_folder)
	metrics_dir = output_dir / "metrics"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)

	org_class_metric_rows = org_summary_metrics.pop("class_metric_rows", [])
	filtered_class_metric_rows = filtered_summary_metrics.pop("class_metric_rows", [])
	org_class_metrics_path = metrics_dir / "ORG_multiclass_class_metrics.csv"
	filtered_class_metrics_path = metrics_dir / f"top_{output_feature_percent_tag}_multiclass_class_metrics.csv"
	if org_class_metric_rows:
		pd.DataFrame(org_class_metric_rows).to_csv(org_class_metrics_path, index=False)
	if filtered_class_metric_rows:
		pd.DataFrame(filtered_class_metric_rows).to_csv(filtered_class_metrics_path, index=False)

	save_json(
		{
			"num_classes": len(class_labels),
			"class_labels": class_labels,
			"macro_average": True,
			"class_metrics_path": str(org_class_metrics_path) if org_class_metric_rows else None,
			**org_summary_metrics,
		},
		metrics_dir / "ORG_test_metrics.json",
	)

	filtered_multiclass_metrics_data = {
		"feature_percent": output_feature_percent,
		"num_classes": len(class_labels),
		"class_labels": class_labels,
		"macro_average": True,
		"class_metrics_path": str(filtered_class_metrics_path) if filtered_class_metric_rows else None,
		**filtered_summary_metrics,
	}
	add_encoded_metric_metadata(filtered_multiclass_metrics_data, feature_percent, dataset_folder)
	save_json(filtered_multiclass_metrics_data, metrics_dir / filtered_metric_filename)

	print("\n[OK] Multi-class one-vs-rest tamamlandi.")
	print(f"[OK] ORG macro test_accuracy: {macro_org_accuracy:.6f}")
	if org_summary_metrics.get("test_precision") is not None:
		print(f"[OK] ORG macro test_precision: {float(org_summary_metrics['test_precision']):.6f}")
	if org_summary_metrics.get("test_recall") is not None:
		print(f"[OK] ORG macro test_recall: {float(org_summary_metrics['test_recall']):.6f}")
	if org_summary_metrics.get("test_f1") is not None:
		print(f"[OK] ORG macro test_f1: {float(org_summary_metrics['test_f1']):.6f}")
	print(f"[OK] {output_feature_label} macro test_accuracy: {macro_filtered_accuracy:.6f}")
	if filtered_summary_metrics.get("test_precision") is not None:
		print(f"[OK] {output_feature_label} macro test_precision: {float(filtered_summary_metrics['test_precision']):.6f}")
	if filtered_summary_metrics.get("test_recall") is not None:
		print(f"[OK] {output_feature_label} macro test_recall: {float(filtered_summary_metrics['test_recall']):.6f}")
	if filtered_summary_metrics.get("test_f1") is not None:
		print(f"[OK] {output_feature_label} macro test_f1: {float(filtered_summary_metrics['test_f1']):.6f}")
	print(f"[OK] Metrik dosyasi: {metrics_dir / 'ORG_test_metrics.json'}")
	if filtered_class_metric_rows:
		print(f"[OK] Sinif bazli multiclass metrik CSV: {filtered_class_metrics_path}")
	return macro_org_accuracy, macro_filtered_accuracy


class ExperimentRunner:
	"""Coordinates one existing experiment without changing its training flow."""

	def __init__(self, config: ExperimentConfig) -> None:
		self.config = config

	def run(self) -> tuple[float, float]:
		config = self.config
		task = config.task.lower().strip()
		if task not in {"classification", "clustering", "regression"}:
			raise ValueError("task 'classification', 'clustering' veya 'regression' olmali.")

		configure_tensorflow_device(config.device)
		set_reproducible(config.random_state)
		if config.random_state is None:
			print("[INFO] random_state: None (rastgele)")
		else:
			print(f"[INFO] random_state: {config.random_state} (sabit)")

		feature_percent = validate_feature_percent(config.feature_percent)
		id_column = normalize_id_column(config.id_column)
		dataset_filename = convert_txt_dataset_to_csv(config.dataset_name)
		dataset_folder = Path(dataset_filename).stem
		print(f"[INFO] Veri yukleniyor: {dataset_filename}")
		df = load_data(dataset_filename, folder="raw", target_column=config.target_column)

		if config.save_encoded_dataset:
			return self._save_encoded_dataset(df, dataset_folder, task, id_column, feature_percent)
		if config.evaluate_dimension_reduction:
			self._validate_dimension_reduction_request(task, dataset_folder)
		if config.validation_dataset_name is not None and config.validation_dataset_name.strip():
			return self._run_explicit_validation(df, dataset_folder, task, id_column, feature_percent)
		return self._run_task(df, dataset_folder, task, id_column, feature_percent)

	def _save_encoded_dataset(
		self,
		df: pd.DataFrame,
		dataset_folder: str,
		task: str,
		id_column: str | None,
		feature_percent: float,
	) -> tuple[float, float]:
		if task != "classification":
			raise ValueError("--save-encoded-dataset su an classification modu icin destekleniyor.")
		config = self.config
		save_dimension_reduced_classification_dataset(
			df=df,
			dataset_folder=dataset_folder,
			target_column=config.target_column,
			id_column=id_column,
			feature_percent=feature_percent,
			random_state=config.random_state,
			autoencoder_early_stopping_patience=config.autoencoder_early_stopping_patience,
			autoencoder_early_stopping_min_delta=config.autoencoder_early_stopping_min_delta,
			save_training_plots=config.save_training_plots,
		)
		return 0.0, 0.0

	def _validate_dimension_reduction_request(self, task: str, dataset_folder: str) -> None:
		if task != "classification":
			raise ValueError("--evaluate-dimension-reduction sadece classification modunda kullanilabilir.")
		if is_encoded_dataset_folder(dataset_folder):
			raise ValueError(
				"Leakage-free evaluation ham dataset ile calismalidir; daha once uretilmis encoded CSV vermeyin."
			)

	def _run_explicit_validation(
		self,
		train_df: pd.DataFrame,
		dataset_folder: str,
		task: str,
		id_column: str | None,
		feature_percent: float,
	) -> tuple[float, float]:
		if task != "classification":
			raise ValueError("--validation-dataset-name simdilik sadece classification modunda destekleniyor.")

		config = self.config
		validation_dataset_filename = convert_txt_dataset_to_csv(config.validation_dataset_name)
		validation_dataset_folder = Path(validation_dataset_filename).stem
		print(f"[INFO] Validation veri yukleniyor: {validation_dataset_filename}")
		validation_df = load_data(validation_dataset_filename, folder="raw", target_column=config.target_column)
		if is_probable_regression_target(train_df[config.target_column]) or is_probable_regression_target(
			validation_df[config.target_column]
		):
			raise ValueError(
				"Train/validation classification modu sürekli sayısal target ile calistirilamaz. "
				"Regression icin ayri akisa gecilmeli."
			)
		if int(train_df[config.target_column].nunique(dropna=True)) > 2 or int(
			validation_df[config.target_column].nunique(dropna=True)
		) > 2:
			raise ValueError("Explicit train/validation modu simdilik sadece binary classification icin kullanilmali.")

		return run_binary_train_validation_experiment(
			train_df=train_df,
			validation_df=validation_df,
			dataset_folder=dataset_folder,
			validation_dataset_folder=validation_dataset_folder,
			target_column=config.target_column,
			id_column=id_column,
			encoding_dim=config.encoding_dim,
			feature_percent=feature_percent,
			random_state=config.random_state,
			classifier_epochs=config.classifier_epochs,
			classifier_hidden_units=config.classifier_hidden_units,
			classifier_dropout_rates=config.classifier_dropout_rates,
			classifier_learning_rate=config.classifier_learning_rate,
			classifier_model=config.classifier_model,
			classifier_early_stopping_patience=config.classifier_early_stopping_patience,
			autoencoder_early_stopping_patience=config.autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor=config.classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta=config.classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta=config.autoencoder_early_stopping_min_delta,
			classifier_class_weight=config.classifier_class_weight,
			classifier_sampling=config.classifier_sampling,
			feature_chunk_size=config.feature_chunk_size,
			chunk_feature_threshold=config.chunk_feature_threshold,
			enable_feature_chunking=config.enable_feature_chunking,
			save_training_plots=config.save_training_plots,
		)

	def _run_task(
		self,
		df: pd.DataFrame,
		dataset_folder: str,
		task: str,
		id_column: str | None,
		feature_percent: float,
	) -> tuple[float, float]:
		config = self.config
		if task == "clustering":
			return run_clustering_experiment(
				df=df,
				dataset_folder=dataset_folder,
				target_column=config.target_column,
				id_column=id_column,
				encoding_dim=config.encoding_dim,
				feature_percent=feature_percent,
				random_state=config.random_state,
				cluster_k=config.cluster_k,
				cluster_min_k=config.cluster_min_k,
				cluster_max_k=config.cluster_max_k,
				feature_chunk_size=config.feature_chunk_size,
				chunk_feature_threshold=config.chunk_feature_threshold,
				enable_feature_chunking=config.enable_feature_chunking,
				save_training_plots=config.save_training_plots,
			)
		if task == "regression":
			return run_regression_experiment(
				df=df,
				dataset_folder=dataset_folder,
				target_column=config.target_column,
				id_column=id_column,
				encoding_dim=config.encoding_dim,
				feature_percent=feature_percent,
				random_state=config.random_state,
				classifier_epochs=config.classifier_epochs,
				classifier_hidden_units=config.classifier_hidden_units,
				classifier_dropout_rates=config.classifier_dropout_rates,
				classifier_learning_rate=config.classifier_learning_rate,
				regression_model=config.regression_model,
				svr_kernel=config.svr_kernel,
				svr_c=config.svr_c,
				svr_epsilon=config.svr_epsilon,
				svr_gamma=config.svr_gamma,
				kmeans_regression_clusters=config.kmeans_regression_clusters,
				kmeans_regression_n_init=config.kmeans_regression_n_init,
				classifier_early_stopping_patience=config.classifier_early_stopping_patience,
				autoencoder_early_stopping_patience=config.autoencoder_early_stopping_patience,
				classifier_early_stopping_min_delta=config.classifier_early_stopping_min_delta,
				autoencoder_early_stopping_min_delta=config.autoencoder_early_stopping_min_delta,
				save_training_plots=config.save_training_plots,
				actual_predicted_top_n=config.actual_predicted_top_n,
			)

		if is_probable_regression_target(df[config.target_column]):
			raise ValueError(
				"Bu veri setinin target kolonu sürekli sayısal görünüyor; classification olarak çalıştırılamaz. "
				"Regression için komutu şu şekilde kullan: --task regression"
			)
		class_count = int(df[config.target_column].nunique(dropna=True))
		if config.evaluate_dimension_reduction:
			return self._run_dimension_reduction(df, dataset_folder, id_column, feature_percent, class_count)
		if class_count > 2:
			return run_multiclass_one_vs_rest(
				df=df,
				dataset_folder=dataset_folder,
				target_column=config.target_column,
				id_column=id_column,
				encoding_dim=config.encoding_dim,
				feature_percent=feature_percent,
				random_state=config.random_state,
				classifier_epochs=config.classifier_epochs,
				classifier_hidden_units=config.classifier_hidden_units,
				classifier_dropout_rates=config.classifier_dropout_rates,
				classifier_learning_rate=config.classifier_learning_rate,
				classifier_model=config.classifier_model,
				classifier_early_stopping_patience=config.classifier_early_stopping_patience,
				autoencoder_early_stopping_patience=config.autoencoder_early_stopping_patience,
				classifier_early_stopping_monitor=config.classifier_early_stopping_monitor,
				classifier_early_stopping_min_delta=config.classifier_early_stopping_min_delta,
				autoencoder_early_stopping_min_delta=config.autoencoder_early_stopping_min_delta,
				classifier_class_weight=config.classifier_class_weight,
				classifier_sampling=config.classifier_sampling,
				feature_chunk_size=config.feature_chunk_size,
				chunk_feature_threshold=config.chunk_feature_threshold,
				enable_feature_chunking=config.enable_feature_chunking,
				save_training_plots=config.save_training_plots,
			)
		return run_binary_experiment(
			df=df,
			dataset_folder=dataset_folder,
			target_column=config.target_column,
			id_column=id_column,
			encoding_dim=config.encoding_dim,
			feature_percent=feature_percent,
			random_state=config.random_state,
			classifier_epochs=config.classifier_epochs,
			classifier_hidden_units=config.classifier_hidden_units,
			classifier_dropout_rates=config.classifier_dropout_rates,
			classifier_learning_rate=config.classifier_learning_rate,
			classifier_model=config.classifier_model,
			classifier_early_stopping_patience=config.classifier_early_stopping_patience,
			autoencoder_early_stopping_patience=config.autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor=config.classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta=config.classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta=config.autoencoder_early_stopping_min_delta,
			classifier_class_weight=config.classifier_class_weight,
			classifier_sampling=config.classifier_sampling,
			feature_chunk_size=config.feature_chunk_size,
			chunk_feature_threshold=config.chunk_feature_threshold,
			enable_feature_chunking=config.enable_feature_chunking,
			save_training_plots=config.save_training_plots,
		)

	def _run_dimension_reduction(
		self,
		df: pd.DataFrame,
		dataset_folder: str,
		id_column: str | None,
		feature_percent: float,
		class_count: int,
	) -> tuple[float, float]:
		config = self.config
		common_kwargs = {
			"df": df,
			"dataset_folder": dataset_folder,
			"target_column": config.target_column,
			"id_column": id_column,
			"encoding_dim": config.encoding_dim,
			"feature_percent": feature_percent,
			"random_state": config.random_state,
			"classifier_epochs": config.classifier_epochs,
			"classifier_hidden_units": config.classifier_hidden_units,
			"classifier_dropout_rates": config.classifier_dropout_rates,
			"classifier_learning_rate": config.classifier_learning_rate,
			"classifier_model": config.classifier_model,
			"classifier_early_stopping_patience": config.classifier_early_stopping_patience,
			"autoencoder_early_stopping_patience": config.autoencoder_early_stopping_patience,
			"classifier_early_stopping_monitor": config.classifier_early_stopping_monitor,
			"classifier_early_stopping_min_delta": config.classifier_early_stopping_min_delta,
			"autoencoder_early_stopping_min_delta": config.autoencoder_early_stopping_min_delta,
			"classifier_class_weight": config.classifier_class_weight,
			"classifier_sampling": config.classifier_sampling,
			"save_training_plots": config.save_training_plots,
		}
		if class_count > 2:
			return run_dimension_reduction_multiclass_one_vs_rest(**common_kwargs)
		return run_dimension_reduction_classification_experiment(**common_kwargs)


def main(
	dataset_name: str = "breast_cancer_data.csv",
	validation_dataset_name: str | None = None,
	target_column: str = "target",
	id_column: str | None = "ID",
	task: str = "classification",
	encoding_dim: int = 8,
	feature_percent: float = 50.0,
	random_state: int | None = RANDOM_STATE,
	classifier_epochs: int = DEFAULT_CLASSIFIER_EPOCHS,
	classifier_hidden_units: tuple[int, ...] = DEFAULT_CLASSIFIER_HIDDEN_UNITS,
	classifier_dropout_rates: tuple[float, ...] | None = None,
	classifier_learning_rate: float = 0.001,
	classifier_model: str = DEFAULT_CLASSIFIER_MODEL,
	regression_model: str = DEFAULT_REGRESSION_MODEL,
	svr_kernel: str = DEFAULT_SVR_KERNEL,
	svr_c: float = DEFAULT_SVR_C,
	svr_epsilon: float = DEFAULT_SVR_EPSILON,
	svr_gamma: str | float = DEFAULT_SVR_GAMMA,
	kmeans_regression_clusters: int = DEFAULT_KMEANS_REGRESSION_CLUSTERS,
	kmeans_regression_n_init: int = DEFAULT_KMEANS_REGRESSION_N_INIT,
	device: str = "auto",
	feature_chunk_size: int = DEFAULT_FEATURE_CHUNK_SIZE,
	chunk_feature_threshold: int = DEFAULT_CHUNK_FEATURE_THRESHOLD,
	enable_feature_chunking: bool = True,
	classifier_early_stopping_patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE,
	autoencoder_early_stopping_patience: int | None = DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE,
	classifier_early_stopping_monitor: str = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR,
	classifier_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	autoencoder_early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	classifier_class_weight: str = "none",
	classifier_sampling: str = "none",
	cluster_k: int | None = None,
	cluster_min_k: int = DEFAULT_CLUSTER_MIN_K,
	cluster_max_k: int = DEFAULT_CLUSTER_MAX_K,
	save_training_plots: bool = False,
	actual_predicted_top_n: int | None = None,
	save_encoded_dataset: bool = False,
	evaluate_dimension_reduction: bool = False,
) -> tuple[float, float]:
	return ExperimentRunner(
		ExperimentConfig(
			dataset_name=dataset_name,
			validation_dataset_name=validation_dataset_name,
			target_column=target_column,
			id_column=id_column,
			task=task,
			encoding_dim=encoding_dim,
			feature_percent=feature_percent,
			random_state=random_state,
			classifier_epochs=classifier_epochs,
			classifier_hidden_units=classifier_hidden_units,
			classifier_dropout_rates=classifier_dropout_rates,
			classifier_learning_rate=classifier_learning_rate,
			classifier_model=classifier_model,
			regression_model=regression_model,
			svr_kernel=svr_kernel,
			svr_c=svr_c,
			svr_epsilon=svr_epsilon,
			svr_gamma=svr_gamma,
			kmeans_regression_clusters=kmeans_regression_clusters,
			kmeans_regression_n_init=kmeans_regression_n_init,
			device=device,
			feature_chunk_size=feature_chunk_size,
			chunk_feature_threshold=chunk_feature_threshold,
			enable_feature_chunking=enable_feature_chunking,
			classifier_early_stopping_patience=classifier_early_stopping_patience,
			autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor=classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
			classifier_class_weight=classifier_class_weight,
			classifier_sampling=classifier_sampling,
			cluster_k=cluster_k,
			cluster_min_k=cluster_min_k,
			cluster_max_k=cluster_max_k,
			save_training_plots=save_training_plots,
			actual_predicted_top_n=actual_predicted_top_n,
			save_encoded_dataset=save_encoded_dataset,
			evaluate_dimension_reduction=evaluate_dimension_reduction,
		)
	).run()


def run_repeated_experiments(
	dataset_name: str,
	validation_dataset_name: str | None,
	target_column: str,
	id_column: str,
	task: str,
	encoding_dim: int,
	feature_percent: float,
	random_state: int | None,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
	classifier_model: str,
	regression_model: str,
	svr_kernel: str,
	svr_c: float,
	svr_epsilon: float,
	svr_gamma: str | float,
	kmeans_regression_clusters: int,
	kmeans_regression_n_init: int,
	device: str,
	feature_chunk_size: int,
	chunk_feature_threshold: int,
	enable_feature_chunking: bool,
	classifier_early_stopping_patience: int | None,
	autoencoder_early_stopping_patience: int | None,
	classifier_early_stopping_monitor: str,
	classifier_early_stopping_min_delta: float,
	autoencoder_early_stopping_min_delta: float,
	classifier_class_weight: str,
	classifier_sampling: str,
	cluster_k: int | None,
	cluster_min_k: int,
	cluster_max_k: int,
	save_training_plots: bool,
	actual_predicted_top_n: int | None,
	evaluate_dimension_reduction: bool,
	repeat_runs: int,
	accuracy_txt_path: Path,
	metric_name: str = "Accuracy",
) -> tuple[list[float], float]:
	"""
	Ayni deneyi repeat_runs kadar calistirir ve metric degerlerini kaydeder.
	Sonuc: (metric_values, average_metric)
	"""
	accuracy_values: list[float] = []
	run_durations: list[float] = []
	classification_run_rows: list[dict] = []
	regression_run_rows: list[dict] = []
	clustering_run_rows: list[dict] = []
	history_frames: list[pd.DataFrame] = []
	dataset_folder = Path(convert_txt_dataset_to_csv(dataset_name)).stem
	result_dataset_folder = dataset_folder
	result_feature_percent = feature_percent
	feature_percent_tag = format_feature_percent_tag(result_feature_percent)
	metric_output_prefix = format_metric_output_prefix(result_feature_percent, result_dataset_folder)
	if evaluate_dimension_reduction:
		metric_output_prefix = f"top_{feature_percent_tag}_dimension_reduction"
	lower_is_better = metric_name in {"Cluster_RMSE"}
	base_config = ExperimentConfig(
		dataset_name=dataset_name,
		validation_dataset_name=validation_dataset_name,
		target_column=target_column,
		id_column=id_column,
		task=task,
		encoding_dim=encoding_dim,
		feature_percent=feature_percent,
		random_state=random_state,
		classifier_epochs=classifier_epochs,
		classifier_hidden_units=classifier_hidden_units,
		classifier_dropout_rates=classifier_dropout_rates,
		classifier_learning_rate=classifier_learning_rate,
		classifier_model=classifier_model,
		regression_model=regression_model,
		svr_kernel=svr_kernel,
		svr_c=svr_c,
		svr_epsilon=svr_epsilon,
		svr_gamma=svr_gamma,
		kmeans_regression_clusters=kmeans_regression_clusters,
		kmeans_regression_n_init=kmeans_regression_n_init,
		device=device,
		feature_chunk_size=feature_chunk_size,
		chunk_feature_threshold=chunk_feature_threshold,
		enable_feature_chunking=enable_feature_chunking,
		classifier_early_stopping_patience=classifier_early_stopping_patience,
		autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
		classifier_early_stopping_monitor=classifier_early_stopping_monitor,
		classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
		autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
		classifier_class_weight=classifier_class_weight,
		classifier_sampling=classifier_sampling,
		cluster_k=cluster_k,
		cluster_min_k=cluster_min_k,
		cluster_max_k=cluster_max_k,
		save_training_plots=save_training_plots,
		actual_predicted_top_n=actual_predicted_top_n,
		evaluate_dimension_reduction=evaluate_dimension_reduction,
	)

	for run_idx in range(1, repeat_runs + 1):
		run_random_state = random_state
		if random_state is not None and repeat_runs > 1:
			run_random_state = int(random_state) + run_idx - 1
		print(f"\n[INFO] Calisma {run_idx}/{repeat_runs} basladi.")
		if repeat_runs > 1:
			print(f"[INFO] Bu calismanin split seed degeri: {run_random_state}")
		run_start_time = time.perf_counter()
		_, filtered_test_accuracy = ExperimentRunner(
			replace(base_config, random_state=run_random_state)
		).run()
		run_elapsed_seconds = time.perf_counter() - run_start_time
		run_durations.append(run_elapsed_seconds)
		print(f"[OK] Calisma {run_idx}/{repeat_runs} suresi: {run_elapsed_seconds:.2f} saniye")
		accuracy_values.append(float(filtered_test_accuracy))
		if task == "classification":
			classification_metrics_filename = format_test_metrics_filename(
				result_feature_percent,
				result_dataset_folder,
			)
			if evaluate_dimension_reduction:
				classification_metrics_filename = f"{metric_output_prefix}_test_metrics.json"
			classification_metrics_path = (
				classification_output_dir(result_dataset_folder)
				/ "metrics"
				/ classification_metrics_filename
			)
			if classification_metrics_path.exists():
				with open(classification_metrics_path, "r", encoding="utf-8") as f:
					classification_metrics = json.load(f)
				classification_run_rows.append(
					{
						"run": run_idx,
						"feature_percent": classification_metrics.get(
							"feature_percent",
							format_encoded_output_percent(feature_percent, result_dataset_folder),
						),
						"selected_feature_count": classification_metrics.get("selected_feature_count"),
						"test_accuracy": classification_metrics.get("test_accuracy", filtered_test_accuracy),
						"test_precision": classification_metrics.get("test_precision"),
						"test_recall": classification_metrics.get("test_recall"),
						"test_f1": classification_metrics.get("test_f1"),
						"average_precision": classification_metrics.get("average_precision"),
						"roc_auc": classification_metrics.get("roc_auc"),
						"classifier_model": classification_metrics.get("classifier_model", classifier_model),
						"classifier_class_weight": classification_metrics.get("classifier_class_weight", classifier_class_weight),
						"classifier_sampling": classification_metrics.get("classifier_sampling", classifier_sampling),
						"split_seed": classification_metrics.get("split_seed", run_random_state),
						"method": classification_metrics.get(
							"method",
							"DimensionReduction" if evaluate_dimension_reduction else "FeatureRank",
						),
						"encoded_dataset": classification_metrics.get("encoded_dataset", False),
						"encoded_source_feature_percent": classification_metrics.get("encoded_source_feature_percent"),
						"encoded_active_feature_percent": classification_metrics.get("encoded_active_feature_percent"),
						"elapsed_seconds": run_elapsed_seconds,
					}
				)
		if task == "regression":
			regression_metrics_path = (
				regression_output_dir(dataset_folder)
				/ "metrics"
				/ f"top_{feature_percent_tag}_test_metrics.json"
			)
			if regression_metrics_path.exists():
				with open(regression_metrics_path, "r", encoding="utf-8") as f:
					regression_metrics = json.load(f)
				regression_run_rows.append(
					{
						"run": run_idx,
						"feature_percent": feature_percent,
						"selected_feature_count": regression_metrics.get("selected_feature_count"),
						"regression_mse": regression_metrics.get("regression_mse"),
						"regression_rmse": regression_metrics.get("regression_rmse"),
						"regression_mae": regression_metrics.get("regression_mae"),
						"regression_r2": regression_metrics.get("regression_r2"),
						"pearson_r": regression_metrics.get("pearson_r"),
						"correlation": regression_metrics.get("correlation", regression_metrics.get("pearson_r")),
						"cosine_similarity": regression_metrics.get("cosine_similarity"),
						"regression_model": regression_metrics.get("regression_model", regression_model),
						"kmeans_regression_clusters": regression_metrics.get("kmeans_regression_clusters"),
						"kmeans_regression_effective_clusters": regression_metrics.get("kmeans_regression_effective_clusters"),
						"kmeans_regression_n_init": regression_metrics.get("kmeans_regression_n_init"),
						"elapsed_seconds": run_elapsed_seconds,
					}
				)
		if task == "clustering":
			clustering_metrics_path = (
				clustering_output_dir(dataset_folder)
				/ "metrics"
				/ f"top_{feature_percent_tag}_cluster_metrics.json"
			)
			if clustering_metrics_path.exists():
				with open(clustering_metrics_path, "r", encoding="utf-8") as f:
					clustering_metrics = json.load(f)
				clustering_run_rows.append(
					{
						"run": run_idx,
						"method": "FeatureRank",
						"feature_percent": feature_percent,
						"selected_feature_count": clustering_metrics.get("selected_feature_count"),
						"best_k": clustering_metrics.get("best_k"),
						"fixed_cluster_k": clustering_metrics.get("fixed_cluster_k"),
						"silhouette_score": clustering_metrics.get("silhouette_score"),
						"cluster_rmse": clustering_metrics.get("cluster_rmse"),
						"inertia": clustering_metrics.get("inertia"),
						"elapsed_seconds": run_elapsed_seconds,
					}
				)
		if save_training_plots and task in {"classification", "regression"}:
			history_df = collect_repeated_run_history(
				dataset_folder=dataset_folder,
				feature_percent=feature_percent,
				run_idx=run_idx,
				task=task,
			)
			if history_df is None and task == "classification":
				history_df = collect_repeated_multiclass_run_history(
					dataset_folder=dataset_folder,
					feature_percent=feature_percent,
					run_idx=run_idx,
				)
			if history_df is not None:
				history_frames.append(history_df)
		sorted_accuracy_values = sorted(accuracy_values, reverse=not lower_is_better)
		classification_metric_text = ""
		regression_error_text = ""
		clustering_silhouette_text = ""
		if task == "classification" and classification_run_rows:
			precision_values = [
				float(row["test_precision"])
				for row in classification_run_rows
				if row.get("test_precision") is not None and not pd.isna(row.get("test_precision"))
			]
			recall_values = [
				float(row["test_recall"])
				for row in classification_run_rows
				if row.get("test_recall") is not None and not pd.isna(row.get("test_recall"))
			]
			f1_values = [
				float(row["test_f1"])
				for row in classification_run_rows
				if row.get("test_f1") is not None and not pd.isna(row.get("test_f1"))
			]
			classification_metric_text = (
				f"\nPrecision listesi: {precision_values}\n"
				f"Recall listesi: {recall_values}\n"
				f"F1 listesi: {f1_values}"
			)
		if task == "regression" and regression_run_rows:
			mse_values = [
				float(row["regression_mse"])
				for row in regression_run_rows
				if row.get("regression_mse") is not None
			]
			rmse_values = [
				float(row["regression_rmse"])
				for row in regression_run_rows
				if row.get("regression_rmse") is not None
			]
			regression_error_text = (
				f"\nMSE listesi: {mse_values}\n"
				f"Sirali MSE (dusuk iyi): {sorted(mse_values)}\n"
				f"RMSE listesi: {rmse_values}\n"
				f"Sirali RMSE (dusuk iyi): {sorted(rmse_values)}"
			)
		if task == "clustering" and clustering_run_rows:
			silhouette_values = [
				float(row["silhouette_score"])
				for row in clustering_run_rows
				if row.get("silhouette_score") is not None and not pd.isna(row.get("silhouette_score"))
			]
			clustering_silhouette_text = (
				f"\nSilhouette listesi: {silhouette_values}\n"
				f"Sirali Silhouette (yuksek iyi): {sorted(silhouette_values, reverse=True)}"
			)
		accuracy_txt_path.write_text(
			f"{accuracy_values}\nSirali {metric_name}: {sorted_accuracy_values}"
			f"{classification_metric_text}"
			f"{regression_error_text}\n"
			f"{clustering_silhouette_text}\n"
			f"Run sureleri saniye: {[round(value, 3) for value in run_durations]}",
			encoding="utf-8",
		)

	average_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0
	std_accuracy = float(np.std(accuracy_values, ddof=1)) if len(accuracy_values) > 1 else 0.0
	average_duration = sum(run_durations) / len(run_durations) if run_durations else 0.0
	std_duration = float(np.std(run_durations, ddof=1)) if len(run_durations) > 1 else 0.0
	sorted_accuracy_values = sorted(accuracy_values, reverse=not lower_is_better)
	classification_metric_text = ""
	regression_error_text = ""
	clustering_silhouette_text = ""
	if task == "classification" and classification_run_rows:
		precision_values = [
			float(row["test_precision"])
			for row in classification_run_rows
			if row.get("test_precision") is not None and not pd.isna(row.get("test_precision"))
		]
		recall_values = [
			float(row["test_recall"])
			for row in classification_run_rows
			if row.get("test_recall") is not None and not pd.isna(row.get("test_recall"))
		]
		f1_values = [
			float(row["test_f1"])
			for row in classification_run_rows
			if row.get("test_f1") is not None and not pd.isna(row.get("test_f1"))
		]
		precision_mean = float(np.mean(precision_values)) if precision_values else float("nan")
		precision_std = float(np.std(precision_values, ddof=1)) if len(precision_values) > 1 else 0.0
		recall_mean = float(np.mean(recall_values)) if recall_values else float("nan")
		recall_std = float(np.std(recall_values, ddof=1)) if len(recall_values) > 1 else 0.0
		f1_mean = float(np.mean(f1_values)) if f1_values else float("nan")
		f1_std = float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else 0.0
		classification_metric_text = (
			f"Precision listesi: {precision_values}\n"
			f"Sirali Precision (yuksek iyi): {sorted(precision_values, reverse=True)}\n"
			f"Ortalama Precision: {precision_mean:.6f}\n"
			f"Std Precision: {precision_std:.6f}\n"
			f"Recall listesi: {recall_values}\n"
			f"Sirali Recall (yuksek iyi): {sorted(recall_values, reverse=True)}\n"
			f"Ortalama Recall: {recall_mean:.6f}\n"
			f"Std Recall: {recall_std:.6f}\n"
			f"F1 listesi: {f1_values}\n"
			f"Sirali F1 (yuksek iyi): {sorted(f1_values, reverse=True)}\n"
			f"Ortalama F1: {f1_mean:.6f}\n"
			f"Std F1: {f1_std:.6f}\n"
		)
	if task == "regression" and regression_run_rows:
		mse_values = [
			float(row["regression_mse"])
			for row in regression_run_rows
			if row.get("regression_mse") is not None
		]
		rmse_values = [
			float(row["regression_rmse"])
			for row in regression_run_rows
			if row.get("regression_rmse") is not None
		]
		correlation_values = [
			float(row["correlation"])
			for row in regression_run_rows
			if row.get("correlation") is not None and not pd.isna(row.get("correlation"))
		]
		mse_mean = float(np.mean(mse_values)) if mse_values else float("nan")
		mse_std = float(np.std(mse_values, ddof=1)) if len(mse_values) > 1 else 0.0
		rmse_mean = float(np.mean(rmse_values)) if rmse_values else float("nan")
		rmse_std = float(np.std(rmse_values, ddof=1)) if len(rmse_values) > 1 else 0.0
		correlation_mean = float(np.mean(correlation_values)) if correlation_values else float("nan")
		correlation_std = float(np.std(correlation_values, ddof=1)) if len(correlation_values) > 1 else 0.0
		regression_error_text = (
			f"MSE listesi: {mse_values}\n"
			f"Sirali MSE (dusuk iyi): {sorted(mse_values)}\n"
			f"Ortalama MSE: {mse_mean:.6f}\n"
			f"Std MSE: {mse_std:.6f}\n"
			f"RMSE listesi: {rmse_values}\n"
			f"Sirali RMSE (dusuk iyi): {sorted(rmse_values)}\n"
			f"Ortalama RMSE: {rmse_mean:.6f}\n"
			f"Std RMSE: {rmse_std:.6f}\n"
			f"Correlation listesi: {correlation_values}\n"
			f"Sirali Correlation (yuksek iyi): {sorted(correlation_values, reverse=True)}\n"
			f"Ortalama Correlation: {correlation_mean:.6f}\n"
			f"Std Correlation: {correlation_std:.6f}\n"
		)
	if task == "clustering" and clustering_run_rows:
		silhouette_values = [
			float(row["silhouette_score"])
			for row in clustering_run_rows
			if row.get("silhouette_score") is not None and not pd.isna(row.get("silhouette_score"))
		]
		silhouette_mean = float(np.mean(silhouette_values)) if silhouette_values else float("nan")
		silhouette_std = float(np.std(silhouette_values, ddof=1)) if len(silhouette_values) > 1 else 0.0
		silhouette_ci_half_width = (
			float(1.96 * silhouette_std / np.sqrt(len(silhouette_values))) if len(silhouette_values) > 1 else 0.0
		)
		clustering_silhouette_text = (
			f"Silhouette listesi: {silhouette_values}\n"
			f"Sirali Silhouette (yuksek iyi): {sorted(silhouette_values, reverse=True)}\n"
			f"Ortalama Silhouette: {silhouette_mean:.6f}\n"
			f"Std Silhouette: {silhouette_std:.6f}\n"
			f"95% CI Silhouette: [{silhouette_mean - silhouette_ci_half_width:.6f}, "
			f"{silhouette_mean + silhouette_ci_half_width:.6f}]\n"
		)
	output_text = (
		f"{accuracy_values}\n"
		f"Sirali {metric_name}: {sorted_accuracy_values}\n"
		f"Ortalama {metric_name}: {average_accuracy:.6f}\n"
		f"Std {metric_name}: {std_accuracy:.6f}\n"
		f"{classification_metric_text}"
		f"{regression_error_text}"
		f"{clustering_silhouette_text}"
		f"Run sureleri saniye: {[round(value, 3) for value in run_durations]}\n"
		f"Ortalama sure saniye: {average_duration:.3f}\n"
		f"Std sure saniye: {std_duration:.3f}"
	)
	accuracy_txt_path.write_text(output_text, encoding="utf-8")

	if task == "clustering":
		plot_output_dir = clustering_output_dir(result_dataset_folder) / "metrics"
		boxplot_metric_name = "Cluster_RMSE"
	elif task == "regression":
		plot_output_dir = regression_output_dir(result_dataset_folder) / "metrics"
	else:
		plot_output_dir = classification_output_dir(result_dataset_folder) / "metrics"
		boxplot_metric_name = "Accuracy"
	if task != "regression":
		save_metric_boxplot(
			metric_values=accuracy_values,
			output_dir=plot_output_dir,
			file_prefix=metric_output_prefix,
			metric_name=boxplot_metric_name,
			normalize_to_unit=boxplot_metric_name == "Cluster_RMSE",
		)
		save_repeated_metric_distribution_plot(
			metric_values=accuracy_values,
			output_dir=plot_output_dir,
			file_prefix=metric_output_prefix,
			metric_name=boxplot_metric_name,
		)

	if task == "classification" and classification_run_rows:
		classification_runs_df = pd.DataFrame(classification_run_rows)
		classification_runs_path = plot_output_dir / f"{metric_output_prefix}_classification_runs.csv"
		classification_runs_df.to_csv(classification_runs_path, index=False)

		def summarize_metric(column_name: str) -> tuple[float, float]:
			values = classification_runs_df[column_name].dropna().astype(float).to_numpy()
			mean_value = float(np.mean(values)) if len(values) else float("nan")
			std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
			return mean_value, std_value

		accuracy_mean, accuracy_std = summarize_metric("test_accuracy")
		precision_mean, precision_std = summarize_metric("test_precision")
		recall_mean, recall_std = summarize_metric("test_recall")
		f1_mean, f1_std = summarize_metric("test_f1")
		selected_feature_counts = classification_runs_df["selected_feature_count"].dropna().astype(int)
		classification_table_summary = {
			"Dataset": dataset_folder,
			"Method": "DimensionReduction" if evaluate_dimension_reduction else "FeatureRank",
			"Feature_Percent": feature_percent,
			"NOF": int(selected_feature_counts.iloc[0]) if not selected_feature_counts.empty else None,
			"Average_Accuracy": accuracy_mean,
			"Accuracy_STD": accuracy_std,
			"Average_Precision": precision_mean,
			"Precision_STD": precision_std,
			"Average_Recall": recall_mean,
			"Recall_STD": recall_std,
			"Average_F1": f1_mean,
			"F1_STD": f1_std,
			"Accuracy_pm_STD": f"{accuracy_mean:.6f} ± {accuracy_std:.6f}",
			"Precision_pm_STD": f"{precision_mean:.6f} ± {precision_std:.6f}",
			"Recall_pm_STD": f"{recall_mean:.6f} ± {recall_std:.6f}",
			"F1_pm_STD": f"{f1_mean:.6f} ± {f1_std:.6f}",
			"repeat_runs": int(len(classification_runs_df)),
			"runs_csv_path": str(classification_runs_path),
		}
		classification_summary_csv_path = plot_output_dir / f"{metric_output_prefix}_classification_table_summary.csv"
		classification_summary_json_path = plot_output_dir / f"{metric_output_prefix}_classification_table_summary.json"
		pd.DataFrame([classification_table_summary]).to_csv(classification_summary_csv_path, index=False)
		save_json(classification_table_summary, classification_summary_json_path)
		print(f"[OK] Classification run CSV: {classification_runs_path}")
		print(f"[OK] Classification tablo ozeti CSV: {classification_summary_csv_path}")

	if task == "clustering" and clustering_run_rows:
		clustering_runs_df = pd.DataFrame(clustering_run_rows)
		clustering_runs_path = plot_output_dir / f"top_{feature_percent_tag}_clustering_runs.csv"
		clustering_runs_df.to_csv(clustering_runs_path, index=False)

		silhouette_values = clustering_runs_df["silhouette_score"].dropna().astype(float).to_numpy()
		selected_feature_counts = clustering_runs_df["selected_feature_count"].dropna().astype(int)
		best_k_values = clustering_runs_df["best_k"].dropna().astype(int)
		n_silhouette_runs = int(len(silhouette_values))
		silhouette_mean = float(np.mean(silhouette_values)) if n_silhouette_runs else float("nan")
		silhouette_std = float(np.std(silhouette_values, ddof=1)) if n_silhouette_runs > 1 else 0.0
		silhouette_ci_half_width = (
			float(1.96 * silhouette_std / np.sqrt(n_silhouette_runs)) if n_silhouette_runs > 1 else 0.0
		)
		if n_silhouette_runs > 0:
			save_metric_boxplot(
				metric_values=silhouette_values.tolist(),
				output_dir=plot_output_dir,
				file_prefix=f"top_{feature_percent_tag}",
				metric_name="Silhouette",
			)
			save_repeated_metric_distribution_plot(
				metric_values=silhouette_values.tolist(),
				output_dir=plot_output_dir,
				file_prefix=f"top_{feature_percent_tag}",
				metric_name="Silhouette",
			)
		clustering_table_summary = {
			"Dataset": dataset_folder,
			"Method": "FeatureRank",
			"Feature_Percent": feature_percent,
			"NOF": int(selected_feature_counts.iloc[0]) if not selected_feature_counts.empty else None,
			"Best_K_Mode": int(best_k_values.mode().iloc[0]) if not best_k_values.empty else None,
			"Average_Silhouette": silhouette_mean,
			"Silhouette_STD": silhouette_std,
			"Silhouette_CI_1": silhouette_mean - silhouette_ci_half_width,
			"Silhouette_CI_2": silhouette_mean + silhouette_ci_half_width,
			"Average_Silhouette_pm_STD": f"{silhouette_mean:.6f} ± {silhouette_std:.6f}",
			"repeat_runs": n_silhouette_runs,
			"runs_csv_path": str(clustering_runs_path),
		}
		clustering_summary_csv_path = plot_output_dir / f"top_{feature_percent_tag}_clustering_table_summary.csv"
		clustering_summary_json_path = plot_output_dir / f"top_{feature_percent_tag}_clustering_table_summary.json"
		pd.DataFrame([clustering_table_summary]).to_csv(clustering_summary_csv_path, index=False)
		save_json(clustering_table_summary, clustering_summary_json_path)
		print(f"[OK] Clustering run CSV: {clustering_runs_path}")
		print(f"[OK] Clustering silhouette tablo ozeti CSV: {clustering_summary_csv_path}")

	if task == "regression" and regression_run_rows:
		regression_runs_df = pd.DataFrame(regression_run_rows)
		regression_runs_path = plot_output_dir / f"top_{feature_percent_tag}_regression_runs.csv"
		regression_runs_df.to_csv(regression_runs_path, index=False)

		rmse_values = regression_runs_df["regression_rmse"].dropna().astype(float).to_numpy()
		if len(rmse_values) > 0:
			save_metric_boxplot(
				metric_values=rmse_values.tolist(),
				output_dir=plot_output_dir,
				file_prefix=f"top_{feature_percent_tag}",
				metric_name="RMSE",
				normalize_to_unit=True,
			)
			save_repeated_metric_distribution_plot(
				metric_values=rmse_values.tolist(),
				output_dir=plot_output_dir,
				file_prefix=f"top_{feature_percent_tag}",
				metric_name="RMSE",
			)
		selected_feature_counts = regression_runs_df["selected_feature_count"].dropna().astype(int)
		n_runs = int(len(rmse_values))
		rmse_mean = float(np.mean(rmse_values)) if n_runs else float("nan")
		rmse_std = float(np.std(rmse_values, ddof=1)) if n_runs > 1 else 0.0
		rmse_ci_half_width = float(1.96 * rmse_std / np.sqrt(n_runs)) if n_runs > 1 else 0.0
		correlation_values = regression_runs_df["correlation"].dropna().astype(float).to_numpy()
		n_correlation_runs = int(len(correlation_values))
		correlation_mean = float(np.mean(correlation_values)) if n_correlation_runs else float("nan")
		correlation_std = float(np.std(correlation_values, ddof=1)) if n_correlation_runs > 1 else 0.0
		correlation_ci_half_width = (
			float(1.96 * correlation_std / np.sqrt(n_correlation_runs)) if n_correlation_runs > 1 else 0.0
		)
		table_summary = {
			"DS": dataset_folder.replace("_data", "").upper(),
			"AL": "FeatureRank",
			"NOF": int(selected_feature_counts.iloc[0]) if not selected_feature_counts.empty else None,
			"ET": average_duration,
			"ER": rmse_mean,
			"ER_STD": rmse_std,
			"ER_CI_1": rmse_mean - rmse_ci_half_width,
			"ER_CI_2": rmse_mean + rmse_ci_half_width,
			"CORR": correlation_mean,
			"CORR_STD": correlation_std,
			"CORR_CI_1": correlation_mean - correlation_ci_half_width,
			"CORR_CI_2": correlation_mean + correlation_ci_half_width,
			"REGRESSION_MODEL": regression_model,
			"KMEANS_REGRESSION_CLUSTERS": kmeans_regression_clusters if regression_model == "kmeans" else None,
			"KMEANS_REGRESSION_N_INIT": kmeans_regression_n_init if regression_model == "kmeans" else None,
			"repeat_runs": n_runs,
			"feature_percent": feature_percent,
			"runs_csv_path": str(regression_runs_path),
		}
		summary_csv_path = plot_output_dir / f"top_{feature_percent_tag}_regression_table_summary.csv"
		summary_json_path = plot_output_dir / f"top_{feature_percent_tag}_regression_table_summary.json"
		pd.DataFrame([table_summary]).to_csv(summary_csv_path, index=False)
		save_json(table_summary, summary_json_path)
		print(f"[OK] Regression run CSV: {regression_runs_path}")
		print(f"[OK] Regression tablo ozeti CSV: {summary_csv_path}")

	if save_training_plots and task in {"classification", "regression"} and history_frames:
		history_dir = task_output_dir(task, dataset_folder) / "training_history"
		save_average_convergence(
			history_frames=history_frames,
			output_dir=history_dir,
			file_prefix=f"top_{feature_percent_tag}",
		)
		final_loss_values = extract_final_history_metric_values(history_frames, "loss")
		save_metric_boxplot(
			metric_values=final_loss_values,
			output_dir=plot_output_dir,
			file_prefix=f"top_{feature_percent_tag}",
			metric_name="Loss",
		)

	return accuracy_values, average_accuracy


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Basit autoencoder egitimi")
	parser.add_argument("--dataset-name", type=str, default="breast_cancer_data.csv", help="Raw data dosyasi (.csv veya .txt)")
	parser.add_argument("--validation-dataset-name", type=str, default="", help="Ayrı validation raw data dosyasi. Verilirse dataset-name train, bu dosya validation olarak kullanilir")
	parser.add_argument("--target-column", type=str, default="target", help="Hedef kolon adi")
	parser.add_argument("--id-column", type=str, default="ID", help="ID kolon adi, kullanmak istemezsen 'none' ver")
	parser.add_argument("--task", type=str, default="classification", choices=["classification", "clustering", "regression"], help="Calisma modu")
	parser.add_argument("--encoding-dim", type=int, default=8, help="Normal FeatureRank autoencoder encoding boyutu. --save-encoded-dataset modunda latent boyut feature-percent ile hesaplanir")
	parser.add_argument(
		"--feature-percent",
		type=str,
		default="20.0",
		help="Secilecek feature yuzdesi. Ornek: 20, 10,20,30 veya all (10-100 arasi 10'arli)",
	)
	parser.add_argument("--random-state", type=str, default=str(RANDOM_STATE), help="Random state. Ornek: 42 veya none")
	parser.add_argument("--repeat-runs", type=int, default=1, help="Ayni deneyi kac kez calistiracagi")
	parser.add_argument("--accuracy-list-txt", type=str, default="", help="Accuracy listesi txt cikti yolu (bos ise varsayilan yol kullanilir)")
	parser.add_argument("--classifier-epochs", type=int, default=DEFAULT_CLASSIFIER_EPOCHS, help="Classifier epoch sayisi")
	parser.add_argument("--classifier-hidden-units", type=str, default="32,16", help="Classifier gizli katman nöronlari. Ornek: 128,64")
	parser.add_argument("--classifier-dropout-rates", type=str, default="", help="Classifier dropout oranlari. Ornek: 0.3,0.2")
	parser.add_argument("--classifier-learning-rate", type=float, default=0.001, help="Classifier ogrenme orani")
	parser.add_argument("--classifier-model", type=str, default=DEFAULT_CLASSIFIER_MODEL, choices=["neural", "logistic", "svm", "random_forest"], help="Classification modeli: neural, logistic, svm veya random_forest")
	parser.add_argument("--regression-model", type=str, default=DEFAULT_REGRESSION_MODEL, choices=["neural", "svr", "kmeans"], help="Regression tahmin modeli: neural, svr veya kmeans")
	parser.add_argument("--svr-kernel", type=str, default=DEFAULT_SVR_KERNEL, choices=["linear", "poly", "rbf", "sigmoid"], help="SVR kernel tipi")
	parser.add_argument("--svr-c", type=float, default=DEFAULT_SVR_C, help="SVR C regularization parametresi")
	parser.add_argument("--svr-epsilon", type=float, default=DEFAULT_SVR_EPSILON, help="SVR epsilon parametresi")
	parser.add_argument("--svr-gamma", type=str, default=DEFAULT_SVR_GAMMA, help="SVR gamma parametresi: scale, auto veya sayisal deger")
	parser.add_argument("--kmeans-regression-clusters", type=int, default=DEFAULT_KMEANS_REGRESSION_CLUSTERS, help="KMeans regression icin cluster sayisi")
	parser.add_argument("--kmeans-regression-n-init", type=int, default=DEFAULT_KMEANS_REGRESSION_N_INIT, help="KMeans regression icin n_init degeri")
	parser.add_argument("--classifier-early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE, help="Classifier icin early stopping patience. 0 verirsen kapanir")
	parser.add_argument("--autoencoder-early-stopping-patience", type=int, default=None, help="Autoencoder icin val_loss tabanli early stopping patience. 0 verirsen kapanir. Regression modunda verilmezse otomatik classifier patience kullanilir")
	parser.add_argument("--classifier-early-stopping-monitor", type=str, default=DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR, choices=["val_loss", "val_accuracy"], help="Classifier early stopping metriği")
	parser.add_argument("--classifier-early-stopping-min-delta", type=float, default=DEFAULT_EARLY_STOPPING_MIN_DELTA, help="Classifier icin minimum validation metric iyilesmesi. Daha kucuk iyilesmeler plato sayilir")
	parser.add_argument("--autoencoder-early-stopping-min-delta", type=float, default=DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA, help="Autoencoder icin minimum val_loss iyilesmesi. Daha kucuk iyilesmeler plato sayilir")
	parser.add_argument("--classifier-class-weight", type=str, default="none", choices=["none", "balanced"], help="Binary classification icin class_weight kullanimi. Dengesiz veri icin balanced verilebilir")
	parser.add_argument("--classifier-sampling", type=str, default="none", choices=["none", "undersample"], help="Binary classification icin train verisinde sampling. undersample: cogunluk sinifini azinlik sinifi sayisina indirir")
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "gpu", "cpu"], help="Cihaz secimi: auto, gpu veya cpu")
	parser.add_argument("--feature-chunk-size", type=int, default=DEFAULT_FEATURE_CHUNK_SIZE, help="Buyuk feature setlerinde her parcadaki feature sayisi")
	parser.add_argument("--chunk-feature-threshold", type=int, default=DEFAULT_CHUNK_FEATURE_THRESHOLD, help="Feature sayisi bu esigi asarsa parcali akis kullanilir")
	parser.add_argument("--disable-feature-chunking", action="store_true", help="Buyuk feature setlerinde otomatik parcali akisi kapatir")
	parser.add_argument("--cluster-k", type=int, default=None, help="Clustering icin sabit k. Verilirse cluster-min-k ve cluster-max-k yerine bu k kullanilir")
	parser.add_argument("--cluster-min-k", type=int, default=DEFAULT_CLUSTER_MIN_K, help="Clustering icin denenecek minimum k")
	parser.add_argument("--cluster-max-k", type=int, default=DEFAULT_CLUSTER_MAX_K, help="Clustering icin denenecek maksimum k")
	parser.add_argument("--save-training-plots", action="store_true", help="Classification egitim history CSV ve PNG grafiklerini kaydeder")
	parser.add_argument(
		"--save-encoded-dataset",
		action="store_true",
		help=(
			"Dimension reduction CSV'lerini disari aktarir. Bu CSV yeniden split edilerek makale performansi "
			"olculmemeli; performans icin --evaluate-dimension-reduction kullanilmalidir"
		),
	)
	parser.add_argument(
		"--evaluate-dimension-reduction",
		action="store_true",
		help=(
			"Ham dataset uzerinde leakage-free dimension reduction classification yapar: "
			"scaler/autoencoder sadece outer train ile fit edilir, test ayri encode edilip degerlendirilir"
		),
	)
	parser.add_argument("--actual-predicted-top-n", type=int, default=None, help="Regression actual-vs-predicted grafiginde en yuksek actual degerli N ornegi gosterir. Bos birakilirsa tum test ornekleri siralanir")


	args = parser.parse_args()
	if args.save_encoded_dataset and args.evaluate_dimension_reduction:
		raise ValueError(
			"--save-encoded-dataset ve --evaluate-dimension-reduction ayni komutta kullanilamaz. "
			"Ilki CSV uretir, ikincisi leakage-free performans olcer."
		)
	feature_percent_values = parse_feature_percent_values(args.feature_percent)
	if args.save_encoded_dataset and args.feature_percent == parser.get_default("feature_percent"):
		feature_percent_values = [float(value) for value in range(10, 101, 10)]
		print("[INFO] --save-encoded-dataset icin feature-percent verilmedi; %10-%100 arasi otomatik calisacak.")
	random_state = parse_random_state(args.random_state)
	svr_gamma = parse_svr_gamma(args.svr_gamma)
	classifier_hidden_units = parse_hidden_units(args.classifier_hidden_units)
	classifier_dropout_rates = parse_dropout_rates(args.classifier_dropout_rates, len(classifier_hidden_units))
	if args.autoencoder_early_stopping_patience is None:
		if args.task == "regression":
			autoencoder_early_stopping_patience = args.classifier_early_stopping_patience
		else:
			autoencoder_early_stopping_patience = DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE
	else:
		autoencoder_early_stopping_patience = args.autoencoder_early_stopping_patience

	cluster_min_k = args.cluster_min_k
	cluster_max_k = args.cluster_max_k
	cluster_k = args.cluster_k
	if args.cluster_k is not None:
		if args.cluster_k < 2:
			raise ValueError("--cluster-k en az 2 olmali.")
		print(
			f"[INFO] Sabit cluster k kullaniliyor: k={args.cluster_k}. "
			f"Grafikler k={cluster_min_k}-{cluster_max_k} araliginda olusturulacak."
		)

	if args.task == "clustering":
		metric_name = "Cluster_RMSE"
	elif args.task == "regression":
		metric_name = "Pearson_r"
	else:
		metric_name = "Accuracy"

	if len(feature_percent_values) > 1:
		print(f"[INFO] Coklu feature-percent calisacak: {feature_percent_values}")

	for percent_idx, feature_percent in enumerate(feature_percent_values, start=1):
		if len(feature_percent_values) > 1:
			print(
				f"\n[INFO] Feature-percent {percent_idx}/{len(feature_percent_values)} basladi: "
				f"%{feature_percent}"
			)

		if args.save_encoded_dataset:
			if args.task != "classification":
				raise ValueError("--save-encoded-dataset su an classification modu icin destekleniyor.")
			configure_tensorflow_device(args.device)
			set_reproducible(random_state)
			dataset_filename = convert_txt_dataset_to_csv(args.dataset_name)
			dataset_folder = Path(dataset_filename).stem
			print(f"[INFO] Sadece encoded CSV uretilecek. Veri yukleniyor: {dataset_filename}")
			df = load_data(dataset_filename, folder="raw", target_column=args.target_column)
			encoded_path = save_dimension_reduced_classification_dataset(
				df=df,
				dataset_folder=dataset_folder,
				target_column=args.target_column,
				id_column=normalize_id_column(args.id_column),
				feature_percent=feature_percent,
				random_state=random_state,
				autoencoder_early_stopping_patience=autoencoder_early_stopping_patience if autoencoder_early_stopping_patience > 0 else None,
				autoencoder_early_stopping_min_delta=args.autoencoder_early_stopping_min_delta,
				save_training_plots=args.save_training_plots,
			)
			print(f"[OK] Encoded dataset hazir: {encoded_path.name}")
			continue

		if args.accuracy_list_txt.strip():
			accuracy_txt_path = Path(args.accuracy_list_txt)
			if len(feature_percent_values) > 1:
				feature_percent_tag = format_feature_percent_tag(feature_percent)
				accuracy_txt_path = (
					accuracy_txt_path.parent
					/ f"{accuracy_txt_path.stem}_top_{feature_percent_tag}{accuracy_txt_path.suffix or '.txt'}"
				)
		else:
			dataset_folder = Path(args.dataset_name).stem
			feature_percent_tag = format_feature_percent_tag(feature_percent)
			output_dataset_folder = dataset_folder
			output_feature_percent = feature_percent
			metric_output_prefix = format_metric_output_prefix(output_feature_percent, output_dataset_folder)
			if args.task == "clustering":
				accuracy_txt_path = (
					clustering_output_dir(output_dataset_folder)
					/ "metrics"
					/ f"{metric_output_prefix}_cluster_rmse_runs.txt"
				)
			elif args.task == "regression":
				accuracy_txt_path = (
					regression_output_dir(output_dataset_folder)
					/ "metrics"
					/ f"{metric_output_prefix}_pearson_r_runs.txt"
				)
			else:
				accuracy_filename = f"{metric_output_prefix}_accuracy_runs.txt"
				if args.evaluate_dimension_reduction:
					accuracy_filename = f"top_{feature_percent_tag}_dimension_reduction_accuracy_runs.txt"
				accuracy_txt_path = (
					classification_output_dir(output_dataset_folder)
					/ "metrics"
					/ accuracy_filename
				)
		ensure_dir(accuracy_txt_path.parent)

		accuracy_values, average_accuracy = run_repeated_experiments(
			dataset_name=args.dataset_name,
			validation_dataset_name=args.validation_dataset_name.strip() or None,
			target_column=args.target_column,
			id_column=args.id_column,
			task=args.task,
			encoding_dim=args.encoding_dim,
			feature_percent=feature_percent,
			random_state=random_state,
			classifier_epochs=args.classifier_epochs,
			classifier_hidden_units=classifier_hidden_units,
			classifier_dropout_rates=classifier_dropout_rates,
			classifier_learning_rate=args.classifier_learning_rate,
			classifier_model=args.classifier_model,
			regression_model=args.regression_model,
			svr_kernel=args.svr_kernel,
			svr_c=args.svr_c,
			svr_epsilon=args.svr_epsilon,
			svr_gamma=svr_gamma,
			kmeans_regression_clusters=args.kmeans_regression_clusters,
			kmeans_regression_n_init=args.kmeans_regression_n_init,
			device=args.device,
			feature_chunk_size=args.feature_chunk_size,
			chunk_feature_threshold=args.chunk_feature_threshold,
			enable_feature_chunking=not args.disable_feature_chunking,
			classifier_early_stopping_patience=args.classifier_early_stopping_patience if args.classifier_early_stopping_patience > 0 else None,
			autoencoder_early_stopping_patience=autoencoder_early_stopping_patience if autoencoder_early_stopping_patience > 0 else None,
			classifier_early_stopping_monitor=args.classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta=args.classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta=args.autoencoder_early_stopping_min_delta,
			classifier_class_weight=args.classifier_class_weight,
			classifier_sampling=args.classifier_sampling,
			cluster_k=cluster_k,
			cluster_min_k=cluster_min_k,
			cluster_max_k=cluster_max_k,
			save_training_plots=args.save_training_plots,
			actual_predicted_top_n=args.actual_predicted_top_n,
			evaluate_dimension_reduction=args.evaluate_dimension_reduction,
			repeat_runs=args.repeat_runs,
			accuracy_txt_path=accuracy_txt_path,
			metric_name=metric_name,
		)

		result_label = format_feature_output_label(feature_percent, Path(args.dataset_name).stem)
		print(f"[OK] {result_label} {metric_name} listesi yazildi: {accuracy_txt_path}")
		print(f"[OK] {result_label} {metric_name} dizisi: {accuracy_values}")
		print(f"[OK] {result_label} Ortalama {metric_name}: {average_accuracy:.6f}")
