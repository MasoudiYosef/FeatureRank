import os
import sys
import argparse
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
	accuracy_score,
	mean_absolute_error,
	mean_squared_error,
	r2_score,
	silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Proje kokunu import path'ine ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RANDOM_STATE
from src.data_loader import convert_txt_dataset_to_csv, load_data
from src.models import build_sigmoid_autoencoder, build_latent_classifier, build_latent_regressor
from src.preprocessing import preprocess_data, scale_data
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
DEFAULT_FEATURE_CHUNK_SIZE = 1000
DEFAULT_CHUNK_FEATURE_THRESHOLD = 50000
DEFAULT_CLUSTER_MIN_K = 2
DEFAULT_CLUSTER_MAX_K = 15
DEFAULT_EARLY_STOPPING_PATIENCE = 0
DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE = 0
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.0
DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA = 0.001
DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR = "val_accuracy"


def set_reproducible(seed: int | None) -> None:
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

	if "average_loss" in average_df.columns:
		plt.figure(figsize=(8, 5))
		plt.plot(average_df["epoch"], average_df["average_loss"], label="average_loss")
		if "average_val_loss" in average_df.columns:
			plt.plot(average_df["epoch"], average_df["average_val_loss"], label="average_val_loss")
		plt.xlabel("Epoch")
		plt.ylabel("Average Loss")
		plt.title(f"{file_prefix} average error convergence")
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
) -> None:
	if not metric_values:
		return

	ensure_dir(output_dir)
	metric_label = metric_name.lower()
	plt.figure(figsize=(6, 5))
	plt.boxplot(metric_values, labels=[metric_name], showmeans=True)
	plt.ylabel(metric_name)
	plt.title(f"{file_prefix} {metric_label} boxplot")
	plt.grid(True, axis="y", alpha=0.3)
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
) -> None:
	if len(y_true) == 0 or len(y_pred) == 0:
		return

	ensure_dir(output_dir)
	y_true = np.asarray(y_true, dtype=float).ravel()
	y_pred = np.asarray(y_pred, dtype=float).ravel()
	min_value = float(min(np.min(y_true), np.min(y_pred)))
	max_value = float(max(np.max(y_true), np.max(y_pred)))

	plt.figure(figsize=(7, 6))
	plt.scatter(y_true, y_pred, alpha=0.75, s=28)
	plt.plot([min_value, max_value], [min_value, max_value], color="#d62728", linestyle="--", linewidth=1.4)
	plt.xlabel("Actual values")
	plt.ylabel("Predicted values")
	plt.title(f"{file_prefix} actual vs predicted")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()

	plot_path = output_dir / f"{file_prefix}_actual_vs_predicted.png"
	plt.savefig(plot_path, dpi=150)
	plt.close()
	print(f"[OK] Actual vs predicted plot: {plot_path}")


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
) -> pd.DataFrame | None:
	feature_percent_tag = format_feature_percent_tag(feature_percent)
	history_dir = Path("outputs") / "autoencoder" / dataset_folder / "training_history"
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
	feature_percent_tag = format_feature_percent_tag(feature_percent)
	metrics_path = (
		Path("outputs")
		/ "autoencoder"
		/ dataset_folder
		/ "metrics"
		/ f"top_{feature_percent_tag}_test_metrics.json"
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
		history_dir = Path("outputs") / "autoencoder" / dataset_folder / binary_dataset_folder / "training_history"
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
	history_dir = Path("outputs") / "autoencoder" / dataset_folder / "training_history"
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
	early_stopping_patience: int | None = None,
	early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	history_output_dir: Path | None = None,
	history_prefix: str | None = None,
) -> tuple[float, tf.keras.Model, tf.keras.Model]:
	autoencoder, encoder = build_sigmoid_autoencoder(
		input_dim=X_train_sub.shape[1],
		encoding_dim=encoding_dim,
		activation="sigmoid",
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
		epochs=AUTOENCODER_EPOCHS,
		batch_size=BATCH_SIZE,
		shuffle=True,
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
	classifier_early_stopping_patience: int | None = None,
	autoencoder_early_stopping_patience: int | None = None,
	classifier_early_stopping_monitor: str = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR,
	classifier_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	autoencoder_early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	history_output_dir: Path | None = None,
	history_prefix: str | None = None,
) -> tuple[float, float, tf.keras.Model, tf.keras.Model, np.ndarray]:
	X_train_sub, X_val, y_train_sub, y_val = train_test_split(
		X_train,
		y_train,
		test_size=CLASSIFIER_VALIDATION_SPLIT,
		random_state=random_state,
		shuffle=True,
		stratify=y_train,
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
	)

	X_train_encoded = encoder.predict(X_train_sub, verbose=0).astype(np.float32)
	X_val_encoded = encoder.predict(X_val, verbose=0).astype(np.float32)
	X_test_encoded = encoder.predict(X_test, verbose=0).astype(np.float32)

	# Get encoded dimension for validation
	encoder_output_dim = X_train_encoded.shape[1]
	
	classifier = build_latent_classifier(
		input_dim=encoder_output_dim,
		hidden_units=classifier_hidden_units,
		dropout_rates=classifier_dropout_rates,
		learning_rate=classifier_learning_rate,
	)
	y_train_fit = y_train_sub.astype(np.float32)
	y_val_fit = y_val.astype(np.float32)
	
	# Verify input/output shapes match model expectations
	if X_train_encoded.shape[1] != encoder_output_dim:
		raise ValueError(
			f"Encoder output dim {X_train_encoded.shape[1]} != expected dim {encoder_output_dim}"
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
		X_train_encoded,
		y_train_fit,
		epochs=classifier_epochs,
		batch_size=BATCH_SIZE,
		validation_data=(X_val_encoded, y_val_fit),
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
	# Handle both single-output (sigmoid) and multi-output predictions
	if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 1:
		y_pred_prob = y_pred_prob.ravel()
	y_pred = (y_pred_prob > THRESHOLD).astype(int).ravel()
	
	if len(y_pred) != len(y_test):
		raise ValueError(
			f"Prediction length {len(y_pred)} != y_test length {len(y_test)}"
		)
	
	test_accuracy = float(accuracy_score(y_test.astype(int), y_pred))
	print("classifier output shape:", classifier.output_shape)
	return test_mse, test_accuracy, autoencoder, encoder,X_train_sub


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
	regressor_early_stopping_patience: int | None = None,
	autoencoder_early_stopping_patience: int | None = None,
	regressor_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	autoencoder_early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	history_output_dir: Path | None = None,
	history_prefix: str | None = None,
) -> tuple[dict, tf.keras.Model, tf.keras.Model, np.ndarray, np.ndarray, np.ndarray]:
	X_train_sub, X_val, y_train_sub, y_val = train_test_split(
		X_train,
		y_train,
		test_size=CLASSIFIER_VALIDATION_SPLIT,
		random_state=random_state,
		shuffle=True,
	)

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
		y_train_sub.astype(np.float32),
		epochs=regressor_epochs,
		batch_size=BATCH_SIZE,
		validation_data=(X_val_encoded, y_val.astype(np.float32)),
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

	y_pred = regressor.predict(X_test_encoded, verbose=0).ravel()
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
	}
	#regression_r2 = model performansı / açıklama gücü
	#pearson_r     = gerçek-tahmin korelasyonu
	print("regressor output shape:", regressor.output_shape)
	return metrics, autoencoder, encoder, X_train_sub, y_true, y_pred


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
) -> tuple[pd.DataFrame, dict, np.ndarray]:
	min_k, max_k = normalize_cluster_k_range(min_k, max_k, X_cluster.shape[0])
	rows: list[dict] = []
	best_labels: np.ndarray | None = None
	best_row: dict | None = None

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
			"silhouette_score": silhouette,
		}
		rows.append(row)
		if not np.isnan(silhouette) and (best_row is None or silhouette > best_row["silhouette_score"]):
			best_row = row
			best_labels = labels

	if best_row is None or best_labels is None:
		raise ValueError("Gecerli silhouette skoru hesaplanamadi. k araligini veya veri boyutunu kontrol edin.")

	return pd.DataFrame(rows), best_row, best_labels


def save_cluster_evaluation_plots(scores_df: pd.DataFrame, output_dir: Path, file_prefix: str) -> None:
	if scores_df.empty or "k" not in scores_df.columns:
		return

	ensure_dir(output_dir)
	if "inertia" in scores_df.columns:
		plt.figure(figsize=(8, 5))
		plt.plot(scores_df["k"], scores_df["inertia"], marker="o")
		plt.xlabel("Number of clusters (k)")
		plt.ylabel("Within-cluster sum of squares (Inertia)")
		plt.title(f"{file_prefix} elbow method")
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
			plt.xlabel("Number of clusters (k)")
			plt.ylabel("Silhouette score")
			plt.title(f"{file_prefix} silhouette score")
			plt.grid(True, alpha=0.3)
			plt.tight_layout()
			silhouette_path = output_dir / f"{file_prefix}_silhouette.png"
			plt.savefig(silhouette_path, dpi=150)
			plt.close()
			print(f"[OK] Silhouette plot: {silhouette_path}")


def save_cluster_pca_scatter(
	X_cluster: np.ndarray,
	labels: np.ndarray,
	output_dir: Path,
	file_prefix: str,
) -> None:
	if X_cluster.shape[0] < 2 or X_cluster.shape[1] < 2:
		return

	ensure_dir(output_dir)
	pca = PCA(n_components=2, random_state=RANDOM_STATE)
	X_2d = pca.fit_transform(X_cluster)
	explained = pca.explained_variance_ratio_ * 100

	plt.figure(figsize=(8, 6))
	scatter = plt.scatter(
		X_2d[:, 0],
		X_2d[:, 1],
		c=labels,
		cmap="tab10",
		s=28,
		alpha=0.8,
		edgecolors="none",
	)
	plt.xlabel(f"PC1 ({explained[0]:.1f}% variance)")
	plt.ylabel(f"PC2 ({explained[1]:.1f}% variance)")
	plt.title(f"{file_prefix} cluster visualization (PCA 2D)")
	plt.grid(True, alpha=0.25)
	plt.colorbar(scatter, label="Cluster")
	plt.tight_layout()

	plot_path = output_dir / f"{file_prefix}_clusters_pca_2d.png"
	plt.savefig(plot_path, dpi=150)
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

	output_dir = Path("outputs") / "autoencoder" / dataset_folder
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

	output_dir = Path("outputs") / "autoencoder" / dataset_folder
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
	cluster_min_k: int,
	cluster_max_k: int,
	feature_chunk_size: int,
	chunk_feature_threshold: int,
	enable_feature_chunking: bool,
	save_training_plots: bool,
) -> tuple[float, float]:
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

	output_dir = Path("outputs") / "clustering" / dataset_folder
	metrics_dir = output_dir / "metrics"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)

	print(f"[INFO] Clustering modu basladi. X shape: {X_raw.shape}")
	print("[INFO] Label varsa clustering egitiminde kullanilmayacak; k degeri silhouette skoruna gore secilecek.")

	effective_min_k = cluster_min_k
	effective_max_k = cluster_max_k
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
	)
	org_scores_path = output_dir / "ORG_cluster_scores.csv"
	org_scores_df.to_csv(org_scores_path, index=False)
	save_cluster_evaluation_plots(
		scores_df=org_scores_df,
		output_dir=output_dir,
		file_prefix="ORG",
	)
	save_cluster_pca_scatter(
		X_cluster=X_org_scaled,
		labels=org_best_labels,
		output_dir=output_dir,
		file_prefix="ORG",
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
	org_metrics_data = {
		"task": "clustering",
		"feature_set": "ORG",
		"original_feature_count": len(feature_names),
		"selected_feature_count": len(feature_names),
		"cluster_min_k": effective_min_k,
		"cluster_max_k": effective_max_k,
		"best_k": int(org_best_row["k"]),
		"silhouette_score": float(org_best_row["silhouette_score"]),
		"inertia": float(org_best_row["inertia"]),
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
	)

	scores_path = output_dir / f"top_{feature_percent_tag}_cluster_scores.csv"
	scores_df.to_csv(scores_path, index=False)
	save_cluster_evaluation_plots(
		scores_df=scores_df,
		output_dir=output_dir,
		file_prefix=f"top_{feature_percent_tag}",
	)
	save_cluster_pca_scatter(
		X_cluster=X_selected_scaled,
		labels=best_labels,
		output_dir=output_dir,
		file_prefix=f"top_{feature_percent_tag}",
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

	metrics_data = {
		"task": "clustering",
		"feature_set": f"top_{feature_percent_tag}",
		"feature_percent": feature_percent,
		"original_feature_count": len(feature_names),
		"selected_feature_count": len(selected_df),
		"cluster_min_k": effective_min_k,
		"cluster_max_k": effective_max_k,
		"best_k": int(best_row["k"]),
		"silhouette_score": float(best_row["silhouette_score"]),
		"inertia": float(best_row["inertia"]),
	}
	metrics_path = metrics_dir / f"top_{feature_percent_tag}_cluster_metrics.json"
	save_json(metrics_data, metrics_path)

	print("\n[OK] Clustering tamamlandi.")
	print(f"[OK] ORG silhouette_score: {float(org_best_row['silhouette_score']):.6f}")
	print(f"[OK] ORG best k: {int(org_best_row['k'])}")
	print(f"[OK] ORG metrik dosyasi: {org_metrics_path}")
	print(f"[OK] Top %{feature_percent} secilen feature sayisi: {len(selected_df)}")
	print(f"[OK] En iyi k: {int(best_row['k'])}")
	print(f"[OK] En iyi silhouette_score: {float(best_row['silhouette_score']):.6f}")
	print(f"[OK] Elbow/Inertia skor CSV: {scores_path}")
	print(f"[OK] Cluster atamalari: {assignments_path}")
	return float(best_row["silhouette_score"]), float(best_row["silhouette_score"])


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
	classifier_early_stopping_patience: int | None,
	autoencoder_early_stopping_patience: int | None,
	classifier_early_stopping_monitor: str,
	classifier_early_stopping_min_delta: float,
	autoencoder_early_stopping_min_delta: float,
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

	output_dir = Path("outputs") / "autoencoder" / dataset_folder
	metrics_dir = output_dir / "metrics"
	chunks_dir = output_dir / "chunks"
	history_dir = output_dir / "training_history"
	filtered_data_dir = Path("data") / "autoencoder" / dataset_folder
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

		chunk_test_mse, chunk_test_accuracy, chunk_autoencoder, _, chunk_train_sub = train_and_evaluate_pipeline(
			X_train_chunk,
			X_test_chunk,
			y_train,
			y_test,
			encoding_dim,
			random_state,
			classifier_epochs,
			classifier_hidden_units,
			classifier_dropout_rates,
			classifier_learning_rate,
			classifier_early_stopping_patience,
			autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta,
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
				"test_accuracy": chunk_test_accuracy,
				"weights_path": str(chunk_weights_path),
				"selected_features_path": str(chunk_selected_path),
			}
		)

		save_json(chunk_summaries[-1], metrics_dir / f"{chunk_name}_test_metrics.json")
		print(
			f"[OK] {chunk_name} tamamlandi. "
			f"Top %{feature_percent}: {len(chunk_selected_df)} feature, "
			f"accuracy: {chunk_test_accuracy:.6f}"
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
	final_test_mse, final_test_accuracy, _, _, _ = train_and_evaluate_pipeline(
		X_train_merged,
		X_test_merged,
		y_train,
		y_test,
		encoding_dim,
		random_state,
		classifier_epochs,
		classifier_hidden_units,
		classifier_dropout_rates,
		classifier_learning_rate,
		classifier_early_stopping_patience,
		autoencoder_early_stopping_patience,
		classifier_early_stopping_monitor,
		classifier_early_stopping_min_delta,
		autoencoder_early_stopping_min_delta,
		history_output_dir=history_dir if save_training_plots else None,
		history_prefix=f"chunked_top_{feature_percent_tag}_final" if save_training_plots else None,
	)

	final_metrics_data = {
		"chunked_feature_selection": True,
		"feature_percent": feature_percent,
		"original_feature_count": len(feature_names),
		"feature_chunk_size": feature_chunk_size,
		"chunk_count": len(feature_chunks),
		"merged_feature_count": len(merged_feature_names),
		"test_mse": final_test_mse,
		"test_accuracy": final_test_accuracy,
		"threshold": THRESHOLD,
		"chunk_summaries": chunk_summaries,
		"all_chunk_selected_features_path": str(all_chunk_selected_path),
		"merged_selected_features_path": str(merged_selected_path),
		"merged_dataset_path": str(merged_dataset_path),
	}
	if current_class_label is not None and class_counts is not None:
		final_metrics_data["current_class_label"] = current_class_label
		final_metrics_data["class_counts"] = class_counts
		final_metrics_data["binary_label_counts"] = {
			"label_0": int(np.sum(y_test == 0)),
			"label_1": int(np.sum(y_test == 1)),
		}

	save_json(final_metrics_data, metrics_dir / f"chunked_top_{feature_percent_tag}_test_metrics.json")
	save_json(final_metrics_data, metrics_dir / f"top_{feature_percent_tag}_test_metrics.json")

	print("\n[OK] Chunked autoencoder akisi tamamlandi.")
	print(f"[OK] Orijinal feature sayisi: {len(feature_names)}")
	print(f"[OK] Chunk sayisi: {len(feature_chunks)}")
	print(f"[OK] Birlesen top feature sayisi: {len(merged_feature_names)}")
	print(f"[OK] Final dataset test_mse: {final_test_mse:.6f}")
	print(f"[OK] Final dataset test_accuracy: {final_test_accuracy:.6f}")
	print(f"[OK] Chunk secim CSV: {all_chunk_selected_path}")
	print(f"[OK] Birlesik feature CSV: {merged_selected_path}")
	print(f"[OK] Birlesik dataset CSV: {merged_dataset_path} (satir: {len(merged_filtered_df)})")
	print(f"[OK] Final metrik dosyasi: {metrics_dir / f'chunked_top_{feature_percent_tag}_test_metrics.json'}")
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
	feature_chunk_size: int = DEFAULT_FEATURE_CHUNK_SIZE,
	chunk_feature_threshold: int = DEFAULT_CHUNK_FEATURE_THRESHOLD,
	enable_feature_chunking: bool = True,
	classifier_early_stopping_patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE,
	autoencoder_early_stopping_patience: int | None = DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE,
	classifier_early_stopping_monitor: str = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR,
	classifier_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	autoencoder_early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	save_training_plots: bool = False,
	current_class_label: int | None = None,
	class_counts: dict[int, int] | None = None,
) -> tuple[float, float]:
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
				classifier_early_stopping_patience=classifier_early_stopping_patience,
				autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
				classifier_early_stopping_monitor=classifier_early_stopping_monitor,
				classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
				autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
				feature_chunk_size=feature_chunk_size,
				save_training_plots=save_training_plots,
				current_class_label=current_class_label,
				class_counts=class_counts,
			)

	X_train, X_test, _ = scale_data(X_train_raw, X_test_raw)
	X_train = X_train.astype(np.float32)
	X_test = X_test.astype(np.float32)

	output_dir = Path("outputs") / "autoencoder" / dataset_folder
	metrics_dir = output_dir / "metrics"
	history_dir = output_dir / "training_history"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)
	if save_training_plots:
		ensure_dir(history_dir)

	test_mse, test_accuracy, autoencoder, _, X_train_sub_used = train_and_evaluate_pipeline(
		X_train,
		X_test,
		y_train,
		y_test,
		encoding_dim,
		random_state,
		classifier_epochs,
		classifier_hidden_units,
		classifier_dropout_rates,
		classifier_learning_rate,
		classifier_early_stopping_patience,
		autoencoder_early_stopping_patience,
		classifier_early_stopping_monitor,
		classifier_early_stopping_min_delta,
		autoencoder_early_stopping_min_delta,
		history_output_dir=history_dir if save_training_plots else None,
		history_prefix="ORG" if save_training_plots else None,
	)

	feature_names = X_train_raw.columns.tolist()
	weights_path = output_dir / "first_layer_W_list.csv"
	save_feature_weighted_lists(autoencoder, X_train_sub_used, feature_names, weights_path)

	feature_percent_tag = format_feature_percent_tag(feature_percent)
	selected_features_path = output_dir / f"top_{feature_percent_tag}_max_abs_features.csv"
	selected_df = save_top_percent_features_by_abs_max_weight(
		weight_list_csv_path=weights_path,
		feature_names=feature_names,
		feature_percent=feature_percent,
		output_path=selected_features_path,
	)

	filtered_data_dir = Path("data") / "autoencoder" / dataset_folder
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
		print("[INFO] Top %100 tum feature'lari iceriyor. ORG sonucu yeniden egitilmeden kullaniliyor.")
	else:
		selected_feature_names = selected_df["feature_name"].tolist()
		X_train_filtered_raw = X_train_raw[selected_feature_names]
		X_test_filtered_raw = X_test_raw[selected_feature_names]
		X_train_filtered, X_test_filtered, _ = scale_data(X_train_filtered_raw, X_test_filtered_raw)
		# Ensure consistent float32 dtype
		X_train_filtered = X_train_filtered.astype(np.float32)
		X_test_filtered = X_test_filtered.astype(np.float32)
		filtered_test_mse, filtered_test_accuracy, _, _,_= train_and_evaluate_pipeline(
			X_train_filtered,
			X_test_filtered,
			y_train_filtered,
			y_test_filtered,
			encoding_dim,
			random_state,
			classifier_epochs,
			classifier_hidden_units,
			classifier_dropout_rates,
			classifier_learning_rate,
			classifier_early_stopping_patience,
			autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta,
			history_output_dir=history_dir if save_training_plots else None,
			history_prefix=f"top_{feature_percent_tag}" if save_training_plots else None,
		)

	org_metrics_data = {
		"test_mse": test_mse,
		"test_accuracy": test_accuracy,
		"threshold": THRESHOLD,
	}
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

	filtered_metrics_data = {
		"feature_percent": feature_percent,
		"selected_feature_count": len(selected_df),
		"test_mse": filtered_test_mse,
		"test_accuracy": filtered_test_accuracy,
		"threshold": THRESHOLD,
	}
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
		metrics_dir / f"top_{feature_percent_tag}_test_metrics.json",
	)

	print("\n[OK] Autoencoder egitimi tamamlandi.")
	#print(f"[OK] test_mse: {test_mse:.6f}")
	print(f"[OK] test_accuracy: {test_accuracy:.6f}")
	#print(f"[OK] Feature weighted listeleri: {weights_path}")
	print(f"[OK] Top %{feature_percent} secilen feature sayisi: {len(selected_df)}")
	print(f"[OK] Secilen feature CSV: {selected_features_path}")
	#print(f"[OK] Filterlenmis dataset CSV: {filtered_dataset_path} (satir: {len(filtered_df)})")
	#print(f"[OK] Top %{feature_percent} dataset test_mse: {filtered_test_mse:.6f}")
	print(f"[OK] Top %{feature_percent} dataset test_accuracy: {filtered_test_accuracy:.6f}")
	filtered_metrics_path = metrics_dir / f"top_{feature_percent_tag}_test_metrics.json"
	print(f"[OK] Top %{feature_percent} metrik dosyasi: {filtered_metrics_path}")
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
	classifier_early_stopping_patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE,
	autoencoder_early_stopping_patience: int | None = DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE,
	classifier_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	autoencoder_early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	save_training_plots: bool = False,
) -> tuple[float, float]:
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

	print(f"[INFO] Regression modu. X_train shape: {X_train_raw.shape}")
	print(f"[INFO] X_test shape : {X_test_raw.shape}")

	X_train, X_test, _ = scale_data(X_train_raw, X_test_raw)
	X_train = X_train.astype(np.float32)
	X_test = X_test.astype(np.float32)

	output_dir = Path("outputs") / "autoencoder" / dataset_folder
	metrics_dir = output_dir / "metrics"
	history_dir = output_dir / "training_history"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)
	if save_training_plots:
		ensure_dir(history_dir)

	org_metrics, autoencoder, _, X_train_sub_used, org_y_true, org_y_pred = train_and_evaluate_regression_pipeline(
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
	)

	feature_names = X_train_raw.columns.tolist()
	weights_path = output_dir / "first_layer_W_list.csv"
	save_feature_weighted_lists(autoencoder, X_train_sub_used, feature_names, weights_path)

	feature_percent_tag = format_feature_percent_tag(feature_percent)
	selected_features_path = output_dir / f"top_{feature_percent_tag}_max_abs_features.csv"
	selected_df = save_top_percent_features_by_abs_max_weight(
		weight_list_csv_path=weights_path,
		feature_names=feature_names,
		feature_percent=feature_percent,
		output_path=selected_features_path,
	)

	filtered_data_dir = Path("data") / "autoencoder" / dataset_folder
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
		print("[INFO] Top %100 tum feature'lari iceriyor. ORG regression sonucu yeniden egitilmeden kullaniliyor.")
	else:
		selected_feature_names = selected_df["feature_name"].tolist()
		X_train_filtered_raw = X_train_raw[selected_feature_names]
		X_test_filtered_raw = X_test_raw[selected_feature_names]
		X_train_filtered, X_test_filtered, _ = scale_data(X_train_filtered_raw, X_test_filtered_raw)
		filtered_metrics, _, _, _, filtered_y_true, filtered_y_pred = train_and_evaluate_regression_pipeline(
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
	)

	org_metrics_data = {"task": "regression", **org_metrics}
	filtered_metrics_data = {
		"task": "regression",
		"feature_percent": feature_percent,
		"selected_feature_count": len(selected_df),
		**filtered_metrics,
	}
	save_json(org_metrics_data, metrics_dir / "ORG_test_metrics.json")
	save_json(filtered_metrics_data, metrics_dir / f"top_{feature_percent_tag}_test_metrics.json")

	print("\n[OK] Regression autoencoder egitimi tamamlandi.")
	print(f"[OK] ORG R2: {org_metrics['regression_r2']:.6f}")
	print(f"[OK] ORG RMSE: {org_metrics['regression_rmse']:.6f}")
	print(f"[OK] Top %{feature_percent} secilen feature sayisi: {len(selected_df)}")
	print(f"[OK] Top %{feature_percent} R2: {filtered_metrics['regression_r2']:.6f}")
	print(f"[OK] Top %{feature_percent} RMSE: {filtered_metrics['regression_rmse']:.6f}")
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
	feature_chunk_size: int,
	classifier_early_stopping_patience: int | None,
	autoencoder_early_stopping_patience: int | None,
	classifier_early_stopping_monitor: str,
	classifier_early_stopping_min_delta: float,
	autoencoder_early_stopping_min_delta: float,
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
				classifier_early_stopping_patience=classifier_early_stopping_patience,
				autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
				classifier_early_stopping_monitor=classifier_early_stopping_monitor,
				classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
				autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
				feature_chunk_size=feature_chunk_size,
				chunk_feature_threshold=chunk_feature_threshold,
				enable_feature_chunking=enable_feature_chunking,
				save_training_plots=save_training_plots,
				current_class_label=class_label,
				class_counts=class_counts,
			)

	feature_percent_tag = format_feature_percent_tag(feature_percent)
	filtered_metric_filename = f"top_{feature_percent_tag}_test_metrics.json"
	macro_filtered_accuracy = compute_multiclass_macro_accuracy(
		dataset_folder=dataset_folder,
		class_labels=class_labels,
		metric_filename=filtered_metric_filename,
	)
	try:
		macro_org_accuracy = compute_multiclass_macro_accuracy(
			dataset_folder=dataset_folder,
			class_labels=class_labels,
			metric_filename="ORG_test_metrics.json",
		)
	except FileNotFoundError:
		macro_org_accuracy = macro_filtered_accuracy

	output_dir = Path("outputs") / "autoencoder" / dataset_folder
	metrics_dir = output_dir / "metrics"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)

	save_json(
		{
			"num_classes": len(class_labels),
			"class_labels": class_labels,
			"macro_average": True,
			"test_accuracy": macro_org_accuracy,
		},
		metrics_dir / "ORG_test_metrics.json",
	)

	save_json(
		{
			"feature_percent": feature_percent,
			"num_classes": len(class_labels),
			"class_labels": class_labels,
			"macro_average": True,
			"test_accuracy": macro_filtered_accuracy,
		},
		metrics_dir / f"top_{feature_percent_tag}_test_metrics.json",
	)

	print("\n[OK] Multi-class one-vs-rest tamamlandi.")
	print(f"[OK] ORG macro test_accuracy: {macro_org_accuracy:.6f}")
	print(f"[OK] Top %{feature_percent} macro test_accuracy: {macro_filtered_accuracy:.6f}")
	print(f"[OK] Metrik dosyasi: {metrics_dir / 'ORG_test_metrics.json'}")
	return macro_org_accuracy, macro_filtered_accuracy


def main(
	dataset_name: str = "breast_cancer_data.csv",
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
	device: str = "auto",
	feature_chunk_size: int = DEFAULT_FEATURE_CHUNK_SIZE,
	chunk_feature_threshold: int = DEFAULT_CHUNK_FEATURE_THRESHOLD,
	enable_feature_chunking: bool = True,
	classifier_early_stopping_patience: int | None = DEFAULT_EARLY_STOPPING_PATIENCE,
	autoencoder_early_stopping_patience: int | None = DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE,
	classifier_early_stopping_monitor: str = DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR,
	classifier_early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
	autoencoder_early_stopping_min_delta: float = DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA,
	cluster_min_k: int = DEFAULT_CLUSTER_MIN_K,
	cluster_max_k: int = DEFAULT_CLUSTER_MAX_K,
	save_training_plots: bool = False,
) -> tuple[float, float]:
	task = task.lower().strip()
	if task not in {"classification", "clustering", "regression"}:
		raise ValueError("task 'classification', 'clustering' veya 'regression' olmali.")
	configure_tensorflow_device(device)
	set_reproducible(random_state)
	if random_state is None:
		print("[INFO] random_state: None (rastgele)")
	else:
		print(f"[INFO] random_state: {random_state} (sabit)")
	feature_percent = validate_feature_percent(feature_percent)
	id_column = normalize_id_column(id_column)

	dataset_filename = convert_txt_dataset_to_csv(dataset_name)
	dataset_folder = Path(dataset_filename).stem

	print(f"[INFO] Veri yukleniyor: {dataset_filename}")
	df = load_data(dataset_filename, folder="raw", target_column=target_column)

	if task == "clustering":
		return run_clustering_experiment(
			df=df,
			dataset_folder=dataset_folder,
			target_column=target_column,
				id_column=id_column,
				encoding_dim=encoding_dim,
				feature_percent=feature_percent,
				random_state=random_state,
				cluster_min_k=cluster_min_k,
				cluster_max_k=cluster_max_k,
				feature_chunk_size=feature_chunk_size,
				chunk_feature_threshold=chunk_feature_threshold,
				enable_feature_chunking=enable_feature_chunking,
				save_training_plots=save_training_plots,
			)

	if task == "regression":
		return run_regression_experiment(
			df=df,
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
			classifier_early_stopping_patience=classifier_early_stopping_patience,
			autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
			classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
			save_training_plots=save_training_plots,
		)

	if task == "classification" and is_probable_regression_target(df[target_column]):
		raise ValueError(
			"Bu veri setinin target kolonu sürekli sayısal görünüyor; classification olarak çalıştırılamaz. "
			"Regression için komutu şu şekilde kullan: --task regression"
		)

	class_count = int(df[target_column].nunique(dropna=True))
	if class_count > 2:
		return run_multiclass_one_vs_rest(
			df=df,
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
			classifier_early_stopping_patience=classifier_early_stopping_patience,
			autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor=classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
			feature_chunk_size=feature_chunk_size,
			chunk_feature_threshold=chunk_feature_threshold,
			enable_feature_chunking=enable_feature_chunking,
			save_training_plots=save_training_plots,
		)

	return run_binary_experiment(
		df=df,
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
		classifier_early_stopping_patience=classifier_early_stopping_patience,
		autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
		classifier_early_stopping_monitor=classifier_early_stopping_monitor,
		classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
		autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
		feature_chunk_size=feature_chunk_size,
		chunk_feature_threshold=chunk_feature_threshold,
		enable_feature_chunking=enable_feature_chunking,
		save_training_plots=save_training_plots,
	)


def run_repeated_experiments(
	dataset_name: str,
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
	device: str,
	feature_chunk_size: int,
	chunk_feature_threshold: int,
	enable_feature_chunking: bool,
	classifier_early_stopping_patience: int | None,
	autoencoder_early_stopping_patience: int | None,
	classifier_early_stopping_monitor: str,
	classifier_early_stopping_min_delta: float,
	autoencoder_early_stopping_min_delta: float,
	cluster_min_k: int,
	cluster_max_k: int,
	save_training_plots: bool,
	repeat_runs: int,
	accuracy_txt_path: Path,
	metric_name: str = "Accuracy",
) -> tuple[list[float], float]:
	"""
	Ayni deneyi repeat_runs kadar calistirir ve metric degerlerini kaydeder.
	Sonuc: (metric_values, average_metric)
	"""
	accuracy_values: list[float] = []
	history_frames: list[pd.DataFrame] = []
	dataset_folder = Path(convert_txt_dataset_to_csv(dataset_name)).stem

	for run_idx in range(1, repeat_runs + 1):
		print(f"\n[INFO] Calisma {run_idx}/{repeat_runs} basladi.")
		_, filtered_test_accuracy = main(
			dataset_name=dataset_name,
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
			device=device,
			feature_chunk_size=feature_chunk_size,
			chunk_feature_threshold=chunk_feature_threshold,
			enable_feature_chunking=enable_feature_chunking,
			classifier_early_stopping_patience=classifier_early_stopping_patience,
			autoencoder_early_stopping_patience=autoencoder_early_stopping_patience,
			classifier_early_stopping_monitor=classifier_early_stopping_monitor,
			classifier_early_stopping_min_delta=classifier_early_stopping_min_delta,
			autoencoder_early_stopping_min_delta=autoencoder_early_stopping_min_delta,
			cluster_min_k=cluster_min_k,
			cluster_max_k=cluster_max_k,
			save_training_plots=save_training_plots,
		)
		accuracy_values.append(float(filtered_test_accuracy))
		if save_training_plots and task in {"classification", "regression"}:
			history_df = collect_repeated_run_history(
				dataset_folder=dataset_folder,
				feature_percent=feature_percent,
				run_idx=run_idx,
			)
			if history_df is None and task == "classification":
				history_df = collect_repeated_multiclass_run_history(
					dataset_folder=dataset_folder,
					feature_percent=feature_percent,
					run_idx=run_idx,
				)
			if history_df is not None:
				history_frames.append(history_df)
		sorted_accuracy_values = sorted(accuracy_values, reverse=True)
		accuracy_txt_path.write_text(
			f"{accuracy_values}\nSirali {metric_name}: {sorted_accuracy_values}",
			encoding="utf-8",
		)

	average_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0
	std_accuracy = float(np.std(accuracy_values, ddof=1)) if len(accuracy_values) > 1 else 0.0
	sorted_accuracy_values = sorted(accuracy_values, reverse=True)
	output_text = (
		f"{accuracy_values}\n"
		f"Sirali {metric_name}: {sorted_accuracy_values}\n"
		f"Ortalama {metric_name}: {average_accuracy:.6f}\n"
		f"Std {metric_name}: {std_accuracy:.6f}"
	)
	accuracy_txt_path.write_text(output_text, encoding="utf-8")

	feature_percent_tag = format_feature_percent_tag(feature_percent)
	if task == "clustering":
		plot_output_dir = Path("outputs") / "clustering" / dataset_folder / "metrics"
		boxplot_metric_name = "Silhouette"
	elif task == "regression":
		plot_output_dir = Path("outputs") / "autoencoder" / dataset_folder / "metrics"
		boxplot_metric_name = "Pearson_r"
	else:
		plot_output_dir = Path("outputs") / "autoencoder" / dataset_folder / "metrics"
		boxplot_metric_name = "Accuracy"
	save_metric_boxplot(
		metric_values=accuracy_values,
		output_dir=plot_output_dir,
		file_prefix=f"top_{feature_percent_tag}",
		metric_name=boxplot_metric_name,
	)
	save_repeated_metric_distribution_plot(
		metric_values=accuracy_values,
		output_dir=plot_output_dir,
		file_prefix=f"top_{feature_percent_tag}",
		metric_name=boxplot_metric_name,
	)

	if save_training_plots and task in {"classification", "regression"} and history_frames:
		history_dir = Path("outputs") / "autoencoder" / dataset_folder / "training_history"
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
	parser.add_argument("--target-column", type=str, default="target", help="Hedef kolon adi")
	parser.add_argument("--id-column", type=str, default="ID", help="ID kolon adi, kullanmak istemezsen 'none' ver")
	parser.add_argument("--task", type=str, default="classification", choices=["classification", "clustering", "regression"], help="Calisma modu")
	parser.add_argument("--encoding-dim", type=int, default=8, help="Encoding boyutu")
	parser.add_argument("--feature-percent", type=float, default=20.0, help="Secilecek feature yuzdesi")
	parser.add_argument("--random-state", type=str, default=str(RANDOM_STATE), help="Random state. Ornek: 42 veya none")
	parser.add_argument("--repeat-runs", type=int, default=1, help="Ayni deneyi kac kez calistiracagi")
	parser.add_argument("--accuracy-list-txt", type=str, default="", help="Accuracy listesi txt cikti yolu (bos ise varsayilan yol kullanilir)")
	parser.add_argument("--classifier-epochs", type=int, default=DEFAULT_CLASSIFIER_EPOCHS, help="Classifier epoch sayisi")
	parser.add_argument("--classifier-hidden-units", type=str, default="32,16", help="Classifier gizli katman nöronlari. Ornek: 128,64")
	parser.add_argument("--classifier-dropout-rates", type=str, default="", help="Classifier dropout oranlari. Ornek: 0.3,0.2")
	parser.add_argument("--classifier-learning-rate", type=float, default=0.001, help="Classifier ogrenme orani")
	parser.add_argument("--classifier-early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE, help="Classifier icin early stopping patience. 0 verirsen kapanir")
	parser.add_argument("--autoencoder-early-stopping-patience", type=int, default=None, help="Autoencoder icin val_loss tabanli early stopping patience. 0 verirsen kapanir. Regression modunda verilmezse otomatik classifier patience kullanilir")
	parser.add_argument("--classifier-early-stopping-monitor", type=str, default=DEFAULT_CLASSIFIER_EARLY_STOPPING_MONITOR, choices=["val_loss", "val_accuracy"], help="Classifier early stopping metriği")
	parser.add_argument("--classifier-early-stopping-min-delta", type=float, default=DEFAULT_EARLY_STOPPING_MIN_DELTA, help="Classifier icin minimum validation metric iyilesmesi. Daha kucuk iyilesmeler plato sayilir")
	parser.add_argument("--autoencoder-early-stopping-min-delta", type=float, default=DEFAULT_AUTOENCODER_EARLY_STOPPING_MIN_DELTA, help="Autoencoder icin minimum val_loss iyilesmesi. Daha kucuk iyilesmeler plato sayilir")
	parser.add_argument("--device", type=str, default="auto", choices=["auto", "gpu", "cpu"], help="Cihaz secimi: auto, gpu veya cpu")
	parser.add_argument("--feature-chunk-size", type=int, default=DEFAULT_FEATURE_CHUNK_SIZE, help="Buyuk feature setlerinde her parcadaki feature sayisi")
	parser.add_argument("--chunk-feature-threshold", type=int, default=DEFAULT_CHUNK_FEATURE_THRESHOLD, help="Feature sayisi bu esigi asarsa parcali akis kullanilir")
	parser.add_argument("--disable-feature-chunking", action="store_true", help="Buyuk feature setlerinde otomatik parcali akisi kapatir")
	parser.add_argument("--cluster-min-k", type=int, default=DEFAULT_CLUSTER_MIN_K, help="Clustering icin denenecek minimum k")
	parser.add_argument("--cluster-max-k", type=int, default=DEFAULT_CLUSTER_MAX_K, help="Clustering icin denenecek maksimum k")
	parser.add_argument("--save-training-plots", action="store_true", help="Classification egitim history CSV ve PNG grafiklerini kaydeder")


	args = parser.parse_args()
	random_state = parse_random_state(args.random_state)
	classifier_hidden_units = parse_hidden_units(args.classifier_hidden_units)
	classifier_dropout_rates = parse_dropout_rates(args.classifier_dropout_rates, len(classifier_hidden_units))
	if args.autoencoder_early_stopping_patience is None:
		if args.task == "regression":
			autoencoder_early_stopping_patience = args.classifier_early_stopping_patience
		else:
			autoencoder_early_stopping_patience = DEFAULT_AUTOENCODER_EARLY_STOPPING_PATIENCE
	else:
		autoencoder_early_stopping_patience = args.autoencoder_early_stopping_patience

	if args.accuracy_list_txt.strip():
		accuracy_txt_path = Path(args.accuracy_list_txt)
	else:
		dataset_folder = Path(args.dataset_name).stem
		feature_percent_tag = format_feature_percent_tag(args.feature_percent)
		if args.task == "clustering":
			accuracy_txt_path = Path("outputs") / "clustering" / dataset_folder / "metrics" / f"top_{feature_percent_tag}_silhouette_runs.txt"
		elif args.task == "regression":
			accuracy_txt_path = Path("outputs") / "autoencoder" / dataset_folder / "metrics" / f"top_{feature_percent_tag}_pearson_r_runs.txt"
		else:
			accuracy_txt_path = Path("outputs") / "autoencoder" / dataset_folder / "metrics" / f"top_{feature_percent_tag}_accuracy_runs.txt"
	ensure_dir(accuracy_txt_path.parent)

	if args.task == "clustering":
		metric_name = "Silhouette"
	elif args.task == "regression":
		metric_name = "Pearson_r"
	else:
		metric_name = "Accuracy"
	accuracy_values, average_accuracy = run_repeated_experiments(
		dataset_name=args.dataset_name,
		target_column=args.target_column,
		id_column=args.id_column,
		task=args.task,
		encoding_dim=args.encoding_dim,
		feature_percent=args.feature_percent,
		random_state=random_state,
		classifier_epochs=args.classifier_epochs,
		classifier_hidden_units=classifier_hidden_units,
		classifier_dropout_rates=classifier_dropout_rates,
		classifier_learning_rate=args.classifier_learning_rate,
		device=args.device,
		feature_chunk_size=args.feature_chunk_size,
		chunk_feature_threshold=args.chunk_feature_threshold,
		enable_feature_chunking=not args.disable_feature_chunking,
		classifier_early_stopping_patience=args.classifier_early_stopping_patience if args.classifier_early_stopping_patience > 0 else None,
		autoencoder_early_stopping_patience=autoencoder_early_stopping_patience if autoencoder_early_stopping_patience > 0 else None,
		classifier_early_stopping_monitor=args.classifier_early_stopping_monitor,
		classifier_early_stopping_min_delta=args.classifier_early_stopping_min_delta,
		autoencoder_early_stopping_min_delta=args.autoencoder_early_stopping_min_delta,
		cluster_min_k=args.cluster_min_k,
		cluster_max_k=args.cluster_max_k,
		save_training_plots=args.save_training_plots,
		repeat_runs=args.repeat_runs,
		accuracy_txt_path=accuracy_txt_path,
		metric_name=metric_name,
	)

	print(f"[OK] {metric_name} listesi yazildi: {accuracy_txt_path}")
	print(f"[OK] {metric_name} dizisi: {accuracy_values}")
	print(f"[OK] Ortalama {metric_name}: {average_accuracy:.6f}")
