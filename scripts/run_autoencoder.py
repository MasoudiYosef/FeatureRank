import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Proje kokunu import path'ine ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RANDOM_STATE
from src.data_loader import convert_txt_dataset_to_csv, load_data
from src.models import build_relu_autoencoder, build_latent_classifier
from src.preprocessing import preprocess_data
from src.autoencoder_feature_selection import (
	save_top_percent_features_by_abs_max_weight,
	save_filtered_dataset_from_selected_features,
	validate_feature_percent,
)
from src.utils import ensure_dir, save_json

AUTOENCODER_EPOCHS = 50
BATCH_SIZE = 16
CLASSIFIER_VALIDATION_SPLIT = 0.1
THRESHOLD = 0.5
DEFAULT_CLASSIFIER_EPOCHS = 50
DEFAULT_CLASSIFIER_HIDDEN_UNITS = (32, 16)


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


def save_feature_weighted_lists(autoencoder, X_train: np.ndarray, feature_names: list[str], output_path: Path) -> None:
	"""
	Her feature icin bagli oldugu nöronlara sample-bazli katkı listesi uretir:
	contribution_list_i[j] = mean_s( abs(x_s,i * w_i,j) )
	"""
	weights = autoencoder.get_layer("enc_dense_1").get_weights()[0]  # (n_features, n_neurons)
	if X_train.ndim != 2:
		raise ValueError(f"X_train 2 boyutlu olmali, gelen shape: {X_train.shape}")

	if weights.shape[0] != X_train.shape[1]:
		raise ValueError(
			f"X_train feature boyutu ({X_train.shape[1]}) ile agirlik satir sayisi ({weights.shape[0]}) eslesmiyor."
		)

	# (n_samples, n_features, 1) * (1, n_features, n_neurons)
	# -> (n_samples, n_features, n_neurons)
	contributions = np.abs(X_train[:, :, np.newaxis] * weights[np.newaxis, :, :])
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


def unpack_processed_arrays(processed: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	X_train = processed["X_train_scaled"]
	X_test = processed["X_test_scaled"]
	y_train = processed["y_train"].to_numpy().astype(np.int32)
	y_test = processed["y_test"].to_numpy().astype(np.int32)
	return X_train, X_test, y_train, y_test


def train_and_evaluate_pipeline(
	X_train: np.ndarray,
	X_test: np.ndarray,
	y_train: np.ndarray,
	y_test: np.ndarray,
	encoding_dim: int,
	classifier_epochs: int,
	classifier_hidden_units: tuple[int, ...],
	classifier_dropout_rates: tuple[float, ...] | None,
	classifier_learning_rate: float,
) -> tuple[float, float, tf.keras.Model, tf.keras.Model]:
	autoencoder, encoder = build_relu_autoencoder(
		input_dim=X_train.shape[1],
		encoding_dim=encoding_dim,
	)

	autoencoder.fit(
		X_train,
		X_train,
		validation_data=(X_test, X_test),
		epochs=AUTOENCODER_EPOCHS,
		batch_size=BATCH_SIZE,
		shuffle=True,
		verbose=1,
	)

	test_mse = float(autoencoder.evaluate(X_test, X_test, verbose=0))

	X_train_encoded = encoder.predict(X_train, verbose=0)
	X_test_encoded = encoder.predict(X_test, verbose=0)

	classifier = build_latent_classifier(
		input_dim=X_train_encoded.shape[1],
		hidden_units=classifier_hidden_units,
		dropout_rates=classifier_dropout_rates,
		learning_rate=classifier_learning_rate,
	)
	y_train_fit = y_train.astype(np.float32)
	classifier.fit(
		X_train_encoded,
		y_train_fit,
		epochs=classifier_epochs,
		batch_size=BATCH_SIZE,
		validation_split=CLASSIFIER_VALIDATION_SPLIT,
		verbose=1,
	)

	y_pred_prob = classifier.predict(X_test_encoded, verbose=0)
	y_pred = (y_pred_prob > THRESHOLD).astype(int).ravel()
		
	test_accuracy = float(accuracy_score(y_test.astype(int), y_pred))
	print("classifier output shape:", classifier.output_shape)
	return test_mse, test_accuracy, autoencoder, encoder


def main(
	dataset_name: str = "breast_cancer_data.csv",
	target_column: str = "target",
	id_column: str | None = "ID",
	encoding_dim: int = 8,
	feature_percent: float = 50.0,
	random_state: int | None = RANDOM_STATE,
	classifier_epochs: int = DEFAULT_CLASSIFIER_EPOCHS,
	classifier_hidden_units: tuple[int, ...] = DEFAULT_CLASSIFIER_HIDDEN_UNITS,
	classifier_dropout_rates: tuple[float, ...] | None = None,
	classifier_learning_rate: float = 0.001,
) -> tuple[float, float]:
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

	processed = preprocess_data(df, target_column=target_column, id_column=id_column, random_state=random_state)
	X_train_raw = processed["X_train"]
	X_train, X_test, y_train, y_test = unpack_processed_arrays(processed)
	if not set(np.unique(y_train)).issubset({0, 1}) or not set(np.unique(y_test)).issubset({0, 1}):
		raise ValueError("Bu script binary etiket bekliyor. Label degerleri sadece 0 ve 1 olmali.")

	print(f"[INFO] X_train shape: {X_train.shape}")
	print(f"[INFO] X_test shape : {X_test.shape}")

	test_mse, test_accuracy, autoencoder, _ = train_and_evaluate_pipeline(
		X_train,
		X_test,
		y_train,
		y_test,
		encoding_dim,
		classifier_epochs,
		classifier_hidden_units,
		classifier_dropout_rates,
		classifier_learning_rate,
	)

	output_dir = Path("outputs") / "autoencoder" / dataset_folder
	metrics_dir = output_dir / "metrics"
	ensure_dir(output_dir)
	ensure_dir(metrics_dir)

	feature_names = X_train_raw.columns.tolist()
	weights_path = output_dir / "first_layer_W_list.csv"
	save_feature_weighted_lists(autoencoder, X_train, feature_names, weights_path)

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

	filtered_train_df = pd.read_csv(filtered_dataset_path)
	processed_filtered = preprocess_data(filtered_train_df, target_column=target_column, id_column=id_column, random_state=random_state)
	X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = unpack_processed_arrays(processed_filtered)
	filtered_test_mse, filtered_test_accuracy, _, _ = train_and_evaluate_pipeline(
		X_train_filtered,
		X_test_filtered,
		y_train_filtered,
		y_test_filtered,
		encoding_dim,
		classifier_epochs,
		classifier_hidden_units,
		classifier_dropout_rates,
		classifier_learning_rate,
	)

	save_json(
		{
			"test_mse": test_mse,
			"test_accuracy": test_accuracy,
			"threshold": THRESHOLD,
		},
		metrics_dir / "ORG_test_metrics.json",
	)

	save_json(
		{
			"feature_percent": feature_percent,
			"selected_feature_count": len(selected_df),
			"test_mse": filtered_test_mse,
			"test_accuracy": filtered_test_accuracy,
			"threshold": THRESHOLD,
		},
		metrics_dir / f"top_{feature_percent_tag}_test_metrics.json",
	)

	print("\n[OK] Autoencoder egitimi tamamlandi.")
	print(f"[OK] test_mse: {test_mse:.6f}")
	print(f"[OK] test_accuracy: {test_accuracy:.6f}")
	print(f"[OK] Feature weighted listeleri: {weights_path}")
	print(f"[OK] Top %{feature_percent} secilen feature sayisi: {len(selected_df)}")
	print(f"[OK] Secilen feature CSV: {selected_features_path}")
	print(f"[OK] Filterlenmis dataset CSV: {filtered_dataset_path} (satir: {len(filtered_df)})")
	print(f"[OK] Top %{feature_percent} dataset test_mse: {filtered_test_mse:.6f}")
	print(f"[OK] Top %{feature_percent} dataset test_accuracy: {filtered_test_accuracy:.6f}")
	filtered_metrics_path = metrics_dir / f"top_{feature_percent_tag}_test_metrics.json"
	print(f"[OK] Top %{feature_percent} metrik dosyasi: {filtered_metrics_path}")
	print(f"[OK] Output klasoru: {output_dir}")
	print(f"[OK] Metrik dosyasi: {metrics_dir / 'ORG_test_metrics.json'}")
	return test_accuracy, filtered_test_accuracy

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Basit autoencoder egitimi")
	parser.add_argument("--dataset-name", type=str, default="breast_cancer_data.csv", help="Raw data dosyasi (.csv veya .txt)")
	parser.add_argument("--target-column", type=str, default="target", help="Hedef kolon adi")
	parser.add_argument("--id-column", type=str, default="ID", help="ID kolon adi, kullanmak istemezsen 'none' ver")
	parser.add_argument("--encoding-dim", type=int, default=8, help="Encoding boyutu")
	parser.add_argument("--feature-percent", type=float, default=20.0, help="Secilecek feature yuzdesi")
	parser.add_argument("--random-state", type=str, default=str(RANDOM_STATE), help="Random state. Ornek: 42 veya none")
	parser.add_argument("--repeat-runs", type=int, default=1, help="Ayni deneyi kac kez calistiracagi")
	parser.add_argument("--accuracy-list-txt", type=str, default="", help="Accuracy listesi txt cikti yolu (bos ise varsayilan yol kullanilir)")
	parser.add_argument("--classifier-epochs", type=int, default=DEFAULT_CLASSIFIER_EPOCHS, help="Classifier epoch sayisi")
	parser.add_argument("--classifier-hidden-units", type=str, default="32,16", help="Classifier gizli katman nöronlari. Ornek: 128,64")
	parser.add_argument("--classifier-dropout-rates", type=str, default="", help="Classifier dropout oranlari. Ornek: 0.3,0.2")
	parser.add_argument("--classifier-learning-rate", type=float, default=0.001, help="Classifier ogrenme orani")


	args = parser.parse_args()
	random_state = parse_random_state(args.random_state)
	classifier_hidden_units = parse_hidden_units(args.classifier_hidden_units)
	classifier_dropout_rates = parse_dropout_rates(args.classifier_dropout_rates, len(classifier_hidden_units))
	if args.repeat_runs <= 0:
		raise ValueError("repeat-runs pozitif tam sayi olmali.")
	if args.classifier_epochs <= 0:
		raise ValueError("classifier-epochs pozitif tam sayi olmali.")
	if args.classifier_learning_rate <= 0:
		raise ValueError("classifier-learning-rate pozitif olmali.")

	accuracy_values: list[float] = []
	if args.accuracy_list_txt.strip():
		accuracy_txt_path = Path(args.accuracy_list_txt)
	else:
		dataset_folder = Path(args.dataset_name).stem
		feature_percent_tag = format_feature_percent_tag(args.feature_percent)
		accuracy_txt_path = Path("outputs") / "autoencoder" / dataset_folder / "metrics" / f"top_{feature_percent_tag}_accuracy_runs.txt"
	ensure_dir(accuracy_txt_path.parent)

	#50 kere çalıştırıyor.
	for run_idx in range(1, args.repeat_runs + 1):
		print(f"\n[INFO] Calisma {run_idx}/{args.repeat_runs} basladi.")
		_, filtered_test_accuracy = main(
			dataset_name=args.dataset_name,
			target_column=args.target_column,
			id_column=args.id_column,
			encoding_dim=args.encoding_dim,
			feature_percent=args.feature_percent,
			random_state=random_state,
			classifier_epochs=args.classifier_epochs,
			classifier_hidden_units=classifier_hidden_units,
			classifier_dropout_rates=classifier_dropout_rates,
			classifier_learning_rate=args.classifier_learning_rate,
		)
		accuracy_values.append(float(filtered_test_accuracy))
		accuracy_txt_path.write_text(str(accuracy_values), encoding="utf-8")

	print(f"[OK] Accuracy listesi yazildi: {accuracy_txt_path}")
	print(f"[OK] Accuracy dizisi: {accuracy_values}")
