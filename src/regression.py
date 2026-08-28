"""Regression workflows for FeatureRank experiments."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.autoencoder_feature_selection import (
    load_selected_features_if_compatible,
    save_filtered_dataset_from_selected_features,
    save_sample_weighted_contributions,
    save_top_percent_features_by_abs_max_weight,
)
from src.config import (
    ACTUAL_PREDICTED_TOP_N,
    CLASSIFIER_EARLY_STOPPING_MIN_DELTA,
    CLASSIFIER_EARLY_STOPPING_PATIENCE,
    CLASSIFIER_EPOCHS,
    ExperimentConfig,
    KMEANS_REGRESSION_CLUSTERS,
    KMEANS_REGRESSION_N_INIT,
    REGRESSION_MODEL,
    BATCH_SIZE,
    CLASSIFIER_VALIDATION_SPLIT,
)
from src.models import build_latent_regressor
from src.output_paths import format_feature_percent_tag, regression_output_dir
from src.preprocessing import preprocess_data, scale_data
from src.utils import ensure_dir, save_json
from src.experiment import (
    save_training_history,
    train_autoencoder_model,
)


def train_and_evaluate_regression_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: ExperimentConfig,
    history_output_dir: Path | None = None,
    history_prefix: str | None = None,
) -> tuple[
    dict, object, tf.keras.Model, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Train the ranking autoencoder and the configured latent regressor."""
    regression_model = REGRESSION_MODEL.lower().strip()
    if regression_model not in {"neural", "kmeans"}:
        raise ValueError("REGRESSION_MODEL 'neural' veya 'kmeans' olmali.")
    random_state = config.random_state

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=CLASSIFIER_VALIDATION_SPLIT,
        random_state=random_state,
        shuffle=True,
    )
    y_scaler = StandardScaler()
    y_train_scaled = (
        y_scaler.fit_transform(np.asarray(y_train_sub).reshape(-1, 1)).ravel().astype(np.float32)
    )
    y_val_scaled = y_scaler.transform(np.asarray(y_val).reshape(-1, 1)).ravel().astype(np.float32)

    autoencoder_mse, autoencoder, encoder = train_autoencoder_model(
        X_train_sub=X_train_sub,
        X_val=X_val,
        X_eval=X_test,
        encoding_dim=config.encoding_dim,
        history_output_dir=history_output_dir,
        history_prefix=history_prefix,
    )

    X_train_encoded = encoder.predict(X_train_sub, verbose=0).astype(np.float32)
    X_val_encoded = encoder.predict(X_val, verbose=0).astype(np.float32)
    X_test_encoded = encoder.predict(X_test, verbose=0).astype(np.float32)

    if regression_model == "kmeans":
        kmeans_regression_clusters = KMEANS_REGRESSION_CLUSTERS
        if kmeans_regression_clusters < 1:
            raise ValueError("kmeans-regression-clusters en az 1 olmali.")
        effective_cluster_count = min(kmeans_regression_clusters, X_train_encoded.shape[0])
        regressor = KMeans(
            n_clusters=effective_cluster_count,
            random_state=random_state,
            n_init=KMEANS_REGRESSION_N_INIT,
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
        )

        callbacks: list[tf.keras.callbacks.Callback] = []
        if CLASSIFIER_EARLY_STOPPING_PATIENCE > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=CLASSIFIER_EARLY_STOPPING_PATIENCE,
                    min_delta=CLASSIFIER_EARLY_STOPPING_MIN_DELTA,
                    restore_best_weights=True,
                    mode="min",
                    verbose=1,
                )
            )

        regressor_history = regressor.fit(
            X_train_encoded,
            y_train_scaled,
            epochs=CLASSIFIER_EPOCHS,
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
    if regression_model == "kmeans":
        metrics.update(
            {
                "kmeans_regression_clusters": int(kmeans_regression_clusters),
                "kmeans_regression_effective_clusters": int(
                    min(kmeans_regression_clusters, X_train_encoded.shape[0])
                ),
                "kmeans_regression_n_init": int(KMEANS_REGRESSION_N_INIT),
            }
        )
    # regression_r2 = model performansı / açıklama gücü
    # pearson_r     = gerçek-tahmin korelasyonu
    if regression_model == "neural":
        print("regressor output shape:", regressor.output_shape)
    else:
        print(
            "regressor model: "
            f"KMeansRegression(k={min(kmeans_regression_clusters, X_train_encoded.shape[0])}, "
            f"n_init={KMEANS_REGRESSION_N_INIT})"
        )
    return metrics, autoencoder, encoder, X_train_sub, y_true, y_pred, y_train_true, y_train_pred


def save_regression_actual_vs_predicted_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    file_prefix: str,
    top_n: int | None = None,
) -> None:
    """Save the normalized actual-versus-predicted diagnostic plot."""
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
    """Save per-sample regression errors and return the CSV path."""
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
            "squared_error": error**2,
        }
    )
    csv_path = output_dir / f"{file_prefix}_prediction_errors.csv"
    predictions_df.to_csv(csv_path, index=False)
    print(f"[OK] Regression prediction errors CSV: {csv_path}")
    return csv_path


def _run_regression_experiment(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[float, float]:
    """Run original and selected-feature regression evaluations."""
    dataset_folder = Path(config.dataset_name).stem
    target_column = config.target_column
    id_column = config.id_column
    feature_percent = config.feature_percent
    random_state = config.random_state
    save_training_plots = config.save_details
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

    (
        org_metrics,
        autoencoder,
        _,
        X_train_sub_used,
        org_y_true,
        org_y_pred,
        org_train_y_true,
        org_train_y_pred,
    ) = train_and_evaluate_regression_pipeline(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        config=config,
        history_output_dir=history_dir if save_training_plots else None,
        history_prefix="ORG" if save_training_plots else None,
    )
    save_regression_actual_vs_predicted_plot(
        y_true=org_y_true,
        y_pred=org_y_pred,
        output_dir=output_dir,
        file_prefix="ORG",
        top_n=ACTUAL_PREDICTED_TOP_N,
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
        save_sample_weighted_contributions(
            autoencoder, X_train_sub_used, feature_names, weights_path
        )
        selected_df = save_top_percent_features_by_abs_max_weight(
            weight_list_csv_path=weights_path,
            feature_names=feature_names,
            feature_percent=feature_percent,
            output_path=selected_features_path,
        )

    filtered_data_dir = Path("data") / "filtered" / dataset_folder
    ensure_dir(filtered_data_dir)
    filtered_dataset_path = (
        filtered_data_dir / f"top_{feature_percent_tag}_max_abs_features_dataset.csv"
    )
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
        print(
            "[INFO] Top %100 tum feature'lari iceriyor. ORG regression sonucu yeniden egitilmeden kullaniliyor."
        )
    else:
        selected_feature_names = selected_df["feature_name"].tolist()
        X_train_filtered_raw = X_train_raw[selected_feature_names]
        X_test_filtered_raw = X_test_raw[selected_feature_names]
        X_train_filtered, X_test_filtered, _ = scale_data(X_train_filtered_raw, X_test_filtered_raw)
        (
            filtered_metrics,
            _,
            _,
            _,
            filtered_y_true,
            filtered_y_pred,
            filtered_train_y_true,
            filtered_train_y_pred,
        ) = train_and_evaluate_regression_pipeline(
            X_train=X_train_filtered.astype(np.float32),
            X_test=X_test_filtered.astype(np.float32),
            y_train=y_train,
            y_test=y_test,
            config=config,
            history_output_dir=history_dir if save_training_plots else None,
            history_prefix=f"top_{feature_percent_tag}" if save_training_plots else None,
        )
    save_regression_actual_vs_predicted_plot(
        y_true=filtered_y_true,
        y_pred=filtered_y_pred,
        output_dir=output_dir,
        file_prefix=f"top_{feature_percent_tag}",
        top_n=ACTUAL_PREDICTED_TOP_N,
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
        filtered_metrics_data["train_prediction_errors_path"] = str(
            filtered_train_prediction_errors_path
        )
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


def run_regression(df: pd.DataFrame, config: ExperimentConfig) -> tuple[float, float]:
    """Run regression using fixed model settings from :mod:`src.config`."""
    return _run_regression_experiment(df, config)
