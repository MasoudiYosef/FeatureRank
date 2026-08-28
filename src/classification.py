"""Classification workflows for FeatureRank experiments."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.autoencoder_feature_selection import (
    load_selected_features_if_compatible,
    save_filtered_dataset_from_selected_features,
    save_sample_weighted_contributions,
    save_top_percent_features_by_abs_max_weight,
)
from src.config import (
    CHUNK_FEATURE_THRESHOLD,
    CLASSIFIER_CLASS_WEIGHT,
    CLASSIFIER_MODEL,
    CLASSIFIER_SAMPLING,
    CLASSIFIER_VALIDATION_SPLIT,
    ENABLE_FEATURE_CHUNKING,
    ExperimentConfig,
    FEATURE_CHUNK_SIZE,
    THRESHOLD,
)
from src.output_paths import (
    add_encoded_metric_metadata,
    classification_output_dir,
    format_encoded_output_percent,
    format_feature_output_label,
    format_feature_percent_tag,
    format_metric_output_prefix,
    format_test_metrics_filename,
    is_encoded_dataset_folder,
)
from src.preprocessing import preprocess_data, scale_data
from src.utils import compute_rmse_from_mse, ensure_dir, save_json
from src.experiment import (
    train_and_evaluate_direct_classifier,
    train_and_evaluate_pipeline,
    train_autoencoder_model,
)


def run_chunked_binary_experiment(
    df: pd.DataFrame,
    processed: dict,
    config: ExperimentConfig,
    dataset_folder: str,
    current_class_label: int | None = None,
    class_counts: dict[int, int] | None = None,
) -> tuple[float, float]:
    """Rank a large binary dataset chunk by chunk, then evaluate the merged set."""
    target_column = config.target_column
    id_column = config.id_column
    encoding_dim = config.encoding_dim
    feature_percent = config.feature_percent
    random_state = config.random_state
    classifier_model = CLASSIFIER_MODEL
    classifier_class_weight = CLASSIFIER_CLASS_WEIGHT
    classifier_sampling = CLASSIFIER_SAMPLING
    feature_chunk_size = FEATURE_CHUNK_SIZE
    save_training_plots = config.save_details
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
            history_output_dir=history_dir if save_training_plots else None,
            history_prefix=chunk_name if save_training_plots else None,
        )

        chunk_weights_path = chunk_dir / "first_layer_W_list.csv"
        save_sample_weighted_contributions(
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

    merged_dataset_path = (
        filtered_data_dir / f"chunked_top_{feature_percent_tag}_max_abs_features_dataset.csv"
    )
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
    config: ExperimentConfig,
    dataset_folder: str | None = None,
    current_class_label: int | None = None,
    class_counts: dict[int, int] | None = None,
) -> tuple[float, float]:
    """Run the standard binary FeatureRank flow and return original/filtered accuracy."""
    dataset_folder = dataset_folder or Path(config.dataset_name).stem
    target_column = config.target_column
    id_column = config.id_column
    encoding_dim = config.encoding_dim
    feature_percent = config.feature_percent
    random_state = config.random_state
    classifier_model = CLASSIFIER_MODEL
    feature_chunk_size = FEATURE_CHUNK_SIZE
    chunk_feature_threshold = CHUNK_FEATURE_THRESHOLD
    enable_feature_chunking = ENABLE_FEATURE_CHUNKING
    classifier_class_weight = CLASSIFIER_CLASS_WEIGHT
    classifier_sampling = CLASSIFIER_SAMPLING
    save_training_plots = config.save_details
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
            config=config,
            dataset_folder=dataset_folder,
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
        print(
            "[INFO] FeatureRank ve ikinci autoencoder uygulanmayacak; direkt reduced feature'lar ile classifier egitilecek."
        )
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
        print(
            f"[OK] {output_feature_label} metrik dosyasi: {metrics_dir / filtered_metrics_filename}"
        )
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
        history_output_dir=history_dir if save_training_plots else None,
        history_prefix="ORG" if save_training_plots else None,
    )

    feature_names = X_train_raw.columns.tolist()
    feature_percent_tag = format_feature_percent_tag(feature_percent)
    output_feature_percent = format_encoded_output_percent(feature_percent, dataset_folder)
    output_feature_prefix = format_metric_output_prefix(feature_percent, dataset_folder)
    output_feature_label = format_feature_output_label(feature_percent, dataset_folder)
    selected_features_path = output_dir / f"top_{feature_percent_tag}_max_abs_features.csv"
    # Feature ranking must come from the autoencoder trained in this run.
    # Reusing an old CSV can silently mix metrics from different seeds or code versions.
    weights_path = output_dir / "first_layer_W_list.csv"
    save_sample_weighted_contributions(autoencoder, X_train_sub_used, feature_names, weights_path)
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

    y_train_filtered = y_train
    y_test_filtered = y_test
    if len(selected_df) == len(feature_names):
        filtered_test_mse = test_mse
        filtered_test_accuracy = test_accuracy
        filtered_y_pred = org_y_pred
        filtered_y_score = org_y_score
        print(
            f"[INFO] {output_feature_label} tum secili/encoded feature'lari iceriyor. ORG sonucu yeniden egitilmeden kullaniliyor."
        )
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
            "label_1": label_1_count,
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
            "label_1": label_1_count_filtered,
        }

    save_json(
        filtered_metrics_data,
        metrics_dir / filtered_metrics_filename,
    )

    print("\n[OK] Autoencoder egitimi tamamlandi.")
    print(f"[OK] test_accuracy: {test_accuracy:.6f}")
    print(f"[OK] {output_feature_label} secilen feature sayisi: {len(selected_df)}")
    print(f"[OK] Secilen feature CSV: {selected_features_path}")
    print(f"[OK] {output_feature_label} dataset test_accuracy: {filtered_test_accuracy:.6f}")
    print(f"[OK] Calisma suresi: {elapsed_seconds:.2f} saniye")
    filtered_metrics_path = metrics_dir / filtered_metrics_filename
    print(f"[OK] {output_feature_label} metrik dosyasi: {filtered_metrics_path}")
    return test_accuracy, filtered_test_accuracy


def run_multiclass_one_vs_rest(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[float, float]:
    """Evaluate each class against the rest and aggregate the class metrics."""
    dataset_folder = Path(config.dataset_name).stem
    target_column = config.target_column
    feature_percent = config.feature_percent
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

        print(
            f"\n[INFO] One-vs-rest egitimi basliyor: class={class_label}, klasor={binary_dataset_folder}"
        )
        run_binary_experiment(
            df=binary_df,
            config=config,
            dataset_folder=nested_binary_folder,
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
    filtered_class_metrics_path = (
        metrics_dir / f"top_{output_feature_percent_tag}_multiclass_class_metrics.csv"
    )
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
        "class_metrics_path": (
            str(filtered_class_metrics_path) if filtered_class_metric_rows else None
        ),
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
        print(
            f"[OK] {output_feature_label} macro test_precision: {float(filtered_summary_metrics['test_precision']):.6f}"
        )
    if filtered_summary_metrics.get("test_recall") is not None:
        print(
            f"[OK] {output_feature_label} macro test_recall: {float(filtered_summary_metrics['test_recall']):.6f}"
        )
    if filtered_summary_metrics.get("test_f1") is not None:
        print(
            f"[OK] {output_feature_label} macro test_f1: {float(filtered_summary_metrics['test_f1']):.6f}"
        )
    print(f"[OK] Metrik dosyasi: {metrics_dir / 'ORG_test_metrics.json'}")
    if filtered_class_metric_rows:
        print(f"[OK] Sinif bazli multiclass metrik CSV: {filtered_class_metrics_path}")
    return macro_org_accuracy, macro_filtered_accuracy


def run_classification(df: pd.DataFrame, config: ExperimentConfig) -> tuple[float, float]:
    """Run binary or one-vs-rest classification from one readable config."""
    if int(df[config.target_column].nunique(dropna=True)) > 2:
        return run_multiclass_one_vs_rest(df, config)
    return run_binary_experiment(df, config)


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
    class_weight = {
        int(class_label): total / (len(classes) * float(count))
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
    before_counts = {int(label): int(count) for label, count in zip(classes, counts)}
    for class_label in classes:
        class_indices = np.where(y_train_int == int(class_label))[0]
        if len(class_indices) > target_count:
            class_indices = rng.choice(class_indices, size=target_count, replace=False)
        selected_indices.append(class_indices)
    balanced_indices = np.concatenate(selected_indices)
    rng.shuffle(balanced_indices)
    balanced_y = y_train[balanced_indices]
    after_classes, after_counts = np.unique(balanced_y.astype(int), return_counts=True)
    after_counts_dict = {
        int(label): int(count) for label, count in zip(after_classes, after_counts)
    }
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
            max_iter=5000, random_state=random_state, class_weight=class_weight
        )
    if classifier_model == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1,
        )
    raise ValueError("classifier-model 'neural', 'logistic' veya 'random_forest' olmali.")


def predict_binary_scores(classifier, X_test_encoded: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        y_score = classifier.predict_proba(X_test_encoded)[:, 1]
    elif hasattr(classifier, "decision_function"):
        decision_scores = classifier.decision_function(X_test_encoded)
        y_score = 1.0 / (1.0 + np.exp(-decision_scores))
    else:
        y_score = classifier.predict(X_test_encoded)
    return np.asarray(y_score, dtype=np.float32).ravel()


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


def ensure_shared_selected_features(processed: dict, config: ExperimentConfig) -> pd.DataFrame:
    """Load or create the FeatureRank list shared by classification and clustering."""
    dataset_folder = Path(config.dataset_name).stem
    feature_percent = config.feature_percent
    random_state = config.random_state
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

    chunked_selected_features_path = (
        output_dir / f"chunked_merged_top_{feature_percent_tag}_features.csv"
    )
    if chunked_selected_features_path.exists():
        selected_df = load_selected_features_if_compatible(
            chunked_selected_features_path, feature_names
        )
        if selected_df is not None:
            print(
                f"[INFO] Ortak chunked feature listesi kullaniliyor: {chunked_selected_features_path}"
            )
            return selected_df

    if should_use_feature_chunking(
        len(feature_names), CHUNK_FEATURE_THRESHOLD, FEATURE_CHUNK_SIZE, ENABLE_FEATURE_CHUNKING
    ):
        return generate_chunked_shared_selected_features(processed, config)

    print(
        "[INFO] Ortak feature listesi bulunamadi. Classification ile ayni train split mantigiyla uretiliyor."
    )
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
        encoding_dim=config.encoding_dim,
    )
    weights_path = output_dir / "first_layer_W_list.csv"
    save_sample_weighted_contributions(autoencoder, X_train_sub, feature_names, weights_path)
    return save_top_percent_features_by_abs_max_weight(
        weight_list_csv_path=weights_path,
        feature_names=feature_names,
        feature_percent=feature_percent,
        output_path=selected_features_path,
    )


def generate_chunked_shared_selected_features(
    processed: dict, config: ExperimentConfig
) -> pd.DataFrame:
    """Generate the shared FeatureRank list in chunks for very wide datasets."""
    dataset_folder = Path(config.dataset_name).stem
    feature_percent = config.feature_percent
    random_state = config.random_state
    X_train_raw = processed["X_train"]
    X_test_raw = processed["X_test"]
    y_train = processed["y_train"].to_numpy().astype(np.int32)
    feature_names = X_train_raw.columns.tolist()
    feature_chunks = split_feature_names_into_chunks(feature_names, FEATURE_CHUNK_SIZE)
    feature_percent_tag = format_feature_percent_tag(feature_percent)
    output_dir = classification_output_dir(dataset_folder)
    chunks_dir = output_dir / "chunks" / "shared_feature_ranking"
    ensure_dir(output_dir)
    ensure_dir(chunks_dir)
    print(
        f"[INFO] Ortak feature listesi chunked uretilecek: {len(feature_names)} feature, {len(feature_chunks)} parca (chunk_size={FEATURE_CHUNK_SIZE})."
    )

    chunk_selected_frames: list[pd.DataFrame] = []
    for chunk_idx, chunk_feature_names in enumerate(feature_chunks, start=1):
        chunk_name = f"chunk_{chunk_idx:03d}"
        chunk_dir = chunks_dir / chunk_name
        ensure_dir(chunk_dir)
        print(
            f"\n[INFO] Ortak {chunk_name}/{len(feature_chunks):03d} feature ranking basliyor (feature sayisi: {len(chunk_feature_names)})."
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
            encoding_dim=config.encoding_dim,
        )
        chunk_weights_path = chunk_dir / "first_layer_W_list.csv"
        save_sample_weighted_contributions(
            chunk_autoencoder, X_train_sub, chunk_feature_names, chunk_weights_path
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
        print(
            f"[OK] Ortak {chunk_name} tamamlandi. Top %{feature_percent}: {len(chunk_selected_df)} feature."
        )

    all_chunk_selected_df = pd.concat(chunk_selected_frames, ignore_index=True)
    all_chunk_selected_path = output_dir / f"chunked_top_{feature_percent_tag}_max_abs_features.csv"
    all_chunk_selected_df.to_csv(all_chunk_selected_path, index=False)
    merged_feature_names = list(dict.fromkeys(all_chunk_selected_df["feature_name"].tolist()))
    merged_selected_df = pd.DataFrame(
        {
            "feature": [f"F{i + 1}" for i in range(len(merged_feature_names))],
            "feature_name": merged_feature_names,
            "source": "chunked_top_features",
        }
    )
    merged_selected_path = output_dir / f"chunked_merged_top_{feature_percent_tag}_features.csv"
    merged_selected_df.to_csv(merged_selected_path, index=False)
    print(f"[OK] Ortak chunked feature listesi olusturuldu: {merged_selected_path}")
    print(f"[OK] Birlesen feature sayisi: {len(merged_selected_df)}")
    return merged_selected_df


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
            "selected_feature_count": metrics.get("selected_feature_count"),
            "accuracy": float(metrics["test_accuracy"]),
            "precision": (
                float(metrics["test_precision"])
                if metrics.get("test_precision") is not None
                else None
            ),
            "recall": (
                float(metrics["test_recall"]) if metrics.get("test_recall") is not None else None
            ),
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
        return float(
            sum(row[metric_name] * row["class_count"] for row in valid_rows) / total_weight
        )

    selected_counts = [
        int(row["selected_feature_count"])
        for row in metric_rows
        if row.get("selected_feature_count") is not None
    ]
    return {
        "selected_feature_count": selected_counts[0] if selected_counts else None,
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
