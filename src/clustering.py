"""Clustering workflows for FeatureRank experiments."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FormatStrFormatter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config import CLUSTER_MAX_K, CLUSTER_MIN_K, RANDOM_STATE, ExperimentConfig
from src.output_paths import clustering_output_dir, format_feature_percent_tag
from src.preprocessing import preprocess_data, scale_data
from src.utils import ensure_dir, save_json
from src.classification import ensure_shared_selected_features


_PCA_CLUSTER_COLORS = ("#0072B2", "#009E73", "#FED000", "#E41A1C", "#8A8A8A", "#A6761D", "#F781BF")


def build_cluster_colormap(cluster_count: int) -> ListedColormap:
    """Return a color map with one stable color for each cluster."""
    if cluster_count <= len(_PCA_CLUSTER_COLORS):
        return ListedColormap(
            _PCA_CLUSTER_COLORS[:cluster_count], name="feature_rank_cluster_dynamic"
        )
    return ListedColormap(
        plt.get_cmap("tab20", cluster_count)(np.arange(cluster_count)),
        name="feature_rank_cluster_dynamic",
    )


def normalize_cluster_k_range(min_k: int, max_k: int, sample_count: int) -> tuple[int, int]:
    """Validate the requested k range and cap it at the available samples."""
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
    """Fit KMeans for each k and return scores plus the selected labels."""
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
        if not np.isnan(silhouette) and (
            best_row is None or silhouette > best_row["silhouette_score"]
        ):
            best_row = row
            best_labels = labels

    if selected_k is not None:
        if selected_row is None or selected_labels is None:
            raise ValueError(
                f"Sabit cluster k={selected_k}, k araliginda bulunamadi: {min_k}-{max_k}"
            )
        if np.isnan(selected_row["silhouette_score"]):
            raise ValueError(
                f"Sabit cluster k={selected_k} icin gecerli silhouette skoru hesaplanamadi."
            )
        return pd.DataFrame(rows), selected_row, selected_labels

    if best_row is None or best_labels is None:
        raise ValueError(
            "Gecerli silhouette skoru hesaplanamadi. k araligini veya veri boyutunu kontrol edin."
        )

    return pd.DataFrame(rows), best_row, best_labels


def save_cluster_evaluation_plots(
    scores_df: pd.DataFrame,
    output_dir: Path,
    file_prefix: str,
    selected_k: int | None = None,
) -> None:
    """Save elbow, silhouette, and combined cluster evaluation plots."""
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
            selected_combined_path = (
                output_dir / f"k_{selected_k}_{file_prefix}_elbow_silhouette.png"
            )
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
    """Save a two-dimensional PCA view of the selected cluster labels."""
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


def _run_clustering_experiment(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[float, float]:
    dataset_folder = Path(config.dataset_name).stem
    target_column = config.target_column
    id_column = config.id_column
    feature_percent = config.feature_percent
    random_state = config.random_state
    cluster_k = config.cluster_k
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
    print(
        "[INFO] Label varsa clustering egitiminde kullanilmayacak; k degeri silhouette skoruna gore secilecek."
    )

    effective_min_k = CLUSTER_MIN_K
    effective_max_k = CLUSTER_MAX_K
    if cluster_k is not None:
        if cluster_k < 2:
            raise ValueError("--cluster-k en az 2 olmali.")
        if cluster_k >= X_raw.shape[0]:
            raise ValueError(
                f"--cluster-k degeri satir sayisindan kucuk olmali. Gelen k={cluster_k}, satir={X_raw.shape[0]}"
            )
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
        config=config,
    )

    feature_percent_tag = format_feature_percent_tag(feature_percent)
    selected_feature_names = selected_df["feature_name"].tolist()
    missing_features = [name for name in selected_feature_names if name not in feature_names]
    if missing_features:
        raise ValueError(
            f"Ortak feature listesinde veri setinde bulunmayan feature var: {missing_features}"
        )
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


def run_clustering(df: pd.DataFrame, config: ExperimentConfig) -> tuple[float, float]:
    """Run clustering using the compact, project-wide experiment config."""
    return _run_clustering_experiment(df, config)
