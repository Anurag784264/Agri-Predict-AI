"""
K-Means clustering and unsupervised pattern analysis pipeline for AgriPredict AI.
Implements feature scaling, cluster validation (Silhouette, Davies-Bouldin),
Elbow inertia curve generation, and 2D PCA projection.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


def compute_elbow_curve(df: pd.DataFrame, feature_cols: List[str], max_k: int = 8) -> pd.DataFrame:
    """
    Computes inertia values across k=2..max_k for the Elbow Method.
    """
    if df.empty or len(feature_cols) < 2 or len(df) < max_k:
        return pd.DataFrame()

    X = df[feature_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    k_range = range(2, min(max_k + 1, len(X)))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append({"k": k, "Inertia": round(float(kmeans.inertia_), 2)})

    return pd.DataFrame(inertias)


def perform_kmeans_clustering(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_clusters: int = 3
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes K-Means clustering on the specified numerical features with full diagnostics.
    """
    if df.empty or len(feature_cols) < 2:
        return pd.DataFrame(), {
            "status": "error",
            "message": "At least 2 numerical features are required for clustering."
        }

    valid_data = df.dropna(subset=feature_cols).copy()
    if len(valid_data) <= n_clusters:
        return pd.DataFrame(), {
            "status": "error",
            "message": f"Sample size ({len(valid_data)}) must exceed number of clusters ({n_clusters})."
        }

    X = valid_data[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    valid_data["Cluster"] = [f"Cluster {i+1}" for i in labels]

    # Cluster quality metrics
    sil_score = None
    db_score = None
    try:
        sil_score = round(float(silhouette_score(X_scaled, labels)), 3)
        db_score = round(float(davies_bouldin_score(X_scaled, labels)), 3)
    except Exception:
        pass

    # 2D PCA Projection for visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    valid_data["PCA_Dim1"] = coords[:, 0].round(3)
    valid_data["PCA_Dim2"] = coords[:, 1].round(3)
    explained_var = [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]

    # Cluster Profile Summary
    summary = valid_data.groupby("Cluster")[feature_cols].mean().round(2).reset_index()
    counts = valid_data["Cluster"].value_counts().reset_index()
    counts.columns = ["Cluster", "Member_Count"]
    summary = pd.merge(counts, summary, on="Cluster")
    summary["Percentage"] = ((summary["Member_Count"] / len(valid_data)) * 100).round(1)

    diagnostics = {
        "status": "success",
        "n_clusters": n_clusters,
        "sample_count": len(valid_data),
        "silhouette_score": sil_score,
        "davies_bouldin_score": db_score,
        "pca_variance_explained": explained_var,
        "cluster_summary": summary
    }

    return valid_data, diagnostics
