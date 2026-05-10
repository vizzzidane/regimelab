"""
clustering.py – K-Means regime clustering on standardised SPY features.

Regime labels are assigned deterministically by sorting cluster centroids
on annualised volatility (ascending):
    lowest Vol_20d  → Calm
    middle Vol_20d  → Choppy
    highest Vol_20d → Stressed

This ordering is reproducible and does not rely on manual inspection.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from regimelab.config import K_FINAL, RANDOM_STATE, N_INIT
from regimelab.features import FEATURE_COLS

REGIME_ORDER = ["Calm", "Choppy", "Stressed"]


def compute_silhouette_scores(
    features_df: pd.DataFrame,
    k_values: list[int] | None = None,
) -> pd.DataFrame:
    """
    Standardise features and compute silhouette scores for each k.

    Parameters
    ----------
    features_df : pd.DataFrame
        Clean feature matrix (no NaNs).
    k_values : list of int, optional
        Values of k to evaluate.  Defaults to [2, 3, 4, 5].

    Returns
    -------
    pd.DataFrame
        Columns: k, silhouette_score.
    """
    if k_values is None:
        k_values = [2, 3, 4, 5]

    scaler = StandardScaler()
    X = scaler.fit_transform(features_df[FEATURE_COLS])

    records = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        records.append({"k": k, "silhouette_score": score})
        print(f"[clustering] k={k}  silhouette={score:.4f}")

    return pd.DataFrame(records)


def fit_regime_model(
    features_df: pd.DataFrame,
    k: int = K_FINAL,
) -> tuple[StandardScaler, KMeans, np.ndarray, pd.DataFrame]:
    """
    Standardise features and fit a K-Means model.

    Parameters
    ----------
    features_df : pd.DataFrame
        Clean feature matrix (no NaNs).
    k : int
        Number of clusters.

    Returns
    -------
    scaler : StandardScaler
        Fitted scaler (needed to inverse-transform centroids).
    kmeans : KMeans
        Fitted K-Means model.
    labels : np.ndarray
        Integer cluster labels aligned with features_df.index.
    centroids : pd.DataFrame
        Cluster centroids in original (unscaled) units.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df[FEATURE_COLS])

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = kmeans.fit_predict(X)

    centroids_scaled = kmeans.cluster_centers_
    centroids_orig = scaler.inverse_transform(centroids_scaled)
    centroids = pd.DataFrame(
        centroids_orig,
        columns=FEATURE_COLS,
        index=pd.RangeIndex(k, name="Cluster"),
    )

    print(f"[clustering] Fitted K-Means k={k}  "
          f"cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")
    return scaler, kmeans, labels, centroids


def assign_regime_labels(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    labels: np.ndarray,
    centroids: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach cluster IDs and regime labels to the main DataFrame.

    Labels are assigned by sorting cluster centroids on Vol_20d (ascending):
        rank 0 → Calm, rank 1 → Choppy, rank 2 → Stressed.

    Parameters
    ----------
    df : pd.DataFrame
        Main SPY DataFrame (will not be mutated).
    features_df : pd.DataFrame
        Feature matrix whose index aligns with *labels*.
    labels : np.ndarray
        Integer cluster labels from fit_regime_model().
    centroids : pd.DataFrame
        Centroids in original units from fit_regime_model().

    Returns
    -------
    pd.DataFrame
        Copy of *df* with 'Cluster' (int) and 'Regime' (str) columns added.
    """
    df = df.copy()

    # Map integer cluster id → regime name via volatility ordering
    sorted_by_vol = centroids["Vol_20d"].sort_values().index.tolist()
    label_map = {cluster_id: name for cluster_id, name in zip(sorted_by_vol, REGIME_ORDER)}

    # Attach cluster IDs
    df["Cluster"] = np.nan
    df.loc[features_df.index, "Cluster"] = labels

    # Attach regime names
    df["Regime"] = df["Cluster"].map(label_map)

    print("[clustering] Regime label assignment:")
    for cid, name in label_map.items():
        vol = centroids.loc[cid, "Vol_20d"]
        mom = centroids.loc[cid, "Mom_20d"]
        dd  = centroids.loc[cid, "Drawdown_60d"]
        print(f"  Cluster {cid} → {name:8s}  "
              f"Vol={vol:.4f}  Mom={mom:.4f}  DD={dd:.4f}")

    return df
