from __future__ import annotations

from typing import Dict
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    silhouette_score,
)


def clustering_accuracy(y_true: np.ndarray, labels: np.ndarray) -> float:
    """Hungarian accuracy. Noise label -1 is treated as a normal predicted cluster."""
    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    true_labs = np.unique(y_true)
    pred_labs = np.unique(labels)
    W = np.zeros((len(pred_labs), len(true_labs)), dtype=np.int64)
    for i, p in enumerate(pred_labs):
        for j, t in enumerate(true_labs):
            W[i, j] = np.sum((labels == p) & (y_true == t))
    row_ind, col_ind = linear_sum_assignment(W.max() - W)
    return float(W[row_ind, col_ind].sum() / len(y_true))


def purity_score(y_true: np.ndarray, labels: np.ndarray) -> float:
    """Purity. Noise label -1 is treated as a normal predicted cluster."""
    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    total = 0
    for c in np.unique(labels):
        idx = labels == c
        _, counts = np.unique(y_true[idx], return_counts=True)
        total += counts.max()
    return float(total / len(y_true))


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(X) < 3:
        return float("nan")
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")


def safe_dbcv(X: np.ndarray, labels: np.ndarray) -> float:
    """DBCV using hdbscan.validity.validity_index when available."""
    try:
        from hdbscan.validity import validity_index
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(validity_index(np.asarray(X, dtype=np.float64), np.asarray(labels)))
    except Exception:
        return float("nan")


def safe_cdbw(X: np.ndarray, labels: np.ndarray) -> float:
    """True CDbw when the optional `cdbw` package is installed.

    Noise label -1 is passed to the package and handled with alg_noise="comb".
    If the package is unavailable or fails, returns NaN.
    """
    try:
        from cdbw import CDbw
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(CDbw(
            np.asarray(X, dtype=np.float64),
            np.asarray(labels),
            metric="euclidean",
            alg_noise="comb",
            intra_dens_inf=False,
            s=3,
            multipliers=False,
        ))
    except Exception:
        return float("nan")


def supervised_metrics_all_points(y_true: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """All supervised metrics treat noise (-1) as a predicted cluster."""
    return {
        "ari": float(adjusted_rand_score(y_true, labels)),
        "nmi": float(normalized_mutual_info_score(y_true, labels)),
        "ami": float(adjusted_mutual_info_score(y_true, labels)),
        "acc": clustering_accuracy(y_true, labels),
        "purity": purity_score(y_true, labels),
        "ari_nmi": float((adjusted_rand_score(y_true, labels) + normalized_mutual_info_score(y_true, labels)) / 2.0),
    }
