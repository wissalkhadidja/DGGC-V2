from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def compute_projection_axes(
    X: np.ndarray,
    classes: Optional[np.ndarray] = None,
    clusters: Optional[np.ndarray] = None,
    methods=("pca", "lda_classes", "lda_clusters"),
) -> Dict[str, dict]:
    """
    Compute 2D projection coordinates and axes.

    For PCA, axes are principal directions in original D-space.
    For LDA, axes are discriminant scalings in original D-space.
    If a projection is impossible, it is skipped.
    """
    out: Dict[str, dict] = {}
    X = np.asarray(X, dtype=np.float32)

    if "pca" in methods:
        pca = PCA(n_components=2, random_state=42).fit(X)
        coords = pca.transform(X)
        out["pca"] = {
            "coords": coords.astype(np.float32),
            "axes": pca.components_.astype(np.float32),
            "origin": pca.mean_.astype(np.float32),
            "model": pca,
        }

    if "lda_classes" in methods and classes is not None:
        y = np.asarray(classes)
        if len(np.unique(y)) >= 2:
            try:
                n_components = min(2, len(np.unique(y)) - 1, X.shape[1])
                lda = LinearDiscriminantAnalysis(n_components=n_components).fit(X, y)
                coords = lda.transform(X)
                if coords.shape[1] == 1:
                    coords = np.c_[coords[:, 0], np.zeros(len(coords))]
                    axes = np.vstack([lda.scalings_[:, 0], np.zeros(X.shape[1])])
                else:
                    axes = lda.scalings_[:, :2].T
                out["lda_classes"] = {
                    "coords": coords.astype(np.float32),
                    "axes": axes.astype(np.float32),
                    "origin": X.mean(axis=0).astype(np.float32),
                    "model": lda,
                }
            except Exception:
                pass

    if "lda_clusters" in methods and clusters is not None:
        y = np.asarray(clusters)
        # noise is allowed as one class, but if all/no meaningful labels, skip
        if len(np.unique(y)) >= 2:
            try:
                n_components = min(2, len(np.unique(y)) - 1, X.shape[1])
                lda = LinearDiscriminantAnalysis(n_components=n_components).fit(X, y)
                coords = lda.transform(X)
                if coords.shape[1] == 1:
                    coords = np.c_[coords[:, 0], np.zeros(len(coords))]
                    axes = np.vstack([lda.scalings_[:, 0], np.zeros(X.shape[1])])
                else:
                    axes = lda.scalings_[:, :2].T
                out["lda_clusters"] = {
                    "coords": coords.astype(np.float32),
                    "axes": axes.astype(np.float32),
                    "origin": X.mean(axis=0).astype(np.float32),
                    "model": lda,
                }
            except Exception:
                pass

    return out


def grid_from_projection(proj: dict, grid_size: int = 150, padding: float = 0.08):
    coords = proj["coords"]
    axes = proj["axes"]
    origin = proj["origin"]
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    dx = x_max - x_min
    dy = y_max - y_min
    x_min -= padding * dx
    x_max += padding * dx
    y_min -= padding * dy
    y_max += padding * dy
    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    plane = np.c_[XX.ravel(), YY.ravel()]
    # Inverse plane map: origin + a*u + b*v
    X_grid = origin[None, :] + plane[:, [0]] * axes[0][None, :] + plane[:, [1]] * axes[1][None, :]
    return XX, YY, X_grid.astype(np.float32)
