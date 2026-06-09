from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs


def embed_to_dim(X2: np.ndarray, dim: int, seed: int) -> np.ndarray:
    if dim == X2.shape[1]:
        return X2.astype(np.float32)
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(X2.shape[1], dim)).astype(np.float32)
    A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-8
    X = X2 @ A
    noise = 0.01 * rng.normal(size=X.shape)
    return (X + noise).astype(np.float32)


def make_gaussian_blobs(n: int, dim: int, seed: int):
    X2, y = make_blobs(n_samples=n, centers=[[-5, 0], [0, 5], [5, 0]], cluster_std=[0.75, 0.85, 0.75], random_state=seed)
    return embed_to_dim(X2, dim, seed), y.astype(int)


def make_density_bridge(n: int, dim: int, seed: int):
    rng = np.random.default_rng(seed)
    n1, n2 = n // 3, n // 3
    nb = n - n1 - n2
    c1 = rng.normal(loc=[-4, 0], scale=[0.7, 0.7], size=(n1, 2))
    c2 = rng.normal(loc=[4, 0], scale=[0.7, 0.7], size=(n2, 2))
    t = rng.uniform(-3.2, 3.2, size=(nb, 1))
    bridge = np.hstack([t, rng.normal(0, 0.22, size=(nb, 1))])
    X2 = np.vstack([c1, c2, bridge])
    y = np.array([0] * n1 + [1] * n2 + [2] * nb)
    return embed_to_dim(X2, dim, seed), y.astype(int)


def make_overlapping_density_blobs(n: int, dim: int, seed: int):
    X2, y = make_blobs(n_samples=n, centers=[[-2, 0], [0, 0.8], [2, 0]], cluster_std=[1.2, 1.3, 1.2], random_state=seed)
    return embed_to_dim(X2, dim, seed), y.astype(int)


def make_imbalanced_anisotropic(n: int, dim: int, seed: int):
    rng = np.random.default_rng(seed)
    sizes = [int(0.60 * n), int(0.25 * n)]
    sizes.append(n - sum(sizes))
    means = np.array([[-4, 0], [1, 2], [4, -1]])
    covs = [np.array([[3.0, 1.2], [1.2, 0.5]]), np.array([[0.3, 0.0], [0.0, 2.0]]), np.array([[1.0, -0.8], [-0.8, 1.2]])]
    Xs, ys = [], []
    for i, (s, m, c) in enumerate(zip(sizes, means, covs)):
        Xs.append(rng.multivariate_normal(m, c, size=s))
        ys.extend([i] * s)
    return embed_to_dim(np.vstack(Xs), dim, seed), np.array(ys, dtype=int)


def make_hubness_stress(n: int, dim: int, seed: int):
    X, y = make_gaussian_blobs(n, dim, seed)
    rng = np.random.default_rng(seed)
    hub = rng.normal(0, 0.15, size=(max(20, n // 100), dim)).astype(np.float32)
    labels = rng.integers(0, 3, size=len(hub))
    return np.vstack([X, hub]).astype(np.float32), np.concatenate([y, labels]).astype(int)


GENERATORS = {
    "gaussian_blobs": make_gaussian_blobs,
    "density_bridge": make_density_bridge,
    "overlapping_density_blobs": make_overlapping_density_blobs,
    "imbalanced_anisotropic": make_imbalanced_anisotropic,
    "hubness_stress": make_hubness_stress,
}
