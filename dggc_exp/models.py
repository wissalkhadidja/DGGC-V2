from __future__ import annotations

from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn


class ScalarField(nn.Module):
    """NCE scalar field f_theta(x)."""

    def __init__(self, dim: int, hidden: Tuple[int, ...] = (128, 128), dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [nn.LayerNorm(dim)]
        prev = dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_gaussian_reference(X: np.ndarray, eps: float = 1e-4):
    mu = X.mean(axis=0).astype(np.float32)
    cov = np.atleast_2d(np.cov(X.T)).astype(np.float32)
    cov = cov + eps * np.eye(X.shape[1], dtype=np.float32)
    return mu, cov


def sample_gaussian_reference(mu: np.ndarray, cov: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mu, cov, size=n).astype(np.float32)
