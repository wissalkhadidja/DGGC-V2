from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, normalize

from .metrics import (
    supervised_metrics_all_points,
    safe_silhouette,
    safe_dbcv,
    safe_cdbw,
)
from .models import ScalarField, fit_gaussian_reference, sample_gaussian_reference
from .projections import compute_projection_axes, grid_from_projection


def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ExperienceConfig:
    name: str = "dggc_experience"
    seed: int = 42
    standardize_initial: bool = True
    hidden: Tuple[int, ...] = (128, 128)
    dropout: float = 0.0
    epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    reference: str = "gaussian_full"
    flow_steps: int = 100
    flow_step_size: float = 0.02
    normalize_gradient: bool = True
    renormalize_each_step: bool = False
    save_every: int = 1
    device: str = "cuda"
    hdbscan_min_cluster_sizes: Tuple[int, ...] = (5, 10, 20, 30, 50, 80, 120, 200)
    hdbscan_min_samples_values: Tuple[int, ...] = (1, 3, 5, 10, 15, 20, 30, 50)
    # Full HDBSCAN search space. The best configuration is selected across all
    # combinations below, not only min_cluster_size/min_samples.
    hdbscan_metrics: Tuple[str, ...] = (
        "euclidean", "manhattan", "chebyshev", "minkowski", "cosine"
    )
    hdbscan_cluster_selection_methods: Tuple[str, ...] = ("eom", "leaf")
    hdbscan_alpha_values: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    hdbscan_minkowski_p_values: Tuple[float, ...] = (1.5, 2.0, 3.0)
    hdbscan_cluster_selection_epsilon_values: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.10)
    hdbscan_allow_single_cluster_values: Tuple[bool, ...] = (False, True)
    hdbscan_leaf_size_values: Tuple[int, ...] = (20, 40)
    # Pure selection: the best HDBSCAN configuration is the one with the
    # largest HDBSCAN relative_validity_ computed on the final DGGC embedding.
    # No penalty, no supervised metric, no hand-tuned correction is used here.
    hdbscan_selection_criterion: str = "relative_validity"
    projection_methods: Tuple[str, ...] = ("pca", "lda_classes", "lda_clusters")
    grid_size_density: int = 150


@dataclass
class Experience:
    """
    Complete DGGC experiment object.

    Attributes
    ----------
    embeddings:
        N x D x K tensor, where K = saved DGGC trajectory steps.
        embeddings[:, :, 0] is the frozen/initial embedding.
    classes:
        N labels for ground-truth classes.
    density_model:
        PyTorch NCE scalar field after training.
    clusters:
        N HDBSCAN labels selected by relative_validity_ on final embedding.
    quality:
        Dictionary of timings, supervised metrics, internal metrics, noise and grid info.
    """

    X_initial: np.ndarray
    classes: np.ndarray
    config: ExperienceConfig = field(default_factory=ExperienceConfig)
    embeddings: Optional[np.ndarray] = None
    density_model: Optional[ScalarField] = None
    density_model_state: Optional[dict] = None
    clusters: Optional[np.ndarray] = None
    quality: Dict[str, float] = field(default_factory=dict)
    best_hdbscan_grid: Dict[str, float] = field(default_factory=dict)
    hdbscan_grid_results: List[dict] = field(default_factory=list)
    projections: Dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        seed_everything(self.config.seed)
        X = np.asarray(self.X_initial, dtype=np.float32)
        if self.config.standardize_initial:
            X = StandardScaler().fit_transform(X).astype(np.float32)
        self.classes = np.asarray(self.classes).astype(int)
        self.embeddings = X[:, :, None].astype(np.float32)
        if self.config.device == "cuda" and not torch.cuda.is_available():
            self.config.device = "cpu"

    @property
    def X0(self) -> np.ndarray:
        return self.embeddings[:, :, 0]

    @property
    def X_final(self) -> np.ndarray:
        return self.embeddings[:, :, -1]

    def _make_reference(self, X: np.ndarray) -> np.ndarray:
        if self.config.reference == "gaussian_full":
            mu, cov = fit_gaussian_reference(X)
            return sample_gaussian_reference(mu, cov, len(X), self.config.seed + 123)
        if self.config.reference == "gaussian_diag":
            mu = X.mean(axis=0).astype(np.float32)
            std = X.std(axis=0).astype(np.float32) + 1e-6
            rng = np.random.default_rng(self.config.seed + 123)
            return rng.normal(mu, std, size=X.shape).astype(np.float32)
        if self.config.reference == "uniform_box":
            rng = np.random.default_rng(self.config.seed + 123)
            lo = X.min(axis=0)
            hi = X.max(axis=0)
            return rng.uniform(lo, hi, size=X.shape).astype(np.float32)
        if self.config.reference == "sphere":
            rng = np.random.default_rng(self.config.seed + 123)
            Z = rng.normal(size=X.shape).astype(np.float32)
            Z = normalize(Z).astype(np.float32)
            radius = np.median(np.linalg.norm(X - X.mean(axis=0), axis=1))
            return (Z * radius + X.mean(axis=0)).astype(np.float32)
        raise ValueError(f"Unknown reference: {self.config.reference}")

    def train_NCE(self):
        """Train the NCE scalar field on the initial embedding and record time."""
        t0 = time.time()
        X = self.X0.astype(np.float32)
        X_ref = self._make_reference(X)
        model = ScalarField(dim=X.shape[1], hidden=self.config.hidden, dropout=self.config.dropout).to(self.config.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        bce = nn.BCEWithLogitsLoss()

        X_all = np.vstack([X, X_ref]).astype(np.float32)
        y_all = np.concatenate([np.ones(len(X)), np.zeros(len(X_ref))]).astype(np.float32)[:, None]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(X_all), torch.tensor(y_all)),
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        model.train()
        for _ in range(self.config.epochs):
            for xb, yb in loader:
                xb = xb.to(self.config.device)
                yb = yb.to(self.config.device)
                opt.zero_grad(set_to_none=True)
                loss = bce(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

        self.density_model = model.eval()
        self.density_model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        self.quality["time_train_NCE_sec"] = time.time() - t0
        return self

    def update_embeddings(self):
        """Run DGGC density ascent and store the embedding trajectory N x D x K."""
        if self.density_model is None:
            raise RuntimeError("Call train_NCE() before update_embeddings().")
        t0 = time.time()
        x = torch.tensor(self.X0, dtype=torch.float32, device=self.config.device)
        saved = [self.X0.astype(np.float32)]

        for step in range(1, self.config.flow_steps + 1):
            x = x.detach().requires_grad_(True)
            score = self.density_model(x).sum()
            grad = torch.autograd.grad(score, x, create_graph=False)[0]
            if self.config.normalize_gradient:
                grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-6)
            x = x + self.config.flow_step_size * grad
            if self.config.renormalize_each_step:
                x = torch.nn.functional.normalize(x, dim=1)
            if step % self.config.save_every == 0 or step == self.config.flow_steps:
                saved.append(x.detach().cpu().numpy().astype(np.float32))

        self.embeddings = np.stack(saved, axis=2).astype(np.float32)
        self.quality["time_update_embeddings_sec"] = time.time() - t0
        self.quality["saved_embedding_steps"] = int(self.embeddings.shape[2])
        return self

    def _hdbscan_algorithm_for_metric(self, metric: str) -> str:
        """Choose a safe HDBSCAN backend for each metric.

        HDBSCAN's fast tree backends do not support every distance. For metrics
        such as cosine/correlation, the generic algorithm is safer although
        slower. Euclidean-like metrics keep algorithm="best".
        """
        generic_metrics = {"cosine", "correlation", "precomputed", "haversine"}
        return "generic" if metric in generic_metrics else "best"

    def _hdbscan_selection_score(self, relative_validity: float, noise: float, n_clusters: int) -> float:
        """Pure HDBSCAN selection score: relative_validity_ only."""
        if not np.isfinite(relative_validity):
            return -np.inf
        return float(relative_validity)

    def clustering(self):
        """Optimize HDBSCAN on final embedding over the full HDBSCAN grid.

        The grid includes:
        - min_cluster_size
        - min_samples
        - metric
        - cluster_selection_method
        - alpha

        By default, selection maximizes HDBSCAN relative_validity_ on the final
        DGGC embedding. The complete grid is saved in hdbscan_grid_results.
        """
        t0 = time.time()
        X = self.X_final
        best = None
        rows = []

        for metric in self.config.hdbscan_metrics:
            algorithm = self._hdbscan_algorithm_for_metric(metric)
            p_values = self.config.hdbscan_minkowski_p_values if metric == "minkowski" else (None,)
            for p_value in p_values:
                for method in self.config.hdbscan_cluster_selection_methods:
                    for alpha in self.config.hdbscan_alpha_values:
                        for eps in self.config.hdbscan_cluster_selection_epsilon_values:
                            for allow_single in self.config.hdbscan_allow_single_cluster_values:
                                for leaf_size in self.config.hdbscan_leaf_size_values:
                                    for mcs in self.config.hdbscan_min_cluster_sizes:
                                        for ms in self.config.hdbscan_min_samples_values:
                                            base_row: Dict[str, Any] = {
                                                "metric": str(metric),
                                                "algorithm": str(algorithm),
                                                "cluster_selection_method": str(method),
                                                "alpha": float(alpha),
                                                "cluster_selection_epsilon": float(eps),
                                                "allow_single_cluster": bool(allow_single),
                                                "leaf_size": int(leaf_size),
                                                "minkowski_p": None if p_value is None else float(p_value),
                                                "min_cluster_size": int(mcs),
                                                "min_samples": int(ms),
                                            }
                                            try:
                                                kwargs = dict(
                                                    min_cluster_size=int(mcs),
                                                    min_samples=int(ms),
                                                    metric=str(metric),
                                                    algorithm=str(algorithm),
                                                    cluster_selection_method=str(method),
                                                    alpha=float(alpha),
                                                    cluster_selection_epsilon=float(eps),
                                                    allow_single_cluster=bool(allow_single),
                                                    leaf_size=int(leaf_size),
                                                    gen_min_span_tree=True,
                                                    prediction_data=False,
                                                )
                                                if p_value is not None:
                                                    kwargs["p"] = float(p_value)
                                                clusterer = hdbscan.HDBSCAN(**kwargs).fit(X)
                                                labels = clusterer.labels_.astype(int)
                                                rel = float(getattr(clusterer, "relative_validity_", np.nan))
                                                if not np.isfinite(rel):
                                                    rel = -np.inf
                                                noise = float(np.mean(labels == -1))
                                                n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
                                                selection_score = self._hdbscan_selection_score(rel, noise, n_clusters)
                                                row = {
                                                    **base_row,
                                                    "relative_validity": rel,
                                                    "selection_score": selection_score,
                                                    "noise": noise,
                                                    "coverage": 1.0 - noise,
                                                    "n_clusters": n_clusters,
                                                }
                                                rows.append(row)

                                                # Pure criterion: relative_validity_; tie-breaks prefer
                                                # coverage then more non-noise clusters.
                                                score_tuple = (selection_score, 1.0 - noise, n_clusters)
                                                best_tuple = (-np.inf, -np.inf, -np.inf) if best is None else best["score_tuple"]
                                                if score_tuple > best_tuple:
                                                    best = {
                                                        "labels": labels,
                                                        "clusterer": clusterer,
                                                        "row": row,
                                                        "score_tuple": score_tuple,
                                                    }
                                            except Exception as e:
                                                rows.append({
                                                    **base_row,
                                                    "relative_validity": float("nan"),
                                                    "selection_score": float("nan"),
                                                    "error": str(e),
                                                })
        if best is None:
            raise RuntimeError("All HDBSCAN grid configurations failed.")
        self.clusters = best["labels"]
        self.best_hdbscan_grid = best["row"]
        self.hdbscan_grid_results = rows
        self.quality["time_clustering_sec"] = time.time() - t0
        self.quality["hdbscan_grid_size"] = int(len(rows))
        self.quality["hdbscan_selection_criterion"] = self.config.hdbscan_selection_criterion
        self.quality.update({f"hdbscan_best_{k}": v for k, v in self.best_hdbscan_grid.items()})
        return self

    def compute_quality(self):
        """Compute supervised metrics and internal validity indices."""
        if self.clusters is None:
            raise RuntimeError("Call clustering() before compute_quality().")
        y = self.classes
        labels = self.clusters
        q = {}
        q.update(supervised_metrics_all_points(y, labels))
        q["noise_percent"] = float(100.0 * np.mean(labels == -1))
        q["coverage_percent"] = float(100.0 * np.mean(labels != -1))
        q["n_clusters_non_noise"] = int(len(set(labels)) - (1 if -1 in labels else 0))

        # Internal indices in both initial and final spaces.
        for name, X in [("initial", self.X0), ("final", self.X_final)]:
            q[f"silhouette_{name}"] = safe_silhouette(X, labels)
            q[f"dbcv_{name}"] = safe_dbcv(X, labels)
            q[f"cdbw_{name}"] = safe_cdbw(X, labels)

        self.quality.update(q)
        return self

    def projection_2D(self, embedding: str = "final", methods: Optional[Tuple[str, ...]] = None):
        """Compute PCA, LDA classes and LDA clusters projections for initial/final embedding."""
        if methods is None:
            methods = self.config.projection_methods
        X = self.X_final if embedding == "final" else self.X0
        self.projections[embedding] = compute_projection_axes(
            X,
            classes=self.classes,
            clusters=self.clusters,
            methods=methods,
        )
        return self.projections[embedding]

    def visu_embeddings(self, out_dir: str | Path, embedding: str = "final", color_by: str = "classes"):
        """Plot 2D embeddings using computed projections."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        projs = self.projection_2D(embedding)
        labels = self.classes if color_by == "classes" or self.clusters is None else self.clusters
        for pname, p in projs.items():
            coords = p["coords"]
            plt.figure(figsize=(7, 6))
            plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, cmap="tab20", alpha=0.85)
            plt.title(f"{self.config.name} | {embedding} | {pname} | color={color_by}")
            plt.xlabel("axis 1")
            plt.ylabel("axis 2")
            plt.tight_layout()
            plt.savefig(out_dir / f"embeddings_{embedding}_{pname}_{color_by}.png", dpi=180)
            plt.close()
        return self

    def visu_density(self, out_dir: str | Path, embedding: str = "final", projection: str = "pca"):
        """Visualize NCE scalar field on a 2D projected plane."""
        if self.density_model is None:
            raise RuntimeError("Need trained density_model.")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        projs = self.projection_2D(embedding)
        if projection not in projs:
            return self
        XX, YY, X_grid = grid_from_projection(projs[projection], grid_size=self.config.grid_size_density)
        self.density_model.eval()
        vals = []
        with torch.no_grad():
            for i in range(0, len(X_grid), 4096):
                xb = torch.tensor(X_grid[i:i+4096], dtype=torch.float32, device=self.config.device)
                vals.append(self.density_model(xb).detach().cpu().numpy().ravel())
        Z = np.concatenate(vals).reshape(XX.shape)
        coords = projs[projection]["coords"]
        plt.figure(figsize=(7, 6))
        plt.contourf(XX, YY, Z, levels=40)
        plt.scatter(coords[:, 0], coords[:, 1], c=self.classes, s=5, cmap="tab20", alpha=0.75)
        plt.title(f"NCE density field | {embedding} | {projection}")
        plt.tight_layout()
        plt.savefig(out_dir / f"density_{embedding}_{projection}.png", dpi=180)
        plt.close()
        return self

    def save_to_file(self, path: str | Path):
        """Save complete instance and individual artifacts."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        np.save(path / "classes.npy", self.classes)
        if self.clusters is not None:
            np.save(path / "clusters.npy", self.clusters)
        with open(path / "quality.json", "w") as f:
            json.dump(self.quality, f, indent=2)
        with open(path / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        with open(path / "best_hdbscan_grid.json", "w") as f:
            json.dump(self.best_hdbscan_grid, f, indent=2)
        with open(path / "hdbscan_grid_results.json", "w") as f:
            json.dump(self.hdbscan_grid_results, f, indent=2)
        if self.density_model is not None:
            torch.save({
                "state_dict": self.density_model_state,
                "config": asdict(self.config),
                "dim": int(self.X0.shape[1]),
            }, path / "density_model.pt")
        with open(path / "experience.pkl", "wb") as f:
            pickle.dump(self, f)
        return self

    @staticmethod
    def load_from_file(path: str | Path):
        with open(Path(path) / "experience.pkl", "rb") as f:
            return pickle.load(f)
