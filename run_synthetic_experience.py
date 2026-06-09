from __future__ import annotations

import argparse
from pathlib import Path

from dggc_exp import Experience, ExperienceConfig, GENERATORS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experience_out")
    parser.add_argument("--dataset", default="gaussian_blobs", choices=list(GENERATORS.keys()))
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--flow_steps", type=int, default=100)
    parser.add_argument("--flow_step_size", type=float, default=0.02)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--min_cluster_sizes", nargs="+", type=int, default=[5, 10, 20, 30, 50, 80, 120, 200])
    parser.add_argument("--min_samples_values", nargs="+", type=int, default=[1, 3, 5, 10, 15, 20, 30, 50])
    parser.add_argument("--hdbscan_metrics", nargs="+", default=["euclidean", "manhattan", "chebyshev", "minkowski", "cosine"])
    parser.add_argument("--hdbscan_cluster_selection_methods", nargs="+", default=["eom", "leaf"])
    parser.add_argument("--hdbscan_alpha_values", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--hdbscan_minkowski_p_values", nargs="+", type=float, default=[1.5, 2.0, 3.0])
    parser.add_argument("--hdbscan_cluster_selection_epsilon_values", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.10])
    parser.add_argument("--hdbscan_allow_single_cluster_values", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--hdbscan_leaf_size_values", nargs="+", type=int, default=[20, 40])
    parser.add_argument("--hdbscan_selection_criterion", default="relative_validity", choices=["relative_validity"])
    args = parser.parse_args()

    X, y = GENERATORS[args.dataset](args.n_samples, args.dim, args.seed)
    cfg = ExperienceConfig(
        name=f"{args.dataset}_d{args.dim}_seed{args.seed}",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        flow_steps=args.flow_steps,
        flow_step_size=args.flow_step_size,
        save_every=args.save_every,
        device=args.device,
        hdbscan_min_cluster_sizes=tuple(args.min_cluster_sizes),
        hdbscan_min_samples_values=tuple(args.min_samples_values),
        hdbscan_metrics=tuple(args.hdbscan_metrics),
        hdbscan_cluster_selection_methods=tuple(args.hdbscan_cluster_selection_methods),
        hdbscan_alpha_values=tuple(args.hdbscan_alpha_values),
        hdbscan_minkowski_p_values=tuple(args.hdbscan_minkowski_p_values),
        hdbscan_cluster_selection_epsilon_values=tuple(args.hdbscan_cluster_selection_epsilon_values),
        hdbscan_allow_single_cluster_values=tuple(bool(v) for v in args.hdbscan_allow_single_cluster_values),
        hdbscan_leaf_size_values=tuple(args.hdbscan_leaf_size_values),
        hdbscan_selection_criterion=args.hdbscan_selection_criterion,
    )

    exp = Experience(X, y, cfg)
    exp.train_NCE()
    exp.update_embeddings()
    exp.clustering()
    exp.compute_quality()

    out_dir = Path(args.out)
    exp.save_to_file(out_dir)

    if args.figures:
        fig_dir = out_dir / "figures"
        exp.visu_embeddings(fig_dir, embedding="initial", color_by="classes")
        exp.visu_embeddings(fig_dir, embedding="final", color_by="classes")
        exp.visu_embeddings(fig_dir, embedding="final", color_by="clusters")
        exp.visu_density(fig_dir, embedding="initial", projection="pca")
        exp.visu_density(fig_dir, embedding="final", projection="pca")

    print("Saved experience to", out_dir)
    print("Best HDBSCAN grid:", exp.best_hdbscan_grid)
    print("Quality:")
    for k, v in exp.quality.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
