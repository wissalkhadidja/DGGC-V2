from __future__ import annotations

import argparse
from dggc_exp import Experience


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    exp = Experience.load_from_file(args.path)
    print("Name:", exp.config.name)
    print("Embeddings shape:", exp.embeddings.shape)
    print("Classes shape:", exp.classes.shape)
    print("Clusters shape:", None if exp.clusters is None else exp.clusters.shape)
    print("Best grid:", exp.best_hdbscan_grid)
    print("Quality:")
    for k, v in exp.quality.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
