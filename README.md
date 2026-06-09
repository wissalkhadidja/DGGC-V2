# DGGC Experience Framework — version pure

Cette version implémente uniquement ce cahier des charges : une classe Python `Experience` qui décrit une expérience DGGC et ses résultats.

Elle ne contient pas encore les autres méthodes de comparaison. Elle fait seulement :

```text
embedding initial
→ entraînement NCE / density model
→ descente de densité DGGC
→ HDBSCAN grid search sur embedding final
→ sélection du meilleur HDBSCAN par relative_validity_
→ calcul des métriques
→ figures
→ sauvegarde complète
```

## Installation

```bash
pip install -r requirements.txt
```

`cdbw` est inclus pour calculer le vrai CDbw. Si l'installation échoue selon ton environnement, le reste du framework fonctionne quand même, mais `cdbw_initial` et `cdbw_final` seront `NaN`.

## Test rapide

```bash
python run_synthetic_experience.py --out exp_test --dataset gaussian_blobs --n_samples 1000 --dim 2 --epochs 10 --flow_steps 50 --device cpu --figures
```

Avec CUDA :

```bash
python run_synthetic_experience.py --out exp_hubness --dataset hubness_stress --n_samples 1000 --dim 10 --epochs 10 --flow_steps 100 --device cuda --figures
```

## Attributs de `Experience`

- `embeddings` : array `N x D x K`, avec `K = T + 1` si `save_every=1`. La matrice `[:, :, 0]` est l'embedding initial frozen, et `[:, :, -1]` est l'embedding DGGC final.
- `classes` : array `N` contenant les labels connus.
- `density_model` : réseau NCE PyTorch entraîné.
- `density_model_state` : poids du modèle sauvegardables.
- `clusters` : labels HDBSCAN sélectionnés sur l'embedding final.
- `quality` : dictionnaire des temps, métriques et informations HDBSCAN.
- `best_hdbscan_grid` : meilleure configuration HDBSCAN selon `relative_validity_`.
- `hdbscan_grid_results` : toutes les configurations testées.
- `projections` : projections 2D calculées par PCA, LDA classes, LDA clusters.

## Méthodes de `Experience`

- `train_NCE()` : entraîne le NCE sur l'embedding initial et mesure le temps.
- `update_embeddings()` : applique la descente de densité DGGC et sauvegarde la trajectoire.
- `clustering()` : optimise HDBSCAN par grid search et sélectionne le max de `relative_validity_`.
- `compute_quality()` : calcule ARI, NMI, AMI, ACC, Purity, Silhouette, DBCV, CDbw, bruit et nombre de clusters.
- `projection_2D()` : calcule PCA, LDA selon les classes, LDA selon les clusters.
- `visu_embeddings()` : visualise les embeddings 2D.
- `visu_density()` : visualise le champ scalaire du NCE dans un plan de projection.
- `save_to_file()` : sauvegarde l'expérience complète.
- `load_from_file()` : recharge une expérience sauvegardée.

## HDBSCAN grid search

La sélection est volontairement pure :

```text
best grid = argmax relative_validity_
```

Aucune pénalité, aucun score supervisé et aucune correction heuristique ne sont utilisés.

La grille par défaut teste :

```text
min_cluster_size:          5 10 20 30 50 80 120 200
min_samples:               1 3 5 10 15 20 30 50
metric:                    euclidean manhattan
cluster_selection_method:  eom leaf
alpha:                     1.0
```

Tu peux changer la grille en ligne de commande :

```bash
python run_synthetic_experience.py \
  --out exp_density \
  --dataset density_bridge \
  --n_samples 1000 \
  --dim 10 \
  --epochs 10 \
  --flow_steps 100 \
  --device cuda \
  --min_cluster_sizes 10 20 30 50 80 120 \
  --min_samples_values 1 3 5 10 15 20 \
  --hdbscan_metrics euclidean manhattan \
  --hdbscan_cluster_selection_methods eom leaf \
  --figures
```

## Métriques

Les métriques supervisées traitent le bruit `-1` comme un cluster normal :

- ARI
- NMI
- AMI
- ACC
- Purity

Les métriques internes sont calculées sur l'embedding initial et final avec les mêmes labels HDBSCAN finaux :

- `silhouette_initial`, `silhouette_final`
- `dbcv_initial`, `dbcv_final`
- `cdbw_initial`, `cdbw_final`

Sont aussi sauvegardés :

- `noise_percent`
- `coverage_percent`
- `n_clusters_non_noise`
- `time_train_NCE_sec`
- `time_update_embeddings_sec`
- `time_clustering_sec`

## Sorties sauvegardées

Chaque expérience sauvegarde :

- `experience.pkl`
- `density_model.pt`
- `embeddings.npy`
- `classes.npy`
- `clusters.npy`
- `quality.json`
- `config.json`
- `best_hdbscan_grid.json`
- `hdbscan_grid_results.json`
- `figures/`

## Expanded HDBSCAN grid

This version keeps the original requirement: the selected clustering is the HDBSCAN configuration with the highest `relative_validity_` on the final DGGC embedding. No penalty and no supervised metric are used for selection.

The grid now explores:

- `metric`: `euclidean`, `manhattan`, `chebyshev`, `minkowski`, `cosine`
- `cluster_selection_method`: `eom`, `leaf`
- `alpha`: `0.5`, `1.0`, `1.5`, `2.0`
- `cluster_selection_epsilon`: `0.0`, `0.01`, `0.05`, `0.10`
- `allow_single_cluster`: `False`, `True`
- `leaf_size`: `20`, `40`
- `minkowski_p`: `1.5`, `2.0`, `3.0` when `metric=minkowski`
- `min_cluster_size`: default `5 10 20 30 50 80 120 200`
- `min_samples`: default `1 3 5 10 15 20 30 50`

The complete grid is saved in `hdbscan_grid_results.json`, and the selected configuration is saved in `best_hdbscan_grid.json`.

Warning: the full default grid is large. For quick tests, reduce the grid from the command line, for example:

```powershell
python run_synthetic_experience.py --out quick --dataset hubness_stress --n_samples 1000 --dim 10 --epochs 5 --flow_steps 50 --device cuda --hdbscan_metrics euclidean manhattan --hdbscan_alpha_values 1.0 --hdbscan_cluster_selection_epsilon_values 0.0 --hdbscan_allow_single_cluster_values 0 --hdbscan_leaf_size_values 40
```
