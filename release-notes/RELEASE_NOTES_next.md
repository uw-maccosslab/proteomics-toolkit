# Release Notes (Next Release)

## Overview

Adds descriptive marker-discovery metrics, multivariate variance partitioning
(PERMANOVA), multi-class permutation importance with bootstrap stability,
silhouette-driven k-means protein clustering, and two new visualization
helpers (PCA loadings biplot, UMAP). These additions support designs with
very small sample sizes per group, where classical per-protein hypothesis
testing is underpowered, by providing rigorous descriptive metrics and an
omnibus permutation test, plus an unsupervised clustering route that pairs
naturally with downstream gene-set enrichment.

## New Features

### `marker_discovery` module

- `method_specificity_score(data, sample_columns, sample_metadata, group_column, ...)`:
  per (protein, group) pair, computes the group mean, distance from the
  second-best group (`delta_top`), specificity vs across-group median, and
  rank. Long-form output sorted by group then rank for easy `groupby().head(N)`.
- `inter_vs_intra_group_variance(data, sample_columns, sample_metadata, group_column, ...)`:
  per-protein ratio of variance across group means to mean within-group
  variance. High ratio = group-discriminating. Descriptive complement to
  ANOVA when n per group is too small for classical tests.
- `cluster_proteins_kmeans(data, sample_columns, ...)`: k-means clustering
  of proteins over samples with silhouette-driven k selection. Returns a
  per-protein cluster assignment plus the silhouette curve so callers can
  document the chosen k. Supports per-protein z-score standardization for
  shape-based clustering. Pairs naturally with `run_enrichment_by_group`
  for per-cluster gene-set / GO enrichment.

### `multivariate` module

- `permanova(data, sample_columns, sample_metadata, factor, ...)`: Anderson 2001
  PERMANOVA on a sample-by-sample distance matrix, with label permutation for
  significance. Pure scipy implementation; supports euclidean, braycurtis,
  cosine, correlation, cityblock metrics. Returns pseudo-F, R^2, p-value, and
  group sizes.

### `classification` module

- `multiclass_feature_importance(data, sample_columns, sample_metadata, group_column, ...)`:
  multi-class random-forest (or XGBoost) classifier with sklearn permutation
  importance, plus a bootstrap stability score (fraction of resamples in which
  the protein lands in the top-K importance ranks). Designed for descriptive
  marker discovery in low-replication designs.

### `visualization` module

- `plot_pca_loadings(data, sample_columns, ...)`: PCA loadings biplot showing
  the top-N proteins driving two principal components. Complements `plot_pca`
  (which shows samples in PC space).
- `plot_umap(data, sample_columns, sample_metadata, group_column, ...)`: UMAP
  projection of samples colored by a metadata group. Lazy import of
  `umap-learn`; install with `pip install proteomics-toolkit[umap]` or
  `pip install umap-learn`.

## Changes

- None to existing public APIs.

## Bug Fixes

- Fixed `plot_box_plot` x-tick labeling to prefer the `Replicate` field from
  `sample_metadata` when present, falling back to `_make_display_labels`
  (batch-suffix stripping) only when `Replicate` is missing.

## Testing

- New `tests/test_marker_discovery.py` covering method_specificity_score,
  inter_vs_intra_group_variance, and cluster_proteins_kmeans with planted
  patterns.
- New `tests/test_multivariate.py` covering PERMANOVA with both null
  (no group structure) and strong-effect synthetic data.
- New `tests/test_multiclass_importance.py` covering RF importance ranking
  and bootstrap stability on synthetic data with planted markers.
- New `tests/test_visualization_loadings_umap.py` for `plot_pca_loadings`;
  UMAP test is skipped if `umap-learn` is not installed.

## Dependencies

- Added optional `umap` extra: `umap-learn>=0.5`. Required only for
  `plot_umap`. Install with `pip install proteomics-toolkit[umap]`.
