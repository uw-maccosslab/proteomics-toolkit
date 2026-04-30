# Release Notes v26.3.0

## Overview

Adds descriptive marker-discovery metrics, multivariate variance partitioning
(PERMANOVA), multi-class permutation importance with bootstrap stability, and
two new visualization helpers (PCA loadings biplot, UMAP). These additions
support designs with very small sample sizes per group, where classical
per-protein hypothesis testing is underpowered, by providing rigorous
descriptive metrics and an omnibus permutation test.

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
  (batch-suffix stripping) only when `Replicate` is missing. The previous
  behavior used `_make_display_labels` unconditionally and lost the
  replicate-name labels that downstream notebooks depend on.

## Testing

- New `tests/test_marker_discovery.py` covering method_specificity_score and
  inter_vs_intra_group_variance with a known-effect synthetic dataset.
- New `tests/test_multivariate.py` covering PERMANOVA with both null
  (no group structure) and strong-effect synthetic data; verifies p-value
  bounds and approximate R^2.
- New `tests/test_multiclass_importance.py` covering RF importance ranking
  and bootstrap stability on synthetic data with planted markers.
- New `tests/test_visualization_loadings_umap.py` for `plot_pca_loadings`;
  UMAP test is skipped if `umap-learn` is not installed.

## Dependencies

- Added optional `umap` extra: `umap-learn>=0.5`. Required only for
  `plot_umap`. Install with `pip install proteomics-toolkit[umap]`.

## Documentation

- README installation section reorganized to show `uv` as the recommended
  install path (`uv sync`, `uv sync --extra dev`, `uv sync --extra umap`,
  `uv add proteomics-toolkit`) alongside the existing `pip` instructions.
- Removed the obsolete `[xgboost]` extra from the README (xgboost has been a
  required dependency since v26.2.1). Added the new `[umap]` extra.
