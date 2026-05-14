# Release Notes v26.4.1

## Overview

Patch release that fixes a long-standing bug in `plot_cluster_heatmap` where
cluster-size labels reflected the truncated row count rather than the true
cluster size, plus changes the default to show every protein per cluster
instead of capping at 50.

## Bug Fixes

- `temporal_clustering.plot_cluster_heatmap` no longer mislabels clusters. The
  function used to first truncate each cluster to `max_proteins_per_cluster=50`
  rows, then count "cluster size" from the truncated DataFrame; large clusters
  therefore all rendered as `(n=50)` regardless of their true size. The fix
  computes cluster sizes from the original input DataFrame so labels always
  show the true `n`. The on-screen band height still reflects however many
  rows are rendered.
- Heatmap now uses `interpolation="nearest"` so dense heatmaps with many rows
  don't blur adjacent rows into a smooth gradient. Cluster boundaries (white
  separators) remain crisp.
- Y-axis "Proteins (n=...)" label now reports the full cluster-input size, not
  the post-truncation row count.

## Changes

- `plot_cluster_heatmap` parameter `max_proteins_per_cluster` now accepts
  `None` (the new default) which renders every protein. The previous default
  was `50`. Callers that explicitly pass an integer keep their previous
  behaviour.
- `TemporalClusteringConfig.max_proteins_per_cluster_heatmap` default changed
  from `30` to `None`. `run_temporal_analysis` now passes this through to
  `plot_cluster_heatmap`, so by default the dispatcher's heatmap shows every
  protein. Set it to an integer in `TemporalClusteringConfig` to restore the
  cap.
- Heatmap figure height now scales with the number of rendered rows
  (`max(8, min(24, n_rows * 0.025 + 3))` inches) so dense heatmaps remain
  legible rather than getting compressed into a fixed 12-inch box.

## Testing

- New `tests/test_temporal_clustering.py` covering: label-uses-true-size when
  capped, label-uses-true-size when uncapped, default `max_proteins_per_cluster_heatmap=None`
  produces no truncation, full-row count on the y-axis.
