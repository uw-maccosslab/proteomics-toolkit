# Release Notes (Next Release)

## Overview

Adds SHAP-based interpretability for the tree-based binary classifiers,
opt-in gene-name relabeling of classifier and SHAP outputs, a new
`plot_sample_clustermap` bi-clustered heatmap, and a major performance
and correctness fix for the intensity-trend moderated linear model
output (iterrows on a results DataFrame previously degraded from
~seconds to ~minutes, and some pandas operations raised a "truth value
of a DataFrame is ambiguous" error).

## New Features

### Classification

- `run_binary_classification` gained a `return_model=False` parameter.
  When set to `True`, the result dict additionally carries
  `final_model`, `scaler`, `X_scaled`, and `y_encoded` so the fitted
  pipeline can be reused downstream (e.g., for SHAP) without
  retraining on a slightly different fold structure.
- New `classification.compute_shap_values(model, X, feature_names=None, annotations=None, id_col=..., gene_col=...)`:
  thin wrapper around `shap.TreeExplainer` for RandomForest /
  XGBoost binary classifiers. Collapses the 3-D
  `(samples, features, classes)` Explanation to the 2-D positive-class
  slice so the result is ready for `shap.plots.beeswarm` or
  `plot_shap_summary`. Raises `ImportError` with an install hint when
  `shap` is missing.
- New `classification.plot_shap_summary(explanation, max_display=20, plot_type='beeswarm', title=None)`:
  renders a beeswarm or bar summary of the top features and returns
  the matplotlib figure.
- New `classification.relabel_features_with_genes(feature_ids, annotation_df, id_col='protein_group', gene_col='leading_gene_name', fallback='id'|'empty')`:
  maps pipeline-internal feature IDs (e.g. PRISM `PG####`
  protein-group identifiers) to human-readable gene symbols at the
  point where the user surface is materialised (plot labels,
  importance Series index, SHAP `Explanation.feature_names`). Falls
  back to the original ID when the gene name is missing.
- `run_binary_classification` and `compute_shap_values` both gained
  opt-in `annotations=`, `id_col=`, `gene_col=` parameters that wire
  the helper into the returned `feature_names`,
  `feature_importances.index`, and SHAP `Explanation.feature_names`.
  Models are still trained on the original IDs internally; original
  IDs are preserved as `feature_ids` when `return_model=True`.

### Visualization

- New `plot_sample_clustermap(data, sample_columns, sample_metadata=None, group_column='Group', ...)`:
  bi-clustered seaborn heatmap with an optional top color bar
  annotating each column by group. Row-z-scores by default for
  pattern-focused reads, auto-sizes the figure from row/column
  counts, drops all-NaN rows (with a warning) and fills remaining
  NaNs with row-mean before clustering. Returns the seaborn
  `ClusterGrid` so callers can further tweak `ax_heatmap`,
  `ax_col_dendrogram`, etc.
- `plot_sample_clustermap` gained a `label_fallback_columns: list[str] | None`
  parameter. When `label_column` is missing for a row, the function now
  walks a fallback chain of annotation columns instead of dropping
  straight to the DataFrame index (which is uninformative after
  `reset_index()`). Defaults to `["leading_uniprot_id", "protein_group"]`
  when those columns exist (PRISM-friendly), so callers passing
  `label_column="leading_gene_name"` get a useful label for custom or
  unmapped entries such as the apoE_E4 spiked standard whose
  `leading_gene_name` is literally the string `"NA"`. Pass an explicit
  list to override the default; pass `[]` to disable fallback and
  reproduce the legacy index-fallback behavior. The missing-value check
  is also now case-insensitive and recognises `""`, `"na"`, `"nan"`,
  `"none"`, `"null"`, and `"n/a"` (previously only the literal
  lowercase `"nan"` was treated as missing).

## Bug Fixes

- `run_moderated_linear_model(..., moderation='intensity_trend')` results
  could break downstream pandas operations. Operations like
  `results.nsmallest(...)`, `results.sort_values(...).head()`, and
  `pd.concat([...])` raised `ValueError: The truth value of a DataFrame
  is ambiguous` because pandas compares `.attrs` for equality on these
  operations and the per-(feature, group) diagnostic points were stored
  as a DataFrame. The points are now stashed as a list of records (one
  dict per row), wrapped in a deepcopy-no-op sentinel
  (`_AttrsPayload`). `get_intensity_trend_points(results)` recovers
  the DataFrame as before and tolerates legacy storage forms.
- `plot_variance_vs_intensity` now routes through
  `get_intensity_trend_points` so both the new records-form and any
  legacy DataFrame-form storage in `attrs` are handled uniformly.

## Performance

- `iterrows` over an intensity-trend moderated results DataFrame is now
  fast. Pandas deep-copies every `attrs` value when propagating
  attributes to per-row Series, so storing a ~17,000-element list of
  dicts in `attrs` previously turned an 8,700-row `iterrows` loop from
  ~2 seconds into ~22 minutes. The `_AttrsPayload` wrapper opts out of
  `copy.deepcopy` (returns `self`), restoring near-constant per-row
  cost.

## Testing

- New `TestReturnModel` in `tests/test_classification.py`: confirms the
  default does not include model artifacts, and that
  `return_model=True` returns the fitted classifier, scaler, scaled
  feature matrix, and encoded labels.
- New `TestShap` (skipped when `shap` is not installed) covering:
  2-D positive-class collapse from `compute_shap_values`, DataFrame
  input preserving feature names, beeswarm and bar variants of
  `plot_shap_summary` running end-to-end, and a `ValueError` on
  invalid `plot_type`.
- Seven new tests for `relabel_features_with_genes` and the classifier
  / SHAP `annotations=` round-trip: id-fallback, empty-fallback,
  custom column names, invalid `fallback`, classifier relabeling
  round-trip, no-annotations backwards compatibility, and
  SHAP+annotations round-trip.
- New `TestPlotSampleClustermap` in `tests/test_visualization.py`:
  basic clustermap with group color bar, clustermap without metadata,
  `row_cluster=False` preserves row order, missing-sample-column
  raises, empty `sample_columns` raises.
- Three new clustermap row-label tests in
  `tests/test_visualization.py`:
  `test_clustermap_label_fallback_to_uniprot_then_protein_group`
  exercises the full PRISM-default chain across rows containing the
  literal string `"NA"`, `np.nan`, lowercase `"nan"`, and an empty
  UniProt forcing fallback to `protein_group`;
  `test_clustermap_label_fallback_explicit_columns` verifies an
  explicit `label_fallback_columns` list overrides the PRISM
  default; `test_clustermap_label_fallback_disabled` checks that
  `label_fallback_columns=[]` reproduces the legacy fallback to the
  DataFrame index.
- New regression tests for the intensity-trend attrs storage:
  `test_intensity_trend_attrs_do_not_break_pandas_ops` verifies
  `nsmallest` / `sort_values` no longer raise, and
  `test_intensity_trend_attrs_do_not_slow_iterrows` asserts
  `iterrows` stays under 1 s on the fixture and that per-row attrs
  share object identity with the parent payload.

## Dependencies

- Added optional `shap` extra: `shap>=0.46`. Required only for
  `compute_shap_values` and `plot_shap_summary`. Install with
  `pip install proteomics-toolkit[shap]` or `uv sync --extra shap`.

## Documentation

- `docs/09-classification.md` gained two new sections:
  "Gene-name labels for plots and importance tables" documenting the
  `annotations=` / `id_col=` / `gene_col=` opt-in across
  `run_binary_classification`, `compute_shap_values`, and the
  standalone `relabel_features_with_genes` helper; and
  "SHAP interpretability for tree models" covering the
  `return_model=True` workflow, `compute_shap_values`, and
  `plot_shap_summary` with the install hint for the `[shap]` extra.
- `docs/07-visualization.md` gained a "Bi-clustered sample heatmap"
  section for `plot_sample_clustermap`, including the row-z-score
  default, NaN handling, label thresholds, and `col_cluster=False`
  for preserving original sample order.
- `README.md` Core Analysis Modules and Module Reference brought
  back in sync with the actual public API: added entries for
  `classification`, `marker_discovery`, and `multivariate` (previously
  missing), and filled in the public functions that had accumulated
  since the README was last updated. Specifically:
  `data_import` gained `load_prism_peptide_data`, `load_diann_data`,
  `load_fasta_sequences`; `statistical_analysis` gained
  `run_moderated_linear_model`, `get_intensity_trend_points`,
  `compute_paired_fold_changes`; `visualization` gained
  `plot_pca_loadings`, `plot_umap`, `plot_missing_value_heatmap`,
  `plot_identifications_per_sample`, `plot_intensity_distributions`,
  `plot_cv_distribution`, `plot_peptide_coverage_map`,
  `plot_variance_vs_intensity`, `plot_variance_vs_peptide_count`,
  and `plot_sample_clustermap`.
