# Results Visualization

[← Back to overview](01-overview.md)

For upstream QC plots (box plots, PCA, missing-value patterns) see
[04-qc-plots.md](04-qc-plots.md). This page covers results-level plots that
consume the output of [06-statistical-analysis.md](06-statistical-analysis.md).

## Volcano plot

```python
ptk.plot_volcano(
    differential_df        = results,
    fc_threshold           = 1.0,              # log2FC threshold (default: 0.5)
    p_threshold            = 0.05,
    title                  = 'KI vs KI Control',
    gene_column            = 'Protein Gene',   # column for point labels
    label_top_n            = 20,
    use_adjusted_pvalue    = 'adjusted',
    enable_pvalue_fallback = True,
    normalization_method   = 'PRISM_corrected',
    point_size             = 80,
    alpha                  = 0.4,
    label_fontsize         = 11,
    axis_label_fontsize    = 20,
    tick_label_fontsize    = 16,
)
```

## Summary table

```python
ptk.display_analysis_summary(
    differential_results = results,
    config               = config,
    label_top_n          = 20,
)
```

## Heatmap of significant proteins

```python
sig = results[results['adj.P.Val'] < 0.05]
ptk.plot_grouped_heatmap(
    data_df        = protein_data.reset_index(),
    value_columns  = exp_sample_cols,
    group_column   = None,           # set to a cluster column if available
    label_column   = 'protein_group',
    title          = 'Significant Proteins',
    zscore         = True,
    pvalue_column  = 'adj.P.Val',
    sort_by_pvalue = True,
)
```

## Bi-clustered sample heatmap

`plot_sample_clustermap` is a seaborn-backed clustermap of samples
(columns) by features (rows) with an optional top color bar
annotating each sample by group. Rows are z-scored by default so the
colormap reflects relative pattern across samples; both axes are
hierarchically clustered.

Useful for: visually confirming that samples cluster by treatment
group on the significant proteins, or for displaying a curated panel
of features grouped by metadata.

```python
sig = results[results['adj.P.Val'] < 0.05].head(50)

g = ptk.plot_sample_clustermap(
    data            = protein_data.loc[sig['Protein']],
    sample_columns  = exp_sample_cols,
    sample_metadata = meta_dict,
    group_column    = 'Group',
    label_column    = 'leading_gene_name',
    title           = 'Top significant proteins',
    zscore          = True,                # row z-score before clustering
    cmap            = 'RdBu_r',
    vmin            = -2.0, vmax           = 2.0,
    row_cluster     = True,
    col_cluster     = True,
    metric          = 'correlation',
    method          = 'average',
)

# g is a seaborn ClusterGrid; tweak g.ax_heatmap, g.ax_col_dendrogram, ...
```

Notes:

- All-NaN rows are dropped (with a warning) and remaining NaNs are
  filled with the row mean before clustering, since seaborn's
  clustermap aborts on NaN.
- Row labels are shown when there are at most 80 rows by default;
  override with `show_row_labels=True` / `False`. Sample labels
  along the x-axis are off by default (`show_col_labels=True` to
  enable).
- Pass `col_cluster=False` to preserve the original sample order
  (e.g., when the column order already encodes time or dose).
- `group_colors=` accepts a `{group_label: color_string}` mapping;
  if omitted, colors are auto-assigned from `tab10`.

## Protein profile and grouped trajectories

For longitudinal or temporal data, see `ptk.plot_grouped_trajectories`
and `ptk.plot_protein_profile`. These are particularly useful alongside
the [linear trend / dose-response recipe](06-statistical-analysis.md#linear-trend--dose-response).
