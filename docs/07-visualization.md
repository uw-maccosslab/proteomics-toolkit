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

## Protein profile and grouped trajectories

For longitudinal or temporal data, see `ptk.plot_grouped_trajectories`
and `ptk.plot_protein_profile`. These are particularly useful alongside
the [linear trend / dose-response recipe](06-statistical-analysis.md#linear-trend--dose-response).
