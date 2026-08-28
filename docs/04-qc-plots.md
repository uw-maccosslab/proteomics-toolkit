# Quality Control Plots

[← Back to overview](01-overview.md)

Run these before [normalization](05-normalization.md) and
[statistical analysis](06-statistical-analysis.md) to spot bad samples,
missing-value patterns, and batch effects early.

Sections:

- [Sample-level QC](#sample-level-qc)
- [Missing values and identifications](#missing-values-and-identifications)
- [Intensity and CV distributions](#intensity-and-cv-distributions)
- [Correlation heatmaps](#correlation-heatmaps)
- [Control pool consistency](#control-pool-consistency)
- [Peptide coverage](#peptide-coverage)
- [Variance-prior diagnostics](#variance-prior-diagnostics)

## Sample-level QC

```python
# Sample intensity distributions
ptk.plot_box_plot(
    data            = protein_data,
    sample_columns  = all_sample_cols,
    sample_metadata = sample_metadata,
    group_colors    = group_colors,
    group_order     = group_order,
    log_transform   = True,
    title           = 'Raw Intensity Distribution',
)

# PCA
ptk.plot_pca(
    data            = protein_data,
    sample_columns  = exp_sample_cols,
    sample_metadata = sample_metadata,
    group_colors    = group_colors,
    title           = 'PCA — All Samples',
    log_transform   = True,
)
```

## Missing values and identifications

All four generic QC plots below accept either a protein- or peptide-level
DataFrame; pass `feature_label="peptide"` when working on peptides so
axis titles read correctly.

```python
# Heatmap of the NA/zero pattern across samples x features
ptk.plot_missing_value_heatmap(protein_data, all_sample_cols)

# Bar chart of identifications per sample, coloured by group
ptk.plot_identifications_per_sample(
    protein_data, all_sample_cols, sample_metadata=sample_metadata
)
```

## Intensity and CV distributions

```python
# Density overlay of per-sample log-intensity (useful before/after normalization)
ptk.plot_intensity_distributions(protein_data, all_sample_cols)

# CV distribution across all samples, or split by group
ptk.plot_cv_distribution(
    protein_data, all_sample_cols, sample_metadata=sample_metadata
)
```

## Correlation heatmaps

```python
# Control sample correlation heatmap
control_cols = [c for c, m in sample_metadata.items() if m.get('Group') in ['Pool', 'Ref']]
ptk.plot_control_correlation(
    data             = protein_data,
    control_columns  = control_cols,
    sample_metadata  = sample_metadata,
    title            = 'Control Sample Correlation',
    log_transform    = False,
    cluster          = True,
    group_colors     = group_colors,
    group_column     = 'Group',
)

# Sample-to-sample correlation heatmap (triangular)
ptk.visualization.plot_sample_correlation_triangular_heatmap(
    data            = protein_data,
    sample_columns  = exp_sample_cols,
    sample_metadata = sample_metadata,
    method          = 'pearson',
    group_colors    = group_colors,
    group_column    = 'Condition',
)
```

## Control pool consistency

```python
ptk.plot_control_correlation_analysis(
    original_data          = protein_data,
    median_normalized_data = protein_data,
    vsn_normalized_data    = protein_data,
    sample_columns         = all_sample_cols,
    sample_metadata        = sample_metadata,
    control_column         = 'Condition',
    control_labels         = ['QC', 'Ref-Pool', 'GW-QC'],
)
```

## Peptide coverage

Sequence-viewer-style coverage map: the parent protein sequence is
rendered as text, peptide bars are drawn underneath at their true
residue positions, long proteins wrap to multiple rows, and bar colour
encodes abundance (default), fold-change between two conditions, or
detection frequency. Load the parent-protein sequences from a FASTA
file first:

```python
sequences = ptk.load_fasta_sequences('uniprot_human.fasta')
# Or for the bundled tutorial data: sequences = ptk.datasets.load_example_sequences()

# Default: colour bars by log2 mean abundance across sample_columns
ptk.plot_peptide_coverage_map(
    peptide_data,
    protein_id='sp|P02768|ALBU_HUMAN',
    protein_sequence=sequences['sp|P02768|ALBU_HUMAN'],
    sample_columns=sample_cols,
    start_column='start_position',   # optional - otherwise located by str.find
)

# Colour bars by Control-vs-Treatment log2 fold-change
ptk.plot_peptide_coverage_map(
    peptide_data,
    protein_id='sp|P02768|ALBU_HUMAN',
    protein_sequence=sequences['sp|P02768|ALBU_HUMAN'],
    sample_columns=sample_cols,
    color_by='fold_change',
    sample_metadata=meta_dict,
    group_column='Group',
    group_labels=('Control', 'Treatment'),
)
```

`color_by` accepts `"abundance"` (default), `"fold_change"`, or
`"detection"`. For DIA data, peptides rarely overlap; overlapping
peptides from missed cleavages are automatically stacked on separate
tracks within a row.

## Variance-prior diagnostics

Two companion plots for the
[moderated linear model](06-statistical-analysis.md#moderated-linear-model--limma-deqms-intensity_trend-or-intensity_peptide_trend):

**`plot_variance_vs_intensity`** (for `moderation="intensity_trend"` or
`"intensity_peptide_trend"`): drawn in the space the prior is actually fit
in — `log(within-group variance)` against `log(group-mean intensity)`, one
point per (feature, group) pair, with the LOWESS prior overlaid.

The title reports the observed log-log slope, and reference slopes of 1 and
2 are drawn for scale. The slope identifies the noise regime directly: 1 is
counting/shot noise (variance proportional to mean), 2 is constant CV
(variance proportional to mean squared), 0 is flat additive background. MS
intensities are ion *rates* rather than counts, so intermediate values are
normal and are exactly why the prior is nonparametric rather than committed
to either extreme.

The right-hand panel shows the residual after the intensity stage. With
`"intensity_peptide_trend"` it is plotted against peptide count with the
second-stage LOWESS overlaid, so a downward trend shows the peptide term
adding real information. With plain `"intensity_trend"` it shows the residual
distribution and its spread.

```python
# After run_moderated_linear_model with moderation='intensity_trend'
# or 'intensity_peptide_trend'
ptk.plot_variance_vs_intensity(results)
```

**`plot_variance_vs_peptide_count`** (for `moderation="deqms"`): per-
protein residual variance against peptide count, with both the limma
global prior and the DEqMS LOWESS prior overlaid. Use this after
`run_moderated_linear_model` with `moderation="deqms"` to verify the
count-conditioned prior.

```python
ptk.plot_variance_vs_peptide_count(deqms_results)
```
