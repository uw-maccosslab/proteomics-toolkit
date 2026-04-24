# Statistical Analysis

[← Back to overview](01-overview.md)

All analyses use `ptk.StatisticalConfig` +
`ptk.run_comprehensive_statistical_analysis`. Pick a recipe below, then
skip to [StatisticalConfig reference](#statisticalconfig-reference) for
the full parameter list.

Recipes:

- [Paired t-test](#paired-t-test-beforeafter-per-subject) (before/after per subject)
- [Unpaired comparison](#unpaired-comparison) (two independent groups)
- [PRISM data — unpaired](#prism-data--unpaired-comparison)
- [Moderated linear model](#moderated-linear-model--limma-deqms-or-intensity_trend) (empirical Bayes variance shrinkage; limma / deqms / intensity_trend)
- [Mixed-effects model](#mixed-effects-model-repeated-measures) (repeated measures)
- [Linear trend over time](#linear-trend--dose-response) (dose-response)

Reference:

- [Log transformation](#log-transformation)
- [StatisticalConfig reference](#statisticalconfig-reference)

## Paired t-test (before/after per subject)

**Design:** Each subject contributes exactly one "before" and one
"after" sample.

```python
config = ptk.StatisticalConfig()
config.analysis_type           = 'paired'
config.statistical_test_method = 'paired_t'      # or 'mixed_effects'

# Required: how samples are paired
config.subject_column = 'Patient_Number'  # links same patient across conditions
config.paired_column  = 'Condition'       # column that labels the two timepoints
config.paired_label1  = 'A'              # baseline label
config.paired_label2  = 'B'              # follow-up label  (effect = B - A)

# Required: which groups to include in the analysis
config.group_column = 'Condition'
config.group_labels = ['A', 'B']

config.log_transform_before_stats = 'auto'
config.log_base                   = 'log2'
config.correction_method          = 'fdr_bh'
config.p_value_threshold          = 0.05
config.fold_change_threshold      = 1.5

config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    normalized_data     = protein_data,
    sample_metadata     = sample_metadata,
    config              = config,
    protein_annotations = protein_annotations,
)
```

**Key output columns:** `Protein`, `Gene`, `logFC`, `P.Value`,
`adj.P.Val`, `n_pairs`, `cohens_d`. `logFC > 0` means higher in
Condition B (after).

## Unpaired comparison

**Use case:** Case vs Control, two independent patient cohorts.

```python
config = ptk.StatisticalConfig()
config.analysis_type           = 'unpaired'
config.statistical_test_method = 'welch_t'   # or 'mann_whitney' (non-parametric)

config.group_column = 'Disease_Status'
config.group_labels = ['Case', 'Control']

config.log_transform_before_stats = 'auto'
config.correction_method          = 'fdr_bh'
config.p_value_threshold          = 0.05
config.fold_change_threshold      = 1.5
config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    normalized_data     = protein_data,
    sample_metadata     = sample_metadata,
    config              = config,
    protein_annotations = protein_annotations,
)
```

## PRISM data — unpaired comparison

**Design:** Two independent groups from PRISM-normalized data.

The key steps are: (1) build the standard 5-column annotation + sample
data, (2) set the DataFrame index to protein accessions, (3) always
pass `protein_annotations`.

```python
import pandas as pd

# Build annotation DataFrame (standard 5-column format)
annot = protein_data[[
    'leading_protein', 'leading_description', 'leading_gene_name',
    'leading_uniprot_id', 'leading_name'
]].copy()
annot.columns = ['Protein', 'Description', 'Protein Gene', 'UniProt_Accession', 'UniProt_Entry_Name']

# Combine annotations + sample data
data = pd.concat([
    annot.reset_index(drop=True),
    protein_data[sample_cols].reset_index(drop=True)
], axis=1)
data.index = data['Protein']   # accession as index — critical for meaningful results

config = ptk.StatisticalConfig()
config.analysis_type           = 'unpaired'
config.statistical_test_method = 'welch_t'
config.group_column            = 'Group'
config.group_labels            = ['KI Control', 'KI']   # [reference, study]
config.log_transform_before_stats = 'auto'
config.correction_method       = 'fdr_bh'
config.p_value_threshold       = 0.05
config.fold_change_threshold   = 1.0
config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    data, sample_meta_dict, config,
    protein_annotations=annot,
)
```

**Output columns:** `Protein`, `logFC`, `P.Value`, `adj.P.Val`,
`AveExpr`, `t`, `Protein Gene`, `Description`, `UniProt_Accession`,
`Gene`.

> The toolkit preserves protein accessions as the DataFrame index
> during statistical testing, so the `Protein` column in results always
> contains real accession numbers (not integer row indices).

## Moderated linear model — limma, deqms, or intensity_trend

**Use case:** Small sample sizes (fewer than ~6 replicates per group)
where raw per-feature t-statistics are under-powered. A single entry
point `run_moderated_linear_model` runs the Smyth per-feature OLS fit
and applies one of three empirical-Bayes variance priors selected via
`config.moderation`:

| Moderation | Prior shape | When to pick |
|---|---|---|
| `"intensity_trend"` *(default)* | Prior varies with mean intensity (Python equivalent of limma's `trend=TRUE`). Captures the Poisson `sd ~ sqrt(intensity)` relationship natural to MS counting noise. | Default for MS data. Works at protein and peptide level. |
| `"limma"` | Single global prior (Smyth 2004). | Use when the variance-intensity trend is flat, or as a conservative baseline. |
| `"deqms"` | Prior conditioned on peptide count (Zhu et al. 2020). | Protein-level only. Useful when peptide counts are informative about identification confidence. |

Set `config.robust = True` to Winsorize extreme `s_i²` values when
estimating the prior hyperparameters (matches limma's `robust=TRUE`).
This prevents a handful of genuinely high-variance features from
inflating the global prior.

```python
# intensity_trend (default) on protein- or peptide-level data.
config = ptk.StatisticalConfig()
config.analysis_type           = 'unpaired'
config.statistical_test_method = 'moderated_linear_model'
config.moderation              = 'intensity_trend'    # default
config.robust                  = False                # optional
config.group_column            = 'Group'
config.group_labels            = ['Control', 'Treatment']
config.log_transform_before_stats = True
config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    data, sample_meta_dict, config, protein_annotations=annot
)

# Diagnostic: (feature, group) SD vs sqrt(intensity) with LOWESS prior
ptk.plot_variance_vs_intensity(results)
```

For DEqMS with PRISM protein data (PRISM emits `n_peptides` in the
protein parquet):

```python
# Include n_peptides in the data DataFrame you pass in.
data_with_counts = pd.concat([
    annot.reset_index(drop=True),
    protein_data[['n_peptides']].reset_index(drop=True),
    protein_data[sample_cols].reset_index(drop=True),
], axis=1)

config.moderation = 'deqms'
# config.peptide_count_column defaults to 'n_peptides'
results = ptk.run_comprehensive_statistical_analysis(
    data_with_counts, sample_meta_dict, config, protein_annotations=annot
)
ptk.plot_variance_vs_peptide_count(results)
```

Output DataFrames include extra columns: `residual_s2`, `residual_df`,
`posterior_s2`, `posterior_df`, `limma_s0_sq`, plus one of
`deqms_s0_sq` + `peptide_count_used` or
`intensity_s0_sq` + `intensity_used` depending on moderation. The
`intensity_trend` results also carry a per-(feature, group) long-form
DataFrame on `results.attrs["intensity_trend_points"]`, accessible via
`ptk.get_intensity_trend_points(results)`.

### Migration from `limma_like` / `deqms_like`

The previous `run_limma_like_analysis` / `run_deqms_like_analysis`
functions and the `statistical_test_method` string values `"limma_like"`
/ `"deqms_like"` have been **removed**. Migrate as follows:

```python
# Before
config.statistical_test_method = 'limma_like'
# After
config.statistical_test_method = 'moderated_linear_model'
config.moderation = 'limma'

# Before
config.statistical_test_method = 'deqms_like'
# After
config.statistical_test_method = 'moderated_linear_model'
config.moderation = 'deqms'
```

## Mixed-effects model (repeated measures)

**Use case:** Comparing groups at multiple timepoints while accounting
for within-subject correlation (e.g., dose × visit interaction).

```python
config = ptk.StatisticalConfig()
config.analysis_type           = 'paired'        # or 'interaction'
config.statistical_test_method = 'mixed_effects'

config.subject_column = 'Subject'
config.paired_column  = 'Visit'
config.paired_label1  = 'D-02'    # baseline visit
config.paired_label2  = 'D-13'    # follow-up visit

config.group_column   = 'DrugDose'
config.group_labels   = ['0', '20', '40', '80']

# Interaction terms: test if dose × visit effect is significant
config.interaction_terms    = ['DrugDose', 'Visit']
config.additional_interactions = []
config.covariates           = []           # optional: e.g. ['Age', 'Sex']
config.force_categorical    = False        # True = treat DrugDose as factors

config.log_transform_before_stats = 'auto'
config.correction_method          = 'fdr_bh'
config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    normalized_data     = protein_data,
    sample_metadata     = sample_metadata,
    config              = config,
    protein_annotations = protein_annotations,
)
```

## Linear trend / dose-response

**Use case:** Test if protein abundance changes linearly with dose or
time. Useful when the interval between timepoints varies across
subjects (e.g., 18-31 days). The model is
`Protein ~ TimeBetweenSamples + (1|Subject)`.

`logFC` in the output represents the **slope per unit time**, so values
are small. Use `fc_threshold=0.01` in volcano plots and
`logfc_threshold=0.0` in enrichment analysis.

```python
config = ptk.StatisticalConfig()
config.analysis_type           = 'linear_trend'
config.statistical_test_method = 'mixed_effects'

config.subject_column = 'Patient_Number'
config.time_column    = 'TimeBetweenSamples'

config.correction_method     = 'fdr_bh'
config.p_value_threshold     = 0.05
config.fold_change_threshold = 1.0

results = ptk.run_comprehensive_statistical_analysis(
    data, meta_dict, config, protein_annotations=annot,
)

# Volcano: use small FC threshold since logFC is per-day
ptk.plot_volcano(results, fc_threshold=0.01, p_threshold=0.05, label_top_n=15)

# Enrichment: no FC cutoff (slopes are small by nature)
enrichment_results = ptk.run_differential_enrichment(
    results, logfc_threshold=0.0, pvalue_threshold=0.05,
)
```

See [08-enrichment.md](08-enrichment.md) for the full enrichment workflow.

## Log transformation

`config.log_transform_before_stats` accepts `'auto'`, `True`, or
`False`.

- `'auto'` inspects `config.normalization_method` (or, absent that, the
  data mean) to decide whether the input is already log-scale.
- Set it to `True` when you know your data is linear (e.g., PRISM
  protein parquet) and you want log2 before stats.
- Set it to `False` when the data is already log-transformed (e.g.,
  after [`vsn_normalize`](05-normalization.md) or `rlr_normalize`).

## StatisticalConfig reference

| Attribute | Type | Description |
|---|---|---|
| `analysis_type` | str | **Required.** `'paired'`, `'unpaired'`, `'linear_trend'`, `'longitudinal'`, `'interaction'` |
| `statistical_test_method` | str | `'paired_t'`, `'mixed_effects'`, `'welch_t'`, `'student_t'`, `'wilcoxon'`, `'mann_whitney'`, `'moderated_linear_model'` |
| `moderation` | str | When `statistical_test_method='moderated_linear_model'`: `'limma'`, `'deqms'`, or `'intensity_trend'` *(default)* |
| `robust` | bool | Winsorize extreme `s_i²` when estimating prior hyperparameters (matches limma's `robust=TRUE`). Default `False`. |
| `subject_column` | str | Metadata column identifying subjects/patients (required for paired/mixed) |
| `group_column` | str | Metadata column with group labels |
| `group_labels` | list | Labels to compare (e.g. `['A', 'B']`) |
| `paired_column` | str | Column distinguishing the two timepoints in a paired design |
| `paired_label1` | str | Baseline/before label in `paired_column` |
| `paired_label2` | str | Follow-up/after label in `paired_column` |
| `time_column` | str | Numeric time/dose column for `linear_trend`/`longitudinal` |
| `interaction_terms` | list | Mixed-effects interaction terms (e.g. `['Group', 'Visit']`) |
| `covariates` | list | Additional covariates to control for (e.g. `['Age', 'Sex']`) |
| `log_transform_before_stats` | str/bool | `'auto'`, `True`, `False` - see [Log transformation](#log-transformation) |
| `log_base` | str | `'log2'` (default), `'log10'`, `'ln'` |
| `correction_method` | str | `'fdr_bh'` (BH), `'bonferroni'`, `'fdr_by'`, etc. |
| `use_adjusted_pvalue` | str | `'adjusted'` or `'unadjusted'` |
| `p_value_threshold` | float | Volcano plot line (default 0.05) |
| `fold_change_threshold` | float | FC threshold for significance calls (default 1.5) |
| `peptide_count_column` | str | Column used by `moderation='deqms'` (default `'n_peptides'`) |

Always call `config.validate()` before running analysis to catch
configuration errors early.

## Next steps

- [Visualise results](07-visualization.md)
- [Run enrichment](08-enrichment.md)
- [Binary classification](09-classification.md)
- [Common pitfalls](11-pitfalls.md)
