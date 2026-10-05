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
  - [Covariate adjustment](#covariate-adjustment) (unpaired only)
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

## Moderated linear model — limma, deqms, intensity_trend, or intensity_peptide_trend

**Use case:** Small sample sizes (fewer than ~6 replicates per group)
where raw per-feature t-statistics are under-powered. A single entry
point `run_moderated_linear_model` runs the Smyth per-feature OLS fit
and applies one of four empirical-Bayes variance priors selected via
`config.moderation`:

| Moderation | Prior shape | When to pick |
|---|---|---|
| `"intensity_peptide_trend"` | Additive two-stage LOWESS: `log(var) = f1(log mean intensity) + f2(log peptide count)`. | **Most accurate for protein-level rollup data.** Requires a peptide-count column. |
| `"intensity_trend"` *(default)* | Nonparametric LOWESS of `log(variance)` on `log(mean intensity)` for the shape, scaled to the design's residuals ([below](#where-the-intensity-priors-level-and-weight-come-from)). | Good default for MS data. Works at protein *and* peptide level. |
| `"limma"` | Single global prior (Smyth 2004). | Use when the variance-intensity trend is flat, or as a conservative baseline. |
| `"deqms"` | Prior conditioned on peptide count alone (Zhu et al. 2020). | Protein-level only. See the caveat below before preferring it. |

**On the shape of the intensity prior.** `intensity_trend` makes *no*
counting-noise assumption. MS intensities are ion rates, not ion counts, so
there is no reason to expect the Poisson `sd ~ sqrt(intensity)` relationship
to hold, and empirically it does not: on a 14-replicate validation dataset the
`log(variance)` vs `log(mean intensity)` slope is 1.42, between the shot-noise
value of 1 and the constant-CV value of 2. The prior is fully nonparametric
precisely so it does not have to commit to either. Fitting an explicit
`a + b*mu + c*mu^2` error model instead gives no measurable improvement
(cross-validated RMSE 1.046 vs 1.043).

**Why peptide count belongs in the prior.** Intensity is not a sufficient
statistic for protein-level variance: at matched intensity, a protein rolled
up from many peptides is better determined than one from few. The two
predictors are only weakly correlated (r = 0.34), so the peptide term adds
largely independent information. Cross-validated RMSE on `log(variance)`,
7,990 (feature, group) points from 14 technical replicates:

| Prior | CV RMSE | vs flat prior |
|---|---|---|
| `"limma"` (constant) | 2.337 | — |
| `"deqms"`-style, peptide count alone | 2.389 | -2.2% |
| `"intensity_trend"` | 1.043 | +55.4% |
| **`"intensity_peptide_trend"`** | **0.831** | **+64.4%** |

Note the trap in row two: peptide count *alone* is slightly **worse** than a
flat prior. It helps only in addition to intensity, never instead of it, so
`"deqms"` is not a substitute for `"intensity_peptide_trend"`.

**Which abundance conditions the prior.** The intensity stage always uses the
abundance of the feature actually being tested: protein abundance for a
protein-level analysis, peptide abundance for a peptide-level one. The prior
reads the same raw feature matrix the model is fit on, so this follows
automatically and needs no configuration. `"intensity_peptide_trend"` is
protein-level only, since a peptide has no peptide count.

**Why the peptide term is not an empirical hack.** Fitting the two terms
jointly and linearly on the validation dataset gives

```
log(var) = c + 1.60 * log(protein abundance) - 0.89 * log(n_peptides)     R2 = 0.88
```

Both coefficients land where theory says they should. If a protein abundance
is a rollup of `n` peptides with independent errors, its variance scales as
`1/n`, i.e. a peptide coefficient of exactly -1; the observed -0.89 says
averaging is nearly ideal, with a small shortfall consistent with correlated
peptide-level error (shared ionisation suppression, interference, and rollup
shrinkage). The abundance coefficient of 1.60 again sits between the
shot-noise and constant-CV extremes. So the second stage is recovering a real
statistical property of the rollup rather than fitting noise.

Because the model is additive in the two log terms, conditioning stage 1 on
mean per-peptide intensity (`log(abundance) - log(n_peptides)`) instead of
protein abundance is only a reparameterisation, and measurably makes no
difference (CV RMSE 0.8325 vs 0.8314). Protein abundance is kept as the more
directly interpretable choice.

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

# Diagnostic, drawn in fit space: log(variance) vs log(mean intensity) with
# the LOWESS prior, the observed log-log slope, and reference slopes of
# 1 (shot noise) and 2 (constant CV).
ptk.plot_variance_vs_intensity(results)
```

For the combined intensity + peptide prior with PRISM protein data (PRISM
emits `n_peptides` in the protein parquet):

```python
# Include n_peptides alongside the annotation and sample columns.
data_with_counts = pd.concat([
    annot.reset_index(drop=True),
    protein_data[sample_cols].reset_index(drop=True),
], axis=1)
data_with_counts['n_peptides'] = protein_data['n_peptides'].values

config.moderation = 'intensity_peptide_trend'
# config.peptide_count_column defaults to 'n_peptides'
results = ptk.run_comprehensive_statistical_analysis(
    data_with_counts, sample_meta_dict, config, protein_annotations=annot
)
# Right-hand panel shows the residual after the intensity stage against
# peptide count, so the value of the second stage is visible.
ptk.plot_variance_vs_intensity(results)
```

For DEqMS (peptide count alone) with PRISM protein data:

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
`intensity_s0_sq` + `intensity_trend_shape` + `intensity_trend_level` +
`intensity_used` depending on moderation
(`intensity_peptide_trend` adds `peptide_count_used` as well).
`intensity_s0_sq` is the prior variance the test used, which is
`intensity_trend_level * intensity_trend_shape`. The
`intensity_trend` and `intensity_peptide_trend` results also carry a
per-(feature, group) long-form DataFrame on
`results.attrs["intensity_trend_points"]`, accessible via
`ptk.get_intensity_trend_points(results)`, with columns
`intensity_log_var_hat` and — for the combined mode —
`peptide_log_var_adj` giving each stage's contribution.

### Where the intensity prior's level and weight come from

A moderated t replaces each feature's own variance estimate, which is noisy
with few samples, by a weighted average of that estimate and a prior:

```
posterior_s2 = (d0 * prior + residual_df * residual_s2) / (d0 + residual_df)
```

The prior has two parts. Its **scale** is the variance expected for a feature
at that intensity. Its **weight**, `d0`, says how far to trust that scale over
the feature's own estimate. The intensity prior gets them from different
places:

1. **The shape comes from groups of samples.** The LOWESS of
   `log(within-group variance)` on `log(within-group mean intensity)` is fitted
   on the design groups, or on dedicated QC and reference injections when
   `variance_prior_group_column` is set (see below). This says how noise
   changes with intensity: low-abundance features are noisier.
2. **The level and the weight come from the design's residuals.** The trend is
   multiplied by one factor, reported as `intensity_trend_level`, and `d0` is
   estimated at the same time, by fitting the residual variances against the
   trend. This is Smyth's (2004) method of moments on `residual_s2 / trend`,
   which is limma's `fitFDist` with the covariate curve supplied rather than
   fitted.

Step 2 exists because no source of groups measures the noise a design is
tested against:

- **QC and reference pools** carry instrument and processing noise only. A
  study residual also carries biology: between-person spread in an unpaired
  design, and week-to-week variation within a person in a paired one. Their
  trend sits too **low**.
- **Design groups under a paired or within-subject model** contain the
  between-subject spread that the model's subject block removes. Their trend
  sits too **high**.
- **Small groups** bias the trend low regardless of source. A LOWESS fits the
  mean of `log(variance)`, and the log of a chi-square draw averages below the
  log of its mean: by a factor of 0.81 at six samples per group, and 0.28 at two.

**Before v26.8.0 there was no step 2.** The trend was used at whatever level
its source gave, and `d0` was estimated separately, from the residuals around
their own global mean. That `d0` was then applied to a different level. When it
came out large, the posterior variance was the QC pools' technical variance
outright. On simulated null data with known truth (2,000 features, five seeds),
the share of null p-values below 0.05 was:

| Design | Trend from | Before v26.8.0 | v26.8.0 |
|---|---|---|---|
| unpaired, 6 vs 6 | QC pools | 40% | 4.9% |
| unpaired, 6 vs 6 | design groups (default) | 7.4% | 5.0% |
| paired, 6 subjects | QC pools | 18.5% | 4.7% |
| paired, 6 subjects | design groups (default) | 0.17% | 5.1% |

A calibrated test gives 5%. The paired default was the worst case for power: it
almost never found anything, false or true. These simulations are in
`tests/test_statistical_analysis.py::TestTrendCalibration`.

On a real longitudinal serum cohort (3,595 proteins, 10 subjects at weeks 0, 2, 4, 6 and 12, 12
reference and QC injections), 40 permutations of each null design gave:

| Null test | Trend from | Before v26.8.0 | v26.8.0 |
|---|---|---|---|
| unpaired 5 vs 5, random split of week 0 | QC pools | 6.1% | 4.6% |
| unpaired 5 vs 5 | design groups | 4.5% | 4.5% |
| paired week 0 vs 12, labels flipped within subject | QC pools | 7.0% | 5.8% |
| paired week 0 vs 12 | design groups | 4.4% | 5.5% |
| trend over weeks, shuffled within subject | QC pools | 5.1% | 4.8% |
| trend over weeks | design groups | 4.5% | 4.8% |
| trend, random weeks on week-0 samples (between-subject) | QC pools | 6.3% | 4.8% |
| trend, random weeks on week-0 samples | design groups | 6.2% | 4.8% |

The real cohort's `d0` is small (about 2.5 against 9 to 39 residual df), so its errors were
modest. The fitted level was about 2 for the QC pools and about 0.7 for paired and within-subject
design groups. Both paired rows stay slightly above 5% after the change, under either source, and
that residue is not explained by the prior.

**Known limitation, to revisit: one level for every intensity.** The level and `d0` are single
numbers, so the calibration can raise or lower the trend but cannot tilt it. When the shape comes
from samples whose noise has a different makeup from the study residual, the two differ in slope
as well as height. QC pools carry only technical noise, which falls steeply with intensity, while
a study residual adds biology that is roughly the same at every intensity. In the simulations
above, each case averaged about 5%, but split into intensity thirds (low / mid / high) the share
of null p-values below 0.05 was:

| Design | Trend from | Low | Mid | High |
|---|---|---|---|---|
| unpaired, 6 vs 6 | QC pools | 2.9% | 4.1% | 7.9% |
| paired, 6 subjects | QC pools | 3.2% | 4.3% | 6.7% |
| paired, 6 subjects | design groups (default) | 7.2% | 4.6% | 3.4% |

The fitted `d0` limits the damage, because a mismatched shape lowers it: 22 rather than about 170
in the unpaired QC case. On the serum cohort, where `d0` was about 2.5, the prior carries only
6-22% of each protein's variance, so the effect there should be much smaller. It has not been
measured on real data. In simulation, fitting a slope on `log(trend)` as well as the level removed
the gradient (4.2-5.6% in every third). Skyline-PRISM uses the same estimator and is pinned to this
one at 1e-9, so any change here has to be made in both together.

**What changes for an existing analysis.** P-values move, in a direction set by
the design and the source. A QC-sourced prior gets less optimistic. A paired or
within-subject analysis on design groups gets more powerful. An unpaired
analysis on design groups barely moves. To reproduce a result from an earlier
version, pin that version (`proteomics-toolkit==26.7.1`); there is no switch
back, because the old estimator is miscalibrated in both directions.

`ptk.plot_variance_vs_intensity(results)` draws the fitted trend solid and,
when the level is not 1, the prior actually used dashed. With
`intensity_peptide_trend` the dashed curve is the intensity stage at the
fitted level; each protein's peptide-count adjustment is shown in the right
panel.

### Taking the shape from QC or reference injections (`variance_prior_group_column`)

Dedicated technical replicates, such as pooled QC, inter-batch references or
system-suitability injections, trace the intensity dependence of the noise
without any biology in the way. To fit the trend's shape on them, name the
metadata column that identifies them, and optionally which of its values
count:

```python
config.variance_prior_group_column = 'Sample Category'
config.variance_prior_groups       = ['Inter-Experiment Reference', 'Inter-Batch Reference']
```

Each value is its own group, so two different materials are not pooled into
one variance. Include those samples in the `sample_metadata` passed to
`run_comprehensive_statistical_analysis`. They usually lack the design's
columns (no `Week`, no group label), so the design fit drops them, but the
prior still sees them.

This changes the shape only. The level is fitted to the design either way, so
pointing the prior at QC pools cannot make the residual noise look like
technical noise. Expect `intensity_trend_level` above 1 with this option: the
study's noise is higher than the pools' by the biology the pools lack.

### Covariate adjustment

**Use case:** Control for nuisance variables (age, sex, batch, BMI,
etc.) when comparing two groups. The treatment statistics (`logFC`,
`t`, `P.Value`, `adj.P.Val`) are reported on the treatment coefficient
after adjusting for the supplied covariates.

Supported for `analysis_type` in `'unpaired'`, `'linear_trend'`, and
`'interaction'`. Setting `config.covariates` with `paired` emits a warning and
is otherwise ignored by the moderated linear model (use the
`mixed_effects` path if you need covariates in those designs).

```python
config = ptk.StatisticalConfig()
config.analysis_type           = 'unpaired'
config.statistical_test_method = 'moderated_linear_model'
config.moderation              = 'intensity_trend'    # or 'limma' / 'deqms'
config.group_column            = 'Group'
config.group_labels            = ['Control', 'Treatment']

# Numeric covariates contribute one design column each.
# Object / category dtypes are reference-dummy-coded via patsy.
config.covariates              = ['Age', 'Sex', 'Batch']

config.log_transform_before_stats = 'auto'
config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    data, sample_meta_dict, config, protein_annotations=annot,
)
```

**Handling of missing values.** Samples missing any covariate value are
listwise-deleted before the per-feature fit. With
`moderation='intensity_trend'`, the variance prior is refit on the same
restricted sample set so the prior matches the coefficient estimates.

### Linear-trend mode (moderated slope test)

**Use case:** Longitudinal designs with three or more timepoints where
a 2-group paired comparison wastes data, but the categorical F-test
(`mixed_effects` + `longitudinal`) is over-conservative with many
timepoints. Fits a 1-df slope test per protein with the same limma /
deqms / intensity_trend variance moderation.

The design is `feature ~ intercept + Time + (optional subject one-hot block)`.
`logFC` in the output is the slope **per unit time**, so use a small
`fc_threshold` in volcano plots. When `moderation='intensity_trend'`,
every unique value of `time_column` contributes an anchor point per
feature to the LOWESS trend — five timepoints give 5× the leverage of
a 2-group paired fit.

```python
config = ptk.StatisticalConfig()
config.statistical_test_method = 'moderated_linear_model'
config.analysis_type           = 'linear_trend'
config.moderation              = 'intensity_trend'    # default; or 'limma' / 'deqms'

config.time_column    = 'Week'        # numeric column in metadata
config.subject_column = 'Subject'     # optional; enables limma-style
                                      # within-subject fixed-effect block
                                      # for repeated-measures designs

config.log_transform_before_stats = 'auto'
config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    data, sample_meta_dict, config, protein_annotations=annot,
)

# Volcano with a slope-appropriate FC threshold
ptk.plot_volcano(results, fc_threshold=0.01, p_threshold=0.05, label_top_n=15)

# Intensity-trend diagnostic shows one point per (feature, unique time value)
ptk.plot_variance_vs_intensity(results)
```

When to pick this over the mixed-effects `linear_trend` route (below):
the moderated path replaces REML inference with limma's empirical-Bayes
variance shrinkage, which is the dominant power gain on small-n MS data
when the variance prior is the bottleneck. The mixed-effects route is
preferable when subject variance is the primary nuisance of interest
or when the slope structure is more complex than a fixed effect.

### Interaction mode (moderated difference of differences)

**Use case:** A 2 x 2 design where the question is whether an effect
*changes* between two conditions: does the treatment-versus-control
difference differ between two visits, or does a collection-method
difference change with processing delay? Tests the interaction with the
same limma / deqms / intensity_trend variance moderation as the other
moderated designs.

The design is
`feature ~ intercept + group:level + group + level (+ optional subject block + covariates)`,
and the `group:level` coefficient is tested. `logFC` is the difference of
differences:

```text
logFC = (alt - ref at paired_label2) - (alt - ref at paired_label1)
```

where `ref`/`alt` are `group_labels[0]`/`group_labels[1]`. All four cells
must contain samples. Design columns that are aliased are dropped
automatically (for example, a `subject_column` block whose subjects are
nested in the group absorbs the group main effect), and the call raises if
the interaction itself is not estimable. With `moderation='intensity_trend'`
and no `variance_prior_group_column`, each of the four cells contributes an
anchor point per feature to the variance trend.

```python
config = ptk.StatisticalConfig()
config.statistical_test_method = 'moderated_linear_model'
config.analysis_type           = 'interaction'
config.moderation              = 'intensity_trend'

config.group_column  = 'Collection'
config.group_labels  = ['Venous', 'Capillary']   # ref, alt
config.paired_column = 'Hours'
config.paired_label1 = '1'                        # reference level
config.paired_label2 = '25'

config.subject_column = 'Subject'                 # optional
config.variance_prior_group_column = 'sample_type'   # optional QC/pool prior
config.variance_prior_groups       = ['reference']

config.log_transform_before_stats = 'auto'
config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    data, sample_meta_dict, config, protein_annotations=annot,
)
```

Unlike the mixed-effects `interaction` route, this path does not use
`interaction_terms`; the two factors and their levels come from
`group_column`/`group_labels` and `paired_column`/`paired_label1`/`paired_label2`.
Residual df is `n - 4` without a subject block or covariates.

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
| `interaction_terms` | list | Mixed-effects interaction terms (e.g. `['Group', 'Visit']`). Not used by the moderated `interaction` path, which reads `group_labels` and `paired_label1`/`paired_label2` instead |
| `covariates` | list | Additional covariates to control for (e.g. `['Age', 'Sex']`). Honored by `mixed_effects` for all designs, and by `moderated_linear_model` for `analysis_type` in `'unpaired'`, `'linear_trend'`, and `'interaction'`. See [Covariate adjustment](#covariate-adjustment). |
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
