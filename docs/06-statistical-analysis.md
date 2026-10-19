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
- [Moderated linear model](#moderated-linear-model) (empirical Bayes variance shrinkage; limma-trend by default, or limma / deqms, or the deprecated intensity_peptide_trend / intensity_trend)
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

## Moderated linear model

**Use case:** Small sample sizes (fewer than ~6 replicates per group)
where raw per-feature t-statistics are under-powered. A single entry
point `run_moderated_linear_model` runs the Smyth per-feature OLS fit
and applies one of five empirical-Bayes variance priors selected via
`config.moderation`:

| Moderation | Prior shape | When to pick |
|---|---|---|
| `"limma_trend"` *(default)* | limma's `eBayes(trend=TRUE)`: a natural-spline curve of the residual `log(variance)` on each feature's mean log abundance (`AveExpr`), fitted to the design's own residuals, with the prior df estimated around it. | The default for MS data, at protein *and* peptide level. Calibrated at every intensity in every design tested ([below](#why-limma-trend-is-the-default)). Pinned to R limma. It does not use peptide counts, and in simulation and on a serum cohort it tilted by them ([below](#does-the-peptide-count-stage-fix-it)). |
| `"limma"` | Single global prior (Smyth 2004). | Use when the variance-intensity trend is flat, or as a conservative baseline. |
| `"deqms"` | Prior conditioned on peptide count alone (Zhu et al. 2020). | Protein-level only. Calibrated by peptide count but tilted by intensity, in simulation and on a serum cohort ([below](#does-the-peptide-count-stage-fix-it)); see also the caveat below. |
| `"intensity_peptide_trend"` *(deprecated)* | Additive two-stage LOWESS: `log(var) = f1(log mean intensity) + f2(log peptide count)`, the shape from groups of samples and the level fitted to the residuals. | Only to reproduce an earlier result; it warns when used. The peptide stage keeps the intensity trend's tilt and, from the pools, adds one by peptide count ([below](#does-the-peptide-count-stage-fix-it)). |
| `"intensity_trend"` *(deprecated)* | LOWESS of `log(variance)` on `log(mean intensity)` over groups of samples (QC/reference pools, or the design groups) for the shape, scaled to the design's residuals ([below](#where-the-intensity-priors-level-and-weight-come-from)). | Only to reproduce an earlier result; it warns when used. Calibrated on average but tilted by intensity ([below](#why-limma-trend-is-the-default)). The default until 26.9.0. |

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
flat prior. It helps only in addition to intensity, never instead of it - which
is also what the null simulation [below](#does-the-peptide-count-stage-fix-it)
finds for `"deqms"`: calibrated by peptide count, tilted by intensity.

These RMSEs measure how well each curve predicts the variance of technical
replicates - the noise the pools carry. They do not say whether a test built
on the curve is calibrated on study samples, which also carry biology; that
is the question [Why limma-trend is the default](#why-limma-trend-is-the-default)
and [Does the peptide-count stage fix it?](#does-the-peptide-count-stage-fix-it)
answer, and where both intensity priors fall short. Peptide count still
belongs in a protein-level prior; what goes wrong is taking the curve's shape
from samples that are not the residuals being tested.

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
estimating the prior hyperparameters. This has the intent of limma's
`robust=TRUE` but not its algorithm: the toolkit Winsorizes the
log-variances' departures from the prior curve (or mean) at four
MAD-scaled deviations, where limma uses `fitFDistRobustly`. It prevents
a handful of genuinely high-variance features from inflating the prior.

```python
# limma_trend (default) on protein- or peptide-level data.
config = ptk.StatisticalConfig()
config.analysis_type           = 'unpaired'
config.statistical_test_method = 'moderated_linear_model'
config.moderation              = 'limma_trend'        # default
config.robust                  = False                # optional
config.group_column            = 'Group'
config.group_labels            = ['Control', 'Treatment']
config.log_transform_before_stats = True
config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    data, sample_meta_dict, config, protein_annotations=annot
)

# The prior the test used against each feature's own residual variance -
# limma's plotSA, by hand.
ax = results.plot.scatter('AveExpr', 'residual_s2', s=4, logy=True)
results.sort_values('AveExpr').plot('AveExpr', 'limma_trend_s0_sq', ax=ax, color='red')
```

`ptk.plot_variance_vs_intensity(results)` draws the intensity priors'
LOWESS and needs a result from `intensity_trend` or
`intensity_peptide_trend`, both deprecated.

To reproduce an earlier result with the deprecated combined intensity +
peptide prior on PRISM protein data (PRISM emits `n_peptides` in the protein
parquet):

```python
# Include n_peptides alongside the annotation and sample columns.
# Deprecated, and warns: tilted by intensity and by peptide count (see
# "Does the peptide-count stage fix it?" below).
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
`posterior_s2`, `posterior_df` (`residual_df + d0`), `total_df` (the df the
p-value is read on; see [Degrees of freedom](#degrees-of-freedom-limmas-cap)),
`limma_s0_sq`, plus one of `limma_trend_s0_sq` (the prior variance used,
per feature), `deqms_s0_sq` + `peptide_count_used`, or
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

### Why limma-trend is the default

Until 26.9.0 the default was the intensity trend, with its shape taken from
the design groups or, with `variance_prior_group_column`, the QC and reference
pools. Its overall false-positive rate was right - about 5% of null p-values
below 0.05 - but split by intensity it was not: optimistic for bright
features and conservative for dim ones, the two cancelling in the average.
The calibration work for 26.8.0 (below) was all measured overall, which is
how it went unseen.

The cause is the curve's shape. On the log2 scale the moderated t works on,
a feature's variance is roughly (CV / ln 2)^2. The pools carry technical noise
only: its absolute SD rises with intensity (shot noise goes as about the
square root of the signal), so its CV - the log-scale noise - falls, towards a
floor set by multiplicative sources such as injection and integration. Study
samples add biological variation, which is multiplicative - a roughly
constant CV - and so close to flat on the log scale. One scale factor cannot
fit a falling curve to a flatter one: scaled to fit on average, the pool
curve sits too low at high intensity and too high at low. The mismatch also
roughly halves the fitted prior df, so the prior moderates less and the test
loses power. limma-trend has no source to get wrong: it fits its curve to the
contrast's own residual variances, so it follows whatever mix of technical
and biological variance the design leaves.

The share of null p-values below 0.05 in the dimmest and brightest fifth of
features (a calibrated test gives 5% in both), from random-label nulls -
splits, label swaps within subject, or weeks shuffled within subject. These
were measured with Skyline-PRISM, whose intensity trend is pinned to this
toolkit's at 1e-9 and whose limma-trend is pinned to inmoose and R limma, as
this toolkit's now is:

| Data, design | intensity trend from pools | from design groups | limma-trend |
|---|---|---|---|
| 327-peptide cohort, unpaired 3 vs 3 | 2.4% -> 8.9% | 6.2% -> 4.7% | 5.0% -> 5.3% |
| 327-peptide cohort, between-subject trend, 10 samples | 3.5% -> 8.0% | n/a | 5.2% -> 6.0% |
| serum cohort (3,595 proteins), paired week 0 vs 12 | 4.6% -> 7.0% | 6.1% -> 3.7% | 5.2% -> 5.6% |
| serum cohort, within-subject trend over weeks 0-12 | 5.0% -> 6.2% | n/a | 5.2% -> 5.7% |
| serum cohort, unpaired 5 vs 5 (week-0 samples) | 3.8% -> 6.9% | 5.0% -> 4.5% | 4.7% -> 5.1% |
| simulated, paired, 6 subjects | 2.1% -> 9.4% | 9.2% -> 2.7% | 5.1% -> 5.3% |

The design-group shape is no fix: in a paired design the groups carry the
between-subject spread that the subject block removes, so it tilts the other
way, and a trend design has no groups at all. Where a paired or
within-subject design's residual really is technical only - repeat draws
from a person that differ only by measurement - the pool shape is right, and
simulation shows it flat at 5% there; real people vary from visit to visit,
and the serum rows show it.

Power, simulated at 5% FDR with a two-fold effect in 10% of features: the
pool-shaped prior found 52% of true effects against limma-trend's 71%
(unpaired 5 vs 5; 30% against 58% of the dim ones), and 89% against 95%
(paired, 6 subjects), with its false hits piled on bright features.

`TestPriorCalibrationByIntensity` in `tests/test_statistical_analysis.py`
reruns the simulated unpaired case on every test run: limma-trend must stay
within 3.5-6.5% in every intensity fifth, and the pool-shaped trend must
still tilt by more than 3 points (if it ever stops, the reason for the
deprecation needs revisiting).

limma-trend here is limma's own estimator, pinned to R limma 3.66.0's
`lmFit` + `eBayes(trend=TRUE)` by `tests/fixtures/limma_ebayes.json`
(`generate_limma_ebayes.R`). One difference is deliberate: with missing
values, features have different residual df, and limma 3.58 and later then
switch to `fitFDistUnequalDF1`; the toolkit keeps `fitFDist`'s moments, as
limma did before 3.58 (`eBayes(legacy=TRUE)`), which is what the golden pins
for that case. With complete data the two are the same.

The intensity trend is still available, so an earlier result can be
reproduced, and warns when used.

### Does the peptide-count stage fix it?

No. `intensity_peptide_trend` adds a second LOWESS stage, on log(peptide
count), to the intensity trend, and was not deprecated with it in 26.9.0
because its calibration by intensity had not been measured. It now has, and it
is deprecated too: it keeps the intensity trend's tilt, and with a pool-sourced
shape it adds one by peptide count.

The reason is the same as for the intensity trend, in a second direction.
Both stages are fitted on the same groups of samples, with only the level and
`d0` fitted to the residuals. A protein's technical noise averages down over
its peptides, about as `1/n` for `n` peptides, and that is what the pools
show - the -0.89 coefficient [above](#moderated-linear-model). But a protein's
biology is a property of the protein, not of its peptides, and does not
average down, so a study residual falls more slowly with peptide count than
the pools do. Scaled to fit on average, the pools' peptide curve sits too low
for proteins with many peptides and too high for those with few.

**The simulation.** `_protein_calibration_null` in
`tests/test_statistical_analysis.py` extends `_calibration_null`, the null
`TestPriorCalibrationByIntensity` uses, to protein level:
each protein has a peptide count (log-uniform from 1 to 64, median about 8,
independent of intensity), technical variance that falls with intensity as
before and is divided by the count (scaled so a protein at the median count
matches the peptide-free null), and biology that does not depend on the count.
Unpaired is 5 vs 5 with between-subject biology; paired is 6 subjects measured
twice, with a subject effect the model removes and visit-to-visit biology
(half the between-subject variance) that it does not. Eight reference and
eight QC injections carry the technical variance only. Six draws of 2,000
proteins give 2,400 per fifth, so a 5% rate has a binomial sd of 0.44 points.
Each row's null rates and prior df are one call to `_protein_null_rates`;
power is the share of true two-fold effects found at 5% FDR (Benjamini-Hochberg)
with one in ten proteins changed, from `_protein_calibration_null(...,
effect_fraction=0.1)` at seeds 20261100-20261105.

The share of null p-values below 0.05 in the dimmest and brightest fifth of
proteins, and in the fifth with the fewest and the most peptides (a calibrated
test gives 5% in all four):

| Unpaired, 5 vs 5 | intensity: dimmest -> brightest | peptides: fewest -> most | prior df | power (dimmest fifth) |
|---|---|---|---|---|
| `limma_trend` | 4.8% -> 4.7% | 7.8% -> 2.6% | 6.3 | 68.9% (39.1%) |
| `deqms` | 6.8% -> 3.1% | 4.6% -> 4.1% | 4.7 | 63.8% (46.0%) |
| `intensity_trend`, from pools | 2.7% -> 7.5% | 7.0% -> 3.9% | 3.0 | 57.5% (16.5%) |
| `intensity_trend`, from design groups | 5.2% -> 4.5% | 7.8% -> 2.6% | 6.1 | 68.8% (40.3%) |
| `intensity_peptide_trend`, from pools | 2.8% -> 6.2% | 3.0% -> 6.7% | 2.1 | 54.9% (26.6%) |
| `intensity_peptide_trend`, from design groups | 4.9% -> 4.7% | 4.5% -> 4.7% | 9.5 | 71.6% (46.8%) |
| prototype: limma-trend with peptide count (not in the toolkit) | 4.4% -> 4.8% | 4.3% -> 4.9% | 10.0 | 72.0% (46.4%) |

| Paired, 6 subjects | intensity: dimmest -> brightest | peptides: fewest -> most | prior df | power (dimmest fifth) |
|---|---|---|---|---|
| `limma_trend` | 5.7% -> 4.4% | 9.0% -> 2.3% | 5.1 | 84.4% (55.6%) |
| `deqms` | 8.8% -> 1.5% | 4.8% -> 4.3% | 3.5 | 80.0% (58.0%) |
| `intensity_trend`, from pools | 3.3% -> 7.4% | 8.0% -> 3.4% | 3.2 | 72.2% (22.4%) |
| `intensity_trend`, from design groups | 7.2% -> 3.5% | 8.8% -> 2.4% | 4.5 | 84.2% (59.6%) |
| `intensity_peptide_trend`, from pools | 2.5% -> 6.7% | 2.5% -> 7.8% | 2.6 | 71.2% (36.0%) |
| `intensity_peptide_trend`, from design groups | 8.4% -> 2.9% | 7.0% -> 3.2% | 6.9 | 86.2% (65.2%) |
| prototype: limma-trend with peptide count (not in the toolkit) | 5.7% -> 4.1% | 4.9% -> 5.0% | 10.3 | 85.9% (62.4%) |

Every overall rate in these tables was between 4.4% and 5.3%, which is why
none of this shows without splitting. Read it by source, as for the intensity
trend:

- **From the pools,** `intensity_peptide_trend` is tilted by intensity in both
  designs, about as much as the intensity trend (3.4 and 4.2 points from
  dimmest to brightest, against its 4.8 and 4.1), and by peptide count the other
  way round from it: optimistic for proteins with many peptides. It also has
  the smallest prior df and the least power of anything tested.
- **From the design groups** it is calibrated in an unpaired design, where the
  groups' spread *is* the residual, and the best of the toolkit's priors
  there in this simulation. In a paired design the groups carry the between-subject spread the
  subject block removes, and it tilts the other way, as the intensity trend
  does: 8.4% of null p-values below 0.05 in the dimmest fifth. Its paired
  power is the highest in the table partly for that reason - a prior that is
  optimistic for dim proteins finds more of them, true and false.

Two checks on how much this depends on the simulation's choices. Dividing the
biology by the peptide count too - the case most favorable to the pools'
peptide curve, since it then has the residuals' slope - removes the
peptide-count tilt and makes the intensity tilt worse: 1.8% -> 7.5% unpaired
and 1.8% -> 8.1% paired from the pools, and 9.5% -> 2.4% paired from design
groups. Making log peptide count track intensity (r = 0.57, against 0.34 on
real data) leaves it much the same: 1.8% -> 7.4% and 1.3% -> 8.5% from the
pools, 7.9% -> 3.8% paired from design groups, and a slight tilt (3.8% ->
6.0%) even unpaired from design groups.

**On a real cohort.** The same comparison on the serum cohort used
[above](#why-limma-trend-is-the-default): PRISM's corrected protein matrix,
3,595 protein groups with a median of 4 peptides (23% have one; log peptide
count correlates 0.41 with intensity), 10 subjects on verapamil sampled at
weeks 0, 2, 4, 6 and 12, 11 on liraglutide, and 6 reference and 6 QC
injections. Three random-label nulls, 40 permutations each: unpaired 5 vs 5
(random splits of the week-0 samples), unpaired 10 vs 11 (random splits of
both studies' pre-drug samples), and paired week 0 vs 12 with the labels
flipped at random within each subject. Peptide count is reported in bins
rather than fifths here, because so many proteins tie at one peptide:

| Null, prior | intensity: dimmest -> brightest fifth | peptides: 1 -> 10 or more | prior df |
|---|---|---|---|
| **unpaired 5 vs 5** | | | |
| `limma_trend` | 5.0% -> 4.7% | 6.9% -> 3.0% | 3.4 |
| `deqms` | 5.5% -> 3.8% | 3.9% -> 3.9% | 2.6 |
| `intensity_peptide_trend`, from pools | 4.4% -> 5.9% | 5.2% -> 4.4% | 4.2 |
| `intensity_peptide_trend`, from design groups | 4.5% -> 5.0% | 5.1% -> 4.1% | 4.9 |
| prototype: limma-trend with peptide count | 4.9% -> 4.9% | 4.4% -> 4.2% | 5.2 |
| **paired, week 0 vs 12** | | | |
| `limma_trend` | 5.0% -> 4.6% | 6.8% -> 3.1% | 4.2 |
| `deqms` | 5.5% -> 3.0% | 3.8% -> 4.0% | 2.5 |
| `intensity_peptide_trend`, from pools | 5.1% -> 4.4% | 5.3% -> 4.2% | 6.0 |
| `intensity_peptide_trend`, from design groups | 5.7% -> 3.1% | 5.7% -> 3.5% | 5.3 |
| prototype: limma-trend with peptide count | 4.9% -> 4.6% | 4.3% -> 4.2% | 6.7 |
| **unpaired 10 vs 11** | | | |
| `limma_trend` | 5.2% -> 6.4% | 6.3% -> 5.1% | 4.5 |
| `deqms` | 5.6% -> 5.9% | 4.8% -> 5.7% | 3.4 |
| `intensity_peptide_trend`, from pools | 4.6% -> 7.4% | 5.1% -> 6.4% | 4.5 |
| `intensity_peptide_trend`, from design groups | 5.2% -> 6.5% | 5.5% -> 5.7% | 5.7 |
| prototype: limma-trend with peptide count | 5.4% -> 6.5% | 5.1% -> 5.9% | 6.2 |

Permutations of one cohort share its proteins, so the uncertainty is the
spread across permutations, not a binomial one: the standard error of a
fifth's or a bin's rate is 0.1-0.3 points in the unpaired 5 vs 5 null, up to
0.7 paired and up to 1.2 in the 10 vs 11 null, largest for bright and
many-peptide proteins. It measures permutation noise on this cohort, not how
another cohort would differ. The 10 vs 11 null runs a little high for every
prior (5.2-5.5% overall, 6.4-7.4% in the brightest fifth), most likely because
a random split mixes two studies' subjects unevenly; it is the weakest of the
three.

The real data agree with the simulation on what matters. Peptide count
carries variance information that intensity does not: at matched intensity,
the residual variance of a one-peptide protein is 2.3-2.9 times that of a
four-peptide one, levelling off at about 0.6-0.7 times from ten peptides on,
where biology dominates, and a spline in log peptide count removes 33-43% of
the spread in log residual variance that the intensity curve leaves (an
intensity x count interaction removes only 1-3 points more). limma-trend, which
does not use it, tilts by it: the rate for one-peptide proteins exceeds the
rate for those with ten or more by 3.9 points (standard error 0.24) unpaired
5 vs 5 and 3.7 (0.50) paired, and by 1.3 (0.90) in the 10 vs 11 null. The
pool-shaped `intensity_peptide_trend` tilts by intensity in both unpaired
nulls (4.4% -> 5.9% and 4.6% -> 7.4%) though not in the paired one here, and
the design-group shape tilts the other way paired (5.7% -> 3.1%), as in
simulation; the peptide-count tilt the simulation shows for the pool shape is
not visible on this cohort. DEqMS is again calibrated by peptide count and
tilted by intensity. The prototype was within half a point of flat in both
directions in the two stronger nulls, with prior df 50-60% above
limma-trend's.

A spike-in on the same splits - one protein in ten shifted, Benjamini-Hochberg
at 5%, 20 permutations - puts a number on what the prototype would buy. At a
two-fold change it found 28.0% of true effects against limma-trend's 23.0%
unpaired 5 vs 5, 67.5% against 66.7% paired, and 52.5% against 52.1% unpaired
10 vs 11; at a 1.5-fold change, 22.9% against 17.2% paired (the unpaired 5 vs
5 split finds almost nothing at 1.5-fold under any prior). The realized false
discovery proportion stayed at or under 5.3% for every prior, except in that
1.5-fold unpaired 5 vs 5 case, where each prior called only a handful of
proteins and the proportion means little. The gain is in
proteins with many peptides (ten or more, unpaired 5 vs 5 at two-fold: 55.0%
against 40.5%), and one-peptide proteins give some up (2.9% against 7.9%) -
the proteins where limma-trend's null rate was 6.9%, so part of what it found
there was false.

**What this says about the other priors.** limma-trend was calibrated by
intensity in every case (4.1-5.7% in every fifth of every variant), but it
does not use peptide counts, so where variance falls with them it tilts by
them: optimistic for proteins with few peptides, conservative for those with
many (7.8% -> 2.6% unpaired, 9.0% -> 2.3% paired; 7.1% -> 1.4% and 9.2% ->
0.8% when the biology falls with the count too). The intensity trend, which
does not use peptide counts either, does the same. DEqMS is the mirror image:
calibrated by peptide count and tilted by intensity (6.8% -> 3.1% unpaired,
8.8% -> 1.5% paired). The serum cohort above shows the same limma-trend and
DEqMS tilts on real data. limma-trend remains the default: it is calibrated by
intensity at both protein and peptide level, and nothing in the toolkit yet
corrects its peptide-count tilt without introducing another.

The prototype rows are limma's `fitFDist` with log(peptide count) as a second
covariate beside the intensity spline (a natural spline of 3 df in log count;
a straight line did as well except where count tracked intensity, where it
left 7.4% in the fewest-peptide fifth, paired), both fitted to the design's
own residuals, so nothing comes from pools or groups. It was flat in both
directions in every simulated variant (4.0-6.0% in every fifth) and on the
serum cohort, had the largest prior df, and matched or beat the best power. It
is not in the toolkit. If it is added, it is a protein-level prior only: a
peptide has no peptide count, and the prior must refuse peptide-level input
rather than run on it.

`TestIntensityPeptideTrendCalibration` pins the three facts the deprecation
and this section rest on: the pool-shaped `intensity_peptide_trend` tilts by
intensity and by peptide count (unpaired), the design-group-shaped one tilts
the other way paired, and limma-trend is calibrated by intensity on this null
but tilts by peptide count. Each bound is 2 points, more than three binomial
sd of a difference between two fifths.

### Degrees of freedom: limma's cap

The moderated t's p-value is read on limma's `df.total`:

```
total_df = min(residual_df + d0, sum of residual_df over every feature fitted)
```

Uncapped, an infinite `d0` reads the t off a normal distribution, claiming
more certainty than all the residuals together hold. The cap binds whenever
`d0` is infinite, and otherwise only when few features are tested and `d0` is
large. Before 26.9.0 the toolkit used `residual_df + d0` uncapped, for every
moderation, so its p-values were smaller than limma's where the cap binds.
For example, 8 features at 4 residual df with an infinite `d0`: a t of 5.49
gave p = 4.0e-8; on the capped 32 df it is 4.8e-6. On 3,000 features at 8 df
the cap moves a p-value by 0.1% at t = 3 and 4.5% at t = 8. `posterior_df`
keeps the uncapped value (it is the posterior variance's df), and `total_df`
is the one the p-value used.

### Where the intensity prior's level and weight come from

This describes the deprecated `intensity_trend` (and the deprecated `intensity_peptide_trend`).
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

**Why it was deprecated: one level for every intensity.** The level and `d0` are single
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
measured on real data at the time. It since has been, and the tilt is there too (see
[Why limma-trend is the default](#why-limma-trend-is-the-default)). In simulation, fitting a slope
on `log(trend)` as well as the level removed the gradient (4.2-5.6% in every third), and a
prototype that kept the pool curve and added a fitted flat floor matched limma-trend but never beat
it, so the default moved to limma-trend instead, in this toolkit (26.9.0) and Skyline-PRISM
(dotnet-v26.29.0) together. The intensity trend itself is unchanged, so both tools still reproduce
earlier results with it.

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

Used only by the intensity-trend priors. limma-trend fits its prior to the
design's own residuals and needs no QC or reference injections; setting this
with any other moderation warns that it is ignored.

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
config.moderation              = 'limma_trend'        # default; or 'limma' / 'deqms'
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
listwise-deleted before the per-feature fit. The variance prior is fitted to
the residuals of that restricted fit (and with the intensity priors, on the
same restricted sample set), so the prior matches the coefficient estimates.

### Linear-trend mode (moderated slope test)

**Use case:** Longitudinal designs with three or more timepoints where
a 2-group paired comparison wastes data, but the categorical F-test
(`mixed_effects` + `longitudinal`) is over-conservative with many
timepoints. Fits a 1-df slope test per protein with the same variance
moderation as the other designs (limma-trend by default).

The design is `feature ~ intercept + Time + (optional subject one-hot block)`.
`logFC` in the output is the slope **per unit time**, so use a small
`fc_threshold` in volcano plots. With the deprecated
`moderation='intensity_trend'`, every unique value of `time_column`
contributes an anchor point per feature to the LOWESS trend.

```python
config = ptk.StatisticalConfig()
config.statistical_test_method = 'moderated_linear_model'
config.analysis_type           = 'linear_trend'
config.moderation              = 'limma_trend'        # default; or 'limma' / 'deqms'

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
same variance moderation as the other moderated designs (limma-trend by
default).

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
the interaction itself is not estimable. With the deprecated
`moderation='intensity_trend'` and no `variance_prior_group_column`, each of
the four cells contributes an anchor point per feature to the variance trend.

```python
config = ptk.StatisticalConfig()
config.statistical_test_method = 'moderated_linear_model'
config.analysis_type           = 'interaction'
config.moderation              = 'limma_trend'        # default

config.group_column  = 'Collection'
config.group_labels  = ['Venous', 'Capillary']   # ref, alt
config.paired_column = 'Hours'
config.paired_label1 = '1'                        # reference level
config.paired_label2 = '25'

config.subject_column = 'Subject'                 # optional

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
| `moderation` | str | When `statistical_test_method='moderated_linear_model'`: `'limma_trend'` *(default)*, `'limma'`, `'deqms'`, or the deprecated `'intensity_peptide_trend'` and `'intensity_trend'` |
| `robust` | bool | Winsorize extreme `s_i²` when estimating prior hyperparameters (the intent of limma's `robust=TRUE`, not its algorithm). Default `False`. |
| `variance_prior_group_column` | str | Intensity-trend priors only: metadata column whose values group the QC/reference samples the trend's shape is fitted on. Ignored, with a warning, by other moderations. |
| `variance_prior_groups` | list | Intensity-trend priors only: which values of `variance_prior_group_column` to use. |
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
