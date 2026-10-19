# Release Notes (Next Release)

## Overview

Feature release with a breaking change to every default moderated p-value. The default variance
prior for `statistical_test_method="moderated_linear_model"` is now limma-trend (limma's
`eBayes(trend=TRUE)`), replacing the intensity trend, which is deprecated. The intensity trend was
calibrated on average but tilted by intensity: optimistic for bright features, conservative for
dim ones, and it found fewer true effects. `intensity_peptide_trend`, which adds a peptide-count
stage to the intensity trend, is deprecated too: the peptide stage keeps the tilt and adds one
by peptide count. Moderated p-values are also now read on limma's capped
degrees of freedom, for every prior. Both are pinned to R limma 3.66.0, and both match
Skyline-PRISM dotnet-v26.29.0, so the two tools agree on default runs again.

## New Features

- **`moderation="limma_trend"`, the new default.** The prior variance follows a natural-spline
  curve of the residual log-variance on each feature's mean log abundance (`AveExpr`), fitted to
  the design's own residuals, and the prior df is estimated around that curve. This is limma's
  `fitFDist` with a covariate, with its spline df (1 to 4, by the number of features). Nothing
  comes from QC or reference samples, so the curve follows whatever mix of technical and
  biological variance the design leaves. It works at protein and peptide level, for unpaired,
  paired, linear-trend and interaction designs, with covariates and `robust=True`. Results carry
  the prior used per feature as `limma_trend_s0_sq`, with `limma_s0_sq` as the global prior for
  comparison.
- **New `total_df` column** on every moderated result: the degrees of freedom the p-value was
  read on (see Bug Fixes). `posterior_df` keeps `residual_df + d0`.

## Bug Fixes

- **Moderated p-values were read on uncapped degrees of freedom, so they were smaller than
  limma's wherever the cap binds.** limma's `eBayes` caps the moderated t's df at the residual
  df pooled over every feature, `pmin(df.residual + df.prior, sum(df.residual))`. The toolkit
  used `residual_df + d0` uncapped, for every prior, so with an infinite prior df it read the t
  off a normal distribution. The cap binds whenever the prior df is infinite, and otherwise only
  when few features are tested and the prior df is large. For example, 8 features at 4 residual
  df with an infinite prior df: a t of 5.49 gave p = 4.0e-8, and gives 4.8e-6 on the capped 32
  df. On 3,000 features at 8 df the cap moves a p-value by 0.1% at t = 3 and 4.5% at t = 8.
  Skyline-PRISM made the same fix in dotnet-v26.29.0 (maccoss/skyline-prism#137).

## Breaking Changes

- **The default prior changed, so every default moderated p-value changes.** A config that does
  not set `moderation` now runs `"limma_trend"`. On random-label nulls the intensity trend gave
  6-9% of p-values below 0.05 for the brightest fifth of features and 2-5% for the dimmest,
  where a calibrated test gives 5% in both (a 327-peptide cohort, a 3,595-protein serum cohort,
  and simulations; unpaired, paired and trend designs). limma-trend gave 4.7-6.0% in both the
  dimmest and the brightest fifth of every case. The intensity trend also roughly halved the
  prior df and lost power: in simulation at 5% FDR it found 52% of true two-fold effects against
  limma-trend's 71% (unpaired 5 vs 5), and 89% against 95% (paired, 6 subjects). Set
  `moderation="intensity_trend"` to reproduce an earlier result. Its estimator is unchanged; only
  its p-values move, and only where the df cap binds.
- **`moderation="intensity_trend"` is deprecated** and emits a `FutureWarning` saying why. It is a
  `FutureWarning` rather than a `DeprecationWarning` because Python hides a `DeprecationWarning`
  raised inside a library, and notebook users are who need to see it.
- **`moderation="intensity_peptide_trend"` is deprecated too**, with its own `FutureWarning`. Its
  peptide-count stage is shaped on the same QC/reference pools or design groups as its intensity
  stage, so it inherits the intensity trend's mismatch instead of correcting it: pooled technical
  noise averages down over a protein's peptides, while the biology in a study residual does not.
  On a simulated protein-level null (peptide counts 1-64, technical variance falling with
  intensity and as 1/count, flat biology), a pool-sourced shape gave 2.8% of null p-values below
  0.05 for the dimmest fifth of proteins and 6.2% for the brightest, 3.0% for the fifth with the
  fewest peptides and 6.7% for the most, the smallest prior df of any prior tested (2.1), and
  found 55% of true two-fold effects at 5% FDR against limma-trend's 69% (unpaired 5 vs 5). A
  design-group shape was calibrated unpaired but tilted the other way paired (8.4% dimmest, 2.9%
  brightest, 6 subjects), as the intensity trend's does. Use it only to reproduce an earlier
  result. The same simulation found that limma-trend, calibrated by intensity in every case
  (4.1-5.7% per fifth), does not use peptide counts and so tilts by them where variance falls
  with them: 7.8% of null p-values below 0.05 for the fewest-peptide fifth and 2.6% for the most
  (unpaired), 9.0% and 2.3% (paired). DEqMS is the reverse, calibrated by count and tilted by
  intensity. A prototype limma-trend with log(peptide count) as a second covariate, fitted to
  the residuals, was calibrated both ways (4.0-6.0%); it is not in this release.
- **`variance_prior_group_column` now warns when it has no effect.** It only ever shaped the
  intensity-trend priors. Until this release that was the default, so a config that sets it was
  written for that prior; under any other moderation it is now ignored with a `UserWarning`
  rather than silently.
- **P-values change wherever the df cap binds, under every prior** (see Bug Fixes). The t
  statistics, posterior variances and prior df are unchanged.

## Testing

- `TestLimmaEbayesGolden` pins the moderated t to R limma 3.66.0's own `lmFit` + `eBayes`, called
  whole: `tests/fixtures/limma_ebayes.json`, written by `tests/fixtures/generate_limma_ebayes.R`,
  which imports nothing from the toolkit. Ten cases cover limma-trend at all three spline df
  (2, 3, 4), finite and infinite prior df, unpaired, paired and linear-trend designs, missing
  values (`legacy=TRUE`), and the global prior. Four of them make the df cap bind, and the rest
  show it changes nothing when it does not. Measured agreement is 1e-11 or better; the stated
  tolerances (1e-12 to 1e-9) are explained in the test's docstring.
- `TestPriorCalibrationByIntensity` reruns a seeded null with real data's variance structure
  (technical noise falling with intensity, flat biology; 6 draws of 2,000 features, unpaired 5
  vs 5) and checks calibration in each intensity fifth, which an overall 5% rate cannot see.
  limma-trend must stay within 3.5-6.5% in every fifth, and the pool-shaped intensity trend must
  still tilt by more than 3 points. Marked `slow` (about 35 s).
- `TestIntensityPeptideTrendCalibration` reruns the protein-level null (`_protein_calibration_null`,
  6 draws of 2,000 proteins with peptide counts) and pins the three facts the
  `intensity_peptide_trend` deprecation rests on: the pool-shaped prior tilts by intensity and by
  peptide count (unpaired), the design-group-shaped one tilts the other way (paired), and
  limma-trend stays within 3.5-6.5% in every intensity fifth while tilting by peptide count. Each
  tilt bound is 2 points, more than three binomial sd. Marked `slow` (about 90 s together).
- `TestTrendCalibration` checks limma-trend's overall null rate on its unpaired and paired
  simulations, beside the intensity trend's.
- New tests for the default, both deprecation warnings and their reasons, the warning for an ignored
  `variance_prior_group_column`, and limma-trend with `robust=True`.

## Documentation

- `docs/06-statistical-analysis.md` explains why limma-trend is the default, with the
  null-calibration measurements by intensity fifth, and documents limma's df cap and the new
  `total_df` column. The intensity trend's sections are marked deprecated, and the examples use
  limma-trend. The one deliberate difference from limma is recorded: with missing values limma
  3.58+ switches to `fitFDistUnequalDF1`, and the toolkit keeps `fitFDist`'s moments.
- `docs/06-statistical-analysis.md` has a new section, "Does the peptide-count stage fix it?",
  with the protein-level null measured by intensity fifth and by peptide-count fifth for every
  prior, unpaired and paired, from pools and from design groups, with power and prior df, and two
  sensitivity checks (biology falling with peptide count too; count tracking intensity). The
  peptide-count prior's earlier cross-validation figures are kept, with a note that they measure
  fit to technical replicates, not calibration on study samples.
- The README, the overview, the QC-plot and pitfalls pages, and the tutorial notebook use
  limma-trend as the default. `plot_variance_vs_intensity` is documented as a diagnostic for the
  intensity priors only, and the docs and tutorial show limma's `plotSA` drawn by hand instead.
