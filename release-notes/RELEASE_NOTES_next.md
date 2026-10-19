# Release Notes (Next Release)

## Overview

Feature release with a breaking change to every default moderated p-value. The default variance
prior for `statistical_test_method="moderated_linear_model"` is now limma-trend (limma's
`eBayes(trend=TRUE)`), replacing the intensity trend, which is deprecated. The intensity trend was
calibrated on average but tilted by intensity: optimistic for bright features, conservative for
dim ones, and it found fewer true effects. Moderated p-values are also now read on limma's capped
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
- **A paired or within-subject feature with a subject entirely missing was dropped, or given a
  wrong result.** The per-feature fit inverted the design restricted to the observed samples,
  and a subject with no value for a feature leaves that design singular. Usually the inverse
  failed and the feature vanished from the results. When the missing subject was the one the
  intercept stands for, the inverse was numerically singular instead, and the feature got a
  meaningless answer: on a 6-subject paired test set, a protein with a real change came back
  t = 0.03, p = 0.98, on 3 residual df where 4 was right. That bad variance also entered the
  prior every other feature was shrunk toward. Each feature is now fitted as limma's `lmFit`
  fits one with missing values: on its observed samples, with the design columns they can
  estimate (left to right, as R's `lm.fit` pivots). A subject absent from a feature drops out of
  that feature's comparison. A feature missing a whole arm reports NaN for its contrast, as
  limma's is NA, but its residual variance and df now go into the prior and the pooled df, as in
  limma. `AveExpr` is reported for every feature with an observed value. All of this is pinned
  to R limma 3.66.0 (`eBayes(legacy = TRUE)`, since missing values make the residual df unequal).
  With complete data nothing changes.

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
  `"intensity_peptide_trend"` is not deprecated, but its intensity stage is built the same way
  and its calibration by intensity has not been measured.
- **`variance_prior_group_column` now warns when it has no effect.** It only ever shaped the
  intensity-trend priors. Until this release that was the default, so a config that sets it was
  written for that prior; under any other moderation it is now ignored with a `UserWarning`
  rather than silently.
- **P-values change wherever the df cap binds, under every prior** (see Bug Fixes). The t
  statistics, posterior variances and prior df are unchanged.

## Testing

- `TestLimmaEbayesGolden` pins the moderated t to R limma 3.66.0's own `lmFit` + `eBayes`, called
  whole: `tests/fixtures/limma_ebayes.json`, written by `tests/fixtures/generate_limma_ebayes.R`,
  which imports nothing from the toolkit. Fourteen cases cover limma-trend at all three spline
  df (2, 3, 4), finite and infinite prior df, unpaired, paired, linear-trend and within-subject
  trend designs, and the global prior. Five of the cases have missing values (`legacy=TRUE`):
  scattered NaNs, an arm entirely missing, a paired subject missing (the reference subject and
  another), and a within-subject trend with whole subjects missing. Five make the df cap bind,
  and the rest show it changes nothing when it does not. Measured agreement is 1e-11 or better;
  the stated tolerances (1e-12 to 1e-9) are explained in the test's docstring.
- `TestMissingValuesInTheDesign` reproduces the dropped and wrong paired results directly: the
  fold change of a feature missing one subject equals the mean within-subject difference over the
  other subjects, on the right residual df, and a feature missing a whole arm reports no contrast
  but contributes its residual df.
- `TestPriorCalibrationByIntensity` reruns a seeded null with real data's variance structure
  (technical noise falling with intensity, flat biology; 6 draws of 2,000 features, unpaired 5
  vs 5) and checks calibration in each intensity fifth, which an overall 5% rate cannot see.
  limma-trend must stay within 3.5-6.5% in every fifth, and the pool-shaped intensity trend must
  still tilt by more than 3 points. Marked `slow` (about 35 s).
- `TestTrendCalibration` checks limma-trend's overall null rate on its unpaired and paired
  simulations, beside the intensity trend's.
- New tests for the default, the deprecation warning and its reason, the warning for an ignored
  `variance_prior_group_column`, and limma-trend with `robust=True`.

## Documentation

- `docs/06-statistical-analysis.md` explains why limma-trend is the default, with the
  null-calibration measurements by intensity fifth, and documents limma's df cap and the new
  `total_df` column. The intensity trend's sections are marked deprecated, and the examples use
  limma-trend. The one deliberate difference from limma is recorded: with missing values limma
  3.58+ switches to `fitFDistUnequalDF1`, and the toolkit keeps `fitFDist`'s moments.
- The README, the overview, the QC-plot and pitfalls pages, and the tutorial notebook use
  limma-trend as the default. `plot_variance_vs_intensity` is documented as a diagnostic for the
  intensity priors only, and the docs and tutorial show limma's `plotSA` drawn by hand instead.
