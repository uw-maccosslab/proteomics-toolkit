# Proteomics Toolkit v26.8.0 Release Notes

## Overview

Feature release with a breaking change to moderated p-values. The intensity-trend variance prior
is now calibrated to the design being tested. The trend still supplies the shape (how noise
changes with intensity), but its level and the prior degrees of freedom are fitted to the
design's own residual variances. Before this release, moderated tests were optimistic when the
prior came from QC or reference injections. They were nearly powerless for a paired or
within-subject analysis on the default design-group prior. Skyline-PRISM's matching change is
pinned to this release's output at 1e-9.

This release also adds a moderated interaction test for 2 x 2 designs: whether an effect changes
between two conditions.

## New Features

### Moderated interaction (difference-of-differences) analysis

- `statistical_test_method="moderated_linear_model"` now accepts `analysis_type="interaction"`:
  a 2 x 2 factorial of `group_labels` (reference, alternative) by `paired_label1` /
  `paired_label2`, testing the `group:level` coefficient. `logFC` is the difference of
  differences, `(alt - ref at paired_label2) - (alt - ref at paired_label1)`, which is how much
  the group effect changes between the two levels. It uses the same limma, deqms and
  intensity-trend moderation as the other moderated designs.
- An optional `subject_column` block and `covariates` are supported. Aliased design columns (for
  example, subjects nested in the group) are dropped automatically, and the call raises a clear
  error if the interaction itself is not estimable. All four cells must contain samples.
- With `moderation="intensity_trend"` and no `variance_prior_group_column`, the trend takes one
  group per design cell, so it sees within-cell variance rather than variance inflated by the
  effects. QC and reference-pool priors work as for the other designs.
- This path does not use `interaction_terms`; the factors and levels come from `group_column` /
  `group_labels` and `paired_column` / `paired_label1` / `paired_label2`.
  `StatisticalConfig.validate()` no longer requires `interaction_terms` for it, and the
  dispatcher requires `group_column`.

## Bug Fixes

- **The intensity-trend prior was used at the wrong level, so moderated p-values were
  miscalibrated in both directions.** `moderation="intensity_trend"` (the default) and
  `"intensity_peptide_trend"` took the prior's scale from a LOWESS fitted on within-group
  variances. They took its weight, `d0`, separately, from the design residuals around their
  own global mean. Those groups are not the residuals the model is tested against. QC and
  reference pools (`variance_prior_group_column`) lack the biology a study residual carries, so
  their trend sits too low. Design groups under a paired or within-subject model contain the
  between-subject spread the subject block removes, so their trend sits too high. On simulated
  null data, the share of p-values below 0.05 was 40% (unpaired, QC prior), 18.5% (paired, QC
  prior) and 0.17% (paired, default prior), where a calibrated test gives 5%. The trend is now
  multiplied by a level fitted, together with `d0`, to `residual_s2 / trend`. This is Smyth's
  method of moments with the trend as a covariate offset (limma's `fitFDist` with the curve
  supplied). All four cases now give 4.7-5.1%. See "Where the intensity prior's level and weight
  come from" in `docs/06-statistical-analysis.md`.

## Breaking Changes

- **P-values from `intensity_trend` and `intensity_peptide_trend` change.** A QC- or
  reference-sourced prior gets less optimistic. A paired or within-subject analysis on the
  default prior gets more powerful. An unpaired analysis on the default prior barely moves.
  There is no switch back, because the old estimator is miscalibrated in both directions. Pin
  `proteomics-toolkit==26.7.1` to reproduce an earlier result.
- `intensity_s0_sq` is now the prior variance the test used, after calibration. The
  uncalibrated trend is the new `intensity_trend_shape` column, and the factor between them is
  `intensity_trend_level`.

## Testing

- `TestTrendCalibration` checks the share of null p-values below 0.05 for unpaired and paired
  designs, with the trend from QC pools and from design groups. The tests use 2,000 simulated
  features with no true effect, heteroscedastic technical noise, and between-person biology
  that the pools lack. It also checks that the level moves in the direction each source's bias
  predicts, and that a known level and `d0` are recovered.
- `test_qc_prior_produces_lower_s0_sq_than_design_prior` asserted the old behavior as the
  option's headline: a QC prior gives smaller posterior variances and larger |t|. It is replaced
  by `test_prior_level_is_fitted_to_the_design_whichever_source`.
- The variance-plot fixture in `tests/test_visualization.py` fitted the model on linear
  intensities against a log-space trend. Calibration exposed it by scaling the trend by 1.6e9.
  It now fits on log2, as the dispatcher does.
- New plot tests check the dashed-curve label under `intensity_peptide_trend`, and that results
  filtered to no rows or narrowed to a few columns still draw the calibrated prior.
- 11 tests for the interaction mode: an exact cell-mean check, planted-effect recovery, nested
  subjects, error paths, and a dispatcher run with a reference-pool prior.

## Documentation

- `docs/06-statistical-analysis.md` explains the shape, level and weight of the intensity prior,
  with the before-and-after calibration numbers, and documents `variance_prior_group_column`.
  It no longer calls `intensity_trend` the Python equivalent of limma's `trend=TRUE`, and
  neither do the code comments or the tutorial.
- `docs/06-statistical-analysis.md` documents the interaction mode.
- `plot_variance_vs_intensity` draws the prior actually used (dashed) beside the fitted trend
  when calibration moved it. With `intensity_peptide_trend` the dashed curve is the intensity
  stage at the fitted level, since each protein's peptide adjustment cannot be drawn as a curve.
- `docs/06-statistical-analysis.md` records a known limitation to revisit, in this toolkit and in
  Skyline-PRISM together: a single level cannot tilt the trend, so in simulation a QC-sourced
  prior still left null p-values graded by intensity (2.9% to 7.9% across intensity thirds)
  while averaging 5%.
