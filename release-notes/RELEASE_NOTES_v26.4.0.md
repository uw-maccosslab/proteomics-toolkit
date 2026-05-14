# Release Notes v26.4.0

## Overview

Adds a moderated linear trend analysis to `run_moderated_linear_model` and
fixes a validator bug that rejected falsy paired labels (e.g. integer `0`
for a "Week 0" baseline).

## New Features

### Moderated linear trend (`analysis_type='linear_trend'`)

- `run_moderated_linear_model` gains support for `analysis_type='linear_trend'`
  in addition to the existing `'paired'` and `'unpaired'` modes. The per-feature
  design is `feature ~ intercept + Time + (optional subject one-hot block)`,
  and the slope coefficient is tested with the same limma / deqms /
  intensity_trend empirical-Bayes variance moderation as the two-group
  modes.
- `logFC` in the output is the slope per unit time, so use a small
  `fc_threshold` in volcano plots (the existing convention for trend tests).
- Required: `config.time_column` (numeric column in metadata). Optional:
  `config.subject_column` to enable a limma-style fixed-effect subject block
  for repeated-measures designs.
- When `moderation='intensity_trend'`, every unique value of `time_column`
  contributes an anchor point per feature to the LOWESS variance trend.
  A 5-timepoint cohort therefore gives 5x the leverage on the prior fit
  compared to a 2-group paired comparison.

## Bug Fixes

- `StatisticalConfig.validate()` now uses `is None` instead of truthiness
  when checking `paired_label1` and `paired_label2`, so valid falsy labels
  like `0` (e.g. "Week 0" baseline stored as an integer) no longer fail
  validation. This removes the need for callers to cast paired labels to
  strings as a workaround.

## Testing

- New `TestModeratedLinearTrend` class in `tests/test_statistical_analysis.py`
  covering schema, slope recovery on planted-effect data, per-(feature, time)
  intensity-trend anchor points, subject blocking presence / absence, all
  three moderation priors, and input validation.
- New `test_validate_paired_accepts_zero_label` regression test for the
  validator falsy-label fix.
