# Release Notes v26.4.2

## Overview

Adds covariate adjustment to the unpaired moderated linear model. Previously,
`config.covariates` was silently ignored unless
`statistical_test_method='mixed_effects'`.

## New Features

### Statistical analysis

- `run_moderated_linear_model` and `run_comprehensive_statistical_analysis`
  now honor `config.covariates` when
  `statistical_test_method='moderated_linear_model'` and
  `analysis_type='unpaired'`. Numeric covariates contribute one design
  column each; categorical (object / category) covariates are
  reference-dummy-coded via patsy. The reported treatment statistics
  (`logFC`, `t`, `P.Value`, `adj.P.Val`) are adjusted for the supplied
  covariates.
- Samples missing any covariate value are listwise-deleted before the
  per-feature fit, and the `intensity_trend` prior is refit on the same
  restricted sample set so the variance prior matches the coefficient
  estimates.
- `analysis_type='paired'` and `'linear_trend'` do not yet apply
  covariates in the moderated linear model and emit a warning when
  `config.covariates` is set.

## Bug Fixes

- `config.covariates` was previously silently ignored by the moderated
  linear model path. It is now applied for `analysis_type='unpaired'`
  and an explicit warning is printed for `paired` / `linear_trend`
  rather than failing silently.

## Testing

- New covariate-adjustment tests in `tests/test_statistical_analysis.py`:
  recovery of a known treatment effect in the presence of a confounder,
  categorical-covariate dummy encoding, multi-covariate adjustment,
  listwise deletion of samples with missing covariate values, error
  on an unknown covariate column, empty-covariate-list baseline
  equivalence, and composition with the `intensity_trend` prior.
