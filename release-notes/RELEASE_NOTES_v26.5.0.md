# Proteomics Toolkit v26.5.0 Release Notes

Adds an opt-in **reference-sample variance prior** for the intensity-trend
moderated linear model so the LOWESS prior can be calibrated on dedicated
technical-replicate classes (BatchQC, BatchRef, pooled QC, system suitability)
instead of the analysis design groups, which often include real biological
heterogeneity that should not be smoothed into the prior.

## New Features

### Statistical Analysis

- `StatisticalConfig` gained two new optional attributes for use with
  `statistical_test_method='moderated_linear_model'` and
  `moderation='intensity_trend'`:
  - `variance_prior_group_column`: name of a metadata column whose values
    define the groups used to fit the intensity-trend LOWESS prior
    (e.g. `'QC_Category'`). When `None` (default), the prior is fit on
    the design groups as before, preserving backward compatibility.
  - `variance_prior_groups`: optional list of values from
    `variance_prior_group_column` to restrict the prior pool
    (e.g. `['BatchQC', 'BatchRef']`).
- `run_moderated_linear_model` and the dispatcher
  `run_comprehensive_statistical_analysis` route the prior fit to those
  samples and skip the design-only restriction when the option is set.
  The per-feature OLS fit on study samples is unchanged; only the
  variance prior changes. Reference samples can be supplied through the
  same `sample_metadata` dict as the study samples; they are excluded
  from the design fit automatically (NaN values in `paired_column` /
  `time_column` are filtered out before model fitting).
- `prepare_metadata_dataframe` now treats `time_column` as a required
  column for `analysis_type='linear_trend'` (and `longitudinal` /
  `dose_response`) so reference samples that carry NaN time values do
  not leak into the design fit. Previously, the required-column check
  did not include the time column, which became apparent when callers
  passed reference samples to populate the new prior.
- The moderated linear-model `linear_trend` design fit now drops samples
  with non-finite time values instead of raising. This is the defensive
  counterpart to the dispatcher change above and accommodates direct
  callers of `run_moderated_linear_model` who pass reference samples
  without going through the dispatcher.

## Why This Matters

For studies where the design groups are not nominal replicates (e.g. a
paired pre/post analysis where each timepoint group spans many subjects
with real biological variability, or a linear-trend analysis where the
"groups" are unique time values each populated by a handful of subjects),
the within-group SD bakes in inter-subject biology. The intensity-trend
LOWESS prior calibrated on that SD is then conservative: it shrinks
per-feature posterior variances toward a noise floor that already
includes signal we are trying to detect, blunting the test.

Dedicated technical-replicate samples (BatchQC, BatchRef, pooled QC) give
a clean technical-noise estimate at each intensity. Using those as the
prior pool produces a less conservative test on biological samples: real
biology rises above the technical floor, while truly noisy features still
get shrunk. On a real plasma EV study (n=70, 3,995 proteins, 14 reference
samples spanning the same intensity range as the study), switching from
the design within-group prior to a BatchQC + BatchRef prior produced
2-3x more significant proteins at the same FDR cutoff without changing
the per-feature OLS fit.

## Backward Compatibility

The new option is fully opt-in. Existing code that does not set
`variance_prior_group_column` reproduces the v26.4.x behavior bit-for-bit.
A unit test (`test_default_unchanged_when_option_not_set`) enforces this.

## Testing

- New `TestVariancePriorGroupColumn` class in
  `tests/test_statistical_analysis.py` with five tests:
  - `test_qc_prior_produces_lower_s0_sq_than_design_prior`: on a fixture
    where reference samples have lower technical SD than the design
    within-group SD, the QC-sourced prior yields a smaller
    `intensity_s0_sq`, smaller `posterior_s2`, and larger median `|t|`.
  - `test_default_unchanged_when_option_not_set`: leaving the new option
    as `None` reproduces the historical result exactly.
  - `test_unknown_prior_column_raises`: validation error when the named
    column is not present in metadata.
  - `test_prior_groups_restriction_is_honored`: when
    `variance_prior_groups` is set, the prior cloud only contains rows
    whose value is in the whitelist.
  - `test_works_in_paired_analysis_type`: confirms the option works for
    `analysis_type='paired'` with reference samples carrying NaN
    `paired_column`.

## Documentation

- `StatisticalConfig` and `run_moderated_linear_model` docstrings explain
  when to use the new option, why it matters, and the requirement that
  reference samples be included in `sample_metadata`.
