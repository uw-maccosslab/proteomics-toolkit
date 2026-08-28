# Proteomics Toolkit v26.7.0 Release Notes

Two additions. A nested cross-validated ElasticNet regression function for
predicting continuous outcomes from proteomics data, the regression
counterpart to `run_rfecv_stability`. And a new variance-prior mode for the
moderated linear model, `moderation="intensity_peptide_trend"`, which
conditions the prior on peptide count as well as intensity — measurably the
best-calibrated prior available for protein-level rollup data.

## New Features

### Regression (new `regression.py` module)

- `run_elasticnet_regression(data, target, ...)`: nested-CV ElasticNet
  regression for a continuous outcome. Wraps `sklearn.linear_model.ElasticNetCV`
  inside an outer `RepeatedKFold` so alpha/l1_ratio tuning never sees the fold
  it is scored on. Returns:
  - `outer_r2_mean` / `outer_r2_std`: honest held-out R2.
  - `outer_rmse_mean`, `outer_spearman_mean`: pooled out-of-fold RMSE and
    Spearman correlation.
  - `selection_frequency`: fraction of outer folds in which each feature kept
    a nonzero coefficient, and `consensus_features` above a configurable
    threshold — same schema as `run_rfecv_stability`, so
    `plot_selection_frequency` works unmodified on regression results.
  - `permutation_p_value`: empirical p-value from a target-shuffle null on R2.
  - `cv_predictions`, `coefficients` (descriptive, from an all-data refit),
    and a `config` echo.
  - Optional `annotations` relabel feature ids to gene symbols in the
    outputs, reusing `relabel_features_with_genes`.
- `plot_regression_scatter(results, ...)`: predicted-vs-true scatter plot
  from `run_elasticnet_regression`, with a 1:1 reference line and
  R2/Spearman/permutation-p annotated in the title.

Both functions are exported at the package top level
(`ptk.run_elasticnet_regression`, `ptk.plot_regression_scatter`).

### Variance prior conditioned on peptide count

- New `config.moderation = "intensity_peptide_trend"` for
  `run_moderated_linear_model`. Fits the additive two-stage model
  `log(var) = f1(log mean intensity) + f2(log peptide count)`, where both
  stages are LOWESS. The existing `"intensity_trend"` mode is stage 1 alone.

  The motivation is that intensity is not a sufficient statistic for
  protein-level variance. At matched intensity, a protein rolled up from many
  peptides is better determined than one from few, and the two predictors are
  only weakly correlated (r = 0.34 on the validation dataset), so the peptide
  term is largely independent information.

  Measured on 7,990 (feature, group) points from 14 technical replicates,
  5-fold cross-validated RMSE on log(variance):

  | Prior | CV RMSE | vs flat |
  |---|---|---|
  | Constant (`limma`, no trend) | 2.337 | — |
  | `intensity_trend` | 1.043 | +55.4% |
  | Explicit `a + b*mu + c*mu^2` | 1.046 | +55.2% |
  | `deqms`-style, peptide count alone | 2.389 | -2.2% |
  | **`intensity_peptide_trend`** | **0.831** | **+64.4%** |

  Two results worth noting. An explicit physical error model buys nothing
  over the nonparametric fit, so there is no reason to impose one. And
  peptide count *alone* is slightly worse than a flat prior — it helps only
  in addition to intensity, not instead of it, so `deqms` is not a substitute
  for this mode.

  The peptide term is theoretically grounded, not just empirically useful.
  Fitting both terms jointly and linearly gives
  `log(var) = c + 1.60*log(abundance) - 0.89*log(n_peptides)` at R2 = 0.88.
  A rollup of `n` independent peptides has variance proportional to `1/n`,
  i.e. a peptide coefficient of exactly -1; the observed -0.89 indicates
  near-ideal averaging with a small shortfall from correlated peptide error.

  The intensity stage always conditions on the abundance of the feature being
  tested — protein abundance for protein-level analyses, peptide abundance
  for peptide-level — because the prior reads the same raw feature matrix the
  model is fit on. `intensity_peptide_trend` is protein-level only.

  Requires a peptide-count column in the data (`config.peptide_count_column`,
  default `"n_peptides"`), matching the existing `deqms` requirement. Falls
  back cleanly to intensity-only behaviour when the counts are degenerate.

- `plot_variance_vs_intensity` now carries `peptide_count_used`,
  `peptide_log_var_adj`, and `intensity_log_var_hat` on the diagnostic
  points, so the contribution of each stage is inspectable.

## Bug Fixes

- `plot_variance_vs_intensity` drew the diagnostic in a space the estimator
  does not use. It plotted within-group SD against **sqrt(mean intensity)**
  on linear axes and overlaid a dashed `sd = k*sqrt(intensity)` "Poisson-like"
  reference line, implying a counting-noise model. The prior is in fact a
  nonparametric LOWESS of `log(variance)` on `log(mean intensity)`, with no
  shot-noise assumption anywhere in it. MS intensities are ion *rates*, not
  counts, so that framing was doubly misleading, and on those axes a
  well-fitting log-log prior looks like a poor fit with scatter fanning out at
  high intensity.

  The plot is now drawn in fit space: `log(variance)` vs `log(mean intensity)`
  with the LOWESS overlaid, annotated with the observed log-log slope and
  reference slopes of 1 (shot noise) and 2 (constant CV) so the noise regime
  can be read directly. A second panel shows the residual after the intensity
  stage against peptide count, making the value of the second stage visible.
  On the validation dataset the observed slope is 1.42 — between the two
  parametric extremes, which is exactly why a nonparametric prior is the right
  default.

## Performance

<!-- Performance improvements with context, e.g.
"Reduced memory from 35 GB to 5 GB for 240-file experiments". -->

## Breaking Changes

<!-- Any changes that require user action (config format changes, removed
options, renamed APIs, etc). Omit this section if there are no breaking
changes. -->

## Testing

- New `tests/test_regression.py` cases covering: a planted-signal dataset
  (out-of-fold R2 above a pure-noise baseline, planted features rank high in
  selection frequency, permutation p < 0.3), a pure-noise dataset (R2 near 0,
  permutation null disabled returns `None`), the return schema, gene-name
  relabeling via `annotations`, consensus-threshold filtering, the `<10
  samples` error path, and `plot_regression_scatter` returning a Figure.
- New `TestModeratedLinearModelIntensityPeptideTrend` covering: the returned
  schema and `peptide_count_used` column, the count column not being consumed
  as an extra sample, the missing-column error path, a custom
  `peptide_count_column`, the peptide stage measurably moving the prior on
  data with planted count-dependent variance, the diagnostic points carrying
  both stages' columns, and graceful degradation to intensity-only when the
  count column is constant.

## Documentation

- New `docs/12-regression.md` with a `run_elasticnet_regression` usage
  recipe, return-value reference, tuning guidance, and caveats. Linked from
  `docs/01-overview.md`'s guide index and typical workflow.
- README feature list and module reference updated with the new
  `regression.py` module.
- `StatisticalConfig.moderation` docstring documents the new
  `intensity_peptide_trend` mode alongside `limma`, `deqms`, and
  `intensity_trend`, including when to prefer each.
