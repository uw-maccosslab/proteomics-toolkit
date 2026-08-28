# Continuous-Outcome Regression

[← Back to overview](01-overview.md)

Predict a continuous outcome (a clinical score, a percent-change
measurement, a dose-response endpoint) from a samples x features
abundance matrix, using nested cross-validated ElasticNet. Mirrors the
honest nested-CV design of [`run_rfecv_stability`](09-classification.md):
an outer loop gives an unbiased held-out performance estimate while an
inner loop tunes regularization strength, and a target-permutation null
attaches an empirical p-value to the observed performance.

## Running ElasticNet regression

`run_elasticnet_regression(data, target, ...)` wraps
`sklearn.linear_model.ElasticNetCV` inside an outer `RepeatedKFold`, so
alpha/l1_ratio tuning never sees the fold it is scored on.

```python
import proteomics_toolkit as ptk

res = ptk.run_elasticnet_regression(
    expr,                 # samples x features
    outcome,               # continuous Series indexed by sample
    annotations=protein_table,   # optional gene relabeling
)
print(res["outer_r2_mean"], res["outer_spearman_mean"], res["permutation_p_value"])
ptk.plot_regression_scatter(res)
```

It returns:

- `outer_r2_mean` / `outer_r2_std`: honest held-out R² across outer folds.
- `outer_rmse_mean`, `outer_spearman_mean`: pooled out-of-fold RMSE and
  Spearman correlation between true and predicted values.
- `selection_frequency`: a Series giving the fraction of outer folds in
  which each feature kept a nonzero ElasticNet coefficient. Features near
  1.0 "consistently survive CV".
- `consensus_features`: features above `consensus_threshold` (default 0.5).
- `permutation_p_value`: empirical p from a target-shuffle null on R².
- `coefficients`: descriptive coefficients from a final refit on all data
  (not cross-validated; for interpretation only).
- `cv_predictions`: DataFrame of pooled out-of-fold `True_Value` /
  `Predicted_Value` per sample, ready for `plot_regression_scatter`.

## Plotting

```python
ptk.plot_regression_scatter(res, title="TLG improvement: ElasticNet")
ptk.plot_selection_frequency(res, top_n=30)  # reused from classification.py
```

`plot_selection_frequency` (from the classification module) works
unmodified on regression results since both return dicts share the same
`selection_frequency` / `config["consensus_threshold"]` schema.

## Tuning parameters

- `l1_ratio`: a single value or sequence passed to `ElasticNetCV`. A
  sequence (the default) lets `ElasticNetCV` search the L1/L2 mix as well
  as alpha along the regularization path.
- `n_alphas`: number of alphas per `l1_ratio` searched by `ElasticNetCV`.
- `outer_cv` / `inner_cv`: `(n_splits, n_repeats)` for the outer
  `RepeatedKFold`, and fold count for the inner `KFold` used by
  `ElasticNetCV`'s internal search.
- `n_permutations`: target-shuffle iterations for the null; `0` disables.

For small cohorts (n < ~50), the default `outer_cv=(5, 10)` (50
evaluations) and `n_permutations=100` are cheap. For larger feature counts
(peptide-scale matrices), reduce `n_permutations` to control runtime.

## Caveats

- With n-much-less-than-p proteomics data, ElasticNet coefficients from
  the final all-data refit are descriptive, not confirmatory — use
  `selection_frequency` / `consensus_features` (computed only from
  held-out folds) as the more honest signal of which features matter.
- A sparse or heavily skewed target (e.g. an ordinal score dominated by a
  couple of values) will underperform a genuinely continuous outcome;
  check the target's distribution before trusting R²/permutation p at
  face value.
