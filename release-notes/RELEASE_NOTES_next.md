# Release Notes (Next Release)

## Overview

Adds SHAP-based interpretability for the tree-based binary classifiers
(`run_binary_classification` with `method='random_forest'` or
`method='xgboost'`), so the protein features driving each prediction
can be inspected with the standard `shap` beeswarm or bar summaries.

## New Features

### Classification

- `run_binary_classification` gained a `return_model=False` parameter.
  When set to `True`, the result dict additionally carries
  `final_model`, `scaler`, `X_scaled`, and `y_encoded` so the fitted
  pipeline can be reused downstream (e.g., for SHAP) without
  retraining on a slightly different fold structure.
- New `classification.compute_shap_values(model, X, feature_names=None)`:
  thin wrapper around `shap.TreeExplainer` for RandomForest /
  XGBoost binary classifiers. Collapses the 3-D
  `(samples, features, classes)` Explanation to the 2-D positive-class
  slice so the result is ready for `shap.plots.beeswarm` or
  `plot_shap_summary`. Raises `ImportError` with an install hint when
  `shap` is missing.
- New `classification.plot_shap_summary(explanation, max_display=20,
  plot_type='beeswarm', title=None)`: renders a beeswarm or bar
  summary of the top features and returns the matplotlib figure.

## Bug Fixes

<!-- Description of the bug and its impact, then what was fixed. -->

## Performance

<!-- Performance improvements with context, e.g.
"Reduced memory from 35 GB to 5 GB for 240-file experiments". -->

## Breaking Changes

<!-- Any changes that require user action (config format changes, removed
options, renamed APIs, etc). Omit this section if there are no breaking
changes. -->

## Testing

- New `TestReturnModel` in `tests/test_classification.py`: confirms the
  default does not include model artifacts, and that
  `return_model=True` returns the fitted classifier, scaler, scaled
  feature matrix, and encoded labels.
- New `TestShap` (skipped when `shap` is not installed) covering:
  2-D positive-class collapse from `compute_shap_values`, DataFrame
  input preserving feature names, beeswarm and bar variants of
  `plot_shap_summary` running end-to-end, and a `ValueError` on
  invalid `plot_type`.

## Dependencies

- Added optional `shap` extra: `shap>=0.46`. Required only for
  `compute_shap_values` and `plot_shap_summary`. Install with
  `pip install proteomics-toolkit[shap]` or `uv sync --extra shap`.

## Documentation

<!-- Documentation updates relevant to users (new recipes, restructured
guides, README changes). -->
