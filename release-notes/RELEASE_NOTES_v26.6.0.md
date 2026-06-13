# Proteomics Toolkit v26.6.0 Release Notes

Adds **recursive feature elimination with cross-validation and stability
selection** (`run_rfecv_stability`) for finding compact classifying signatures
in the n-much-less-than-p proteomics regime without the optimistic bias of a
plain scikit-learn `RFECV`.

## New Features

### Classification

- `run_rfecv_stability(data, labels, ...)`: recursive feature elimination with
  cross-validation under an honest outer `RepeatedStratifiedKFold`, so feature
  selection never sees the held-out fold it is scored on. Returns:
  - `outer_auc_mean` / `outer_auc_std`: leakage-free held-out AUC, plus
    `fold_roc_data` so the result plugs directly into `plot_roc_curve` (mean ROC
    with a +/- 1 SD band).
  - `selection_frequency`: a `pandas.Series` giving the fraction of outer folds
    in which each feature survived RFE. Features near 1.0 "consistently survive
    cross-validation"; `consensus_features` collects those above
    `consensus_threshold` (default 0.5).
  - `permutation_p_value`: an empirical p-value from a label-shuffle null on the
    held-out AUC, so a weak signal can be tested against chance.
  - `cv_predictions`, `n_features_per_fold`, and a `config` echo.
  - Supports two linear estimators via `estimator=`: `"linear_svm"`
    (`LinearSVC`, scored by `decision_function` to avoid a nested calibration
    CV) and `"logistic_l1"` (L1-penalized logistic regression). A per-fold
    variance prefilter (`prefilter_top_var`) and fractional RFE `step` keep
    peptide-scale matrices (tens of thousands of features) tractable.
  - Optional `annotations` relabel feature ids to gene symbols in the outputs,
    reusing `relabel_features_with_genes`.
- `plot_selection_frequency(results, top_n=30)`: lollipop plot of the most
  stable features from `run_rfecv_stability`, with the consensus threshold drawn
  as a reference line.

Both functions are exported at the package top level
(`ptk.run_rfecv_stability`, `ptk.plot_selection_frequency`).

## Testing

- New `tests/test_classification.py` cases covering: the estimator factory and
  continuous-score helpers; a planted-signal dataset (the discriminative
  features rank top by selection frequency, AUC well above chance, permutation
  p < 0.05); a pure-noise dataset (AUC near chance, diffuse selection,
  non-significant permutation p); both estimators; the return schema and
  `plot_roc_curve` compatibility; and error paths (unknown estimator, too few
  samples, non-binary labels).
- A regression test asserts the L1 logistic estimator emits no `penalty`
  deprecation warning.

## Documentation

- `docs/09-classification.md` gains a "Recursive feature elimination with
  stability selection" section with a usage example and guidance for
  peptide-scale runs (`prefilter_top_var`, lower `n_permutations`).
- README feature list and classification API reference updated.

## Notes

- The L1 logistic estimator uses scikit-learn's non-deprecated `l1_ratio=1.0`
  API (with the `saga` solver) rather than the deprecated `penalty="l1"`,
  avoiding a deprecation warning on every fit under scikit-learn 1.8+.
