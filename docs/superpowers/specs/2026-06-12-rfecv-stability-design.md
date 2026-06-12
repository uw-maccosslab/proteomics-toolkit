# Design: Recursive Feature Elimination with Cross-Validation and Stability Selection

Date: 2026-06-12
Module: `proteomics_toolkit.classification`
Status: Approved for implementation planning

## Motivation

The EV-Opioid-Pregnancy study asks whether a compact set of EV proteins or
peptides can classify opioid- or SSRI-exposed pregnancies. This is a classic
n-much-less-than-p problem: roughly 44-80 samples against 5,486 proteins or
66,171 peptides. Differential analysis (notebooks 01/02) finds a small opioid
signal dominated by CHGA and no SSRI signal.

Recursive feature elimination (RFE) is attractive for finding a minimal
classifying signature, but in this regime a naive `sklearn.RFECV` reports badly
optimistic performance: the same cross-validation used to choose the feature
count also sees every sample, so feature selection leaks into the score. We need
a procedure that (a) gives an honest held-out performance estimate, (b) reports
which features *consistently* survive selection across resampling, and (c) tells
us whether any apparent signature beats chance.

## Goals

1. Add a public `run_rfecv_stability` function to `classification.py`.
2. Honest performance via an outer CV that wraps the entire RFE-CV pipeline.
3. Per-feature selection frequency across outer folds ("consensus signature").
4. Permutation (label-shuffle) null to attach an empirical p-value to AUC.
5. Reuse the existing mean-ROC-with-SD-band plotting (`plot_roc_curve`).
6. Scale to peptide-level matrices (tens of thousands of features) in minutes.
7. Apply it inside notebooks 01 (proteins) and 02 (peptides) for the opioid and
   SSRI contrasts, `preg_only + lenient` config, with both estimators.

## Non-goals

- No multi-class RFE (binary only, matching the contrasts of interest).
- No probability calibration (AUC/ROC are rank-based; see decision_function).
- No new notebooks; extend 01 and 02 in place.

## Public API

```python
def run_rfecv_stability(
    data: pd.DataFrame,            # samples x features (intensities)
    labels: pd.Series,             # indexed by sample id -> binary class
    estimator: str = "linear_svm", # "linear_svm" | "logistic_l1"
    outer_cv: tuple[int, int] = (5, 10),   # (n_splits, n_repeats) RepeatedStratifiedKFold
    inner_cv: int = 5,             # StratifiedKFold for RFECV feature-count search
    step: float = 0.1,             # fraction of features eliminated per RFE iteration
    prefilter_top_var: int | None = 2000,  # top-N by TRAIN-fold variance before RFE
    scoring: str = "roc_auc",
    min_features_to_select: int = 1,
    log_transform: str | bool = "auto",
    n_permutations: int = 100,     # label-shuffle null; 0 disables
    annotations: pd.DataFrame | None = None,
    id_col: str = "protein_group",
    gene_col: str = "leading_gene_name",
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict
```

### Estimators

- `"linear_svm"`: `LinearSVC` (small C, e.g. 0.01, `max_iter=5000`). RFE ranks by
  `abs(coef_)`. Held-out scores from `decision_function` (rank-equivalent to a
  calibrated probability for AUC/ROC, but avoids a nested calibration CV).
- `"logistic_l1"`: `LogisticRegression(penalty="l1", solver="saga")`. RFE ranks
  by `abs(coef_)`. Held-out scores from `predict_proba`.

## Algorithm

For each outer split from `RepeatedStratifiedKFold(n_splits, n_repeats)`:

1. Split samples into train / test.
2. Fit-on-train-only preprocessing: optional `log2` (auto-detect raw-scale),
   `StandardScaler`, then variance prefilter to the top `prefilter_top_var`
   features by training-fold variance. Apply the same transforms to test.
3. Run `RFECV(estimator, step=step, cv=StratifiedKFold(inner_cv),
   scoring=scoring, min_features_to_select=..., n_jobs=n_jobs)` on the
   transformed training data. Its internal CV chooses the feature count and the
   selected subset, using train only.
4. Record the selected feature set for this fold.
5. Refit the estimator on the training fold restricted to selected features;
   score the held-out test fold (AUC, balanced accuracy, per-fold ROC).

Aggregate across outer folds:

- Performance: mean +/- SD of outer-test AUC and balanced accuracy.
- Stability: `selection_frequency[feature] = folds_selected / total_folds`,
  sorted descending. Consensus set = features with frequency >= 0.5 (threshold
  configurable by the caller from the returned full curve).
- Permutation null: repeat the entire outer-CV performance estimate with
  shuffled labels `n_permutations` times; empirical p-value =
  (1 + #{null_mean_auc >= observed_mean_auc}) / (1 + n_permutations).
- Optional final `RFECV` on all data -> single deployable feature set, flagged
  as not independently validated.

## Return value (dict)

| Key | Type | Meaning |
|-----|------|---------|
| `estimator` | str | Echo of estimator used |
| `outer_auc_mean`, `outer_auc_std` | float | Honest held-out AUC |
| `auc_roc`, `auc_std` | float | Aliases for `plot_roc_curve` compatibility |
| `balanced_accuracy` | float | Mean held-out balanced accuracy |
| `per_fold_scores` | list[float] | Per-outer-fold AUC |
| `selection_frequency` | pd.Series | feature -> fraction of folds selected (sorted) |
| `consensus_features` | list[str] | features above frequency threshold |
| `n_features_per_fold` | list[int] | RFECV-chosen feature count per fold |
| `permutation_auc_null` | np.ndarray | null mean-AUC distribution (empty if disabled) |
| `permutation_p_value` | float \| None | empirical p for observed AUC |
| `cv_predictions` | pd.DataFrame | sample, y_true, y_pred, y_score, fold |
| `fold_roc_data` | list | per-fold (fpr, tpr) for `plot_roc_curve` |
| `class_names` | list[str] | label classes |
| `final_features` | list[str] | all-data RFECV set (optional) |
| `config` | dict | echo of key parameters |

Feature names are gene-relabeled via `relabel_features_with_genes` when
`annotations` is provided.

## Plotting

- New `plot_selection_frequency(results, top_n=30, title=...)`: horizontal
  lollipop of the most stable features (gene-labeled), with the consensus
  threshold drawn as a reference line.
- Reuse existing `plot_roc_curve(results)`: the return dict carries
  `fold_roc_data`, `auc_roc`, `auc_std`, and `class_names`, so the mean ROC with
  +/- 1 SD band renders with no changes, matching notebook 01's current style.

## Performance and scalability

Keep total runtime in minutes, not hours:

- Train-fold variance prefilter caps RFE input (peptides never start at 66k).
- `step=0.1` removes 10% of features per RFE iteration (log-scaled iterations).
- `decision_function` for SVM avoids a nested calibration CV.
- `n_jobs=-1` parallelizes RFECV's inner CV and the permutation loop.
- Peptide notebook uses a smaller `n_permutations` (e.g. 50) and a tighter
  prefilter (e.g. 3000); a runtime estimate is printed before heavy cells.

## Testing (TDD)

1. Planted-signal dataset (~40 samples, 200 features, 5 discriminative + 195
   noise): the 5 planted features rank at the top of `selection_frequency`, AUC
   well above 0.5, permutation p < 0.05.
2. Pure-noise dataset: AUC approx 0.5, no feature near frequency 1.0,
   permutation p not significant. Guards against the optimistic-RFECV failure.
3. Both estimators run and return the documented schema.
4. Error paths: unknown estimator, fewer than 10 samples, single-class labels.

Tests follow the existing `tests/test_classification.py` conventions (class per
concept, descriptive names, fixtures in `conftest.py`).

## Notebook application (01 and 02)

Add an RFECV section to each notebook:

- Build the samples x features matrix and the binary `opioid_use` / `ssri_use`
  label vector for the `preg_only + lenient` config.
- Run `run_rfecv_stability` with `estimator="linear_svm"` and `"logistic_l1"`.
- Report mean AUC +/- SD against the permutation null, the stable consensus
  signature (gene-labeled), and the mean ROC with SD band.

Expected outcome (a check on the method, not just the biology): the opioid
contrast yields a small but above-chance, CHGA-centric signature; the SSRI
contrast returns at chance with no stable consensus, the correct negative
control.
