# Binary Classification

[← Back to overview](01-overview.md)

Classify subjects into two groups (e.g., responder vs non-responder)
using protein fold-changes as features, with cross-validated performance
metrics.

Sections:

- [Feature selection and leakage](#feature-selection-and-leakage)
- [Computing per-subject fold-changes](#computing-per-subject-fold-changes)
- [PCA of fold-changes](#pca-of-fold-changes)
- [Running classification](#running-classification)
- [Nested differential-abundance selection](#nested-differential-abundance-selection)
- [Comparing multiple methods](#comparing-multiple-methods)

## Feature selection and leakage

`run_binary_classification` takes a `feature_selection` argument that
controls how the top `n_top_features` are chosen. The default is the
leakage-free MAD ranking.

| Strategy | Uses outcome labels? | Recommended for |
|---|---|---|
| `"mad"` *(default)* | no (unsupervised MAD across subjects) | small datasets; the safe default |
| `"differential_abundance"` | yes, but **nested** - training split only per fold | small datasets when you want a supervised ranker |
| `"fold_change"` | no in principle, but can correlate with outcome if the treatment effect is asymmetric | legacy behaviour - avoid for small N |
| explicit `feature_proteins=...` | up to you | features from a truly independent source (different cohort, prior biology) |

Why it matters: on a small cohort, ranking features by `|fold-change|`
or by a full-dataset t-statistic and then running cross-validation on
the same subjects lets information about the held-out test split leak
into the feature list. Reported AUC is inflated. `"mad"` avoids this by
never consulting the outcome; `"differential_abundance"` avoids it by
re-ranking features using only the training split of each fold.

## Computing per-subject fold-changes

```python
# Log2 transform first (PRISM data is in linear space)
log2_data = data.copy()
log2_data[study_cols] = np.log2(data[study_cols].clip(lower=1e-6))

# Compute paired differences (post - pre) per subject
fc_matrix = ptk.compute_paired_fold_changes(log2_data, meta_dict, config)

# Build group labels per subject
subject_response = {}
for col, meta in meta_dict.items():
    subj = meta.get('Subject')
    resp = meta.get('Response')
    if subj and resp:
        subject_response[subj] = resp
group_labels = pd.Series(subject_response)
group_labels = group_labels.loc[group_labels.index.intersection(fc_matrix.index)]
```

## PCA of fold-changes

```python
ptk.plot_fold_change_pca(
    fc_matrix, group_labels,
    group_colors={'NR': '#1f77b4', 'R': '#ff7f0e'},
    title='PCA of Treatment Response',
)
```

## Running classification

Four methods are available: `logistic_regression`, `random_forest`,
`linear_svm`, `xgboost`. The default feature selector is `"mad"`.

```python
# MAD-based selection of the 50 most variable features, with 5-fold CV
cr = ptk.run_binary_classification(
    fc_matrix, group_labels,
    n_top_features=50,
    method='logistic_regression',
    cv_method=5,
)

# Individual ROC curve with mean +/- SD band
ptk.plot_roc_curve(cr, title='ROC: Logistic Regression')
```

You can inspect the MAD ranking directly:

```python
top = ptk.select_features_by_mad(fc_matrix, n_top_features=50)
```

## Nested differential-abundance selection

When you do want a supervised feature ranker but the dataset is small,
use nested DA. Feature ranking is performed with a Welch t-test inside
each CV fold, on the training split only; the held-out test split
never influences selection.

```python
cr = ptk.run_binary_classification(
    fc_matrix, group_labels,
    feature_selection='differential_abundance',
    n_top_features=30,
    method='logistic_regression',
    cv_method=5,
)
```

The `feature_names` entry in the returned dict is the list picked on
the *full dataset* (used to fit the final model for reporting
`feature_importances`). The cross-validated AUC reflects the per-fold
selections rather than this full-data list.

## Comparing multiple methods

```python
methods = ['logistic_regression', 'random_forest', 'linear_svm', 'xgboost']
method_results = {
    method: ptk.run_binary_classification(
        fc_matrix, group_labels,
        feature_selection='mad',          # or 'differential_abundance'
        n_top_features=50,
        method=method, cv_method=5,
    )
    for method in methods
}

# Overlay all methods on one ROC plot
ptk.plot_roc_comparison(method_results, title='ROC Comparison')
```

**Classification result keys:** `accuracy`, `balanced_accuracy`,
`auc_roc`, `auc_std`, `confusion_matrix`, `cv_predictions`,
`feature_importances`, `classification_report`, `feature_selection`,
`fold_roc_data`.

**Label inversion:** The module automatically detects and corrects
label inversion (AUC < 0.5) by flipping probabilities, predictions,
and per-fold ROC curves.

See [06-statistical-analysis.md](06-statistical-analysis.md) for upstream
fold-change generation details.
