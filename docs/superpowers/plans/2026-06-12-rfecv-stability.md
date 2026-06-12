# RFECV Stability-Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable `run_rfecv_stability` function to `proteomics_toolkit.classification` that runs recursive feature elimination with cross-validation under an honest outer CV, reports per-feature selection frequency and a permutation null, then apply it to the opioid/SSRI contrasts in notebooks 01 and 02.

**Architecture:** A single public function wraps `sklearn.RFECV` inside an outer `RepeatedStratifiedKFold`. All preprocessing (log2, scaling, variance prefilter) and RFE selection happen on training folds only, so held-out AUC carries no selection leakage. Per-fold selected feature sets are accumulated into a selection-frequency Series; a label-shuffle permutation loop yields an empirical p-value. The return dict is shaped to plug into the existing `plot_roc_curve`. A `plot_selection_frequency` helper visualizes the stable signature.

**Tech Stack:** Python, scikit-learn (`RFECV`, `LinearSVC`, `LogisticRegression`, `RepeatedStratifiedKFold`, `StratifiedKFold`), numpy, pandas, matplotlib, pytest.

**Working directory:** `/home/maccoss/GitHub/uw-maccosslab/proteomics-toolkit`
**Run tests with:** `uv run pytest tests/test_classification.py -v`

---

## File Structure

- Modify: `proteomics_toolkit/classification.py` — add `_make_rfe_estimator`, `_continuous_scores`, `_run_rfecv_outer_cv` (private), `run_rfecv_stability`, `plot_selection_frequency` (public).
- Modify: `proteomics_toolkit/__init__.py` — export the two new public symbols.
- Modify: `tests/test_classification.py` — add test classes for the new functions.
- Modify: `docs/09-classification.md` — document the new function.
- Modify: `README.md` — one-line feature mention.
- Modify: `../collab-uw-vojech/2026-05-EV-Opioid-Pregnancy/notebooks/01_opioid_ssri_differential_and_clustering.ipynb` — add an RFECV section (proteins).
- Modify: `../collab-uw-vojech/2026-05-EV-Opioid-Pregnancy/notebooks/02_opioid_ssri_peptide_differential_and_clustering.ipynb` — add an RFECV section (peptides).

---

## Task 1: Estimator factory and score helpers

**Files:**
- Modify: `proteomics_toolkit/classification.py` (add after `select_features_by_mad`, near line 41)
- Test: `tests/test_classification.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_classification.py`:

```python
from proteomics_toolkit.classification import (
    _make_rfe_estimator,
    _continuous_scores,
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


class TestRfeEstimatorFactory:
    def test_linear_svm_returns_linearsvc_with_coef(self):
        est = _make_rfe_estimator("linear_svm", random_state=0)
        assert isinstance(est, LinearSVC)

    def test_logistic_l1_returns_l1_logistic(self):
        est = _make_rfe_estimator("logistic_l1", random_state=0)
        assert isinstance(est, LogisticRegression)
        assert est.penalty == "l1"

    def test_unknown_estimator_raises(self):
        with pytest.raises(ValueError, match="estimator"):
            _make_rfe_estimator("random_forest", random_state=0)

    def test_continuous_scores_prefers_proba_then_decision(self):
        import numpy as np
        X = np.random.RandomState(0).normal(size=(20, 4))
        y = (X[:, 0] > 0).astype(int)
        lr = LogisticRegression().fit(X, y)
        svm = LinearSVC(dual="auto").fit(X, y)
        s_lr = _continuous_scores(lr, X)
        s_svm = _continuous_scores(svm, X)
        assert s_lr.shape == (20,)
        assert s_svm.shape == (20,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_classification.py -k "RfeEstimatorFactory or continuous_scores" -v`
Expected: FAIL with `ImportError` / `cannot import name '_make_rfe_estimator'`.

- [ ] **Step 3: Implement the helpers**

Add to `proteomics_toolkit/classification.py` after `select_features_by_mad` (line 41):

```python
def _make_rfe_estimator(estimator, random_state=42):
    """Build a linear estimator exposing ``coef_`` for RFE ranking.

    Args:
        estimator: ``"linear_svm"`` or ``"logistic_l1"``.
        random_state: Seed for reproducibility.

    Returns:
        An unfitted scikit-learn estimator.

    Raises:
        ValueError: If ``estimator`` is not a supported name.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    if estimator == "linear_svm":
        # Small C favors a wide margin; dual="auto" handles n_features >> n_samples.
        return LinearSVC(C=0.01, random_state=random_state, max_iter=5000, dual="auto")
    if estimator == "logistic_l1":
        return LogisticRegression(
            penalty="l1", solver="saga", C=1.0, max_iter=5000, random_state=random_state
        )
    raise ValueError(f"Unknown estimator: {estimator!r}. Use 'linear_svm' or 'logistic_l1'.")


def _continuous_scores(clf, X):
    """Return a continuous positive-class score for each row of ``X``.

    AUC and ROC are rank-based, so a calibrated probability is unnecessary:
    ``decision_function`` (the signed distance from a linear SVM's boundary)
    ranks samples identically to a calibrated probability and avoids an extra
    calibration cross-validation. Logistic regression uses ``predict_proba``.
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_classification.py -k "RfeEstimatorFactory or continuous_scores" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add proteomics_toolkit/classification.py tests/test_classification.py
git commit -m "Added RFE estimator factory and score helpers"
```

---

## Task 2: Core `run_rfecv_stability` (outer CV, stability, permutation null)

**Files:**
- Modify: `proteomics_toolkit/classification.py` (add after the helpers from Task 1)
- Test: `tests/test_classification.py`

- [ ] **Step 1: Add shared synthetic-data fixtures to `tests/conftest.py`**

Add to `tests/conftest.py`:

```python
@pytest.fixture
def rfecv_signal_data():
    """40 samples x 200 features; 5 planted discriminative features + noise."""
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(0)
    n, p, n_signal = 40, 200, 5
    y = np.array([0] * 20 + [1] * 20)
    X = rng.normal(size=(n, p))
    # Planted features shift by +2 in class 1 (columns 0..4).
    X[y == 1, :n_signal] += 2.0
    samples = [f"S{i}" for i in range(n)]
    data = pd.DataFrame(X, index=samples, columns=[f"F{j}" for j in range(p)])
    labels = pd.Series(["ctrl" if v == 0 else "case" for v in y], index=samples)
    signal_features = [f"F{j}" for j in range(n_signal)]
    return data, labels, signal_features


@pytest.fixture
def rfecv_noise_data():
    """40 samples x 200 features of pure noise; labels independent of features."""
    import numpy as np
    import pandas as pd

    rng = np.random.RandomState(1)
    n, p = 40, 200
    X = rng.normal(size=(n, p))
    samples = [f"S{i}" for i in range(n)]
    data = pd.DataFrame(X, index=samples, columns=[f"F{j}" for j in range(p)])
    labels = pd.Series(["ctrl"] * 20 + ["case"] * 20, index=samples)
    return data, labels
```

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_classification.py`:

```python
from proteomics_toolkit.classification import run_rfecv_stability


class TestRunRfecvStabilitySignal:
    def test_planted_features_rank_top_and_auc_high(self, rfecv_signal_data):
        data, labels, signal_features = rfecv_signal_data
        res = run_rfecv_stability(
            data, labels, estimator="linear_svm",
            outer_cv=(5, 4), inner_cv=3, prefilter_top_var=None,
            n_permutations=30, random_state=0,
        )
        # Honest held-out AUC clearly above chance.
        assert res["outer_auc_mean"] > 0.75
        # The 5 planted features are the most frequently selected.
        top5 = list(res["selection_frequency"].head(5).index)
        assert set(signal_features).issubset(set(top5))
        # Permutation null says this beats chance.
        assert res["permutation_p_value"] < 0.05

    def test_return_schema_and_plot_compatibility_keys(self, rfecv_signal_data):
        data, labels, _ = rfecv_signal_data
        res = run_rfecv_stability(
            data, labels, outer_cv=(5, 2), inner_cv=3,
            prefilter_top_var=None, n_permutations=0, random_state=0,
        )
        for key in [
            "estimator", "outer_auc_mean", "outer_auc_std", "auc_roc", "auc_std",
            "balanced_accuracy", "per_fold_scores", "selection_frequency",
            "consensus_features", "n_features_per_fold", "permutation_auc_null",
            "permutation_p_value", "cv_predictions", "fold_roc_data", "class_names",
            "n_features", "y_true", "y_prob", "config",
        ]:
            assert key in res, f"missing key {key}"
        import pandas as pd
        assert isinstance(res["selection_frequency"], pd.Series)
        assert isinstance(res["cv_predictions"], pd.DataFrame)
        # plot_roc_curve must accept the result without raising.
        from proteomics_toolkit.classification import plot_roc_curve
        fig = plot_roc_curve(res)
        assert fig is not None

    def test_logistic_l1_estimator_runs(self, rfecv_signal_data):
        data, labels, signal_features = rfecv_signal_data
        res = run_rfecv_stability(
            data, labels, estimator="logistic_l1",
            outer_cv=(5, 2), inner_cv=3, prefilter_top_var=None,
            n_permutations=0, random_state=0,
        )
        assert res["estimator"] == "logistic_l1"
        assert res["outer_auc_mean"] > 0.75


class TestRunRfecvStabilityNoise:
    def test_noise_auc_near_chance_and_not_significant(self, rfecv_noise_data):
        data, labels = rfecv_noise_data
        res = run_rfecv_stability(
            data, labels, estimator="linear_svm",
            outer_cv=(5, 4), inner_cv=3, prefilter_top_var=None,
            n_permutations=30, random_state=0,
        )
        assert 0.35 < res["outer_auc_mean"] < 0.65
        # No feature is selected in almost every fold.
        assert res["selection_frequency"].max() < 0.9
        # Permutation test does not flag a signal.
        assert res["permutation_p_value"] > 0.05


class TestRunRfecvStabilityErrors:
    def test_too_few_samples_raises(self):
        import pandas as pd
        data = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]], index=["a", "b"], columns=["F0", "F1"]
        )
        labels = pd.Series(["x", "y"], index=["a", "b"])
        with pytest.raises(ValueError, match="at least 10"):
            run_rfecv_stability(data, labels)

    def test_non_binary_labels_raise(self, rfecv_signal_data):
        data, labels, _ = rfecv_signal_data
        labels = labels.copy()
        labels.iloc[0] = "third_class"
        with pytest.raises(ValueError, match="exactly two classes"):
            run_rfecv_stability(data, labels, n_permutations=0)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_classification.py -k "RunRfecvStability" -v`
Expected: FAIL with `cannot import name 'run_rfecv_stability'`.

- [ ] **Step 4: Implement the private outer-CV runner and the public function**

Add to `proteomics_toolkit/classification.py` after the Task 1 helpers:

```python
def _run_rfecv_outer_cv(
    X, y, estimator, outer_cv, inner_cv, step, prefilter_top_var,
    scoring, min_features_to_select, random_state, n_jobs, collect,
):
    """Run the nested outer-CV RFE-CV procedure once for labels ``y``.

    All preprocessing and RFE selection happen on training folds only. When
    ``collect`` is False (permutation runs), per-feature counts, ROC curves and
    predictions are skipped for speed and only per-fold AUCs are returned.

    Returns:
        dict with ``fold_aucs`` (list[float]); when ``collect`` is True also
        ``sel_counts`` (np.ndarray over all features), ``n_features_per_fold``
        (list[int]), ``fold_roc`` (list[(fpr, tpr, auc)]), and ``predictions``
        (list of (sample_index, y_true, y_pred, score, fold_index)).
    """
    import numpy as np
    from sklearn.feature_selection import RFECV
    from sklearn.metrics import auc as sk_auc
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n_splits, n_repeats = outer_cv
    outer = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    inner = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)

    n_features_total = X.shape[1]
    sel_counts = np.zeros(n_features_total) if collect else None
    n_feat_per_fold = [] if collect else None
    fold_roc = [] if collect else None
    predictions = [] if collect else None
    fold_aucs = []

    for fold_i, (tr, te) in enumerate(outer.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Variance prefilter on the training fold only (keeps peptide-scale
        # matrices tractable without leaking test-fold information).
        if prefilter_top_var is not None and prefilter_top_var < n_features_total:
            variances = X_tr_s.var(axis=0)
            keep = np.argsort(variances)[::-1][:prefilter_top_var]
        else:
            keep = np.arange(n_features_total)

        rfecv = RFECV(
            estimator=_make_rfe_estimator(estimator, random_state),
            step=step,
            cv=inner,
            scoring=scoring,
            min_features_to_select=min_features_to_select,
            n_jobs=n_jobs,
        )
        rfecv.fit(X_tr_s[:, keep], y_tr)

        scores = _continuous_scores(rfecv, X_te_s[:, keep])
        y_pred = rfecv.predict(X_te_s[:, keep])

        if len(np.unique(y_te)) == 2:
            fold_aucs.append(roc_auc_score(y_te, scores))
        else:
            fold_aucs.append(np.nan)

        if collect:
            selected_global = keep[rfecv.support_]
            sel_counts[selected_global] += 1
            n_feat_per_fold.append(int(rfecv.n_features_))
            if len(np.unique(y_te)) == 2:
                fpr, tpr, _ = roc_curve(y_te, scores)
                fold_roc.append((fpr, tpr, sk_auc(fpr, tpr)))
            for j, idx in enumerate(te):
                predictions.append((int(idx), int(y_te[j]), int(y_pred[j]), float(scores[j]), fold_i))

    out = {"fold_aucs": fold_aucs}
    if collect:
        out.update(
            sel_counts=sel_counts,
            n_features_per_fold=n_feat_per_fold,
            fold_roc=fold_roc,
            predictions=predictions,
        )
    return out


def run_rfecv_stability(
    data,
    labels,
    estimator="linear_svm",
    outer_cv=(5, 10),
    inner_cv=5,
    step=0.1,
    prefilter_top_var=2000,
    scoring="roc_auc",
    min_features_to_select=1,
    log_transform="auto",
    n_permutations=100,
    consensus_threshold=0.5,
    annotations=None,
    id_col="protein_group",
    gene_col="leading_gene_name",
    random_state=42,
    n_jobs=-1,
):
    """Recursive feature elimination with CV, stability selection, and a null.

    Wraps :class:`sklearn.feature_selection.RFECV` inside an outer
    ``RepeatedStratifiedKFold`` so that feature selection never sees the
    held-out fold it is scored on. This gives an honest performance estimate in
    the n-much-less-than-p proteomics regime, plus a per-feature *selection
    frequency* (how often each feature survives RFE across folds) and a
    label-permutation null for the held-out AUC.

    Args:
        data: DataFrame, samples (rows) x features (columns), numeric.
        labels: Series indexed by sample id with exactly two classes.
        estimator: ``"linear_svm"`` or ``"logistic_l1"``; ranks features by
            ``abs(coef_)``.
        outer_cv: ``(n_splits, n_repeats)`` for the outer RepeatedStratifiedKFold.
        inner_cv: Number of folds for RFECV's internal feature-count search.
        step: Fraction (or count) of features eliminated per RFE iteration.
        prefilter_top_var: Keep this many highest-variance features (computed on
            each training fold) before RFE; ``None`` disables prefiltering.
        scoring: Scoring metric passed to RFECV (default ``"roc_auc"``).
        min_features_to_select: Smallest feature subset RFECV may choose.
        log_transform: ``True``/``False`` or ``"auto"`` (log2 when the matrix
            looks raw-scale, i.e. max value > 100).
        n_permutations: Label-shuffle iterations for the null; ``0`` disables.
        consensus_threshold: Selection-frequency cutoff for ``consensus_features``.
        annotations: Optional DataFrame for relabeling feature ids to gene names.
        id_col, gene_col: Columns in ``annotations`` for the id->gene mapping.
        random_state: Seed for reproducibility.
        n_jobs: Parallelism for RFECV.

    Returns:
        dict with honest performance, ``selection_frequency`` (pd.Series),
        ``consensus_features``, ``permutation_p_value``, ``cv_predictions``,
        ``fold_roc_data`` (compatible with :func:`plot_roc_curve`), and a
        ``config`` echo.

    Raises:
        ValueError: If fewer than 10 shared samples, labels are not binary, or
            ``estimator`` is unknown.
    """
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import LabelEncoder

    if estimator not in ("linear_svm", "logistic_l1"):
        raise ValueError(f"Unknown estimator: {estimator!r}. Use 'linear_svm' or 'logistic_l1'.")

    common = data.index.intersection(labels.index)
    if len(common) < 10:
        raise ValueError(f"Need at least 10 shared samples; got {len(common)}.")

    X_df = data.loc[common].dropna(axis=1, how="any")
    y_raw = labels.loc[common]
    if y_raw.nunique() != 2:
        raise ValueError(f"labels must have exactly two classes; got {y_raw.nunique()}.")

    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)

    do_log = log_transform is True or (log_transform == "auto" and np.nanmax(X) > 100)
    if do_log:
        X = np.log2(np.clip(X, 1.0, None))

    le = LabelEncoder()
    y = le.fit_transform(y_raw.to_numpy())
    class_names = list(le.classes_)

    obs = _run_rfecv_outer_cv(
        X, y, estimator, outer_cv, inner_cv, step, prefilter_top_var,
        scoring, min_features_to_select, random_state, n_jobs, collect=True,
    )

    total_folds = outer_cv[0] * outer_cv[1]
    selection_frequency = (
        pd.Series(obs["sel_counts"] / total_folds, index=feature_names)
        .sort_values(ascending=False)
    )
    consensus_features = list(
        selection_frequency[selection_frequency >= consensus_threshold].index
    )

    fold_aucs = np.asarray(obs["fold_aucs"], dtype=float)
    outer_auc_mean = float(np.nanmean(fold_aucs))
    outer_auc_std = float(np.nanstd(fold_aucs))

    # Permutation null: rerun the whole outer-CV estimate on shuffled labels.
    null_aucs = []
    if n_permutations > 0:
        rng = np.random.RandomState(random_state)
        for _ in range(n_permutations):
            y_perm = rng.permutation(y)
            perm = _run_rfecv_outer_cv(
                X, y_perm, estimator, outer_cv, inner_cv, step, prefilter_top_var,
                scoring, min_features_to_select, random_state, n_jobs, collect=False,
            )
            null_aucs.append(float(np.nanmean(perm["fold_aucs"])))
    null_aucs = np.asarray(null_aucs, dtype=float)
    permutation_p_value = (
        float((1 + np.sum(null_aucs >= outer_auc_mean)) / (1 + n_permutations))
        if n_permutations > 0
        else None
    )

    # Pooled predictions across repeated folds (each sample appears n_repeats times).
    preds = obs["predictions"]
    if preds:
        balanced_acc = float(
            balanced_accuracy_score([p[1] for p in preds], [p[2] for p in preds])
        )
    else:
        balanced_acc = float("nan")
    cv_predictions = pd.DataFrame(
        {
            "Sample": [common[p[0]] for p in preds],
            "True_Label": le.inverse_transform([p[1] for p in preds]),
            "Predicted_Label": le.inverse_transform([p[2] for p in preds]),
            "Score": [p[3] for p in preds],
            "Fold": [p[4] for p in preds],
        }
    )

    if annotations is not None:
        gene_labels = relabel_features_with_genes(
            selection_frequency.index, annotations, id_col=id_col, gene_col=gene_col
        )
        selection_frequency.index = gene_labels
        consensus_features = relabel_features_with_genes(
            consensus_features, annotations, id_col=id_col, gene_col=gene_col
        )

    n_features = int(np.median(obs["n_features_per_fold"])) if obs["n_features_per_fold"] else 0

    logger.info(
        "RFECV stability: estimator=%s, outer AUC=%.3f +/- %.3f, "
        "median features/fold=%d, consensus(>=%.2f)=%d, perm p=%s",
        estimator, outer_auc_mean, outer_auc_std, n_features,
        consensus_threshold, len(consensus_features), permutation_p_value,
    )

    return {
        "estimator": estimator,
        "outer_auc_mean": outer_auc_mean,
        "outer_auc_std": outer_auc_std,
        "auc_roc": outer_auc_mean,
        "auc_std": outer_auc_std,
        "balanced_accuracy": balanced_acc,
        "per_fold_scores": fold_aucs.tolist(),
        "selection_frequency": selection_frequency,
        "consensus_features": consensus_features,
        "n_features_per_fold": obs["n_features_per_fold"],
        "permutation_auc_null": null_aucs,
        "permutation_p_value": permutation_p_value,
        "cv_predictions": cv_predictions,
        "fold_roc_data": obs["fold_roc"],
        "class_names": class_names,
        "n_features": n_features,
        "y_true": np.array([p[1] for p in preds]),
        "y_prob": np.array([p[3] for p in preds]),
        "config": {
            "outer_cv": outer_cv,
            "inner_cv": inner_cv,
            "step": step,
            "prefilter_top_var": prefilter_top_var,
            "scoring": scoring,
            "consensus_threshold": consensus_threshold,
            "log_transform_applied": bool(do_log),
            "n_permutations": n_permutations,
        },
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_classification.py -k "RunRfecvStability" -v`
Expected: PASS (all tests in the three new classes). If the signal-data AUC assertion is flaky, it indicates a real implementation bug, not a threshold problem — do not loosen the threshold without investigating.

- [ ] **Step 6: Commit**

```bash
git add proteomics_toolkit/classification.py tests/test_classification.py tests/conftest.py
git commit -m "Added run_rfecv_stability with nested CV and permutation null"
```

---

## Task 3: `plot_selection_frequency` helper

**Files:**
- Modify: `proteomics_toolkit/classification.py` (add after `plot_roc_curve`)
- Test: `tests/test_classification.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_classification.py`:

```python
class TestPlotSelectionFrequency:
    def test_returns_figure(self, rfecv_signal_data):
        import matplotlib
        matplotlib.use("Agg")
        from proteomics_toolkit.classification import plot_selection_frequency

        data, labels, _ = rfecv_signal_data
        res = run_rfecv_stability(
            data, labels, outer_cv=(5, 2), inner_cv=3,
            prefilter_top_var=None, n_permutations=0, random_state=0,
        )
        fig = plot_selection_frequency(res, top_n=10)
        assert fig is not None
        ax = fig.axes[0]
        # At most top_n features are drawn.
        assert len(ax.get_yticklabels()) <= 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_classification.py -k "PlotSelectionFrequency" -v`
Expected: FAIL with `cannot import name 'plot_selection_frequency'`.

- [ ] **Step 3: Implement the plot**

Add to `proteomics_toolkit/classification.py` after `plot_roc_curve`:

```python
def plot_selection_frequency(
    results, top_n=30, title="RFECV Selection Frequency", figsize=(8, 9), color="#1f77b4"
):
    """Lollipop plot of the most stable features from ``run_rfecv_stability``.

    Args:
        results: The dict returned by :func:`run_rfecv_stability`.
        top_n: Number of top features (by selection frequency) to show.
        title: Plot title.
        figsize: Figure size.
        color: Marker/stem color.

    Returns:
        matplotlib Figure. The ``consensus_threshold`` is drawn as a dashed
        reference line.
    """
    import matplotlib.pyplot as plt

    freq = results["selection_frequency"].head(top_n)
    threshold = results["config"].get("consensus_threshold", 0.5)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = range(len(freq))
    ax.hlines(y=y_pos, xmin=0, xmax=freq.values, color=color, alpha=0.6)
    ax.plot(freq.values, list(y_pos), "o", color=color)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(freq.index)
    ax.invert_yaxis()  # most frequent at top
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=1, label=f"consensus >= {threshold:g}")
    ax.set_xlabel("Selection frequency across outer CV folds")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_classification.py -k "PlotSelectionFrequency" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add proteomics_toolkit/classification.py tests/test_classification.py
git commit -m "Added plot_selection_frequency for RFECV stability results"
```

---

## Task 4: Export from package top level

**Files:**
- Modify: `proteomics_toolkit/__init__.py`
- Test: `tests/test_classification.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_classification.py`:

```python
class TestRfecvExports:
    def test_top_level_imports(self):
        import proteomics_toolkit as ptk

        assert hasattr(ptk, "run_rfecv_stability")
        assert hasattr(ptk, "plot_selection_frequency")
        assert "run_rfecv_stability" in ptk.__all__
        assert "plot_selection_frequency" in ptk.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_classification.py -k "RfecvExports" -v`
Expected: FAIL with `AttributeError: module 'proteomics_toolkit' has no attribute 'run_rfecv_stability'`.

- [ ] **Step 3: Add the exports**

In `proteomics_toolkit/__init__.py`, find the classification import block (around line 119-129) and add the two names alphabetically. For example, after the line `relabel_features_with_genes,` add:

```python
    run_rfecv_stability,  # Nested-CV RFE with stability selection + null
    plot_selection_frequency,  # Lollipop of RFECV selection frequency
```

Then in the `__all__` list (around line 367), add:

```python
    "run_rfecv_stability",  # Nested-CV RFE with stability selection + null
    "plot_selection_frequency",  # Lollipop of RFECV selection frequency
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_classification.py -k "RfecvExports" -v`
Expected: PASS.

- [ ] **Step 5: Run the full classification test module**

Run: `uv run pytest tests/test_classification.py -v`
Expected: PASS (all prior tests plus the new ones).

- [ ] **Step 6: Lint**

Run: `uv run ruff check proteomics_toolkit/classification.py proteomics_toolkit/__init__.py`
Expected: no errors. Fix any reported issues, then re-run.

- [ ] **Step 7: Commit**

```bash
git add proteomics_toolkit/__init__.py tests/test_classification.py
git commit -m "Exported run_rfecv_stability and plot_selection_frequency"
```

---

## Task 5: Documentation

**Files:**
- Modify: `docs/09-classification.md`
- Modify: `README.md`

- [ ] **Step 1: Append a section to `docs/09-classification.md`**

Add at the end of the file:

```markdown
## Recursive feature elimination with stability selection

`run_rfecv_stability(data, labels, ...)` finds a compact classifying signature
in the n-much-less-than-p regime without the optimistic bias of a plain
`sklearn.RFECV`. It wraps RFECV in an outer `RepeatedStratifiedKFold`, so
feature selection never sees the fold it is scored on.

It returns:

- `outer_auc_mean` / `outer_auc_std`: honest held-out AUC (with `fold_roc_data`
  for `plot_roc_curve`, which draws the mean ROC with a +/- 1 SD band).
- `selection_frequency`: a Series giving the fraction of outer folds in which
  each feature survived RFE. Features near 1.0 "consistently survive CV".
- `consensus_features`: features above `consensus_threshold` (default 0.5).
- `permutation_p_value`: empirical p from a label-shuffle null on the AUC.

```python
import proteomics_toolkit as ptk

res = ptk.run_rfecv_stability(
    expr,                 # samples x features
    labels,               # binary Series indexed by sample
    estimator="linear_svm",
    annotations=protein_table,   # optional gene relabeling
)
print(res["outer_auc_mean"], res["permutation_p_value"])
ptk.plot_selection_frequency(res, top_n=30)
ptk.plot_roc_curve(res)
```

For peptide-scale matrices, keep `prefilter_top_var` set (e.g. 2000-3000) and
lower `n_permutations` to control runtime.
```

- [ ] **Step 2: Add a bullet to `README.md`**

In `README.md`, find the `- **classification**:` feature line and append to its description:

```
; recursive feature elimination with cross-validation, per-feature stability selection, and a permutation null (`run_rfecv_stability`)
```

- [ ] **Step 3: Commit**

```bash
git add docs/09-classification.md README.md
git commit -m "Documented run_rfecv_stability"
```

---

## Task 6: Apply in notebook 01 (proteins)

**Files:**
- Modify: `../collab-uw-vojech/2026-05-EV-Opioid-Pregnancy/notebooks/01_opioid_ssri_differential_and_clustering.ipynb`

**Context:** The protein matrix is features x samples (`corrected_proteins.parquet`, ~5,486 proteins x ~80 experimental samples). Sample metadata carries `opioid_use` / `ssri_use` (0/1) and `sample_type`. The `preg_only + lenient` config = pregnant experimental samples with known drug status. `run_rfecv_stability` wants a samples x features matrix, so transpose. Reuse the notebook's existing matrix/metadata variables rather than reloading; inspect the notebook to find their exact names before writing the cells.

- [ ] **Step 1: Read the notebook to find existing variable names**

Run: `cd ../collab-uw-vojech && uv run jupyter nbconvert --to script --stdout 2026-05-EV-Opioid-Pregnancy/notebooks/01_opioid_ssri_differential_and_clustering.ipynb | grep -nE "protein|metadata|preg|opioid_use|experimental|corrected" | head -60`

Identify: the protein intensity DataFrame (features x samples), the sample-metadata DataFrame, the `protein_group`/`leading_gene_name` annotation columns, and the mask defining `preg_only + lenient`.

- [ ] **Step 2: Add a markdown cell**

Insert near the end of the notebook (after the clustering section):

```markdown
## Recursive feature elimination with cross-validation (RFECV)

Can a compact set of EV proteins classify opioid (or SSRI) exposure and survive
cross-validation? We run `ptk.run_rfecv_stability`, which wraps RFECV in an
outer repeated CV (honest AUC), reports how often each protein survives RFE
across folds, and tests the AUC against a label-shuffle null. SSRI serves as a
built-in negative control (no differential hits upstream).
```

- [ ] **Step 3: Add a code cell that builds inputs and runs both estimators**

Adapt the variable names from Step 1 (placeholders `protein_matrix`, `meta`, `annot` shown here):

```python
import proteomics_toolkit as ptk
import pandas as pd

# preg_only + lenient: pregnant experimental samples with known drug status.
preg_lenient = meta[
    (meta["sample_type"] == "experimental") & meta["fetal_sex"].notna()
]

# samples x features for the toolkit (transpose the features x samples matrix).
expr = protein_matrix[preg_lenient["sample"]].T
expr.index = preg_lenient["sample"].to_numpy()

rfecv_results = {}
for drug in ["opioid_use", "ssri_use"]:
    labels = (
        preg_lenient.set_index("sample")[drug]
        .reindex(expr.index)
        .dropna()
        .map({0: f"non_{drug}", 1: drug})
    )
    X = expr.loc[labels.index]
    for est in ["linear_svm", "logistic_l1"]:
        print(f"=== {drug} / {est} (n={len(labels)}) ===")
        res = ptk.run_rfecv_stability(
            X, labels, estimator=est,
            prefilter_top_var=2000, n_permutations=100,
            annotations=annot, id_col="protein_group", gene_col="leading_gene_name",
            random_state=0,
        )
        rfecv_results[(drug, est)] = res
        print(
            f"AUC = {res['outer_auc_mean']:.3f} +/- {res['outer_auc_std']:.3f}, "
            f"perm p = {res['permutation_p_value']:.3f}, "
            f"consensus = {res['consensus_features']}"
        )
```

- [ ] **Step 4: Add a code cell for plots (opioid, linear SVM)**

```python
res = rfecv_results[("opioid_use", "linear_svm")]
ptk.plot_roc_curve(res, title="Opioid RFECV ROC (linear SVM)")
ptk.plot_selection_frequency(res, top_n=25, title="Opioid: protein selection frequency")
```

- [ ] **Step 5: Run the notebook end-to-end to confirm it executes**

Run: `cd ../collab-uw-vojech && uv run jupyter nbconvert --to notebook --execute --inplace 2026-05-EV-Opioid-Pregnancy/notebooks/01_opioid_ssri_differential_and_clustering.ipynb --ExecutePreprocessor.timeout=1800`
Expected: completes without errors. Confirm the printed opioid AUC is above the SSRI AUC and that the SSRI permutation p is not significant (the negative-control expectation). If the notebook is too slow, lower `n_permutations` to 50 in the code cell and rerun.

- [ ] **Step 6: Commit**

```bash
cd ../collab-uw-vojech
git add 2026-05-EV-Opioid-Pregnancy/notebooks/01_opioid_ssri_differential_and_clustering.ipynb
git commit -m "Added RFECV stability analysis to protein notebook"
cd ../proteomics-toolkit
```

---

## Task 7: Apply in notebook 02 (peptides)

**Files:**
- Modify: `../collab-uw-vojech/2026-05-EV-Opioid-Pregnancy/notebooks/02_opioid_ssri_peptide_differential_and_clustering.ipynb`

**Context:** The peptide matrix is ~66,171 peptides x ~80 samples (`corrected_peptides.parquet`); features are `PeptideModifiedSequenceUnimodIds`, mapped to genes via `GroupID`/protein-group annotation. Same `preg_only + lenient` masking as Task 6. Because of feature count, use a tighter prefilter and fewer permutations to stay in the minutes range.

- [ ] **Step 1: Read the notebook to find existing variable names**

Run: `cd ../collab-uw-vojech && uv run jupyter nbconvert --to script --stdout 2026-05-EV-Opioid-Pregnancy/notebooks/02_opioid_ssri_peptide_differential_and_clustering.ipynb | grep -nE "peptide|metadata|preg|opioid_use|GroupID|Gene|experimental|corrected" | head -60`

Identify: the peptide intensity DataFrame (features x samples), the peptide annotation columns for the id->gene relabeling, and the `preg_only + lenient` mask.

- [ ] **Step 2: Add a markdown cell**

Insert near the end of the notebook:

```markdown
## Recursive feature elimination with cross-validation (RFECV)

Peptide-level RFECV with the same honest outer-CV, stability, and permutation-
null procedure as the protein notebook. With ~66k peptides we keep a training-
fold variance prefilter and a coarse elimination step so the run stays in the
minutes range.
```

- [ ] **Step 3: Add a code cell that builds inputs and runs RFECV**

Adapt variable names from Step 1 (placeholders `peptide_matrix`, `meta`, `pep_annot`, with `id_col`/`gene_col` matching the peptide annotation):

```python
import proteomics_toolkit as ptk

preg_lenient = meta[
    (meta["sample_type"] == "experimental") & meta["fetal_sex"].notna()
]
expr = peptide_matrix[preg_lenient["sample"]].T
expr.index = preg_lenient["sample"].to_numpy()

pep_rfecv_results = {}
for drug in ["opioid_use", "ssri_use"]:
    labels = (
        preg_lenient.set_index("sample")[drug]
        .reindex(expr.index)
        .dropna()
        .map({0: f"non_{drug}", 1: drug})
    )
    X = expr.loc[labels.index]
    print(f"=== {drug} / linear_svm (n={len(labels)}, peptides={X.shape[1]}) ===")
    res = ptk.run_rfecv_stability(
        X, labels, estimator="linear_svm",
        prefilter_top_var=3000, step=0.1, n_permutations=50,
        annotations=pep_annot, id_col="PeptideModifiedSequenceUnimodIds", gene_col="Gene",
        random_state=0,
    )
    pep_rfecv_results[drug] = res
    print(
        f"AUC = {res['outer_auc_mean']:.3f} +/- {res['outer_auc_std']:.3f}, "
        f"perm p = {res['permutation_p_value']:.3f}, "
        f"consensus = {res['consensus_features']}"
    )
```

Note: replace `id_col`/`gene_col` with the actual peptide-id and gene columns found in Step 1.

- [ ] **Step 4: Add a plotting cell (opioid)**

```python
res = pep_rfecv_results["opioid_use"]
ptk.plot_roc_curve(res, title="Opioid peptide RFECV ROC (linear SVM)")
ptk.plot_selection_frequency(res, top_n=25, title="Opioid: peptide selection frequency")
```

- [ ] **Step 5: Run the notebook end-to-end**

Run: `cd ../collab-uw-vojech && uv run jupyter nbconvert --to notebook --execute --inplace 2026-05-EV-Opioid-Pregnancy/notebooks/02_opioid_ssri_peptide_differential_and_clustering.ipynb --ExecutePreprocessor.timeout=3600`
Expected: completes without errors. If runtime is excessive, lower `n_permutations` to 25 and/or `prefilter_top_var` to 2000 and rerun. Confirm opioid consensus peptides are dominated by CHGA (the upstream differential result) and SSRI is at chance.

- [ ] **Step 6: Commit**

```bash
cd ../collab-uw-vojech
git add 2026-05-EV-Opioid-Pregnancy/notebooks/02_opioid_ssri_peptide_differential_and_clustering.ipynb
git commit -m "Added RFECV stability analysis to peptide notebook"
cd ../proteomics-toolkit
```

---

## Final verification

- [ ] **Run the full toolkit test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass.

- [ ] **Lint the whole package**

Run: `uv run ruff check proteomics_toolkit/`
Expected: no errors.

- [ ] **Confirm the public API**

Run: `uv run python -c "import proteomics_toolkit as ptk; print(ptk.run_rfecv_stability.__doc__.splitlines()[0]); print('plot_selection_frequency' in ptk.__all__)"`
Expected: prints the docstring summary line and `True`.
