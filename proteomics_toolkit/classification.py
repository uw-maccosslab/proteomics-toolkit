"""
Classification and Feature-Importance Module for Proteomics Data

Provides:

- Binary classification (logistic regression / RF / SVM / XGBoost) with
  cross-validated performance metrics, ROC plotting, and per-fold
  diagnostics. Designed for paired-fold-change designs.
- Multi-class permutation-importance ranking with bootstrap stability
  scoring, for descriptive feature discovery in low-replication designs
  where per-protein hypothesis tests are underpowered.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def select_features_by_mad(fold_change_matrix, n_top_features=50):
    """Rank features by median absolute deviation across subjects (unsupervised).

    MAD is a label-free measure of variability, so using it for feature
    selection cannot leak outcome information into downstream classification.
    Preferred over fold-change-based selection for small datasets.

    Args:
        fold_change_matrix: DataFrame with subjects as rows and proteins
            as columns.
        n_top_features: Number of features to return.

    Returns:
        List of top-N feature column names (ordered most-variable first).
    """
    medians = fold_change_matrix.median(axis=0)
    mad = (fold_change_matrix.sub(medians, axis=1)).abs().median(axis=0)
    top = mad.nlargest(min(n_top_features, len(mad)))
    return list(top.index)


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
        return LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=5000, random_state=random_state)
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


def _run_rfecv_outer_cv(
    X,
    y,
    estimator,
    outer_cv,
    inner_cv,
    step,
    prefilter_top_var,
    scoring,
    min_features_to_select,
    random_state,
    n_jobs,
    collect,
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
    from sklearn.feature_selection import RFECV
    from sklearn.metrics import auc as sk_auc
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n_splits, n_repeats = outer_cv
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
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
        id_col: Column in ``annotations`` matching feature ids.
        gene_col: Column in ``annotations`` holding gene symbols.
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
        X,
        y,
        estimator,
        outer_cv,
        inner_cv,
        step,
        prefilter_top_var,
        scoring,
        min_features_to_select,
        random_state,
        n_jobs,
        collect=True,
    )

    total_folds = outer_cv[0] * outer_cv[1]
    selection_frequency = pd.Series(obs["sel_counts"] / total_folds, index=feature_names).sort_values(ascending=False)
    consensus_features = list(selection_frequency[selection_frequency >= consensus_threshold].index)

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
                X,
                y_perm,
                estimator,
                outer_cv,
                inner_cv,
                step,
                prefilter_top_var,
                scoring,
                min_features_to_select,
                random_state,
                n_jobs,
                collect=False,
            )
            null_aucs.append(float(np.nanmean(perm["fold_aucs"])))
    null_aucs = np.asarray(null_aucs, dtype=float)
    permutation_p_value = (
        float((1 + np.sum(null_aucs >= outer_auc_mean)) / (1 + n_permutations)) if n_permutations > 0 else None
    )

    # Pooled predictions across repeated folds (each sample appears n_repeats times).
    preds = obs["predictions"]
    if preds:
        balanced_acc = float(balanced_accuracy_score([p[1] for p in preds], [p[2] for p in preds]))
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
        estimator,
        outer_auc_mean,
        outer_auc_std,
        n_features,
        consensus_threshold,
        len(consensus_features),
        permutation_p_value,
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


def _select_top_by_ttest(X_train, y_train, n_top_features):
    """Rank columns of X_train by |Welch t-statistic| between the two classes.

    Private helper used for nested differential-abundance feature selection
    inside cross-validation. Features are ranked using only the training
    subset of each fold so the held-out test split never influences
    selection.
    """
    from scipy.stats import ttest_ind

    classes = np.unique(y_train)
    if len(classes) != 2:
        # Degenerate fold - fall back to MAD ranking on the training split
        train_df = pd.DataFrame(X_train)
        return list(select_features_by_mad(train_df, n_top_features=n_top_features))
    mask0 = y_train == classes[0]
    mask1 = y_train == classes[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        tstat, _ = ttest_ind(X_train[mask0], X_train[mask1], axis=0, equal_var=False, nan_policy="omit")
    tstat = np.abs(np.asarray(tstat, dtype=float))
    tstat[~np.isfinite(tstat)] = -np.inf
    order = np.argsort(-tstat)  # descending
    n = min(n_top_features, len(order))
    return order[:n].tolist()


def _select_top_by_variance_ratio(X_train, y_train, n_top_features):
    """Rank columns of X_train by inter-group / intra-group variance ratio.

    For each feature, computes ``var(group_means) / mean(within_group_var)``
    using only the training subset. This is the same ranking criterion as
    :func:`~proteomics_toolkit.marker_discovery.inter_vs_intra_group_variance`
    but computed per-fold so the held-out test labels never enter feature
    ranking. For 2+ groups it is monotonically related to the one-way ANOVA
    F-statistic.

    Falls back to MAD ranking when a fold has fewer than two distinct
    classes (degenerate stratification).
    """
    classes = np.unique(y_train)
    if len(classes) < 2:
        train_df = pd.DataFrame(X_train)
        return list(select_features_by_mad(train_df, n_top_features=n_top_features))

    means_list, vars_list = [], []
    for c in classes:
        mask = y_train == c
        means_list.append(np.nanmean(X_train[mask], axis=0))
        # ddof=1 sample variance, matching numpy/limma conventions
        vars_list.append(np.nanvar(X_train[mask], axis=0, ddof=1))
    means_mat = np.column_stack(means_list)
    vars_mat = np.column_stack(vars_list)

    with np.errstate(invalid="ignore", divide="ignore"):
        inter = np.nanvar(means_mat, axis=1, ddof=1)
        intra = np.nanmean(vars_mat, axis=1)
        ratio = np.where(intra > 0, inter / intra, np.nan)

    ratio = np.asarray(ratio, dtype=float)
    ratio[~np.isfinite(ratio)] = -np.inf
    order = np.argsort(-ratio)  # descending
    n = min(n_top_features, len(order))
    return order[:n].tolist()


def run_binary_classification(
    fold_change_matrix,
    group_labels,
    feature_proteins=None,
    n_top_features=50,
    feature_selection="mad",
    method="logistic_regression",
    cv_method=5,
    return_model=False,
    annotations=None,
    id_col="protein_group",
    gene_col="leading_gene_name",
):
    """Run binary classification with feature selection and cross-validation.

    Classifies subjects into two groups using protein fold-changes as features.
    Uses stratified k-fold cross-validation (default 5-fold) to estimate
    out-of-sample performance. Per-fold ROC data is stored for plotting
    mean +/- SD ROC curves.

    Feature selection strategies (``feature_selection`` argument):

    - ``"mad"`` *(default, recommended)*: unsupervised selection by median
      absolute deviation across subjects. The outcome label is never
      consulted, so there is no selection-to-classification leakage.
    - ``"differential_abundance"``: nested supervised selection. Inside
      each CV fold, a per-feature Welch t-test is fit on the training
      split only; the top ``n_top_features`` by |t| are used to train
      the classifier and score the held-out test split. This is the
      statistically correct way to use a supervised ranker without
      leakage.
    - ``"variance_ratio"``: nested supervised selection by inter-group /
      intra-group variance ratio (the F-ratio used by
      :func:`~proteomics_toolkit.marker_discovery.inter_vs_intra_group_variance`).
      Per-fold: rank features by ``var(group_means) / mean(within_group_var)``
      on the training subset only, take the top ``n_top_features``. Like
      ``"differential_abundance"`` this avoids leakage by re-selecting per
      fold; useful when you want to rank by "are the group means well
      separated relative to within-group spread" rather than a pairwise
      t-statistic.
    - ``"fold_change"``: legacy behaviour - top by absolute mean
      fold-change across all subjects. **Leaky on small datasets** when
      classes differ systematically; use MAD, nested DA, or nested
      variance_ratio instead.

    Passing an explicit ``feature_proteins`` list overrides the automatic
    selection for every fold. This is appropriate when the feature list
    comes from a truly independent source (different cohort, prior
    biological knowledge).

    Args:
        fold_change_matrix: DataFrame with subjects as rows and proteins as
            columns. Values are typically per-subject fold-changes.
        group_labels: Series mapping subject IDs (matching fold_change_matrix
            index) to binary group labels (e.g., 'R' / 'NR').
        feature_proteins: Optional list of protein IDs to use as features.
            When supplied, bypasses ``feature_selection`` and uses these
            features in every fold.
        n_top_features: Number of top features to select when
            ``feature_proteins`` is None. Applies to all three selection
            strategies.
        feature_selection: One of ``"mad"`` (default),
            ``"differential_abundance"``, ``"variance_ratio"``, or
            ``"fold_change"``. See above for details.
        method: Classification method. One of 'logistic_regression',
            'random_forest', 'linear_svm', or 'xgboost'.
        cv_method: Cross-validation strategy. Integer for k-fold (default 5),
            or 'loo' for leave-one-out (not recommended for ROC curves).
        return_model: If True, also include the final fitted classifier,
            its StandardScaler, the scaled feature matrix, and the
            encoded labels in the returned dict (keys ``final_model``,
            ``scaler``, ``X_scaled``, ``y_encoded``). Use this when
            you need the trained model for downstream interpretability
            (e.g., :func:`compute_shap_values`).
        annotations: Optional DataFrame containing ``id_col`` and
            ``gene_col``. When provided, the returned ``feature_names``
            list and ``feature_importances`` Series are relabeled from
            pipeline IDs to gene symbols via
            :func:`relabel_features_with_genes`. This keeps plots and
            tables interpretable without the caller having to remap
            ``protein_group`` -> gene every time. The model is still
            trained on the original feature IDs internally; only the
            return values are relabeled.
        id_col: Identifier column in ``annotations`` matching the
            columns of ``fold_change_matrix``. Ignored if
            ``annotations`` is None.
        gene_col: Gene-name column in ``annotations``. Ignored if
            ``annotations`` is None.

    Returns:
        Dict with keys:
            'accuracy': Overall classification accuracy.
            'balanced_accuracy': Accuracy balanced across classes.
            'auc_roc': Mean AUC-ROC across folds.
            'auc_std': Standard deviation of AUC across folds.
            'confusion_matrix': Confusion matrix as numpy array.
            'cv_predictions': DataFrame with true and predicted labels.
            'feature_importances': Series of feature importance scores
                (from final fit on all data; for nested selectors
                ``differential_abundance`` and ``variance_ratio`` this
                uses the full-data ranking and is meant as a descriptive
                summary, not a CV-validated metric).
            'classification_report': Text classification report.
            'n_features': Number of features used per fold.
            'feature_names': For ``mad`` / ``fold_change`` / explicit
                ``feature_proteins``: the fixed feature set. For
                ``differential_abundance`` / ``variance_ratio``: the
                feature set selected from the full data (distinct from
                the per-fold sets that drove CV performance).
            'feature_selection': Echo of the strategy used.
            'y_true': Encoded true labels array.
            'y_prob': Predicted probability array.
            'class_names': Array of class name strings.
            'fold_roc_data': List of (fpr, tpr, auc) tuples per fold.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        auc,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    # Align subjects
    common_subjects = fold_change_matrix.index.intersection(group_labels.index)
    if len(common_subjects) < 10:
        raise ValueError(
            f"Only {len(common_subjects)} subjects have both fold-change data and group labels. Need at least 10."
        )

    X = fold_change_matrix.loc[common_subjects].copy()
    y = group_labels.loc[common_subjects].copy()

    # Drop proteins with any NaN across subjects
    X = X.dropna(axis=1)

    # Feature-selection resolution
    valid_selection = {"mad", "differential_abundance", "variance_ratio", "fold_change"}
    if feature_selection not in valid_selection:
        raise ValueError(f"feature_selection must be one of {sorted(valid_selection)}; got {feature_selection!r}")

    # When set, this is the per-fold ranker for a nested-CV strategy.
    # None means feature selection (if any) was already done up-front and
    # the per-fold loop runs without re-selecting.
    nested_selector_fn = None
    if feature_proteins is not None:
        available = [p for p in feature_proteins if p in X.columns]
        if len(available) < 2:
            logger.warning(
                "Fewer than 2 specified feature proteins found in data. "
                "Falling back to %s selection with n_top_features=%d.",
                feature_selection,
                n_top_features,
            )
            feature_proteins = None
        else:
            X = X[available]

    if feature_proteins is None:
        if feature_selection == "mad":
            top_names = select_features_by_mad(X, n_top_features=n_top_features)
            X = X[top_names]
        elif feature_selection == "fold_change":
            mean_abs_fc = X.abs().mean(axis=0)
            top_features = mean_abs_fc.nlargest(min(n_top_features, len(mean_abs_fc)))
            X = X[top_features.index]
        elif feature_selection == "differential_abundance":
            nested_selector_fn = _select_top_by_ttest
        elif feature_selection == "variance_ratio":
            nested_selector_fn = _select_top_by_variance_ratio

    feature_names = list(X.columns)
    n_features = min(n_top_features, len(feature_names)) if nested_selector_fn is not None else len(feature_names)
    if nested_selector_fn is not None:
        print(
            f"Classification using nested {feature_selection} selection "
            f"(top {n_features} per fold, from {len(feature_names)} candidates), "
            f"{len(common_subjects)} subjects"
        )
    else:
        print(f"Classification using {n_features} features, {len(common_subjects)} subjects")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    # Scale features
    scaler = StandardScaler()

    # Set up cross-validation
    if cv_method == "loo":
        cv = LeaveOneOut()
        cv_label = "Leave-One-Out"
    else:
        cv = StratifiedKFold(n_splits=int(cv_method), shuffle=True, random_state=42)
        cv_label = f"{cv_method}-Fold"

    # Set up classifier factory (create fresh per fold)
    def make_clf():
        if method == "logistic_regression":
            return LogisticRegression(max_iter=5000, solver="saga", penalty="l1", C=1.0, random_state=42)
        elif method == "random_forest":
            return RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1
            )
        elif method == "linear_svm":
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.svm import LinearSVC

            svm_base = LinearSVC(C=0.01, random_state=42, max_iter=5000)
            return CalibratedClassifierCV(svm_base, cv=3)
        elif method == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'logistic_regression', 'random_forest', 'linear_svm', or 'xgboost'."
            )

    # Cross-validated predictions and per-fold ROC data
    y_pred_all = np.zeros(len(y_encoded), dtype=int)
    y_prob_all = np.zeros(len(y_encoded))
    fold_roc_data = []

    X_values = X.values
    for train_idx, test_idx in cv.split(X_values, y_encoded):
        X_train_raw = X_values[train_idx]
        X_test_raw = X_values[test_idx]
        y_train = y_encoded[train_idx]
        y_test = y_encoded[test_idx]

        if nested_selector_fn is not None:
            # Rank features on the training split only, then subset both splits.
            top_cols = nested_selector_fn(X_train_raw, y_train, n_top_features)
            X_train_raw = X_train_raw[:, top_cols]
            X_test_raw = X_test_raw[:, top_cols]

        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        clf = make_clf()
        clf.fit(X_train, y_train)
        y_pred_all[test_idx] = clf.predict(X_test)

        if hasattr(clf, "predict_proba"):
            fold_probs = clf.predict_proba(X_test)[:, 1]
            y_prob_all[test_idx] = fold_probs

            # Compute per-fold ROC curve (only if enough test samples)
            if len(test_idx) >= 2 and len(np.unique(y_test)) == 2:
                fpr_fold, tpr_fold, _ = roc_curve(y_test, fold_probs)
                auc_fold = auc(fpr_fold, tpr_fold)
                fold_roc_data.append((fpr_fold, tpr_fold, auc_fold))

    # Check for label inversion before computing metrics.
    # LabelEncoder assigns labels alphabetically (NR=0, R=1), but the classifier
    # may learn the mapping in reverse. Detect this via AUC < 0.5 and flip.
    #
    # When flipping, each per-fold ROC must be reflected across y = x
    # (swap fpr <-> tpr, new_auc = 1 - old_auc). Previously only the scalar
    # per-fold auc was swapped to max(a, 1-a) while the (fpr, tpr) arrays were
    # left untouched, which left plotted curves below the diagonal even though
    # the reported AUC was >= 0.5.
    try:
        auc_overall = roc_auc_score(y_encoded, y_prob_all)
        if auc_overall < 0.5:
            auc_overall = 1.0 - auc_overall
            y_prob_all = 1.0 - y_prob_all
            y_pred_all = 1 - y_pred_all
            fold_roc_data = [(tpr, fpr, 1.0 - a) for fpr, tpr, a in fold_roc_data]
    except ValueError:
        auc_overall = None

    # Compute metrics after any label inversion correction
    acc = accuracy_score(y_encoded, y_pred_all)
    bal_acc = balanced_accuracy_score(y_encoded, y_pred_all)
    cm = confusion_matrix(y_encoded, y_pred_all)
    report = classification_report(y_encoded, y_pred_all, target_names=class_names)

    # Fold AUC statistics
    if fold_roc_data:
        fold_aucs = [a for _, _, a in fold_roc_data]
        auc_mean = np.mean(fold_aucs)
        auc_std = np.std(fold_aucs)
    else:
        auc_mean = auc_overall
        auc_std = 0.0

    # Fit final model on all data for feature importances. For nested
    # selectors, re-rank features using the *full dataset*; this is
    # separate from the CV performance (which used per-fold selections)
    # and is meant as a descriptive summary of which features the method
    # prefers when fed the whole cohort.
    if nested_selector_fn is not None:
        final_top_cols = nested_selector_fn(X_values, y_encoded, n_top_features)
        final_feature_names = [feature_names[i] for i in final_top_cols]
        X_final = X_values[:, final_top_cols]
    else:
        final_feature_names = feature_names
        X_final = X_values

    X_scaled = scaler.fit_transform(X_final)
    final_clf = make_clf()
    final_clf.fit(X_scaled, y_encoded)

    if method == "logistic_regression":
        importances = pd.Series(final_clf.coef_[0], index=final_feature_names).abs()
    elif method == "linear_svm":
        # CalibratedClassifierCV wraps the SVC; train a bare LinearSVC for coefficients
        from sklearn.svm import LinearSVC

        svm_for_coef = LinearSVC(C=0.01, random_state=42, max_iter=5000)
        svm_for_coef.fit(X_scaled, y_encoded)
        importances = pd.Series(np.abs(svm_for_coef.coef_[0]), index=final_feature_names)
    elif method == "xgboost":
        importances = pd.Series(final_clf.feature_importances_, index=final_feature_names)
    else:
        importances = pd.Series(final_clf.feature_importances_, index=final_feature_names)
    importances = importances.sort_values(ascending=False)

    # Build predictions DataFrame
    cv_predictions = pd.DataFrame(
        {
            "Subject": common_subjects,
            "True_Label": le.inverse_transform(y_encoded),
            "Predicted_Label": le.inverse_transform(y_pred_all),
            "Predicted_Probability": y_prob_all,
            "Correct": y_encoded == y_pred_all,
        }
    ).set_index("Subject")

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Binary Classification Results ({method})")
    print(f"{'=' * 50}")
    print(f"Classes: {class_names[0]} vs {class_names[1]}")
    print(
        f"Subjects: {len(common_subjects)} ({(y_encoded == 0).sum()} {class_names[0]}, "
        f"{(y_encoded == 1).sum()} {class_names[1]})"
    )
    print(f"Features: {n_features}")
    print(f"CV Method: {cv_label}")
    print(f"\nAccuracy: {acc:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    if auc_mean is not None:
        print(f"AUC-ROC: {auc_mean:.3f} +/- {auc_std:.3f}")
    print("\nConfusion Matrix:")
    print(f"  Predicted:  {class_names[0]:>6}  {class_names[1]:>6}")
    for i, label in enumerate(class_names):
        print(f"  {label:>10}  {cm[i, 0]:>6}  {cm[i, 1]:>6}")
    print(f"\n{report}")

    # If annotations are provided, relabel the returned feature_names list
    # and feature_importances Series index from pipeline IDs (e.g. PG####)
    # to gene symbols, so downstream plots and tables are interpretable
    # without the caller having to remap.
    if annotations is not None:
        gene_labels = relabel_features_with_genes(
            final_feature_names,
            annotations,
            id_col=id_col,
            gene_col=gene_col,
        )
        feature_names_out = gene_labels
        importances_out = pd.Series(importances.values, index=gene_labels).sort_values(ascending=False)
    else:
        feature_names_out = final_feature_names
        importances_out = importances

    result = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "auc_roc": auc_mean,
        "auc_std": auc_std,
        "confusion_matrix": cm,
        "cv_predictions": cv_predictions,
        "feature_importances": importances_out,
        "classification_report": report,
        "n_features": n_features,
        "feature_names": feature_names_out,
        "feature_selection": ("explicit" if feature_proteins is not None else feature_selection),
        "y_true": y_encoded,
        "y_prob": y_prob_all,
        "class_names": class_names,
        "fold_roc_data": fold_roc_data,
    }
    if return_model:
        result["final_model"] = final_clf
        result["scaler"] = scaler
        result["X_scaled"] = X_scaled
        result["y_encoded"] = y_encoded
        # Keep the original (pre-relabel) feature IDs so SHAP can map
        # gene labels back if the caller needs to look up which raw
        # column drove each importance.
        result["feature_ids"] = final_feature_names
    return result


def relabel_features_with_genes(
    feature_ids,
    annotation_df,
    id_col="protein_group",
    gene_col="leading_gene_name",
    fallback="id",
):
    """Map protein-level feature IDs to gene-name labels for plotting.

    Pipeline-internal identifiers (e.g. PRISM ``protein_group`` values like
    ``"PG0833"``) are uninterpretable to most readers; gene symbols are not.
    This helper produces a parallel list of human-readable labels suitable
    for any plot that takes feature names, by looking up each ID in an
    annotation table.

    Args:
        feature_ids: Iterable of feature identifiers (e.g. ``protein_group``
            values) in the order they appear as columns in the model's
            feature matrix.
        annotation_df: DataFrame containing at least ``id_col`` and ``gene_col``.
            Typically the protein-level data table.
        id_col: Column in ``annotation_df`` whose values match the
            entries in ``feature_ids``. Default ``"protein_group"``.
        gene_col: Column in ``annotation_df`` holding the human-readable
            gene symbol. Default ``"leading_gene_name"``.
        fallback: Behavior when ``gene_col`` is missing or empty for a
            given ID:

            - ``"id"`` *(default)*: use the original feature ID. Safer for
              traceability; always produces a non-empty label.
            - ``"empty"``: emit an empty string. Useful if you want to
              hide unannotated features in dense plots.

    Returns:
        list[str]: Labels in the same order as ``feature_ids``.

    Raises:
        ValueError: If ``fallback`` is not ``"id"`` or ``"empty"``.
    """
    if fallback not in ("id", "empty"):
        raise ValueError(f"fallback must be 'id' or 'empty'; got {fallback!r}")
    gene_map = annotation_df.set_index(id_col)[gene_col].fillna("").to_dict()
    if fallback == "id":
        return [gene_map.get(fid, fid) or fid for fid in feature_ids]
    return [gene_map.get(fid, "") for fid in feature_ids]


def compute_shap_values(
    model, X, feature_names=None, annotations=None, id_col="protein_group", gene_col="leading_gene_name"
):
    """Compute SHAP values for a fitted tree-based binary classifier.

    Thin wrapper around :class:`shap.TreeExplainer` for
    :class:`~sklearn.ensemble.RandomForestClassifier` and
    :class:`~xgboost.XGBClassifier`. For binary classifiers SHAP returns
    a 3-D ``(samples, features, classes)`` Explanation; this function
    collapses to the positive class (class 1 under
    :class:`~sklearn.preprocessing.LabelEncoder`'s alphabetical ordering)
    so the returned Explanation is 2-D and ready for ``shap.plots.beeswarm``
    or :func:`plot_shap_summary`.

    Use with ``run_binary_classification(..., return_model=True)``: pass
    ``result['final_model']`` and ``result['X_scaled']`` to keep features
    on the same scale the model was trained on.

    Args:
        model: A fitted tree-based binary classifier. Linear and SVM
            models are not supported by ``TreeExplainer``; use
            :class:`shap.LinearExplainer` directly for those.
        X: Feature matrix (numpy array or DataFrame), rows = samples.
            Must be on the same scale used during training (e.g., the
            ``X_scaled`` produced by ``run_binary_classification``).
        feature_names: Optional list of column names to attach to the
            returned Explanation. Inferred from ``X.columns`` when ``X``
            is a DataFrame.
        annotations: Optional DataFrame containing ``id_col`` and
            ``gene_col``. When provided alongside ``feature_names``, the
            feature labels attached to the Explanation are remapped from
            pipeline IDs to gene symbols via
            :func:`relabel_features_with_genes`. Strongly recommended for
            any SHAP plot intended for a non-bioinformatician audience.
        id_col: Identifier column in ``annotations`` matching
            ``feature_names``. Ignored if ``annotations`` is None.
        gene_col: Gene-name column in ``annotations``. Ignored if
            ``annotations`` is None.

    Returns:
        :class:`shap.Explanation` (2-D) for the positive class, with
        ``feature_names`` populated.

    Raises:
        ImportError: If ``shap`` is not installed. Install with
            ``pip install proteomics-toolkit[shap]`` or ``pip install shap``.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "shap is required for compute_shap_values. Install via "
            "'pip install proteomics-toolkit[shap]' or 'pip install shap'."
        ) from e

    if hasattr(X, "values"):
        X_arr = X.values
        if feature_names is None:
            feature_names = list(X.columns)
    else:
        X_arr = np.asarray(X)

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_arr)

    # Binary classifiers produce a 3-D (samples, features, 2) Explanation;
    # keep only the positive class so downstream plots work directly.
    if getattr(explanation, "values", None) is not None and explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]

    if feature_names is not None:
        if annotations is not None:
            feature_names = relabel_features_with_genes(
                feature_names,
                annotations,
                id_col=id_col,
                gene_col=gene_col,
            )
        explanation.feature_names = list(feature_names)
    return explanation


def plot_shap_summary(explanation, max_display=20, plot_type="beeswarm", title=None):
    """Plot a SHAP summary (beeswarm or bar) for one classifier.

    Args:
        explanation: :class:`shap.Explanation` from :func:`compute_shap_values`
            (2-D, positive class only).
        max_display: Number of top features to show.
        plot_type: ``'beeswarm'`` (per-sample colored points; default) or
            ``'bar'`` (mean |SHAP| bar plot).
        title: Optional title to set on the figure.

    Returns:
        :class:`matplotlib.figure.Figure` of the plot.

    Raises:
        ImportError: If ``shap`` is not installed.
        ValueError: If ``plot_type`` is not ``'beeswarm'`` or ``'bar'``.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "shap is required for plot_shap_summary. Install via "
            "'pip install proteomics-toolkit[shap]' or 'pip install shap'."
        ) from e

    import matplotlib.pyplot as plt

    if plot_type == "beeswarm":
        shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    elif plot_type == "bar":
        shap.plots.bar(explanation, max_display=max_display, show=False)
    else:
        raise ValueError(f"plot_type must be 'beeswarm' or 'bar'; got {plot_type!r}")

    fig = plt.gcf()
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_fold_change_pca(
    fold_change_matrix,
    group_labels,
    group_colors=None,
    title="PCA of Per-Subject Fold Changes",
    figsize=(10, 10),
    annotate_subjects=False,
):
    """PCA visualization of per-subject fold-changes colored by group.

    Projects the subject-level fold-change profiles into the first two
    principal components and colors points by group membership.

    Args:
        fold_change_matrix: DataFrame with subjects as rows and proteins
            as columns.
        group_labels: Series mapping subject IDs to group labels.
        group_colors: Optional dict mapping group labels to colors.
        title: Plot title.
        figsize: Figure size tuple.
        annotate_subjects: If True, label each point with its subject ID.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Align
    common = fold_change_matrix.index.intersection(group_labels.index)
    X = fold_change_matrix.loc[common].dropna(axis=1)
    y = group_labels.loc[common]

    # Scale and run PCA
    X_scaled = StandardScaler().fit_transform(X.values)
    pca = PCA(n_components=min(2, X_scaled.shape[1]))
    coords = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_ * 100

    # Default colors
    if group_colors is None:
        unique_groups = y.unique()
        default_palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        group_colors = {g: default_palette[i % len(default_palette)] for i, g in enumerate(sorted(unique_groups))}

    fig, ax = plt.subplots(figsize=figsize)

    for group in sorted(y.unique()):
        mask = y == group
        color = group_colors.get(group, "#999999")
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            label=f"{group} (n={mask.sum()})",
            s=100,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

    if annotate_subjects:
        for i, subj in enumerate(common):
            ax.annotate(str(subj), (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.7)

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)", fontsize=14)
    if len(var_explained) > 1:
        ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig


def plot_roc_curve(
    classification_results,
    title="ROC Curve",
    figsize=(8, 7),
    color="#1f77b4",
):
    """Plot mean ROC curve with +/- 1 SD band from cross-validation folds.

    For k-fold CV, interpolates per-fold ROC curves onto a common FPR grid
    and plots the mean TPR with a shaded +/- 1 standard deviation band.
    Falls back to a single aggregated ROC curve if per-fold data is not
    available (e.g., LOO-CV).

    Args:
        classification_results: Dict returned by run_binary_classification().
            Uses 'fold_roc_data' for per-fold curves when available.
        title: Plot title.
        figsize: Figure size tuple.
        color: Line and band color for the ROC curve.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    fold_roc_data = classification_results.get("fold_roc_data", [])
    n_features = classification_results["n_features"]
    classification_results["class_names"]

    fig, ax = plt.subplots(figsize=figsize)

    if len(fold_roc_data) >= 2:
        # Interpolate per-fold curves onto common FPR grid
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        for fpr_fold, tpr_fold, auc_fold in fold_roc_data:
            interp_tpr = np.interp(mean_fpr, fpr_fold, tpr_fold)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc_fold)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        # Mean ROC curve
        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            lw=2,
            label=f"AUC = {mean_auc:.2f} +/- {std_auc:.2f} ({n_features} features)",
        )

        # +/- 1 SD band
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)
        ax.fill_between(
            mean_fpr,
            tpr_lower,
            tpr_upper,
            alpha=0.2,
            color=color,
        )
    else:
        # Fallback: single aggregated ROC curve (e.g., LOO-CV)
        y_true = classification_results["y_true"]
        y_prob = classification_results["y_prob"]
        auc_val = classification_results.get("auc_roc", None)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr) if auc_val is None else auc_val

        ax.plot(
            fpr,
            tpr,
            color=color,
            lw=2,
            label=f"AUC = {roc_auc:.3f} ({n_features} features)",
        )

    # Diagonal chance line
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Chance")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig


# Default colors for classifier methods
METHOD_COLORS = {
    "logistic_regression": "#1f77b4",  # blue
    "random_forest": "#2ca02c",  # green
    "linear_svm": "#ff7f0e",  # orange
    "xgboost": "#9467bd",  # purple
}

METHOD_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "linear_svm": "Linear SVM",
    "xgboost": "XGBoost",
}


def plot_selection_frequency(results, top_n=30, title="RFECV Selection Frequency", figsize=(8, 9), color="#1f77b4"):
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
    ax.axvline(
        threshold,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"consensus >= {threshold:g}",
    )
    ax.set_xlabel("Selection frequency across outer CV folds")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_roc_comparison(
    results_dict,
    title="ROC Comparison",
    figsize=(10, 8),
    method_colors=None,
):
    """Overlay ROC curves from multiple classification methods on one plot.

    Each method gets a mean ROC curve with a +/- 1 SD shaded band, using
    per-fold data from cross-validation.

    Args:
        results_dict: Dict mapping method name (str) to classification results
            dict as returned by run_binary_classification().
        title: Plot title.
        figsize: Figure size tuple.
        method_colors: Optional dict mapping method name to color string.
            Defaults to MODULE_COLORS if method name matches, otherwise
            cycles through a default palette.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    if method_colors is None:
        method_colors = {}

    fig, ax = plt.subplots(figsize=figsize)
    mean_fpr = np.linspace(0, 1, 100)

    # Fallback palette for unknown method names
    fallback_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    color_idx = 0

    for method_name, cr in results_dict.items():
        # Determine color
        color = method_colors.get(method_name)
        if color is None:
            color = METHOD_COLORS.get(method_name)
        if color is None:
            color = fallback_colors[color_idx % len(fallback_colors)]
            color_idx += 1

        label = METHOD_LABELS.get(method_name, method_name)
        fold_roc_data = cr.get("fold_roc_data", [])

        if len(fold_roc_data) >= 2:
            tprs = []
            aucs_list = []
            for fpr_fold, tpr_fold, auc_fold in fold_roc_data:
                interp_tpr = np.interp(mean_fpr, fpr_fold, tpr_fold)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs_list.append(auc_fold)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            std_tpr = np.std(tprs, axis=0)
            mean_auc = np.mean(aucs_list)
            std_auc = np.std(aucs_list)

            ax.plot(
                mean_fpr,
                mean_tpr,
                color=color,
                lw=2,
                label=f"{label} (AUC = {mean_auc:.2f} +/- {std_auc:.2f})",
            )
            tpr_upper = np.minimum(mean_tpr + std_tpr, 1.0)
            tpr_lower = np.maximum(mean_tpr - std_tpr, 0.0)
            ax.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.15, color=color)
        else:
            # Fallback: single aggregated curve
            y_true = cr["y_true"]
            y_prob = cr["y_prob"]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = cr.get("auc_roc", auc(fpr, tpr))
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Chance (AUC = 0.50)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)
    return fig


def multiclass_feature_importance(
    data,
    sample_columns,
    sample_metadata,
    group_column,
    method="random_forest",
    n_repeats=30,
    n_estimators=500,
    bootstrap_iters=200,
    top_k_for_stability=50,
    log_transform=True,
    random_state=0,
    annotation_columns=None,
):
    """Multi-class feature importance with bootstrap stability scoring.

    Trains a multi-class classifier on protein abundances (samples as rows,
    proteins as columns), then computes sklearn permutation importance for
    each protein. Bootstrapping over samples (with replacement) gives a
    stability score per protein: the fraction of bootstrap resamples in
    which the protein lands in the top-K importance ranks.

    Designed for descriptive marker discovery, *not* hypothesis testing.
    Permutation importance with a tuned RF can reveal proteins that jointly
    discriminate groups even when no single per-protein test would survive
    multiple-testing correction. The bootstrap stability score is the
    appropriate uncertainty metric in this setting (see Meinshausen &
    Buhlmann 2010).

    Args:
        data: Wide protein DataFrame with annotation columns followed by
            sample columns.
        sample_columns: List of sample column names to include.
        sample_metadata: Dict-of-dicts keyed by sample column name; each
            value must contain ``group_column``.
        group_column: Metadata field naming each sample's class label.
        method: ``"random_forest"`` (default) or ``"xgboost"``.
        n_repeats: Permutation-importance repeat count for each fit.
        n_estimators: Trees per forest.
        bootstrap_iters: Number of bootstrap resamples for the stability
            score. Set to 0 to skip bootstrapping (returns NaN stability).
        top_k_for_stability: A protein contributes to the stability score
            on a given bootstrap if its importance rank that bootstrap is
            <= top_k_for_stability.
        log_transform: log2-transform abundance before fitting. Pass
            False if the input is already log-transformed.
        random_state: Seed for reproducibility.
        annotation_columns: Optional list of annotation columns to carry
            through. Defaults to the standard 5-column annotation set
            plus skyline-prism leading_* columns.

    Returns:
        DataFrame with one row per protein, sorted by ``importance_mean``
        descending. Columns: annotation columns, ``importance_mean``,
        ``importance_std``, ``stability``. The OOB score (RF only) and
        class labels are attached via ``df.attrs``.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    if method not in ("random_forest", "xgboost"):
        raise ValueError(f"method must be 'random_forest' or 'xgboost'; got {method!r}")

    labels = []
    keep_cols = []
    for col in sample_columns:
        meta = sample_metadata.get(col)
        if meta is None:
            continue
        val = meta.get(group_column)
        if val is None or (isinstance(val, float) and np.isnan(val)) or val == "":
            continue
        labels.append(str(val))
        keep_cols.append(col)
    if len(keep_cols) < 6:
        raise ValueError(f"Need >=6 samples with non-empty {group_column!r}; got {len(keep_cols)}.")
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError(f"Need >=2 distinct classes for {group_column!r}; got {len(classes)}.")

    default_annot = (
        "Protein",
        "Description",
        "Protein Gene",
        "UniProt_Accession",
        "UniProt_Entry_Name",
        "leading_gene_name",
        "leading_uniprot_id",
        "leading_protein",
        "leading_name",
        "leading_description",
        "protein_group",
    )
    sample_set = set(keep_cols)
    if annotation_columns is not None:
        annot_cols = [c for c in annotation_columns if c in data.columns and c not in sample_set]
    else:
        annot_cols = [c for c in default_annot if c in data.columns and c not in sample_set]

    abundance = data[keep_cols].apply(pd.to_numeric, errors="coerce")
    if log_transform:
        with np.errstate(invalid="ignore", divide="ignore"):
            abundance = np.log2(abundance.where(abundance > 0))
    abundance = abundance.dropna(axis=0, how="any")
    if abundance.empty:
        raise ValueError("No proteins remain after dropping rows with NaN values.")

    X = abundance.to_numpy(dtype=float).T  # (n_samples, n_proteins)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    rng = np.random.default_rng(random_state)

    def make_clf(seed):
        if method == "random_forest":
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=int(seed),
                n_jobs=-1,
                oob_score=True,
            )
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=n_estimators,
            max_depth=4,
            learning_rate=0.1,
            random_state=int(seed),
            eval_metric="mlogloss",
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = make_clf(random_state)
    clf.fit(X_scaled, y)
    oob = float(getattr(clf, "oob_score_", float("nan"))) if method == "random_forest" else float("nan")

    # Use n_jobs=1 for permutation_importance to avoid memory blowup: with -1
    # each worker copies the full (n_samples, n_features) matrix, which OOM-kills
    # at large feature counts on small machines.
    perm = permutation_importance(clf, X_scaled, y, n_repeats=n_repeats, random_state=int(random_state), n_jobs=1)
    importance_mean = perm.importances_mean
    importance_std = perm.importances_std

    n_features = X.shape[1]
    if bootstrap_iters > 0:
        in_top_count = np.zeros(n_features, dtype=int)
        n_samples = X.shape[0]
        completed = 0
        max_attempts = bootstrap_iters * 3
        for _ in range(max_attempts):
            if completed >= bootstrap_iters:
                break
            seed_b = int(rng.integers(0, 2**31 - 1))
            idx = rng.integers(0, n_samples, size=n_samples)
            if len(np.unique(y[idx])) < 2:
                continue
            scaler_b = StandardScaler()
            X_b = scaler_b.fit_transform(X[idx])
            clf_b = make_clf(seed_b)
            clf_b.fit(X_b, y[idx])
            perm_b = permutation_importance(
                clf_b,
                X_b,
                y[idx],
                n_repeats=max(5, n_repeats // 3),
                random_state=seed_b,
                n_jobs=1,
            )
            order_b = np.argsort(-perm_b.importances_mean, kind="stable")
            top_b = order_b[:top_k_for_stability]
            in_top_count[top_b] += 1
            completed += 1
        stability = in_top_count / float(max(completed, 1))
    else:
        stability = np.full(n_features, np.nan)

    out = pd.DataFrame(
        {
            "importance_mean": importance_mean,
            "importance_std": importance_std,
            "stability": stability,
        }
    )
    if annot_cols:
        annot_df = data.loc[abundance.index, annot_cols].reset_index(drop=True)
        out = pd.concat([annot_df, out], axis=1)
    else:
        out.insert(0, "Protein", [f"Protein_{i}" for i in abundance.index])

    out = out.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    out.attrs["oob_score"] = oob
    out.attrs["classes"] = list(le.classes_)
    out.attrs["n_samples"] = int(X.shape[0])
    out.attrs["n_features"] = int(n_features)

    logger.info(
        "multiclass_feature_importance: method=%s n_samples=%d n_features=%d classes=%s oob=%.3f",
        method,
        X.shape[0],
        n_features,
        list(le.classes_),
        oob,
    )
    return out
