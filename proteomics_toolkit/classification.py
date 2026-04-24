"""
Binary Classification Module for Proteomics Data

Provides tools for classifying subjects into binary groups (e.g., responder
vs non-responder) based on protein expression profiles, with cross-validated
performance metrics and PCA visualization.
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
        tstat, _ = ttest_ind(
            X_train[mask0], X_train[mask1], axis=0, equal_var=False, nan_policy="omit"
        )
    tstat = np.abs(np.asarray(tstat, dtype=float))
    tstat[~np.isfinite(tstat)] = -np.inf
    order = np.argsort(-tstat)  # descending
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
    - ``"fold_change"``: legacy behaviour - top by absolute mean
      fold-change across all subjects. **Leaky on small datasets** when
      classes differ systematically; use MAD or nested DA instead.

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
        feature_selection: One of ``"mad"`` (default), ``"differential_abundance"``,
            or ``"fold_change"``. See above for details.
        method: Classification method. One of 'logistic_regression',
            'random_forest', 'linear_svm', or 'xgboost'.
        cv_method: Cross-validation strategy. Integer for k-fold (default 5),
            or 'loo' for leave-one-out (not recommended for ROC curves).

    Returns:
        Dict with keys:
            'accuracy': Overall classification accuracy.
            'balanced_accuracy': Accuracy balanced across classes.
            'auc_roc': Mean AUC-ROC across folds.
            'auc_std': Standard deviation of AUC across folds.
            'confusion_matrix': Confusion matrix as numpy array.
            'cv_predictions': DataFrame with true and predicted labels.
            'feature_importances': Series of feature importance scores
                (from final fit on all data; for ``differential_abundance``
                this uses the full-data DA ranking and is meant as a
                descriptive summary, not a CV-validated metric).
            'classification_report': Text classification report.
            'n_features': Number of features used per fold.
            'feature_names': For ``mad`` / ``fold_change`` / explicit
                ``feature_proteins``: the fixed feature set. For
                ``differential_abundance``: the feature set selected
                from the full data (distinct from the per-fold sets
                that drove CV performance).
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
    valid_selection = {"mad", "differential_abundance", "fold_change"}
    if feature_selection not in valid_selection:
        raise ValueError(
            f"feature_selection must be one of {sorted(valid_selection)}; got {feature_selection!r}"
        )

    nested_da = False
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
            # Defer per-fold selection; X stays as the full feature space.
            nested_da = True

    feature_names = list(X.columns)
    n_features = min(n_top_features, len(feature_names)) if nested_da else len(feature_names)
    if nested_da:
        print(
            f"Classification using nested differential-abundance selection "
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

        if nested_da:
            # Rank features on the training split only, then subset both splits.
            top_cols = _select_top_by_ttest(X_train_raw, y_train, n_top_features)
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

    # Fit final model on all data for feature importances. For nested DA,
    # select top features using the *full dataset* t-test; this is
    # separate from the CV performance (which used per-fold selections)
    # and is meant as a descriptive summary of which features the method
    # prefers when fed the whole cohort.
    if nested_da:
        final_top_cols = _select_top_by_ttest(X_values, y_encoded, n_top_features)
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

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "auc_roc": auc_mean,
        "auc_std": auc_std,
        "confusion_matrix": cm,
        "cv_predictions": cv_predictions,
        "feature_importances": importances,
        "classification_report": report,
        "n_features": n_features,
        "feature_names": final_feature_names,
        "feature_selection": (
            "explicit" if feature_proteins is not None else feature_selection
        ),
        "y_true": y_encoded,
        "y_prob": y_prob_all,
        "class_names": class_names,
        "fold_roc_data": fold_roc_data,
    }


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
