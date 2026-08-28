"""
Continuous-Outcome Regression Module for Proteomics Data

Provides nested cross-validated ElasticNet regression for predicting a
continuous outcome (e.g., a clinical score, a percent-change measurement)
from a samples x features abundance matrix. Mirrors the honest nested-CV
design of :func:`proteomics_toolkit.classification.run_rfecv_stability`:
an outer loop gives an unbiased held-out performance estimate while an
inner loop tunes regularization strength, and a label(-value)-permutation
null attaches an empirical p-value to the observed performance.
"""

import logging

import numpy as np
import pandas as pd

from .classification import relabel_features_with_genes

logger = logging.getLogger(__name__)


def _run_elasticnet_outer_cv(
    X,
    y,
    outer_cv,
    inner_cv,
    l1_ratio,
    alphas,
    random_state,
    n_jobs,
    collect,
):
    """Run the nested outer-CV ElasticNetCV procedure once for target ``y``.

    All preprocessing and hyperparameter tuning happen on training folds
    only. When ``collect`` is False (permutation runs), per-feature
    nonzero-coefficient counts and pooled predictions are skipped for speed
    and only per-fold scores are returned.

    Returns:
        dict with ``fold_r2`` (list[float]); when ``collect`` is True also
        ``nonzero_counts`` (np.ndarray over all features), ``n_features_per_fold``
        (list[int]), ``chosen_l1_ratio_per_fold`` (list[float]), and
        ``predictions`` (list of (sample_index, y_true, y_pred, fold_index)).
    """
    from sklearn.linear_model import ElasticNetCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, RepeatedKFold
    from sklearn.preprocessing import StandardScaler

    n_splits, n_repeats = outer_cv
    outer = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    inner = KFold(n_splits=inner_cv, shuffle=True, random_state=random_state)

    n_features_total = X.shape[1]
    nonzero_counts = np.zeros(n_features_total) if collect else None
    n_feat_per_fold = [] if collect else None
    chosen_l1_ratio_per_fold = [] if collect else None
    predictions = [] if collect else None
    fold_r2 = []

    for fold_i, (tr, te) in enumerate(outer.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = ElasticNetCV(
            l1_ratio=l1_ratio,
            alphas=alphas,
            cv=inner,
            random_state=random_state,
            n_jobs=n_jobs,
            max_iter=10000,
        )
        model.fit(X_tr_s, y_tr)

        y_pred = model.predict(X_te_s)
        fold_r2.append(r2_score(y_te, y_pred) if len(te) >= 2 else np.nan)

        if collect:
            nonzero = np.abs(model.coef_) > 0
            nonzero_counts[nonzero] += 1
            n_feat_per_fold.append(int(nonzero.sum()))
            chosen_l1_ratio_per_fold.append(float(model.l1_ratio_))
            for j, idx in enumerate(te):
                predictions.append((int(idx), float(y_te[j]), float(y_pred[j]), fold_i))

    out = {"fold_r2": fold_r2}
    if collect:
        out.update(
            nonzero_counts=nonzero_counts,
            n_features_per_fold=n_feat_per_fold,
            chosen_l1_ratio_per_fold=chosen_l1_ratio_per_fold,
            predictions=predictions,
        )
    return out


def run_elasticnet_regression(
    data,
    target,
    outer_cv=(5, 10),
    inner_cv=5,
    l1_ratio=(0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0),
    alphas=100,
    log_transform="auto",
    n_permutations=100,
    consensus_threshold=0.5,
    annotations=None,
    id_col="protein_group",
    gene_col="leading_gene_name",
    random_state=42,
    n_jobs=-1,
):
    """Nested cross-validated ElasticNet regression for a continuous outcome.

    Wraps :class:`sklearn.linear_model.ElasticNetCV` inside an outer
    ``RepeatedKFold`` so that regularization-strength and l1_ratio tuning
    never see the held-out fold they are scored on. This gives an honest
    performance estimate in the n-much-less-than-p proteomics regime, plus
    a per-feature *selection frequency* (how often each feature keeps a
    nonzero coefficient across folds) and a target-permutation null.

    Args:
        data: DataFrame, samples (rows) x features (columns), numeric.
        target: Series indexed by sample id with a continuous outcome.
        outer_cv: ``(n_splits, n_repeats)`` for the outer RepeatedKFold.
        inner_cv: Number of folds for ElasticNetCV's internal alpha/l1_ratio search.
        l1_ratio: l1_ratio value(s) passed to ElasticNetCV. A sequence lets
            ElasticNetCV search over the L1/L2 mix as well as alpha.
        alphas: Number of alphas along the regularization path searched by
            ElasticNetCV per l1_ratio (or an explicit array of alpha values).
        log_transform: ``True``/``False`` or ``"auto"`` (log2 when the matrix
            looks raw-scale, i.e. max value > 100).
        n_permutations: Target-shuffle iterations for the null; ``0`` disables.
        consensus_threshold: Selection-frequency cutoff for ``consensus_features``.
        annotations: Optional DataFrame for relabeling feature ids to gene names.
        id_col: Column in ``annotations`` matching feature ids.
        gene_col: Column in ``annotations`` holding gene symbols.
        random_state: Seed for reproducibility.
        n_jobs: Parallelism for ElasticNetCV.

    Returns:
        dict with honest performance (``outer_r2_mean``/``std``,
        ``outer_rmse_mean``/``std``, ``outer_spearman_mean``/``std``),
        ``selection_frequency`` (pd.Series), ``consensus_features``,
        ``permutation_p_value``, ``cv_predictions``, ``coefficients``
        (all-data refit), and a ``config`` echo.

    Raises:
        ValueError: If fewer than 10 shared samples.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import ElasticNetCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    common = data.index.intersection(target.index)
    if len(common) < 10:
        raise ValueError(f"Need at least 10 shared samples; got {len(common)}.")

    X_df = data.loc[common].dropna(axis=1, how="any")
    y_raw = target.loc[common].astype(float)

    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=float)
    y = y_raw.to_numpy(dtype=float)

    do_log = log_transform is True or (log_transform == "auto" and np.nanmax(X) > 100)
    if do_log:
        X = np.log2(np.clip(X, 1.0, None))

    obs = _run_elasticnet_outer_cv(
        X,
        y,
        outer_cv,
        inner_cv,
        l1_ratio,
        alphas,
        random_state,
        n_jobs,
        collect=True,
    )

    total_folds = outer_cv[0] * outer_cv[1]
    selection_frequency = pd.Series(
        obs["nonzero_counts"] / total_folds, index=feature_names
    ).sort_values(ascending=False)
    consensus_features = list(selection_frequency[selection_frequency >= consensus_threshold].index)

    fold_r2 = np.asarray(obs["fold_r2"], dtype=float)
    outer_r2_mean = float(np.nanmean(fold_r2))
    outer_r2_std = float(np.nanstd(fold_r2))

    preds = obs["predictions"]
    y_true_pooled = np.array([p[1] for p in preds])
    y_pred_pooled = np.array([p[2] for p in preds])
    outer_rmse_mean = float(np.sqrt(np.mean((y_true_pooled - y_pred_pooled) ** 2))) if preds else float("nan")
    if len(preds) >= 2 and np.std(y_true_pooled) > 0 and np.std(y_pred_pooled) > 0:
        outer_spearman_mean = float(spearmanr(y_true_pooled, y_pred_pooled).statistic)
    else:
        outer_spearman_mean = float("nan")
    pooled_r2 = float(r2_score(y_true_pooled, y_pred_pooled)) if preds else float("nan")

    # Permutation null: rerun the whole outer-CV estimate on shuffled targets.
    null_r2 = []
    if n_permutations > 0:
        rng = np.random.RandomState(random_state)
        for _ in range(n_permutations):
            y_perm = rng.permutation(y)
            perm = _run_elasticnet_outer_cv(
                X,
                y_perm,
                outer_cv,
                inner_cv,
                l1_ratio,
                alphas,
                random_state,
                n_jobs,
                collect=False,
            )
            null_r2.append(float(np.nanmean(perm["fold_r2"])))
    null_r2 = np.asarray(null_r2, dtype=float)
    permutation_p_value = (
        float((1 + np.sum(null_r2 >= outer_r2_mean)) / (1 + n_permutations)) if n_permutations > 0 else None
    )

    cv_predictions = pd.DataFrame(
        {
            "Sample": [common[p[0]] for p in preds],
            "True_Value": y_true_pooled,
            "Predicted_Value": y_pred_pooled,
            "Fold": [p[3] for p in preds],
        }
    )

    # Final refit on all data for a descriptive coefficient table.
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    final_model = ElasticNetCV(
        l1_ratio=l1_ratio,
        alphas=alphas,
        cv=KFold(n_splits=inner_cv, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=n_jobs,
        max_iter=10000,
    )
    final_model.fit(X_scaled, y)
    coefficients = pd.Series(final_model.coef_, index=feature_names).sort_values(
        key=lambda s: s.abs(), ascending=False
    )

    if annotations is not None:
        gene_labels = relabel_features_with_genes(
            selection_frequency.index, annotations, id_col=id_col, gene_col=gene_col
        )
        selection_frequency.index = gene_labels
        consensus_features = relabel_features_with_genes(
            consensus_features, annotations, id_col=id_col, gene_col=gene_col
        )
        coefficients.index = relabel_features_with_genes(
            coefficients.index, annotations, id_col=id_col, gene_col=gene_col
        )

    n_features = int(np.median(obs["n_features_per_fold"])) if obs["n_features_per_fold"] else 0

    logger.info(
        "ElasticNet regression: outer R2=%.3f +/- %.3f, median nonzero features/fold=%d, "
        "consensus(>=%.2f)=%d, perm p=%s",
        outer_r2_mean,
        outer_r2_std,
        n_features,
        consensus_threshold,
        len(consensus_features),
        permutation_p_value,
    )

    return {
        "outer_r2_mean": outer_r2_mean,
        "outer_r2_std": outer_r2_std,
        "outer_rmse_mean": outer_rmse_mean,
        "outer_spearman_mean": outer_spearman_mean,
        "pooled_r2": pooled_r2,
        "per_fold_scores": fold_r2.tolist(),
        "selection_frequency": selection_frequency,
        "consensus_features": consensus_features,
        "n_features_per_fold": obs["n_features_per_fold"],
        "chosen_l1_ratio_per_fold": obs["chosen_l1_ratio_per_fold"],
        "permutation_r2_null": null_r2,
        "permutation_p_value": permutation_p_value,
        "cv_predictions": cv_predictions,
        "coefficients": coefficients,
        "n_features": n_features,
        "final_model": final_model,
        "config": {
            "outer_cv": outer_cv,
            "inner_cv": inner_cv,
            "l1_ratio": l1_ratio,
            "alphas": alphas,
            "consensus_threshold": consensus_threshold,
            "log_transform_applied": bool(do_log),
            "n_permutations": n_permutations,
        },
    }


def plot_regression_scatter(
    results,
    title="ElasticNet: Predicted vs. True",
    figsize=(7, 7),
    color="#1f77b4",
):
    """Scatter of out-of-fold predictions vs. true values from ``run_elasticnet_regression``.

    Args:
        results: The dict returned by :func:`run_elasticnet_regression`.
        title: Plot title.
        figsize: Figure size.
        color: Marker color.

    Returns:
        matplotlib Figure with a 1:1 reference line and R2/Spearman/permutation
        p annotated in the title.
    """
    import matplotlib.pyplot as plt

    preds = results["cv_predictions"]
    y_true = preds["True_Value"].to_numpy()
    y_pred = preds["Predicted_Value"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, color=color, alpha=0.7, edgecolors="black", linewidths=0.5)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="1:1")

    perm_p = results.get("permutation_p_value")
    perm_txt = "n/a" if perm_p is None else f"{perm_p:.3f}"
    subtitle = (
        f"R2={results['outer_r2_mean']:.3f} +/- {results['outer_r2_std']:.3f}, "
        f"rho={results['outer_spearman_mean']:.3f}, perm p={perm_txt}"
    )
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value (out-of-fold)")
    ax.set_title(f"{title}\n{subtitle}")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
