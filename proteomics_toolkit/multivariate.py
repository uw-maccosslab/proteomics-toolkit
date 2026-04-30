"""
Multivariate Variance-Partitioning Tests
========================================

Non-parametric methods for asking "does this metadata factor explain a
significant fraction of total variance in the proteome?" without
assuming normality or balanced designs. Useful when per-protein tests
are underpowered (small n per group) but the global structure is still
informative.

The implementation here is a pure-scipy PERMANOVA (Anderson 2001) on
a sample-by-sample distance matrix derived from the protein abundance
table. It permutes group labels rather than relying on parametric
F-distribution tail areas, so it is valid at small sample sizes.

References:
    Anderson, M.J. (2001). A new method for non-parametric multivariate
    analysis of variance. Austral Ecology 26: 32-46.
    https://doi.org/10.1111/j.1442-9993.2001.01070.pp.x
"""

import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

SUPPORTED_METRICS = ("euclidean", "braycurtis", "cosine", "correlation", "cityblock")


def _build_distance_matrix(
    data: pd.DataFrame,
    sample_columns: list[str],
    metric: str,
    log_transform: bool,
) -> np.ndarray:
    """Build a sample-by-sample condensed distance matrix.

    Drops proteins with any NaN across the requested samples (PERMANOVA
    requires complete cases for the distance computation).
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {SUPPORTED_METRICS}; got {metric!r}")

    sub = data[sample_columns].apply(pd.to_numeric, errors="coerce")
    if log_transform:
        with np.errstate(invalid="ignore", divide="ignore"):
            sub = np.log2(sub.where(sub > 0))

    sub = sub.dropna(axis=0, how="any")
    if sub.empty:
        raise ValueError(
            "No proteins remain after dropping rows with NaN values. "
            "Filter or impute before calling permanova."
        )
    logger.info("permanova distance matrix uses %d proteins x %d samples.", sub.shape[0], sub.shape[1])
    # rows = proteins, cols = samples; pdist needs samples as rows.
    sample_matrix = sub.to_numpy(dtype=float).T
    return squareform(pdist(sample_matrix, metric=metric))


def _permanova_pseudo_f(distance_sq: np.ndarray, labels: np.ndarray) -> float:
    """Compute the PERMANOVA pseudo-F statistic given squared distances and integer labels.

    distance_sq: (n_samples, n_samples) of squared pairwise distances.
    labels: (n_samples,) integer-encoded group labels.

    Implements Anderson (2001) eqs. 4-6:
        SS_total = sum_{i<j} d_ij^2 / n
        SS_within = sum over groups of sum_{i<j in group} d_ij^2 / n_g
        SS_among  = SS_total - SS_within
        F = (SS_among / (a - 1)) / (SS_within / (n - a))
    """
    n = distance_sq.shape[0]
    groups = np.unique(labels)
    a = len(groups)

    iu = np.triu_indices(n, k=1)
    ss_total = distance_sq[iu].sum() / n

    ss_within = 0.0
    for g in groups:
        members = np.where(labels == g)[0]
        n_g = len(members)
        if n_g < 2:
            continue
        sub = distance_sq[np.ix_(members, members)]
        iu_g = np.triu_indices(n_g, k=1)
        ss_within += sub[iu_g].sum() / n_g

    ss_among = ss_total - ss_within
    if ss_within <= 0 or (n - a) <= 0 or (a - 1) <= 0:
        return float("nan")
    return float((ss_among / (a - 1)) / (ss_within / (n - a)))


def permanova(
    data: pd.DataFrame,
    sample_columns: list[str],
    sample_metadata: dict,
    factor: str,
    metric: str = "euclidean",
    log_transform: bool = True,
    n_permutations: int = 999,
    random_state: int | None = 0,
) -> dict:
    """Permutational MANOVA (Anderson 2001) on a metadata factor.

    Tests whether ``factor`` (a categorical metadata field) explains a
    significant fraction of the multivariate variance in the proteome,
    using label permutation. Returns the pseudo-F statistic, R^2
    (fraction of variance explained), and a permutation p-value.

    Args:
        data: Wide protein DataFrame; annotation columns plus sample
            columns. Proteins with any NaN across ``sample_columns`` are
            dropped before the distance computation.
        sample_columns: Sample column names to include.
        sample_metadata: Dict-of-dicts keyed by sample column name; each
            value must contain ``factor``.
        factor: Metadata field whose values define the groups.
        metric: scipy.spatial.distance metric. One of
            ``"euclidean"``, ``"braycurtis"``, ``"cosine"``,
            ``"correlation"``, ``"cityblock"``. Euclidean on log2 data
            is a sensible default for proteomics.
        log_transform: log2-transform abundance before distances. Pass
            False if the input is already log-transformed.
        n_permutations: Number of label permutations. The p-value is
            ``(1 + number of perm F >= observed F) / (1 + n_permutations)``,
            following the standard add-one convention so p > 0.
        random_state: Seed for the permutation RNG.

    Returns:
        Dict with keys:
            ``factor``: echo of the input factor name.
            ``F``: observed pseudo-F.
            ``R2``: SS_among / SS_total (fraction of variance explained).
            ``p_value``: permutation p-value.
            ``n_permutations``: number of permutations actually run.
            ``n_samples``: number of samples in the test.
            ``groups``: list of group labels used.
            ``group_sizes``: dict of label -> sample count.
            ``metric``: echo of the distance metric.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1.")

    # Resolve labels for each requested sample
    labels_raw = []
    keep_cols = []
    for col in sample_columns:
        meta = sample_metadata.get(col)
        if meta is None:
            continue
        val = meta.get(factor)
        if val is None or (isinstance(val, float) and np.isnan(val)) or val == "":
            continue
        labels_raw.append(str(val))
        keep_cols.append(col)
    if len(keep_cols) < 4:
        raise ValueError(
            f"Need >=4 samples with non-empty {factor!r}; got {len(keep_cols)}."
        )
    labels_arr, group_names = pd.factorize(pd.Series(labels_raw), sort=True)
    if len(group_names) < 2:
        raise ValueError(f"Need >=2 distinct groups for {factor!r}; got 1.")

    distance = _build_distance_matrix(data, keep_cols, metric=metric, log_transform=log_transform)
    distance_sq = distance**2

    f_obs = _permanova_pseudo_f(distance_sq, labels_arr)

    # SS_total / R^2 are derived from the same squared-distance bookkeeping
    n = len(labels_arr)
    iu = np.triu_indices(n, k=1)
    ss_total = distance_sq[iu].sum() / n
    ss_within = 0.0
    for g in np.unique(labels_arr):
        members = np.where(labels_arr == g)[0]
        n_g = len(members)
        if n_g < 2:
            continue
        sub = distance_sq[np.ix_(members, members)]
        iu_g = np.triu_indices(n_g, k=1)
        ss_within += sub[iu_g].sum() / n_g
    r2 = (ss_total - ss_within) / ss_total if ss_total > 0 else float("nan")

    # Permutation null
    rng = np.random.default_rng(random_state)
    permuted_labels = labels_arr.copy()
    n_at_or_above = 0
    for _ in range(n_permutations):
        rng.shuffle(permuted_labels)
        f_perm = _permanova_pseudo_f(distance_sq, permuted_labels)
        if np.isfinite(f_perm) and f_perm >= f_obs:
            n_at_or_above += 1

    p_value = (1 + n_at_or_above) / (1 + n_permutations)

    group_sizes = pd.Series(labels_raw).value_counts().to_dict()
    logger.info(
        "permanova[%s]: F=%.3f R2=%.3f p=%.4f (%d perms, %d samples)",
        factor,
        f_obs,
        r2,
        p_value,
        n_permutations,
        n,
    )
    return {
        "factor": factor,
        "F": f_obs,
        "R2": float(r2) if np.isfinite(r2) else float("nan"),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "n_samples": n,
        "groups": list(group_names),
        "group_sizes": group_sizes,
        "metric": metric,
    }
