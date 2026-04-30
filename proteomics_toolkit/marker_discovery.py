"""
Descriptive Marker-Discovery Metrics
====================================

Per-protein ranking and clustering helpers for marker discovery. The
metrics here are descriptive effect sizes (no p-values) and are
appropriate for small-n exploratory designs where classical hypothesis
testing is underpowered.

- ``method_specificity_score``: per (protein, group) pair, computes how
  much higher the group mean is than the next-best group, the ratio of
  the group mean to the across-group median, and the rank of the group
  among all groups for that protein. Long-form output suitable for
  sorting per group.
- ``inter_vs_intra_group_variance``: per protein, computes the ratio of
  variance across group means to the mean within-group variance. High
  ratio = group-discriminating. Useful as an unsupervised sort when n
  per group is too small for ANOVA but exceeds 1.
- ``cluster_proteins_kmeans``: k-means clustering of proteins over
  samples, with optional silhouette-driven selection of k. Returns a
  per-protein cluster assignment plus the silhouette curve so callers
  can document the chosen k.

All functions accept the standard ``proteomics_toolkit`` data layout:
a wide protein DataFrame with annotation columns followed by sample
columns, plus a dict-of-dicts ``sample_metadata`` keyed by sample
column name.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Annotation columns that callers may want carried through to the output
# table when present in the input. Sample columns are everything else.
DEFAULT_ANNOTATION_COLUMNS = (
    "Protein",
    "Description",
    "Protein Gene",
    "UniProt_Accession",
    "UniProt_Entry_Name",
    # Aliases used by skyline-prism / DIA-NN outputs:
    "leading_gene_name",
    "leading_uniprot_id",
    "leading_protein",
    "leading_name",
    "leading_description",
    "protein_group",
)


def _resolve_annotation_columns(
    data: pd.DataFrame,
    sample_columns: list[str],
    annotation_columns: list[str] | None,
) -> list[str]:
    """Pick annotation columns to carry through to the output table.

    If the caller passes ``annotation_columns`` explicitly, use that list
    intersected with the data columns. Otherwise default to the well-known
    annotation column names that exist in the data.
    """
    sample_set = set(sample_columns)
    if annotation_columns is not None:
        return [c for c in annotation_columns if c in data.columns and c not in sample_set]
    return [c for c in DEFAULT_ANNOTATION_COLUMNS if c in data.columns and c not in sample_set]


def _group_to_samples(
    sample_columns: list[str],
    sample_metadata: dict,
    group_column: str,
) -> dict[str, list[str]]:
    """Bucket sample column names by group label.

    Samples whose metadata is missing the group_column or whose value is
    None / NaN / empty are skipped with a warning.
    """
    buckets: dict[str, list[str]] = {}
    skipped = 0
    for sample in sample_columns:
        meta = sample_metadata.get(sample)
        if meta is None:
            skipped += 1
            continue
        label = meta.get(group_column)
        if label is None or (isinstance(label, float) and np.isnan(label)) or label == "":
            skipped += 1
            continue
        buckets.setdefault(str(label), []).append(sample)
    if skipped:
        logger.info("Skipped %d samples missing %r in metadata.", skipped, group_column)
    if not buckets:
        raise ValueError(
            f"No samples have a non-empty value for group_column={group_column!r} "
            "in sample_metadata."
        )
    return buckets


def method_specificity_score(
    data: pd.DataFrame,
    sample_columns: list[str],
    sample_metadata: dict,
    group_column: str,
    log_transform: bool = True,
    annotation_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-(protein, group) marker-discovery metrics.

    For each protein, computes the per-group mean abundance, then for each
    (protein, group) row reports:

    - ``group_mean``: mean of the protein in that group's samples (in log2
      space if ``log_transform=True``).
    - ``second_best_group_mean``: mean of the next-highest group.
    - ``delta_top``: ``group_mean - second_best_group_mean``. In log2 space
      this is a log2 fold-change vs the second-best group.
    - ``specificity``: ``group_mean - median_across_groups`` (log2) or
      ``group_mean / median_across_groups`` (linear). Captures how much
      this group stands out from the typical group.
    - ``rank``: 1 = highest mean for this protein, len(groups) = lowest.
    - ``n_samples``: number of samples in that group with a non-NaN value
      for this protein.

    Args:
        data: Wide protein DataFrame with annotation columns followed by
            sample columns.
        sample_columns: List of sample column names to consider.
        sample_metadata: Dict-of-dicts keyed by sample column name. Each
            value must contain ``group_column``.
        group_column: Name of the metadata field whose values define the
            groups (e.g., ``"EV_Method"``).
        log_transform: If True, log2-transform abundances before computing
            means. Pass False if the input is already log-transformed
            (e.g., output of VSN / RLR / LOESS normalization).
        annotation_columns: Optional list of annotation columns to carry
            through to the output. If None, the standard annotation
            columns are detected automatically.

    Returns:
        Long-form DataFrame with one row per (protein, group). Columns:
        annotation columns, ``Group``, ``group_mean``,
        ``second_best_group_mean``, ``delta_top``, ``specificity``,
        ``rank``, ``n_samples``.

        Sorted by (Group ascending, delta_top descending) so that
        ``df.groupby("Group").head(25)`` returns the top markers per
        group.
    """
    if not sample_columns:
        raise ValueError("sample_columns must be non-empty.")
    missing = [c for c in sample_columns if c not in data.columns]
    if missing:
        raise ValueError(f"sample_columns not found in data: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    annot_cols = _resolve_annotation_columns(data, sample_columns, annotation_columns)
    buckets = _group_to_samples(sample_columns, sample_metadata, group_column)
    groups = sorted(buckets.keys())

    abundance = data[sample_columns].apply(pd.to_numeric, errors="coerce")
    if log_transform:
        # log2(x) with non-positive values -> NaN
        with np.errstate(invalid="ignore", divide="ignore"):
            abundance = np.log2(abundance.where(abundance > 0))

    # Per-group mean / count matrices: shape (n_proteins, n_groups)
    group_means = pd.DataFrame(index=data.index, columns=groups, dtype=float)
    group_counts = pd.DataFrame(index=data.index, columns=groups, dtype=int)
    for g in groups:
        cols = buckets[g]
        sub = abundance[cols]
        group_means[g] = sub.mean(axis=1, skipna=True)
        group_counts[g] = sub.notna().sum(axis=1)

    # For each protein: across-group median, sorted-descending mean values to
    # derive delta_top (mean - second_best) and rank.
    means_array = group_means.to_numpy(dtype=float)
    sorted_desc = np.sort(np.where(np.isnan(means_array), -np.inf, means_array), axis=1)[:, ::-1]
    second_best = sorted_desc[:, 1] if sorted_desc.shape[1] >= 2 else np.full(sorted_desc.shape[0], np.nan)
    second_best = np.where(np.isfinite(second_best), second_best, np.nan)
    median_across = np.nanmedian(means_array, axis=1)

    # Rank descending: highest mean = 1.
    ranks = np.argsort(np.argsort(-means_array, axis=1, kind="stable"), axis=1, kind="stable") + 1

    # Build long-form output
    rows = []
    for j, g in enumerate(groups):
        block = pd.DataFrame(
            {
                "Group": g,
                "group_mean": group_means[g].values,
                "second_best_group_mean": second_best,
                "delta_top": group_means[g].values - second_best,
                "specificity": (
                    group_means[g].values - median_across
                    if log_transform
                    else group_means[g].values / median_across
                ),
                "rank": ranks[:, j],
                "n_samples": group_counts[g].values,
            },
            index=data.index,
        )
        # Mark non-best groups: delta_top and specificity computed against the
        # protein's own best group, so for non-rank-1 rows delta_top is
        # negative or zero. We keep all rows; users typically filter rank == 1.
        rows.append(pd.concat([data[annot_cols], block], axis=1) if annot_cols else block.reset_index(drop=True))

    long = pd.concat(rows, axis=0, ignore_index=True)

    # When a group is not the top, replace delta_top with (group_mean -
    # top_group_mean) so the values are interpretable as "log2 distance from
    # the best group". This matches conventional one-vs-best framing.
    top_means = sorted_desc[:, 0]
    is_top = (long["rank"] == 1).to_numpy()
    n_proteins = means_array.shape[0]
    n_groups = len(groups)
    top_means_repeated = np.tile(top_means, n_groups)
    long_means = long["group_mean"].to_numpy()
    long.loc[~is_top, "delta_top"] = (long_means[~is_top] - top_means_repeated[~is_top])

    # Sort: best (rank 1) per group first, then descending delta_top.
    long = long.sort_values(["Group", "rank", "delta_top"], ascending=[True, True, False]).reset_index(drop=True)

    logger.info(
        "method_specificity_score: %d proteins x %d groups -> %d rows.",
        n_proteins,
        n_groups,
        len(long),
    )
    return long


def inter_vs_intra_group_variance(
    data: pd.DataFrame,
    sample_columns: list[str],
    sample_metadata: dict,
    group_column: str,
    log_transform: bool = True,
    annotation_columns: list[str] | None = None,
    min_per_group: int = 2,
) -> pd.DataFrame:
    """Per-protein ratio of inter-group to intra-group variance.

    The ratio is ``var(group_means) / mean(within_group_var)``. High values
    indicate proteins whose group means are more spread apart than the
    typical within-group spread, i.e. group-discriminating proteins. This
    is descriptive (no p-value) and is useful when n per group is too low
    for ANOVA / Kruskal-Wallis to be reliable.

    Groups with fewer than ``min_per_group`` non-NaN observations for a
    given protein are excluded from that protein's calculation. If fewer
    than two groups remain the ratio is NaN.

    Args:
        data: Wide protein DataFrame with annotation columns followed by
            sample columns.
        sample_columns: Sample column names to consider.
        sample_metadata: Dict-of-dicts keyed by sample column name.
        group_column: Metadata field naming the group of each sample.
        log_transform: If True, log2-transform before computing variances.
        annotation_columns: Optional explicit list of annotation columns
            to carry through; defaults to standard annotation columns
            present in ``data``.
        min_per_group: Minimum non-NaN samples a group must have to count
            toward the ratio for a given protein.

    Returns:
        DataFrame with one row per protein, sorted by ``ratio`` descending.
        Columns: annotation columns, ``inter_var`` (variance of group
        means), ``intra_var`` (mean within-group variance), ``ratio``,
        ``n_groups_used`` (groups passing min_per_group), ``n_samples``
        (total samples used).
    """
    if not sample_columns:
        raise ValueError("sample_columns must be non-empty.")
    if min_per_group < 2:
        raise ValueError("min_per_group must be >= 2 (variance is undefined for n=1).")

    annot_cols = _resolve_annotation_columns(data, sample_columns, annotation_columns)
    buckets = _group_to_samples(sample_columns, sample_metadata, group_column)

    abundance = data[sample_columns].apply(pd.to_numeric, errors="coerce")
    if log_transform:
        with np.errstate(invalid="ignore", divide="ignore"):
            abundance = np.log2(abundance.where(abundance > 0))

    n_proteins = abundance.shape[0]
    group_means = np.full((n_proteins, len(buckets)), np.nan)
    group_vars = np.full((n_proteins, len(buckets)), np.nan)
    group_counts = np.zeros((n_proteins, len(buckets)), dtype=int)

    for j, (_g, cols) in enumerate(buckets.items()):
        sub = abundance[cols].to_numpy(dtype=float)
        counts = np.sum(~np.isnan(sub), axis=1)
        with np.errstate(invalid="ignore"):
            mean_g = np.nanmean(sub, axis=1)
            # ddof=1 sample variance; np.nanvar does not support ddof in older
            # numpy, so compute manually. var(x) = mean((x - mean)^2) * n/(n-1).
            centered = sub - mean_g[:, None]
            ssq = np.nansum(centered**2, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                var_g = np.where(counts > 1, ssq / (counts - 1), np.nan)
        # Mask groups below min_per_group as NaN
        valid = counts >= min_per_group
        group_means[:, j] = np.where(valid, mean_g, np.nan)
        group_vars[:, j] = np.where(valid, var_g, np.nan)
        group_counts[:, j] = counts

    n_groups_used = np.sum(~np.isnan(group_means), axis=1)
    with np.errstate(invalid="ignore"):
        # Variance of the group means across groups (sample variance).
        means_centered = group_means - np.nanmean(group_means, axis=1, keepdims=True)
        inter_var = np.where(
            n_groups_used >= 2,
            np.nansum(means_centered**2, axis=1) / np.maximum(n_groups_used - 1, 1),
            np.nan,
        )
        intra_var = np.nanmean(group_vars, axis=1)
        ratio = np.where((intra_var > 0) & np.isfinite(intra_var), inter_var / intra_var, np.nan)

    out = pd.DataFrame(
        {
            "inter_var": inter_var,
            "intra_var": intra_var,
            "ratio": ratio,
            "n_groups_used": n_groups_used,
            "n_samples": np.sum(group_counts, axis=1),
        },
        index=data.index,
    )
    if annot_cols:
        out = pd.concat([data[annot_cols].reset_index(drop=True), out.reset_index(drop=True)], axis=1)
    else:
        out = out.reset_index(drop=True)

    out = out.sort_values("ratio", ascending=False, na_position="last").reset_index(drop=True)
    logger.info(
        "inter_vs_intra_group_variance: %d proteins, top ratio = %.3f.",
        n_proteins,
        float(out["ratio"].iloc[0]) if len(out) and np.isfinite(out["ratio"].iloc[0]) else float("nan"),
    )
    return out


def cluster_proteins_kmeans(
    data: pd.DataFrame,
    sample_columns: list[str],
    k: int | None = None,
    k_range: tuple[int, int] = (2, 10),
    log_transform: bool = True,
    standardize: bool = True,
    annotation_columns: list[str] | None = None,
    random_state: int = 0,
    n_init: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-means clustering of proteins over samples, with silhouette-driven k selection.

    Each protein is treated as a point in sample space (vector of length
    ``len(sample_columns)``), so clustering groups proteins that follow
    similar abundance patterns across samples. Useful for identifying
    sets of proteins that vary together (e.g., proteins enriched by a
    particular EV method).

    If ``k`` is None, scans ``k_range`` and selects the value with the
    highest mean silhouette score. The full silhouette curve is returned
    so callers can record the choice.

    Args:
        data: Wide protein DataFrame with annotation columns followed
            by sample columns. Proteins with any NaN across the requested
            samples are dropped before clustering.
        sample_columns: Sample column names to use as features.
        k: Number of clusters. If None, selected via silhouette.
        k_range: Inclusive (k_min, k_max) range to scan when ``k`` is
            None. Each value of k must satisfy 2 <= k < n_proteins.
        log_transform: log2-transform abundances before clustering.
            Pass False if the input is already log-transformed.
        standardize: Per-protein z-score (subtract the protein's own
            mean, divide by its own std) before clustering. This is
            standard for "shape" clustering, where proteins are grouped
            by the *pattern* of variation across samples rather than by
            absolute abundance. Set False to cluster by raw abundance
            shape.
        annotation_columns: Optional annotation columns to carry through
            to the per-protein output. Defaults to standard annotation
            columns present in the data.
        random_state: Seed for KMeans initialization.
        n_init: Number of KMeans random initializations.

    Returns:
        Tuple ``(assignments, silhouette_scan)``:

        - ``assignments``: DataFrame with one row per non-NaN protein,
          containing the requested annotation columns plus a ``cluster``
          column (integer cluster id, 0-indexed) and a ``silhouette``
          column (per-protein silhouette score for the chosen k).
        - ``silhouette_scan``: DataFrame with columns ``k`` and
          ``mean_silhouette``. If ``k`` was passed in explicitly, this
          contains a single row.
    """
    if not sample_columns:
        raise ValueError("sample_columns must be non-empty.")
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

    abundance = data[sample_columns].apply(pd.to_numeric, errors="coerce")
    if log_transform:
        with np.errstate(invalid="ignore", divide="ignore"):
            abundance = np.log2(abundance.where(abundance > 0))
    abundance = abundance.dropna(axis=0, how="any")
    if abundance.empty:
        raise ValueError(
            "No proteins remain after dropping rows with NaN values. "
            "Filter or impute before calling cluster_proteins_kmeans."
        )

    matrix = abundance.to_numpy(dtype=float)  # rows = proteins, cols = samples
    if standardize:
        # Per-protein z-score: cluster by shape, not by absolute level
        means = matrix.mean(axis=1, keepdims=True)
        stds = matrix.std(axis=1, ddof=0, keepdims=True)
        stds = np.where(stds > 0, stds, 1.0)
        matrix = (matrix - means) / stds

    n_proteins = matrix.shape[0]
    annot_cols = _resolve_annotation_columns(data, sample_columns, annotation_columns)

    if k is None:
        k_min, k_max = k_range
        if k_min < 2:
            raise ValueError("k_range[0] must be >= 2.")
        if k_max >= n_proteins:
            k_max = max(k_min, n_proteins - 1)
        scan_rows = []
        best_k = None
        best_score = -np.inf
        for k_candidate in range(k_min, k_max + 1):
            km = KMeans(n_clusters=k_candidate, random_state=random_state, n_init=n_init)
            labels_k = km.fit_predict(matrix)
            score = silhouette_score(matrix, labels_k)
            scan_rows.append({"k": k_candidate, "mean_silhouette": float(score)})
            if score > best_score:
                best_score = score
                best_k = k_candidate
        silhouette_scan = pd.DataFrame(scan_rows)
        chosen_k = best_k
        logger.info("cluster_proteins_kmeans: silhouette-selected k=%d (score=%.3f)", chosen_k, best_score)
    else:
        if k < 2 or k >= n_proteins:
            raise ValueError(f"k must satisfy 2 <= k < n_proteins ({n_proteins}); got {k}.")
        chosen_k = k
        silhouette_scan = pd.DataFrame(columns=["k", "mean_silhouette"])

    km = KMeans(n_clusters=chosen_k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(matrix)
    per_point_silhouette = silhouette_samples(matrix, labels)
    if k is None:
        silhouette_scan = silhouette_scan.assign(
            chosen=lambda d: d["k"] == chosen_k,
        )
    else:
        silhouette_scan = pd.DataFrame(
            [{"k": chosen_k, "mean_silhouette": float(per_point_silhouette.mean()), "chosen": True}]
        )

    assignments = pd.DataFrame(
        {
            "cluster": labels.astype(int),
            "silhouette": per_point_silhouette.astype(float),
        },
        index=abundance.index,
    )
    if annot_cols:
        annot_df = data.loc[abundance.index, annot_cols].reset_index(drop=True)
        assignments = pd.concat([annot_df, assignments.reset_index(drop=True)], axis=1)
    else:
        assignments = assignments.reset_index(drop=True)

    return assignments, silhouette_scan

