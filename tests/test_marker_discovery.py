"""Tests for proteomics_toolkit.marker_discovery."""

import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.marker_discovery import (
    cluster_proteins_kmeans,
    inter_vs_intra_group_variance,
    method_specificity_score,
)


def _make_planted_data(n_per_group=3, n_proteins=20, seed=0):
    """Three-group dataset with proteins planted as markers in each group."""
    rng = np.random.default_rng(seed)
    groups = ["A", "B", "C"]
    sample_names = []
    sample_metadata = {}
    for g in groups:
        for i in range(n_per_group):
            name = f"{g}_{i + 1}"
            sample_names.append(name)
            sample_metadata[name] = {"Method": g}

    # Background log2 abundance ~ N(10, 0.3^2). Plant 3 markers per group at +3.
    bg = rng.normal(10.0, 0.3, size=(n_proteins, len(sample_names)))
    columns = sample_names
    proteins = [f"P{i:02d}" for i in range(n_proteins)]

    # Plant markers: P00..P02 in A, P03..P05 in B, P06..P08 in C.
    def col_indices_for(group_letter):
        return [i for i, n in enumerate(columns) if n.startswith(group_letter + "_")]

    for j in col_indices_for("A"):
        bg[0:3, j] += 3.0
    for j in col_indices_for("B"):
        bg[3:6, j] += 3.0
    for j in col_indices_for("C"):
        bg[6:9, j] += 3.0

    # Convert log2 -> linear (the function log2-transforms internally by default)
    linear = np.power(2.0, bg)
    df = pd.DataFrame(linear, columns=columns)
    df.insert(0, "Protein", proteins)
    df.insert(1, "Description", [f"Description {p}" for p in proteins])
    df.insert(2, "leading_gene_name", [f"Gene_{p}" for p in proteins])
    return df, columns, sample_metadata


def test_method_specificity_score_returns_one_row_per_protein_group():
    df, cols, meta = _make_planted_data()
    out = method_specificity_score(df, cols, meta, group_column="Method")
    assert len(out) == 20 * 3  # 20 proteins x 3 groups
    assert {"Group", "group_mean", "delta_top", "specificity", "rank", "n_samples"} <= set(out.columns)
    # Sanity: for the rank-1 group, delta_top should be positive on planted markers.
    top = out[out["rank"] == 1]
    assert len(top) == 20


def test_method_specificity_score_identifies_planted_markers():
    df, cols, meta = _make_planted_data()
    out = method_specificity_score(df, cols, meta, group_column="Method")
    top_per_group = out[out["rank"] == 1]
    # Top three by delta_top in group A should be the planted A markers (P00, P01, P02).
    a_top3 = top_per_group[top_per_group["Group"] == "A"].nlargest(3, "delta_top")["Protein"].tolist()
    assert set(a_top3) == {"P00", "P01", "P02"}
    b_top3 = top_per_group[top_per_group["Group"] == "B"].nlargest(3, "delta_top")["Protein"].tolist()
    assert set(b_top3) == {"P03", "P04", "P05"}
    c_top3 = top_per_group[top_per_group["Group"] == "C"].nlargest(3, "delta_top")["Protein"].tolist()
    assert set(c_top3) == {"P06", "P07", "P08"}


def test_method_specificity_score_log2_units_are_consistent():
    df, cols, meta = _make_planted_data()
    out = method_specificity_score(df, cols, meta, group_column="Method", log_transform=True)
    a_top = out[(out["Group"] == "A") & (out["rank"] == 1) & (out["Protein"] == "P00")]
    assert not a_top.empty
    # delta_top is group_mean - second-best group_mean. We planted +3 log2.
    assert a_top["delta_top"].iloc[0] == pytest.approx(3.0, abs=0.5)


def test_method_specificity_score_handles_missing_metadata():
    df, cols, meta = _make_planted_data()
    # Drop metadata for one sample; function should warn-and-skip, not error.
    meta_partial = {k: v for k, v in meta.items() if k != cols[0]}
    out = method_specificity_score(df, cols, meta_partial, group_column="Method")
    assert len(out) == 20 * 3


def test_inter_vs_intra_group_variance_ranks_planted_markers_first():
    df, cols, meta = _make_planted_data(n_per_group=4)
    out = inter_vs_intra_group_variance(df, cols, meta, group_column="Method")
    # Top 9 by ratio should include the 9 planted markers.
    top9 = set(out.head(9)["Protein"].tolist())
    planted = {f"P{i:02d}" for i in range(9)}
    assert top9 == planted


def test_inter_vs_intra_group_variance_rejects_min_per_group_lt_2():
    df, cols, meta = _make_planted_data()
    with pytest.raises(ValueError, match="min_per_group"):
        inter_vs_intra_group_variance(df, cols, meta, group_column="Method", min_per_group=1)


def test_method_specificity_score_rejects_unknown_sample_columns():
    df, cols, meta = _make_planted_data()
    with pytest.raises(ValueError, match="not found"):
        method_specificity_score(df, [*cols, "missing_col"], meta, group_column="Method")


def _make_pattern_clustered_data(n_per_pattern=4, n_samples=12, seed=0):
    """Build a dataset where proteins fall into three clear shape clusters.

    Each cluster of proteins has a distinct mean profile across samples,
    so silhouette should select k=3 cleanly.
    """
    rng = np.random.default_rng(seed)
    sample_names = [f"S{i:02d}" for i in range(n_samples)]
    n_proteins = n_per_pattern * 3

    profile_a = np.linspace(8.0, 12.0, n_samples)  # ramp up
    profile_b = np.linspace(12.0, 8.0, n_samples)  # ramp down
    profile_c = np.tile([10.0, 11.0, 9.0, 10.5], n_samples // 4 + 1)[:n_samples]  # zigzag

    rows = []
    proteins = []
    for i in range(n_per_pattern):
        rows.append(profile_a + rng.normal(0, 0.1, n_samples))
        proteins.append(f"A{i:02d}")
    for i in range(n_per_pattern):
        rows.append(profile_b + rng.normal(0, 0.1, n_samples))
        proteins.append(f"B{i:02d}")
    for i in range(n_per_pattern):
        rows.append(profile_c + rng.normal(0, 0.1, n_samples))
        proteins.append(f"C{i:02d}")

    matrix = np.power(2.0, np.array(rows))  # convert log2 -> linear
    df = pd.DataFrame(matrix, columns=sample_names)
    df.insert(0, "Protein", proteins)
    df.insert(1, "leading_gene_name", [f"Gene_{p}" for p in proteins])
    return df, sample_names, n_proteins


def test_cluster_proteins_kmeans_recovers_three_planted_patterns():
    df, samples, _n = _make_pattern_clustered_data()
    assignments, scan = cluster_proteins_kmeans(
        df, samples, k=None, k_range=(2, 6), log_transform=True, random_state=0
    )
    # Silhouette should pick k=3
    chosen_k = int(scan.loc[scan["chosen"], "k"].iloc[0])
    assert chosen_k == 3
    # Each planted group of proteins (A*, B*, C*) should land in a single cluster
    for prefix in ("A", "B", "C"):
        sub = assignments[assignments["Protein"].str.startswith(prefix)]
        assert sub["cluster"].nunique() == 1
    # The three planted groups should occupy three distinct clusters
    cluster_per_prefix = {
        prefix: assignments[assignments["Protein"].str.startswith(prefix)]["cluster"].iloc[0]
        for prefix in ("A", "B", "C")
    }
    assert len(set(cluster_per_prefix.values())) == 3


def test_cluster_proteins_kmeans_explicit_k_returns_single_row_scan():
    df, samples, _n = _make_pattern_clustered_data()
    assignments, scan = cluster_proteins_kmeans(
        df, samples, k=3, log_transform=True, random_state=0
    )
    assert len(scan) == 1
    assert scan["k"].iloc[0] == 3
    assert assignments["cluster"].nunique() == 3
    assert (assignments["silhouette"] >= -1).all() and (assignments["silhouette"] <= 1).all()


def test_cluster_proteins_kmeans_rejects_invalid_k():
    df, samples, _n = _make_pattern_clustered_data()
    with pytest.raises(ValueError, match="k must satisfy"):
        cluster_proteins_kmeans(df, samples, k=1, log_transform=True)
