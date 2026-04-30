"""Tests for proteomics_toolkit.marker_discovery."""

import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.marker_discovery import (
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
