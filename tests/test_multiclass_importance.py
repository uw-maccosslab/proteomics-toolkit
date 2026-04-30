"""Tests for proteomics_toolkit.classification.multiclass_feature_importance."""

import numpy as np
import pandas as pd

from proteomics_toolkit.classification import multiclass_feature_importance


def _dataset_with_planted_markers(n_per_group=6, n_proteins=40, seed=42):
    """Three-group dataset with strong markers planted in proteins P00-P05."""
    rng = np.random.default_rng(seed)
    groups = ["A", "B", "C"]
    samples = []
    metadata = {}
    for g in groups:
        for i in range(n_per_group):
            name = f"{g}_{i + 1}"
            samples.append(name)
            metadata[name] = {"Method": g}
    values = rng.normal(10.0, 0.4, size=(n_proteins, len(samples)))
    # Plant markers: P00-P01 high in A, P02-P03 high in B, P04-P05 high in C
    for j, name in enumerate(samples):
        g = metadata[name]["Method"]
        if g == "A":
            values[0:2, j] += 3.0
        elif g == "B":
            values[2:4, j] += 3.0
        elif g == "C":
            values[4:6, j] += 3.0
    linear = np.power(2.0, values)
    df = pd.DataFrame(linear, columns=samples)
    df.insert(0, "Protein", [f"P{i:02d}" for i in range(n_proteins)])
    df.insert(1, "Description", [f"Description {i}" for i in range(n_proteins)])
    df.insert(2, "leading_gene_name", [f"Gene_{i:02d}" for i in range(n_proteins)])
    return df, samples, metadata


def test_multiclass_feature_importance_returns_expected_columns():
    df, samples, meta = _dataset_with_planted_markers()
    out = multiclass_feature_importance(
        df, samples, meta, group_column="Method",
        n_repeats=10, n_estimators=100, bootstrap_iters=0, random_state=0,
    )
    assert {"importance_mean", "importance_std", "stability"} <= set(out.columns)
    assert len(out) == 40
    assert "oob_score" in out.attrs
    assert out.attrs["classes"] == ["A", "B", "C"]


def test_multiclass_feature_importance_ranks_planted_markers_high():
    df, samples, meta = _dataset_with_planted_markers(n_per_group=8)
    out = multiclass_feature_importance(
        df, samples, meta, group_column="Method",
        n_repeats=20, n_estimators=200, bootstrap_iters=0, random_state=0,
    )
    top6 = set(out.head(6)["Protein"].tolist())
    planted = {f"P{i:02d}" for i in range(6)}
    # Allow one miss to keep test robust to RF stochasticity
    assert len(top6 & planted) >= 5


def test_multiclass_feature_importance_bootstrap_stability_in_unit_interval():
    df, samples, meta = _dataset_with_planted_markers(n_per_group=8)
    out = multiclass_feature_importance(
        df, samples, meta, group_column="Method",
        n_repeats=10, n_estimators=100,
        bootstrap_iters=10, top_k_for_stability=10, random_state=0,
    )
    assert (out["stability"] >= 0).all()
    assert (out["stability"] <= 1).all()
    # Planted markers should have higher mean stability than non-planted
    planted_stability = out[out["Protein"].isin([f"P{i:02d}" for i in range(6)])]["stability"].mean()
    non_planted_stability = out[~out["Protein"].isin([f"P{i:02d}" for i in range(6)])]["stability"].mean()
    assert planted_stability > non_planted_stability


def test_multiclass_feature_importance_rejects_too_few_samples():
    df, samples, meta = _dataset_with_planted_markers(n_per_group=1)  # 3 samples total
    import pytest

    with pytest.raises(ValueError, match=">=6 samples"):
        multiclass_feature_importance(
            df, samples, meta, group_column="Method",
            n_repeats=5, n_estimators=20, bootstrap_iters=0,
        )
