"""Tests for proteomics_toolkit.multivariate (PERMANOVA)."""

import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.multivariate import permanova


def _null_dataset(n_per_group=4, n_proteins=30, seed=1):
    """Two groups with identical distributions: PERMANOVA p should be uniform-ish, R^2 small."""
    rng = np.random.default_rng(seed)
    samples = []
    metadata = {}
    for g in ("A", "B"):
        for i in range(n_per_group):
            name = f"{g}_{i + 1}"
            samples.append(name)
            metadata[name] = {"Method": g}
    values = rng.normal(10.0, 0.5, size=(n_proteins, len(samples)))
    df = pd.DataFrame(np.power(2.0, values), columns=samples)
    df.insert(0, "Protein", [f"P{i:02d}" for i in range(n_proteins)])
    return df, samples, metadata


def _strong_effect_dataset(n_per_group=4, n_proteins=30, seed=2, effect=3.0):
    rng = np.random.default_rng(seed)
    samples = []
    metadata = {}
    for g in ("A", "B"):
        for i in range(n_per_group):
            name = f"{g}_{i + 1}"
            samples.append(name)
            metadata[name] = {"Method": g}
    values = rng.normal(10.0, 0.5, size=(n_proteins, len(samples)))
    # Shift first 10 proteins up in group B
    for j, name in enumerate(samples):
        if metadata[name]["Method"] == "B":
            values[0:10, j] += effect
    df = pd.DataFrame(np.power(2.0, values), columns=samples)
    df.insert(0, "Protein", [f"P{i:02d}" for i in range(n_proteins)])
    return df, samples, metadata


def test_permanova_returns_expected_keys():
    df, samples, meta = _strong_effect_dataset()
    res = permanova(df, samples, meta, factor="Method", n_permutations=99, random_state=0)
    expected_keys = {"factor", "F", "R2", "p_value", "n_permutations", "n_samples", "groups", "group_sizes", "metric"}
    assert expected_keys <= set(res)
    assert res["factor"] == "Method"
    assert res["n_permutations"] == 99
    assert res["n_samples"] == 8
    assert sorted(res["groups"]) == ["A", "B"]


def test_permanova_strong_effect_is_significant():
    df, samples, meta = _strong_effect_dataset(effect=3.0)
    res = permanova(df, samples, meta, factor="Method", n_permutations=499, random_state=0)
    assert res["F"] > 5.0
    assert res["R2"] > 0.4
    # p-value bound: with 499 perms and a strong effect, observed should be the
    # most extreme; minimum achievable p is 1/500 = 0.002.
    assert res["p_value"] <= 0.05


def test_permanova_null_effect_pvalue_is_high_on_average():
    """With no real effect, p-value distribution should be approximately uniform."""
    rng = np.random.default_rng(123)
    p_vals = []
    for _ in range(20):
        seed = int(rng.integers(0, 10**6))
        df, samples, meta = _null_dataset(seed=seed)
        res = permanova(df, samples, meta, factor="Method", n_permutations=99, random_state=seed)
        p_vals.append(res["p_value"])
    # Under the null, mean p should be close to 0.5; we accept a wide range to avoid flakiness.
    assert 0.2 < np.mean(p_vals) < 0.8


def test_permanova_rejects_single_group():
    df, samples, meta = _strong_effect_dataset()
    # Collapse all to one label
    meta_one = {k: {"Method": "A"} for k in meta}
    with pytest.raises(ValueError, match="2 distinct groups"):
        permanova(df, samples, meta_one, factor="Method", n_permutations=99)


def test_permanova_rejects_too_few_samples():
    df, samples, meta = _strong_effect_dataset(n_per_group=2)
    # Keep only 3 samples (filter the dict)
    keep = samples[:3]
    meta_3 = {k: meta[k] for k in keep}
    with pytest.raises(ValueError, match=">=4 samples"):
        permanova(df, keep, meta_3, factor="Method", n_permutations=99)


def test_permanova_rejects_invalid_metric():
    df, samples, meta = _strong_effect_dataset()
    with pytest.raises(ValueError, match="metric"):
        permanova(df, samples, meta, factor="Method", metric="nonsense", n_permutations=10)
