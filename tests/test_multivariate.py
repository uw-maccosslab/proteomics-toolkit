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


def _stratified_dataset(seed=5, stratum_shift=4.0, group_effect=0.0, n_proteins=40):
    """Two strata with a large offset and groups spread unevenly over them.

    Stratum S1 has 1 A and 7 B samples, S2 has 7 A and 1 B, and every protein in S2 is shifted
    by ``stratum_shift``. Without strata the offset looks like a group difference.
    """
    rng = np.random.default_rng(seed)
    samples, metadata = [], {}
    layout = [("S1", "A", 1), ("S1", "B", 7), ("S2", "A", 7), ("S2", "B", 1)]
    for stratum, group, n in layout:
        for i in range(n):
            name = f"{stratum}_{group}_{i}"
            samples.append(name)
            metadata[name] = {"Group": group, "Day": stratum}
    values = rng.normal(10.0, 0.3, size=(n_proteins, len(samples)))
    for j, name in enumerate(samples):
        if metadata[name]["Day"] == "S2":
            values[:, j] += stratum_shift
        if metadata[name]["Group"] == "B":
            values[:10, j] += group_effect
    df = pd.DataFrame(np.power(2.0, values), columns=samples)
    df.insert(0, "Protein", [f"P{i:02d}" for i in range(n_proteins)])
    return df, samples, metadata


def test_permanova_strata_removes_stratum_offset():
    df, samples, metadata = _stratified_dataset()
    plain = permanova(df, samples, metadata, factor="Group", n_permutations=199)
    stratified = permanova(df, samples, metadata, factor="Group", strata="Day", n_permutations=199)
    assert plain["p_value"] < 0.05  # the stratum offset masquerades as a group effect
    assert stratified["p_value"] > 0.05
    assert stratified["R2"] < 0.2
    assert stratified["strata"] == "Day"
    assert stratified["n_strata"] == 2
    assert plain["n_strata"] == 1


def test_permanova_strata_still_finds_a_real_effect():
    df, samples, metadata = _stratified_dataset(group_effect=2.0)
    res = permanova(df, samples, metadata, factor="Group", strata="Day", n_permutations=199)
    assert res["p_value"] < 0.05


def test_permanova_strata_permutes_only_within_strata():
    # One sample of each group per stratum: only 2^4 = 16 labelings exist within strata, so the
    # smallest attainable p is about 1/16, never the 1/1000 an unrestricted shuffle allows.
    rng = np.random.default_rng(0)
    samples, metadata = [], {}
    for s in range(4):
        for g in ("A", "B"):
            name = f"S{s}_{g}"
            samples.append(name)
            metadata[name] = {"Group": g, "Day": f"S{s}"}
    values = rng.normal(10, 0.2, size=(20, len(samples)))
    values[:, 1::2] += 3.0  # a huge group effect
    df = pd.DataFrame(np.power(2.0, values), columns=samples)
    res = permanova(df, samples, metadata, factor="Group", strata="Day", n_permutations=999)
    assert res["p_value"] > 0.03


def test_permanova_strata_drops_samples_without_a_stratum():
    df, samples, metadata = _stratified_dataset()
    metadata[samples[0]] = {"Group": "A"}  # no Day
    res = permanova(df, samples, metadata, factor="Group", strata="Day", n_permutations=9)
    assert res["n_samples"] == len(samples) - 1


def test_permanova_strata_rejects_braycurtis():
    df, samples, metadata = _stratified_dataset()
    with pytest.raises(ValueError):
        permanova(df, samples, metadata, factor="Group", strata="Day", metric="braycurtis")
