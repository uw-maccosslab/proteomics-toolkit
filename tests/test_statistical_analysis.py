"""Tests for the statistical_analysis module."""

import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.statistical_analysis import (
    StatisticalConfig,
    _calibrate_trend_to_design,
    _fit_limma_prior,
    _sanitize_formula_term,
    _trigamma_inverse,
    apply_multiple_testing_correction,
    get_intensity_trend_points,
    run_comprehensive_statistical_analysis,
    run_mann_whitney_test,
    run_mixed_effects_analysis,
    run_moderated_linear_model,
    run_paired_t_test,
    run_unpaired_t_test,
    run_wilcoxon_test,
)

# ---------------------------------------------------------------------------
# Fixtures specific to statistical tests
# ---------------------------------------------------------------------------


def _make_unpaired_data():
    """Create protein_data + metadata_df for unpaired tests."""
    rng = np.random.default_rng(42)
    samples_a = [f"A_{i}" for i in range(5)]
    samples_b = [f"B_{i}" for i in range(5)]
    all_samples = samples_a + samples_b

    proteins = [f"Protein_{i}" for i in range(10)]
    values = rng.uniform(1e5, 1e7, size=(10, 10))
    # Make group B consistently higher for first protein
    values[0, 5:] += 5e6

    protein_data = pd.DataFrame(values, index=proteins, columns=all_samples)

    metadata_df = pd.DataFrame(
        {
            "Sample": all_samples,
            "Group": ["Control"] * 5 + ["Treatment"] * 5,
        }
    )

    config = StatisticalConfig()
    config.analysis_type = "unpaired"
    config.group_column = "Group"
    config.group_labels = ["Control", "Treatment"]
    config.log_transform_before_stats = False

    return protein_data, metadata_df, config


def _make_paired_data():
    """Create protein_data + metadata_df for paired tests."""
    rng = np.random.default_rng(42)
    subjects = ["S1", "S2", "S3", "S4", "S5"]
    samples_pre = [f"{s}_Pre" for s in subjects]
    samples_post = [f"{s}_Post" for s in subjects]
    all_samples = samples_pre + samples_post

    proteins = [f"Protein_{i}" for i in range(10)]
    values = rng.uniform(1e5, 1e7, size=(10, 10))
    # Make post consistently higher for first protein
    values[0, 5:] += 5e6

    protein_data = pd.DataFrame(values, index=proteins, columns=all_samples)

    rows = []
    for s in subjects:
        rows.append({"Sample": f"{s}_Pre", "Subject": s, "Timepoint": "Pre", "Group": "A"})
        rows.append({"Sample": f"{s}_Post", "Subject": s, "Timepoint": "Post", "Group": "A"})
    metadata_df = pd.DataFrame(rows)

    config = StatisticalConfig()
    config.analysis_type = "paired"
    config.group_column = "Group"
    config.group_labels = ["A"]
    config.subject_column = "Subject"
    config.paired_column = "Timepoint"
    config.paired_label1 = "Pre"
    config.paired_label2 = "Post"
    config.log_transform_before_stats = False

    return protein_data, metadata_df, config


# ---------------------------------------------------------------------------
# StatisticalConfig
# ---------------------------------------------------------------------------


class TestStatisticalConfig:
    def test_defaults(self):
        config = StatisticalConfig()
        assert config.p_value_threshold == 0.05
        assert config.fold_change_threshold == 1.5
        assert config.correction_method == "fdr_bh"

    def test_validate_raises_without_analysis_type(self):
        config = StatisticalConfig()
        with pytest.raises(ValueError, match="analysis_type must be set"):
            config.validate()

    def test_validate_paired_requires_labels(self):
        config = StatisticalConfig()
        config.analysis_type = "paired"
        config.group_column = "Group"
        config.group_labels = ["A", "B"]
        with pytest.raises(ValueError, match="paired_label1"):
            config.validate()

    def test_validate_unpaired_passes(self):
        config = StatisticalConfig()
        config.analysis_type = "unpaired"
        config.group_column = "Group"
        config.group_labels = ["A", "B"]
        assert config.validate() is True

    def test_validate_linear_trend_requires_time_column(self):
        config = StatisticalConfig()
        config.analysis_type = "linear_trend"
        with pytest.raises(ValueError, match="time_column"):
            config.validate()

    def test_validate_paired_accepts_zero_label(self):
        # Regression: a paired_label of 0 (e.g. "Week 0" stored as int) used
        # to be rejected because the validator checked truthiness instead of
        # `is None`.
        config = StatisticalConfig()
        config.analysis_type = "paired"
        config.statistical_test_method = "paired_t"  # avoid mixed_effects subject requirement
        config.group_column = "Week"
        config.group_labels = [0, 12]
        config.paired_label1 = 0
        config.paired_label2 = 12
        assert config.validate() is True


# ---------------------------------------------------------------------------
# _sanitize_formula_term
# ---------------------------------------------------------------------------


class TestSanitizeFormulaTerm:
    def test_simple_term_unchanged(self):
        assert _sanitize_formula_term("Group") == "Group"

    def test_term_with_space_is_quoted(self):
        assert _sanitize_formula_term("Time Point") == 'Q("Time Point")'

    def test_term_with_special_chars_is_quoted(self):
        assert _sanitize_formula_term("dose-response") == 'Q("dose-response")'


# ---------------------------------------------------------------------------
# Unpaired tests
# ---------------------------------------------------------------------------


class TestUnpairedTTest:
    def test_returns_dataframe(self):
        protein_data, metadata_df, config = _make_unpaired_data()
        result = run_unpaired_t_test(protein_data, metadata_df, config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(protein_data)

    def test_result_has_pvalue_column(self):
        protein_data, metadata_df, config = _make_unpaired_data()
        result = run_unpaired_t_test(protein_data, metadata_df, config)
        assert "P.Value" in result.columns or "p_value" in result.columns


class TestMannWhitneyTest:
    def test_returns_dataframe(self):
        protein_data, metadata_df, config = _make_unpaired_data()
        result = run_mann_whitney_test(protein_data, metadata_df, config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(protein_data)


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------


class TestPairedTTest:
    def test_returns_dataframe(self):
        protein_data, metadata_df, config = _make_paired_data()
        result = run_paired_t_test(protein_data, metadata_df, config)
        assert isinstance(result, pd.DataFrame)

    def test_result_has_pvalue_column(self):
        protein_data, metadata_df, config = _make_paired_data()
        result = run_paired_t_test(protein_data, metadata_df, config)
        assert "P.Value" in result.columns or "p_value" in result.columns


class TestWilcoxonTest:
    def test_returns_dataframe(self):
        protein_data, metadata_df, config = _make_paired_data()
        result = run_wilcoxon_test(protein_data, metadata_df, config)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# apply_multiple_testing_correction
# ---------------------------------------------------------------------------


class TestMultipleTestingCorrection:
    def test_fdr_bh_correction(self):
        results_df = pd.DataFrame(
            {
                "Protein": [f"P{i}" for i in range(5)],
                "P.Value": [0.01, 0.04, 0.03, 0.20, 0.50],
                "logFC": [1.0, -0.5, 0.8, 0.1, -0.1],
            }
        )
        config = StatisticalConfig()
        config.correction_method = "fdr_bh"
        corrected = apply_multiple_testing_correction(results_df, config)
        assert isinstance(corrected, pd.DataFrame)
        assert "adj.P.Val" in corrected.columns

    def test_bonferroni_correction(self):
        results_df = pd.DataFrame(
            {
                "Protein": [f"P{i}" for i in range(3)],
                "P.Value": [0.01, 0.04, 0.03],
                "logFC": [1.0, -0.5, 0.8],
            }
        )
        config = StatisticalConfig()
        config.correction_method = "bonferroni"
        corrected = apply_multiple_testing_correction(results_df, config)
        assert isinstance(corrected, pd.DataFrame)
        assert "adj.P.Val" in corrected.columns


# ---------------------------------------------------------------------------
# Peptide-level statistics
#
# The existing statistical functions are row-indexed and do not assume a
# protein identifier. These tests verify that peptide rows flow through the
# same pipeline.
# ---------------------------------------------------------------------------


def _make_unpaired_peptide_data():
    """Create peptide_data + metadata_df for unpaired tests."""
    rng = np.random.default_rng(7)
    samples_a = [f"A_{i}" for i in range(5)]
    samples_b = [f"B_{i}" for i in range(5)]
    all_samples = samples_a + samples_b

    peptides = [f"PEPTIDE_{i}" for i in range(15)]
    values = rng.uniform(1e4, 1e6, size=(15, 10))
    values[0, 5:] += 5e5  # make first peptide clearly different in group B

    peptide_data = pd.DataFrame(values, index=peptides, columns=all_samples)

    metadata_df = pd.DataFrame(
        {
            "Sample": all_samples,
            "Group": ["Control"] * 5 + ["Treatment"] * 5,
        }
    )

    config = StatisticalConfig()
    config.analysis_type = "unpaired"
    config.group_column = "Group"
    config.group_labels = ["Control", "Treatment"]
    config.log_transform_before_stats = False

    return peptide_data, metadata_df, config


def _make_paired_peptide_data():
    """Create peptide_data + metadata_df for paired tests."""
    rng = np.random.default_rng(8)
    subjects = ["S1", "S2", "S3", "S4", "S5"]
    samples_pre = [f"{s}_Pre" for s in subjects]
    samples_post = [f"{s}_Post" for s in subjects]
    all_samples = samples_pre + samples_post

    peptides = [f"PEPTIDE_{i}" for i in range(15)]
    values = rng.uniform(1e4, 1e6, size=(15, 10))
    values[0, 5:] += 5e5

    peptide_data = pd.DataFrame(values, index=peptides, columns=all_samples)

    rows = []
    for s in subjects:
        rows.append({"Sample": f"{s}_Pre", "Subject": s, "Timepoint": "Pre", "Group": "A"})
        rows.append({"Sample": f"{s}_Post", "Subject": s, "Timepoint": "Post", "Group": "A"})
    metadata_df = pd.DataFrame(rows)

    config = StatisticalConfig()
    config.analysis_type = "paired"
    config.group_column = "Group"
    config.group_labels = ["A"]
    config.subject_column = "Subject"
    config.paired_column = "Timepoint"
    config.paired_label1 = "Pre"
    config.paired_label2 = "Post"
    config.log_transform_before_stats = False

    return peptide_data, metadata_df, config


class TestPeptideLevelStatistics:
    def test_unpaired_t_test_on_peptides(self):
        peptide_data, metadata_df, config = _make_unpaired_peptide_data()
        result = run_unpaired_t_test(peptide_data, metadata_df, config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(peptide_data)

    def test_mann_whitney_on_peptides(self):
        peptide_data, metadata_df, config = _make_unpaired_peptide_data()
        result = run_mann_whitney_test(peptide_data, metadata_df, config)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(peptide_data)

    def test_paired_t_test_on_peptides(self):
        peptide_data, metadata_df, config = _make_paired_peptide_data()
        result = run_paired_t_test(peptide_data, metadata_df, config)
        assert isinstance(result, pd.DataFrame)

    def test_wilcoxon_on_peptides(self):
        peptide_data, metadata_df, config = _make_paired_peptide_data()
        result = run_wilcoxon_test(peptide_data, metadata_df, config)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# limma_like / deqms_like
# ---------------------------------------------------------------------------


def _make_limma_fixture(with_effect_rows=10, total_rows=60, seed=42):
    """Build a protein DataFrame + metadata + config for limma/DEqMS tests.

    Values are log-space so ``log_transform_before_stats = False`` is safe.
    The first ``with_effect_rows`` features get a +1.5 treatment shift and
    everyone else is pure noise.
    """
    rng = np.random.default_rng(seed)
    samples_a = [f"A_{i}" for i in range(6)]
    samples_b = [f"B_{i}" for i in range(6)]
    all_samples = samples_a + samples_b

    values = rng.normal(loc=10, scale=0.5, size=(total_rows, 12))
    values[:with_effect_rows, 6:] += 1.5

    features = [f"P{i:04d}" for i in range(total_rows)]
    feature_data = pd.DataFrame(values, index=features, columns=all_samples)

    metadata_df = pd.DataFrame(
        {
            "Sample": all_samples,
            "Group": ["Control"] * 6 + ["Treatment"] * 6,
        }
    )

    config = StatisticalConfig()
    config.analysis_type = "unpaired"
    config.group_column = "Group"
    config.group_labels = ["Control", "Treatment"]
    config.log_transform_before_stats = False
    config.statistical_test_method = "limma_like"

    return feature_data, metadata_df, config


class TestTrigammaInverse:
    def test_identity_round_trip(self):
        # ψ'(ψ'⁻¹(x)) should recover x for x in (0, ∞).
        from scipy.special import polygamma

        for target in [0.01, 0.1, 0.5, 1.0, 5.0]:
            inv = _trigamma_inverse(target)
            assert np.isclose(polygamma(1, inv), target, rtol=1e-6)


class TestModeratedLinearModelLimma:
    def test_returns_standard_schema(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "limma"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        for col in ("Protein", "logFC", "AveExpr", "t", "P.Value", "n_group1", "n_group2"):
            assert col in result.columns
        assert len(result) == len(feature_data)

    def test_ranks_differential_features_first(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "limma"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        top10 = set(result.sort_values("P.Value").head(10)["Protein"])
        expected = {f"P{i:04d}" for i in range(10)}
        assert top10 == expected

    def test_fold_change_sign_matches_direction(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "limma"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        spiked = result[result["Protein"].str.match(r"P000[0-9]$")]
        assert (spiked["logFC"] > 0).all()

    def test_peptide_level_works(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "limma"
        feature_data.index = [f"PEPTIDE_{i}" for i in range(len(feature_data))]
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        assert len(result) == len(feature_data)
        assert result["Protein"].iloc[0].startswith("PEPTIDE_")


class TestModeratedLinearModelDeqms:
    def test_returns_standard_schema_with_count_col(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        rng = np.random.default_rng(0)
        feature_data["n_peptides"] = rng.integers(2, 20, size=len(feature_data))
        config.moderation = "deqms"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        for col in ("Protein", "logFC", "t", "P.Value", "peptide_count_used", "deqms_s0_sq"):
            assert col in result.columns

    def test_missing_count_column_raises(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "deqms"
        with pytest.raises(ValueError, match="peptide-count column"):
            run_moderated_linear_model(feature_data, metadata_df, config)

    def test_custom_count_column_honoured(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        rng = np.random.default_rng(1)
        feature_data["custom_count"] = rng.integers(2, 20, size=len(feature_data))
        config.moderation = "deqms"
        config.peptide_count_column = "custom_count"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        assert result["peptide_count_used"].notna().all()

    def test_ranks_differential_features_first(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        rng = np.random.default_rng(2)
        feature_data["n_peptides"] = rng.integers(2, 20, size=len(feature_data))
        config.moderation = "deqms"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        top10 = set(result.sort_values("P.Value").head(10)["Protein"])
        expected = {f"P{i:04d}" for i in range(10)}
        assert top10 == expected


class TestModeratedLinearModelIntensityPeptideTrend:
    """The additive intensity + peptide-count variance prior.

    Peptide count carries variance information that intensity alone misses:
    at matched intensity a protein rolled up from many peptides is better
    determined than one from few. These tests pin that the second stage is
    wired in, is actually used, and degrades gracefully.
    """

    def test_returns_intensity_and_peptide_columns(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        rng = np.random.default_rng(0)
        feature_data["n_peptides"] = rng.integers(2, 30, size=len(feature_data))
        config.moderation = "intensity_peptide_trend"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        for col in ("Protein", "logFC", "P.Value", "intensity_s0_sq", "peptide_count_used"):
            assert col in result.columns
        assert result["peptide_count_used"].notna().all()

    def test_peptide_count_column_is_not_treated_as_a_sample(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        rng = np.random.default_rng(1)
        feature_data["n_peptides"] = rng.integers(2, 30, size=len(feature_data))
        config.moderation = "intensity_peptide_trend"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        # One row per feature: the count column must have been dropped before
        # the design fit rather than being consumed as an extra sample.
        assert len(result) == len(feature_data)

    def test_missing_count_column_raises(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "intensity_peptide_trend"
        with pytest.raises(ValueError, match="peptide-count column"):
            run_moderated_linear_model(feature_data, metadata_df, config)

    def test_custom_count_column_honoured(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        rng = np.random.default_rng(2)
        feature_data["my_counts"] = rng.integers(2, 30, size=len(feature_data))
        config.moderation = "intensity_peptide_trend"
        config.peptide_count_column = "my_counts"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        assert result["peptide_count_used"].notna().all()

    def test_peptide_stage_changes_the_prior(self):
        """The second stage must actually move the prior, not silently no-op.

        Counts are made strongly informative (variance inflated for
        low-count features) so the peptide stage has real signal to find.
        """
        feature_data, metadata_df, config = _make_limma_fixture()
        rng = np.random.default_rng(3)
        counts = rng.integers(2, 30, size=len(feature_data))
        sample_cols = [c for c in feature_data.columns if c.startswith("S")]
        noise = rng.normal(0, 1, size=(len(feature_data), len(sample_cols)))
        inflate = (30.0 / counts)[:, None]
        feature_data[sample_cols] = feature_data[sample_cols].to_numpy() + noise * inflate

        config.moderation = "intensity_trend"
        without = run_moderated_linear_model(feature_data.copy(), metadata_df, config)

        fd = feature_data.copy()
        fd["n_peptides"] = counts
        config.moderation = "intensity_peptide_trend"
        with_pep = run_moderated_linear_model(fd, metadata_df, config)

        assert not np.allclose(
            without["intensity_s0_sq"].to_numpy(dtype=float),
            with_pep["intensity_s0_sq"].to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_points_carry_peptide_diagnostic_columns(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        rng = np.random.default_rng(4)
        feature_data["n_peptides"] = rng.integers(2, 30, size=len(feature_data))
        config.moderation = "intensity_peptide_trend"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        pts = get_intensity_trend_points(result)
        assert {"peptide_count_used", "peptide_log_var_adj", "intensity_log_var_hat"}.issubset(pts.columns)

    def test_constant_peptide_count_degrades_to_intensity_only(self):
        """A degenerate count column must not blow up or corrupt the prior."""
        feature_data, metadata_df, config = _make_limma_fixture()
        feature_data["n_peptides"] = 5  # no variation -> nothing for stage 2
        config.moderation = "intensity_peptide_trend"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        assert np.isfinite(result["intensity_s0_sq"].to_numpy(dtype=float)).all()


class TestModeratedLinearModelIntensityTrend:
    def test_returns_intensity_columns_and_attrs(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "intensity_trend"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        for col in ("Protein", "logFC", "t", "P.Value", "intensity_s0_sq", "intensity_used"):
            assert col in result.columns
        # Per-(feature, group) points are stashed on attrs as a list of records
        # (one dict per row) to avoid tripping pandas attrs-equality. Recover
        # the DataFrame via the canonical accessor.
        pts = get_intensity_trend_points(result)
        assert {"feature_idx", "group", "mean_intensity", "sd_intensity", "predicted_sd"}.issubset(pts.columns)

    def test_get_intensity_trend_points_accessor(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "intensity_trend"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        pts = get_intensity_trend_points(result)
        assert len(pts) > 0
        assert "mean_intensity" in pts.columns

    def test_get_intensity_trend_points_raises_when_missing(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "limma"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        with pytest.raises(ValueError, match="intensity_trend"):
            get_intensity_trend_points(result)

    def test_ranks_differential_features_first(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "intensity_trend"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        top10 = set(result.sort_values("P.Value").head(10)["Protein"])
        expected = {f"P{i:04d}" for i in range(10)}
        assert top10 == expected

    def test_intensity_trend_attrs_do_not_break_pandas_ops(self):
        """Regression: prior to records-form storage, attaching a DataFrame
        to ``results_df.attrs["intensity_trend_points"]`` made any subsequent
        nsmallest / sort_values / concat raise "The truth value of a
        DataFrame is ambiguous" because pandas compares attrs by equality
        on concat and DataFrame == DataFrame returns a DataFrame.
        """
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "intensity_trend"
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        # These ops must not raise; before the fix they raised ValueError.
        _ = result.nsmallest(5, "P.Value")
        _ = result.sort_values("P.Value").head(5)

    def test_intensity_trend_attrs_do_not_slow_iterrows(self):
        """Regression: storing the per-(feature, group) points as a plain
        list of dicts in attrs makes pandas deep-copy the entire list every
        time it propagates attrs to a new object - including the Series
        yielded by every iterrows step. For a real protein matrix (~8k
        proteins * 2 groups = ~17k records) that turns an 8k-row iterrows
        loop from ~2s into ~22 minutes. Wrapping the records in
        ``_AttrsPayload`` (a deepcopy-no-op sentinel) avoids the cost.

        This test asserts that iterrows over the result is fast and that
        the per-row Series carries the same attrs object identity rather
        than a fresh copy.
        """
        import time

        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "intensity_trend"
        result = run_moderated_linear_model(feature_data, metadata_df, config)

        # Sanity check: payload is present and non-trivial in size
        pts = get_intensity_trend_points(result)
        assert len(pts) > 50

        # iterrows must stay quick - small fixture only ~30 rows but the same
        # deepcopy-per-row code path runs. Allow generous headroom (50ms is
        # still ~3 orders of magnitude faster than the bug's behaviour).
        t0 = time.time()
        rows = list(result.iterrows())
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"iterrows took {elapsed:.3f}s - attrs deepcopy may be back"

        # Sentinel sanity: the wrapped payload in attrs is the same object
        # propagated to each row's attrs (no deepcopy).
        parent_payload = result.attrs.get("intensity_trend_points")
        for _, row in rows[:3]:
            assert row.attrs.get("intensity_trend_points") is parent_payload


def _make_confounded_fixture(
    n_features=60,
    n_planted=10,
    treatment_effect=1.5,
    age_effect=0.05,
    seed=7,
):
    """Build a fixture where treatment is confounded with a continuous
    covariate (age) and shares variance with a categorical covariate (sex).

    Without covariate adjustment, the estimated treatment effect on the
    planted features is inflated by ``age_effect * (mean_age_treat -
    mean_age_control)``. After adjustment, the estimate should recover
    ``treatment_effect``.
    """
    rng = np.random.default_rng(seed)
    n_control = 8
    n_treatment = 8
    samples_a = [f"A_{i}" for i in range(n_control)]
    samples_b = [f"B_{i}" for i in range(n_treatment)]
    all_samples = samples_a + samples_b
    n_samples = len(all_samples)

    age = np.concatenate(
        [rng.normal(30.0, 2.0, size=n_control), rng.normal(40.0, 2.0, size=n_treatment)]
    )
    sex = np.array(["F", "M"] * (n_samples // 2))
    rng.shuffle(sex)
    treat = np.array([0] * n_control + [1] * n_treatment, dtype=float)

    values = rng.normal(loc=10.0, scale=0.4, size=(n_features, n_samples))
    # Plant treatment + age effects on the first n_planted features
    values[:n_planted, :] += treatment_effect * treat[np.newaxis, :]
    values[:n_planted, :] += age_effect * age[np.newaxis, :]

    features = [f"P{i:04d}" for i in range(n_features)]
    feature_data = pd.DataFrame(values, index=features, columns=all_samples)
    metadata_df = pd.DataFrame(
        {
            "Sample": all_samples,
            "Group": ["Control"] * n_control + ["Treatment"] * n_treatment,
            "Age": age,
            "Sex": sex,
        }
    )

    config = StatisticalConfig()
    config.analysis_type = "unpaired"
    config.group_column = "Group"
    config.group_labels = ["Control", "Treatment"]
    config.log_transform_before_stats = False
    config.statistical_test_method = "moderated_linear_model"
    config.moderation = "limma"
    return feature_data, metadata_df, config, treatment_effect


class TestModeratedLinearModelCovariates:
    """Covariate adjustment in the unpaired moderated-linear-model path."""

    def test_covariate_adjustment_recovers_true_effect(self):
        feature_data, metadata_df, config, true_effect = _make_confounded_fixture()
        config.covariates = ["Age"]
        adjusted = run_moderated_linear_model(feature_data, metadata_df, config)
        config_no_cov = StatisticalConfig()
        config_no_cov.analysis_type = "unpaired"
        config_no_cov.group_column = "Group"
        config_no_cov.group_labels = ["Control", "Treatment"]
        config_no_cov.log_transform_before_stats = False
        config_no_cov.statistical_test_method = "moderated_linear_model"
        config_no_cov.moderation = "limma"
        config_no_cov.covariates = []
        unadjusted = run_moderated_linear_model(feature_data, metadata_df, config_no_cov)

        planted = [f"P{i:04d}" for i in range(10)]
        adj_fc = adjusted.set_index("Protein").loc[planted, "logFC"].mean()
        unadj_fc = unadjusted.set_index("Protein").loc[planted, "logFC"].mean()

        # Unadjusted is inflated by ~age_effect * delta_age = 0.05 * 10 = 0.5
        assert unadj_fc > true_effect + 0.2, (
            f"unadjusted mean logFC was {unadj_fc:.3f}; expected > {true_effect + 0.2}"
        )
        # Adjusted should be close to the true effect (within 0.25)
        assert abs(adj_fc - true_effect) < 0.25, (
            f"adjusted mean logFC was {adj_fc:.3f}; expected close to {true_effect}"
        )

    def test_categorical_covariate_dummy_encoded(self):
        feature_data, metadata_df, config, _ = _make_confounded_fixture()
        config.covariates = ["Sex"]
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        # Schema is preserved
        for col in ("Protein", "logFC", "P.Value", "n_group1", "n_group2"):
            assert col in result.columns
        assert len(result) == len(feature_data)
        # Planted features still rank near the top despite the extra (uninformative) covariate
        top20 = set(result.sort_values("P.Value").head(20)["Protein"])
        planted = {f"P{i:04d}" for i in range(10)}
        assert len(top20 & planted) >= 8

    def test_multiple_covariates(self):
        feature_data, metadata_df, config, true_effect = _make_confounded_fixture()
        config.covariates = ["Age", "Sex"]
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        planted = [f"P{i:04d}" for i in range(10)]
        adj_fc = result.set_index("Protein").loc[planted, "logFC"].mean()
        assert abs(adj_fc - true_effect) < 0.25

    def test_missing_covariate_listwise_deletion(self):
        feature_data, metadata_df, config, _ = _make_confounded_fixture()
        # Drop Age for two samples (one per group) -> they should be excluded
        metadata_df = metadata_df.copy()
        metadata_df.loc[metadata_df["Sample"] == "A_0", "Age"] = np.nan
        metadata_df.loc[metadata_df["Sample"] == "B_0", "Age"] = np.nan
        config.covariates = ["Age"]
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        # We started with 8 + 8 = 16 samples; expect 14 after listwise deletion
        # n_group1 + n_group2 reflects per-feature counts; AveExpr-supported features should use 14
        n_total = (result["n_group1"] + result["n_group2"]).max()
        assert n_total == 14

    def test_missing_covariate_column_raises(self):
        feature_data, metadata_df, config, _ = _make_confounded_fixture()
        config.covariates = ["NotAColumn"]
        with pytest.raises(ValueError, match="Covariate columns not present"):
            run_moderated_linear_model(feature_data, metadata_df, config)

    def test_empty_covariate_list_preserves_baseline(self):
        feature_data, metadata_df, config, _ = _make_confounded_fixture()
        config.covariates = []
        baseline = run_moderated_linear_model(feature_data, metadata_df, config)
        config.covariates = None
        none_variant = run_moderated_linear_model(feature_data, metadata_df, config)
        # Both empty-covariate paths should produce identical logFC
        pd.testing.assert_series_equal(
            baseline.set_index("Protein")["logFC"],
            none_variant.set_index("Protein")["logFC"],
            check_names=False,
        )

    def test_covariates_compose_with_intensity_trend(self):
        feature_data, metadata_df, config, true_effect = _make_confounded_fixture()
        config.moderation = "intensity_trend"
        config.covariates = ["Age"]
        result = run_moderated_linear_model(feature_data, metadata_df, config)
        # Intensity-trend output columns present
        for col in ("intensity_s0_sq", "intensity_used"):
            assert col in result.columns
        # Effect still recovered after covariate adjustment + intensity_trend prior
        planted = [f"P{i:04d}" for i in range(10)]
        adj_fc = result.set_index("Protein").loc[planted, "logFC"].mean()
        assert abs(adj_fc - true_effect) < 0.25


def _make_linear_trend_fixture(
    n_features=200,
    n_planted=20,
    timepoints=(0.0, 2.0, 4.0, 6.0, 12.0),
    n_subjects=10,
    slope=0.05,
    subject_sigma=0.4,
    noise_sigma=0.3,
    seed=11,
):
    """Build a longitudinal fixture for moderated linear-trend tests.

    Generates ``n_subjects`` × len(timepoints) samples. The first
    ``n_planted`` features get a real slope of ``slope`` per unit time
    plus a per-subject random intercept; remaining features are pure
    noise plus the same subject intercepts.

    Returns
    -------
    feature_data_raw : pd.DataFrame
        Raw (linear-scale) feature intensities. Suitable for the
        intensity_trend moderation path which expects raw values.
    feature_data_log : pd.DataFrame
        Log2 of feature_data_raw. Used directly with
        ``config.log_transform_before_stats = False`` for limma/deqms
        moderation tests.
    metadata_df : pd.DataFrame
        Long-format metadata with Sample / Subject / Week columns.
    config : StatisticalConfig
        Pre-populated for linear_trend with subject blocking.
    planted_features : list[str]
        IDs of the features with a planted slope.
    """
    rng = np.random.default_rng(seed)
    n_t = len(timepoints)
    n_samples = n_subjects * n_t

    samples = [f"S{s:02d}_W{int(t)}" for s in range(n_subjects) for t in timepoints]
    weeks = np.array([t for _ in range(n_subjects) for t in timepoints], dtype=float)
    subjects = [f"S{s:02d}" for s in range(n_subjects) for _ in timepoints]

    # Subject random intercepts shared across all features (mean expression centred at 10).
    subj_intercepts = rng.normal(0.0, subject_sigma, size=n_subjects)
    subj_intercept_per_sample = np.array([subj_intercepts[s] for s in range(n_subjects) for _ in timepoints])

    # Log-space feature matrix
    log_values = rng.normal(loc=10.0, scale=noise_sigma, size=(n_features, n_samples))
    log_values += subj_intercept_per_sample[np.newaxis, :]
    log_values[:n_planted, :] += slope * weeks[np.newaxis, :]

    features = [f"P{i:04d}" for i in range(n_features)]
    feature_data_log = pd.DataFrame(log_values, index=features, columns=samples)
    # Raw scale (intensity_trend expects raw / pre-log).
    feature_data_raw = pd.DataFrame(np.exp(log_values * np.log(2.0)), index=features, columns=samples)

    metadata_df = pd.DataFrame({"Sample": samples, "Subject": subjects, "Week": weeks})

    config = StatisticalConfig()
    config.analysis_type = "linear_trend"
    config.statistical_test_method = "moderated_linear_model"
    config.time_column = "Week"
    config.subject_column = "Subject"
    config.log_transform_before_stats = False
    config.moderation = "intensity_trend"

    planted_features = features[:n_planted]
    return feature_data_log, feature_data_raw, metadata_df, config, planted_features


class TestModeratedLinearTrend:
    """Tests for moderated linear-trend (slope) analysis.

    Direct callers of ``run_moderated_linear_model`` bypass the dispatcher's
    log-transform + raw-data stash. We pass log-space data as feature_data
    so the slope is on the natural ``log2/week`` scale, and stash a raw
    view on ``config._raw_feature_data`` for the intensity_trend prior.
    """

    def test_returns_standard_schema_linear_trend(self):
        log_data, raw, meta, config, _ = _make_linear_trend_fixture()
        config._raw_feature_data = raw
        result = run_moderated_linear_model(log_data, meta, config)
        for col in (
            "Protein",
            "logFC",
            "AveExpr",
            "t",
            "P.Value",
            "residual_s2",
            "posterior_s2",
            "residual_df",
            "posterior_df",
            "limma_s0_sq",
            "test_method",
            "intensity_s0_sq",
            "intensity_used",
        ):
            assert col in result.columns, f"missing column {col!r}"

    def test_recovers_planted_slope(self):
        log_data, raw, meta, config, planted = _make_linear_trend_fixture()
        config._raw_feature_data = raw
        result = run_moderated_linear_model(log_data, meta, config)
        planted_set = set(planted)
        # Slope is in log2 units per week. logFC for planted features should
        # be near the planted slope (0.05). Null features should sit near 0.
        planted_logfc = result.loc[result["Protein"].isin(planted_set), "logFC"]
        null_logfc = result.loc[~result["Protein"].isin(planted_set), "logFC"]
        assert abs(planted_logfc.median() - 0.05) < 0.02
        assert abs(null_logfc.median()) < 0.01

    def test_ranks_differential_features_first(self):
        log_data, raw, meta, config, planted = _make_linear_trend_fixture()
        config._raw_feature_data = raw
        result = run_moderated_linear_model(log_data, meta, config)
        top20 = set(result.sort_values("P.Value").head(20)["Protein"])
        # All 20 planted features should rank in the top 20 by P.Value with
        # this much signal vs noise; require at least 15 to remain robust
        # against the LOWESS prior occasionally upweighting borderline nulls.
        assert len(top20 & set(planted)) >= 15

    def test_intensity_trend_points_one_row_per_feature_per_time(self):
        log_data, raw, meta, config, _ = _make_linear_trend_fixture()
        config._raw_feature_data = raw
        result = run_moderated_linear_model(log_data, meta, config)
        pts = get_intensity_trend_points(result)
        # 5 unique weeks in the fixture
        assert pts["group"].nunique() == 5
        # 200 features x 5 groups = 1000 rows
        assert len(pts) == 200 * 5

    def test_subject_blocking_changes_posterior_df(self):
        log_data, _, meta, config, _ = _make_linear_trend_fixture()
        config.moderation = "limma"  # design check; not testing variance prior here
        # With subject blocking
        config.subject_column = "Subject"
        with_subj = run_moderated_linear_model(log_data, meta, config)
        # Without subject blocking
        config.subject_column = None
        no_subj = run_moderated_linear_model(log_data, meta, config)
        # With 10 subjects in the design, residual_df shrinks by ~9 when
        # subject is included as a fixed-effect block.
        assert with_subj["residual_df"].median() < no_subj["residual_df"].median()
        diff = no_subj["residual_df"].median() - with_subj["residual_df"].median()
        assert 8 <= diff <= 10  # n_subjects - 1 = 9

    def test_all_three_moderation_modes_run(self):
        for moderation in ("limma", "deqms", "intensity_trend"):
            log_data, raw, meta, config, _ = _make_linear_trend_fixture()
            config.moderation = moderation
            if moderation == "deqms":
                # Attach a peptide-count column for deqms moderation.
                log_data = log_data.copy()
                log_data["n_peptides"] = np.random.default_rng(1).integers(1, 20, size=len(log_data))
                result = run_moderated_linear_model(log_data, meta, config)
                assert "deqms_s0_sq" in result.columns
                assert "peptide_count_used" in result.columns
            elif moderation == "intensity_trend":
                config._raw_feature_data = raw
                result = run_moderated_linear_model(log_data, meta, config)
                assert "intensity_s0_sq" in result.columns
                assert "intensity_used" in result.columns
            else:  # limma
                result = run_moderated_linear_model(log_data, meta, config)
                assert "limma_s0_sq" in result.columns
            # Common columns across all moderation modes:
            for col in ("Protein", "logFC", "P.Value", "posterior_s2", "test_method"):
                assert col in result.columns

    def test_requires_time_column(self):
        log_data, raw, meta, config, _ = _make_linear_trend_fixture()
        config._raw_feature_data = raw
        config.time_column = None
        config.dose_column = None
        with pytest.raises(ValueError, match="time_column"):
            run_moderated_linear_model(log_data, meta, config)

    def test_requires_at_least_two_unique_times(self):
        log_data, raw, meta, config, _ = _make_linear_trend_fixture()
        config._raw_feature_data = raw
        # Collapse all timepoints to a single value
        meta = meta.copy()
        meta["Week"] = 0.0
        with pytest.raises(ValueError, match="unique"):
            run_moderated_linear_model(log_data, meta, config)

    def test_covariate_adjustment_recovers_slope(self):
        # Age confounds the score -> expression slope: age = 50 + 30*score, so an
        # age effect leaks into the unadjusted slope. Adjusting for Age recovers it.
        rng = np.random.default_rng(7)
        n_samples, n_features, n_planted = 48, 150, 15
        score = rng.uniform(0.0, 1.0, size=n_samples)
        age = 50.0 + 30.0 * score + rng.normal(0.0, 3.0, size=n_samples)
        true_slope, age_coef = 1.0, 0.02
        log_values = rng.normal(10.0, 0.3, size=(n_features, n_samples))
        log_values[:n_planted, :] += true_slope * score[np.newaxis, :]
        log_values[:n_planted, :] += age_coef * age[np.newaxis, :]
        feats = [f"P{i:04d}" for i in range(n_features)]
        samples = [f"S{i:02d}" for i in range(n_samples)]
        log_data = pd.DataFrame(log_values, index=feats, columns=samples)
        meta = pd.DataFrame({"Sample": samples, "Score": score, "Age": age})

        def _run(covs):
            cfg = StatisticalConfig()
            cfg.analysis_type = "linear_trend"
            cfg.statistical_test_method = "moderated_linear_model"
            cfg.time_column = "Score"
            cfg.moderation = "limma"
            cfg.log_transform_before_stats = False
            cfg.covariates = covs
            return run_moderated_linear_model(log_data, meta, cfg)

        planted = feats[:n_planted]
        adj = _run(["Age"]).set_index("Protein").loc[planted, "logFC"].mean()
        unadj = _run([]).set_index("Protein").loc[planted, "logFC"].mean()
        # Unadjusted slope is inflated by ~age_coef * d(age)/d(score) = 0.02 * 30 = 0.6.
        assert unadj > true_slope + 0.25, f"unadjusted slope {unadj:.3f} not inflated"
        # Adjusting for Age recovers the planted slope.
        assert abs(adj - true_slope) < 0.2, f"adjusted slope {adj:.3f} != {true_slope}"

    def test_linear_trend_covariates_compose_with_intensity_trend(self):
        log_data, raw, meta, config, planted = _make_linear_trend_fixture()
        config._raw_feature_data = raw
        config.subject_column = None
        meta = meta.copy()
        meta["Age"] = np.random.default_rng(3).normal(60.0, 8.0, size=len(meta))
        config.covariates = ["Age"]
        result = run_moderated_linear_model(log_data, meta, config)
        for col in ("intensity_s0_sq", "intensity_used", "logFC", "P.Value"):
            assert col in result.columns, f"missing column {col!r}"
        # Planted slope still recovered with the (uninformative) covariate present.
        med = result.set_index("Protein").loc[planted, "logFC"].median()
        assert abs(med - 0.05) < 0.02

    def test_linear_trend_covariate_listwise_deletion(self):
        log_data, raw, meta, config, _ = _make_linear_trend_fixture()
        config._raw_feature_data = raw
        config.subject_column = None  # 50 samples, no subject block
        meta = meta.copy()
        meta["Age"] = np.random.default_rng(5).normal(60.0, 8.0, size=len(meta))
        meta.loc[meta.index[:3], "Age"] = np.nan  # 3 samples lack the covariate
        config.covariates = ["Age"]
        result = run_moderated_linear_model(log_data, meta, config)
        # n_group1 + n_group2 == samples used per feature; 50 - 3 = 47.
        assert int((result["n_group1"] + result["n_group2"]).max()) == 47


def _make_interaction_fixture(n_per_cell=4, n_features=200, n_planted=10, effect=1.5, seed=7):
    """Build a 2 x 2 (Group x Visit) fixture for moderated interaction tests.

    The first ``n_planted`` features carry a pure interaction (+``effect`` only in
    the Treatment/Post cell). The next ``n_planted`` carry large Group and Visit
    main effects but no interaction, so a correct interaction test must leave them
    null. Each subject is measured at both visits, nested within one group.

    Returns
    -------
    log_data : pd.DataFrame
        Log2 feature intensities (rows = features, columns = samples).
    metadata_df : pd.DataFrame
        Sample / Subject / Group / Visit metadata.
    config : StatisticalConfig
        Pre-populated for a limma-moderated interaction fit on log data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for group in ("Control", "Treatment"):
        for i in range(n_per_cell):
            subject = f"{group[0]}{i}"
            for visit in ("Pre", "Post"):
                rows.append({"Sample": f"{subject}_{visit}", "Subject": subject, "Group": group, "Visit": visit})
    metadata_df = pd.DataFrame(rows)
    is_treat = metadata_df["Group"].eq("Treatment").to_numpy(dtype=float)
    is_post = metadata_df["Visit"].eq("Post").to_numpy(dtype=float)

    values = rng.normal(10.0, 0.3, size=(n_features, len(metadata_df)))
    values[:n_planted] += effect * (is_treat * is_post)
    values[n_planted : 2 * n_planted] += 2.0 * is_treat + 1.5 * is_post
    features = [f"P{i:04d}" for i in range(n_features)]
    log_data = pd.DataFrame(values, index=features, columns=metadata_df["Sample"])

    config = StatisticalConfig()
    config.analysis_type = "interaction"
    config.statistical_test_method = "moderated_linear_model"
    config.moderation = "limma"
    config.group_column = "Group"
    config.group_labels = ["Control", "Treatment"]
    config.paired_column = "Visit"
    config.paired_label1 = "Pre"
    config.paired_label2 = "Post"
    config.log_transform_before_stats = False
    return log_data, metadata_df, config


def _difference_of_differences(log_data, metadata_df):
    cell = metadata_df.set_index("Sample")[["Group", "Visit"]].loc[log_data.columns]
    means = log_data.T.groupby([cell["Group"], cell["Visit"]]).mean().T
    return (means[("Treatment", "Post")] - means[("Control", "Post")]) - (
        means[("Treatment", "Pre")] - means[("Control", "Pre")]
    )


class TestModeratedInteraction:
    """2 x 2 difference-of-differences test in the moderated linear model."""

    def test_logfc_equals_difference_of_cell_mean_differences(self):
        log_data, metadata_df, config = _make_interaction_fixture()
        result = run_moderated_linear_model(log_data, metadata_df, config).set_index("Protein")
        expected = _difference_of_differences(log_data, metadata_df)
        np.testing.assert_allclose(result.loc[expected.index, "logFC"], expected, atol=1e-10)

    def test_residual_df_is_samples_minus_four_cells(self):
        log_data, metadata_df, config = _make_interaction_fixture()
        result = run_moderated_linear_model(log_data, metadata_df, config)
        assert (result["residual_df"] == len(metadata_df) - 4).all()

    def test_planted_interaction_detected_and_main_effects_ignored(self):
        log_data, metadata_df, config = _make_interaction_fixture()
        result = run_moderated_linear_model(log_data, metadata_df, config).set_index("Protein")
        planted = [f"P{i:04d}" for i in range(10)]
        main_only = [f"P{i:04d}" for i in range(10, 20)]
        top = set(result.sort_values("P.Value").head(10).index)
        assert len(top & set(planted)) >= 9
        # Large main effects without interaction must not leak into the contrast.
        assert result.loc[main_only, "logFC"].abs().max() < 0.75
        assert result.loc[main_only, "P.Value"].min() > 1e-3

    def test_nested_subject_block_keeps_estimate_and_drops_aliased_column(self):
        log_data, metadata_df, config = _make_interaction_fixture()
        without_subjects = run_moderated_linear_model(log_data, metadata_df, config).set_index("Protein")
        config.subject_column = "Subject"
        with_subjects = run_moderated_linear_model(log_data, metadata_df, config).set_index("Protein")
        # Balanced design: the subject block changes the variance, not the estimate.
        np.testing.assert_allclose(with_subjects["logFC"], without_subjects["logFC"], atol=1e-10)
        # 8 subjects nested in 2 groups: 8 subject means + 1 visit + 1 interaction = 10 params.
        assert (with_subjects["residual_df"] == len(metadata_df) - 10).all()

    def test_intensity_trend_prior_uses_one_group_per_cell(self):
        log_data, metadata_df, config = _make_interaction_fixture()
        config.moderation = "intensity_trend"
        config._raw_feature_data = np.exp2(log_data)
        result = run_moderated_linear_model(log_data, metadata_df, config)
        points = get_intensity_trend_points(result)
        assert points["group"].nunique() == 4
        assert {"intensity_s0_sq", "intensity_used"} <= set(result.columns)

    def test_missing_cell_raises(self):
        log_data, metadata_df, config = _make_interaction_fixture()
        keep = ~(metadata_df["Group"].eq("Treatment") & metadata_df["Visit"].eq("Post"))
        metadata_df = metadata_df.loc[keep]
        with pytest.raises(ValueError, match="all four"):
            run_moderated_linear_model(log_data[metadata_df["Sample"]], metadata_df, config)

    def test_confounded_covariate_raises_not_estimable(self):
        log_data, metadata_df, config = _make_interaction_fixture()
        metadata_df = metadata_df.assign(
            TreatPost=(metadata_df["Group"].eq("Treatment") & metadata_df["Visit"].eq("Post")).astype(float)
        )
        config.covariates = ["TreatPost"]
        with pytest.raises(ValueError, match="not estimable"):
            run_moderated_linear_model(log_data, metadata_df, config)

    def test_validate_accepts_moderated_interaction_without_formula_terms(self):
        _, _, config = _make_interaction_fixture()
        assert config.interaction_terms == []
        assert config.validate()

    def test_validate_requires_paired_labels_for_moderated_interaction(self):
        _, _, config = _make_interaction_fixture()
        config.paired_label2 = None
        with pytest.raises(ValueError, match="paired_label1 and paired_label2"):
            config.validate()

    def test_validate_still_requires_formula_terms_for_mixed_effects_interaction(self):
        _, _, config = _make_interaction_fixture()
        config.statistical_test_method = "mixed_effects"
        config.subject_column = "Subject"
        with pytest.raises(ValueError, match="interaction_terms"):
            config.validate()

    def test_dispatcher_runs_interaction_with_reference_pool_prior(self):
        log_data, metadata_df, config = _make_interaction_fixture()
        rng = np.random.default_rng(3)
        pools = [f"Pool_{i}" for i in range(3)]
        pool_values = pd.DataFrame(rng.normal(10.0, 0.1, size=(len(log_data), 3)), index=log_data.index, columns=pools)
        raw = np.exp2(pd.concat([log_data, pool_values], axis=1))
        # The dispatcher reads samples from column 6 onward (5 standard annotation columns).
        annotation_columns = ["Protein", "Description", "Protein Gene", "UniProt_Accession", "UniProt_Entry_Name"]
        annotations = pd.DataFrame({column: raw.index for column in annotation_columns})
        normalized_data = pd.concat([annotations, raw.reset_index(drop=True)], axis=1)
        sample_metadata = {
            row.Sample: {"Group": row.Group, "Visit": row.Visit, "sample_type": "experimental"}
            for row in metadata_df.itertuples()
        }
        sample_metadata.update({p: {"Group": np.nan, "Visit": np.nan, "sample_type": "reference"} for p in pools})
        config.moderation = "intensity_trend"
        config.log_transform_before_stats = "auto"
        config.variance_prior_group_column = "sample_type"
        config.variance_prior_groups = ["reference"]
        config.validate()

        result = run_comprehensive_statistical_analysis(normalized_data, sample_metadata, config).set_index("Protein")

        expected = _difference_of_differences(log_data, metadata_df)
        # The 'auto' log transform adds a small pseudocount, so agreement is close, not exact.
        np.testing.assert_allclose(result.loc[expected.index, "logFC"], expected, atol=0.02)
        assert (result["residual_df"] == len(metadata_df) - 4).all()
        assert "adj.P.Val" in result.columns


class TestModerationOptionValidation:
    def test_invalid_moderation_raises(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "bogus"
        with pytest.raises(ValueError, match="moderation must be one of"):
            run_moderated_linear_model(feature_data, metadata_df, config)


class TestVariancePriorGroupColumn:
    """Tests for the ``variance_prior_group_column`` option that takes the
    intensity_trend LOWESS SHAPE from a separate sample pool (typically
    dedicated technical-replicate classes) instead of the design groups.

    The pools trace how variance depends on intensity without any biology in
    the way. They do not say how large the variance is for the design being
    tested, which is why the trend's level is fitted to the design residuals
    afterwards; see TestTrendCalibration.
    """

    @staticmethod
    def _fixture(seed=42):
        """Linear-trend fixture with extra QC samples whose technical
        variance is much smaller than the design's between-subject spread.

        Returns
        -------
        log_data, raw_data : DataFrames
            Feature data on log and raw scales, including both study and
            QC sample columns.
        metadata_df : DataFrame
            Long-form metadata including a ``QC_Category`` column. Study
            samples have a finite Week; QC samples have NaN Week so they
            are naturally excluded from the design fit.
        config : StatisticalConfig
            Pre-populated for linear_trend; caller toggles
            variance_prior_group_column on/off.
        """
        rng = np.random.default_rng(seed)
        # 8 subjects x 5 weeks = 40 study samples.
        subjects = [f"S{i:02d}" for i in range(8)]
        weeks = [0.0, 2.0, 4.0, 6.0, 12.0]
        study_samples = [f"{s}_W{int(w)}" for s in subjects for w in weeks]
        study_week = np.array([w for _ in subjects for w in weeks], dtype=float)
        study_subj = np.array([s for s in subjects for _ in weeks])
        # Large between-subject biology so the design within-group SD is
        # noticeably larger than the QC technical SD.
        subj_intercepts = rng.normal(0.0, 0.6, size=len(subjects))
        subj_per_sample = np.repeat(subj_intercepts, len(weeks))

        # 6 BatchQC and 6 BatchRef samples sharing one "subject" each but
        # with low technical noise (sigma = 0.1 in log space) -- the
        # technical-replicate noise floor.
        qc_samples = [f"BatchQC_{i:02d}" for i in range(6)] + [
            f"BatchRef_{i:02d}" for i in range(6)
        ]
        qc_class = ["BatchQC"] * 6 + ["BatchRef"] * 6

        n_features = 150
        n_planted = 15
        all_samples = study_samples + qc_samples
        n_total = len(all_samples)

        log_values = np.empty((n_features, n_total))
        # Study columns: noise + subject intercept + (planted slope x week)
        study_block = rng.normal(loc=10.0, scale=0.3, size=(n_features, len(study_samples)))
        study_block += subj_per_sample[np.newaxis, :]
        study_block[:n_planted, :] += 0.05 * study_week[np.newaxis, :]
        log_values[:, : len(study_samples)] = study_block
        # QC columns: same baseline but only tight technical noise
        qc_block = rng.normal(loc=10.0, scale=0.1, size=(n_features, len(qc_samples)))
        log_values[:, len(study_samples):] = qc_block

        features = [f"P{i:04d}" for i in range(n_features)]
        log_data = pd.DataFrame(log_values, index=features, columns=all_samples)
        raw_data = pd.DataFrame(np.exp(log_values * np.log(2.0)), index=features, columns=all_samples)

        metadata_df = pd.DataFrame(
            {
                "Sample": all_samples,
                "Subject": list(study_subj) + qc_samples,
                "Week": list(study_week) + [np.nan] * len(qc_samples),
                "QC_Category": ["Study"] * len(study_samples) + qc_class,
            }
        )

        config = StatisticalConfig()
        config.analysis_type = "linear_trend"
        config.statistical_test_method = "moderated_linear_model"
        config.time_column = "Week"
        config.subject_column = "Subject"
        config.log_transform_before_stats = False
        config.moderation = "intensity_trend"
        config._raw_feature_data = raw_data
        return log_data, raw_data, metadata_df, config

    def test_prior_level_is_fitted_to_the_design_whichever_source(self):
        """The two sources are wrong in opposite directions, and the calibration
        corrects both.

        In this fixture the within-subject residual is the 0.3 study noise. The
        QC pools carry only 0.1 noise, so their trend sits far BELOW the
        residuals. The design groups are the five weeks, and each spans all
        eight subjects, whose 0.6 intercepts the subject block removes, so their
        trend sits far ABOVE. Until v26.8.0 each was used at its own level. The
        QC prior then shrank every residual toward technical noise. This test
        once asserted exactly that, as the option's headline behavior: smaller
        posterior variances and larger |t|.
        """
        log_data, _, meta, config = self._fixture()
        res_default = run_moderated_linear_model(log_data, meta, config)
        config.variance_prior_group_column = "QC_Category"
        config.variance_prior_groups = ["BatchQC", "BatchRef"]
        res_qc = run_moderated_linear_model(log_data, meta, config)

        assert res_default["intensity_trend_level"].iloc[0] < 0.5  # design groups over-state
        assert res_qc["intensity_trend_level"].iloc[0] > 4.0  # technical pools under-state

        # After calibration both priors sit at the residuals' own level, so the
        # source no longer decides how much every test is shrunk.
        median_resid = np.nanmedian(res_default["residual_s2"])
        for res in (res_default, res_qc):
            assert 0.7 < np.nanmedian(res["intensity_s0_sq"]) / median_resid < 1.4
            assert 0.7 < np.nanmedian(res["posterior_s2"]) / median_resid < 1.4
        ratio_t = np.nanmedian(np.abs(res_qc["t"])) / np.nanmedian(np.abs(res_default["t"]))
        assert 0.9 < ratio_t < 1.1

    def test_default_unchanged_when_option_not_set(self):
        """Backward compatibility: leaving variance_prior_group_column as
        None must reproduce the historical design-group prior bit-for-bit."""
        log_data, _, meta, config = self._fixture()
        res_a = run_moderated_linear_model(log_data, meta, config)
        # Explicitly set both override knobs to None and re-run.
        config.variance_prior_group_column = None
        config.variance_prior_groups = None
        res_b = run_moderated_linear_model(log_data, meta, config)
        # Drop the trend-points attrs payload before comparing frames so
        # the _AttrsPayload identity equality doesn't trip pandas.equals.
        for r in (res_a, res_b):
            r.attrs.pop("intensity_trend_points", None)
        pd.testing.assert_frame_equal(res_a, res_b)

    def test_unknown_prior_column_raises(self):
        log_data, _, meta, config = self._fixture()
        config.variance_prior_group_column = "NotAColumn"
        with pytest.raises(ValueError, match="variance_prior_group_column"):
            run_moderated_linear_model(log_data, meta, config)

    def test_prior_groups_restriction_is_honored(self):
        """When variance_prior_groups is set, the prior cloud only sees
        rows whose prior-column value is in the whitelist."""
        log_data, _, meta, config = self._fixture()
        config.variance_prior_group_column = "QC_Category"
        config.variance_prior_groups = ["BatchQC"]  # exclude BatchRef
        result = run_moderated_linear_model(log_data, meta, config)
        pts = get_intensity_trend_points(result)
        assert set(pts["group"].unique()) == {"BatchQC"}

    def test_works_in_paired_analysis_type(self):
        """Same option, paired design. Build a minimal paired fixture with
        QC samples missing Timepoint and confirm both prior modes run."""
        rng = np.random.default_rng(0)
        n_subj, n_feat = 10, 80
        subjects = [f"S{i:02d}" for i in range(n_subj)]
        study_samples = [f"{s}_T{t}" for s in subjects for t in (1, 2)]
        subj_intercept = np.repeat(rng.normal(0, 0.6, size=n_subj), 2)
        timepoint = np.tile([1.0, 2.0], n_subj)

        log_values = rng.normal(loc=10.0, scale=0.3, size=(n_feat, len(study_samples)))
        log_values += subj_intercept[np.newaxis, :]
        log_values[:10, :] += 0.6 * (timepoint - 1.0)[np.newaxis, :]

        qc_samples = [f"QC_{i:02d}" for i in range(8)]
        qc_block = rng.normal(loc=10.0, scale=0.1, size=(n_feat, len(qc_samples)))

        all_samples = study_samples + qc_samples
        all_values = np.column_stack([log_values, qc_block])
        features = [f"P{i:04d}" for i in range(n_feat)]
        log_data = pd.DataFrame(all_values, index=features, columns=all_samples)
        raw_data = pd.DataFrame(np.exp(all_values * np.log(2.0)), index=features, columns=all_samples)

        meta = pd.DataFrame(
            {
                "Sample": all_samples,
                "Subject": list(np.repeat(subjects, 2)) + qc_samples,
                "Timepoint": list(timepoint) + [np.nan] * len(qc_samples),
                "QC_Category": ["Study"] * len(study_samples) + ["BatchQC"] * len(qc_samples),
            }
        )
        config = StatisticalConfig()
        config.analysis_type = "paired"
        config.statistical_test_method = "moderated_linear_model"
        config.subject_column = "Subject"
        config.paired_column = "Timepoint"
        config.paired_label1 = 1.0
        config.paired_label2 = 2.0
        config.log_transform_before_stats = False
        config.moderation = "intensity_trend"
        config._raw_feature_data = raw_data
        config.variance_prior_group_column = "QC_Category"
        config.variance_prior_groups = ["BatchQC"]

        result = run_moderated_linear_model(log_data, meta, config)
        # Sanity: same number of features, prior column populated.
        assert len(result) == n_feat
        assert result["intensity_s0_sq"].notna().all()
        # Prior cloud must have used only the QC samples.
        pts = get_intensity_trend_points(result)
        assert set(pts["group"].unique()) == {"BatchQC"}


class TestTrendCalibration:
    """The intensity trend's level and the prior df are fitted to the design residuals.

    Calibration is checked where it matters: the share of null p-values below
    0.05. A calibrated test gives about 5%. Before v26.8.0 the trend was used at
    its source's level, with d0 estimated separately around the residuals' global
    mean. On these same simulations that gave:

    | design   | trend source  | null p < 0.05 before | after |
    |----------|---------------|----------------------|-------|
    | unpaired | QC pools      | 40%                  | 4.9%  |
    | unpaired | design groups | 7.4%                 | 5.0%  |
    | paired   | QC pools      | 18.5%                | 4.7%  |
    | paired   | design groups | 0.17%                | 5.1%  |

    (means over five seeds of 2,000 features). The pools under-state the noise
    by all the biology they lack. Small design groups under-state it through
    the log-chi-square bias of a LOWESS on log variance. Design groups under a
    paired model over-state it by the between-subject spread the subject block
    removes, which left the toolkit's default paired test with almost no false
    positives and correspondingly little power.
    """

    @staticmethod
    def _null_fixture(design, seed=0, n_feat=2000, n_subj=6, bio=0.4, tech=0.15, n_qc=8):
        """No true effects. Biology `bio` between people, heteroscedastic technical noise."""
        rng = np.random.default_rng(seed)
        level = rng.uniform(8, 20, size=n_feat)
        tech_f = tech * (1.6 - 0.06 * (level - 8))  # noise falls with abundance
        if design == "unpaired":
            samples = [f"S{j:02d}" for j in range(2 * n_subj)]
            groups = ["A"] * n_subj + ["B"] * n_subj
            subjects = samples
            study = (level[:, None] + rng.normal(0, bio, size=(n_feat, 2 * n_subj))
                     + tech_f[:, None] * rng.normal(size=(n_feat, 2 * n_subj)))
        else:
            samples = [f"P{j:02d}_{t}" for t in ("T1", "T2") for j in range(n_subj)]
            groups = ["T1"] * n_subj + ["T2"] * n_subj
            subjects = [f"P{j:02d}" for _ in range(2) for j in range(n_subj)]
            person = rng.normal(0, bio, size=(n_feat, n_subj))
            # Some biology within a person too, which no pooled injection carries.
            study = (level[:, None] + np.hstack([person, person])
                     + 0.5 * bio * rng.normal(size=(n_feat, 2 * n_subj))
                     + tech_f[:, None] * rng.normal(size=(n_feat, 2 * n_subj)))
        qc = [f"QC{j}" for j in range(n_qc)]
        values = np.hstack([study, level[:, None] + tech_f[:, None] * rng.normal(size=(n_feat, n_qc))])
        features = [f"F{i:05d}" for i in range(n_feat)]
        columns = samples + qc
        log_data = pd.DataFrame(values, index=features, columns=columns)
        meta = pd.DataFrame({
            "Sample": columns,
            "Group": groups + [np.nan] * n_qc,
            "Subject": subjects + qc,
            "Category": ["Study"] * len(samples) + ["Pool"] * n_qc,
        })
        config = StatisticalConfig()
        config.statistical_test_method = "moderated_linear_model"
        config.moderation = "intensity_trend"
        config.log_transform_before_stats = False
        config._raw_feature_data = pd.DataFrame(2.0 ** values, index=features, columns=columns)
        if design == "unpaired":
            config.analysis_type = "unpaired"
            config.group_column = "Group"
            config.group_labels = ["A", "B"]
        else:
            config.analysis_type = "paired"
            config.subject_column = "Subject"
            config.paired_column = config.group_column = "Group"
            config.paired_label1, config.paired_label2 = "T1", "T2"
            config.group_labels = ["T1", "T2"]
        return log_data, meta, config

    @pytest.mark.parametrize("design", ["unpaired", "paired"])
    @pytest.mark.parametrize("source", ["pools", "design groups"])
    def test_null_p_values_are_calibrated(self, design, source):
        log_data, meta, config = self._null_fixture(design)
        if source == "pools":
            config.variance_prior_group_column = "Category"
            config.variance_prior_groups = ["Pool"]
        res = run_moderated_linear_model(log_data, meta, config)
        frac = float((res["P.Value"] < 0.05).mean())
        # 2,000 independent null features: the binomial SD of the share is 0.005,
        # so this is about +/-4 SD around the nominal 5%.
        assert 0.03 < frac < 0.07, f"{design}, trend from {source}: {frac:.1%} of null p < 0.05"

    def test_level_direction_follows_what_the_source_lacks(self):
        """Pools lack biology, so they are scaled UP. Paired design groups carry the
        between-subject spread the model removes, so they are scaled DOWN."""
        log_data, meta, config = self._null_fixture("paired")
        res_groups = run_moderated_linear_model(log_data, meta, config)
        config.variance_prior_group_column = "Category"
        config.variance_prior_groups = ["Pool"]
        res_pools = run_moderated_linear_model(log_data, meta, config)
        assert res_pools["intensity_trend_level"].iloc[0] > 1.5
        assert res_groups["intensity_trend_level"].iloc[0] < 0.6

    def test_prior_used_is_the_level_times_the_shape(self):
        log_data, meta, config = self._null_fixture("unpaired", n_feat=300)
        res = run_moderated_linear_model(log_data, meta, config)
        np.testing.assert_allclose(
            res["intensity_s0_sq"], res["intensity_trend_level"] * res["intensity_trend_shape"], rtol=1e-12)
        # And it is what the posterior was built from.
        d0 = res["posterior_df"] - res["residual_df"]
        expected = (d0 * res["intensity_s0_sq"] + res["residual_df"] * res["residual_s2"]) / res["posterior_df"]
        np.testing.assert_allclose(res["posterior_s2"], expected, rtol=1e-12)

    def test_calibration_recovers_a_known_level_and_prior_df(self):
        """Draw true variances from the prior the model assumes - level * shape * d0 / chi2(d0) - and
        sample variances around them. The fit must return that level and d0."""
        rng = np.random.default_rng(11)
        n, d, d0_true, level_true = 20000, 6.0, 8.0, 2.5
        shape = np.exp(rng.uniform(-4, 0, size=n))  # an arbitrary per-feature trend
        sigma2 = level_true * shape * d0_true / rng.chisquare(d0_true, size=n)
        s2 = sigma2 * rng.chisquare(d, size=n) / d
        fit = {"s2": s2, "df": np.full(n, d)}
        level, d0 = _calibrate_trend_to_design(fit, shape)
        assert level == pytest.approx(level_true, rel=0.05)
        assert d0 == pytest.approx(d0_true, rel=0.15)
        # It IS the limma prior fit on the ratio, nothing more.
        assert (level, d0) == _fit_limma_prior(s2 / shape, np.full(n, d))


class TestLimmaPriorRobust:
    def test_robust_tightens_prior_df_when_outliers_present(self):
        # Outliers inflate the variance of log(s^2), pushing d0 toward 0
        # under the plain estimator. Robust Winsorization (median/MAD-based
        # threshold) should keep d0 near the true prior df.
        rng = np.random.default_rng(7)
        n = 400
        # Generate from a true (s0^2, d0) = (0.04, 20) inverse chi-square so
        # the plain estimator targets d0 = 20.
        s2 = 0.04 * 20.0 / rng.chisquare(df=20.0, size=n)
        # Inject 8 extreme outliers
        s2[:8] = s2[:8] * 200
        d = np.full(n, 10.0)

        _, d0_plain = _fit_limma_prior(s2, d, robust=False)
        _, d0_robust = _fit_limma_prior(s2, d, robust=True)
        # Outliers inflate e_var under plain fit, so d0_plain is pulled
        # low. Robust fit should give a larger d0 (stronger prior).
        assert d0_robust > d0_plain


class TestMixedEffectsProteinNameLookup:
    """Regression tests for the protein_annotations lookup in run_mixed_effects_analysis.

    v26.2.0 introduced an index reassignment in run_comprehensive_statistical_analysis
    that made filtered_protein_data.index hold protein IDs, while protein_annotations
    retained its original integer RangeIndex. That broke the
    ``protein_annotations.loc[protein_idx, "Protein"]`` lookup with a KeyError when
    annotations were provided.
    """

    @staticmethod
    def _make_longitudinal_fixture():
        rng = np.random.default_rng(0)
        subjects = ["S1", "S2", "S3", "S4", "S5", "S6"]
        weeks = [0, 4, 8]
        samples, rows = [], []
        for s in subjects:
            for w in weeks:
                name = f"{s}_W{w}"
                samples.append(name)
                rows.append({"Sample": name, "BRI Subject ID": s, "Week": w})
        metadata_df = pd.DataFrame(rows)

        proteins = [f"sp|P{idx:04d}|PROT{idx:04d}_HUMAN" for idx in range(5)]
        values = rng.uniform(1e5, 1e7, size=(len(proteins), len(samples)))
        protein_values = pd.DataFrame(values, columns=samples)
        annotations = pd.DataFrame(
            {
                "Protein": proteins,
                "Description": [f"desc {p}" for p in proteins],
                "Protein Gene": [f"GENE{i}" for i in range(len(proteins))],
            }
        )
        normalized_data = pd.concat([annotations.reset_index(drop=True), protein_values.reset_index(drop=True)], axis=1)

        config = StatisticalConfig()
        config.analysis_type = "dose_response"
        config.statistical_test_method = "mixed_effects"
        config.dose_column = "Week"
        config.subject_column = "BRI Subject ID"
        config.log_transform_before_stats = False
        config.correction_method = "fdr_bh"

        sample_metadata = {
            row["Sample"]: {k: row[k] for k in ("BRI Subject ID", "Week")} for _, row in metadata_df.iterrows()
        }

        return normalized_data, sample_metadata, config, annotations, proteins

    def test_dose_response_with_annotations_does_not_keyerror(self):
        normalized_data, sample_metadata, config, annotations, expected_proteins = self._make_longitudinal_fixture()

        results = run_comprehensive_statistical_analysis(
            normalized_data=normalized_data,
            sample_metadata=sample_metadata,
            config=config,
            protein_annotations=annotations,
        )

        assert "Protein" in results.columns
        assert set(results["Protein"]) == set(expected_proteins)

    def test_direct_call_with_integer_indexed_annotations(self):
        rng = np.random.default_rng(1)
        subjects = ["S1", "S2", "S3", "S4"]
        weeks = [0, 4]
        samples, rows = [], []
        for s in subjects:
            for w in weeks:
                name = f"{s}_W{w}"
                samples.append(name)
                rows.append({"Sample": name, "Subject": s, "Week": w})
        metadata_df = pd.DataFrame(rows)

        proteins = [f"sp|Q{idx:04d}|P{idx:04d}_HUMAN" for idx in range(3)]
        values = rng.uniform(1e5, 1e7, size=(len(proteins), len(samples)))
        protein_data = pd.DataFrame(values, index=proteins, columns=samples)
        annotations = pd.DataFrame({"Protein": proteins})

        config = StatisticalConfig()
        config.analysis_type = "dose_response"
        config.statistical_test_method = "mixed_effects"
        config.dose_column = "Week"
        config.subject_column = "Subject"
        config.log_transform_before_stats = False

        results = run_mixed_effects_analysis(protein_data, metadata_df, config, annotations)

        assert set(results["Protein"]) == set(proteins)
