"""Tests for the statistical_analysis module."""

import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.statistical_analysis import (
    StatisticalConfig,
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


class TestModerationOptionValidation:
    def test_invalid_moderation_raises(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "bogus"
        with pytest.raises(ValueError, match="moderation must be one of"):
            run_moderated_linear_model(feature_data, metadata_df, config)


class TestVariancePriorGroupColumn:
    """Tests for the ``variance_prior_group_column`` option that sources the
    intensity_trend LOWESS from a separate sample pool (typically dedicated
    technical-replicate classes) instead of the design groups.

    Why this matters: when the design groups are not nominal replicates
    (e.g. a paired pre/post analysis where each timepoint group still spans
    many subjects), the design within-group SD bakes in inter-subject
    biology that the prior should not be calibrating against. Pointing the
    prior at QC / reference samples gives a cleaner technical-noise floor
    and a less conservative test.
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

    def test_qc_prior_produces_lower_s0_sq_than_design_prior(self):
        """The headline behaviour: with QC samples that have lower technical
        variance than the design within-group spread, the QC-sourced prior
        should yield a smaller intensity_s0_sq, which propagates to smaller
        posterior variances and larger |t|."""
        log_data, _, meta, config = self._fixture()

        # Default: design within-group SD (one group per unique week)
        res_default = run_moderated_linear_model(log_data, meta, config)

        # New option: prior from QC samples only
        config.variance_prior_group_column = "QC_Category"
        config.variance_prior_groups = ["BatchQC", "BatchRef"]
        res_qc = run_moderated_linear_model(log_data, meta, config)

        assert np.nanmedian(res_qc["intensity_s0_sq"]) < np.nanmedian(res_default["intensity_s0_sq"])
        assert np.nanmedian(res_qc["posterior_s2"]) < np.nanmedian(res_default["posterior_s2"])
        assert np.nanmedian(np.abs(res_qc["t"])) > np.nanmedian(np.abs(res_default["t"]))

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
