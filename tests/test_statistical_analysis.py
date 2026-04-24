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
        # Per-(feature, group) points DataFrame is stashed on attrs
        pts = result.attrs.get("intensity_trend_points")
        assert pts is not None
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


class TestModerationOptionValidation:
    def test_invalid_moderation_raises(self):
        feature_data, metadata_df, config = _make_limma_fixture()
        config.moderation = "bogus"
        with pytest.raises(ValueError, match="moderation must be one of"):
            run_moderated_linear_model(feature_data, metadata_df, config)


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
            row["Sample"]: {k: row[k] for k in ("BRI Subject ID", "Week")}
            for _, row in metadata_df.iterrows()
        }

        return normalized_data, sample_metadata, config, annotations, proteins

    def test_dose_response_with_annotations_does_not_keyerror(self):
        normalized_data, sample_metadata, config, annotations, expected_proteins = (
            self._make_longitudinal_fixture()
        )

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
