"""Tests for the statistical_analysis module."""

import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.statistical_analysis import (
    StatisticalConfig,
    _sanitize_formula_term,
    apply_multiple_testing_correction,
    run_mann_whitney_test,
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
