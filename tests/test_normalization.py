"""Tests for the normalization module."""

import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.normalization import (
    calculate_normalization_stats,
    get_normalization_characteristics,
    is_normalization_log_transformed,
    loess_normalize,
    mad_normalize,
    median_normalize,
    quantile_normalize,
    rlr_normalize,
    vsn_normalize,
    z_score_normalize,
)

SAMPLE_NAMES = [
    "Sample_A1", "Sample_A2", "Sample_A3",
    "Sample_B1", "Sample_B2", "Sample_B3",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_annotation_preserved(original, normalized):
    """Check that the 5 annotation columns survived normalization."""
    annotation_cols = list(original.columns[:5])
    for col in annotation_cols:
        assert col in normalized.columns
        pd.testing.assert_series_equal(
            original[col].reset_index(drop=True),
            normalized[col].reset_index(drop=True),
            check_names=False,
        )


def _assert_no_nans_in_samples(df, sample_cols):
    """Normalized sample values should not be NaN when input had no NaN."""
    for col in sample_cols:
        assert df[col].isna().sum() == 0, f"NaN found in column {col}"


# ---------------------------------------------------------------------------
# get_normalization_characteristics / is_normalization_log_transformed
# ---------------------------------------------------------------------------


class TestNormalizationMetadata:
    def test_characteristics_keys(self):
        chars = get_normalization_characteristics()
        assert "median" in chars
        assert "vsn" in chars
        assert "quantile" in chars

    def test_median_is_not_log_transformed(self):
        assert is_normalization_log_transformed("median") is False

    def test_vsn_is_log_transformed(self):
        assert is_normalization_log_transformed("vsn") is True

    def test_rlr_is_log_transformed(self):
        assert is_normalization_log_transformed("rlr") is True

    def test_unknown_method_returns_false(self):
        assert is_normalization_log_transformed("unknown_method") is False


# ---------------------------------------------------------------------------
# median_normalize
# ---------------------------------------------------------------------------


class TestMedianNormalize:
    def test_output_shape_matches_input(self, standardized_protein_data):
        result = median_normalize(standardized_protein_data, SAMPLE_NAMES)
        assert result.shape == standardized_protein_data.shape

    def test_annotation_columns_preserved(self, standardized_protein_data):
        result = median_normalize(standardized_protein_data, SAMPLE_NAMES)
        _assert_annotation_preserved(standardized_protein_data, result)

    def test_sample_medians_equalized(self, standardized_protein_data):
        result = median_normalize(standardized_protein_data, SAMPLE_NAMES)
        medians = result[SAMPLE_NAMES].median()
        # After median normalization, all sample medians should be close
        assert medians.std() < medians.mean() * 0.01

    def test_no_nans_introduced(self, standardized_protein_data):
        result = median_normalize(standardized_protein_data, SAMPLE_NAMES)
        _assert_no_nans_in_samples(result, SAMPLE_NAMES)


# ---------------------------------------------------------------------------
# quantile_normalize
# ---------------------------------------------------------------------------


class TestQuantileNormalize:
    def test_output_shape(self, standardized_protein_data):
        result = quantile_normalize(standardized_protein_data, SAMPLE_NAMES)
        assert result.shape == standardized_protein_data.shape

    def test_distributions_identical_after_normalization(self, standardized_protein_data):
        result = quantile_normalize(standardized_protein_data, SAMPLE_NAMES)
        sorted_vals = np.sort(result[SAMPLE_NAMES].values, axis=0)
        # Each row of sorted values should be nearly identical across columns
        for row in sorted_vals:
            assert np.std(row) < 1e-6

    def test_annotation_preserved(self, standardized_protein_data):
        result = quantile_normalize(standardized_protein_data, SAMPLE_NAMES)
        _assert_annotation_preserved(standardized_protein_data, result)


# ---------------------------------------------------------------------------
# z_score_normalize
# ---------------------------------------------------------------------------


class TestZScoreNormalize:
    def test_output_shape(self, standardized_protein_data):
        result = z_score_normalize(standardized_protein_data, SAMPLE_NAMES)
        assert result.shape == standardized_protein_data.shape

    def test_annotation_preserved(self, standardized_protein_data):
        result = z_score_normalize(standardized_protein_data, SAMPLE_NAMES)
        _assert_annotation_preserved(standardized_protein_data, result)

    def test_values_changed(self, standardized_protein_data):
        result = z_score_normalize(standardized_protein_data, SAMPLE_NAMES)
        # Normalized values should differ from originals
        assert not np.allclose(
            result[SAMPLE_NAMES].values,
            standardized_protein_data[SAMPLE_NAMES].values,
        )


# ---------------------------------------------------------------------------
# mad_normalize
# ---------------------------------------------------------------------------


class TestMadNormalize:
    def test_output_shape(self, standardized_protein_data):
        result = mad_normalize(standardized_protein_data, SAMPLE_NAMES)
        assert result.shape == standardized_protein_data.shape

    def test_annotation_preserved(self, standardized_protein_data):
        result = mad_normalize(standardized_protein_data, SAMPLE_NAMES)
        _assert_annotation_preserved(standardized_protein_data, result)


# ---------------------------------------------------------------------------
# vsn_normalize
# ---------------------------------------------------------------------------


class TestVsnNormalize:
    def test_output_shape(self, standardized_protein_data):
        result = vsn_normalize(standardized_protein_data, SAMPLE_NAMES)
        assert result.shape == standardized_protein_data.shape

    def test_values_are_transformed(self, standardized_protein_data):
        result = vsn_normalize(standardized_protein_data, SAMPLE_NAMES)
        # VSN uses arcsinh, so values should be much smaller than raw
        assert result[SAMPLE_NAMES].mean().mean() < standardized_protein_data[SAMPLE_NAMES].mean().mean()


# ---------------------------------------------------------------------------
# rlr_normalize
# ---------------------------------------------------------------------------


class TestRlrNormalize:
    def test_output_shape(self, standardized_protein_data):
        result = rlr_normalize(standardized_protein_data, SAMPLE_NAMES)
        assert result.shape == standardized_protein_data.shape

    def test_annotation_preserved(self, standardized_protein_data):
        result = rlr_normalize(standardized_protein_data, SAMPLE_NAMES)
        _assert_annotation_preserved(standardized_protein_data, result)


# ---------------------------------------------------------------------------
# loess_normalize
# ---------------------------------------------------------------------------


class TestLoessNormalize:
    def test_output_shape(self, standardized_protein_data):
        result = loess_normalize(standardized_protein_data, SAMPLE_NAMES)
        assert result.shape == standardized_protein_data.shape

    def test_annotation_preserved(self, standardized_protein_data):
        result = loess_normalize(standardized_protein_data, SAMPLE_NAMES)
        _assert_annotation_preserved(standardized_protein_data, result)


# ---------------------------------------------------------------------------
# calculate_normalization_stats
# ---------------------------------------------------------------------------


class TestCalculateNormalizationStats:
    def test_returns_dict(self, standardized_protein_data):
        normalized = median_normalize(standardized_protein_data, SAMPLE_NAMES)
        # Pass only sample columns to avoid log2 on annotation strings
        stats = calculate_normalization_stats(
            standardized_protein_data[SAMPLE_NAMES],
            normalized[SAMPLE_NAMES],
            method="median",
        )
        assert isinstance(stats, dict)
