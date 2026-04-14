"""Tests for the validation module."""

import pytest

from proteomics_toolkit.validation import (
    ControlSampleError,
    SampleMatchingError,
    validate_metadata_data_consistency,
)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class TestCustomExceptions:
    def test_sample_matching_error_is_exception(self):
        with pytest.raises(SampleMatchingError):
            raise SampleMatchingError("test message")

    def test_control_sample_error_is_exception(self):
        with pytest.raises(ControlSampleError):
            raise ControlSampleError("test message")

    def test_exception_message_preserved(self):
        try:
            raise SampleMatchingError("specific error")
        except SampleMatchingError as e:
            assert "specific error" in str(e)


# ---------------------------------------------------------------------------
# validate_metadata_data_consistency
# ---------------------------------------------------------------------------


class TestValidateMetadataDataConsistency:
    def test_valid_data_passes(self, sample_metadata, sample_columns):
        protein_columns = ["Protein", "Description"] + sample_columns
        result = validate_metadata_data_consistency(
            metadata=sample_metadata,
            metadata_sample_names=sample_columns,
            protein_columns=protein_columns,
            control_column="Group",
            control_labels=["Control"],
            verbose=False,
        )
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_missing_samples_flagged(self, sample_metadata):
        # Protein data only has a subset of samples
        protein_columns = ["Protein", "Sample_A1", "Sample_A2"]
        metadata_names = list(sample_metadata["Replicate"])
        result = validate_metadata_data_consistency(
            metadata=sample_metadata,
            metadata_sample_names=metadata_names,
            protein_columns=protein_columns,
            control_column="Group",
            control_labels=["Control"],
            verbose=False,
        )
        # Should have warnings or errors about missing samples
        assert len(result["warnings"]) > 0 or len(result["errors"]) > 0
