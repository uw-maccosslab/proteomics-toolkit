"""Tests for the bundled example dataset."""

import pandas as pd

from proteomics_toolkit.data_import import BATCH_SUFFIX_DELIMITER
from proteomics_toolkit.datasets import load_example_data


class TestLoadExampleData:
    def test_returns_four_objects(self):
        result = load_example_data()
        assert len(result) == 4

    def test_protein_shape(self):
        protein_data, _, _, sample_cols = load_example_data()
        assert isinstance(protein_data, pd.DataFrame)
        assert len(protein_data) == 80
        # Standard PRISM annotation columns
        assert "leading_protein" in protein_data.columns
        assert "leading_description" in protein_data.columns
        assert len(sample_cols) == 12

    def test_peptide_shape(self):
        _, peptide_data, _, _ = load_example_data()
        assert len(peptide_data) == 60
        assert "peptide_sequence" in peptide_data.columns
        assert "leading_protein" in peptide_data.columns

    def test_metadata_has_expected_columns(self):
        _, _, metadata, _ = load_example_data()
        assert "Replicate" in metadata.columns
        assert "Group" in metadata.columns
        assert len(metadata) == 12
        assert set(metadata["Group"].unique()) == {"Control", "Treatment"}

    def test_sample_columns_use_batch_suffix(self):
        _, _, _, sample_cols = load_example_data()
        assert all(BATCH_SUFFIX_DELIMITER in col for col in sample_cols)
