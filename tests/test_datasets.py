"""Tests for the bundled example dataset."""

import pandas as pd

from proteomics_toolkit.data_import import BATCH_SUFFIX_DELIMITER
from proteomics_toolkit.datasets import load_example_data, load_example_sequences


class TestLoadExampleData:
    def test_returns_five_objects(self):
        result = load_example_data()
        assert len(result) == 5

    def test_protein_shape(self):
        protein_data, _, _, sample_cols, _ = load_example_data()
        assert isinstance(protein_data, pd.DataFrame)
        assert len(protein_data) == 80
        assert "leading_protein" in protein_data.columns
        assert "leading_description" in protein_data.columns
        assert len(sample_cols) == 12

    def test_peptide_shape(self):
        _, peptide_data, _, _, _ = load_example_data()
        assert len(peptide_data) == 60
        assert "peptide_sequence" in peptide_data.columns
        assert "leading_protein" in peptide_data.columns
        assert "start_position" in peptide_data.columns

    def test_metadata_has_expected_columns(self):
        _, _, metadata, _, _ = load_example_data()
        assert "Replicate" in metadata.columns
        assert "Group" in metadata.columns
        assert len(metadata) == 12
        assert set(metadata["Group"].unique()) == {"Control", "Treatment"}

    def test_sample_columns_use_batch_suffix(self):
        _, _, _, sample_cols, _ = load_example_data()
        assert all(BATCH_SUFFIX_DELIMITER in col for col in sample_cols)

    def test_protein_sequences_bundled(self):
        _, peptide_data, _, _, sequences = load_example_data()
        # Every peptide's parent protein must have a sequence
        for leading_protein in peptide_data["leading_protein"].unique():
            assert leading_protein in sequences, f"Missing sequence for {leading_protein}"
            assert len(sequences[leading_protein]) >= 50

    def test_peptide_sequences_appear_in_parent(self):
        _, peptide_data, _, _, sequences = load_example_data()
        # Every peptide must be a substring of its parent-protein sequence
        # at its recorded start position.
        for _, row in peptide_data.iterrows():
            parent_seq = sequences[row["leading_protein"]]
            start = int(row["start_position"])
            length = int(row["peptide_length"])
            assert parent_seq[start - 1 : start - 1 + length] == row["peptide_sequence"]


class TestLoadExampleSequences:
    def test_accessor_returns_dict(self):
        seqs = load_example_sequences()
        assert isinstance(seqs, dict)
        # At least one known demo protein
        assert "sp|P02768|ALBU_HUMAN" in seqs

    def test_uniprot_accession_alias(self):
        seqs = load_example_sequences()
        # P02768 is an alias for sp|P02768|ALBU_HUMAN
        assert seqs["P02768"] == seqs["sp|P02768|ALBU_HUMAN"]
