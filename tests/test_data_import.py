"""Tests for the data_import module."""

import pandas as pd
import pytest

from proteomics_toolkit.data_import import (
    BATCH_SUFFIX_DELIMITER,
    clean_description,
    clean_sample_names,
    create_sample_column_mapping,
    detect_batch_suffix,
    identify_sample_columns,
    load_skyline_data,
    parse_gene_from_description,
    parse_uniprot_identifier,
    strip_batch_suffix,
)


# ---------------------------------------------------------------------------
# parse_uniprot_identifier
# ---------------------------------------------------------------------------


class TestParseUniprotIdentifier:
    def test_swissprot_format(self):
        result = parse_uniprot_identifier("sp|P12345|ALBU_HUMAN")
        assert result["accession"] == "P12345"
        assert result["database"] == "SwissProt"
        assert result["entry_name"] == "ALBU_HUMAN"

    def test_trembl_format(self):
        result = parse_uniprot_identifier("tr|Q9ABC1|Q9ABC1_MOUSE")
        assert result["accession"] == "Q9ABC1"
        assert result["database"] == "TrEMBL"

    def test_bare_accession_falls_back_to_regex(self):
        result = parse_uniprot_identifier("P12345")
        assert result["accession"] == "P12345"
        assert result["database"] == ""

    def test_nan_returns_empty(self):
        result = parse_uniprot_identifier(float("nan"))
        assert result["accession"] == ""

    def test_unrecognized_returns_empty(self):
        result = parse_uniprot_identifier("some random text 123")
        assert result["accession"] == ""


# ---------------------------------------------------------------------------
# parse_gene_from_description / clean_description
# ---------------------------------------------------------------------------


class TestDescriptionParsing:
    def test_parse_gene_from_gn_field(self):
        desc = "Albumin OS=Homo sapiens OX=9606 GN=ALB PE=1 SV=2"
        assert parse_gene_from_description(desc) == "ALB"

    def test_parse_gene_returns_empty_when_missing(self):
        assert parse_gene_from_description("No gene here") == ""

    def test_parse_gene_nan_returns_empty(self):
        assert parse_gene_from_description(float("nan")) == ""

    def test_clean_description_strips_uniprot_fields(self):
        desc = "Albumin OS=Homo sapiens OX=9606 GN=ALB PE=1 SV=2"
        cleaned = clean_description(desc)
        assert "OS=" not in cleaned
        assert "GN=" not in cleaned
        assert cleaned.startswith("Albumin")

    def test_clean_description_nan_returns_empty(self):
        assert clean_description(float("nan")) == ""


# ---------------------------------------------------------------------------
# load_skyline_data
# ---------------------------------------------------------------------------


class TestLoadSkylineData:
    def test_loads_protein_and_metadata(self, tmp_csv_files):
        protein_data, metadata, peptide_data = load_skyline_data(
            tmp_csv_files["protein_file"], tmp_csv_files["metadata_file"]
        )
        assert isinstance(protein_data, pd.DataFrame)
        assert isinstance(metadata, pd.DataFrame)
        assert peptide_data is None
        assert len(protein_data) == 5
        assert len(metadata) == 6

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_skyline_data(str(tmp_path / "no_such.csv"), str(tmp_path / "also_no.csv"))


# ---------------------------------------------------------------------------
# Batch suffix handling
# ---------------------------------------------------------------------------


class TestBatchSuffix:
    def test_detect_batch_suffix_single_batch(self):
        cols = ["Sample1__@__Batch1", "Sample2__@__Batch1"]
        suffix = detect_batch_suffix(cols)
        assert suffix == "__@__Batch1"

    def test_detect_batch_suffix_no_delimiter(self):
        cols = ["Sample1", "Sample2"]
        assert detect_batch_suffix(cols) is None

    def test_detect_batch_suffix_multiple_batches_returns_none(self):
        cols = ["Sample1__@__Batch1", "Sample2__@__Batch2"]
        assert detect_batch_suffix(cols) is None

    def test_strip_batch_suffix_removes_suffix(self):
        cols = ["Sample1__@__Batch1", "Sample2__@__Batch1"]
        mapping = strip_batch_suffix(cols)
        assert mapping["Sample1__@__Batch1"] == "Sample1"
        assert mapping["Sample2__@__Batch1"] == "Sample2"


# ---------------------------------------------------------------------------
# identify_sample_columns / clean_sample_names / create_sample_column_mapping
# ---------------------------------------------------------------------------


class TestSampleColumnHelpers:
    def test_identify_sample_columns(self, raw_protein_data, sample_metadata):
        cols = identify_sample_columns(raw_protein_data, sample_metadata)
        assert len(cols) == 6
        assert "Sample_A1" in cols

    def test_clean_sample_names_returns_dict(self, sample_columns):
        result = clean_sample_names(sample_columns)
        assert isinstance(result, dict)

    def test_create_sample_column_mapping(self, sample_columns):
        # data_columns, metadata_sample_names
        mapping = create_sample_column_mapping(sample_columns, sample_columns)
        assert isinstance(mapping, dict)
