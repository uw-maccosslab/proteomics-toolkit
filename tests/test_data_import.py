"""Tests for the data_import module."""

import pandas as pd
import pytest

from proteomics_toolkit.data_import import (
    clean_description,
    clean_sample_names,
    create_sample_column_mapping,
    detect_batch_suffix,
    identify_sample_columns,
    load_diann_data,
    load_fasta_sequences,
    load_prism_peptide_data,
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


# ---------------------------------------------------------------------------
# load_diann_data
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_diann_pg_matrix(tmp_path):
    """Write a tiny DIA-NN-style pg_matrix.tsv for testing."""
    df = pd.DataFrame(
        {
            "Protein.Group": ["P12345", "P67890", "Q11111"],
            "Protein.Ids": ["P12345", "P67890;P67891", "Q11111"],
            "Protein.Names": ["ALBU_HUMAN", "TRFE_HUMAN;TRFE_MOUSE", "Q11111_HUMAN"],
            "Genes": ["ALB", "TF;TF", ""],
            "First.Protein.Description": [
                "Albumin",
                "Serotransferrin",
                "Uncharacterized protein",
            ],
            "/data/sampleA.raw.mzML": [1.0e6, 2.0e6, 3.0e6],
            "/data/sampleB.raw.mzML": [1.5e6, 2.5e6, 3.5e6],
            "/data/sampleC.raw.mzML": [1.2e6, 2.2e6, 3.2e6],
        }
    )
    path = tmp_path / "report.pg_matrix.tsv"
    df.to_csv(path, sep="\t", index=False)
    return str(path)


class TestLoadDiannData:
    def test_loads_tsv(self, tmp_diann_pg_matrix):
        protein_data, metadata, sample_cols = load_diann_data(tmp_diann_pg_matrix)
        assert isinstance(protein_data, pd.DataFrame)
        assert metadata is None
        assert len(sample_cols) == 3

    def test_sample_columns_are_file_paths(self, tmp_diann_pg_matrix):
        _, _, sample_cols = load_diann_data(tmp_diann_pg_matrix)
        assert all(col.startswith("/data/") for col in sample_cols)

    def test_standardized_annotation_prefix(self, tmp_diann_pg_matrix):
        protein_data, _, _ = load_diann_data(tmp_diann_pg_matrix)
        expected = ["Protein", "Description", "Protein Gene", "UniProt_Accession", "UniProt_Entry_Name"]
        assert list(protein_data.columns[:5]) == expected

    def test_multi_id_fields_take_first(self, tmp_diann_pg_matrix):
        protein_data, _, _ = load_diann_data(tmp_diann_pg_matrix)
        # Second row had "P67890;P67891" in Protein.Ids
        assert protein_data.loc[1, "UniProt_Accession"] == "P67890"
        # Second row had "TRFE_HUMAN;TRFE_MOUSE" in Protein.Names
        assert protein_data.loc[1, "UniProt_Entry_Name"] == "TRFE_HUMAN"
        # Second row had "TF;TF" in Genes
        assert protein_data.loc[1, "Protein Gene"] == "TF"

    def test_original_diann_columns_preserved(self, tmp_diann_pg_matrix):
        protein_data, _, _ = load_diann_data(tmp_diann_pg_matrix)
        for col in ("Protein.Group", "Protein.Ids", "Genes"):
            assert col in protein_data.columns

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_diann_data(str(tmp_path / "does_not_exist.tsv"))

    def test_loads_with_metadata(self, tmp_diann_pg_matrix, tmp_path):
        metadata_df = pd.DataFrame({"Replicate": ["A", "B", "C"], "Group": ["X", "X", "Y"]})
        meta_path = tmp_path / "meta.csv"
        metadata_df.to_csv(meta_path, index=False)
        _, metadata, _ = load_diann_data(tmp_diann_pg_matrix, metadata_file=str(meta_path))
        assert metadata is not None
        assert len(metadata) == 3


# ---------------------------------------------------------------------------
# load_prism_peptide_data
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_prism_peptide_parquet(tmp_path):
    """Write a minimal PRISM-style peptide parquet for testing."""
    df = pd.DataFrame(
        {
            "peptide_sequence": ["PEPTIDEA", "PEPTIDEB", "PEPTIDEC"],
            "leading_protein": ["sp|P12345|ALBU_HUMAN", "sp|P67890|TRFE_HUMAN", "sp|P12345|ALBU_HUMAN"],
            "Sample1__@__Batch1": [1.0e5, 2.0e5, 3.0e5],
            "Sample2__@__Batch1": [1.1e5, 2.1e5, 3.1e5],
        }
    )
    path = tmp_path / "corrected_peptides.parquet"
    df.to_parquet(path, index=False)
    return str(path)


class TestLoadPrismPeptideData:
    def test_loads_parquet(self, tmp_prism_peptide_parquet):
        peptide_data, metadata, sample_cols = load_prism_peptide_data(tmp_prism_peptide_parquet)
        assert isinstance(peptide_data, pd.DataFrame)
        assert metadata is None
        assert len(sample_cols) == 2
        assert len(peptide_data) == 3

    def test_preserves_peptide_sequence_column(self, tmp_prism_peptide_parquet):
        peptide_data, _, _ = load_prism_peptide_data(tmp_prism_peptide_parquet)
        assert "peptide_sequence" in peptide_data.columns


# ---------------------------------------------------------------------------
# load_fasta_sequences
# ---------------------------------------------------------------------------


class TestLoadFastaSequences:
    def test_parses_uniprot_style_headers(self, tmp_path):
        fasta = tmp_path / "mini.fasta"
        fasta.write_text(
            ">sp|P02768|ALBU_HUMAN Serum albumin OS=Homo sapiens\n"
            "MKWVTFISLLLLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKAL\n"
            "VLIAFAQYLQQCPFEDHVKLVNEVTEFAKT\n"
            ">tr|Q9ABC1|Q9ABC1_MOUSE Hypothetical\n"
            "MSEQUENCEFORAMOUSEPROTEINVAKT\n"
        )
        seqs = load_fasta_sequences(str(fasta))
        assert "P02768" in seqs
        assert "ALBU_HUMAN" in seqs
        assert seqs["P02768"] == seqs["ALBU_HUMAN"]
        assert seqs["P02768"].startswith("MKWVTFISLL")
        # Multi-line sequence concatenates without spaces/newlines
        assert "\n" not in seqs["P02768"]
        # Second record parsed independently
        assert seqs["Q9ABC1"].startswith("MSEQUENCE")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_fasta_sequences(str(tmp_path / "does-not-exist.fasta"))

    def test_non_uniprot_header_uses_first_token(self, tmp_path):
        fasta = tmp_path / "plain.fasta"
        fasta.write_text(">MyProtein extra notes here\nACDEFGHIK\n")
        seqs = load_fasta_sequences(str(fasta))
        assert seqs["MyProtein"] == "ACDEFGHIK"
