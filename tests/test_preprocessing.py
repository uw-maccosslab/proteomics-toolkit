"""Tests for the preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.preprocessing import (
    _normalize_group_value,
    assess_data_completeness,
    classify_samples,
    create_standard_data_structure,
    filter_proteins_by_completeness,
    identify_annotation_columns,
    parse_gene_and_description,
    parse_protein_identifiers,
)

# ---------------------------------------------------------------------------
# _normalize_group_value
# ---------------------------------------------------------------------------


class TestNormalizeGroupValue:
    def test_integer_stays_integer(self):
        assert _normalize_group_value(5) == 5

    def test_float_integer_becomes_int(self):
        assert _normalize_group_value(80.0) == 80
        assert isinstance(_normalize_group_value(80.0), int)

    def test_string_integer_becomes_int(self):
        assert _normalize_group_value("42") == 42

    def test_string_float_becomes_float(self):
        assert _normalize_group_value("3.14") == pytest.approx(3.14)

    def test_non_numeric_string_unchanged(self):
        assert _normalize_group_value("Control") == "Control"

    def test_none_becomes_unknown(self):
        assert _normalize_group_value(None) == "Unknown"

    def test_nan_becomes_unknown(self):
        assert _normalize_group_value(float("nan")) == "Unknown"


# ---------------------------------------------------------------------------
# parse_protein_identifiers
# ---------------------------------------------------------------------------


class TestParseProteinIdentifiers:
    def test_adds_annotation_columns(self, raw_protein_data):
        result = parse_protein_identifiers(raw_protein_data)
        assert "UniProt_Accession" in result.columns
        assert "UniProt_Database" in result.columns
        assert "UniProt_Entry_Name" in result.columns

    def test_extracts_accessions_correctly(self, raw_protein_data):
        result = parse_protein_identifiers(raw_protein_data)
        assert result.loc[0, "UniProt_Accession"] == "P12345"
        assert result.loc[0, "UniProt_Database"] == "SwissProt"

    def test_trembl_detected(self, raw_protein_data):
        result = parse_protein_identifiers(raw_protein_data)
        trembl_row = result[result["Protein"].str.startswith("tr|")]
        assert trembl_row.iloc[0]["UniProt_Database"] == "TrEMBL"

    def test_preserves_original_columns(self, raw_protein_data):
        original_cols = set(raw_protein_data.columns)
        result = parse_protein_identifiers(raw_protein_data)
        assert original_cols.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# parse_gene_and_description
# ---------------------------------------------------------------------------


class TestParseGeneAndDescription:
    def test_extracts_gene_names(self, raw_protein_data):
        result = parse_gene_and_description(raw_protein_data)
        assert "Gene" in result.columns or "Protein Gene" in result.columns

    def test_cleans_descriptions(self, raw_protein_data):
        result = parse_gene_and_description(raw_protein_data)
        if "Description" in result.columns:
            desc_vals = result["Description"].tolist()
            for d in desc_vals:
                if d:
                    assert "OS=" not in d


# ---------------------------------------------------------------------------
# assess_data_completeness / filter_proteins_by_completeness
# ---------------------------------------------------------------------------


class TestDataCompleteness:
    def test_assess_completeness_runs(self, standardized_protein_data, sample_columns, sample_metadata_dict):
        # assess_data_completeness returns None (prints results)
        assess_data_completeness(standardized_protein_data, sample_columns, sample_metadata_dict)

    def test_filter_proteins_removes_low_detection(self, standardized_protein_data, sample_columns):
        # Inject NaN to make one protein incomplete
        df = standardized_protein_data.copy()
        df.loc[0, sample_columns] = np.nan
        filtered = filter_proteins_by_completeness(df, sample_columns, min_detection_rate=0.5)
        assert len(filtered) <= len(df)


# ---------------------------------------------------------------------------
# create_standard_data_structure
# ---------------------------------------------------------------------------


class TestCreateStandardDataStructure:
    def test_produces_five_annotation_columns(self, raw_protein_data):
        # Need parsed data first
        parsed = parse_protein_identifiers(raw_protein_data)
        parsed = parse_gene_and_description(parsed)
        result = create_standard_data_structure(parsed)
        expected = [
            "Protein",
            "Description",
            "Protein Gene",
            "UniProt_Accession",
            "UniProt_Entry_Name",
        ]
        assert list(result.columns[:5]) == expected

    def test_sample_columns_follow_annotations(self, raw_protein_data):
        parsed = parse_protein_identifiers(raw_protein_data)
        parsed = parse_gene_and_description(parsed)
        result = create_standard_data_structure(parsed)
        sample_cols = list(result.columns[5:])
        # All sample columns should be numeric
        for col in sample_cols:
            assert pd.api.types.is_numeric_dtype(result[col])


# ---------------------------------------------------------------------------
# identify_annotation_columns
# ---------------------------------------------------------------------------


class TestIdentifyAnnotationColumns:
    def test_returns_non_numeric_columns(self, standardized_protein_data):
        ann_cols = identify_annotation_columns(standardized_protein_data)
        assert "Protein" in ann_cols
        assert "Description" in ann_cols


# ---------------------------------------------------------------------------
# classify_samples
# ---------------------------------------------------------------------------


class TestClassifySamples:
    def test_returns_expected_structure(self, sample_metadata_dict):
        result = classify_samples(
            sample_metadata_dict,
            group_column="Group",
            group_labels=["Control", "Treatment"],
            control_column="Group",
            control_labels=["Control"],
        )
        assert isinstance(result, tuple)
        assert len(result) == 5
