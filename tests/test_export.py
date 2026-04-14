"""Tests for the export module."""

import os

import pandas as pd

from proteomics_toolkit.export import (
    create_config_dict_from_notebook_vars,
    export_analysis_results,
    export_timestamped_config,
)

# ---------------------------------------------------------------------------
# export_analysis_results
# ---------------------------------------------------------------------------


class TestExportAnalysisResults:
    def test_exports_csv_files(self, tmp_path, standardized_protein_data, sample_metadata_dict):
        prefix = str(tmp_path / "test_export")
        result = export_analysis_results(
            normalized_data=standardized_protein_data,
            sample_metadata=sample_metadata_dict,
            output_prefix=prefix,
        )
        assert isinstance(result, dict)
        assert "normalized_data" in result
        # Check the normalized data file was created
        assert os.path.exists(result["normalized_data"])

    def test_exports_with_differential_results(self, tmp_path, standardized_protein_data, sample_metadata_dict):
        diff_results = pd.DataFrame(
            {
                "Protein": standardized_protein_data["Protein"].tolist(),
                "fold_change": [1.5, 0.8, 2.0, 1.1, 0.5],
                "p_value": [0.01, 0.05, 0.001, 0.3, 0.8],
            }
        )
        prefix = str(tmp_path / "test_diff")
        result = export_analysis_results(
            normalized_data=standardized_protein_data,
            sample_metadata=sample_metadata_dict,
            differential_results=diff_results,
            filtered_data=standardized_protein_data,
            output_prefix=prefix,
        )
        assert "differential_results" in result


# ---------------------------------------------------------------------------
# export_timestamped_config
# ---------------------------------------------------------------------------


class TestExportTimestampedConfig:
    def test_creates_config_file(self, tmp_path):
        config_dict = {
            "normalization_method": "median",
            "p_value_threshold": 0.05,
        }
        output_prefix = str(tmp_path / "config")
        result_path = export_timestamped_config(config_dict, output_prefix=output_prefix)
        assert os.path.exists(result_path)

    def test_config_file_is_nonempty(self, tmp_path):
        config_dict = {"method": "quantile", "threshold": 0.01}
        output_prefix = str(tmp_path / "config2")
        result_path = export_timestamped_config(config_dict, output_prefix=output_prefix)
        assert os.path.getsize(result_path) > 0


# ---------------------------------------------------------------------------
# create_config_dict_from_notebook_vars
# ---------------------------------------------------------------------------


class TestCreateConfigDict:
    def test_returns_dict(self):
        result = create_config_dict_from_notebook_vars(
            normalization_method="median",
            p_value_threshold=0.05,
        )
        assert isinstance(result, dict)
        assert result["normalization_method"] == "median"
