"""Tests for the enrichment module.

Network-dependent tests (Enrichr API calls) are marked with @pytest.mark.network
so they can be skipped in offline CI environments with: pytest -m "not network"
"""

import pandas as pd
import pytest

from proteomics_toolkit.enrichment import (
    LIBRARY_COLORS,
    EnrichmentConfig,
    get_available_libraries,
    merge_enrichment_results,
    parse_enrichr_results,
)


# ---------------------------------------------------------------------------
# EnrichmentConfig
# ---------------------------------------------------------------------------


class TestEnrichmentConfig:
    def test_defaults(self):
        config = EnrichmentConfig()
        assert config.pvalue_cutoff == 0.05
        assert config.top_n == 20
        assert config.min_genes == 5
        assert len(config.enrichr_libraries) > 0

    def test_custom_libraries(self):
        config = EnrichmentConfig(enrichr_libraries=["KEGG_2021_Human"])
        assert config.enrichr_libraries == ["KEGG_2021_Human"]

    def test_library_colors_populated(self):
        assert len(LIBRARY_COLORS) > 0
        assert "KEGG_2021_Human" in LIBRARY_COLORS


# ---------------------------------------------------------------------------
# get_available_libraries
# ---------------------------------------------------------------------------


class TestGetAvailableLibraries:
    def test_returns_list(self):
        libs = get_available_libraries()
        assert isinstance(libs, list)
        assert len(libs) > 0


# ---------------------------------------------------------------------------
# merge_enrichment_results
# ---------------------------------------------------------------------------


class TestMergeEnrichmentResults:
    def test_merges_two_dataframes(self):
        df1 = pd.DataFrame({"Term": ["A"], "P-value": [0.01], "Library": ["KEGG"]})
        df2 = pd.DataFrame({"Term": ["B"], "P-value": [0.02], "Library": ["GO"]})
        merged = merge_enrichment_results({"group1": df1, "group2": df2})
        assert len(merged) == 2

    def test_handles_empty_dict(self):
        merged = merge_enrichment_results({})
        assert isinstance(merged, pd.DataFrame)
        assert len(merged) == 0


# ---------------------------------------------------------------------------
# parse_enrichr_results
# ---------------------------------------------------------------------------


class TestParseEnrichrResults:
    def test_parses_mock_response(self):
        # Simulate the structure returned by Enrichr API
        mock_data = {
            "KEGG_2021_Human": [
                # Each entry: [rank, term, p-value, z-score, combined_score,
                #              overlapping_genes, adjusted_p, old_p, old_adj_p]
                [1, "Glycolysis", 0.001, -2.0, 50.0, ["GAPDH", "ENO1"], 0.01, 0.001, 0.01],
                [2, "TCA cycle", 0.05, -1.0, 10.0, ["IDH1"], 0.1, 0.05, 0.1],
            ]
        }
        result = parse_enrichr_results(mock_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1
