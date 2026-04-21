"""Tests for the QC visualization functions.

Covers the five QC plots added in v26.2.0. Uses matplotlib's non-interactive
``Agg`` backend so tests do not open GUI windows.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.visualization import (
    plot_cv_distribution,
    plot_identifications_per_sample,
    plot_intensity_distributions,
    plot_missing_value_heatmap,
    plot_peptide_coverage_map,
)

SAMPLE_NAMES = ["S1", "S2", "S3", "S4", "S5", "S6"]


@pytest.fixture
def protein_df():
    """Tiny protein-level DataFrame with some missing values."""
    rng = np.random.default_rng(0)
    n = 20
    data = {name: rng.uniform(1e5, 1e7, size=n) for name in SAMPLE_NAMES}
    df = pd.DataFrame(data)
    # Inject missingness
    df.loc[0, "S1"] = np.nan
    df.loc[1, "S2"] = 0
    df.loc[2, ["S3", "S4"]] = np.nan
    df.insert(0, "Protein", [f"P{i:04d}" for i in range(n)])
    return df


@pytest.fixture
def peptide_df():
    """Tiny peptide-level DataFrame."""
    rng = np.random.default_rng(1)
    n = 12
    data = {name: rng.uniform(1e4, 1e6, size=n) for name in SAMPLE_NAMES}
    df = pd.DataFrame(data)
    df.loc[0, "S1"] = np.nan
    df.insert(0, "peptide_sequence", [f"PEPTIDE{i}" for i in range(n)])
    df.insert(1, "leading_protein", ["sp|P12345|ALBU_HUMAN"] * 6 + ["sp|P67890|TRFE_HUMAN"] * 6)
    return df


@pytest.fixture
def sample_meta():
    return {
        "S1": {"Group": "Control"},
        "S2": {"Group": "Control"},
        "S3": {"Group": "Control"},
        "S4": {"Group": "Treatment"},
        "S5": {"Group": "Treatment"},
        "S6": {"Group": "Treatment"},
    }


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


class TestPlotMissingValueHeatmap:
    def test_protein_level(self, protein_df):
        fig = plot_missing_value_heatmap(protein_df, SAMPLE_NAMES)
        assert isinstance(fig, plt.Figure)

    def test_peptide_level(self, peptide_df):
        fig = plot_missing_value_heatmap(peptide_df, SAMPLE_NAMES, feature_label="peptide")
        assert isinstance(fig, plt.Figure)


class TestPlotIdentificationsPerSample:
    def test_protein_level_no_metadata(self, protein_df):
        fig = plot_identifications_per_sample(protein_df, SAMPLE_NAMES)
        assert isinstance(fig, plt.Figure)

    def test_protein_level_with_metadata(self, protein_df, sample_meta):
        fig = plot_identifications_per_sample(protein_df, SAMPLE_NAMES, sample_metadata=sample_meta)
        assert isinstance(fig, plt.Figure)

    def test_peptide_level(self, peptide_df, sample_meta):
        fig = plot_identifications_per_sample(
            peptide_df, SAMPLE_NAMES, sample_metadata=sample_meta, feature_label="peptide"
        )
        assert isinstance(fig, plt.Figure)


class TestPlotIntensityDistributions:
    def test_protein_level(self, protein_df):
        fig = plot_intensity_distributions(protein_df, SAMPLE_NAMES)
        assert isinstance(fig, plt.Figure)

    def test_peptide_level_no_log(self, peptide_df):
        fig = plot_intensity_distributions(peptide_df, SAMPLE_NAMES, log_transform=False, feature_label="peptide")
        assert isinstance(fig, plt.Figure)


class TestPlotCvDistribution:
    def test_protein_level_no_metadata(self, protein_df):
        fig = plot_cv_distribution(protein_df, SAMPLE_NAMES)
        assert isinstance(fig, plt.Figure)

    def test_protein_level_by_group(self, protein_df, sample_meta):
        fig = plot_cv_distribution(protein_df, SAMPLE_NAMES, sample_metadata=sample_meta)
        assert isinstance(fig, plt.Figure)

    def test_peptide_level(self, peptide_df, sample_meta):
        fig = plot_cv_distribution(peptide_df, SAMPLE_NAMES, sample_metadata=sample_meta, feature_label="peptide")
        assert isinstance(fig, plt.Figure)


class TestPlotPeptideCoverageMap:
    def test_basic_coverage(self, peptide_df):
        fig = plot_peptide_coverage_map(
            peptide_df,
            protein_id="sp|P12345|ALBU_HUMAN",
            sample_columns=SAMPLE_NAMES,
        )
        assert isinstance(fig, plt.Figure)

    def test_raises_on_unknown_protein(self, peptide_df):
        with pytest.raises(ValueError, match="No peptides found"):
            plot_peptide_coverage_map(
                peptide_df,
                protein_id="sp|Q99999|UNKNOWN",
                sample_columns=SAMPLE_NAMES,
            )

    def test_with_explicit_start_positions(self, peptide_df):
        df = peptide_df.copy()
        df["start_position"] = [10, 30, 50, 75, 100, 120, 10, 30, 50, 75, 100, 120]
        fig = plot_peptide_coverage_map(
            df,
            protein_id="sp|P12345|ALBU_HUMAN",
            sample_columns=SAMPLE_NAMES,
            start_column="start_position",
            protein_length=200,
        )
        assert isinstance(fig, plt.Figure)
