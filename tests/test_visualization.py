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
    plot_variance_vs_intensity,
    plot_variance_vs_peptide_count,
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
    """Tests for the sequence-aware coverage map.

    The fixture ``peptide_df`` ships peptides whose ``peptide_sequence``
    values are ``"PEPTIDE0"``, ``"PEPTIDE1"``, etc. For the coverage-map
    we need a parent protein sequence that contains those peptides at
    known positions; we construct it inline per test.
    """

    def _make_parent_sequence(self, peptide_df):
        """Build a parent protein containing all ALBU peptides in order.

        Returns ``(sequence, start_positions_for_albu_peptides)``.
        """
        albu_peps = peptide_df[
            peptide_df["leading_protein"] == "sp|P12345|ALBU_HUMAN"
        ]["peptide_sequence"].tolist()
        seq_parts = []
        starts = []
        pos = 1
        linker = "GGGGG"
        for pep in albu_peps:
            seq_parts.append(linker)
            pos += len(linker)
            seq_parts.append(pep)
            starts.append(pos)
            pos += len(pep)
        seq_parts.append(linker)
        return "".join(seq_parts), starts

    def test_basic_coverage_with_sequence(self, peptide_df):
        parent_seq, _ = self._make_parent_sequence(peptide_df)
        fig = plot_peptide_coverage_map(
            peptide_df,
            protein_id="sp|P12345|ALBU_HUMAN",
            protein_sequence=parent_seq,
            sample_columns=SAMPLE_NAMES,
        )
        assert isinstance(fig, plt.Figure)

    def test_missing_protein_sequence_raises(self, peptide_df):
        with pytest.raises(ValueError, match="protein_sequence is required"):
            plot_peptide_coverage_map(
                peptide_df,
                protein_id="sp|P12345|ALBU_HUMAN",
                protein_sequence="",
                sample_columns=SAMPLE_NAMES,
            )

    def test_raises_on_unknown_protein(self, peptide_df):
        with pytest.raises(ValueError, match="No peptides found"):
            plot_peptide_coverage_map(
                peptide_df,
                protein_id="sp|Q99999|UNKNOWN",
                protein_sequence="A" * 100,
                sample_columns=SAMPLE_NAMES,
            )

    def test_locates_peptides_via_sequence_search(self, peptide_df):
        """Without start_column, each peptide should be located by str.find."""
        parent_seq, expected_starts = self._make_parent_sequence(peptide_df)
        # No start_column argument - the function must find each peptide
        fig = plot_peptide_coverage_map(
            peptide_df,
            protein_id="sp|P12345|ALBU_HUMAN",
            protein_sequence=parent_seq,
            sample_columns=SAMPLE_NAMES,
        )
        assert isinstance(fig, plt.Figure)
        # Verify the parent sequence actually contains the first peptide
        albu_first = peptide_df[
            peptide_df["leading_protein"] == "sp|P12345|ALBU_HUMAN"
        ]["peptide_sequence"].iloc[0]
        assert albu_first in parent_seq

    def test_color_by_abundance(self, peptide_df):
        parent_seq, _ = self._make_parent_sequence(peptide_df)
        fig = plot_peptide_coverage_map(
            peptide_df,
            protein_id="sp|P12345|ALBU_HUMAN",
            protein_sequence=parent_seq,
            sample_columns=SAMPLE_NAMES,
            color_by="abundance",
        )
        assert isinstance(fig, plt.Figure)

    def test_color_by_detection(self, peptide_df):
        parent_seq, _ = self._make_parent_sequence(peptide_df)
        fig = plot_peptide_coverage_map(
            peptide_df,
            protein_id="sp|P12345|ALBU_HUMAN",
            protein_sequence=parent_seq,
            sample_columns=SAMPLE_NAMES,
            color_by="detection",
        )
        assert isinstance(fig, plt.Figure)

    def test_color_by_fold_change_via_value_column(self, peptide_df):
        parent_seq, _ = self._make_parent_sequence(peptide_df)
        df = peptide_df.copy()
        df["logFC"] = np.arange(len(df), dtype=float) - (len(df) / 2.0)
        fig = plot_peptide_coverage_map(
            df,
            protein_id="sp|P12345|ALBU_HUMAN",
            protein_sequence=parent_seq,
            sample_columns=SAMPLE_NAMES,
            color_by="fold_change",
            value_column="logFC",
        )
        assert isinstance(fig, plt.Figure)

    def test_color_by_fold_change_computed(self, peptide_df, sample_meta):
        parent_seq, _ = self._make_parent_sequence(peptide_df)
        fig = plot_peptide_coverage_map(
            peptide_df,
            protein_id="sp|P12345|ALBU_HUMAN",
            protein_sequence=parent_seq,
            sample_columns=SAMPLE_NAMES,
            color_by="fold_change",
            sample_metadata=sample_meta,
            group_column="Group",
            group_labels=("Control", "Treatment"),
        )
        assert isinstance(fig, plt.Figure)

    def test_invalid_color_by_raises(self, peptide_df):
        parent_seq, _ = self._make_parent_sequence(peptide_df)
        with pytest.raises(ValueError, match="color_by"):
            plot_peptide_coverage_map(
                peptide_df,
                protein_id="sp|P12345|ALBU_HUMAN",
                protein_sequence=parent_seq,
                sample_columns=SAMPLE_NAMES,
                color_by="bogus",
            )


class TestPlotVarianceVsPeptideCount:
    def test_basic_diagnostic(self):
        rng = np.random.default_rng(3)
        n = 40
        counts = rng.integers(2, 25, size=n)
        # Variance that genuinely decays with peptide count, plus noise
        variances = np.exp(-np.log(counts)) * rng.uniform(0.5, 1.5, size=n)
        results = pd.DataFrame(
            {
                "Protein": [f"P{i:04d}" for i in range(n)],
                "peptide_count_used": counts,
                "residual_s2": variances,
            }
        )
        fig = plot_variance_vs_peptide_count(results)
        assert isinstance(fig, plt.Figure)

    def test_missing_columns_raises(self):
        results = pd.DataFrame({"Protein": ["P0"], "logFC": [1.0]})
        with pytest.raises(ValueError, match="missing required columns"):
            plot_variance_vs_peptide_count(results)


class TestPlotVarianceVsIntensity:
    def _build_intensity_trend_result(self):
        """Produce a results DataFrame with the attrs payload that the
        diagnostic plot expects."""
        import proteomics_toolkit as ptk

        rng = np.random.default_rng(5)
        n_feat = 30
        samples_a = [f"A_{i}" for i in range(6)]
        samples_b = [f"B_{i}" for i in range(6)]
        all_samples = samples_a + samples_b

        base = rng.lognormal(mean=12, sigma=1.5, size=n_feat)
        # Poisson-like: sd proportional to sqrt(mean), i.e. relative sd ~ 1/sqrt(mean)
        rel_sd = 1.0 / np.sqrt(base)
        values = base[:, None] * np.exp(rng.normal(0, rel_sd[:, None], size=(n_feat, 12)))

        features = [f"P{i:04d}" for i in range(n_feat)]
        data = pd.DataFrame(values, index=features, columns=all_samples)

        metadata_df = pd.DataFrame(
            {"Sample": all_samples, "Group": ["Control"] * 6 + ["Treatment"] * 6}
        )

        config = ptk.StatisticalConfig()
        config.analysis_type = "unpaired"
        config.group_column = "Group"
        config.group_labels = ["Control", "Treatment"]
        config.log_transform_before_stats = True
        config.moderation = "intensity_trend"
        # When called directly (bypassing the dispatcher), the raw data *is*
        # the linear-space data.
        config._raw_feature_data = data

        return ptk.run_moderated_linear_model(data, metadata_df, config)

    def test_basic_intensity_diagnostic(self):
        result = self._build_intensity_trend_result()
        fig = plot_variance_vs_intensity(result)
        assert isinstance(fig, plt.Figure)

    def test_missing_attrs_raises(self):
        results = pd.DataFrame({"Protein": ["P0"]})
        with pytest.raises(ValueError, match="intensity-trend points"):
            plot_variance_vs_intensity(results)
