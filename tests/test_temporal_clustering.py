"""Tests for proteomics_toolkit.temporal_clustering."""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from proteomics_toolkit.temporal_clustering import (
    TemporalClusteringConfig,
    plot_cluster_heatmap,
)


def _make_clustered_temporal_df(cluster_sizes=(120, 80, 40, 15)):
    """Build a fake merged_df with `Cluster`/`Cluster_Name` and Week_* columns."""
    rng = np.random.default_rng(42)
    week_columns = ["Week_0", "Week_2", "Week_4", "Week_6", "Week_12"]

    rows = []
    for cluster_id, n in enumerate(cluster_sizes):
        # Each cluster gets a distinct mean trajectory so the heatmap colour-codes
        # them differently. Real data noise comes from rng.normal().
        if cluster_id == 0:
            mean_traj = np.array([1.0, 0.5, 0.0, -0.5, -1.0])  # monotonic down
        elif cluster_id == 1:
            mean_traj = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # monotonic up
        elif cluster_id == 2:
            mean_traj = np.array([-0.5, -0.2, 1.0, -0.2, -0.5])  # peak at W4
        else:
            mean_traj = np.array([0.5, 0.0, -1.0, 0.0, 0.5])  # trough at W4
        for i in range(n):
            values = mean_traj + rng.normal(0, 0.3, size=len(week_columns))
            row = {
                "Protein": f"PG{cluster_id:02d}_{i:04d}",
                "Gene": f"G{cluster_id}_{i}",
                "Cluster": cluster_id,
                "Cluster_Name": f"Cluster {cluster_id + 1}",
                "P.Value": rng.uniform(1e-5, 1e-2),
            }
            for c, v in zip(week_columns, values):
                row[c] = v
            rows.append(row)
    return pd.DataFrame(rows), week_columns


class TestPlotClusterHeatmapLabels:
    """Regression: cluster size labels must reflect TRUE cluster sizes, not the
    rendered/truncated row count."""

    def test_no_cap_shows_all_proteins(self):
        df, week_cols = _make_clustered_temporal_df(cluster_sizes=(120, 80, 40, 15))
        fig = plot_cluster_heatmap(
            df,
            week_cols,
            max_proteins_per_cluster=None,
            show_genes=False,
        )
        # twin x-axis carries the cluster-size labels
        text_labels = []
        for ax in fig.axes:
            for t in ax.get_yticklabels():
                if "n=" in t.get_text():
                    text_labels.append(t.get_text())
        # We expect labels matching the true sizes (120, 80, 40, 15).
        joined = " ".join(text_labels)
        for true_n in (120, 80, 40, 15):
            assert f"n={true_n}" in joined, f"missing n={true_n} in labels: {joined!r}"

    def test_label_uses_true_size_even_when_capped(self):
        """With ``max_proteins_per_cluster=50``, large clusters used to be
        labeled n=50 (the truncated count). Labels must still report the true
        size; only the rendered band height changes."""
        df, week_cols = _make_clustered_temporal_df(cluster_sizes=(120, 80, 40, 15))
        fig = plot_cluster_heatmap(
            df,
            week_cols,
            max_proteins_per_cluster=50,
            show_genes=False,
        )
        text_labels = []
        for ax in fig.axes:
            for t in ax.get_yticklabels():
                if "n=" in t.get_text():
                    text_labels.append(t.get_text())
        joined = " ".join(text_labels)
        # Even though the 120- and 80-protein clusters are capped to 50 rows on
        # screen, the labels must reflect the TRUE counts.
        assert "n=120" in joined, f"missing n=120 in labels: {joined!r}"
        assert "n=80" in joined, f"missing n=80 in labels: {joined!r}"
        assert "n=40" in joined, f"missing n=40 in labels: {joined!r}"
        assert "n=15" in joined, f"missing n=15 in labels: {joined!r}"

    def test_default_max_proteins_per_cluster_is_no_cap(self):
        """The default behaviour (max_proteins_per_cluster=None) shows every
        protein and the y-axis "Proteins (n=...)" reflects the full row count."""
        df, week_cols = _make_clustered_temporal_df(cluster_sizes=(60, 50, 40, 30))
        fig = plot_cluster_heatmap(df, week_cols, show_genes=False)
        # Left axis carries the proteins-total label
        ylabels = [ax.get_ylabel() for ax in fig.axes if ax.get_ylabel()]
        proteins_label = next((label for label in ylabels if "Proteins" in label), "")
        assert "n=180" in proteins_label, f"expected n=180 in {proteins_label!r}"


class TestTemporalClusteringConfigDefaults:
    def test_max_proteins_per_cluster_heatmap_default_is_none(self):
        cfg = TemporalClusteringConfig()
        assert cfg.max_proteins_per_cluster_heatmap is None
