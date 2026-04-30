"""Tests for plot_pca_loadings and plot_umap."""

import importlib

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.visualization import plot_pca_loadings, plot_umap


def _structured_dataset(n_per_group=4, n_proteins=30, seed=7):
    """Two-group dataset with planted markers so PC1 is meaningful."""
    rng = np.random.default_rng(seed)
    samples = []
    metadata = {}
    for g in ("A", "B"):
        for i in range(n_per_group):
            name = f"{g}_{i + 1}"
            samples.append(name)
            metadata[name] = {"Method": g}
    values = rng.normal(10.0, 0.3, size=(n_proteins, len(samples)))
    for j, name in enumerate(samples):
        if metadata[name]["Method"] == "B":
            values[0:5, j] += 3.0
    df = pd.DataFrame(np.power(2.0, values), columns=samples)
    df.insert(0, "Protein", [f"P{i:02d}" for i in range(n_proteins)])
    df.insert(1, "leading_gene_name", [f"Gene_{i:02d}" for i in range(n_proteins)])
    return df, samples, metadata


def test_plot_pca_loadings_returns_figure_and_labels_planted_proteins():
    df, samples, _meta = _structured_dataset()
    fig = plot_pca_loadings(df, samples, top_n=5, annotation_column="leading_gene_name")
    assert isinstance(fig, plt.Figure)
    # Planted markers (Gene_00..Gene_04) should appear in annotations
    ax = fig.axes[0]
    annotated = {child.get_text() for child in ax.texts}
    planted = {f"Gene_{i:02d}" for i in range(5)}
    assert len(annotated & planted) >= 3
    plt.close(fig)


def test_plot_pca_loadings_rejects_when_too_few_samples_for_components():
    df, samples, _meta = _structured_dataset()
    with pytest.raises(ValueError, match="Need at least"):
        plot_pca_loadings(df, samples[:2], pc1=0, pc2=1, top_n=3)


def test_plot_pca_loadings_falls_back_when_annotation_column_missing():
    df, samples, _meta = _structured_dataset()
    fig = plot_pca_loadings(df, samples, top_n=3, annotation_column="not_a_real_column")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


@pytest.mark.skipif(importlib.util.find_spec("umap") is None, reason="umap-learn not installed")
def test_plot_umap_returns_figure():
    df, samples, meta = _structured_dataset()
    fig = plot_umap(df, samples, meta, group_column="Method", n_neighbors=4, random_state=0)
    assert isinstance(fig, plt.Figure)
    # Two groups -> two scatter point sets
    ax = fig.axes[0]
    assert len(ax.collections) >= 2
    plt.close(fig)


def test_plot_umap_raises_clean_error_when_umap_not_installed(monkeypatch):
    """Sanity check that the lazy import path raises ImportError, not AttributeError."""
    import sys

    # Force ImportError by removing the umap module from cache and blocking its import.
    monkeypatch.setitem(sys.modules, "umap", None)
    df, samples, meta = _structured_dataset()
    with pytest.raises((ImportError, TypeError)):
        plot_umap(df, samples, meta, group_column="Method", n_neighbors=3, random_state=0)
