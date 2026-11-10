"""Tests for the ``show`` parameter of the plot functions that used to return nothing.

Each function shows its figure and returns None by default, as before, and returns
the figure without showing it when ``show=False``, so notebooks can save it. Uses
matplotlib's non-interactive ``Agg`` backend so tests do not open GUI windows.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.visualization import (
    plot_box_plot,
    plot_comparative_pca,
    plot_control_correlation_analysis,
    plot_normalization_comparison,
    plot_pca,
    plot_sample_correlation_heatmap,
    plot_sample_correlation_triangular_heatmap,
    plot_volcano,
)

SAMPLES = [f"S{i}" for i in range(1, 9)]


@pytest.fixture
def intensities():
    """Positive linear intensities, 40 proteins by 8 samples, no missing values."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.lognormal(mean=15, sigma=1, size=(40, len(SAMPLES))), columns=SAMPLES)
    df.insert(0, "Protein", [f"PG{i:04d}" for i in range(40)])
    return df


@pytest.fixture
def metadata():
    """Two groups of four; two samples of each group are pools for the control plots."""
    return {
        s: {
            "Group": "Control" if i < 4 else "Treatment",
            "Study": "Control" if i < 4 else "Treatment",
            "Subject": "Pool" if i % 4 < 2 else f"M{i}",
        }
        for i, s in enumerate(SAMPLES)
    }


@pytest.fixture
def differential():
    """Differential results with a few hits on each side."""
    rng = np.random.default_rng(1)
    n = 60
    log_fc = rng.normal(0, 0.6, n)
    p = rng.uniform(0, 1, n)
    p[:6] = 1e-5
    return pd.DataFrame(
        {
            "Protein": [f"PG{i:04d}" for i in range(n)],
            "Gene": [f"Gene{i}" for i in range(n)],
            "logFC": log_fc,
            "P.Value": p,
            "adj.P.Val": np.minimum(p * 10, 1),
        }
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _calls(intensities, metadata, differential):
    log2 = intensities.copy()
    log2[SAMPLES] = np.log2(log2[SAMPLES])
    return {
        "plot_box_plot": lambda **kw: plot_box_plot(intensities, SAMPLES, metadata, **kw),
        "plot_volcano": lambda **kw: plot_volcano(differential, **kw),
        "plot_normalization_comparison": lambda **kw: plot_normalization_comparison(
            intensities, intensities, SAMPLES, **kw
        ),
        "plot_sample_correlation_heatmap": lambda **kw: plot_sample_correlation_heatmap(log2, SAMPLES, metadata, **kw),
        "plot_sample_correlation_triangular_heatmap": lambda **kw: plot_sample_correlation_triangular_heatmap(
            log2, SAMPLES, metadata, **kw
        ),
        "plot_pca": lambda **kw: plot_pca(intensities, SAMPLES, metadata, log_transform=True, **kw),
        "plot_comparative_pca": lambda **kw: plot_comparative_pca(
            intensities, intensities, log2, SAMPLES, metadata, **kw
        ),
        "plot_control_correlation_analysis": lambda **kw: plot_control_correlation_analysis(
            intensities, intensities, log2, SAMPLES, metadata, **kw
        ),
    }


FUNCTIONS = [
    "plot_box_plot",
    "plot_volcano",
    "plot_normalization_comparison",
    "plot_sample_correlation_heatmap",
    "plot_sample_correlation_triangular_heatmap",
    "plot_pca",
    "plot_comparative_pca",
    "plot_control_correlation_analysis",
]


@pytest.mark.parametrize("name", FUNCTIONS)
def test_show_false_returns_the_figure(name, intensities, metadata, differential, tmp_path):
    fig = _calls(intensities, metadata, differential)[name](show=False)
    assert isinstance(fig, plt.Figure)
    assert fig.axes
    out = tmp_path / f"{name}.png"
    fig.savefig(out)
    assert out.stat().st_size > 0


@pytest.mark.parametrize("name", FUNCTIONS)
def test_default_shows_and_returns_none(name, intensities, metadata, differential):
    assert _calls(intensities, metadata, differential)[name]() is None


def test_volcano_with_no_rows_returns_none():
    assert plot_volcano(pd.DataFrame(columns=["logFC", "P.Value", "adj.P.Val"]), show=False) is None


@pytest.mark.parametrize("name", ["plot_pca", "plot_comparative_pca"])
def test_default_group_colors_raise_no_warning(name, intensities, metadata, differential):
    """The default tab10 colors are RGBA arrays; passing one as ``c=`` made matplotlib warn
    that it looks like a single numeric RGB(A) sequence, and plot_comparative_pca called
    ``plt.cm.get_cmap``, which matplotlib 3.9 removed."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r".*\*c\* argument")
        warnings.filterwarnings("error", category=DeprecationWarning)
        fig = _calls(intensities, metadata, differential)[name](show=False)
    assert isinstance(fig, plt.Figure)
