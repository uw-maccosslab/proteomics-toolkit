"""Regression tests for the LOWESS delta speedup in the moderated-LM variance priors.

The intensity_trend / deqms variance priors call statsmodels ``lowess`` with
``delta = 0.01 * range(x)`` (statsmodels' documented large-n recommendation and
R ``lowess()``'s default) instead of the O(n^2) ``delta=0`` default, which
otherwise dominates runtime on peptide-scale inputs (~80k feature-group points).

These assert (a) the approximation is numerically indistinguishable from the
exact fit on the smooth variance-vs-intensity trend, and (b) the intensity_trend
path still runs end to end and recovers a planted effect with the delta applied.
"""
import numpy as np
import pandas as pd

from proteomics_toolkit.statistical_analysis import (
    StatisticalConfig,
    run_moderated_linear_model,
)


def test_delta_lowess_matches_exact_on_smooth_trend():
    """delta = 0.01 * range reproduces the exact (delta=0) LOWESS curve to < 1e-3."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0.0, 20.0, 4000))
    y = 3.0 - 0.12 * x + rng.normal(0.0, 0.05, x.size)  # smooth monotone trend + noise

    exact = lowess(y, x, frac=0.5, it=3, return_sorted=True)
    fast = lowess(y, x, frac=0.5, it=3, return_sorted=True, delta=0.01 * np.ptp(x))

    grid = np.linspace(x.min(), x.max(), 300)
    ye = np.interp(grid, exact[:, 0], exact[:, 1])
    yf = np.interp(grid, fast[:, 0], fast[:, 1])
    assert np.max(np.abs(ye - yf)) < 1e-3


def _unpaired_intensity_fixture(total_rows=200, with_effect_rows=20, seed=1):
    """Unpaired intensity_trend fixture with a wide dynamic range so the log-mean
    spread is large enough that ``delta`` genuinely skips points."""
    rng = np.random.default_rng(seed)
    samples = [f"C{i}" for i in range(6)] + [f"T{i}" for i in range(6)]
    base = rng.uniform(6.0, 16.0, size=(total_rows, 1))
    values = base + rng.normal(0.0, 0.5, size=(total_rows, 12))  # log2 scale
    values[:with_effect_rows, 6:] += 1.5  # planted effect in the treatment group
    feats = [f"P{i:04d}" for i in range(total_rows)]
    feature_data = pd.DataFrame(values, index=feats, columns=samples)
    meta = pd.DataFrame({"Sample": samples, "Group": ["Control"] * 6 + ["Treatment"] * 6})
    config = StatisticalConfig()
    config.analysis_type = "unpaired"
    config.group_column = "Group"
    config.group_labels = ["Control", "Treatment"]
    config.log_transform_before_stats = False
    config.statistical_test_method = "moderated_linear_model"
    config.moderation = "intensity_trend"
    config._raw_feature_data = 2 ** feature_data  # trend prior expects raw intensities
    return feature_data, meta, config


def test_intensity_trend_runs_and_ranks_with_delta():
    """The delta-using intensity_trend path executes and ranks the planted effect."""
    feat, meta, config = _unpaired_intensity_fixture()
    result = run_moderated_linear_model(feat, meta, config).set_index("Protein")
    assert result["intensity_s0_sq"].notna().any()
    top20 = set(result.sort_values("P.Value").head(20).index)
    planted = {f"P{i:04d}" for i in range(20)}
    assert len(top20 & planted) >= 15
