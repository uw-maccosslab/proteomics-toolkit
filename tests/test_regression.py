"""Tests for the regression module."""

import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.regression import (
    plot_regression_scatter,
    run_elasticnet_regression,
)


@pytest.fixture
def elasticnet_signal_data():
    """40 samples x 200 features; 5 planted predictive features + noise.

    Target is a linear combination of the planted features plus noise, so a
    well-tuned ElasticNet should recover positive out-of-fold R2 and rank the
    planted features high in selection frequency.
    """
    rng = np.random.RandomState(0)
    n, p, n_signal = 40, 200, 5
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:n_signal] = 2.0
    y = X @ beta + rng.normal(scale=1.0, size=n)
    samples = [f"S{i}" for i in range(n)]
    data = pd.DataFrame(X, index=samples, columns=[f"F{j}" for j in range(p)])
    target = pd.Series(y, index=samples)
    signal_features = [f"F{j}" for j in range(n_signal)]
    return data, target, signal_features


@pytest.fixture
def elasticnet_noise_data():
    """40 samples x 200 features of pure noise; target independent of features."""
    rng = np.random.RandomState(1)
    n, p = 40, 200
    X = rng.normal(size=(n, p))
    y = rng.normal(size=n)
    samples = [f"S{i}" for i in range(n)]
    data = pd.DataFrame(X, index=samples, columns=[f"F{j}" for j in range(p)])
    target = pd.Series(y, index=samples)
    return data, target


class TestRunElasticnetRegression:
    def test_returns_expected_keys(self, elasticnet_signal_data):
        data, target, _ = elasticnet_signal_data
        result = run_elasticnet_regression(
            data, target, outer_cv=(5, 2), inner_cv=3, n_permutations=0, random_state=0
        )
        for key in (
            "outer_r2_mean",
            "outer_r2_std",
            "outer_rmse_mean",
            "outer_spearman_mean",
            "pooled_r2",
            "per_fold_scores",
            "selection_frequency",
            "consensus_features",
            "n_features_per_fold",
            "chosen_l1_ratio_per_fold",
            "permutation_r2_null",
            "permutation_p_value",
            "cv_predictions",
            "coefficients",
            "n_features",
            "final_model",
            "config",
        ):
            assert key in result, f"missing key {key}"
        assert isinstance(result["selection_frequency"], pd.Series)
        assert isinstance(result["cv_predictions"], pd.DataFrame)
        assert isinstance(result["coefficients"], pd.Series)

    def test_recovers_signal_above_noise(self, elasticnet_signal_data, elasticnet_noise_data):
        data, target, signal_features = elasticnet_signal_data
        signal_result = run_elasticnet_regression(
            data, target, outer_cv=(5, 2), inner_cv=3, n_permutations=0, random_state=0
        )

        noise_data, noise_target = elasticnet_noise_data
        noise_result = run_elasticnet_regression(
            noise_data, noise_target, outer_cv=(5, 2), inner_cv=3, n_permutations=0, random_state=0
        )

        assert signal_result["outer_r2_mean"] > noise_result["outer_r2_mean"]
        assert signal_result["outer_r2_mean"] > 0.1

    def test_planted_features_rank_high_in_selection_frequency(self, elasticnet_signal_data):
        data, target, signal_features = elasticnet_signal_data
        result = run_elasticnet_regression(
            data, target, outer_cv=(5, 2), inner_cv=3, n_permutations=0, random_state=0
        )
        top_features = set(result["selection_frequency"].head(len(signal_features)).index)
        # At least half of the planted features should be in the top-N by selection frequency.
        assert len(top_features & set(signal_features)) >= len(signal_features) // 2

    def test_permutation_p_value_low_for_signal(self, elasticnet_signal_data):
        data, target, _ = elasticnet_signal_data
        result = run_elasticnet_regression(
            data, target, outer_cv=(5, 2), inner_cv=3, n_permutations=20, random_state=0
        )
        assert result["permutation_p_value"] is not None
        assert result["permutation_p_value"] < 0.3

    def test_permutation_disabled_returns_none(self, elasticnet_noise_data):
        data, target = elasticnet_noise_data
        result = run_elasticnet_regression(
            data, target, outer_cv=(5, 2), inner_cv=3, n_permutations=0, random_state=0
        )
        assert result["permutation_p_value"] is None
        assert len(result["permutation_r2_null"]) == 0

    def test_too_few_samples_raises(self):
        data = pd.DataFrame(
            np.random.RandomState(0).normal(size=(5, 10)),
            index=[f"S{i}" for i in range(5)],
            columns=[f"F{j}" for j in range(10)],
        )
        target = pd.Series(np.random.RandomState(0).normal(size=5), index=data.index)
        with pytest.raises(ValueError, match="at least 10"):
            run_elasticnet_regression(data, target, n_permutations=0)

    def test_annotations_relabels_selection_frequency_and_coefficients(self, elasticnet_signal_data):
        data, target, _ = elasticnet_signal_data
        annot = pd.DataFrame(
            {
                "protein_group": list(data.columns),
                "leading_gene_name": [f"GENE_{c}" for c in data.columns],
            }
        )
        result = run_elasticnet_regression(
            data,
            target,
            outer_cv=(5, 2),
            inner_cv=3,
            n_permutations=0,
            random_state=0,
            annotations=annot,
        )
        assert all(idx.startswith("GENE_") for idx in result["selection_frequency"].index)
        assert all(idx.startswith("GENE_") for idx in result["coefficients"].index)

    def test_consensus_threshold_filters_features(self, elasticnet_signal_data):
        data, target, _ = elasticnet_signal_data
        result = run_elasticnet_regression(
            data,
            target,
            outer_cv=(5, 2),
            inner_cv=3,
            n_permutations=0,
            random_state=0,
            consensus_threshold=0.9,
        )
        for feat in result["consensus_features"]:
            assert result["selection_frequency"][feat] >= 0.9


class TestPlotRegressionScatter:
    def test_returns_figure(self, elasticnet_signal_data):
        import matplotlib

        matplotlib.use("Agg")

        data, target, _ = elasticnet_signal_data
        result = run_elasticnet_regression(
            data, target, outer_cv=(5, 2), inner_cv=3, n_permutations=0, random_state=0
        )
        fig = plot_regression_scatter(result)
        assert fig is not None
        assert len(fig.axes) == 1
