"""Tests for the classification module."""

import numpy as np
import pandas as pd
from sklearn.metrics import auc as sklearn_auc

from proteomics_toolkit.classification import (
    run_binary_classification,
    select_features_by_mad,
)


class TestRunBinaryClassification:
    def test_logistic_regression_returns_expected_keys(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix,
            group_labels,
            method="logistic_regression",
            cv_method=3,
        )
        assert "accuracy" in result
        assert "auc_roc" in result
        assert "confusion_matrix" in result
        assert "feature_importances" in result

    def test_random_forest_runs(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix,
            group_labels,
            method="random_forest",
            cv_method=3,
        )
        assert "accuracy" in result
        assert result["accuracy"] >= 0.0

    def test_linear_svm_runs(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix,
            group_labels,
            method="linear_svm",
            cv_method=3,
        )
        assert "accuracy" in result

    def test_n_top_features_limits_features(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix,
            group_labels,
            n_top_features=5,
            cv_method=3,
        )
        assert result["n_features"] == 5

    def test_custom_feature_proteins(self, fold_change_matrix, group_labels):
        selected = list(fold_change_matrix.columns[:3])
        result = run_binary_classification(
            fold_change_matrix,
            group_labels,
            feature_proteins=selected,
            cv_method=3,
        )
        assert result["n_features"] == 3

    def test_accuracy_is_reasonable(self, fold_change_matrix, group_labels):
        """With well-separated groups, accuracy should be above chance."""
        result = run_binary_classification(
            fold_change_matrix,
            group_labels,
            method="logistic_regression",
            cv_method=3,
        )
        # Groups are well-separated in the fixture, expect > 60%
        assert result["accuracy"] > 0.5

    def test_per_fold_roc_is_consistent_with_auc_after_inversion(self):
        """Per-fold (fpr, tpr) curves must match their reported AUC scalars even
        when the classifier learned the labels in reverse.

        Regression test: previously the auto-inversion branch replaced the per-fold
        auc scalar with max(a, 1-a) but did not transform the (fpr, tpr) arrays.
        That left plotted curves below the diagonal while the printed AUC read
        >= 0.5. The fix reflects each fold across y=x (swap fpr/tpr, auc->1-auc)
        so the stored tuples are internally consistent.
        """
        # Construct a classifier-adversarial fixture: features are strongly
        # anti-correlated with labels, so the classifier's raw probability output
        # for class 1 is low when the true label is 1. The auto-flip branch
        # should fire and the resulting fold_roc_data must be self-consistent.
        rng = np.random.default_rng(0)
        n_per_group = 12
        subjects = [f"S_{i}" for i in range(2 * n_per_group)]
        proteins = [f"P_{i}" for i in range(15)]
        fc = rng.normal(loc=0.0, scale=0.3, size=(2 * n_per_group, 15))
        # Group A (first half) gets +1 offset, Group B (second half) gets -1.
        fc[:n_per_group, :] += 1.0
        fc[n_per_group:, :] -= 1.0
        fc_df = pd.DataFrame(fc, index=subjects, columns=proteins)
        # Assign labels so alphabetical encoding puts the "well-predicted" group
        # at class 0 rather than class 1 -> classifier learns the reverse.
        labels = pd.Series(["B_pos"] * n_per_group + ["A_neg"] * n_per_group, index=subjects)

        result = run_binary_classification(
            fc_df, labels, method="logistic_regression", cv_method=3,
        )

        # Every per-fold AUC scalar must match the integral of its (fpr, tpr).
        for fpr, tpr, stored_auc in result["fold_roc_data"]:
            recomputed = sklearn_auc(fpr, tpr)
            assert abs(recomputed - stored_auc) < 1e-9, (
                f"Per-fold AUC {stored_auc} does not match its (fpr, tpr) curve "
                f"area {recomputed}; the (fpr, tpr) arrays were not reflected when "
                f"the auto-flip triggered."
            )
        # And the aggregated AUC must be >= 0.5 after the auto-flip correction.
        assert result["auc_roc"] >= 0.5


# ---------------------------------------------------------------------------
# Feature selection: MAD (unsupervised) and nested differential abundance
# ---------------------------------------------------------------------------


class TestSelectFeaturesByMad:
    def test_returns_top_n_by_mad(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "lowvar": rng.normal(0, 0.01, size=20),
                "midvar": rng.normal(0, 0.5, size=20),
                "hivar": rng.normal(0, 2.0, size=20),
            }
        )
        top = select_features_by_mad(df, n_top_features=2)
        assert top == ["hivar", "midvar"]

    def test_default_classification_uses_mad(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix,
            group_labels,
            cv_method=3,
        )
        assert result["feature_selection"] == "mad"


class TestNestedDifferentialAbundance:
    def test_runs_and_returns_schema(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix,
            group_labels,
            feature_selection="differential_abundance",
            n_top_features=10,
            cv_method=3,
        )
        assert result["feature_selection"] == "differential_abundance"
        assert result["n_features"] == 10
        # Final-model feature names should be the 10 picked from full data
        assert len(result["feature_names"]) == 10
        assert "accuracy" in result
        assert "auc_roc" in result


class TestFoldChangeSelectionStillAvailable:
    def test_explicit_fold_change_mode(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix,
            group_labels,
            feature_selection="fold_change",
            n_top_features=5,
            cv_method=3,
        )
        assert result["feature_selection"] == "fold_change"
        assert result["n_features"] == 5


class TestInvalidFeatureSelection:
    def test_unknown_selection_raises(self, fold_change_matrix, group_labels):
        import pytest

        with pytest.raises(ValueError, match="feature_selection"):
            run_binary_classification(
                fold_change_matrix,
                group_labels,
                feature_selection="not_a_method",
                cv_method=3,
            )
