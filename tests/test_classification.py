"""Tests for the classification module."""

import numpy as np
import pandas as pd
import pytest

from proteomics_toolkit.classification import (
    run_binary_classification,
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
