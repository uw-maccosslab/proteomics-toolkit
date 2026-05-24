"""Tests for the classification module."""

import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc as sklearn_auc

from proteomics_toolkit.classification import (
    compute_shap_values,
    plot_shap_summary,
    relabel_features_with_genes,
    run_binary_classification,
    select_features_by_mad,
)

shap_installed = importlib.util.find_spec("shap") is not None
requires_shap = pytest.mark.skipif(not shap_installed, reason="shap is an optional dependency")


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
        with pytest.raises(ValueError, match="feature_selection"):
            run_binary_classification(
                fold_change_matrix,
                group_labels,
                feature_selection="not_a_method",
                cv_method=3,
            )


class TestReturnModel:
    def test_default_does_not_return_model(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix, group_labels, method="random_forest", cv_method=3,
        )
        assert "final_model" not in result
        assert "scaler" not in result
        assert "X_scaled" not in result

    def test_return_model_includes_fit_artifacts(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix, group_labels, method="random_forest",
            cv_method=3, return_model=True,
        )
        assert isinstance(result["final_model"], RandomForestClassifier)
        assert result["X_scaled"].shape[0] == len(group_labels)
        assert result["X_scaled"].shape[1] == result["n_features"]
        assert result["scaler"] is not None
        assert len(result["y_encoded"]) == len(group_labels)


@requires_shap
class TestShap:
    def test_compute_shap_values_returns_2d_explanation(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix, group_labels, method="random_forest",
            cv_method=3, return_model=True,
        )
        explanation = compute_shap_values(
            result["final_model"], result["X_scaled"], feature_names=result["feature_names"],
        )
        # Positive-class slice -> 2-D values matching (n_samples, n_features)
        assert explanation.values.ndim == 2
        assert explanation.values.shape == (len(group_labels), result["n_features"])
        assert list(explanation.feature_names) == list(result["feature_names"])

    def test_compute_shap_values_works_with_dataframe(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix, group_labels, method="random_forest",
            cv_method=3, return_model=True,
        )
        X_df = pd.DataFrame(result["X_scaled"], columns=result["feature_names"])
        explanation = compute_shap_values(result["final_model"], X_df)
        assert list(explanation.feature_names) == list(result["feature_names"])

    def test_plot_shap_summary_beeswarm_runs(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix, group_labels, method="random_forest",
            cv_method=3, return_model=True,
        )
        explanation = compute_shap_values(
            result["final_model"], result["X_scaled"], feature_names=result["feature_names"],
        )
        fig = plot_shap_summary(explanation, max_display=5, plot_type="beeswarm")
        assert fig is not None

    def test_plot_shap_summary_bar_runs(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix, group_labels, method="random_forest",
            cv_method=3, return_model=True,
        )
        explanation = compute_shap_values(
            result["final_model"], result["X_scaled"], feature_names=result["feature_names"],
        )
        fig = plot_shap_summary(explanation, plot_type="bar")
        assert fig is not None

    def test_plot_shap_summary_invalid_plot_type_raises(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix, group_labels, method="random_forest",
            cv_method=3, return_model=True,
        )
        explanation = compute_shap_values(
            result["final_model"], result["X_scaled"], feature_names=result["feature_names"],
        )
        with pytest.raises(ValueError, match="plot_type"):
            plot_shap_summary(explanation, plot_type="violin")


class TestRelabelFeaturesWithGenes:
    def _annot_df(self):
        return pd.DataFrame({
            "protein_group": ["PG0001", "PG0002", "PG0003", "PG0004"],
            "leading_gene_name": ["PIGR", "CHGA", "", None],  # PG0003 empty, PG0004 NaN
        })

    def test_id_fallback_uses_pg_when_gene_empty(self):
        labels = relabel_features_with_genes(
            ["PG0001", "PG0002", "PG0003", "PG0004", "PG0099"],
            self._annot_df(),
        )
        # PG0003 has empty gene -> id fallback; PG0004 has NaN -> id fallback;
        # PG0099 not in annotation_df -> id fallback (gene_map.get default).
        assert labels == ["PIGR", "CHGA", "PG0003", "PG0004", "PG0099"]

    def test_empty_fallback_uses_blank(self):
        labels = relabel_features_with_genes(
            ["PG0001", "PG0003", "PG0099"],
            self._annot_df(),
            fallback="empty",
        )
        assert labels == ["PIGR", "", ""]

    def test_custom_column_names(self):
        annot = pd.DataFrame({"my_id": ["A", "B"], "my_gene": ["GENE_A", "GENE_B"]})
        labels = relabel_features_with_genes(
            ["A", "B"], annot, id_col="my_id", gene_col="my_gene",
        )
        assert labels == ["GENE_A", "GENE_B"]

    def test_invalid_fallback_raises(self):
        with pytest.raises(ValueError, match="fallback"):
            relabel_features_with_genes(["PG0001"], self._annot_df(), fallback="bogus")


class TestRunBinaryClassificationAnnotations:
    def test_annotations_relabels_feature_names_and_importances(self, fold_change_matrix, group_labels):
        # Build an annotation DF that maps the fixture's feature IDs to gene-like names
        annot = pd.DataFrame({
            "protein_group": list(fold_change_matrix.columns),
            "leading_gene_name": [f"GENE_{c}" for c in fold_change_matrix.columns],
        })
        result = run_binary_classification(
            fold_change_matrix, group_labels, method="random_forest",
            cv_method=3, annotations=annot, return_model=True,
        )
        # feature_names should now all be GENE_*
        assert all(n.startswith("GENE_") for n in result["feature_names"])
        # feature_importances.index uses the same gene labels
        assert all(idx.startswith("GENE_") for idx in result["feature_importances"].index)
        # Original IDs preserved under feature_ids when return_model=True
        assert all(not fid.startswith("GENE_") for fid in result["feature_ids"])

    def test_no_annotations_keeps_original_ids(self, fold_change_matrix, group_labels):
        result = run_binary_classification(
            fold_change_matrix, group_labels, method="random_forest", cv_method=3,
        )
        assert all(not n.startswith("GENE_") for n in result["feature_names"])
        # feature_ids only present when return_model=True
        assert "feature_ids" not in result


@requires_shap
class TestShapAnnotations:
    def test_compute_shap_with_annotations_uses_gene_labels(self, fold_change_matrix, group_labels):
        annot = pd.DataFrame({
            "protein_group": list(fold_change_matrix.columns),
            "leading_gene_name": [f"GENE_{c}" for c in fold_change_matrix.columns],
        })
        result = run_binary_classification(
            fold_change_matrix, group_labels, method="random_forest",
            cv_method=3, return_model=True,
        )
        explanation = compute_shap_values(
            result["final_model"], result["X_scaled"],
            feature_names=result["feature_names"], annotations=annot,
        )
        # All labels in the Explanation are gene-prefixed
        assert all(n.startswith("GENE_") for n in explanation.feature_names)
