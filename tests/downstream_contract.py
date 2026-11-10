"""What code outside this repository uses, pinned so that a change which breaks it fails a test.

The toolkit is called from the lab's analysis notebooks, from Skyline-PRISM's parity fixtures and by
anyone who installs it from PyPI. Several of the lab's repositories install this folder editable, so
they run whatever is checked out here. ``tests/test_public_api.py`` checks these tables.

The tables only ever grow; the one thing that may shrink is the list of parameters a call requires.
When a check fails, the change broke a caller: keep the old name, position or keyword working (an
alias, a new parameter added keyword-only at the end with a default, a FutureWarning for a release
before a behavior change) rather than editing the entry.

``PUBLIC_NAMES`` is every public name as of 26.10.1. The other tables come from a scan of the lab's
notebooks and scripts in November 2026, when 26.10.1 was current. To find usage added since, run
``python tests/scan_downstream_usage.py <folder> [<folder> ...]`` over the folders holding those
repositories; it lists what is in use that these tables do not pin yet.

Paths are relative to the package: ``"normalization.median_normalize"`` is
``proteomics_toolkit.normalization.median_normalize``, and a path without a module, such as
``"median_normalize"``, is the name re-exported from ``proteomics_toolkit`` itself.
"""

# Every public function and class each module defines, and the package's ``__all__`` (key ""), as
# of 26.10.1. Each must still exist at that path, and the "" names must stay in ``__all__``, since
# ``from proteomics_toolkit import *`` binds only what ``__all__`` lists.
PUBLIC_NAMES = {
    "": (
        "analyze_negative_values apply_systematic_color_scheme BATCH_SUFFIX_DELIMITER calculate_temporal_means"
        " classification classify_samples classify_trend_pattern clean_sample_names cluster_proteins_kmeans"
        " cluster_temporal_trends compute_paired_fold_changes compute_shap_values ControlSampleError"
        " create_config_dict_from_notebook_vars create_sample_column_mapping data_import datasets detect_batch_suffix"
        " display_analysis_summary enhanced_sample_processing enrichment EnrichmentConfig export"
        " export_analysis_results export_complete_analysis export_results export_significant_proteins_summary"
        " export_timestamped_config filter_significant_proteins generate_sample_matching_diagnostic_report"
        " get_available_libraries get_intensity_trend_points handle_negative_values inter_vs_intra_group_variance"
        " load_diann_data load_fasta_sequences load_prism_data load_prism_peptide_data load_skyline_data"
        " loess_normalize mad_normalize marker_discovery median_normalize merge_enrichment_results"
        " merge_with_statistics method_specificity_score multiclass_feature_importance multivariate"
        " name_clusters_by_pattern normalization parse_enrichr_results parse_protein_identifiers permanova"
        " plot_box_plot plot_cluster_heatmap plot_cluster_parallel_coordinates plot_comparative_pca"
        " plot_control_correlation plot_control_correlation_analysis plot_control_group_correlation_analysis"
        " plot_cv_distribution plot_enrichment_barplot plot_enrichment_comparison plot_fold_change_pca"
        " plot_grouped_heatmap plot_grouped_trajectories plot_identifications_per_sample"
        " plot_individual_control_pool_analysis plot_intensity_distributions plot_missing_value_heatmap"
        " plot_normalization_comparison plot_pca plot_pca_loadings plot_peptide_coverage_map plot_protein_profile"
        " plot_regression_scatter plot_roc_comparison plot_roc_curve plot_sample_clustermap plot_selection_frequency"
        " plot_shap_summary plot_umap plot_variance_vs_intensity plot_variance_vs_peptide_count plot_volcano"
        " preprocessing quantile_normalize query_enrichr regression relabel_features_with_genes rlr_normalize"
        " run_binary_classification run_comprehensive_statistical_analysis run_differential_enrichment"
        " run_elasticnet_regression run_enrichment_analysis run_enrichment_by_cluster run_enrichment_by_group"
        " run_moderated_linear_model run_rfecv_stability run_statistical_analysis run_temporal_analysis"
        " SampleMatchingError select_features_by_mad statistical_analysis StatisticalConfig strip_batch_suffix"
        " temporal_clustering TemporalClusteringConfig validate_metadata_data_consistency validation visualization"
        " vsn_normalize z_score_normalize"
    ),
    "classification": (
        "compute_shap_values multiclass_feature_importance plot_fold_change_pca plot_roc_comparison plot_roc_curve"
        " plot_selection_frequency plot_shap_summary relabel_features_with_genes run_binary_classification"
        " run_rfecv_stability select_features_by_mad"
    ),
    "data_import": (
        "clean_description clean_sample_names create_sample_column_mapping detect_batch_suffix"
        " identify_and_classify_controls identify_sample_columns load_diann_data load_fasta_sequences load_prism_data"
        " load_prism_peptide_data load_skyline_data match_samples_to_metadata parse_gene_from_description"
        " parse_uniprot_identifier strip_batch_suffix"
    ),
    "datasets": "load_example_data load_example_sequences",
    "enrichment": (
        "EnrichmentConfig get_available_libraries merge_enrichment_results parse_enrichr_results"
        " plot_enrichment_barplot plot_enrichment_comparison query_enrichr run_differential_enrichment"
        " run_enrichment_analysis run_enrichment_by_group"
    ),
    "export": (
        "create_config_dict_from_notebook_vars export_analysis_results export_complete_analysis export_results"
        " export_significant_proteins_summary export_timestamped_config"
    ),
    "marker_discovery": "cluster_proteins_kmeans inter_vs_intra_group_variance method_specificity_score",
    "multivariate": "permanova",
    "normalization": (
        "analyze_negative_values calculate_detailed_normalization_stats calculate_normalization_stats"
        " get_normalization_characteristics handle_negative_values is_normalization_log_transformed loess_normalize"
        " log_transform mad_normalize median_normalize quantile_normalize rlr_normalize vsn_normalize z_score_normalize"
    ),
    "preprocessing": (
        "apply_systematic_color_scheme assess_data_completeness calculate_group_colors classify_samples"
        " create_standard_data_structure filter_proteins_by_completeness identify_annotation_columns"
        " parse_gene_and_description parse_protein_identifiers"
    ),
    "regression": "plot_regression_scatter run_elasticnet_regression",
    "statistical_analysis": (
        "apply_multiple_testing_correction compute_paired_fold_changes display_analysis_summary export_results"
        " get_intensity_trend_points prepare_metadata_dataframe run_comprehensive_statistical_analysis"
        " run_mann_whitney_test run_mixed_effects_analysis run_moderated_linear_model run_paired_t_test"
        " run_statistical_analysis run_unpaired_t_test run_wilcoxon_test StatisticalConfig"
    ),
    "temporal_clustering": (
        "calculate_temporal_means classify_all_patterns classify_trend_pattern cluster_temporal_trends"
        " determine_optimal_clusters filter_significant_proteins get_week_columns merge_with_statistics"
        " name_clusters_by_pattern parse_enrichr_results plot_cluster_heatmap plot_cluster_parallel_coordinates"
        " plot_enrichment_barplot plot_enrichment_comparison plot_silhouette_analysis query_enrichr"
        " run_enrichment_by_cluster run_temporal_analysis TemporalClusteringConfig"
    ),
    "validation": (
        "ControlSampleError enhanced_sample_processing generate_sample_matching_diagnostic_report SampleMatchingError"
        " validate_metadata_data_consistency"
    ),
    "visualization": (
        "plot_box_plot plot_comparative_pca plot_control_correlation plot_control_correlation_analysis"
        " plot_control_cv_distribution plot_control_group_correlation_analysis plot_cv_distribution"
        " plot_grouped_heatmap plot_grouped_trajectories plot_identifications_per_sample"
        " plot_individual_control_pool_analysis plot_intensity_distributions plot_missing_value_heatmap"
        " plot_normalization_comparison plot_pca plot_pca_loadings plot_peptide_coverage_map plot_protein_profile"
        " plot_sample_clustermap plot_sample_correlation_heatmap plot_sample_correlation_triangular_heatmap plot_umap"
        " plot_variance_vs_intensity plot_variance_vs_peptide_count plot_volcano"
    ),
}

# Names callers read without calling them.
DOWNSTREAM_ATTRIBUTES = {
    "": "__version__ BATCH_SUFFIX_DELIMITER SampleMatchingError",
    "enrichment": "LIBRARY_COLORS",
}

# Calls made downstream: path -> (the parameters callers fill by position, in order; the keywords
# callers pass; the parameters the callable required, with no default, when the entry was added). The
# first two are what the scanned calls pass between them, not what each one passes, so a parameter
# outside the third may be left out by some caller: it must never become required. A required one
# may become optional; drop it from the third then. The private names are here because callers use
# them: Skyline-PRISM references its priors to _fit_intensity_trend_prior and
# _fit_count_dependent_prior, and notebooks import the rest.
DOWNSTREAM_CALLS = {
    "analyze_negative_values": ("", "data normalization_method sample_columns", "data normalization_method"),
    "classify_samples": (
        "sample_metadata",
        "apply_systematic_colors control_column control_labels group_column group_labels sample_metadata"
        " systematic_color_palette",
        "sample_metadata group_column group_labels control_column control_labels",
    ),
    "cluster_proteins_kmeans": (
        "data sample_columns",
        "k k_range log_transform n_init random_state standardize",
        "data sample_columns",
    ),
    "compute_paired_fold_changes": ("protein_data sample_metadata config", "", "protein_data sample_metadata config"),
    "compute_shap_values": ("model X", "feature_names", "model X"),
    "display_analysis_summary": (
        "differential_results config",
        "config differential_results label_top_n",
        "differential_results config",
    ),
    "enhanced_sample_processing": (
        "",
        "control_column control_labels group_column group_labels metadata protein_data strict_validation"
        " toolkit_module",
        "metadata protein_data group_column group_labels control_column control_labels toolkit_module",
    ),
    "EnrichmentConfig": ("", "enrichr_libraries min_genes pvalue_cutoff top_n", ""),
    "export_complete_analysis": (
        "",
        "analysis_description config_dict differential_results filtered_data normalized_data output_prefix"
        " sample_metadata",
        "normalized_data sample_metadata config_dict",
    ),
    "generate_sample_matching_diagnostic_report": ("validation_results", "", "validation_results"),
    "get_intensity_trend_points": ("results_df", "", "results_df"),
    "inter_vs_intra_group_variance": (
        "data sample_columns sample_metadata",
        "group_column log_transform min_per_group",
        "data sample_columns sample_metadata group_column",
    ),
    "load_prism_data": ("parquet_file metadata_file", "", "parquet_file"),
    "load_prism_peptide_data": ("parquet_file", "peptide_cols", "parquet_file"),
    "load_skyline_data": ("", "metadata_file peptide_file protein_file", "protein_file metadata_file"),
    "method_specificity_score": (
        "data sample_columns sample_metadata",
        "group_column log_transform",
        "data sample_columns sample_metadata group_column",
    ),
    "parse_enrichr_results": ("results config", "", "results"),
    "permanova": (
        "data sample_columns sample_metadata",
        "factor log_transform metric n_permutations random_state",
        "data sample_columns sample_metadata factor",
    ),
    "plot_box_plot": (
        "data sample_columns sample_metadata",
        "data figsize group_colors group_order log_transform sample_columns sample_metadata title",
        "data sample_columns sample_metadata",
    ),
    "plot_control_correlation": (
        "",
        "cluster control_columns data group_colors sample_metadata title",
        "data control_columns",
    ),
    "plot_control_correlation_analysis": (
        "",
        "control_column control_labels median_normalized_data original_data sample_columns sample_metadata"
        " vsn_normalized_data",
        "original_data median_normalized_data vsn_normalized_data sample_columns sample_metadata",
    ),
    "plot_enrichment_barplot": ("enrichment_df", "title top_n", "enrichment_df"),
    "plot_enrichment_comparison": ("enrichment_dict", "title top_n_per_group", "enrichment_dict"),
    "plot_fold_change_pca": (
        "fold_change_matrix group_labels",
        "group_colors title",
        "fold_change_matrix group_labels",
    ),
    "plot_pca": (
        "data sample_columns sample_metadata",
        "data figsize group_colors log_transform sample_columns sample_metadata title",
        "data sample_columns sample_metadata",
    ),
    "plot_pca_loadings": ("data sample_columns", "annotation_column log_transform top_n", "data sample_columns"),
    "plot_regression_scatter": ("results", "title", "results"),
    "plot_roc_comparison": ("results_dict", "title", "results_dict"),
    "plot_roc_curve": ("classification_results", "color title", "classification_results"),
    "plot_selection_frequency": ("results", "title top_n", "results"),
    "plot_shap_summary": ("explanation", "max_display plot_type title", "explanation"),
    "plot_variance_vs_intensity": ("results", "figsize", "results"),
    "plot_volcano": (
        "differential_df",
        "alpha axis_label_fontsize differential_df enable_pvalue_fallback fc_threshold figsize gene_column"
        " label_fontsize label_top_n legend_loc normalization_method p_threshold point_size tick_label_fontsize title"
        " use_adjusted_pvalue",
        "differential_df",
    ),
    "query_enrichr": ("gene_list config", "description", "gene_list"),
    "run_binary_classification": (
        "fold_change_matrix group_labels",
        "annotations cv_method feature_proteins feature_selection fold_change_matrix gene_col group_labels id_col"
        " method n_top_features return_model",
        "fold_change_matrix group_labels",
    ),
    "run_comprehensive_statistical_analysis": (
        "normalized_data sample_metadata config",
        "config normalized_data protein_annotations sample_metadata",
        "normalized_data sample_metadata config",
    ),
    "run_differential_enrichment": (
        "results_df",
        "config gene_column logfc_column logfc_threshold pvalue_column pvalue_threshold results_df",
        "results_df",
    ),
    "run_elasticnet_regression": (
        "data target",
        "annotations gene_col id_col log_transform n_jobs n_permutations outer_cv random_state",
        "data target",
    ),
    "run_enrichment_analysis": ("gene_list", "config description", "gene_list"),
    "run_moderated_linear_model": ("feature_data metadata_df config", "", "feature_data metadata_df config"),
    "run_rfecv_stability": (
        "data labels",
        "annotations consensus_threshold estimator gene_col id_col inner_cv log_transform n_jobs n_permutations"
        " outer_cv prefilter_top_var random_state step",
        "data labels",
    ),
    "StatisticalConfig": ("", "", ""),
    "strip_batch_suffix": ("column_names", "", "column_names"),
    "validate_metadata_data_consistency": (
        "",
        "control_column control_labels metadata metadata_sample_names protein_columns verbose",
        "metadata metadata_sample_names protein_columns control_column control_labels",
    ),
    "classification.compute_shap_values": ("", "X annotations feature_names gene_col id_col model", "model X"),
    "classification.plot_roc_comparison": ("", "figsize results_dict title", "results_dict"),
    "classification.plot_shap_summary": ("explanation", "max_display plot_type title", "explanation"),
    "classification.run_binary_classification": (
        "",
        "annotations cv_method feature_proteins feature_selection fold_change_matrix gene_col group_labels id_col"
        " method n_top_features return_model",
        "fold_change_matrix group_labels",
    ),
    "classification.select_features_by_mad": (
        "fold_change_matrix",
        "fold_change_matrix n_top_features",
        "fold_change_matrix",
    ),
    "data_import.clean_description": ("protein_description", "", "protein_description"),
    "data_import.clean_sample_names": ("sample_columns", "auto_detect common_prefix common_suffix", "sample_columns"),
    "data_import.identify_and_classify_controls": ("sample_metadata", "", "sample_metadata"),
    "data_import.identify_sample_columns": ("data metadata", "", "data metadata"),
    "data_import.load_prism_data": ("", "metadata_file parquet_file", "parquet_file"),
    "data_import.load_skyline_data": ("", "metadata_file peptide_file protein_file", "protein_file metadata_file"),
    "data_import.match_samples_to_metadata": (
        "cleaned_sample_names metadata",
        "include_unmatched",
        "cleaned_sample_names metadata",
    ),
    "data_import.parse_gene_from_description": ("protein_description", "", "protein_description"),
    "data_import.parse_uniprot_identifier": ("protein_id", "", "protein_id"),
    "data_import.strip_batch_suffix": ("column_names", "", "column_names"),
    "enrichment.EnrichmentConfig": ("", "enrichr_libraries min_genes pvalue_cutoff top_n", ""),
    "enrichment.get_available_libraries": ("", "", ""),
    "enrichment.merge_enrichment_results": ("enrichment_dict", "", "enrichment_dict"),
    "enrichment.parse_enrichr_results": ("results config", "", "results"),
    "enrichment.plot_enrichment_barplot": ("enrichment_df", "figsize title top_n", "enrichment_df"),
    "enrichment.plot_enrichment_comparison": ("enrichment_dict", "figsize title top_n_per_group", "enrichment_dict"),
    "enrichment.query_enrichr": ("gene_list config", "", "gene_list"),
    "enrichment.run_differential_enrichment": (
        "results_df",
        "config logfc_threshold pvalue_threshold verbose",
        "results_df",
    ),
    "enrichment.run_enrichment_analysis": ("gene_list config", "verbose", "gene_list"),
    "enrichment.run_enrichment_by_group": (
        "data_df",
        "config gene_column group_column verbose",
        "data_df group_column",
    ),
    "export._add_annotations_to_normalized_data": (
        "",
        "annotation_cols filtered_data normalized_data",
        "normalized_data filtered_data annotation_cols",
    ),
    "export.create_config_dict_from_notebook_vars": (
        "",
        "FORCE_CATEGORICAL additional_interactions analysis_description analysis_type control_column control_labels"
        " covariates enable_pvalue_fallback export_results final_sample_columns fold_change_threshold group_colors"
        " group_column group_labels group_order handle_negatives interaction_terms label_top_proteins log_base"
        " log_pseudocount log_transform_before_stats metadata_file min_detection_rate min_positive_replacement"
        " min_samples_per_group negative_handling_method normalization_method optimize_vsn output_prefix"
        " p_value_threshold paired_column paired_label1 paired_label2 peptide_file protein_file q_value_max"
        " random_seed remove_common_prefix statistical_test_method subject_column systematic_color_palette"
        " toolkit_path use_adjusted_pvalue use_systematic_colors",
        "",
    ),
    "export.export_analysis_results": (
        "",
        "normalized_data output_prefix sample_metadata",
        "normalized_data sample_metadata",
    ),
    "export.export_complete_analysis": (
        "",
        "analysis_description config_dict differential_results filtered_data normalized_data output_prefix"
        " sample_metadata",
        "normalized_data sample_metadata config_dict",
    ),
    "export.export_significant_proteins_summary": (
        "",
        "config_dict differential_results output_prefix",
        "differential_results config_dict",
    ),
    "export.export_timestamped_config": (
        "",
        "analysis_description computed_values config_dict output_prefix",
        "config_dict",
    ),
    "marker_discovery.inter_vs_intra_group_variance": (
        "",
        "annotation_columns data group_column log_transform min_per_group sample_columns sample_metadata",
        "data sample_columns sample_metadata group_column",
    ),
    "multivariate.permanova": (
        "",
        "data factor log_transform metric n_permutations random_state sample_columns sample_metadata",
        "data sample_columns sample_metadata factor",
    ),
    "normalization.analyze_negative_values": (
        "data",
        "normalization_method sample_columns",
        "data normalization_method",
    ),
    "normalization.calculate_detailed_normalization_stats": (
        "original_data normalized_data",
        "control_labels method normalized_data original_data sample_metadata",
        "original_data normalized_data method",
    ),
    "normalization.calculate_normalization_stats": (
        "data normalized_data method",
        "method",
        "data normalized_data method",
    ),
    "normalization.get_normalization_characteristics": ("", "", ""),
    "normalization.handle_negative_values": ("data", "data method min_positive_value sample_columns", "data"),
    "normalization.is_normalization_log_transformed": ("normalization_method", "", "normalization_method"),
    "normalization.loess_normalize": ("data", "sample_columns span", "data"),
    "normalization.log_transform": ("data", "base", "data"),
    "normalization.mad_normalize": ("data sample_columns", "sample_columns", "data"),
    "normalization.median_normalize": ("data sample_columns", "sample_columns", "data"),
    "normalization.quantile_normalize": ("data", "sample_columns", "data"),
    "normalization.rlr_normalize": ("data sample_columns", "sample_columns", "data"),
    "normalization.vsn_normalize": ("data", "optimize_params sample_columns", "data"),
    "normalization.z_score_normalize": ("data sample_columns", "sample_columns", "data"),
    "preprocessing._normalize_group_value": ("value", "", "value"),
    "preprocessing.apply_systematic_color_scheme": (
        "sample_metadata",
        "control_labels group_labels systematic_color_palette",
        "sample_metadata group_labels control_labels systematic_color_palette",
    ),
    "preprocessing.assess_data_completeness": (
        "data sample_columns sample_metadata",
        "",
        "data sample_columns sample_metadata",
    ),
    "preprocessing.calculate_group_colors": ("sample_metadata", "", "sample_metadata"),
    "preprocessing.classify_samples": (
        "sample_metadata",
        "control_column control_labels group_column group_labels",
        "sample_metadata group_column group_labels control_column control_labels",
    ),
    "preprocessing.create_standard_data_structure": ("data cleaned_sample_names", "cleaned_sample_names data", "data"),
    "preprocessing.filter_proteins_by_completeness": (
        "data sample_columns min_detection_rate",
        "min_detection_rate",
        "data sample_columns",
    ),
    "preprocessing.identify_annotation_columns": ("data", "", "data"),
    "preprocessing.parse_gene_and_description": ("data", "", "data"),
    "preprocessing.parse_protein_identifiers": ("data", "protein_col", "data"),
    "statistical_analysis._apply_log_transformation_if_needed": ("data config", "", "data config"),
    "statistical_analysis._create_empty_mixed_effects_result": ("protein_idx reason", "", "protein_idx reason"),
    "statistical_analysis._create_empty_result": ("protein_idx reason", "", "protein_idx reason"),
    "statistical_analysis._fit_count_dependent_prior": ("fit counts", "", "fit counts"),
    "statistical_analysis._fit_intensity_trend_prior": (
        "fit raw_feature_data metadata_df config",
        "",
        "fit raw_feature_data metadata_df config",
    ),
    "statistical_analysis.apply_multiple_testing_correction": ("results_df config", "", "results_df config"),
    "statistical_analysis.display_analysis_summary": (
        "",
        "config differential_results label_top_n",
        "differential_results config",
    ),
    "statistical_analysis.prepare_metadata_dataframe": (
        "sample_metadata_dict sample_columns config",
        "",
        "sample_metadata_dict sample_columns config",
    ),
    "statistical_analysis.run_comprehensive_statistical_analysis": (
        "normalized_data sample_metadata config",
        "config normalized_data protein_annotations sample_metadata",
        "normalized_data sample_metadata config",
    ),
    "statistical_analysis.run_mann_whitney_test": (
        "protein_data metadata_df config",
        "",
        "protein_data metadata_df config",
    ),
    "statistical_analysis.run_paired_t_test": (
        "protein_data metadata_df config",
        "",
        "protein_data metadata_df config",
    ),
    "statistical_analysis.run_unpaired_t_test": (
        "protein_data metadata_df config",
        "",
        "protein_data metadata_df config",
    ),
    "statistical_analysis.run_wilcoxon_test": (
        "protein_data metadata_df config",
        "",
        "protein_data metadata_df config",
    ),
    "statistical_analysis.StatisticalConfig": ("", "", ""),
    "temporal_clustering.calculate_temporal_means": (
        "",
        "data_df metadata_dict subject_column week_column",
        "data_df metadata_dict",
    ),
    "temporal_clustering.classify_trend_pattern": ("trajectory", "", "trajectory"),
    "temporal_clustering.cluster_temporal_trends": (
        "temporal_df week_columns config",
        "config temporal_df week_columns",
        "temporal_df week_columns",
    ),
    "temporal_clustering.determine_optimal_clusters": (
        "X_scaled",
        "X_scaled k_range plot prefer_higher_k random_seed",
        "X_scaled",
    ),
    "temporal_clustering.filter_significant_proteins": ("merged_df config", "", "merged_df"),
    "temporal_clustering.get_week_columns": ("temporal_df", "", "temporal_df"),
    "temporal_clustering.merge_with_statistics": (
        "temporal_df stats_df cluster_labels cluster_names",
        "",
        "temporal_df stats_df cluster_labels cluster_names",
    ),
    "temporal_clustering.name_clusters_by_pattern": (
        "X_scaled labels week_columns",
        "",
        "X_scaled labels week_columns",
    ),
    "temporal_clustering.plot_silhouette_analysis": (
        "X_scaled labels",
        "cluster_names title",
        "X_scaled labels cluster_names",
    ),
    "temporal_clustering.run_temporal_analysis": (
        "",
        "config data_df metadata_dict output_prefix run_enrichment stats_df treatment_name",
        "data_df metadata_dict stats_df treatment_name",
    ),
    "temporal_clustering.TemporalClusteringConfig": (
        "",
        "alpha auto_detect_clusters clustering_method cmap enrichment_pvalue_cutoff enrichment_top_n enrichr_libraries"
        " figsize linewidth max_clusters min_clusters min_fold_change min_genes_for_enrichment n_clusters"
        " p_value_threshold random_seed use_adjusted_pvalue",
        "",
    ),
    "visualization.plot_box_plot": (
        "",
        "data figsize group_colors group_order log_transform sample_columns sample_metadata title",
        "data sample_columns sample_metadata",
    ),
    "visualization.plot_comparative_pca": (
        "",
        "figsize group_colors group_order median_normalized_data original_data sample_columns sample_metadata"
        " vsn_normalized_data",
        "original_data median_normalized_data vsn_normalized_data sample_columns sample_metadata",
    ),
    "visualization.plot_control_correlation_analysis": (
        "",
        "control_column control_labels figsize median_normalized_data original_data sample_columns sample_metadata"
        " vsn_normalized_data",
        "original_data median_normalized_data vsn_normalized_data sample_columns sample_metadata",
    ),
    "visualization.plot_control_cv_distribution": (
        "",
        "control_column control_labels cv_threshold data figsize normalization_method sample_columns sample_metadata"
        " title_suffix",
        "data sample_columns sample_metadata control_column control_labels",
    ),
    "visualization.plot_individual_control_pool_analysis": (
        "",
        "control_column control_labels data figsize normalization_method sample_columns sample_metadata",
        "data sample_columns sample_metadata",
    ),
    "visualization.plot_normalization_comparison": (
        "",
        "method normalized_data original_data sample_columns",
        "original_data normalized_data sample_columns",
    ),
    "visualization.plot_pca": (
        "",
        "data figsize group_colors log_transform sample_columns sample_metadata title",
        "data sample_columns sample_metadata",
    ),
    "visualization.plot_sample_clustermap": (
        "data",
        "cmap col_cluster data figsize group_colors group_column label_column legend_title method metric row_cluster"
        " sample_columns sample_metadata show_col_labels show_row_labels title vmax vmin zscore",
        "data sample_columns",
    ),
    "visualization.plot_sample_correlation_heatmap": (
        "data",
        "figsize sample_columns sample_metadata",
        "data sample_columns sample_metadata",
    ),
    "visualization.plot_sample_correlation_triangular_heatmap": (
        "",
        "annot_fontsize data figsize group_colors group_column label_fontsize method sample_columns sample_metadata"
        " show_clustering",
        "data sample_columns sample_metadata",
    ),
    "visualization.plot_variance_vs_intensity": ("results", "figsize", "results"),
    "visualization.plot_volcano": (
        "",
        "differential_df enable_pvalue_fallback fc_threshold figsize label_fontsize label_top_n legend_loc"
        " normalization_method p_threshold title use_adjusted_pvalue",
        "differential_df",
    ),
}

# Attributes callers set or read on a config object built with no arguments. Setting an attribute the
# config does not have raises nothing and changes nothing, so renaming one silently drops the setting.
CONFIG_ATTRIBUTES = {
    "enrichment.EnrichmentConfig": "enrichr_libraries min_genes pvalue_cutoff rate_limit_delay timeout top_n",
    "statistical_analysis.StatisticalConfig": (
        "additional_interactions analysis_type assume_equal_variance correction_method covariates dose_column"
        " enable_pvalue_fallback fold_change_threshold force_categorical group_column group_labels interaction_terms"
        " log_base log_pseudocount log_transform_before_stats moderation normalization_method p_value_threshold"
        " paired_column paired_label1 paired_label2 peptide_count_column robust statistical_test_method subject_column"
        " time_column use_adjusted_pvalue validate variance_prior_group_column variance_prior_groups"
    ),
    "temporal_clustering.TemporalClusteringConfig": (
        "auto_detect_clusters clustering_method enrichr_libraries max_clusters min_clusters min_fold_change n_clusters"
        " p_value_threshold random_seed use_adjusted_pvalue"
    ),
}
