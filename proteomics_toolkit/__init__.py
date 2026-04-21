"""
Proteomics Analysis Toolkit
===========================

A comprehensive Python library for analyzing mass spectrometry-based proteomics data,
particularly designed for Skyline quantitation outputs. This toolkit provides a complete
workflow from data import through statistical analysis and visualization.

QUICK START EXAMPLE:
-------------------
    import proteomics_toolkit as ptk

    # 1. Load data
    protein_data, metadata, _ = ptk.load_skyline_data('proteins.csv', 'metadata.csv')

    # 2. Process and normalize
    processed = ptk.parse_protein_identifiers(protein_data)
    normalized = ptk.median_normalize(processed, sample_columns=['Sample1', 'Sample2'])

    # 3. Statistical analysis
    config = ptk.StatisticalConfig()
    results = ptk.run_comprehensive_statistical_analysis(normalized, sample_metadata, config)

    # 4. Visualization and export
    ptk.plot_volcano(results)
    ptk.export_complete_analysis(normalized, sample_metadata, config, results)

MODULE OVERVIEW:
===============

data_import
    Purpose: Load and parse Skyline quantitation files and metadata
    Key functions: load_skyline_data(), clean_sample_names()
    Use when: Starting analysis, need to load protein/peptide data

preprocessing
    Purpose: Data cleaning, quality assessment, and sample classification
    Key functions: parse_protein_identifiers(), classify_samples(), filter_proteins_by_completeness()
    Use when: Need to clean data, assess completeness, classify control vs study samples

normalization
    Purpose: Various normalization methods to reduce technical variation
    Key functions: median_normalize(), vsn_normalize(), quantile_normalize()
    Use when: Raw data needs normalization before statistical analysis

statistical_analysis
    Purpose: Differential analysis, mixed-effects models, and statistical testing
    Key functions: run_comprehensive_statistical_analysis(), StatisticalConfig()
    Use when: Performing statistical comparisons between groups

visualization
    Purpose: Quality control plots, results visualization, and publication-ready figures
    Key functions: plot_volcano(), plot_box_plot(), plot_pca()
    Use when: Need QC plots, want to visualize results, create figures

temporal_clustering
    Purpose: K-means clustering of temporal protein trends and gene set enrichment
    Key functions: TemporalAnalyzer, TemporalClusteringConfig
    Use when: Analyzing longitudinal proteomics data with multiple time points

validation
    Purpose: Data validation, error checking, and diagnostic reporting
    Key functions: validate_metadata_data_consistency(), enhanced_sample_processing()
    Use when: Need to validate metadata matches data, troubleshoot sample mismatches

export
    Purpose: Export results, configurations, and create reproducible analysis records
    Key functions: export_complete_analysis(), export_timestamped_config()
    Use when: Saving results, creating configuration backups, sharing analysis

TYPICAL WORKFLOW:
================
1. ptk.load_skyline_data() - Load protein data and metadata
2. ptk.validate_metadata_data_consistency() - Validate data consistency
3. ptk.parse_protein_identifiers() - Clean and annotate protein data
4. ptk.classify_samples() - Identify control vs study samples
5. ptk.median_normalize() - Normalize data (or other normalization methods)
6. ptk.run_comprehensive_statistical_analysis() - Perform statistical analysis
7. ptk.plot_volcano() - Visualize results
8. ptk.export_complete_analysis() - Export everything for reproducibility

ERROR HANDLING:
==============
The toolkit includes comprehensive validation with clear error messages:
- SampleMatchingError: When samples in metadata don't match protein data
- ControlSampleError: When control samples are missing or misspecified
- Use ptk.validate_metadata_data_consistency() to diagnose issues early
"""

# =============================================================================
# MODULE IMPORTS - Core functionality organized by analysis stage
# =============================================================================

from . import (
    classification,  # Binary classification and fold-change PCA
    data_import,  # Data loading and parsing
    datasets,  # Bundled example datasets
    enrichment,  # Gene set enrichment analysis
    export,  # Results export and configuration management
    normalization,  # Normalization methods
    preprocessing,  # Data cleaning and quality assessment
    statistical_analysis,  # Statistical testing and modeling
    temporal_clustering,  # Temporal trend analysis and clustering
    validation,  # Data validation and error checking
    visualization,  # Plotting and visualization
)

__version__ = "26.2.0"
__author__ = "Michael MacCoss Lab, University of Washington"

# =============================================================================
# CONVENIENCE IMPORTS - Most commonly used functions available at top level
# =============================================================================

# DATA LOADING - Essential functions for starting any analysis
# CLASSIFICATION - Binary classification and fold-change PCA
from .classification import (
    plot_fold_change_pca,  # PCA of per-subject fold-changes by group
    plot_roc_comparison,  # Overlay ROC curves from multiple methods
    plot_roc_curve,  # ROC curve from classification results
    run_binary_classification,  # LOO/k-fold CV binary classification
)
from .data_import import (
    BATCH_SUFFIX_DELIMITER,  # The batch suffix delimiter: __@__
    clean_sample_names,  # Clean up sample column names automatically
    create_sample_column_mapping,  # Map metadata sample names to data columns
    detect_batch_suffix,  # Detect skyline-prism batch suffix in column names
    load_diann_data,  # Load DIA-NN report.pg_matrix.tsv protein-group matrix
    load_prism_data,  # Load PRISM parquet output (corrected_proteins.parquet)
    load_prism_peptide_data,  # Load PRISM peptide parquet (corrected_peptides.parquet)
    load_skyline_data,  # Main function: Load protein/peptide data + metadata
    strip_batch_suffix,  # Strip batch suffix to get short sample names
)

# GENE SET ENRICHMENT - General-purpose enrichment analysis via Enrichr API
from .enrichment import (
    EnrichmentConfig,  # Configuration for enrichment analysis
    get_available_libraries,  # List available Enrichr libraries
    merge_enrichment_results,  # Merge multiple enrichment DataFrames
    parse_enrichr_results,  # Parse raw Enrichr results to DataFrame
    plot_enrichment_barplot,  # Bar plot of enrichment results
    plot_enrichment_comparison,  # Compare enrichment across groups
    query_enrichr,  # Query Enrichr API directly
    run_differential_enrichment,  # Run enrichment on up/down-regulated genes
    run_enrichment_analysis,  # Run enrichment on a gene list
    run_enrichment_by_group,  # Run enrichment for groups (clusters, categories, etc.)
)

# DATA EXPORT - Save results and create reproducible analysis records
from .export import (
    create_config_dict_from_notebook_vars,  # Create config from notebook variables
    export_analysis_results,  # Export data files only
    export_complete_analysis,  # Main function: Export everything (data + config)
    export_results,  # General results export function
    export_significant_proteins_summary,  # Export summary of significant results
    export_timestamped_config,  # Export configuration with timestamp
)

# NORMALIZATION - All normalization methods for reducing technical variation
from .normalization import (
    analyze_negative_values,  # Analyze negative value patterns
    handle_negative_values,  # Handle negative values from VSN normalization
    loess_normalize,  # Advanced: LOESS intensity-dependent correction
    mad_normalize,  # Robust: Median absolute deviation normalization
    median_normalize,  # Most common: Simple, robust median normalization
    quantile_normalize,  # Strong: Force identical sample distributions
    rlr_normalize,  # Advanced: Robust linear regression
    vsn_normalize,  # Advanced: Variance stabilizing normalization
    z_score_normalize,  # Standard: Z-score standardization
)

# DATA PREPROCESSING - Core data preparation functions
from .preprocessing import (
    apply_systematic_color_scheme,  # Apply consistent colors for visualization
    classify_samples,  # Classify samples into study vs control groups
    parse_protein_identifiers,  # Extract UniProt IDs, gene names, descriptions
)

# STATISTICAL ANALYSIS - Core statistical functions and configuration
from .statistical_analysis import (
    StatisticalConfig,  # Configuration class for analysis parameters
    compute_paired_fold_changes,  # Per-subject fold-change matrix for paired designs
    display_analysis_summary,  # Display analysis results summary
    run_comprehensive_statistical_analysis,  # Main function: Complete statistical analysis
    run_statistical_analysis,  # Lower-level statistical analysis function
)

# TEMPORAL CLUSTERING - Longitudinal trend analysis and gene set enrichment
from .temporal_clustering import (
    TemporalClusteringConfig,  # Configuration for temporal analysis
    calculate_temporal_means,  # Calculate mean abundance at each timepoint
    classify_trend_pattern,  # Classify individual protein trends
    cluster_temporal_trends,  # Cluster proteins by temporal patterns
    filter_significant_proteins,  # Filter to statistically significant proteins
    merge_with_statistics,  # Merge temporal data with statistical results
    name_clusters_by_pattern,  # Assign descriptive names to clusters
    plot_cluster_heatmap,  # Heatmap visualization of clusters (temporal-specific)
    plot_cluster_parallel_coordinates,  # Parallel coordinate plots for clusters (temporal-specific)
    run_enrichment_by_cluster,  # Run enrichment analysis per cluster (temporal-specific)
    run_temporal_analysis,  # Main function: Complete temporal analysis pipeline
)

# DATA VALIDATION - Comprehensive data validation and error handling
from .validation import (
    ControlSampleError,  # Exception: Control sample configuration issues
    SampleMatchingError,  # Exception: Samples don't match between files
    enhanced_sample_processing,  # Sample processing with built-in validation
    generate_sample_matching_diagnostic_report,  # Detailed diagnostic reports
    validate_metadata_data_consistency,  # Main validation: Check metadata vs data consistency
)

# VISUALIZATION - Publication-ready plots and quality control visualizations
# GROUPED VISUALIZATIONS - General-purpose heatmaps and trajectory plots
from .visualization import (
    plot_box_plot,  # QC plot: Sample intensity distributions
    plot_comparative_pca,  # Advanced: Compare multiple normalization methods
    plot_control_correlation,  # QC plot: Simple control sample correlation heatmap
    plot_control_correlation_analysis,  # QC plot: Control sample correlation analysis
    plot_control_group_correlation_analysis,  # QC plot: Group-wise control correlations
    plot_cv_distribution,  # QC plot: CV distribution across all samples (or by group)
    plot_grouped_heatmap,  # Heatmap for any grouped data
    plot_grouped_trajectories,  # Line plots for grouped trajectories
    plot_identifications_per_sample,  # QC plot: #features identified per sample (bar)
    plot_individual_control_pool_analysis,  # Detailed: Individual control pool analysis
    plot_intensity_distributions,  # QC plot: Density overlay of intensity per sample
    plot_missing_value_heatmap,  # QC plot: Missing-value pattern across samples x features
    plot_normalization_comparison,  # QC plot: Before/after normalization comparison
    plot_pca,  # QC plot: Principal component analysis
    plot_peptide_coverage_map,  # QC plot: Peptide positions along a protein sequence
    plot_protein_profile,  # Single protein profile plot
    plot_volcano,  # Main results plot: Volcano plot for differential analysis
)

# =============================================================================
# PUBLIC API - All functions available for import
# =============================================================================

__all__ = [
    # MODULES - Full module access for advanced workflows
    "data_import",  # Complete data import functionality
    "datasets",  # Bundled example datasets
    "preprocessing",  # Complete preprocessing functionality
    "normalization",  # All normalization methods
    "statistical_analysis",  # Complete statistical analysis suite
    "visualization",  # All plotting and visualization functions
    "temporal_clustering",  # Temporal trend analysis and clustering
    "enrichment",  # Gene set enrichment analysis
    "validation",  # Data validation and error handling
    "export",  # Results export and configuration management
    "classification",  # Binary classification and fold-change PCA
    # DATA LOADING - Start here for any analysis
    "load_skyline_data",  # ESSENTIAL: Load protein data + metadata
    "load_prism_data",  # Load PRISM parquet output (corrected_proteins.parquet)
    "load_prism_peptide_data",  # Load PRISM peptide parquet (corrected_peptides.parquet)
    "load_diann_data",  # Load DIA-NN report.pg_matrix.tsv protein-group matrix
    "clean_sample_names",  # Clean and standardize sample column names
    "detect_batch_suffix",  # Detect skyline-prism batch suffix in column names
    "strip_batch_suffix",  # Strip batch suffix to get short sample names
    "create_sample_column_mapping",  # Map metadata sample names to data columns
    "BATCH_SUFFIX_DELIMITER",  # The batch suffix delimiter: __@__
    # PREPROCESSING - Data preparation and quality control
    "parse_protein_identifiers",  # Extract protein annotations (UniProt, Gene, etc.)
    "classify_samples",  # Separate control vs study samples
    "apply_systematic_color_scheme",  # Consistent colors for groups
    # NORMALIZATION - Choose the best method for your data
    "median_normalize",  # Good Starting Point: Robust, preserves scale
    "vsn_normalize",  # Use for highly variable samples: Handles heteroscedasticity
    "quantile_normalize",  # Strong normalization (identical distributions)
    "mad_normalize",  # Robust to outliers
    "z_score_normalize",  # Standardize to mean=0, std=1
    "rlr_normalize",  # Robust linear regression
    "loess_normalize",  # LOESS intensity-dependent correction
    "handle_negative_values",  # Handle VSN negative values
    "analyze_negative_values",  # Analyze negative value patterns
    # STATISTICAL ANALYSIS - The core of differential analysis
    "run_comprehensive_statistical_analysis",  # MAIN FUNCTION: Complete analysis
    "display_analysis_summary",  # Show results summary
    "StatisticalConfig",  # Configuration for statistical analysis
    "run_statistical_analysis",  # Lower-level analysis function
    "compute_paired_fold_changes",  # Per-subject fold-change matrix
    # VALIDATION - Catch errors early, get helpful diagnostics
    "validate_metadata_data_consistency",  # RECOMMENDED: Validate before analysis
    "enhanced_sample_processing",  # Sample processing with validation
    "generate_sample_matching_diagnostic_report",  # Detailed error diagnostics
    "SampleMatchingError",  # Custom exception for sample mismatches
    "ControlSampleError",  # Custom exception for control sample issues
    # EXPORT - Save your work and create reproducible records
    "export_complete_analysis",  # MAIN FUNCTION: Export everything
    "export_analysis_results",  # Export data files only
    "export_timestamped_config",  # Export configuration backup
    "create_config_dict_from_notebook_vars",  # Create config from notebook variables
    "export_significant_proteins_summary",  # Export significant results summary
    "export_results",  # General export function
    # VISUALIZATION - Publication-ready plots and QC
    "plot_volcano",  # ESSENTIAL: Volcano plot (main results plot)
    "plot_box_plot",  # ESSENTIAL: Sample distributions (QC)
    "plot_normalization_comparison",  # Before/after normalization (QC)
    "plot_pca",  # Principal component analysis (QC)
    "plot_comparative_pca",  # Compare normalization methods
    "plot_control_correlation",  # Simple control correlation heatmap
    "plot_control_correlation_analysis",  # Control sample QC
    "plot_control_group_correlation_analysis",  # Group-wise control QC
    "plot_individual_control_pool_analysis",  # Detailed control analysis
    "plot_grouped_heatmap",  # Heatmap for any grouped data
    "plot_grouped_trajectories",  # Trajectory plots for groups
    "plot_protein_profile",  # Single protein profile
    "plot_missing_value_heatmap",  # QC: missing-value pattern (proteins or peptides)
    "plot_identifications_per_sample",  # QC: #features identified per sample
    "plot_intensity_distributions",  # QC: density overlay of intensity per sample
    "plot_cv_distribution",  # QC: CV distribution across all samples (or by group)
    "plot_peptide_coverage_map",  # QC: peptide positions along a protein sequence
    # TEMPORAL CLUSTERING - Longitudinal trend analysis
    "TemporalClusteringConfig",  # Configuration for temporal analysis
    "run_temporal_analysis",  # MAIN FUNCTION: Complete temporal analysis pipeline
    "calculate_temporal_means",  # Calculate mean abundance at each timepoint
    "cluster_temporal_trends",  # Cluster proteins by temporal patterns
    "name_clusters_by_pattern",  # Assign descriptive names to clusters
    "classify_trend_pattern",  # Classify individual protein trends
    "merge_with_statistics",  # Merge temporal data with statistical results
    "filter_significant_proteins",  # Filter to significant proteins only
    "run_enrichment_by_cluster",  # Run enrichment analysis per cluster
    "plot_cluster_heatmap",  # Heatmap visualization of clusters (temporal-specific)
    "plot_cluster_parallel_coordinates",  # Parallel coordinate plots for clusters (temporal-specific)
    # GENE SET ENRICHMENT - General-purpose enrichment analysis
    "EnrichmentConfig",  # Configuration for enrichment analysis
    "query_enrichr",  # Query Enrichr API directly
    "parse_enrichr_results",  # Parse raw Enrichr results
    "run_enrichment_analysis",  # Run enrichment on a gene list
    "run_enrichment_by_group",  # Run enrichment by group (clusters, categories)
    "run_differential_enrichment",  # Enrichment on up/down-regulated genes
    "plot_enrichment_barplot",  # Bar plot of enrichment results
    "plot_enrichment_comparison",  # Compare enrichment across groups
    "get_available_libraries",  # List available Enrichr libraries
    "merge_enrichment_results",  # Merge multiple enrichment DataFrames
    # CLASSIFICATION - Binary group discrimination
    "run_binary_classification",  # LOO/k-fold CV binary classification
    "plot_fold_change_pca",  # PCA of per-subject fold-changes by group
    "plot_roc_curve",  # ROC curve from classification results
    "plot_roc_comparison",  # Overlay ROC curves from multiple methods
]
