"""
Export Module for Proteomics Analysis Toolkit

This module handles exporting analysis results, configurations, and processed data
from proteomics experiments. It provides functions for creating timestamped
configuration files and exporting data with proper annotations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def export_analysis_results(
    normalized_data: pd.DataFrame,
    sample_metadata: Dict[str, Dict[str, Any]],
    differential_results: Optional[pd.DataFrame] = None,
    filtered_data: Optional[pd.DataFrame] = None,
    output_prefix: str = "proteomics_analysis",
    annotation_cols: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Export comprehensive analysis results including normalized data, metadata,
    and annotated differential results.

    Parameters:
    -----------
    normalized_data : pd.DataFrame
        Normalized protein intensity data
    sample_metadata : dict
        Dictionary mapping sample names to their metadata
    differential_results : pd.DataFrame, optional
        Statistical analysis results
    filtered_data : pd.DataFrame, optional
        Original filtered data with annotations
    output_prefix : str
        Prefix for output filenames
    annotation_cols : list, optional
        List of annotation columns to include

    Returns:
    --------
    dict
        Dictionary of exported files
    """

    print("Exporting analysis results...")

    exported_files = {}

    # Default annotation columns for standardized data structure
    if annotation_cols is None:
        # Use the exact 5 columns from our standardized structure
        annotation_cols = [
            "Protein",
            "Description",
            "Protein Gene",
            "UniProt_Accession",
            "UniProt_Entry_Name",
        ]

    # Export normalized data with annotations
    normalized_file = f"{output_prefix}_normalized_data.csv"

    # Add annotations to normalized data if filtered_data is available
    if filtered_data is not None:
        annotated_normalized_data = _add_annotations_to_normalized_data(normalized_data, filtered_data, annotation_cols)
        annotated_normalized_data.to_csv(normalized_file, index=False)
        print(f"Normalized data (with annotations) exported to: {normalized_file}")
    else:
        normalized_data.to_csv(normalized_file)
        print(f"Normalized data exported to: {normalized_file}")

    exported_files["normalized_data"] = normalized_file

    # Export sample metadata
    metadata_export = pd.DataFrame.from_dict(sample_metadata, orient="index")
    metadata_export.index.name = "Sample_ID"  # Add header for the sample names column
    metadata_file = f"{output_prefix}_sample_metadata.csv"
    metadata_export.to_csv(metadata_file)
    exported_files["sample_metadata"] = metadata_file
    print(f"Sample metadata exported to: {metadata_file}")

    # Export differential results with annotations if available
    if differential_results is not None and not differential_results.empty and filtered_data is not None:
        annotated_results = _add_annotations_to_results(differential_results, filtered_data, annotation_cols)

        differential_file = f"{output_prefix}_differential_results_annotated.csv"
        annotated_results.to_csv(differential_file, index=False)
        exported_files["differential_results"] = differential_file
        print(f"Differential analysis results (with annotations) exported to: {differential_file}")

        # Show preview
        _display_results_preview(annotated_results)

    elif differential_results is not None and not differential_results.empty:
        # Export results without annotations
        differential_file = f"{output_prefix}_differential_results.csv"
        differential_results.to_csv(differential_file, index=False)
        exported_files["differential_results"] = differential_file
        print(f"Differential analysis results exported to: {differential_file}")

    return exported_files


def _add_annotations_to_results(
    differential_results: pd.DataFrame,
    filtered_data: pd.DataFrame,
    annotation_cols: List[str],
) -> pd.DataFrame:
    """
    Add protein annotations to differential analysis results.

    Parameters:
    -----------
    differential_results : pd.DataFrame
        Statistical analysis results
    filtered_data : pd.DataFrame
        Original data with annotations
    annotation_cols : list
        List of annotation columns to include

    Returns:
    --------
    pd.DataFrame
        Results with annotations added
    """

    # Get available annotation columns
    available_annotation_cols = [col for col in annotation_cols if col in filtered_data.columns]
    print(f"Adding annotation columns: {available_annotation_cols}")

    # Create annotations dataframe
    protein_annotations = filtered_data[available_annotation_cols].copy()

    # Merge differential results with annotations
    differential_results_annotated = differential_results.merge(protein_annotations, on="Protein", how="left")

    # Reorder columns to put annotations first
    annotation_first_cols = [col for col in available_annotation_cols if col in differential_results_annotated.columns]
    statistical_cols = [col for col in differential_results_annotated.columns if col not in annotation_first_cols]

    # Create final column order: annotations first, then statistical results
    final_column_order = annotation_first_cols + statistical_cols
    return differential_results_annotated[final_column_order]


def _add_annotations_to_normalized_data(
    normalized_data: pd.DataFrame,
    filtered_data: pd.DataFrame,
    annotation_cols: List[str],
) -> pd.DataFrame:
    """
    Add protein annotations to normalized data with CLEAN column ordering.

    EXACTLY THE COLUMN ORDER REQUESTED:
    1) Protein
    2) Description (the short description)
    3) Protein Gene
    4) UniProt_Accession
    5) UniProt_Entry_Name
    6-N) Sample Data Columns (cleaned names with shared prefix/suffix removed)

    Parameters:
    -----------
    normalized_data : pd.DataFrame
        Normalized protein intensity data
    filtered_data : pd.DataFrame
        Original data with annotations
    annotation_cols : list
        List of annotation columns to include

    Returns:
    --------
    pd.DataFrame
        Normalized data with annotations in the EXACT order specified
    """

    print("Creating CLEAN normalized data with proper column ordering...")

    # With standardized data structure, sample columns start at index 5
    if len(normalized_data.columns) > 5:
        sample_cols = list(normalized_data.columns[5:])  # Everything after first 5 annotation columns
        print(f"Using standardized structure: {len(sample_cols)} sample columns (columns 6+)")
    else:
        # Fallback for legacy data
        numeric_cols = normalized_data.select_dtypes(include=[np.number]).columns.tolist()
        sample_cols = [col for col in numeric_cols if col not in annotation_cols]
        print(f"Using legacy detection: {len(sample_cols)} sample columns")

    # Validate data alignment
    if len(normalized_data) != len(filtered_data):
        raise ValueError(
            f"Mismatch: normalized_data has {len(normalized_data)} rows, filtered_data has {len(filtered_data)} rows"
        )

    # Start with a fresh DataFrame in the EXACT order requested
    result_data = pd.DataFrame()

    # EXACT COLUMN ORDER AS REQUESTED:

    # 1) Protein
    if "Protein" in filtered_data.columns:
        result_data["Protein"] = filtered_data["Protein"].tolist()
        print("Added: Protein")
    else:
        print("⚠ Warning: Protein column not found")

    # 2) Description (short description) - use the CLEAN name "Description"
    if "Description" in filtered_data.columns:
        result_data["Description"] = filtered_data["Description"].tolist()
        print("Added: Description")
    elif "Protein Description" in filtered_data.columns:
        result_data["Description"] = filtered_data["Protein Description"].tolist()
        print("Added: Description (from 'Protein Description')")
    else:
        print("⚠ Warning: Description column not found")

    # 3) Protein Gene - keep the name "Protein Gene" as requested
    if "Protein Gene" in filtered_data.columns:
        result_data["Protein Gene"] = filtered_data["Protein Gene"].tolist()
        print("Added: Protein Gene")
    elif "Gene" in filtered_data.columns:
        result_data["Protein Gene"] = filtered_data["Gene"].tolist()
        print("Added: Protein Gene (from 'Gene')")
    else:
        print("⚠ Warning: Gene column not found")

    # 4) UniProt_Accession
    if "UniProt_Accession" in filtered_data.columns:
        result_data["UniProt_Accession"] = filtered_data["UniProt_Accession"].tolist()
        print("Added: UniProt_Accession")
    else:
        print("⚠ Warning: UniProt_Accession column not found")

    # 5) UniProt_Entry_Name
    if "UniProt_Entry_Name" in filtered_data.columns:
        result_data["UniProt_Entry_Name"] = filtered_data["UniProt_Entry_Name"].tolist()
        print("Added: UniProt_Entry_Name")
    else:
        print("⚠ Warning: UniProt_Entry_Name column not found")

    # 6-N) Sample Data Columns (cleaned names with shared prefix/suffix removed)
    for col in sample_cols:
        if col in normalized_data.columns:
            result_data[col] = normalized_data[col].tolist()

    annotation_count = len(
        [
            col
            for col in [
                "Protein",
                "Description",
                "Protein Gene",
                "UniProt_Accession",
                "UniProt_Entry_Name",
            ]
            if col in result_data.columns
        ]
    )

    print(f"Done: Final column order: {annotation_count} annotations + {len(sample_cols)} samples")
    print("Done: CLEAN structure: No redundant or scattered columns!")
    print(f"Column order: {list(result_data.columns[:5])}... + {len(sample_cols)} sample columns")
    print("Note: NO redundant columns like 'Gene', 'UniProt_Database', extra 'Description' etc.")

    return result_data


def _display_results_preview(annotated_results: pd.DataFrame) -> None:
    """Display a preview of exported results."""

    print(f"\nExported results preview (columns: {list(annotated_results.columns[:8])}...):")
    print(f"Total proteins: {len(annotated_results)}")

    if len(annotated_results) > 0:
        print("Sample of first few entries:")
        preview_cols = [
            col
            for col in [
                "Protein",
                "Gene",
                "UniProt_Accession",
                "logFC",
                "P.Value",
                "adj.P.Val",
            ]
            if col in annotated_results.columns
        ]
        # Use pandas display if available, otherwise print
        try:
            from IPython.display import display

            display(annotated_results[preview_cols].head(3))
        except ImportError:
            print(annotated_results[preview_cols].head(3))


def export_timestamped_config(
    config_dict: Dict[str, Any],
    output_prefix: str = "proteomics_analysis",
    analysis_description: str = "Proteomics analysis",
    computed_values: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Export analysis configuration as a timestamped Python file.

    Parameters:
    -----------
    config_dict : dict
        Dictionary containing all configuration parameters
    output_prefix : str
        Prefix for the configuration filename
    analysis_description : str
        Description of the analysis type
    computed_values : dict, optional
        Additional computed values to include as comments

    Returns:
    --------
    str
        Path to the exported configuration file
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_file = f"{output_prefix}_config_{timestamp}.py"

    print(f"Exporting analysis configuration to: {config_file}")

    with open(config_file, "w", encoding="utf-8") as f:
        # Write header
        f.write("# =============================================================================\n")
        f.write("# PROTEOMICS ANALYSIS CONFIGURATION\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Analysis: {analysis_description}\n")
        f.write("# =============================================================================\n\n")

        # Define configuration sections with their possible parameters
        section_configs = [
            (
                "INPUT FILES AND PATHS",
                [
                    "toolkit_path",
                    "metadata_file",
                    "protein_file",
                    "peptide_file",
                    "remove_common_prefix",
                ],
            ),
            ("DATA FILTERING PARAMETERS", ["min_detection_rate", "min_samples_per_group"]),
            ("NORMALIZATION STRATEGY", ["normalization_method", "optimize_vsn"]),
            (
                "NEGATIVE VALUE HANDLING STRATEGY",
                [
                    "handle_negatives",
                    "negative_handling_method",
                    "min_positive_replacement",
                ],
            ),
            (
                "LOG TRANSFORMATION SETTINGS",
                [
                    "log_transform_before_stats",
                    "log_base",
                    "log_pseudocount",
                ],
            ),
            (
                "STATISTICAL ANALYSIS STRATEGY",
                ["statistical_test_method", "analysis_type"],
            ),
            (
                "EXPERIMENTAL DESIGN CONFIGURATION",
                [
                    "subject_column",
                    "paired_column",
                    "paired_label1",
                    "paired_label2",
                    "group_column",
                    "group_labels",
                    "FORCE_CATEGORICAL",
                ],
            ),
            (
                "MIXED-EFFECTS MODEL CONFIGURATION",
                ["interaction_terms", "additional_interactions", "covariates"],
            ),
            ("CONTROL SAMPLE CONFIGURATION", ["control_column", "control_labels"]),
            (
                "VISUALIZATION SETTINGS",
                [
                    "use_systematic_colors",
                    "systematic_color_palette",
                    "group_order",
                    "group_colors",
                ],
            ),
            (
                "SIGNIFICANCE THRESHOLDS",
                [
                    "p_value_threshold",
                    "fold_change_threshold",
                    "q_value_max",
                    "use_adjusted_pvalue",
                    "enable_pvalue_fallback",
                ],
            ),
            (
                "OUTPUT AND EXPORT SETTINGS",
                [
                    "export_results",
                    "output_prefix",
                    "label_top_proteins",
                    "random_seed",
                ],
            ),
        ]

        # Only write sections that have at least one parameter present in config_dict
        section_counter = 1
        for section_name, param_names in section_configs:
            # Check if any parameters from this section exist in the config
            params_in_section = [p for p in param_names if p in config_dict]

            if params_in_section:
                _write_config_section(f, section_name, config_dict, params_in_section, section_counter)
                section_counter += 1

        # Write computed values as comments
        if computed_values:
            f.write("# =============================================================================\n")
            f.write("# COMPUTED VALUES (for reference)\n")
            f.write("# =============================================================================\n")

            for key, value in computed_values.items():
                if key == "group_colors" and isinstance(value, dict):
                    f.write("# Group colors assigned:\n")
                    for group, color in value.items():
                        f.write(f"#   {group}: {color}\n")
                else:
                    f.write(f"# {key}: {value}\n")

    return config_file


def _write_config_section(
    file_handle,
    section_name: str,
    config_dict: Dict[str, Any],
    param_names: List[str],
    section_number: int = 1,
) -> None:
    """Write a configuration section to file."""

    file_handle.write("# =============================================================================\n")
    file_handle.write(f"# {section_number}. {section_name}\n")
    file_handle.write("# =============================================================================\n")

    for param in param_names:
        if param in config_dict:
            value = config_dict[param]
            file_handle.write(f"{param} = {repr(value)}\n")

    file_handle.write("\n")


def export_complete_analysis(
    normalized_data: pd.DataFrame,
    sample_metadata: Dict[str, Dict[str, Any]],
    config_dict: Dict[str, Any],
    differential_results: Optional[pd.DataFrame] = None,
    filtered_data: Optional[pd.DataFrame] = None,
    output_prefix: str = "proteomics_analysis",
    analysis_description: str = "Comprehensive proteomics analysis",
) -> Dict[str, str]:
    """
    Export complete analysis including data, results, and timestamped configuration.

    This is the main export function that combines data export and configuration export.

    Parameters:
    -----------
    normalized_data : pd.DataFrame
        Normalized protein intensity data
    sample_metadata : dict
        Dictionary mapping sample names to their metadata
    config_dict : dict
        Complete configuration dictionary
    differential_results : pd.DataFrame, optional
        Statistical analysis results
    filtered_data : pd.DataFrame, optional
        Original filtered data with annotations
    output_prefix : str
        Prefix for output filenames
    analysis_description : str
        Description for the configuration header

    Returns:
    --------
    dict
        Dictionary of all exported files
    """

    # Export data files
    exported_files = export_analysis_results(
        normalized_data=normalized_data,
        sample_metadata=sample_metadata,
        differential_results=differential_results,
        filtered_data=filtered_data,
        output_prefix=output_prefix,
    )

    # Prepare computed values for config
    computed_values = {}
    if "group_colors" in config_dict:
        computed_values["group_colors"] = config_dict.get("group_colors")
    if filtered_data is not None:
        computed_values["Total proteins analyzed"] = len(filtered_data)
    if "final_sample_columns" in config_dict:
        computed_values["Total samples"] = len(config_dict.get("final_sample_columns", []))
    if "formula" in config_dict:
        computed_values["Model formula"] = config_dict.get("formula")

    # Export timestamped configuration
    config_file = export_timestamped_config(
        config_dict=config_dict,
        output_prefix=output_prefix,
        analysis_description=analysis_description,
        computed_values=computed_values,
    )

    exported_files["configuration"] = config_file

    # Print comprehensive summary
    _print_export_summary(exported_files, config_file)

    return exported_files


def _print_export_summary(exported_files: Dict[str, str], config_file: str) -> None:
    """Print a comprehensive summary of exported files."""

    print("\n" + "=" * 60)
    print("Done: All analysis results and configuration exported successfully!")
    print("Files created:")

    if "normalized_data" in exported_files:
        print(f"  • {exported_files['normalized_data']} - Normalized protein data")
    if "sample_metadata" in exported_files:
        print(f"  • {exported_files['sample_metadata']} - Sample metadata")
    if "differential_results" in exported_files:
        print(f"  • {exported_files['differential_results']} - Differential results with annotations")
    if "configuration" in exported_files:
        print(f"  • {exported_files['configuration']} - Python configuration (timestamped)")

    print("=" * 60)

    print("\nCONFIGURATION REPRODUCIBILITY:")
    print("The complete analysis configuration has been saved as a Python file")
    print("with timestamp for easy identification and reuse:")
    print("")
    print("Python Configuration (timestamped):")
    print(f"   {config_file}")
    print("   - Complete analysis settings in Python format")
    print("   - Ready to copy variables to recreate this exact analysis")
    print("   - Easy to parse, edit, and version control")
    print("   - Can be imported: exec(open('config_file.py').read())")
    print("")

    if "differential_results" in exported_files:
        print("Annotated Results:")
        print(f"   {exported_files['differential_results']}")
        print("   - Includes Protein, Gene, UniProt_Accession, Description")
        print("   - No row indices - protein annotations are proper columns")
        print("   - Ready for publication or downstream analysis")

    print("=" * 60)

    print("\nREPRODUCIBILITY TIP:")
    print("To reproduce this analysis:")
    print(f"1. Copy the configuration variables from: {config_file}")
    print("2. Paste them into the configuration cell of a new notebook")
    print("3. Run the analysis with identical settings")
    print(f"4. Or import directly: exec(open('{config_file}').read())")


def create_config_dict_from_notebook_vars(**kwargs) -> Dict[str, Any]:
    """
    Create a configuration dictionary from notebook variables.

    This function takes all the configuration variables from a notebook
    and packages them into a dictionary suitable for export. Only includes
    the parameters that are actually provided, making it generalizable to
    different analysis types.

    Parameters:
    -----------
    **kwargs : various
        All configuration variables from the notebook

    Returns:
    --------
    dict
        Configuration dictionary containing only provided parameters
    """

    # Define minimal defaults only for essential parameters
    minimal_defaults = {
        # Only the most basic file path defaults
        "toolkit_path": ".",
        # Everything else should come from the user's configuration
    }

    # Start with minimal defaults
    config_dict = minimal_defaults.copy()

    # Add only the parameters that were actually provided
    # This ensures the export only contains what the user configured
    config_dict.update(kwargs)

    return config_dict


def export_significant_proteins_summary(
    differential_results: pd.DataFrame,
    config_dict: Dict[str, Any],
    output_prefix: str = "proteomics_analysis",
) -> str:
    """
    Export a summary of significant proteins with key statistics.

    Parameters:
    -----------
    differential_results : pd.DataFrame
        Statistical analysis results
    config_dict : dict
        Configuration parameters
    output_prefix : str
        Prefix for output filename

    Returns:
    --------
    str
        Path to exported summary file
    """

    p_threshold = config_dict.get("p_value_threshold", 0.05)
    fc_threshold = config_dict.get("fold_change_threshold", 1.5)

    # Filter to significant proteins
    significant_results = differential_results[differential_results["adj.P.Val"] < p_threshold]

    if len(significant_results) == 0:
        print("No significant proteins found - skipping summary export")
        return ""

    # Create summary
    summary_file = f"{output_prefix}_significant_proteins_summary.csv"

    # Select key columns for summary
    summary_cols = [
        "Protein",
        "Gene",
        "UniProt_Accession",
        "Description",
        "logFC",
        "P.Value",
        "adj.P.Val",
    ]
    available_cols = [col for col in summary_cols if col in significant_results.columns]

    summary_data = significant_results[available_cols].copy()

    # Add significance categories
    summary_data["Regulation"] = summary_data["logFC"].apply(
        lambda x: "Up" if x > fc_threshold else ("Down" if x < -fc_threshold else "Unchanged")
    )

    # Sort by significance
    summary_data = summary_data.sort_values("adj.P.Val")

    # Export
    summary_data.to_csv(summary_file, index=False)
    print(f"Significant proteins summary exported to: {summary_file}")
    print(f"  • Total significant: {len(summary_data)}")
    print(f"  • Upregulated: {(summary_data['Regulation'] == 'Up').sum()}")
    print(f"  • Downregulated: {(summary_data['Regulation'] == 'Down').sum()}")

    return summary_file


# Backwards compatibility functions
def export_results(differential_df: pd.DataFrame, output_file: str, include_all: bool = True) -> None:
    """
    Export differential analysis results to CSV file.

    This is a backwards compatibility function. Use export_analysis_results for new code.

    Parameters:
    -----------
    differential_df : pd.DataFrame
        Differential analysis results
    output_file : str
        Output CSV filename
    include_all : bool
        Whether to include all proteins or only significant ones
    """

    if not include_all:
        # Check for 'Significant' column or use adj.P.Val threshold
        if "Significant" in differential_df.columns:
            export_df = differential_df[differential_df["Significant"]].copy()
        elif "adj.P.Val" in differential_df.columns:
            export_df = differential_df[differential_df["adj.P.Val"] < 0.05].copy()
        else:
            export_df = differential_df.copy()
        print(f"Exporting {len(export_df)} significant proteins to {output_file}")
    else:
        export_df = differential_df.copy()
        print(f"Exporting all {len(export_df)} proteins to {output_file}")

    export_df.to_csv(output_file, index=False)
    print("Results exported successfully!")
