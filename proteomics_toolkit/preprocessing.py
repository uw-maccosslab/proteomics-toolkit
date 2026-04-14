"""
Data Preprocessing Module for Proteomics Analysis Toolkit

Functions for data cleaning, quality assessment, and protein annotation parsing.
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _normalize_group_value(value: Any) -> Union[int, float, str]:
    """
    Normalize group values to consistent types for sorting and comparison.

    Keeps numeric values as numbers when possible, only converts to string
    when necessary for non-numeric values.

    Parameters:
    -----------
    value : any
        The group value to normalize

    Returns:
    --------
    int, float, or str
        Normalized value
    """

    # Handle None, empty string, or NaN
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"

    # If it's already a number (int or float), keep it as a number
    if isinstance(value, (int, float)):
        # Convert float integers to int (80.0 -> 80)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        else:
            return value

    # If it's a string, try to convert to number if possible
    if isinstance(value, str):
        try:
            # Try integer first
            if "." not in value:
                return int(value)
            else:
                # Try float
                float_val = float(value)
                if float_val.is_integer():
                    return int(float_val)
                else:
                    return float_val
        except ValueError:
            # Not a number, return as string
            return value

    # For any other type, convert to string
    return str(value)


def parse_protein_identifiers(data: pd.DataFrame, protein_col: str = "Protein") -> pd.DataFrame:
    """
    Parse UniProt identifiers and add annotation columns.

    Parameters:
    -----------
    data : pd.DataFrame
        Protein data with identifier column
    protein_col : str
        Name of column containing protein identifiers

    Returns:
    --------
    pd.DataFrame : Data with added UniProt annotation columns
    """

    def parse_uniprot_identifier(protein_id: str) -> Dict[str, str]:
        """Parse single UniProt identifier"""
        if pd.isna(protein_id):
            return {"accession": "", "database": "", "entry_name": ""}

        protein_id = str(protein_id).strip()

        # Pattern: sp|P12345|PROT_HUMAN or tr|Q9ABC1|Q9ABC1_MOUSE
        match = re.match(r"^(sp|tr)\|([A-Z0-9]+)\|([A-Z0-9_]+)$", protein_id)
        if match:
            db = "SwissProt" if match.group(1) == "sp" else "TrEMBL"
            return {
                "accession": match.group(2),
                "database": db,
                "entry_name": match.group(3),
            }

        # If no match, try to find accession pattern
        acc_match = re.search(r"([A-Z][A-Z0-9]{5,9})", protein_id)
        if acc_match:
            return {"accession": acc_match.group(1), "database": "", "entry_name": ""}

        return {"accession": "", "database": "", "entry_name": ""}

    print("=== PARSING PROTEIN IDENTIFIERS ===\n")

    # Parse identifiers
    parsed_proteins = data[protein_col].apply(parse_uniprot_identifier)
    protein_info = pd.DataFrame(parsed_proteins.tolist())

    # Add to original data
    result_data = data.copy()
    result_data["UniProt_Accession"] = protein_info["accession"]
    result_data["UniProt_Database"] = protein_info["database"]
    result_data["UniProt_Entry_Name"] = protein_info["entry_name"]

    # Show parsing results
    total_proteins = len(result_data)
    has_accession = (result_data["UniProt_Accession"] != "").sum()
    has_database = (result_data["UniProt_Database"] != "").sum()

    print(f"Total proteins: {total_proteins}")
    print(f"Accessions extracted: {has_accession} ({has_accession / total_proteins * 100:.1f}%)")
    print(f"Database identified: {has_database} ({has_database / total_proteins * 100:.1f}%)")

    if has_database > 0:
        db_counts = result_data["UniProt_Database"].value_counts()
        print(f"Database distribution: {dict(db_counts)}")

    return result_data


def parse_gene_and_description(data: pd.DataFrame) -> pd.DataFrame:
    """
    Parse gene names and clean descriptions from protein annotations.

    Parameters:
    -----------
    data : pd.DataFrame
        Protein data with annotation columns

    Returns:
    --------
    pd.DataFrame : Data with cleaned Gene and Description columns
    """

    def parse_gene_from_description(protein_description: str) -> str:
        """Extract gene name from description using GN= pattern"""
        if pd.isna(protein_description):
            return ""
        desc_str = str(protein_description).strip()
        gene_match = re.search(r"GN=([^=\s]+)", desc_str)
        return gene_match.group(1).strip() if gene_match else ""

    def clean_description(protein_description: str) -> str:
        """Clean description by removing UniProt fields"""
        if pd.isna(protein_description):
            return ""
        desc_str = str(protein_description).strip()
        # Remove patterns like OS=..., OX=..., GN=..., PE=..., SV=...
        clean_desc = re.sub(r"\s+(?:OS|OX|GN|PE|SV)=[^=]*(?=\s+(?:OS|OX|GN|PE|SV)=|$)", "", desc_str)
        return clean_desc.strip()

    print("=== PARSING GENE NAMES AND DESCRIPTIONS ===\n")

    result_data = data.copy()

    # Find existing gene column
    gene_col_found = None
    gene_columns = ["Protein Gene", "Gene", "Gene Symbol", "Gene Name"]
    for col in gene_columns:
        if col in result_data.columns:
            gene_col_found = col
            break

    # Find description column
    desc_col_found = None
    desc_columns = ["Protein Description", "Description", "Protein Name", "Name"]
    for col in desc_columns:
        if col in result_data.columns:
            desc_col_found = col
            break

    # Handle Gene column creation
    if gene_col_found:
        print(f"Found existing gene column: '{gene_col_found}' - using as primary Gene source")
        result_data["Gene"] = result_data[gene_col_found].fillna("")

        # Supplement missing genes from description if available
        if desc_col_found:
            missing_genes = result_data["Gene"] == ""
            if missing_genes.sum() > 0:
                print(f"Supplementing {missing_genes.sum()} missing genes from description parsing...")
                parsed_genes = result_data[desc_col_found].apply(parse_gene_from_description)
                result_data.loc[missing_genes, "Gene"] = parsed_genes[missing_genes]
    else:
        print("No existing gene column found")
        if desc_col_found:
            print(f"Parsing gene names from description column: '{desc_col_found}'")
            result_data["Gene"] = result_data[desc_col_found].apply(parse_gene_from_description)
        else:
            print("No description column found either - creating empty Gene column")
            result_data["Gene"] = ""

    # Handle Description cleaning
    if desc_col_found:
        print(f"Cleaning descriptions from column: '{desc_col_found}'")
        result_data["Description"] = result_data[desc_col_found].apply(clean_description)
    else:
        print("No description column found - creating empty Description column")
        result_data["Description"] = ""

    # Show results
    total_proteins = len(result_data)
    has_gene = (result_data["Gene"] != "").sum()
    has_desc = (result_data["Description"] != "").sum()

    print("\nFinal results:")
    print(f"Gene names available: {has_gene} ({has_gene / total_proteins * 100:.1f}%)")
    print(f"Descriptions cleaned: {has_desc} ({has_desc / total_proteins * 100:.1f}%)")

    return result_data


def create_standard_data_structure(data: pd.DataFrame, cleaned_sample_names: Dict[str, str] = None) -> pd.DataFrame:
    """
    Create standardized data structure with exactly 5 annotation columns followed by cleaned sample columns.

    Expected structure:
    1. Protein (original UniProt ID)
    2. Description (cleaned protein name)
    3. Protein Gene (gene symbol)
    4. UniProt_Accession (extracted accession)
    5. UniProt_Entry_Name (extracted entry name)
    6. Sample columns (with cleaned names)

    Parameters:
    -----------
    data : pd.DataFrame
        Input data that has been processed with parse_protein_identifiers() and parse_gene_and_description()
    cleaned_sample_names : Dict[str, str], optional
        Mapping from original sample names to cleaned names

    Returns:
    --------
    pd.DataFrame
        Standardized data structure with proper column order
    """

    print("=== CREATING STANDARD DATA STRUCTURE ===\n")

    # Required annotation columns in exact order
    required_annotation_cols = [
        "Protein",
        "Description",
        "Protein Gene",
        "UniProt_Accession",
        "UniProt_Entry_Name",
    ]

    # Check that all required annotation columns exist
    missing_cols = [col for col in required_annotation_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required annotation columns: {missing_cols}. "
            f"Please run parse_protein_identifiers() and parse_gene_and_description() first."
        )

    # Identify sample columns (all columns that aren't annotation columns)
    all_annotation_cols = [
        "Protein",
        "Protein Description",
        "Protein Gene",
        "Gene",
        "Description",
        "UniProt_Accession",
        "UniProt_Entry_Name",
        "UniProt_Database",
    ]

    sample_columns = [col for col in data.columns if col not in all_annotation_cols]

    print(f"Found {len(sample_columns)} sample columns")
    print(
        f"Sample column range: {sample_columns[0] if sample_columns else 'None'}"
        f" ... {sample_columns[-1] if sample_columns else 'None'}"
    )

    # Create the result dataframe with proper column order
    result_data = pd.DataFrame(index=data.index)

    # Add the 5 required annotation columns in order
    for col in required_annotation_cols:
        result_data[col] = data[col]

    # Handle sample column naming and ordering
    if cleaned_sample_names:
        print("Applying cleaned sample names...")
        # Apply cleaned names and add sample columns
        for original_name in sample_columns:
            cleaned_name = cleaned_sample_names.get(original_name, original_name)
            result_data[cleaned_name] = data[original_name]
            if original_name != cleaned_name:
                print(f"  {original_name} -> {cleaned_name}")
    else:
        # Use original sample column names
        for col in sample_columns:
            result_data[col] = data[col]

    # Verify the structure
    final_columns = list(result_data.columns)
    annotation_part = final_columns[:5]
    sample_part = final_columns[5:]

    print("\nFinal structure verification:")
    print(f"Total columns: {len(final_columns)}")
    print(f"Annotation columns (1-5): {annotation_part}")
    print(
        f"Sample columns ({len(sample_part)}): {sample_part[0] if sample_part else 'None'}"
        f" ... {sample_part[-1] if sample_part else 'None'}"
    )

    # Check that annotation columns are exactly what we expect
    if annotation_part != required_annotation_cols:
        raise ValueError(
            f"Annotation columns don't match expected order. "
            f"Got: {annotation_part}, Expected: {required_annotation_cols}"
        )

    print("✅ Data structure standardization complete!")

    return result_data


def assess_data_completeness(data: pd.DataFrame, sample_columns: List[str], sample_metadata: Dict[str, Dict]) -> None:
    """
    Assess and visualize data completeness across samples.

    Parameters:
    -----------
    data : pd.DataFrame
        Protein quantitation data
    sample_columns : List[str]
        List of sample column names
    sample_metadata : Dict[str, Dict]
        Sample metadata mapping
    """

    print("=== ASSESSING DATA COMPLETENESS ===\n")

    # Calculate completeness statistics
    sample_data = data[sample_columns]

    # Overall statistics
    total_values = sample_data.shape[0] * sample_data.shape[1]
    non_zero_values = (sample_data != 0).sum().sum()
    non_null_values = sample_data.notna().sum().sum()

    print("Data completeness summary:")
    print(f"Total possible values: {total_values:,}")
    print(f"Non-null values: {non_null_values:,} ({non_null_values / total_values * 100:.1f}%)")
    print(f"Non-zero values: {non_zero_values:,} ({non_zero_values / total_values * 100:.1f}%)")

    # Per-sample statistics
    print("\nPer-sample completeness:")
    for sample in sample_columns:
        non_zero = (data[sample] != 0).sum()
        total = len(data)

        group = sample_metadata.get(sample, {}).get("Group", "Unknown")
        print(f"{sample}: {non_zero}/{total} non-zero ({non_zero / total * 100:.1f}%) - Group: {group}")

    # Per-protein statistics
    print("\nProtein detection summary:")
    proteins_per_sample = (sample_data != 0).sum(axis=1)
    print(f"Proteins detected in all samples: {(proteins_per_sample == len(sample_columns)).sum()}")
    print(f"Proteins detected in >75% samples: {(proteins_per_sample > 0.75 * len(sample_columns)).sum()}")
    print(f"Proteins detected in >50% samples: {(proteins_per_sample > 0.5 * len(sample_columns)).sum()}")


def filter_proteins_by_completeness(
    data: pd.DataFrame, sample_columns: List[str], min_detection_rate: float = 0.5
) -> pd.DataFrame:
    """
    Filter proteins based on detection completeness across samples.

    Parameters:
    -----------
    data : pd.DataFrame
        Protein quantitation data
    sample_columns : List[str]
        List of sample column names
    min_detection_rate : float
        Minimum fraction of samples where protein must be detected (default: 0.5)

    Returns:
    --------
    pd.DataFrame : Filtered data
    """

    print("=== FILTERING PROTEINS BY COMPLETENESS ===\n")

    sample_data = data[sample_columns]

    # Calculate detection rates
    detection_rates = (sample_data != 0).sum(axis=1) / len(sample_columns)

    # Apply filter
    keep_proteins = detection_rates >= min_detection_rate
    filtered_data = data[keep_proteins].copy()

    print(f"Original proteins: {len(data)}")
    print(f"Proteins with ≥{min_detection_rate * 100:.0f}% detection rate: {len(filtered_data)}")
    print(f"Removed: {len(data) - len(filtered_data)} proteins")

    return filtered_data


def calculate_group_colors(
    sample_metadata: Dict[str, Dict],
) -> Tuple[Dict[str, str], pd.Series]:
    """
    Calculate colors for experimental groups and group counts.
    Orders control samples (pools) on the right with distinct colors.

    Parameters:
    -----------
    sample_metadata : Dict[str, Dict]
        Sample metadata mapping

    Returns:
    --------
    Tuple[Dict[str, str], pd.Series] : Group colors and counts
    """

    # Get unique groups, handling NaN values
    groups = []
    for meta in sample_metadata.values():
        group = meta.get("Group", "Unknown")
        # Convert NaN to 'Unknown'
        if pd.isna(group):
            group = "Unknown"
        groups.append(group)

    unique_groups = set(groups)

    # Count samples per group
    group_counts = pd.Series(groups).value_counts()

    # Separate control/pool samples from study samples
    control_groups = []
    study_groups = []

    for group in unique_groups:
        if pd.isna(group):
            continue
        group_lower = str(group).lower()
        if any(keyword in group_lower for keyword in ["pool", "control", "qc", "standard", "blank", "reference"]):
            control_groups.append(group)
        else:
            study_groups.append(group)

    # Sort groups: study groups first (alphabetically), then control groups (alphabetically)
    ordered_groups = sorted(study_groups) + sorted(control_groups)

    # Reorder group_counts to match this order
    group_counts = group_counts.reindex(ordered_groups, fill_value=0)

    # Check if we have meaningful groups
    meaningful_groups = len(unique_groups) > 1 or (len(unique_groups) == 1 and "Unknown" not in unique_groups)

    if meaningful_groups:
        # Use different color palettes for study vs control samples
        study_colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#e377c2",
            "#17becf",
            "#bcbd22",
        ]  # Standard colors for study groups

        # Distinct colors for different control types
        control_color_map = {
            "eisaipool": "#8B4513",  # SaddleBrown
            "gwpool": "#696969",  # DimGray
            "hoofpool": "#A0522D",  # Sienna
            "pool": "#708090",  # SlateGray
            "qc": "#2F4F4F",  # DarkSlateGray
            "control": "#556B2F",  # DarkOliveGreen
            "standard": "#800080",  # Purple
            "blank": "#191970",  # MidnightBlue
            "reference": "#8B008B",  # DarkMagenta
        }

        group_colors = {}

        # Assign colors to study groups
        for i, group in enumerate(study_groups):
            group_colors[group] = study_colors[i % len(study_colors)]

        # Assign specific colors to control groups
        for group in control_groups:
            group_lower = str(group).lower()
            color_assigned = False

            for keyword, color in control_color_map.items():
                if keyword in group_lower:
                    group_colors[group] = color
                    color_assigned = True
                    break

            # Fallback color if no specific keyword matched
            if not color_assigned:
                group_colors[group] = "#696969"  # DimGray as default control color

    else:
        # No meaningful groups - use grey
        group_colors = {group: "#7f7f7f" for group in unique_groups}

    return group_colors, group_counts


def identify_annotation_columns(data: pd.DataFrame) -> List[str]:
    """
    Identify columns containing protein annotation information.

    Parameters:
    -----------
    data : pd.DataFrame
        Protein data

    Returns:
    --------
    List[str] : List of annotation column names
    """

    # Common annotation column patterns
    annotation_patterns = [
        "protein",
        "gene",
        "description",
        "name",
        "accession",
        "entry",
        "symbol",
        "uniprot",
        "database",
        "organism",
    ]

    # Find columns that are likely annotations (non-numeric)
    annotation_columns = []

    for col in data.columns:
        col_lower = col.lower()

        # Check if column name matches annotation patterns
        if any(pattern in col_lower for pattern in annotation_patterns):
            annotation_columns.append(col)
        # Check if column is primarily non-numeric
        elif data[col].dtype == "object":
            # Check if most values are strings (not numbers as strings)
            try:
                pd.to_numeric(data[col], errors="raise")
            except (ValueError, TypeError):
                annotation_columns.append(col)

    return annotation_columns


def classify_samples(
    sample_metadata: Dict,
    group_column: str,
    group_labels: List[str],
    control_column: str,
    control_labels: List[str],
    apply_systematic_colors: bool = True,
    systematic_color_palette: str = "Set1",
) -> Tuple[Dict, List[str], List[str], Dict, Optional[Dict]]:
    """
    Classify samples into study groups and control groups based on configuration.
    Automatically applies systematic color scheme by default for consistent visualization.

    Parameters:
    -----------
    sample_metadata : dict
        Sample metadata dictionary with sample names as keys
    group_column : str
        Column name to use for study group classification
    group_labels : list of str
        Labels to use for study groups
    control_column : str
        Column name to use for control identification
    control_labels : list of str
        Labels in control_column that identify control samples
    apply_systematic_colors : bool, default True
        Whether to automatically apply systematic color scheme for consistent visualization
    systematic_color_palette : str, default "Set1"
        Color palette to use for systematic color assignment

    Returns:
    --------
    tuple: (group_distribution, control_samples, study_samples, corrected_metadata, group_colors)
        - group_distribution: dict with group names and sample counts
        - control_samples: list of control sample names
        - study_samples: list of study sample names
        - corrected_metadata: dict with corrected Group assignments for visualization
        - group_colors: dict mapping group names to colors (None if apply_systematic_colors=False)
    """

    print("SAMPLE CLASSIFICATION SUMMARY")
    print("=" * 50)

    # Separate control identification from study group classification
    control_samples = []
    study_samples = []
    group_distribution = {}
    corrected_sample_metadata = {}

    for sample_name, sample_info in sample_metadata.items():
        corrected_info = sample_info.copy()

        # First check if this is a control sample based on control configuration
        is_control = False
        if control_column in sample_info:
            sample_label = sample_info[control_column]
            if sample_label in control_labels:
                is_control = True
                control_samples.append(sample_name)
                # Use the control label directly as the group name and ensure it's a string
                sample_label = str(sample_label)
                if sample_label not in group_distribution:
                    group_distribution[sample_label] = 0
                group_distribution[sample_label] += 1

                # Tag controls with their specific control type for visualization
                corrected_info["Group"] = sample_label  # Use the control label as group

        if not is_control:
            # This is a study sample - use the group_column for classification
            study_samples.append(sample_name)
            if group_column in sample_info:
                study_group = sample_info[group_column]
                if pd.notna(study_group) and study_group != "" and study_group is not None:
                    # Normalize the group value to handle various data types consistently
                    study_group_normalized = _normalize_group_value(study_group)

                    if study_group_normalized not in group_distribution:
                        group_distribution[study_group_normalized] = 0
                    group_distribution[study_group_normalized] += 1
                    corrected_info["Group"] = study_group_normalized
                else:
                    # This should not happen for study samples, but handle gracefully
                    if "Unknown_Group" not in group_distribution:
                        group_distribution["Unknown_Group"] = 0
                    group_distribution["Unknown_Group"] += 1
                    corrected_info["Group"] = "Unknown_Group"
            else:
                if "No_Group_Column" not in group_distribution:
                    group_distribution["No_Group_Column"] = 0
                group_distribution["No_Group_Column"] += 1
                corrected_info["Group"] = "No_Group_Column"

        corrected_sample_metadata[sample_name] = corrected_info

    print(f"Total samples: {len(sample_metadata)}")
    print(f"Control samples: {len(control_samples)}")
    print(f"Study samples: {len(study_samples)}")

    print("\nSAMPLE DISTRIBUTION:")
    # Use configuration order instead of sorting - respect group_labels and control_labels order
    ordered_groups = []

    # First add study groups in the order specified by group_labels
    for target_group in group_labels:
        # Normalize the target group to match our normalization
        normalized_target = _normalize_group_value(target_group)
        if normalized_target in group_distribution:
            ordered_groups.append((normalized_target, group_distribution[normalized_target]))

    # Then add control groups in the order specified by control_labels
    for control_label in control_labels:
        if control_label in group_distribution:
            ordered_groups.append((control_label, group_distribution[control_label]))

    # Finally add any remaining groups not in configuration (shouldn't happen normally)
    for group_name, count in group_distribution.items():
        if not any(group_name == existing[0] for existing in ordered_groups):
            ordered_groups.append((group_name, count))

    for group_name, count in ordered_groups:
        if str(group_name).startswith("Control_"):
            status = " (Control Pool)"
        else:
            status = " (Study Group)"
        print(f"  {group_name}: {count} samples{status}")

    print("\nCONFIGURATION:")
    print(f"  Study groups from column: '{group_column}' -> {group_labels}")
    print(f"  Control identification: '{control_column}' -> {control_labels}")

    # Verify study group samples exist
    study_group_counts = {}
    for sample_name, sample_info in sample_metadata.items():
        if sample_name in study_samples and group_column in sample_info:
            study_group = sample_info[group_column]
            if pd.notna(study_group):
                # Convert to string, handling float-to-int conversion for cleaner labels
                if isinstance(study_group, (int, float)) and float(study_group).is_integer():
                    study_group_str = str(int(float(study_group)))  # Convert 80.0 -> '80'
                else:
                    study_group_str = str(study_group)

                # Check if this matches any of the expected group labels
                for target_group in group_labels:
                    if study_group_str == target_group:
                        if target_group not in study_group_counts:
                            study_group_counts[target_group] = 0
                        study_group_counts[target_group] += 1
                        break

    print("\nSTUDY GROUP VERIFICATION:")
    for target_group in group_labels:
        count = study_group_counts.get(target_group, 0)
        print(f"  {target_group}: {count} samples")

    print("\nSamples are now properly classified for analysis and visualization")

    # Reorder group_distribution to match configuration order
    ordered_group_distribution = {}

    # First add study groups in configuration order
    for target_group in group_labels:
        # Normalize the target group to match our normalization
        normalized_target = _normalize_group_value(target_group)
        if normalized_target in group_distribution:
            ordered_group_distribution[normalized_target] = group_distribution[normalized_target]

    # Then add control groups in configuration order
    for control_label in control_labels:
        if control_label in group_distribution:
            ordered_group_distribution[control_label] = group_distribution[control_label]

    # Finally add any remaining groups not in configuration (shouldn't happen normally)
    for group_name, count in group_distribution.items():
        if group_name not in ordered_group_distribution:
            ordered_group_distribution[group_name] = count

    # Use the ordered distribution
    group_distribution = ordered_group_distribution

    # Automatically apply systematic color scheme if requested
    group_colors = None
    if apply_systematic_colors:
        print("\n" + "=" * 50)
        print("AUTOMATIC SYSTEMATIC COLOR ASSIGNMENT")
        print("=" * 50)
        group_colors, _ = apply_systematic_color_scheme(
            sample_metadata=corrected_sample_metadata,
            group_labels=group_labels,
            control_labels=control_labels,
            systematic_color_palette=systematic_color_palette,
            use_systematic_colors=True,
        )
        print("\nDone: Systematic color scheme applied automatically")
        print("  All visualization functions will use consistent colors")
        print("  To customize colors, use the optional color customization cell")

    return (
        group_distribution,
        control_samples,
        study_samples,
        corrected_sample_metadata,
        group_colors,
    )


def apply_systematic_color_scheme(
    sample_metadata: Dict,
    group_labels: List[str],
    control_labels: List[str],
    systematic_color_palette: str,
    use_systematic_colors: bool = True,
) -> Tuple[Dict, pd.Series]:
    """
    Apply systematic color scheme to sample groups with proper ordering.

    Parameters:
    -----------
    sample_metadata : dict
        Corrected sample metadata with Group assignments
    group_labels : list of str
        Expected study group labels in desired order
    control_labels : list of str
        Expected control labels in desired order
    systematic_color_palette : str
        Color palette name (e.g., 'Set1', 'tab10')
    use_systematic_colors : bool
        Whether to use systematic colors or automatic assignment

    Returns:
    --------
    tuple: (group_colors, group_counts)
        - group_colors: dict mapping group names to colors
        - group_counts: pandas Series with group counts
    """

    # Calculate group colors with the corrected metadata
    updated_group_colors, updated_group_counts = calculate_group_colors(sample_metadata)

    # Apply systematic color scheme if enabled
    if use_systematic_colors:
        # Create configuration-ordered group list
        config_ordered_groups = []

        # Add study groups in configuration order
        for target_group in group_labels:
            normalized_target = _normalize_group_value(target_group)
            if normalized_target in updated_group_counts:
                config_ordered_groups.append(normalized_target)

        # Add control groups in configuration order
        for control_label in control_labels:
            if control_label in updated_group_counts:
                config_ordered_groups.append(control_label)

        # Add any remaining groups not in configuration
        for group in updated_group_counts.keys():
            if group not in config_ordered_groups:
                config_ordered_groups.append(group)

        # Assign distinct high-contrast colors
        high_contrast_colors = [
            "#1f77b4",  # Blue (dose 0)
            "#ff7f0e",  # Orange (dose 20)
            "#2ca02c",  # Green (dose 40)
            "#d62728",  # Red (dose 80)
            "#9467bd",  # Purple (HoofPool)
            "#8c564b",  # Brown (GWPool)
            "#e377c2",  # Pink (EISAIPool)
            "#7f7f7f",  # Gray
            "#bcbd22",  # Olive
            "#17becf",  # Cyan
        ]

        systematic_group_colors = {}
        for i, group in enumerate(config_ordered_groups):
            if i < len(high_contrast_colors):
                systematic_group_colors[group] = high_contrast_colors[i]
            else:
                systematic_group_colors[group] = high_contrast_colors[
                    i % len(high_contrast_colors)
                ]  # Create ordered group list: study groups first, then control pools
        control_groups = []
        study_groups = []

        for group_name in updated_group_counts.index:
            if group_name in control_labels:
                control_groups.append(group_name)
            else:
                study_groups.append(group_name)

        # Sort study groups to match group_labels order if possible
        study_groups_sorted = []
        for target_group in group_labels:
            if target_group in study_groups:
                study_groups_sorted.append(target_group)

        # Add any study groups not in group_labels
        for group in study_groups:
            if group not in study_groups_sorted:
                study_groups_sorted.append(group)

        # Sort control groups to match control_labels order
        control_groups_sorted = []
        for control_label in control_labels:
            if control_label in control_groups:
                control_groups_sorted.append(control_label)

        # Add any control groups not in control_labels
        for group in control_groups:
            if group not in control_groups_sorted:
                control_groups_sorted.append(group)

        # Final ordered list: study groups first, then controls
        ordered_groups = study_groups_sorted + control_groups_sorted

        # Simple, reliable color assignment with distinct colors for each group
        # Use a predefined set of high-contrast colors that work well for data visualization
        high_contrast_colors = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple (for controls)
            "#8c564b",  # Brown (for controls)
            "#e377c2",  # Pink (for controls)
            "#7f7f7f",  # Gray
            "#bcbd22",  # Olive
            "#17becf",  # Cyan
        ]

        systematic_group_colors = {}
        for i, group in enumerate(ordered_groups):
            if i < len(high_contrast_colors):
                systematic_group_colors[group] = high_contrast_colors[i]
            else:
                # Fallback to cycling through colors if we have more groups than colors
                systematic_group_colors[group] = high_contrast_colors[i % len(high_contrast_colors)]
                systematic_group_colors[group] = updated_group_colors.get(group, "#1f77b4")

        # Update the group colors dictionary
        updated_group_colors = systematic_group_colors

        print("High-contrast systematic color assignments:")
        color_scheme_type = "high-contrast systematic (default)"
    else:
        print("Automatic color assignments:")
        color_scheme_type = "automatic"

    for group, count in updated_group_counts.items():
        color = updated_group_colors.get(str(group), "#1f77b4")  # Convert to string and provide fallback
        # Determine if this is a control group
        is_control_group = str(group) in control_labels
        group_type = " (Control)" if is_control_group else " (Study)"
        print(f"  {group}: {count} samples (color: {color}){group_type}")

    print(f"\n{color_scheme_type.title()} colors applied for optimal visual distinction")
    print("All sample groups will be clearly distinguishable in plots")

    return updated_group_colors, updated_group_counts
