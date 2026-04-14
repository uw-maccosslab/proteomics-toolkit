"""
Statistical Analysis Module for Proteomics Data

This module provides a clean, configuration-driven approach to differential
protein expression analysis with support for various statistical methods.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_1samp, ttest_ind, wilcoxon
from statsmodels.stats.multitest import multipletests

from .normalization import is_normalization_log_transformed
from .preprocessing import _normalize_group_value

# Try to import statsmodels for mixed-effects models
try:
    from statsmodels.formula.api import mixedlm

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    mixedlm = None
    warnings.warn("statsmodels not available. Mixed-effects models will be disabled.")


def _sanitize_formula_term(term):
    """
    Sanitize a column name for use in statsmodels formulas.
    Wraps terms containing spaces or special characters in Q().

    Parameters:
    -----------
    term : str
        Column name to sanitize

    Returns:
    --------
    str
        Sanitized term safe for use in formulas
    """
    # Check if term needs quoting (contains spaces or special characters)
    if " " in term or any(char in term for char in [":", "-", "+", "*", "/", "(", ")", "[", "]"]):
        return f'Q("{term}")'
    return term


def _apply_log_transformation_if_needed(data, config):
    """
    Apply log transformation to data if needed based on configuration.
    Uses existing normalization infrastructure to determine if data is already log-transformed.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data with numeric columns to potentially transform
    config : StatisticalConfig
        Configuration object containing log transformation settings

    Returns:
    --------
    pd.DataFrame
        Data with log transformation applied if needed
    """
    # Get numeric columns (sample data)
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    if config.log_transform_before_stats == "auto":
        # Use existing normalization infrastructure to determine if data is already log-transformed
        if hasattr(config, "normalization_method") and config.normalization_method:
            already_log_transformed = is_normalization_log_transformed(config.normalization_method)

            if already_log_transformed:
                apply_log_transform = False
                print(
                    f"Log transformation: AUTO-DETECTED "
                    f"(not needed - {config.normalization_method} already log-transforms data)"
                )
            else:
                apply_log_transform = True
                print(
                    f"Log transformation: AUTO-DETECTED "
                    f"(needed - {config.normalization_method} preserves original scale)"
                )
        else:
            # No normalization method info, check data range as fallback
            if numeric_columns:
                sample_data_range = data[numeric_columns]
                mean_value = sample_data_range.mean().mean()
                apply_log_transform = mean_value > 50
                status = "needed" if apply_log_transform else "not needed"
                print(f"Log transformation: AUTO-DETECTED ({status} - mean value {mean_value:.1f})")
            else:
                apply_log_transform = False
                print("Log transformation: AUTO-DETECTED (no numeric columns found)")

    elif str(config.log_transform_before_stats).lower() in ["true", "1", "yes", "on"]:
        apply_log_transform = True
        print("Log transformation: ENABLED (forced by configuration)")
    else:
        apply_log_transform = False
        print("Log transformation: DISABLED (by configuration)")

    if not apply_log_transform or not numeric_columns:
        print("Using data as-is for statistical analysis")
        return data

    # Apply log transformation
    print(f"Applying {config.log_base} transformation for statistical analysis...")

    # Create a copy to avoid modifying original data
    transformed_data = data.copy()
    sample_data_subset = transformed_data[numeric_columns]

    # Handle negative values if any exist
    if (sample_data_subset < 0).any().any():
        print("  -> Handling negative values...")
        min_val = sample_data_subset.min().min()
        shift_amount = abs(min_val) + 1
        transformed_data[numeric_columns] = transformed_data[numeric_columns] + shift_amount
        print(f"     Shifted all values by +{shift_amount:.2f}")

    # Determine pseudocount
    if config.log_pseudocount is None:
        pseudocount = max(1e-6, sample_data_subset.min().min() / 100) if sample_data_subset.min().min() > 0 else 0.1
    else:
        pseudocount = config.log_pseudocount

    # Apply appropriate log transformation
    if config.log_base == "log2":
        transformed_data[numeric_columns] = np.log2(transformed_data[numeric_columns] + pseudocount)
    elif config.log_base == "log10":
        transformed_data[numeric_columns] = np.log10(transformed_data[numeric_columns] + pseudocount)
    elif config.log_base == "ln":
        transformed_data[numeric_columns] = np.log(transformed_data[numeric_columns] + pseudocount)
    else:
        raise ValueError(f"Unknown log base: {config.log_base}")

    print(f"  -> Applied {config.log_base} transformation with pseudocount {pseudocount}")

    # Verify transformation
    new_mean = transformed_data[numeric_columns].mean().mean()
    print(
        f"  -> New data range: {transformed_data[numeric_columns].min().min():.2f}"
        f" to {transformed_data[numeric_columns].max().max():.2f}"
    )
    print(f"  -> New mean: {new_mean:.2f}")

    return transformed_data


class StatisticalConfig:
    """Configuration class for statistical analysis parameters

    Supports multiple analysis types:
    - 'paired': Paired group comparison (requires group_column, paired_label1, paired_label2)
    - 'unpaired': Unpaired group comparison (requires group_column, group_labels)
    - 'linear_trend': Linear trend over time/dose (requires time_column, tests slope != 0)
    - 'longitudinal': Any change over time (requires time_column, F-test on time as factor)
    - 'interaction': Group × Time interaction (requires group_column, paired_column, interaction_terms)

    Note: 'dose_response' is accepted as an alias for 'linear_trend' for backward compatibility.
    """

    def __init__(self):
        # Basic analysis parameters
        self.statistical_test_method = "mixed_effects"
        self.analysis_type = (
            None  # Must be set by user: 'paired', 'unpaired', 'linear_trend', 'longitudinal', 'interaction'
        )
        self.p_value_threshold = 0.05
        self.fold_change_threshold = 1.5

        # Experimental design - set based on analysis type
        self.subject_column = None  # Required for mixed-effects models
        self.time_column = None  # For linear_trend/longitudinal analysis (Week, Time, etc.)
        self.dose_column = None  # Alias for time_column (backward compatibility)

        # Group comparison parameters (for paired/unpaired/interaction analyses)
        self.group_column = None
        self.group_labels = []

        # Paired comparison parameters (for paired analysis)
        self.paired_column = None
        self.paired_label1 = None
        self.paired_label2 = None

        # Mixed-effects specific
        self.interaction_terms = []
        self.additional_interactions = []
        self.covariates = []

        # Variable treatment control
        self.force_categorical = False  # True to treat numeric variables as categorical factors

        # T-test variance assumption
        self.assume_equal_variance = False  # True for Student's t-test, False for Welch's t-test

        # Multiple testing correction
        self.correction_method = "fdr_bh"

        # P-value selection parameters
        self.use_adjusted_pvalue = "adjusted"  # "adjusted" or "unadjusted"
        self.enable_pvalue_fallback = True

        # Log transformation parameters
        self.log_transform_before_stats = "auto"  # "auto", True, False
        self.log_base = "log2"  # "log2", "log10", "ln"
        self.log_pseudocount = None  # None for auto, or specific value

        # Normalization method (used for auto log transformation)
        self.normalization_method = None  # Set this to the normalization method used

    def validate(self):
        """Validate that required parameters are set for the chosen analysis type"""
        if not self.analysis_type:
            raise ValueError(
                "analysis_type must be set. Choose: 'paired', 'unpaired', "
                "'linear_trend', 'longitudinal', or 'interaction'"
            )

        # Get time column (time_column preferred, dose_column for backward compatibility)
        time_col = self.time_column or self.dose_column

        # Validate linear_trend analysis (formerly dose_response)
        if self.analysis_type in ("linear_trend", "dose_response"):
            if not time_col:
                raise ValueError("linear_trend analysis requires time_column to be set")
            if self.statistical_test_method == "mixed_effects" and not self.subject_column:
                raise ValueError("Mixed-effects linear_trend analysis requires subject_column")

        # Validate longitudinal analysis (F-test for any change over time)
        elif self.analysis_type == "longitudinal":
            if not time_col:
                raise ValueError("longitudinal analysis requires time_column to be set")
            if self.statistical_test_method == "mixed_effects" and not self.subject_column:
                raise ValueError("Mixed-effects longitudinal analysis requires subject_column")

        # Validate paired group comparison
        elif self.analysis_type == "paired":
            if not self.group_column or not self.group_labels:
                raise ValueError("paired analysis requires group_column and group_labels")
            if not self.paired_label1 or not self.paired_label2:
                raise ValueError("paired analysis requires paired_label1 and paired_label2")
            if self.statistical_test_method == "mixed_effects" and not self.subject_column:
                raise ValueError("Mixed-effects paired analysis requires subject_column")

        # Validate unpaired group comparison
        elif self.analysis_type == "unpaired":
            if not self.group_column or not self.group_labels:
                raise ValueError("unpaired analysis requires group_column and group_labels")

        # Validate interaction analysis
        elif self.analysis_type == "interaction":
            if not self.group_column or not self.group_labels:
                raise ValueError("interaction analysis requires group_column and group_labels")
            if not self.paired_column:
                raise ValueError("interaction analysis requires paired_column")
            if not self.interaction_terms:
                raise ValueError("interaction analysis requires interaction_terms")
            if self.statistical_test_method == "mixed_effects" and not self.subject_column:
                raise ValueError("Mixed-effects interaction analysis requires subject_column")

        return True


def prepare_metadata_dataframe(sample_metadata_dict, sample_columns, config):
    """Convert sample metadata dictionary to DataFrame suitable for analysis"""

    print(f"Preparing metadata for {len(sample_columns)} samples...")

    # Create DataFrame from metadata dictionary
    metadata_rows = []
    for sample_name in sample_columns:
        if sample_name in sample_metadata_dict:
            row = sample_metadata_dict[sample_name].copy()
            row["Sample"] = sample_name
            metadata_rows.append(row)
        else:
            print(f"Warning: No metadata found for sample {sample_name}")

    if not metadata_rows:
        print("Warning: No metadata found for any samples - returning empty DataFrame")
        # Return empty DataFrame with expected columns for graceful handling
        expected_cols = ["Sample"]
        if config.subject_column:
            expected_cols.append(config.subject_column)
        if config.paired_column:
            expected_cols.append(config.paired_column)
        if hasattr(config, "group_column") and config.group_column:
            expected_cols.append(config.group_column)
        return pd.DataFrame(columns=expected_cols)

    metadata_df = pd.DataFrame(metadata_rows)

    # Skip validation if no data (graceful handling for edge cases)
    if len(metadata_df) == 0:
        print("Empty metadata - skipping validation")
        return metadata_df

    # Build list of required columns based on analysis configuration
    required_cols = []

    # Subject column is always required for mixed-effects models
    if config.subject_column:
        required_cols.append(config.subject_column)

    # Paired/time column is usually required
    if config.paired_column:
        required_cols.append(config.paired_column)

    # Group column is only required for group comparison analyses
    # NOT required for dose-response or continuous predictor models
    if hasattr(config, "group_column") and config.group_column:
        # Only require if we're using interaction terms that include the group column
        # or if analysis_type indicates group comparison
        if (config.interaction_terms and config.group_column in config.interaction_terms) or (
            hasattr(config, "analysis_type") and config.analysis_type in ["paired", "unpaired"]
        ):
            required_cols.append(config.group_column)

    # Validate required columns exist
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required metadata columns: {missing_cols}")

    # CRITICAL: Filter out samples with missing values in required columns
    print(f"  Before filtering: {len(metadata_df)} samples")
    for col in required_cols:
        before_count = len(metadata_df)
        metadata_df = metadata_df.dropna(subset=[col])
        after_count = len(metadata_df)
        if before_count != after_count:
            print(f"  Removed {before_count - after_count} samples missing {col}")

    if len(metadata_df) == 0:
        raise ValueError("No samples remain after filtering for required metadata")

    print(f"  After filtering: {len(metadata_df)} samples")

    # Only print subject info if subject_column is defined
    if config.subject_column and config.subject_column in metadata_df.columns:
        print(f"  Subjects: {metadata_df[config.subject_column].nunique()}")

    # Only print group info if group_column is defined and exists in metadata
    if hasattr(config, "group_column") and config.group_column and config.group_column in metadata_df.columns:
        print(f"  Groups: {metadata_df[config.group_column].value_counts().to_dict()}")

        # Handle categorical vs continuous variable treatment
        if hasattr(config, "force_categorical") and config.force_categorical:
            # Convert group column to categorical (string) type to force statsmodels to treat as factors
            metadata_df[config.group_column] = metadata_df[config.group_column].astype(str)
            print("  Group variable treatment: CATEGORICAL (converted to string factors)")
        else:
            # Ensure group column preserves numeric type if possible (for continuous treatment)
            # Apply normalization to ensure consistency but preserve numeric types
            metadata_df[config.group_column] = metadata_df[config.group_column].apply(_normalize_group_value)
            group_col_type = metadata_df[config.group_column].dtype
            print(f"  Group variable treatment: CONTINUOUS (type: {group_col_type})")

    if config.paired_column and config.paired_column in metadata_df.columns:
        print(f"  Timepoints: {metadata_df[config.paired_column].value_counts().to_dict()}")

    return metadata_df


def run_paired_t_test(protein_data, metadata_df, config):
    """Run paired t-test analysis"""

    print("Running paired t-test analysis...")

    results = []
    n_proteins = len(protein_data)

    for i, (protein_idx, protein_values) in enumerate(protein_data.iterrows()):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{n_proteins} proteins...")

        # Get data for this protein
        protein_df = pd.DataFrame({"Sample": protein_values.index, "Intensity": protein_values.values})

        # Merge with metadata
        protein_df = protein_df.merge(metadata_df, on="Sample", how="inner")

        # Remove missing values
        protein_df = protein_df.dropna(subset=["Intensity"])

        if len(protein_df) < 4:  # Need at least some data
            results.append(_create_empty_result(protein_idx, "Insufficient data"))
            continue

        # Calculate paired differences for each subject
        baseline_data = protein_df[protein_df[config.paired_column] == config.paired_label1]
        followup_data = protein_df[protein_df[config.paired_column] == config.paired_label2]

        # Merge on subject to get paired data
        paired_data = baseline_data.merge(followup_data, on=config.subject_column, suffixes=("_baseline", "_followup"))

        if len(paired_data) < 3:  # Need at least 3 pairs
            results.append(_create_empty_result(protein_idx, "Insufficient paired data"))
            continue

        # Calculate differences (followup - baseline)
        differences = paired_data["Intensity_followup"] - paired_data["Intensity_baseline"]

        try:
            # Paired t-test (test if mean difference != 0)
            t_stat, p_value = ttest_1samp(differences, 0)

            # Calculate effect size (Cohen's d for paired data)
            mean_diff = differences.mean()
            std_diff = differences.std()
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0

            # Log fold change (approximate)
            log_fc = mean_diff  # Already in log-like space if VSN normalized

            result = {
                "Protein": protein_idx,
                "logFC": log_fc,
                "AveExpr": protein_df["Intensity"].mean(),
                "t": t_stat,
                "P.Value": p_value,
                "B": np.nan,  # Not applicable for t-test
                "n_pairs": len(paired_data),
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "cohens_d": cohens_d,
                "test_method": "Paired t-test",
            }

            results.append(result)

        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            results.append(_create_empty_result(protein_idx, f"Analysis failed: {e}"))

    print(f"Done: Paired t-test completed for {len(results)} proteins")
    return pd.DataFrame(results)


def compute_paired_fold_changes(protein_data, sample_metadata, config):
    """Compute per-subject fold-changes (post minus pre) for each protein.

    Returns a DataFrame with subjects as rows and proteins as columns,
    where each value is the difference (paired_label2 - paired_label1)
    for that subject and protein. Useful for downstream classification
    or clustering of subjects based on protein response profiles.

    Args:
        protein_data: DataFrame with proteins as rows, samples as columns.
            First 5 columns are annotations (Protein, Description, etc.),
            remaining columns are sample intensities.
        sample_metadata: Dict mapping sample column names to metadata dicts.
            Each dict must contain the keys specified by config.subject_column
            and config.paired_column.
        config: StatisticalConfig with paired_column, paired_label1,
            paired_label2, and subject_column set.

    Returns:
        DataFrame with subjects as rows (index = subject IDs) and proteins
        as columns (column names = protein identifiers from the data index).
        Values are fold-changes (paired_label2 - paired_label1) in the same
        scale as the input data (log2 if input is log2-transformed).
    """
    # Build metadata DataFrame from dict
    meta_records = []
    for sample_name, meta in sample_metadata.items():
        record = dict(meta)
        record["Sample"] = sample_name
        meta_records.append(record)
    metadata_df = pd.DataFrame(meta_records)

    # Ensure paired column values are comparable
    metadata_df[config.paired_column] = metadata_df[config.paired_column].apply(
        lambda x: float(x) if isinstance(x, (int, float, np.integer, np.floating)) else x
    )

    # Filter to samples with valid subject and timepoint
    valid_mask = metadata_df[config.subject_column].notna() & metadata_df[config.paired_column].notna()
    metadata_df = metadata_df[valid_mask]

    # Split by timepoint
    baseline = metadata_df[metadata_df[config.paired_column] == config.paired_label1].set_index(config.subject_column)
    followup = metadata_df[metadata_df[config.paired_column] == config.paired_label2].set_index(config.subject_column)

    # Find subjects present at both timepoints
    paired_subjects = baseline.index.intersection(followup.index)
    if len(paired_subjects) == 0:
        raise ValueError(
            f"No paired subjects found. Check that config.paired_label1="
            f"{config.paired_label1} and config.paired_label2="
            f"{config.paired_label2} match values in the "
            f"'{config.paired_column}' column."
        )

    print(f"Computing per-subject fold-changes for {len(paired_subjects)} subjects...")

    # Extract sample intensity columns (skip annotation columns)
    sample_cols = [c for c in protein_data.columns if c in sample_metadata]

    # Build the fold-change matrix: subjects x proteins
    fc_rows = {}
    for subject in paired_subjects:
        baseline_sample = baseline.loc[subject, "Sample"]
        followup_sample = followup.loc[subject, "Sample"]

        if baseline_sample in sample_cols and followup_sample in sample_cols:
            diff = protein_data[followup_sample].values - protein_data[baseline_sample].values
            fc_rows[subject] = diff

    fc_matrix = pd.DataFrame(fc_rows, index=protein_data.index).T
    fc_matrix.index.name = config.subject_column

    print(f"  Result: {fc_matrix.shape[0]} subjects x {fc_matrix.shape[1]} proteins")
    return fc_matrix


def run_mixed_effects_analysis(protein_data, metadata_df, config, protein_annotations=None):
    """Run mixed-effects model analysis

    Supports multiple analysis types:
    - linear_trend (or dose_response): Protein ~ Time + (1|Subject) with continuous time
      Tests if there is a linear trend over time (slope != 0)
    - longitudinal: Protein ~ C(Time) + (1|Subject) with categorical time
      Tests if protein changes at all over time (F-test on time factor)
    - interaction: Protein ~ Group * Time + (1|Subject)
      Tests if time effect differs between groups
    """

    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is required for mixed-effects analysis")

    # Get time column (time_column preferred, dose_column for backward compatibility)
    time_col = getattr(config, "time_column", None) or getattr(config, "dose_column", None)

    # Determine if this is a longitudinal (categorical time) analysis
    is_longitudinal = hasattr(config, "analysis_type") and config.analysis_type == "longitudinal"
    is_linear_trend = hasattr(config, "analysis_type") and config.analysis_type in ("linear_trend", "dose_response")

    # Determine model description for logging
    all_interaction_terms = config.interaction_terms + config.additional_interactions

    if len(all_interaction_terms) >= 2:
        term1 = _sanitize_formula_term(all_interaction_terms[0])
        term2 = _sanitize_formula_term(all_interaction_terms[1])
        model_desc = f"Protein ~ {term1} * {term2}"
        if len(all_interaction_terms) > 2:
            model_desc += " + " + " + ".join(_sanitize_formula_term(t) for t in all_interaction_terms[2:])
    elif len(all_interaction_terms) == 1:
        term = _sanitize_formula_term(all_interaction_terms[0])
        model_desc = f"Protein ~ {term}"
    elif time_col:
        time_term = _sanitize_formula_term(time_col)
        if is_longitudinal:
            model_desc = f"Protein ~ C({time_term})"  # Categorical time for F-test
        else:
            model_desc = f"Protein ~ {time_term}"  # Continuous time for linear trend
    else:
        model_desc = "Protein ~ 1"  # Intercept only

    if config.covariates:
        sanitized_covariates = [_sanitize_formula_term(cov) for cov in config.covariates]
        model_desc += " + " + " + ".join(sanitized_covariates)

    model_desc += f" + (1|{config.subject_column})"

    print("Running mixed-effects analysis...")
    if is_longitudinal:
        print("  Analysis type: LONGITUDINAL (any change over time, F-test)")
        print("  Tests: Do any timepoints differ from each other?")
    elif is_linear_trend:
        print("  Analysis type: LINEAR TREND (continuous time)")
        print("  Tests: Is there a linear trend over time (slope ≠ 0)?")
    print(f"  Model: {model_desc}")

    results = []
    n_proteins = len(protein_data)

    for i, (protein_idx, protein_values) in enumerate(protein_data.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_proteins} proteins...")

        # Get actual protein name if annotations are provided
        if protein_annotations is not None and "Protein" in protein_annotations.columns:
            # Use protein_idx (the actual row index) to get the correct protein name
            actual_protein_name = protein_annotations.loc[protein_idx, "Protein"]
        else:
            actual_protein_name = protein_idx  # Fallback to index

        # Prepare data for this protein
        protein_df = pd.DataFrame({"Sample": protein_values.index, "Intensity": protein_values.values})

        # Merge with metadata
        protein_df = protein_df.merge(metadata_df, on="Sample", how="inner")

        # Remove missing values
        protein_df = protein_df.dropna(subset=["Intensity"])

        if len(protein_df) < 8:  # Need sufficient data for mixed model
            results.append(_create_empty_mixed_effects_result(actual_protein_name, "Insufficient data"))
            continue

        try:
            # Build formula - supports both interaction models and additive models
            all_interaction_terms = config.interaction_terms + config.additional_interactions

            formula = None

            if len(all_interaction_terms) >= 2:
                # Build interaction model (first two terms)
                term1 = _sanitize_formula_term(all_interaction_terms[0])
                term2 = _sanitize_formula_term(all_interaction_terms[1])
                formula = f"Intensity ~ {term1} * {term2}"
                # Add additional interaction terms as main effects
                if len(all_interaction_terms) > 2:
                    additional_terms = " + ".join(_sanitize_formula_term(t) for t in all_interaction_terms[2:])
                    formula += f" + {additional_terms}"

            elif len(all_interaction_terms) == 1:
                # Build additive model with single main effect
                term = _sanitize_formula_term(all_interaction_terms[0])
                formula = f"Intensity ~ {term}"

            elif time_col:
                # Build time-based model
                time_term = _sanitize_formula_term(time_col)
                if is_longitudinal:
                    # LONGITUDINAL: Treat time as categorical factor for F-test
                    formula = f"Intensity ~ C({time_term})"
                else:
                    # LINEAR TREND: Treat time as continuous for slope test
                    formula = f"Intensity ~ {time_term}"

            else:
                # No main effects specified - need at least one predictor
                raise ValueError("Need at least one predictor variable (interaction_terms, time_column, or covariates)")

            # Add covariates if specified
            if config.covariates:
                sanitized_covariates = [_sanitize_formula_term(cov) for cov in config.covariates]
                formula += " + " + " + ".join(sanitized_covariates)

            # Fit mixed-effects model
            if mixedlm is None:
                raise ImportError("statsmodels required for mixed-effects analysis")

            # Suppress convergence warnings during fitting
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message=".*convergence.*")
                warnings.filterwarnings("ignore", message=".*singular.*")

                model = mixedlm(formula, protein_df, groups=protein_df[config.subject_column])

                # Try robust BFGS method first (more stable than LBFGS)
                # Fallback to other methods if needed
                fitted_model = None
                for method in ["bfgs", "nm", "powell", "lbfgs"]:
                    try:
                        fitted_model = model.fit(method=method, disp=False)  # type: ignore[call-arg]
                        break
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        continue

                if fitted_model is None:
                    raise RuntimeError("All optimization methods failed")

            # Extract results
            params = fitted_model.params
            pvalues = fitted_model.pvalues

            # Initialize all effect variables
            interaction_coef = np.nan
            interaction_pvalue = np.nan
            group_effect = np.nan
            group_pvalue = np.nan
            time_effect = np.nan
            time_pvalue = np.nan

            # Determine primary effect based on model type
            if len(all_interaction_terms) >= 2:
                # INTERACTION MODEL: Extract interaction and main effects

                # Find interaction parameters (contain both variable names and ":")
                interaction_candidates = [
                    p
                    for p in params.index
                    if config.interaction_terms[0] in p and config.interaction_terms[1] in p and ":" in p
                ]

                if interaction_candidates:
                    # Use the first (and typically only) interaction term
                    term_name = interaction_candidates[0]
                    interaction_coef = params[term_name]
                    interaction_pvalue = pvalues[term_name]

                # Find group effect parameters
                group_candidates = [
                    p for p in params.index if config.interaction_terms[0] in p and ":" not in p and p != "Intercept"
                ]

                if group_candidates:
                    term_name = group_candidates[0]
                    group_effect = params[term_name]
                    group_pvalue = pvalues[term_name]

                # Find time effect parameters
                time_candidates = [
                    p for p in params.index if config.interaction_terms[1] in p and ":" not in p and p != "Intercept"
                ]

                if time_candidates:
                    term_name = time_candidates[0]
                    time_effect = params[term_name]
                    time_pvalue = pvalues[term_name]

            else:
                # ADDITIVE MODEL: Extract primary predictor effect
                primary_predictor = None

                if len(all_interaction_terms) == 1:
                    primary_predictor = all_interaction_terms[0]
                elif time_col:
                    primary_predictor = time_col

                if primary_predictor:
                    # Find the parameter for the primary predictor
                    primary_candidates = [
                        p for p in params.index if primary_predictor in p and ":" not in p and p != "Intercept"
                    ]

                    if is_longitudinal and len(primary_candidates) > 1:
                        # LONGITUDINAL: Multiple time coefficients (categorical)
                        # Use Wald test on all time coefficients together (F-test equivalent)
                        try:
                            # Build constraint string for joint Wald test
                            # Test that ALL time coefficients = 0 simultaneously
                            constraints = [f"{c} = 0" for c in primary_candidates]
                            wald_test = fitted_model.wald_test(constraints, scalar=True)
                            group_pvalue = wald_test.pvalue
                            # For logFC, use max absolute coefficient as effect size
                            time_coefficients = [params[c] for c in primary_candidates]
                            max_abs_idx = np.argmax(np.abs(time_coefficients))
                            group_effect = time_coefficients[max_abs_idx]
                        except Exception:
                            # Fallback: use most significant individual coefficient
                            min_pval_idx = np.argmin([pvalues[c] for c in primary_candidates])
                            term_name = primary_candidates[min_pval_idx]
                            group_effect = params[term_name]
                            group_pvalue = pvalues[term_name]
                    elif primary_candidates:
                        # LINEAR TREND: Single time coefficient (continuous)
                        term_name = primary_candidates[0]
                        group_effect = params[term_name]
                        group_pvalue = pvalues[term_name]

            # Determine primary effect for logFC and P.Value
            # For interaction models: use interaction term
            # For additive models: use primary predictor (group_effect)
            if len(all_interaction_terms) >= 2:
                primary_logfc = interaction_coef
                primary_pvalue = interaction_pvalue
            else:
                primary_logfc = group_effect
                primary_pvalue = group_pvalue

            # Determine test method description
            if is_longitudinal:
                test_method = "Mixed-effects model (Wald F-test on time)"
            elif is_linear_trend:
                test_method = "Mixed-effects model (linear trend)"
            else:
                test_method = "Mixed-effects model"

            result = {
                "Protein": actual_protein_name,  # Use actual protein name
                "logFC": primary_logfc,  # Primary effect (interaction or main effect)
                "AveExpr": protein_df["Intensity"].mean(),
                "t": np.nan,  # Not applicable for mixed model
                "P.Value": primary_pvalue,  # Primary p-value
                "B": np.nan,  # Not applicable
                "group_effect": group_effect,
                "group_pvalue": group_pvalue,
                "time_effect": time_effect,
                "time_pvalue": time_pvalue,
                "interaction_effect": interaction_coef,
                "interaction_pvalue": interaction_pvalue,
                "aic": fitted_model.aic if fitted_model else np.nan,
                "bic": fitted_model.bic if fitted_model else np.nan,
                "n_obs": len(protein_df),
                "test_method": test_method,
            }

            results.append(result)

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            error_message = f"Model failed: {e}"
            if i < 3:  # Show details for first few failures
                print(f"  Protein {i + 1} failed: {error_message}")
            results.append(_create_empty_mixed_effects_result(actual_protein_name, error_message))

    print(f"Done: Mixed-effects analysis completed for {len(results)} proteins")
    return pd.DataFrame(results)


def run_unpaired_t_test(protein_data, metadata_df, config):
    """Run unpaired t-test analysis"""

    print("Running unpaired t-test analysis...")

    results = []
    n_proteins = len(protein_data)

    # Filter to specific timepoint if needed
    if config.paired_column and config.paired_label2:
        metadata_df = metadata_df[metadata_df[config.paired_column] == config.paired_label2]
        print(f"  Analyzing {config.paired_label2} timepoint only")

    for i, (protein_idx, protein_values) in enumerate(protein_data.iterrows()):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{n_proteins} proteins...")

        # Get data for this protein
        protein_df = pd.DataFrame({"Sample": protein_values.index, "Intensity": protein_values.values})

        # Merge with metadata
        protein_df = protein_df.merge(metadata_df, on="Sample", how="inner")
        protein_df = protein_df.dropna(subset=["Intensity"])

        if len(protein_df) < 4:
            results.append(_create_empty_result(protein_idx, "Insufficient data"))
            continue

        # Split into groups
        group1_data = protein_df[protein_df[config.group_column] == config.group_labels[0]]["Intensity"]
        group2_data = protein_df[protein_df[config.group_column] == config.group_labels[1]]["Intensity"]

        if len(group1_data) < 2 or len(group2_data) < 2:
            results.append(_create_empty_result(protein_idx, "Insufficient group data"))
            continue

        try:
            # Use Student's t-test (equal variance) or Welch's t-test based on config
            equal_var = getattr(config, "assume_equal_variance", False)
            t_stat, p_value = ttest_ind(group2_data, group1_data, equal_var=equal_var)
            test_name = "Student's t-test" if equal_var else "Welch's t-test"

            # Calculate effect size
            pooled_std = np.sqrt(
                ((len(group1_data) - 1) * group1_data.var() + (len(group2_data) - 1) * group2_data.var())
                / (len(group1_data) + len(group2_data) - 2)
            )
            cohens_d = (group2_data.mean() - group1_data.mean()) / pooled_std if pooled_std > 0 else 0

            # Log fold change
            log_fc = group2_data.mean() - group1_data.mean()

            result = {
                "Protein": protein_idx,
                "logFC": log_fc,
                "AveExpr": protein_df["Intensity"].mean(),
                "t": t_stat,
                "P.Value": p_value,
                "B": np.nan,
                "n_group1": len(group1_data),
                "n_group2": len(group2_data),
                "cohens_d": cohens_d,
                "test_method": test_name,
            }

            results.append(result)

        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            results.append(_create_empty_result(protein_idx, f"Analysis failed: {e}"))

    print(f"Done: Unpaired t-test completed for {len(results)} proteins")
    return pd.DataFrame(results)


def run_wilcoxon_test(protein_data, metadata_df, config):
    """Run paired non-parametric Wilcoxon signed-rank test"""

    print("Running Wilcoxon signed-rank test analysis...")

    results = []
    n_proteins = len(protein_data)

    for i, (protein_idx, protein_values) in enumerate(protein_data.iterrows()):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{n_proteins} proteins...")

        # Get data for this protein
        protein_df = pd.DataFrame({"Sample": protein_values.index, "Intensity": protein_values.values})

        # Merge with metadata
        protein_df = protein_df.merge(metadata_df, on="Sample", how="inner")

        # Remove missing values
        protein_df = protein_df.dropna(subset=["Intensity"])

        if len(protein_df) < 4:  # Need at least some data
            results.append(_create_empty_result(protein_idx, "Insufficient data"))
            continue

        # Calculate paired differences for each subject
        baseline_data = protein_df[protein_df[config.paired_column] == config.paired_label1]
        followup_data = protein_df[protein_df[config.paired_column] == config.paired_label2]

        # Merge on subject to get paired data
        paired_data = baseline_data.merge(followup_data, on=config.subject_column, suffixes=("_baseline", "_followup"))

        if len(paired_data) < 3:  # Need at least 3 pairs
            results.append(_create_empty_result(protein_idx, "Insufficient paired data"))
            continue

        # Calculate differences (followup - baseline)
        differences = paired_data["Intensity_followup"] - paired_data["Intensity_baseline"]

        # Remove zero differences for Wilcoxon test
        non_zero_diffs = differences[differences != 0]

        if len(non_zero_diffs) < 3:
            results.append(_create_empty_result(protein_idx, "Insufficient non-zero differences"))
            continue

        try:
            # Wilcoxon signed-rank test
            statistic, p_value = wilcoxon(non_zero_diffs, alternative="two-sided")

            # Calculate effect size (r = z / sqrt(N))
            # For Wilcoxon, we use median and IQR
            mean_diff = differences.mean()
            median_diff = differences.median()

            # Pseudo-Cohen's d using median and MAD (more robust)
            mad = np.median(np.abs(differences - median_diff)) * 1.4826  # Scale factor for normality
            effect_size = median_diff / mad if mad > 0 else 0

            result = {
                "Protein": protein_idx,
                "logFC": median_diff,  # Use median for non-parametric
                "AveExpr": protein_df["Intensity"].mean(),
                "statistic": statistic,
                "P.Value": p_value,
                "B": np.nan,  # Not applicable for non-parametric tests
                "n_pairs": len(paired_data),
                "mean_diff": mean_diff,
                "median_diff": median_diff,
                "Effect_Size": effect_size,
                "test_method": "Wilcoxon signed-rank",
            }

            results.append(result)

        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            results.append(_create_empty_result(protein_idx, f"Analysis failed: {e}"))

    print(f"Done: Wilcoxon signed-rank test completed for {len(results)} proteins")
    return pd.DataFrame(results)


def run_mann_whitney_test(protein_data, metadata_df, config):
    """Run unpaired non-parametric Mann-Whitney U test"""

    print("Running Mann-Whitney U test analysis...")

    results = []
    n_proteins = len(protein_data)

    # Filter to specific timepoint if needed (only if paired_column is present in metadata)
    if (
        config.paired_column
        and config.paired_label2
        and hasattr(config, "paired_column")
        and config.paired_column in metadata_df.columns
    ):
        metadata_df = metadata_df[metadata_df[config.paired_column] == config.paired_label2]
        print(f"  Analyzing {config.paired_label2} timepoint only")

    for i, (protein_idx, protein_values) in enumerate(protein_data.iterrows()):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{n_proteins} proteins...")

        # Get data for this protein
        protein_df = pd.DataFrame({"Sample": protein_values.index, "Intensity": protein_values.values})

        # Merge with metadata
        protein_df = protein_df.merge(metadata_df, on="Sample", how="inner")
        protein_df = protein_df.dropna(subset=["Intensity"])

        if len(protein_df) < 4:
            results.append(_create_empty_result(protein_idx, "Insufficient data"))
            continue

        # Split into groups
        group1_data = protein_df[protein_df[config.group_column] == config.group_labels[0]]["Intensity"]
        group2_data = protein_df[protein_df[config.group_column] == config.group_labels[1]]["Intensity"]

        if len(group1_data) < 2 or len(group2_data) < 2:
            results.append(_create_empty_result(protein_idx, "Insufficient group data"))
            continue

        try:
            # Mann-Whitney U test
            # Convert to float arrays to ensure scipy compatibility
            group1_arr = np.asarray(group1_data, dtype=float)
            group2_arr = np.asarray(group2_data, dtype=float)
            statistic, p_value = mannwhitneyu(group2_arr, group1_arr, alternative="two-sided")

            # Calculate effect size (r = z / sqrt(N))
            # For Mann-Whitney, we use median and IQR-based effect size
            median1 = group1_data.median()
            median2 = group2_data.median()

            # Pooled MAD for effect size
            mad1 = np.median(np.abs(group1_data - median1)) * 1.4826
            mad2 = np.median(np.abs(group2_data - median2)) * 1.4826
            pooled_mad = np.sqrt((mad1**2 + mad2**2) / 2)

            effect_size = (median2 - median1) / pooled_mad if pooled_mad > 0 else 0

            # Log fold change using medians
            log_fc = median2 - median1

            result = {
                "Protein": protein_idx,
                "logFC": log_fc,
                "AveExpr": protein_df["Intensity"].mean(),
                "statistic": statistic,
                "P.Value": p_value,
                "B": np.nan,
                "n_group1": len(group1_data),
                "n_group2": len(group2_data),
                "Effect_Size": effect_size,
                "test_method": "Mann-Whitney U",
            }

            results.append(result)

        except (ValueError, RuntimeError, ZeroDivisionError) as e:
            results.append(_create_empty_result(protein_idx, f"Analysis failed: {e}"))

    print(f"Done: Mann-Whitney U test completed for {len(results)} proteins")
    return pd.DataFrame(results)


def _create_empty_result(protein_idx, reason):
    """Create empty result for failed analysis"""
    return {
        "Protein": protein_idx,
        "logFC": np.nan,
        "AveExpr": np.nan,
        "t": np.nan,
        "P.Value": np.nan,
        "B": np.nan,
        "test_method": f"Failed: {reason}",
    }


def _create_empty_mixed_effects_result(protein_idx, reason):
    """Create empty result for failed mixed-effects analysis"""
    return {
        "Protein": protein_idx,
        "logFC": np.nan,
        "AveExpr": np.nan,
        "t": np.nan,
        "P.Value": np.nan,
        "B": np.nan,
        "group_effect": np.nan,
        "group_pvalue": np.nan,
        "time_effect": np.nan,
        "time_pvalue": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "n_obs": 0,
        "test_method": f"Mixed-effects failed: {reason}",
    }


def apply_multiple_testing_correction(results_df, config):
    """Apply multiple testing correction"""

    if "P.Value" not in results_df.columns:
        print("Warning: No P.Value column found for correction")
        return results_df

    # Get valid p-values
    valid_pvalues = results_df["P.Value"].dropna()

    if len(valid_pvalues) == 0:
        print("Warning: No valid p-values found")
        results_df["adj.P.Val"] = np.nan
        results_df["Significant"] = False
        return results_df

    # Check if correction should be applied
    correction_method = getattr(config, "correction_method", config.use_adjusted_pvalue)
    if correction_method == "none" or config.use_adjusted_pvalue == "none":
        # No correction - adjusted p-values are same as raw p-values
        results_df["adj.P.Val"] = results_df["P.Value"]
        results_df["Significant"] = results_df["P.Value"] < config.p_value_threshold
        print("Multiple testing correction applied:")
        print("  Method: none (no correction)")
        print(
            f"  Significant proteins (p < {config.p_value_threshold}): "
            f"{(results_df['P.Value'] < config.p_value_threshold).sum()}"
        )
    else:
        # Apply correction
        all_pvalues = results_df["P.Value"].fillna(1.0)
        rejected, adj_pvalues, _, _ = multipletests(all_pvalues, method=correction_method)

        results_df["adj.P.Val"] = adj_pvalues
        results_df["Significant"] = rejected

        print("Multiple testing correction applied:")
        print(f"  Method: {correction_method}")
        print(f"  Significant proteins (FDR < 0.05): {(results_df['adj.P.Val'] < 0.05).sum()}")

    # Add significance categories
    results_df["Significance"] = "Not significant"
    results_df.loc[results_df["adj.P.Val"] < 0.05, "Significance"] = "Significant (FDR < 0.05)"
    results_df.loc[results_df["adj.P.Val"] < 0.01, "Significance"] = "Highly significant (FDR < 0.01)"

    return results_df


def export_results(differential_df: pd.DataFrame, output_file: str, include_all: bool = True) -> None:
    """
    Export differential analysis results to CSV file.

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
        export_df = differential_df[differential_df["Significant"]].copy()
        print(f"Exporting {len(export_df)} significant proteins to {output_file}")
    else:
        export_df = differential_df.copy()
        print(f"Exporting all {len(export_df)} proteins to {output_file}")

    export_df.to_csv(output_file, index=False)
    print("Results exported successfully!")


def run_comprehensive_statistical_analysis(normalized_data, sample_metadata, config, protein_annotations=None):
    """
    Statistical analysis with automatic dataset validation and subject pairing

    Parameters:
    -----------
    normalized_data : pd.DataFrame
        Protein expression data (proteins x samples)
    sample_metadata : dict
        Dictionary mapping sample names to metadata
    config : StatisticalConfig
        Configuration object with analysis parameters
    protein_annotations : pd.DataFrame, optional
        DataFrame with protein annotations including 'Protein' column

    Returns:
    --------
    pd.DataFrame
        Results of statistical analysis
    """

    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        raise ValueError(f"Configuration error: {e}") from e

    # Step 0: Handle log transformation if needed
    statistical_data = _apply_log_transformation_if_needed(normalized_data, config)

    # Step 1: Clean and validate metadata
    print("Step 1: Cleaning and validating sample metadata...")

    # Clean subject IDs to fix whitespace issues
    cleaned_sample_metadata = {}
    for sample_name, metadata in sample_metadata.items():
        cleaned_metadata = metadata.copy()
        if config.subject_column in cleaned_metadata and cleaned_metadata[config.subject_column]:
            cleaned_metadata[config.subject_column] = str(cleaned_metadata[config.subject_column]).strip()
        cleaned_sample_metadata[sample_name] = cleaned_metadata

    sample_metadata = cleaned_sample_metadata

    # Step 2: Prepare metadata dataframe
    # IMPORTANT: With standardized data structure, sample columns start at index 5
    # First 5 columns are always: Protein, Description, Protein Gene, UniProt_Accession, UniProt_Entry_Name
    if len(normalized_data.columns) > 5:
        all_sample_columns = list(normalized_data.columns[5:])  # Everything after first 5 annotation columns
        print(f"  Using standardized data structure: {len(all_sample_columns)} sample columns (columns 6+)")
    else:
        # Fallback for legacy data (shouldn't happen with create_standard_data_structure)
        all_sample_columns = normalized_data.select_dtypes(include=[np.number]).columns.tolist()
        print(f"  Using legacy detection: {len(all_sample_columns)} sample columns")

    # Filter to only samples that have metadata
    sample_columns = [col for col in all_sample_columns if col in sample_metadata]

    if len(sample_columns) < len(all_sample_columns):
        print(f"  Filtered to {len(sample_columns)} samples with metadata (from {len(all_sample_columns)} total)")

    print(f"  Sample columns: {sample_columns[:3]}{'...' if len(sample_columns) > 3 else ''}")

    metadata_df = prepare_metadata_dataframe(sample_metadata, sample_columns, config)

    # Step 3: Analyze experimental design
    print("\nStep 2: Analyzing experimental design...")

    # Check if this is a time-based analysis (no group comparison)
    is_time_based = (
        hasattr(config, "analysis_type") and config.analysis_type in ("linear_trend", "dose_response", "longitudinal")
    ) or (
        not hasattr(config, "group_labels")
        or not config.group_labels
        or not hasattr(config, "group_column")
        or not config.group_column
    )

    if is_time_based:
        # For time-based analyses, all samples are valid (no group filtering needed)
        valid_samples = sample_metadata.copy()
        if config.analysis_type == "longitudinal":
            print("  Analysis type: LONGITUDINAL (F-test for any change over time)")
        elif config.analysis_type in ("linear_trend", "dose_response"):
            print("  Analysis type: LINEAR TREND (testing if slope ≠ 0)")
        else:
            print("  Analysis type: Time-based (no group filtering)")
        print(f"  Valid experimental samples: {len(valid_samples)}")
    else:
        # Filter to experimental samples only (exclude controls) for group comparison
        valid_samples = {}
        # Normalize group labels once for efficiency
        normalized_group_labels = [_normalize_group_value(label) for label in config.group_labels]

        for sample_name, metadata in sample_metadata.items():
            comparison_value = metadata.get(config.group_column)
            # Normalize the comparison value for consistent comparison
            normalized_comparison = _normalize_group_value(comparison_value)

            if normalized_comparison in normalized_group_labels:
                valid_samples[sample_name] = metadata

        print(f"  Valid experimental samples: {len(valid_samples)}")

    # Analyze subject pairing structure (only for paired/interaction designs with subject + paired columns)
    has_pairing_info = (
        not is_time_based
        and config.subject_column
        and config.paired_column
        and hasattr(config, "analysis_type")
        and config.analysis_type in ("paired", "interaction")
    )
    if has_pairing_info:
        pairing_data = {}
        for sample_name, metadata in valid_samples.items():
            subject = metadata.get(config.subject_column)
            visit = metadata.get(config.paired_column)
            comparison = metadata.get(config.group_column)

            # Use 'is not None' instead of truthy check to handle comparison=0
            if subject and visit and comparison is not None:
                if subject not in pairing_data:
                    pairing_data[subject] = {}
                pairing_data[subject][visit] = {
                    "sample": sample_name,
                    "comparison": comparison,
                }

        # Check for complete pairs - handle both categorical and continuous analysis
        complete_pairs = []
        incomplete_subjects = []

        # Determine if we're doing continuous analysis (FORCE_CATEGORICAL = False for numeric variables)
        is_continuous_analysis = (
            hasattr(config, "force_categorical")
            and not config.force_categorical
            and all(str(label).replace(".", "").replace("-", "").isdigit() for label in config.group_labels)
        )

        for subject, visits in pairing_data.items():
            if config.paired_label1 in visits and config.paired_label2 in visits:
                baseline = visits[config.paired_label1]
                followup = visits[config.paired_label2]

                # For continuous analysis, we expect same dose at both timepoints (dose-response over time)
                # For categorical analysis, we expect same group at both timepoints
                if baseline["comparison"] == followup["comparison"]:
                    complete_pairs.append(
                        {
                            "subject": subject,
                            "group": baseline["comparison"],
                            "baseline_sample": baseline["sample"],
                            "followup_sample": followup["sample"],
                        }
                    )
                else:
                    incomplete_subjects.append(f"{subject} (mixed groups)")
            else:
                available_visits = list(visits.keys())
                incomplete_subjects.append(f"{subject} (missing visits: {available_visits})")

        # Group complete pairs by treatment group - normalize for comparison
        group_pairs = {}
        for group_label in config.group_labels:
            normalized_group = _normalize_group_value(group_label)
            group_pairs[group_label] = [
                p for p in complete_pairs if _normalize_group_value(p["group"]) == normalized_group
            ]

        if is_continuous_analysis:
            print("  Analysis type: CONTINUOUS dose-response (group variable as numeric)")
            print(f"  Complete paired subjects: {len(complete_pairs)} (group maintained across timepoints)")
            print("  Distribution across complete pairs:")
            for group, pairs in group_pairs.items():
                print(f"    {group}: {len(pairs)} subjects")
        else:
            print("  Analysis type: CATEGORICAL group comparison")
            print("  Complete paired subjects by group:")
            for group, pairs in group_pairs.items():
                print(f"    {group}: {len(pairs)} subjects")

        if incomplete_subjects:
            print(f"  Incomplete subjects: {len(incomplete_subjects)}")
            if len(incomplete_subjects) <= 5:  # Show details if few
                for subject_info in incomplete_subjects:
                    print(f"    {subject_info}")

    # Step 4: Run statistical analysis
    print(f"\nStep 3: Running {config.statistical_test_method} analysis...")

    # Filter protein data to samples with metadata
    available_samples = metadata_df["Sample"].tolist()
    filtered_protein_data = statistical_data[available_samples].copy()

    # Ensure the index contains actual protein identifiers (not integer row numbers)
    # so that iterrows() in test functions yields meaningful Protein values
    if "Protein" in statistical_data.columns:
        filtered_protein_data.index = statistical_data["Protein"].values

    print(f"  Method: {config.statistical_test_method}")
    print(f"  Analysis type: {config.analysis_type if config.analysis_type else 'not specified'}")
    print(f"  Proteins: {len(filtered_protein_data)}")
    print(f"  Samples: {len(available_samples)}")

    # Print analysis-specific parameters
    time_col = getattr(config, "time_column", None) or getattr(config, "dose_column", None)
    if config.analysis_type in ("linear_trend", "dose_response", "longitudinal"):
        if time_col:
            print(f"  Time variable: {time_col}")
    elif config.analysis_type in ["paired", "unpaired", "interaction"]:
        if config.group_labels:
            print(f"  Groups: {config.group_labels}")
        if config.paired_label1 and config.paired_label2:
            print(f"  Timepoints: {config.paired_label1} -> {config.paired_label2}")

    # Print subject grouping for mixed-effects
    if config.subject_column:
        print(f"  Subject grouping: {config.subject_column}")

    if config.statistical_test_method == "mixed_effects":
        print(f"  Interaction terms: {config.interaction_terms + config.additional_interactions}")
        if config.covariates:
            print(f"  Covariates: {config.covariates}")

    # Run appropriate analysis
    if config.statistical_test_method == "mixed_effects":
        results_df = run_mixed_effects_analysis(filtered_protein_data, metadata_df, config, protein_annotations)
    elif config.statistical_test_method in ["paired_t", "paired_welch"]:
        results_df = run_paired_t_test(filtered_protein_data, metadata_df, config)
    elif config.statistical_test_method in ["welch_t", "student_t"]:
        results_df = run_unpaired_t_test(filtered_protein_data, metadata_df, config)
    elif config.statistical_test_method == "wilcoxon":
        results_df = run_wilcoxon_test(filtered_protein_data, metadata_df, config)
    elif config.statistical_test_method == "mann_whitney":
        results_df = run_mann_whitney_test(filtered_protein_data, metadata_df, config)
    else:
        raise ValueError(
            f"Unknown statistical method: {config.statistical_test_method}. "
            f"Supported methods: mixed_effects, paired_t, paired_welch, welch_t, student_t, wilcoxon, mann_whitney"
        )

    # Apply multiple testing correction
    results_df = apply_multiple_testing_correction(results_df, config)

    # Merge with protein annotations if provided
    if protein_annotations is not None and len(protein_annotations) > 0:
        print("\nStep 6: Adding protein annotations...")

        # Get annotation columns (exclude sample columns and duplicates)
        annotation_cols = ["Protein"]
        potential_cols = [
            "Description",
            "Gene",
            "Protein Gene",
            "UniProt_Accession",
            "UniProt_Entry_Name",
            "UniProt_Database",
        ]

        for col in potential_cols:
            if col in protein_annotations.columns:
                annotation_cols.append(col)

        # Merge statistical results with annotations
        annotations_subset = protein_annotations[annotation_cols].copy()
        results_df = results_df.merge(annotations_subset, on="Protein", how="left")

        # Add Gene column if we have 'Protein Gene' but not 'Gene'
        if "Protein Gene" in results_df.columns and "Gene" not in results_df.columns:
            results_df["Gene"] = results_df["Protein Gene"]

        print(f"  Added annotation columns: {annotation_cols[1:]}")  # Skip 'Protein' as it's the key
    else:
        print("\nStep 6: No protein annotations provided - skipping annotation merge")

    # Sort by p-value
    results_df = results_df.sort_values("P.Value")

    print("\nDone: Statistical analysis completed!")
    print(f"  Total proteins analyzed: {len(results_df)}")
    print(f"  Proteins with valid results: {results_df['P.Value'].notna().sum()}")
    print(f"  Significant proteins (FDR < 0.05): {(results_df['adj.P.Val'] < 0.05).sum()}")

    return results_df


def display_analysis_summary(differential_results, config, label_top_n=10):
    """
    Display summary of statistical analysis results

    Parameters:
    -----------
    differential_results : pd.DataFrame
        Results from statistical analysis
    config : StatisticalConfig
        Configuration object with analysis parameters
    label_top_n : int
        Number of top significant proteins to display

    Returns:
    --------
    dict
        Summary statistics for downstream use
    """

    if differential_results is None or len(differential_results) == 0:
        print("Warning: No differential analysis results available")
        return {}

    print("=" * 60)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 60)

    # Basic statistics
    total_proteins = len(differential_results)
    valid_results = differential_results["P.Value"].notna().sum()
    significant_005 = (differential_results["adj.P.Val"] < 0.05).sum()
    significant_001 = (differential_results["adj.P.Val"] < 0.01).sum()

    print("Analysis Overview:")
    print(f"  Method: {config.statistical_test_method.upper()}")
    print(f"  Total proteins analyzed: {total_proteins:,}")
    print(f"  Proteins with valid results: {valid_results:,}")
    print(f"  Significant proteins (FDR < 0.05): {significant_005:,}")
    print(f"  Highly significant (FDR < 0.01): {significant_001:,}")

    if valid_results == 0:
        print("\nError: No valid statistical results found")
        return {}

    # Show top significant results
    successful_results = differential_results[differential_results["P.Value"].notna()]

    if len(successful_results) > 0:
        print(f"\n=== TOP {label_top_n} MOST SIGNIFICANT PROTEINS ===")

        top_results = successful_results.nsmallest(label_top_n, "P.Value")

        # Choose appropriate columns based on analysis type.
        # Prefer human-readable identifiers over the internal PG#### index.
        id_cols = [c for c in ["Gene", "UniProt_Accession", "Description"] if c in top_results.columns]
        if not id_cols:
            id_cols = ["Protein"]  # fall back to PG index if no annotations present

        if config.statistical_test_method == "mixed_effects":
            # Mixed-effects model results - use primary columns only
            display_cols = id_cols + ["logFC", "P.Value", "adj.P.Val", "n_obs"]

            # Note: logFC and P.Value already contain interaction results
            # No need to show duplicate interaction_coef and interaction_pvalue

        else:
            # Traditional statistical test results
            display_cols = id_cols + ["logFC", "P.Value", "adj.P.Val"]
            if "Effect_Size" in top_results.columns:
                display_cols.insert(len(id_cols), "Effect_Size")

        # Filter to available columns
        available_cols = [col for col in display_cols if col in top_results.columns]

        if available_cols:
            # Create a clean display dataframe with only the specified columns
            display_df = pd.DataFrame()
            for col in available_cols:
                display_df[col] = top_results[col].copy()

            # Format columns for better display
            for col in display_df.columns:
                if col in ["P.Value", "adj.P.Val"]:
                    display_df[col] = display_df[col].apply(
                        lambda x: f"{x:.2e}" if pd.notna(x) and x < 0.01 else f"{x:.6f}" if pd.notna(x) else "N/A"
                    )
                elif col in ["logFC", "Effect_Size"]:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
                elif col == "Description":
                    # Truncate long descriptions to keep the table readable
                    display_df[col] = display_df[col].apply(
                        lambda x: (x[:50] + "…") if isinstance(x, str) and len(x) > 51 else x
                    )

            print(display_df.to_string(index=False))

        # Additional analysis-specific summary
        if config.statistical_test_method == "mixed_effects":
            # Use P.Value column since it contains the interaction p-values
            interaction_significant = (successful_results["P.Value"] < 0.05).sum()
            print("\nInteraction Effects:")
            print(f"  Significant {' × '.join(config.interaction_terms)} interactions: {interaction_significant}")

    else:
        print("\nError: No proteins with valid statistical results")

        # Show failure reasons if available
        failed_results = differential_results[differential_results["P.Value"].isna()]
        if "test_method" in failed_results.columns and len(failed_results) > 0:
            print("\nFailure Analysis:")
            failure_reasons = failed_results["test_method"].value_counts()
            for reason, count in failure_reasons.items():
                print(f"  {reason}: {count}")

    # Create summary dictionary for return
    summary = {
        "total_proteins": total_proteins,
        "valid_results": valid_results,
        "significant_005": significant_005,
        "significant_001": significant_001,
        "analysis_method": config.statistical_test_method,
        "success_rate": valid_results / total_proteins if total_proteins > 0 else 0,
    }

    print("\nDone: Analysis summary complete!")

    return summary


# Maintain backwards compatibility
def run_statistical_analysis(normalized_data, sample_metadata, config, protein_annotations=None):
    """Backwards compatible wrapper for run_comprehensive_statistical_analysis"""
    return run_comprehensive_statistical_analysis(normalized_data, sample_metadata, config, protein_annotations)
