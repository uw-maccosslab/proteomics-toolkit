"""
Data Normalization Module for Proteomics Analysis Toolkit

Functions for normalizing proteomics data using various methods.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize


def _extract_standard_annotation_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and standardize the 5 required annotation columns from processed data:
    1. Protein (unchanged)
    2. Description (from existing parsed Description column)
    3. Protein Gene (unchanged)
    4. UniProt_Accession (from existing parsed column)
    5. UniProt_Entry_Name (from existing parsed column)

    This function assumes the data has already been processed by preprocessing functions
    that parse protein identifiers and create the necessary annotation columns.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataframe with processed protein annotation data

    Returns:
    --------
    pd.DataFrame
        DataFrame with exactly 5 standardized annotation columns
    """
    # Define the exact 5 annotation columns we want (in order)
    required_cols = [
        "Protein",
        "Description",
        "Protein Gene",
        "UniProt_Accession",
        "UniProt_Entry_Name",
    ]

    # Extract only these columns
    result = data[required_cols].copy()

    return result


def _create_standardized_column_order(annotation_data: pd.DataFrame, sample_data: pd.DataFrame) -> List[str]:
    """
    Create standardized column order with exactly 5 annotation columns first, then sample columns.

    ENFORCES the exact standardized structure to prevent annotation column scattering.

    Parameters:
    -----------
    annotation_data : pd.DataFrame
        DataFrame containing exactly the 5 standardized annotation columns
    sample_data : pd.DataFrame
        DataFrame containing sample columns

    Returns:
    --------
    List[str]
        Ordered list of column names in standardized format

    Raises:
    -------
    ValueError: If annotation_data doesn't have the expected structure
    """
    expected_annotation_cols = [
        "Protein",
        "Description",
        "Protein Gene",
        "UniProt_Accession",
        "UniProt_Entry_Name",
    ]

    actual_cols = list(annotation_data.columns)
    if actual_cols != expected_annotation_cols:
        raise ValueError(
            f"Annotation data must have standardized columns.\nExpected: {expected_annotation_cols}\nGot: {actual_cols}"
        )

    # Combine standardized annotation columns + sample columns
    final_column_order = expected_annotation_cols + list(sample_data.columns)

    return final_column_order


def get_normalization_characteristics() -> Dict[str, Dict[str, Any]]:
    """
    Get characteristics of each normalization method.

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary with normalization method characteristics
    """
    return {
        "median": {
            "preserves_scale": True,
            "log_transformed": False,
            "description": "Median normalization - keeps data on original scale",
        },
        "vsn": {
            "preserves_scale": False,
            "log_transformed": True,
            "description": "Variance Stabilizing Normalization - arcsinh transformed",
        },
        "quantile": {
            "preserves_scale": True,
            "log_transformed": False,
            "description": "Quantile normalization - keeps data on original scale",
        },
        "mad": {
            "preserves_scale": True,
            "log_transformed": False,
            "description": "Median Absolute Deviation - keeps data on original scale",
        },
        "z-score": {
            "preserves_scale": True,
            "log_transformed": False,
            "description": "Z-score normalization - keeps data on original scale",
        },
        "rlr": {
            "preserves_scale": False,
            "log_transformed": True,
            "description": "Robust Linear Regression - log2 transformed",
        },
        "loess": {
            "preserves_scale": False,
            "log_transformed": True,
            "description": "LOESS normalization - log2 transformed",
        },
        "none": {
            "preserves_scale": True,
            "log_transformed": False,
            "description": "No normalization applied",
        },
    }


def is_normalization_log_transformed(normalization_method: str) -> bool:
    """
    Check if a normalization method produces log-transformed data.

    Parameters:
    -----------
    normalization_method : str
        Name of the normalization method

    Returns:
    --------
    bool
        True if method produces log-transformed data, False otherwise
    """
    characteristics = get_normalization_characteristics()
    method_lower = normalization_method.lower()

    if method_lower in characteristics:
        return characteristics[method_lower]["log_transformed"]
    else:
        # Unknown method - assume it's not log transformed
        return False


def _separate_sample_and_annotation_data(data: pd.DataFrame, sample_columns: Optional[list] = None) -> tuple:
    """
    Separate sample and annotation data from standardized data structure.

    EXPECTS standardized data structure from create_standard_data_structure():
    - Columns 0-4: EXACTLY (Protein, Description, Protein Gene, UniProt_Accession, UniProt_Entry_Name)
    - Columns 5+: Sample columns

    This function is STRICT to prevent annotation columns from being scattered.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data with standardized structure (must have been processed by create_standard_data_structure)
    sample_columns : Optional[list]
        List of sample column names. If None, uses columns 5+ from standardized structure

    Returns:
    --------
    tuple : (sample_data, annotation_data)
        - sample_data: DataFrame with only sample columns
        - annotation_data: DataFrame with only the 5 standardized annotation columns

    Raises:
    -------
    ValueError: If data doesn't have the expected standardized structure
    """
    expected_annotation_cols = [
        "Protein",
        "Description",
        "Protein Gene",
        "UniProt_Accession",
        "UniProt_Entry_Name",
    ]

    # Strict validation: Check if data has the expected standardized structure
    if len(data.columns) < 5:
        raise ValueError(
            f"Data must have at least 5 columns for standardized structure. Got {len(data.columns)} columns. "
            f"Use create_standard_data_structure() first."
        )

    actual_annotation_cols = list(data.columns[:5])
    if actual_annotation_cols != expected_annotation_cols:
        raise ValueError(
            f"Data does not have standardized annotation structure.\n"
            f"Expected: {expected_annotation_cols}\n"
            f"Got: {actual_annotation_cols}\n"
            f"Use create_standard_data_structure() to standardize the data first."
        )

    # Extract the standardized annotation and sample data
    annotation_data = data.iloc[:, :5].copy()  # First 5 columns (standardized)

    if sample_columns is None:
        # Use all columns after the first 5 as samples
        sample_data = data.iloc[:, 5:].copy() if len(data.columns) > 5 else pd.DataFrame(index=data.index)
    else:
        # Use specified sample columns (must be in columns 5+)
        sample_data = data[sample_columns].copy()

    return sample_data, annotation_data


def median_normalize(data: pd.DataFrame, sample_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Median normalization - divide by sample median, multiply by global median.
    This keeps data on the original scale.

    Handles DataFrames with mixed annotation and sample columns automatically.

    Parameters:
    -----------
    data : pd.DataFrame
        Raw intensity data (can contain annotation columns)
    sample_columns : Optional[list]
        List of sample column names. If None, auto-detects numeric columns

    Returns:
    --------
    pd.DataFrame : Median normalized data with same structure as input
    """

    print("Applying median normalization...")

    # Separate sample and annotation data using standardized structure
    sample_data, annotation_data = _separate_sample_and_annotation_data(data, sample_columns)

    if sample_data.empty:
        print("Warning: No numeric sample columns found for normalization")
        return data.copy()

    # Normalize only the sample columns
    normalized_sample_data = sample_data.copy()

    # Calculate median for each sample
    sample_medians = sample_data.median(axis=0)

    # Calculate global median across all samples
    global_median = sample_medians.median()

    # Normalize: divide by sample median, multiply by global median
    for col in sample_data.columns:
        if sample_medians[col] > 0:  # Avoid division by zero
            normalized_sample_data[col] = (sample_data[col] / sample_medians[col]) * global_median

    # Reconstruct the complete DataFrame with standardized column order
    if not annotation_data.empty:
        # Create standardized column order: annotation columns first, then sample columns
        final_column_order = _create_standardized_column_order(annotation_data, normalized_sample_data)

        # Combine annotation columns with normalized sample data
        result = pd.concat([annotation_data, normalized_sample_data], axis=1)

        # Reorder columns to match standardized order
        result = result[final_column_order]
    else:
        result = normalized_sample_data

    print(f"Median normalization completed for {len(sample_data.columns)} samples")

    return result


def vsn_normalize(
    data: pd.DataFrame,
    optimize_params: bool = False,
    sample_columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Variance Stabilizing Normalization (VSN) using arcsinh transformation.

    This method applies an arcsinh transformation to stabilize variance across
    intensity ranges.

    Parameters:
    -----------
    data : pd.DataFrame
        Raw intensity data (original scale)
    optimize_params : bool
        Whether to optimize VSN parameters (default False for speed)
    sample_columns : list, optional
        List of sample column names. If None, will attempt to detect automatically.

    Returns:
    --------
    pd.DataFrame : VSN normalized data with preserved annotation columns
    """
    print("Starting VSN normalization...")

    # Separate annotation and sample columns
    sample_data, annotation_data = _separate_sample_and_annotation_data(data, sample_columns)

    def vsn_transformation_scipy(data_series: pd.Series, optimize_params: bool = False) -> pd.Series:
        """Apply VSN transformation to a single sample"""
        data_values = np.asarray(data_series.values)  # Convert to numpy array

        # Remove zeros and negative values for parameter estimation
        data_clean = data_values[data_values > 0]
        data_clean = np.asarray(data_clean)  # Convert to numpy array for calculations

        if len(data_clean) == 0:
            return pd.Series(np.zeros_like(data_values), index=data_series.index)

        if optimize_params:
            # Optimize VSN parameters for variance stabilization
            def variance_heterogeneity(params):
                a, b = params
                if a <= 0:  # Ensure positive scaling
                    return 1e6

                # Apply transformation
                transformed = np.arcsinh(a * data_clean + b)

                # Sort by original intensity
                sorted_indices = np.argsort(data_clean)
                sorted_transformed = transformed[sorted_indices]

                # Calculate rolling window variances
                window_size = max(100, len(data_clean) // 20)
                variances = []

                for i in range(0, len(sorted_transformed) - window_size, window_size // 4):
                    window_data = sorted_transformed[i : i + window_size]
                    if len(window_data) > 10:
                        variances.append(np.var(window_data))

                # Return coefficient of variation of variances (want this small)
                if len(variances) > 1:
                    mean_var = np.mean(variances)
                    if mean_var > 0:
                        return np.std(variances) / mean_var
                    else:
                        return 1e6
                else:
                    return 1e6

            # Try multiple starting points for robustness
            best_result = None
            best_score = float("inf")

            starting_points = [
                [1.0, 0.0],
                [0.1, 0.0],
                [10.0, 0.0],
                [1 / np.median(data_clean), 0.0],
                [1 / np.quantile(data_clean, 0.5), 0.0],
            ]

            for start_params in starting_points:
                try:
                    result = optimize.minimize(
                        variance_heterogeneity,
                        start_params,
                        method="Nelder-Mead",
                        options={"maxiter": 500},
                    )

                    if result.success and result.fun < best_score:
                        best_result = result
                        best_score = result.fun
                except (ValueError, RuntimeError):
                    continue

            if best_result is not None and best_result.success:
                a_opt, b_opt = best_result.x
            else:
                # Fallback to quantile-based approach
                a_opt = 1.0 / np.quantile(data_clean, 0.5)
                b_opt = 0.0
        else:
            # Use a conservative quantile-based approach (faster)
            q50 = np.quantile(data_clean, 0.5)  # median
            a_opt = 1.0 / q50  # Scale based on median
            b_opt = 0.0

        # Apply transformation to all data (including zeros)
        # Handle zeros by adding small offset
        data_for_transform = np.where(data_values == 0, 1e-6, data_values)
        transformed = np.arcsinh(a_opt * data_for_transform + b_opt)

        return pd.Series(transformed, index=data_series.index)

    # Apply VSN transformation to each sample column
    normalized_sample_data = sample_data.copy()

    print(f"Applying VSN transformation to {len(sample_data.columns)} samples...")
    for i, col in enumerate(sample_data.columns):
        if (i + 1) % 5 == 0 or i == 0:  # Progress indicator
            print(f"Processing sample {i + 1}/{len(sample_data.columns)}: {col}")

        normalized_sample_data[col] = vsn_transformation_scipy(sample_data[col], optimize_params=optimize_params)

    print("VSN transformation completed!")

    # Combine annotation data with normalized sample data using standardized column order
    if annotation_data is not None:
        # Create standardized column order: annotation columns first, then sample columns
        final_column_order = _create_standardized_column_order(annotation_data, normalized_sample_data)

        result = pd.concat(
            [
                annotation_data.reset_index(drop=True),
                normalized_sample_data.reset_index(drop=True),
            ],
            axis=1,
        )

        # Reorder columns to match standardized order
        result = result[final_column_order]
        result.index = data.index
    else:
        result = normalized_sample_data
        result.index = data.index

    return result


def quantile_normalize(data: pd.DataFrame, sample_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Quantile normalization - makes the distribution of each sample identical.

    Parameters:
    -----------
    data : pd.DataFrame
        Raw intensity data
    sample_columns : list, optional
        List of sample column names. If None, will attempt to detect automatically.

    Returns:
    --------
    pd.DataFrame : Quantile normalized data with preserved annotation columns
    """
    print("Starting quantile normalization...")

    # Separate annotation and sample columns
    sample_data, annotation_data = _separate_sample_and_annotation_data(data, sample_columns)

    print("Applying quantile normalization...")

    # Convert to numpy for faster processing
    data_matrix = sample_data.values

    # Sort each column
    sorted_indices = np.argsort(data_matrix, axis=0)
    sorted_data = np.sort(data_matrix, axis=0)

    # Calculate row means (quantile means)
    quantile_means = np.mean(sorted_data, axis=1)

    # Create normalized data by replacing sorted values with quantile means
    normalized_matrix = np.empty_like(data_matrix)

    for i in range(data_matrix.shape[1]):
        # Restore original order using sorted indices
        normalized_matrix[sorted_indices[:, i], i] = quantile_means

    # Convert back to DataFrame
    normalized_sample_data = pd.DataFrame(normalized_matrix, index=sample_data.index, columns=sample_data.columns)

    print(f"Quantile normalization completed for {len(sample_data.columns)} samples")

    # Combine annotation data with normalized sample data using standardized column order
    if annotation_data is not None:
        # Create standardized column order: annotation columns first, then sample columns
        final_column_order = _create_standardized_column_order(annotation_data, normalized_sample_data)

        result = pd.concat(
            [
                annotation_data.reset_index(drop=True),
                normalized_sample_data.reset_index(drop=True),
            ],
            axis=1,
        )

        # Reorder columns to match standardized order
        result = result[final_column_order]
        result.index = data.index
    else:
        result = normalized_sample_data
        result.index = data.index

    return result


def log_transform(data: pd.DataFrame, base: str = "log2", pseudocount: Optional[float] = None) -> pd.DataFrame:
    """
    Apply log transformation to data.

    Parameters:
    -----------
    data : pd.DataFrame
        Data to transform
    base : str
        Log base ('log2', 'log10', or 'ln')
    pseudocount : float, optional
        Small value to add before log transform (auto-calculated if None)

    Returns:
    --------
    pd.DataFrame : Log-transformed data
    """

    # Calculate pseudocount if not provided
    if pseudocount is None:
        min_positive = data[data > 0].min().min()
        pseudocount = min_positive / 10 if min_positive > 0 else 1e-6

    # Add pseudocount to handle zeros
    data_with_pseudo = data + pseudocount

    # Apply appropriate log transformation
    if base == "log2":
        transformed_data = np.log2(data_with_pseudo)
    elif base == "log10":
        transformed_data = np.log10(data_with_pseudo)
    elif base == "ln":
        transformed_data = np.log(data_with_pseudo)
    else:
        raise ValueError("base must be 'log2', 'log10', or 'ln'")

    print(f"Applied {base} transformation with pseudocount {pseudocount}")

    # Return as DataFrame with original index and column names
    return pd.DataFrame(transformed_data, index=data.index, columns=data.columns)


def mad_normalize(data: pd.DataFrame, sample_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Median Absolute Deviation (MAD) normalization.

    More robust than median normalization as it uses MAD instead of standard deviation.
    Formula: (x - median) / MAD, then rescaled to original median range

    Parameters:
    -----------
    data : pd.DataFrame
        Raw intensity data
    sample_columns : list, optional
        List of sample column names. If None, will attempt to detect automatically.

    Returns:
    --------
    pd.DataFrame : MAD normalized data with preserved annotation columns
    """
    print("Starting MAD normalization...")

    # Separate annotation and sample columns
    sample_data, annotation_data = _separate_sample_and_annotation_data(data, sample_columns)

    print("Applying MAD (Median Absolute Deviation) normalization...")

    normalized_sample_data = sample_data.copy()

    # Calculate median and MAD for each sample
    sample_medians = sample_data.median(axis=0)
    global_median = sample_medians.median()

    # Calculate MAD for each sample
    sample_mads = pd.Series(index=sample_data.columns, dtype=float)
    for col in sample_data.columns:
        sample_mads[col] = np.median(np.abs(sample_data[col] - sample_medians[col]))

    global_mad = sample_mads.median()

    # Normalize: (x - sample_median) / sample_MAD * global_MAD + global_median
    for col in sample_data.columns:
        if sample_mads.loc[col] > 0:  # Avoid division by zero
            normalized_sample_data[col] = (
                (sample_data[col] - sample_medians[col]) / sample_mads.loc[col]
            ) * global_mad + global_median

    print(f"MAD normalization completed for {len(sample_data.columns)} samples")

    # Combine annotation data with normalized sample data using standardized column order
    if annotation_data is not None:
        # Create standardized column order: annotation columns first, then sample columns
        final_column_order = _create_standardized_column_order(annotation_data, normalized_sample_data)

        result = pd.concat(
            [
                annotation_data.reset_index(drop=True),
                normalized_sample_data.reset_index(drop=True),
            ],
            axis=1,
        )

        # Reorder columns to match standardized order
        result = result[final_column_order]
        result.index = data.index
    else:
        result = normalized_sample_data
        result.index = data.index

    return result


def z_score_normalize(data: pd.DataFrame, sample_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Z-score normalization (standardization).

    Each sample is transformed to have mean=0 and std=1, then rescaled to original range.
    Formula: (x - mean) / std

    Parameters:
    -----------
    data : pd.DataFrame
        Raw intensity data
    sample_columns : list, optional
        List of sample column names. If None, will attempt to detect automatically.

    Returns:
    --------
    pd.DataFrame : Z-score normalized data with preserved annotation columns
    """
    print("Starting Z-score normalization...")

    # Separate annotation and sample columns
    sample_data, annotation_data = _separate_sample_and_annotation_data(data, sample_columns)

    print("Applying Z-score normalization...")

    # Calculate sample-wise statistics
    sample_means = sample_data.mean(axis=0)
    sample_stds = sample_data.std(axis=0)
    global_mean = sample_means.mean()
    global_std = sample_stds.mean()

    # Z-score normalize each sample, then rescale
    normalized_sample_data = sample_data.copy()
    for col in sample_data.columns:
        if sample_stds.loc[col] > 0:  # Avoid division by zero
            normalized_sample_data[col] = (
                (sample_data[col] - sample_means[col]) / sample_stds.loc[col]
            ) * global_std + global_mean

    print(f"Z-score normalization completed for {len(sample_data.columns)} samples")

    # Combine annotation data with normalized sample data using standardized column order
    if annotation_data is not None:
        # Create standardized column order: annotation columns first, then sample columns
        final_column_order = _create_standardized_column_order(annotation_data, normalized_sample_data)

        result = pd.concat(
            [
                annotation_data.reset_index(drop=True),
                normalized_sample_data.reset_index(drop=True),
            ],
            axis=1,
        )

        # Reorder columns to match standardized order
        result = result[final_column_order]
        result.index = data.index
    else:
        result = normalized_sample_data
        result.index = data.index

    return result


def rlr_normalize(data: pd.DataFrame, sample_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Robust Linear Regression (RLR) normalization.

    Uses robust regression to normalize against a pseudo-reference sample.
    More robust to outliers than simple linear normalization.

    Parameters:
    -----------
    data : pd.DataFrame
        Raw intensity data
    sample_columns : list, optional
        List of sample column names. If None, will attempt to detect automatically.

    Returns:
    --------
    pd.DataFrame : RLR normalized data with preserved annotation columns
    """
    print("Starting RLR normalization...")

    # Separate annotation and sample columns
    sample_data, annotation_data = _separate_sample_and_annotation_data(data, sample_columns)

    print("Applying RLR (Robust Linear Regression) normalization...")

    # Create pseudo-reference sample (median across all samples)
    pseudo_reference = sample_data.median(axis=1)

    # Log transform for linear regression
    log_data = np.log2(sample_data.replace(0, np.nan))
    log_reference = np.log2(pseudo_reference.replace(0, np.nan))

    normalized_sample_data = sample_data.copy()

    for col in sample_data.columns:
        # Get valid (non-NaN) data points
        col_data = log_data[col]
        valid_mask = ~(pd.isna(col_data) | pd.isna(log_reference))

        if valid_mask.sum() > 10:  # Need sufficient points for regression
            # Convert to numpy arrays for processing
            x_values = np.array(log_reference[valid_mask])
            y_values = np.array(col_data[valid_mask])

            # Simple robust regression (using median-based approach)
            # Calculate slope using Theil-Sen estimator approximation
            differences = []
            for i in range(0, len(x_values) - 1, max(1, len(x_values) // 100)):  # Sample points to avoid O(n²)
                for j in range(i + 1, min(i + 50, len(x_values))):  # Limited pairwise comparisons
                    if abs(x_values[j] - x_values[i]) > 1e-6:  # Avoid division by near-zero
                        slope = (y_values[j] - y_values[i]) / (x_values[j] - x_values[i])
                        differences.append(slope)

            if differences:
                robust_slope = np.median(differences)
                robust_intercept = np.median(y_values - robust_slope * x_values)

                # Apply correction: divide by slope to normalize
                if robust_slope != 0:
                    correction_factor = float(2 ** (robust_intercept / robust_slope))
                    normalized_sample_data[col] = normalized_sample_data[col] / correction_factor

    print(f"RLR normalization completed for {len(sample_data.columns)} samples")

    # Combine annotation data with normalized sample data using standardized column order
    if annotation_data is not None:
        # Create standardized column order: annotation columns first, then sample columns
        final_column_order = _create_standardized_column_order(annotation_data, normalized_sample_data)

        result = pd.concat(
            [
                annotation_data.reset_index(drop=True),
                normalized_sample_data.reset_index(drop=True),
            ],
            axis=1,
        )

        # Reorder columns to match standardized order
        result = result[final_column_order]
        result.index = data.index
    else:
        result = normalized_sample_data
        result.index = data.index

    return result


def loess_normalize(data: pd.DataFrame, span: float = 0.75, sample_columns: Optional[list] = None) -> pd.DataFrame:
    """
    LOESS (LOcally WEighted Scatterplot Smoothing) normalization.

    Uses local weighted regression to normalize against intensity-dependent effects.
    Good for correcting systematic biases that vary with intensity.

    Parameters:
    -----------
    data : pd.DataFrame
        Raw intensity data
    span : float
        Fraction of data used for each local regression (0.1 to 1.0)
    sample_columns : list, optional
        List of sample column names. If None, will attempt to detect automatically.

    Returns:
    --------
    pd.DataFrame : LOESS normalized data with preserved annotation columns
    """
    print("Starting LOESS normalization...")

    # Separate annotation and sample columns
    sample_data, annotation_data = _separate_sample_and_annotation_data(data, sample_columns)

    print(f"Applying LOESS normalization (span={span})...")

    # Create pseudo-reference sample (median across all samples)
    pseudo_reference = sample_data.median(axis=1)

    # Log transform
    log_data = np.log2(sample_data.replace(0, np.nan))
    log_reference = np.log2(pseudo_reference.replace(0, np.nan))

    normalized_sample_data = sample_data.copy()

    for col in sample_data.columns:
        # Get valid data points
        col_data = log_data[col]
        valid_mask = ~(pd.isna(col_data) | pd.isna(log_reference))

        if valid_mask.sum() > 20:  # Need sufficient points
            # Convert to numpy arrays
            x = np.array(log_reference[valid_mask])
            y = np.array(col_data[valid_mask])

            # Simple LOESS approximation using rolling median
            # Sort by intensity for local smoothing
            sort_idx = np.argsort(x)
            x_sorted = x[sort_idx]
            y_sorted = y[sort_idx]

            # Calculate local medians
            window_size = max(10, int(len(x_sorted) * span))
            smoothed = np.zeros_like(y_sorted)

            for i in range(len(x_sorted)):
                # Define local window
                start = max(0, i - window_size // 2)
                end = min(len(x_sorted), i + window_size // 2)

                # Local median difference
                local_diff = np.median(y_sorted[start:end] - x_sorted[start:end])
                smoothed[i] = x_sorted[i] + local_diff

            # Interpolate back to original order
            smooth_correction = np.interp(x, x_sorted, smoothed[sort_idx])

            # Apply correction
            correction = y - smooth_correction

            # Get the original indices for valid data
            valid_indices = np.where(valid_mask)[0]

            for idx, corr in zip(valid_indices, correction):
                original_idx = sample_data.index[idx]
                normalized_sample_data.loc[original_idx, col] = sample_data.loc[original_idx, col] / (2**corr)

    print(f"LOESS normalization completed for {len(sample_data.columns)} samples")

    # Combine annotation data with normalized sample data using standardized column order
    if annotation_data is not None:
        # Create standardized column order: annotation columns first, then sample columns
        final_column_order = _create_standardized_column_order(annotation_data, normalized_sample_data)

        result = pd.concat(
            [
                annotation_data.reset_index(drop=True),
                normalized_sample_data.reset_index(drop=True),
            ],
            axis=1,
        )

        # Reorder columns to match standardized order
        result = result[final_column_order]
        result.index = data.index
    else:
        result = normalized_sample_data
        result.index = data.index

    return result


def calculate_normalization_stats(data: pd.DataFrame, normalized_data: pd.DataFrame, method: str) -> Dict[str, float]:
    """
    Calculate statistics to assess normalization effectiveness.

    Parameters:
    -----------
    data : pd.DataFrame
        Original data
    normalized_data : pd.DataFrame
        Normalized data
    method : str
        Normalization method name

    Returns:
    --------
    Dict[str, float] : Normalization statistics
    """

    # Log transform both datasets for comparison (if not VSN)
    if method.lower() != "vsn":
        log2_original = pd.DataFrame(np.log2(data.replace(0, np.nan)), index=data.index, columns=data.columns)
        log2_normalized = pd.DataFrame(
            np.log2(normalized_data.replace(0, np.nan)),
            index=normalized_data.index,
            columns=normalized_data.columns,
        )
    else:
        log2_original = pd.DataFrame(np.log2(data.replace(0, np.nan)), index=data.index, columns=data.columns)
        log2_normalized = normalized_data  # VSN is already transformed

    # Calculate statistics
    stats = {
        "original_median_range": (log2_original.median(axis=0).max() - log2_original.median(axis=0).min()),
        "normalized_median_range": (log2_normalized.median(axis=0).max() - log2_normalized.median(axis=0).min()),
        "original_cv_median": (log2_original.std(axis=0) / log2_original.mean(axis=0)).median(),
        "normalized_cv_median": (log2_normalized.std(axis=0) / log2_normalized.mean(axis=0)).median(),
    }

    # Calculate reduction in median range
    stats["median_range_reduction"] = 1 - (stats["normalized_median_range"] / stats["original_median_range"])

    return stats


def calculate_detailed_normalization_stats(
    original_data: pd.DataFrame,
    normalized_data: pd.DataFrame,
    method: str,
    sample_metadata: dict = None,
    control_labels: list = None,
) -> dict:
    """
    Calculate detailed statistics to assess normalization effectiveness, with control-specific reporting.

    Parameters:
    -----------
    original_data : pd.DataFrame
        Original data (sample columns only)
    normalized_data : pd.DataFrame
        Normalized data (sample columns only)
    method : str
        Normalization method name
    sample_metadata : dict, optional
        Sample metadata dictionary {sample_name: metadata_dict}
    control_labels : list, optional
        List of control sample identifiers (e.g., ['HoofPool', 'GWPool', 'EISAIPool'])

    Returns:
    --------
    Dict[str, any] : Detailed normalization statistics
    """

    # Determine which scale to use for statistics
    if method.lower() == "vsn":
        # VSN produces variance-stabilized data (already transformed)
        original_for_stats = pd.DataFrame(
            np.log2(original_data.replace(0, np.nan)),
            index=original_data.index,
            columns=original_data.columns,
        )
        normalized_for_stats = normalized_data.copy()  # VSN is already transformed
        scale_type = "log2"
    elif method.lower() in ["rlr", "loess"]:
        # These methods apply log2 transformation during normalization
        original_for_stats = pd.DataFrame(
            np.log2(original_data.replace(0, np.nan)),
            index=original_data.index,
            columns=original_data.columns,
        )
        normalized_for_stats = normalized_data.copy()  # Already log2 transformed
        scale_type = "log2"
    else:
        # Median, Quantile, MAD, Z-score preserve linear scale
        original_for_stats = original_data.copy()
        normalized_for_stats = normalized_data.copy()
        scale_type = "linear"

    # Overall statistics
    original_sample_medians = original_for_stats.median(axis=0)
    normalized_sample_medians = normalized_for_stats.median(axis=0)

    overall_stats = {
        "method": method.upper(),
        "scale_type": scale_type,
        "original_sample_median_range": original_sample_medians.max() - original_sample_medians.min(),
        "normalized_sample_median_range": normalized_sample_medians.max() - normalized_sample_medians.min(),
        "original_cv_median_all_samples": (original_for_stats.std(axis=0) / original_for_stats.mean(axis=0)).median(),
        "normalized_cv_median_all_samples": (
            normalized_for_stats.std(axis=0) / normalized_for_stats.mean(axis=0)
        ).median(),
    }

    # Calculate range reduction
    if overall_stats["original_sample_median_range"] > 0:
        overall_stats["sample_median_range_reduction"] = 1 - (
            overall_stats["normalized_sample_median_range"] / overall_stats["original_sample_median_range"]
        )
    else:
        overall_stats["sample_median_range_reduction"] = 0.0

    # Control sample statistics (if available)
    control_stats = {}
    if sample_metadata and control_labels:
        for control_type in control_labels:
            # Find samples matching this control type
            control_samples = [
                sample
                for sample, meta in sample_metadata.items()
                if meta.get("Subject", "") == control_type and sample in original_for_stats.columns
            ]

            if len(control_samples) >= 2:  # Need at least 2 samples to calculate CV
                orig_control_data = original_for_stats[control_samples]
                norm_control_data = normalized_for_stats[control_samples]

                # Calculate CV for each protein across control replicates
                orig_cvs = (orig_control_data.std(axis=1) / orig_control_data.mean(axis=1)).replace(
                    [np.inf, -np.inf], np.nan
                )
                norm_cvs = (norm_control_data.std(axis=1) / norm_control_data.mean(axis=1)).replace(
                    [np.inf, -np.inf], np.nan
                )

                control_stats[control_type] = {
                    "n_samples": len(control_samples),
                    "original_median_cv": orig_cvs.median(),
                    "normalized_median_cv": norm_cvs.median(),
                    "cv_improvement": orig_cvs.median() - norm_cvs.median(),
                    "samples": control_samples,
                }

    return {"overall": overall_stats, "control_samples": control_stats}


def analyze_negative_values(data, normalization_method, sample_columns=None):
    """
    Analyze negative values in proteomics data.

    Parameters
    ----------
    data : pd.DataFrame
        Proteomics data to analyze
    normalization_method : str
        The normalization method used
    sample_columns : list, optional
        List of sample columns to analyze. If None, assumes all numeric columns

    Returns
    -------
    Dict[str, Any] : Analysis results including counts, statistics, and summaries
    """

    print("=== NEGATIVE VALUE ANALYSIS ===")
    print(f"Normalization method: {normalization_method}")
    print(f"Data shape: {data.shape}")

    # Separate annotation columns from sample data if needed
    if sample_columns is not None:
        sample_data = data[sample_columns]
    else:
        # Try to identify numeric columns automatically
        sample_data = data.select_dtypes(include=[np.number])
        if sample_data.empty:
            # Fallback: assume data is all numeric
            sample_data = data

    # Check for negative values
    negative_mask = sample_data < 0
    negative_count = negative_mask.sum().sum()
    total_values = sample_data.size

    print(f"\nNegative values found: {negative_count:,} out of {total_values:,} total values")
    print(f"Percentage of negative values: {negative_count / total_values * 100:.3f}%")

    analysis_results = {
        "negative_count": negative_count,
        "total_values": total_values,
        "negative_percentage": negative_count / total_values * 100,
        "has_negatives": negative_count > 0,
    }

    if negative_count > 0:
        # Show distribution of negative values
        min_value = sample_data.min().min()
        print(f"Most negative value: {min_value:.6f}")
        analysis_results["min_value"] = min_value

        # Count negative values per sample and per protein
        negative_per_sample = negative_mask.sum(axis=0)
        negative_per_protein = negative_mask.sum(axis=1)

        print("\nNegative values per sample:")
        print(f"  Range: {negative_per_sample.min()} - {negative_per_sample.max()}")
        print(f"  Mean: {negative_per_sample.mean():.1f}")

        print("\nNegative values per protein:")
        print(f"  Range: {negative_per_protein.min()} - {negative_per_protein.max()}")
        print(f"  Mean: {negative_per_protein.mean():.1f}")

        # Show samples with most negative values
        top_negative_samples = negative_per_sample.nlargest(5)
        print("\nSamples with most negative values:")
        for sample, count in top_negative_samples.items():
            print(f"  {sample}: {count} negative values")

        # Show proteins with negative values across many samples
        proteins_with_negatives = (negative_per_protein > 0).sum()
        print(f"\nProteins with at least one negative value: {proteins_with_negatives}")
        print(f"Percentage of proteins affected: {proteins_with_negatives / len(data) * 100:.1f}%")

        analysis_results.update(
            {
                "negative_per_sample": negative_per_sample,
                "negative_per_protein": negative_per_protein,
                "proteins_with_negatives": proteins_with_negatives,
                "top_negative_samples": top_negative_samples,
            }
        )

        # If VSN, explain this is normal
        if normalization_method.lower() == "vsn":
            print("\nNOTE: Negative values after VSN normalization are NORMAL and expected")
            print("   VSN uses sinh transformation which can produce negative values for low-abundance proteins")
            print("   These negative values represent proteins with intensities below the baseline")
            print("   They are biologically meaningful and mathematically valid in VSN space")
            print("   However, they may need handling for certain downstream analyses")

    else:
        print("No negative values detected")
        analysis_results["min_value"] = sample_data.min().min()

    zero_count = (sample_data == 0).sum().sum()
    print(f"\nZero values found: {zero_count:,}")
    print(f"Data range: {sample_data.min().min():.6f} to {sample_data.max().max():.6f}")

    # Fix the quantile calculation for flattened data
    flattened_data = sample_data.values.flatten()
    percentiles = np.percentile(flattened_data, [0, 25, 50, 75, 100])
    print(f"Data distribution (5 percentiles): {np.round(percentiles, 4).tolist()}")

    analysis_results.update(
        {
            "zero_count": zero_count,
            "data_range": (sample_data.min().min(), sample_data.max().max()),
            "percentiles": np.round(percentiles, 4).tolist(),
        }
    )

    print("\nDone: Negative value analysis completed")

    return analysis_results


def handle_negative_values(
    data: pd.DataFrame,
    method: str = "min_positive",
    min_positive_value: Optional[float] = None,
    replacement_value: Optional[float] = None,
    sample_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Handle negative values in normalized proteomics data.

    Parameters:
    -----------
    data : pd.DataFrame
        Normalized data that may contain negative values. Can contain both annotation
        columns (Protein, Gene, Description) and sample columns.
    method : str
        Method for handling negative values:
        - "min_positive": Replace with small positive value (1/10th of min positive)
        - "keep": Keep negative values unchanged
        - "zero": Replace with zero
        - "nan": Replace with NaN
        - "small_positive": Legacy alias for "min_positive"
        - "shift_global": Shift all data by making minimum value positive
        - "replace_value": Replace with specific value
    min_positive_value : float, optional
        For "min_positive" method, use this as the replacement value.
        If None, uses 1/10th of the smallest positive value
    replacement_value : float, optional
        For "replace_value" method, specific value to use
    sample_columns : list, optional
        List of sample column names. If None, will try to identify numeric columns automatically.

    Returns:
    --------
    pd.DataFrame : Data with negative values handled
    """

    print(f"\nApplying negative value handling: {method}")

    # Separate annotation columns from sample data if needed
    if sample_columns is not None:
        # Use specified sample columns
        sample_data = data[sample_columns]
        annotation_columns = [col for col in data.columns if col not in sample_columns]
    else:
        # Try to identify sample columns automatically (numeric columns)
        sample_data = data.select_dtypes(include=[np.number])
        if sample_data.empty:
            # Fallback: assume all data is numeric
            sample_data = data
            annotation_columns = []
        else:
            annotation_columns = [col for col in data.columns if col not in sample_data.columns]

    # Count negative values before handling
    negative_count = (sample_data < 0).sum().sum()
    print(f"  Found {negative_count} negative values to handle")

    if negative_count == 0:
        print("  No negative values found - returning original data")
        return data.copy()

    # Handle negative values based on method
    if method == "keep":
        print("  Keeping all negative values (no changes)")
        return data.copy()

    elif method in ["min_positive", "small_positive"]:  # Accept both names
        # Replace negatives with fraction of minimum positive value
        min_positive = sample_data[sample_data > 0].min().min()
        replacement_value = min_positive_value if min_positive_value is not None else min_positive / 10
        sample_data = sample_data.where(sample_data >= 0, replacement_value)
        print(f"  Replaced negatives with {replacement_value:.6f} (1/10th of min positive: {min_positive:.6f})")

    elif method == "zero":
        sample_data = sample_data.where(sample_data >= 0, 0)
        print("  Replaced negatives with zero")

    elif method == "nan":
        sample_data = sample_data.where(sample_data >= 0, np.nan)
        print("  Replaced negatives with NaN")

    elif method == "shift_global":
        min_value = sample_data.min().min()
        shift_amount = abs(min_value) + 0.001  # Make minimum value 0.001
        sample_data = sample_data + shift_amount
        print(f"  Shifted all values by: +{shift_amount:.6f}")
        print(f"  New minimum value: {sample_data.min().min():.6f}")

    elif method == "replace_value":
        if replacement_value is None:
            replacement_value = 1e-6
        sample_data = sample_data.where(sample_data >= 0, replacement_value)
        print(f"  Replaced negative values with: {replacement_value:.2e}")

    else:
        raise ValueError(f"Unknown negative handling method: {method}")

    # Reconstruct the full DataFrame using standardized column order
    if annotation_columns:
        # Get the annotation data
        annotation_data = data[annotation_columns]

        # Create standardized column order: annotation columns first, then sample columns
        final_column_order = _create_standardized_column_order(annotation_data, sample_data)

        result_data = pd.concat([data[annotation_columns], sample_data], axis=1)

        # Reorder columns to match standardized order
        result_data = result_data[final_column_order]
    else:
        result_data = sample_data

    # Verification
    final_negative_count = (result_data.select_dtypes(include=[np.number]) < 0).sum().sum()
    print(f"  Final negative count: {final_negative_count}")

    return result_data
