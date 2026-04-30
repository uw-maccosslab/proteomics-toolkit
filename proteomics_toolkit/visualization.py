"""
Visualization Module for Proteomics Analysis Toolkit

Functions for creating plots and visualizations for proteomics data analysis.
"""

import logging
import warnings
from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_import import BATCH_SUFFIX_DELIMITER

logger = logging.getLogger(__name__)


def _make_display_labels(sample_columns: List[str]) -> List[str]:
    """
    Create display-friendly labels from sample column names by stripping
    the PRISM batch suffix (__@__<batch_name>).

    For a single batch, the suffix is simply removed.
    For multiple batches, a short batch number (B1, B2, ...) is appended.
    """
    if not sample_columns:
        return []

    # Separate columns with and without the delimiter
    batch_names = []
    for col in sample_columns:
        if BATCH_SUFFIX_DELIMITER in col:
            batch_names.append(col.split(BATCH_SUFFIX_DELIMITER, 1)[1])
        else:
            batch_names.append(None)

    unique_batches = sorted(set(b for b in batch_names if b is not None))
    multi_batch = len(unique_batches) > 1
    batch_index = {name: i + 1 for i, name in enumerate(unique_batches)}

    labels = []
    for col, batch in zip(sample_columns, batch_names):
        if batch is not None:
            short = col.split(BATCH_SUFFIX_DELIMITER, 1)[0]
            if multi_batch:
                short = f"{short} (B{batch_index[batch]})"
            labels.append(short)
        else:
            labels.append(col)
    return labels


def _color_to_rgba(color: Union[str, tuple, np.ndarray]) -> tuple:
    """
    Convert any color format (hex string, named color, RGB tuple, etc.) to RGBA tuple.

    Parameters:
    -----------
    color : str, tuple, or np.ndarray
        Color in any matplotlib-compatible format

    Returns:
    --------
    tuple
        RGBA tuple with values in range [0, 1]
    """
    if isinstance(color, np.ndarray):
        # Already an array (e.g., from colormap)
        return tuple(color)
    elif isinstance(color, (list, tuple)):
        # Already RGB/RGBA tuple
        return tuple(color)
    else:
        # String color (hex or named) - convert to RGBA
        return mcolors.to_rgba(color)


def plot_box_plot(
    data: pd.DataFrame,
    sample_columns: List[str],
    sample_metadata: Dict[str, Dict],
    group_colors: Optional[Dict[str, str]] = None,
    group_order: Optional[List[str]] = None,
    log_transform: bool = True,
    figsize: Tuple[int, int] = (16, 8),
    title: str = "Protein Intensity Distribution by Sample",
) -> None:
    """
    Create box plot of protein intensities by sample.

    Parameters:
    -----------
    data : pd.DataFrame
        Protein quantitation data
    sample_columns : List[str]
        List of sample column names
    sample_metadata : Dict[str, Dict]
        Sample metadata mapping
    group_colors : Dict[str, str], optional
        Colors for each group
    log_transform : bool
        Whether to log2 transform data for plotting
    figsize : Tuple[int, int]
        Figure size (width, height)
    title : str
        Plot title
    """

    # Prepare data
    sample_data = data[sample_columns]

    if log_transform:
        plot_data = np.log2(sample_data.replace(0, np.nan))
        ylabel = "Log2 Intensity"
    else:
        plot_data = sample_data
        ylabel = "Intensity"

    # Group samples by experimental group
    samples_by_group = {}
    for sample in sample_columns:
        group = sample_metadata.get(sample, {}).get("Group", "Unknown")
        # Handle NaN values by converting them to 'Unknown'
        if pd.isna(group):
            group = "Unknown"
        if group not in samples_by_group:
            samples_by_group[group] = []
        samples_by_group[group].append(sample)

    # Set up default colors if not provided
    if group_colors is None:
        unique_groups = list(samples_by_group.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        group_colors = {group: colors[i] for i, group in enumerate(unique_groups)}

    # Build display labels: use Replicate field from sample_metadata when available,
    # otherwise fall back to stripping the batch suffix from the column name.
    fallback_labels = _make_display_labels(sample_columns)
    display_labels = {
        col: (sample_metadata.get(col, {}).get("Replicate") or fallback)
        for col, fallback in zip(sample_columns, fallback_labels)
    }

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    positions = []
    box_data = []
    colors = []
    labels = []
    pos = 0

    # Sort groups for consistent ordering - use provided order or fallback to alphabetical
    if group_order:
        # Use the provided group order, filtering to only groups that exist in the data
        final_group_order = [g for g in group_order if g in samples_by_group.keys()]
    else:
        # Fallback to alphabetical sorting - convert to strings to handle mixed types
        final_group_order = sorted(samples_by_group.keys(), key=str)

    # Debug information
    print(f"Debug: Group order: {final_group_order}")
    print(f"Debug: samples_by_group keys: {list(samples_by_group.keys())}")
    print(f"Debug: Total samples to plot: {sum(len(samples_by_group[g]) for g in final_group_order)}")

    for group in final_group_order:
        if group in samples_by_group:  # Additional safety check
            for sample in sorted(samples_by_group[group]):
                if sample in plot_data.columns:  # Ensure sample exists in data
                    values = plot_data[sample].dropna()
                    if len(values) > 0:  # Only add if we have data
                        box_data.append(values)
                        positions.append(pos)
                        colors.append(group_colors.get(group, "#7f7f7f"))
                        labels.append(display_labels[sample])
                        pos += 1
                    else:
                        print(f"Warning: No data for sample {sample}")
                else:
                    print(f"Warning: Sample {sample} not found in data columns")
        else:
            print(f"Warning: Group {group} not found in samples_by_group")

        pos += 0.5  # Add space between groups

    print(
        f"Debug: Final arrays lengths - box_data: {len(box_data)}, "
        f"positions: {len(positions)}, colors: {len(colors)}, labels: {len(labels)}"
    )

    if len(box_data) == 0:
        print("Error: No data to plot!")
        return

    # Ensure all arrays have the same length
    assert len(box_data) == len(positions) == len(colors) == len(labels), (
        f"Array length mismatch: box_data={len(box_data)}, positions={len(positions)}, "
        f"colors={len(colors)}, labels={len(labels)}"
    )

    # Create box plots
    bp = ax.boxplot(
        box_data,
        positions=positions,
        patch_artist=True,
        widths=0.8,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.5},
    )

    # Color the boxes
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Customize plot
    ax.set_xlabel("Sample", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add legend - use same ordering as the plot
    legend_elements = []
    for group in final_group_order:
        legend_elements.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=group_colors.get(group, "#7f7f7f"),
                alpha=0.7,
                label=group,
            )
        )
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("Box plot summary:")
    print(f"Total samples plotted: {len(box_data)}")
    print(f"Average proteins per sample: {np.mean([len(data) for data in box_data]):.0f}")


def plot_volcano(
    differential_df: pd.DataFrame,
    fc_threshold: float = 0.5,
    p_threshold: float = 0.05,
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None,
    gene_column: str = "Gene",
    label_top_n: int = 10,
    use_adjusted_pvalue: str = "adjusted",
    enable_pvalue_fallback: bool = True,
    normalization_method: Optional[str] = None,
    point_size: int = 80,
    alpha: float = 0.4,
    label_fontsize: int = 11,
    axis_label_fontsize: int = 20,
    tick_label_fontsize: int = 16,
    legend_loc: str = "upper right",
) -> None:
    """
    Create volcano plot for differential analysis results.

    Parameters:
    -----------
    differential_df : pd.DataFrame
        Differential analysis results
    fc_threshold : float
        Fold change threshold for significance (log2 for traditional data,
        VSN-space difference for VSN data)
    p_threshold : float
        P-value threshold (applied to selected p-value type)
    figsize : Tuple[int, int]
        Figure size (width, height)
    title : str, optional
        Plot title
    gene_column : str
        Column name for gene labels
    label_top_n : int
        Number of top significant proteins to label
    use_adjusted_pvalue : str
        "adjusted" to use FDR-corrected p-values, "unadjusted" for raw p-values
    enable_pvalue_fallback : bool
        If True, fallback to unadjusted p-values when no adjusted significant results
    normalization_method : str, optional
        Normalization method used ("VSN", "Median", etc.) to determine appropriate
        X-axis label. If None, defaults to "Log2 Fold Change"
    point_size : int
        Size of scatter plot points (default: 80)
    alpha : float
        Transparency of scatter points, 0-1 (default: 0.4, more transparent)
    label_fontsize : int
        Font size for gene labels on significant proteins (default: 11)
    axis_label_fontsize : int
        Font size for x and y axis labels (default: 20)
    tick_label_fontsize : int
        Font size for tick labels (default: 16)
    legend_loc : str
        Legend location (default: "upper right"). Options: "upper left", "upper right",
        "lower left", "lower right", "center left", "center right", etc.
    """

    if len(differential_df) == 0:
        print("No data to plot")
        return

    # Make a copy to avoid modifying original data
    df = differential_df.copy()

    # Auto-detect gene column if specified column doesn't exist
    gene_col_used = gene_column
    if gene_column not in df.columns:
        # Try common gene column names
        gene_column_options = ["Gene", "Gene_Names_Display", "gene", "GeneName", "gene_name", "Symbol"]
        for col in gene_column_options:
            if col in df.columns:
                gene_col_used = col
                break
        else:
            gene_col_used = None  # No gene column found

    # Determine which p-value column to use
    p_col_used = None
    fallback_used = False

    if use_adjusted_pvalue == "adjusted" and "adj.P.Val" in df.columns:
        # First try adjusted p-values
        significant_count = (df["adj.P.Val"] < p_threshold).sum()

        if significant_count > 0 or not enable_pvalue_fallback:
            p_col_used = "adj.P.Val"
            p_type_label = "FDR"
        elif enable_pvalue_fallback and "P.Value" in df.columns:
            # Fallback to unadjusted if no significant adjusted results
            p_col_used = "P.Value"
            p_type_label = "P-value"
            fallback_used = True
            print(f"Warning: No significant proteins found using adjusted p-values (FDR < {p_threshold})")
            print("Note: Automatically falling back to unadjusted p-values for visualization")
            print("    Note: Results shown use raw p-values, interpret with caution")
        else:
            p_col_used = "adj.P.Val"
            p_type_label = "FDR"
    elif use_adjusted_pvalue == "unadjusted" and "P.Value" in df.columns:
        p_col_used = "P.Value"
        p_type_label = "P-value"
    else:
        # Default fallback
        if "adj.P.Val" in df.columns:
            p_col_used = "adj.P.Val"
            p_type_label = "FDR"
        elif "P.Value" in df.columns:
            p_col_used = "P.Value"
            p_type_label = "P-value"
        else:
            print("ERROR: No p-value columns found (need 'P.Value' or 'adj.P.Val')")
            return

    # Calculate -log10(p-value) for volcano plot
    df["neg_log10_p"] = -np.log10(df[p_col_used])

    # Determine effective threshold based on normalization method
    if normalization_method and normalization_method.upper() == "VSN":
        # For VSN/arcsinh data, use smaller thresholds since arcsinh space is compressed
        effective_threshold = fc_threshold * 0.5  # Scale down the threshold for arcsinh space
    else:
        # For log2 data, use the original threshold
        effective_threshold = fc_threshold

    # Create color categories based on selected p-value
    colors = []
    for _, row in df.iterrows():
        if row[p_col_used] < p_threshold and abs(row["logFC"]) > effective_threshold:
            if row["logFC"] > effective_threshold:
                colors.append("red")  # Increased
            else:
                colors.append("blue")  # Decreased
        elif row[p_col_used] < p_threshold:
            colors.append("orange")  # Significant but small fold change
        else:
            colors.append("gray")  # Not significant

    df["color"] = colors

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot points by color category
    for color in ["gray", "orange", "blue", "red"]:
        subset = df[df["color"] == color]
        if len(subset) > 0:
            label = {
                "gray": "Not significant",
                "orange": "Significant",
                "blue": "Decreased",
                "red": "Increased",
            }[color]
            ax.scatter(
                subset["logFC"],
                subset["neg_log10_p"],
                c=color,
                alpha=alpha,
                s=point_size,
                label=label,
                edgecolors="white",
                linewidth=0.5,
            )

    # Add significance thresholds
    ax.axhline(y=-np.log10(p_threshold), color="black", linestyle="--", alpha=0.5)

    # Set normalization-appropriate fold change thresholds (visual lines)
    ax.axvline(x=effective_threshold, color="black", linestyle="--", alpha=0.5)
    ax.axvline(x=-effective_threshold, color="black", linestyle="--", alpha=0.5)

    # Label top significant proteins
    if label_top_n > 0 and gene_col_used is not None and gene_col_used in df.columns:
        significant = (
            df[(df[p_col_used] < p_threshold) & (abs(df["logFC"]) > effective_threshold)]
            .sort_values(p_col_used)
            .head(label_top_n)
        )

        for _, row in significant.iterrows():
            # Offset labels based on position to reduce overlap
            x_offset = 8 if row["logFC"] > 0 else -8
            ha = "left" if row["logFC"] > 0 else "right"
            ax.annotate(
                row[gene_col_used],
                (row["logFC"], row["neg_log10_p"]),
                xytext=(x_offset, 5),
                textcoords="offset points",
                fontsize=label_fontsize,
                fontweight="bold",
                alpha=0.9,
                ha=ha,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3, lw=0.5),
            )

    # Customize plot
    # Set normalization-appropriate X-axis label
    if normalization_method and normalization_method.upper() == "VSN":
        x_label = "Arcsinh Transform Fold Change"
        fc_description = "Arcsinh FC"
    else:
        x_label = "Log2 Fold Change"
        fc_description = "FC"

    # Print title as a statement before the plot
    if title is None:
        title_suffix = f" (using {'raw' if fallback_used else use_adjusted_pvalue} p-values)" if fallback_used else ""
        plot_title = f"Volcano Plot ({fc_description} > {fc_threshold}, {p_type_label} < {p_threshold}){title_suffix}"
    else:
        plot_title = title

    print(f"\n{plot_title}")

    # Customize plot appearance
    ax.set_xlabel(x_label, fontsize=axis_label_fontsize, fontweight="bold")
    ax.set_ylabel(f"-Log10 {p_type_label}", fontsize=axis_label_fontsize, fontweight="bold")

    # Remove top and right spines, make bottom and left thicker
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    # Increase tick label size
    ax.tick_params(axis="both", which="major", labelsize=tick_label_fontsize, width=2, length=8)

    ax.grid(True, alpha=0.3)

    # Set x-axis limits to show the full data range with some padding
    logfc_min = df["logFC"].min()
    logfc_max = df["logFC"].max()
    logfc_range = logfc_max - logfc_min
    padding = logfc_range * 0.05  # 5% padding on each side
    ax.set_xlim(logfc_min - padding, logfc_max + padding)

    # Position legend to avoid blocking data
    ax.legend(loc=legend_loc, frameon=True, fancybox=True, shadow=True, fontsize=11)

    plt.tight_layout()
    plt.show()

    # Print summary
    n_significant = len(df[df[p_col_used] < p_threshold])
    n_up = len(df[(df["logFC"] > fc_threshold) & (df[p_col_used] < p_threshold)])
    n_down = len(df[(df["logFC"] < -fc_threshold) & (df[p_col_used] < p_threshold)])

    print("Volcano plot summary:")
    print(f"Total proteins: {len(df)}")
    print(f"P-value type used: {p_type_label} ({'fallback from FDR' if fallback_used else use_adjusted_pvalue})")
    print(f"Significant ({p_type_label} < {p_threshold}): {n_significant}")
    print(f"Up-regulated ({fc_description} > {fc_threshold}, {p_type_label} < {p_threshold}): {n_up}")
    print(f"Down-regulated ({fc_description} < -{fc_threshold}, {p_type_label} < {p_threshold}): {n_down}")


def plot_normalization_comparison(
    original_data: pd.DataFrame,
    normalized_data: pd.DataFrame,
    sample_columns: List[str],
    method: str = "Normalized",
    figsize: Tuple[int, int] = (15, 6),
) -> None:
    """
    Compare data distributions before and after normalization.

    Parameters:
    -----------
    original_data : pd.DataFrame
        Original data
    normalized_data : pd.DataFrame
        Normalized data
    sample_columns : List[str]
        Sample column names
    method : str
        Normalization method name for plot title
    figsize : Tuple[int, int]
        Figure size (width, height)
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Original data
    log2_original = np.log2(original_data[sample_columns].replace(0, np.nan))

    # Create line plots instead of histograms for better visibility
    for _, col in enumerate(sample_columns):
        data = log2_original[col].dropna()
        if len(data) > 0:
            # Use more bins for smoother curves
            counts, bins = np.histogram(data, bins=100, density=True)
            # Convert to line plot using bin centers
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Optional: Apply light smoothing for even smoother curves
            from scipy.ndimage import gaussian_filter1d

            smoothed_counts = gaussian_filter1d(counts, sigma=0.8)

            ax1.plot(bin_centers, smoothed_counts, alpha=0.7, linewidth=1.5)

    ax1.set_xlabel("Log2 Intensity")
    ax1.set_ylabel("Density")
    ax1.set_title("Before Normalization")
    ax1.grid(True, alpha=0.3)

    # Normalized data
    if method.lower() == "vsn":
        # VSN data is already transformed
        plot_normalized = normalized_data[sample_columns]
        xlabel = "VSN Transformed Intensity"
    else:
        # Log transform other normalized data
        plot_normalized = np.log2(normalized_data[sample_columns].replace(0, np.nan))
        xlabel = "Log2 Intensity"

    for _, col in enumerate(sample_columns):
        data = plot_normalized[col].dropna()
        if len(data) > 0:
            # Use more bins for smoother curves
            counts, bins = np.histogram(data, bins=100, density=True)
            # Convert to line plot using bin centers
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Optional: Apply light smoothing for even smoother curves
            from scipy.ndimage import gaussian_filter1d

            smoothed_counts = gaussian_filter1d(counts, sigma=0.8)

            ax2.plot(bin_centers, smoothed_counts, alpha=0.7, linewidth=1.5)

    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Density")
    ax2.set_title(f"After {method} Normalization")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print statistics
    original_medians = log2_original.median()
    if method.lower() == "vsn":
        norm_medians = plot_normalized.median()
    else:
        norm_medians = plot_normalized.median()

    print(f"Normalization comparison ({method}):")
    print(f"Original median range: {original_medians.max() - original_medians.min():.3f}")
    print(f"Normalized median range: {norm_medians.max() - norm_medians.min():.3f}")
    print(
        "Range reduction: "
        f"{1 - (norm_medians.max() - norm_medians.min()) / (original_medians.max() - original_medians.min()):.1%}"
    )


def plot_sample_correlation_heatmap(
    data: pd.DataFrame,
    sample_columns: List[str],
    sample_metadata: Dict[str, Dict],
    figsize: Tuple[int, int] = (12, 10),
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    group_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Plot correlation heatmap between samples.

    Parameters:
    -----------
    data : pd.DataFrame
        Expression data
    sample_columns : List[str]
        Sample column names
    sample_metadata : Dict[str, Dict]
        Sample metadata
    figsize : Tuple[int, int]
        Figure size
    method : str
        Correlation method ('pearson', 'spearman')
    group_colors : Optional[Dict[str, str]]
        Dictionary mapping group names to colors. If None, colors are auto-generated.
    """

    # Calculate correlation matrix
    sample_data = data[sample_columns]
    correlation_matrix = sample_data.corr(method=method)

    # Create group color annotation
    groups = []
    for sample in sample_columns:
        group = sample_metadata.get(sample, {}).get("Group", "Unknown")
        # Handle NaN values by converting them to 'Unknown'
        if pd.isna(group):
            group = "Unknown"
        groups.append(group)
    unique_groups = list(set(groups))

    # Use provided group colors or create color palette for groups
    if group_colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        group_color_map = {group: colors[i] for i, group in enumerate(unique_groups)}
    else:
        group_color_map = group_colors.copy()
        # Add any missing groups with default colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        for i, group in enumerate(unique_groups):
            if group not in group_color_map:
                group_color_map[group] = colors[i % len(colors)]

    # Create row/col colors for annotation
    row_colors = []
    for sample in sample_columns:
        group = sample_metadata.get(sample, {}).get("Group", "Unknown")
        if pd.isna(group):
            group = "Unknown"
        row_colors.append(group_color_map[group])

    # Create clustermap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Calculate actual correlation range for better color scaling
        correlation_values = correlation_matrix.values
        # Get upper triangle (exclude diagonal which is always 1)
        upper_triangle = correlation_values[np.triu_indices_from(correlation_values, k=1)]
        actual_min = np.min(upper_triangle)
        actual_max = np.max(upper_triangle)

        # Use a color scale that emphasizes the actual correlation range
        # For Pearson correlation, we want red (bad/low) to blue (good/high)
        g = sns.clustermap(
            correlation_matrix,
            figsize=figsize,
            cmap="RdYlBu",  # Red-Yellow-Blue: red=low, yellow=medium, blue=high
            center=(actual_min + actual_max) / 2,  # Center on middle of actual range
            vmin=actual_min,
            vmax=1.0,  # Scale from actual minimum to perfect correlation
            square=True,
            linewidths=0.1,
            row_colors=row_colors,
            col_colors=row_colors,
            cbar_kws={"label": f"{method.title()} Correlation\n(Red=Low, Blue=High)"},
        )

        # Customize
        g.ax_heatmap.set_xlabel("Samples")
        g.ax_heatmap.set_ylabel("Samples")
        g.fig.suptitle(f"Sample Correlation Heatmap ({method.title()})", fontsize=16, y=1.02)

    plt.show()

    # Print correlation statistics
    correlation_values = correlation_matrix.values
    # Get upper triangle (exclude diagonal)
    upper_triangle = correlation_values[np.triu_indices_from(correlation_values, k=1)]

    print(f"Correlation summary ({method}):")
    print(f"Mean correlation: {np.mean(upper_triangle):.3f}")
    print(f"Min correlation: {np.min(upper_triangle):.3f}")
    print(f"Max correlation: {np.max(upper_triangle):.3f}")


def plot_sample_correlation_triangular_heatmap(
    data: pd.DataFrame,
    sample_columns: List[str],
    sample_metadata: Dict[str, Dict],
    figsize: Tuple[int, int] = (16, 14),
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    group_colors: Optional[Dict[str, str]] = None,
    show_clustering: bool = True,
    label_fontsize: int = 14,
    annot_fontsize: int = 10,
    group_column: str = "Study",
    max_samples_for_annotations: int = 30,
    cmap: str = "coolwarm",
) -> None:
    """
    Plot triangular correlation heatmap between samples with color bars for group identification.

    Parameters:
    -----------
    data : pd.DataFrame
        Expression data
    sample_columns : List[str]
        Sample column names
    sample_metadata : Dict[str, Dict]
        Sample metadata
    figsize : Tuple[int, int]
        Figure size
    method : str
        Correlation method ('pearson', 'spearman', 'kendall')
    group_colors : Optional[Dict[str, str]]
        Dictionary mapping group names to colors. If None, colors are auto-generated.
    show_clustering : bool
        Whether to perform hierarchical clustering
    label_fontsize : int
        Font size for sample labels
    annot_fontsize : int
        Font size for correlation value annotations
    group_column : str
        Column name in metadata to use for grouping (default: "Study")
    max_samples_for_annotations : int
        Maximum number of samples to show correlation values (default: 30)
    cmap : str
        Colormap for correlation values (default: "coolwarm")
    """
    from matplotlib.patches import Patch

    # Calculate correlation matrix
    sample_data = data[sample_columns]
    correlation_matrix = sample_data.corr(method=method)

    # Helper function to check for NA-like values (None, NaN, "na", "nan", "")
    def _is_na_value(val):
        if val is None:
            return True
        if isinstance(val, float) and pd.isna(val):
            return True
        if isinstance(val, str) and val.lower() in ("na", "nan", ""):
            return True
        return False

    # Create group color annotations for sample labels
    groups = []
    for sample in sample_columns:
        meta = sample_metadata.get(sample, {})
        # Try the specified group_column first, then fall back to alternatives
        group = meta.get(group_column)

        if _is_na_value(group):
            group = meta.get("Sample Category")
        if _is_na_value(group):
            group = meta.get("Group")
        if _is_na_value(group):
            group = "Unknown"
        groups.append(str(group))

    unique_groups = sorted(set(groups))

    # Use provided group colors or create color palette for groups
    if group_colors is None:
        # Use a distinct color palette
        if len(unique_groups) <= 10:
            color_palette = plt.cm.tab10(np.linspace(0, 1, 10))
        else:
            color_palette = plt.cm.tab20(np.linspace(0, 1, 20))
        group_color_map = {group: color_palette[i % len(color_palette)] for i, group in enumerate(unique_groups)}
    else:
        group_color_map = group_colors.copy()
        # Add any missing groups with default colors
        color_palette = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, group in enumerate(unique_groups):
            if group not in group_color_map:
                group_color_map[group] = color_palette[i % len(color_palette)]

    # Create color mapping for individual samples (in original order)
    sample_colors_orig = []
    for sample in sample_columns:
        meta = sample_metadata.get(sample, {})
        group = meta.get(group_column)
        # Reuse the same NA check logic
        if _is_na_value(group):
            group = meta.get("Sample Category")
        if _is_na_value(group):
            group = meta.get("Group")
        if _is_na_value(group):
            group = "Unknown"
        sample_colors_orig.append(group_color_map[str(group)])

    if show_clustering:
        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform

        # Create distance matrix from correlation (1 - correlation for hierarchical clustering)
        distance_matrix = 1 - correlation_matrix.fillna(0)

        # Perform hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix), method="average")

        # Get the order from clustering
        dendro = dendrogram(linkage_matrix, no_plot=True)
        cluster_order = dendro["leaves"]

        # Reorder correlation matrix and labels
        ordered_samples = [sample_columns[i] for i in cluster_order]
        correlation_matrix = correlation_matrix.loc[ordered_samples, ordered_samples]
        sample_colors = [sample_colors_orig[i] for i in cluster_order]
    else:
        ordered_samples = sample_columns
        sample_colors = sample_colors_orig

    n_samples = len(ordered_samples)
    show_annotations = n_samples <= max_samples_for_annotations

    # Create figure - we'll use mpl_toolkits for proper alignment
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax_heatmap = plt.subplots(figsize=figsize)

    # Create a mask for the upper triangle to show only lower triangle (excluding diagonal)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

    # Create the triangular heatmap with improved colorbar settings
    heatmap = sns.heatmap(
        correlation_matrix,
        mask=mask,  # Show only lower triangle + diagonal
        annot=show_annotations,  # Only show correlation values if not too many samples
        fmt=".2f" if show_annotations else "",  # Format correlation values
        cmap=cmap,  # Better colormap for correlation
        vmin=0.5,  # Set reasonable min for correlation (adjust based on your data)
        vmax=1.0,  # Max correlation
        square=True,  # Square aspect ratio
        linewidths=0.5 if n_samples <= 50 else 0.1,  # Thinner lines for many samples
        cbar_kws={
            "label": f"{method.title()} Correlation",
            "shrink": 0.5,  # Reduced height to better match triangular heatmap
            "aspect": 20,  # Narrower colorbar
        },
        ax=ax_heatmap,
        annot_kws={"size": annot_fontsize} if show_annotations else {},
    )

    # Increase colorbar label and tick font sizes
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_ylabel(f"{method.title()} Correlation", fontsize=label_fontsize + 2, fontweight="bold")
    cbar.ax.tick_params(labelsize=label_fontsize)

    # Set tick labels
    if n_samples <= 50:
        ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=label_fontsize)
        ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0, fontsize=label_fontsize)
    else:
        # Too many samples - hide tick labels
        ax_heatmap.set_xticklabels([])
        ax_heatmap.set_yticklabels([])

    ax_heatmap.set_xlabel("")
    ax_heatmap.set_ylabel("")

    clustering_text = " (Clustered)" if show_clustering else ""
    ax_heatmap.set_title(
        f"Sample Correlation Heatmap{clustering_text}\n{method.title()} Correlation",
        fontsize=label_fontsize + 6,
        fontweight="bold",
    )

    # Convert colors to RGBA format for imshow (handles hex strings, named colors, etc.)
    sample_colors_rgba = [_color_to_rgba(c) for c in sample_colors]

    # Use make_axes_locatable for properly aligned color bars
    divider = make_axes_locatable(ax_heatmap)

    # Create left color bar - append to left side, shares y-axis with heatmap
    ax_left_colorbar = divider.append_axes("left", size="1.5%", pad=0.05)
    left_colors = np.array(sample_colors_rgba).reshape(-1, 1, 4)
    ax_left_colorbar.imshow(left_colors, aspect="auto", origin="upper", extent=[0, 1, n_samples, 0])
    ax_left_colorbar.set_xlim(0, 1)
    ax_left_colorbar.set_ylim(n_samples, 0)  # Match heatmap orientation (0 at top)
    ax_left_colorbar.set_xticks([])
    ax_left_colorbar.set_yticks([])
    ax_left_colorbar.tick_params(left=False, bottom=False)  # Remove tick marks
    for spine in ax_left_colorbar.spines.values():
        spine.set_visible(False)  # Remove border
    ax_left_colorbar.set_ylabel("Group", fontsize=label_fontsize + 4, fontweight="bold")

    # Create bottom color bar - append to bottom, shares x-axis with heatmap
    ax_bottom_colorbar = divider.append_axes("bottom", size="1.5%", pad=0.05)
    bottom_colors = np.array(sample_colors_rgba).reshape(1, -1, 4)
    ax_bottom_colorbar.imshow(bottom_colors, aspect="auto", origin="upper", extent=[0, n_samples, 1, 0])
    ax_bottom_colorbar.set_xlim(0, n_samples)
    ax_bottom_colorbar.set_ylim(1, 0)
    ax_bottom_colorbar.set_xticks([])
    ax_bottom_colorbar.set_yticks([])
    ax_bottom_colorbar.tick_params(left=False, bottom=False)  # Remove tick marks
    for spine in ax_bottom_colorbar.spines.values():
        spine.set_visible(False)  # Remove border
    ax_bottom_colorbar.set_xlabel("Group", fontsize=label_fontsize + 4, fontweight="bold")

    # Add legend for groups - only include groups that actually appear in the data
    # Filter out "na"-like values that were replaced by Sample Category fallback
    legend_groups = [g for g in unique_groups if not _is_na_value(g)]
    legend_handles = [
        Patch(facecolor=_color_to_rgba(group_color_map[g]), edgecolor="black", label=g)
        for g in legend_groups
        if groups.count(g) > 0
    ]

    # Place legend close to the plot (upper right, just outside the heatmap)
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=label_fontsize,
        title="Groups",
        title_fontsize=label_fontsize + 2,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout()
    plt.show()

    # Print correlation statistics
    correlation_values = correlation_matrix.values
    lower_triangle = correlation_values[np.tril_indices_from(correlation_values, k=-1)]

    print(f"\nCorrelation summary ({method}):")
    print(f"  Mean correlation: {np.nanmean(lower_triangle):.3f}")
    print(f"  Min correlation: {np.nanmin(lower_triangle):.3f}")
    print(f"  Max correlation: {np.nanmax(lower_triangle):.3f}")
    print(f"  Correlation range: {np.nanmax(lower_triangle) - np.nanmin(lower_triangle):.3f}")

    if not show_annotations:
        print(f"\n  Note: Correlation values hidden ({n_samples} samples > {max_samples_for_annotations} threshold)")

    # Group-wise correlation summary
    print(f"\nGroup composition (by {group_column}):")
    for group in legend_groups:  # Use filtered legend_groups instead of unique_groups
        count = groups.count(group)
        if count > 0:
            print(f"  {group}: {count} samples")


def plot_pca(
    data: pd.DataFrame,
    sample_columns: List[str],
    sample_metadata: Dict[str, Dict],
    group_colors: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Principal Component Analysis",
    log_transform: bool = False,
) -> None:
    """
    Plot PCA of samples.

    Parameters:
    -----------
    data : pd.DataFrame
        Expression data
    sample_columns : List[str]
        Sample column names
    sample_metadata : Dict[str, Dict]
        Sample metadata
    group_colors : Dict[str, str], optional
        Colors for groups
    figsize : Tuple[int, int]
        Figure size
    title : str
        Plot title
    log_transform : bool
        If True, apply log2 transform before PCA
    """

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("scikit-learn not available for PCA plot")
        return

    # Prepare data
    sample_data = data[sample_columns]

    if log_transform:
        sample_data = np.log2(sample_data.clip(lower=1))

    # Remove proteins with missing values
    complete_data = sample_data.dropna()

    if len(complete_data) == 0:
        print("No complete data available for PCA")
        return

    # Transpose for PCA (samples as rows)
    pca_data = complete_data.T

    # Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pca_data)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Group samples
    groups = []
    for sample in sample_columns:
        group = sample_metadata.get(sample, {}).get("Group", "Unknown")
        # Handle NaN values by converting them to 'Unknown'
        if pd.isna(group):
            group = "Unknown"
        groups.append(group)
    unique_groups = list(set(groups))

    # Set colors
    if group_colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
        group_colors = {group: colors[i] for i, group in enumerate(unique_groups)}

    # Plot each group
    for group in unique_groups:
        group_indices = [i for i, g in enumerate(groups) if g == group]
        ax.scatter(
            pca_result[group_indices, 0],
            pca_result[group_indices, 1],
            c=group_colors[group],
            label=group,
            alpha=0.7,
            s=100,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("PCA summary:")
    print(f"PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
    print(f"PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")
    print(f"Total variance explained: {pca.explained_variance_ratio_[:2].sum():.1%}")


def plot_comparative_pca(
    original_data: pd.DataFrame,
    median_normalized_data: pd.DataFrame,
    vsn_normalized_data: pd.DataFrame,
    sample_columns: List[str],
    sample_metadata: Dict[str, Dict],
    group_colors: Optional[Dict[str, str]] = None,
    group_order: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (24, 8),
) -> None:
    """
    Plot comparative PCA analysis across normalization methods.

    Parameters:
    -----------
    original_data : pd.DataFrame
        Original expression data
    median_normalized_data : pd.DataFrame
        Median normalized data
    vsn_normalized_data : pd.DataFrame
        VSN normalized data
    sample_columns : List[str]
        Sample column names
    sample_metadata : Dict[str, Dict]
        Sample metadata
    group_colors : Dict[str, str], optional
        Colors for groups
    figsize : Tuple[int, int]
        Figure size
    """

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("scikit-learn not available for PCA plot")
        return

    def perform_pca_analysis(data, title_suffix, log_transform=True):
        """Perform PCA on a dataset and return results"""

        if log_transform:
            # Apply log2 transform
            pca_input = np.log2(data[sample_columns].replace(0, np.nan))
        else:
            # Use data as-is (for VSN which is already transformed)
            pca_input = data[sample_columns]

        # Remove proteins with missing values across samples
        complete_data = pca_input.dropna()

        if len(complete_data) == 0:
            print(f"No complete data available for {title_suffix}")
            return None, None

        print(f"Using {len(complete_data)} proteins for {title_suffix} PCA")

        # Transpose for PCA (samples as rows, proteins as columns)
        pca_data = complete_data.T

        # Standardize features (proteins)
        scaler = StandardScaler()
        pca_input_scaled = scaler.fit_transform(pca_data)

        # Perform PCA
        n_components = min(10, len(sample_columns) - 1, pca_input_scaled.shape[1])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(pca_input_scaled)

        # Create DataFrame with results
        pca_df = pd.DataFrame(pca_result[:, :2], columns=["PC1", "PC2"], index=complete_data.columns)

        # Add group information
        pca_df["Group"] = [sample_metadata.get(sample, {}).get("Group", "Unknown") for sample in pca_df.index]
        pca_df["Sample"] = pca_df.index

        return pca_df, pca

    def plot_single_pca(ax, pca_df, pca, title_suffix):
        """Create a single PCA plot on the given axis"""
        if pca_df is None or pca is None:
            ax.text(
                0.5,
                0.5,
                f"No data\navailable\nfor {title_suffix}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title(title_suffix, fontsize=16, fontweight="bold")
            return

        # Get unique groups and set colors - use group_order if provided for consistent legend ordering
        if group_order is not None:
            # Use provided group order, but only include groups that exist in the data
            unique_groups = [group for group in group_order if group in pca_df["Group"].values]
            # Add any remaining groups not in group_order
            remaining_groups = [group for group in pca_df["Group"].unique() if group not in unique_groups]
            unique_groups.extend(remaining_groups)
        else:
            unique_groups = pca_df["Group"].unique()

        if group_colors is None:
            import matplotlib.pyplot as plt

            tab10_cmap = plt.cm.get_cmap("tab10")
            colors = [tab10_cmap(i / max(1, len(unique_groups) - 1)) for i in range(len(unique_groups))]
            current_group_colors = {group: colors[i] for i, group in enumerate(unique_groups)}
        else:
            current_group_colors = group_colors.copy()

        # Plot each group
        for group in unique_groups:
            group_data = pca_df[pca_df["Group"] == group]
            if len(group_data) > 0:
                ax.scatter(
                    group_data["PC1"],
                    group_data["PC2"],
                    c=current_group_colors.get(group, "#7f7f7f"),
                    label=group,
                    alpha=0.7,
                    s=100,
                    edgecolors="black",
                    linewidth=0.5,
                )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=14)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=14)
        ax.set_title(title_suffix, fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)

    # Perform PCA on all datasets
    print("Performing comparative PCA analysis...")
    original_pca_df, original_pca = perform_pca_analysis(original_data, "Original Data", log_transform=True)
    median_pca_df, median_pca = perform_pca_analysis(median_normalized_data, "Median Normalized", log_transform=True)
    vsn_pca_df, vsn_pca = perform_pca_analysis(vsn_normalized_data, "VSN Normalized", log_transform=False)

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    plot_single_pca(axes[0], original_pca_df, original_pca, "Original Data (Log2)")
    plot_single_pca(axes[1], median_pca_df, median_pca, "Median Normalized (Log2)")
    plot_single_pca(axes[2], vsn_pca_df, vsn_pca, "VSN Normalized")

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n=== PCA SUMMARY STATISTICS ===")

    def print_pca_summary(pca, pca_df, dataset_name):
        """Print summary statistics for PCA results"""
        if pca is None or pca_df is None:
            print(f"**{dataset_name}:** No data available")
            return

        print(f"**{dataset_name}:**")
        print(f"  PC1 variance: {pca.explained_variance_ratio_[0]:.1%}")
        print(f"  PC2 variance: {pca.explained_variance_ratio_[1]:.1%}")
        print(f"  Total variance (PC1-PC2): {pca.explained_variance_ratio_[:2].sum():.1%}")

        # Calculate group separation (average distance between group centroids)
        unique_groups = pca_df["Group"].unique()
        if len(unique_groups) > 1:
            group_centroids = {}
            for group in unique_groups:
                group_data = pca_df[pca_df["Group"] == group]
                if len(group_data) > 0:
                    group_centroids[group] = (
                        group_data["PC1"].mean(),
                        group_data["PC2"].mean(),
                    )

            if len(group_centroids) > 1:
                distances = []
                groups = list(group_centroids.keys())
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        g1, g2 = groups[i], groups[j]
                        dist = np.sqrt(
                            (group_centroids[g1][0] - group_centroids[g2][0]) ** 2
                            + (group_centroids[g1][1] - group_centroids[g2][1]) ** 2
                        )
                        distances.append(dist)

                if distances:
                    print(f"  Inter-group distance: {np.mean(distances):.2f} ± {np.std(distances):.2f}")
        print()

    print_pca_summary(original_pca, original_pca_df, "Original Data")
    print_pca_summary(median_pca, median_pca_df, "Median Normalized")
    print_pca_summary(vsn_pca, vsn_pca_df, "VSN Normalized")


def plot_control_correlation(
    data: pd.DataFrame,
    control_columns: List[str],
    sample_metadata: Optional[Dict[str, Dict]] = None,
    title: str = "Control Sample Correlation",
    log_transform: bool = False,
    figsize: Tuple[int, int] = (8, 7),
    cluster: bool = False,
    group_colors: Optional[Dict[str, str]] = None,
    group_column: str = "Group",
) -> None:
    """
    Plot a Pearson correlation heatmap for a set of control/QC samples.

    Parameters:
    -----------
    data : pd.DataFrame
        Protein expression data
    control_columns : List[str]
        Column names of the control samples to correlate
    sample_metadata : Dict[str, Dict], optional
        Metadata dict (used for short display names and group labels)
    title : str
        Plot title
    log_transform : bool
        If True, apply log2 transform before computing correlations
    figsize : Tuple[int, int]
        Figure size
    cluster : bool
        If True, apply hierarchical clustering to rows and columns
    group_colors : Dict[str, str], optional
        Map of group name → color for row/col color annotations
    group_column : str
        Key in sample_metadata dicts to use for group labels
    """
    import seaborn as sns

    control_data = data[control_columns].copy()

    if log_transform:
        control_data = np.log2(control_data.clip(lower=1))

    control_data = control_data.dropna()
    if len(control_data) == 0:
        print("No valid data for correlation analysis")
        return

    # Use short names for display if metadata available
    display_names = {}
    for col in control_columns:
        if sample_metadata and col in sample_metadata:
            display_names[col] = sample_metadata[col].get("Replicate", col)
        else:
            display_names[col] = col
    control_data = control_data.rename(columns=display_names)

    corr_matrix = control_data.corr(method="pearson")

    if cluster:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        # Compute clustering order
        dist = 1 - corr_matrix.fillna(0).values
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist)
        link = linkage(condensed, method="average")
        order = leaves_list(link)
        ordered_labels = [corr_matrix.index[i] for i in order]
        corr_matrix = corr_matrix.loc[ordered_labels, ordered_labels]

        # Build group color bar if metadata and group_colors provided
        group_bar = None
        unique_groups_ordered = None
        if sample_metadata and group_colors:
            # Map ordered display names back to original columns
            rev_display = {v: k for k, v in display_names.items()}
            group_labels = []
            for label in ordered_labels:
                orig_col = rev_display.get(label, label)
                grp = sample_metadata.get(orig_col, {}).get(group_column, "Unknown")
                group_labels.append(grp)
            group_bar = [group_colors.get(g, "#888888") for g in group_labels]
            unique_groups_ordered = dict.fromkeys(group_labels)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            square=True,
            linewidths=0.5,
            vmin=corr_matrix.min().min(),
            vmax=1,
            cbar_kws={"label": "Pearson Correlation"},
            ax=ax,
            xticklabels=True,
            yticklabels=True,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)

        # Color-code tick labels by group instead of drawing rectangles
        if group_bar is not None:
            for i, color in enumerate(group_bar):
                ax.get_xticklabels()[i].set_color(color)
                ax.get_xticklabels()[i].set_fontweight("bold")
                ax.get_yticklabels()[i].set_color(color)
                ax.get_yticklabels()[i].set_fontweight("bold")

            # Legend below the plot
            from matplotlib.patches import Patch

            handles = [Patch(facecolor=group_colors.get(g, "#888888"), label=g) for g in unique_groups_ordered]
            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                frameon=True,
                title="Group",
                ncol=len(unique_groups_ordered),
            )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            square=True,
            linewidths=0.5,
            vmin=max(-1, corr_matrix.min().min()),
            vmax=1,
            cbar_kws={"label": "Pearson Correlation"},
            ax=ax,
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=10)

    # Print summary
    off_diag = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    off_diag = off_diag[~np.isnan(off_diag)]
    if len(off_diag) > 0:
        print(f"Mean r = {np.mean(off_diag):.3f}, Min r = {np.min(off_diag):.3f}, Max r = {np.max(off_diag):.3f}")


def plot_control_correlation_analysis(
    original_data: pd.DataFrame,
    median_normalized_data: pd.DataFrame,
    vsn_normalized_data: pd.DataFrame,
    sample_columns: List[str],
    sample_metadata: Dict[str, Dict],
    control_column: str = "Subject",
    control_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (24, 8),
) -> None:
    """
    Plot control sample correlation analysis across normalization methods.

    Parameters:
    -----------
    original_data : pd.DataFrame
        Original expression data
    median_normalized_data : pd.DataFrame
        Median normalized data
    vsn_normalized_data : pd.DataFrame
        VSN normalized data
    sample_columns : List[str]
        Sample column names
    sample_metadata : Dict[str, Dict]
        Sample metadata
    control_column : str
        Column name to look for control samples
    control_labels : List[str], optional
        Labels in the control column that identify control samples.
        If None, function will look for samples containing common control patterns
        like 'Pool', 'Control', 'QC', 'Standard', etc.
    figsize : Tuple[int, int]
        Figure size
    """

    try:
        import seaborn as sns  # noqa: F401  # Used for optional enhanced plotting
    except ImportError:
        print("Required packages (seaborn, scipy) not available for correlation analysis")
        return

    # Handle default control labels - look for common control patterns
    if control_labels is None:
        # Find unique values in the control column
        unique_control_values = set()
        for sample in sample_columns:
            control_value = sample_metadata.get(sample, {}).get(control_column, "")
            if control_value and not pd.isna(control_value):
                unique_control_values.add(str(control_value))

        # Look for common control patterns
        control_patterns = [
            "Pool",
            "Control",
            "QC",
            "Standard",
            "Blank",
            "pool",
            "control",
            "qc",
            "standard",
            "blank",
        ]
        control_labels = []

        for value in unique_control_values:
            if any(pattern in value for pattern in control_patterns):
                control_labels.append(value)

        if not control_labels:
            print(f"No control samples found automatically in column '{control_column}'.")
            print(f"Available values: {sorted(unique_control_values)}")
            print("Please specify control_labels parameter explicitly.")
            return

    print("=== CONTROL CORRELATION ANALYSIS ===\n")
    print(f"Looking for control samples in column '{control_column}' with labels: {control_labels}")

    # Find control samples using simplified approach
    all_control_samples = []

    for sample in sample_columns:
        # Get the value from the specified control column
        control_value = sample_metadata.get(sample, {}).get(control_column, "")

        # Check if this sample matches any of our control labels
        if control_value in control_labels:
            all_control_samples.append(sample)

    print(f"\nFound control samples: {all_control_samples}")
    print(f"Total control samples: {len(all_control_samples)}")

    if len(all_control_samples) < 2:
        print("\n⚠ Warning: Less than 2 control samples found!")
        print(f"Check that your control_column ('{control_column}') and control_labels ({control_labels}) are correct.")
        return

    def create_control_heatmap(data, title, ax, log_transform=False):
        """Create correlation heatmap for control samples"""

        # Extract control data first
        control_data = data[all_control_samples]

        if log_transform:
            # Apply log2 transformation to numeric control data only
            plot_data = np.log2(control_data.replace(0, np.nan))
            title_suffix = " (Log2)"
        else:
            plot_data = control_data
            title_suffix = ""

        # Remove missing values
        plot_data = plot_data.dropna()

        if len(plot_data) == 0:
            ax.text(
                0.5,
                0.5,
                "No valid data\nfor correlation",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_title(title + title_suffix, fontsize=14, fontweight="bold")
            return None

        print(f"Using {len(plot_data)} proteins for {title} correlation")

        # Calculate correlation matrix
        corr_matrix = plot_data.corr(method="pearson")

        # Create heatmap with clustering
        mask = corr_matrix.isna()

        # Use seaborn's clustermap for hierarchical clustering
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        try:
            # Create distance matrix from correlation (1 - correlation for hierarchical clustering)
            distance_matrix = 1 - corr_matrix.fillna(0)

            # Perform hierarchical clustering
            linkage_matrix = linkage(squareform(distance_matrix), method="average")

            # Create clustermap
            g = sns.clustermap(
                corr_matrix,
                row_linkage=linkage_matrix,
                col_linkage=linkage_matrix,
                figsize=(8, 8),
                cmap="RdYlBu_r",
                center=0 if corr_matrix.min().min() < 0 else None,
                square=True,
                linewidths=0.5,
                cbar_kws={"label": "Pearson Correlation"},
                mask=mask,
                vmin=max(-1, corr_matrix.min().min()),
                vmax=1,
                annot=True,
                fmt=".3f",
                annot_kws={"size": 8},
            )

            # Update the subplot in the main figure
            # Clear the original axis
            ax.clear()

            # Copy the clustered heatmap to the original axis
            # Get the reordered correlation matrix from clustermap
            reordered_corr = g.data2d

            # Close the clustermap figure (we'll recreate it in our axis)
            plt.close(g.fig)  # type: ignore

            # Now create the heatmap in our designated axis with the clustered order
            sns.heatmap(
                reordered_corr,
                annot=True,
                fmt=".3f",
                cmap="RdYlBu_r",
                center=0 if reordered_corr.min().min() < 0 else None,
                square=True,
                linewidths=0.5,
                cbar_kws={"label": "Pearson Correlation"},
                ax=ax,
                vmin=max(-1, reordered_corr.min().min()),
                vmax=1,
            )

        except Exception as e:  # noqa: BLE001
            print(f"Clustering failed ({e}), using original order")
            # Fallback to non-clustered heatmap
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt=".3f",
                cmap="RdYlBu_r",
                center=0 if corr_matrix.min().min() < 0 else None,
                square=True,
                linewidths=0.5,
                cbar_kws={"label": "Pearson Correlation"},
                ax=ax,
                mask=mask,
                vmin=max(-1, corr_matrix.min().min()),
                vmax=1,
            )

        ax.set_title(title + title_suffix, fontsize=14, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", rotation=0, labelsize=10)

        return corr_matrix

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    print("\nCreating correlation heatmaps...")
    corr_original = create_control_heatmap(
        original_data,
        "Control Samples Correlation\n(Original Data)",
        axes[0],
        log_transform=True,
    )
    corr_median = create_control_heatmap(
        median_normalized_data,
        "Control Samples Correlation\n(Median Normalized)",
        axes[1],
        log_transform=True,
    )
    corr_vsn = create_control_heatmap(
        vsn_normalized_data,
        "Control Samples Correlation\n(VSN Normalized)",
        axes[2],
        log_transform=False,
    )

    plt.tight_layout()
    plt.show()

    # Print correlation summary
    print("\n=== CONTROL CORRELATION SUMMARY ===")

    def print_correlation_summary(corr_matrix, dataset_name):
        if corr_matrix is not None:
            # Get off-diagonal correlations
            off_diagonal = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            off_diagonal = off_diagonal[~np.isnan(off_diagonal)]

            if len(off_diagonal) > 0:
                print(f"\n**{dataset_name}:**")
                print(f"  Mean correlation: {np.mean(off_diagonal):.3f}")
                print(f"  Median correlation: {np.median(off_diagonal):.3f}")
                print(f"  Min correlation: {np.min(off_diagonal):.3f}")
                print(f"  Max correlation: {np.max(off_diagonal):.3f}")

                # Count high correlations
                high_corr = np.sum(off_diagonal > 0.8)
                total = len(off_diagonal)
                print(f"  High correlation (>0.8): {high_corr}/{total} ({100 * high_corr / total:.1f}%)")

    print_correlation_summary(corr_original, "Original Data")
    print_correlation_summary(corr_median, "Median Normalized")
    print_correlation_summary(corr_vsn, "VSN Normalized")


def plot_control_group_correlation_analysis(
    data: pd.DataFrame,
    sample_columns: List[str],
    _sample_metadata: Dict[str, Dict],
    control_patterns: Dict[str, List[str]],
    normalization_method: str = "VSN",
    figsize: Tuple[int, int] = (15, 12),
) -> None:
    """
    Create detailed correlation analysis plots for each control group separately.
    Shows correlation matrices with scatter plots and histograms.

    Parameters:
    -----------
    data : pd.DataFrame
        Normalized proteomics data
    sample_columns : List[str]
        Sample column names
    sample_metadata : Dict[str, Dict]
        Sample metadata
    control_patterns : Dict[str, List[str]]
        Control sample patterns by type
    normalization_method : str
        Name of normalization method for title
    figsize : Tuple[int, int]
        Figure size (width, height)
    """

    print(f"\n=== INDIVIDUAL CONTROL GROUP CORRELATION ANALYSIS ({normalization_method}) ===")

    # Find control samples by type
    control_groups = {}
    for control_type, patterns in control_patterns.items():
        found_samples = []
        for sample in sample_columns:
            if any(pattern in sample for pattern in patterns):
                found_samples.append(sample)

        if found_samples:
            control_groups[control_type] = list(set(found_samples))

    if not control_groups:
        print("No control samples found matching the patterns!")
        return

    # Process each control group that has multiple samples
    valid_groups = {k: v for k, v in control_groups.items() if len(v) >= 2}

    if not valid_groups:
        print("No control groups with 2+ samples found!")
        return

    print(f"\nAnalyzing {len(valid_groups)} control groups:")
    for group_name, samples in valid_groups.items():
        print(f"  {group_name}: {len(samples)} samples - {samples}")

    # Create subplot grid

    for _, (group_name, group_samples) in enumerate(valid_groups.items()):
        if len(group_samples) < 2:
            continue

        print(f"\n--- {group_name} Correlation Analysis ---")

        # Extract data for this control group
        if normalization_method.upper() == "VSN":
            # VSN data doesn't need log transformation
            control_data = data[group_samples].dropna()
        else:
            # Apply log2 transformation for other methods
            control_data = np.log2(data[group_samples].replace(0, np.nan)).dropna()

        if len(control_data) == 0:
            print(f"No valid data for {group_name}")
            continue

        print(f"Using {len(control_data)} proteins for {group_name}")

        # Calculate correlation matrix
        corr_matrix = control_data.corr(method="pearson")
        n_samples = len(group_samples)

        # Create correlation matrix plot with pairwise scatter plots
        fig, axes = plt.subplots(n_samples, n_samples, figsize=figsize)
        fig.suptitle(
            f"Correlation Matrix - {group_name} ({normalization_method} Normalized)",
            fontsize=16,
            fontweight="bold",
        )

        # Handle single sample case (shouldn't happen but just in case)
        if n_samples == 1:
            axes = np.array([[axes]])
        elif n_samples == 2:
            axes = axes.reshape(2, 2)

        for row in range(n_samples):
            for col in range(n_samples):
                ax = axes[row, col]

                sample_x = group_samples[col]
                sample_y = group_samples[row]

                if row == col:
                    # Diagonal: histogram
                    values = control_data[sample_x].dropna()
                    ax.hist(values, bins=30, alpha=0.7, color="skyblue", density=True)
                    ax.set_title(sample_x, fontsize=10, fontweight="bold")
                    ax.set_ylabel("Density")
                elif row > col:
                    # Lower triangle: scatter plot
                    x_vals = control_data[sample_x]
                    y_vals = control_data[sample_y]

                    # Remove NaN pairs
                    valid_mask = ~(x_vals.isna() | y_vals.isna())
                    x_clean = x_vals[valid_mask]
                    y_clean = y_vals[valid_mask]

                    if len(x_clean) > 0:
                        ax.scatter(x_clean, y_clean, alpha=0.6, s=1, color="blue")

                        # Add trend line
                        if len(x_clean) > 1:
                            z = np.polyfit(x_clean, y_clean, 1)
                            p = np.poly1d(z)
                            ax.plot(x_clean, p(x_clean), "r-", alpha=0.8)

                        # Calculate correlation
                        correlation = corr_matrix.loc[sample_y, sample_x]

                        # Add correlation text
                        ax.text(
                            0.05,
                            0.95,
                            f"r = {correlation:.3f}",
                            transform=ax.transAxes,
                            fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                            verticalalignment="top",
                        )

                    ax.set_xlabel(sample_x, fontsize=9)
                    ax.set_ylabel(sample_y, fontsize=9)
                else:
                    # Upper triangle: correlation value
                    correlation = corr_matrix.loc[sample_y, sample_x]
                    ax.text(
                        0.5,
                        0.5,
                        f"{correlation:.3f}***" if correlation > 0.95 else f"{correlation:.3f}",
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                    )
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Adjust tick labels
                ax.tick_params(axis="both", which="major", labelsize=8)

        plt.tight_layout()
        plt.show()

        # Print correlation summary for this group
        off_diagonal = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        off_diagonal = off_diagonal[~np.isnan(off_diagonal)]

        if len(off_diagonal) > 0:
            print(f"\n{group_name} Correlation Summary:")
            print(f"  Mean correlation: {np.mean(off_diagonal):.3f}")
            print(f"  Median correlation: {np.median(off_diagonal):.3f}")
            print(f"  Min correlation: {np.min(off_diagonal):.3f}")
            print(f"  Max correlation: {np.max(off_diagonal):.3f}")
            high_corr = np.sum(off_diagonal > 0.9)
            total = len(off_diagonal)
            print(f"  High correlation (>0.9): {high_corr}/{total} ({100 * high_corr / total:.1f}%)")

        print(f"\n{group_name} analysis complete!")
        print("-" * 50)


def plot_individual_control_pool_analysis(
    data: pd.DataFrame,
    sample_columns: List[str],
    sample_metadata: Dict[str, Dict],
    control_column: str = "Subject",
    control_labels: Optional[List[str]] = None,
    normalization_method: str = "VSN",
    figsize: Tuple[int, int] = (15, 12),
) -> None:
    """
    Create detailed correlation analysis plots for individual control pools.
    Shows correlation matrices with scatter plots and histograms for each control pool.

    Parameters:
    -----------
    data : pd.DataFrame
        Normalized proteomics data
    sample_columns : List[str]
        Sample column names
    sample_metadata : Dict[str, Dict]
        Sample metadata
    control_column : str
        Column name to look for control samples
    control_labels : List[str], optional
        Labels in the control column that identify control samples.
        If None, function will look for samples containing common control patterns
        like 'Pool', 'Control', 'QC', 'Standard', etc.
    normalization_method : str
        Name of normalization method for title
    figsize : Tuple[int, int]
        Figure size (width, height)
    """

    # Handle default control labels - look for common control patterns
    if control_labels is None:
        # Find unique values in the control column
        unique_control_values = set()
        for sample in sample_columns:
            control_value = sample_metadata.get(sample, {}).get(control_column, "")
            if control_value and not pd.isna(control_value):
                unique_control_values.add(str(control_value))

        # Look for common control patterns
        control_patterns = [
            "Pool",
            "Control",
            "QC",
            "Standard",
            "Blank",
            "pool",
            "control",
            "qc",
            "standard",
            "blank",
        ]
        control_labels = []

        for value in unique_control_values:
            if any(pattern in value for pattern in control_patterns):
                control_labels.append(value)

        if not control_labels:
            print(f"No control samples found automatically in column '{control_column}'.")
            print(f"Available values: {sorted(unique_control_values)}")
            print("Please specify control_labels parameter explicitly.")
            return

    print(f"\n=== INDIVIDUAL CONTROL POOL CORRELATION ANALYSIS ({normalization_method}) ===")

    # Find control samples using simplified approach
    pool_samples = {}
    for control_label in control_labels:
        found_samples = []
        for sample in sample_columns:
            sample_value = sample_metadata.get(sample, {}).get(control_column, "")
            if sample_value == control_label:
                found_samples.append(sample)

        if found_samples:
            pool_samples[control_label] = found_samples

    if not pool_samples:
        print(f"No control pool samples found in column '{control_column}' with labels {control_labels}!")
        return

    # Process each control pool that has multiple samples
    valid_pools = {k: v for k, v in pool_samples.items() if len(v) >= 2}

    if not valid_pools:
        print("No control pools with 2+ samples found!")
        return

    print(f"\nAnalyzing {len(valid_pools)} control pools:")
    for pool_name, samples in valid_pools.items():
        print(f"  {pool_name}: {len(samples)} samples - {samples}")

    # Create subplot for each pool
    for _, (pool_name, pool_sample_list) in enumerate(valid_pools.items()):
        if len(pool_sample_list) < 2:
            continue

        print(f"\n--- {pool_name} Correlation Analysis ---")

        # Extract data for this control pool
        if normalization_method.upper() == "VSN":
            # VSN data doesn't need log transformation
            control_data = data[pool_sample_list].dropna()
        else:
            # Apply log2 transformation for other methods
            control_data = np.log2(data[pool_sample_list].replace(0, np.nan)).dropna()

        if len(control_data) == 0:
            print(f"No valid data for {pool_name}")
            continue

        print(f"Using {len(control_data)} proteins for {pool_name}")

        # Calculate correlation matrix
        corr_matrix = control_data.corr(method="pearson")
        n_samples = len(pool_sample_list)

        # Create correlation matrix plot with pairwise scatter plots
        fig, axes = plt.subplots(n_samples, n_samples, figsize=figsize)
        fig.suptitle(
            f"Correlation Matrix - {pool_name} ({normalization_method} Normalized)",
            fontsize=16,
            fontweight="bold",
        )

        # Handle single sample case (shouldn't happen but just in case)
        if n_samples == 1:
            axes = np.array([[axes]])
        elif n_samples == 2:
            axes = axes.reshape(2, 2)

        for row in range(n_samples):
            for col in range(n_samples):
                ax = axes[row, col]

                sample_x = pool_sample_list[col]
                sample_y = pool_sample_list[row]

                if row == col:
                    # Diagonal: histogram
                    values = control_data[sample_x].dropna()
                    ax.hist(values, bins=30, alpha=0.7, color="skyblue", density=True)
                    ax.set_title(sample_x, fontsize=10, fontweight="bold")
                    ax.set_ylabel("Density")
                elif row > col:
                    # Lower triangle: scatter plot
                    x_vals = control_data[sample_x]
                    y_vals = control_data[sample_y]

                    # Remove NaN pairs
                    valid_mask = ~(x_vals.isna() | y_vals.isna())
                    x_clean = x_vals[valid_mask]
                    y_clean = y_vals[valid_mask]

                    if len(x_clean) > 0:
                        ax.scatter(x_clean, y_clean, alpha=0.6, s=1, color="blue")

                        # Add trend line
                        if len(x_clean) > 1:
                            z = np.polyfit(x_clean, y_clean, 1)
                            p = np.poly1d(z)
                            ax.plot(x_clean, p(x_clean), "r-", alpha=0.8)

                        # Calculate correlation
                        correlation = corr_matrix.loc[sample_y, sample_x]

                        # Add correlation text
                        ax.text(
                            0.05,
                            0.95,
                            f"r = {correlation:.3f}",
                            transform=ax.transAxes,
                            fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                            verticalalignment="top",
                        )

                    ax.set_xlabel(sample_x, fontsize=9)
                    ax.set_ylabel(sample_y, fontsize=9)
                else:
                    # Upper triangle: correlation value
                    correlation = corr_matrix.loc[sample_y, sample_x]
                    ax.text(
                        0.5,
                        0.5,
                        f"{correlation:.3f}***" if correlation > 0.95 else f"{correlation:.3f}",
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                    )
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks([])
                    ax.set_yticks([])

                # Adjust tick labels
                ax.tick_params(axis="both", which="major", labelsize=8)

        plt.tight_layout()
        plt.show()

        # Print correlation summary for this pool
        off_diagonal = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        off_diagonal = off_diagonal[~np.isnan(off_diagonal)]

        if len(off_diagonal) > 0:
            print(f"\n{pool_name} Correlation Summary:")
            print(f"  Mean correlation: {np.mean(off_diagonal):.3f}")
            print(f"  Median correlation: {np.median(off_diagonal):.3f}")
            print(f"  Min correlation: {np.min(off_diagonal):.3f}")
            print(f"  Max correlation: {np.max(off_diagonal):.3f}")
            high_corr = np.sum(off_diagonal > 0.9)
            total = len(off_diagonal)
            print(f"  High correlation (>0.9): {high_corr}/{total} ({100 * high_corr / total:.1f}%)")

        print(f"\n{pool_name} analysis complete!")
        print("-" * 50)


def plot_control_cv_distribution(
    data: pd.DataFrame,
    sample_columns: List[str],
    sample_metadata: Dict[str, Dict],
    control_column: str,
    control_labels: List[str],
    normalization_method: str = "Median",
    figsize: Tuple[int, int] = (18, 6),
    cv_threshold: float = 20.0,
    title_suffix: str = "",
) -> Dict[str, List[float]]:
    """
    Create CV distribution histograms for control samples by control type.

    This function analyzes the coefficient of variance (CV) distribution for each
    control type to assess reproducibility between control replicates. Uses median
    normalized, non-log transformed data for CV calculations.

    Parameters:
    -----------
    data : pd.DataFrame
        Protein quantitation data (should be median normalized, non-log transformed)
    sample_columns : List[str]
        List of sample column names in the data
    sample_metadata : Dict[str, Dict]
        Sample metadata mapping {sample_name: {metadata_dict}}
    control_column : str
        Column name in metadata containing control sample designations
    control_labels : List[str]
        Labels identifying QC/control samples (e.g., ["HoofPool", "GWPool", "PlatePool"])
    normalization_method : str, optional
        Name of normalization method used (for plot title), default "Median"
    figsize : Tuple[int, int], optional
        Figure size (width, height), default (18, 6)
    cv_threshold : float, optional
        CV threshold line to display (%), default 20.0
    title_suffix : str, optional
        Additional text to append to plot title

    Returns:
    --------
    Dict[str, List[float]]
        Dictionary mapping control type to list of CV values for that type

    Notes:
    ------
    - CV is calculated as (std/mean * 100) for each protein across control samples
    - Only proteins with positive mean values are included in CV calculations
    - Lower CV values indicate better reproducibility between control replicates
    - At least 2 samples of each control type are required for CV calculation
    """

    print("=== CONTROL SAMPLE CV DISTRIBUTION ANALYSIS ===")

    # Get control samples for each control type
    control_samples_by_type = {}
    for control_label in control_labels:
        control_samples_by_type[control_label] = []
        for sample_name, metadata_dict in sample_metadata.items():
            if sample_name in sample_columns and metadata_dict.get(control_column) == control_label:
                control_samples_by_type[control_label].append(sample_name)

    print("Control samples by type:")
    for control_type, samples in control_samples_by_type.items():
        print(f"  {control_type}: {len(samples)} samples")

    # Calculate CV for each control type
    cv_data = {}
    for control_type, control_sample_list in control_samples_by_type.items():
        if len(control_sample_list) > 1:  # Need at least 2 samples to calculate CV
            # Get the control sample data
            control_data = data[control_sample_list]

            # Calculate CV for each protein across control samples
            mean_values = control_data.mean(axis=1)
            std_values = control_data.std(axis=1)

            # Calculate CV (std/mean * 100) for proteins with non-zero means
            cv_values = []
            for i in range(len(mean_values)):
                if mean_values.iloc[i] > 0:  # Avoid division by zero
                    cv = (std_values.iloc[i] / mean_values.iloc[i]) * 100
                    cv_values.append(cv)

            cv_data[control_type] = cv_values
            print(f"  {control_type}: {len(cv_values)} proteins with calculable CV")
        else:
            cv_data[control_type] = []
            print(f"  {control_type}: Insufficient samples for CV calculation")

    # Create the histogram plot
    fig, axes = plt.subplots(1, len(control_labels), figsize=figsize)
    if len(control_labels) == 1:
        axes = [axes]  # Ensure axes is always iterable

    title = f"Coefficient of Variance Distribution for Control Samples\n({normalization_method} Normalized Data)"
    if title_suffix:
        title += f" - {title_suffix}"
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for idx, control_type in enumerate(control_labels):
        ax = axes[idx]
        cv_values = cv_data.get(control_type, [])

        if len(cv_values) > 0:
            # Create histogram
            n, bins, patches = ax.hist(cv_values, bins=50, alpha=0.7, color="skyblue", edgecolor="black", linewidth=0.5)

            # Calculate statistics
            median_cv = np.median(cv_values)
            mean_cv = np.mean(cv_values)
            cv_under_threshold = (np.array(cv_values) < cv_threshold).sum() / len(cv_values) * 100
            cv_under_10 = (np.array(cv_values) < 10).sum() / len(cv_values) * 100

            # Add vertical line for median CV
            ax.axvline(median_cv, color="red", linestyle="--", linewidth=2, label=f"Median CV: {median_cv:.1f}%")

            # Add vertical line at CV threshold
            ax.axvline(
                cv_threshold,
                color="orange",
                linestyle=":",
                linewidth=2,
                alpha=0.8,
                label=f"{cv_threshold}% CV threshold",
            )

            # Add text annotation for CV statistics
            stats_text = f"{cv_under_threshold:.1f}% of proteins\nhave CV < {cv_threshold}%"
            ax.text(
                0.95,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            # Formatting
            ax.set_title(
                f"{control_type} Controls\n({len(control_samples_by_type[control_type])} samples)",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("Coefficient of Variance (%)", fontsize=12)
            ax.set_ylabel("Number of Proteins", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Set reasonable x-axis limits
            ax.set_xlim(0, min(100, np.percentile(cv_values, 99)))

            # Print detailed statistics
            print(f"\n{control_type} Statistics:")
            print(f"  Median CV: {median_cv:.2f}%")
            print(f"  Mean CV: {mean_cv:.2f}%")
            print(f"  Proteins with CV < {cv_threshold}%: {cv_under_threshold:.1f}%")
            print(f"  Proteins with CV < 10%: {cv_under_10:.1f}%")

        else:
            # Handle case with insufficient data
            ax.text(
                0.5,
                0.5,
                f"Insufficient data\nfor {control_type}\n(need ≥2 samples)",
                transform=ax.transAxes,
                fontsize=14,
                horizontalalignment="center",
                verticalalignment="center",
            )
            ax.set_title(
                f"{control_type} Controls\n({len(control_samples_by_type.get(control_type, []))} samples)",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xlabel("Coefficient of Variance (%)", fontsize=12)
            ax.set_ylabel("Number of Proteins", fontsize=12)

    plt.tight_layout()
    plt.show()

    print("\nDone: Control sample CV distribution analysis complete")
    print(f"Note: CV calculated from {normalization_method.lower()} normalized (non-log transformed) data")
    print("Lower CV values indicate better reproducibility between control replicates")

    return cv_data


# =============================================================================
# GROUPED DATA VISUALIZATIONS
# =============================================================================
# These functions create heatmaps and parallel coordinate plots for any grouped data,
# including temporal trends, dose-response, treatment groups, clusters, etc.


def plot_grouped_heatmap(
    data_df: pd.DataFrame,
    value_columns: List[str],
    group_column: str,
    label_column: Optional[str] = None,
    title: str = "Expression Heatmap by Group",
    cmap: str = "RdBu_r",
    zscore: bool = True,
    vmin: float = -2,
    vmax: float = 2,
    show_labels: bool = True,
    max_per_group: int = 50,
    sort_by_pvalue: bool = True,
    pvalue_column: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Create a heatmap of expression data organized by groups (clusters, treatments, etc).

    This is a general-purpose heatmap function that can be used for:
    - Temporal trends (weeks as columns, clusters as groups)
    - Dose-response (doses as columns, treatment groups as groups)
    - Any categorical grouping of proteins

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing the data to visualize
    value_columns : List[str]
        Column names containing the values to plot (e.g., Week_0, Week_2, Week_4)
    group_column : str
        Column name for grouping (e.g., 'Cluster', 'Treatment', 'Category')
    label_column : str, optional
        Column name for row labels (e.g., 'Gene'). If None, no labels shown.
    title : str
        Plot title
    cmap : str
        Matplotlib colormap name
    zscore : bool
        Whether to z-score normalize each row
    vmin, vmax : float
        Color scale limits (only used if zscore=True, otherwise auto-scaled)
    show_labels : bool
        Whether to show row labels (requires label_column)
    max_per_group : int
        Maximum rows to show per group
    sort_by_pvalue : bool
        Whether to sort by p-value within groups
    pvalue_column : str, optional
        Column name containing p-values for sorting
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None.

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> # Temporal heatmap by cluster
    >>> fig = plot_grouped_heatmap(
    ...     merged_df,
    ...     value_columns=['Week_0', 'Week_2', 'Week_4', 'Week_8'],
    ...     group_column='Cluster_Name',
    ...     label_column='Gene',
    ...     title='Temporal Protein Expression by Cluster'
    ... )

    >>> # Dose-response heatmap by treatment
    >>> fig = plot_grouped_heatmap(
    ...     dose_df,
    ...     value_columns=['Dose_0', 'Dose_10', 'Dose_50', 'Dose_100'],
    ...     group_column='Response_Type',
    ...     label_column='Gene'
    ... )
    """
    # Get data matrix
    X = data_df[value_columns].values.astype(float)

    # Z-score if requested
    if zscore:
        X_means = np.nanmean(X, axis=1, keepdims=True)
        X_stds = np.nanstd(X, axis=1, keepdims=True)
        X_stds[X_stds == 0] = 1
        X_z = (X - X_means) / X_stds
    else:
        X_z = X
        vmin, vmax = np.nanmin(X_z), np.nanmax(X_z)

    # Sort by group, then optionally by p-value
    sorted_df = data_df.copy()
    sorted_df["_z_data"] = list(X_z)

    sort_cols = [group_column]
    ascending = [True]
    if sort_by_pvalue and pvalue_column and pvalue_column in sorted_df.columns:
        sort_cols.append(pvalue_column)
        ascending.append(True)

    sorted_df = sorted_df.sort_values(sort_cols, ascending=ascending)

    # Limit rows per group
    limited_dfs = []
    for group in sorted_df[group_column].unique():
        group_subset = sorted_df[sorted_df[group_column] == group].head(max_per_group)
        limited_dfs.append(group_subset)
    sorted_df = pd.concat(limited_dfs)

    # Create heatmap data
    heatmap_data = np.vstack(sorted_df["_z_data"].values)

    # Clean column labels
    col_labels = [str(c).replace("Week_", "W").replace("Dose_", "D") for c in value_columns]

    # Row labels
    row_labels = sorted_df[label_column].tolist() if label_column and show_labels else None

    # Calculate figure size
    if figsize is None:
        fig_height = min(12, max(6, len(sorted_df) * 0.12))
        figsize = (12, fig_height)

    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(right=0.75, left=0.15)

    # Plot heatmap
    im = ax.imshow(heatmap_data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # X-axis labels
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Condition", fontsize=12, fontweight="bold")

    # Y-axis labels
    if row_labels and len(sorted_df) <= 50:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
    else:
        ax.set_ylabel(f"Proteins (n={len(sorted_df)})", fontsize=12)
        ax.set_yticks([])

    # Add group separators and labels
    cluster_bounds = []
    current_idx = 0
    for group in sorted_df[group_column].unique():
        n_in_group = (sorted_df[group_column] == group).sum()
        cluster_bounds.append((current_idx, group, n_in_group))
        if current_idx > 0:
            ax.axhline(y=current_idx - 0.5, color="white", linewidth=2)
        current_idx += n_in_group

    # Group labels on right side
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    group_mids = [start + n / 2 for start, _, n in cluster_bounds]
    group_labels_text = [f"{name}\n(n={n})" for _, name, n in cluster_bounds]
    ax2.set_yticks(group_mids)
    ax2.set_yticklabels(group_labels_text, fontsize=10, fontweight="bold")

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Z-score" if zscore else "Value", fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    return fig


def plot_grouped_trajectories(
    data_df: pd.DataFrame,
    value_columns: List[str],
    group_column: str,
    title: str = "Expression Trajectories by Group",
    x_values: Optional[List[float]] = None,
    x_label: str = "Condition",
    y_label: str = "Z-scored Value",
    zscore: bool = True,
    alpha: float = 0.3,
    linewidth: float = 1.5,
    line_color: str = "#1f4e79",
    mean_color: str = "black",
    show_mean: bool = True,
    n_cols: int = 2,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Create parallel coordinate / trajectory plots for each group.

    This function creates a grid of line plots showing individual trajectories
    and mean trajectory for each group. Useful for:
    - Temporal trends across time points
    - Dose-response curves
    - Any continuous variable across conditions

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing the data to visualize
    value_columns : List[str]
        Column names containing the values (e.g., Week_0, Week_2, etc.)
    group_column : str
        Column name for grouping
    title : str
        Overall plot title
    x_values : List[float], optional
        Numeric x-axis values. If None, extracted from column names or uses indices.
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    zscore : bool
        Whether to z-score normalize each row
    alpha : float
        Transparency for individual lines
    linewidth : float
        Line width for individual lines
    line_color : str
        Color for individual trajectory lines
    mean_color : str
        Color for mean trajectory line
    show_mean : bool
        Whether to show mean trajectory with highlighted line
    n_cols : int
        Number of columns in subplot grid
    figsize : tuple, optional
        Figure size. Auto-calculated if None.

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> # Temporal trajectories by cluster
    >>> fig = plot_grouped_trajectories(
    ...     merged_df,
    ...     value_columns=['Week_0', 'Week_2', 'Week_4', 'Week_8'],
    ...     group_column='Cluster_Name',
    ...     x_values=[0, 2, 4, 8],
    ...     x_label='Week',
    ...     title='Temporal Patterns by Cluster'
    ... )

    >>> # Dose-response by sensitivity group
    >>> fig = plot_grouped_trajectories(
    ...     dose_df,
    ...     value_columns=['Dose_0', 'Dose_10', 'Dose_50', 'Dose_100'],
    ...     group_column='Sensitivity',
    ...     x_values=[0, 10, 50, 100],
    ...     x_label='Dose (mg)',
    ...     title='Dose Response by Sensitivity'
    ... )
    """
    # Get data and optionally z-score
    X = data_df[value_columns].values.astype(float)

    if zscore:
        X_means = np.nanmean(X, axis=1, keepdims=True)
        X_stds = np.nanstd(X, axis=1, keepdims=True)
        X_stds[X_stds == 0] = 1
        X_z = (X - X_means) / X_stds
        X_z = np.nan_to_num(X_z, nan=0, posinf=0, neginf=0)
    else:
        X_z = X

    # Extract x-values from column names if not provided
    if x_values is None:
        # Try to extract numbers from column names
        x_values = []
        for col in value_columns:
            # Try common patterns: Week_0, Week0, W0, Dose_10, D10, etc.
            import re

            match = re.search(r"[-+]?\d*\.?\d+", str(col))
            if match:
                x_values.append(float(match.group()))
            else:
                x_values.append(len(x_values))  # Fallback to index

    # Get unique groups
    groups = data_df[group_column].unique()
    n_groups = len(groups)

    n_rows = (n_groups + n_cols - 1) // n_cols

    if figsize is None:
        figsize = (7 * n_cols, max(4 * n_rows, 8))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, group_name in enumerate(groups):
        ax = axes[idx]

        mask = data_df[group_column] == group_name
        n_items = mask.sum()

        if n_items > 0:
            # Plot individual trajectories
            for i, is_in_group in enumerate(mask):
                if is_in_group:
                    ax.plot(x_values, X_z[i], color=line_color, alpha=alpha, linewidth=linewidth)

            # Plot mean trajectory
            if show_mean:
                group_mean = X_z[mask].mean(axis=0)
                ax.plot(x_values, group_mean, color=mean_color, linewidth=4, label=f"Mean (n={n_items})")
                ax.plot(x_values, group_mean, color="orange", linewidth=2.5, linestyle="--")

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"{group_name}\n(n={n_items})", fontsize=12, fontweight="bold")
        ax.set_xticks(x_values)
        if zscore:
            ax.set_ylim(-2.5, 2.5)
        if show_mean:
            ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    # Hide unused subplots
    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    return fig


def plot_protein_profile(
    data_df: pd.DataFrame,
    protein_id: str,
    value_columns: List[str],
    protein_column: str = "Protein",
    gene_column: Optional[str] = "Gene",
    x_values: Optional[List[float]] = None,
    x_label: str = "Condition",
    y_label: str = "Abundance",
    title: Optional[str] = None,
    color: str = "#1f4e79",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot the expression profile of a single protein across conditions.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing protein data
    protein_id : str
        Protein identifier to plot
    value_columns : List[str]
        Column names containing values across conditions
    protein_column : str
        Column name containing protein identifiers
    gene_column : str, optional
        Column name containing gene names for title
    x_values : List[float], optional
        X-axis values. If None, uses indices.
    x_label, y_label : str
        Axis labels
    title : str, optional
        Plot title. If None, auto-generated from protein/gene name.
    color : str
        Line and marker color
    figsize : tuple
        Figure size

    Returns
    -------
    Figure
        Matplotlib figure object
    """
    # Find the protein
    mask = data_df[protein_column] == protein_id
    if not mask.any():
        raise ValueError(f"Protein '{protein_id}' not found in data")

    protein_row = data_df[mask].iloc[0]
    values = protein_row[value_columns].values.astype(float)

    if x_values is None:
        x_values = list(range(len(value_columns)))

    # Generate title
    if title is None:
        gene = protein_row.get(gene_column, protein_id) if gene_column else protein_id
        title = f"{gene} ({protein_id})"

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x_values, values, "o-", color=color, linewidth=2, markersize=10)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x_values)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Generic QC plots
#
# These work on either protein-level or peptide-level DataFrames. They operate
# on the sample columns plus a ``feature_label`` (default "protein") used only
# for axis titles.
# ---------------------------------------------------------------------------


def plot_missing_value_heatmap(
    data: pd.DataFrame,
    sample_columns: List[str],
    feature_label: str = "protein",
    figsize: Tuple[float, float] = (12, 8),
    max_features: int = 500,
    cluster_features: bool = True,
) -> plt.Figure:
    """Visualise the missing-value pattern across samples x features.

    Produces a binary heatmap where dark cells indicate missing (NaN or zero)
    values and light cells indicate detected values. Useful for spotting
    systematic missingness tied to specific samples or batches.

    Parameters
    ----------
    data : pd.DataFrame
        Protein- or peptide-level intensity DataFrame.
    sample_columns : list of str
        Names of the sample columns to include.
    feature_label : str
        Label for the feature axis (e.g. "protein" or "peptide").
    figsize : tuple
        Figure size.
    max_features : int
        If the DataFrame has more rows than this, sample a random subset to
        keep the heatmap readable.
    cluster_features : bool
        If True, order features by detection count (most-detected at top).

    Returns
    -------
    matplotlib.figure.Figure
    """
    sample_data = data[sample_columns]
    is_missing = sample_data.isna() | (sample_data == 0)

    if cluster_features:
        detection_count = (~is_missing).sum(axis=1)
        order = detection_count.sort_values(ascending=False).index
        is_missing = is_missing.loc[order]

    if len(is_missing) > max_features:
        is_missing = is_missing.sample(n=max_features, random_state=0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(is_missing.astype(int).values, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xticks(range(len(sample_columns)))
    ax.set_xticklabels(_make_display_labels(sample_columns), rotation=90, fontsize=8)
    ax.set_xlabel("Sample")
    ax.set_ylabel(f"{feature_label.capitalize()}s (ordered by detection)")
    ax.set_title(f"Missing-value pattern ({len(is_missing)} {feature_label}s shown)")
    plt.tight_layout()
    return fig


def plot_identifications_per_sample(
    data: pd.DataFrame,
    sample_columns: List[str],
    sample_metadata: Optional[Dict[str, Dict]] = None,
    group_column: str = "Group",
    feature_label: str = "protein",
    figsize: Tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Bar chart of the number of features identified per sample.

    Parameters
    ----------
    data : pd.DataFrame
        Protein- or peptide-level intensity DataFrame.
    sample_columns : list of str
        Names of the sample columns to include.
    sample_metadata : dict, optional
        Mapping from sample name to metadata dict; used to colour bars by group.
    group_column : str
        Key in ``sample_metadata`` entries to use for group colouring.
    feature_label : str
        Label for the feature axis (e.g. "protein" or "peptide").
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    sample_data = data[sample_columns]
    counts = (~(sample_data.isna() | (sample_data == 0))).sum(axis=0)

    # Assign colours by group if metadata is available
    if sample_metadata is not None:
        groups = [sample_metadata.get(s, {}).get(group_column, "Unknown") for s in sample_columns]
        unique_groups = sorted(set(groups))
        palette = sns.color_palette("Set1", n_colors=max(len(unique_groups), 3))
        group_to_color = {g: palette[i % len(palette)] for i, g in enumerate(unique_groups)}
        bar_colors = [group_to_color[g] for g in groups]
    else:
        bar_colors = "steelblue"
        group_to_color = None

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sample_columns))
    ax.bar(x, counts.values, color=bar_colors)
    ax.set_xticks(x)
    ax.set_xticklabels(_make_display_labels(sample_columns), rotation=90, fontsize=8)
    ax.set_ylabel(f"{feature_label.capitalize()}s identified")
    ax.set_title(f"{feature_label.capitalize()} identifications per sample")

    if group_to_color is not None:
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in group_to_color.values()]
        ax.legend(handles, group_to_color.keys(), title=group_column, loc="best", fontsize=8)

    plt.tight_layout()
    return fig


def plot_intensity_distributions(
    data: pd.DataFrame,
    sample_columns: List[str],
    log_transform: bool = True,
    feature_label: str = "protein",
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Density plot overlay of (log) intensity per sample.

    Useful for visually checking that samples are on comparable scales before
    and after normalization.

    Parameters
    ----------
    data : pd.DataFrame
        Protein- or peptide-level intensity DataFrame.
    sample_columns : list of str
        Names of the sample columns to include.
    log_transform : bool
        If True, log2-transform non-zero values before plotting.
    feature_label : str
        Label used in the plot title (e.g. "protein" or "peptide").
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    display_names = _make_display_labels(sample_columns)

    for col, label in zip(sample_columns, display_names):
        values = data[col].replace(0, np.nan).dropna().values
        if log_transform:
            values = np.log2(values)
        if len(values) > 1:
            sns.kdeplot(values, ax=ax, label=label, linewidth=1, alpha=0.7)

    ax.set_xlabel("log2(intensity)" if log_transform else "intensity")
    ax.set_ylabel("Density")
    ax.set_title(f"{feature_label.capitalize()} intensity distribution per sample")
    ax.legend(fontsize=7, loc="best", ncol=2)
    plt.tight_layout()
    return fig


def plot_cv_distribution(
    data: pd.DataFrame,
    sample_columns: List[str],
    sample_metadata: Optional[Dict[str, Dict]] = None,
    group_column: str = "Group",
    feature_label: str = "protein",
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Coefficient-of-variation distribution across all samples (or by group).

    For each feature, computes CV = std / mean across the supplied samples.
    When ``sample_metadata`` is provided and contains ``group_column``,
    a separate CV distribution is drawn per group (overlayed).

    Parameters
    ----------
    data : pd.DataFrame
        Protein- or peptide-level intensity DataFrame.
    sample_columns : list of str
        Names of the sample columns to include.
    sample_metadata : dict, optional
        Mapping from sample name to metadata dict.
    group_column : str
        Key in ``sample_metadata`` entries to use for per-group distributions.
    feature_label : str
        Label used in the plot title (e.g. "protein" or "peptide").
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """

    def _cv(frame: pd.DataFrame) -> pd.Series:
        mean = frame.mean(axis=1)
        std = frame.std(axis=1)
        cv = (std / mean).replace([np.inf, -np.inf], np.nan).dropna()
        return cv * 100  # percent

    fig, ax = plt.subplots(figsize=figsize)

    if sample_metadata is not None:
        groups = [sample_metadata.get(s, {}).get(group_column, "Unknown") for s in sample_columns]
        group_to_samples: Dict[str, List[str]] = {}
        for s, g in zip(sample_columns, groups):
            group_to_samples.setdefault(g, []).append(s)
        palette = sns.color_palette("Set1", n_colors=max(len(group_to_samples), 3))
        for i, (group, cols) in enumerate(sorted(group_to_samples.items())):
            if len(cols) >= 2:
                cv_values = _cv(data[cols])
                sns.kdeplot(cv_values, ax=ax, label=f"{group} (n={len(cols)})", color=palette[i % len(palette)])
        ax.legend(title=group_column, loc="best")
    else:
        cv_values = _cv(data[sample_columns])
        sns.kdeplot(cv_values, ax=ax, color="steelblue", fill=True, alpha=0.3)

    ax.set_xlabel("CV (%)")
    ax.set_ylabel("Density")
    ax.set_title(f"{feature_label.capitalize()}-level CV distribution")
    plt.tight_layout()
    return fig


def _assign_peptide_tracks(intervals: List[Tuple[int, int]]) -> List[int]:
    """Greedy interval-to-track assignment.

    Given a list of ``(start, end)`` intervals (both 1-based, inclusive on
    start and exclusive on end), return a list of track indices such that
    no two overlapping intervals share a track. Tracks are reused as soon
    as they're free. Input order is preserved; for DIA data with mostly
    non-overlapping peptides this typically returns all zeros.
    """
    track_ends: List[int] = []
    tracks: List[int] = []
    for start, end in intervals:
        placed = False
        for t, t_end in enumerate(track_ends):
            if start >= t_end:
                tracks.append(t)
                track_ends[t] = end
                placed = True
                break
        if not placed:
            tracks.append(len(track_ends))
            track_ends.append(end)
    return tracks


def plot_peptide_coverage_map(
    peptide_data: pd.DataFrame,
    protein_id: str,
    protein_sequence: str,
    sample_columns: List[str],
    protein_column: str = "leading_protein",
    sequence_column: str = "peptide_sequence",
    start_column: Optional[str] = None,
    color_by: str = "abundance",
    value_column: Optional[str] = None,
    sample_metadata: Optional[Dict[str, Dict]] = None,
    group_column: Optional[str] = None,
    group_labels: Optional[Tuple[str, str]] = None,
    wrap_residues: int = 60,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Sequence-aware peptide coverage map for a single protein.

    Renders the protein amino-acid sequence as text along the top of each
    row, with peptide bars drawn beneath at their true start/end
    residue positions. Long proteins wrap to multiple rows at
    ``wrap_residues`` (default 60). Peptide bars are coloured by
    per-peptide **abundance**, **fold-change**, or **detection frequency**
    depending on ``color_by``. Overlapping peptides (e.g. from missed
    cleavages) stack vertically within a row.

    Parameters
    ----------
    peptide_data : pd.DataFrame
        Peptide-level DataFrame. Must contain ``protein_column`` and
        ``sequence_column``.
    protein_id : str
        Protein identifier to match in ``protein_column``.
    protein_sequence : str
        Amino-acid sequence of the parent protein. **Required.** Use
        :func:`~proteomics_toolkit.data_import.load_fasta_sequences` to
        read it from a FASTA file.
    sample_columns : list of str
        Sample columns used to compute abundance / fold-change /
        detection values (whichever ``color_by`` selects).
    protein_column : str
        Column holding the parent-protein identifier for each peptide.
    sequence_column : str
        Column holding the peptide amino-acid sequence.
    start_column : str, optional
        Column holding 1-based start positions along the parent protein.
        If ``None``, each peptide is located by ``str.find`` in
        ``protein_sequence``; peptides that don't match are skipped with
        a warning.
    color_by : str
        One of ``"abundance"`` (default), ``"fold_change"``, or
        ``"detection"``.
    value_column : str, optional
        When ``color_by="fold_change"``, a pre-computed per-peptide
        log2 fold-change column in ``peptide_data`` to use directly.
        If omitted, the fold-change is computed from
        ``sample_metadata`` + ``group_column`` + ``group_labels``.
    sample_metadata : dict, optional
        Required for ``color_by="fold_change"`` when ``value_column``
        is not supplied. Maps sample column name -> metadata dict.
    group_column : str, optional
        Key in ``sample_metadata`` entries to split samples for fold-change.
    group_labels : tuple of (str, str), optional
        ``(reference_group, study_group)`` labels for fold-change.
    wrap_residues : int
        Residues per sequence row. Default 60.
    cmap : str, optional
        Matplotlib colormap name. Defaults: ``"viridis"`` for abundance,
        ``"coolwarm"`` for fold-change, ``"Blues"`` for detection.
    figsize : tuple, optional
        Figure size. If None, chosen based on the number of rows needed.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``protein_sequence`` is empty, if no peptides match
        ``protein_id``, or if ``color_by`` is invalid / its required
        extra inputs are missing.
    """
    import matplotlib as mpl
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    valid_color_by = {"abundance", "fold_change", "detection"}
    if color_by not in valid_color_by:
        raise ValueError(f"color_by must be one of {sorted(valid_color_by)}; got {color_by!r}")
    if not isinstance(protein_sequence, str) or len(protein_sequence) == 0:
        raise ValueError(
            "protein_sequence is required and must be a non-empty amino-acid string. "
            "Use ptk.load_fasta_sequences(path) to read it from a FASTA file."
        )

    subset = peptide_data[peptide_data[protein_column] == protein_id].copy()
    if subset.empty:
        raise ValueError(f"No peptides found matching {protein_column}={protein_id!r}")

    protein_length = len(protein_sequence)
    sequences = subset[sequence_column].astype(str).tolist()
    lengths = [len(seq) for seq in sequences]

    # Resolve start positions
    if start_column is not None and start_column in subset.columns:
        starts = [int(x) for x in subset[start_column].tolist()]
    else:
        starts = []
        for seq in sequences:
            pos = protein_sequence.find(seq)
            starts.append(pos + 1 if pos >= 0 else -1)  # 1-based

    # Drop peptides that couldn't be located
    keep_mask = [s > 0 and (s - 1 + l_) <= protein_length for s, l_ in zip(starts, lengths)]
    n_dropped = sum(1 for k in keep_mask if not k)
    if n_dropped:
        logger.warning(
            "%d of %d peptides for %s could not be located in the protein sequence and were skipped.",
            n_dropped,
            len(sequences),
            protein_id,
        )
    keep_idx = [i for i, k in enumerate(keep_mask) if k]
    if not keep_idx:
        raise ValueError(
            f"No peptides for {protein_id!r} could be located in the supplied protein sequence."
        )
    sequences = [sequences[i] for i in keep_idx]
    starts = [starts[i] for i in keep_idx]
    lengths = [lengths[i] for i in keep_idx]
    subset = subset.iloc[keep_idx]

    # Compute per-peptide colour values
    sample_frame = subset[sample_columns]
    if color_by == "detection":
        values = (~(sample_frame.isna() | (sample_frame == 0))).sum(axis=1).to_numpy() / len(sample_columns)
        colour_label = "Detection frequency"
        default_cmap = "Blues"
        norm: "Normalize | TwoSlopeNorm" = Normalize(vmin=0.0, vmax=1.0)
    elif color_by == "abundance":
        mean_int = sample_frame.mean(axis=1).to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.log2(np.where(mean_int > 0, mean_int, np.nan))
        colour_label = "log2(mean intensity)"
        default_cmap = "viridis"
        finite = values[np.isfinite(values)]
        norm = Normalize(vmin=float(np.nanmin(finite)), vmax=float(np.nanmax(finite)))
    else:  # fold_change
        if value_column is not None:
            if value_column not in subset.columns:
                raise ValueError(f"value_column {value_column!r} not in peptide_data columns.")
            values = subset[value_column].to_numpy(dtype=float)
        else:
            if not (sample_metadata and group_column and group_labels and len(group_labels) == 2):
                raise ValueError(
                    "color_by='fold_change' requires either value_column or "
                    "(sample_metadata, group_column, group_labels=(ref, study))."
                )
            ref_label, study_label = group_labels[0], group_labels[1]
            ref_cols = [c for c in sample_columns if sample_metadata.get(c, {}).get(group_column) == ref_label]
            study_cols = [c for c in sample_columns if sample_metadata.get(c, {}).get(group_column) == study_label]
            if not ref_cols or not study_cols:
                raise ValueError(
                    f"No samples found matching group_labels ref={ref_label!r} or study={study_label!r}."
                )
            ref_mean = subset[ref_cols].mean(axis=1).to_numpy(dtype=float)
            study_mean = subset[study_cols].mean(axis=1).to_numpy(dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                values = np.log2(np.where((ref_mean > 0) & (study_mean > 0), study_mean / ref_mean, np.nan))
        colour_label = "log2 fold-change"
        default_cmap = "coolwarm"
        finite = values[np.isfinite(values)]
        if finite.size:
            vmax = float(np.nanmax(np.abs(finite)))
            vmax = vmax if vmax > 0 else 1.0
        else:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    cmap_obj = mpl.colormaps.get_cmap(cmap or default_cmap)

    # Figure out how many rows we need; compute per-row track counts for
    # stacking height.
    n_rows = int(np.ceil(protein_length / wrap_residues))
    # For each peptide, figure out which rows it touches and register the
    # sub-interval on each.
    peptide_row_spans: List[List[Tuple[int, int, int]]] = [[] for _ in range(len(sequences))]
    for pi, (start, length) in enumerate(zip(starts, lengths)):
        end = start + length  # exclusive
        first_row = (start - 1) // wrap_residues
        last_row = (end - 2) // wrap_residues
        for r in range(first_row, last_row + 1):
            row_start_res = r * wrap_residues + 1
            row_end_res = (r + 1) * wrap_residues + 1
            seg_start = max(start, row_start_res)
            seg_end = min(end, row_end_res)
            peptide_row_spans[pi].append((r, seg_start, seg_end))

    # Assign tracks per row. We process peptides in global start order so
    # tracks assigned in one row are consistent with the next.
    pep_order = sorted(range(len(sequences)), key=lambda i: starts[i])
    row_track_assignments: List[Dict[int, int]] = [dict() for _ in range(n_rows)]
    row_intervals_for_assign: List[List[Tuple[int, int]]] = [[] for _ in range(n_rows)]
    row_peptides: List[List[int]] = [[] for _ in range(n_rows)]
    for pi in pep_order:
        for r, seg_start, seg_end in peptide_row_spans[pi]:
            row_intervals_for_assign[r].append((seg_start, seg_end))
            row_peptides[r].append(pi)

    for r in range(n_rows):
        tracks = _assign_peptide_tracks(row_intervals_for_assign[r])
        for pep_idx, track in zip(row_peptides[r], tracks):
            # Store the largest track used by this peptide on this row;
            # different row segments may land on different tracks.
            prev = row_track_assignments[r].get(pep_idx)
            if prev is None or track > prev:
                row_track_assignments[r][pep_idx] = track

    max_tracks_per_row = [
        max(row_track_assignments[r].values()) + 1 if row_track_assignments[r] else 1 for r in range(n_rows)
    ]

    # Layout: each row reserves 1 unit for the sequence text + n tracks
    # worth of peptide bars + a small gap.
    row_heights = [1 + max_tracks_per_row[r] * 0.9 + 0.6 for r in range(n_rows)]
    total_height = sum(row_heights)
    if figsize is None:
        fig_h = max(3.0, 0.8 * total_height)
        figsize = (max(12.0, wrap_residues * 0.18), fig_h)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw rows from top to bottom: top of figure = residues 1..wrap
    row_top = total_height
    row_tops = []
    for r in range(n_rows):
        row_top -= row_heights[r]
        row_tops.append(row_top)

    for r in range(n_rows):
        row_start_res = r * wrap_residues + 1
        row_end_res = min((r + 1) * wrap_residues, protein_length)
        sequence_y = row_tops[r] + max_tracks_per_row[r] * 0.9 + 0.2

        # Sequence text (one character per residue)
        for res_pos in range(row_start_res, row_end_res + 1):
            ax.text(
                res_pos,
                sequence_y,
                protein_sequence[res_pos - 1],
                ha="center",
                va="center",
                family="monospace",
                fontsize=9,
            )
        # Residue-number ticks every 10 positions
        tick_positions = [p for p in range(row_start_res, row_end_res + 1) if p % 10 == 0 or p == row_start_res]
        for tp in tick_positions:
            ax.text(tp, sequence_y + 0.6, str(tp), ha="center", va="center", fontsize=7, color="gray")

        # Peptide bars on this row
        for pep_idx, track in row_track_assignments[r].items():
            # Find this peptide's segment on this row
            seg_start, seg_end = None, None
            for rr, s, e in peptide_row_spans[pep_idx]:
                if rr == r:
                    seg_start, seg_end = s, e
                    break
            if seg_start is None:
                continue
            val = values[pep_idx]
            face = cmap_obj(norm(val)) if np.isfinite(val) else (0.8, 0.8, 0.8, 1.0)
            y = row_tops[r] + (max_tracks_per_row[r] - 1 - track) * 0.9 + 0.1
            ax.add_patch(
                Rectangle(
                    (seg_start, y),
                    seg_end - seg_start,
                    0.7,
                    facecolor=face,
                    edgecolor="black",
                    linewidth=0.3,
                )
            )

    ax.set_xlim(0, wrap_residues + 1)
    ax.set_ylim(0, total_height)
    ax.set_xlabel("Residue position (within row)")
    ax.set_yticks([])
    ax.set_title(f"Peptide coverage: {protein_id}")
    # Hide the spines for a cleaner sequence-viewer look
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    # Colourbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label(colour_label)

    plt.tight_layout()
    return fig


def plot_variance_vs_peptide_count(
    results: pd.DataFrame,
    count_column: str = "peptide_count_used",
    variance_column: str = "residual_s2",
    limma_prior_column: str = "limma_s0_sq",
    figsize: Tuple[float, float] = (8, 6),
    lowess_frac: float = 0.5,
) -> plt.Figure:
    """Diagnostic plot for the DEqMS peptide-count-conditioned variance prior.

    Reproduces the standard DEqMS diagnostic (cf. Zhu et al. 2020 Fig. 1B
    and Zhu et al. 2026 Nat. Protoc.): a scatter of per-protein pooled
    within-group residual variance against peptide count, with both the
    limma global prior (horizontal line) and the DEqMS
    count-dependent prior (LOWESS curve) overlaid so the two priors can
    be compared directly.

    Axes follow the DEqMS convention: ``ln(variance)`` on Y and
    ``log2(peptide count)`` on X. The variance being plotted is the
    pooled within-group residual variance from the full-cohort linear
    model (same quantity that both priors are fit against), so the
    scatter uses all samples in all groups.

    Expects the output of :func:`run_deqms_like_analysis`. Will also
    work on a :func:`run_limma_like_analysis` result if a peptide-count
    column is merged in; in that case the LOWESS curve is just a
    smoothed trend and no DEqMS prior was actually applied.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain ``count_column`` and ``variance_column``. The
        ``limma_prior_column`` is used for the horizontal reference line
        if present; otherwise it is omitted.
    count_column : str
        Column of peptide counts. Default ``"peptide_count_used"``.
    variance_column : str
        Column of per-feature pooled within-group residual variances.
        Default ``"residual_s2"``.
    limma_prior_column : str
        Column holding the limma global prior variance. Default
        ``"limma_s0_sq"``.
    figsize : tuple
        Figure size.
    lowess_frac : float
        LOWESS smoothing parameter (fraction of points used per local fit).

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``count_column`` or ``variance_column`` is missing.
    """
    from statsmodels.nonparametric.smoothers_lowess import lowess

    missing = [c for c in (count_column, variance_column) if c not in results.columns]
    if missing:
        raise ValueError(
            f"Results DataFrame missing required columns: {missing}. "
            f"Pass the output of run_deqms_like_analysis (includes 'peptide_count_used' "
            f"and 'residual_s2')."
        )

    counts = np.asarray(results[count_column], dtype=float)
    variances = np.asarray(results[variance_column], dtype=float)
    mask = np.isfinite(counts) & np.isfinite(variances) & (counts >= 1) & (variances > 0)
    if mask.sum() < 3:
        raise ValueError("Need at least 3 features with valid counts and variances to plot.")

    log2_counts = np.log2(counts[mask])
    ln_var = np.log(variances[mask])

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(log2_counts, ln_var, s=12, alpha=0.5, color="black", label="Proteins")

    # DEqMS LOWESS prior
    smoothed = lowess(ln_var, log2_counts, frac=lowess_frac, it=3, return_sorted=True)
    ax.plot(smoothed[:, 0], smoothed[:, 1], color="crimson", linewidth=2.5, label="DEqMS prior (LOWESS)")

    # Limma global prior (horizontal line) if available
    if limma_prior_column in results.columns:
        limma_vals = np.asarray(results[limma_prior_column], dtype=float)
        limma_vals = limma_vals[np.isfinite(limma_vals) & (limma_vals > 0)]
        if len(limma_vals) > 0:
            limma_ln = float(np.log(limma_vals[0]))  # constant across proteins
            ax.axhline(
                limma_ln,
                color="gray",
                linewidth=2,
                linestyle="-",
                label="Limma prior (global)",
            )

    ax.set_xlabel("log2(peptide count)")
    ax.set_ylabel("ln(variance)")
    ax.set_title("Estimation of variance prior (DEqMS diagnostic)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_variance_vs_intensity(
    results: pd.DataFrame,
    figsize: Tuple[float, float] = (8, 6),
) -> plt.Figure:
    """Diagnostic for the ``intensity_trend`` moderation prior.

    Renders one point per (feature, group) pair: Y = within-group
    standard deviation across that group's samples, X = √(within-group
    mean intensity). Under pure Poisson noise the points lie on a line
    through the origin with slope `k`, so a straight linear cloud with
    slope ≈ 1 suggests counting-noise-dominated data.

    Overlays the LOWESS fit (same curve used by the prior) and a dashed
    reference line ``sd = k·√intensity`` with `k` estimated from the
    cloud.

    Parameters
    ----------
    results : pd.DataFrame
        The output of :func:`~proteomics_toolkit.statistical_analysis.run_moderated_linear_model`
        with ``moderation="intensity_trend"``. The per-(feature, group)
        cloud is read from
        ``results.attrs["intensity_trend_points"]`` (populated
        automatically; also retrievable via
        :func:`~proteomics_toolkit.statistical_analysis.get_intensity_trend_points`).
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``results`` does not carry the intensity-trend points (i.e.
        was not produced with ``moderation="intensity_trend"``).
    """
    pts = None
    if hasattr(results, "attrs"):
        pts = results.attrs.get("intensity_trend_points")
    if pts is None:
        raise ValueError(
            "Results DataFrame does not carry intensity-trend points. "
            "Run run_moderated_linear_model with config.moderation='intensity_trend' "
            "first, or pass get_intensity_trend_points(results)."
        )

    mean_intensity = pts["mean_intensity"].to_numpy(dtype=float)
    sd_intensity = pts["sd_intensity"].to_numpy(dtype=float)
    predicted_sd = pts["predicted_sd"].to_numpy(dtype=float) if "predicted_sd" in pts.columns else None

    mask = np.isfinite(mean_intensity) & np.isfinite(sd_intensity) & (mean_intensity > 0) & (sd_intensity > 0)
    if mask.sum() < 3:
        raise ValueError("Need at least 3 (feature, group) points with valid mean/SD to plot.")

    sqrt_mean = np.sqrt(mean_intensity[mask])
    sd = sd_intensity[mask]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(sqrt_mean, sd, s=10, alpha=0.4, color="black", label="(feature, group)")

    # LOWESS curve from the stored predicted SD, sorted by sqrt(mean)
    if predicted_sd is not None:
        finite = np.isfinite(mean_intensity) & np.isfinite(predicted_sd) & (mean_intensity > 0)
        if finite.sum() > 2:
            order = np.argsort(np.sqrt(mean_intensity[finite]))
            ax.plot(
                np.sqrt(mean_intensity[finite])[order],
                predicted_sd[finite][order],
                color="crimson",
                linewidth=2.5,
                label="LOWESS prior",
            )

    # Poisson reference line sd = k·√intensity, k fit to the cloud
    k = float(np.sum(sd * sqrt_mean) / np.sum(sqrt_mean * sqrt_mean))
    x_ref = np.linspace(0, sqrt_mean.max() * 1.02, 100)
    ax.plot(
        x_ref,
        k * x_ref,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"sd = {k:.3g}·√intensity (Poisson-like)",
    )

    ax.set_xlabel("√(mean intensity)")
    ax.set_ylabel("Within-group standard deviation")
    ax.set_title("Variance prior (intensity_trend diagnostic)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_pca_loadings(
    data: pd.DataFrame,
    sample_columns: List[str],
    pc1: int = 0,
    pc2: int = 1,
    top_n: int = 20,
    annotation_column: str = "leading_gene_name",
    log_transform: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "PCA Loadings",
) -> plt.Figure:
    """PCA loadings biplot showing the top-N proteins driving two PCs.

    Computes PCA on the protein abundance matrix (samples as rows,
    proteins as columns), then plots each protein's loading on the two
    requested principal components. The top-N proteins by Euclidean
    norm of their (pc1, pc2) loading vector are labeled.

    This complements ``plot_pca`` (which shows samples in PC space) by
    showing which proteins drive the structure.

    Args:
        data: Wide protein DataFrame with annotation columns followed by
            sample columns.
        sample_columns: Sample column names to include.
        pc1: Index (0-based) of the principal component for the x-axis.
        pc2: Index (0-based) of the principal component for the y-axis.
        top_n: Number of proteins to label.
        annotation_column: Column in ``data`` to use for protein labels.
            Falls back to a numeric index if absent.
        log_transform: log2-transform abundance before PCA. Pass False if
            the input is already log-transformed.
        figsize: Figure size tuple.
        title: Plot title.

    Returns:
        matplotlib Figure object.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    abundance = data[sample_columns].apply(pd.to_numeric, errors="coerce")
    if log_transform:
        with np.errstate(invalid="ignore", divide="ignore"):
            abundance = np.log2(abundance.where(abundance > 0))
    abundance = abundance.dropna(axis=0, how="any")
    if abundance.empty:
        raise ValueError("No proteins remain after dropping rows with NaN values.")

    n_components = max(pc1, pc2) + 1
    # After mean-centering, the rank of the data matrix is at most n_samples - 1,
    # so PCA needs strictly more samples than the number of components requested.
    if abundance.shape[1] <= n_components:
        raise ValueError(
            f"Need at least {n_components + 1} samples for PC{pc1 + 1}/PC{pc2 + 1}; "
            f"have {abundance.shape[1]}."
        )

    # Samples as rows -> PCA components are loadings on proteins.
    sample_matrix = abundance.to_numpy(dtype=float).T
    sample_matrix = StandardScaler().fit_transform(sample_matrix)
    pca = PCA(n_components=n_components)
    pca.fit(sample_matrix)
    var = pca.explained_variance_ratio_ * 100

    # Loadings: rows of components_ are eigenvectors over proteins.
    loadings = pca.components_  # (n_components, n_proteins)
    x = loadings[pc1]
    y = loadings[pc2]
    norms = np.sqrt(x**2 + y**2)
    top_idx = np.argsort(-norms)[:top_n]

    if annotation_column in data.columns:
        labels_all = data.loc[abundance.index, annotation_column].astype(str).tolist()
    else:
        logger.info("annotation_column %r not in data; using numeric labels.", annotation_column)
        labels_all = [str(i) for i in abundance.index]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=10, alpha=0.3, color="#888888", linewidth=0)
    ax.scatter(x[top_idx], y[top_idx], s=40, alpha=0.9, color="#d62728", linewidth=0.5, edgecolor="white")

    for i in top_idx:
        ax.annotate(
            labels_all[i],
            xy=(x[i], y[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(f"PC{pc1 + 1} loading ({var[pc1]:.1f}% variance)")
    ax.set_ylabel(f"PC{pc2 + 1} loading ({var[pc2]:.1f}% variance)")
    ax.set_title(f"{title} (top {top_n} proteins labeled)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_umap(
    data: pd.DataFrame,
    sample_columns: List[str],
    sample_metadata: Dict[str, Dict],
    group_column: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    log_transform: bool = True,
    group_colors: Optional[Dict[str, str]] = None,
    random_state: int = 0,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "UMAP projection",
) -> plt.Figure:
    """UMAP projection of samples colored by a metadata group.

    Useful as a non-linear complement to ``plot_pca`` when the dominant
    structure is not captured by the first two principal components.
    Imports umap-learn lazily; install with ``pip install umap-learn``.

    Args:
        data: Wide protein DataFrame with annotation columns followed by
            sample columns.
        sample_columns: Sample column names to include.
        sample_metadata: Dict-of-dicts keyed by sample column name.
        group_column: Metadata field naming each sample's group label
            for coloring.
        n_neighbors: UMAP n_neighbors parameter (controls local-vs-global
            tradeoff). Default 15.
        min_dist: UMAP min_dist parameter (cluster compactness). Default 0.1.
        metric: Distance metric passed to umap.UMAP. Default "euclidean".
        log_transform: log2-transform abundance before UMAP. Pass False
            if the input is already log-transformed.
        group_colors: Optional dict mapping group label to color string.
            Auto-generated from the seaborn palette if None.
        random_state: Seed for reproducibility.
        figsize: Figure size tuple.
        title: Plot title.

    Returns:
        matplotlib Figure object.
    """
    try:
        import umap  # type: ignore
    except ImportError as e:  # pragma: no cover - environment-specific
        raise ImportError(
            "plot_umap requires the umap-learn package. "
            "Install with: pip install umap-learn"
        ) from e

    abundance = data[sample_columns].apply(pd.to_numeric, errors="coerce")
    if log_transform:
        with np.errstate(invalid="ignore", divide="ignore"):
            abundance = np.log2(abundance.where(abundance > 0))
    abundance = abundance.dropna(axis=0, how="any")
    if abundance.empty:
        raise ValueError("No proteins remain after dropping rows with NaN values.")

    sample_matrix = abundance.to_numpy(dtype=float).T
    n_samples = sample_matrix.shape[0]
    if n_samples < 4:
        raise ValueError(f"UMAP needs >=4 samples; got {n_samples}.")

    # Resolve group labels in the same order as keep_cols
    labels = []
    for col in sample_columns:
        meta = sample_metadata.get(col, {})
        labels.append(str(meta.get(group_column, "Unknown")))

    effective_n_neighbors = min(n_neighbors, max(2, n_samples - 1))
    reducer = umap.UMAP(
        n_neighbors=effective_n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = reducer.fit_transform(sample_matrix)

    if group_colors is None:
        unique = sorted(set(labels))
        palette = sns.color_palette("tab10", n_colors=len(unique))
        group_colors = {g: palette[i] for i, g in enumerate(unique)}

    fig, ax = plt.subplots(figsize=figsize)
    for g in sorted(set(labels)):
        mask = np.array([lbl == g for lbl in labels])
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=80,
            alpha=0.85,
            color=group_colors.get(g, "#999999"),
            edgecolors="white",
            linewidths=0.5,
            label=f"{g} (n={int(mask.sum())})",
        )

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
