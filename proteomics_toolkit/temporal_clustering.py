"""
Temporal Clustering and Trend Analysis Module

This module provides tools for analyzing temporal protein expression patterns
in longitudinal proteomics experiments, including:

- K-means clustering with automatic optimal cluster detection (silhouette scores)
- Pattern classification (sustained increase/decrease, transient patterns)
- Integration with differential expression results from statistical_analysis module
- Visualization (heatmaps, parallel coordinate plots) - uses visualization module
- Gene set enrichment analysis via Enrichr API - uses enrichment module

Statistical Methodology:
------------------------
The statistical testing for differential expression is performed PRIOR to calling
this module, typically using the statistical_analysis module which performs:
  - Limma (linear models for microarrays) for repeated measures designs
  - Mixed-effects models for longitudinal data
  - T-tests for simple two-group comparisons

This module takes those statistical results (stats_df) and:
  1. Clusters proteins by their temporal trajectory using K-means
  2. Automatically detects optimal cluster count using silhouette scores
  3. Names clusters based on their expression pattern (e.g., "Sustained Increase")
  4. Filters to significant proteins (P < 0.05 by default)
  5. Runs pathway enrichment on each cluster via Enrichr API

Note: Visualization and enrichment functions are now also available in
the dedicated visualization and enrichment modules for broader use cases.

Author: MacCoss Lab
Version: 1.2.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import seaborn as sns  # noqa: F401 - used in heatmap functions
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings
import requests
import json
import time

# Import from our reusable modules - these provide general-purpose versions
# of enrichment and visualization functions that can be used elsewhere
from . import enrichment as enrich_module

# Re-export key classes for backward compatibility
EnrichmentConfig = enrich_module.EnrichmentConfig


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TemporalClusteringConfig:
    """Configuration for temporal clustering analysis."""
    
    # Clustering settings
    n_clusters: int = 4
    auto_detect_clusters: bool = True  # Automatically determine optimal clusters
    min_clusters: int = 2  # Minimum clusters to test for auto-detection
    max_clusters: int = 8  # Maximum clusters to test for auto-detection
    clustering_method: str = 'kmeans'  # 'kmeans' or 'hierarchical'
    random_seed: int = 42
    
    # Experimental design
    subject_column: str = 'BRI Subject ID'  # Column identifying subjects for within-subject normalization
    
    # Significance thresholds
    p_value_threshold: float = 0.05
    use_adjusted_pvalue: bool = False
    min_fold_change: float = 0.0
    
    # Pattern classification thresholds
    pattern_change_threshold: float = 0.3  # Z-score threshold for pattern detection
    transient_threshold: float = 0.5  # Threshold for transient pattern detection
    
    # Enrichment settings
    enrichr_libraries: List[str] = field(default_factory=lambda: [
        'GO_Biological_Process_2023',
        'GO_Molecular_Function_2023', 
        'KEGG_2021_Human',
        'Reactome_2022',
        'WikiPathway_2023_Human'
    ])
    enrichment_pvalue_cutoff: float = 0.05
    enrichment_top_n: int = 20
    min_genes_for_enrichment: int = 5
    
    # Visualization settings
    figsize: Tuple[int, int] = (14, 8)  # Reduced height
    heatmap_figsize: Tuple[int, int] = (16, 10)  # Separate heatmap size
    linewidth: float = 2.0
    alpha: float = 0.4
    cmap: str = 'RdBu_r'
    max_proteins_per_cluster_heatmap: int = 30  # Limit proteins shown per cluster
    
    # Cluster colors
    cluster_colors: Dict[int, str] = field(default_factory=lambda: {
        0: '#e41a1c',  # Red
        1: '#377eb8',  # Blue
        2: '#4daf4a',  # Green
        3: '#984ea3',  # Purple
        4: '#ff7f00',  # Orange
        5: '#ffff33',  # Yellow
        6: '#a65628',  # Brown
        7: '#f781bf',  # Pink
        8: '#66c2a5',  # Teal
        9: '#8da0cb',  # Light blue
    })


# =============================================================================
# TEMPORAL DATA CALCULATION
# =============================================================================

def calculate_temporal_means(
    data_df: pd.DataFrame,
    metadata_dict: Dict[str, Dict],
    week_column: str = 'Week',
    subject_column: str = 'BRI Subject ID'
) -> Tuple[pd.DataFrame, List]:
    """
    Calculate mean protein abundance at each timepoint across subjects.
    
    Uses within-subject Z-score normalization to leverage the paired design:
    1. For each protein, Z-score normalize within each subject
    2. Average the Z-scored values across subjects for each timepoint
    
    This approach ensures that between-subject variation in baseline levels
    doesn't dominate the clustering, and each subject serves as their own control.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Protein abundance data with Protein and Gene columns
    metadata_dict : dict
        Dictionary mapping sample names to metadata (including week and subject info)
    week_column : str
        Name of the metadata field containing timepoint information
    subject_column : str
        Name of the metadata field containing subject IDs
        
    Returns
    -------
    temporal_means : pd.DataFrame
        DataFrame with proteins as rows and weeks as columns (Z-scored within subjects, then averaged)
    unique_weeks : list
        Sorted list of unique timepoints
    """
    # Get samples, their weeks, and their subjects
    sample_weeks = {s: m.get(week_column) for s, m in metadata_dict.items()}
    sample_subjects = {s: m.get(subject_column) for s, m in metadata_dict.items()}
    unique_weeks = sorted([w for w in set(sample_weeks.values()) if w is not None])
    unique_subjects = set([s for s in sample_subjects.values() if s is not None])
    
    # Get sample columns that exist in data
    sample_cols = [c for c in data_df.columns if c in metadata_dict]
    
    # Extract protein data matrix
    protein_data = data_df[sample_cols].values.astype(float)
    n_proteins = protein_data.shape[0]
    n_weeks = len(unique_weeks)
    
    # Initialize array for Z-scored means
    zscore_means = np.zeros((n_proteins, n_weeks))
    zscore_counts = np.zeros((n_proteins, n_weeks))  # Track how many subjects contributed
    
    # Process each subject: Z-score within subject, then accumulate
    for subject in unique_subjects:
        # Get samples for this subject
        subject_samples = [s for s in sample_cols 
                          if sample_subjects.get(s) == subject]
        
        if len(subject_samples) < 2:
            # Need at least 2 timepoints to Z-score within subject
            continue
        
        # Get this subject's data (proteins x timepoints for this subject)
        subject_data = data_df[subject_samples].values.astype(float)
        
        # Z-score normalize within this subject (across their timepoints)
        subject_means = np.nanmean(subject_data, axis=1, keepdims=True)
        subject_stds = np.nanstd(subject_data, axis=1, keepdims=True)
        subject_stds[subject_stds == 0] = 1  # Avoid division by zero
        subject_zscore = (subject_data - subject_means) / subject_stds
        
        # Map this subject's Z-scored values to the appropriate week columns
        for i, sample in enumerate(subject_samples):
            week = sample_weeks.get(sample)
            if week is not None and week in unique_weeks:
                week_idx = unique_weeks.index(week)
                # Add Z-scored value (handling NaN)
                valid_mask = ~np.isnan(subject_zscore[:, i])
                zscore_means[valid_mask, week_idx] += subject_zscore[valid_mask, i]
                zscore_counts[valid_mask, week_idx] += 1
    
    # Calculate mean Z-score across subjects for each week
    zscore_counts[zscore_counts == 0] = 1  # Avoid division by zero
    zscore_means = zscore_means / zscore_counts
    
    # Build output DataFrame
    temporal_means = pd.DataFrame()
    temporal_means['Protein'] = data_df['Protein']
    temporal_means['Gene'] = data_df['Gene'] if 'Gene' in data_df.columns else data_df['Protein']
    
    for i, week in enumerate(unique_weeks):
        temporal_means[f'Week_{week}'] = zscore_means[:, i]
    
    return temporal_means, unique_weeks


def get_week_columns(temporal_df: pd.DataFrame) -> List[str]:
    """Extract week column names from temporal DataFrame."""
    return [c for c in temporal_df.columns if c.startswith('Week_')]


def determine_optimal_clusters(
    X_scaled: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    random_seed: int = 42,
    plot: bool = True,
    prefer_higher_k: bool = True,
    silhouette_tolerance: float = 0.05
) -> Tuple[int, Figure, Dict]:
    """
    Determine optimal number of clusters using elbow and silhouette methods.
    
    The selection logic:
    1. Find k with best silhouette score
    2. If prefer_higher_k=True, consider k values with silhouette within tolerance
       of the best, and pick the higher k (more biologically meaningful clusters)
    3. Consider elbow method as a secondary signal
    
    Parameters
    ----------
    X_scaled : np.ndarray
        Z-score normalized temporal matrix
    k_range : tuple
        Range of k values to test (min, max)
    random_seed : int
        Random seed for reproducibility
    plot : bool
        Whether to create diagnostic plot
    prefer_higher_k : bool
        If True, prefer higher k when silhouette scores are similar
    silhouette_tolerance : float
        Consider k values within this tolerance of the best silhouette
        
    Returns
    -------
    optimal_k : int
        Recommended number of clusters
    fig : Figure or None
        Diagnostic plot showing elbow and silhouette analysis
    scores : dict
        Dictionary with k values, inertias, and silhouette scores
    """
    from sklearn.metrics import silhouette_score
    
    k_values = list(range(k_range[0], k_range[1] + 1))
    inertias = []
    silhouette_scores = []
    
    for k in k_values:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            
            if k > 1:
                sil_score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
    
    # Find elbow using second derivative
    inertias_arr = np.array(inertias)
    diffs = np.diff(inertias_arr)
    diffs2 = np.diff(diffs)
    elbow_idx = np.argmax(diffs2) + 2 if len(diffs2) > 0 else 2
    elbow_k = k_values[elbow_idx] if elbow_idx < len(k_values) else k_values[-1]
    
    # Find best silhouette score
    sil_arr = np.array(silhouette_scores)
    best_sil = np.max(sil_arr)
    best_sil_idx = np.argmax(sil_arr)
    best_sil_k = k_values[best_sil_idx]
    
    # Smart selection: if prefer_higher_k, find highest k within tolerance of best
    if prefer_higher_k:
        # Find all k values with silhouette within tolerance of best
        acceptable_mask = sil_arr >= (best_sil - silhouette_tolerance)
        acceptable_indices = np.where(acceptable_mask)[0]
        # Pick the highest acceptable k
        optimal_idx = acceptable_indices[-1]  # Last one = highest k
        optimal_k = k_values[optimal_idx]
        
        # Also consider elbow: if elbow suggests more clusters and it's acceptable, use it
        if elbow_k > optimal_k:
            elbow_idx_in_list = k_values.index(elbow_k) if elbow_k in k_values else -1
            if elbow_idx_in_list >= 0 and acceptable_mask[elbow_idx_in_list]:
                optimal_k = elbow_k
    else:
        optimal_k = best_sil_k
    
    scores = {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'elbow_k': elbow_k,
        'best_silhouette_k': best_sil_k,
        'selected_k': optimal_k
    }
    
    fig = None
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        axes[0].plot(k_values, inertias, 'b-o', linewidth=2, markersize=10)
        axes[0].axvline(x=elbow_k, color='orange', linestyle='--', linewidth=2, label=f'Elbow: k={elbow_k}')
        axes[0].axvline(x=optimal_k, color='red', linestyle='-', linewidth=2, label=f'Selected: k={optimal_k}')
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-cluster SS)', fontsize=12)
        axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(k_values)
        
        # Silhouette plot with tolerance band
        axes[1].plot(k_values, silhouette_scores, 'g-o', linewidth=2, markersize=10)
        axes[1].axhline(y=best_sil - silhouette_tolerance, color='gray', linestyle=':', 
                       label=f'Tolerance band (-{silhouette_tolerance})')
        axes[1].axvline(x=best_sil_k, color='green', linestyle='--', linewidth=2, 
                       label=f'Best silhouette: k={best_sil_k} ({best_sil:.3f})')
        axes[1].axvline(x=optimal_k, color='red', linestyle='-', linewidth=2, 
                       label=f'Selected: k={optimal_k}')
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=9, loc='upper right')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(k_values)
        
        # Add annotation explaining the selection
        selection_reason = ""
        if optimal_k == best_sil_k:
            selection_reason = f"Selected k={optimal_k} (best silhouette score)"
        elif optimal_k == elbow_k:
            selection_reason = f"Selected k={optimal_k} (elbow method, within silhouette tolerance)"
        else:
            selection_reason = f"Selected k={optimal_k} (highest k within silhouette tolerance of {silhouette_tolerance})"
        
        plt.suptitle(f'Cluster Selection Analysis\n{selection_reason}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    return optimal_k, fig, scores


def plot_silhouette_analysis(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    cluster_names: Dict[int, str],
    title: str = 'Silhouette Analysis of Temporal Clusters'
) -> Figure:
    """
    Create a detailed silhouette plot showing the silhouette coefficient for each sample.
    
    This visualization shows:
    - The silhouette value for each protein within each cluster
    - The average silhouette score for each cluster
    - The overall average silhouette score
    
    Parameters
    ----------
    X_scaled : np.ndarray
        Z-score normalized temporal matrix (proteins x timepoints)
    labels : np.ndarray
        Cluster labels for each protein
    cluster_names : dict
        Mapping from cluster ID to descriptive name
    title : str
        Plot title
        
    Returns
    -------
    fig : Figure
        The silhouette plot figure
    """
    from sklearn.metrics import silhouette_score, silhouette_samples
    import matplotlib.cm as cm
    
    n_clusters = len(np.unique(labels))
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X_scaled, labels)
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Silhouette plot for each sample
    y_lower = 10
    
    # Use a colormap
    colors = cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        # Get silhouette scores for samples in this cluster
        cluster_silhouette_values = sample_silhouette_values[labels == i]
        cluster_silhouette_values.sort()
        
        cluster_size = cluster_silhouette_values.shape[0]
        y_upper = y_lower + cluster_size
        
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, cluster_silhouette_values,
            facecolor=colors[i], edgecolor=colors[i], alpha=0.7
        )
        
        # Label the cluster
        cluster_label = cluster_names.get(i, f'Cluster {i}')
        ax1.text(-0.05, y_lower + 0.5 * cluster_size, cluster_label,
                fontsize=10, fontweight='bold', va='center')
        
        y_lower = y_upper + 10
    
    # Add vertical line for average silhouette score
    ax1.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
                label=f'Average: {silhouette_avg:.3f}')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    ax1.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax1.set_ylabel('Proteins (sorted by cluster)', fontsize=12)
    ax1.set_title('Silhouette Plot', fontsize=14, fontweight='bold')
    ax1.set_xlim([-0.3, 1])
    ax1.set_yticks([])
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Right plot: Bar chart of average silhouette per cluster
    cluster_avgs = []
    cluster_labels_list = []
    cluster_sizes = []
    
    for i in range(n_clusters):
        cluster_mask = labels == i
        cluster_avg = np.mean(sample_silhouette_values[cluster_mask])
        cluster_avgs.append(cluster_avg)
        cluster_labels_list.append(cluster_names.get(i, f'Cluster {i}'))
        cluster_sizes.append(np.sum(cluster_mask))
    
    ax2.barh(range(n_clusters), cluster_avgs, color=colors, edgecolor='black', alpha=0.8)
    
    # Add cluster sizes as labels
    for i, (avg, size) in enumerate(zip(cluster_avgs, cluster_sizes)):
        ax2.text(avg + 0.02, i, f'n={size}', va='center', fontsize=10)
    
    ax2.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
                label=f'Overall avg: {silhouette_avg:.3f}')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    ax2.set_yticks(range(n_clusters))
    ax2.set_yticklabels(cluster_labels_list, fontsize=10)
    ax2.set_xlabel('Average Silhouette Score', fontsize=12)
    ax2.set_title('Average Score by Cluster', fontsize=14, fontweight='bold')
    ax2.set_xlim([-0.3, 1])
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add interpretation text
    interpretation = ""
    if silhouette_avg > 0.5:
        interpretation = "Strong cluster structure"
    elif silhouette_avg > 0.25:
        interpretation = "Reasonable cluster structure"
    else:
        interpretation = "Weak cluster structure - clusters may overlap"
    
    plt.suptitle(f'{title}\n{interpretation} (avg silhouette = {silhouette_avg:.3f})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# =============================================================================
# CLUSTERING
# =============================================================================

def cluster_temporal_trends(
    temporal_df: pd.DataFrame,
    week_columns: List[str],
    config: Optional[TemporalClusteringConfig] = None
) -> Tuple[np.ndarray, np.ndarray, Any, Optional[Any]]:
    """
    Cluster proteins by their temporal trajectory patterns using K-means or hierarchical clustering.
    
    Parameters
    ----------
    temporal_df : pd.DataFrame
        DataFrame with temporal means (proteins x weeks)
    week_columns : list
        List of column names containing temporal data
    config : TemporalClusteringConfig, optional
        Configuration object (use config.auto_detect_clusters to enable auto-detection)
        
    Returns
    -------
    labels : np.ndarray
        Cluster labels for each protein
    X_scaled : np.ndarray
        Z-score normalized temporal matrix
    model : KMeans or linkage matrix
        The fitted clustering model
    cluster_selection_fig : matplotlib.Figure or None
        Figure showing cluster selection analysis (if auto_detect_clusters=True)
    """
    if config is None:
        config = TemporalClusteringConfig()
    
    # Get the temporal data matrix
    X = temporal_df[week_columns].values.astype(float)
    
    # Handle missing values - use float conversion for nanmean
    global_mean = float(np.nanmean(X))
    X = np.nan_to_num(X, nan=global_mean)
    
    # Z-score normalize each protein's trajectory (focus on shape, not magnitude)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.T).T
    
    # Handle any remaining NaN/inf from scaling
    X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
    
    # Determine optimal number of clusters
    cluster_selection_fig = None
    n_clusters = config.n_clusters
    
    if config.auto_detect_clusters:
        print("   Determining optimal number of clusters...", flush=True)
        n_clusters, cluster_selection_fig, _ = determine_optimal_clusters(
            X_scaled, 
            k_range=(config.min_clusters, config.max_clusters),
            random_seed=config.random_seed,
            plot=True
        )
        print(f"   Optimal clusters: {n_clusters}", flush=True)
    
    if config.clustering_method == 'kmeans':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = KMeans(
                n_clusters=n_clusters, 
                random_state=config.random_seed, 
                n_init=10
            )
            labels = model.fit_predict(X_scaled)
        return labels, X_scaled, model, cluster_selection_fig
    else:
        # Hierarchical clustering
        Z = linkage(X_scaled, method='ward')
        labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
        return labels, X_scaled, Z, cluster_selection_fig


def name_clusters_by_pattern(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    week_columns: List[str]
) -> Dict[int, str]:
    """
    Name clusters with simple numbered labels.
    
    Using simple numbered names avoids confusion when multiple clusters
    have similar trajectory patterns (e.g., multiple "Sustained Decrease" clusters).
    
    Returns
    -------
    cluster_names : dict
        Mapping of cluster ID to simple name like "Cluster 1", "Cluster 2", etc.
    """
    cluster_names = {}
    n_clusters = len(np.unique(labels))
    
    for cluster_id in range(n_clusters):
        # Use 1-based numbering for user-friendly display
        cluster_names[cluster_id] = f"Cluster {cluster_id + 1}"
    
    return cluster_names


# =============================================================================
# PATTERN CLASSIFICATION
# =============================================================================

def classify_trend_pattern(
    trajectory: np.ndarray,
    config: Optional[TemporalClusteringConfig] = None
) -> Tuple[str, float]:
    """
    Classify a protein's temporal trajectory into one of several patterns.
    
    Parameters
    ----------
    trajectory : np.ndarray
        Array of z-scored values across timepoints
    config : TemporalClusteringConfig, optional
        Configuration object
        
    Returns
    -------
    pattern : str
        Pattern name (e.g., "Up & Stay Up", "Down & Stay Down", etc.)
    confidence : float
        Confidence score based on magnitude of change
    """
    if config is None:
        config = TemporalClusteringConfig()
    
    threshold = config.pattern_change_threshold
    transient_threshold = config.transient_threshold
    
    n = len(trajectory)
    if n < 2:
        return "Stable", 0.0
    
    start, end = trajectory[0], trajectory[-1]
    
    # Note: mid value calculated but not currently used in pattern classification
    # Kept for potential future use in more complex pattern detection
    _ = trajectory[n // 2] if n >= 3 else (start + end) / 2  # noqa: F841
    
    net_change = end - start
    peak_idx = np.argmax(trajectory)
    trough_idx = np.argmin(trajectory)
    peak_val = trajectory[peak_idx]
    trough_val = trajectory[trough_idx]
    
    # Determine dominant pattern
    if net_change > threshold:  # Net increase
        if peak_idx >= n - 2:  # Peak at or near end
            return "Up & Stay Up", abs(net_change)
        else:
            return "Up then Down", abs(peak_val - start)
    elif net_change < -threshold:  # Net decrease
        if trough_idx >= n - 2:  # Trough at or near end
            return "Down & Stay Down", abs(net_change)
        else:
            return "Down then Up", abs(start - trough_val)
    else:  # Minimal net change
        if peak_val - start > transient_threshold:
            return "Up then Down", peak_val - start
        elif start - trough_val > transient_threshold:
            return "Down then Up", start - trough_val
        else:
            return "Stable", 0.1
    
    return "Stable", 0.1


def classify_all_patterns(
    temporal_df: pd.DataFrame,
    week_columns: List[str],
    config: Optional[TemporalClusteringConfig] = None
) -> Tuple[List[str], List[float]]:
    """
    Classify temporal patterns for all proteins in a DataFrame.
    
    Returns
    -------
    patterns : list
        Pattern name for each protein
    confidence_scores : list
        Confidence score for each protein
    """
    if config is None:
        config = TemporalClusteringConfig()
    
    X = temporal_df[week_columns].values.astype(float)
    
    # Z-score normalize
    X_means = np.nanmean(X, axis=1, keepdims=True)
    X_stds = np.nanstd(X, axis=1, keepdims=True)
    X_stds[X_stds == 0] = 1
    X_z = (X - X_means) / X_stds
    
    patterns = []
    confidence_scores = []
    
    for i in range(len(X_z)):
        trajectory = X_z[i]
        if np.any(np.isnan(trajectory)):
            patterns.append("Unknown")
            confidence_scores.append(0.0)
        else:
            pattern, confidence = classify_trend_pattern(trajectory, config)
            patterns.append(pattern)
            confidence_scores.append(confidence)
    
    return patterns, confidence_scores


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def merge_with_statistics(
    temporal_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_names: Dict[int, str]
) -> pd.DataFrame:
    """
    Merge temporal data with statistical results and cluster assignments.
    
    Parameters
    ----------
    temporal_df : pd.DataFrame
        Temporal means DataFrame
    stats_df : pd.DataFrame
        Statistical results with P.Value, logFC, etc.
    cluster_labels : np.ndarray
        Cluster assignments from K-means clustering
    cluster_names : dict
        Mapping of cluster ID to descriptive name (derived from cluster centroids)
        
    Returns
    -------
    merged_df : pd.DataFrame
        Combined DataFrame with all information
    """
    result = temporal_df.copy()
    
    # Add cluster information
    result['Cluster'] = cluster_labels
    result['Cluster_Name'] = [cluster_names.get(c, f"Cluster {c}") for c in cluster_labels]
    
    # Merge with statistical results using Protein as key
    if stats_df is not None and len(stats_df) > 0:
        stats_cols = ['Protein']
        for col in ['Gene', 'logFC', 'P.Value', 'adj.P.Val']:
            if col in stats_df.columns:
                stats_cols.append(col)
        
        stats_subset = stats_df[stats_cols].copy()
        
        # Rename Gene to avoid conflicts
        if 'Gene' in stats_subset.columns:
            stats_subset = stats_subset.rename(columns={'Gene': 'Gene_Stats'})
        
        result = result.merge(stats_subset, on='Protein', how='left')
        
        # Use parsed Gene names if available
        if 'Gene_Stats' in result.columns:
            result['Gene'] = result['Gene_Stats'].fillna(result['Gene'])
            result = result.drop(columns=['Gene_Stats'])
    
    return result


def filter_significant_proteins(
    merged_df: pd.DataFrame,
    config: Optional[TemporalClusteringConfig] = None
) -> pd.DataFrame:
    """
    Filter to statistically significant proteins.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged DataFrame with P.Value column
    config : TemporalClusteringConfig, optional
        Configuration with p_value_threshold
        
    Returns
    -------
    sig_df : pd.DataFrame
        Filtered DataFrame with only significant proteins
    """
    if config is None:
        config = TemporalClusteringConfig()
    
    p_col = 'adj.P.Val' if config.use_adjusted_pvalue and 'adj.P.Val' in merged_df.columns else 'P.Value'
    
    if p_col not in merged_df.columns:
        print(f"Warning: {p_col} column not found. Returning all proteins.")
        return merged_df
    
    sig_df = merged_df[merged_df[p_col] < config.p_value_threshold].copy()
    
    if config.min_fold_change > 0 and 'logFC' in merged_df.columns:
        sig_df = sig_df[sig_df['logFC'].abs() >= config.min_fold_change]
    
    return sig_df


# =============================================================================
# ENRICHR API FUNCTIONS
# =============================================================================

def query_enrichr(
    gene_list: List[str],
    config: Optional[TemporalClusteringConfig] = None
) -> Dict[str, List]:
    """
    Query Enrichr API for gene set enrichment analysis.
    
    Parameters
    ----------
    gene_list : list
        List of gene symbols
    config : TemporalClusteringConfig, optional
        Configuration with enrichr_libraries
        
    Returns
    -------
    results : dict
        Results from each library
    """
    if config is None:
        config = TemporalClusteringConfig()
    
    # Clean gene list
    clean_genes = [g for g in gene_list if pd.notna(g) and str(g).strip() != '']
    
    if len(clean_genes) < config.min_genes_for_enrichment:
        print(f"  Warning: Only {len(clean_genes)} genes provided, need at least {config.min_genes_for_enrichment}")
        return {}
    
    ENRICHR_URL = 'https://maayanlab.cloud/Enrichr'
    
    genes_str = '\n'.join(clean_genes)
    payload = {
        'list': (None, genes_str),
        'description': (None, 'Temporal Clustering Analysis')
    }
    
    try:
        response = requests.post(f'{ENRICHR_URL}/addList', files=payload, timeout=30)
        if not response.ok:
            print(f"  Error submitting gene list: {response.status_code}")
            return {}
        
        data = json.loads(response.text)
        user_list_id = data['userListId']
        
    except Exception as e:
        print(f"  Error connecting to Enrichr: {e}")
        return {}
    
    # Query each library
    results = {}
    for library in config.enrichr_libraries:
        try:
            time.sleep(0.5)  # Rate limiting
            response = requests.get(
                f'{ENRICHR_URL}/enrich',
                params={'userListId': user_list_id, 'backgroundType': library},
                timeout=30
            )
            
            if response.ok:
                enrichment_results = json.loads(response.text)
                if library in enrichment_results:
                    results[library] = enrichment_results[library]
            
        except Exception as e:
            print(f"  Error querying {library}: {e}")
            continue
    
    return results


def parse_enrichr_results(
    results: Dict[str, List],
    config: Optional[TemporalClusteringConfig] = None
) -> pd.DataFrame:
    """
    Parse Enrichr results into a DataFrame.
    
    Enrichr result format per term:
    [0] Rank, [1] Term name, [2] P-value, [3] Z-score,
    [4] Combined score, [5] Overlapping genes, [6] Adjusted p-value
    """
    if config is None:
        config = TemporalClusteringConfig()
    
    parsed_results = []
    
    for library, terms in results.items():
        for term_data in terms[:config.enrichment_top_n]:
            if len(term_data) >= 7:
                pval = term_data[2]
                adj_pval = term_data[6]
                
                if pval <= config.enrichment_pvalue_cutoff:
                    parsed_results.append({
                        'Library': library,
                        'Term': term_data[1],
                        'P_Value': pval,
                        'Adj_P_Value': adj_pval,
                        'Z_Score': term_data[3],
                        'Combined_Score': term_data[4],
                        'Genes': ';'.join(term_data[5]) if isinstance(term_data[5], list) else term_data[5],
                        'N_Genes': len(term_data[5]) if isinstance(term_data[5], list) else 1
                    })
    
    if parsed_results:
        return pd.DataFrame(parsed_results).sort_values('Combined_Score', ascending=False)
    else:
        return pd.DataFrame()


def run_enrichment_by_cluster(
    merged_df: pd.DataFrame,
    cluster_column: str = 'Cluster_Name',
    gene_column: str = 'Gene',
    config: Optional[TemporalClusteringConfig] = None
) -> Dict[str, pd.DataFrame]:
    """
    Run enrichment analysis for each cluster.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame with cluster assignments and gene names
    cluster_column : str
        Column name containing cluster assignments
    gene_column : str
        Column name containing gene symbols
    config : TemporalClusteringConfig, optional
        Configuration object
        
    Returns
    -------
    enrichment_results : dict
        Dictionary mapping cluster name to enrichment DataFrame
    """
    if config is None:
        config = TemporalClusteringConfig()
    
    enrichment_results = {}
    
    for cluster_name in merged_df[cluster_column].unique():
        subset = merged_df[merged_df[cluster_column] == cluster_name]
        gene_list = subset[gene_column].dropna().tolist()
        
        print(f"\n{cluster_name}: {len(gene_list)} genes", flush=True)
        
        if len(gene_list) >= config.min_genes_for_enrichment:
            raw_results = query_enrichr(gene_list, config)
            if raw_results:
                enrichment_df = parse_enrichr_results(raw_results, config)
                enrichment_results[cluster_name] = enrichment_df
                if not enrichment_df.empty:
                    print(f"  Found {len(enrichment_df)} enriched terms", flush=True)
                else:
                    enrichment_results[cluster_name] = pd.DataFrame()
                    print("  No significant enrichment", flush=True)
            else:
                enrichment_results[cluster_name] = pd.DataFrame()
        else:
            enrichment_results[cluster_name] = pd.DataFrame()
            print(f"  Skipping - need at least {config.min_genes_for_enrichment} genes", flush=True)
    
    return enrichment_results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_cluster_heatmap(
    merged_df: pd.DataFrame,
    week_columns: List[str],
    cluster_column: str = 'Cluster_Name',
    title: str = 'Temporal Protein Expression by Cluster',
    config: Optional[TemporalClusteringConfig] = None,
    show_genes: bool = True,
    max_proteins_per_cluster: int = 50
) -> Figure:
    """
    Create a heatmap of protein expression organized by cluster.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame with temporal data and cluster assignments
    week_columns : list
        Columns containing temporal data
    cluster_column : str
        Column containing cluster assignments
    title : str
        Plot title
    config : TemporalClusteringConfig, optional
        Configuration object
    show_genes : bool
        Whether to show gene names on y-axis
    max_proteins_per_cluster : int
        Maximum proteins to show per cluster
        
    Returns
    -------
    fig : matplotlib.Figure
        The heatmap figure
    """
    if config is None:
        config = TemporalClusteringConfig()
    
    # Get data matrix and z-score
    X = merged_df[week_columns].values.astype(float)
    X_means = np.nanmean(X, axis=1, keepdims=True)
    X_stds = np.nanstd(X, axis=1, keepdims=True)
    X_stds[X_stds == 0] = 1
    X_z = (X - X_means) / X_stds
    
    # Sort by cluster, then by p-value (most significant first)
    sorted_df = merged_df.copy()
    sorted_df['_z_data'] = list(X_z)
    
    # Determine sort column - prefer p-value, fall back to cluster only
    sort_cols = [cluster_column]
    ascending = [True]
    for pval_col in ['P.Value', 'P_Value', 'adj.P.Val']:
        if pval_col in sorted_df.columns:
            sort_cols.append(pval_col)
            ascending.append(True)  # Lower p-value = more significant = first
            break
    
    sorted_df = sorted_df.sort_values(sort_cols, ascending=ascending)
    
    # Limit proteins per cluster
    limited_dfs = []
    for cluster in sorted_df[cluster_column].unique():
        cluster_subset = sorted_df[sorted_df[cluster_column] == cluster].head(max_proteins_per_cluster)
        limited_dfs.append(cluster_subset)
    sorted_df = pd.concat(limited_dfs)
    
    # Create heatmap data
    heatmap_data = np.vstack(sorted_df['_z_data'].values)
    
    # Week labels
    week_labels = [c.replace('Week_', 'W') for c in week_columns]
    
    # Gene labels
    gene_labels = sorted_df['Gene'].tolist() if show_genes else None
    
    # Create figure with better layout for colorbar
    # Limit height to be more readable (max ~15 inches)
    fig_height = min(12, max(6, len(sorted_df) * 0.12))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    
    # Leave space on right for cluster labels and colorbar
    plt.subplots_adjust(right=0.75, left=0.15)
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, aspect='auto', cmap=config.cmap, vmin=-2, vmax=2)
    
    # Labels
    ax.set_xticks(range(len(week_labels)))
    ax.set_xticklabels(week_labels, fontsize=11, fontweight='bold')
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    
    if show_genes and len(sorted_df) <= 50:
        ax.set_yticks(range(len(gene_labels)))
        ax.set_yticklabels(gene_labels, fontsize=8)
    else:
        ax.set_ylabel(f'Proteins (n={len(sorted_df)})', fontsize=12)
        ax.set_yticks([])
    
    # Add cluster separators
    cluster_bounds = []
    current_idx = 0
    for cluster in sorted_df[cluster_column].unique():
        n_in_cluster = (sorted_df[cluster_column] == cluster).sum()
        cluster_bounds.append((current_idx, cluster, n_in_cluster))
        if current_idx > 0:
            ax.axhline(y=current_idx - 0.5, color='white', linewidth=2)
        current_idx += n_in_cluster
    
    # Add cluster labels on right side  
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    cluster_mids = [start + n/2 for start, _, n in cluster_bounds]
    cluster_labels_text = [f"{name}\n(n={n})" for _, name, n in cluster_bounds]
    ax2.set_yticks(cluster_mids)
    ax2.set_yticklabels(cluster_labels_text, fontsize=10, fontweight='bold')
    
    # Colorbar - position it further right to avoid overlap with cluster labels
    cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Z-score', fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    return fig


def plot_cluster_parallel_coordinates(
    merged_df: pd.DataFrame,
    week_columns: List[str],
    cluster_column: str = 'Cluster_Name',
    title: str = 'Temporal Patterns by Cluster',
    config: Optional[TemporalClusteringConfig] = None
) -> Figure:
    """
    Create parallel coordinate plots for each cluster.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame with temporal data and cluster assignments
    week_columns : list
        Columns containing temporal data
    cluster_column : str
        Column containing cluster assignments
    title : str
        Plot title
    config : TemporalClusteringConfig, optional
        Configuration object
        
    Returns
    -------
    fig : matplotlib.Figure
        The parallel coordinate figure
    """
    if config is None:
        config = TemporalClusteringConfig()
    
    # Get weeks as numeric
    weeks = [float(c.replace('Week_', '').replace('Week', '')) for c in week_columns]
    
    # Get data and z-score
    X = merged_df[week_columns].values.astype(float)
    X_means = np.nanmean(X, axis=1, keepdims=True)
    X_stds = np.nanstd(X, axis=1, keepdims=True)
    X_stds[X_stds == 0] = 1
    X_z = (X - X_means) / X_stds
    X_z = np.nan_to_num(X_z, nan=0, posinf=0, neginf=0)
    
    # Get unique clusters
    clusters = merged_df[cluster_column].unique()
    n_clusters = len(clusters)
    
    # Use single column layout for taller individual plots
    n_cols = 2  # Two columns for side-by-side comparison
    n_rows = (n_clusters + n_cols - 1) // n_cols
    
    # Make each subplot taller (4 inches height per row)
    fig_height = max(4 * n_rows, 8)
    fig_width = 14
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_clusters == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Use consistent dark blue color for all clusters
    line_color = '#1f4e79'  # Dark blue
    mean_color = 'black'
    
    for idx, cluster_name in enumerate(clusters):
        ax = axes[idx]
        
        mask = merged_df[cluster_column] == cluster_name
        n_proteins = mask.sum()
        
        if n_proteins > 0:
            # Plot individual trajectories in dark blue
            for i, is_in_cluster in enumerate(mask):
                if is_in_cluster:
                    ax.plot(weeks, X_z[i], color=line_color, alpha=config.alpha, linewidth=config.linewidth)
            
            # Plot mean trajectory with thick black line and dashed overlay
            cluster_mean = X_z[mask].mean(axis=0)
            ax.plot(weeks, cluster_mean, color=mean_color, linewidth=4, label=f'Mean (n={n_proteins})')
            ax.plot(weeks, cluster_mean, color='orange', linewidth=2.5, linestyle='--')
        
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Z-scored Abundance', fontsize=12)
        ax.set_title(f'{cluster_name}\n(n={n_proteins} proteins)', fontsize=12, fontweight='bold')
        ax.set_xticks(weeks)
        ax.set_ylim(-2.5, 2.5)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    # Hide unused subplots
    for idx in range(n_clusters, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def plot_enrichment_barplot(
    enrichment_df: pd.DataFrame,
    title: str,
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[Figure]:
    """
    Create a horizontal bar plot of enrichment results.
    
    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Enrichment results from parse_enrichr_results()
    title : str
        Plot title
    top_n : int
        Number of top terms to show
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.Figure or None
        The bar plot figure
    """
    if enrichment_df.empty:
        print(f"  No significant enrichment results for: {title}")
        return None
    
    # Take top N by combined score
    plot_df = enrichment_df.head(top_n).copy()
    plot_df = plot_df.sort_values('Combined_Score', ascending=True)
    
    # Color by library
    library_colors = {
        'GO_Biological_Process_2023': '#1f77b4',
        'GO_Molecular_Function_2023': '#2ca02c',
        'KEGG_2021_Human': '#d62728',
        'Reactome_2022': '#9467bd',
        'WikiPathway_2023_Human': '#ff7f0e'
    }
    
    colors = [library_colors.get(lib, 'gray') for lib in plot_df['Library']]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Truncate long term names
    term_labels = [t[:55] + '...' if len(t) > 55 else t for t in plot_df['Term']]
    
    ax.barh(range(len(plot_df)), plot_df['Combined_Score'], color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(term_labels, fontsize=9)
    ax.set_xlabel('Combined Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add gene count annotations
    for i, (score, n_genes) in enumerate(zip(plot_df['Combined_Score'], plot_df['N_Genes'])):
        ax.text(score + 0.5, i, f'({n_genes})', va='center', fontsize=8, color='gray')
    
    # Legend for libraries
    legend_elements = [
        Rectangle((0,0), 1, 1, facecolor=c, alpha=0.8, 
                      label=lib.replace('_', ' ').replace('2023', '').replace('2022', '').replace('2021', '').strip()) 
        for lib, c in library_colors.items() if lib in plot_df['Library'].values
    ]
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_enrichment_comparison(
    enrichment_dict: Dict[str, pd.DataFrame],
    title: str,
    top_n_per_group: int = 5,
    figsize: Tuple[int, int] = (14, 10)
) -> Optional[Figure]:
    """
    Create a dot plot comparing enrichment across multiple clusters.
    
    Parameters
    ----------
    enrichment_dict : dict
        Dictionary mapping cluster name to enrichment DataFrame
    title : str
        Plot title
    top_n_per_group : int
        Number of top terms per cluster
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib.Figure or None
        The comparison plot
    """
    # Collect all terms
    all_terms = []
    for cluster_name, enrichment_df in enrichment_dict.items():
        if not enrichment_df.empty:
            top_terms = enrichment_df.head(top_n_per_group).copy()
            top_terms['Cluster'] = cluster_name
            all_terms.append(top_terms)
    
    if not all_terms:
        print(f"  No terms to plot for: {title}")
        return None
    
    combined_df = pd.concat(all_terms, ignore_index=True)
    
    # Get unique terms and clusters
    unique_terms = combined_df.groupby('Term')['Combined_Score'].max().sort_values(ascending=False).head(20).index.tolist()
    clusters = list(enrichment_dict.keys())
    active_clusters = [c for c in clusters if not enrichment_dict[c].empty]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, term in enumerate(unique_terms):
        for j, cluster in enumerate(active_clusters):
            df = enrichment_dict[cluster]
            term_row = df[df['Term'] == term]
            
            if not term_row.empty:
                pval = term_row['P_Value'].values[0]
                n_genes = term_row['N_Genes'].values[0]
                
                size = min(n_genes * 30, 400)
                color_val = min(-np.log10(pval + 1e-10), 10)
                
                ax.scatter(j, i, s=size, c=[color_val], cmap='Reds', 
                          vmin=0, vmax=10, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    ax.set_xticks(range(len(active_clusters)))
    ax.set_xticklabels(active_clusters, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(unique_terms)))
    ax.set_yticklabels([t[:50] + '...' if len(t) > 50 else t for t in unique_terms], fontsize=9)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Enriched Term', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=10))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('-log10(P-value)', fontsize=10)
    
    ax.set_xlim(-0.5, len(active_clusters) - 0.5)
    ax.set_ylim(-0.5, len(unique_terms) - 0.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


# =============================================================================
# HIGH-LEVEL ANALYSIS FUNCTION
# =============================================================================

def run_temporal_analysis(
    data_df: pd.DataFrame,
    metadata_dict: Dict[str, Dict],
    stats_df: pd.DataFrame,
    treatment_name: str,
    week_column: str = 'Week',
    config: Optional[TemporalClusteringConfig] = None,
    output_prefix: Optional[str] = None,
    run_enrichment: bool = True
) -> Dict[str, Any]:
    """
    Run complete temporal clustering analysis pipeline.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        Protein abundance data
    metadata_dict : dict
        Sample metadata
    stats_df : pd.DataFrame
        Statistical results
    treatment_name : str
        Name of treatment for titles
    week_column : str
        Metadata field containing timepoint info
    config : TemporalClusteringConfig, optional
        Configuration object
    output_prefix : str, optional
        Prefix for output files
    run_enrichment : bool
        Whether to run enrichment analysis
        
    Returns
    -------
    results : dict
        Dictionary containing all analysis results
    """
    if config is None:
        config = TemporalClusteringConfig()
    
    # Disable interactive plotting to prevent duplicate figure rendering
    plt.ioff()
    
    print("="*80)
    print(f"{treatment_name.upper()}: TEMPORAL CLUSTERING ANALYSIS", flush=True)
    print("="*80, flush=True)
    
    results = {}
    
    # 1. Calculate temporal means (Z-scored within subjects first, then averaged)
    print("\n1. Calculating temporal means (within-subject Z-score normalization)...", flush=True)
    print(f"   Subject column: {config.subject_column}", flush=True)
    temporal_df, unique_weeks = calculate_temporal_means(
        data_df, metadata_dict, week_column, 
        subject_column=config.subject_column
    )
    week_columns = get_week_columns(temporal_df)
    print(f"   Timepoints: {unique_weeks}", flush=True)
    print(f"   Proteins: {len(temporal_df)}", flush=True)
    results['temporal_df'] = temporal_df
    results['week_columns'] = week_columns
    results['weeks'] = unique_weeks
    
    # 2. Cluster temporal trends
    if config.auto_detect_clusters:
        print(f"\n2. Auto-detecting optimal number of clusters (range: {config.min_clusters}-{config.max_clusters})...", flush=True)
    else:
        print(f"\n2. Clustering proteins into {config.n_clusters} clusters...", flush=True)
    cluster_labels, X_scaled, _, cluster_selection_fig = cluster_temporal_trends(temporal_df, week_columns, config)
    cluster_names = name_clusters_by_pattern(X_scaled, cluster_labels, week_columns)
    print(f"   Final cluster count: {len(cluster_names)}", flush=True)
    print("   Cluster distribution:", flush=True)
    for cid, name in cluster_names.items():
        n = (cluster_labels == cid).sum()
        print(f"     {name}: {n} proteins", flush=True)
    results['cluster_labels'] = cluster_labels
    results['cluster_names'] = cluster_names
    results['X_scaled'] = X_scaled
    if cluster_selection_fig is not None:
        results['fig_cluster_selection'] = cluster_selection_fig
    
    # 3. Merge with statistics
    print("\n3. Merging with statistical results...", flush=True)
    print("   (Statistical testing was performed in the differential analysis step,", flush=True)
    print("    typically using limma for longitudinal proteomics data)", flush=True)
    merged_df = merge_with_statistics(
        temporal_df, stats_df, cluster_labels, cluster_names
    )
    results['merged_df'] = merged_df
    
    # 4. Filter significant proteins (with fallback to unadjusted p-values)
    print(f"\n4. Filtering significant proteins (p < {config.p_value_threshold})...", flush=True)
    
    # Determine p-value label for titles
    pval_label = "FDR" if config.use_adjusted_pvalue else "p"
    fallback_used = False
    
    sig_df = filter_significant_proteins(merged_df, config)
    
    # Fallback to unadjusted p-values if no FDR-significant proteins found
    if len(sig_df) == 0 and config.use_adjusted_pvalue and 'P.Value' in merged_df.columns:
        # Check if there are proteins significant with unadjusted p-values
        unadj_count = (merged_df['P.Value'] < config.p_value_threshold).sum()
        if unadj_count > 0:
            print(f"   Warning: No proteins found with FDR < {config.p_value_threshold}")
            print(f"   Note: Automatically falling back to unadjusted p-values ({unadj_count} proteins)")
            print("      Note: Results use raw p-values, interpret with caution")
            # Create a temporary config with unadjusted p-values
            fallback_config = TemporalClusteringConfig(
                p_value_threshold=config.p_value_threshold,
                use_adjusted_pvalue=False,
                min_fold_change=config.min_fold_change
            )
            sig_df = filter_significant_proteins(merged_df, fallback_config)
            pval_label = "p (unadjusted)"
            fallback_used = True
    
    print(f"   Significant proteins: {len(sig_df)}", flush=True)
    results['significant_df'] = sig_df
    results['fallback_used'] = fallback_used
    
    # 5. Create visualizations
    print("\n5. Creating visualizations...", flush=True)
    
    # Check if there are any significant proteins
    if len(sig_df) == 0:
        print(f"   Warning: No significant proteins found ({pval_label} < {config.p_value_threshold})")
        print("   Skipping heatmap and parallel coordinate plots")
        print("   Consider using unadjusted p-values (set use_adjusted_pvalue=False in config)")
        results['fig_heatmap'] = None
        results['fig_parallel'] = None
    else:
        # Heatmap
        print("   - Cluster heatmap", flush=True)
        fig_heatmap = plot_cluster_heatmap(
            sig_df, week_columns, 
            title=f'{treatment_name}: Significant Proteins by Cluster ({pval_label}<{config.p_value_threshold})',
            config=config
        )
        results['fig_heatmap'] = fig_heatmap
        
        # Parallel coordinate plots
        print("   - Parallel coordinate plots", flush=True)
        fig_parallel = plot_cluster_parallel_coordinates(
            sig_df, week_columns,
            title=f'{treatment_name}: Temporal Patterns ({pval_label}<{config.p_value_threshold})',
            config=config
        )
        results['fig_parallel'] = fig_parallel
    
    # Silhouette analysis plot - shows ALL proteins used for clustering
    print("   - Silhouette analysis plot", flush=True)
    fig_silhouette = plot_silhouette_analysis(
        X_scaled, cluster_labels, cluster_names,
        title=f'{treatment_name}: Cluster Quality Analysis (all {len(cluster_labels)} proteins)'
    )
    results['fig_silhouette'] = fig_silhouette
    
    # 6. Enrichment analysis
    if run_enrichment and len(sig_df) > 0:
        print("\n6. Running enrichment analysis...", flush=True)
        enrichment_results = run_enrichment_by_cluster(sig_df, config=config)
        results['enrichment_results'] = enrichment_results
        
        # Create enrichment plots
        for cluster_name, enrich_df in enrichment_results.items():
            if not enrich_df.empty:
                fig = plot_enrichment_barplot(
                    enrich_df, 
                    f'{treatment_name}: {cluster_name} Enrichment'
                )
                results[f'fig_enrichment_{cluster_name}'] = fig
        
        # Comparison plot
        if any(not df.empty for df in enrichment_results.values()):
            fig_comparison = plot_enrichment_comparison(
                enrichment_results,
                f'{treatment_name}: Pathway Enrichment Comparison'
            )
            results['fig_enrichment_comparison'] = fig_comparison
    elif run_enrichment:
        print("\n6. Skipping enrichment analysis (no significant proteins)", flush=True)
    
    # 7. Export results
    if output_prefix:
        print(f"\n7. Exporting results to {output_prefix}...", flush=True)
        merged_df.to_csv(f'{output_prefix}_all_proteins.csv', index=False)
        sig_df.to_csv(f'{output_prefix}_significant.csv', index=False)
        
        if run_enrichment and 'enrichment_results' in results:
            all_enrichment = []
            for cluster_name, enrich_df in results['enrichment_results'].items():
                if not enrich_df.empty:
                    enrich_copy = enrich_df.copy()
                    enrich_copy['Cluster'] = cluster_name
                    all_enrichment.append(enrich_copy)
            if all_enrichment:
                pd.concat(all_enrichment).to_csv(f'{output_prefix}_enrichment.csv', index=False)
    
    print("\n" + "="*80, flush=True)
    print(f"Done: {treatment_name} temporal analysis complete", flush=True)
    print("="*80, flush=True)
    
    # Close all figures to prevent them from leaking into subsequent notebook cells
    # The figures are stored in results dict and can still be displayed with display()
    plt.close('all')
    
    # Re-enable interactive plotting
    plt.ion()
    
    return results
