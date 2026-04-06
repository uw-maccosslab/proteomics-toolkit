"""
Gene Set Enrichment Analysis Module

This module provides tools for performing gene set enrichment analysis on
protein/gene lists using the Enrichr API. It can be used with:

- Clusters from temporal or other grouping analyses
- Up-regulated or down-regulated proteins from differential abundance analysis
- Any user-defined gene lists

The module is designed to be general-purpose and not tied to any specific
analysis type (temporal, dose-response, etc.).

Author: MacCoss Lab
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import requests
import json
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EnrichmentConfig:
    """Configuration for gene set enrichment analysis.
    
    Attributes
    ----------
    enrichr_libraries : List[str]
        Gene set libraries to query from Enrichr
    pvalue_cutoff : float
        P-value threshold for significant enrichment
    top_n : int
        Maximum number of top terms to return per library
    min_genes : int
        Minimum number of genes required to run enrichment
    rate_limit_delay : float
        Delay between API requests (seconds) for rate limiting
    timeout : int
        Request timeout in seconds
        
    Examples
    --------
    >>> config = EnrichmentConfig()
    >>> config.enrichr_libraries = ['KEGG_2021_Human', 'GO_Biological_Process_2023']
    >>> config.pvalue_cutoff = 0.01
    """
    
    # Libraries to query
    enrichr_libraries: List[str] = field(default_factory=lambda: [
        'GO_Biological_Process_2023',
        'GO_Molecular_Function_2023', 
        'KEGG_2021_Human',
        'Reactome_2022',
        'WikiPathway_2023_Human'
    ])
    
    # Significance thresholds
    pvalue_cutoff: float = 0.05
    top_n: int = 20
    
    # Gene list requirements
    min_genes: int = 5
    
    # API settings
    rate_limit_delay: float = 0.5
    timeout: int = 30
    
    # Visualization settings
    bar_figsize: Tuple[int, int] = (12, 8)
    comparison_figsize: Tuple[int, int] = (14, 10)


# Default library colors for consistent visualization
LIBRARY_COLORS = {
    'GO_Biological_Process_2023': '#1f77b4',
    'GO_Biological_Process_2021': '#1f77b4',
    'GO_Molecular_Function_2023': '#2ca02c',
    'GO_Molecular_Function_2021': '#2ca02c',
    'GO_Cellular_Component_2023': '#17becf',
    'GO_Cellular_Component_2021': '#17becf',
    'KEGG_2021_Human': '#d62728',
    'KEGG_2019_Human': '#d62728',
    'Reactome_2022': '#9467bd',
    'Reactome_2016': '#9467bd',
    'WikiPathway_2023_Human': '#ff7f0e',
    'WikiPathways_2019_Human': '#ff7f0e',
    'MSigDB_Hallmark_2020': '#8c564b',
    'BioPlanet_2019': '#e377c2',
}


# =============================================================================
# ENRICHR API FUNCTIONS
# =============================================================================

def query_enrichr(
    gene_list: List[str],
    config: Optional[EnrichmentConfig] = None,
    description: str = 'Gene Set Enrichment Analysis'
) -> Dict[str, List]:
    """
    Query Enrichr API for gene set enrichment analysis.
    
    This function submits a gene list to the Enrichr web service and returns
    enrichment results for the specified gene set libraries.
    
    Parameters
    ----------
    gene_list : List[str]
        List of gene symbols (e.g., ['BRCA1', 'TP53', 'EGFR'])
    config : EnrichmentConfig, optional
        Configuration object. Uses defaults if not provided.
    description : str
        Description for the gene list submission
        
    Returns
    -------
    Dict[str, List]
        Dictionary mapping library name to list of enrichment results.
        Each result is a list: [rank, term, pval, zscore, combined_score, genes, adj_pval]
        Returns empty dict if submission fails or too few genes.
        
    Examples
    --------
    >>> genes = ['BRCA1', 'BRCA2', 'TP53', 'ATM', 'CHEK2']
    >>> results = query_enrichr(genes)
    >>> if 'KEGG_2021_Human' in results:
    ...     print(f"Found {len(results['KEGG_2021_Human'])} KEGG terms")
    
    Notes
    -----
    Enrichr is a free web service. Please cite:
    - Chen EY et al. (2013) Enrichr: interactive and collaborative HTML5 gene list enrichment analysis tool. BMC Bioinformatics.
    - Kuleshov MV et al. (2016) Enrichr: a comprehensive gene set enrichment analysis web server 2016 update. Nucleic Acids Research.
    """
    if config is None:
        config = EnrichmentConfig()
    
    # Clean gene list - remove NaN, empty strings, whitespace
    clean_genes = []
    for g in gene_list:
        if pd.notna(g):
            gene_str = str(g).strip()
            if gene_str and gene_str.lower() not in ['nan', 'none', '']:
                clean_genes.append(gene_str)
    
    if len(clean_genes) < config.min_genes:
        print(f"  Warning: Only {len(clean_genes)} genes provided, need at least {config.min_genes}")
        return {}
    
    ENRICHR_URL = 'https://maayanlab.cloud/Enrichr'
    
    genes_str = '\n'.join(clean_genes)
    payload = {
        'list': (None, genes_str),
        'description': (None, description)
    }
    
    # Submit gene list
    try:
        response = requests.post(
            f'{ENRICHR_URL}/addList', 
            files=payload, 
            timeout=config.timeout
        )
        if not response.ok:
            print(f"  Error submitting gene list: {response.status_code}")
            return {}
        
        data = json.loads(response.text)
        user_list_id = data['userListId']
        
    except requests.exceptions.Timeout:
        print("  Error: Enrichr request timed out")
        return {}
    except requests.exceptions.ConnectionError:
        print("  Error: Could not connect to Enrichr (check internet connection)")
        return {}
    except Exception as e:
        print(f"  Error connecting to Enrichr: {e}")
        return {}
    
    # Query each library
    results = {}
    for library in config.enrichr_libraries:
        try:
            time.sleep(config.rate_limit_delay)  # Rate limiting
            response = requests.get(
                f'{ENRICHR_URL}/enrich',
                params={'userListId': user_list_id, 'backgroundType': library},
                timeout=config.timeout
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
    config: Optional[EnrichmentConfig] = None
) -> pd.DataFrame:
    """
    Parse Enrichr API results into a tidy DataFrame.
    
    Parameters
    ----------
    results : Dict[str, List]
        Raw results from query_enrichr()
    config : EnrichmentConfig, optional
        Configuration object for filtering thresholds
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Library: Source library name
        - Term: Enriched term/pathway name
        - P_Value: Unadjusted p-value
        - Adj_P_Value: Benjamini-Hochberg adjusted p-value  
        - Z_Score: Enrichr z-score
        - Combined_Score: Enrichr combined score (log(p) * z)
        - Genes: Semicolon-separated list of overlapping genes
        - N_Genes: Number of genes in overlap
        
        Sorted by Combined_Score descending. Returns empty DataFrame if no significant results.
        
    Notes
    -----
    Enrichr result format per term:
    [0] Rank, [1] Term name, [2] P-value, [3] Z-score,
    [4] Combined score, [5] Overlapping genes, [6] Adjusted p-value
    """
    if config is None:
        config = EnrichmentConfig()
    
    parsed_results = []
    
    for library, terms in results.items():
        for term_data in terms[:config.top_n]:
            if len(term_data) >= 7:
                pval = term_data[2]
                adj_pval = term_data[6]
                
                if pval <= config.pvalue_cutoff:
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


# =============================================================================
# HIGH-LEVEL ENRICHMENT FUNCTIONS
# =============================================================================

def run_enrichment_analysis(
    gene_list: List[str],
    config: Optional[EnrichmentConfig] = None,
    description: str = 'Gene Set Enrichment Analysis',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run complete enrichment analysis on a gene list.
    
    This is a convenience function that combines query_enrichr() and 
    parse_enrichr_results() into a single call.
    
    Parameters
    ----------
    gene_list : List[str]
        List of gene symbols
    config : EnrichmentConfig, optional
        Configuration object
    description : str
        Description for the analysis
    verbose : bool
        Whether to print progress messages
        
    Returns
    -------
    pd.DataFrame
        Parsed enrichment results DataFrame
        
    Examples
    --------
    >>> # Enrichment on up-regulated genes
    >>> upregulated = results_df[results_df['logFC'] > 1]['Gene'].tolist()
    >>> enrichment = run_enrichment_analysis(upregulated, description='Upregulated genes')
    
    >>> # Enrichment on cluster members
    >>> cluster1_genes = clustered_df[clustered_df['Cluster'] == 1]['Gene'].tolist()
    >>> enrichment = run_enrichment_analysis(cluster1_genes, description='Cluster 1')
    """
    if config is None:
        config = EnrichmentConfig()
    
    # Clean gene list for counting
    clean_genes = [g for g in gene_list if pd.notna(g) and str(g).strip() != '']
    
    if verbose:
        print(f"Running enrichment on {len(clean_genes)} genes...", flush=True)
    
    raw_results = query_enrichr(gene_list, config, description)
    
    if not raw_results:
        if verbose:
            print("  No results returned from Enrichr", flush=True)
        return pd.DataFrame()
    
    enrichment_df = parse_enrichr_results(raw_results, config)
    
    if verbose:
        if not enrichment_df.empty:
            print(f"  Found {len(enrichment_df)} significant terms", flush=True)
        else:
            print("  No significant enrichment found", flush=True)
    
    return enrichment_df


def run_enrichment_by_group(
    data_df: pd.DataFrame,
    group_column: str,
    gene_column: str = 'Gene',
    config: Optional[EnrichmentConfig] = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run enrichment analysis for each group in a DataFrame.
    
    This function is useful for analyzing enrichment across:
    - Clusters from clustering analysis
    - Up/down regulated groups from differential expression
    - Treatment groups, dose groups, time points, etc.
    
    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame containing gene names and group assignments
    group_column : str
        Column name containing group identifiers
    gene_column : str
        Column name containing gene symbols
    config : EnrichmentConfig, optional
        Configuration object
    verbose : bool
        Whether to print progress messages
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping group name to enrichment results DataFrame
        
    Examples
    --------
    >>> # Enrichment by cluster
    >>> enrichment_by_cluster = run_enrichment_by_group(
    ...     clustered_df, 
    ...     group_column='Cluster_Name',
    ...     gene_column='Gene'
    ... )
    
    >>> # Enrichment for up vs down regulated
    >>> results_df['Direction'] = np.where(results_df['logFC'] > 0, 'Up', 'Down')
    >>> enrichment_by_direction = run_enrichment_by_group(
    ...     results_df[results_df['adj.P.Val'] < 0.05],
    ...     group_column='Direction',
    ...     gene_column='Gene'
    ... )
    """
    if config is None:
        config = EnrichmentConfig()
    
    enrichment_results = {}
    
    for group_name in data_df[group_column].unique():
        subset = data_df[data_df[group_column] == group_name]
        gene_list = subset[gene_column].dropna().tolist()
        
        if verbose:
            print(f"\n{group_name}: {len(gene_list)} genes", flush=True)
        
        if len(gene_list) >= config.min_genes:
            enrichment_df = run_enrichment_analysis(
                gene_list, 
                config, 
                description=f"Group: {group_name}",
                verbose=verbose
            )
            enrichment_results[group_name] = enrichment_df
        else:
            enrichment_results[group_name] = pd.DataFrame()
            if verbose:
                print(f"  Skipping - need at least {config.min_genes} genes", flush=True)
    
    return enrichment_results


def run_differential_enrichment(
    results_df: pd.DataFrame,
    gene_column: str = 'Gene',
    logfc_column: str = 'logFC',
    pvalue_column: str = 'adj.P.Val',
    logfc_threshold: float = 1.0,
    pvalue_threshold: float = 0.05,
    config: Optional[EnrichmentConfig] = None,
    verbose: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run enrichment analysis on up-regulated and down-regulated gene sets.
    
    This function is specifically designed for differential expression results,
    automatically splitting genes into up- and down-regulated groups based
    on fold change and significance thresholds.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Statistical results with fold change and p-values
    gene_column : str
        Column containing gene symbols
    logfc_column : str
        Column containing log2 fold change values
    pvalue_column : str
        Column containing (adjusted) p-values
    logfc_threshold : float
        Absolute log2FC threshold for significance (default 1.0 = 2-fold)
    pvalue_threshold : float
        P-value threshold for significance
    config : EnrichmentConfig, optional
        Configuration object
    verbose : bool
        Whether to print progress messages
        
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys 'Upregulated' and 'Downregulated', 
        each mapping to an enrichment results DataFrame
        
    Examples
    --------
    >>> # Run enrichment on differential expression results
    >>> enrichment = run_differential_enrichment(
    ...     stats_results,
    ...     logfc_threshold=0.5,  # 1.4-fold change
    ...     pvalue_threshold=0.05
    ... )
    >>> print(f"Up-regulated pathways: {len(enrichment['Upregulated'])}")
    >>> print(f"Down-regulated pathways: {len(enrichment['Downregulated'])}")
    """
    if config is None:
        config = EnrichmentConfig()
    
    # Get significant genes
    sig_mask = results_df[pvalue_column] < pvalue_threshold
    
    # Split by direction
    up_mask = sig_mask & (results_df[logfc_column] > logfc_threshold)
    down_mask = sig_mask & (results_df[logfc_column] < -logfc_threshold)
    
    up_genes = results_df.loc[up_mask, gene_column].dropna().tolist()
    down_genes = results_df.loc[down_mask, gene_column].dropna().tolist()
    
    if verbose:
        print(f"Significant genes: {up_mask.sum()} up, {down_mask.sum()} down")
    
    enrichment_results = {}
    
    # Up-regulated
    if verbose:
        print(f"\nUpregulated ({len(up_genes)} genes):", flush=True)
    if len(up_genes) >= config.min_genes:
        enrichment_results['Upregulated'] = run_enrichment_analysis(
            up_genes, config, description='Upregulated genes', verbose=verbose
        )
    else:
        enrichment_results['Upregulated'] = pd.DataFrame()
        if verbose:
            print(f"  Skipping - need at least {config.min_genes} genes", flush=True)
    
    # Down-regulated
    if verbose:
        print(f"\nDownregulated ({len(down_genes)} genes):", flush=True)
    if len(down_genes) >= config.min_genes:
        enrichment_results['Downregulated'] = run_enrichment_analysis(
            down_genes, config, description='Downregulated genes', verbose=verbose
        )
    else:
        enrichment_results['Downregulated'] = pd.DataFrame()
        if verbose:
            print(f"  Skipping - need at least {config.min_genes} genes", flush=True)
    
    return enrichment_results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_enrichment_barplot(
    enrichment_df: pd.DataFrame,
    title: str = 'Gene Set Enrichment',
    top_n: int = 15,
    figsize: Optional[Tuple[int, int]] = None,
    library_colors: Optional[Dict[str, str]] = None
) -> Optional[Figure]:
    """
    Create a horizontal bar plot of enrichment results.
    
    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Enrichment results from parse_enrichr_results() or run_enrichment_analysis()
    title : str
        Plot title
    top_n : int
        Number of top terms to display
    figsize : tuple, optional
        Figure size (width, height)
    library_colors : dict, optional
        Custom colors for each library
        
    Returns
    -------
    Figure or None
        Matplotlib figure, or None if no data to plot
        
    Examples
    --------
    >>> enrichment = run_enrichment_analysis(gene_list)
    >>> fig = plot_enrichment_barplot(enrichment, title='My Enrichment Results')
    >>> fig.savefig('enrichment.png', dpi=150, bbox_inches='tight')
    """
    if enrichment_df.empty:
        print(f"  No significant enrichment results for: {title}")
        return None
    
    if figsize is None:
        figsize = (12, 8)
    
    if library_colors is None:
        library_colors = LIBRARY_COLORS
    
    # Take top N by combined score
    plot_df = enrichment_df.head(top_n).copy()
    plot_df = plot_df.sort_values('Combined_Score', ascending=True)
    
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
    title: str = 'Enrichment Comparison',
    top_n_per_group: int = 5,
    figsize: Optional[Tuple[int, int]] = None
) -> Optional[Figure]:
    """
    Create a dot plot comparing enrichment across multiple groups.
    
    This visualization shows which pathways are enriched in which groups,
    useful for comparing clusters, treatment groups, or up/down regulated sets.
    
    Parameters
    ----------
    enrichment_dict : Dict[str, pd.DataFrame]
        Dictionary mapping group name to enrichment DataFrame
    title : str
        Plot title
    top_n_per_group : int
        Number of top terms to include per group
    figsize : tuple, optional
        Figure size
        
    Returns
    -------
    Figure or None
        Matplotlib figure, or None if no data to plot
        
    Examples
    --------
    >>> enrichment_by_cluster = run_enrichment_by_group(clustered_df, 'Cluster')
    >>> fig = plot_enrichment_comparison(enrichment_by_cluster, title='Cluster Pathways')
    """
    if figsize is None:
        figsize = (14, 10)
    
    # Collect all terms
    all_terms = []
    for group_name, enrichment_df in enrichment_dict.items():
        if not enrichment_df.empty:
            top_terms = enrichment_df.head(top_n_per_group).copy()
            top_terms['Group'] = group_name
            all_terms.append(top_terms)
    
    if not all_terms:
        print(f"  No terms to plot for: {title}")
        return None
    
    combined_df = pd.concat(all_terms, ignore_index=True)
    
    # Get unique terms (top by combined score) and groups
    unique_terms = combined_df.groupby('Term')['Combined_Score'].max().sort_values(ascending=False).head(20).index.tolist()
    groups = list(enrichment_dict.keys())
    active_groups = [g for g in groups if not enrichment_dict[g].empty]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, term in enumerate(unique_terms):
        for j, group in enumerate(active_groups):
            df = enrichment_dict[group]
            term_row = df[df['Term'] == term]
            
            if not term_row.empty:
                pval = term_row['P_Value'].values[0]
                n_genes = term_row['N_Genes'].values[0]
                
                size = min(n_genes * 30, 400)
                color_val = min(-np.log10(pval + 1e-10), 10)
                
                ax.scatter(j, i, s=size, c=[color_val], cmap='Reds', 
                          vmin=0, vmax=10, alpha=0.8, edgecolors='black', linewidths=0.5)
    
    ax.set_xticks(range(len(active_groups)))
    ax.set_xticklabels(active_groups, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(unique_terms)))
    ax.set_yticklabels([t[:50] + '...' if len(t) > 50 else t for t in unique_terms], fontsize=9)
    ax.set_xlabel('Group', fontsize=12)
    ax.set_ylabel('Enriched Term', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=10))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('-log10(P-value)', fontsize=10)
    
    ax.set_xlim(-0.5, len(active_groups) - 0.5)
    ax.set_ylim(-0.5, len(unique_terms) - 0.5)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return fig


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_available_libraries() -> List[str]:
    """
    Get list of commonly used Enrichr libraries.
    
    Returns
    -------
    List[str]
        List of library names that can be used in EnrichmentConfig.enrichr_libraries
        
    Notes
    -----
    This is not exhaustive. See https://maayanlab.cloud/Enrichr/#libraries
    for the complete list of available libraries.
    """
    return [
        # Gene Ontology
        'GO_Biological_Process_2023',
        'GO_Molecular_Function_2023',
        'GO_Cellular_Component_2023',
        # Pathway databases
        'KEGG_2021_Human',
        'Reactome_2022',
        'WikiPathway_2023_Human',
        'BioPlanet_2019',
        # Disease/phenotype
        'DisGeNET',
        'OMIM_Disease',
        'ClinVar_2019',
        # Transcription factors
        'ENCODE_TF_ChIP-seq_2015',
        'ChEA_2022',
        'TRANSFAC_and_JASPAR_PWMs',
        # Drug/chemical
        'DrugMatrix',
        'DSigDB',
        'L1000_Kinase_and_GPCR_Perturbations_up',
        # Other
        'MSigDB_Hallmark_2020',
        'Human_Gene_Atlas',
        'GTEx_Tissue_Expression_Up',
        'GTEx_Tissue_Expression_Down',
    ]


def merge_enrichment_results(
    enrichment_dict: Dict[str, pd.DataFrame],
    add_group_column: bool = True
) -> pd.DataFrame:
    """
    Merge multiple enrichment result DataFrames into one.
    
    Parameters
    ----------
    enrichment_dict : Dict[str, pd.DataFrame]
        Dictionary mapping group name to enrichment DataFrame
    add_group_column : bool
        Whether to add a 'Group' column with the group name
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all enrichment results
    """
    all_dfs = []
    for group_name, df in enrichment_dict.items():
        if not df.empty:
            df_copy = df.copy()
            if add_group_column:
                df_copy['Group'] = group_name
            all_dfs.append(df_copy)
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()
