# Proteomics Analysis Toolkit

[![CI](https://github.com/uw-maccosslab/proteomics-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/uw-maccosslab/proteomics-toolkit/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/proteomics-toolkit)](https://pypi.org/project/proteomics-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/proteomics-toolkit)](https://pypi.org/project/proteomics-toolkit/)
[![License](https://img.shields.io/pypi/l/proteomics-toolkit)](https://github.com/uw-maccosslab/proteomics-toolkit/blob/main/LICENSE)

A Python toolkit for analyzing mass spectrometry-based proteomics data, supporting both Skyline CSV and PRISM parquet workflows.

## Features

### Core Analysis Modules
- **data_import**: Load Skyline CSV or PRISM parquet data, handle batch suffixes, manage sample metadata
- **preprocessing**: Protein identifier parsing, sample classification, data quality assessment
- **normalization**: Seven normalization methods (median, VSN, quantile, MAD, z-score, RLR, LOESS)
- **statistical_analysis**: Differential protein analysis — t-tests, Wilcoxon, Mann-Whitney, mixed-effects models
- **visualization**: Publication-ready plots — volcano, PCA, box plots, heatmaps, correlation, trajectories
- **enrichment**: Gene set enrichment via Enrichr API
- **temporal_clustering**: K-means clustering of temporal protein trends
- **validation**: Metadata/data consistency checking with diagnostic reports
- **export**: Standardized result export with timestamped configs

## Installation

### With `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package and project
manager. Install it once with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from a clone of this repository:

```bash
git clone https://github.com/uw-maccosslab/proteomics-toolkit.git
cd proteomics-toolkit
uv sync                  # creates .venv and installs runtime deps from uv.lock
uv sync --extra dev      # also installs pytest + pytest-cov for running tests
uv sync --extra umap     # also installs umap-learn for plot_umap

# Run commands inside the managed venv
uv run pytest tests/ -v
uv run python -c "import proteomics_toolkit as ptk; print(ptk.__version__)"
```

Or add it to an existing `uv`-managed project:

```bash
uv add proteomics-toolkit
uv add 'proteomics-toolkit[umap]'   # with optional UMAP support
```

### With `pip`

```bash
# Install from PyPI
pip install proteomics-toolkit

# With optional UMAP support (for plot_umap)
pip install proteomics-toolkit[umap]

# Install from GitHub (latest development version)
pip install git+https://github.com/uw-maccosslab/proteomics-toolkit.git

# For development (editable install from local clone)
git clone https://github.com/uw-maccosslab/proteomics-toolkit.git
cd proteomics-toolkit
pip install -e '.[dev]'
```

## Quick Start

### PRISM Workflow (recommended for batch-corrected data)

```python
import proteomics_toolkit as ptk
import pandas as pd

# 1. Load PRISM data
protein_data, metadata, sample_cols = ptk.load_prism_data(
    'PRISM-Output/corrected_proteins.parquet',
    'PRISM-Output/sample_metadata.csv',
)

# 2. Map batch-suffixed column names to short replicate IDs
col_map = ptk.strip_batch_suffix(sample_cols)  # {full_col: short_name}
short_to_col = {v: k for k, v in col_map.items()}

# 3. Build sample metadata dict (keys = full PRISM column names)
meta_dict = {}
for _, row in metadata.iterrows():
    full_col = short_to_col.get(row['Replicate'])
    if full_col:
        meta_dict[full_col] = row.to_dict()

# 4. Build annotation + sample data for stats
annot = protein_data[[
    'leading_protein', 'leading_description', 'leading_gene_name',
    'leading_uniprot_id', 'leading_name'
]].copy()
annot.columns = ['Protein', 'Description', 'Protein Gene', 'UniProt_Accession', 'UniProt_Entry_Name']
data = pd.concat([annot.reset_index(drop=True),
                   protein_data[sample_cols].reset_index(drop=True)], axis=1)
data.index = data['Protein']  # accession as index

# 5. Statistical analysis
config = ptk.StatisticalConfig()
config.analysis_type = 'unpaired'
config.statistical_test_method = 'welch_t'
config.group_column = 'Group'
config.group_labels = ['Control', 'Treatment']  # [reference, study]
config.correction_method = 'fdr_bh'
config.p_value_threshold = 0.05
config.fold_change_threshold = 1.0
config.log_transform_before_stats = True
config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    data, meta_dict, config, protein_annotations=annot
)

# 6. Visualization
ptk.plot_volcano(results, fc_threshold=1.0, gene_column='Protein Gene', label_top_n=15)
ptk.display_analysis_summary(results, config)

# 7. Enrichment
enrich_config = ptk.EnrichmentConfig(
    enrichr_libraries=['GO_Biological_Process_2023', 'KEGG_2021_Human'],
    pvalue_cutoff=0.05,
)
enrich = ptk.run_differential_enrichment(
    results, gene_column='Protein Gene', logfc_column='logFC',
    pvalue_column='adj.P.Val', config=enrich_config,
)
```

### Skyline CSV Workflow

```python
# 1. Load data
protein_data, metadata, peptide_data = ptk.load_skyline_data(
    protein_file='protein_quant.csv',
    metadata_file='metadata.csv',
)

# 2. Process sample names
sample_columns = ptk.data_import.identify_sample_columns(protein_data, metadata)
cleaned_names = ptk.clean_sample_names(sample_columns)

# 3. Parse annotations and filter
processed_data = ptk.parse_protein_identifiers(protein_data)

# 4. Normalize (skip for PRISM — already normalized)
normalized = ptk.median_normalize(processed_data, sample_columns=list(cleaned_names.values()))

# 5. QC plots
ptk.plot_box_plot(normalized, list(cleaned_names.values()), sample_metadata)
ptk.plot_pca(normalized, list(cleaned_names.values()), sample_metadata)
```

## Statistical Analysis

All statistical analyses use `StatisticalConfig` + `run_comprehensive_statistical_analysis()`.

### Unpaired comparison (two independent groups)
```python
config = ptk.StatisticalConfig()
config.analysis_type = 'unpaired'
# Options: 'welch_t', 'mann_whitney', 'limma_like', 'deqms_like'
config.statistical_test_method = 'welch_t'
config.group_column = 'Group'
config.group_labels = ['Control', 'Treatment']
config.log_transform_before_stats = 'auto'
config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    data, sample_metadata, config, protein_annotations=annot
)
```

For small sample sizes, prefer empirical Bayes variance shrinkage: set
`config.statistical_test_method = 'limma_like'` (works on proteins or
peptides), or `'deqms_like'` (protein-level only; uses the `n_peptides`
column from PRISM output to build a peptide-count-conditioned variance
prior). See [docs/06-statistical-analysis.md](docs/06-statistical-analysis.md#limma_like-and-deqms_like-empirical-bayes-variance-shrinkage) for details.

### Paired comparison (before/after per subject)
```python
config = ptk.StatisticalConfig()
config.analysis_type = 'paired'
config.statistical_test_method = 'paired_t'
config.subject_column = 'Subject'
config.paired_column = 'Condition'
config.paired_label1 = 'Before'
config.paired_label2 = 'After'
config.group_column = 'Condition'
config.group_labels = ['Before', 'After']
config.validate()
```

### Mixed-effects model (repeated measures)
```python
config = ptk.StatisticalConfig()
config.analysis_type = 'paired'
config.statistical_test_method = 'mixed_effects'
config.subject_column = 'Subject'
config.paired_column = 'Visit'
config.paired_label1 = 'Baseline'
config.paired_label2 = 'Follow-up'
config.group_column = 'Treatment'
config.group_labels = ['Placebo', 'Drug']
config.interaction_terms = ['Treatment', 'Visit']
config.validate()
```

**Output columns:** `Protein`, `logFC`, `P.Value`, `adj.P.Val`, `AveExpr`, `t`, `Protein Gene`, `Description`, `UniProt_Accession`, `Gene`

## Enrichment

Enrichment results use these column names (not the Enrichr web-UI names):

| Column | Description |
|---|---|
| `Term` | Pathway / GO term name |
| `P_Value` | Unadjusted p-value |
| `Adj_P_Value` | BH-adjusted p-value |
| `Z_Score` | Enrichr z-score |
| `Combined_Score` | log(p) × z — used for ranking |
| `Genes` | Semicolon-separated gene list |
| `N_Genes` | Number of overlapping genes |
| `Library` | Source Enrichr library |

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- statsmodels >= 0.12.0
- requests >= 2.25.0 (for Enrichr API)
- pyarrow >= 8.0.0 (for PRISM parquet files)

## Module Reference

### data_import.py
- `load_skyline_data()` — Load Skyline protein/peptide CSVs + metadata
- `load_prism_data()` — Load PRISM parquet + metadata
- `identify_sample_columns()` — Auto-detect sample columns
- `clean_sample_names()` — Remove common prefixes/suffixes
- `detect_batch_suffix()` — Detect PRISM `__@__` batch suffix
- `strip_batch_suffix()` — Map batch-suffixed names → short names
- `create_sample_column_mapping()` — Map data columns to metadata sample names
- `match_samples_to_metadata()` — Link samples to metadata rows
- `BATCH_SUFFIX_DELIMITER` — Constant: `"__@__"`

### preprocessing.py
- `parse_protein_identifiers()` — Extract UniProt accessions and databases
- `parse_gene_and_description()` — Parse gene names from descriptions
- `classify_samples()` — Classify samples into groups / controls with color assignment
- `apply_systematic_color_scheme()` — Generate consistent group colors
- `create_standard_data_structure()` — Build standard 5-column annotation + sample layout
- `assess_data_completeness()` — Evaluate missing data patterns
- `filter_proteins_by_completeness()` — Remove proteins below detection threshold
- `calculate_group_colors()` — Generate group color mapping
- `identify_annotation_columns()` — Auto-detect annotation vs sample columns

### normalization.py
- `median_normalize()` — Median-based normalization (preserves original scale)
- `vsn_normalize()` — Variance Stabilizing Normalization (arcsinh-transformed)
- `quantile_normalize()` — Force identical distributions
- `mad_normalize()` — Median absolute deviation normalization
- `z_score_normalize()` — Standardize to mean=0, sd=1
- `rlr_normalize()` — Robust linear regression (log2-transformed)
- `loess_normalize()` — LOESS intensity-dependent (log2-transformed)
- `handle_negative_values()` — Handle negative values from VSN
- `analyze_negative_values()` — Analyze negative value patterns
- `calculate_normalization_stats()` — Evaluate normalization effectiveness

### statistical_analysis.py
- `StatisticalConfig` — Configuration class (zero-arg constructor, set attributes individually)
- `run_comprehensive_statistical_analysis()` — Main analysis entry point
- `display_analysis_summary()` — Print/return summary of results
- `run_statistical_analysis()` — Backward-compatible wrapper

### visualization.py
- `plot_box_plot()` — Sample intensity distributions by group
- `plot_volcano()` — Volcano plot with labeled top hits
- `plot_pca()` — PCA with group coloring, optional log-transform
- `plot_comparative_pca()` — Compare PCA across normalization methods
- `plot_normalization_comparison()` — Before/after normalization QC
- `plot_sample_correlation_heatmap()` — Full correlation matrix
- `plot_sample_correlation_triangular_heatmap()` — Lower-triangle correlation
- `plot_control_correlation()` — Control sample correlation with optional clustering
- `plot_control_correlation_analysis()` — Multi-panel control QC
- `plot_control_group_correlation_analysis()` — Group-wise control QC
- `plot_individual_control_pool_analysis()` — Individual control analysis
- `plot_control_cv_distribution()` — CV distribution for control samples
- `plot_grouped_heatmap()` — Heatmap for any grouped data
- `plot_grouped_trajectories()` — Line plots for temporal/dose-response data
- `plot_protein_profile()` — Single protein expression profile

### enrichment.py
- `EnrichmentConfig` — Configuration dataclass (libraries, thresholds, API settings)
- `query_enrichr()` — Query Enrichr API with a gene list
- `parse_enrichr_results()` — Parse raw results into a tidy DataFrame
- `run_enrichment_analysis()` — Complete enrichment on a gene list
- `run_enrichment_by_group()` — Enrichment for each group in a DataFrame
- `run_differential_enrichment()` — Split by up/down-regulated, run enrichment on each
- `plot_enrichment_barplot()` — Horizontal bar plot by Combined Score
- `plot_enrichment_comparison()` — Dot plot comparing enrichment across groups
- `get_available_libraries()` — List common Enrichr libraries
- `merge_enrichment_results()` — Merge multiple enrichment DataFrames

### temporal_clustering.py
- `TemporalClusteringConfig` — Configuration dataclass
- `run_temporal_analysis()` — Complete pipeline: clustering → visualization → enrichment
- `calculate_temporal_means()` — Mean abundance per timepoint across subjects
- `cluster_temporal_trends()` — K-means or hierarchical clustering
- `name_clusters_by_pattern()` — Assign descriptive cluster names
- `classify_trend_pattern()` — Classify individual protein trends
- `merge_with_statistics()` — Merge temporal data with statistical results
- `filter_significant_proteins()` — Filter to significant proteins
- `run_enrichment_by_cluster()` — Enrichment per cluster
- `plot_cluster_heatmap()` — Cluster-organized heatmap
- `plot_cluster_parallel_coordinates()` — Parallel coordinate plots

### validation.py
- `validate_metadata_data_consistency()` — Check metadata matches data columns
- `enhanced_sample_processing()` — Sample processing with validation
- `generate_sample_matching_diagnostic_report()` — Detailed mismatch diagnostics
- `SampleMatchingError` — Exception for sample matching failures
- `ControlSampleError` — Exception for control sample configuration issues

### export.py
- `export_complete_analysis()` — Full export: data + config + results
- `export_analysis_results()` — Export normalized data + differential results
- `export_timestamped_config()` — Save analysis config with timestamp
- `create_config_dict_from_notebook_vars()` — Build config dict from notebook variables
- `export_significant_proteins_summary()` — Export significant results summary
- `export_results()` — General-purpose result export

## See Also

- [User Guide](docs/01-overview.md) -- Index of topic-focused recipe pages under `docs/`
- [Tutorial notebook](docs/tutorial.ipynb) -- End-to-end workflow on the bundled example dataset
- [CLAUDE.md](../CLAUDE.md) — Project conventions and data prep patterns
