# proteomics_toolkit — Analysis Skills Guide

A practical recipe book for using `proteomics_toolkit` (`ptk`) for common proteomics analysis patterns in this repository.

---

## Table of Contents
1. [Installation & Import](#1-installation--import)
2. [Loading Data](#2-loading-data)
   - [From Skyline CSV (v1-style)](#21-from-skyline-csv-v1-style)
   - [From PRISM Parquet (v2-style)](#22-from-prism-parquet-v2-style)
3. [Sample Metadata and Classification](#3-sample-metadata-and-classification)
4. [Quality Control Plots](#4-quality-control-plots)
5. [Normalization](#5-normalization)
6. [Statistical Analysis Recipes](#6-statistical-analysis-recipes)
   - [Paired t-test (before/after per patient)](#61-paired-t-test-beforeafter-per-patient)
   - [Unpaired comparison (two groups)](#62-unpaired-comparison-two-independent-groups)
   - [PRISM data — unpaired comparison](#63-prism-data--unpaired-comparison)
   - [Mixed-effects model (repeated measures)](#64-mixed-effects-model-repeated-measures)
   - [Dose-response / linear trend](#65-dose-response--linear-trend-over-time)
7. [Results Visualization](#7-results-visualization)
8. [Gene Set Enrichment](#8-gene-set-enrichment)
9. [Binary Classification](#9-binary-classification)
10. [Export](#10-export)
11. [StatisticalConfig Reference](#11-statisticalconfig-reference)
12. [Enrichment Column Reference](#12-enrichment-column-reference)

---

## 1. Installation & Import

```bash
# Install from GitHub
pip install git+https://github.com/uw-maccosslab/proteomics-toolkit.git

# For development (editable install from local clone)
pip install -e .
```

```python
import proteomics_toolkit as ptk
```

All commonly-used functions are re-exported at the top level (`ptk.<function>`).
Modules are also accessible directly: `ptk.visualization`, `ptk.statistical_analysis`, etc.

---

## 2. Loading Data

### 2.1 From Skyline CSV (v1-style)

```python
protein_data, metadata, peptide_data = ptk.load_skyline_data(
    protein_file  = 'proteins.csv',
    metadata_file = 'metadata.csv',
    peptide_file  = 'peptides.csv',   # optional
)

# Identify sample columns and parse protein annotations
sample_columns = ptk.data_import.identify_sample_columns(protein_data, metadata)
protein_data   = ptk.preprocessing.parse_protein_identifiers(protein_data)
protein_data   = ptk.preprocessing.parse_gene_and_description(protein_data)

# Strip common prefix from sample names and match to metadata
cleaned_names   = ptk.clean_sample_names(sample_columns, common_prefix=None)
sample_metadata = ptk.data_import.match_samples_to_metadata(cleaned_names, metadata)

# Rename columns in protein_data to match cleaned names
protein_data    = protein_data.rename(columns=cleaned_names)
sample_columns  = list(cleaned_names.values())
```

### 2.2 From PRISM Parquet (v2-style)

PRISM column names have the format `<sample>__@__<batch>`.
Use `ptk.load_prism_data()` for the standard load, or `ptk.strip_batch_suffix()` for manual column mapping.

#### Quick load (recommended)

```python
protein_data, metadata, sample_cols = ptk.load_prism_data(
    'PRISM-Output/corrected_proteins.parquet',
    'PRISM-Output/sample_metadata.csv',
)

# Map full PRISM column names → short replicate IDs for metadata joining
col_map = ptk.strip_batch_suffix(sample_cols)   # {full_col: short_name}
```

#### Building metadata dict and annotation DataFrame

```python
import pandas as pd

# Build sample_metadata dict keyed by full PRISM column names
short_to_col = {short: full for full, short in col_map.items()}
sample_metadata = {}
for _, row in metadata.iterrows():
    full_col = short_to_col.get(row['Replicate'])  # or row['sample']
    if full_col:
        sample_metadata[full_col] = row.to_dict()

# Build protein_annotations DataFrame (standard 5-column format)
# PRISM annotation columns → standard statistical analysis columns:
#   leading_protein     → Protein
#   leading_description → Description
#   leading_gene_name   → Protein Gene
#   leading_uniprot_id  → UniProt_Accession
#   leading_name        → UniProt_Entry_Name
protein_annotations = protein_data[[
    'leading_protein', 'leading_description', 'leading_gene_name',
    'leading_uniprot_id', 'leading_name'
]].copy()
protein_annotations.columns = [
    'Protein', 'Description', 'Protein Gene', 'UniProt_Accession', 'UniProt_Entry_Name'
]
# Set index to protein accessions for mixed-effects annotation lookups
protein_annotations = protein_annotations.set_index('Protein')
protein_annotations.index.name = None           # avoid index/column name clash
protein_annotations['Protein'] = protein_annotations.index  # re-add for lookups
```

**When to use PRISM parquet vs Skyline CSV:**
| | Skyline CSV | PRISM Parquet |
|---|---|---|
| Normalization | Apply via ptk | Already done by PRISM |
| Protein grouping | Apply via ptk | Already done by PRISM |
| Data format | Wide CSV | Wide Parquet with `__@__` suffix |
| Typical use | v1 notebooks | v2+ notebooks |

---

## 3. Sample Metadata and Classification

```python
# Classify samples into study groups and controls; assign consistent colors
group_distribution, control_samples, study_samples, sample_metadata, group_colors = \
    ptk.classify_samples(
        sample_metadata          = sample_metadata,
        group_column             = 'Condition',          # column with group labels
        group_labels             = ['A', 'B'],           # study groups
        control_column           = 'Condition',          # column with control labels
        control_labels           = ['QC', 'Ref-Pool'],   # QC / pool labels
        apply_systematic_colors  = True,
        systematic_color_palette = 'Set1',
    )

# Convenience subsets
exp_sample_cols = [
    col for col, meta in sample_metadata.items()
    if meta.get('Condition') in ['A', 'B']
]
all_sample_cols = list(sample_metadata.keys())

# Ordered list for consistent plot axes
group_order = ['A', 'B'] + [g for g in group_distribution if g not in ['A', 'B']]
```

---

## 4. Quality Control Plots

```python
# Sample intensity distributions
ptk.plot_box_plot(
    data            = protein_data,
    sample_columns  = all_sample_cols,
    sample_metadata = sample_metadata,
    group_colors    = group_colors,
    group_order     = group_order,
    log_transform   = True,
    title           = 'Raw Intensity Distribution',
)

# PCA
ptk.plot_pca(
    data            = protein_data,
    sample_columns  = exp_sample_cols,
    sample_metadata = sample_metadata,
    group_colors    = group_colors,
    title           = 'PCA — All Samples',      # custom title
    log_transform   = True,                       # log2-transform before PCA
)

# Control sample correlation heatmap
control_cols = [c for c, m in sample_metadata.items() if m.get('Group') in ['Pool', 'Ref']]
ptk.plot_control_correlation(
    data             = protein_data,
    control_columns  = control_cols,
    sample_metadata  = sample_metadata,
    title            = 'Control Sample Correlation',
    log_transform    = False,
    cluster          = True,                      # hierarchical clustering
    group_colors     = group_colors,
    group_column     = 'Group',
)

# Sample-to-sample correlation heatmap (triangular)
ptk.visualization.plot_sample_correlation_triangular_heatmap(
    data            = protein_data,
    sample_columns  = exp_sample_cols,
    sample_metadata = sample_metadata,
    method          = 'pearson',          # or 'spearman'
    group_colors    = group_colors,
    group_column    = 'Condition',
)

# QC pool consistency
ptk.plot_control_correlation_analysis(
    original_data          = protein_data,
    median_normalized_data = protein_data,   # same if already normalized
    vsn_normalized_data    = protein_data,
    sample_columns         = all_sample_cols,
    sample_metadata        = sample_metadata,
    control_column         = 'Condition',
    control_labels         = ['QC', 'Ref-Pool', 'GW-QC'],
)
```

---

## 5. Normalization

**Skip this section when loading PRISM parquet** — normalization is pre-applied.
For Skyline CSV data, apply one of:

```python
# Most common: simple, robust, preserves original scale
normalized = ptk.median_normalize(protein_data, sample_columns=sample_columns)

# VSN: handles heteroscedastic data; produces log-like values
normalized = ptk.vsn_normalize(protein_data, optimize_params=False, sample_columns=sample_columns)

# Quantile: forces identical distributions (strong normalization)
normalized = ptk.quantile_normalize(protein_data, sample_columns=sample_columns)
```

Compare before / after:
```python
ptk.plot_normalization_comparison(
    original_data    = protein_data,
    normalized_data  = normalized,
    sample_columns   = sample_columns,
    method           = 'Median',
)
```

---

## 6. Statistical Analysis Recipes

All analyses use `ptk.StatisticalConfig` + `ptk.run_comprehensive_statistical_analysis`.

### 6.1 Paired t-test (before/after per patient)

**Design:** Each patient contributes exactly one "before" and one "after" sample.
**Use case:** Paired clinical study (Condition A → Condition B).

```python
config = ptk.StatisticalConfig()
config.analysis_type           = 'paired'
config.statistical_test_method = 'paired_t'      # or 'mixed_effects'

# Required: how samples are paired
config.subject_column = 'Patient_Number'  # links same patient across conditions
config.paired_column  = 'Condition'       # column that labels the two timepoints
config.paired_label1  = 'A'              # baseline label
config.paired_label2  = 'B'              # follow-up label  (effect = B - A)

# Required: which groups to include in the analysis
config.group_column = 'Condition'
config.group_labels = ['A', 'B']

# Log transformation (set 'auto' when data is linear-scale, e.g. PRISM output)
config.log_transform_before_stats = 'auto'
config.log_base                   = 'log2'

config.correction_method      = 'fdr_bh'
config.use_adjusted_pvalue    = 'adjusted'
config.p_value_threshold      = 0.05
config.fold_change_threshold  = 1.5

config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    normalized_data     = protein_data,
    sample_metadata     = sample_metadata,
    config              = config,
    protein_annotations = protein_annotations,   # optional but recommended
)
```

**Key output columns:** `Protein`, `Gene`, `logFC`, `P.Value`, `adj.P.Val`, `n_pairs`, `cohens_d`

**Interpretation:** `logFC > 0` means higher in Condition B (after).

---

### 6.2 Unpaired comparison (two independent groups)

**Use case:** Case vs Control, two independent patient cohorts.

```python
config = ptk.StatisticalConfig()
config.analysis_type           = 'unpaired'
config.statistical_test_method = 'welch_t'   # or 'mann_whitney' (non-parametric)

config.group_column = 'Disease_Status'
config.group_labels = ['Case', 'Control']

config.log_transform_before_stats = 'auto'
config.correction_method          = 'fdr_bh'
config.p_value_threshold          = 0.05
config.fold_change_threshold      = 1.5
config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    normalized_data     = protein_data,
    sample_metadata     = sample_metadata,
    config              = config,
    protein_annotations = protein_annotations,
)
```

---

### 6.3 PRISM data — unpaired comparison

**Design:** Two independent groups from PRISM-normalized data (e.g., Carroll Lab EV Pilot).
**Use case:** Disease model vs control in mouse EV proteomics.

The key steps are: (1) build the standard 5-column annotation + sample data, (2) set the DataFrame index to protein accessions, (3) always pass `protein_annotations`.

```python
import pandas as pd

# Filter low-confidence proteins
protein_data_filtered = protein_data[protein_data['low_confidence'] != True].copy()

# Build annotation DataFrame (standard 5-column format)
annot = protein_data_filtered[[
    'leading_protein', 'leading_description', 'leading_gene_name',
    'leading_uniprot_id', 'leading_name'
]].copy()
annot.columns = ['Protein', 'Description', 'Protein Gene', 'UniProt_Accession', 'UniProt_Entry_Name']

# Combine annotations + sample data
data = pd.concat([
    annot.reset_index(drop=True),
    protein_data_filtered[sample_cols].reset_index(drop=True)
], axis=1)
data.index = data['Protein']   # accession as index — critical for meaningful results

# Configure comparison
config = ptk.StatisticalConfig()
config.analysis_type           = 'unpaired'
config.statistical_test_method = 'welch_t'
config.group_column            = 'Group'
config.group_labels            = ['KI Control', 'KI']   # [reference, study]
config.log_transform_before_stats = 'auto'
config.correction_method       = 'fdr_bh'
config.p_value_threshold       = 0.05
config.fold_change_threshold   = 1.0
config.validate()

# Run — always pass protein_annotations so results include gene names
results = ptk.run_comprehensive_statistical_analysis(
    data, sample_meta_dict, config,
    protein_annotations=annot,
)
```

**Output columns:** `Protein`, `logFC`, `P.Value`, `adj.P.Val`, `AveExpr`, `t`, `Protein Gene`, `Description`, `UniProt_Accession`, `Gene`

> **Note:** The toolkit automatically preserves protein accessions as the DataFrame index
> during statistical testing, so the `Protein` column in results always contains real
> accession numbers (not integer row indices).

---

### 6.4 Mixed-effects model (repeated measures)

**Use case:** Comparing groups at multiple timepoints; accounts for within-subject correlation.
This is the v1 design: drug dose × visit interaction.

```python
config = ptk.StatisticalConfig()
config.analysis_type           = 'paired'        # or 'interaction'
config.statistical_test_method = 'mixed_effects'

config.subject_column = 'Subject'
config.paired_column  = 'Visit'
config.paired_label1  = 'D-02'    # baseline visit
config.paired_label2  = 'D-13'    # follow-up visit

config.group_column   = 'DrugDose'
config.group_labels   = ['0', '20', '40', '80']

# Interaction terms: test if dose × visit effect is significant
config.interaction_terms    = ['DrugDose', 'Visit']
config.additional_interactions = []
config.covariates           = []           # optional: e.g. ['Age', 'Sex']
config.force_categorical    = False        # True = treat DrugDose as factors

config.log_transform_before_stats = 'auto'
config.correction_method          = 'fdr_bh'
config.validate()

results = ptk.run_comprehensive_statistical_analysis(
    normalized_data     = protein_data,
    sample_metadata     = sample_metadata,
    config              = config,
    protein_annotations = protein_annotations,
)
```

---

### 6.5 Dose-response / linear trend over time

**Use case:** Test if protein abundance changes linearly with dose or time.
Useful when the interval between timepoints varies across subjects (e.g., 18-31 days).
The model is: `Protein ~ TimeBetweenSamples + (1|Subject)`

logFC in the output represents the **slope per unit time** (e.g., per day), so values
are small. Use `fc_threshold=0.01` in volcano plots and `logfc_threshold=0.0` in
enrichment analysis.

```python
config = ptk.StatisticalConfig()
config.analysis_type           = 'linear_trend'
config.statistical_test_method = 'mixed_effects'

config.subject_column = 'Patient_Number'
config.time_column    = 'TimeBetweenSamples'  # 0 for baseline, 18-31 for post

config.correction_method      = 'fdr_bh'
config.p_value_threshold      = 0.05
config.fold_change_threshold  = 1.0

results = ptk.run_comprehensive_statistical_analysis(
    data, meta_dict, config, protein_annotations=annot,
)

# Volcano: use small FC threshold since logFC is per-day
ptk.plot_volcano(results, fc_threshold=0.01, p_threshold=0.05, label_top_n=15)

# Enrichment: no FC cutoff (slopes are small by nature)
enrichment_results = ptk.run_differential_enrichment(
    results, logfc_threshold=0.0, pvalue_threshold=0.05,
)
```

---

## 7. Results Visualization

```python
# Volcano plot
ptk.plot_volcano(
    differential_df        = results,
    fc_threshold           = 1.0,              # log2FC threshold (default: 0.5)
    p_threshold            = 0.05,
    title                  = 'KI vs KI Control',
    gene_column            = 'Protein Gene',   # column for point labels
    label_top_n            = 20,
    use_adjusted_pvalue    = 'adjusted',
    enable_pvalue_fallback = True,
    normalization_method   = 'PRISM_corrected',
    point_size             = 80,               # scatter point size
    alpha                  = 0.4,              # point transparency
    label_fontsize         = 11,               # gene label font size
    axis_label_fontsize    = 20,
    tick_label_fontsize    = 16,
)

# Summary table
ptk.display_analysis_summary(
    differential_results = results,
    config               = config,
    label_top_n          = 20,
)

# Heatmap of significant proteins (see notebooks for full paired heatmap code)
sig = results[results['adj.P.Val'] < 0.05]
ptk.plot_grouped_heatmap(
    data_df       = protein_data.reset_index(),
    value_columns = exp_sample_cols,
    group_column  = None,           # set to a cluster column if available
    label_column  = 'protein_group',
    title         = 'Significant Proteins',
    zscore        = True,
    pvalue_column = 'adj.P.Val',
    sort_by_pvalue = True,
)
```

---

## 8. Gene Set Enrichment

Use the Enrichr API via `ptk.run_differential_enrichment()` for pathway analysis on differential results.

```python
# Configure enrichment
enrich_config = ptk.EnrichmentConfig(
    enrichr_libraries=['GO_Biological_Process_2023', 'GO_Molecular_Function_2023', 'KEGG_2021_Human'],
    pvalue_cutoff=0.05,
    top_n=20,
)

# Run on differential results — splits into Upregulated / Downregulated
enrich_results = ptk.run_differential_enrichment(
    results_df=results,
    gene_column='Protein Gene',
    logfc_column='logFC',
    pvalue_column='adj.P.Val',
    config=enrich_config,
)

# Display and plot results
if enrich_results:
    for direction, enrich_df in enrich_results.items():
        if not enrich_df.empty:
            print(f"\n{direction} enrichment — top results:")
            # NOTE: columns are Adj_P_Value and N_Genes (not 'Adjusted P-value' / 'Overlap')
            print(enrich_df[['Term', 'Adj_P_Value', 'N_Genes']].head(10).to_string(index=False))
            ptk.plot_enrichment_barplot(
                enrich_df,
                title=f'KI vs KI Control — {direction} proteins',
                top_n=15,
            )
            plt.show()
```

**Enrichment DataFrame columns:** `Term`, `P_Value`, `Adj_P_Value`, `Z_Score`, `Combined_Score`, `Genes`, `N_Genes`, `Library`

```python
# Comparison dot plot: upregulated vs downregulated side by side
ptk.plot_enrichment_comparison(
    enrich_results,                   # dict with 'Upregulated' / 'Downregulated' keys
    title='Enrichment Comparison',
    top_n_per_group=10,               # top terms per direction before deduplication
)
```

See [Section 12](#12-enrichment-column-reference) for the full column reference.

---

## 9. Binary Classification

Classify subjects into two groups (e.g., responder vs non-responder) using protein fold-changes as features with cross-validated performance metrics.

### Computing per-subject fold-changes

```python
# Log2 transform first (PRISM data is in linear space)
log2_data = data.copy()
log2_data[study_cols] = np.log2(data[study_cols].clip(lower=1e-6))

# Compute paired differences (post - pre) per subject
fc_matrix = ptk.compute_paired_fold_changes(log2_data, meta_dict, config)

# Build group labels per subject
subject_response = {}
for col, meta in meta_dict.items():
    subj = meta.get('Subject')
    resp = meta.get('Response')
    if subj and resp:
        subject_response[subj] = resp
group_labels = pd.Series(subject_response)
group_labels = group_labels.loc[group_labels.index.intersection(fc_matrix.index)]
```

### PCA of fold-changes

```python
ptk.plot_fold_change_pca(
    fc_matrix, group_labels,
    group_colors={'NR': '#1f77b4', 'R': '#ff7f0e'},
    title='PCA of Treatment Response',
)
```

### Running classification

Four methods are available: `logistic_regression`, `random_forest`, `linear_svm`, `xgboost`.

```python
# Select feature proteins from statistical results
feature_proteins = results_me.loc[results_me['adj.P.Val'] < 0.05, 'Protein'].tolist()

# Run classification with 5-fold stratified CV
cr = ptk.run_binary_classification(
    fc_matrix, group_labels,
    feature_proteins=feature_proteins,
    method='logistic_regression',    # or 'random_forest', 'linear_svm', 'xgboost'
    cv_method=5,
)

# Individual ROC curve with mean +/- SD band
ptk.plot_roc_curve(cr, title='ROC: Logistic Regression')
```

### Comparing multiple methods

```python
methods = ['logistic_regression', 'random_forest', 'linear_svm', 'xgboost']
method_results = {}
for method in methods:
    method_results[method] = ptk.run_binary_classification(
        fc_matrix, group_labels,
        feature_proteins=feature_proteins,
        method=method, cv_method=5,
    )

# Overlay all methods on one ROC plot
ptk.plot_roc_comparison(method_results, title='ROC Comparison')
```

**Classification result keys:** `accuracy`, `balanced_accuracy`, `auc_roc`, `auc_std`, `confusion_matrix`, `cv_predictions`, `feature_importances`, `classification_report`, `fold_roc_data`

**Label inversion:** The module automatically detects and corrects label inversion (AUC < 0.5) by flipping probabilities, predictions, and per-fold ROC curves.

---

## 10. Export

```python
# Quick CSV export
results.to_csv('results-differential.csv', index=False)

# Full export (data + config + results with timestamp)
ptk.export_complete_analysis(
    normalized_data      = protein_data[exp_sample_cols],
    sample_metadata      = sample_metadata,
    config_dict          = config_dict,          # plain dict of all analysis parameters
    differential_results = results,
    filtered_data        = protein_data,         # full protein matrix
    output_prefix        = 'Analysis-Paired',
    analysis_description = 'Paired A vs B analysis',
)

# Significant proteins summary only
ptk.export_significant_proteins_summary(
    differential_results = results,
    config_dict          = config_dict,
    output_prefix        = 'Analysis-Paired',
)
```

---

## 11. StatisticalConfig Reference

| Attribute | Type | Description |
|---|---|---|
| `analysis_type` | str | **Required.** `'paired'`, `'unpaired'`, `'linear_trend'`, `'longitudinal'`, `'interaction'` |
| `statistical_test_method` | str | `'paired_t'`, `'mixed_effects'`, `'welch_t'`, `'student_t'`, `'wilcoxon'`, `'mann_whitney'` |
| `subject_column` | str | Metadata column identifying subjects/patients (required for paired/mixed) |
| `group_column` | str | Metadata column with group labels |
| `group_labels` | list | Labels to compare (e.g. `['A', 'B']`) |
| `paired_column` | str | Column distinguishing the two timepoints in a paired design |
| `paired_label1` | str | Baseline/before label in `paired_column` |
| `paired_label2` | str | Follow-up/after label in `paired_column` |
| `time_column` | str | Numeric time/dose column for `linear_trend`/`longitudinal` |
| `interaction_terms` | list | Mixed-effects interaction terms (e.g. `['Group', 'Visit']`) |
| `covariates` | list | Additional covariates to control for (e.g. `['Age', 'Sex']`) |
| `log_transform_before_stats` | str/bool | `'auto'`, `True`, `False` |
| `log_base` | str | `'log2'` (default), `'log10'`, `'ln'` |
| `correction_method` | str | `'fdr_bh'` (BH), `'bonferroni'`, `'fdr_by'`, etc. |
| `use_adjusted_pvalue` | str | `'adjusted'` or `'unadjusted'` |
| `p_value_threshold` | float | Volcano plot line (default 0.05) |
| `fold_change_threshold` | float | FC threshold for significance calls (default 1.5) |

Always call `config.validate()` before running analysis to catch configuration errors early.

---

## Common Pitfalls

| Problem | Solution |
|---|---|
| `Patient_Number` is `float` (84.0) | Convert to string: `str(int(v))` when building `sample_metadata` |
| PRISM batch suffix in column names | Use `ptk.strip_batch_suffix()` to map to metadata sample names |
| No significant proteins at FDR < 0.05 | Try `use_adjusted_pvalue='unadjusted'` or lower `FDR_THRESHOLD`; check `enable_pvalue_fallback=True` |
| Log transform applied twice | Set `log_transform_before_stats=False` if data is already log-transformed |
| `SampleMatchingError` | Call `ptk.validate_metadata_data_consistency()` for diagnostics |
| Mixed-effects model fails | Ensure `n ≥ 8` per protein and `subject_column` has no NaNs |
| `Protein` column has integer indices | Set `data.index = data['Protein']` before calling `run_comprehensive_statistical_analysis()` — the toolkit now preserves this index automatically |
| `KeyError: "['Adjusted P-value', 'Overlap'] not in index"` | Enrichment columns are `Adj_P_Value` and `N_Genes`, not `Adjusted P-value` / `Overlap` |
| Enrichment results missing gene names | Always pass `protein_annotations` to `run_comprehensive_statistical_analysis()` so `Protein Gene` column is present in results |

---

## 12. Enrichment Column Reference

DataFrames returned by `ptk.run_differential_enrichment()`, `ptk.run_enrichment_analysis()`, and `ptk.parse_enrichr_results()` have these columns:

| Column | Type | Description |
|---|---|---|
| `Term` | str | Enriched pathway / GO term name |
| `P_Value` | float | Unadjusted p-value from Enrichr |
| `Adj_P_Value` | float | Benjamini-Hochberg adjusted p-value |
| `Z_Score` | float | Enrichr z-score |
| `Combined_Score` | float | log(p-value) × z-score — used for bar plot ranking |
| `Genes` | str | Semicolon-separated list of overlapping genes |
| `N_Genes` | int | Number of genes in overlap |
| `Library` | str | Source Enrichr library name |

> **Warning:** These column names use underscores (`Adj_P_Value`, `N_Genes`), not the Enrichr
> web-UI names (`Adjusted P-value`, `Overlap`). This is a common source of `KeyError`.
