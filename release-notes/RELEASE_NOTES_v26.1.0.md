# Release Notes v26.1.0

## Overview

Initial release of `proteomics-toolkit`, a Python library for analyzing mass
spectrometry-based proteomics data. The toolkit was extracted from the
`collab-uw-zeng` project and provides a modular, configuration-driven workflow
from data import through statistical analysis, visualization, and export.

## Features

### Data Import (`data_import`)
- Load Skyline protein/peptide quantitation CSV files and sample metadata
- Load PRISM parquet output (`corrected_proteins.parquet`)
- Automatic batch suffix detection and stripping for Skyline-PRISM files
- Sample column identification and name cleaning

### Preprocessing (`preprocessing`)
- UniProt identifier parsing (SwissProt, TrEMBL, bare accession formats)
- Gene name extraction from `GN=` fields and description cleaning
- Sample classification into study and control groups with color assignment
- Data completeness assessment and protein filtering by detection rate
- Standardized 5-column annotation data structure

### Normalization (`normalization`)
- Seven normalization methods: median, VSN, quantile, MAD, z-score, RLR, LOESS
- All methods preserve annotation columns and return standardized structure
- Normalization statistics and method characteristic metadata

### Statistical Analysis (`statistical_analysis`)
- Configuration-driven analysis via `StatisticalConfig`
- Paired and unpaired t-tests (Welch's and Student's)
- Wilcoxon signed-rank and Mann-Whitney U non-parametric tests
- Linear mixed-effects models via `statsmodels`
- Multiple testing correction (FDR, Bonferroni)
- Automatic log transformation detection based on normalization method

### Visualization (`visualization`)
- Volcano plots, box plots, PCA, and comparative PCA
- Normalization comparison (before/after)
- Control sample correlation analysis and CV distributions
- Grouped heatmaps, trajectory plots, and protein profiles
- Sample correlation heatmaps (full and triangular)

### Temporal Clustering (`temporal_clustering`)
- K-means clustering of temporal protein abundance trends
- Automatic cluster naming by trend pattern
- Silhouette analysis for optimal cluster count
- Integration with enrichment analysis per cluster

### Enrichment (`enrichment`)
- Gene set enrichment via the Enrichr API
- Support for GO, KEGG, Reactome, WikiPathway, and other libraries
- Differential enrichment on up/down-regulated gene lists
- Enrichment comparison bar plots across groups

### Classification (`classification`)
- Binary classification with leave-one-out and k-fold cross-validation
- Logistic regression, random forest, linear SVM, and XGBoost methods
- ROC curve plotting and multi-method ROC comparison
- PCA visualization of per-subject fold-changes

### Validation (`validation`)
- Metadata-to-data consistency validation
- Custom exceptions: `SampleMatchingError`, `ControlSampleError`
- Diagnostic report generation for sample matching issues

### Export (`export`)
- Complete analysis export (data, config, results)
- Timestamped configuration files for reproducibility
- Significant protein summary export

## Testing

- 101 passing tests across 8 test files
- Coverage of all modules: data_import, preprocessing, normalization,
  statistical_analysis, validation, classification, enrichment, export
- Shared fixtures in `tests/conftest.py` with realistic proteomics data
- pytest configuration in `pyproject.toml` with `slow` and `network` markers

## Dependencies

- Python >= 3.10
- pandas >= 2.0, numpy >= 1.24, scipy >= 1.10
- statsmodels >= 0.14, scikit-learn >= 1.3
- matplotlib >= 3.7, seaborn >= 0.12
- pyarrow >= 12.0, requests >= 2.25
- Optional: xgboost >= 1.7
