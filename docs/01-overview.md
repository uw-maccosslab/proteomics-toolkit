# proteomics_toolkit — User Guide

A practical recipe book for using `proteomics_toolkit` (`ptk`) for common
proteomics analysis patterns. Each topic lives in its own file; this page
is the index.

## Installation

```bash
# From PyPI
pip install proteomics-toolkit

# With XGBoost (for the classification module)
pip install proteomics-toolkit[xgboost]

# From source (editable install)
git clone https://github.com/uw-maccosslab/proteomics-toolkit.git
cd proteomics-toolkit
pip install -e .
```

```python
import proteomics_toolkit as ptk
```

All commonly-used functions are re-exported at the top level (`ptk.<function>`).
Modules are accessible directly as well: `ptk.visualization`,
`ptk.statistical_analysis`, etc.

## Bundled tutorial

A small end-to-end example dataset ships with the package. The Jupyter
notebook [tutorial.ipynb](tutorial.ipynb) runs the full pipeline on it.

```python
protein_data, peptide_data, metadata, sample_cols = ptk.datasets.load_example_data()
```

## Guide index

| Topic | File | What it covers |
|---|---|---|
| Data import | [02-data-import.md](02-data-import.md) | Skyline CSV, PRISM parquet, DIA-NN `pg_matrix.tsv` |
| Sample metadata | [03-metadata.md](03-metadata.md) | Building the sample-metadata dict, classifying study vs control |
| QC plots | [04-qc-plots.md](04-qc-plots.md) | Box plots, PCA, correlation heatmaps, missing-value maps, identification counts |
| Normalization | [05-normalization.md](05-normalization.md) | Median / VSN / quantile / MAD / z-score / RLR / LOESS |
| Statistical analysis | [06-statistical-analysis.md](06-statistical-analysis.md) | Paired/unpaired t-test, Wilcoxon, mixed-effects, linear trend, `limma_like`, `deqms_like`, full `StatisticalConfig` reference |
| Results visualization | [07-visualization.md](07-visualization.md) | Volcano, summary tables, grouped heatmaps |
| Gene set enrichment | [08-enrichment.md](08-enrichment.md) | Enrichr API via `run_differential_enrichment`; column reference |
| Binary classification | [09-classification.md](09-classification.md) | Per-subject fold-changes, PCA, LOO/k-fold CV, ROC comparison |
| Export | [10-export.md](10-export.md) | Timestamped reproducible exports |
| Common pitfalls | [11-pitfalls.md](11-pitfalls.md) | Gotchas and fixes |

## Typical workflow

1. **Load** your data: [02-data-import.md](02-data-import.md)
2. **Build metadata** and classify samples: [03-metadata.md](03-metadata.md)
3. **Run QC plots** to spot bad samples early: [04-qc-plots.md](04-qc-plots.md)
4. **Normalize** (skip for PRISM parquet — already normalized):
   [05-normalization.md](05-normalization.md)
5. **Run statistical analysis**: [06-statistical-analysis.md](06-statistical-analysis.md)
6. **Visualize results** (volcano, heatmap): [07-visualization.md](07-visualization.md)
7. **Enrichment / classification** as needed: [08-enrichment.md](08-enrichment.md),
   [09-classification.md](09-classification.md)
8. **Export** for reproducibility: [10-export.md](10-export.md)
