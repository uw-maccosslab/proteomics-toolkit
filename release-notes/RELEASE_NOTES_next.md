# Release Notes (Next Release)

## Overview

Expands input format coverage, adds first-class peptide-level data support,
fills in missing QC plots, and ships an example dataset with a Jupyter
tutorial notebook. Motivated by feature comparison with BludauLab's
[proteopy](https://github.com/UKHD-NP/proteopy).

## New Features

- **DIA-NN reader** (`load_diann_data`): load `report.pg_matrix.tsv`
  directly and have DIA-NN annotation columns (`Protein.Group`,
  `Protein.Ids`, `Protein.Names`, `Genes`, `First.Protein.Description`)
  mapped onto the toolkit's 5-column standard annotation prefix.
- **PRISM peptide loader** (`load_prism_peptide_data`): convenience
  wrapper around `load_prism_data` making the
  `corrected_peptides.parquet` path discoverable.
- **Bundled example dataset**: `proteomics_toolkit.datasets.load_example_data()`
  returns a small PRISM-style protein + peptide + metadata bundle
  (80 proteins, 60 peptides, 12 samples across two groups) for use in
  tutorials, smoke tests, and demos.
- **Tutorial notebook**: `docs/tutorial.ipynb` walks through the full
  pipeline on the bundled dataset: load -> QC -> stats -> volcano/PCA ->
  peptide coverage -> export.

## QC Plots

Five new visualization functions, each of which accepts either a
protein-level or peptide-level DataFrame:

- `plot_missing_value_heatmap` — NA/zero pattern across samples x features.
- `plot_identifications_per_sample` — bar chart of #features identified
  per sample, coloured by group.
- `plot_intensity_distributions` — KDE overlay of (log) intensity per
  sample.
- `plot_cv_distribution` — CV distribution across all samples, or split
  by group (contrast with the existing `plot_control_cv_distribution`,
  which is controls-only).
- `plot_peptide_coverage_map` — horizontal bar per peptide spanning its
  residue positions along a protein, opacity encoding detection
  frequency across samples.

## Peptide-Level Data Support

- `load_prism_peptide_data` loader (see above).
- Generic QC plots accept a `feature_label` argument so axis titles read
  correctly for peptide DataFrames.
- Statistical functions (`run_unpaired_t_test`, `run_paired_t_test`,
  `run_wilcoxon_test`, `run_mann_whitney_test`) verified to operate on
  peptide-level DataFrames unchanged.
- Peptide-to-protein rollup is explicitly out of scope for this release.

## Testing

- Test count: 101 -> 132 (31 new tests).
- New test files: `tests/test_visualization.py` (13),
  `tests/test_datasets.py` (5).
- Expansions: `tests/test_data_import.py` (+9 for DIA-NN and peptide
  loaders), `tests/test_statistical_analysis.py` (+4 for peptide-level
  runs).
- Tutorial notebook is executed end-to-end as a manual verification step
  before release.

## Dependencies

No additions. Existing dependencies (pandas, numpy, scipy, statsmodels,
scikit-learn, matplotlib, seaborn, pyarrow, requests) are sufficient.

## Packaging

- `pyproject.toml` now declares `proteomics_toolkit.datasets` as a
  subpackage with `*.parquet` and `*.csv` bundled via
  `[tool.setuptools.package-data]`.
