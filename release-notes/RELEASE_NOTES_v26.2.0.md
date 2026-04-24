# Release Notes v26.2.0

## Overview

A large feature release. Expands input format coverage (DIA-NN), adds
first-class peptide-level data support, redesigns the peptide coverage
map as a true sequence viewer, consolidates the moderated-analysis API
around a new `run_moderated_linear_model` with three variance-prior
modes (including a Python equivalent of limma's `trend=TRUE`), adds
leakage-free feature selection to the binary classifier, fills in
missing QC plots, and ships a bundled example dataset plus a Jupyter
tutorial notebook. Documentation has been split into topic-focused
pages under `docs/` with a numeric-prefixed index. Motivated in part
by feature comparison with BludauLab's
[proteopy](https://github.com/UKHD-NP/proteopy).

**Breaking changes** (see individual sections for migration snippets):
- `run_limma_like_analysis` / `run_deqms_like_analysis` removed -
  replaced by `run_moderated_linear_model` + `config.moderation`.
- `plot_peptide_coverage_map` now requires a `protein_sequence`
  argument and defaults to `color_by="abundance"` instead of
  detection frequency.
- `datasets.load_example_data()` now returns a 5-tuple
  `(protein_data, peptide_data, metadata, sample_columns, protein_sequences)`.
- Classification default feature selection changed from fold-change
  to MAD (leakage-free). Old behaviour available via
  `feature_selection="fold_change"`.

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

## Documentation

- Removed the `low_confidence` filter step from the PRISM quickstart in
  `README.md` and from the unpaired-comparison recipe in `docs/guide.md`.
  The `low_confidence` column flagged proteins inferred only via shared
  peptides; it did not indicate low identification confidence and the
  name misled users into dropping legitimate hits. The column is being
  removed from PRISM in a future release.
- `load_prism_data` still recognises `low_confidence` in its known-annotation
  set so legacy parquets continue to load cleanly, but its docstring now
  notes the column is deprecated.
- Regenerated the bundled example dataset to match the new PRISM schema
  (no `low_confidence` column).
- Added `docs/guide.md` §6.3.1 describing when to use `limma_like` vs
  `deqms_like` and showing example configs for each.

## Statistical Modeling

- **`limma_like`** — new `StatisticalConfig.statistical_test_method` value.
  Native NumPy/SciPy reimplementation of limma's moderated t-statistic
  (Smyth 2004): per-feature linear model + empirical Bayes shrinkage of
  residual variances toward a global prior `(s0^2, d0)`. No R or `rpy2`
  dependency. Works on both protein-level and peptide-level DataFrames.
- **`deqms_like`** — new `StatisticalConfig.statistical_test_method` value.
  Builds on `limma_like` with a peptide-count-conditioned variance prior
  (Zhu et al. 2020). The prior `s0^2` is a LOWESS function of
  `log(n_peptides)` so proteins quantified from more peptides benefit
  from tighter shrinkage. Protein-level only; reads the peptide-count
  column named in `StatisticalConfig.peptide_count_column` (default
  `"n_peptides"`, which PRISM protein parquets ship by default).
- Both functions are exported at the top level:
  `ptk.run_limma_like_analysis`, `ptk.run_deqms_like_analysis`. The
  comprehensive dispatcher routes the new method names automatically.
- Output DataFrames include extra diagnostic columns: `residual_s2`,
  `residual_df`, `posterior_s2`, `posterior_df`, and (for `deqms_like`)
  `peptide_count_used`.

## QC Plots (cont.)

- **`plot_variance_vs_peptide_count`** — new DEqMS diagnostic. Scatter of
  per-protein residual variance against peptide count (both log-scaled)
  with a LOWESS fit overlaid. Use after `run_deqms_like_analysis` to
  confirm the peptide-count-conditioned variance prior is doing useful
  work (a clear downward trend means proteins with more peptides have
  genuinely lower residual variance).

## Moderated linear model: unified API, intensity-trend prior, and robust fitting

- **`run_moderated_linear_model`** is the new unified entry point for
  limma-/DEqMS-style moderated differential analysis. Selects the
  variance prior via `StatisticalConfig.moderation`:
  - `"intensity_trend"` *(new default)* — Python equivalent of limma's
    `trend=TRUE`. Prior conditioned on per-(feature, group) mean
    intensity via LOWESS. Captures the Poisson-ish `sd ∝ √intensity`
    relationship that dominates MS counting noise.
  - `"limma"` — global empirical Bayes prior (Smyth 2004).
  - `"deqms"` — peptide-count-conditioned prior (Zhu 2020).
- **`config.robust = True`** (new, default False) enables Huber-style
  Winsorization of per-feature `log(s²)` using median/MAD thresholds
  when estimating the prior hyperparameters. Matches limma's
  `robust=TRUE`; prevents a handful of genuinely high-variance
  features from inflating `s0²` and blunting every test.
- New diagnostic plot **`plot_variance_vs_intensity(results)`** shows
  one point per (feature, group) pair: Y = within-group SD, X =
  √(within-group mean intensity). Overlays the LOWESS prior and a
  dashed `sd = k·√intensity` reference line so Poisson-like noise
  reads as slope ≈ 1.
- New helper **`get_intensity_trend_points(results)`** returns the
  long-form per-(feature, group) DataFrame used by the plot.
- Output DataFrames carry additional per-feature columns:
  `residual_s2`, `residual_df`, `posterior_s2`, `posterior_df`,
  `limma_s0_sq`, plus one of `deqms_s0_sq`+`peptide_count_used` or
  `intensity_s0_sq`+`intensity_used` depending on the moderation.

**Breaking change:** `run_limma_like_analysis` and
`run_deqms_like_analysis` are removed, along with the
`statistical_test_method` string values `"limma_like"` /
`"deqms_like"`. Migration:

```python
# Before
config.statistical_test_method = 'limma_like'
# After
config.statistical_test_method = 'moderated_linear_model'
config.moderation = 'limma'

# Before
config.statistical_test_method = 'deqms_like'
# After
config.statistical_test_method = 'moderated_linear_model'
config.moderation = 'deqms'
```

The dispatcher raises a `ValueError` on the removed strings that points
callers at the new API.

## Peptide coverage map rewritten

- **`plot_peptide_coverage_map`** is now a sequence-viewer-style plot
  matching the canonical proteomics coverage-map layout (cf.
  proteomics literature). The parent protein sequence is rendered as
  monospaced text, peptide bars are drawn at their true residue
  start/end positions underneath, and long proteins wrap to multiple
  rows (default 60 residues per row).
- **New required argument: `protein_sequence`** (string of amino
  acids). Callers must pass the parent-protein sequence so the
  function can render it and locate peptides. Missing / empty values
  raise a clear `ValueError`.
- **Colour modes** via `color_by`: `"abundance"` *(default, log2 mean
  intensity across sample_columns)*, `"fold_change"`
  *(log2(study/ref), diverging cmap centred at 0)*, or `"detection"`
  *(fraction of samples detected; the previous default, now opt-in)*.
  For fold-change, pass either a pre-computed `value_column` or
  `(sample_metadata, group_column, group_labels=(ref, study))`.
- Peptides rarely overlap in DIA data, but overlapping peptides from
  missed cleavages are automatically stacked on separate tracks within
  a row.
- If `start_column` is not supplied, positions are inferred from
  `protein_sequence.find(peptide_sequence)`; unlocatable peptides are
  logged and skipped.
- **Breaking change:** the previous default behaviour (detection-only,
  no sequence rendering) is no longer reachable with the same
  arguments. Update call sites to pass `protein_sequence` and choose a
  `color_by`.

## FASTA loading helper

- **`ptk.load_fasta_sequences(path)`** parses a FASTA file into
  `{accession -> sequence}` (and aliases the UniProt entry name to the
  same sequence). No Biopython dependency; handles
  `>sp|ACC|ENTRY_NAME ...` and `>tr|...|...` headers, plus plain
  headers that use the first whitespace-delimited token.

## Bundled dataset updates

- `proteomics_toolkit.datasets.load_example_data()` now returns a
  **5-tuple**: `(protein_data, peptide_data, metadata, sample_columns,
  protein_sequences)`. Existing code that unpacks four values needs
  updating.
- New accessor `ptk.datasets.load_example_sequences()` returns just the
  protein-sequence dict.
- New bundled file `example_sequences.json` holds amino-acid sequences
  for the 20 proteins that have peptide-level data. The generator now
  samples peptides *from within* those parent sequences at real start
  positions, so the coverage map's sequence lookup works on the
  bundled example without any manual setup.
- `pyproject.toml` `[tool.setuptools.package-data]` now includes
  `*.json` so the sequences file ships in the wheel.

## Classification - leakage-free feature selection

- **`run_binary_classification`** now takes a ``feature_selection``
  argument and defaults to ``"mad"`` (unsupervised median absolute
  deviation across subjects). On small datasets, the previous default
  of ranking by absolute mean fold-change could leak outcome
  information into downstream classification when the treatment effect
  was asymmetric between groups; MAD is label-free and eliminates that
  path.
- Added ``"differential_abundance"`` mode: nested supervised selection.
  A per-feature Welch t-test is fit on the training split of each CV
  fold, and the top N by |t| are used to train that fold's classifier
  and score its held-out test split. This is the statistically correct
  way to use a supervised feature ranker on a small cohort.
- Legacy ``"fold_change"`` ranking remains available but is no longer
  the default. The previous fixed behaviour corresponds to
  ``feature_selection="fold_change"``.
- **`select_features_by_mad`** is exposed at the top level
  (`ptk.select_features_by_mad`) so users can inspect the MAD ranking
  directly.
- Returned dict gained a ``"feature_selection"`` key echoing the
  strategy used; for ``"differential_abundance"`` the ``"feature_names"``
  list reflects features picked on the full cohort (used only for the
  final-model feature importances), while cross-validated performance
  reflects the per-fold selections.
- [docs/09-classification.md](../docs/09-classification.md) rewritten
  with a leakage-aware recipe guide.
