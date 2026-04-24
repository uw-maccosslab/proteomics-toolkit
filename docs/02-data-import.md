# Data Import

[← Back to overview](01-overview.md)

Three input formats are supported out of the box:

- [Skyline CSV](#skyline-csv) - legacy v1-style workflow
- [PRISM parquet](#prism-parquet) - recommended for batch-corrected data
- [DIA-NN pg_matrix.tsv](#dia-nn-pg_matrixtsv) - DIA-NN protein-group matrix

## Skyline CSV

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

## PRISM parquet

PRISM column names have the format `<sample>__@__<batch>`. PRISM always
emits both a protein parquet (`corrected_proteins.parquet`) and a peptide
parquet (`corrected_peptides.parquet`).

### Quick load (recommended)

```python
# Protein-level
protein_data, metadata, sample_cols = ptk.load_prism_data(
    'PRISM-Output/corrected_proteins.parquet',
    'PRISM-Output/sample_metadata.csv',
)

# Peptide-level
peptide_data, _, _ = ptk.load_prism_peptide_data(
    'PRISM-Output/corrected_peptides.parquet',
)

# Map full PRISM column names → short replicate IDs for metadata joining
col_map = ptk.strip_batch_suffix(sample_cols)   # {full_col: short_name}
```

### Building metadata dict and annotation DataFrame

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

> The `low_confidence` column shipped by older PRISM versions was
> previously used as a filter. **Do not filter on it** - it flagged
> proteins inferred via shared peptides, not low-confidence
> identifications. The column is being removed from PRISM output.

## DIA-NN `pg_matrix.tsv`

```python
protein_data, metadata, sample_cols = ptk.load_diann_data(
    'diann-output/report.pg_matrix.tsv',
    'metadata.csv',
)
# DIA-NN sample columns are file paths; clean them for display
col_map = ptk.clean_sample_names(sample_cols, auto_detect=True)
```

DIA-NN annotation columns (`Protein.Group`, `Protein.Ids`,
`Protein.Names`, `Genes`, `First.Protein.Description`) are mapped onto
the standard 5-column annotation prefix and the originals are preserved
alongside for reference.

## When to use each format

| | Skyline CSV | PRISM Parquet | DIA-NN pg_matrix |
|---|---|---|---|
| Normalization | Apply via `ptk` (see [05-normalization.md](05-normalization.md)) | Already done by PRISM | Apply via `ptk` |
| Protein grouping | Apply via `ptk` | Already done by PRISM | Already done by DIA-NN |
| Data format | Wide CSV | Wide Parquet with `__@__` suffix | Wide TSV |
| Peptide-level output | Separate CSV | `corrected_peptides.parquet` | `report.pr_matrix.tsv` (not yet supported) |
| Peptide count per protein | N/A | `n_peptides` column | N/A (not emitted) |
| Typical use | v1 notebooks | v2+ notebooks, PRISM pipeline | DIA-NN users |

## Next steps

- [Build your sample metadata](03-metadata.md)
- [Run QC plots](04-qc-plots.md)
- [Run statistical analysis](06-statistical-analysis.md)
