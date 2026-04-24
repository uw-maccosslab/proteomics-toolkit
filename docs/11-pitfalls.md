# Common Pitfalls

[← Back to overview](01-overview.md)

| Problem | Solution |
|---|---|
| `Patient_Number` is `float` (84.0) | Convert to string when building `sample_metadata`: `str(int(v))` |
| PRISM batch suffix in column names | Use `ptk.strip_batch_suffix()` to map to metadata sample names — see [02-data-import.md](02-data-import.md#prism-parquet) |
| No significant proteins at FDR < 0.05 | Try `use_adjusted_pvalue='unadjusted'`, or use `limma_like`/`deqms_like` for small sample sizes — see [06-statistical-analysis.md](06-statistical-analysis.md#limma_like-and-deqms_like-empirical-bayes-variance-shrinkage) |
| Log transform applied twice | Set `log_transform_before_stats=False` if data is already log-transformed — see [06-statistical-analysis.md § Log transformation](06-statistical-analysis.md#log-transformation) |
| `SampleMatchingError` | Call `ptk.validate_metadata_data_consistency()` — see [03-metadata.md](03-metadata.md#validating-metadatadata-consistency) |
| Mixed-effects model fails | Ensure `n ≥ 8` per protein and `subject_column` has no NaNs |
| `Protein` column has integer indices | Set `data.index = data['Protein']` before calling `run_comprehensive_statistical_analysis()` — the toolkit now preserves this index automatically |
| `KeyError: "['Adjusted P-value', 'Overlap'] not in index"` | Enrichment columns are `Adj_P_Value` and `N_Genes`, not `Adjusted P-value` / `Overlap` — see [08-enrichment.md](08-enrichment.md#column-reference) |
| Enrichment results missing gene names | Always pass `protein_annotations` to `run_comprehensive_statistical_analysis()` so `Protein Gene` column is present in results |
| Filtering on PRISM `low_confidence` column | Don't. It flagged shared-peptide inference, not low identification confidence. The column is being removed from PRISM output. |
