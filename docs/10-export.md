# Export

[← Back to overview](01-overview.md)

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

The full export writes:

- `<prefix>_normalized_data.csv` — normalized intensities with annotations
- `<prefix>_sample_metadata.csv` — sample metadata
- `<prefix>_differential_results_annotated.csv` — differential results
  with gene annotations merged in
- `<prefix>_config_<timestamp>.py` — human-readable Python file
  recording every parameter used

This makes an analysis directory self-contained and reproducible. See
[06-statistical-analysis.md](06-statistical-analysis.md) for how to build
`config_dict` from a `StatisticalConfig` object.
