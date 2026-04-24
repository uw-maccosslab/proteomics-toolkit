# Sample Metadata and Classification

[← Back to overview](01-overview.md)

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

The `group_colors` dict is consumed by the [QC plots](04-qc-plots.md) and
[results visualization functions](07-visualization.md) to keep group
colouring consistent across every plot in an analysis.

## Validating metadata/data consistency

If the downstream analysis fails with `SampleMatchingError`, run:

```python
report = ptk.validate_metadata_data_consistency(
    metadata=metadata,
    metadata_sample_names=list(metadata['Replicate']),
    protein_columns=list(protein_data.columns),
    control_column='Condition',
    control_labels=['QC', 'Ref-Pool'],
)
```

This prints a diagnostic report listing unmatched samples on either side.
See also [11-pitfalls.md](11-pitfalls.md).
