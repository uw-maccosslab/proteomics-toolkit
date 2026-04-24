# Normalization

[← Back to overview](01-overview.md)

**Skip this page when loading [PRISM parquet](02-data-import.md#prism-parquet)** — normalization is pre-applied.

For Skyline CSV data, pick one of the methods below. All of them
preserve the standard 5-column annotation prefix and return a DataFrame
in the same structure as the input.

```python
# Most common: simple, robust, preserves original scale
normalized = ptk.median_normalize(protein_data, sample_columns=sample_columns)

# VSN: handles heteroscedastic data; produces log-like (arcsinh) values
normalized = ptk.vsn_normalize(protein_data, optimize_params=False, sample_columns=sample_columns)

# Quantile: forces identical distributions (strong normalization)
normalized = ptk.quantile_normalize(protein_data, sample_columns=sample_columns)

# Other options: ptk.mad_normalize, ptk.z_score_normalize,
# ptk.rlr_normalize, ptk.loess_normalize
```

## Comparing before / after

```python
ptk.plot_normalization_comparison(
    original_data    = protein_data,
    normalized_data  = normalized,
    sample_columns   = sample_columns,
    method           = 'Median',
)
```

See also [04-qc-plots.md](04-qc-plots.md#intensity-and-cv-distributions) for
per-sample density overlays that make normalization effects easy to
read.

## Method characteristics

| Method | Preserves scale | Log-transformed output | Notes |
|---|---|---|---|
| `median_normalize` | yes | no | Good default |
| `quantile_normalize` | yes | no | Strong: forces identical distributions |
| `mad_normalize` | yes | no | Robust to outliers |
| `z_score_normalize` | yes | no | Mean 0, std 1 per sample |
| `vsn_normalize` | no | yes (arcsinh) | Handles heteroscedasticity |
| `rlr_normalize` | no | yes (log2) | Robust linear regression |
| `loess_normalize` | no | yes (log2) | Intensity-dependent correction |

The statistical analysis pipeline auto-detects log-transformed inputs
via `config.normalization_method` - see
[06-statistical-analysis.md](06-statistical-analysis.md#log-transformation).
