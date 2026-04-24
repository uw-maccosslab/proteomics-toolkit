# Gene Set Enrichment

[← Back to overview](01-overview.md)

Pathway analysis on differential results via the Enrichr API. See
[Column reference](#column-reference) for the exact output schema.

## Differential enrichment

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
            print(enrich_df[['Term', 'Adj_P_Value', 'N_Genes']].head(10).to_string(index=False))
            ptk.plot_enrichment_barplot(
                enrich_df,
                title=f'{direction} proteins',
                top_n=15,
            )
```

> **Common mistake:** columns are `Adj_P_Value` and `N_Genes`, not
> `Adjusted P-value` / `Overlap`. See [Column reference](#column-reference).

## Comparison dot plot

```python
ptk.plot_enrichment_comparison(
    enrich_results,                   # dict with 'Upregulated' / 'Downregulated' keys
    title='Enrichment Comparison',
    top_n_per_group=10,
)
```

## Column reference

DataFrames returned by `ptk.run_differential_enrichment`,
`ptk.run_enrichment_analysis`, and `ptk.parse_enrichr_results` have
these columns:

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

> These column names use underscores (`Adj_P_Value`, `N_Genes`), not
> the Enrichr web-UI names (`Adjusted P-value`, `Overlap`). Using the
> web-UI names raises `KeyError`.
