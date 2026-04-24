# Release Notes v26.2.1

## Overview

Patch release. Fixes a `KeyError` in `run_mixed_effects_analysis` when
resolving protein names from a reindexed annotation table, and promotes
`xgboost` from an optional extra to a required dependency.

## Bug Fixes

- Fixed `KeyError` in `run_mixed_effects_analysis` when resolving protein names.
  The function now defaults to the protein index and only looks up
  `protein_annotations.loc[...]` when the annotation index actually contains
  the `protein_idx`. This preserves behavior for older callers that pass
  integer-indexed DataFrames while correctly handling the case where
  `run_comprehensive_statistical_analysis` has reindexed data by protein ID.

## Testing

- Added regression tests for `run_mixed_effects_analysis` covering both the
  reindexed and direct-call scenarios.

## Dependencies

- Promoted `xgboost>=1.7` from an optional extra to a required dependency.
  The `xgboost` and `all` extras have been removed from `pyproject.toml`.
