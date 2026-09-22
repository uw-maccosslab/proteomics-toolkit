# Proteomics Toolkit v26.7.1 Release Notes

## Overview

Maintenance release. There are no changes to analysis behaviour, function
signatures, or results: code that works on 26.7.0 produces identical output on
26.7.1. What changed is the release plumbing. The v26.7.0 CI run failed its
lint job while all three test jobs passed, and because the publish workflow
gated only on tests, 26.7.0 was still published to PyPI from a tree that did
not pass lint. This release fixes the lint failure and closes the gap that let
it through.

## Bug Fixes

- Fixed an unsorted import block in `proteomics_toolkit/__init__.py` that
  failed ruff's `I001` check. When `run_elasticnet_regression` and
  `plot_regression_scatter` were added in 26.7.0, the `from .regression import
  (...)` block was placed between `.classification` and `.data_import`. The
  top-level convenience imports are one contiguous block that ruff sorts
  alphabetically, so the insertion point left it out of order. The block now
  sits between `.preprocessing` and `.statistical_analysis`. Import ordering
  only; the exported API is unchanged and all 114 top-level names still
  resolve.

## Changes

- The ruff version is now pinned to `0.16.8` in the `dev` extra in
  `pyproject.toml`, and both workflows read that pin through ruff-action's
  `version-file` input. Previously CI installed whatever ruff was latest at
  the time of the run, so a newly released rule could turn a green tree red
  with no change to the code, and there was no pinned ruff in the project at
  all for contributors to run locally. `pip install -e ".[dev]"` now installs
  the same version CI uses, making `ruff check .` locally reproduce the CI
  gate exactly. `pyproject.toml` is the single source of truth for the
  version.
- `publish.yml` now runs a `lint` job and the `publish` job depends on
  `[lint, test]` rather than `test` alone. A tree that fails lint can no
  longer be published to PyPI, which is precisely what happened with 26.7.0.

## Testing

- No new tests. The existing suite passes unchanged: 279 passed, 1 skipped on
  Python 3.12 (the skip is `plot_pca_loadings`'s UMAP case, which requires the
  optional `umap` extra).
