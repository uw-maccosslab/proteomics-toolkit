# Release Notes (Next Release)

## Overview

<!-- One paragraph: what this release is about. -->

## New Features

<!-- New functions/modules, with the problem each solves. -->

## Bug Fixes

- **The LICENSE file disagreed with the declared license.** From 26.1.0 through
  26.10.0 the package declared Apache-2.0 in `pyproject.toml`, and PyPI lists it that
  way, but the `LICENSE` file shipped in the sdist and wheel held the MIT text.
  `LICENSE` now holds the Apache License 2.0 text, matching the metadata and the lab's
  other tools.
- **Group colours and legend order changed from run to run.** `plot_pca` and
  `plot_sample_correlation_heatmap` took their groups from `list(set(groups))`.
  Python randomises string hashing in each process, so which group got which tab10
  colour, and the legend order, varied between runs of the same notebook. The data
  and statistics were unaffected, but rerunning a notebook changed its figures. Both
  now sort the groups through a shared helper, which falls back to sorting by text
  when labels of different types cannot be compared, and keeps labels with the same
  text (`1` and `"1"`) in first-seen order.
  `plot_sample_correlation_triangular_heatmap` and `plot_identifications_per_sample`,
  which already sorted, use the helper too, so mixed-type labels no longer make them
  raise `TypeError`.
- **`plot_control_group_correlation_analysis` shuffled its samples.** It
  de-duplicated each control group's samples through a set, so their order on the
  correlation grid varied between runs. It now keeps the order of `sample_columns`.

## Performance

<!-- Performance improvements with context, e.g.
"Reduced memory from 35 GB to 5 GB for 240-file experiments". -->

## Breaking Changes

<!-- Any changes that require user action (config format changes, removed
options, renamed APIs, etc). Omit this section if there are no breaking
changes. -->

## Testing

- New `tests/test_visualization_group_order.py` renders `plot_pca`,
  `plot_sample_correlation_heatmap` and `plot_control_group_correlation_analysis` in
  subprocesses with four different `PYTHONHASHSEED` values and checks that the images
  are identical, and checks that the PCA legend is sorted. Both checks fail on the
  previous code.

## Documentation

<!-- Docs added or updated. -->
