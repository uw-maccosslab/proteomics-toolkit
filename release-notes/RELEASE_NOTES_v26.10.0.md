# Proteomics Toolkit v26.10.0 Release Notes

## Overview

Feature release with no breaking changes. The older plot functions in `visualization`
can now hand back their figure, so notebooks can save volcano, PCA and QC plots to
files. Two plotting bugs are fixed: a matplotlib warning from the default PCA colors,
and a crash in `plot_comparative_pca` with matplotlib 3.9 or later.

## New Features

- **`show` parameter on the plot functions that returned nothing.** `plot_volcano`,
  `plot_box_plot`, `plot_pca`, `plot_comparative_pca`,
  `plot_normalization_comparison`, `plot_sample_correlation_heatmap`,
  `plot_sample_correlation_triangular_heatmap` and
  `plot_control_correlation_analysis` called `plt.show()` and returned None, so
  their figures could not be saved. With `show=False` they return the
  `matplotlib.figure.Figure` without showing it
  (`fig = ptk.plot_volcano(results, show=False); fig.savefig(...)`). The default,
  `show=True`, keeps the old behavior: the figure is shown and None is returned.
  Returning the figure in both cases would draw it twice in a notebook whenever the
  call is the last line of a cell, because Jupyter displays both the shown figure
  and the returned one.

## Bug Fixes

- **Matplotlib warning from the default PCA colors.** `plot_pca` and
  `plot_comparative_pca` passed each group's color, an RGBA array from the tab10
  colormap when `group_colors` is not given, as `scatter(c=...)`. Matplotlib warned
  that the `*c*` argument "looks like a single numeric RGB or RGBA sequence" on
  every call. They now pass it as `color=`. `classification.plot_fold_change_pca`
  had the same call and is fixed too.
- **`plot_comparative_pca` failed with matplotlib 3.9 or later** when `group_colors`
  was not given, because it called `plt.cm.get_cmap`, which matplotlib 3.9 removed.
  It now calls `plt.get_cmap`.

## Testing

- New `tests/test_visualization_show.py` covers all eight functions:
  - with `show=False` each returns a figure that saves to PNG;
  - by default each returns None;
  - `plot_volcano` with no rows still returns None;
  - `plot_pca` and `plot_comparative_pca` with the default colors raise neither the
    `*c*` warning nor a deprecation warning.

## Documentation

- `docs/07-visualization.md` has a "Saving a figure" section on `show=False`.
