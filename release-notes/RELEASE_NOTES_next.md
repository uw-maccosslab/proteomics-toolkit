# Release Notes (Next Release)

## Overview

Tools for showing that a treatment changes nothing measurable: a per-protein equivalence test on the moderated model, the moderated standard error in its results, and PERMANOVA within strata (batches, days or cohorts).

## New Features

- **`equivalence_test()`** (`statistical_analysis`). A study that finds no differential
  protein has not shown that none changes. The new function runs two one-sided tests (TOST) per
  feature on the moderated linear model's estimate, standard error and posterior degrees of
  freedom, and reports the 90% interval, the largest change the data are compatible with
  (`max_abs_logFC`), the TOST p-value, its FDR-adjusted value and an `Equivalent` flag. For
  example, `equivalence_test(results, margin=np.log2(1.2))` lists the proteins shown to change by
  less than 1.2-fold.
- **`SE` column in moderated results.** `run_moderated_linear_model` (and so
  `run_comprehensive_statistical_analysis` with the moderated model) now reports the moderated
  standard error of `logFC`, the `se` the t-statistic is built from (`logFC / SE == t`).
- **`permanova(strata=...)`** (`multivariate`). Tests the factor within strata: each protein is
  centred on its mean within each stratum before the distances, and labels are permuted only
  within strata. Without it, a batch or day offset with the groups spread unevenly over the
  batches shows up as a group effect. The result also reports `strata` and `n_strata`.
  `metric="braycurtis"` is refused with `strata`, since centring makes values negative.

## Bug Fixes

<!-- Description of the bug and its impact, then what was fixed. -->

## Performance

<!-- Performance improvements with context, e.g.
"Reduced memory from 35 GB to 5 GB for 240-file experiments". -->

## Breaking Changes

<!-- Any changes that require user action (config format changes, removed
options, renamed APIs, etc). Omit this section if there are no breaking
changes. -->

## Testing

- `TestEquivalenceTest` (`tests/test_statistical_analysis.py`): TOST p-value and interval
  against hand computation (t and normal), interval-inside-margin equals unadjusted TOST,
  adjustment over the tested features only, the `|logFC / t|` fallback, input validation, and the
  `SE` column of the moderated model.
- `tests/test_multivariate.py`: PERMANOVA with `strata` removes a stratum offset that otherwise
  looks like a group effect, still finds a real effect, permutes only within strata, drops samples
  with no stratum, and refuses Bray-Curtis.

## Documentation

- `docs/06-statistical-analysis.md`: new section, "Equivalence test: showing that nothing changed".
- `README.md`: `equivalence_test()` and `permanova(strata=...)` in the function lists.
