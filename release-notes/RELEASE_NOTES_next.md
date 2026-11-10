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
  centered on its mean within each stratum before the distances, and labels are permuted only
  within strata. Without it, a batch or day offset with the groups spread unevenly over the
  batches shows up as a group effect. The result also reports `strata` and `n_strata`.
  `metric="braycurtis"` is refused with `strata`, since centering makes values negative.

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

- **New `tests/test_public_api.py` checks that a change keeps working what code outside
  this repository calls.** The lab's notebooks, Skyline-PRISM's parity fixtures and
  PyPI users call the toolkit, and until now no test noticed a renamed keyword, a
  reordered parameter or a removed name. Its tables, in `tests/downstream_contract.py`,
  pin:
  - every public name as of 26.10.1, and that the package's `__all__` still lists its
    names, since `from proteomics_toolkit import *` binds only those;
  - from a scan of the lab's notebooks and scripts, the 138 calls they make into the
    package, private functions included: for each, the parameters callers fill by
    position, in order, the keywords they pass, and the parameters it required, so that
    one some callers leave out can never become required;
  - the attributes callers set on `StatisticalConfig`, `EnrichmentConfig` and
    `TemporalClusteringConfig`, since setting one that does not exist is silently
    ignored.

  Smoke tests make the most-used calls on the bundled example data, chained as the
  PRISM notebooks chain them, and check the result columns callers read. They include
  the run behind Skyline-PRISM's end-to-end fixture, and its call to
  `_fit_intensity_trend_prior` with a bare config that has only the nine attributes PRISM
  sets. Each of these fails the test: renaming a keyword, swapping two parameters,
  removing a name, dropping a name from `__all__`, making an optional parameter required,
  renaming a config attribute, or making the intensity-trend prior read one more config
  attribute.
- **New `tests/scan_downstream_usage.py` finds usage the tables do not pin yet.** Run it
  over the folders that hold the lab's repositories. It prints new calls as entries ready
  to paste, and lists calls to names that no longer exist and config settings that do
  nothing. It reads code in source order, a scope at a time, as Python runs it. A star
  import binds the names it really binds. A variable reused for a second config counts
  toward the second from then on, and a function's parameter hides a variable of the
  same name. `tests/test_scan_downstream_usage.py` tests it.
- `TestEquivalenceTest` (`tests/test_statistical_analysis.py`): TOST p-value and interval
  against hand computation (t and normal), interval-inside-margin equals unadjusted TOST,
  adjustment over the tested features only, the `|logFC / t|` fallback, input validation, and the
  `SE` column of the moderated model.
- `tests/test_multivariate.py`: PERMANOVA with `strata` removes a stratum offset that otherwise
  looks like a group effect, still finds a real effect, permutes only within strata, drops samples
  with no stratum, and refuses Bray-Curtis.

## Documentation

- **American spelling throughout.** Docstrings, comments, the user guide, the tutorial
  notebook and the notes of seven earlier releases used British spellings in places
  (`colour`, `behaviour`, `visualise`, `centred`). They now use American ones. The
  GitHub release pages for 26.2.0, 26.4.1, 26.4.3, 26.7.0, 26.7.1, 26.10.0 and 26.10.1
  were corrected to match their files. No function, parameter or result column changed
  name. `CLAUDE.md` now asks for American spelling.
- `docs/06-statistical-analysis.md`: new section, "Equivalence test: showing that nothing changed".
- `README.md`: `equivalence_test()` and `permanova(strata=...)` in the function lists.
