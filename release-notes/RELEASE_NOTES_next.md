# Release Notes (Next Release)

## Overview

<!-- One paragraph: what this release is about. -->

## New Features

<!-- New functions/modules, with the problem each solves. -->

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
  - every public name as of 26.10.1;
  - from a scan of the lab's notebooks and scripts, the 138 calls they make into the
    package, private functions included: for each, the parameters callers fill by
    position, in order, and the keywords they pass;
  - the attributes callers set on `StatisticalConfig`, `EnrichmentConfig` and
    `TemporalClusteringConfig`, since setting one that does not exist is silently
    ignored.

  Smoke tests make the most-used calls on the bundled example data, chained as the
  PRISM notebooks chain them, and check the result columns callers read. They include
  the run behind Skyline-PRISM's end-to-end fixture, and its call to
  `_fit_intensity_trend_prior` with a bare config that has only the nine attributes PRISM
  sets. Renaming a keyword, swapping two parameters, removing a name, renaming a config
  attribute, or making the intensity-trend prior read one more config attribute each
  fails the test.
- **New `tests/scan_downstream_usage.py` finds usage the tables do not pin yet.** Run it
  over the folders that hold the lab's repositories. It prints new calls as entries ready
  to paste, and lists calls to names that no longer exist and config settings that do
  nothing.

## Documentation

<!-- Docs added or updated. -->
