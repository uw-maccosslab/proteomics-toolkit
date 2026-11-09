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

## Performance

<!-- Performance improvements with context, e.g.
"Reduced memory from 35 GB to 5 GB for 240-file experiments". -->

## Breaking Changes

<!-- Any changes that require user action (config format changes, removed
options, renamed APIs, etc). Omit this section if there are no breaking
changes. -->

## Testing

<!-- New or changed test coverage. -->

## Documentation

<!-- Docs added or updated. -->
