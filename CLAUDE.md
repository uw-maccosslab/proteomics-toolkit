# Proteomics Toolkit

## Project Overview

A Python library for analyzing mass spectrometry-based proteomics data, focused on
Skyline quantitation outputs. Provides a complete workflow from data import through
statistical analysis, visualization, and export.

## Versioning

This project uses **year.version.minor** versioning:

- **Year**: two-digit calendar year (e.g., 26 for 2026)
- **Version**: incremented for each feature release within the year
- **Minor**: incremented for bug fixes and patches

Example: `26.1.0` is the first release in 2026. A bug fix becomes `26.1.1`.
A new feature release becomes `26.2.0`.

The version is defined in two places that must stay in sync:
- `pyproject.toml` (`version = "..."`)
- `proteomics_toolkit/__init__.py` (`__version__ = "..."`)

## Testing

Tests live in `tests/` and use pytest. Run with:

```
python -m pytest tests/ -v
```

**Every new feature or capability must include corresponding tests.** Place tests in
the appropriate `tests/test_<module>.py` file, or create a new one if adding a module.
Shared fixtures are in `tests/conftest.py`.

Run tests before committing. All tests must pass.

## Release Notes

Release notes are kept in `release-notes/` following the naming convention:

```
RELEASE_NOTES_v<VERSION>.md
```

For example: `RELEASE_NOTES_v26.1.0.md`. Each release note should include sections for
an overview, new features/changes, bug fixes, and testing status. See existing files in
that directory for the format.

### Working release notes

During development, new features, bug fixes, and changes are documented in
`release-notes/RELEASE_NOTES_next.md`. This is a living document that accumulates
entries between releases.

**Release workflow:**

1. During development, add entries to `RELEASE_NOTES_next.md` as work is completed.
2. When ready to release, rename `RELEASE_NOTES_next.md` to
   `RELEASE_NOTES_v<VERSION>.md` and update the heading/overview to reflect the
   final version number.
3. Update the version in `pyproject.toml` and `proteomics_toolkit/__init__.py`.
4. Commit, push, and create a GitHub release to trigger PyPI publishing.
5. After the release, create a fresh `RELEASE_NOTES_next.md` from the template
   to begin tracking the next release.

## Code Style

- Python: PEP 8, `ruff` for linting (`ruff check .`)
- Line length: 120 characters
- Google-style docstrings
- Modern type hints (`float | None` not `Optional[float]`)
- `logging.getLogger(__name__)` at module level; no bare `print()` in library code
- Domain-specific variable names from mass spectrometry are preferred
  (`mz`, `rt`, `precursor_mz`, `fragment_charge`)

## Project Structure

- `proteomics_toolkit/` -- package source (data_import, preprocessing, normalization,
  statistical_analysis, visualization, temporal_clustering, enrichment, validation,
  export, classification)
- `tests/` -- pytest test suite
- `docs/` -- user guide (topic-focused markdown files + tutorial notebook).
  Start at [docs/01-overview.md](docs/01-overview.md), which indexes the
  per-topic pages (data import, metadata, QC plots, normalization,
  statistical analysis, visualization, enrichment, classification,
  export, pitfalls).
- `release-notes/` -- per-version release notes

## Git

- Commit messages in past tense ("Added feature" not "Add feature")
- Small, focused commits
- Include `Co-Authored-By: Claude <noreply@anthropic.com>` when AI-assisted
- Do not push to remote without asking first

## Data Structure Convention

All processed data uses a standardized 5-column annotation prefix:
1. Protein
2. Description
3. Protein Gene
4. UniProt_Accession
5. UniProt_Entry_Name

Sample columns follow after these five annotation columns.
