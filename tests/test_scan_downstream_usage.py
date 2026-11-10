"""Tests for ``tests/scan_downstream_usage.py``, which finds usage the contract tables do not pin yet."""

import json
import textwrap

from tests import scan_downstream_usage as scanner


def _scan(tmp_path, *cells):
    """Scan one file: a script when given one source, a notebook with one code cell per source otherwise."""
    if len(cells) == 1:
        path = tmp_path / "script.py"
        path.write_text(textwrap.dedent(cells[0]), encoding="utf-8")
    else:
        path = tmp_path / "notebook.ipynb"
        notebook = {"cells": [{"cell_type": "code", "source": textwrap.dedent(cell)} for cell in cells]}
        path.write_text(json.dumps(notebook), encoding="utf-8")
    usage = scanner.Usage()
    usage.scan_file(str(path))
    return usage


class TestStarImports:
    def test_star_import_of_the_package_binds_its_names(self, tmp_path):
        usage = _scan(
            tmp_path,
            """
            from proteomics_toolkit import *
            median_normalize(df, sample_columns=cols)
            """,
        )
        assert usage.max_positional["median_normalize"] == 1
        assert usage.keywords["median_normalize"] == {"sample_columns"}
        assert "*" not in usage.names

    def test_star_import_of_a_module_skips_what_the_module_imported(self, tmp_path):
        usage = _scan(
            tmp_path,
            """
            from proteomics_toolkit.normalization import *
            np.mean(x)
            median_normalize(df)
            """,
        )
        assert usage.max_positional["normalization.median_normalize"] == 1
        assert not [path for path in usage.max_positional if "np" in path.split(".")]

    def test_star_import_of_a_removed_module_is_reported_broken(self, tmp_path):
        usage = _scan(tmp_path, "from proteomics_toolkit.differential import *\n")
        lines = scanner.report(usage)
        assert lines[:2] == ["Already broken: these names no longer exist", "    differential"]


class TestBindings:
    def test_annotated_config_assignment_is_checked(self, tmp_path):
        usage = _scan(
            tmp_path,
            """
            import proteomics_toolkit as ptk
            cfg: ptk.StatisticalConfig = ptk.StatisticalConfig()
            cfg.multiple_testing_correction = "fdr_bh"
            """,
        )
        assert any("do nothing" in line and "multiple_testing_correction" in line for line in scanner.report(usage))

    def test_reused_variable_is_read_in_source_order(self, tmp_path):
        usage = _scan(
            tmp_path,
            """
            import proteomics_toolkit as ptk
            config = ptk.StatisticalConfig()
            config.moderation = "global"
            config = ptk.EnrichmentConfig()
            config.top_n = 5
            """,
        )
        assert dict(usage.attributes) == {"StatisticalConfig": {"moderation"}, "EnrichmentConfig": {"top_n"}}

    def test_each_function_has_its_own_scope(self, tmp_path):
        usage = _scan(
            tmp_path,
            """
            import proteomics_toolkit as ptk

            def statistics():
                config = ptk.StatisticalConfig()
                config.moderation = "global"

            def enrichment():
                config = ptk.EnrichmentConfig()
                config.top_n = 5
            """,
        )
        assert dict(usage.attributes) == {"StatisticalConfig": {"moderation"}, "EnrichmentConfig": {"top_n"}}

    def test_parameter_hides_a_variable_of_the_same_name(self, tmp_path):
        usage = _scan(
            tmp_path,
            """
            import proteomics_toolkit as ptk
            config = ptk.StatisticalConfig()

            def pytest_configure(config):
                config.addinivalue_line("markers", "slow")
            """,
        )
        assert dict(usage.attributes) == {}

    def test_name_rebound_to_something_else_no_longer_names_the_package(self, tmp_path):
        usage = _scan(
            tmp_path,
            """
            from proteomics_toolkit import *
            validation = load_rows()
            validation.head()
            """,
        )
        assert "validation.head" not in usage.max_positional
        assert "validation.head" not in usage.names

    def test_notebook_cells_share_one_scope(self, tmp_path):
        usage = _scan(
            tmp_path,
            "import proteomics_toolkit as ptk",
            "cfg = ptk.StatisticalConfig()",
            "cfg.moderation = 'global'\nptk.median_normalize(df)",
        )
        assert usage.files == 1
        assert dict(usage.attributes) == {"StatisticalConfig": {"moderation"}}
        assert usage.max_positional["median_normalize"] == 1


class TestReport:
    def test_new_call_is_printed_with_its_required_parameters(self, tmp_path, monkeypatch):
        unpinned = {path: entry for path, entry in scanner.DOWNSTREAM_CALLS.items() if path != "median_normalize"}
        monkeypatch.setattr(scanner, "DOWNSTREAM_CALLS", unpinned)
        usage = _scan(
            tmp_path,
            """
            import proteomics_toolkit as ptk
            ptk.median_normalize(df, sample_columns=cols)
            """,
        )
        assert '    "median_normalize": ("data", "sample_columns", "data"),' in scanner.report(usage)
