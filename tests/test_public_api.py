"""Contract tests: what notebooks, Skyline-PRISM and other callers use keeps working.

The tables are in ``tests/downstream_contract.py``, whose docstring says where they come from and
how to extend them. The first tests check them against the package: every pinned name still exists,
the package's ``__all__`` still lists its pinned names, every call made downstream still binds, and
every config attribute callers set still exists.

The smoke tests then make the calls notebooks make most, on the bundled example data, chained the
way the PRISM-era notebooks chain them, and check the columns and keys callers read from the
results. They check shape, not numbers; the module tests check the numbers. Uses matplotlib's
non-interactive ``Agg`` backend so the plots do not open windows.
"""

import importlib
import inspect
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import proteomics_toolkit as ptk
from proteomics_toolkit.datasets import load_example_data
from proteomics_toolkit.preprocessing import _normalize_group_value
from proteomics_toolkit.statistical_analysis import _fit_count_dependent_prior, _fit_intensity_trend_prior
from tests.downstream_contract import CONFIG_ATTRIBUTES, DOWNSTREAM_ATTRIBUTES, DOWNSTREAM_CALLS, PUBLIC_NAMES

# Notebooks call the plot functions with their default show=True, as the smoke tests do.
pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")

P = inspect.Parameter


def _resolve(path):
    """Return the object at ``path`` as ``from proteomics_toolkit.<module> import <name>`` finds it."""
    module, _, name = path.rpartition(".")
    owner = importlib.import_module(f"proteomics_toolkit.{module}" if module else "proteomics_toolkit")
    return getattr(owner, name)


def _paths(table):
    """Yield the path of every name in a table of space-separated names keyed by module."""
    for module, names in table.items():
        for name in names.split():
            yield f"{module}.{name}" if module else name


def _missing(paths):
    missing = []
    for path in paths:
        try:
            _resolve(path)
        except (ImportError, AttributeError):
            missing.append(path)
    return missing


def _signature_problems(signature, positional, keywords, required):
    """Say why calls passing ``positional`` by position and ``keywords`` by name would now fail.

    Args:
        signature: The callable's current ``inspect.Signature``.
        positional: The parameters callers fill by position, in order, as named when pinned.
        keywords: The keyword names callers pass.
        required: The parameters the callable required when pinned. ``positional`` and ``keywords``
            pool every caller, so only these are known to be passed by all of them.

    Returns:
        A list of problems; empty when every such call still binds.
    """
    problems = []
    params = list(signature.parameters.values())
    # The parameters callers fill by position must still come first, in the same order: a reorder
    # binds every value to the wrong parameter without an error.
    now = []
    for param in params:
        if param.kind in (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD):
            now.append(param.name)
        elif param.kind is P.VAR_POSITIONAL:
            now += [f"*{param.name}"] * len(positional)
            break
        else:
            break
    now = now[: len(positional)]
    if now != positional:
        problems.append(f"callers pass ({', '.join(positional)}) by position, but it now takes ({', '.join(now)})")
    for keyword in keywords:
        try:
            signature.bind_partial(**{keyword: None})
        except TypeError:
            problems.append(f"no longer accepts the keyword {keyword!r}")
    for param in params:
        if (
            param.default is P.empty
            and param.kind not in (P.VAR_POSITIONAL, P.VAR_KEYWORD)
            and param.name not in required
        ):
            problems.append(f"{param.name!r} is required, but was not when callers were pinned; give it a default")
    return problems


class TestNamesStillExist:
    def test_public_names(self):
        missing = _missing(_paths(PUBLIC_NAMES))
        assert not missing, f"Public names removed or moved: {missing}"

    def test_names_callers_read(self):
        missing = _missing(_paths(DOWNSTREAM_ATTRIBUTES))
        assert not missing, f"Names callers read are gone: {missing}"

    def test_names_callers_call(self):
        missing = _missing(DOWNSTREAM_CALLS)
        assert not missing, f"Names callers call are gone: {missing}"

    def test_package_all_keeps_pinned_names(self):
        # A name can stay importable and still drop out of ``__all__``, and then
        # ``from proteomics_toolkit import *`` no longer binds it.
        dropped = sorted(set(PUBLIC_NAMES[""].split()) - set(ptk.__all__))
        assert not dropped, f"Names dropped from proteomics_toolkit.__all__: {dropped}"


class TestCallsStillBind:
    def test_downstream_calls(self):
        problems = []
        for path, (positional, keywords, required) in DOWNSTREAM_CALLS.items():
            try:
                signature = inspect.signature(_resolve(path))
            except (ImportError, AttributeError):
                continue  # reported by TestNamesStillExist
            found = _signature_problems(signature, positional.split(), keywords.split(), required.split())
            problems += [f"{path}: {p}" for p in found]
        assert not problems, "Calls made downstream would now fail:\n" + "\n".join(problems)

    def test_config_attributes_callers_set(self):
        missing = []
        for path, attributes in CONFIG_ATTRIBUTES.items():
            config = _resolve(path)()
            missing += [f"{path}.{name}" for name in attributes.split() if not hasattr(config, name)]
        assert not missing, f"Config attributes callers set are gone, so their settings would be ignored: {missing}"


class TestSignatureCheck:
    """The check above catches each way a signature change breaks a caller, and passes the safe ones."""

    @staticmethod
    def pinned(data, sample_columns=None, method="median"):
        pass

    def problems(self, changed):
        return _signature_problems(inspect.signature(changed), ["data"], ["sample_columns", "method"], ["data"])

    def test_unchanged_signature_passes(self):
        assert self.problems(self.pinned) == []

    def test_keyword_only_parameter_added_at_end_passes(self):
        def changed(data, sample_columns=None, method="median", *, span=0.75):
            pass

        assert self.problems(changed) == []

    def test_renamed_keyword_fails(self):
        def changed(data, columns=None, method="median"):
            pass

        assert self.problems(changed) == ["no longer accepts the keyword 'sample_columns'"]

    def test_reordered_positional_parameters_fail(self):
        def changed(sample_columns=None, data=None, method="median"):
            pass

        assert self.problems(changed) == [
            "callers pass (data) by position, but it now takes (sample_columns)",
        ]

    def test_parameter_made_keyword_only_fails(self):
        def changed(*, data, sample_columns=None, method="median"):
            pass

        assert self.problems(changed) == ["callers pass (data) by position, but it now takes ()"]

    def test_new_required_parameter_fails(self):
        def changed(data, sample_columns=None, method="median", *, span):
            pass

        assert self.problems(changed) == ["'span' is required, but was not when callers were pinned; give it a default"]

    def test_optional_parameter_made_required_fails(self):
        # Some caller passes method, so it is among the pinned keywords, but a caller that leaves it
        # out would now fail.
        def changed(data, sample_columns=None, *, method):
            pass

        assert self.problems(changed) == [
            "'method' is required, but was not when callers were pinned; give it a default"
        ]

    def test_required_parameter_made_optional_passes(self):
        def changed(data=None, sample_columns=None, method="median"):
            pass

        assert self.problems(changed) == []

    def test_var_keyword_accepts_any_keyword(self):
        def changed(data, **kwargs):
            pass

        assert self.problems(changed) == []


# ---------------------------------------------------------------------------
# Smoke tests on the bundled example data
# ---------------------------------------------------------------------------

# PRISM's annotation columns, and the toolkit's standard names notebooks rename them to.
PRISM_ANNOTATIONS = [
    "leading_protein",
    "leading_description",
    "leading_gene_name",
    "leading_uniprot_id",
    "leading_name",
]
ANNOTATION_COLUMNS = ["Protein", "Description", "Protein Gene", "UniProt_Accession", "UniProt_Entry_Name"]
GROUPS = ["Control", "Treatment"]
GROUP_COLORS = {"Control": "#1f77b4", "Treatment": "#ff7f0e"}

# Result columns notebooks read from every two-group comparison.
RESULT_COLUMNS = ["Protein", "Gene", "logFC", "AveExpr", "t", "P.Value", "adj.P.Val"]
# What Skyline-PRISM's end-to-end fixture also reads from a moderated run.
MODERATED_COLUMNS = ["residual_s2", "residual_df", "posterior_df"]
INTENSITY_TREND_COLUMNS = ["intensity_trend_shape", "intensity_trend_level"]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def prism_files(tmp_path_factory):
    """The example dataset written out as a PRISM run writes it: a protein parquet and a metadata CSV."""
    protein_data, _, metadata, _, _ = load_example_data()
    folder = tmp_path_factory.mktemp("prism")
    protein_data.to_parquet(folder / "corrected_proteins.parquet")
    metadata.to_csv(folder / "sample_metadata.csv", index=False)
    return str(folder / "corrected_proteins.parquet"), str(folder / "sample_metadata.csv")


@pytest.fixture
def run(prism_files):
    """The example run loaded and arranged the way the PRISM notebooks arrange theirs."""
    protein_data, metadata, sample_columns = ptk.load_prism_data(*prism_files)
    short_names = ptk.strip_batch_suffix(sample_columns)
    by_replicate = metadata.set_index("Replicate")
    sample_metadata = {
        column: {"Replicate": name, **by_replicate.loc[name].to_dict()} for column, name in short_names.items()
    }
    annotations = protein_data[PRISM_ANNOTATIONS].set_axis(ANNOTATION_COLUMNS, axis=1).reset_index(drop=True)
    data = pd.concat([annotations, protein_data[sample_columns].reset_index(drop=True)], axis=1)
    data.index = data["Protein"]
    annotations = annotations.set_index("Protein")
    annotations.index.name = None
    annotations["Protein"] = annotations.index
    return SimpleNamespace(
        protein_data=protein_data,
        metadata=metadata,
        sample_columns=sample_columns,
        short_names=short_names,
        sample_metadata=sample_metadata,
        data=data,
        annotations=annotations,
    )


def _config(**settings):
    config = ptk.StatisticalConfig()
    config.analysis_type = "unpaired"
    config.group_column = "Group"
    config.group_labels = GROUPS
    for name, value in settings.items():
        setattr(config, name, value)
    return config


class TestLoading:
    def test_load_prism_data(self, prism_files):
        protein_data, metadata, sample_columns = ptk.load_prism_data(*prism_files)
        assert (len(protein_data), len(sample_columns)) == (80, 12)
        assert set(PRISM_ANNOTATIONS + ["protein_group", "n_peptides"]) <= set(protein_data.columns)
        assert {"Replicate", "Group"} <= set(metadata.columns)

    def test_strip_batch_suffix_gives_the_metadata_names(self, run):
        assert list(run.short_names) == run.sample_columns
        assert set(run.short_names.values()) == set(run.metadata["Replicate"])

    def test_clean_sample_names(self, run):
        cleaned = ptk.clean_sample_names(run.sample_columns, auto_detect=True)
        assert set(cleaned) == set(run.sample_columns)

    def test_validate_metadata_data_consistency(self, run):
        report = ptk.validate_metadata_data_consistency(
            metadata=run.metadata,
            metadata_sample_names=list(run.metadata["Replicate"]),
            protein_columns=list(run.short_names.values()),
            control_column="Group",
            control_labels=[],
            verbose=False,
        )
        assert {"is_valid", "errors", "diagnostics"} <= set(report)

    def test_classify_samples(self, run):
        result = ptk.classify_samples(
            run.sample_metadata,
            group_column="Group",
            group_labels=GROUPS,
            control_column="Group",
            control_labels=["__none__"],
        )
        assert len(result) == 5
        sample_metadata = result[3]
        assert set(sample_metadata) == set(run.sample_columns)

    @pytest.mark.parametrize(
        "value, expected", [(80.0, 80), ("80", 80), ("2.5", 2.5), ("Week 4", "Week 4"), (None, "Unknown")]
    )
    def test_normalize_group_value(self, value, expected):
        assert _normalize_group_value(value) == expected


class TestSkylineFormat:
    """The older Skyline-export path: parse identifiers, filter, then build the standard structure."""

    def test_preprocessing_chain(self, run):
        skyline = pd.concat(
            [
                pd.DataFrame(
                    {
                        "Protein": run.protein_data["leading_protein"],
                        "Protein Description": run.protein_data["leading_description"],
                        "Protein Gene": run.protein_data["leading_gene_name"],
                    }
                ),
                run.protein_data[run.sample_columns].rename(columns=run.short_names),
            ],
            axis=1,
        )
        sample_columns = list(run.short_names.values())
        sample_metadata = {run.short_names[column]: meta for column, meta in run.sample_metadata.items()}

        processed = ptk.preprocessing.parse_protein_identifiers(skyline)
        processed = ptk.preprocessing.parse_gene_and_description(processed)
        ptk.preprocessing.assess_data_completeness(processed, sample_columns, sample_metadata)
        filtered = ptk.preprocessing.filter_proteins_by_completeness(processed, sample_columns, min_detection_rate=0.5)
        standardized = ptk.preprocessing.create_standard_data_structure(filtered)

        assert list(standardized.columns[:5]) == ANNOTATION_COLUMNS
        assert set(sample_columns) <= set(standardized.columns)
        assert len(standardized) == len(filtered)


class TestNormalization:
    @pytest.mark.parametrize(
        "name, settings",
        [
            ("median_normalize", {}),
            ("vsn_normalize", {"optimize_params": False}),
            ("quantile_normalize", {}),
            ("mad_normalize", {}),
            ("z_score_normalize", {}),
            ("rlr_normalize", {}),
            ("loess_normalize", {"span": 0.75}),
        ],
    )
    def test_keeps_rows_and_columns(self, run, name, settings):
        normalized = getattr(ptk.normalization, name)(run.data, sample_columns=run.sample_columns, **settings)
        assert len(normalized) == len(run.data)
        assert list(normalized.columns[:5]) == ANNOTATION_COLUMNS
        assert set(run.sample_columns) <= set(normalized.columns)

    def test_normalization_statistics(self, run):
        normalized = ptk.normalization.median_normalize(run.data, sample_columns=run.sample_columns)
        # Notebooks pass the sample columns only, as the docstring asks.
        stats = ptk.normalization.calculate_detailed_normalization_stats(
            original_data=run.data[run.sample_columns],
            normalized_data=normalized[run.sample_columns],
            method="median",
            sample_metadata=run.sample_metadata,
            control_labels=["Control"],
        )
        assert {"overall", "control_samples"} <= set(stats)

    def test_negative_values(self, run):
        z_scored = ptk.normalization.z_score_normalize(run.data, sample_columns=run.sample_columns)
        report = ptk.analyze_negative_values(
            data=z_scored, normalization_method="z_score", sample_columns=run.sample_columns
        )
        assert {"negative_count", "negative_percentage"} <= set(report)
        handled = ptk.normalization.handle_negative_values(z_scored, sample_columns=run.sample_columns)
        assert len(handled) == len(z_scored)


class TestPlots:
    def test_qc_plots(self, run):
        ptk.plot_box_plot(run.protein_data, run.sample_columns, run.sample_metadata, group_colors=GROUP_COLORS)
        ptk.plot_pca(
            run.protein_data, run.sample_columns, run.sample_metadata, group_colors=GROUP_COLORS, log_transform=True
        )
        assert len(plt.get_fignums()) == 2

    def test_control_cv_distribution(self, run):
        cvs = ptk.visualization.plot_control_cv_distribution(
            data=run.data,
            sample_columns=run.sample_columns,
            sample_metadata=run.sample_metadata,
            control_column="Group",
            control_labels=["Control"],
        )
        assert "Control" in cvs


class TestStatistics:
    @pytest.mark.parametrize("method, extra_columns", [("moderated_linear_model", MODERATED_COLUMNS), ("welch_t", [])])
    def test_two_group_comparison(self, run, method, extra_columns):
        config = _config(statistical_test_method=method)
        results = ptk.run_comprehensive_statistical_analysis(
            run.data, run.sample_metadata, config, protein_annotations=run.annotations
        )
        assert len(results) == len(run.data)
        assert set(RESULT_COLUMNS + extra_columns) <= set(results.columns)

        assert isinstance(ptk.display_analysis_summary(results, config), dict)
        ptk.plot_volcano(results, fc_threshold=1.0, p_threshold=0.05)
        assert len(plt.get_fignums()) == 1

    def test_intensity_trend_run_skyline_prism_makes(self, run):
        """The run behind Skyline-PRISM's end-to-end fixture, and the diagnostics notebooks plot from it."""
        config = _config(
            statistical_test_method="moderated_linear_model",
            moderation="intensity-trend",
            variance_prior_group_column="Group",
            variance_prior_groups=GROUPS,
            log_transform_before_stats="auto",
            log_base="log2",
            normalization_method="prism",
            correction_method="fdr_bh",
            subject_column=None,
        )
        results = ptk.run_comprehensive_statistical_analysis(
            normalized_data=run.data,
            sample_metadata=run.sample_metadata,
            config=config,
            protein_annotations=run.annotations,
        )
        assert set(RESULT_COLUMNS + MODERATED_COLUMNS + INTENSITY_TREND_COLUMNS) <= set(results.columns)
        assert isinstance(ptk.plot_variance_vs_intensity(results), Figure)
        assert isinstance(ptk.get_intensity_trend_points(results), pd.DataFrame)

    def test_prepare_metadata_dataframe(self, run):
        metadata = ptk.statistical_analysis.prepare_metadata_dataframe(
            run.sample_metadata, run.sample_columns, _config()
        )
        assert len(metadata) == len(run.sample_columns)

    def test_config_dict_keeps_notebook_settings(self):
        settings = {"normalization_method": "median", "group_column": "Group", "group_labels": GROUPS}
        config_dict = ptk.create_config_dict_from_notebook_vars(**settings)
        assert set(settings) <= set(config_dict)


class TestSkylinePrismReferences:
    """Skyline-PRISM's parity fixtures call these private fits directly; these mirror its calls."""

    def test_intensity_trend_prior_takes_prisms_bare_config(self):
        # Exactly the attributes PRISM's fixture generator gives its config, and no others: a fit
        # that starts reading another attribute breaks PRISM.
        class Cfg:
            analysis_type = "unpaired"
            group_column = "Group"
            group_labels = ["A", "B"]
            log_base = "log2"
            variance_prior_group_column = None
            variance_prior_groups = None
            paired_column = None
            time_column = None
            dose_column = None

        rng = np.random.default_rng(29)
        base = np.array([1.0e4, 3.0e4, 1.0e5, 4.0e5, 9.0e5, 2.0e6, 5.0e6, 1.2e7])
        raw = np.abs(base[:, None] * (1.0 + 0.15 * rng.normal(size=(8, 8))))
        raw[:, 4:] *= 1.4
        samples = [f"S{i}" for i in range(8)]
        features = [f"f{i}" for i in range(8)]

        prior, _ = _fit_intensity_trend_prior(
            {"features": features},
            pd.DataFrame(raw, index=features, columns=samples),
            pd.DataFrame({"Sample": samples, "Group": ["A"] * 4 + ["B"] * 4}),
            Cfg(),
        )
        prior = np.asarray(prior, dtype=float)
        assert prior.shape == (8,)
        assert np.all(np.isfinite(prior)) and np.all(prior > 0)

    def test_count_dependent_prior(self):
        rng = np.random.default_rng(101)
        counts = rng.integers(1, 30, size=40).astype(float)
        variances = rng.gamma(4.0, 0.05, size=40) / np.sqrt(counts)
        fit = {"features": [f"f{i}" for i in range(40)], "s2": variances}

        prior = _fit_count_dependent_prior(fit, counts)
        if isinstance(prior, tuple):  # PRISM takes the array from either return form
            prior = prior[0]
        prior = np.asarray(prior, dtype=float)
        assert prior.shape == (40,)
        assert np.all(np.isfinite(prior)) and np.all(prior > 0)
