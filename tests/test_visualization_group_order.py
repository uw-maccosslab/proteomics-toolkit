"""Plots must draw the same figure on every run.

``plot_pca`` and ``plot_sample_correlation_heatmap`` took their group order from
``list(set(groups))``, and ``plot_control_group_correlation_analysis`` took its sample
order from ``list(set(found_samples))``. Python randomises string hashing per process
(``PYTHONHASHSEED``), so the group-to-colour mapping, the legend order and the sample
order on the axes changed from one run to the next. Within one process the order is
fixed, so the check runs the plots in subprocesses with different hash seeds and
compares the rendered images.
"""

import hashlib
import json
import os
import subprocess
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteomics_toolkit.visualization import _ordered_groups, plot_pca

GROUPS = ["Xray", "Control", "reference", "qc"]
SAMPLES = [f"{g}_{i}" for g in GROUPS for i in range(3)]

_SCRIPT = textwrap.dedent(
    """
    import hashlib, io, json, sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from proteomics_toolkit.visualization import (
        plot_control_group_correlation_analysis,
        plot_pca,
        plot_sample_correlation_heatmap,
    )

    groups = json.loads(sys.argv[1])
    samples = [f"{g}_{i}" for g in groups for i in range(3)]
    metadata = {s: {"Group": s.split("_")[0]} for s in samples}
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.lognormal(15, 1, size=(30, len(samples))), columns=samples)
    log2 = np.log2(data)

    def digest(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", metadata={"Software": None})
        return hashlib.sha256(buf.getvalue()).hexdigest()

    out = {}
    fig = plot_pca(data, samples, metadata, log_transform=True, show=False)
    out["pca_legend"] = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    out["pca"] = digest(fig)
    out["heatmap"] = digest(plot_sample_correlation_heatmap(log2, samples, metadata, show=False))
    plt.close("all")
    plot_control_group_correlation_analysis(log2, samples, metadata, {"QC": ["qc_"]})
    out["control"] = [digest(plt.figure(n)) for n in plt.get_fignums()]
    print(json.dumps(out))
    """
)


def _run(seed: int) -> dict:
    env = {**os.environ, "PYTHONHASHSEED": str(seed), "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, json.dumps(GROUPS)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_figures_do_not_depend_on_the_hash_seed():
    runs = [_run(seed) for seed in (1, 2, 3, 4)]
    assert runs[0]["control"], "the control-group plot drew no figure"
    for run in runs[1:]:
        assert run == runs[0]


def test_pca_legend_is_sorted():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.lognormal(15, 1, size=(30, len(SAMPLES))), columns=SAMPLES)
    metadata = {s: {"Group": s.split("_")[0]} for s in SAMPLES}
    fig = plot_pca(data, SAMPLES, metadata, log_transform=True, show=False)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == sorted(GROUPS)
    plt.close("all")


def test_ordered_groups_sorts_comparable_labels_naturally():
    assert _ordered_groups(["b", "a", "b", "c"]) == ["a", "b", "c"]
    # Numbers sort as numbers, not as text (which would give 1, 10, 2).
    assert _ordered_groups([10, 2, 1, 2]) == [1, 2, 10]


def test_ordered_groups_handles_mixed_types():
    # Ints and strings cannot be compared, so they sort by text.
    assert _ordered_groups(["b", 1, "a"]) == [1, "a", "b"]
    # 1 and "1" have the same text, so they keep first-seen order rather than set order.
    assert _ordered_groups(["1", 1]) == ["1", 1]
    assert _ordered_groups([1, "1"]) == [1, "1"]


def test_digest_helper_is_stable():
    """Guards the subprocess comparison: the same figure rendered twice hashes the same."""
    import io

    def digest():
        fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 0])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", metadata={"Software": None})
        plt.close(fig)
        return hashlib.sha256(buf.getvalue()).hexdigest()

    assert digest() == digest()
