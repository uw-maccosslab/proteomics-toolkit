"""Example datasets bundled with proteomics-toolkit.

Use :func:`load_example_data` to obtain a small, PRISM-style protein and
peptide dataset suitable for tutorials, smoke tests, and demos.
:func:`load_example_sequences` returns the parent-protein amino-acid
sequences for the 20 proteins that also appear at peptide level, keyed
by ``leading_protein`` (e.g. ``"sp|P02768|ALBU_HUMAN"``) and by UniProt
accession (e.g. ``"P02768"``).

Example
-------
>>> import proteomics_toolkit as ptk
>>> protein_data, peptide_data, metadata, sample_cols, sequences = (
...     ptk.datasets.load_example_data()
... )
>>> albu_seq = sequences["sp|P02768|ALBU_HUMAN"]
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

_DATA_DIR = Path(__file__).parent


def load_example_sequences() -> Dict[str, str]:
    """Return the bundled protein sequences dict.

    Keys are both ``leading_protein`` strings (e.g.
    ``"sp|P02768|ALBU_HUMAN"``) and bare UniProt accessions (e.g.
    ``"P02768"``); both forms point at the same sequence.
    """
    with open(_DATA_DIR / "example_sequences.json") as fh:
        return json.load(fh)


def load_example_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], Dict[str, str]]:
    """Load the bundled PRISM-style example dataset.

    The dataset is a small, simulated proteomics experiment with two groups
    (``Control`` and ``Treatment``), six samples per group, and a single
    batch. Column names follow the Skyline-PRISM ``<sample>__@__<batch>``
    convention so the batch-suffix code paths are exercised. A subset of
    proteins is differentially abundant between groups.

    Returns
    -------
    protein_data : pd.DataFrame
        Protein-level quantitation in PRISM format (columns include
        ``leading_protein``, ``leading_description``, etc.).
    peptide_data : pd.DataFrame
        Peptide-level quantitation (~3 peptides per protein for the
        20 most abundant proteins). Includes ``start_position`` and
        ``peptide_length`` columns for use with
        :func:`~proteomics_toolkit.visualization.plot_peptide_coverage_map`.
    metadata : pd.DataFrame
        Sample metadata with ``Replicate``, ``Group``, and ``Subject`` columns.
    sample_columns : list of str
        Sample column names in the parquet files (with batch suffix).
    protein_sequences : Dict[str, str]
        Parent-protein amino-acid sequences for the 20 proteins that have
        peptide-level coverage. Keyed by both ``leading_protein`` and
        bare UniProt accession.

    Example
    -------
    >>> import proteomics_toolkit as ptk
    >>> protein_data, peptide_data, metadata, sample_cols, sequences = (
    ...     ptk.datasets.load_example_data()
    ... )
    >>> print(protein_data.shape)
    (80, 16)
    """
    protein_path = _DATA_DIR / "example_proteins.parquet"
    peptide_path = _DATA_DIR / "example_peptides.parquet"
    metadata_path = _DATA_DIR / "example_metadata.csv"

    protein_data = pd.read_parquet(protein_path)
    peptide_data = pd.read_parquet(peptide_path)
    metadata = pd.read_csv(metadata_path)
    protein_sequences = load_example_sequences()

    # Sample columns are everything carrying the batch suffix
    from ..data_import import BATCH_SUFFIX_DELIMITER

    sample_columns = [col for col in protein_data.columns if BATCH_SUFFIX_DELIMITER in col]
    return protein_data, peptide_data, metadata, sample_columns, protein_sequences
