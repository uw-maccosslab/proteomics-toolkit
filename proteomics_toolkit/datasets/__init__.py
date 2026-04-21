"""Example datasets bundled with proteomics-toolkit.

Use :func:`load_example_data` to obtain a small, PRISM-style protein and
peptide dataset suitable for tutorials, smoke tests, and demos.

Example
-------
>>> import proteomics_toolkit as ptk
>>> protein_data, peptide_data, metadata, sample_cols = ptk.datasets.load_example_data()
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd

_DATA_DIR = Path(__file__).parent


def load_example_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
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
        20 most abundant proteins).
    metadata : pd.DataFrame
        Sample metadata with ``Replicate``, ``Group``, and ``Subject`` columns.
    sample_columns : list of str
        Sample column names in the parquet files (with batch suffix).

    Example
    -------
    >>> import proteomics_toolkit as ptk
    >>> protein_data, peptide_data, metadata, sample_cols = ptk.datasets.load_example_data()
    >>> print(protein_data.shape)
    (80, 16)
    """
    protein_path = _DATA_DIR / "example_proteins.parquet"
    peptide_path = _DATA_DIR / "example_peptides.parquet"
    metadata_path = _DATA_DIR / "example_metadata.csv"

    protein_data = pd.read_parquet(protein_path)
    peptide_data = pd.read_parquet(peptide_path)
    metadata = pd.read_csv(metadata_path)

    # Sample columns are everything carrying the batch suffix
    from ..data_import import BATCH_SUFFIX_DELIMITER

    sample_columns = [col for col in protein_data.columns if BATCH_SUFFIX_DELIMITER in col]
    return protein_data, peptide_data, metadata, sample_columns
