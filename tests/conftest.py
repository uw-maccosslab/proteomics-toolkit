"""Shared fixtures for proteomics_toolkit tests."""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Annotation column helpers
# ---------------------------------------------------------------------------

ANNOTATION_COLS = [
    "Protein",
    "Description",
    "Protein Gene",
    "UniProt_Accession",
    "UniProt_Entry_Name",
]

SAMPLE_NAMES = ["Sample_A1", "Sample_A2", "Sample_A3", "Sample_B1", "Sample_B2", "Sample_B3"]


# ---------------------------------------------------------------------------
# Core data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_columns():
    """Return the standard sample column names used across tests."""
    return list(SAMPLE_NAMES)


@pytest.fixture
def raw_protein_data():
    """Minimal protein quantitation DataFrame without annotation columns.

    Five proteins, six samples (two groups of three).
    Values are realistic positive intensities.
    """
    rng = np.random.default_rng(42)
    n_proteins = 5

    data = {
        "Protein": [
            "sp|P12345|ALBU_HUMAN",
            "sp|P67890|TRFE_HUMAN",
            "tr|Q11111|Q11111_HUMAN",
            "sp|P99999|CYTC_HUMAN",
            "sp|O75533|SF3B1_HUMAN",
        ],
        "Protein Gene": ["ALB", "TF", "UNKNOWN1", "CYCS", "SF3B1"],
        "Description": [
            "Albumin OS=Homo sapiens OX=9606 GN=ALB PE=1 SV=2",
            "Serotransferrin OS=Homo sapiens OX=9606 GN=TF PE=1 SV=3",
            "Uncharacterized protein OS=Homo sapiens OX=9606 PE=4 SV=1",
            "Cytochrome c OS=Homo sapiens OX=9606 GN=CYCS PE=1 SV=2",
            "Splicing factor 3B subunit 1 OS=Homo sapiens OX=9606 GN=SF3B1 PE=1 SV=3",
        ],
    }
    for sample in SAMPLE_NAMES:
        data[sample] = rng.uniform(1e5, 1e7, size=n_proteins)

    return pd.DataFrame(data)


@pytest.fixture
def standardized_protein_data():
    """Protein data with the 5-column annotation structure expected by normalization.

    Columns: Protein, Description, Protein Gene, UniProt_Accession,
             UniProt_Entry_Name, then six sample columns.
    """
    rng = np.random.default_rng(42)
    n_proteins = 5

    data = {
        "Protein": [
            "sp|P12345|ALBU_HUMAN",
            "sp|P67890|TRFE_HUMAN",
            "tr|Q11111|Q11111_HUMAN",
            "sp|P99999|CYTC_HUMAN",
            "sp|O75533|SF3B1_HUMAN",
        ],
        "Description": [
            "Albumin",
            "Serotransferrin",
            "Uncharacterized protein",
            "Cytochrome c",
            "Splicing factor 3B subunit 1",
        ],
        "Protein Gene": ["ALB", "TF", "UNKNOWN1", "CYCS", "SF3B1"],
        "UniProt_Accession": ["P12345", "P67890", "Q11111", "P99999", "O75533"],
        "UniProt_Entry_Name": [
            "ALBU_HUMAN",
            "TRFE_HUMAN",
            "Q11111_HUMAN",
            "CYTC_HUMAN",
            "SF3B1_HUMAN",
        ],
    }
    for sample in SAMPLE_NAMES:
        data[sample] = rng.uniform(1e5, 1e7, size=n_proteins)

    return pd.DataFrame(data)


@pytest.fixture
def sample_metadata():
    """Metadata DataFrame with Replicate, Group, and Subject columns."""
    return pd.DataFrame(
        {
            "Replicate": SAMPLE_NAMES,
            "Group": ["Control", "Control", "Control", "Treatment", "Treatment", "Treatment"],
            "Subject": ["S1", "S2", "S3", "S1", "S2", "S3"],
        }
    )


@pytest.fixture
def sample_metadata_dict():
    """Metadata as a dict-of-dicts, keyed by sample name."""
    groups = ["Control", "Control", "Control", "Treatment", "Treatment", "Treatment"]
    subjects = ["S1", "S2", "S3", "S1", "S2", "S3"]
    return {
        name: {"Group": g, "Subject": s}
        for name, g, s in zip(SAMPLE_NAMES, groups, subjects)
    }


@pytest.fixture
def tmp_csv_files(tmp_path, raw_protein_data, sample_metadata):
    """Write raw_protein_data and sample_metadata to temporary CSV files.

    Returns a dict with keys 'protein_file' and 'metadata_file'.
    """
    protein_file = tmp_path / "proteins.csv"
    metadata_file = tmp_path / "metadata.csv"
    raw_protein_data.to_csv(protein_file, index=False)
    sample_metadata.to_csv(metadata_file, index=False)
    return {"protein_file": str(protein_file), "metadata_file": str(metadata_file)}


@pytest.fixture
def fold_change_matrix():
    """Per-subject fold-change matrix for classification tests.

    Rows = subjects, columns = proteins.  Group A has generally positive
    fold-changes; Group B has generally negative.
    """
    rng = np.random.default_rng(99)
    subjects = [f"Subj_{i}" for i in range(10)]
    proteins = [f"Protein_{i}" for i in range(20)]
    fc = rng.normal(loc=0, scale=1, size=(10, 20))
    # Make first 5 subjects trend positive and last 5 negative
    fc[:5, :] += 0.8
    fc[5:, :] -= 0.8
    return pd.DataFrame(fc, index=subjects, columns=proteins)


@pytest.fixture
def group_labels():
    """Binary group labels matching fold_change_matrix subjects."""
    subjects = [f"Subj_{i}" for i in range(10)]
    labels = ["Responder"] * 5 + ["NonResponder"] * 5
    return pd.Series(labels, index=subjects, name="Group")
