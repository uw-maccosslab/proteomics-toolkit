"""Generator script for the bundled example dataset.

Run manually (once) to produce:
    - example_proteins.parquet
    - example_peptides.parquet
    - example_metadata.csv
    - example_sequences.json

The generated data is deterministic (fixed seed) and simulates a small
PRISM-style proteomics experiment with two groups (Control, Treatment),
six samples per group, one batch, 80 proteins, and 60 peptides across
the 20 most abundant proteins. Parent protein sequences are generated
synthetically, and peptides are sampled *from within* each parent
sequence at real start positions so the coverage-map visualization can
locate peptides directly.

Usage:
    python -m proteomics_toolkit.datasets._generate
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 26
N_PROTEINS = 80
N_PEPTIDE_PROTEINS = 20  # Proteins that also appear at peptide level
PEPTIDES_PER_PROTEIN = 3
N_SAMPLES_PER_GROUP = 6
BATCH_NAME = "Batch1"
N_DIFFERENTIAL_PROTEINS = 12  # Proteins with a real between-group effect

REAL_GENES = [
    ("sp|P02768|ALBU_HUMAN", "ALB", "Serum albumin"),
    ("sp|P02787|TRFE_HUMAN", "TF", "Serotransferrin"),
    ("sp|P00738|HPT_HUMAN", "HP", "Haptoglobin"),
    ("sp|P01023|A2MG_HUMAN", "A2M", "Alpha-2-macroglobulin"),
    ("sp|P02765|FETUA_HUMAN", "AHSG", "Alpha-2-HS-glycoprotein"),
    ("sp|P02647|APOA1_HUMAN", "APOA1", "Apolipoprotein A-I"),
    ("sp|P02652|APOA2_HUMAN", "APOA2", "Apolipoprotein A-II"),
    ("sp|P04114|APOB_HUMAN", "APOB", "Apolipoprotein B-100"),
    ("sp|P02671|FIBA_HUMAN", "FGA", "Fibrinogen alpha chain"),
    ("sp|P02675|FIBB_HUMAN", "FGB", "Fibrinogen beta chain"),
    ("sp|P02679|FIBG_HUMAN", "FGG", "Fibrinogen gamma chain"),
    ("sp|P00450|CERU_HUMAN", "CP", "Ceruloplasmin"),
    ("sp|P02763|A1AG1_HUMAN", "ORM1", "Alpha-1-acid glycoprotein 1"),
    ("sp|P01009|A1AT_HUMAN", "SERPINA1", "Alpha-1-antitrypsin"),
    ("sp|P01008|ANT3_HUMAN", "SERPINC1", "Antithrombin-III"),
    ("sp|P00747|PLMN_HUMAN", "PLG", "Plasminogen"),
    ("sp|P00734|THRB_HUMAN", "F2", "Prothrombin"),
    ("sp|P02790|HEMO_HUMAN", "HPX", "Hemopexin"),
    ("sp|P10909|CLUS_HUMAN", "CLU", "Clusterin"),
    ("sp|P02749|APOH_HUMAN", "APOH", "Beta-2-glycoprotein 1"),
]


def _make_protein_annotations(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build protein annotation rows; reuse real entries then synthesize extras."""
    rows = []
    for i in range(n):
        if i < len(REAL_GENES):
            uniprot_full, gene, description = REAL_GENES[i]
        else:
            uniprot_full = f"sp|Q{i:05d}|SIM{i}_HUMAN"
            gene = f"SIM{i}"
            description = f"Simulated protein {i}"
        acc = uniprot_full.split("|")[1]
        name = uniprot_full.split("|")[2]
        rows.append(
            {
                "leading_protein": uniprot_full,
                "leading_description": description,
                "leading_gene_name": gene,
                "leading_uniprot_id": acc,
                "leading_name": name,
                "protein_group": acc,
                "n_peptides": int(rng.integers(2, 20)),
                "n_unique_peptides": int(rng.integers(2, 15)),
            }
        )
    return pd.DataFrame(rows)


def _make_sample_matrix(n_features: int, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Draw log-normal intensities for one batch's worth of samples."""
    base = rng.lognormal(mean=14, sigma=1.5, size=n_features)  # per-feature baseline
    noise = rng.normal(loc=0, scale=0.15, size=(n_features, n_samples))
    return np.maximum(base[:, None] * np.exp(noise), 1.0)


def generate(output_dir: Path) -> None:
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # Protein-level table
    # ------------------------------------------------------------------
    protein_annot = _make_protein_annotations(N_PROTEINS, rng)

    sample_ids_control = [f"Ctrl_{i + 1:02d}" for i in range(N_SAMPLES_PER_GROUP)]
    sample_ids_treatment = [f"Trt_{i + 1:02d}" for i in range(N_SAMPLES_PER_GROUP)]
    sample_ids = sample_ids_control + sample_ids_treatment
    sample_cols = [f"{sid}__@__{BATCH_NAME}" for sid in sample_ids]

    protein_intensities = _make_sample_matrix(N_PROTEINS, len(sample_cols), rng)

    # Inject differential expression into the first N_DIFFERENTIAL_PROTEINS proteins
    fold_changes = rng.choice([1.8, 2.5, 0.4, 0.55], size=N_DIFFERENTIAL_PROTEINS)
    treatment_mask = np.array([sid.startswith("Trt_") for sid in sample_ids])
    for i, fc in enumerate(fold_changes):
        protein_intensities[i, treatment_mask] *= fc

    # Inject ~5% random missingness
    missing_mask = rng.random(protein_intensities.shape) < 0.05
    protein_intensities[missing_mask] = np.nan

    protein_df = protein_annot.copy()
    for col, values in zip(sample_cols, protein_intensities.T):
        protein_df[col] = values.astype("float64")

    # ------------------------------------------------------------------
    # Parent protein sequences (top 20 proteins) + peptides drawn from them
    # ------------------------------------------------------------------
    # Amino-acid probabilities approximately match Swiss-Prot abundance.
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    aa_probs = np.array(
        [
            0.083,  # A
            0.014,  # C
            0.055,  # D
            0.068,  # E
            0.040,  # F
            0.071,  # G
            0.023,  # H
            0.060,  # I
            0.058,  # K
            0.096,  # L
            0.024,  # M
            0.040,  # N
            0.047,  # P
            0.039,  # Q
            0.057,  # R
            0.069,  # S
            0.053,  # T
            0.066,  # V
            0.011,  # W
            0.030,  # Y
        ]
    )
    aa_probs = aa_probs / aa_probs.sum()

    protein_sequences = {}
    for pi in range(N_PEPTIDE_PROTEINS):
        seq_len = int(rng.integers(200, 450))
        seq = "".join(rng.choice(alphabet, size=seq_len, p=aa_probs))
        protein_sequences[str(protein_annot.loc[pi, "leading_protein"])] = seq
        # Also register under the bare UniProt accession for alias lookups.
        protein_sequences[str(protein_annot.loc[pi, "leading_uniprot_id"])] = seq

    peptide_rows = []
    for pi in range(N_PEPTIDE_PROTEINS):
        leading_protein = protein_annot.loc[pi, "leading_protein"]
        leading_description = protein_annot.loc[pi, "leading_description"]
        leading_gene = protein_annot.loc[pi, "leading_gene_name"]
        parent_seq = protein_sequences[leading_protein]
        parent_len = len(parent_seq)
        # Sample non-overlapping peptides in DIA style (most peptides do
        # not overlap except for missed cleavages - we don't model those
        # here).
        starts_used: list[tuple[int, int]] = []
        for _ in range(PEPTIDES_PER_PROTEIN):
            for _attempt in range(50):
                length = int(rng.integers(7, 20))
                start = int(rng.integers(1, parent_len - length))  # 1-based
                # Reject if it overlaps an already-placed peptide
                if all(start + length <= s0 or s0 + l0 <= start for s0, l0 in starts_used):
                    starts_used.append((start, length))
                    break
            else:
                # Fallback: accept even if overlapping
                length = int(rng.integers(7, 20))
                start = int(rng.integers(1, parent_len - length))
                starts_used.append((start, length))
            sequence = parent_seq[start - 1 : start - 1 + length]
            peptide_rows.append(
                {
                    "peptide_sequence": sequence,
                    "leading_protein": leading_protein,
                    "leading_description": leading_description,
                    "leading_gene_name": leading_gene,
                    "start_position": start,
                    "peptide_length": length,
                }
            )
    peptide_annot = pd.DataFrame(peptide_rows)

    peptide_intensities = _make_sample_matrix(len(peptide_annot), len(sample_cols), rng)
    # Mirror fold-change pattern for peptides of differential proteins
    peptides_per_diff = {pi: [] for pi in range(N_DIFFERENTIAL_PROTEINS)}
    for row_idx, row in peptide_annot.iterrows():
        for pi in range(N_DIFFERENTIAL_PROTEINS):
            if row["leading_protein"] == protein_annot.loc[pi, "leading_protein"]:
                peptides_per_diff[pi].append(row_idx)
    for pi, fc in enumerate(fold_changes):
        for row_idx in peptides_per_diff.get(pi, []):
            peptide_intensities[row_idx, treatment_mask] *= fc

    pep_missing = rng.random(peptide_intensities.shape) < 0.08
    peptide_intensities[pep_missing] = np.nan

    peptide_df = peptide_annot.copy()
    for col, values in zip(sample_cols, peptide_intensities.T):
        peptide_df[col] = values.astype("float64")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    metadata_df = pd.DataFrame(
        {
            "Replicate": sample_ids,
            "Group": ["Control"] * N_SAMPLES_PER_GROUP + ["Treatment"] * N_SAMPLES_PER_GROUP,
            "Subject": [f"Subj{i + 1}" for i in range(N_SAMPLES_PER_GROUP)] * 2,
            "Batch": [BATCH_NAME] * (2 * N_SAMPLES_PER_GROUP),
        }
    )

    # ------------------------------------------------------------------
    # Write files
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    protein_df.to_parquet(output_dir / "example_proteins.parquet", index=False)
    peptide_df.to_parquet(output_dir / "example_peptides.parquet", index=False)
    metadata_df.to_csv(output_dir / "example_metadata.csv", index=False)
    with open(output_dir / "example_sequences.json", "w") as fh:
        json.dump(protein_sequences, fh, indent=2)

    print(
        f"Wrote {len(protein_df)} proteins, {len(peptide_df)} peptides, "
        f"{len(metadata_df)} samples, {len(protein_sequences)} sequence entries"
    )
    print(f"  -> {output_dir / 'example_proteins.parquet'}")
    print(f"  -> {output_dir / 'example_peptides.parquet'}")
    print(f"  -> {output_dir / 'example_metadata.csv'}")
    print(f"  -> {output_dir / 'example_sequences.json'}")


if __name__ == "__main__":
    generate(Path(__file__).parent)
