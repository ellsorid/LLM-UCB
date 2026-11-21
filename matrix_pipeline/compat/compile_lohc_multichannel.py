
# ============================================================
# compile_lohc_multichannel.py
#
# LOHC SMILES -> 12-Channel Molecular Matrix
#
# One molecule = one N x N x 12 tensor
#
# Atom channels (0-6): diagonal only
# Bond channels (7-11): off-diagonal only
#
# Default state: lean / unsaturated
#
# Dependencies:
#   numpy
#   pandas
#   rdkit
#
# ============================================================

import os
import json
import argparse
import traceback

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.warning")


# ============================================================
# 1. Matrix configuration
# ============================================================

N_CHANNELS = 12

ATOM_CHANNELS = 7
BOND_CHANNELS = 5

CHANNEL_NAMES = [
    "atomic_number",
    "formal_charge",
    "total_hydrogens",
    "aromatic_atom",
    "isotope_mass",
    "radical_electrons",
    "atom_chirality",

    "bond_order",
    "aromatic_bond",
    "conjugated_bond",
    "ring_bond",
    "bond_stereochemistry"
]

# ------------------------------------------------------------
# Atom chirality encoding
#
# 0 = unspecified / absent
# 1 = R
# 2 = S
# 3 = r
# 4 = s
# 5 = specified but CIP unavailable / other
# ------------------------------------------------------------

ATOM_STEREO_MAP = {
    "R": 1,
    "S": 2,
    "r": 3,
    "s": 4
}

# ------------------------------------------------------------
# Bond stereochemistry encoding
#
# 0 = unspecified / absent
# 1 = E
# 2 = Z
# 3 = CIS
# 4 = TRANS
# 5 = ANY
# 6 = OTHER
# ------------------------------------------------------------

BOND_STEREO_MAP = {
    "STEREOE": 1,
    "STEREOZ": 2,
    "STEREOCIS": 3,
    "STEREOTRANS": 4,
    "STEREOANY": 5
}


# ============================================================
# 2. SMILES parsing
# ============================================================

def parse_smiles(smiles):

    smiles = str(smiles).strip()

    if not smiles:
        raise ValueError("Empty SMILES")

    mol = Chem.MolFromSmiles(
        smiles,
        sanitize=True
    )

    if mol is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    # Preserve heavy-atom molecular representation.
    mol = Chem.RemoveHs(mol)

    if mol.GetNumAtoms() == 0:
        raise ValueError("No atoms")

    # Assign stereochemistry.
    Chem.AssignStereochemistry(
        mol,
        cleanIt=True,
        force=True
    )

    # Prefer modern CIP assignment.
    try:

        from rdkit.Chem import rdCIPLabeler

        rdCIPLabeler.AssignCIPLabels(mol)

    except Exception:

        # Fall back to RDKit's standard assignment.
        Chem.AssignStereochemistry(
            mol,
            cleanIt=True,
            force=True
        )

    return mol


# ============================================================
# 3. Atom chirality
# ============================================================

def get_atom_chirality(atom):

    if atom.HasProp("_CIPCode"):

        code = atom.GetProp("_CIPCode")

        if code in ATOM_STEREO_MAP:

            return ATOM_STEREO_MAP[code]

    if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:

        return 5

    return 0


# ============================================================
# 4. Bond stereochemistry
# ============================================================

def get_bond_stereochemistry(bond):

    stereo_name = str(
        bond.GetStereo()
    )

    if stereo_name in BOND_STEREO_MAP:

        return BOND_STEREO_MAP[stereo_name]

    if stereo_name == "STEREONONE":

        return 0

    return 6


# ============================================================
# 5. Bond order
# ============================================================

def get_bond_order(bond):

    if bond.GetIsAromatic():

        return 1.5

    return float(
        bond.GetBondTypeAsDouble()
    )


# ============================================================
# 6. Build one molecular matrix
# ============================================================

def smiles_to_matrix(smiles):

    """
    Convert one SMILES into:

        M: [N, N, 12]

    Channels 0-6:
        Atom information on diagonal.

    Channels 7-11:
        Bond information off diagonal.
    """

    mol = parse_smiles(smiles)

    n = mol.GetNumAtoms()

    M = np.zeros(
        (n, n, N_CHANNELS),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # A. Atom information: diagonal
    # --------------------------------------------------------

    for i, atom in enumerate(
        mol.GetAtoms()
    ):

        atom_features = [

            # A1: Atomic number
            atom.GetAtomicNum(),

            # A2: Formal charge
            atom.GetFormalCharge(),

            # A3: Total hydrogen count
            atom.GetTotalNumHs(
                includeNeighbors=True
            ),

            # A4: Aromatic atom
            int(atom.GetIsAromatic()),

            # A5: Isotope mass
            atom.GetIsotope(),

            # A6: Radical electrons
            atom.GetNumRadicalElectrons(),

            # A7: Atom stereochemistry
            get_atom_chirality(atom)

        ]

        M[i, i, 0:7] = np.asarray(
            atom_features,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # B. Bond information: off diagonal
    # --------------------------------------------------------

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()

        j = bond.GetEndAtomIdx()

        bond_features = [

            # B1: Bond order
            get_bond_order(bond),

            # B2: Aromatic bond
            int(bond.GetIsAromatic()),

            # B3: Conjugated bond
            int(bond.GetIsConjugated()),

            # B4: Ring bond
            int(bond.IsInRing()),

            # B5: Bond stereochemistry
            get_bond_stereochemistry(bond)

        ]

        # Symmetric storage.
        M[i, j, 7:12] = bond_features

        M[j, i, 7:12] = bond_features

    validate_matrix(M)

    return M


# ============================================================
# 7. Matrix validation
# ============================================================

def validate_matrix(M):

    if M.ndim != 3:

        raise ValueError(
            "Matrix must have 3 dimensions"
        )

    n1, n2, channels = M.shape

    if n1 != n2:

        raise ValueError(
            "Matrix must be square"
        )

    if channels != N_CHANNELS:

        raise ValueError(
            "Expected 12 channels"
        )

    if not np.isfinite(M).all():

        raise ValueError(
            "Matrix contains NaN or Inf"
        )

    # --------------------------------------------------------
    # Atom channels must be diagonal only.
    # --------------------------------------------------------

    offdiag = ~np.eye(
        n1,
        dtype=bool
    )

    if not np.allclose(
        M[offdiag, 0:7],
        0
    ):

        raise ValueError(
            "Atom information found outside diagonal"
        )

    # --------------------------------------------------------
    # Bond channels must have zero diagonal.
    # --------------------------------------------------------

    diag = np.eye(
        n1,
        dtype=bool
    )

    if not np.allclose(
        M[diag, 7:12],
        0
    ):

        raise ValueError(
            "Bond information found on diagonal"
        )

    # --------------------------------------------------------
    # Bond matrix symmetry.
    # --------------------------------------------------------

    if not np.allclose(
        M[:, :, 7:12],
        M[:, :, 7:12].transpose(1, 0, 2)
    ):

        raise ValueError(
            "Bond matrix is not symmetric"
        )

    # --------------------------------------------------------
    # Atomic number validation.
    # --------------------------------------------------------

    atomic_numbers = M[diag, 0]

    if np.any(atomic_numbers <= 0):

        raise ValueError(
            "Invalid atomic number"
        )

    # --------------------------------------------------------
    # Aromatic flags.
    # --------------------------------------------------------

    for channel in [3, 8, 9, 10]:

        values = M[:, :, channel]

        if not np.isin(
            values,
            [0, 1]
        ).all():

            raise ValueError(
                f"Invalid binary channel: {channel}"
            )

    # --------------------------------------------------------
    # Atom chirality.
    # --------------------------------------------------------

    if not np.isin(
        M[diag, 6],
        [0, 1, 2, 3, 4, 5]
    ).all():

        raise ValueError(
            "Invalid atom chirality"
        )

    # --------------------------------------------------------
    # Bond stereochemistry.
    # --------------------------------------------------------

    if not np.isin(
        M[:, :, 11],
        [0, 1, 2, 3, 4, 5, 6]
    ).all():

        raise ValueError(
            "Invalid bond stereochemistry"
        )

    return True


# ============================================================
# 8. Numeric label conversion
# ============================================================

def get_numeric_column(df, column):

    if column not in df.columns:

        return np.full(
            len(df),
            np.nan,
            dtype=np.float64
        )

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    return values.to_numpy(
        dtype=np.float64
    )


# ============================================================
# 9. Matrix reconstruction helper
# ============================================================

def reconstruct_matrix(npz_data, index):

    start = int(
        npz_data["matrix_ptr"][index]
    )

    end = int(
        npz_data["matrix_ptr"][index + 1]
    )

    n = int(
        npz_data["atom_counts"][index]
    )

    M = npz_data["M_data"][
        start:end
    ].reshape(
        n,
        n,
        N_CHANNELS
    )

    return M


# ============================================================
# 10. Compile CSV
# ============================================================

def compile_csv(

    input_file,

    output_file,

    state="lean",

    smiles_column=None,

    target_column="delta_H",

    max_rows=None

):

    print("\nLoading CSV...")

    df = pd.read_csv(
        input_file,
        dtype=str,
        keep_default_na=False
    )

    if state == "lean":

        default_column = "unsat_SMILE"

    elif state == "rich":

        default_column = "sat_SMILE"

    else:

        raise ValueError(
            "State must be lean or rich"
        )

    smiles_column = (
        smiles_column
        if smiles_column is not None
        else default_column
    )

    if smiles_column not in df.columns:

        raise KeyError(
            f"SMILES column not found: {smiles_column}"
        )

    if target_column not in df.columns:

        raise KeyError(
            f"Target column not found: {target_column}"
        )

    if max_rows is not None:

        df = df.iloc[:max_rows].copy()

    total_rows = len(df)

    print("Input:", input_file)

    print("State:", state)

    print("SMILES column:", smiles_column)

    print("Total rows:", total_rows)

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    matrices = []

    atom_counts = []

    matrix_ptr = [0]

    atom_ptr = [0]

    ids = []

    source_rows = []

    input_smiles = []

    external_ids = []

    all_status = []

    all_errors = []

    failures = []

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    label_columns = list(dict.fromkeys([

        target_column,

        "delta_H",

        "nH2",

        "pH2"

    ]))

    label_columns = [

        c for c in label_columns

        if c in df.columns

    ]

    all_labels = {

        c: get_numeric_column(
            df, c
        )

        for c in label_columns

    }

    valid_labels = {

        c: []

        for c in label_columns

    }

    # --------------------------------------------------------
    # Optional external ID
    # --------------------------------------------------------

    has_external_id = (
        "id" in df.columns
    )

    # --------------------------------------------------------
    # Row-by-row compilation
    # --------------------------------------------------------

    for source_row, row in enumerate(
        df.itertuples(index=False, name=None)
    ):

        smiles = str(
            df.iloc[source_row][smiles_column]
        ).strip()

        matrix_id = f"row_{source_row:08d}"

        try:

            if not smiles:

                raise ValueError(
                    "Empty SMILES"
                )

            M = smiles_to_matrix(
                smiles
            )

            n = M.shape[0]

            # Store flattened matrix.
            matrices.append(
                M.reshape(-1, N_CHANNELS)
            )

            atom_counts.append(n)

            matrix_ptr.append(
                matrix_ptr[-1] + n * n
            )

            atom_ptr.append(
                atom_ptr[-1] + n
            )

            ids.append(matrix_id)

            source_rows.append(
                source_row
            )

            input_smiles.append(
                smiles
            )

            if has_external_id:

                external_ids.append(
                    str(
                        df.iloc[source_row]["id"]
                    )
                )

            else:

                external_ids.append("")

            for column in label_columns:

                valid_labels[column].append(
                    all_labels[column][source_row]
                )

            all_status.append("success")

            all_errors.append("")

        except Exception as e:

            failures.append({

                "matrix_id": matrix_id,

                "source_row": source_row,

                "smiles": smiles,

                "reason": str(e)

            })

            all_status.append("failed")

            all_errors.append(
                str(e)
            )

        if (source_row + 1) % 500 == 0:

            print(
                f"Processed {source_row+1}/{total_rows}"
            )

    # --------------------------------------------------------
    # Check compilation result
    # --------------------------------------------------------

    if len(matrices) == 0:

        raise RuntimeError(
            "No valid molecular matrices generated"
        )

    M_data = np.concatenate(
        matrices,
        axis=0
    ).astype(np.float32)

    # --------------------------------------------------------
    # Construct valid labels
    # --------------------------------------------------------

    label_arrays = {

        column: np.asarray(
            values,
            dtype=np.float64
        )

        for column, values in valid_labels.items()

    }

    # --------------------------------------------------------
    # Metadata for ALL source rows
    #
    # Failed rows are not discarded from the audit.
    # source_rows remain original CSV row numbers.
    # --------------------------------------------------------

    all_ids = np.asarray([

        f"row_{i:08d}"

        for i in range(total_rows)

    ], dtype=str)

    all_smiles = df[
        smiles_column
    ].astype(str).to_numpy()

    # --------------------------------------------------------
    # Channel metadata
    # --------------------------------------------------------

    encoding = {

        "schema_version": "LOHC_12CH_V1",

        "matrix_shape": "N x N x 12",

        "storage_dtype": "float32",

        "atom_channels": {

            "0": "atomic_number",

            "1": "formal_charge",

            "2": "total_hydrogens",

            "3": "aromatic_atom",

            "4": "isotope_mass",

            "5": "radical_electrons",

            "6": "atom_chirality"

        },

        "bond_channels": {

            "7": "bond_order",

            "8": "aromatic_bond",

            "9": "conjugated_bond",

            "10": "ring_bond",

            "11": "bond_stereochemistry"

        },

        "atom_chirality_encoding": {

            "0": "unspecified",

            "1": "R",

            "2": "S",

            "3": "r",

            "4": "s",

            "5": "specified_other_or_unresolved"

        },

        "bond_stereo_encoding": {

            "0": "unspecified",

            "1": "E",

            "2": "Z",

            "3": "CIS",

            "4": "TRANS",

            "5": "ANY",

            "6": "OTHER"

        },

        "bond_order_encoding": {

            "1.0": "single",

            "1.5": "aromatic",

            "2.0": "double",

            "3.0": "triple"

        },

        "atom_map_numbers": (

            "Not included as a model channel"

        ),

        "stereochemistry_note": (

            "Atom chirality uses CIP labels where available. "

            "Unresolved or nonstandard stereochemistry is "

            "marked separately. Bond stereo is stored "

            "symmetrically using E/Z or related categories."

        ),

        "state": state,

        "smiles_column": smiles_column

    }

    # --------------------------------------------------------
    # Save NPZ
    # --------------------------------------------------------

    save_dict = {

        # Core matrix data
        "M_data": M_data,

        "matrix_ptr": np.asarray(
            matrix_ptr,
            dtype=np.int64
        ),

        "atom_ptr": np.asarray(
            atom_ptr,
            dtype=np.int64
        ),

        "atom_counts": np.asarray(
            atom_counts,
            dtype=np.int32
        ),

        # Valid molecule identifiers
        "ids": np.asarray(
            ids,
            dtype=str
        ),

        "source_rows": np.asarray(
            source_rows,
            dtype=np.int64
        ),

        "external_ids": np.asarray(
            external_ids,
            dtype=str
        ),

        "input_smiles": np.asarray(
            input_smiles,
            dtype=str
        ),

        # All original source rows
        "all_ids": all_ids,

        "all_source_rows": np.arange(
            total_rows,
            dtype=np.int64
        ),

        "all_input_smiles": all_smiles,

        "all_status": np.asarray(
            all_status,
            dtype=str
        ),

        "all_errors": np.asarray(
            all_errors,
            dtype=str
        ),

        # Channel definitions
        "channel_names": np.asarray(
            CHANNEL_NAMES,
            dtype=str
        ),

        "encoding_json": np.asarray(
            json.dumps(
                encoding,
                ensure_ascii=False
            )
        ),

        "state": np.asarray(state),

        "smiles_column": np.asarray(
            smiles_column
        ),

        "schema_version": np.asarray(
            "LOHC_12CH_V1"
        )

    }

    # Add labels for valid matrices.
    for column, values in label_arrays.items():

        save_dict[column] = values

    # Add labels for all original rows.
    for column, values in all_labels.items():

        save_dict[
            f"all_{column}"
        ] = values

    np.savez_compressed(
        output_file,
        **save_dict
    )

    # --------------------------------------------------------
    # Save failure report
    # --------------------------------------------------------

    failure_file = output_file.replace(
        ".npz",
        "_failed.csv"
    )

    pd.DataFrame(
        failures,
        columns=[
            "matrix_id",
            "source_row",
            "smiles",
            "reason"
        ]
    ).to_csv(

        failure_file,

        index=False,

        encoding="utf-8-sig"

    )

    # --------------------------------------------------------
    # Save complete audit
    # --------------------------------------------------------

    audit = pd.DataFrame({

        "matrix_id": all_ids,

        "source_row": np.arange(
            total_rows
        ),

        "smiles": all_smiles,

        "status": all_status,

        "error": all_errors

    })

    for column, values in all_labels.items():

        audit[column] = values

    audit_file = output_file.replace(
        ".npz",
        "_audit.csv"
    )

    audit.to_csv(

        audit_file,

        index=False,

        encoding="utf-8-sig"

    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n===================================")

    print("12-CHANNEL MATRIX COMPILATION DONE")

    print("===================================")

    print("Total source rows:", total_rows)

    print("Valid molecules:", len(matrices))

    print("Failed molecules:", len(failures))

    print("Matrix data shape:", M_data.shape)

    print("Total matrix positions:", len(M_data))

    print("Channels:", N_CHANNELS)

    print("Matrix format: N x N x 12")

    print("State:", state)

    print("Output:", output_file)

    print("Failure report:", failure_file)

    print("Audit report:", audit_file)

    print("===================================\n")

    return output_file


# ============================================================
# 11. CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "LOHC 12-channel molecular matrix compiler"
        )

    )

    parser.add_argument(

        "--input",

        required=True

    )

    parser.add_argument(

        "--output",

        default="lohc_multichannel_12ch_v1.npz"

    )

    parser.add_argument(

        "--state",

        choices=["lean", "rich"],

        default="lean"

    )

    parser.add_argument(

        "--smiles-col",

        default=None

    )

    parser.add_argument(

        "--target-col",

        default="delta_H"

    )

    parser.add_argument(

        "--max-rows",

        type=int,

        default=None

    )

    args = parser.parse_args()

    compile_csv(

        input_file=args.input,

        output_file=args.output,

        state=args.state,

        smiles_column=args.smiles_col,

        target_column=args.target_col,

        max_rows=args.max_rows

    )


if __name__ == "__main__":

    main()
