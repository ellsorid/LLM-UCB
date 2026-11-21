
# ============================================================
# Matrix Data Module
#
# Reads M_data directly.
# Does not read SMILES for model input.
# Does not construct graph objects.
# ============================================================

import hashlib
import numpy as np
import torch

from torch.utils.data import Dataset

from config import (
    NUM_CHANNELS,
    CHANNEL_NAMES,
    SCHEMA_VERSION
)


# ============================================================
# Matrix validation
# ============================================================

def validate_matrix(M, index=None):

    prefix = f"Molecule {index}: " if index is not None else ""

    if M.ndim != 3:
        raise ValueError(prefix + "Matrix must be 3D")

    n1, n2, channels = M.shape

    if n1 != n2 or channels != NUM_CHANNELS:
        raise ValueError(prefix + "Invalid matrix dimensions")

    if not np.isfinite(M).all():
        raise ValueError(prefix + "NaN or Inf detected")

    diag = np.eye(n1, dtype=bool)
    offdiag = ~diag

    # Atom channels: diagonal only
    if not np.allclose(M[offdiag, :7], 0):
        raise ValueError(prefix + "Atom channels outside diagonal")

    # Bond channels: off-diagonal only
    if not np.allclose(M[diag, 7:12], 0):
        raise ValueError(prefix + "Bond channels on diagonal")

    # Atomic number
    Z = M[diag, 0]

    if np.any(Z < 1) or np.any(Z > 118):
        raise ValueError(prefix + "Invalid atomic number")

    # Atom channels must be integer encoded
    if not np.allclose(M[:, :, :7], np.round(M[:, :, :7])):
        raise ValueError(prefix + "Non-integer atom channel")

    # Bond symmetry
    if not np.allclose(
        M[:, :, 7:12],
        M.transpose(1, 0, 2)[:, :, 7:12]
    ):
        raise ValueError(prefix + "Asymmetric bond matrix")

    # Bond order
    bond_order = M[:, :, 7]

    allowed_orders = [0.0, 1.0, 1.5, 2.0, 3.0]

    valid = np.zeros_like(
        bond_order,
        dtype=bool
    )

    for value in allowed_orders:
        valid |= np.isclose(bond_order, value)

    if not valid.all():
        raise ValueError(prefix + "Unsupported bond order")

    # Binary channels
    for channel in [8, 9, 10]:

        if not np.isin(
            M[:, :, channel],
            [0, 1]
        ).all():

            raise ValueError(
                prefix + f"Invalid binary channel {channel}"
            )

    # Atom chirality
    if not np.isin(
        M[diag, 6],
        [0, 1, 2, 3, 4, 5]
    ).all():

        raise ValueError(prefix + "Invalid atom chirality")

    # Bond stereochemistry
    if not np.isin(
        M[:, :, 11],
        [0, 1, 2, 3, 4, 5, 6]
    ).all():

        raise ValueError(prefix + "Invalid bond stereochemistry")

    # Unbonded positions must have zero bond features
    no_bond = bond_order == 0

    if not np.allclose(M[:, :, 8:12][no_bond], 0):
        raise ValueError(prefix + "Features found on absent bonds")

    return True


# ============================================================
# NPZ loader
# ============================================================

def load_matrix_archive(path, target_col="delta_H"):

    print("Loading 12-channel matrix archive...")

    with np.load(path, allow_pickle=False) as z:

        required = [
            "M_data",
            "matrix_ptr",
            "atom_counts",
            "ids",
            "source_rows",
            "channel_names",
            "schema_version",
            target_col
        ]

        missing = [
            key for key in required
            if key not in z.files
        ]

        if missing:
            raise KeyError(
                f"Missing NPZ fields: {missing}"
            )

        archive = {

            "M_data": np.asarray(
                z["M_data"],
                dtype=np.float32
            ),

            "matrix_ptr": np.asarray(
                z["matrix_ptr"],
                dtype=np.int64
            ),

            "atom_counts": np.asarray(
                z["atom_counts"],
                dtype=np.int64
            ),

            "ids": z["ids"].astype(str),

            "source_rows": np.asarray(
                z["source_rows"],
                dtype=np.int64
            ),

            "targets": np.asarray(
                z[target_col],
                dtype=np.float64
            ),

            "channel_names": z["channel_names"].astype(str),

            "schema_version": str(
                z["schema_version"].item()
            )

        }

    M_data = archive["M_data"]
    ptr = archive["matrix_ptr"]
    counts = archive["atom_counts"]

    n = len(counts)

    if M_data.ndim != 2 or M_data.shape[1] != 12:
        raise ValueError(
            "M_data must have shape [total_positions, 12]"
        )

    if archive["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "Unexpected matrix schema version"
        )

    if not np.array_equal(
        archive["channel_names"],
        np.asarray(CHANNEL_NAMES)
    ):
        raise ValueError(
            "Channel order does not match model configuration"
        )

    if len(ptr) != n + 1:
        raise ValueError("matrix_ptr length mismatch")

    if ptr[0] != 0:
        raise ValueError("matrix_ptr must start at zero")

    if not np.array_equal(
        np.diff(ptr),
        counts ** 2
    ):
        raise ValueError("matrix_ptr and atom_counts mismatch")

    if ptr[-1] != len(M_data):
        raise ValueError("M_data pointer mismatch")

    for key in ["ids", "source_rows", "targets"]:

        if len(archive[key]) != n:
            raise ValueError(
                f"{key} length mismatch"
            )

    if len(set(archive["ids"].tolist())) != n:
        raise ValueError("Duplicate matrix IDs")

    if len(set(archive["source_rows"].tolist())) != n:
        raise ValueError("Duplicate source row indices")

    if not np.isfinite(archive["targets"]).all():
        raise ValueError("Invalid target values")

    if np.any(counts <= 0):
        raise ValueError("Invalid atom counts")

    print("Molecules:", n)
    print("M_data:", M_data.shape)
    print("Channels:", M_data.shape[-1])
    print("Target:", "delta_H")
    print("Schema:", archive["schema_version"])

    print("\nValidating molecular matrices...")

    for i in range(n):

        start = ptr[i]
        end = ptr[i + 1]

        matrix = M_data[start:end].reshape(
            counts[i],
            counts[i],
            NUM_CHANNELS
        )

        validate_matrix(matrix, i)

        if (i + 1) % 1000 == 0:
            print(f"Validated {i+1}/{n}")

    print("Matrix validation passed.")

    return archive


# ============================================================
# Retrieve one molecule matrix
# ============================================================

def get_matrix(archive, index):

    start = archive["matrix_ptr"][index]
    end = archive["matrix_ptr"][index + 1]

    n = archive["atom_counts"][index]

    return archive["M_data"][start:end].reshape(
        n, n, NUM_CHANNELS
    )


# ============================================================
# Matrix-native molecular identity
#
# Weisfeiler-Lehman style color refinement.
# Used only for grouped cross-validation.
# No graph objects are constructed.
# ============================================================

def compress_colors(signatures):

    unique = sorted(
        set(signatures),
        key=repr
    )

    lookup = {
        signature: i
        for i, signature in enumerate(unique)
    }

    return [
        lookup[x]
        for x in signatures
    ]


def matrix_identity_hash(M):

    n = M.shape[0]

    # Initial atomic labels
    atom_labels = []

    for i in range(n):

        atom_labels.append(
            tuple(
                np.round(
                    M[i, i, :7]
                ).astype(int).tolist()
            )
        )

    colors = compress_colors(atom_labels)

    # Iterative neighborhood refinement
    for _ in range(max(1, n)):

        signatures = []

        for i in range(n):

            neighbors = []

            for j in range(n):

                if i == j:
                    continue

                if M[i, j, 7] <= 0:
                    continue

                bond_features = tuple(
                    np.round(
                        M[i, j, 7:12] * 2
                    ).astype(int).tolist()
                )

                neighbors.append(
                    (bond_features, colors[j])
                )

            neighbors.sort()

            signatures.append(
                (colors[i], tuple(neighbors))
            )

        new_colors = compress_colors(
            signatures
        )

        if new_colors == colors:
            break

        colors = new_colors

    # Final node and edge signature
    node_signature = tuple(
        sorted(colors)
    )

    edge_signature = []

    for i in range(n):

        for j in range(i + 1, n):

            if M[i, j, 7] <= 0:
                continue

            bond_features = tuple(
                np.round(
                    M[i, j, 7:12] * 2
                ).astype(int).tolist()
            )

            edge_signature.append(
                (
                    min(colors[i], colors[j]),
                    max(colors[i], colors[j]),
                    bond_features
                )
            )

    edge_signature.sort()

    payload = repr((
        n,
        node_signature,
        tuple(edge_signature)
    ))

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def build_matrix_groups(archive, n_samples=None):

    n = (
        len(archive["ids"])
        if n_samples is None
        else n_samples
    )

    groups = []

    print("\nBuilding matrix-native molecular groups...")

    for i in range(n):

        M = get_matrix(archive, i)

        groups.append(
            matrix_identity_hash(M)
        )

        if (i + 1) % 1000 == 0:
            print(f"Grouped {i+1}/{n}")

    return np.asarray(groups)


# ============================================================
# Dataset
# ============================================================

class MatrixDataset(Dataset):

    def __init__(
        self,
        archive,
        targets,
        indices
    ):

        self.archive = archive
        self.targets = targets

        self.indices = np.asarray(
            indices,
            dtype=np.int64
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        i = int(self.indices[idx])

        M = get_matrix(
            self.archive,
            i
        )

        return (
            torch.from_numpy(M),
            float(self.targets[i])
        )


# ============================================================
# Batch collation
#
# M:    [B, Nmax, Nmax, 12]
# Mask: [B, Nmax]
# Y:    [B]
# ============================================================

def collate_matrix_batch(batch):

    batch_size = len(batch)

    max_atoms = max(
        item[0].shape[0]
        for item in batch
    )

    M_batch = torch.zeros(
        batch_size,
        max_atoms,
        max_atoms,
        NUM_CHANNELS,
        dtype=torch.float32
    )

    mask = torch.zeros(
        batch_size,
        max_atoms,
        dtype=torch.float32
    )

    y = torch.zeros(
        batch_size,
        dtype=torch.float32
    )

    for i, (M, target) in enumerate(batch):

        n = M.shape[0]

        M_batch[i, :n, :n, :] = M

        mask[i, :n] = 1.0

        y[i] = float(target)

    return M_batch, mask, y
