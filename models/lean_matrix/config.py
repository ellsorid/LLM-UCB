
# ============================================================
# Configuration: 12-Channel Matrix GNN
# ============================================================

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

NUM_CHANNELS = 12
NUM_ATOM_CHANNELS = 7
NUM_BOND_CHANNELS = 5

HIDDEN_DIM = 128
NUM_GNN_LAYERS = 3

DEFAULTS = {
    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "dropout": 0.1,
    "ema_decay": 0.995,
    "warmup_epochs": 10,
    "early_stop_patience": 10,
    "n_splits": 5,
    "gradient_clip": 1.0,
    "num_workers": 0,
    "seed": 42
}

SCHEMA_VERSION = "LOHC_12CH_V1"
