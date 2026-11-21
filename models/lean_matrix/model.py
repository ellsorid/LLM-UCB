
# ============================================================
# Dense Matrix GIN
#
# Direct input:
#   M: [B,N,N,12]
#
# No SMILES.
# No RDKit.
# No PyG.
# No edge_index.
# No graph conversion.
#
# Atom and bond categorical properties are embedded
# internally, directly from the matrix channels.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import NUM_CHANNELS, HIDDEN_DIM, NUM_GNN_LAYERS
except ImportError:
    from config import NUM_CHANNELS, HIDDEN_DIM, NUM_GNN_LAYERS


# ============================================================
# Bond order encoding
# ============================================================

def encode_bond_order(x):

    idx = torch.zeros_like(
        x,
        dtype=torch.long
    )

    idx[
        torch.abs(x - 1.0) < 1e-4
    ] = 1

    idx[
        torch.abs(x - 1.5) < 1e-4
    ] = 2

    idx[
        torch.abs(x - 2.0) < 1e-4
    ] = 3

    idx[
        torch.abs(x - 3.0) < 1e-4
    ] = 4

    return idx


# ============================================================
# Atom encoder
# ============================================================

class AtomMatrixEncoder(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.embeddings = nn.ModuleList([

            nn.Embedding(
                120,
                hidden_dim,
                padding_idx=0
            ),

            nn.Embedding(
                17,
                hidden_dim
            ),

            nn.Embedding(
                16,
                hidden_dim,
                padding_idx=0
            ),

            nn.Embedding(
                2,
                hidden_dim,
                padding_idx=0
            ),

            nn.Embedding(
                301,
                hidden_dim,
                padding_idx=0
            ),

            nn.Embedding(
                9,
                hidden_dim,
                padding_idx=0
            ),

            nn.Embedding(
                6,
                hidden_dim,
                padding_idx=0
            )

        ])

        self.norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(self, atom_features, mask):

        z = atom_features[..., 0].round().long()
        charge = atom_features[..., 1].round().long()
        hydrogen = atom_features[..., 2].round().long()
        aromatic = atom_features[..., 3].round().long()
        isotope = atom_features[..., 4].round().long()
        radical = atom_features[..., 5].round().long()
        chirality = atom_features[..., 6].round().long()

        indices = [

            z.clamp(0, 119),

            (charge + 8).clamp(0, 16),

            hydrogen.clamp(0, 15),

            aromatic.clamp(0, 1),

            isotope.clamp(0, 300),

            radical.clamp(0, 8),

            chirality.clamp(0, 5)

        ]

        h = 0

        for embedding, index in zip(
            self.embeddings,
            indices
        ):

            h = h + embedding(index)

        h = self.norm(h)

        h = h * mask.unsqueeze(-1)

        return h


# ============================================================
# Bond encoder
# ============================================================

class BondMatrixEncoder(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.embeddings = nn.ModuleList([

            nn.Embedding(
                5,
                hidden_dim,
                padding_idx=0
            ),

            nn.Embedding(
                2,
                hidden_dim,
                padding_idx=0
            ),

            nn.Embedding(
                2,
                hidden_dim,
                padding_idx=0
            ),

            nn.Embedding(
                2,
                hidden_dim,
                padding_idx=0
            ),

            nn.Embedding(
                7,
                hidden_dim,
                padding_idx=0
            )

        ])

        self.norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(self, bond_features):

        bond_order = encode_bond_order(
            bond_features[..., 0]
        )

        aromatic = (
            bond_features[..., 1]
            .round()
            .long()
            .clamp(0, 1)
        )

        conjugated = (
            bond_features[..., 2]
            .round()
            .long()
            .clamp(0, 1)
        )

        ring = (
            bond_features[..., 3]
            .round()
            .long()
            .clamp(0, 1)
        )

        stereo = (
            bond_features[..., 4]
            .round()
            .long()
            .clamp(0, 6)
        )

        indices = [

            bond_order,

            aromatic,

            conjugated,

            ring,

            stereo

        ]

        edge = 0

        for embedding, index in zip(
            self.embeddings,
            indices
        ):

            edge = edge + embedding(index)

        has_bond = bond_order > 0

        edge = self.norm(edge)

        edge = edge * has_bond.unsqueeze(-1)

        return edge, has_bond


# ============================================================
# Dense GIN layer
# ============================================================

class DenseGINLayer(nn.Module):

    def __init__(
        self,
        hidden_dim,
        dropout=0.1
    ):

        super().__init__()

        self.eps = nn.Parameter(
            torch.zeros(1)
        )

        self.mlp = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )

        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        h,
        edge,
        has_bond,
        mask
    ):

        # h: [B,N,H]
        # edge: [B,N,N,H]

        valid_nodes = (
            mask > 0
        )

        valid_pairs = (

            valid_nodes.unsqueeze(1) &

            valid_nodes.unsqueeze(2) &

            has_bond

        )

        # Dense pairwise message calculation.
        # h.unsqueeze(1) represents source atoms j.
        messages = F.relu(
            h.unsqueeze(1) + edge
        )

        messages = messages * (
            valid_pairs.unsqueeze(-1)
        )

        # Aggregate neighbors for each atom i.
        aggregate = messages.sum(
            dim=2
        )

        combined = (

            (1.0 + self.eps) * h +

            aggregate

        )

        update = self.mlp(
            combined
        )

        update = self.dropout(
            update
        )

        out = self.norm(
            h + update
        )

        out = F.relu(out)

        out = out * mask.unsqueeze(-1)

        return out


# ============================================================
# Single Matrix GIN
# ============================================================

class MatrixGIN(nn.Module):

    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        dropout=0.1,
        num_layers=NUM_GNN_LAYERS
    ):

        super().__init__()

        self.atom_encoder = AtomMatrixEncoder(
            hidden_dim
        )

        self.bond_encoder = BondMatrixEncoder(
            hidden_dim
        )

        self.layers = nn.ModuleList([

            DenseGINLayer(
                hidden_dim,
                dropout
            )

            for _ in range(num_layers)

        ])

        # Sum + Mean + Max pooling
        self.fc1 = nn.Linear(
            hidden_dim * 3,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            hidden_dim // 2
        )

        self.fc3 = nn.Linear(
            hidden_dim // 2,
            1
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(
        self,
        M,
        mask
    ):

        # ----------------------------------------------------
        # Extract diagonal atom channels.
        # Result: [B,N,7]
        # ----------------------------------------------------

        atom_features = torch.diagonal(

            M[:, :, :, :7],

            dim1=1,

            dim2=2

        ).transpose(1, 2).contiguous()

        # ----------------------------------------------------
        # Extract off-diagonal bond channels.
        # Result: [B,N,N,5]
        # ----------------------------------------------------

        bond_features = M[:, :, :, 7:12]

        # ----------------------------------------------------
        # Embedding
        # ----------------------------------------------------

        h = self.atom_encoder(
            atom_features,
            mask
        )

        edge, has_bond = self.bond_encoder(
            bond_features
        )

        # ----------------------------------------------------
        # Three Dense GIN layers
        # ----------------------------------------------------

        for layer in self.layers:

            h = layer(
                h,
                edge,
                has_bond,
                mask
            )

        # ----------------------------------------------------
        # Molecular pooling
        # ----------------------------------------------------

        masked_h = (
            h * mask.unsqueeze(-1)
        )

        sum_pool = masked_h.sum(
            dim=1
        )

        atom_count = mask.sum(
            dim=1,
            keepdim=True
        ).clamp_min(1.0)

        mean_pool = (
            sum_pool / atom_count
        )

        max_input = h.masked_fill(

            mask.unsqueeze(-1) == 0,

            float("-inf")

        )

        max_pool = max_input.max(
            dim=1
        ).values

        max_pool = torch.nan_to_num(
            max_pool,
            nan=0.0,
            neginf=0.0
        )

        molecular_embedding = torch.cat([

            sum_pool,

            mean_pool,

            max_pool

        ], dim=-1)

        # ----------------------------------------------------
        # Regression head
        # ----------------------------------------------------

        out = F.relu(
            self.fc1(molecular_embedding)
        )

        out = self.dropout(out)

        out = F.relu(
            self.fc2(out)
        )

        out = self.dropout(out)

        return self.fc3(out).squeeze(-1)
