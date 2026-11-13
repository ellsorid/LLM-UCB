"""Encode rich and lean states and their transformation into AR, AL and P."""
import math

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from lohc_matrix_deepseek_pilot import aligned_bonds


def canonical_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid SMILES')
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True))


def encode(pair, enthalpy=None):
    lean, rich = map(canonical_molecule, pair)
    left, right, aligned, mapping, _, ties = aligned_bonds((lean, rich))
    n = len(mapping)

    def state(mol, adjacency):
        labels = [[a.GetAtomicNum(), a.GetFormalCharge(), a.GetTotalNumHs(), a.GetIsotope(),
                   int(a.GetIsAromatic())] for a in mol.GetAtoms()]
        return np.column_stack((adjacency, labels)).tolist()

    transform = np.zeros((n, n))
    transform[np.arange(n), mapping] = 1
    delta = aligned - left
    hydrogen = [sum(a.GetAtomicNum() == 1 for a in Chem.AddHs(mol).GetAtoms()) for mol in (lean, rich)]
    n_h2 = (hydrogen[1] - hydrogen[0]) / 2
    capacity = 100 * n_h2 * 2.01588 / Descriptors.MolWt(rich) if n_h2 > 0 else None
    if enthalpy is not None and not math.isfinite(enthalpy):
        enthalpy = None
    # All rows in P have the same width. Null in a property row means unavailable or padding.
    p = np.column_stack((transform, delta, np.zeros((n, 3)))).tolist()
    p.append([None] * (2*n) + [capacity, enthalpy, ties])
    return {'AR': state(rich, right), 'AL': state(lean, left), 'P': p}
