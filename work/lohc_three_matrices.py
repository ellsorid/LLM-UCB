"""Build AR, AL, P for a shared heavy-atom LOHC graph and two global properties."""

import argparse
import csv
import gzip
import math
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from lohc_matrix_deepseek_pilot import aligned_bonds
from lohc_structure_score_trial import load, save


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


def prebuild(run, sample, out):
    requested = {tuple(r['pair']): r['id'] for r in load(sample)['scores']}
    lean_index = {pair[0] for pair in requested}
    observed = {p: {} for p in requested}
    for arm in ('native', 'llm'):
        with gzip.open(run / arm / 'archive.csv.gz', 'rt', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                if row['smi'] not in lean_index or not row['sat_smi']:
                    continue
                pair = row['smi'], Chem.MolToSmiles(canonical_molecule(row['sat_smi']))
                if pair not in requested:
                    continue
                try:
                    h = float(row['delta_H_per_H2'])
                except ValueError:
                    continue
                if math.isfinite(h):
                    observed[pair][arm] = h
    results, issues = {}, {}
    for pair, identifier in requested.items():
        values = list(observed[pair].values())
        if values and max(values) - min(values) > 1e-6:
            issues[identifier] = 'Inconsistent GNN enthalpy across arms'
            values = []
        elif not values:
            issues[identifier] = 'No matching GNN enthalpy in the archives'
        try:
            results[identifier] = encode(pair, values[0] if values else None)
        except ValueError as error:
            issues[identifier] = str(error)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'matrices.json', results)
    save(out / 'manifest.json', {'source': str(sample), 'run': str(run), 'encoded': len(results),
        'with_enthalpy': sum(r['P'][-1][-2] is not None for r in results.values()), 'issues': issues,
        'schema': {'AR/AL': 'n rows: n bond columns then atomic number, formal charge, H count, isotope, aromatic flag',
                   'P': 'first n rows: correspondence C (lean rows, rich columns), delta C AR_bonds C.T - AL_bonds, 3 padding zeros. Last row: null padding, H2 wt%, GNN dehydrogenation enthalpy kJ/mol H2, equivalent mapping count'}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--sample', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    prebuild(args.run, args.sample, args.out)
