"""Typed bond changes on aligned molecular pairs."""
from collections import Counter, defaultdict

import numpy as np
from rdkit import Chem

from lohc_matrix_deepseek_pilot import matrices, unweighted_graph


MOTIFS = {name: Chem.MolFromSmarts(smarts) for name, smarts in {
    'phenol': '[c]-[O;H1]', 'aryl_ether': '[c]-[O;X2]-[#6]', 'aromatic_n': '[#7;a]',
    'furan_o': '[o;r5]', 'aryl_exocyclic_cn': '[c]!@[#7]', 'carbonyl': '[#6]=[#8]',
    'aliphatic_amine': '[N;X3;!$(N-C=O)]',
}.items()}


def edge_types(molecule):
    rings = molecule.GetRingInfo().BondRings()
    out = defaultdict(set)
    for ring in rings:
        label = f'{len(ring)}' + ('ar' if all(molecule.GetBondWithIdx(b).GetIsAromatic() for b in ring) else 'nonar')
        for b in ring:
            out[b].add(label)
    return {b.GetIdx(): '/'.join(sorted(out[b.GetIdx()])) or 'nonring' for b in molecule.GetBonds()}


def describe(pair):
    lean, rich = [Chem.MolFromSmiles(s) for s in pair]
    matrix, _ = matrices((lean, rich))
    left = {(i, j): order for i, j, order in matrix['lean_upper']}
    right = {(i, j): order for i, j, order in matrix['rich_upper']}
    lean_types = edge_types(lean)
    # Use the same selected atom mapping to label the rich graph.
    candidates = unweighted_graph(rich).GetSubstructMatches(unweighted_graph(lean), uniquify=False, maxMatches=4096)
    rich_adjacency = Chem.GetAdjacencyMatrix(rich, useBO=True)
    expected = np.zeros_like(rich_adjacency)
    for (i, j), order in right.items():
        expected[i, j] = expected[j, i] = order
    mapping = min(m for m in candidates if np.array_equal(rich_adjacency[np.ix_(m, m)], expected))
    assert all(rich.GetAtomWithIdx(mapping[i]).GetTotalNumHs() == h for i, _, _, h in matrix['atoms'])
    rich_types = edge_types(rich)
    active = {i for i, j, _ in matrix['delta_rich_minus_lean']} | {j for i, j, _ in matrix['delta_rich_minus_lean']}
    distances = Chem.GetDistanceMatrix(lean)
    facts = {'T0': f'两态重原子连接相同；等价候选映射数 {matrix["equally_good_mappings"]}'}
    typed = Counter()
    for k, (i, j, _) in enumerate(matrix['delta_rich_minus_lean'], 1):
        element = '-'.join(sorted([matrix['atoms'][i][1], matrix['atoms'][j][1]]))
        a = lean_types[lean.GetBondBetweenAtoms(i, j).GetIdx()]
        b = rich_types[rich.GetBondBetweenAtoms(mapping[i], mapping[j]).GetIdx()]
        change = f'{left[i,j]:g}>{right[i,j]:g}'
        facts[f'E{k:02}'] = f'{i}-{j} {element} {change} {a}>{b}'
        typed[f'edit:{element}:{change}:{a}>{b}'] += 1
    motif_id = 0
    for phase, mol, lookup in [('lean', lean, dict(enumerate(range(lean.GetNumAtoms())))),
                               ('rich', rich, {index: i for i, index in enumerate(mapping)})]:
        for name, query in MOTIFS.items():
            for atoms in mol.GetSubstructMatches(query):
                indices = sorted(lookup[i] for i in atoms)
                distance = int(min(distances[i, j] for i in indices for j in active))
                motif_id += 1
                facts[f'G{motif_id:02}'] = f'{phase} {name} atoms={indices} dist={distance}'
                typed[f'motif:{phase}:{name}:{min(distance, 3)}'] += 1
    delta = np.zeros_like(expected)
    for i, j, value in matrix['delta_rich_minus_lean']:
        delta[i, j] = delta[j, i] = value
    values = np.linalg.eigvalsh(delta)
    delta_spectrum = np.r_[np.quantile(values, np.linspace(0, 1, 9)),
                           np.mean(abs(values)), np.sqrt(np.mean(values**2))]
    return matrix, facts, dict(typed), delta_spectrum.tolist()
