"""Heavy-atom alignment and bond matrix extraction."""
import numpy as np
from rdkit import Chem


def unweighted_graph(molecule):
    result = Chem.RWMol()
    for atom in molecule.GetAtoms():
        result.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    for bond in molecule.GetBonds():
        result.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return result.GetMol()


def sparse(matrix):
    return [[int(i), int(j), float(matrix[i, j])] for i, j in zip(*np.nonzero(np.triu(matrix, 1)))]


def aligned_bonds(molecules):
    lean, rich = molecules
    if lean.GetNumAtoms() != rich.GetNumAtoms() or lean.GetNumBonds() != rich.GetNumBonds():
        raise ValueError('A shared heavy-atom graph is required for this pilot')
    first, second = (Chem.GetAdjacencyMatrix(mol, useBO=True) for mol in molecules)
    mappings = unweighted_graph(rich).GetSubstructMatches(unweighted_graph(lean), uniquify=False, maxMatches=4096)
    if not mappings or len(mappings) >= 4096:
        raise ValueError('Atom correspondence unavailable or search limit reached')
    alternatives = []
    for match in mappings:
        delta = second[np.ix_(match, match)] - first
        alternatives.append((int(np.count_nonzero(delta)), float(np.abs(delta).sum()), match))
    changed, magnitude, mapping = min(alternatives)
    aligned = second[np.ix_(mapping, mapping)]
    return first, second, aligned, mapping, changed, sum(x[:2] == (changed, magnitude) for x in alternatives)


def matrices(molecules):
    lean, rich = molecules
    first, _, aligned, mapping, changed, ties = aligned_bonds(molecules)
    edits = sparse(aligned - first)
    atoms = [[i, atom.GetSymbol(), atom.GetTotalNumHs(), rich.GetAtomWithIdx(mapping[i]).GetTotalNumHs()]
             for i, atom in enumerate(lean.GetAtoms())]
    assert np.array_equal(first != 0, aligned != 0) and changed == 2 * len(edits)
    input_data = {'shape': [len(atoms), len(atoms)], 'atoms': atoms, 'lean_upper': sparse(first),
                  'rich_upper': sparse(aligned), 'delta_rich_minus_lean': edits,
                  'equally_good_mappings': ties}
    risk_context = np.zeros(10)
    changed_atoms = set()
    for i, j, delta in edits:
        left, right = lean.GetAtomWithIdx(i), lean.GetAtomWithIdx(j)
        bond = lean.GetBondBetweenAtoms(i, j)
        risk_context[0] += 1
        risk_context[1] += abs(delta)
        risk_context[2] += bond.GetIsAromatic()
        risk_context[3] += bond.IsInRing()
        risk_context[4] += any(a.GetAtomicNum() == 7 for a in (left, right))
        risk_context[5] += any(a.GetAtomicNum() == 8 for a in (left, right))
        risk_context[6] += bond.IsInRingSize(5)
        risk_context[7] += bond.IsInRingSize(6)
        changed_atoms.update([i, j])
    risk_context[8] = sum(atoms[i][3] - atoms[i][2] for i in changed_atoms)
    risk_context[9] = len(changed_atoms)
    return input_data, risk_context.tolist()
