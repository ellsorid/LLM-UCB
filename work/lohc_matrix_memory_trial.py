"""JSON requests and matrix identity checks for the online evidence memory."""
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from rdkit import Chem


def request_json(url, params):
    request = Request(url + '?' + urlencode(params), headers={'User-Agent': 'LOHC-Matrix-Evidence-Pilot/1.0'})
    with urlopen(request, timeout=18) as response:
        return json.load(response)


def graph_identity(state):
    n = len(state)
    mol = Chem.RWMol()
    bonds = {1: Chem.BondType.SINGLE, 1.5: Chem.BondType.AROMATIC, 2: Chem.BondType.DOUBLE,
             3: Chem.BondType.TRIPLE}
    for row in state:
        z, charge, hydrogens, isotope, aromatic = map(int, row[n:])
        atom = Chem.Atom(z)
        atom.SetFormalCharge(charge)
        atom.SetNumExplicitHs(hydrogens)
        atom.SetNoImplicit(True)
        atom.SetIsotope(isotope)
        atom.SetIsAromatic(bool(aromatic))
        mol.AddAtom(atom)
    for i, row in enumerate(state):
        for j in range(i+1, n):
            if row[j]:
                mol.AddBond(i, j, bonds[row[j]])
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def matrix_pair(matrices):
    return tuple(graph_identity(matrices[k]) for k in ('AL', 'AR'))
