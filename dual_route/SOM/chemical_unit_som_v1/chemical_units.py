"""Extract chemistry-defined, atom-indexed units from aligned LOHC matrices."""
import argparse
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np


KINDS = ("functional_group", "ring_or_conjugation", "attachment", "reaction_change")
STATES = ("lean", "rich")
CHANGE_CHANNELS = {1:"formal_charge", 2:"hydrogen_count", 7:"bond_order", 8:"aromatic_bond", 9:"conjugation"}


def load_pair(record):
    from matrix_pipeline.compat.pair_core import dense_from_state
    states = record.get("whole_structure", record.get("full"))
    return np.stack([dense_from_state(states[s]) for s in STATES])


def extract(pair, uid):
    from matrix_pipeline.matrix_structure import extract_state
    from rdkit import Chem
    from matrix_pipeline.core import to_mol

    if pair.ndim != 4 or pair.shape[0] != 2 or pair.shape[1] != pair.shape[2] or pair.shape[3] != 12:
        raise ValueError("Expected aligned [2,n,n,12] matrix")
    if not np.array_equal(np.diag(pair[0, :, :, 0]), np.diag(pair[1, :, :, 0])):
        raise ValueError("States disagree on aligned elements")
    output = {kind:[] for kind in KINDS}
    n = pair.shape[1]
    graphs = [nx.from_numpy_array(matrix[:, :, 7] != 0) for matrix in pair]
    events = []
    for i in range(n):
        for j in range(i, n):
            for c in ((1, 2) if i==j else (7, 8, 9)):
                before, after = float(pair[0,i,j,c]), float(pair[1,i,j,c])
                if before != after:
                    events.append({"atoms":[i] if i==j else [i,j], "channel":CHANGE_CHANNELS[c],
                                   "lean":before, "rich":after})

    def add(kind, name, state, members, anchor=()):
        ids = sorted(set(members))
        index = STATES.index(state)
        graph, matrix = graphs[index], pair[index]
        selected = set(ids)
        if not selected:
            return
        bonds = [[i,j,float(matrix[i,j,7])] for i,j in sorted(graph.edges)
                 if i in selected and j in selected]
        boundary = [[i,j,float(matrix[i,j,7])] for i in ids for j in sorted(graph[i]) if j not in selected]
        key = (name, state, ids)
        if any((r["name"],r["state"],r["parent_ids"]) == key for r in output[kind]):
            return
        output[kind].append({
            "unit_id":f"{kind}/{state}/{len(output[kind])}", "name":name, "state":state, "parent_ids":ids,
            "anchor_ids":sorted(set(anchor)), "internal_bonds":bonds, "boundary_bonds":boundary,
            "atom_labels":[[i,int(matrix[i,i,0]),int(matrix[i,i,1]),int(matrix[i,i,2]),bool(matrix[i,i,3])] for i in ids],
            "paired_changes":[e for e in events if selected.intersection(e["atoms"])],
        })

    smarts = {"ether":"[#6;!$([#6]=[O,S,N])]-[O;X2;H0]-[#6;!$([#6]=[O,S,N])]",
              "ester":"[#6](=[#8])-[#8;X2]-[#6]", "phenol":"[c]-[O;H1]",
              "thiol":"[#6]-[S;H1]", "sulfone":"[#6]-[S](=[O])(=[O])-[#6]",
              "imine":"[#6;!a]=[#7;!a]"}
    for state, matrix, graph in zip(STATES,pair,graphs):
        view = extract_state(matrix,state=state,molecule_id=uid)
        for group in view["functional_groups"]:
            add("functional_group", group["kind"], state, group["parent_ids"])
        molecule = to_mol(matrix)
        if molecule.GetNumAtoms() != n or any(
            atom.GetAtomicNum() != matrix[i,i,0] for i,atom in enumerate(molecule.GetAtoms())
        ):
            raise ValueError("RDKit atom index differs from matrix index")
        for name, pattern in smarts.items():
            for found in molecule.GetSubstructMatches(Chem.MolFromSmarts(pattern)):
                add("functional_group", name, state, found)
        ring_atoms = set(view["whole"]["ring_parent_ids"])
        for ring in view["ring_systems"]:
            add("ring_or_conjugation", "ring_system", state, ring["parent_ids"])
        conjugated = nx.Graph()
        conjugated.add_edges_from((i,j) for i,j in graph.edges if matrix[i,j,9])
        for component in nx.connected_components(conjugated):
            add("ring_or_conjugation", "conjugated_component", state, component)
        for component in nx.connected_components(graph.subgraph(set(graph)-ring_atoms)):
            ia = [(i,j) for i in component for j in graph[i] if j in ring_atoms]
            for i,j in ia:
                add("attachment", f"{view['atoms'][j]['element']}_attached_chain",
                    state, component | {j}, (i,j))
        changed = set(i for e in events for i in e["atoms"])
        if changed:
            union = nx.compose(*graphs)
            for component in nx.connected_components(union.subgraph(changed)):
                support = set(component)
                for ring in view["ring_systems"]:
                    if support.intersection(ring["parent_ids"]):
                        support.update(ring["parent_ids"])
                add("reaction_change", "net_hydrogenation_region", state, support, component)
    return {"molecule_id":uid,"units":output,"total_changes":len(events)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0,str(args.runtime))
    record = json.loads(args.record.read_text(encoding="utf-8"))
    result = extract(load_pair(record), record.get("record_id", args.record.stem))
    args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
