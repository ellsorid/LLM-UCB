"""Fixed-grid configuration, candidate tracing and parent export shared by C."""

from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolHash

PROPERTIES = ["h2_wt_percent", "delta_H_per_H2", "SA score"]
SOURCE_FILES = ("utils.py", "adaptive_ga.py", "predict.py", "GNN2.py", "main.py", "enthalpy_window.py")
PATTERNS = {
    "ring_aryl_ether": Chem.MolFromSmarts("[c]-[O;R]-[C;X4;R]"),
    "sidechain_aryl_ether": Chem.MolFromSmarts("[c]-[O;!R]-[C;X4;!R]"),
    "free_phenol": Chem.MolFromSmarts("[c]-[O;H1]"),
    "aromatic_nitrogen": Chem.MolFromSmarts("[n]"),
}

WINDOW_FIELDS = {"dh_target": 54.0, "dh_window_low": 40.0, "dh_window_high": 70.0}

ARCHIVE_COLS = ("smi", "sat_smi", "fitness_evaluated", "chemical_behavior_cell", "qd_quality",
                "qd_target_hit", *PROPERTIES)


@lru_cache(maxsize=6000)
def labels(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return tuple(name for name, pattern in PATTERNS.items() if mol is not None and mol.HasSubstructMatch(pattern))


class Catalog:
    def __init__(self):
        self.cells = {}
        self.valid = set()
        self.members = {}
        self.totals = Counter()
        self.group_stats = {}

    def add(self, frame):
        for row in frame[list(ARCHIVE_COLS)].to_dict("records"):
            smi, cell = row["smi"], row["chemical_behavior_cell"]
            if not isinstance(smi, str) or pd.isna(cell) or cell < 0:
                continue
            cell = int(cell)
            if smi not in self.cells:
                self.totals[cell] += 1
                self.cells[smi] = cell
            if (smi in self.valid or row["fitness_evaluated"] != 1
                    or any(not np.isfinite(row[name]) for name in PROPERTIES)):
                continue
            self.valid.add(smi)
            motifs = labels(smi)
            self.members[smi] = motifs
            for motif in ("all_valid", *motifs):
                group = self.group_stats.setdefault((cell, motif), {"n": 0, "hits": 0, "top": None})
                group["n"] += 1
                group["hits"] += int(row["qd_target_hit"] == 1)
                best = group["top"]
                quality = row["qd_quality"]
                if pd.notna(quality) and (best is None or quality > best["qd_quality"]):
                    group["top"] = {key: row[key] for key in ("smi", "sat_smi", "qd_quality", *PROPERTIES)}



def pair_trace(pairs, ranked, seen, generation, offset):
    rows = {row["smi"]: row for row in ranked.to_dict("records")}
    records = []
    for index, (lean, rich) in enumerate(pairs, offset + 1):
        row = rows.get(lean)
        valid = row is not None and all(pd.notna(row[name]) and np.isfinite(row[name]) for name in PROPERTIES)
        novel = bool(valid and lean not in seen)
        if novel:
            seen.add(lean)
        records.append({
            "index": index, "generation": generation, "smi": lean, "rich_proxy_smi": rich,
            "new_valid_evaluation": novel,
            "new_native_hit": bool(novel and row["qd_target_hit"] == 1),
            "chemical_cell": int(row["chemical_behavior_cell"]) if valid else None,
            "target_cell": row["qd_cell"] if valid else None,
            "qd_quality": float(row["qd_quality"]) if valid else None,
        })
    return records


def checked_pair(lean, rich):
    if isinstance(lean, str) and lean and isinstance(rich, str) and rich:
        a, b = Chem.MolFromSmiles(lean), Chem.MolFromSmiles(rich)
        if a is None or b is None or rdMolHash.MolHash(a, rdMolHash.HashFunction.ElementGraph) != rdMolHash.MolHash(
                b, rdMolHash.HashFunction.ElementGraph):
            return None
        net = sum(atom.GetTotalNumHs() for atom in b.GetAtoms()) - sum(atom.GetTotalNumHs() for atom in a.GetAtoms())
        if net > 0 and net % 2 == 0:
            return (Chem.MolToSmiles(a), Chem.MolToSmiles(b))
    return None


def candidate(row):
    pair = checked_pair(row.get("smi"), row.get("sat_smi"))
    if not pair:
        return None
    return {"smi": row["smi"], "pair": list(pair), "scaffold": str(row.get("scaffold_smi") or "(acyclic)"),
            "cell": row["qd_cell"], "quality": float(row["qd_quality"]),
            "properties": {k: float(row[k]) for k in ("h2_wt_percent", "delta_H_per_H2")}}


def parent_archive(frame):
    return [item for row in frame.to_dict("records") if row.get("qd_cell") and (item := candidate(row))]
