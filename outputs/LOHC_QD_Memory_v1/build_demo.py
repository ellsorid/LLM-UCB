"""Build and replay an offline LOHC memory and one fixed-cell planning example."""

import argparse
import ast
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE = Path(__file__).resolve().parent
GA_CODE = HERE.parents[1] / "work" / "lohc_operator_ac_runtime_01"
OBJECTIVES = ["obj_h2_score", "obj_dh_score", "obj_sa_score", "obj_scaffold_novelty"]
PROPERTIES = ["h2_wt_percent", "delta_H_per_H2", "SA score"]
GA_CONFIG = {"chemical_behavior_cells": 96, "chemical_behavior_seed": 1729,
             "chemical_local_archive_k": 3, "random_seed": 11}
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def load_native_qd(root):
    # Importing all of utils initializes the GNN and creates model_checkpoints.
    # Load the original, self-contained QD definitions from the read-only source.
    namespace = {"np": np, "pd": pd, "Chem": Chem, "DataStructs": DataStructs,
                 "AllChem": AllChem, "rdmd": rdMolDescriptors}
    names = {
        "adaptive_ga.py": {"BIO_PLATFORM_IDS"},
        "utils.py": {
            "QD_H2_EDGES", "QD_DH_EDGES", "QD_SA_EDGES", "_qd_finite", "_qd_bin",
            "_qd_cell_from_row", "_qd_quality_from_row", "annotate_qd_cells",
            "_chemical_behavior_vector_from_mol", "_chemical_behavior_centroids",
            "annotate_chemical_behavior", "_fingerprint_novelty_scores",
            "annotate_chemical_qd", "local_pareto_archive", "_copy_with_mol",
            "_non_dominated_sort_indices", "_pareto_front_indices_nd",
        },
    }
    for filename, required in names.items():
        path = root / filename
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        definitions = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in required:
                definitions[node.name] = node
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in required:
                    definitions[name] = node
        if definitions.keys() != required:
            raise ValueError(f"Missing native QD definitions in {path}: {required - definitions.keys()}")
        source = ast.Module(body=list(definitions.values()), type_ignores=[])
        exec(compile(source, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def write_json(path, value):
    path.write_text(json_text(value) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(smiles, stereo=True):
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=stereo) if mol else None


def scaffold(smiles):
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles) if smiles else None


def load_seed(path):
    records = read_json(path)
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Duplicate literature evidence IDs")
    for record in records:
        for field in ("lean", "rich"):
            key = field + "_smiles"
            smiles = canonical(record[key])
            if record[key] and not smiles:
                raise ValueError(f"Invalid literature structure {record['id']}: {key}")
            record[key] = smiles
            record[field + "_inchikey"] = (
                Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)) if smiles else None
            )
        if not all(record[key] for key in ("doi", "source_url", "source_anchor", "evidence_type")):
            raise ValueError(f"Evidence lacks provenance: {record['id']}")
        record["lean_scaffold"] = scaffold(record["lean_smiles"])
    return records


def import_ga(archive, ga):
    frame = pd.read_csv(archive, low_memory=False)
    frame.insert(0, "candidate_id", [f"G{i:05d}" for i in range(1, len(frame) + 1)])
    frame["canonical_smi"] = frame["smi"].map(canonical)
    frame["canonical_sat_smi"] = frame["sat_smi"].map(canonical)
    frame["source_chemical_cell"] = frame["chemical_behavior_cell"]
    frame = ga.annotate_chemical_behavior(frame, GA_CONFIG)
    frame = ga.annotate_qd_cells(frame)
    numeric = frame[PROPERTIES + OBJECTIVES].apply(pd.to_numeric, errors="coerce")
    frame[PROPERTIES + OBJECTIVES] = numeric
    frame["selection_score"] = frame.get("selection_score", 0.0)
    if "pareto_objective_sum" not in frame:
        frame["pareto_objective_sum"] = numeric[OBJECTIVES].sum(axis=1, min_count=len(OBJECTIVES))
    frame["pareto_eligible"] = (
        pd.to_numeric(frame["fitness_evaluated"], errors="coerce").eq(1)
        & frame["canonical_smi"].notna()
        & frame["canonical_sat_smi"].notna()
        & np.isfinite(numeric).all(axis=1)
    )
    frame["target_gap"] = (
        (4.2 - frame["h2_wt_percent"]).clip(lower=0) / 1.5
        + (48 - frame["delta_H_per_H2"]).clip(lower=0) / 10
        + (frame["delta_H_per_H2"] - 62).clip(lower=0) / 10
        + (frame["SA score"] - 4).clip(lower=0) / 2
    )
    return frame


def snapshots(frame):
    fields = ["candidate_id", "canonical_smi", "canonical_sat_smi", "scaffold_smi", "platform_id",
              "h2_wt_percent", "delta_H_per_H2", "SA score", "qd_cell", "qd_target_hit", "target_gap"]
    return json.loads(frame[fields].to_json(orient="records", double_precision=10))


def literature_matches(seed, candidates, limit=5):
    queries = [(item["canonical_smi"], item["canonical_sat_smi"]) for item in candidates]
    scored = []
    for evidence in seed:
        lean, rich = evidence["lean_smiles"], evidence["rich_smiles"]
        if not lean:
            continue
        fp = FP.GetFingerprint(Chem.MolFromSmiles(lean))
        matches = []
        for query_lean, query_rich in queries:
            pair = (canonical(lean, False), canonical(rich, False))
            query_pair = (canonical(query_lean, False), canonical(query_rich, False))
            if all(pair) and pair == query_pair:
                match = (3, 1.0, "same_connectivity_pair_stereo_unresolved")
                if (lean, rich) == (query_lean, query_rich) and evidence["rich_identity"] == "achiral":
                    match = (4, 1.0, "same_pair")
            elif pair[0] == query_pair[0]:
                match = (2, 1.0, "same_lean_only")
            else:
                similarity = DataStructs.TanimotoSimilarity(
                    fp, FP.GetFingerprint(Chem.MolFromSmiles(query_lean))
                )
                same_scaffold = bool(evidence["lean_scaffold"]) and (
                    evidence["lean_scaffold"] == scaffold(query_lean)
                )
                match = (int(same_scaffold), similarity,
                         "same_lean_scaffold_analogy" if same_scaffold else "fingerprint_analogy")
            matches.append(match)
        if matches:
            priority, similarity, relationship = max(matches)
            scored.append((priority, similarity, evidence["id"], relationship, evidence))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [{**record, "relationship_to_retrieved_examples": relationship,
             "max_or_prioritized_lean_similarity": round(similarity, 4)}
            for _, similarity, _, relationship, record in scored[:limit]]


def make_packet(cell, ga, seed, cell_id):
    eligible = cell[cell["pareto_eligible"]].copy()
    if len(eligible) < 4:
        raise ValueError(f"Cell {cell_id} has fewer than four eligible rows")
    top = ga.local_pareto_archive(eligible, GA_CONFIG, top_k=3)
    misses = eligible[
        eligible["qd_target_hit"].eq(0) & ~eligible["candidate_id"].isin(top["candidate_id"])
    ].sort_values(["target_gap", "candidate_id"]).head(3)
    representatives = snapshots(top)
    near_misses = snapshots(misses)
    return {
        "parent_cell": cell_id,
        "fixed_grid": GA_CONFIG,
        "ga_archive_scope": "Archived rows. Upstream invalid and duplicate attempts are unavailable.",
        "statistics": {"archived_rows": len(cell), "pareto_eligible_rows": len(eligible),
                       "target_hits": int(eligible["qd_target_hit"].sum())},
        "native_local_pareto_representatives": representatives,
        "near_misses": near_misses,
        "lohc_evidence": literature_matches(seed, representatives + near_misses),
        "previous_plans": [],
    }


def live_plan(packet, prompt, output):
    base_url, model, key = (os.environ.get(name) for name in
                            ("LOHC_LLM_BASE_URL", "LOHC_LLM_MODEL", "LOHC_LLM_API_KEY"))
    if not all((base_url, model, key)):
        raise ValueError("Set LOHC_LLM_BASE_URL, LOHC_LLM_MODEL and LOHC_LLM_API_KEY for --live")
    body = {"model": model, "temperature": 0, "max_tokens": 900,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": prompt},
                         {"role": "user", "content": json_text(packet)}]}
    request = Request(base_url.rstrip("/") + "/chat/completions",
                      data=json.dumps(body).encode("utf-8"),
                      headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urlopen(request, timeout=45) as response:
        result = json.load(response)
    message = result["choices"][0]["message"]["content"]
    write_json(output / "live_audit.json", {"requested_model": model, "returned_model": result.get("model"),
               "parameters": {"temperature": 0, "max_tokens": 900},
               "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
               "usage": result.get("usage"), "raw_content": message})
    return json.loads(message)


def replay_plan(plan, packet, cell, ga):
    expected_keys = {"parent_cell", "hypothesis", "evidence_ids", "rules", "test", "budget_slots"}
    if set(plan) != expected_keys or plan["parent_cell"] != packet["parent_cell"]:
        raise ValueError("Plan schema or parent cell mismatch")
    if not all(isinstance(plan[field], str) and plan[field].strip() for field in ("hypothesis", "test")):
        raise ValueError("Plan requires a hypothesis and test")
    available = {evidence["id"] for evidence in packet["lohc_evidence"]}
    if not isinstance(plan["evidence_ids"], list) or not plan["evidence_ids"] or any(
        not isinstance(item, str) or item not in available for item in plan["evidence_ids"]
    ):
        raise ValueError("Plan evidence IDs must be present in the packet")
    if type(plan["budget_slots"]) is not int or not 1 <= plan["budget_slots"] <= 512:
        raise ValueError("budget_slots must be an integer in [1,512]")
    rules = plan["rules"]
    if not isinstance(rules, list) or not 1 <= len(rules) <= 3:
        raise ValueError("Provide one to three SMARTS rules")
    labels, patterns = [], []
    for rule in rules:
        if set(rule) != {"label", "smarts"} or not all(
            isinstance(value, str) and value.strip() for value in rule.values()
        ):
            raise ValueError("Each rule requires a label and SMARTS string")
        label, smarts = rule["label"], rule["smarts"]
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None or label in {*labels, "remainder"}:
            raise ValueError(f"Invalid rule: {label}")
        labels.append(label)
        patterns.append(pattern)
    assignments = []
    for row in cell.itertuples():
        matches = [label for label, pattern in zip(labels, patterns)
                   if row.mol is not None and row.mol.HasSubstructMatch(pattern)]
        assignments.append({"candidate_id": row.candidate_id, "subcell": matches[0] if matches else "remainder",
                            "all_matching_rules": matches, "pareto_eligible": bool(row.pareto_eligible)})
    replay = cell.copy()
    replay["subcell"] = [row["subcell"] for row in assignments]
    groups = {}
    for label, group in replay.groupby("subcell", sort=True):
        eligible = group[group["pareto_eligible"]]
        groups[label] = {"archived_rows": len(group), "eligible_rows": len(eligible),
                         "native_local_pareto_representatives": snapshots(
                             ga.local_pareto_archive(eligible, GA_CONFIG, top_k=3)
                         ) if not eligible.empty else []}
    if len([group for group in groups.values() if group["eligible_rows"] >= 2]) < 2:
        raise ValueError("Split needs two subcells with at least two eligible examples each")
    if any(groups.get(label, {}).get("eligible_rows", 0) < 2 for label in labels):
        raise ValueError("Every explicit subcell needs at least two eligible examples")
    return {"groups": groups, "assignments": assignments,
            "overlapping_rules": sum(len(row["all_matching_rules"]) > 1 for row in assignments),
            "next_batch_status": "proposed_only"}


def save_memory(path, frame, seed, packet, plan, replay, provenance):
    columns = ["candidate_id", "smi", "sat_smi", "canonical_smi", "canonical_sat_smi", "scaffold_smi",
               "platform_id", "discovery_generation", "operation_generation", "origin_type",
               "parent_smi", "mutation_operator", "fitness_evaluated", "pass_fast_prefilter",
               *PROPERTIES, *OBJECTIVES, "source_chemical_cell", "chemical_behavior_cell", "qd_cell",
               "qd_target_hit", "pareto_eligible", "target_gap"]
    with sqlite3.connect(path) as db:
        frame[columns].to_sql("generation_memory", db, index=False)
        db.executescript("""
            CREATE UNIQUE INDEX generation_id ON generation_memory(candidate_id);
            CREATE INDEX generation_cell ON generation_memory(chemical_behavior_cell);
            CREATE INDEX generation_structure ON generation_memory(canonical_smi, canonical_sat_smi);
            CREATE TABLE lohc_memory (
                evidence_id TEXT PRIMARY KEY, family TEXT, lean_smiles TEXT, rich_smiles TEXT,
                doi TEXT NOT NULL, source_anchor TEXT NOT NULL, record_json TEXT NOT NULL
            );
            CREATE INDEX lohc_structure ON lohc_memory(lean_smiles, rich_smiles);
            CREATE TABLE planning_memory (
                plan_id TEXT PRIMARY KEY, parent_cell INTEGER, provenance TEXT NOT NULL,
                packet_json TEXT NOT NULL, plan_json TEXT NOT NULL, replay_json TEXT NOT NULL
            );
        """)
        db.executemany("INSERT INTO lohc_memory VALUES (?, ?, ?, ?, ?, ?, ?)", [
            (row["id"], row["family"], row["lean_smiles"], row["rich_smiles"], row["doi"],
             row["source_anchor"], json_text(row)) for row in seed
        ])
        if plan is not None:
            db.execute("INSERT INTO planning_memory VALUES (?, ?, ?, ?, ?, ?)", (
                "demo-001", packet["parent_cell"], provenance, json_text(packet),
                json_text(plan), json_text(replay)))
        if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise RuntimeError("SQLite integrity check failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ga-code", type=Path, default=GA_CODE)
    parser.add_argument("--archive", type=Path, default=HERE / "ga_archive.csv")
    parser.add_argument("--seed-memory", type=Path, default=HERE / "lohc_seed.json")
    parser.add_argument("--cell", type=int, default=17)
    parser.add_argument("--out", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", type=Path, help="Replay a reviewed plan or a saved model response")
    modes.add_argument("--live", action="store_true", help="Explicit opt-in to one API call")
    args = parser.parse_args()
    ga_path = args.ga_code.resolve()
    ga = load_native_qd(ga_path)
    RDLogger.DisableLog("rdApp.*")
    seed = load_seed(args.seed_memory)
    frame = import_ga(args.archive, ga)
    cell = frame[frame["chemical_behavior_cell"].eq(args.cell)].copy()
    packet = make_packet(cell, ga, seed, args.cell)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "planning_packet.json", packet)
    prompt_path = HERE / "planner_prompt.txt"
    if args.live:
        plan, provenance = live_plan(packet, prompt_path.read_text(encoding="utf-8"), output), "external_api"
    elif args.plan:
        plan, provenance = read_json(args.plan), "reviewed_offline_plan"
    else:
        plan, provenance = None, "packet_only"
    replay = replay_plan(plan, packet, cell, ga) if plan is not None else None
    if plan is not None:
        write_json(output / "plan.json", plan)
        write_json(output / "replay.json", replay)
    save_memory(output / "memory.sqlite", frame, seed, packet, plan, replay, provenance)
    original = frame["source_chemical_cell"].notna()
    mismatches = int((frame.loc[original, "source_chemical_cell"]
                      != frame.loc[original, "chemical_behavior_cell"]).sum())
    manifest = {"evidence_records": len(seed), "evidence_dois": len({row["doi"] for row in seed}),
                "ga_rows": len(frame), "ga_eligible_rows": int(frame["pareto_eligible"].sum()),
                "historical_cell_labels_compared": int(original.sum()),
                "historical_cell_label_mismatches": mismatches,
                "parent_cell": args.cell, **packet["statistics"], "plan_provenance": provenance,
                "rdkit_version": rdBase.rdkitVersion, "python_version": sys.version.split()[0],
                "ga_source": str(ga_path), "ga_source_sha256": {
                    name: sha256(ga_path / name) for name in ("utils.py", "adaptive_ga.py")},
                "archive_sha256": sha256(args.archive), "seed_sha256": sha256(args.seed_memory),
                "prompt_sha256": sha256(prompt_path)}
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
