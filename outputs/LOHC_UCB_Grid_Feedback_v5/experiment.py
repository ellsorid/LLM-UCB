"""Continue saved native-UCB checkpoints with grid-conditioned reward feedback."""

import argparse
from collections import Counter
from contextlib import nullcontext
from functools import lru_cache
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
RUNTIME = HERE.parents[1] / "work" / "lohc_v2_runtime"
sys.path[:0] = [str(OUTPUTS / "LOHC_Agents_v2"), str(OUTPUTS / "LOHC_QD_Memory_v1")]
import agents
import build_demo as demo

SMOKE_ROOT = HERE.parents[1] / "runs" / "smoke"
HISTORY = {11: "both_seed11_hashseed", 23: "both_seed23"}
SOURCE_FILES = ("utils.py", "adaptive_ga.py", "predict.py", "GNN2.py", "main.py")
PATTERNS = {
    "ring_aryl_ether": Chem.MolFromSmarts("[c]-[O;R]-[C;X4;R]"),
    "sidechain_aryl_ether": Chem.MolFromSmarts("[c]-[O;!R]-[C;X4;!R]"),
    "free_phenol": Chem.MolFromSmarts("[c]-[O;H1]"),
    "aromatic_nitrogen": Chem.MolFromSmarts("[n]"),
}
TERMINAL = Chem.MolFromSmarts("[#6;X4;H3;!R]-[#6;X4;!R]")
MODES = ("retain_on_proxy_improvement", "shorten_alkyl_and_retain")


@lru_cache(maxsize=20000)
def structure(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, (), None
    motifs = tuple(name for name, pattern in PATTERNS.items() if mol.HasSubstructMatch(pattern))
    return mol, motifs, demo.scaffold(smi)


@lru_cache(maxsize=3000)
def shortened(smi):
    parent = structure(smi)[0]
    results = set()
    for atom, _ in parent.GetSubstructMatches(TERMINAL):
        edit = Chem.RWMol(parent)
        edit.RemoveAtom(atom)
        if Chem.SanitizeMol(edit, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE:
            results.add(Chem.MolToSmiles(edit))
    return results


def reward_signal(row, rule, cell, tolerance):
    if rule is None or rule["cell"] != cell or int(row["child_valid"]) != 1:
        return 0
    parent = demo.canonical(row["parent_smi"])
    child = demo.canonical(row["child_smi"])
    if not parent or not child or parent == child:
        return 0
    _, before, scaffold = structure(parent)
    _, after, child_scaffold = structure(child)
    if rule["motif"] not in before:
        return 0
    if rule["mode"] == "shorten_alkyl_and_retain" and child not in shortened(parent):
        return 0
    if rule["motif"] not in after or scaffold != child_scaffold:
        return -1
    h2, dh = row["delta_h2"], row["delta_dh_gain"]
    if np.isfinite(h2) and np.isfinite(dh) and h2 > 0 and dh >= -tolerance:
        return 1
    return 0


def bootstrap(ga, root, seed):
    folder = root / HISTORY[seed]
    meta = demo.read_json(folder / "controlled_config.json")
    cfg = {**meta["config"], **demo.GA_CONFIG, "random_seed": seed}
    archive = demo.import_ga(folder / "frames" / "all_generated_molecules.csv", ga)
    population = pd.read_csv(folder / "frames" / "last_pop.csv", low_memory=False)
    population = ga.annotate_chemical_behavior(population, cfg)
    history = pd.read_csv(folder / "mutation_transitions.csv", low_memory=False)
    complete = archive[archive["pareto_eligible"]].drop_duplicates("canonical_smi")
    return cfg, archive, population, history, complete


def history_map(frame):
    return {smi: int(cell) for smi, cell in zip(frame["canonical_smi"], frame["chemical_behavior_cell"])
            if smi and pd.notna(cell) and cell >= 0}


def make_packet(boot, evidence, protocol):
    pooled = pd.concat([item[4] for item in boot.values()], ignore_index=True)
    pooled = pooled.drop_duplicates("canonical_smi").copy()
    pooled["motifs"] = pooled["canonical_smi"].map(lambda smi: structure(smi)[1])
    events = pd.concat([item[3] for item in boot.values()], ignore_index=True)
    events["canonical_parent"] = events["parent_smi"].map(demo.canonical)
    shortlist, options = [], {}
    groups = []
    for cell, frame in pooled.groupby("chemical_behavior_cell"):
        for motif in PATTERNS:
            group = frame[frame["motifs"].map(lambda values: motif in values)]
            if len(group) >= protocol["minimum_subgrid_parents"]:
                groups.append((int(cell), motif, frame, group))
    groups.sort(key=lambda group: (-len(group[3]), group[0], group[1]))
    for cell, motif, cell_frame, group in groups[:protocol["max_representative_groups"]]:
        eligible = events[events["canonical_parent"].isin(group["canonical_smi"])]
        valid = eligible[eligible["child_valid"].eq(1)]
        representative = group.sort_values(["qd_quality", "canonical_smi"], ascending=[False, True]).iloc[0]
        ident = f"{cell}:{motif}"
        modes = [MODES[0]]
        if group["mol"].map(lambda mol: mol.HasSubstructMatch(TERMINAL)).sum() >= protocol["minimum_subgrid_parents"]:
            modes.append(MODES[1])
        ids = []
        for mode in modes:
            option = f"{ident}:{mode}"
            options[option] = {"cell": cell, "motif": motif, "mode": mode}
            ids.append(option)
        shortlist.append({
            "id": ident, "fixed_cell_n": len(cell_frame), "motif_n": len(group),
            "logged_edits": len(eligible), "valid_logged": len(valid),
            "unique_logged_parents": int(eligible["canonical_parent"].nunique()),
            "hits_among_valid_logged": int(valid["child_target_hit"].sum()),
            "representative": {
                "smi": representative["canonical_smi"], "rich_proxy_smi": representative["canonical_sat_smi"],
                "properties": {key: round(float(representative[key]), 3) for key in demo.PROPERTIES},
            },
            "options": ids,
        })
    return {"groups": shortlist, "lohc_evidence": evidence, "notes": {
        MODES[0]: "if scaffold and motif remain, H2 increases, and distance to 54 kJ/mol worsens by at most 2, +1. On scaffold or motif loss, -1. Otherwise 0.",
        MODES[1]: "apply the same conditions only after exact one-terminal-carbon shortening.",
        "missing": "All figures are observed or native proxies. Invalid or unavailable feedback is zero.",
    }}, options


def validated_rule(response, options, packet):
    if set(response) != {"option_id", "evidence_ids", "hypothesis", "assessments"}:
        raise ValueError("Invalid decision fields")
    if response["option_id"] not in {*options, "abstain"}:
        raise ValueError("Unknown rule")
    if not isinstance(response["hypothesis"], str) or not response["hypothesis"].strip():
        raise ValueError("Missing hypothesis")
    ids = response["evidence_ids"]
    if not isinstance(ids, list) or not set(ids) <= {item["id"] for item in packet["lohc_evidence"]}:
        raise ValueError("Invalid evidence IDs")
    items = response["assessments"]
    expected = {item["id"] for item in packet["groups"]}
    if len(items) != len(expected) or {item["id"] for item in items} != expected:
        raise ValueError("Representative IDs differ")
    if any(set(item) != {"id", "note"} or not item["note"].strip() for item in items):
        raise ValueError("Invalid representative assessment")
    return options.get(response["option_id"])


def diagnostics(rule, checkpoints):
    if rule is None:
        return []
    summaries = []
    for seed, (_, _, _, events, complete) in checkpoints.items():
        cell = complete[complete["chemical_behavior_cell"].eq(rule["cell"])]
        membership = {
            row.canonical_smi: rule["motif"] in structure(row.canonical_smi)[1]
            for row in cell.itertuples()
        }
        table = events.assign(parent_key=events["parent_smi"].map(demo.canonical))
        table = table[table["parent_key"].isin(membership) & table["child_valid"].eq(1)].copy()
        table["matches_motif"] = table["parent_key"].map(membership)
        by_operator = {}
        for operator, rows in table.groupby("mutation_operator", sort=True):
            by_operator[operator] = {}
            for label, subset in (("inside", rows[rows["matches_motif"]]),
                                  ("outside", rows[~rows["matches_motif"]])):
                by_operator[operator][label] = {
                    "valid_logged": len(subset),
                    "unique_parents": int(subset["parent_key"].nunique()),
                    "positive_native_reward": int(subset["operator_reward"].gt(0).sum()),
                    "negative_native_reward": int(subset["operator_reward"].lt(0).sum()),
                }
        summaries.append({
            "seed": seed, "fixed_cell_molecules": len(membership), "motif_molecules": sum(membership.values()),
            "valid_logged_in_cell": len(table), "operators": by_operator,
        })
    return summaries


def softmax(scores, temperature):
    weights = np.exp(np.clip((scores - scores.max()) / temperature, -30, 0))
    return weights / weights.sum()


def continue_seed(ga, torch, native, data, seed, rule, protocol, output, routing=None):
    cfg, archive, population, history, complete = data
    seed_value = seed + 24000
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    ga.GA_FITNESS_CONFIG = cfg
    ga.DELTA_H_CACHE_RING.clear()
    ga.DELTA_H_UNCERTAINTY_CACHE_RING.clear()
    default_rng = np.random.default_rng
    stream = default_rng(seed_value)

    def seeded_rng(arg=None):
        return default_rng(int(stream.integers(0, 2**32)) if arg is None else arg)

    def policy():
        result = native.OperatorExperiencePolicy(
            mode=cfg["adaptive_mode"], beta=cfg["operator_ucb_beta"],
            random_fraction=cfg["operator_random_fraction"], warmup_generations=cfg["operator_warmup_generations"],
            temperature=cfg["operator_temperature"], seed=seed_value,
        )
        result.update(history)
        return result

    coupled, shadow = policy(), policy()
    if coupled.stats != shadow.stats:
        raise RuntimeError("Initial native UCB states differ")
    initial_state_equal = True
    operators = native.MUTATION_OPERATORS
    pool = ga.get_biobased_seed_pool(float(cfg["sa_max_generation"]), float(cfg["max_mol_weight"]))
    if pool.empty:
        pool = population[["smi"]].drop_duplicates().copy()
    archive_smis = set(archive["smi"].astype(str))
    cell_by_smiles = history_map(complete)
    if routing is not None:
        routing.initialize(policy(), coupled, history, cell_by_smiles)
    evaluated = set(complete["canonical_smi"])
    tracker = native.ExplorationTracker(cfg)
    tracker.update(complete, generation=protocol["checkpoint_generation"])
    screening = cfg["survivor_screening"]
    original_choose = coupled.choose_order
    draws, updates, rounds = [], [], []
    generation = protocol["checkpoint_generation"]

    def choose(context, gen):
        if routing is None:
            order, actual = original_choose(context, gen)
            c_scores = coupled._scores(context)
        else:
            order, actual, c_scores, route = routing.choose(context, gen, original_choose)
        a_scores = shadow._scores(context)
        normal = softmax(a_scores, shadow.temperature)
        c_probs = np.array([actual[name] for name in operators])
        if np.allclose(c_probs, 1 / len(operators)):
            normal[:] = 1 / len(operators)
        draws.append({
            "generation": gen + 1, "context": context, "first_operator": operators[order[0]],
            "actual_probabilities": actual, "shadow_same_context_probabilities": dict(zip(operators, normal.tolist())),
            "ucb_max_delta": float(np.max(np.abs(c_scores - a_scores))),
            "max_probability_delta": float(np.max(np.abs(c_probs - normal))),
        })
        if routing is not None:
            draws[-1]["route"] = route
        return order, actual

    coupled.choose_order = choose
    started = time.perf_counter()
    native_prediction = ga.predict_delta_H
    predicted_pairs = 0

    def count_prediction(sat, arom):
        nonlocal predicted_pairs
        predicted_pairs += len(arom)
        return native_prediction(sat, arom)

    ga.predict_delta_H = count_prediction
    try:
        for _ in range(protocol["continuation_generations"]):
            for smi, cell in zip(population["smi"], population["chemical_behavior_cell"]):
                key = demo.canonical(smi)
                if key is not None:
                    cell_by_smiles[key] = int(cell)
            begin, previous_pairs = len(draws), predicted_pairs
            np.random.default_rng = seeded_rng
            try:
                with (routing.observe_parents(ga) if routing is not None else nullcontext()):
                    candidates, elite, _ = ga.generate_candidate_pool(
                        population, pool, cfg, generation, 0, archive_smis=archive_smis, operator_policy=coupled,
                    )
            finally:
                np.random.default_rng = default_rng
            ranked = ga.constrain_and_rank_candidates(
                candidates, elite, screening, compute_full_hydrogenation=False, cfg=cfg,
            )
            ranked = ga.annotate_chemical_behavior(ranked, cfg)
            metrics = tracker.update(ranked, generation=generation + 1)
            transitions = native.build_transition_records(ranked, generation=generation + 1, config=screening)
            changed = transitions.copy()
            signals = []
            for index, row in transitions.iterrows():
                cell = cell_by_smiles.get(demo.canonical(row["parent_smi"]))
                signal = reward_signal(row, rule, cell, protocol["max_enthalpy_target_distance_loss"])
                signals.append(signal)
                if signal:
                    adjusted = float(np.clip(row["operator_reward"] + protocol["reward_weight"] * signal, -1, 1))
                    changed.loc[index, "operator_reward"] = adjusted
                    updates.append({
                        "generation": generation + 1, "operator": row["mutation_operator"],
                        "context": row["experience_context"], "cell": cell, "parent": row["parent_smi"],
                        "child": row["child_smi"], "signal": signal,
                        "native_reward": float(row["operator_reward"]), "fused_reward": adjusted,
                    })
            shadow.update(transitions)
            if routing is None:
                coupled.update(changed)
            else:
                coupled.update(transitions)
                routing.update_local(changed)
            counts_equal = all(
                coupled._entry(context, op)["count"] == shadow._entry(context, op)["count"]
                for context in set(coupled.stats) | set(shadow.stats) for op in operators
            )
            if not counts_equal:
                raise RuntimeError("Native and coupled attribution counts differ")
            fresh = ranked[ranked["fitness_evaluated"].eq(1)].copy()
            fresh["canonical_smi"] = fresh["smi"].map(demo.canonical)
            fresh = fresh[fresh["canonical_smi"].notna() & ~fresh["canonical_smi"].isin(evaluated)]
            fresh = fresh.drop_duplicates("canonical_smi")
            evaluated.update(fresh["canonical_smi"])
            archive_smis.update(candidates["smi"].astype(str))
            ranked = ga.apply_frontier_parent_scores(ga.dedupe_and_sort_population(ranked), cfg)
            population, _, _ = ga.select_next_population(
                ranked, cfg["pop_size"], screening, cfg["keep_core_ratio"], cfg["keep_cluster_ratio"],
                cfg["keep_platform_ratio"], cfg["keep_qd_ratio"], 0.0, random_state=generation + 1, cfg=cfg,
            )
            population = ga.apply_frontier_parent_scores(population, cfg)
            used = draws[begin:]
            rounds.append({
                "generation": generation + 1, "transitions": len(transitions), "native_operator_draws": len(used),
                "nonzero_feedback": sum(signal != 0 for signal in signals),
                "modified_reward": sum(u["fused_reward"] != u["native_reward"]
                                       for u in updates if u["generation"] == generation + 1),
                "max_probability_delta": max((row["max_probability_delta"] for row in used), default=0.0),
                "decisions_with_probability_delta": sum(row["max_probability_delta"] > 1e-10 for row in used),
                "pair_submissions": predicted_pairs - previous_pairs,
                "new_unique_evaluated": len(fresh), "new_unique_native_hits": int(fresh["qd_target_hit"].sum()),
                "new_evaluated_chemical_cells": int(fresh["chemical_behavior_cell"].nunique()),
                "native_tracker_target_hits": int(metrics["generation_target_hits"]),
            })
            print(seed, rounds[-1], flush=True)
            generation += 1
    finally:
        ga.predict_delta_H = native_prediction
    demo.write_json(output / f"seed{seed}_detail.json", {
        "seed": seed, "native_initial_reward_state_matched": initial_state_equal,
        "elapsed_seconds": round(time.perf_counter() - started, 3), "rounds": rounds, "updates": updates,
        "operator_draws": draws, "attribution_counts_matched_shadow": True,
    })
    return {"seed": seed, "rounds": rounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--history-root", type=Path, default=SMOKE_ROOT)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    output, runtime, history_root = args.out.resolve(), args.runtime.resolve(), args.history_root.resolve()
    key_file = args.key_file.resolve() if args.key_file else None
    replay = args.replay.resolve() if args.replay else None
    if os.environ.get("PYTHONHASHSEED") != "11":
        raise ValueError("Set PYTHONHASHSEED=11 before launch")
    protocol = demo.read_json(HERE / "protocol.json")
    if protocol["max_llm_calls"] != 1:
        raise ValueError("This protocol allows one shared planning call")
    output.mkdir(parents=True, exist_ok=False)
    hashes = {file: demo.sha256(runtime / file) for file in SOURCE_FILES}
    if hashes != {file: demo.sha256(demo.GA_CODE / file) for file in SOURCE_FILES}:
        raise RuntimeError("Native runtime source differs from GAcode")
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    import torch
    import utils as ga
    import adaptive_ga as native
    import predict
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(2)
    predict.PREDICT_CFG["device"] = "cpu"
    boot = {seed: bootstrap(ga, history_root, seed) for seed in protocol["seeds"]}
    gate = {str(seed): {"unique": len(data[4]), "cells": int(data[4]["chemical_behavior_cell"].nunique())}
            for seed, data in boot.items()}
    demo.write_json(output / "preflight.json", {
        "gate": gate, "sources": hashes, "protocol_sha256": demo.sha256(HERE / "protocol.json"),
        "weights": {path.name: demo.sha256(path) for path in sorted((runtime / "model_checkpoints").glob("fold*_best.pth"))},
        "history": {str(seed): {file: demo.sha256(history_root / HISTORY[seed] / file) for file in (
            "controlled_config.json", "mutation_transitions.csv", "frames/last_pop.csv",
            "frames/all_generated_molecules.csv")} for seed in boot},
    })
    if any(entry["unique"] < protocol["minimum_unique_evaluated"]
           or entry["cells"] < protocol["minimum_occupied_chemical_cells"] for entry in gate.values()):
        raise RuntimeError("Historical checkpoint does not pass coverage gate")
    memory = demo.read_json(OUTPUTS / "LOHC_Grid_Guided_GA_v4" / "live_run" / "frozen_memory.json")
    evidence = [{key: item[key] for key in ("id", "family", "finding", "doi", "source_anchor")} for item in memory]
    packet, options = make_packet(boot, evidence, protocol)
    demo.write_json(output / "input.json", packet)
    packet_hash = demo.sha256(output / "input.json")
    result = {"input_sha256": packet_hash, "rule": None, "response": None, "fallback": None}
    try:
        if replay:
            previous = demo.read_json(replay / "decision.json")
            if previous["input_sha256"] != packet_hash:
                raise ValueError("Frozen decision packet differs")
            result = previous
            if result["response"] is not None:
                validated_rule(result["response"], options, packet)
        else:
            response = agents.DeepSeek(agents.credential(key_file), output).run("grid_ucb", protocol["prompt"], packet)
            result.update(rule=validated_rule(response, options, packet), response=response)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError) as error:
        if replay:
            raise
        result["fallback"] = type(error).__name__
    demo.write_json(output / "decision.json", result)
    rule = result["rule"]
    fit = diagnostics(rule, boot)
    demo.write_json(output / "fixed_grid_diagnostics.json", fit)
    runs = [continue_seed(ga, torch, native, boot[seed], seed, rule, protocol, output) for seed in boot]
    demo.write_json(output / "summary.json", {
        "scope": "C continuation from saved native UCB generation 8 checkpoints; shadow A uses identical C transitions; no independently generated post-checkpoint A",
        "gate": gate, "rule": rule, "grid_diagnostics": fit, "runs": runs,
    })


if __name__ == "__main__":
    main()
