"""Single-seed, three-arm, full occupied-grid experiment using the 40–70 window."""

import argparse
from collections import Counter
from contextlib import nullcontext
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
V7 = HERE.parent / "LOHC_Full_Grid_Budget_v7"
sys.path.insert(0, str(V7))
import full_grid as v7
from stage_runtime import stage

v5, demo = v7.v5, v7.demo
SEED, POPULATION, GENERATIONS, PENDING_LIMIT = 11, 250, 3, 600
ARMS = ("A_native", "B_fixed_llm", "C_managed_llm")
WINDOW_FIELDS = {"dh_target": 54.0, "dh_window_low": 40.0, "dh_window_high": 70.0}
WINDOW_PROMPT = (
    v7.PROMPT + "本试验唯一的焓变接受窗口是40–70 kJ/mol H2，两端均包含。"
    "UCB和QD的优选参考值是54。native_hits已按新窗口重算。"
)


def boundary_checks(ga, native, cfg, sample):
    values = [39.999, 40.0, 47.999, 48.0, 54.0, 62.0, 70.0, 70.001]
    expected = [0, 1, 1, 1, 1, 1, 1, 0]
    base = {"platform_id": native.BIO_PLATFORM_IDS[0], "h2_wt_percent": 5.0, "SA score": 2.0}
    frame = pd.DataFrame([{**base, "delta_H_per_H2": value} for value in values])
    actual = ga.annotate_qd_cells(frame)["qd_target_hit"].tolist()
    tracker = native.ExplorationTracker(cfg)
    tracked = [int(tracker._target_cell(row) is not None) for row in frame.to_dict("records")]
    states = [int(native.classify_objective_state(row, cfg["survivor_screening"])[1] == "in_window")
              for row in frame.to_dict("records")]
    prototype = sample.iloc[0].to_dict()
    filters = pd.DataFrame([{**prototype, **row} for row in frame.to_dict("records")])
    accepted = ga.apply_multicriteria_filter(filters, cfg["survivor_screening"])["pass_enthalpy_window"].tolist()
    transitions = frame.assign(
        origin_type="guided_mutation", operation_generation=1, parent_h2_wt_percent=5.0,
        parent_delta_H_per_H2=54.0, parent_SA_score=2.0, parent_selected_pareto=0,
    )
    rewarded = native.build_transition_records(transitions, 1, cfg["survivor_screening"])["child_target_hit"].tolist()
    series = {"qd": actual, "tracker": tracked, "ucb_state": states, "screening": accepted, "ucb_bonus": rewarded}
    if any(result != expected for result in series.values()):
        raise RuntimeError(f"Enthalpy boundary mismatch: {series}")
    return {"delta_H": values, "expected": expected, **series,
            "qd_bins": [ga._qd_bin(value, ga.QD_DH_EDGES) if passed else None
                        for value, passed in zip(values, expected)]}


def scored_catalog(archive, native, dh_quality):
    cols = [*v7.ARCHIVE_COLS, "platform_id"]
    catalog = v7.Catalog()
    seen_all, evaluated = set(), set()
    archived_hits = 0
    for frame in pd.read_csv(archive, usecols=cols, chunksize=20000):
        seen_all.update(frame["smi"].dropna())
        h2, dh, sa = (frame[key] for key in demo.PROPERTIES)
        finite = np.isfinite(h2) & np.isfinite(dh) & np.isfinite(sa)
        evaluated.update(frame.loc[finite & frame["fitness_evaluated"].eq(1), "smi"].dropna())
        hit = frame["platform_id"].isin(native.BIO_PLATFORM_IDS) & h2.ge(4.2) & dh.between(40, 70) & sa.le(4)
        frame["qd_target_hit"] = (hit & finite).astype(int)
        archived_hits += int((hit & finite & frame["fitness_evaluated"].eq(1)).sum())
        frame["qd_quality"] = (np.clip((h2 - 4.2) / 2.8, 0, 1) + dh_quality(dh)
                               + np.clip((4 - sa) / 3, 0, 1)) / 3
        catalog.add(frame)
    return catalog, seen_all, evaluated, archived_hits


def clone_catalog(source):
    catalog = v7.Catalog()
    catalog.cells = source.cells.copy()
    catalog.members = source.members.copy()
    catalog.totals = source.totals.copy()
    catalog.valid = source.valid.copy()
    return catalog


def new_policy(native, cfg, seed):
    return native.OperatorExperiencePolicy(
        mode=cfg["adaptive_mode"], beta=cfg["operator_ucb_beta"],
        random_fraction=cfg["operator_random_fraction"], warmup_generations=cfg["operator_warmup_generations"],
        temperature=cfg["operator_temperature"], seed=seed,
    )


def fused_global(transitions, catalog, rules):
    fused = transitions.copy()
    signs = Counter()
    for index, row in transitions.iterrows():
        cell = catalog.cells.get(row["parent_smi"])
        motif = rules.get(cell)
        if motif is None:
            continue
        rule = {"cell": cell, "motif": motif, "mode": "retain_on_proxy_improvement"}
        signal = v5.reward_signal(row, rule, cell, 2.0)
        if signal:
            fused.loc[index, "operator_reward"] = np.clip(row["operator_reward"] + 0.2 * signal, -1, 1)
            signs[signal] += 1
    return fused, signs


def pair_trace(pairs, ranked, seen, generation, offset):
    rows = {row["smi"]: row for row in ranked.to_dict("records")}
    records = []
    for index, (lean, rich) in enumerate(pairs, offset + 1):
        row = rows.get(lean)
        valid = row is not None and all(pd.notna(row[name]) and np.isfinite(row[name]) for name in demo.PROPERTIES)
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


def measure(records):
    hits = [row for row in records if row["new_native_hit"]]
    return {
        "gnn_pairs_submitted": len(records), "new_unique_evaluated": sum(row["new_valid_evaluation"] for row in records),
        "new_unique_native_hits": len(hits), "first_new_hit": hits[0]["index"] if hits else None,
        "best_native_quality": max((row["qd_quality"] for row in records if row["new_valid_evaluation"]), default=None),
        "chemical_cells": len({row["chemical_cell"] for row in records if row["new_valid_evaluation"]}),
        "target_cells": len({row["target_cell"] for row in hits}),
        "hit_scaffolds": len({demo.scaffold(row["smi"]) for row in hits} - {None, ""}),
    }


def run_arm(ga, native, torch, cfg, start, name, initial, all_archive, original_seen, base_catalog,
            decisions, seed_pool, out, *, monitor=None, generations=GENERATIONS):
    population, known, seen = initial.copy(), all_archive.copy(), original_seen.copy()
    catalog = clone_catalog(base_catalog)
    continuation_seed = SEED + 48000
    random.seed(continuation_seed)
    np.random.seed(continuation_seed)
    torch.manual_seed(continuation_seed)
    ga.DELTA_H_CACHE_RING.clear()
    ga.DELTA_H_UNCERTAINTY_CACHE_RING.clear()
    policy = new_policy(native, cfg, continuation_seed)
    router = v7.Router(native, policy, catalog) if name == ARMS[2] else None
    if router:
        router.configure(decisions, [])
    rules = {row["cell"]: row["motif"] for row in decisions if row["plan"] == "split"}
    rng = np.random.default_rng
    stream = rng(continuation_seed)

    def seeded_rng(value=None):
        return rng(int(stream.integers(0, 2**32)) if value is None else value)

    predict = ga.predict_delta_H
    pairs, records, rounds = [], [], []

    def observe(sat, aromatic):
        pairs.extend(zip(aromatic, sat))
        return predict(sat, aromatic)

    ga.predict_delta_H = observe
    screening = cfg["survivor_screening"]
    try:
        for generation in range(start, start + generations):
            begun = time.perf_counter()
            pairs.clear()
            if router:
                router.reads.clear()
            try:
                np.random.default_rng = seeded_rng
                with (router.connect(ga) if router else nullcontext()), (
                    monitor.capture_attempts(ga, catalog) if monitor is not None else nullcontext()
                ):
                    candidates, elite, _ = ga.generate_candidate_pool(
                        population, seed_pool, cfg, generation, 0, archive_smis=known, operator_policy=policy,
                    )
            finally:
                np.random.default_rng = rng
            ranked = ga.constrain_and_rank_candidates(candidates, elite, screening, False, cfg)
            ranked = ga.annotate_chemical_behavior(ranked, cfg)
            transitions = native.build_transition_records(ranked, generation + 1, screening)
            counts = {}
            if monitor is not None or (name == ARMS[1] and not transitions.empty):
                if monitor is not None:
                    adjusted, signs = monitor.adjust(transitions, catalog)
                else:
                    adjusted, signs = fused_global(transitions, catalog, rules)
                policy.update(adjusted)
                counts = {"modified_rewards": sum(signs.values()), "positive": signs[1], "negative": signs[-1]}
            else:
                policy.update(transitions)
                if router:
                    router.update(transitions.to_dict("records"), fused=True)
                    counts = dict(router.reads)
            trace = pair_trace(pairs, ranked, seen, generation + 1, len(records))
            records.extend(trace)
            catalog.add(ranked)
            known.update(candidates["smi"])
            ranked = ga.apply_frontier_parent_scores(ga.dedupe_and_sort_population(ranked), cfg)
            population, _, _ = ga.select_next_population(
                ranked, cfg["pop_size"], screening, cfg["keep_core_ratio"], cfg["keep_cluster_ratio"],
                cfg["keep_platform_ratio"], cfg["keep_qd_ratio"], 0.0, random_state=generation + 1, cfg=cfg,
            )
            population = ga.apply_frontier_parent_scores(population, cfg)
            if monitor is not None:
                monitor.review(generation + 1, generation + 1 < start + generations, population, trace, catalog)
            row = {"generation": generation + 1, **measure(trace), "transition_records": len(transitions),
                   "ucb": counts, "elapsed_seconds": round(time.perf_counter() - begun, 3)}
            rounds.append(row)
            print(name, row, flush=True)
    finally:
        ga.predict_delta_H = predict
    demo.write_json(out / f"{name}.json", {"rounds": rounds, "pairs": records})
    return rounds, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--runtime", type=Path, default=HERE.parents[1] / "work" / "lohc_v8_deterministic_runtime")
    parser.add_argument("--base-runtime", type=Path, default=v5.RUNTIME)
    parser.add_argument("--history", type=Path, default=v7.HISTORY)
    parser.add_argument("--replay-plan", type=Path)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "11":
        raise ValueError("Set PYTHONHASHSEED=11 before launch")
    output, runtime, history = (path.resolve() for path in (args.out, args.runtime, args.history))
    replay = args.replay_plan.resolve() if args.replay_plan else None
    key = args.key_file.resolve() if args.key_file else None
    if not runtime.exists():
        stage(args.base_runtime.resolve(), runtime)
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    import torch
    import utils as ga
    import adaptive_ga as native
    import main as original
    import predict
    from enthalpy_window import dh_quality
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(2)
    predict.PREDICT_CFG["device"] = "cpu"
    cfg = v7.history_config(history, original.CONFIG)
    cfg.update(WINDOW_FIELDS, lohc_dh_low=40.0, lohc_dh_high=70.0, pop_size=POPULATION, num_parents=POPULATION,
               max_pending_gnn_per_generation=PENDING_LIMIT)
    screening = {**cfg["survivor_screening"], **WINDOW_FIELDS}
    cfg["survivor_screening"] = screening
    ga.GA_FITNESS_CONFIG = cfg
    run_history = pd.read_csv(history / "run_summary.csv").iloc[0]
    if int(run_history["random_seed"]) != SEED:
        raise RuntimeError("Incorrect seed checkpoint")
    start = int(run_history["generations_completed"])
    saved = pd.read_csv(history / "frames" / "last_pop.csv", low_memory=False).drop(columns=["mol"], errors="ignore")
    boundaries = boundary_checks(ga, native, cfg, saved)
    saved = ga.annotate_pareto_population(saved, screening)
    saved = ga.apply_frontier_parent_scores(saved, cfg)
    initial, _, _ = ga.select_next_population(
        saved, POPULATION, screening, cfg["keep_core_ratio"], cfg["keep_cluster_ratio"],
        cfg["keep_platform_ratio"], cfg["keep_qd_ratio"], 0.0, random_state=SEED, cfg=cfg,
    )
    initial = ga.apply_frontier_parent_scores(initial, cfg)
    if len(initial) != POPULATION:
        raise RuntimeError("Unexpected checkpoint population size")
    archive = history / "frames" / "all_generated_molecules.csv"
    catalog, archived, evaluated, archived_hits = scored_catalog(archive, native, dh_quality)
    demo.write_json(output / "preflight.json", {
        "seed": SEED, "checkpoint": start, "window": WINDOW_FIELDS, "boundary_checks": boundaries,
        "population": POPULATION, "generations_per_arm": GENERATIONS, "pending_limit_per_generation": PENDING_LIMIT,
        "archive_rows_with_valid_properties": len(evaluated), "archived_native_hits_recomputed": archived_hits,
        "occupied_chemical_cells": len(catalog.totals), "initial_population_smis": initial["smi"].tolist(),
        "sources": {name: demo.sha256(runtime / name) for name in (*v5.SOURCE_FILES, "enthalpy_window.py")},
        "weights": {path.name: demo.sha256(path) for path in sorted((runtime / "model_checkpoints").glob("fold*_best.pth"))},
    })
    print("preflight", len(catalog.totals), archived_hits, boundaries["expected"], flush=True)
    history_rows = []
    for frame in pd.read_csv(history / "mutation_transitions.csv", chunksize=25000):
        history_rows.extend(frame[frame["generation"].gt(start - v7.WINDOW)].to_dict("records"))
    cells = v7.planning_cells(catalog, initial, history_rows)
    evidence = demo.read_json(HERE.parent / "LOHC_Grid_Guided_GA_v4" / "live_run" / "frozen_memory.json")
    evidence = [{field: row[field] for field in ("id", "family", "finding")} for row in evidence]
    llm_output = output / "full_grid"
    llm_output.mkdir()
    demo.write_json(llm_output / "input.json", {"cells": cells, "evidence": evidence, "prompt": WINDOW_PROMPT})
    input_hash = demo.sha256(llm_output / "input.json")
    if replay:
        if demo.sha256(replay / "input.json") != input_hash:
            raise RuntimeError("Frozen full-grid plan input changed")
        decisions = demo.read_json(replay / "decisions.json")
        usage = demo.read_json(replay / "llm_usage.json")
        demo.write_json(llm_output / "decisions.json", decisions)
        demo.write_json(llm_output / "llm_usage.json", usage)
    else:
        llm = v5.agents.DeepSeek(v5.agents.credential(key), llm_output)
        decisions, usage = v7.request_round(llm, llm_output, cells, evidence, 6,
                                           prompt=WINDOW_PROMPT, preserve_reviews=True)
    if len(decisions) != len(cells):
        raise RuntimeError("Full network review did not cover all occupied cells")
    print("full review", usage, flush=True)
    seed_pool = ga.get_biobased_seed_pool(cfg["sa_max_generation"], cfg["max_mol_weight"])
    if seed_pool.empty:
        seed_pool = initial[["smi"]].drop_duplicates()
    data = {name: run_arm(ga, native, torch, cfg, start, name, initial, archived, evaluated,
                          catalog, decisions, seed_pool, output) for name in ARMS}
    first = [[(row["smi"], row["rich_proxy_smi"], row["new_native_hit"], row["qd_quality"]) for row in data[name][1]
              if row["generation"] == start + 1] for name in ARMS]
    if first[0] != first[1] or first[0] != first[2]:
        raise RuntimeError("Arms differed before their first UCB feedback update")
    common = min(len(trace) for _, trace in data.values())
    demo.write_json(output / "comparison.json", {
        "seed": SEED, "window": WINDOW_FIELDS, "checkpoint": start, "shared_first_round_verified": True,
        "full_grid_usage": usage, "llm_requests_this_execution": 0 if replay else usage["actual_requests"],
        "rule_fallback_cells": [row["cell"] for row in decisions if "fallback" in row],
        "common_gnn_pair_budget": common,
        "comparison_at_common_budget": {name: measure(trace[:common]) for name, (_, trace) in data.items()},
        "arm_totals": {name: measure(trace) for name, (_, trace) in data.items()},
        "rounds": {name: rounds for name, (rounds, _) in data.items()},
    })


if __name__ == "__main__":
    main()
