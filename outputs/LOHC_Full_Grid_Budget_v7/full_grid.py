"""Measure full occupied-grid DeepSeek usage during three native GA generations."""

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
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
V5 = HERE.parent / "LOHC_UCB_Grid_Feedback_v5"
sys.path.insert(0, str(V5))
import experiment as v5

demo = v5.demo
HISTORY = HERE.parents[1] / "runs" / "native_history"
SEED, GENERATIONS, WINDOW, MIN_PARENTS, BATCH_GROUPS = 11, 3, 20, 5, 8
PROMPT = (
    "你管理高通量LOHC GA固定化学格的局部UCB经验。评价每个输入结构子群一次，"
    "并对每个格选择 keep（共享原生经验）或 split（选一个允许拆分的 motif，只对目标和其余组分别积累经验）。"
    "已知性质是代理值。文献只用于风险与保留项，不证明生成分子真实来源或实测循环。"
    "只输出JSON，格式为 {\"cells\":[{\"cell\":31,\"plan\":\"split\",\"motif\":\"aromatic_nitrogen\","
    "\"groups\":{\"aromatic_nitrogen\":\"retain\",\"free_phenol\":\"investigate\"}}]}。"
    "group值仅允许 retain、investigate、insufficient。keep 的 motif 为 null。"
    "输入所有格和全部子群必须逐一覆盖。没有充分的候选和历史时选 keep。"
)
NAMES = tuple(v5.PATTERNS)
STATS = ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
ARCHIVE_COLS = ("smi", "sat_smi", "fitness_evaluated", "chemical_behavior_cell", "qd_quality",
                "qd_target_hit", *demo.PROPERTIES)


@lru_cache(maxsize=6000)
def labels(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return tuple(name for name, pattern in v5.PATTERNS.items() if mol is not None and mol.HasSubstructMatch(pattern))


def history_config(path, defaults):
    cfg = dict(defaults)
    with (path / "hyperparams.csv").open(encoding="utf-8-sig", newline="") as handle:
        for name, value in list(csv.reader(handle))[1:]:
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                pass
            cfg["survivor_screening" if name == "screening_config" else name] = value
    return cfg


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
                    or any(not np.isfinite(row[name]) for name in demo.PROPERTIES)):
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
                    group["top"] = {key: row[key] for key in ("smi", "sat_smi", "qd_quality", *demo.PROPERTIES)}

    def record(self, cell, motif, recent, parents):
        group = self.group_stats.get((cell, motif), {"n": 0, "hits": 0, "top": None})
        top = group["top"]
        rep = None if top is None else {
            key: (None if pd.isna(value) else round(float(value), 3) if isinstance(value, (float, np.floating)) else value)
            for key, value in top.items()
        }
        return {
            "n": group["n"], "native_hits": group["hits"], "active_parents": parents[cell, motif],
            "recent_edits": dict(recent[cell, motif]), "representative": rep,
        }


def planning_cells(catalog, population, history):
    recent, parents = defaultdict(Counter), Counter()
    for row in history:
        smi = row["parent_smi"]
        cell = catalog.cells.get(smi)
        if cell is not None:
            for label in ("all_valid", *catalog.members.get(smi, ())):
                recent[cell, label][row["mutation_operator"]] += 1
    for smi in population["smi"]:
        cell = catalog.cells.get(smi)
        if cell is not None:
            for label in ("all_valid", *catalog.members.get(smi, ())):
                parents[cell, label] += 1
    cells = []
    for cell in sorted(catalog.totals):
        motifs = [name for name in NAMES if catalog.group_stats.get((cell, name), {}).get("n", 0) >= MIN_PARENTS]
        shown = motifs or ["all_valid"]
        cells.append({
            "cell": cell, "archived_structures": catalog.totals[cell],
            "split_options": [name for name in motifs if parents[cell, name] > 0],
            "groups": {name: catalog.record(cell, name, recent, parents) for name in shown},
        })
    return cells


def batches(cells):
    result, batch, groups = [], [], 0
    for cell in cells:
        weight = len(cell["groups"])
        if batch and groups + weight > BATCH_GROUPS:
            result.append(batch)
            batch, groups = [], 0
        batch.append(cell)
        groups += weight
    if batch:
        result.append(batch)
    return result


def validate_plan(raw, cells, preserve_reviews=False):
    expected = {row["cell"]: row for row in cells}
    if (set(raw) != {"cells"} or len(raw["cells"]) != len(expected)
            or {row["cell"] for row in raw["cells"]} != set(expected)):
        raise ValueError("Incomplete grid coverage")
    decisions = []
    for row in raw["cells"]:
        cell = expected[row["cell"]]
        if (set(row) != {"cell", "plan", "motif", "groups"} or set(row["groups"]) != set(cell["groups"])
                or set(row["groups"].values()) - {"retain", "investigate", "insufficient"}):
            raise ValueError("Incomplete subgroup coverage")
        if row["plan"] == "keep" and row["motif"] is None:
            decisions.append(row)
            continue
        if row["plan"] != "split" or row["motif"] not in cell["split_options"]:
            if preserve_reviews and row["plan"] in {"keep", "split"}:
                decisions.append({**row, "plan": "keep", "motif": None, "fallback": "unsupported_plan"})
                continue
            raise ValueError("Unsupported split")
        decisions.append(row)
    return decisions


def request_round(llm, folder, cells, evidence, workers, *, prompt=PROMPT, preserve_reviews=False):
    packets = batches(cells)

    def ask(job):
        index, batch = job
        packet = {"literature": evidence, "cells": batch}
        all_attempts = []
        for attempt in (1, 2):
            role = f"batch_{index:02d}_attempt_{attempt}"
            try:
                response = llm.run(role, prompt, packet)
                return validate_plan(response, batch, preserve_reviews), all_attempts
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
                all_attempts.append(type(error).__name__)
        return [], all_attempts

    with ThreadPoolExecutor(max_workers=workers) as executor:
        responses = list(executor.map(ask, enumerate(packets, start=1)))
    decisions = [row for records, _ in responses for row in records]
    cost = Counter()
    audits = sorted(folder.glob("*_audit.json"))
    for file in audits:
        stats = demo.read_json(file).get("usage") or {}
        cost.update({key: stats.get(key, 0) for key in STATS})
    demo.write_json(folder / "packets.json", packets)
    demo.write_json(folder / "decisions.json", decisions)
    result = {
        "occupied_cells": len(cells), "subgroups_sent": sum(len(cell["groups"]) for cell in cells),
        "cells_evaluated": len(decisions), "subgroups_evaluated": sum(len(row["groups"]) for row in decisions),
        "split_plans": sum(row["plan"] == "split" for row in decisions),
        "planned_batches": len(packets), "actual_requests": len(audits),
        "batch_errors": [errors for _, errors in responses if errors], **cost,
    }
    demo.write_json(folder / "llm_usage.json", result)
    return decisions, result


class Router:
    def __init__(self, native, pooled, catalog):
        self.native = native
        self.pooled = pooled
        self.catalog = catalog
        self.local = native.OperatorExperiencePolicy(
            mode=pooled.mode, beta=pooled.beta, random_fraction=pooled.random_fraction,
            warmup_generations=pooled.warmup_generations, temperature=pooled.temperature,
        )
        self.local.rng = pooled.rng
        self.rules = {}
        self.warmed = set()
        self.support = defaultdict(set)
        self.reads = Counter()
        self.parent = None

    def configure(self, decisions, history):
        self.rules = {row["cell"]: row["motif"] for row in decisions if row["plan"] == "split"}
        new = set(self.rules.items()) - self.warmed
        self.update(history, active=new)
        self.warmed.update(new)
        self.reads.clear()

    def key(self, row, motif, cell):
        parent = row["parent_smi"]
        group = "target" if motif in self.catalog.members.get(parent, ()) else "remainder"
        return f"{cell}:{motif}:{group}|{row['experience_context']}", group

    def update(self, rows, active=None, fused=False):
        active = set(self.rules.items()) if active is None else active
        if not active:
            return
        motifs = dict(active)
        updates = []
        for row in rows:
            parent = row["parent_smi"]
            cell = self.catalog.cells.get(parent)
            motif = motifs.get(cell)
            if motif is None:
                continue
            key, group = self.key(row, motif, cell)
            self.support[key].add(parent)
            update = dict(row, experience_context=key)
            if fused and group == "target":
                rule = {"cell": cell, "motif": motif, "mode": "retain_on_proxy_improvement"}
                signal = v5.reward_signal(row, rule, cell, 2.0)
                if signal:
                    self.reads["modified_rewards"] += 1
                    update["operator_reward"] = float(np.clip(row["operator_reward"] + 0.2 * signal, -1, 1))
            updates.append(update)
        self.local.update(pd.DataFrame(updates))

    @contextmanager
    def connect(self, ga):
        classify, choose = ga.classify_objective_state, self.pooled.choose_order

        def observe(row, cfg=None):
            self.parent = row["smi"]
            return classify(row, cfg)

        def routed(context, generation):
            cell = self.catalog.cells.get(self.parent)
            motif = self.rules.get(cell)
            if motif:
                key, group = self.key({"parent_smi": self.parent, "experience_context": context}, motif, cell)
                if len(self.support[key]) >= MIN_PARENTS:
                    self.reads[group + "_local"] += 1
                    return self.local.choose_order(key, generation)
            self.reads["pooled"] += 1
            return choose(context, generation)

        ga.classify_objective_state = observe
        self.pooled.choose_order = routed
        try:
            yield
        finally:
            ga.classify_objective_state = classify
            self.pooled.choose_order = choose
            self.parent = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--history", type=Path, default=HISTORY)
    parser.add_argument("--runtime", type=Path, default=v5.RUNTIME)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    output, history, runtime = (p.resolve() for p in (args.out, args.history, args.runtime))
    key = args.key_file.resolve() if args.key_file else None
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    import torch
    import utils as ga
    import adaptive_ga as native
    import main as original
    import predict
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(2)
    predict.PREDICT_CFG["device"] = "cpu"
    cfg = history_config(history, original.CONFIG)
    ga.GA_FITNESS_CONFIG = cfg
    actual_checkpoint = pd.read_csv(history / "run_summary.csv")["generations_completed"].iloc[0]
    start = int(actual_checkpoint)
    population = pd.read_csv(history / "frames" / "last_pop.csv", low_memory=False).drop(columns=["mol"], errors="ignore")
    population = ga.annotate_chemical_behavior(population, cfg)
    archive_file = history / "frames" / "all_generated_molecules.csv"
    catalog = Catalog()
    for frame in pd.read_csv(archive_file, usecols=list(ARCHIVE_COLS), chunksize=20000):
        catalog.add(frame)
    print("archive", len(catalog.cells), len(catalog.valid), len(catalog.totals), flush=True)
    transitions_file = history / "mutation_transitions.csv"
    recent = []
    for frame in pd.read_csv(transitions_file, chunksize=25000):
        recent.extend(frame[frame["generation"].gt(start - WINDOW)].to_dict("records"))
    evidence = demo.read_json(HERE.parent / "LOHC_Grid_Guided_GA_v4" / "live_run" / "frozen_memory.json")
    evidence = [{key: row[key] for key in ("id", "family", "finding")} for row in evidence]
    continuation_seed = SEED + 47000
    random.seed(continuation_seed)
    np.random.seed(continuation_seed)
    torch.manual_seed(continuation_seed)
    default_rng = np.random.default_rng
    stream = default_rng(continuation_seed)

    def seeded_rng(value=None):
        return default_rng(int(stream.integers(0, 2**32)) if value is None else value)

    pooled = native.OperatorExperiencePolicy(
        mode=cfg["adaptive_mode"], beta=cfg["operator_ucb_beta"], random_fraction=cfg["operator_random_fraction"],
        warmup_generations=cfg["operator_warmup_generations"], temperature=cfg["operator_temperature"],
        seed=continuation_seed,
    )
    pooled.stats = demo.read_json(history / "operator_experience_state.json")
    router = Router(native, pooled, catalog)
    seed_pool = ga.get_biobased_seed_pool(cfg["sa_max_generation"], cfg["max_mol_weight"])
    if seed_pool.empty:
        seed_pool = population[["smi"]].drop_duplicates()
    screening = cfg["survivor_screening"]
    credential = v5.agents.credential(key)
    records = []
    native_predict = ga.predict_delta_H
    submitted = 0

    def count(sat, arom):
        nonlocal submitted
        submitted += len(arom)
        return native_predict(sat, arom)

    ga.predict_delta_H = count
    try:
        for generation in range(start, start + GENERATIONS):
            begun, before = time.perf_counter(), submitted
            folder = output / f"generation_{generation + 1}"
            folder.mkdir()
            cells = planning_cells(catalog, population, recent)
            decisions, usage = request_round(v5.agents.DeepSeek(credential, folder), folder, cells, evidence, args.workers)
            print(generation + 1, "LLM", usage, flush=True)
            router.configure(decisions, recent)
            try:
                np.random.default_rng = seeded_rng
                with router.connect(ga):
                    candidates, elite, _ = ga.generate_candidate_pool(
                        population, seed_pool, cfg, generation, 0, archive_smis=set(catalog.cells), operator_policy=pooled,
                    )
            finally:
                np.random.default_rng = default_rng
            ranked = ga.constrain_and_rank_candidates(candidates, elite, screening, False, cfg)
            ranked = ga.annotate_chemical_behavior(ranked, cfg)
            transition_frame = native.build_transition_records(ranked, generation + 1, screening)
            new_records = transition_frame.to_dict("records")
            pooled.update(transition_frame)
            router.update(new_records, fused=True)
            recent = [row for row in recent + new_records if row["generation"] > generation + 1 - WINDOW]
            fresh = ranked[ranked["fitness_evaluated"].eq(1) & ~ranked["smi"].isin(catalog.valid)]
            fresh = fresh.drop_duplicates("smi")
            catalog.add(ranked)
            ranked = ga.apply_frontier_parent_scores(ga.dedupe_and_sort_population(ranked), cfg)
            population, _, _ = ga.select_next_population(
                ranked, cfg["pop_size"], screening, cfg["keep_core_ratio"], cfg["keep_cluster_ratio"],
                cfg["keep_platform_ratio"], cfg["keep_qd_ratio"], 0.0, random_state=generation + 1, cfg=cfg,
            )
            population = ga.apply_frontier_parent_scores(population, cfg)
            row = {"generation": generation + 1, **usage, "gnn_pairs_submitted": submitted - before,
                   "new_evaluated": len(fresh), "new_native_hits": int(fresh["qd_target_hit"].sum()),
                   "after_ga_occupied_cells": len(catalog.totals), "ucb": dict(router.reads),
                   "elapsed_seconds": round(time.perf_counter() - begun, 3)}
            records.append(row)
            demo.write_json(folder / "generation.json", row)
            print("completed", row, flush=True)
    finally:
        ga.predict_delta_H = native_predict
    demo.write_json(output / "summary.json", {
        "checkpoint": start, "seed": SEED, "pop_size": cfg["pop_size"], "max_batch_subgroups": BATCH_GROUPS,
        "archived_population_file": str(history / "frames" / "last_pop.csv"),
        "local_experience_window": WINDOW, "generations": records,
        "total": {key: sum(row[key] for row in records) for key in (*STATS, "actual_requests", "gnn_pairs_submitted")},
    })


if __name__ == "__main__":
    main()
