"""Run the LLM-UCB workflow from generation zero."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import pandas as pd
from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNTIME = ROOT / "work" / "lohc_operator_ac_runtime_01"
SEED_DATA = HERE / "seed_data"
sys.path[:0] = [str(ROOT / "outputs" / "LOHC_Agents_v2"), str(ROOT / "work")]

import io_helpers as io
from m3os_search import SLOTS
from m3os_sp import MatrixSPMoleculeSearch
from matrix_code_reward import MatrixBook
from runtime_support import Catalog, PROPERTIES, SOURCE_FILES, WINDOW_FIELDS, pair_trace, parent_archive


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(io.json_text(data) + "\n", encoding="utf-8")
    for retry in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if retry == 5:
                raise
            time.sleep(0.04)


def clean_frame(frame, destination):
    temporary = destination.with_name(destination.name + ".tmp")
    frame.drop(columns=["mol"], errors="ignore").to_csv(temporary, index=False, compression="gzip")
    temporary.replace(destination)


class Progress:
    def __init__(self, output, generations):
        self.output = output
        self.stop = output / "STOP"
        self.state = {"arm": "llm", "pid": os.getpid(), "completed_generation": 0, "current_generation": 0,
                      "max_generations": generations, "gnn_pairs": 0, "new_unique_hits": 0,
                      "occupied_chemical_cells": 0, "hit_chemical_cells": 0, "llm_calls": 0, "llm_tokens": 0}

    def publish(self, phase, **fields):
        self.state.update(fields, phase=phase, stop_requested=self.stop.exists(),
                          updated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        atomic_json(self.output / "status.json", self.state)


def run(args, status):
    runtime = args.runtime.resolve()
    seed = 11
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise ValueError("Set PYTHONHASHSEED=11 before launch")
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    import torch
    import utils as ga
    import predict
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(args.threads)
    predict.PREDICT_CFG["device"] = "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = io.read_json(SEED_DATA / "ga_config.json")
    cfg.update(WINDOW_FIELDS, lohc_dh_low=40.0, lohc_dh_high=70.0, pop_size=args.population,
               num_parents=args.population, max_pending_gnn_per_generation=args.gnn_pending, generations=args.generations,
               random_seed=seed, runtime_postprocess=False, runtime_stats_export=False,
               adaptive_output_dir=str(status.output), compute_full_hydrogenation=False)
    cfg["survivor_screening"] = {**cfg["survivor_screening"], **WINDOW_FIELDS}
    cfg["survivor_screening"].update(sa_max=4.0, sa_target_max=4.0,
                                     pareto_objectives=["obj_h2_score", "obj_dh_score", "obj_scaffold_novelty"])
    cfg.update(pareto_objectives=cfg["survivor_screening"]["pareto_objectives"], sa_max_generation=4.0)
    matrix_path = SEED_DATA / "matrix_book.json"
    atomic_json(status.output / "configuration.json", {
        "cfg": cfg, "arm": "llm", "from_generation_zero": True,
        "matrix_book_sha256": io.sha256(matrix_path),
        "m3os_graph": True,
        "m3os_full": True,
        "starting_library_sha256": io.sha256(runtime / "random_start_pop.csv"),
        "native_sources": {name: io.sha256(runtime / name) for name in SOURCE_FILES},
        "weights": {path.name: io.sha256(path) for path in sorted((runtime / "model_checkpoints").glob("fold*_best.pth"))},
    })
    catalog = Catalog()
    matrix_book = MatrixBook(matrix_path)
    molecule_search = MatrixSPMoleculeSearch(status.output, matrix_book, args.key_file, atomic_json)
    molecule_search.enable_online(args.generations)
    seen, hit_cells = set(), set()
    total_pairs = 0
    generation = 0
    pair_inputs = []
    latest = []
    total_hits = 0
    generation_rows = []
    initial_function, generate, rank, select = (
        ga.generate_random_start_pop, ga.generate_candidate_pool, ga.constrain_and_rank_candidates, ga.select_next_population,
    )
    predictor = ga.predict_delta_H
    tracker_class = ga.ExplorationTracker
    original_append = ga.append_csv
    prefilter = ga.apply_gnn_budget_prefilter
    default_rng = np.random.default_rng
    stream = default_rng(seed)
    def seeded_rng(value=None):
        return default_rng(int(stream.integers(0, 2**32)) if value is None else value)

    def observe_prediction(sat, arom):
        nonlocal total_pairs
        pair_inputs.extend(zip(arom, sat))
        total_pairs += len(arom)
        status.publish("GNN 评价", gnn_pairs=total_pairs)
        return predictor(sat, arom)

    def init(*pos, **kwargs):
        status.publish("初始化第 0 代")
        original_population = initial_function(*pos, **kwargs)
        frame = ga.annotate_chemical_behavior(ga.annotate_qd_cells(original_population), cfg)
        molecule_search.ingest(frame, 0, ga)
        atomic_json(status.output / "matrix_scores_0000.json", matrix_book.annotate(parent_archive(frame)))
        if "fitness_evaluated" not in frame:
            frame["fitness_evaluated"] = 1
        catalog.add(frame)
        seen.update(catalog.valid)
        ordered = frame["smi"].astype(str).tolist()
        clean_frame(frame, status.output / "initial_population.csv.gz")
        digest = hashlib.sha256(io.json_text(ordered).encode()).hexdigest()
        status.publish("第 0 代已评价", gnn_pairs=total_pairs, initial_population_sha256=digest,
                       initial_valid_molecules=len(catalog.valid), occupied_chemical_cells=len(catalog.totals))
        return original_population

    def candidates(*pos, **kwargs):
        nonlocal generation
        generation = int(kwargs["generation"]) + 1
        pair_inputs.clear()
        status.publish("生成候选", current_generation=generation)
        np.random.default_rng = seeded_rng
        try:
            result = generate(*pos, **kwargs)
        finally:
            np.random.default_rng = default_rng
        if not status.stop.exists():
            pool, elite, rate = result
            pool = molecule_search.proposals(generation, pool, ga, cfg)
            return pool, elite, rate
        return result

    def reserve_graph_evaluations(cached, pending, configuration):
        if 'm3os_round' not in pending or not pending['m3os_round'].eq(generation).any():
            return prefilter(cached, pending, configuration)
        in_round = pending['m3os_round'].eq(generation)
        graph, _ = prefilter(cached, pending.loc[in_round], {**configuration, 'max_pending_gnn_per_generation': SLOTS})
        limit = int(configuration['max_pending_gnn_per_generation']) - len(graph)
        ordinary, _ = prefilter(cached, pending.loc[~in_round], {**configuration, 'max_pending_gnn_per_generation': limit})
        kept = pd.concat([ordinary, graph], sort=False)
        return kept, len(pending) - len(kept)

    def evaluated(*pos, **kwargs):
        nonlocal latest, total_hits
        status.publish("筛选与 GNN 评价", current_generation=generation)
        result = rank(*pos, **kwargs)
        annotated = ga.annotate_chemical_behavior(result, cfg)
        molecule_search.finish(generation, annotated, pair_inputs, ga)
        latest = pair_trace(pair_inputs, annotated, seen, generation, total_pairs - len(pair_inputs))
        evaluation = []
        for item in latest:
            pair = (item["smi"], item["rich_proxy_smi"])
            grade = matrix_book.evaluate(pair) if item["qd_quality"] is not None else None
            evaluation.append({**item, "matrix_score": grade["score"] if grade else None})
        atomic_json(status.output / f"graph_evaluations_{generation:04d}.json", evaluation)
        hits = [row for row in latest if row["new_native_hit"]]
        total_hits += len(hits)
        hit_cells.update(row["chemical_cell"] for row in hits if row["chemical_cell"] is not None)
        return result

    def survivors(*pos, **kwargs):
        ranked = kwargs.get("df_ranked")
        if ranked is not None:
            catalog.add(ga.annotate_chemical_behavior(ranked, cfg))
        result = select(*pos, **kwargs)
        if generation:
            frame = result[0] if isinstance(result, tuple) else result
            frame = ga.annotate_qd_cells(frame)
            molecule_search.observe_survivors(generation, frame)
            result = (frame, *result[1:]) if isinstance(result, tuple) else frame
            parents = parent_archive(frame)
            atomic_json(status.output / f"parents_{generation:04d}.json", parents)
            atomic_json(status.output / f"matrix_scores_{generation:04d}.json", matrix_book.annotate(parents))
            clean_frame(frame, status.output / "latest_population.csv.gz")
        return result

    def append(frame, path):
        original_append(frame, path)
        if Path(path).name != "stage_timings.csv":
            return
        parents = io.read_json(status.output / f"parents_{generation:04d}.json")
        rows = io.read_json(status.output / f"matrix_scores_{generation:04d}.json")
        scores = [row["matrix_score"]["score"] for row in rows if row["matrix_score"] is not None]
        metric = {
            "generation": generation, "gnn_pairs": total_pairs, "new_unique_hits": total_hits,
            "occupied_chemical_cells": len(catalog.totals), "hit_chemical_cells": len(hit_cells),
            "new_hits_this_generation": sum(row["new_native_hit"] for row in latest),
            "qualified_parents": len(parents), "parent_population": args.population,
            "parent_cells": len({row["cell"] for row in parents}),
            "matrix_scored": len(scores), "matrix_mean": float(np.mean(scores)) if scores else None,
            "matrix_top": max(scores, default=None),
            **molecule_search.stats,
            "llm_calls": molecule_search.total_calls, "llm_tokens": molecule_search.total_tokens,
        }
        generation_rows.append(metric)
        atomic_json(status.output / "progress_history.json", generation_rows)
        status.publish("本代完成", completed_generation=generation,
                       **{key: value for key, value in metric.items() if key != "generation"})
        print("progress", io.json_text(metric), flush=True)

    class StoppableTracker(tracker_class):
        def is_saturated(self):
            return status.stop.exists() or super().is_saturated()

    def export(**kwargs):
        status.publish("保存最终结果")
        molecule_search.export()
        for key, filename in (("df_pop", "last_pop.csv.gz"), ("df_fittest", "fittest.csv.gz"),
                              ("df_archive", "archive.csv.gz")):
            clean_frame(kwargs[key], status.output / filename)

    ga.predict_delta_H = observe_prediction
    ga.generate_random_start_pop = init
    ga.generate_candidate_pool = candidates
    ga.constrain_and_rank_candidates = evaluated
    ga.select_next_population = survivors
    ga.ExplorationTracker = StoppableTracker
    ga.append_csv = append
    ga.apply_gnn_budget_prefilter = reserve_graph_evaluations
    ga.post_process = export
    if status.stop.exists():
        status.publish("已停止")
        return
    ga.run_simplified_workflow(cfg)
    status.publish("已停止" if status.stop.exists() else "已完成")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--key-file", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=1000)
    parser.add_argument("--gnn-pending", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.generations < 1 or args.population < 1 or args.threads < 1 or args.gnn_pending <= SLOTS:
        parser.error(f"Generation, population and thread counts must be positive, and GNN budget must exceed {SLOTS}.")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = Progress(output, args.generations)
    status.publish("进程已启动")
    try:
        run(args, status)
    except BaseException as error:
        traceback.print_exc()
        status.publish("运行失败", error=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
