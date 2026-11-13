"""Run one zero-start arm of the native 40–70 fixed-grid GA."""

import argparse
from contextlib import nullcontext
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
OUT = HERE.parent
V8 = OUT / "LOHC_Window40_70_Ablation_v8"
V9 = OUT / "LOHC_Event_Triggered_UCB_v9"
sys.path[:0] = [str(V8), str(V9)]
import run_ablation as v8
from monitor import Monitor

v7, v5, demo = v8.v7, v8.v5, v8.demo
RUNTIME = HERE.parents[1] / "work" / "lohc_v8_deterministic_runtime"


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(demo.json_text(data) + "\n", encoding="utf-8")
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
    def __init__(self, output, arm, generations):
        self.output = output
        self.stop = output / "STOP"
        self.state = {"arm": arm, "pid": os.getpid(), "completed_generation": 0, "current_generation": 0,
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
    key = args.key_file.resolve() if args.key_file else None
    rule_path = args.rule_book.resolve() if args.rule_book else None
    matrix_path = args.matrix_book.resolve() if args.matrix_book else None
    percentile_path = args.percentile_reference.resolve() if args.percentile_reference else None
    sys.path.insert(0, str(runtime))
    os.chdir(runtime)
    import torch
    import utils as ga
    import adaptive_ga as native
    import main as original
    import predict
    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(args.threads)
    predict.PREDICT_CFG["device"] = "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = v7.history_config(v7.HISTORY, original.CONFIG)
    cfg.update(v8.WINDOW_FIELDS, lohc_dh_low=40.0, lohc_dh_high=70.0, pop_size=args.population,
               num_parents=args.population, max_pending_gnn_per_generation=args.gnn_pending, generations=args.generations,
               random_seed=seed, runtime_postprocess=False, runtime_stats_export=False,
               adaptive_output_dir=str(status.output), compute_full_hydrogenation=False)
    cfg["survivor_screening"] = {**cfg["survivor_screening"], **v8.WINDOW_FIELDS}
    if args.qd_ac:
        cfg["survivor_screening"].update(sa_max=4.0, sa_target_max=4.0,
                                         pareto_objectives=["obj_h2_score", "obj_dh_score", "obj_scaffold_novelty"])
        cfg.update(pareto_objectives=cfg["survivor_screening"]["pareto_objectives"],
                   sa_max_generation=4.0)
    atomic_json(status.output / "configuration.json", {
        "cfg": cfg, "arm": args.arm, "from_generation_zero": True,
        "matrix_book_sha256": demo.sha256(matrix_path) if matrix_path else None,
        "percentile_reference_sha256": demo.sha256(percentile_path) if percentile_path else None,
        "m3os_graph": args.graph_m3os,
        "m3os_full": args.full_m3os,
        "starting_library_sha256": demo.sha256(runtime / "random_start_pop.csv"),
        "native_sources": {name: demo.sha256(runtime / name) for name in (*v5.SOURCE_FILES, "enthalpy_window.py")},
        "weights": {path.name: demo.sha256(path) for path in sorted((runtime / "model_checkpoints").glob("fold*_best.pth"))},
    })
    catalog = v7.Catalog()
    feedback = None
    paired_feedback = None
    qd_feedback = None
    steering = None
    rule_book = None
    matrix_book = None
    matrix_reward = None
    matrix_updates = {}
    percentile_run = None
    molecule_search = None
    if matrix_path is not None:
        from matrix_code_reward import MatrixBook, MatrixReward
        matrix_book = MatrixBook(matrix_path)
        if args.full_m3os and args.arm == "llm":
            if args.matrix_sp:
                from m3os_sp import MatrixSPMoleculeSearch as Search
            else:
                from m3os_full import FullMoleculeSearch as Search
            molecule_search = Search(status.output, matrix_book, key, atomic_json)
            if args.online_memory:
                molecule_search.enable_online(args.generations)
        elif args.graph_m3os and not args.full_m3os:
            from m3os_search import MoleculeSearch
            molecule_search = MoleculeSearch(status.output, matrix_book, key if args.arm == "llm" else None, atomic_json)
        if args.arm == "code":
            matrix_reward = MatrixReward(matrix_book, native.MUTATION_OPERATORS)
        if percentile_path is not None:
            from percentile_selection import EvidenceReview, PercentileSelection
            reviewer = EvidenceReview(key, status.output / "scoring_agent") if args.arm == "llm" else None
            percentile_run = PercentileSelection(matrix_book, percentile_path, status.output, reviewer)
    if rule_path is not None:
        from rule_steering import RuleBook
        rule_book = RuleBook(rule_path)
    steering_book = matrix_book if matrix_book is not None else rule_book
    if args.qd_ac:
        import ac_qd
        if args.arm == "llm" and steering_book is not None and not args.graph_m3os:
            from rule_steering import RuleSteering
            steering = RuleSteering(key, status.output / "operator_llm", args.operator_requests,
                                    native.MUTATION_OPERATORS, int(cfg.get("operator_warmup_generations", 5)),
                                    steering_book, tail=percentile_run)
        elif args.arm == "llm" and not args.graph_m3os:
            qd_feedback = ac_qd.QDFeedback(ac_qd.Review(key, status.output / "qd_llm",
                                                        ac_qd.evidence_records(), args.qd_requests), status.output)
            if args.operator_llm:
                from operator_guidance import Steering
                steering = Steering(key, status.output / "operator_llm", args.operator_requests,
                                    native.MUTATION_OPERATORS, int(cfg.get("operator_warmup_generations", 5)))
    spatial = args.spatial
    if spatial:
        from spatial_feedback import Feedback, Reviews, candidate
    if args.paired:
        from paired_reward import PairedFeedback, PairedReviews
    seen, hit_cells = set(), set()
    monitor = None
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
    policy_factory = ga.OperatorExperiencePolicy
    tracker_class = ga.ExplorationTracker
    original_append = ga.append_csv
    prefilter = ga.apply_gnn_budget_prefilter
    default_rng = np.random.default_rng
    stream = default_rng(seed)
    review_path = status.output / "reviews"
    review_path.mkdir()
    memory = demo.read_json(HERE / "seed_data" / "qd_evidence.json")["evidence"]
    protocol = demo.read_json(V9 / "protocol.json")
    protocol["maximum_requests_per_trial"] = args.generations

    def seeded_rng(value=None):
        return default_rng(int(stream.integers(0, 2**32)) if value is None else value)

    def observe_prediction(sat, arom):
        nonlocal total_pairs
        pair_inputs.extend(zip(arom, sat))
        total_pairs += len(arom)
        status.publish("GNN 评价", gnn_pairs=total_pairs)
        return predictor(sat, arom)

    def init(*pos, **kwargs):
        nonlocal monitor, feedback, paired_feedback
        status.publish("初始化第 0 代")
        original_population = initial_function(*pos, **kwargs)
        frame = ga.annotate_chemical_behavior(ga.annotate_qd_cells(original_population), cfg)
        if molecule_search is not None:
            molecule_search.ingest(frame, 0, ga)
        if matrix_reward is not None:
            matrix_reward.set_parents(frame)
        if matrix_book is not None:
            atomic_json(status.output / "matrix_scores_0000.json", matrix_book.annotate(ac_qd.parent_archive(frame)))
        if steering is not None:
            steering.parent_frame(frame)
        if "fitness_evaluated" not in frame:
            frame["fitness_evaluated"] = 1
        catalog.add(frame)
        seen.update(catalog.valid)
        ordered = frame["smi"].astype(str).tolist()
        clean_frame(frame, status.output / "initial_population.csv.gz")
        digest = hashlib.sha256(demo.json_text(ordered).encode()).hexdigest()
        if not args.qd_ac and not args.paired and (args.arm == "llm" or spatial):
            monitor = Monitor(v5, native, protocol, [], frame, catalog, memory,
                              v5.agents.DeepSeek(v5.agents.credential(key), review_path), review_path)
        if args.paired and args.arm == "llm":
            evidence = demo.read_json(OUT / "LOHC_Grid_Guided_GA_v4" / "live_run" / "frozen_memory.json")
            paired_feedback = PairedFeedback(PairedReviews(key, status.output / "paired_guidance", evidence,
                                                           args.paired_requests))
            paired_feedback.set_parents(frame)
        elif spatial and args.arm == "llm":
            evidence = demo.read_json(OUT / "LOHC_Grid_Guided_GA_v4" / "live_run" / "frozen_memory.json")
            feedback = Feedback(Reviews(key, status.output / "bayesian_guidance", evidence, args.bayes_requests))
            feedback.observe([candidate(r) for r in frame.to_dict("records") if r["qd_target_hit"] == 1], ())
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
            with (monitor.capture_attempts(ga, catalog) if monitor else nullcontext()):
                result = generate(*pos, **kwargs)
        finally:
            np.random.default_rng = default_rng
        if molecule_search is not None and not status.stop.exists():
            pool, elite, rate = result
            pool = molecule_search.proposals(generation, pool, ga, cfg)
            return pool, elite, rate
        return result

    def reserve_graph_evaluations(cached, pending, configuration):
        if 'm3os_round' not in pending or not pending['m3os_round'].eq(generation).any():
            return prefilter(cached, pending, configuration)
        from m3os_search import SLOTS
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
        if matrix_reward is not None:
            matrix_reward.set_children(result)
        if steering_book is not None and steering is not None:
            steering.observe_children(result)
        annotated = ga.annotate_chemical_behavior(result, cfg)
        if molecule_search is not None:
            molecule_search.finish(generation, annotated, pair_inputs, ga)
        if matrix_book is not None and args.arm != "code" and {"smi", "sat_smi"}.issubset(annotated):
            pairs = annotated[["smi", "sat_smi"]].dropna().drop_duplicates().itertuples(index=False, name=None)
            scored = [{"pair": pair, "matrix_score": matrix_book.evaluate(pair)} for pair in pairs]
            atomic_json(status.output / f"matrix_candidates_{generation:04d}.json", scored)
        latest = v8.pair_trace(pair_inputs, annotated, seen, generation, total_pairs - len(pair_inputs))
        if args.graph_m3os:
            evaluation = []
            for item in latest:
                pair = (item['smi'], item['rich_proxy_smi'])
                grade = matrix_book.evaluate(pair) if item['qd_quality'] is not None else None
                evaluation.append({**item, 'matrix_score': grade['score'] if grade else None})
            atomic_json(status.output / f'graph_evaluations_{generation:04d}.json', evaluation)
        hits = [row for row in latest if row["new_native_hit"]]
        total_hits += len(hits)
        hit_cells.update(row["chemical_cell"] for row in hits if row["chemical_cell"] is not None)
        rows = []
        if spatial:
            hit_smiles = {row["smi"] for row in hits}
            archived = [candidate(r) for r in annotated.drop_duplicates("smi").to_dict("records")
                        if r["smi"] in hit_smiles]
            atomic_json(status.output / f"spatial_candidates_{generation:04d}.json", archived)
            if feedback is not None:
                feedback.observe(archived, hit_smiles)
        if paired_feedback is not None:
            novel = {row["smi"] for row in latest if row["new_valid_evaluation"]}
            paired_feedback.set_children(annotated, novel)
        if hits:
            matched = annotated[annotated["smi"].isin({row["smi"] for row in hits})]
            for cell, group in matched.groupby("chemical_behavior_cell", sort=True):
                ordered = group.sort_values(["qd_quality", "smi"], ascending=[False, True])
                if "scaffold_smi" in ordered:
                    ordered = ordered.drop_duplicates("scaffold_smi")
                for item in ordered.head(2).to_dict("records"):
                    rows.append({"chemical_cell": int(cell), "smi": item["smi"], "rich_proxy_smi": item["sat_smi"],
                                 "platform": item["platform_id"], "quality": float(item["qd_quality"]),
                                 "properties": {name: float(item[name]) for name in demo.PROPERTIES}})
        atomic_json(status.output / f"candidates_{generation:04d}.json", rows)
        return result

    def policy(*pos, **kwargs):
        configured = policy_factory(*pos, **kwargs)
        update = configured.update

        def coupled(transitions):
            if matrix_reward is not None:
                adjusted, scored_edits, eligible = matrix_reward.apply(transitions)
                result = update(adjusted)
                for row in scored_edits:
                    stats = configured.stats[row["context"]][row["operator"]]
                    row.update(ucb_count_after=int(stats["count"]), ucb_reward_sum_after=float(stats["reward_sum"]))
                matrix_updates.update(matrix_eligible=eligible, matrix_paired=len(scored_edits),
                                      matrix_updates=sum(row["before"] != row["after"] for row in scored_edits))
                atomic_json(status.output / f"matrix_updates_{generation:04d}.json",
                            {**matrix_updates, "transitions": scored_edits})
                return result
            if steering is not None:
                steering.remember(transitions)
            if monitor is not None:
                transitions, _ = monitor.adjust(transitions, catalog)
            details = []
            if feedback is not None:
                transitions, details = feedback.adjust(transitions, generation, native.MUTATION_OPERATORS)
            elif paired_feedback is not None and not status.stop.exists():
                transitions, details = paired_feedback.adjust(transitions, generation, native.MUTATION_OPERATORS)
            result = update(transitions)
            if details:
                for row in details:
                    entry = configured.stats[row["context"]][row["operator"]]
                    row.update(ucb_count_after=int(entry["count"]), ucb_reward_sum_after=float(entry["reward_sum"]))
                prefix = "paired_updates" if paired_feedback is not None else "bayesian_updates"
                atomic_json(status.output / f"{prefix}_{generation:04d}.json", details)
            return result

        configured.update = coupled
        if steering is not None:
            steering.attach(configured)
        return configured

    def survivors(*pos, **kwargs):
        ranked = kwargs.get("df_ranked")
        if ranked is not None:
            catalog.add(ga.annotate_chemical_behavior(ranked, cfg))
        result = select(*pos, **kwargs)
        if generation:
            frame = result[0] if isinstance(result, tuple) else result
            if args.qd_ac:
                frame = ga.annotate_qd_cells(frame)
                if args.full_m3os and molecule_search is not None:
                    molecule_search.observe_survivors(generation, frame)
                elite_replacements = 0
                if percentile_run is not None and args.arm == "llm" and ranked is not None and not status.stop.exists():
                    status.publish("四目标 Pareto 留种", current_generation=generation)
                    frame, elite_replacements = percentile_run.select(generation, ranked, frame, ga)
                if qd_feedback is not None and ranked is not None and not status.stop.exists():
                    ranked = ga.annotate_qd_cells(ranked)
                    status.publish("QD 格内精评与留种", current_generation=generation)
                    frame = qd_feedback.update(generation, ranked, frame, ga)
                    if steering is not None:
                        status.publish("LLM 审查算子优先级", current_generation=generation)
                        steering.refresh(generation, frame, args.generations, qd_feedback.reviewer)
                if steering_book is not None and steering is not None and not status.stop.exists():
                    status.publish("DeepSeek 评估算子收益", current_generation=generation)
                    steering.refresh(generation, frame, args.generations)
                result = (frame, *result[1:]) if isinstance(result, tuple) else frame
                parents = ac_qd.parent_archive(frame)
                if percentile_run is not None:
                    percentile_run.observe(generation, latest, parents, elite_replacements)
                atomic_json(status.output / f"parents_{generation:04d}.json", parents)
                if matrix_reward is not None:
                    matrix_reward.set_parents(frame)
                if matrix_book is not None:
                    scores = matrix_book.annotate(parents) if percentile_run is None else [
                        {**row, 'matrix_score': percentile_run.grade(tuple(row['pair']))} for row in parents]
                    atomic_json(status.output / f"matrix_scores_{generation:04d}.json", scores)
                if rule_book is not None:
                    atomic_json(status.output / f"rule_scores_{generation:04d}.json", rule_book.annotate(parents))
            clean_frame(frame, status.output / "latest_population.csv.gz")
            if paired_feedback is not None:
                paired_feedback.set_parents(ga.annotate_chemical_behavior(frame, cfg))
            if monitor is not None and len(catalog.valid) >= 500 and len(catalog.totals) >= 20 and not status.stop.exists():
                status.publish("检查 LLM 审查门槛", current_generation=generation)
                monitor.review(generation, generation < args.generations, frame, latest, catalog)
                atomic_json(status.output / "event_state.json", {"gates": monitor.gates, "reviews": monitor.reviews,
                                                                "current_rules": monitor.rules})
                if feedback is not None and generation < args.generations:
                    feedback.refresh(generation)
        return result

    def append(frame, path):
        original_append(frame, path)
        if Path(path).name != "stage_timings.csv":
            return
        tokens = sum((demo.read_json(file).get("usage") or {}).get("total_tokens", 0)
                     for file in review_path.glob("*_audit.json"))
        metric = {
            "generation": generation, "gnn_pairs": total_pairs, "new_unique_hits": total_hits,
            "occupied_chemical_cells": len(catalog.totals), "hit_chemical_cells": len(hit_cells),
            "new_hits_this_generation": sum(row["new_native_hit"] for row in latest),
            "llm_calls": monitor.requests if monitor else 0, "llm_tokens": tokens,
        }
        if feedback is not None:
            from spatial_feedback import usage
            metric.update(bayesian_requests=feedback.reviews.calls, bayesian_updates=feedback.changed,
                          bayesian_tokens=usage(feedback.reviews.output).get("total_tokens", 0))
        if paired_feedback is not None:
            metric.update(llm_calls=paired_feedback.reviews.calls, llm_tokens=paired_feedback.reviews.tokens,
                          paired_sampled=paired_feedback.sampled, paired_updates=paired_feedback.changed,
                          paired_usable=paired_feedback.usable, paired_fallbacks=paired_feedback.fallbacks)
        if args.qd_ac:
            parents = demo.read_json(status.output / f"parents_{generation:04d}.json")
            metric.update(qualified_parents=len(parents), parent_population=args.population,
                          parent_cells=len({r["cell"] for r in parents}))
        if rule_book is not None:
            scores = [r["score"] for r in demo.read_json(status.output / f"rule_scores_{generation:04d}.json")
                      if r["score"] is not None]
            metric.update(rule_scored=len(scores), rule_mean=float(np.mean(scores)) if scores else None,
                          rule_at_least_50=sum(score >= 50 for score in scores))
        if matrix_book is not None:
            rows = demo.read_json(status.output / f"matrix_scores_{generation:04d}.json")
            scores = [r['matrix_score']['score'] for r in rows if r['matrix_score'] is not None]
            metric.update(matrix_scored=len(scores), matrix_mean=float(np.mean(scores)) if scores else None,
                          matrix_top=max(scores, default=None), **matrix_updates)
            candidate_path = status.output / f"matrix_candidates_{generation:04d}.json"
            if candidate_path.exists():
                scored = demo.read_json(candidate_path)
                metric.update(matrix_candidate_pairs=len(scored),
                              matrix_candidates_scored=sum(row["matrix_score"] is not None for row in scored))
        if qd_feedback is not None:
            metric.update(llm_calls=qd_feedback.reviewer.calls,
                          llm_tokens=ac_qd.usage(qd_feedback.reviewer.output).get("total_tokens", 0),
                          qd_replacements=qd_feedback.replacements, qd_subdivided_cells=len(qd_feedback.plans),
                          qd_reviewed=len(qd_feedback.reviewer.grades))
        if steering is not None:
            events = steering.write_generation(generation)
            metric.update(operator_llm_calls=steering.calls, operator_steered=steering.counters["steered_decisions"],
                          operator_plans=sum(p["operator"] is not None for p in steering.history))
            metric["llm_calls"] += steering.calls
            metric["llm_tokens"] += events["usage"].get("total_tokens", 0)
            if steering_book is not None:
                metric.update({f"rule_{k}": v for k, v in events['transition_summary'].items()})
        if percentile_run is not None:
            metric.update(percentile_run.stats)
            metric['llm_calls'] += percentile_run.stats.get('scoring_agent_calls', 0)
            metric['llm_tokens'] += percentile_run.stats.get('scoring_agent_tokens', 0)
        if molecule_search is not None:
            metric.update(molecule_search.stats, llm_calls=molecule_search.total_calls, llm_tokens=molecule_search.total_tokens)
        generation_rows.append(metric)
        atomic_json(status.output / "progress_history.json", generation_rows)
        status.publish("本代完成", completed_generation=generation,
                       **{key: value for key, value in metric.items() if key != "generation"})
        print("progress", demo.json_text(metric), flush=True)

    class StoppableTracker(tracker_class):
        def is_saturated(self):
            return status.stop.exists() or super().is_saturated()

    def export(**kwargs):
        status.publish("保存最终结果")
        if molecule_search is not None:
            molecule_search.export()
        for key, filename in (("df_pop", "last_pop.csv.gz"), ("df_fittest", "fittest.csv.gz"),
                              ("df_archive", "archive.csv.gz")):
            clean_frame(kwargs[key], status.output / filename)

    ga.predict_delta_H = observe_prediction
    ga.generate_random_start_pop = init
    ga.generate_candidate_pool = candidates
    ga.constrain_and_rank_candidates = evaluated
    ga.OperatorExperiencePolicy = policy
    ga.select_next_population = survivors
    ga.ExplorationTracker = StoppableTracker
    ga.append_csv = append
    if molecule_search is not None:
        ga.apply_gnn_budget_prefilter = reserve_graph_evaluations
    ga.post_process = export
    if status.stop.exists():
        status.publish("已停止")
        return
    ga.run_simplified_workflow(cfg)
    status.publish("已停止" if status.stop.exists() else "已完成")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("native", "llm", "code"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--population", type=int, default=1000)
    parser.add_argument("--gnn-pending", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--spatial", action="store_true", help="Compare current LLM-UCB with Bayesian feedback")
    parser.add_argument("--bayes-requests", type=int, default=12)
    parser.add_argument("--paired", action="store_true", help="Compare native UCB with direct paired LLM reward")
    parser.add_argument("--paired-requests", type=int, default=32)
    parser.add_argument("--qd-ac", action="store_true", help="Run the shared A/C QD protocol")
    parser.add_argument("--qd-requests", type=int, default=32)
    parser.add_argument("--operator-llm", action="store_true")
    parser.add_argument("--operator-requests", type=int, default=8)
    parser.add_argument("--rule-book", type=Path, help="Frozen graph rules and cluster centers for the A/C trial")
    parser.add_argument("--matrix-book", type=Path, help="Frozen local matrix scoring for the native/code trial")
    parser.add_argument("--percentile-reference", type=Path, help="Use four-objective cell selection and a frozen score reference")
    parser.add_argument("--graph-m3os", action="store_true", help="Reserve part of each generation for M3OS molecular graph expansion")
    parser.add_argument("--full-m3os", action="store_true", help="Compare native GA with complete M3OS orchestration")
    parser.add_argument("--matrix-sp", action="store_true", help="Use validated matrix SP scores for C graph reward")
    parser.add_argument("--online-memory", action="store_true", help="Use a live evidence memory and intervene through the final generation")
    args = parser.parse_args()
    args.graph_m3os = args.graph_m3os or args.full_m3os
    if args.matrix_sp and not args.full_m3os:
        parser.error('--matrix-sp requires --full-m3os')
    if args.online_memory and not args.matrix_sp:
        parser.error('--online-memory requires --matrix-sp')
    if (args.spatial and not args.paired and not args.qd_ac or
            (args.paired or args.qd_ac) and args.arm == "llm") and not args.key_file:
        parser.error("This mode requires --key-file")
    if args.qd_ac and (args.paired or args.spatial or args.runtime.resolve() == RUNTIME.resolve()):
        parser.error("--qd-ac requires its own staged runtime and cannot be combined with legacy LLM modes")
    if args.operator_llm and not args.qd_ac:
        parser.error("--operator-llm requires --qd-ac")
    if args.rule_book and (not args.qd_ac or args.operator_llm):
        parser.error("--rule-book requires --qd-ac and selects its own operator steering")
    if args.arm == "code" and not args.matrix_book:
        parser.error("The code arm requires --matrix-book")
    if args.matrix_book and (not args.qd_ac or args.rule_book or args.operator_llm):
        parser.error("--matrix-book requires the fixed-grid protocol and selects its own scoring book")
    if args.percentile_reference and (not args.matrix_book or args.arm == "code"):
        parser.error("--percentile-reference requires --matrix-book and a native or LLM arm")
    if args.graph_m3os and (not args.matrix_book or args.percentile_reference or args.arm == 'code' or args.gnn_pending <= 32):
        parser.error('--graph-m3os requires --matrix-book and a GNN budget above 32, with native or LLM arm')
    args.spatial = args.spatial or args.paired
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=not (args.spatial or args.qd_ac))
    status = Progress(output, args.arm, args.generations)
    status.publish("进程已启动")
    try:
        run(args, status)
    except BaseException as error:
        traceback.print_exc()
        status.publish("运行失败", error=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
