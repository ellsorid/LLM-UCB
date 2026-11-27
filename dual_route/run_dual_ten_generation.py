"""Ten-generation dual-route experiment with reproducible bootstrap."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import threading

import numpy as np
import torch
from furan_resume_evidence import scoped_furan_analogue


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def _compute_sa_penalty(sa_val, threshold=3.5, weight=2.0, cap=12.0):
    """Smooth hinge penalty above threshold SA score to discourage over-branched poly-alkynes."""
    if sa_val is None or float(weight) <= 0.0:
        return 0.0
    excess = max(0.0, float(sa_val) - float(threshold))
    return float(min(float(cap), float(weight) * excess))


def run(
    archive,
    out,
    key,
    retry_budget=10,
    resume=None,
    inspect_only=False,
    inputs=None,
    max_generations=10,
    strict_local_gate=False,
    gateway_override=None,
    evidence_finder=None,
    initial_pool_override=None,
    enable_evidence_search=False,
    search_budget=24,
    search_cache=None,
    search_gap_threshold=0.75,
    bootstrap_size=8,
    batch_size=8,
    max_parents=8,
    elite_quota=5,
    native_ga_budget=16,
    shortlist_size=32,
    api_call_limit=100,
    sa_penalty_weight=2.0,
    sa_threshold=3.5,
    protect_directed_parent=True,
    root_cooldown=4.0,
    enable_chemical_rules=False,
):
    if inspect_only and (key or gateway_override):
        raise ValueError("Offline inspection does not accept API credentials")
    if max_generations < 1:
        raise ValueError("max_generations must be at least 1")
    archive = archive.resolve()
    inputs = inputs.resolve() if inputs else archive
    out = out.resolve()
    key = key.resolve() if key else None
    resume = resume.resolve() if resume else None
    runtime = archive/"workflow_runtime_20261125/runtime"
    input_runtime = inputs/"workflow_runtime_20261125/runtime"
    som = archive/"SOM/chemical_unit_som_v1"
    registry = read(archive/"CHEMICAL_UNIT_WORKFLOW.json")
    model_path = archive/registry["som"]["checkpoint"]
    memory_path = inputs/registry["memory"]["fixture"]
    global_memory_path = inputs/"runs/frontier_small_001/memory_store/memory/global.json"
    assignments_path = inputs/"SOM/chemical_unit_som_v1/artifacts/assignments.npz"
    matrices_path = input_runtime/"outputs/matrix_som_v2/matrix_inputs.npz"
    immigrant_path = input_runtime/"outputs/matrix_map_v1/data_split.csv"
    if initial_pool_override is None:
        required = (model_path, memory_path, global_memory_path, assignments_path,
                    matrices_path, immigrant_path)
    else:
        required = (model_path, memory_path, global_memory_path, immigrant_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required workflow files are missing:\n" + "\n".join(missing))
    os.environ.update(LOHC_PROJECT_ROOT=str(runtime),LOHC_BUNDLED_ONLY="1")
    sys.path[:0] = [str(som),str(archive/"LLM/overall_local_v1"),str(archive/"LLM/ucb_chemical_units_v1"),
                    str(archive/"LLM/frontier_loop"),str(runtime)]
    from chemical_som import encode_pairs, encode_token
    from chemical_units import extract
    from chemical_memory_replay import fingerprint
    from evidence_finder import AutonomousEvidenceFinder
    from frontier import family, local_key
    from mapping import KalmanMap, som_laplacian_prior, som_manifold_features
    from matrix_pipeline.core import Molecule, identity, properties, synthesis_score, to_mol
    from matrix_pipeline.temperature_experiment import OfficialTemperatureGateway
    from memory_store import MemoryStore
    from overall_local_probe import TASK, local_result
    from property_feedback import (EditCandidate, EditEvidence, Evaluation, PopulationState,
                                   commit_batch, generation_quotas, select_batch, select_next_parents)
    from rdkit import Chem
    from structure import StructureBridge
    from chemical_rule_surrogate import ChemicalRuleBank, HybridChemicalSurrogate, induce_rules_from_generation
    from dual_route_loop import DESIGN_TASK, validate_design
    from population_ucb import PopulationUCB, score_delta
    from ten_generation_native_ga import generate as native_generate

    class LegalEdits(StructureBridge):
        def __init__(self):
            pass

        def describe(self, molecules):
            return [{"uid":m.uid,"molecule":m} for m in molecules]

    torch.set_num_threads(2)
    model = torch.load(model_path,map_location="cpu",weights_only=False)
    centers = model["centers"].numpy()
    bandwidth = max(float(np.median(((centers[:,None]-centers[None,:])**2).sum(2))),1e-6)
    prior_cov = som_laplacian_prior(model["topology"])
    sa_tag = f"|sa_pen:w{float(sa_penalty_weight):.2f}_t{float(sa_threshold):.2f}" if float(sa_penalty_weight) > 0.0 else ""
    context = (registry["scoring"]["version"]+"|unspecified_conditions|kalman_mode:som_laplacian_v1|SOM:"
               +sha256(model_path.read_bytes()).hexdigest()+sa_tag)
    if (archive/registry["scoring"]["protocol"]).read_text(encoding="utf-8")!=TASK:
        raise ValueError("Scoring protocol mismatch")
    memory = read(memory_path)
    global_memory = read(global_memory_path)["records"]
    memory_store = MemoryStore(memory, global_memory, protocol_hash=sha256(TASK.encode()).hexdigest())
    bridge = LegalEdits()
    unit_summary_by_uid = {}
    search_lock = threading.RLock()
    active_search = evidence_finder is not None or (
        enable_evidence_search and not inspect_only and (key is not None or gateway_override is not None)
    )

    def _safe_sa(pair):
        try:
            return float(synthesis_score(pair))
        except Exception:
            return None

    def encode(m, parent, site, operator):
        z = encode_pairs([m.pair],model["transform"])[0].numpy().astype(float)
        cell, response, psi = som_manifold_features(z, centers, bandwidth)
        units = extract(m.pair,m.uid)
        tags, payload = memory_store.prepare_local(units)
        whole = memory_store.whole_prior(m.pair,m.uid)
        tokens = frozenset(encode_token((u["name"],u["state"],sorted(fingerprint(u).items(),key=repr)))
                           for group in units["units"].values() for u in group)
        kinds = {t["kind"] for t in tags if t["used_for_judgement"] and t["similarity"]>=.5}
        unit_summary = sorted({f"{u['state']}:{u['name']}" for group in units["units"].values() for u in group})
        unit_summary_by_uid[m.uid] = set(unit_summary)
        missing_kinds = sorted(set(units["units"]) - kinds)
        candidate = EditCandidate(m.uid,parent.uid,site,cell,psi,response,
            (family(parent.pair),local_key(parent.pair,site),operator),tokens,
            1-len(kinds)/len(units["units"]),frozenset(t["reference"] for t in tags if t["used_for_judgement"]))
        return {"candidate":candidate,"molecule":m,"tags":tags,"packet":payload,"whole":whole,
                "units":units,"unit_summary":unit_summary,"missing_kinds":missing_kinds}

    def is_clean_neutral_pair(pair):
        return (
            not np.any(np.diagonal(pair[:, :, :, 5], axis1=1, axis2=2) != 0)
            and not np.any(np.diagonal(pair[:, :, :, 1], axis1=1, axis2=2) != 0)
        )

    def ready(r):
        if active_search:
            return True
        if strict_local_gate:
            return bool(r["packet"]) and r["whole"]["score"] is not None
        return r["whole"]["score"] is not None or bool(r["packet"])

    def stratified_pool():
        if initial_pool_override is not None:
            result, seen = [], set()
            for idx, m in enumerate(initial_pool_override):
                if m.uid in seen or not is_clean_neutral_pair(m.pair):
                    continue
                try:
                    row = encode(m, m, 0, "bootstrap")
                except (ValueError, RuntimeError):
                    continue
                seen.add(m.uid)
                row["source_row"] = int(m.source_row if m.source_row is not None else idx)
                result.append(row)
            return result
        rng = np.random.default_rng(42)
        with np.load(assignments_path,allow_pickle=False) as assigned:
            cells, viable = assigned["node"],assigned["eligible"]
        groups = defaultdict(list)
        for i in rng.permutation(len(cells)):
            if viable[i]:
                groups[int(cells[i])].append(i)
        with np.load(matrices_path,allow_pickle=False) as raw:
            matrices,masks = raw["x"],raw["mask"]
        seen, result = set(),[]
        # Cycle through occupied cells to preserve sparse regions.
        keys = rng.permutation(sorted(groups))
        for depth in range(max(map(len,groups.values()))):
            for cell in keys:
                if depth>=len(groups[cell]):
                    continue
                i = groups[cell][depth]
                mask = masks[i]
                pair = matrices[i][:,mask][:,:,mask]
                if not is_clean_neutral_pair(pair):
                    continue
                try:
                    names = tuple(Chem.MolToSmiles(to_mol(m),isomericSmiles=True) for m in pair)
                    uid = identity(names)
                    if uid in seen:
                        continue
                    m = Molecule(uid,pair,names,uid)
                    row = encode(m,m,0,"bootstrap")
                except (ValueError,RuntimeError):
                    continue
                seen.add(uid)
                row["source_row"] = int(i)
                result.append(row)
                if len(result)==256:
                    return result
        raise RuntimeError("Fewer than 256 distinct matrices passed the input checks")

    def bootstrap_queue(pool):
        target_count = bootstrap_size if initial_pool_override is None else min(bootstrap_size, len(pool))
        selected, covered = [], set()
        remaining = [r for r in pool if ready(r)]
        while remaining and len(selected)<target_count:
            def order(r):
                c = r["candidate"]
                references = c.evidence_refs
                both_ready = bool(r["packet"]) and r["whole"]["score"] is not None
                whole_score = float(r["whole"]["score"]) if r["whole"]["score"] is not None else -1.0
                # Prioritize both-ready, distinct SOM cells, then unexplored chemical units.
                return (both_ready, c.cell not in {x["candidate"].cell for x in selected},
                        len(c.units-covered),len(references),whole_score,c.uid)
            row = max(remaining,key=order)
            selected.append(row)
            covered.update(row["candidate"].units)
            remaining = [r for r in remaining if r is not row]
        return selected

    def propose(parents, excluded):
        generated, invalid = {},[]
        for parent in parents:
            for op in ("methyl","amino","hydroxyl","vinyl_ethyl","ethynyl_ethyl"):
                rows, errors = bridge.children({"uid":parent.uid,"molecule":parent},op,max_sites=3)
                invalid.extend(errors)
                for r in rows:
                    uid = r["uid"]
                    if uid in excluded or uid in generated:
                        continue
                    if not is_clean_neutral_pair(r["molecule"].pair):
                        invalid.append({"operator":op,"site":int(r["edit"]["site"]),"error":"Radical or charged species rejected"})
                        continue
                    generated[uid] = encode(r["molecule"],parent,int(r["edit"]["site"]),op)
        return generated,invalid

    pool = stratified_pool()
    ready_pool = [r for r in pool if ready(r)]
    seeds = bootstrap_queue(pool)
    out.mkdir(parents=True)
    legacy_root = out.resolve()/"legacy_ga_runtime"
    shutil.copytree(archive/"LLM/ucb_chemical_units_v1/legacy_ga_v1",legacy_root)
    initial = [{"uid":r["candidate"].uid,"cell":r["candidate"].cell,"source_row":r["source_row"],
                "overall_available":r["whole"]["score"] is not None,"local_matches":len(r["packet"]),
                "score_ready":ready(r)} for r in pool]
    write(out/"INITIAL_POOL.json",initial)
    report = {"status":"coverage_checked","model_context":context,"strict_local_gate":bool(strict_local_gate),
              "initial_pool":{"size":len(pool),"cells":len({r["cell"] for r in initial}),
              "score_ready":len(ready_pool),"score_ready_cells":len({r["candidate"].cell for r in ready_pool})},
              "bootstrap_candidates":[r["candidate"].uid for r in seeds],"rounds":[],"critic_calls":0,
              "search_calls":0,"coupling_memory_used":True}
    write(out/"RUN_REPORT.json",report)
    print(json.dumps({"initial_pool":report["initial_pool"],"bootstrap_selected":len(seeds)}),flush=True)
    if key is None and resume is None and gateway_override is None:
        return report
    min_seeds = min(bootstrap_size, len(pool)) if initial_pool_override is not None else bootstrap_size
    if len(seeds)<min_seeds:
        report["status"] = "blocked_insufficient_bootstrap_coverage"
        write(out/"RUN_REPORT.json",report)
        return report

    prior = read(resume/"RUN_REPORT.json") if resume else None
    if prior and (prior["model_context"]!=context or prior["bootstrap_candidates"]!=report["bootstrap_candidates"]
                  or read(resume/"INITIAL_POOL.json")!=initial):
        raise ValueError("The checkpoint and reconstructed pool differ")
    continuing = bool(prior and prior.get("max_generations")==max_generations)
    if gateway_override is not None:
        gateway = gateway_override
    elif key is not None:
        gateway = OfficialTemperatureGateway(key,out/"api",limit=api_call_limit-(prior["api_calls"] if continuing else 0))
        gateway.request_timeout = 45
    else:
        gateway = None
    if evidence_finder is None and enable_evidence_search and gateway is not None:
        evidence_finder = AutonomousEvidenceFinder(
            gateway=gateway,
            out_dir=out / "evidence_search",
            max_search_requests=search_budget,
            cache_path=search_cache,
        )
    token_fields = ("prompt_tokens","completion_tokens","total_tokens","prompt_cache_hit_tokens")
    retries = int(prior.get("retries",0)) if continuing else 0
    prior_api = prior["api_calls"] if continuing else 0
    design_usages = []
    report["max_generations"] = max_generations
    report["call_limit"] = getattr(gateway, "limit", api_call_limit) if gateway else api_call_limit
    current_generation = 0
    mapping = None

    def _soft_fallback(row, result, reason):
        c = row["candidate"]
        overall = row["whole"]
        if overall["score"] is not None:
            raw_fallback = float(overall["score"])
            status = "overall_only_novel"
            conf = float(min(0.4, max(0.1, float(overall["weight"]) * 0.4)))
        else:
            if mapping is not None:
                pred_mean, _ = mapping.predict(c.psi)
                raw_fallback = float(np.clip(pred_mean[0] * 100.0, 0.0, 100.0))
            else:
                raw_fallback = 50.0
            status = "prior_only_frontier"
            conf = 0.15
        sa_val = _safe_sa(row["molecule"].pair)
        sa_pen = _compute_sa_penalty(sa_val, sa_threshold, sa_penalty_weight)
        fallback_score = float(np.clip(raw_fallback - sa_pen, 0.0, 100.0))
        memory_store.record_open_hypothesis(
            c.uid, c.cell, row["molecule"].identity, row["missing_kinds"], current_generation, reason
        )
        result.update(
            raw_combined=raw_fallback,
            sa_score=sa_val,
            sa_penalty=sa_pen,
            combined=fallback_score,
            overall=overall["score"],
            local=result.get("local") or {"score": None, "weight": 0.0, "judgements": []},
            evidence_refs=[],
            evidence_status=status,
            fallback_reason=reason,
        )
        return Evaluation(c.uid, context, fallback_score, conf, frozenset(), True, evidence_status=status), result

    def score_candidate(row, stage, suffix="first"):
        c = row["candidate"]
        refreshed_gap = float(c.evidence_gap)
        if evidence_finder is not None and suffix == "first" and (
            not row["packet"] or float(c.evidence_gap) >= float(search_gap_threshold)
        ):
            with search_lock:
                report["search_calls"] += 1
            added = evidence_finder(row["molecule"], row["units"], memory_store)
            if added:
                tags, payload = memory_store.prepare_local(row["units"])
                kinds = {t["kind"] for t in tags if t["used_for_judgement"] and t["similarity"]>=.5}
                missing_kinds = sorted(set(row["units"]["units"]) - kinds)
                refreshed_gap = 1 - len(kinds)/len(row["units"]["units"])
                row = row | {"tags": tags, "packet": payload, "missing_kinds": missing_kinds}
        if not row["packet"]:
            result = {"uid":c.uid,"cell":c.cell,"attempt":suffix,"api_status":"no_local_evidence",
                      "http_status":None,"usage":None}
            if strict_local_gate:
                result["error"] = "No local evidence packet"
                return None, result
            return _soft_fallback(row, result, "no_local_evidence")

        cached_judgements, uncached_packet = memory_store.lookup_cached(row["packet"], row["tags"])
        if uncached_packet:
            if gateway is None:
                result = {"uid":c.uid,"cell":c.cell,"attempt":suffix,"api_status":"no_gateway",
                          "http_status":None,"usage":None,"error":"No gateway configured"}
                return None, result
            payload = {"observations":uncached_packet}
            answer,audit = gateway.call("deepseek-flash","t05",f"{stage}_{c.uid[:12]}_{suffix}",TASK,payload,900)
        else:
            answer = {"judgements": []}
            audit = {"status": "ok", "http_status": 200, "usage": {k: 0 for k in token_fields}, "cached": True}

        result = {"uid":c.uid,"cell":c.cell,"attempt":suffix,"api_status":audit["status"],
                  "http_status":audit["http_status"],"usage":audit.get("usage"),
                  "cache_hits":len(cached_judgements),"cache_misses":len(uncached_packet)}
        if audit["status"]=="ok":
            try:
                if uncached_packet:
                    if not isinstance(answer,dict) or not isinstance(answer.get("judgements"),list):
                        raise ValueError("Invalid judgement envelope")
                    memory_store.store_cached(uncached_packet, row["tags"], answer["judgements"])
                    merged_judgements = cached_judgements + answer["judgements"]
                else:
                    merged_judgements = cached_judgements
                local = local_result(row["packet"],{"judgements":merged_judgements},row["tags"])
                result["local"] = local
                if local["score"] is None:
                    if evidence_finder is not None and suffix == "first":
                        with search_lock:
                            report["search_calls"] += 1
                        if evidence_finder(row["molecule"], row["units"], memory_store):
                            tags2, payload2 = memory_store.prepare_local(row["units"])
                            if len(payload2) > len(row["packet"]):
                                return score_candidate(row | {"tags": tags2, "packet": payload2}, stage, suffix="evidence_search")
                    if strict_local_gate:
                        result["error"] = "No directional local judgement"
                        return None,result
                    return _soft_fallback(row, result, "no_directional_local_judgement")
                overall = row["whole"]
                if overall["score"] is not None:
                    raw_combined = (overall["score"]*overall["weight"]+local["score"]*local["weight"])/(overall["weight"]+local["weight"])
                else:
                    raw_combined = float(local["score"])
                sa_val = _safe_sa(row["molecule"].pair)
                sa_pen = _compute_sa_penalty(sa_val, sa_threshold, sa_penalty_weight)
                combined = float(np.clip(raw_combined - sa_pen, 0.0, 100.0))
                ids = {j["id"] for j in local["judgements"] if j["direction"] in ("favorable","unfavorable")}
                refs = frozenset(t["reference"] for t in row["tags"] if t["id"] in ids)
                result.update(raw_combined=float(raw_combined),sa_score=sa_val,sa_penalty=sa_pen,
                              combined=combined,overall=overall["score"],evidence_refs=sorted(refs),
                              evidence_status="full",refreshed_evidence_gap=refreshed_gap)
                return Evaluation(c.uid,context,combined,local["weight"],refs,True,evidence_status="full"),result
            except (TypeError,KeyError,ValueError) as error:
                result["error"] = str(error)
        return None,result

    def supplemental(row):
        m = row["molecule"]
        tags,packet = scoped_furan_analogue(m.pair,extract(m.pair,m.uid))
        return row | {"tags":tags,"packet":packet} if packet else None

    def evaluate_batch(queue, stage):
        nonlocal retries
        evaluations, logs = {},[]
        folder = out/stage
        folder.mkdir()
        inputs_log = [
            {"uid":r["candidate"].uid,"payload":r["packet"],"catalyst_metadata":r["tags"],
             "target_catalyst":None,"overall_memory":r["whole"]} for r in queue]
        write(folder/"INPUTS.json",inputs_log)
        missing = queue
        if resume and (stage=="BOOTSTRAP" or continuing) and (resume/stage/"RESULTS.json").exists():
            if read(resume/stage/"INPUTS.json")!=inputs_log:
                raise ValueError(f"Saved inputs differ for {stage}")
            logs = read(resume/stage/"RESULTS.json")
            by_uid = {r["candidate"].uid:r for r in queue}
            for entry in logs:
                if "combined" not in entry:
                    continue
                uid = entry["uid"]
                row = by_uid[uid]
                status = entry.get("evidence_status", "full")
                if status in ("overall_only_novel", "prior_only_frontier"):
                    conf = 0.25 if status == "overall_only_novel" else 0.15
                    evaluations[uid] = Evaluation(uid, context, float(entry["combined"]), conf, frozenset(), True, evidence_status=status)
                    continue
                if entry["attempt"]=="evidence":
                    row = supplemental(row)
                    if row is None:
                        raise ValueError(f"Cannot reconstruct supplemental evidence for {uid}")
                local = local_result(row["packet"],{"judgements":entry["local"]["judgements"]},row["tags"])
                overall = row["whole"]
                if overall["score"] is not None:
                    total = (overall["score"]*overall["weight"]+local["score"]*local["weight"])/(overall["weight"]+local["weight"])
                else:
                    total = float(local["score"])
                sa_val = _safe_sa(row["molecule"].pair)
                total = float(np.clip(total - _compute_sa_penalty(sa_val, sa_threshold, sa_penalty_weight), 0.0, 100.0))
                if not np.isclose(total,entry["combined"],rtol=0,atol=1e-9):
                    raise ValueError(f"Saved score differs for {uid}")
                evaluations[uid] = Evaluation(uid,context,total,local["weight"],frozenset(entry["evidence_refs"]),True,evidence_status="full")
            missing = [r for r in queue if r["candidate"].uid not in evaluations]
            write(folder/"RESULTS.json",logs)
        if inspect_only:
            if missing:
                write(folder/"BLOCKED_STRUCTURES.json",[
                    {"uid":r["candidate"].uid,"smiles":r["molecule"].identity,
                     "site":r["candidate"].parent_site,
                     "units":extract(r["molecule"].pair,r["candidate"].uid)["units"]} for r in missing])
            return evaluations,{k:sum((v.get("usage") or {}).get(k,0) for v in logs) for k in token_fields}
        supplements = {}
        diagnosed = {r["uid"] for r in logs if r.get("error")=="No directional local judgement"}
        def add_supplement(row):
            uid = row["candidate"].uid
            extra = supplemental(row)
            if extra:
                supplements[uid] = extra
            return extra is not None

        assigned = [r for r in missing if r["candidate"].uid not in diagnosed or not add_supplement(r)]
        with ThreadPoolExecutor(4) as workers:
            handles = [workers.submit(score_candidate,r,stage) for r in assigned]
            retry = []
            for row,handle in zip(assigned,handles):
                e,log = handle.result()
                logs.append(log)
                if e is not None:
                    evaluations[e.uid] = e
                elif log.get("error")=="No directional local judgement" and resume:
                    add_supplement(row)
                elif log["api_status"] not in ("ok","soft_fallback","no_local_evidence") and log["http_status"] not in (401,402,403):
                    retry.append(row)
                write(folder/"RESULTS.json",logs)
            permitted = (retry[:max(0,min(retry_budget-retries,gateway.limit-gateway.calls))]
                         if (gateway and not getattr(gateway, "quota_blocked", False)) else [])
            retries += len(permitted)
            for handle in [workers.submit(score_candidate,r,stage,"retry") for r in permitted]:
                e,log = handle.result()
                logs.append(log)
                if e is not None:
                    evaluations[e.uid] = e
                write(folder/"RESULTS.json",logs)
            if supplements and gateway and not getattr(gateway, "quota_blocked", False):
                permitted = list(supplements.values())[:max(0,min(retry_budget-retries,gateway.limit-gateway.calls))]
                retries += len(permitted)
                write(folder/"SUPPLEMENT.json",[
                    {"uid":r["candidate"].uid,"metadata":r["tags"],"payload":r["packet"]} for r in permitted])
                for handle in [workers.submit(score_candidate,r,stage,"evidence") for r in permitted]:
                    e,log = handle.result()
                    logs.append(log)
                    if e is not None:
                        evaluations[e.uid] = e
                    write(folder/"RESULTS.json",logs)
        totals = {k:sum((r.get("usage") or {}).get(k,0) for r in logs) for k in token_fields}
        return evaluations,totals

    init_scores, tokens = evaluate_batch(seeds,"BOOTSTRAP")
    report["bootstrap"] = {"selected":len(seeds),"scored":len(init_scores),"tokens":tokens,
        "reused":bool(resume and (resume/"BOOTSTRAP/RESULTS.json").exists()),
        "scores":{uid:e.score for uid,e in init_scores.items()},
        "evidence_statuses":{uid:e.evidence_status for uid,e in init_scores.items()}}
    if len(init_scores)!=len(seeds):
        report.update(status="blocked_waiting_for_bootstrap_scores",api_calls=(gateway.calls if gateway else 0))
        write(out/"RUN_REPORT.json",report)
        return report
    features = {r["candidate"].uid:r["candidate"] for r in seeds}
    mapping = KalmanMap(len(seeds[0]["candidate"].psi),context,prior_cov=prior_cov)
    for uid,e in init_scores.items():
        obs_var = (.02+.18*(1-e.evidence_confidence)**2) if e.evidence_status=="full" else (.15+.20*(1-e.evidence_confidence)**2)
        mapping.update(uid,features[uid].psi,e.score/100,obs_var,context)
    state = PopulationState(mapping,features,init_scores)
    excluded = {r["candidate"].uid for r in pool}
    parents = [r["molecule"] for r in seeds]
    molecules = {m.uid:m for m in parents}
    whole_map,local_map = (KalmanMap(len(mapping.mean),context,prior_cov=prior_cov) for _ in range(2))
    population_policy = PopulationUCB()
    context_scores = {}
    rule_bank = ChemicalRuleBank() if enable_chemical_rules else None
    hybrid_surrogate = HybridChemicalSurrogate(prior_cov, context) if enable_chemical_rules else None
    rule_observations = []
    rule_induction_usages = []
    seen_claim_refs = {
        row.get("reference")
        for group in memory_store.local_memory["unit_index"].values()
        for row in group
    }
    if enable_chemical_rules:
        for uid, e in init_scores.items():
            obs_var = (.02+.18*(1-e.evidence_confidence)**2) if e.evidence_status=="full" else (.15+.20*(1-e.evidence_confidence)**2)
            nh2_0, h2_wt_0 = properties(molecules[uid].pair)
            rule_observations.append({
                "uid": uid,
                "psi": features[uid].psi,
                "pair": molecules[uid].pair,
                "score_01": float(e.score) / 100.0,
                "noise_var": float(obs_var),
                "lean_smiles": molecules[uid].identity[0],
                "sa_score": _safe_sa(molecules[uid].pair),
                "h2_wt_percent": float(h2_wt_0),
            })
        hybrid_surrogate.fit(rule_observations, rule_bank, auto_falsify=False)

    def update_axes(rows,stage,evaluations):
        latest = {r["uid"]:r for r in read(out/stage/"RESULTS.json") if "combined" in r}
        inputs_map = {r["uid"]:r for r in read(out/stage/"INPUTS.json")}
        for row in rows:
            x = latest[row.uid]
            ev = evaluations[row.uid]
            overall_score = x.get("overall")
            local_info = x.get("local") or {}
            local_score = local_info.get("score")
            lw = float(local_info.get("weight", 0.0) or 0.0)
            gw = float(inputs_map[row.uid]["overall_memory"].get("weight", 0.0) or 0.0)
            g_val = float(overall_score) if overall_score is not None else float(ev.score)
            l_val = float(local_score) if local_score is not None else float(ev.score)
            blend_w = lw / (lw + gw) if (lw + gw) > 0 else 0.0
            context_scores[row.uid] = (g_val, l_val, blend_w)
            if overall_score is not None:
                whole_map.update(row.uid,row.psi,float(overall_score)/100,.10,context)
            if local_score is not None:
                local_map.update(row.uid,row.psi,float(local_score)/100,.10+.15*(1-lw),context)

    def write_telemetry(stage_name, committed_rows, role_map, evaluations):
        w_means, w_sds = whole_map.cell_surface()
        l_means, l_sds = local_map.cell_surface()
        c_means, c_sds = state.mapping.cell_surface()
        cell_members = defaultdict(list)
        for uid, cand in state.members.items():
            cell_members[cand.cell].append(cand)
        surface = []
        for cid in range(len(centers)):
            members = cell_members.get(cid, [])
            best_score = max((state.evaluations[c.uid].score for c in members if c.uid in state.evaluations), default=None)
            avg_gap = float(np.mean([c.evidence_gap for c in members])) if members else 1.0
            surface.append({
                "cell": cid,
                "occupied": len(members),
                "best_score": best_score,
                "avg_evidence_gap": avg_gap,
                "combined_mean": round(float(c_means[cid] * 100.0), 3),
                "combined_sd": round(float(c_sds[cid] * 100.0), 3),
                "whole_mean": round(float(w_means[cid] * 100.0), 3),
                "whole_sd": round(float(w_sds[cid] * 100.0), 3),
                "local_mean": round(float(l_means[cid] * 100.0), 3),
                "local_sd": round(float(l_sds[cid] * 100.0), 3),
            })
        write(out/f"{stage_name}_SOM_SURFACE.json", {
            "stage": stage_name,
            "generation": current_generation,
            "occupied_cells": sum(1 for s in surface if s["occupied"] > 0),
            "cells": surface,
        })
        for uid in committed_rows:
            mol = molecules[uid]
            cand = state.members[uid]
            ev = evaluations[uid]
            nh2, h2_wt = properties(mol.pair)
            sa_val = _safe_sa(mol.pair)
            sa_pen = _compute_sa_penalty(sa_val, sa_threshold, sa_penalty_weight)
            trajectory.append({
                "generation": current_generation,
                "stage": stage_name,
                "uid": uid,
                "lean_smiles": mol.identity[0],
                "rich_smiles": mol.identity[1],
                "parent_uid": cand.parent_uid,
                "role": role_map.get(uid, "bootstrap"),
                "operator": cand.arm[-1],
                "cell": int(cand.cell),
                "score": float(ev.score),
                "evidence_status": ev.evidence_status,
                "evidence_gap": float(cand.evidence_gap),
                "nH2": float(nh2),
                "h2_wt_percent": float(h2_wt),
                "sa_score": float(sa_val) if sa_val is not None else None,
                "sa_penalty": float(sa_pen),
            })
        write(out/"TRAJECTORY.json", trajectory)
        memory_store.save(out/"MEMORY_LIVE.json")

    trajectory = []
    update_axes(list(features.values()),"BOOTSTRAP",init_scores)
    for uid, ev in init_scores.items():
        latest_bs = {r["uid"]: r for r in read(out/"BOOTSTRAP/RESULTS.json")}
        entry = latest_bs.get(uid, {})
        memory_store.add_global_record(
            molecules[uid], ev.score, entry.get("overall"), (entry.get("local") or {}).get("score"),
            context, ev.evidence_refs, 0,
        )
    write_telemetry("BOOTSTRAP", [r["candidate"].uid for r in seeds], {r["candidate"].uid: "bootstrap" for r in seeds}, init_scores)

    def design_menu(pool, root_uid, remaining, visited_dst_cells=None):
        root = state.members[root_uid]
        root_mol = molecules[root_uid]
        root_sa = _safe_sa(root_mol.pair)
        root_sa_pen = _compute_sa_penalty(root_sa, sa_threshold, sa_penalty_weight)
        root_nh2, root_h2 = properties(root_mol.pair)
        root_units = unit_summary_by_uid.get(root_uid, set())
        w_means, w_sds = whole_map.cell_surface()
        l_means, l_sds = local_map.cell_surface()
        cell_counts = Counter(m.cell for m in state.members.values())
        visited_counts = visited_dst_cells or {}

        def _sa_aware_sort_key(r):
            c = r["candidate"]
            overall, _ = whole_map.delta(root.psi, c.psi)
            local, _ = local_map.delta(root.psi, c.psi)
            cand_sa = _safe_sa(r["molecule"].pair)
            delta_sa_pen = _compute_sa_penalty(cand_sa, sa_threshold, sa_penalty_weight) - root_sa_pen
            net_gain = local * 100.0 + 0.5 * overall * 100.0 - delta_sa_pen
            return (-net_gain, c.uid)

        def details(r, step, base_uid):
            c = r["candidate"]
            mol = r["molecule"]
            overall,ov = whole_map.delta(root.psi,c.psi)
            local,lv = local_map.delta(root.psi,c.psi)
            cand_sa = _safe_sa(mol.pair)
            cand_sa_pen = _compute_sa_penalty(cand_sa, sa_threshold, sa_penalty_weight)
            delta_sa = (cand_sa - root_sa) if (cand_sa is not None and root_sa is not None) else None
            delta_sa_pen = cand_sa_pen - root_sa_pen
            nh2_v, h2_wt_v = properties(mol.pair)
            base_units = unit_summary_by_uid.get(base_uid, root_units)
            child_units = set(r["unit_summary"])
            return {"site":c.parent_site,"operator":c.arm[-1],"target_cell":c.cell,
                "lean_smiles":mol.identity[0],"rich_smiles":mol.identity[1],
                "sa_score":round(float(cand_sa),3) if cand_sa is not None else None,
                "delta_sa":round(float(delta_sa),3) if delta_sa is not None else None,
                "sa_penalty_points":round(float(cand_sa_pen),3),
                "net_pred_delta_points":round(float(overall*100+local*100-delta_sa_pen),3),
                "nH2":float(nh2_v),"h2_wt_percent":round(float(h2_wt_v),3),
                "delta_h2_wt_percent":round(float(h2_wt_v-root_h2),3),
                "dst_cell_already_directed":bool(visited_counts.get(int(c.cell), 0) > 0),
                "destination_response_delta":round(float(c.response[c.cell]-root.response[c.cell]),6),
                "whole_delta_points":round(overall*100,3),"local_delta_points":round(local*100,3),
                "whole_sd_points":round(ov**.5*100,2),"local_sd_points":round(lv**.5*100,2),
                "evidence_gap":c.evidence_gap,"step":step,
                "added_units":sorted(child_units - base_units),
                "removed_units":sorted(base_units - child_units),
                "target_cell_kalman":{
                    "whole_mean":round(float(w_means[c.cell]*100),2),
                    "whole_sd":round(float(w_sds[c.cell]*100),2),
                    "local_mean":round(float(l_means[c.cell]*100),2),
                    "local_sd":round(float(l_sds[c.cell]*100),2),
                    "occupied_count":int(cell_counts[c.cell]),
                },
                "matched_memory":[{"source_id":t["source_id"],"kind":t["kind"],
                                   "state":t["state"],"similarity":round(float(t["similarity"]),3)}
                                  for t in r["tags"][:3]],
                "uncovered_unit_kinds":r["missing_kinds"]}
        options = sorted((r for r in pool if r["candidate"].parent_uid==root_uid and ready(r)),
                         key=_sa_aware_sort_key)
        paths,payload = {},[]
        for i,row in enumerate(options[:5],1):
            edit_id = f"A{i}"
            paths[(edit_id,)]=row
            item = {"id":edit_id,**details(row,1,root_uid),"continuations":[]}
            if i<=3:
                more,_ = propose([row["molecule"]],remaining)
                options2 = sorted((v for v in more.values() if ready(v)),
                                  key=_sa_aware_sort_key)
                for j,r in enumerate(options2[:2],1):
                    child = r["candidate"]
                    branch = f"{edit_id}B{j}"
                    proxy = EditCandidate(child.uid,root_uid,child.parent_site,child.cell,child.psi,
                        child.response,(child.arm[0],child.arm[1],"composite_design"),child.units,
                        child.evidence_gap,child.evidence_refs)
                    paths[(edit_id,branch)] = r|{"candidate":proxy}
                    item["continuations"].append({"id":branch,**details(r,2,row["candidate"].uid)})
            payload.append(item)
        root_summary = {
            "uid":root_uid,"cell":int(root.cell),
            "lean_smiles":root_mol.identity[0],"rich_smiles":root_mol.identity[1],
            "score":round(float(state.evaluations[root_uid].score),3),
            "sa_score":round(float(root_sa),3) if root_sa is not None else None,
            "sa_penalty_points":round(float(root_sa_pen),3),
            "nH2":float(root_nh2),"h2_wt_percent":round(float(root_h2),3),
            "units":sorted(root_units),
        }
        design_out = {"goal":"Improve local performance and H2 storage while retaining overall score, penalizing high SA, and covering unexplored SOM cells",
                      "seed_cell":root.cell,"root_molecule":root_summary,
                      "sa_penalty_rule":{"threshold":float(sa_threshold),"weight":float(sa_penalty_weight)},
                      "scoring_observations":len(whole_map.observed),"allowed_edit_paths":payload}
        if rule_bank is not None and rule_bank.active_rules():
            design_out["discovered_chemical_rules"] = rule_bank.summary_for_prompt(limit=6)
        return paths,design_out

    write(out/"RUN_REPORT.json",report)
    design_root_counts = Counter()
    directed_dst_cells = Counter()
    directed_uids = set()
    prev_directed_uid = None
    directed_parent_retention_hits = 0
    stop = "ten_generations_committed" if max_generations == 10 else f"{max_generations}_generations_committed"
    for generation in range(1,max_generations+1):
        current_generation = generation
        stage = f"G{generation}"
        known = excluded | state.members.keys()
        candidates,invalid = propose(parents,known)
        matrix_candidates = list(candidates.values())
        if not matrix_candidates:
            stop = "blocked_no_legal_matrix_edits"
            break
        # Distinct whole vs local gaps drive hierarchical PopulationUCB selection.
        whole_gap = float(np.clip(np.mean([1.0 - float(r["whole"]["weight"]) for r in matrix_candidates]), 0.0, 1.0))
        local_gap = float(np.clip(np.mean([float(r["candidate"].evidence_gap) for r in matrix_candidates]), 0.0, 1.0))
        policy,policy_audit = population_policy.choose({"whole":whole_gap,"local":local_gap})
        ga = native_generate(legacy_root,input_runtime,
                             parents,known|candidates.keys(),budget=native_ga_budget,policy=policy,generation=generation)
        native_children,rejected = ga.pop("valid"),ga.pop("rejected")
        for m,origin in native_children:
            # Legacy fragment crossover only preserves the donor pool, so
            # use a scored reference for ranking without asserting parentage.
            reference = molecules.get(origin["parent_uid"])
            reference = reference if reference is not None else parents[0]
            row = encode(m,reference,0,origin["operation"])
            candidates[m.uid] = row | {"native":origin,"reference_uid":reference.uid}
        routes_by_operation = Counter(o["operation"] for _,o in native_children)
        root_options = [
            r["candidate"].parent_uid for r in matrix_candidates if ready(r)
        ]
        root_uid = max(
            set(root_options),
            key=lambda uid: (state.evaluations[uid].score - float(root_cooldown) * design_root_counts[uid], uid),
            default=None,
        )
        paths,design_packet = design_menu(
            matrix_candidates, root_uid, known|candidates.keys(), visited_dst_cells=directed_dst_cells
        ) if root_uid else ({},{})
        step = {"generation":generation,"candidate_count":len(candidates),
                "matrix_candidates":len(matrix_candidates),"native_ga":{
                    **ga,"valid_by_operation":dict(routes_by_operation),"rejected":len(rejected),**policy_audit},
                "rejected_matrix_edits":len(invalid),"population_before":len(state.members),
                "design_paths":len(paths),"design_root":root_uid,
                "whole_gap":whole_gap,"local_gap":local_gap}
        report["rounds"].append(step)
        if not paths:
            stop = "blocked_no_evidence_ready_directed_path"
            break
        write(out/f"{stage}_DESIGN_INPUT.json",design_packet)
        root_mol = molecules[root_uid]
        root_cand = state.members[root_uid]
        root_nh2, root_h2_wt = properties(root_mol.pair)
        menu_lookup = {}
        for item in design_packet.get("allowed_edit_paths", []):
            menu_lookup[(item["id"],)] = item
            for sub in item.get("continuations", []):
                menu_lookup[(item["id"], sub["id"])] = sub
        root_sa = _safe_sa(root_mol.pair)
        design_cand_rows = []
        cand_ids = []
        cand_psis = []
        for ep, row in sorted(paths.items(), key=lambda kv: kv[0]):
            m_obj = row["molecule"]
            c_obj = row["candidate"]
            nh2_v, h2_wt_v = properties(m_obj.pair)
            cand_sa = _safe_sa(m_obj.pair)
            cand_sa_pen = _compute_sa_penalty(cand_sa, sa_threshold, sa_penalty_weight)
            meta = menu_lookup.get(ep, {})
            cid_str = str(meta.get("id", ep[-1]))
            cand_ids.append(cid_str)
            cand_psis.append(np.asarray(c_obj.psi, dtype=float))
            design_cand_rows.append({
                "edit_path": list(ep),
                "id": cid_str,
                "step": int(meta.get("step", len(ep))),
                "uid": m_obj.uid,
                "cell": int(c_obj.cell),
                "operator": str(meta.get("operator", c_obj.arm[-1])),
                "site": int(meta.get("site", c_obj.parent_site)),
                "lean_smiles": m_obj.identity[0],
                "rich_smiles": m_obj.identity[1],
                "sa_score": float(cand_sa) if cand_sa is not None else None,
                "sa_penalty_points": float(cand_sa_pen),
                "nH2": float(nh2_v),
                "h2_wt_percent": float(h2_wt_v),
                "whole_delta_points": float(meta.get("whole_delta_points", 0.0)),
                "local_delta_points": float(meta.get("local_delta_points", 0.0)),
                "whole_sd_points": float(meta.get("whole_sd_points", 0.0)),
                "local_sd_points": float(meta.get("local_sd_points", 0.0)),
                "destination_response_delta": float(meta.get("destination_response_delta", 0.0)),
                "evidence_gap": float(meta.get("evidence_gap", c_obj.evidence_gap)),
            })
        write(out/f"{stage}_DESIGN_CANDIDATES.json", {
            "root": {
                "uid": root_uid,
                "cell": int(root_cand.cell),
                "lean_smiles": root_mol.identity[0],
                "rich_smiles": root_mol.identity[1],
                "score": float(state.evaluations[root_uid].score),
                "sa_score": float(root_sa) if root_sa is not None else None,
                "sa_penalty_points": float(_compute_sa_penalty(root_sa, sa_threshold, sa_penalty_weight)),
                "nH2": float(root_nh2),
                "h2_wt_percent": float(root_h2_wt),
            },
            "candidates": design_cand_rows,
        })
        np.savez_compressed(
            out/f"{stage}_DESIGN_PSI.npz",
            root_psi=np.asarray(root_cand.psi, dtype=float),
            candidate_ids=np.asarray(cand_ids, dtype=str),
            candidate_psi=np.stack(cand_psis) if cand_psis else np.zeros((0, len(root_cand.psi)), dtype=float),
        )
        if inspect_only and not (continuing and (resume/f"{stage}_DESIGN_RESULT.json").exists()):
            stop = "preflight_ready"
            break
        prior_design = resume/f"{stage}_DESIGN_RESULT.json" if continuing else None
        if prior_design and prior_design.exists():
            saved = read(prior_design)
            if read(resume/f"{stage}_DESIGN_INPUT.json")!=design_packet:
                raise ValueError("Saved directed design menu differs")
            answer,audit = saved["answer"],saved["audit"]
        else:
            if gateway.calls>=gateway.limit:
                stop = "blocked_call_budget_before_design"
                break
            answer,audit = gateway.call("deepseek-flash","t05",f"{stage}_design",DESIGN_TASK,design_packet,650)
            audit = {k:audit.get(k) for k in ("status","usage","http_status")}
        write(out/f"{stage}_DESIGN_RESULT.json",{"answer":answer,"audit":audit})
        design_usages.append(audit.get("usage") or {})
        step["design_tokens"] = (audit.get("usage") or {}).get("total_tokens",0)
        try:
            directed = validate_design(answer,paths)
        except ValueError as error:
            step["design_error"] = str(error)
            stop = "blocked_invalid_directed_design"
            break
        lead = directed["candidate"]
        design_root_counts[root_uid] += 1
        directed_dst_cells[int(lead.cell)] += 1
        directed_uids.add(lead.uid)
        step["directed_edit_ids"] = answer["edit_ids"]
        step["directed_reason"] = answer["reason"]
        candidates[lead.uid] = directed
        eligible = [r["candidate"] for r in candidates.values() if ready(r)]
        ordinary = [c for c in eligible if c.uid!=lead.uid and c.uid not in state.members]
        plan = generation_quotas(state,ordinary,min(shortlist_size,len(ordinary)),directed_budget=max(1,batch_size//5))
        shortlist = [candidates[r["uid"]]["candidate"] for r in plan["routes"]]
        if prev_directed_uid is not None:
            shortlist_uids = {c.uid for c in shortlist}
            relay_cands = [
                c for c in ordinary
                if c.parent_uid == prev_directed_uid and c.uid not in shortlist_uids
            ]
            if relay_cands:
                shortlist = shortlist + relay_cands[:3]
        native_eligible = [r for r in ordinary if candidates[r.uid].get("native")]
        native_cap = min(max(2, batch_size // 5), len(native_eligible))
        native_pick,_ = select_batch(state,native_eligible,native_cap)
        reserved = {r.uid for r,_ in native_pick}
        relay_pick = []
        if prev_directed_uid is not None:
            relay_pool = [
                c for c in ordinary
                if c.parent_uid == prev_directed_uid and c.uid not in reserved and not candidates[c.uid].get("native")
            ]
            if relay_pool:
                relay_pick, relay_audit = select_batch(state, relay_pool, 1)
                for c_r, _ in relay_pick:
                    reserved.add(c_r.uid)
        picked,audit = select_batch(state,[r for r in shortlist if r.uid not in reserved],max(1,(batch_size-1)-len(native_pick)-len(relay_pick)))
        if prev_directed_uid is not None and relay_pick:
            audit = {**audit, **relay_audit}
        ordinary_selection = native_pick+relay_pick+picked
        selected = [(lead,"directed_parent"),*ordinary_selection]
        step.update(score_ready_count=len(eligible),proposed_routes=plan["routes"],
                    selected=[{"uid":r.uid,"role":role,"cell":r.cell,
                               "parent":(candidates[r.uid].get("native") or {}).get("parent_uid") if "native" in candidates[r.uid] else r.parent_uid,
                               "ranking_reference":r.parent_uid,"arm":r.arm,
                               "origin":candidates[r.uid].get("native",{}),**audit.get(r.uid,{})}
                              for r,role in selected])
        write(out/"RUN_REPORT.json",report)
        if not ordinary_selection:
            stop = "blocked_no_ordinary_candidate"
            break
        already = set()
        if continuing and (resume/stage/"RESULTS.json").exists():
            already = {r["uid"] for r in read(resume/stage/"RESULTS.json") if "combined" in r}
        if gateway and gateway.limit-gateway.calls < sum(r.uid not in already for r,_ in selected):
            stop = "blocked_call_budget_before_scoring"
            break
        print(json.dumps({"generation":generation,"candidates":len(candidates),
                          "score_ready":len(eligible),"selected":len(selected),"native":dict(routes_by_operation)}),flush=True)
        queue = [r for r,_ in selected]
        observations,tokens = evaluate_batch([candidates[r.uid] for r in queue],stage)
        edit_evidence = {}
        for r, role in selected:
            if "native" not in candidates[r.uid] and r.parent_uid in state.evaluations:
                ev_refs = observations[r.uid].evidence_refs if r.uid in observations and observations[r.uid].evidence_refs else r.evidence_refs
                edit_evidence[r.uid] = EditEvidence(
                    r.parent_uid, r.uid, r.parent_site, context, ev_refs, True
                )
        previous = state
        state, gate = commit_batch(previous,queue,observations,edit_evidence,strict_local_gate=strict_local_gate)
        step.update(scored=len(observations),scores={uid:e.score for uid,e in observations.items()},
                    evidence_statuses={uid:e.evidence_status for uid,e in observations.items()},
                    tokens=tokens,commit=gate,population_after=len(state.members),
                    cells=len({r.cell for r in state.members.values()}),
                    new_cells=len({r.cell for r in state.members.values()}-{r.cell for r in previous.members.values()}),
                    new_local_signatures=len(set().union(*(r.units for r in state.members.values()))-
                                               set().union(*(r.units for r in previous.members.values()))))
        write(out/"RUN_REPORT.json",report)
        if not gate["committed"]:
            stop = "blocked_waiting_for_valid_scores"
            break
        update_axes(queue,stage,observations)
        parent_scores = [context_scores[p.uid] for p in parents if p.uid in context_scores]
        pool_baseline = tuple(float(np.mean([row[idx] for row in parent_scores])) for idx in range(3)) if parent_scores else None
        matrix_deltas = [
            score_delta(context_scores[r.parent_uid], context_scores[r.uid])
            for r in queue
            if not candidates[r.uid].get("native") and r.parent_uid in context_scores
        ]
        native_deltas = []
        for r in queue:
            native_info = candidates[r.uid].get("native")
            if not native_info:
                continue
            ref_uid = native_info.get("parent_uid")
            base_score = context_scores.get(ref_uid) if ref_uid else pool_baseline
            if base_score is not None:
                native_deltas.append(score_delta(base_score, context_scores[r.uid]))
        cohort_delta = matrix_deltas + native_deltas
        native_requested = max(1, sum(ga.get("requested", {}).values()))
        native_legal_yield = min(1.0, len(native_children) / native_requested) if native_deltas else None
        if cohort_delta:
            step["coarse_feedback"] = population_policy.observe(
                stage,
                cohort_delta,
                step["new_cells"]/len(queue),
                step["new_local_signatures"]/(step["new_local_signatures"]+len(queue)),
                policy=(policy.name if native_deltas else None),
                legal_yield=native_legal_yield,
            )
        step["native_policy_reward_updated"] = bool(native_deltas)
        step["memory_snapshot"] = memory_store.snapshot()
        state.mapping.save(out/f"{stage}_MAPPING.npz")
        molecules.update({candidates[r.uid]["molecule"].uid: candidates[r.uid]["molecule"] for r in queue})
        latest_stage = {r["uid"]: r for r in read(out/stage/"RESULTS.json") if "combined" in r}
        for r in queue:
            ev = observations[r.uid]
            entry = latest_stage.get(r.uid, {})
            memory_store.add_global_record(
                molecules[r.uid], ev.score, entry.get("overall"), (entry.get("local") or {}).get("score"),
                context, ev.evidence_refs, generation,
            )
        parents, parent_audit = select_next_parents(
            state,
            molecules,
            max_parents=max_parents,
            elite_quota=elite_quota,
            protected_uids=([lead.uid] if protect_directed_parent else ()),
        )
        if lead.uid in {m.uid for m in parents}:
            directed_parent_retention_hits += 1
        prev_directed_uid = lead.uid
        step["parent_selection"] = parent_audit
        if enable_chemical_rules and rule_bank is not None and hybrid_surrogate is not None:
            for r in queue:
                ev = observations[r.uid]
                mol_r = molecules[r.uid]
                obs_var = (.02+.18*(1-ev.evidence_confidence)**2) if ev.evidence_status=="full" else (.15+.20*(1-ev.evidence_confidence)**2)
                nh2_r, h2_wt_r = properties(mol_r.pair)
                rule_observations.append({
                    "uid": r.uid,
                    "psi": r.psi,
                    "pair": mol_r.pair,
                    "score_01": float(ev.score) / 100.0,
                    "noise_var": float(obs_var),
                    "lean_smiles": mol_r.identity[0],
                    "sa_score": _safe_sa(mol_r.pair),
                    "h2_wt_percent": float(h2_wt_r),
                })
            hybrid_surrogate.fit(rule_observations, rule_bank, auto_falsify=True)
            new_claims_gen = []
            for kind_k, rows_k in memory_store.local_memory["unit_index"].items():
                for row_k in rows_k:
                    ref_k = row_k.get("reference")
                    if ref_k not in seen_claim_refs:
                        seen_claim_refs.add(ref_k)
                        ev_k = row_k.get("evidence", {})
                        su_k = row_k.get("source_unit", {})
                        new_claims_gen.append({
                            "source_id": row_k.get("source_id"),
                            "reference": ref_k,
                            "kind": kind_k,
                            "unit_name": su_k.get("name", ""),
                            "state": su_k.get("state", ""),
                            "stream": row_k.get("stream", "local"),
                            "scope_tier": ev_k.get("scope_tier"),
                            "observation": ev_k.get("observation", ""),
                        })
            if not new_claims_gen:
                for r in queue[:4]:
                    for t_k in (candidates[r.uid].get("tags") or [])[:2]:
                        if t_k.get("used_for_judgement"):
                            new_claims_gen.append({
                                "source_id": t_k.get("source_id"),
                                "reference": t_k.get("reference"),
                                "kind": t_k.get("kind", ""),
                                "unit_name": t_k.get("unit", ""),
                                "state": t_k.get("state", ""),
                                "stream": t_k.get("stream", "local"),
                                "scope_tier": t_k.get("scope_tier"),
                                "observation": t_k.get("observation", ""),
                            })
            mmp_pairs_gen = []
            for r in queue:
                if r.parent_uid in state.evaluations and r.parent_uid in molecules:
                    p_mol = molecules[r.parent_uid]
                    c_mol = molecules[r.uid]
                    _, p_h2 = properties(p_mol.pair)
                    _, c_h2 = properties(c_mol.pair)
                    p_sa = _safe_sa(p_mol.pair)
                    c_sa = _safe_sa(c_mol.pair)
                    mmp_pairs_gen.append({
                        "parent_uid": r.parent_uid,
                        "child_uid": r.uid,
                        "parent_smiles": p_mol.identity[0],
                        "child_smiles": c_mol.identity[0],
                        "operator": r.arm[-1],
                        "delta_score": round(float(observations[r.uid].score - state.evaluations[r.parent_uid].score), 3),
                        "delta_sa": round(float(c_sa - p_sa), 3) if (c_sa is not None and p_sa is not None) else None,
                        "delta_h2_wt": round(float(c_h2 - p_h2), 3),
                    })
            mmp_pairs_gen.sort(key=lambda x: -abs(x["delta_score"]))
            ind_res = induce_rules_from_generation(
                gateway=gateway,
                stage=stage,
                generation=generation,
                observations=rule_observations,
                surrogate=hybrid_surrogate,
                rule_bank=rule_bank,
                new_claims=new_claims_gen,
                mmp_pairs=mmp_pairs_gen,
            )
            fit_summary = hybrid_surrogate.fit(rule_observations, rule_bank, auto_falsify=True)
            rule_induction_usages.append(ind_res.get("usage") or {})
            step["rule_induction_status"] = ind_res.get("status")
            step["chemical_rules_summary"] = {
                "total_rules": len(rule_bank.rules),
                "active_rules": fit_summary["active_rules"],
                "supported_rules": sum(1 for r_x in rule_bank.rules.values() if r_x.status in ("supported", "revised")),
                "falsified_rules": sum(1 for r_x in rule_bank.rules.values() if r_x.status == "falsified"),
                "newly_falsified": fit_summary["newly_falsified"],
            }
            write(out/f"{stage}_CHEMICAL_RULES.json", {
                "generation": generation,
                "induction": {k: v for k, v in ind_res.items() if k != "raw_answer"},
                "fit_summary": fit_summary,
                **rule_bank.to_dict(),
            })
        write_telemetry(stage, [r.uid for r in queue], {r.uid: role for r, role in selected}, observations)
        write(out/f"{stage}_PARENTS.json",[{"uid":m.uid,"states":m.identity,"parent":m.parent,
                                         "edit":m.action} for m in parents])
        write(out/"RUN_REPORT.json",report)
        if generation==max_generations:
            downstream,bad = propose(parents,excluded | state.members.keys())
            report["following_candidates"] = {"valid":len(downstream),"invalid":len(bad)}
    batches = ([] if report["bootstrap"]["reused"] else [report["bootstrap"]])+report["rounds"]
    direct_children_of_directed = [
        row["uid"] for row in trajectory
        if row.get("parent_uid") in directed_uids and row["uid"] not in directed_uids
    ]
    lineage_set = set(directed_uids)
    frontier_q = list(directed_uids)
    while frontier_q:
        curr_p = frontier_q.pop(0)
        for row in trajectory:
            if row.get("parent_uid") == curr_p and row["uid"] not in lineage_set:
                lineage_set.add(row["uid"])
                frontier_q.append(row["uid"])
    sa_all = [row["sa_score"] for row in trajectory if row.get("sa_score") is not None]
    sa_dir = [row["sa_score"] for row in trajectory if row.get("role") == "directed_parent" and row.get("sa_score") is not None]
    directed_lineage = {
        "directed_count": len(directed_uids),
        "directed_in_parents_generations": int(directed_parent_retention_hits),
        "direct_children_of_directed": len(direct_children_of_directed),
        "total_descendants_of_directed": int(len(lineage_set - directed_uids)),
        "unique_design_roots": len(design_root_counts),
        "unique_directed_cells": len(directed_dst_cells),
        "mean_sa_all": round(float(np.mean(sa_all)), 3) if sa_all else None,
        "mean_sa_directed": round(float(np.mean(sa_dir)), 3) if sa_dir else None,
    }
    report.update(status=stop,api_calls=(gateway.calls if gateway else 0)+prior_api,retries=retries,total_tokens={
        k:sum(b.get("tokens",{}).get(k,0) for b in batches)+sum(v.get(k,0) for v in design_usages)+sum(v.get(k,0) for v in rule_induction_usages) for k in token_fields},
        final_population=len(state.members),actual_generations=sum(r.get("commit",{}).get("committed",False)
                                                                   for r in report["rounds"]),
        final_memory=memory_store.snapshot(),
        directed_lineage=directed_lineage,
        experiment_config={
            "sa_penalty_weight": float(sa_penalty_weight),
            "sa_threshold": float(sa_threshold),
            "protect_directed_parent": bool(protect_directed_parent),
            "root_cooldown": float(root_cooldown),
            "enable_chemical_rules": bool(enable_chemical_rules),
        },
        chemical_rules=rule_bank.to_dict() if rule_bank is not None else None,
        ucb_operator_posts={ "|".join(k): v for k, v in state.posts.items() },
        population_ucb={"coarse": {k: vars(v) for k, v in population_policy.coarse.items()},
                          "fine": {k: vars(v) for k, v in population_policy.fine.items()}})
    if evidence_finder is not None and hasattr(evidence_finder, "summary"):
        ev_summary = evidence_finder.summary()
        report["evidence_search"] = ev_summary
        report["search_calls"] = ev_summary.get("search_requests", report["search_calls"])
    write(out/"RUN_REPORT.json",report)
    print(json.dumps({k:report[k] for k in ("status","api_calls","total_tokens","actual_generations")}),flush=True)
    return report


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--inputs",type=Path,help="External data root with the archive's relative paths")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--key",type=Path)
    p.add_argument("--resume",type=Path)
    p.add_argument("--inspect-only",action="store_true")
    p.add_argument("--max-generations",type=int,default=10)
    p.add_argument("--strict-local-gate",action="store_true")
    p.add_argument("--enable-evidence-search",action="store_true")
    p.add_argument("--search-budget",type=int,default=24)
    p.add_argument("--search-cache",type=Path)
    p.add_argument("--search-gap-threshold",type=float,default=0.75)
    p.add_argument("--bootstrap-size",type=int,default=8)
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--max-parents",type=int,default=8)
    p.add_argument("--elite-quota",type=int,default=5)
    p.add_argument("--native-ga-budget",type=int,default=16)
    p.add_argument("--shortlist-size",type=int,default=32)
    p.add_argument("--api-call-limit",type=int,default=100)
    p.add_argument("--sa-penalty-weight",type=float,default=2.0)
    p.add_argument("--sa-threshold",type=float,default=3.5)
    p.add_argument("--no-protect-directed-parent",action="store_true")
    p.add_argument("--root-cooldown",type=float,default=4.0)
    p.add_argument("--enable-chemical-rules",action="store_true")
    args = p.parse_args()
    run(args.archive,args.out,args.key,resume=args.resume,inspect_only=args.inspect_only,
        inputs=args.inputs,max_generations=args.max_generations,strict_local_gate=args.strict_local_gate,
        enable_evidence_search=args.enable_evidence_search,search_budget=args.search_budget,
        search_cache=args.search_cache,search_gap_threshold=args.search_gap_threshold,
        bootstrap_size=args.bootstrap_size,batch_size=args.batch_size,
        max_parents=args.max_parents,elite_quota=args.elite_quota,
        native_ga_budget=args.native_ga_budget,shortlist_size=args.shortlist_size,
        api_call_limit=args.api_call_limit,
        sa_penalty_weight=args.sa_penalty_weight,
        sa_threshold=args.sa_threshold,
        protect_directed_parent=not args.no_protect_directed_parent,
        root_cooldown=args.root_cooldown,
        enable_chemical_rules=args.enable_chemical_rules)
