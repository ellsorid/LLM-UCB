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

import numpy as np
import torch
from furan_resume_evidence import scoped_furan_analogue


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def run(archive, out, key, retry_budget=10, resume=None, inspect_only=False, inputs=None):
    if inspect_only and key:
        raise ValueError("Offline inspection does not accept API credentials")
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
    required = (model_path, memory_path, global_memory_path, assignments_path,
                matrices_path, immigrant_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required workflow files are missing:\n" + "\n".join(missing))
    os.environ.update(LOHC_PROJECT_ROOT=str(runtime),LOHC_BUNDLED_ONLY="1")
    sys.path[:0] = [str(som),str(archive/"LLM/overall_local_v1"),str(archive/"LLM/ucb_chemical_units_v1"),
                    str(archive/"LLM/frontier_loop"),str(runtime)]
    from chemical_som import encode_pairs, encode_token
    from chemical_units import extract
    from chemical_memory_replay import fingerprint
    from frontier import family, local_key
    from mapping import KalmanMap
    from matrix_pipeline.core import Molecule, identity, to_mol
    from matrix_pipeline.temperature_experiment import OfficialTemperatureGateway
    from overall_local_probe import TASK, local_result, prepare_local, whole_prior
    from property_feedback import (EditCandidate, Evaluation, PopulationState,
                                   commit_batch, generation_quotas, select_batch)
    from rdkit import Chem
    from structure import StructureBridge
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
    context = registry["scoring"]["version"]+"|unspecified_conditions|SOM:"+sha256(model_path.read_bytes()).hexdigest()
    if (archive/registry["scoring"]["protocol"]).read_text(encoding="utf-8")!=TASK:
        raise ValueError("Scoring protocol mismatch")
    memory = read(memory_path)
    global_memory = read(global_memory_path)["records"]
    bridge = LegalEdits()

    def encode(m, parent, site, operator):
        z = encode_pairs([m.pair],model["transform"])[0].numpy().astype(float)
        d2 = ((centers-z)**2).sum(1)
        response = np.exp(-d2/(2*bandwidth))
        response /= np.linalg.norm(response)
        units = extract(m.pair,m.uid)
        tags, payload = prepare_local(units,memory)
        whole = whole_prior(m.pair,m.uid,global_memory)
        tokens = frozenset(encode_token((u["name"],u["state"],sorted(fingerprint(u).items(),key=repr)))
                           for group in units["units"].values() for u in group)
        kinds = {t["kind"] for t in tags if t["used_for_judgement"] and t["similarity"]>=.5}
        candidate = EditCandidate(m.uid,parent.uid,site,int(d2.argmin()),np.r_[1.,z,.25*response],response,
            (family(parent.pair),local_key(parent.pair,site),operator),tokens,
            1-len(kinds)/len(units["units"]),frozenset(t["reference"] for t in tags if t["used_for_judgement"]))
        return {"candidate":candidate,"molecule":m,"tags":tags,"packet":payload,"whole":whole}

    def ready(r):
        return bool(r["packet"]) and r["whole"]["score"] is not None

    def stratified_pool():
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
        selected, covered = [], set()
        remaining = [r for r in pool if ready(r)]
        while remaining and len(selected)<8:
            def order(r):
                c = r["candidate"]
                references = c.evidence_refs
                # Prioritize distinct SOM cells, then unexplored chemical units.
                return (c.cell not in {x["candidate"].cell for x in selected},
                        len(c.units-covered),len(references),r["whole"]["score"],c.uid)
            row = max(remaining,key=order)
            selected.append(row)
            covered.update(row["candidate"].units)
            remaining = [r for r in remaining if r is not row]
        return selected

    def propose(parents, excluded):
        generated, invalid = {},[]
        for parent in parents:
            for op in ("methyl","amino","hydroxyl"):
                rows, errors = bridge.children({"uid":parent.uid,"molecule":parent},op,max_sites=3)
                invalid.extend(errors)
                for r in rows:
                    uid = r["uid"]
                    if uid in excluded or uid in generated:
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
    report = {"status":"coverage_checked","model_context":context,
              "initial_pool":{"size":len(pool),"cells":len({r["cell"] for r in initial}),
              "score_ready":len(ready_pool),"score_ready_cells":len({r["candidate"].cell for r in ready_pool})},
              "bootstrap_candidates":[r["candidate"].uid for r in seeds],"rounds":[],"critic_calls":0,
              "search_calls":0,"coupling_memory_used":False}
    write(out/"RUN_REPORT.json",report)
    print(json.dumps({"initial_pool":report["initial_pool"],"bootstrap_selected":len(seeds)}),flush=True)
    if key is None and resume is None:
        return report
    if len(seeds)<8:
        report["status"] = "blocked_insufficient_bootstrap_coverage"
        write(out/"RUN_REPORT.json",report)
        return report

    prior = read(resume/"RUN_REPORT.json") if resume else None
    if prior and (prior["model_context"]!=context or prior["bootstrap_candidates"]!=report["bootstrap_candidates"]
                  or read(resume/"INITIAL_POOL.json")!=initial):
        raise ValueError("The checkpoint and reconstructed pool differ")
    continuing = bool(prior and prior.get("max_generations")==10)
    gateway = OfficialTemperatureGateway(key,out/"api",limit=100-(prior["api_calls"] if continuing else 0)) if key else None
    if gateway:
        gateway.request_timeout = 45
    token_fields = ("prompt_tokens","completion_tokens","total_tokens","prompt_cache_hit_tokens")
    retries = int(prior.get("retries",0)) if continuing else 0
    prior_api = prior["api_calls"] if continuing else 0
    design_usages = []
    report["max_generations"] = 10
    report["call_limit"] = 100

    def score_candidate(row, stage, suffix="first"):
        c = row["candidate"]
        payload = {"observations":row["packet"]}
        answer,audit = gateway.call("deepseek-flash","t05",f"{stage}_{c.uid[:12]}_{suffix}",TASK,payload,900)
        result = {"uid":c.uid,"cell":c.cell,"attempt":suffix,"api_status":audit["status"],
                  "http_status":audit["http_status"],"usage":audit.get("usage")}
        if audit["status"]=="ok":
            try:
                local = local_result(row["packet"],answer,row["tags"])
                result["local"] = local
                if local["score"] is None:
                    result["error"] = "No directional local judgement"
                    return None,result
                overall = row["whole"]
                combined = (overall["score"]*overall["weight"]+local["score"]*local["weight"])/(overall["weight"]+local["weight"])
                ids = {j["id"] for j in local["judgements"] if j["direction"] in ("favorable","unfavorable")}
                refs = frozenset(t["reference"] for t in row["tags"] if t["id"] in ids)
                result.update(combined=combined,overall=overall["score"],evidence_refs=sorted(refs))
                return Evaluation(c.uid,context,combined,local["weight"],refs,True),result
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
        inputs = [
            {"uid":r["candidate"].uid,"payload":r["packet"],"catalyst_metadata":r["tags"],
             "target_catalyst":None,"overall_memory":r["whole"]} for r in queue]
        write(folder/"INPUTS.json",inputs)
        missing = queue
        if resume and (stage=="BOOTSTRAP" or continuing) and (resume/stage/"RESULTS.json").exists():
            if read(resume/stage/"INPUTS.json")!=inputs:
                raise ValueError(f"Saved inputs differ for {stage}")
            logs = read(resume/stage/"RESULTS.json")
            by_uid = {r["candidate"].uid:r for r in queue}
            for entry in logs:
                if "combined" not in entry:
                    continue
                uid = entry["uid"]
                row = by_uid[uid]
                if entry["attempt"]=="evidence":
                    row = supplemental(row)
                    if row is None:
                        raise ValueError(f"Cannot reconstruct supplemental evidence for {uid}")
                local = local_result(row["packet"],{"judgements":entry["local"]["judgements"]},row["tags"])
                overall = row["whole"]
                total = (overall["score"]*overall["weight"]+local["score"]*local["weight"])/(overall["weight"]+local["weight"])
                if not np.isclose(total,entry["combined"],rtol=0,atol=1e-9):
                    raise ValueError(f"Saved score differs for {uid}")
                evaluations[uid] = Evaluation(uid,context,total,local["weight"],frozenset(entry["evidence_refs"]),True)
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
                elif log["api_status"]!="ok" and log["http_status"] not in (401,402,403):
                    retry.append(row)
                write(folder/"RESULTS.json",logs)
            permitted = retry[:max(0,min(retry_budget-retries,gateway.limit-gateway.calls))] if not gateway.quota_blocked else []
            retries += len(permitted)
            for handle in [workers.submit(score_candidate,r,stage,"retry") for r in permitted]:
                e,log = handle.result()
                logs.append(log)
                if e is not None:
                    evaluations[e.uid] = e
                write(folder/"RESULTS.json",logs)
            if supplements and not gateway.quota_blocked:
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
        "scores":{uid:e.score for uid,e in init_scores.items()}}
    if len(init_scores)!=len(seeds):
        report.update(status="blocked_waiting_for_bootstrap_scores",api_calls=(gateway.calls if gateway else 0))
        write(out/"RUN_REPORT.json",report)
        return report
    features = {r["candidate"].uid:r["candidate"] for r in seeds}
    mapping = KalmanMap(len(seeds[0]["candidate"].psi),context)
    for uid,e in init_scores.items():
        mapping.update(uid,features[uid].psi,e.score/100,.02+.18*(1-e.evidence_confidence)**2,context)
    state = PopulationState(mapping,features,init_scores)
    excluded = {r["candidate"].uid for r in pool}
    parents = [r["molecule"] for r in seeds]
    molecules = {m.uid:m for m in parents}
    whole_map,local_map = (KalmanMap(len(mapping.mean),context) for _ in range(2))
    population_policy = PopulationUCB()
    context_scores = {}

    def update_axes(rows,stage):
        latest = {r["uid"]:r for r in read(out/stage/"RESULTS.json") if "combined" in r}
        inputs = {r["uid"]:r for r in read(out/stage/"INPUTS.json")}
        for row in rows:
            x = latest[row.uid]
            lw,gw = x["local"]["weight"],inputs[row.uid]["overall_memory"]["weight"]
            context_scores[row.uid]=(x["overall"],x["local"]["score"],lw/(lw+gw))
            whole_map.update(row.uid,row.psi,x["overall"]/100,.10,context)
            local_map.update(row.uid,row.psi,x["local"]["score"]/100,.10+.15*(1-lw),context)

    update_axes(list(features.values()),"BOOTSTRAP")

    def design_menu(pool, root_uid, remaining):
        root = state.members[root_uid]
        def details(r, step):
            c = r["candidate"]
            overall,ov = whole_map.delta(root.psi,c.psi)
            local,lv = local_map.delta(root.psi,c.psi)
            return {"site":c.parent_site,"operator":c.arm[-1],"target_cell":c.cell,
                "destination_response_delta":round(float(c.response[c.cell]-root.response[c.cell]),6),
                "whole_delta_points":round(overall*100,3),"local_delta_points":round(local*100,3),
                "whole_sd_points":round(ov**.5*100,2),"local_sd_points":round(lv**.5*100,2),
                "evidence_gap":c.evidence_gap,"step":step}
        options = sorted((r for r in pool if r["candidate"].parent_uid==root_uid and ready(r)),
                         key=lambda r:(-local_map.delta(root.psi,r["candidate"].psi)[0],r["candidate"].uid))
        paths,payload = {},[]
        for i,row in enumerate(options[:4],1):
            edit_id = f"A{i}"
            paths[(edit_id,)]=row
            item = {"id":edit_id,**details(row,1),"continuations":[]}
            if i<=2:
                more,_ = propose([row["molecule"]],remaining)
                options2 = sorted((v for v in more.values() if ready(v)),
                                  key=lambda r:(-local_map.delta(root.psi,r["candidate"].psi)[0],r["candidate"].uid))
                for j,r in enumerate(options2[:2],1):
                    child = r["candidate"]
                    branch = f"{edit_id}B{j}"
                    proxy = EditCandidate(child.uid,root_uid,child.parent_site,child.cell,child.psi,
                        child.response,(child.arm[0],child.arm[1],"composite_design"),child.units,
                        child.evidence_gap,child.evidence_refs)
                    paths[(edit_id,branch)] = r|{"candidate":proxy}
                    item["continuations"].append({"id":branch,**details(r,2)})
            payload.append(item)
        return paths,{"goal":"Improve local performance while retaining overall performance and exploring reachable SOM cells",
                      "seed_cell":root.cell,"scoring_observations":len(whole_map.observed),"allowed_edit_paths":payload}

    write(out/"RUN_REPORT.json",report)
    stop = "ten_generations_committed"
    for generation in range(1,11):
        stage = f"G{generation}"
        known = excluded | state.members.keys()
        candidates,invalid = propose(parents,known)
        matrix_candidates = list(candidates.values())
        if not matrix_candidates:
            stop = "blocked_no_legal_matrix_edits"
            break
        # The dual-route control layer reads population-level feedback and
        # uses the documented GA operators with bounded policy quotas.
        gap = float(np.mean([r["candidate"].evidence_gap for r in matrix_candidates]))
        policy,policy_audit = population_policy.choose({"whole":gap,"local":gap})
        ga = native_generate(legacy_root,input_runtime,
                             parents,known|candidates.keys(),budget=16,policy=policy,generation=generation)
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
        root_uid = max(set(root_options),key=lambda uid:(state.evaluations[uid].score,uid),default=None)
        paths,design_packet = design_menu(matrix_candidates,root_uid,known|candidates.keys()) if root_uid else ({},{})
        step = {"generation":generation,"candidate_count":len(candidates),
                "matrix_candidates":len(matrix_candidates),"native_ga":{
                    **ga,"valid_by_operation":dict(routes_by_operation),"rejected":len(rejected),**policy_audit},
                "rejected_matrix_edits":len(invalid),"population_before":len(state.members),
                "design_paths":len(paths),"design_root":root_uid}
        report["rounds"].append(step)
        if not paths:
            stop = "blocked_no_evidence_ready_directed_path"
            break
        write(out/f"{stage}_DESIGN_INPUT.json",design_packet)
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
        step["directed_edit_ids"] = answer["edit_ids"]
        step["directed_reason"] = answer["reason"]
        candidates[lead.uid] = directed
        eligible = [r["candidate"] for r in candidates.values() if ready(r)]
        ordinary = [c for c in eligible if c.uid!=lead.uid and c.uid not in state.members]
        plan = generation_quotas(state,ordinary,min(32,len(ordinary)))
        shortlist = [candidates[r["uid"]]["candidate"] for r in plan["routes"]]
        native_eligible = [r for r in ordinary if candidates[r.uid].get("native")]
        native_pick,_ = select_batch(state,native_eligible,1)
        reserved = {r.uid for r,_ in native_pick}
        picked,audit = select_batch(state,[r for r in shortlist if r.uid not in reserved],7-len(native_pick))
        ordinary_selection = native_pick+picked
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
        previous = state
        state, gate = commit_batch(previous,queue,observations,{})
        step.update(scored=len(observations),scores={uid:e.score for uid,e in observations.items()},
                    tokens=tokens,commit=gate,population_after=len(state.members),
                    cells=len({r.cell for r in state.members.values()}),
                    new_cells=len({r.cell for r in state.members.values()}-{r.cell for r in previous.members.values()}),
                    new_local_signatures=len(set().union(*(r.units for r in state.members.values()))-
                                               set().union(*(r.units for r in previous.members.values()))))
        write(out/"RUN_REPORT.json",report)
        if not gate["committed"]:
            stop = "blocked_waiting_for_valid_scores"
            break
        update_axes(queue,stage)
        cohort_delta = [score_delta(context_scores[r.parent_uid],context_scores[r.uid])
                        for r in queue if not candidates[r.uid].get("native") or candidates[r.uid]["native"]["parent_uid"]]
        if cohort_delta:
            step["coarse_feedback"] = population_policy.observe(
                stage,cohort_delta,step["new_cells"]/len(queue),
                step["new_local_signatures"]/(step["new_local_signatures"]+len(queue)))
        # Avoid attributing the mixed matrix/native population's payoff to a
        # single legacy policy without paired scores across its offspring.
        step["native_policy_reward_updated"] = False
        state.mapping.save(out/f"{stage}_MAPPING.npz")
        parents = [candidates[r.uid]["molecule"] for r in queue]
        molecules.update({m.uid:m for m in parents})
        write(out/f"{stage}_PARENTS.json",[{"uid":m.uid,"states":m.identity,"parent":m.parent,
                                         "edit":m.action} for m in parents])
        write(out/"RUN_REPORT.json",report)
        if generation==10:
            downstream,bad = propose(parents,excluded | state.members.keys())
            report["following_candidates"] = {"valid":len(downstream),"invalid":len(bad)}
    batches = ([] if report["bootstrap"]["reused"] else [report["bootstrap"]])+report["rounds"]
    report.update(status=stop,api_calls=(gateway.calls if gateway else 0)+prior_api,retries=retries,total_tokens={
        k:sum(b.get("tokens",{}).get(k,0) for b in batches)+sum(v.get(k,0) for v in design_usages) for k in token_fields},
        final_population=len(state.members),actual_generations=sum(r.get("commit",{}).get("committed",False)
                                                                   for r in report["rounds"]))
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
    args = p.parse_args()
    run(args.archive,args.out,args.key,resume=args.resume,inspect_only=args.inspect_only,inputs=args.inputs)
