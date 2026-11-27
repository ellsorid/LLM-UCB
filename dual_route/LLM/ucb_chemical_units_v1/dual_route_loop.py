"""Run a directed-parent lane alongside the established population expansion."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


DESIGN_TASK = """Design one molecule-vision heuristic breeding parent to complement hierarchical UCB.
Inspect root_molecule and each legal candidate's SMILES structure (lean_smiles),
synthetic accessibility (sa_score, delta_sa, sa_penalty_points), H2 capacity
(h2_wt_percent, delta_h2_wt_percent), predicted whole/local score changes,
net_pred_delta_points, SOM response shift (destination_response_delta),
target_cell_kalman stats, and evidence_gap. Your chosen candidate will enter
next-generation breeding parents for UCB local refinement, so balance
cross-cell exploration (preferring cells not yet visited in dst_cell_already_directed)
and overall/local + H2 improvement against SA penalty (avoid gratuitous poly-alkyne
or high-SA chains when a synthetically cleaner edit achieves strong gain).
Choose one first edit, then optionally one of its linked continuation edits.
Return JSON exactly {"edit_ids":["first_id"],"reason":"short explanation"}
or the same object with two IDs. Use only IDs linked in the menu."""


def validate_design(answer, routes):
    if not isinstance(answer,dict) or set(answer)!={"edit_ids","reason"}:
        raise ValueError("Invalid design fields")
    ids = answer["edit_ids"]
    if (not isinstance(ids,list) or not 1<=len(ids)<=2 or
        not all(isinstance(x,str) for x in ids) or tuple(ids) not in routes or
        not isinstance(answer["reason"],str) or not answer["reason"].strip()):
        raise ValueError("Design must be an executable edit sequence")
    return routes[tuple(ids)]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")


def run(archive, out, key, legacy_runtime=None, replay=None, reference_cohort=None):
    registry = read(archive/"CHEMICAL_UNIT_WORKFLOW.json")
    som = archive/"SOM/chemical_unit_som_v1"
    llm = archive/"LLM/overall_local_v1"
    runtime = archive/"workflow_runtime_20261125/runtime"
    os.environ.update(LOHC_PROJECT_ROOT=str(runtime),LOHC_BUNDLED_ONLY="1")
    sys.path[:0] = [str(som),str(llm),str(archive/"LLM/ucb_chemical_units_v1"),
                   str(archive/"LLM/frontier_loop"),str(runtime)]

    from chemical_som import encode_pairs, encode_token
    from chemical_units import extract, load_pair
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

    class LegalEdits(StructureBridge):
        """Reuse archived validated operators; encode results only with the new SOM."""
        def __init__(self):
            pass

        def describe(self, molecules):
            return [{"uid":m.uid,"molecule":m} for m in molecules]

    torch.set_num_threads(2)
    model_path = archive/registry["som"]["checkpoint"]
    model = torch.load(model_path,map_location="cpu",weights_only=False)
    centers = model["centers"].numpy()
    bandwidth = max(float(np.median(((centers[:,None]-centers[None,:])**2).sum(2))),1e-6)
    context = registry["scoring"]["version"]+"|unspecified_conditions|SOM:"+sha256(model_path.read_bytes()).hexdigest()
    if (archive/registry["scoring"]["protocol"]).read_text(encoding="utf-8") != TASK:
        raise ValueError("Scoring protocol mismatch")
    memory = read(archive/registry["memory"]["fixture"])
    global_memory = read(archive/"runs/frontier_small_001/memory_store/memory/global.json")["records"]
    seed_record = read(llm/"artifacts/target_record.json")
    seed_meta = read(archive/registry["scoring"]["catalyst_metadata"])
    seed_score = read(archive/registry["scoring"]["short_test"])
    pair = load_pair(seed_record)
    names = tuple(Chem.MolToSmiles(to_mol(m),isomericSmiles=True) for m in pair)
    seed = Molecule(identity(names),pair,names,seed_record["record_id"])
    if seed.uid!=seed_record["record_id"] or seed.uid!=seed_meta["uid"] or seed_score["status"]!="ok":
        raise ValueError("Seed score and matrix identities differ")
    bridge = LegalEdits()
    descriptors = {}

    def encode(m, parent, site, op):
        z = encode_pairs([m.pair],model["transform"])[0].numpy().astype(float)
        d2 = ((z-centers)**2).sum(1)
        response = np.exp(-d2/(2*bandwidth))
        response /= np.linalg.norm(response)
        units = extract(m.pair,m.uid)
        tags, payload = prepare_local(units,memory)
        descriptors[m.uid] = (tags,payload)
        tokens = frozenset(encode_token((u["name"],u["state"],sorted(fingerprint(u).items(),key=repr)))
                           for group in units["units"].values() for u in group)
        covered_kinds = {t["kind"] for t in tags if t["used_for_judgement"] and t["similarity"]>=.5}
        return EditCandidate(m.uid,parent.uid,site,int(d2.argmin()),np.r_[1,z,.25*response],response,
            (family(parent.pair),local_key(parent.pair,site),op),tokens,
            1-len(covered_kinds)/len(units["units"]),
            frozenset(t["reference"] for t in tags if t["used_for_judgement"]))

    def proposed(parents, excluded):
        generated, rejected = {}, []
        for parent in parents:
            for op in ("methyl","amino","hydroxyl"):
                rows, errors = bridge.children({"uid":parent.uid,"molecule":parent},op,max_sites=3)
                rejected.extend(errors)
                for row in rows:
                    if row["uid"] in excluded or row["uid"] in generated:
                        continue
                    m = row["molecule"]
                    site = int(row["edit"]["site"])
                    generated[m.uid] = (encode(m,parent,site,op),m,parent)
        return generated,rejected

    root = encode(seed,seed,0,"seed")
    model_state = KalmanMap(len(root.psi),context)
    meaningful_ids = {r["id"] for r in seed_score["local"]["judgements"] if r["direction"] in ("favorable","unfavorable")}
    seed_refs = frozenset(r["reference"] for r in seed_meta["evidence_catalyst_labels"] if r["id"] in meaningful_ids)
    if not seed_refs:
        raise ValueError("Seed score lacks a traceable local observation")
    e = Evaluation(seed.uid,context,float(seed_score["combined"]),float(seed_score["local"]["weight"]),seed_refs,True)
    model_state.update(seed.uid,root.psi,e.score/100,.02+.18*(1-e.evidence_confidence)**2,context)
    state = PopulationState(model_state,{seed.uid:root},{seed.uid:e})
    candidates, invalid = proposed([seed],state.members)
    whole_map,local_map = (KalmanMap(len(root.psi),context) for _ in range(2))
    whole_map.update(seed.uid,root.psi,seed_score["overall_memory"]["score"]/100,.10,context)
    local_map.update(seed.uid,root.psi,seed_score["local"]["score"]/100,.15,context)
    prior = read(archive/"runs/chemical_unit_full_smoke_001/RUN_REPORT.json")
    known = set()
    for r in prior["evaluations"]:
        uid = r["uid"]
        if uid not in candidates or "combined" not in r or r["score_context"]!=context:
            continue
        h = candidates[uid][0].psi
        whole_map.update(uid,h,r["overall"]["score"]/100,.10,context)
        local_map.update(uid,h,r["local"]["score"]/100,.10+.15*(1-r["local"]["weight"]),context)
        known.add(uid)
    def describe_route(row, parent):
        dg,vg = whole_map.delta(root.psi,row.psi)
        dl,vl = local_map.delta(root.psi,row.psi)
        return {"site":row.parent_site,"operator":row.arm[-1],"som_target_cell":row.cell,
                "som_target_response_delta":float(row.response[row.cell]-root.response[row.cell]),
                "overall_delta_points":100*dg,"local_delta_points":100*dl,
                "overall_delta_sd_points":100*vg**.5,"local_delta_sd_points":100*vl**.5,
                "evidence_gap":row.evidence_gap,"step_parent_cell":parent.cell}

    routes, menu = {}, []
    eligible = [t for t in candidates.values() if t[0].uid not in known and descriptors[t[0].uid][1]]
    for i,(row,molecule,parent) in enumerate(eligible[:6],1):
        edit_id = f"A{i}"
        routes[(edit_id,)] = (row,molecule,parent)
        item = {"id":edit_id,**describe_route(row,root),"continuations":[]}
        further,_ = proposed([molecule],set(candidates)|known|{seed.uid})
        shortlist = [t for t in further.values() if descriptors[t[0].uid][1]]
        for j,(c,child,parent) in enumerate(shortlist[:2],1):
            path_id = f"{edit_id}B{j}"
            # Rebind the approved edit sequence to the evaluated root for the
            # batch gate; no single-edit UCB attribution is made for a sequence.
            rebound = EditCandidate(c.uid,seed.uid,c.parent_site,c.cell,c.psi,c.response,
                                    (c.arm[0],c.arm[1],"composite_design"),c.units,c.evidence_gap,c.evidence_refs)
            routes[(edit_id,path_id)] = (rebound,child,parent)
            item["continuations"].append({"id":path_id,**describe_route(c,row)})
        menu.append(item)
    if not routes:
        raise RuntimeError("No evidence-ready directed design")
    out.mkdir()
    report = {"status":"staged","score_context":context,"seed":seed.uid,"seed_score":e.score,
              "valid_candidate_count":len(candidates),"rejected_matrix_edits":invalid,
              "design_menu_paths":len(routes),
              "critic_calls":0,"search_calls":0,"coupling_memory_used":False,
              "population_before":len(state.members),"evaluations":[]}
    design_packet = {"goal":"Improve local performance while retaining the overall score and covering reachable SOM cells",
                     "seed_cell":root.cell,"scoring_observations":len(whole_map.observed),
                     "allowed_edit_paths":menu}
    write(out/"DESIGN_INPUT.json",design_packet)
    write(out/"RUN_REPORT.json",report)
    if key is None and replay is None:
        return report

    prior_run = read(replay/"RUN_REPORT.json") if replay else None
    if prior_run:
        if prior_run["score_context"]!=context or read(replay/"DESIGN_INPUT.json")!=design_packet:
            raise ValueError("Saved design inputs differ from this run")
        answer = prior_run["design"]["response"]
        design_audit = {"status":prior_run["design"]["api_status"],"usage":prior_run["design"]["usage"]}
        gateway = None
    else:
        gateway = OfficialTemperatureGateway(key,out/"api",limit=3)
        gateway.request_timeout = 45
        answer,design_audit = gateway.call("deepseek-flash","t05","design_parent",DESIGN_TASK,design_packet,900)
    report["design"] = {"api_status":design_audit["status"],"usage":design_audit.get("usage"),"response":answer}
    try:
        directed = validate_design(answer,routes)
    except ValueError as error:
        report.update(status="blocked_directed_design",error=str(error),api_calls=gateway.calls if gateway else 0)
        write(out/"RUN_REPORT.json",report)
        return report
    ordinary,preflight = select_batch(state,[t[0] for t in eligible if t[0].uid!=directed[0].uid],1)
    if not ordinary:
        raise RuntimeError("No independent population-expansion candidate")
    queue = [directed[0],ordinary[0][0]]
    report["selected"] = [
        {"uid":queue[0].uid,"role":"directed_parent","cell":queue[0].cell,"edit_ids":answer["edit_ids"]},
        {"uid":queue[1].uid,"role":"population_expansion","cell":queue[1].cell,**preflight[queue[1].uid]},
    ]
    candidates[queue[0].uid] = directed
    write(out/"RUN_REPORT.json",report)
    evaluations = {}
    for slot,row in enumerate(queue,1):
        tags, packet = descriptors[row.uid]
        stage = {"uid":row.uid,"score_context":context,"target_catalyst":None,
                 "evidence_catalyst_labels":tags}
        m = candidates[row.uid][1]
        overall = whole_prior(m.pair,row.uid,global_memory)
        if packet and overall["score"] is not None:
            payload = {"observations":packet}
            write(out/f"LLM_INPUT_{slot}.json",payload)
            if prior_run:
                old = prior_run["evaluations"][slot-1]
                if old["uid"]!=row.uid or read(replay/f"LLM_INPUT_{slot}.json")!=payload:
                    raise ValueError("Saved local scoring inputs differ")
                answer = {"judgements":old["local"]["judgements"]}
                audit = {"status":old["api_status"],"http_status":old["http_status"],"usage":old["usage"]}
            else:
                answer, audit = gateway.call("deepseek-flash","t05",f"local_{slot}",TASK,payload,900)
            stage.update(api_status=audit["status"],http_status=audit["http_status"],usage=audit.get("usage"))
            if audit["status"]=="ok":
                try:
                    local = local_result(packet,answer,tags)
                    stage.update(overall=overall,local=local)
                    if local["score"] is None:
                        stage["error"] = "No directional local judgement"
                    else:
                        merged = (overall["score"]*overall["weight"]+local["score"]*local["weight"])/(overall["weight"]+local["weight"])
                        ids = {j["id"] for j in local["judgements"] if j["direction"] in ("favorable","unfavorable")}
                        refs = frozenset(t["reference"] for t in tags if t["id"] in ids)
                        stage["combined"] = merged
                        evaluations[row.uid] = Evaluation(row.uid,context,merged,local["weight"],refs,True)
                except (TypeError,KeyError,ValueError) as error:
                    stage["error"] = str(error)
        else:
            stage["error"] = "Matching did not supply both input branches"
        report["evaluations"].append(stage)
        write(out/"RUN_REPORT.json",report)

    # The simplified judgement protocol does not estimate edit causality.
    # Only scores go to the property map. Leave operator payoff on its prior.
    updated, commit = commit_batch(state,queue,evaluations,{})
    report.update(api_calls=gateway.calls if gateway else 0,replayed_api_calls=prior_run["api_calls"] if prior_run else 0,
                  commit=commit,population_after=len(updated.members),
                  observed_before=len(state.mapping.observed),observed_after=len(updated.mapping.observed),
                  tokens={field:sum((ev.get("usage") or {}).get(field,0) for ev in [report["design"],*report["evaluations"]])
                          for field in ("prompt_tokens","completion_tokens","total_tokens")})
    if commit["committed"]:
        children, bad = proposed([candidates[r.uid][1] for r in queue],updated.members)
        next_plan = generation_quotas(updated,[t[0] for t in children.values()],3)
        if not next_plan["routes"]:
            report["status"] = "committed_no_next_candidates"
        else:
            report["status"] = "one_generation_committed_next_candidates_ready"
        report["next_generation"] = next_plan | {"valid_candidates":len(children),"rejected_matrix_edits":bad}
        origins = {r.uid:role["role"] for r,role in zip(queue,report["selected"])}
        report["next_generation"]["by_parent_route"] = {
            route:sum(r.parent_uid==uid for r,_,_ in children.values()) for uid,route in origins.items()
        }
        report["selected_cells"] = sorted({r.cell for r in updated.members.values()})
        report["prediction_before_after"] = [
            {"uid":r.uid,"before":float(state.mapping.predict(r.psi)[0][0]),
             "after":float(updated.mapping.predict(r.psi)[0][0])}
            for r,_,_ in children.values()
        ]
        write(out/"NEXT_CANDIDATES.json",[
            {"uid":uid,"parent_uid":r.parent_uid,"site":r.parent_site,"cell":r.cell,"arm":r.arm,
             "parent_route":origins[r.parent_uid],
             "matrix_difference":bridge.differential_packet(
                {"uid":parent.uid,"molecule":parent},
                {"uid":uid,"molecule":m,"edit":m.action})["sparse_delta"]}
            for uid,(r,m,parent) in children.items()
        ])
        updated.mapping.save(out/"UPDATED_MAPPING.npz")
        if legacy_runtime:
            from legacy_sidechain_expansion import generate as legacy_generate
            cohort = [seed,*[candidates[r.uid][1] for r in queue]]
            if reference_cohort:
                history = read(reference_cohort/"RUN_REPORT.json")
                if history["model_context"]!=context:
                    raise ValueError("Reference cohort uses a different score/map version")
                scored = history["bootstrap"]["scores"]
                pool = read(reference_cohort/"INITIAL_POOL.json")
                with np.load(runtime/"outputs/matrix_som_v2/matrix_inputs.npz",allow_pickle=False) as raw:
                    matrices,masks = raw["x"],raw["mask"]
                for r in pool:
                    if r["uid"] not in scored or any(m.uid==r["uid"] for m in cohort):
                        continue
                    i = r["source_row"]
                    pair = matrices[i][:,masks[i]][:,:,masks[i]]
                    names = tuple(Chem.MolToSmiles(to_mol(a),isomericSmiles=True) for a in pair)
                    if identity(names)!=r["uid"]:
                        raise ValueError("Reference parent misalignment")
                    cohort.append(Molecule(r["uid"],pair,names,r["uid"]))
            legacy = legacy_generate(legacy_runtime,runtime,cohort,set(updated.members)|set(children))
            native = []
            for m,metadata in legacy.pop("valid"):
                z = encode_pairs([m.pair],model["transform"])[0].numpy()
                cell = int(((z-centers)**2).sum(1).argmin())
                native.append({"uid":m.uid,"pair":m.identity,"cell":cell,**metadata,
                               "route":"population_expansion","status":"unscored_candidate"})
            write(out/"NATIVE_GA_CANDIDATES.json",native)
            report["native_ga"] = legacy | {"parent_pool_size":len(cohort),"valid_by_operation":{
                op:sum(row["operation"]==op for row in native) for op in legacy["requested"]}}
    else:
        report["status"] = "blocked_waiting_for_valid_scores"
    write(out/"RUN_REPORT.json",report)
    return report


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--key",type=Path)
    p.add_argument("--legacy-runtime",type=Path)
    p.add_argument("--replay",type=Path)
    p.add_argument("--reference-cohort",type=Path)
    args = p.parse_args()
    legacy = args.legacy_runtime or Path(__file__).parent/"legacy_ga_v1"
    result = run(args.archive,args.out,args.key,legacy if legacy.is_dir() else None,args.replay,args.reference_cohort)
    print(json.dumps({k:result[k] for k in ("status","valid_candidate_count","population_before",
          "population_after","tokens") if k in result}),flush=True)
