"""Single live probe with program-side overall Memory and catalyst metadata."""
import argparse
import math
from pathlib import Path
import sys

from chemical_memory_replay import retrieve
from chemical_units import load_pair
from judgement_only_probe import load, save
from relaxed_evidence import match, score


# Audited summaries of the four stored abstracts for this one-molecule pilot.
# A None entry means the intervention changed the catalyst, not the molecule.
OBSERVATIONS = {
    "FRESH_AMENDE_2014_ABSTRACT": (
        "Pt(111)", "The source observed initial C–H activation near the ring nitrogen during dehydrogenation."),
    "FRESH_GLEICHWEIT_2014_ABSTRACT": (
        "Pt(111)", "After stepwise dehydrogenation, the source observed cleavage of the N–alkyl C–N bond."),
    "P3140ddbb70ff": ("Na-modified Pt/Al2O3", None),
    "SRC_EDGAR": (
        "Pd/Al2O3", "In a different hydrogen carrier the pyrrolidine ring dehydrogenated first."),
}

TASK = """For each given local observation, judge the direction of its possible
effect on overall carrier performance. Useful conversion can be favorable,
carrier degradation unfavorable, and a pathway by itself can be unknown.
These are related compounds, not measurements of the target. Do not attribute
an observation to the edit merely because it belongs to the same molecule.
Return JSON exactly {"judgements":[{"id":"input ID",
"direction":"favorable|unfavorable|mixed|unknown","reason":"one short sentence"}]}.
Use each given ID once. Do not assign numeric scores or perform retrieval."""


def whole_prior(pair, uid, memory):
    candidates = []
    for r in memory:
        e = r.get("evaluation",{})
        prior = e.get("score",{})
        if r["case"]==uid or not e.get("critic",{}).get("accept") or e.get("source")!="LIVE_LLM":
            continue
        val = prior.get("overall",{}).get("score")
        if prior.get("molecule_id")!=r["case"] or type(val) not in (int,float) or not math.isfinite(val):
            continue
        ref = {"full":r["input"]["structure"]["whole_states"]}
        similarity,_,_ = match(pair,load_pair(ref),None,[])
        if similarity < .3:
            continue
        candidates.append({"molecule":r["case"],"stored_score":val,"similarity":similarity,
                           "weight":similarity**2,"score_context":e["score_context"]})
    total = sum(r["weight"] for r in candidates)
    return {"matches":candidates,"score":sum(r["weight"]*r["stored_score"] for r in candidates)/total
            if total else None,"weight":min(1,total)}


def prepare_local(target, memory):
    rows, audit = {}, []
    endpoints = {r["evidence"]["endpoint"] for group in memory["unit_index"].values() for r in group}
    for endpoint in sorted(endpoints):
        for hit in retrieve(target,memory,endpoint,limit=10):
            if hit["stream"] != "local" or hit["source_id"] in rows:
                continue
            source = hit["source_id"]
            if source not in OBSERVATIONS:
                continue
            rows[source] = hit
            catalyst, observation = OBSERVATIONS[source]
            audit.append(hit | {"id":f"E{len(audit)+1}","source_catalyst":catalyst,
                                "molecular_observation":observation,
                                "used_for_judgement":observation is not None})
    return audit, [{"id":r["id"],"unit_type":r["kind"],"parent_ids":r["target_ids"],"state":r["state"],
                    "observation":r["molecular_observation"]} for r in audit if r["used_for_judgement"]]


def local_result(packet, result, audit):
    candidates = {r["id"]:r for r in packet}
    rows = {r["id"]:r for r in audit}
    if not isinstance(result,dict) or set(result)!={"judgements"} or not isinstance(result["judgements"],list):
        raise ValueError("Invalid judgement envelope")
    allowed = {"favorable":1,"unfavorable":-1,"mixed":0,"unknown":0}
    seen = set()
    items = []
    for r in result["judgements"]:
        if (not isinstance(r,dict) or set(r)!={"id","direction","reason"} or
                r["id"] not in candidates or r["id"] in seen or r["direction"] not in allowed or
                not isinstance(r["reason"],str) or not r["reason"].strip()):
            raise ValueError("Invalid judgement row")
        seen.add(r["id"])
        if allowed[r["direction"]]:
            hit = rows[r["id"]]
            items.append({"source_id":hit["source_id"],"similarity":hit["similarity"],
                          "direction":allowed[r["direction"]],"quality":.7,"applicability":.5})
    if seen != set(candidates):
        raise ValueError("Missing program-selected evidence")
    local = score([],items,[])
    return {"score":local["score"],"weight":local["coverage"]["local"],"judgements":result["judgements"]}


if __name__=="__main__":
    p = argparse.ArgumentParser()
    for name in ("archive","memory","targets","out","key"):
        p.add_argument(f"--{name}",required=True,type=Path)
    args = p.parse_args()
    sys.path.insert(0,str(args.archive/"workflow_runtime_20261125/runtime"))
    from matrix_pipeline.temperature_experiment import OfficialTemperatureGateway

    target = load(args.memory/"UNITS.json")[0]
    pair = load_pair(load(args.targets))
    uid = target["molecule_id"]
    if load(args.targets)["record_id"]!=uid:
        raise ValueError("Target unit and matrix identities differ")
    origin = args.archive/"runs/frontier_small_001/memory_store/memory/global.json"
    overall = whole_prior(pair,uid,load(origin)["records"])
    tags, packet = prepare_local(target,load(args.memory/"MEMORY_V2.json"))
    if not packet:
        raise ValueError("No matched molecular observations")
    args.out.mkdir()
    save(args.out/"LLM_INPUT.json",packet)
    save(args.out/"EVALUATION_METADATA.json",{"uid":uid,"target_catalyst":None,
         "evidence_catalyst_labels":tags,"coupling_used":False})
    api = OfficialTemperatureGateway(args.key,args.out/"api",limit=1)
    result, audit = api.call("deepseek-flash","t05","local_effect",TASK,{"observations":packet},900)
    report = {"overall_memory":overall,"usage":audit.get("usage"),"api_calls":api.calls,"status":audit["status"],
              "critic_calls":0,"search_calls":0,"coupling_used":False,"observations_sent":len(packet),
              "excluded_interventions":sum(not r["used_for_judgement"] for r in tags)}
    if audit["status"]=="ok":
        try:
            local = local_result(packet,result,tags)
            parts = [x for x in (overall,local) if x["score"] is not None and x["weight"]>0]
            report |= {"local":local,"combined":sum(x["score"]*x["weight"] for x in parts)/
                       sum(x["weight"] for x in parts) if parts else None, "scope":"conditional pilot"}
        except (ValueError,KeyError,TypeError) as error:
            report["validation_error"] = str(error)
    save(args.out/"RESULT.json",report)
    print(report,flush=True)
