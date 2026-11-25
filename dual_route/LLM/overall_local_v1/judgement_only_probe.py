"""One live judgement-only request with programmatic Memory and scoring."""
import argparse
import json
from pathlib import Path
import sys

from chemical_memory_replay import retrieve
from relaxed_evidence import score


TASK = """Assess conditional net effects on the current hydrogen carrier. Use
only supplied candidate references and target facts. A finding can represent
a catalytic mechanism, material loss, a side reaction, or storage behavior.
Assessment of an existing structural unit differs from the incremental effect
of an edit elsewhere. Set edit_direction to unknown unless the evidence
explicitly supports the edit comparison; a related carrier is an analogue.
Do not assume the operating conditions match. Give direction and reasoning
for each supplied ID; do not assign numbers, perform retrieval or review
your own answer. Attribute molecular_structure only when the evidence identifies
an effect of the target structural unit or its analogous chemistry. Use
external_intervention when the main driver is a different catalyst, additive
or operating intervention, even if the affected site is in the target unit.
Such evidence can remain contextual. Return JSON exactly {"judgements":[{"id":"supplied ID",
"direction":"favorable|unfavorable|mixed|unknown",
"edit_direction":"favorable|unfavorable|mixed|unknown",
"directness":"measured_target|experimental_analogue|mechanistic_inference|unknown",
"attribution":"molecular_structure|external_intervention|unclear",
"reason":"observed effect -> possible whole-carrier benefit or cost",
"limits":"condition or structural uncertainty"}]}."""
LABELS = {"favorable":1, "unfavorable":-1, "mixed":0, "unknown":0}
DIRECTNESS = {"experimental_analogue":.7,"mechanistic_inference":.45,"unknown":0}
ATTRIBUTIONS = {"molecular_structure","external_intervention","unclear"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def request_packet(target, memory, edit_site):
    matches = {}
    endpoints = {r["evidence"]["endpoint"] for rows in memory["unit_index"].values() for r in rows}
    for endpoint in sorted(endpoints):
        for hit in retrieve(target,memory,endpoint,limit=8):
            # Coupling needs independent joint support. This probe evaluates local claims.
            if hit["stream"] != "local":
                continue
            key = (hit["source_id"], hit["quote"], hit["state"])
            matches.setdefault(key,hit | {"evidence_endpoint":endpoint})
    units = {u["unit_id"]:u for rows in target["units"].values() for u in rows}
    candidates = []
    for i, hit in enumerate(matches.values(), 1):
        unit = units[hit["unit"]]
        candidates.append(hit | {"id":f"E{i}","unit_name":unit["name"],
            "atom_labels":unit["atom_labels"], "anchor_ids":unit["anchor_ids"],
            "unit_changed_by_edit":edit_site in hit["target_ids"]})
    return {"target":target["molecule_id"], "edit_parent_site":edit_site,
            "operating_conditions":"unspecified","candidates":candidates}


def apply(packet, response):
    if not isinstance(response,dict) or set(response)!={"judgements"} or not isinstance(response["judgements"],list):
        raise ValueError("Invalid judgement envelope")
    hits = {c["id"]:c for c in packet["candidates"]}
    accepted = []
    seen = set()
    for row in response["judgements"]:
        required = {"id","direction","edit_direction","directness","attribution","reason","limits"}
        if (not isinstance(row,dict) or set(row)!=required or row["id"] not in hits or row["id"] in seen
            or row["direction"] not in LABELS or row["edit_direction"] not in LABELS
            or row["directness"] not in DIRECTNESS
            or row["attribution"] not in ATTRIBUTIONS
            or not all(isinstance(row[key],str) and row[key].strip() for key in ("reason","limits"))):
            raise ValueError("Invalid judgement row or direct-measurement claim from analogue data")
        hit = hits[row["id"]]
        seen.add(row["id"])
        if (not hit["unit_changed_by_edit"] or row["attribution"]!="molecular_structure") and row["edit_direction"] != "unknown":
            raise ValueError("Edit effect requires structural attribution and a matching edited unit")
        accepted.append(row | {"source_id":hit["source_id"],"reference":hit["reference"],
            "similarity":hit["similarity"],"quality":DIRECTNESS[row["directness"]],"applicability":.5})
    if seen != set(hits):
        raise ValueError("Every program-selected evidence row needs a judgement")
    def weighted(field):
        return [row | {"direction":LABELS[row[field]]} for row in accepted
                if row[field] in ("favorable","unfavorable") and row["quality"] > 0
                and row["attribution"]=="molecular_structure"]
    return {"judgements":accepted,
            "provisional_local_efficiency":score([],weighted("direction"),[]),
            "provisional_edit_effect":score([],weighted("edit_direction"),[])}


if __name__=="__main__":
    p = argparse.ArgumentParser()
    for name in ("archive","memory","out","key"):
        p.add_argument(f"--{name}",required=True,type=Path)
    args = p.parse_args()
    sys.path.insert(0,str(args.archive/"workflow_runtime_20261125/runtime"))
    from matrix_pipeline.temperature_experiment import OfficialTemperatureGateway

    target = load(args.memory/"UNITS.json")[0]
    replay = load(args.memory/"REPLAY.json")
    edit_site = replay["fixed_window_comparison"][0]["edit_site"]
    packet = request_packet(target,load(args.memory/"MEMORY_V2.json"),edit_site)
    if not packet["candidates"]:
        raise SystemExit("No evidence was retrieved; no API call made")
    args.out.mkdir()
    save(args.out/"INPUT.json",packet)
    gateway = OfficialTemperatureGateway(args.key,args.out/"api",limit=1)
    gateway.request_timeout = 45
    answer,audit = gateway.call("deepseek-flash","t05","interpret_matched_evidence",TASK,packet,1400)
    report = {"api_calls":gateway.calls,"status":audit["status"],"usage":audit.get("usage"),
              "candidate_count":len(packet["candidates"]),"critic_calls":0,"search_calls":0}
    if audit["status"]=="ok":
        try:
            report.update(apply(packet,answer))
        except (ValueError,TypeError,KeyError) as error:
            report["validation_error"] = str(error)
    save(args.out/"RESULT.json",report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
