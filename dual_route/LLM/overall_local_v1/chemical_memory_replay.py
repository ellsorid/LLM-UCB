"""Four-unit evidence index and leakage-free score replay."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from chemical_units import KINDS, extract, load_pair
from generation_gate import matrix_sites
from relaxed_evidence import score, similarity


ENDPOINTS = {"stability":"storage_stability", "reversibility":"cycle_reversibility"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")


def fingerprint(unit):
    labels = {i:(z,arom) for i,z,_,_,arom in unit["atom_labels"]}
    result = Counter(("atom",z,charge,h,arom) for _,z,charge,h,arom in unit["atom_labels"])
    result.update(("bond",order,*sorted((labels[i],labels[j]))) for i,j,order in unit["internal_bonds"])
    result.update(("connection",order,labels[i]) for i,_,order in unit["boundary_bonds"])
    if unit["name"] == "net_hydrogenation_region":
        result.update(("change",e["channel"],e["lean"],e["rich"]) for e in unit["paired_changes"])
    return result


def build_memory(run):
    by_molecule = {}
    for name in ("warm_start","child"):
        rec = read(run/"evaluations"/name/"evaluation/target_record.json")
        uid = rec["record_id"]
        by_molecule[uid] = extract(load_pair(rec),uid)["units"]
    index = {kind:[] for kind in KINDS}
    whole, unresolved = [], []
    for path in sorted((run/"memory_store/memory").glob("*.json")):
        for r in read(path)["records"]:
            units = by_molecule[r["case"]]
            for c in r["evidence"]["claims"]:
                row = {"source_molecule":r["case"],"reference":r["record_id"]+"/"+c["claim_id"],
                       "source_id":c["source_id"],"stream":r["agent"],"evidence":c}
                if r["agent"]=="global":
                    whole.append(row)
                    continue
                sites = set(c["site_parent_ids"])
                found = False
                for kind, choices in units.items():
                    for unit in choices:
                        if unit["state"]!=c["target_state"] or not sites or not sites <= set(unit["parent_ids"]):
                            continue
                        index[kind].append(row | {"source_unit":unit})
                        found = True
                if not found:
                    unresolved.append(row)
    return {"unit_index":index,"whole_evidence":whole,"unresolved_local_evidence":unresolved,
            "source_molecules":list(by_molecule)}


def retrieve(target, memory, endpoint, floor=.30, limit=3):
    """Same state and endpoint, then rank analogous units by structure."""
    hits = []
    for kind, units in target["units"].items():
        for row in memory["unit_index"][kind]:
            evidence = row["evidence"]
            if row["source_molecule"]==target["molecule_id"] or evidence["endpoint"]!=endpoint:
                continue
            source = row["source_unit"]
            for unit in units:
                if (unit["name"],unit["state"]) != (source["name"],source["state"]):
                    continue
                value = similarity(fingerprint(source),fingerprint(unit))
                if value < floor:
                    continue
                hits.append({"kind":kind,"unit":unit["unit_id"],"state":unit["state"],
                             "target_ids":unit["parent_ids"],"similarity":value,
                             "source_id":row["source_id"],"reference":row["reference"],
                             "stream":row["stream"],"scope_tier":evidence.get("scope_tier"),
                             "source_conditions":evidence.get("conditions", ""),
                             "source_catalyst":evidence.get("catalyst"),
                             "quote":evidence.get("quote", ""),
                             "observation":evidence.get("observation"),
                             "attribution":evidence.get("attribution", "molecular_structure")})
    unique = {}
    for row in sorted(hits,key=lambda r:(-r["similarity"],r["reference"],r["unit"])):
        unique.setdefault((row["stream"],row["kind"],row["source_id"],row["state"]),row)
    return [r for stream in ("local","coupling")
            for r in [v for v in unique.values() if v["stream"]==stream][:limit]]


def evaluate(uid, units, memory, archived, catalog):
    rows = []
    for endpoint, memory_endpoint in ENDPOINTS.items():
        hits = retrieve(units,memory,memory_endpoint)
        accepted = {"local":[], "coupling":[]}
        for h in hits:
            stream = h["stream"]
            assessment = archived["assessments"][stream]
            if stream=="coupling" and not any(
                link["endpoint"]==endpoint and link["interaction_status"]=="source_supported"
                for link in assessment["coupling_links"]
            ):
                continue
            for f in assessment["findings"]:
                direction = {"favorable":1,"unfavorable":-1}.get(f["direction"])
                refs = [catalog[r] for r in f["evidence_refs"] if r in catalog]
                if (f["endpoint"]==endpoint and direction is not None and f["site_parent_ids"]
                        and set(f["site_parent_ids"]) <= set(h["target_ids"])
                        and any(c["source_id"]==h["source_id"] and c.get("target_state")==h["state"]
                                and c["endpoint"]==memory_endpoint for c in refs)):
                    # Conditions are not controlled in the archived test. Keep these analogues provisional.
                    accepted[stream].append(h | {"direction":direction,"quality":.7,"applicability":.5})
        provisional = score([], accepted["local"], accepted["coupling"])
        rows.append({"endpoint":endpoint,"archived_llm_score":archived["score"][endpoint]["score"],
                     "retrieved":hits,"scoring_evidence":accepted,"provisional_local_score":provisional,
                     "status":"provisional_only" if provisional["score"] is not None else "insufficient_evidence"})
    return {"uid":uid,"endpoints":rows,
            "catalytic_mechanism_context":retrieve(units,memory,"catalytic_pathway")}


def score_controls():
    item = {"source_id":"test_fixture","direction":1,"quality":1.0,"applicability":1.0}
    weak, strong = item | {"similarity":.25}, item | {"similarity":.8}
    a,b = (score([], [e], [])["score"] for e in (weak,strong))
    c = score([], [strong], [strong | {"direction":-1}])["score"]
    assert a < b and c < b and score([], [], [])["score"] is None
    return {"weak_fixture":a,"strong_fixture":b,"strong_with_negative_coupling_fixture":c,
            "empty_evidence_score":None}


def replay(archive, live, out):
    memory = build_memory(archive/"runs/frontier_small_001")
    out.mkdir()
    write(out/"MEMORY_V2.json",memory)
    rounds = read(live/"family_1/frontier/rounds.json")
    targets, evaluation, comparisons = [], [], []
    for step in rounds:
        for i,item in enumerate(step["evaluation_queue"],1):
            tag = f"g{step['generation']}_slot{i}"
            path = live/"family_1/frontier/evaluations"/tag
            pair = load_pair(read(path/"target_record.json"))
            uid = item["uid"]
            assert uid not in memory["source_molecules"]
            units = extract(pair,uid)
            edit_site = next(row["site"] for row in step["admissions"] if row["uid"]==uid)
            window = {state:{r["site_parent_id"] for r in rows} for state,rows in matrix_sites(pair,edit_site).items()}
            comparisons.append({"uid":uid,"edit_site":edit_site,"units_outside_fixed_window":{
                kind:sum(not set(u["parent_ids"]) <= window[u["state"]] for u in choices)
                for kind,choices in units["units"].items()}})
            for group in units["units"].values():
                for unit in group:
                    s = int(unit["state"]=="rich")
                    for a,b,order in unit["internal_bonds"]+unit["boundary_bonds"]:
                        assert pair[s,a,b,7]==order
            targets.append(units)
            histories = [folder for folder in live.parent.glob("three_gen_recovery_*")
                         if (folder/f"evaluations/{tag}/full_score/score_result.json").exists()]
            folder = sorted(histories)[-1]
            archived = read(folder/f"evaluations/{tag}/full_score/score_result.json")
            assert archived["critic"]["accept"] and archived["score"]["molecule_id"]==uid
            requests = [read(f)["user"] for f in (folder/"protocol_requests").glob("*_score_full.json")]
            catalog = next(r["evidence_catalog"] for r in requests if r["molecule_id"]==uid)
            evaluation.append(evaluate(uid,units,memory,archived,catalog))
    write(out/"UNITS.json",targets)
    report = {"targets":len(targets),"archive_scoring_replays":evaluation,
              "fixed_window_comparison":comparisons,"score_controls":score_controls(),
              "unit_counts":{k:sum(len(r["units"][k]) for r in targets) for k in KINDS},
              "index_rows":{k:len(v) for k,v in memory["unit_index"].items()},
              "whole_background_count":len(memory["whole_evidence"]),
              "unresolved_count":len(memory["unresolved_local_evidence"]),
              "index_endpoint_counts":dict(Counter(
                  row["evidence"]["endpoint"] for row in {r["reference"]:r
                      for group in memory["unit_index"].values() for r in group}.values())),
              "api_calls":0,"score_scope":"Provisional local contribution only; no new validated overall score."}
    write(out/"REPLAY.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="archive_scoring_replays"}))
    print([(r["uid"][:8],[(p["endpoint"],len(p["retrieved"]),p["provisional_local_score"]["score"])
                           for p in r["endpoints"]]) for r in evaluation])


if __name__=="__main__":
    p = argparse.ArgumentParser()
    for option in ("archive","live","out"):
        p.add_argument(f"--{option}",type=Path,required=True)
    args = p.parse_args()
    sys.path.insert(0,str(args.archive/"workflow_runtime_20261125/runtime"))
    replay(args.archive,args.live,args.out)
