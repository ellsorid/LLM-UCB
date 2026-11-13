"""Run bounded DeepSeek agents against the Cell 17 prototype and native GA pilot."""

import argparse
import hashlib
import http.client
import itertools
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
PROTO = HERE.parent / "LOHC_QD_Memory_v1"
sys.path.insert(0, str(PROTO))
import build_demo as demo
import europe_pmc

KEY = re.compile(r"(?<![\w-])sk-[A-Za-z0-9_-]{16,}")
ROUTE_DOI = "10.1002/cssc.202000485"
MOTIFS = {
    "free_phenol": "[c]-[O;H1]",
    "aryl_alkyl_ether": "[c]-[O;X2]-[C;X4]",
    "aryl_aldehyde": "[c]-[CX3H1](=O)",
    "nitrile": "[C]#[N]",
}


def credential(path):
    if path is None:
        return os.environ["LOHC_LLM_API_KEY"]
    data = path.read_bytes()
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    lines = data.decode(encoding).splitlines()
    matches = set(KEY.findall("\n".join(s for s in lines if "deepseek" in s.lower())))
    matches = matches or set(KEY.findall("\n".join(lines)))
    if len(matches) != 1:
        raise ValueError("Expected one unambiguous DeepSeek credential")
    return matches.pop()


class DeepSeek:
    def __init__(self, key, output):
        self.key = key
        self.output = output

    def run(self, role, instruction, packet):
        payload = {
            "model": "deepseek-flash", "thinking": {"type": "disabled"},
            "temperature": 0, "max_tokens": 1100, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": instruction},
                         {"role": "user", "content": demo.json_text(packet)}],
        }
        data = demo.json_text(payload).encode("utf-8")
        conn = http.client.HTTPSConnection("api.deepseek.com", timeout=65)
        start = time.perf_counter()
        status, result = None, {}
        try:
            conn.request("POST", "/chat/completions", body=data,
                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.key})
            response = conn.getresponse()
            status = response.status
            if status != 200:
                raise RuntimeError(f"DeepSeek HTTP {status} in {role}")
            result = json.loads(response.read().decode("utf-8").replace(self.key, "[redacted]"))
        finally:
            conn.close()
            audit = {"role": role, "http_status": status, "elapsed_seconds": round(time.perf_counter()-start, 3),
                     "request_sha256": hashlib.sha256(data).hexdigest(), "model": result.get("model"),
                     "response_id": result.get("id"), "usage": result.get("usage")}
            demo.write_json(self.output / f"{role}_audit.json", audit)
        choice = result["choices"][0]
        content = KEY.sub("[redacted]", choice["message"].get("content") or "")
        (self.output / f"{role}_response.txt").write_text(content, encoding="utf-8")
        if choice.get("finish_reason") == "length" or not content.strip():
            raise ValueError(f"Empty or truncated response in {role}")
        return json.loads(content)


def literature_candidates(gaps):
    output, seen = [], {}
    for gap in gaps:
        query = gap["query"]
        for item in europe_pmc.search(query)["records"]:
            record = europe_pmc.normalize(item)
            key = europe_pmc.identity(record)
            if key is not None and key in seen:
                existing = seen[key]
                if query not in existing["queries"]:
                    existing["queries"].append(query)
                for field, value in record.items():
                    if not existing[field] and value:
                        existing[field] = value
                continue
            record.update(query=query, queries=[query])
            output.append(record)
            if key is not None:
                seen[key] = record
    return output


def evidence_agent(llm, output):
    # Triggers come from the real archive: phenol has no verified route in the seed memory.
    gaps = [{"missing": "biomass route for parent candidate phenol",
             "query": 'TITLE:"Efficient Conversion of Pine Wood Lignin to Phenol"'},
            {"missing": "additional LOHC release evidence",
             "query": 'TITLE_ABS:("liquid organic hydrogen carrier" AND "methylindole" AND dehydrogenation)'}]
    papers = literature_candidates(gaps)
    demo.write_json(output / "discovered_papers.json", papers)
    route = next((item for item in papers if item["doi"] == ROUTE_DOI and item["abstract"]), None)
    if route is None:
        raise ValueError("The reviewed phenol route requires an abstract before evidence extraction")
    packet = {"target": {"name": "phenol", "smiles": "Oc1ccccc1"}, "paper": route}
    proposal = llm.run(
        "evidence",
        '提取苯酚的生物质原料路线证据。仅返回 JSON，示例 {"doi":"10.xxx","source_quote":"原文逐字短句",'
        '"feedstock":"原料","target":"phenol","status":"route_reported"}。只引用给定摘要，'
        '原文短句须同时包含松木或木质素和苯酚；样品身份另行核验。',
        packet,
    )
    quote = " ".join(proposal.get("source_quote", "").split())
    valid = (proposal.get("doi", "").lower() == ROUTE_DOI and quote in route["abstract"]
             and len(quote.split()) <= 25 and "phenol" in quote.lower()
             and ("lignin" in quote.lower() or "biomass" in quote.lower())
             and proposal.get("status") == "route_reported" and proposal.get("target") == "phenol")
    # Admit the exact fact independently verified in this primary paper.
    # Never make the sample claim or promote unreviewed discovery records.
    record = {
        "id": "PHENOL-PINE-ROUTE", "family": "biomass_route", "name": "Phenol from pinewood lignin",
        "lean_smiles": demo.canonical("Oc1ccccc1"), "rich_smiles": None, "rich_identity": "unknown",
        "lean_scaffold": demo.scaffold("Oc1ccccc1"), "direction": "biomass_production",
        "evidence_type": "published_experimental_route", "conditions": {"feedstock": "pinewood lignin"},
        "observations": {}, "finding": "原论文报告由松木木质素制得苯酚。该路线不提供 GA 样品的实际原料或含量。",
        "doi": ROUTE_DOI, "source_url": route["source_url"], "source_anchor": "Abstract",
    }
    demo.write_json(output / "evidence_intake.json", {
        "papers_found": len(papers), "agent_proposal_passed": valid,
        "accepted_by": "source_quote_gate_and_reviewed_primary_fact" if valid else "reviewed_primary_fact",
        "accepted": record, "other_discoveries": "pending_review",
    })
    return record


def provenance_agent(llm, examples, route, output):
    source = {key: route[key] for key in ("id", "doi", "lean_smiles", "finding", "evidence_type")}
    cases = [{"id": item["candidate_id"], "smiles": item["canonical_smi"], "platform": item["platform_id"],
              "exact_route_structure": item["canonical_smi"] == route["lean_smiles"], "sample_evidence": None}
             for item in examples]
    raw = llm.run(
        "provenance",
        '给每个分子分别判定路线与样品。只输出 JSON，例如 {"assessments":[{"id":"G00000",'
        '"route_status":"route_reported","sample_status":"unknown","evidence_ids":["记录ID"]}]}。'
        '仅 exact_route_structure=true 可判 route_reported；若只有平台线索判 platform_only，'
        '其余 unknown；没有样品证据一律 unknown。平台关联不可继承前体路线。',
        {"cases": cases, "route_evidence": source},
    )
    proposals = {row["id"]: row for row in raw["assessments"]}
    records = []
    for case in cases:
        route_status = "unknown"
        if case["exact_route_structure"]:
            route_status = "route_reported"
        elif case["platform"] in {"phenol", "guaiacol_like"}:
            route_status = "platform_only"
        expected = {"id": case["id"], "route_status": route_status, "sample_status": "unknown",
                    "evidence_ids": [route["id"]] if route_status == "route_reported" else []}
        records.append({**expected, "model_agreed": proposals.get(case["id"]) == expected})
    demo.write_json(output / "provenance.json", records)
    return records


def grid_options(cell):
    patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in MOTIFS.items()}
    eligible = cell[cell["pareto_eligible"]]
    raw = {name: {row.candidate_id for row in eligible.itertuples() if row.mol.HasSubstructMatch(pattern)}
           for name, pattern in patterns.items()}
    options = []
    for sequence in itertools.permutations(MOTIFS, 2):
        first, second = sequence
        counts = [len(raw[first]), len(raw[second] - raw[first])]
        if min(counts) < 2:
            continue
        options.append({"id": "+".join(sequence), "rules": [
            {"label": name, "smarts": MOTIFS[name]} for name in sequence],
            "raw_counts": [len(raw[name]) for name in sequence],
            "ordered_counts": counts, "overlap": len(raw[first] & raw[second]),
            "remainder": len(eligible) - sum(counts)})
    return options


def grid_agent(llm, packet, cell, ga, output):
    options = grid_options(cell)
    evidence_ids = {item["id"] for item in packet["lohc_evidence"]}
    input_packet = {**packet, "prechecked_options": options}
    demo.write_json(output / "grid_input.json", input_packet)
    raw = llm.run(
        "grid",
        '选择一个最值得验证的预检查分组方案，按化学差异解释。返回 JSON，例如 '
        '{"option_id":"某个输入ID","hypothesis":"可证伪假设","evidence_ids":["输入ID"],'
        '"test":"独立检验","budget_slots":96,"reversibility_support":"analog_only"}。'
        '只选输入方案和证据 ID，reversibility_support 只能为 analog_only 或 unknown。'
        '样本计数按 ordered_counts 使用，生物质路线只支持路线判断。评价保留未知项。',
        input_packet,
    )
    option = next(item for item in options if item["id"] == raw["option_id"])
    if not set(raw["evidence_ids"]) <= evidence_ids or raw["reversibility_support"] not in {"analog_only", "unknown"}:
        raise ValueError("Invalid evidence or reversibility class")
    plan = {key: raw[key] for key in ("hypothesis", "evidence_ids", "test", "budget_slots")}
    plan.update(parent_cell=packet["parent_cell"], rules=option["rules"])
    replay = demo.replay_plan(plan, packet, cell, ga)
    actual = [replay["groups"][rule["label"]]["eligible_rows"] for rule in plan["rules"]]
    if actual != option["ordered_counts"]:
        raise RuntimeError("Grid preflight and replay counts differ")
    demo.write_json(output / "grid_plan.json", {"option": option, "plan": plan, "evaluation": raw})
    demo.write_json(output / "grid_replay.json", replay)
    return plan, replay


def run_foundation(llm, output):
    RDLogger.DisableLog("rdApp.*")
    ga = demo.load_native_qd(demo.GA_CODE)
    frame = demo.import_ga(PROTO / "ga_archive.csv", ga)
    seed = demo.load_seed(PROTO / "lohc_seed.json")
    cell = frame[frame["chemical_behavior_cell"].eq(17)].copy()
    packet = demo.make_packet(cell, ga, seed, 17)
    route = evidence_agent(llm, output)
    packet["lohc_evidence"].append({**route, "relationship_to_retrieved_examples": "phenol_route_only"})
    examples = packet["native_local_pareto_representatives"] + packet["near_misses"]
    sources = provenance_agent(llm, examples, route, output)
    plan, replay = grid_agent(llm, packet, cell, ga, output)
    memory = output / "memory.sqlite"
    demo.save_memory(memory, frame, seed + [route], packet, plan, replay, "deepseek_with_prechecked_options")
    with sqlite3.connect(memory) as db:
        db.execute("CREATE TABLE provenance (candidate_id TEXT PRIMARY KEY, record_json TEXT NOT NULL)")
        db.executemany("INSERT INTO provenance VALUES (?,?)", [(row["id"], demo.json_text(row)) for row in sources])


def strategy_agent(llm, output, pilot_path):
    pilot = demo.read_json(pilot_path)
    runs = [{k: v for k, v in row.items() if k != "candidates"} for row in pilot["results"]]
    indexed = {(row["seed"], row["strategy"]): row for row in runs}
    differences = [
        {"seed": seed, "delta_hits_diverse_minus_high_quality":
         indexed[(seed, "diverse_parents")]["target_hits"] - indexed[(seed, "high_quality_parents")]["target_hits"]}
        for seed in sorted({row["seed"] for row in runs})
    ]
    totals = {}
    for strategy in ("diverse_parents", "high_quality_parents"):
        subset = [row for row in runs if row["strategy"] == strategy]
        totals[strategy] = {field: sum(row[field] for row in subset) for field in (
            "target_hits", "evaluated", "parent_draws", "operator_attempts", "operator_failures",
            "novel_proposed", "valid_pair_predictions")}
    summary = {"runs": runs, "paired_differences": differences, "totals": totals,
               "scope": "single-batch native GA/GNN proxy only"}
    demo.write_json(output / "strategy_statistics.json", summary)
    plan = llm.run(
        "strategy",
        '根据程序统计提出一个下一批配对策略，不编造分数，不以小样本判定普遍胜负。返回 JSON，'
        '例如 {"preferred_strategy":"balanced_control","control_strategy":"high_quality_parents",'
        '"slots_per_arm":20,"reason":"根据差值解释","next_test":"对照设计"}。'
        'preferred_strategy 只选 diverse_parents、high_quality_parents 或 balanced_control，'
        'control_strategy 只选 diverse_parents 或 high_quality_parents，两个策略不得相同。',
        summary,
    )
    arms = {"diverse_parents", "high_quality_parents"}
    if (plan["preferred_strategy"] not in arms | {"balanced_control"}
            or plan["control_strategy"] not in arms
            or plan["preferred_strategy"] == plan["control_strategy"]
            or type(plan["slots_per_arm"]) is not int or not 1 <= plan["slots_per_arm"] <= 512):
        raise ValueError("Invalid strategy plan")
    demo.write_json(output / "strategy_plan.json", {**plan, "status": "proposed_only"})


def evaluate_molecules(llm, output):
    packet = demo.read_json(PROTO / "demo" / "planning_packet.json")
    seed = demo.load_seed(PROTO / "lohc_seed.json")
    candidates = packet["native_local_pareto_representatives"] + packet["near_misses"]
    cases = [{"candidate": row, "matches": demo.literature_matches(seed, [row], limit=2)}
             for row in candidates]
    raw = llm.run(
        "molecule_evaluation",
        '评价每个分子对的可逆储放氢证据。仅返回 JSON，例如 {"assessments":[{"id":"G00000",'
        '"support":"analog_only","evidence_ids":["ID"],"risk_hypothesis":"待验证的风险",'
        '"verification":"下一步验证"}]}。允许 support 为 analog_only 或 unknown；'
        '输入若无同分子对实测，禁止给出实际可逆性结论。类似物或副反应研究只能提出风险假设。'
        'unknown 可以使用空 evidence_ids。每个输入 ID 恰好出现一次。',
        {"cases": cases},
    )
    rows = raw["assessments"]
    indexed = {row["id"]: row for row in rows}
    expected = {case["candidate"]["candidate_id"] for case in cases}
    if len(rows) != len(indexed) or indexed.keys() != expected:
        raise ValueError("Molecule assessment IDs do not match the input")
    for case in cases:
        row = indexed[case["candidate"]["candidate_id"]]
        if (row["support"] not in {"analog_only", "unknown"}
                or not set(row["evidence_ids"]) <= {item["id"] for item in case["matches"]}
                or (row["support"] == "analog_only" and not row["evidence_ids"])
                or not row["verification"].strip()):
            raise ValueError(f"Invalid molecule assessment for {row['id']}")
    demo.write_json(output / "molecule_evaluation.json", {
        "assessments": rows, "status": "evidence_links_validated_scientific_review_pending",
        "inputs": cases,
    })


def heldout_grid(llm, output):
    RDLogger.DisableLog("rdApp.*")
    ga = demo.load_native_qd(demo.GA_CODE)
    seed = demo.load_seed(PROTO / "lohc_seed.json")
    frame = demo.import_ga(PROTO / "ga_archive.csv", ga)
    cases = {}
    for cell_id, cell in frame.groupby("chemical_behavior_cell", sort=True):
        if cell_id < 0 or cell_id == 17 or cell["pareto_eligible"].sum() < 4:
            continue
        options = grid_options(cell)
        if not options:
            continue
        packet = demo.make_packet(cell, ga, seed, int(cell_id))
        cases[int(cell_id)] = (cell, packet, options[:3])
        if len(cases) == 5:
            break
    if len(cases) < 5:
        raise ValueError("Fewer than five eligible held-out cells")
    payload = [{
        "parent_cell": cell_id, "options": options,
        "representative_smiles": [row["canonical_smi"] for row in packet["native_local_pareto_representatives"]],
        "evidence": [{key: item[key] for key in ("id", "finding", "relationship_to_retrieved_examples")}
                     for item in packet["lohc_evidence"][:2]],
    } for cell_id, (_, packet, options) in cases.items()]
    raw = llm.run(
        "heldout_grid",
        '为每个父格各选一个输入提供的合法 option_id，返回 JSON，例如 {"plans":[{"parent_cell":0,'
        '"option_id":"输入方案ID","evidence_ids":["输入ID"],"hypothesis":"可证伪结构差异",'
        '"test":"独立复核问题","budget_slots":24}]}。只使用该父格的 evidence ID。'
        '每条假设和验证各用一句话，区分类比与实际测量。',
        payload,
    )
    indexed = {item["parent_cell"]: item for item in raw["plans"]}
    results = []
    for cell_id, (cell, packet, options) in cases.items():
        try:
            item = indexed[cell_id]
            option = next(option for option in options if option["id"] == item["option_id"])
            evidence = next(row["evidence"] for row in payload if row["parent_cell"] == cell_id)
            if not set(item["evidence_ids"]) <= {row["id"] for row in evidence}:
                raise ValueError("Evidence was absent from the LLM input")
            plan = {name: item[name] for name in ("parent_cell", "evidence_ids", "hypothesis", "test", "budget_slots")}
            plan["rules"] = option["rules"]
            replay = demo.replay_plan(plan, packet, cell, ga)
            results.append({"cell": cell_id, "validation": "passed", "plan": plan,
                            "offered_options": len(options), "groups": {
                                label: group["eligible_rows"] for label, group in replay["groups"].items()}})
        except (KeyError, StopIteration, ValueError, TypeError) as exc:
            results.append({"cell": cell_id, "validation": "failed", "error_type": type(exc).__name__})
    demo.write_json(output / "heldout_grid.json", {
        "selection": "first five sorted cells with four eligible rows and at least one feasible option",
        "scope": "first-pass technical validation conditional on feasible motif options",
        "inputs": payload, "results": results,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    operations = {"foundation": run_foundation, "evaluate": evaluate_molecules, "heldout": heldout_grid}
    parser.add_argument("mode", choices=[*operations, "strategy"])
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pilot-results", type=Path, default=HERE / "ga_pilot_reproducible.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    llm = DeepSeek(credential(args.key_file), args.out)
    if args.mode == "strategy":
        strategy_agent(llm, args.out, args.pilot_results)
    else:
        operations[args.mode](llm, args.out)


if __name__ == "__main__":
    main()
