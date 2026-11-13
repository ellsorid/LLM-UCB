"""Review equal per-arm LOHC candidate samples without changing either GA."""

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "LOHC_Agents_v2"))
import agents
from dashboard import ARMS, read_json
from worker import atomic_json

PROMPT = (
    "独立评价候选储氢/放氢分子对是否值得做下一步验证。输入数值是代码代理性质。"
    "选择 priority（建议开展具体独立验证）、evidence_gap（先补决定性证据）、low（具体问题导致低优先级）。"
    "关注结构对是否合理、可能竞争反应、液态范围和验证步骤。无同分子实测时，可逆性与实际生物基来源保留未知。"
    "相关论文只作证据类比，若无适用证据用空列表。输出严格JSON，顶层只有 assessments，"
    "数组每项只含 id、priority、reason、first_test、evidence_ids。id逐项复制输入，reason及first_test各不超过35个汉字。"
)
TERMINAL = {"已停止", "已立即停止", "已完成", "运行失败"}


def compress(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else smi


def sample(rows, count, reviewed):
    ordered = sorted(rows, key=lambda row: (row["chemical_cell"] in reviewed, -row["quality"], row["smi"]))
    first, rest, cells = [], [], set()
    for row in ordered:
        cell = row["chemical_cell"]
        (rest if cell in cells else first).append(row)
        cells.add(cell)
    return (first + rest)[:count]


def validate(raw, ids, evidence):
    rows = raw["assessments"]
    if set(raw) != {"assessments"} or len(rows) != len(ids) or {r["id"] for r in rows} != ids:
        raise ValueError("Incomplete candidate review")
    for row in rows:
        if (set(row) != {"id", "priority", "reason", "first_test", "evidence_ids"}
                or row["priority"] not in {"priority", "evidence_gap", "low"}
                or not isinstance(row["reason"], str) or not row["reason"].strip()
                or not isinstance(row["first_test"], str) or not row["first_test"].strip()
                or not isinstance(row["evidence_ids"], list) or not set(row["evidence_ids"]) <= evidence):
            raise ValueError("Invalid candidate review")
    return {row["id"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "live")
    parser.add_argument("--key-file", required=True, type=Path)
    parser.add_argument("--per-arm", type=int, default=3)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    run = args.run.resolve()
    directory = run / "candidate_audit"
    directory.mkdir(exist_ok=False)
    llm = agents.DeepSeek(agents.credential(args.key_file), directory)
    evidence = read_json(HERE.parent / "LOHC_Grid_Guided_GA_v4" / "live_run" / "frozen_memory.json")
    evidence = [{key: row[key] for key in ("id", "family", "lean_smiles", "rich_smiles", "finding", "doi")}
                for row in evidence]
    ids = {row["id"] for row in evidence}
    reviewed = {arm: set() for arm in ARMS}
    priorities = {arm: set() for arm in ARMS}
    totals = {arm: Counter() for arm in ARMS}
    cache = {}
    generation = 1
    tokens = 0
    while True:
        states = {arm: read_json(run / arm / "status.json", {}) or {} for arm in ARMS}
        if any((run / arm / "STOP").exists() or (run / arm / "hard_stop.json").exists() for arm in ARMS):
            break
        common = min(state.get("completed_generation", 0) for state in states.values())
        if generation > common:
            if any(state.get("phase") in TERMINAL for state in states.values()):
                break
            time.sleep(3)
            continue
        pool = {arm: read_json(run / arm / f"candidates_{generation:04d}.json", []) or [] for arm in ARMS}
        count = min(args.per_arm, *(len(pool[arm]) for arm in ARMS))
        selected = {arm: sample(pool[arm], count, reviewed[arm]) for arm in ARMS}
        assigned, proposed = {}, {}
        for arm in ARMS:
            assigned[arm] = []
            for row in selected[arm]:
                lean, rich = compress(row["smi"]), compress(row["rich_proxy_smi"])
                key = lean, rich
                assigned[arm].append((row["chemical_cell"], key))
                if key not in cache and key not in proposed:
                    proposed[key] = {"id": f"P{len(proposed) + 1:02d}", "lean_smi": lean, "rich_smi": rich,
                                     "platform": row["platform"], "properties": row["properties"]}
        error = None
        if proposed:
            role = f"candidate_{generation:04d}"
            packet = {"candidates": list(proposed.values()), "verified_memory": evidence}
            atomic_json(directory / f"{role}_input.json", packet)
            try:
                raw = llm.run(role, PROMPT, packet)
                review = validate(raw, {row["id"] for row in proposed.values()}, ids)
                cache.update({key: review[row["id"]] for key, row in proposed.items()})
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                error = type(exc).__name__
            usage = (read_json(directory / f"{role}_audit.json", {}) or {}).get("usage") or {}
            tokens += usage.get("total_tokens", 0)
        report = {}
        for arm in ARMS:
            evaluated = []
            for cell, key in assigned[arm]:
                if key not in cache:
                    continue
                row = cache[key]
                reviewed[arm].add(cell)
                totals[arm][row["priority"]] += 1
                if row["priority"] == "priority":
                    priorities[arm].add(cell)
                evaluated.append({"cell": cell, "lean_smi": key[0], "rich_smi": key[1], **row})
            report[arm] = evaluated
            atomic_json(run / arm / "candidate_review_summary.json", {
                "through_generation": generation, "reviewed_cells": len(reviewed[arm]),
                "priority_cells": len(priorities[arm]), "decision_counts": dict(totals[arm]),
                "shared_audit_tokens": tokens,
            })
        atomic_json(directory / f"generation_{generation:04d}.json", {
            "generation": generation, "planned_per_arm": count, "reviews": report, "fallback": error,
        })
        print(generation, count, [len(priorities[arm]) for arm in ARMS], tokens, error, flush=True)
        generation += 1


if __name__ == "__main__":
    main()
