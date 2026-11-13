"""Blinded structure-pair extension scores and dashboard color data."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import random
from pathlib import Path
from statistics import median

from candidate_audit import ARMS, HERE, agents, atomic_json, compress, read_json
from dashboard import NX, NY, H_RANGE, D_RANGE

DIMENSIONS = ("thermodynamics", "liquid", "selectivity", "cycling")
PROMPT = """你在盲评 LOHC 分子对。四维评分是有依据的结构推断等级，0=明显不利，1=偏不利，2=有利不利均有，3=偏有利，4=有很强的支持。无法判断方向则填 null。thermodynamics 看反应对与相态/平衡证据；候选焓变有意隐藏，不得推断它的数值、所需温度或声称其焓变已通过。liquid 看两端的液态/流动/挥发风险；selectivity 看目标产物相对竞争路径；cycling 看化学稳定与循环风险。类比和推断不等于实测。只返回 JSON 顶层 assessments 数组，每项含 id, scores（严格包含上述四键）, basis（同四键，短理由）, evidence_ids（只允许输入ID）, recommendation（priority/gap/low）, first_test。逐项复制输入 ID。理由尽量短。"""
CAP = 26810
N = 12


def display_bin(h, d):
    if not (H_RANGE[0] <= h <= H_RANGE[1] and D_RANGE[0] <= d <= D_RANGE[1]):
        return None
    return (min(NX - 1, int(NX * (h - H_RANGE[0]) / (H_RANGE[1] - H_RANGE[0]))),
            min(NY - 1, int(NY * (d - D_RANGE[0]) / (D_RANGE[1] - D_RANGE[0]))))


def pool(run, arm):
    history = read_json(run / arm / "progress_history.json")
    cutoff = max((row for row in history if row["gnn_pairs"] <= CAP), key=lambda row: row["generation"])
    seen = set()
    groups = defaultdict(list)
    for generation in range(1, cutoff["generation"] + 1):
        rows = read_json(run / arm / f"candidates_{generation:04d}.json", [])
        for row in rows:
            pair = compress(row["smi"]), compress(row["rich_proxy_smi"])
            props = row["properties"]
            key = display_bin(props["h2_wt_percent"], props["delta_H_per_H2"])
            if key is None or pair in seen:
                continue
            seen.add(pair)
            groups[key].append({**row, "pair": pair, "generation": generation})
    return cutoff, groups


def validate(raw, expected, evidence):
    rows = raw["assessments"]
    if set(raw) != {"assessments"} or len(rows) != len(expected) or {row["id"] for row in rows} != expected:
        raise ValueError("Review IDs differ")
    for row in rows:
        if (set(row) != {"id", "scores", "basis", "evidence_ids", "recommendation", "first_test"}
                or set(row["scores"]) != set(DIMENSIONS) or set(row["basis"]) != set(DIMENSIONS)
                or any(s is not None and (type(s) is not int or not 0 <= s <= 4) for s in row["scores"].values())
                or any(not isinstance(s, str) or not s.strip() for s in row["basis"].values())
                or not isinstance(row["evidence_ids"], list) or not set(row["evidence_ids"]) <= evidence
                or row["recommendation"] not in {"priority", "gap", "low"}
                or not isinstance(row["first_test"], str) or not row["first_test"].strip()):
            raise ValueError("Review failed schema")
    return rows


def scores(row, review):
    native = 100 * row["quality"]
    if not review:
        return {"native": native, "llm": None, "composite": None, "llm_bounds": [0, 100],
                "composite_bounds": [0.25 * native, 0.25 * native + 75]}
    values = list(review["scores"].values())
    unknown = values.count(None)
    low = sum(s for s in values if s is not None) * 100 / 16
    high = low + unknown * 25
    return {"native": native, "llm": low if not unknown else None,
            "composite": 0.25 * native + 0.75 * low if not unknown else None,
            "llm_bounds": [low, high], "composite_bounds": [0.25 * native + 0.75 * low, 0.25 * native + 0.75 * high]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "live")
    parser.add_argument("--key-file", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = run / "extended_audit_v2"
    output.mkdir(exist_ok=False)
    data = {arm: pool(run, arm) for arm in ARMS}
    common = set(data["native"][1]) & set(data["llm"][1])
    keys = sorted(common, key=lambda key: (-min(len(data[arm][1][key]) for arm in ARMS), key))[:N]
    if not keys:
        raise RuntimeError("No overlapping property bins")
    rng = random.Random(11)
    sampled, anonymous = {}, {}
    for arm in ARMS:
        sampled[arm] = {}
        for key in keys:
            row = rng.choice(sorted(data[arm][1][key], key=lambda item: item["pair"]))
            sampled[arm][key] = row
            anonymous.setdefault(row["pair"], row["properties"])
    pairs = sorted(anonymous)
    rng.shuffle(pairs)
    mapping = {key: f"P{index:03d}" for index, key in enumerate(pairs, 1)}
    evidence = read_json(HERE.parent / "LOHC_Grid_Guided_GA_v4" / "live_run" / "frozen_memory.json")
    evidence = [{key: row[key] for key in ("id", "lean_smiles", "rich_smiles", "finding", "doi")} for row in evidence]
    llm = agents.DeepSeek(agents.credential(args.key_file), output)
    frozen = {"budget_cap": CAP, "cutoffs": {arm: data[arm][0] for arm in ARMS},
              "samples": {arm: [{"bin": list(key), "id": mapping[row["pair"]], **row} for key, row in rows.items()]
                          for arm, rows in sampled.items()}}
    atomic_json(output / "sample.json", frozen)

    def ask(job):
        number, group = job
        packet = {"pairs": [{"id": mapping[key], "lean": key[0], "rich": key[1],
                             "proxy_properties": {name: value for name, value in anonymous[key].items()
                                                  if name != "delta_H_per_H2"}}
                            for key in group], "evidence": evidence}
        role = f"blind_{number:02d}"
        atomic_json(output / f"{role}_input.json", packet)
        for suffix in ("", "_retry"):
            try:
                return validate(llm.run(role + suffix, PROMPT, packet),
                                {mapping[key] for key in group}, {row["id"] for row in evidence})
            except (OSError, ValueError, TypeError, RuntimeError, KeyError):
                continue
        return []

    groups = [pairs[i:i+3] for i in range(0, len(pairs), 3)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        reviews = {row["id"]: row for result in executor.map(ask, enumerate(groups, 1)) for row in result}
    results, summaries = {}, {}
    for arm in ARMS:
        results[arm] = {}
        for key, row in sampled[arm].items():
            review = reviews.get(mapping[row["pair"]])
            results[arm][f"{key[0]},{key[1]}"] = {**row, **scores(row, review), "review": review}
        values = list(results[arm].values())
        exact = [row["llm"] for row in values if row["llm"] is not None]
        summaries[arm] = {
            "sampled": len(values), "reviewed": sum(bool(row["review"]) for row in values), "exact_scores": len(exact),
            "priority": sum(row["review"] and row["review"]["recommendation"] == "priority" or False for row in values),
            "llm_median_exact": median(exact) if exact else None,
            "llm_median_bounds": [median(row["llm_bounds"][i] for row in values) for i in (0, 1)],
            "composite_median_bounds": [median(row["composite_bounds"][i] for row in values) for i in (0, 1)],
        }
    audits = list(output.glob("*_audit.json"))
    usage = sum(((read_json(path) or {}).get("usage") or {}).get("total_tokens", 0) for path in audits)
    report = {"budget_cap": CAP, "cutoffs": frozen["cutoffs"], "common_bins": len(keys), "summaries": summaries,
              "bins": results, "evidence": evidence, "requests": len(audits), "tokens": usage}
    atomic_json(run / "extended_evaluation.json", report)
    print(summaries, "requests", len(audits), "tokens", usage)


if __name__ == "__main__":
    main()
