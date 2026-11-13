"""Budgeted literature-anchored review between local Pareto and GA survival."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math

import numpy as np
import pandas as pd
from rdkit import Chem

from extended_audit import HERE, agents, atomic_json, read_json
from paired_reward import checked_pair
from spatial_feedback import held_out, rank, usage

DIMENSIONS = ("reversible", "selectivity", "cycling", "liquid")
HIGH = 60
SPLIT = 0.60
VERSION = "literature-anchored-qd-ac-v1"
PROMPT = """你是 LOHC 结构评审。输入为规范的储氢/放氢态分子对及已核验论文。四个独立维度 reversible（目标储放氢反应路径可行性）、selectivity（目标产物相对竞争反应）、cycling（多轮保持）、liquid（20–25摄氏度两态液相可用性；实测任一端为固体，至多1分）。采用固定文献比较尺度：0=明确严重不利，1=明显弱于相关可工作体系，2=利弊相当，3=有依据接近已报道可工作的体系，4=有可比同分子直接证据，达到或超过该参照。结构推断不能报 4。不同条件的转化率、产率、实际放氢量不可混用；转化率不能证明选择性。文献只用于 scopes 指定维度。绝不推断未测物性数值或生物质来源。
每项返回 grade（整数0到4或null）、evidence（exact、analog、structural、unknown之一）、reference（引用输入文献id或null）、reason（至多25汉字）。只有能指明相关文献参照和具体结构依据时给等级；缺证据写null，尤其不能从一个SMILES声称液相或循环表现。逐项复制输入 ID。只返回 JSON。完整格式如下：
""" + json.dumps({"assessments": [{"id": "P01", "dimensions": {
    d: {"grade": None, "evidence": "unknown", "reference": None, "reason": "缺少证据"} for d in DIMENSIONS
}}]}, ensure_ascii=False)
PLAN = """你管理 LOHC 性质 QD。输入是一个父格已独立评审的骨架样本和支持候选的维度。只在两类候选对应不同、有化学意义的性能方向时，从 allowed 中选一个维度来分两个子格。已冻结边界0.60，不建议改动；证据薄弱则不细分。只返回 JSON。弃权例：{"dimension":null,"reason":"证据不足","references":[]}。选择维度时 dimension 复制 allowed 的名字，references 填至少一条输入证据ID。"""


def evidence_records():
    records = read_json(HERE.parent / "LOHC_Grid_Guided_GA_v4" / "live_run" / "frozen_memory.json")
    for record in records:
        if record["id"] == "NEC-SIX-CYCLES":
            record["rich_smiles"] = "CCN1C2CCCCC2C2CCCCC21"
            record["observations"] = {"reported_hydrogenation_pct": 99.0, "reported_dehydrogenation_pct": 91.7,
                                      "cycles": 6}
    records += [
        {"id": "NEC-PHASE", "doi": "10.1021/acs.jced.5b00679",
         "lean_smiles": "CCn1c2ccccc2c2ccccc21", "rich_smiles": "CCN1C2CCCCC2C2CCCCC21",
         "finding": "测量不同氢化程度N-乙基咔唑及两端混合物的固液平衡。纯脱氢态在室温固化，液相可用性须与指定工况比较。"},
        {"id": "MCH-LIQUID", "doi": "10.1021/acs.jced.5b00789",
         "lean_smiles": "Cc1ccccc1", "rich_smiles": "CC1CCCCC1",
         "finding": "原研究分别测量甲苯、甲基环己烷液相中的氢溶解度。支持两态液相参照，不能外推黏度或挥发损失。"},
        {"id": "BT-DBT-PHASE", "doi": "10.1016/j.ijhydene.2020.03.210",
         "finding": "BT/DBT 低温工程参照仅在原文条件下比较；商品异构体混合物不能作为单个候选的同分子证据。"},
    ]
    scopes = {
        "MCH-CYCLE": ["reversible", "selectivity", "cycling"], "DBT-CYCLE": ["reversible", "cycling"],
        "23DMID-H": ["reversible", "selectivity"], "23DMID-D": ["reversible", "selectivity"],
        "INDOLE-PATH": ["reversible", "selectivity"], "NMID-STEREO": ["reversible", "selectivity"],
        "GUAIACOL-HDO": ["selectivity"], "ARYL-ETHER-CLEAVAGE": ["selectivity"],
        "NEC-SIX-CYCLES": ["reversible", "cycling"], "NEC-PHASE": ["liquid"],
        "MCH-LIQUID": ["liquid"], "BT-DBT-PHASE": ["liquid"],
    }
    return [{**r, "scopes": scopes.get(r["id"], [])} for r in records]


def literature_packet(evidence):
    keys = ("id", "lean_smiles", "rich_smiles", "conditions", "observations", "finding", "doi", "scopes")
    return [{k: row[k] for k in keys if k in row} for row in evidence]


def validated(raw, rows, evidence):
    items = raw["assessments"]
    expected = {row["id"]: row for row in rows}
    if set(raw) != {"assessments"} or len(items) != len(rows) or {r["id"] for r in items} != set(expected):
        raise ValueError("Invalid review IDs")
    for item in items:
        dimensions = item["dimensions"]
        if set(item) != {"id", "dimensions"} or set(dimensions) != set(DIMENSIONS):
            raise ValueError("Invalid dimensions")
        for dimension, value in dimensions.items():
            if set(value) != {"grade", "evidence", "reference", "reason"}:
                raise ValueError("Invalid dimension schema")
            grade, level, reference = value["grade"], value["evidence"], value["reference"]
            if not isinstance(value["reason"], str) or not value["reason"].strip():
                raise ValueError("Missing reason")
            if grade is None:
                if level != "unknown" or reference is not None:
                    raise ValueError("Unknown grade must stay unknown")
                continue
            if type(grade) is not int or not 0 <= grade <= 4 or level not in {"exact", "analog", "structural"}:
                raise ValueError("Invalid grade")
            if reference not in evidence or grade == 4 and level != "exact":
                raise ValueError("Missing or invalid reference")
            if dimension not in evidence[reference]["scopes"]:
                value.update(grade=None, evidence="unknown", reference=None, reason="参考文献不覆盖该指标")
                continue
            if level == "exact":
                record = evidence[reference]
                if checked_pair(record.get("lean_smiles"), record.get("rich_smiles")) != tuple(expected[item["id"]]["pair"]):
                    # A wrong exact-identity claim supplies no score.
                    value.update(grade=None, evidence="unknown", reference=None, reason="文献与候选结构对不匹配")
                elif dimension == "liquid" and reference == "NEC-PHASE":
                    value.update(grade=min(1, grade), reason="已知脱氢态室温为固体")
    return {r["id"]: r["dimensions"] for r in items}


def interval(value):
    if value["grade"] is None:
        return 0., 1.
    x = value["grade"] / 4
    half = {"exact": .04, "analog": .12, "structural": .19}[value["evidence"]]
    return max(0., x - half), min(1., x + half)


def score(quality, dimensions):
    bounds = [interval(dimensions[name]) for name in DIMENSIONS]
    lo, hi = [100 * (.25 * quality + .75 * sum(v[i] for v in bounds) / 4) for i in (0, 1)]
    known = sum(dimensions[k]["grade"] is not None for k in DIMENSIONS)
    usable = (known >= 3 and dimensions["reversible"]["grade"] is not None and
              dimensions["selectivity"]["grade"] is not None)
    return {"low": lo, "high": hi, "mid": (lo + hi) / 2, "usable": usable,
            "high_f": usable and lo >= HIGH}


def candidate(row):
    pair = checked_pair(row.get("smi"), row.get("sat_smi"))
    if not pair:
        return None
    return {"smi": row["smi"], "pair": list(pair), "scaffold": str(row.get("scaffold_smi") or "(acyclic)"),
            "cell": row["qd_cell"], "quality": float(row["qd_quality"]),
            "properties": {k: float(row[k]) for k in ("h2_wt_percent", "delta_H_per_H2")}}


class Review:
    def __init__(self, key_file, output, evidence, limit):
        output.mkdir()
        self.output = output
        self.client = agents.DeepSeek(agents.credential(key_file), output)
        self.limit = limit
        self.calls = 0
        self.evidence = {row["id"]: row for row in evidence}
        self.payload = literature_packet(evidence)
        self.grades = {}
        self.attempted = set()
        atomic_json(output / "protocol.json", {"version": VERSION, "limit": limit, "high_f_lower_bound": HIGH,
                    "split_boundary": SPLIT, "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                    "planning_prompt_sha256": hashlib.sha256(PLAN.encode()).hexdigest(), "evidence": self.payload,
                    "model": "deepseek-flash", "temperature": 0, "max_tokens": 1100})

    def ask(self, job):
        role, prompt, payload, parser = job
        atomic_json(self.output / f"{role}_input.json", payload)
        try:
            return parser(self.client.run(role, prompt, payload))
        except (KeyError, ValueError, RuntimeError, OSError, AttributeError, TypeError, IndexError) as exc:
            atomic_json(self.output / f"{role}_failure.json", {"error": type(exc).__name__})
            return None

    def run(self, specs):
        results = [None] * len(specs)
        pending = list(enumerate(specs))
        with ThreadPoolExecutor(max_workers=4) as pool:
            for _ in range(2):
                jobs = pending[:self.limit - self.calls]
                args = [(f"call_{self.calls + i + 1:03d}", *spec) for i, (_, spec) in enumerate(jobs)]
                self.calls += len(args)
                pending = []
                for (index, spec), result in zip(jobs, pool.map(self.ask, args)):
                    if result is None:
                        pending.append((index, spec))
                    else:
                        results[index] = result
                if not pending:
                    break
        return results

    def review(self, rows):
        rows = list({tuple(r["pair"]): r for r in rows if tuple(r["pair"]) not in self.attempted}.values())
        rows = rows[:max(0, self.limit - self.calls) * 2]
        jobs = []
        for start in range(0, len(rows), 2):
            batch = [{"id": f"P{n+1:02d}", "pair": r["pair"]} for n, r in enumerate(rows[start:start+2])]
            for row in batch:
                self.attempted.add(tuple(row["pair"]))
            packet = {"molecules": batch, "evidence": self.payload}
            jobs.append((PROMPT, packet, lambda data, b=batch: validated(data, b, self.evidence)))
        for start, result in zip(range(0, len(rows), 2), self.run(jobs)):
            if result:
                for n, row in enumerate(rows[start:start+2], 1):
                    self.grades[tuple(row["pair"])] = result[f"P{n:02d}"]

    def plan(self, cell, allowed, sample):
        payload = {"cell": cell, "allowed": allowed, "samples": sample, "evidence": self.payload}

        def check(raw):
            d, refs = raw["dimension"], raw["references"]
            if (set(raw) != {"dimension", "reason", "references"} or d not in [None, *allowed]
                    or not isinstance(refs, list) or not set(refs) <= self.evidence.keys()
                    or (d is not None and not refs) or not isinstance(raw["reason"], str) or not raw["reason"].strip()):
                raise ValueError("Invalid subgrid proposal")
            return raw
        return self.run([(PLAN, payload, check)])[0]


def subcell(dimension, dimensions):
    lo, hi = interval(dimensions[dimension])
    return "high" if lo >= SPLIT else "low" if hi < SPLIT else None


def source_status(row, evidence):
    lean = row["pair"][0]
    routes = [e for e in evidence if e.get("id") in {"PHENOL-PINE-ROUTE", "PROPYLGUAIACOL-BIRCH"}]
    for record in routes:
        for smi in [record.get("lean_smiles"), record.get("rich_smiles")]:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None and Chem.MolToSmiles(mol) == lean:
                return {"status": "published_exact_route", "doi": record["doi"]}
    scaffold = row["scaffold"]
    related = [e["doi"] for e in routes if e.get("lean_scaffold") == scaffold]
    return {"status": "analog_route" if related else "unknown", "references": related}


class QDFeedback:
    def __init__(self, reviewer, output):
        self.reviewer = reviewer
        self.output = output
        self.records = defaultdict(dict)
        self.plans = {}
        self.last_reviewed = {}
        self.replacements = 0
        self.plan_tried = set()

    def shortlist(self, ranked, survivors, ga):
        groups = {}
        eligible = ranked[ranked["qd_cell"].ne("") & ranked["qd_cell"].isin(survivors["qd_cell"])]
        eligible = eligible.loc[~eligible["smi"].map(held_out)]
        for cell, group in eligible.groupby("qd_cell"):
            arr = group[["obj_h2_score", "obj_dh_score", "obj_scaffold_novelty"]].to_numpy(dtype=float)
            good = np.isfinite(arr).all(axis=1)
            if not good.any():
                continue
            front = ga._non_dominated_sort_indices(arr[good])[0]
            front_group = group.iloc[np.flatnonzero(good)[front]]
            ordered = ga._sort_by_pareto_priority(front_group).drop_duplicates("scaffold_smi")
            # Keep a few lower fronts as a fallback when the first front is very small.
            reserve = ga._sort_by_pareto_priority(group[~group["smi"].isin(ordered["smi"])]).drop_duplicates("scaffold_smi")
            rows = pd.concat([ordered, reserve]).drop_duplicates("scaffold_smi").head(16)
            groups[cell] = [item for row in rows.to_dict("records") if (item := candidate(row))]
        return groups

    def propose(self, generation, cell):
        examples = self.records[cell]
        if cell in self.plans or len(examples) < 6 or self.reviewer.calls >= self.reviewer.limit:
            return
        key = (cell, len(examples))
        if key in self.plan_tried:
            return
        allowed = []
        for d in DIMENSIONS:
            counts = Counter(subcell(d, rec["dimensions"]) for rec in examples.values())
            if counts["high"] >= 2 and counts["low"] >= 2:
                allowed.append(d)
        if not allowed:
            return
        self.plan_tried.add(key)
        plan = self.reviewer.plan(cell, allowed, list(examples.values())[:12])
        if plan and plan["dimension"]:
            self.plans[cell] = {**plan, "generation": generation, "boundary": SPLIT}

    def update(self, generation, ranked, survivors, ga):
        groups = self.shortlist(ranked, survivors, ga)
        budget = self.reviewer.limit - self.reviewer.calls
        queue = []
        if budget:
            for cell, rows in groups.items():
                unseen = [r for r in rows if tuple(r["pair"]) not in self.reviewer.attempted]
                support = len({r["scaffold"] for r in rows})
                if support >= 6 and len(unseen) >= 4 and generation - self.last_reviewed.get(cell, -10) >= 2:
                    queue.append((cell, unseen))
        if queue:
            cell, unseen = min(queue, key=lambda p: (cell_priority(self.records[p[0]], p[1]), rank(p[0])))
            self.reviewer.review(unseen[:6])
            self.last_reviewed[cell] = generation
        edits = []
        result = survivors.copy()
        for cell, rows in groups.items():
            for row in rows:
                grades = self.reviewer.grades.get(tuple(row["pair"]))
                if grades is not None:
                    self.records[cell].setdefault(row["scaffold"], {
                        "pair": row["pair"], "dimensions": grades, "score": score(row["quality"], grades)})
            self.propose(generation, cell)
            dimension = self.plans.get(cell, {}).get("dimension")
            present = result[result["qd_cell"] == cell]
            if present.empty:
                continue
            selected = []
            for row in rows:
                grades = self.reviewer.grades.get(tuple(row["pair"]))
                if grades is not None:
                    selected.append((row, grades, score(row["quality"], grades)))
            selected.sort(key=lambda item: (-item[2]["low"], rank(item[0]["smi"])))
            # Reserve at most two existing positions in a split cell, otherwise one.
            elite = []
            if dimension and len(present) >= 2:
                for band in ("high", "low"):
                    best = next((r for r, g, s in selected if s["usable"] and subcell(dimension, g) == band), None)
                    if best:
                        elite.append((best, band))
            if not elite:
                best = next((r for r, _, s in selected if s["high_f"]), None)
                if best:
                    elite.append((best, None))
            protected = {row["smi"] for row, _ in elite}
            slots = list(ga._sort_by_pareto_priority(present[~present["smi"].isin(protected)]).index[::-1])
            for (row, band) in elite:
                if row["smi"] in set(result["smi"]) or not slots:
                    continue
                old = slots.pop(0)
                entrant = ranked.loc[ranked["smi"].eq(row["smi"])].iloc[0]
                previous = result.loc[old, "smi"]
                result.loc[old] = entrant.reindex(result.columns)
                edits.append({"cell": cell, "retained": row["smi"], "replaced": previous, "subcell": band,
                              "dimension": dimension, "performance": score(
                                  row["quality"], self.reviewer.grades[tuple(row["pair"])]),
                              "bio_route": source_status(row, self.reviewer.evidence.values())})
        self.replacements += len(edits)
        atomic_json(self.output / f"qd_guidance_{generation:04d}.json",
                    {"generation": generation, "reviewed_pairs": len(self.reviewer.grades),
                     "calls": self.reviewer.calls, "edits": edits, "plans": self.plans})
        return result.reset_index(drop=True)


def cell_priority(records, unseen):
    # Prefer an evidence gap near a cell already being evaluated, then unexplored cells.
    return (0 if 0 < len(records) < 8 else 1, -len(unseen))


def parent_archive(frame):
    return [item for row in frame.to_dict("records") if row.get("qd_cell") and (item := candidate(row))]
