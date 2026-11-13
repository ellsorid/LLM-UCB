"""Score sampled real edits and credit only the corresponding UCB transition."""

from collections import Counter
import hashlib
import math
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolHash

from spatial_feedback import agents, atomic_json, held_out, rank
from extended_audit import read_json

WEIGHTS = {"reversibility": 35, "selectivity": 25, "cycling": 25, "liquid": 15}
MANDATORY = {"reversibility", "selectivity"}
COEFFICIENT = 0.20
VERSION = "lohc-paired-edit-v2"
PROMPT = """你盲评两种候选 LOHC。每个 edit 的 A、B 各包含放氢态 lean 和储氢态 rich；A/B 的次序随机。用同一尺度并排评价 A 和 B，不偏好任一标签。判断结构证据，不填物性数值或确认实测结论。
reversibility: 目标加/脱氢能否在结构上闭合，骨架和关键官能团能否保留。
selectivity: 竞争还原位点、键断裂或其他副反应。
cycling: 重复储放氢中累积降解的结构风险，与单次反应选择性区分。
liquid: 两态结晶、流动性不足或挥发的结构风险。
各维分别给 A、B 整数 0 明显不利、1 偏不利、2 利弊并存、3 偏有利、4 有强证据支持；无法判断填 null。类比仅作结构判断线索。即使有文献，也不从 SMILES 确认生物基来源。
仅返回 JSON {"assessments":[{"id":"复制输入","change":"描述 A/B 的一个具体结构差别","dimensions":{"reversibility":{"a":2,"b":3,"basis":"结构依据","level":"structure","evidence_ids":[]},"selectivity":同格式,"cycling":同格式,"liquid":同格式}}]}。
level 只能是 direct（文献同分子）、analog（文献类似物）、structure（结构推断）或 unknown（两端均未知）。direct/analog 必须引用输入的文献 ID，其他两级用空列表。basis 每项一句简短中文。"""


def validate(raw, expected, evidence_ids):
    rows = raw["assessments"]
    if set(raw) != {"assessments"} or len(rows) != len(expected) or {r["id"] for r in rows} != expected:
        raise ValueError("Unexpected edit IDs")
    for row in rows:
        if set(row) != {"id", "change", "dimensions"} or set(row["dimensions"]) != WEIGHTS.keys():
            raise ValueError("Unexpected assessment fields")
        if not isinstance(row["change"], str) or not row["change"].strip():
            raise ValueError("Missing structure change")
        for grade in row["dimensions"].values():
            if set(grade) != {"a", "b", "basis", "level", "evidence_ids"}:
                raise ValueError("Unexpected dimension fields")
            if any(s is not None and (type(s) is not int or not 0 <= s <= 4) for s in (grade["a"], grade["b"])):
                raise ValueError("Invalid grade")
            if not isinstance(grade["basis"], str) or not grade["basis"].strip():
                raise ValueError("Missing evidence basis")
            level = grade["level"]
            ids = grade["evidence_ids"]
            if level not in {"direct", "analog", "structure", "unknown"} or not isinstance(ids, list):
                raise ValueError("Invalid evidence level")
            if not set(ids) <= evidence_ids or bool(ids) != (level in {"direct", "analog"}):
                raise ValueError("Invalid evidence citation")
            if level == "unknown" and (grade["a"] is not None or grade["b"] is not None):
                raise ValueError("Unknown dimension supplied grades")
    return {row["id"]: row for row in rows}


def difference(review, reversed_order=False):
    dimensions = review["dimensions"]
    usable = {name for name, item in dimensions.items() if item["a"] is not None and item["b"] is not None}
    coverage = sum(WEIGHTS[name] for name in usable)
    if coverage < 75 or not MANDATORY <= usable:
        return None, coverage
    delta = sum(weight * (dimensions[name]["b"] - dimensions[name]["a"])
                for name, weight in WEIGHTS.items() if name in usable) / 400
    return -delta if reversed_order else delta, coverage


def agreement(edit, forward, reverse):
    if not forward or not reverse:
        return None, 0, "api_fallback"
    a, ca = difference(forward, edit["reverse"])
    b, cb = difference(reverse, not edit["reverse"])
    coverage = min(ca, cb)
    if a is None or b is None:
        return None, coverage, "insufficient_evidence"
    if a == b == 0:
        return 0.0, coverage, "unchanged"
    if a * b <= 0:
        return None, coverage, "order_disagreement"
    return (a + b) / 2, coverage, "accepted"


class PairedReviews:
    def __init__(self, key_file, output, evidence, limit):
        output.mkdir()
        self.output = output
        self.client = agents.DeepSeek(agents.credential(key_file), output)
        self.evidence = [{k: r[k] for k in ("id", "lean_smiles", "rich_smiles", "finding", "doi")} for r in evidence]
        self.evidence_ids = {row["id"] for row in self.evidence}
        self.limit = limit
        self.calls = 0
        self.tokens = 0
        atomic_json(output / "protocol.json", {
            "version": VERSION, "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "model": "deepseek-flash", "temperature": 0, "thinking": "disabled", "max_tokens": 1100,
            "weights_percent": WEIGHTS, "minimum_common_percent": 75, "mandatory": sorted(MANDATORY),
            "coefficient": COEFFICIENT, "request_limit": limit, "evidence": self.evidence,
            "order_check": "average two orientations only if the parent-to-child signs agree"})

    def review(self, edits):
        def pair(value):
            return {"lean": value[0], "rich": value[1]}

        packet = {"edits": [{"id": e["id"], "A": pair(e["child"] if e["reverse"] else e["parent"]),
                             "B": pair(e["parent"] if e["reverse"] else e["child"])}
                            for e in edits], "evidence": self.evidence}
        for _ in range(2):
            if self.calls >= self.limit:
                break
            self.calls += 1
            role = f"paired_{self.calls:03d}"
            atomic_json(self.output / f"{role}_input.json", packet)
            try:
                result = validate(self.client.run(role, PROMPT, packet), {e["id"] for e in edits}, self.evidence_ids)
                return result
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError, IndexError) as exc:
                atomic_json(self.output / f"{role}_failure.json", {"error": type(exc).__name__})
            finally:
                audit = read_json(self.output / f"{role}_audit.json", {})
                self.tokens += (audit.get("usage") or {}).get("total_tokens", 0)
        return {}


@lru_cache(maxsize=65536)
def checked_pair(lean, rich):
    if isinstance(lean, str) and lean and isinstance(rich, str) and rich:
        a, b = Chem.MolFromSmiles(lean), Chem.MolFromSmiles(rich)
        if a is None or b is None or rdMolHash.MolHash(a, rdMolHash.HashFunction.ElementGraph) != rdMolHash.MolHash(
                b, rdMolHash.HashFunction.ElementGraph):
            return None
        net = sum(atom.GetTotalNumHs() for atom in b.GetAtoms()) - sum(atom.GetTotalNumHs() for atom in a.GetAtoms())
        if net > 0 and net % 2 == 0:
            return (Chem.MolToSmiles(a), Chem.MolToSmiles(b))
    return None


def pair_from(row):
    return checked_pair(row.get("smi"), row.get("sat_smi"))


class PairedFeedback:
    def __init__(self, reviews):
        self.reviews = reviews
        self.parents = {}
        self.children = {}
        self.by_group = Counter()
        self.by_scaffold = Counter()
        self.sampled = 0
        self.changed = 0
        self.fallbacks = 0
        self.usable = 0

    def set_parents(self, frame):
        self.parents = {row["smi"]: (pair, int(row["chemical_behavior_cell"]), row.get("scaffold_smi"))
                        for row in frame.to_dict("records") if (pair := pair_from(row))}

    def set_children(self, frame, novel):
        self.children = {row["smi"]: pair for row in frame.to_dict("records")
                         if row["smi"] in novel and (pair := pair_from(row))}

    def adjust(self, transitions, generation, operators):
        if self.reviews.limit - self.reviews.calls < 2 or transitions is None or transitions.empty:
            return transitions, []
        candidates = []
        for index, row in transitions.iterrows():
            parent, child = row["parent_smi"], row["child_smi"]
            op = row["mutation_operator"]
            if (not row["child_valid"] or op not in operators or parent not in self.parents or child not in self.children
                    or parent == child or held_out(parent) or held_out(child) or not math.isfinite(row["operator_reward"])):
                continue
            pp, cell, scaffold = self.parents[parent]
            candidates.append((index, row, pp, self.children[child], (cell, op), str(scaffold)))
        edits, used = [], set()
        for _ in range(2):
            remaining = [c for c in candidates if c[3][0] not in used]
            if not remaining:
                break
            index, row, parent, child, group, scaffold = min(
                remaining, key=lambda c: (self.by_group[c[4]], self.by_scaffold[c[5]], rank(generation, c[2], c[3])))
            used.add(child[0])
            self.by_group[group] += 1
            self.by_scaffold[scaffold] += 1
            edits.append({"id": f"G{generation}-{len(edits)+1}", "index": index, "parent": parent, "child": child,
                          "reverse": int(rank(parent, child)[0], 16) % 2 == 1,
                          "context": row["experience_context"], "operator": row["mutation_operator"],
                          "chemical_cell": group[0], "before": float(row["operator_reward"])})
        if not edits:
            return transitions, []
        result = transitions.copy()
        reviews = self.reviews.review(edits)
        reverse_reviews = self.reviews.review([{**edit, "reverse": not edit["reverse"]} for edit in edits]) if reviews else {}
        for edit in edits:
            review = reviews.get(edit["id"])
            reverse = reverse_reviews.get(edit["id"])
            delta, coverage, decision = agreement(edit, review, reverse)
            before = edit["before"]
            after = before if delta is None else float(np.clip(before + COEFFICIENT * delta, -1, 1))
            result.at[edit.pop("index"), "operator_reward"] = after
            edit.update(review=review, reverse_review=reverse, delta=delta, coverage_percent=coverage, after=after,
                        decision=("applied" if before != after else "unchanged") if decision == "accepted" else decision)
            self.sampled += 1
            self.usable += delta is not None
            self.changed += before != after
            self.fallbacks += delta is None
        return result, edits
