"""Low-frequency LLM operator recommendations with bounded native UCB priors."""
from collections import Counter, defaultdict, deque
import hashlib
import math

import numpy as np

from ac_qd import DIMENSIONS, evidence_records, rank, score
from extended_audit import agents, atomic_json
from spatial_feedback import usage

BONUS = 0.25
TTL = 3
MIN_RECORDED = 8
PROMPT = """你负责为一个 LOHC QD 父格选择下一批优先变异算子。只允许从 packet.allowed 选择已有算子。目标是增加符合硬约束、绝对性能分更高的后代，SA≤4只准入，焓变40–70窗内等价，勿向54靠拢。性能分F的物理部分按容量与焓变给分；四个文献基准分项是可逆性、选择性、循环稳定性和两态液相，未知保持未知。结合当前父代结构和已观察到的真实算子产出，推荐一个优先算子。存在可核对的代理改善（例如容量上升且窗内命中）时，即使缺实测也可作搜索建议；不得虚构性能分。参考文献只在相关时引用。只返回 JSON 对象，示例 {"cell":"复制输入编号","operator":null,"target":"physical","reference":null,"reason":"无足够证据时弃权"}。有建议时 operator 用 allowed 中的原名，target 只能是 physical、reversible、selectivity、cycling、liquid；非 physical 必须引用输入内确实支持该维度的 evidence ID。reason 简明说明结构或统计依据。"""


def validate_decision(raw, packet):
    if set(raw) != {"cell", "operator", "target", "reference", "reason"} or raw["cell"] != packet["cell"]:
        raise ValueError("Wrong response fields or cell")
    op = raw["operator"]
    target = raw["target"]
    if op is not None and op not in packet["allowed"]:
        raise ValueError("Operator not in shortlist")
    if target not in ("physical", *DIMENSIONS) or not isinstance(raw["reason"], str) or not raw["reason"].strip():
        raise ValueError("Missing supported goal or reason")
    refs = {r["id"]: r["scopes"] for r in packet["evidence"]}
    if raw["reference"] is not None and raw["reference"] not in refs:
        raise ValueError("Reference not in packet")
    if op is not None and target != "physical" and target not in refs.get(raw["reference"], []):
        raise ValueError("Non-physical recommendation needs in-scope evidence")
    return raw


def decide(api, output, role, packet):
    atomic_json(output / f"{role}_input.json", packet)
    return validate_decision(api.run(role, PROMPT, packet), packet)


def summarize_operators(events):
    groups = defaultdict(list)
    for row in events:
        groups[row["mutation_operator"]].append(row)
    result = {}
    for op, rows in groups.items():
        result[op] = {
            "records": len(rows), "distinct_parents": len({r["parent_smi"] for r in rows}),
            "valid_children": sum(r["child_valid"] == 1 for r in rows),
            "window_hits": sum(r["child_target_hit"] == 1 for r in rows),
            "mean_reward": round(float(np.mean([r["operator_reward"] for r in rows])), 4),
            "median_capacity_change": round(float(np.median([r["delta_h2"] for r in rows
                                                              if math.isfinite(r["delta_h2"])])), 4)
            if any(math.isfinite(r["delta_h2"]) for r in rows) else None,
        }
    return result


def probabilities(scores, temperature):
    p = np.exp(np.clip((scores - np.max(scores)) / temperature, -30.0, 30.0))
    return p / p.sum()


class Steering:
    def __init__(self, key, output, limit, operators, start_generation):
        output.mkdir()
        self.output = output
        self.api = agents.DeepSeek(agents.credential(key), output)
        self.limit = limit
        self.calls = 0
        self.operators = tuple(operators)
        self.start_generation = start_generation
        self.cells = {}
        self.events = defaultdict(lambda: deque(maxlen=180))
        self.plans = {}
        self.history = []
        self.counters = Counter()
        self.recommended = Counter()
        self.last_review = {}
        self.decisions = defaultdict(list)
        self.evidence = [{k: row.get(k) for k in ("id", "lean_smiles", "finding", "scopes", "doi")}
                         for row in evidence_records() if row["scopes"]]
        atomic_json(output / "protocol.json", {"bonus": BONUS, "ttl_generations": TTL,
                    "review_budget": limit, "first_review_after_generation": start_generation,
                    "min_transition_records": MIN_RECORDED, "model": "deepseek-flash", "temperature": 0,
                    "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()})

    def remember(self, transitions):
        if transitions is None or transitions.empty:
            return
        for row in transitions.to_dict("records"):
            cell = self.cells.get(row["parent_smi"])
            if cell and row["mutation_operator"] in self.operators:
                self.events[cell].append(row)

    def parent_frame(self, frame):
        self.cells = {r["smi"]: r["qd_cell"] for r in frame.to_dict("records") if r.get("qd_cell")}

    def refresh(self, generation, frame, last_generation, qd_reviewer):
        self.parent_frame(frame)
        if self.calls >= self.limit or generation < self.start_generation or generation >= last_generation:
            return
        candidate_cells = []
        for cell, subset in frame[frame["qd_cell"].ne("")].groupby("qd_cell"):
            if generation - self.last_review.get(cell, -20) < TTL or len(subset) < 3:
                continue
            stats = summarize_operators(self.events[cell])
            allowed = [op for op in self.operators if stats.get(op, {}).get("records", 0) >= 2
                       and stats[op]["valid_children"] >= 1]
            support = sum(stats[op]["records"] for op in allowed)
            if len(allowed) >= 2 and support >= MIN_RECORDED:
                candidate_cells.append((cell, subset, stats, allowed))
        if not candidate_cells:
            return
        cell, subset, stats, allowed = min(
            candidate_cells, key=lambda item: (self.last_review.get(item[0], -20), -len(item[1]), rank(item[0])))
        self.last_review[cell] = generation
        sample = []
        for row in subset.drop_duplicates("scaffold_smi").head(4).to_dict("records"):
            pair = row["smi"], row.get("sat_smi")
            perf = qd_reviewer.grades.get(pair)
            sample.append({"lean": pair[0], "rich": pair[1], "scaffold": row.get("scaffold_smi"),
                           "capacity_wt_pct": round(row["h2_wt_percent"], 3),
                           "enthalpy_per_h2": round(row["delta_H_per_H2"], 3),
                           "physical_score": round(row["qd_quality"], 3),
                           "literature_anchored_F": score(row["qd_quality"], perf) if perf else None})
        packet = {"cell": cell, "allowed": allowed, "operator_statistics": stats,
                  "parents": sample, "evidence": self.evidence}
        for _ in range(2):
            if self.calls >= self.limit:
                return
            self.calls += 1
            role = f"operator_{self.calls:03d}"
            try:
                decision = decide(self.api, self.output, role, packet)
            except (OSError, RuntimeError, ValueError, KeyError, AttributeError, TypeError) as exc:
                atomic_json(self.output / f"{role}_failure.json", {"error": type(exc).__name__})
                continue
            if decision["operator"] is not None:
                self.plans[cell] = {**decision, "from_generation": generation,
                                    "until_generation": generation + TTL}
            self.history.append({"review_generation": generation, **decision})
            return

    def attach(self, policy):
        original_choose = policy.choose_order
        original_scores = policy._scores

        def choose(key, generation, parent_smi=None):
            cell = self.cells.get(parent_smi)
            plan = self.plans.get(cell)
            if not plan or not plan["from_generation"] <= generation < plan["until_generation"]:
                return original_choose(key, generation, parent_smi=parent_smi)
            operator = plan["operator"]
            index = self.operators.index(operator)
            current = {}

            def adjusted_scores(context):
                base = np.asarray(original_scores(context))
                updated = base.copy()
                updated[index] += BONUS
                current["base_probability"] = float(probabilities(base, policy.temperature)[index])
                return updated

            policy._scores = adjusted_scores
            try:
                order, p = original_choose(key, generation, parent_smi=parent_smi)
            finally:
                policy._scores = original_scores
            if current:
                self.counters["steered_decisions"] += 1
                self.recommended[operator] += 1
                self.decisions[generation + 1].append({
                    "parent": parent_smi, "cell": cell, "context": key, "recommended": operator,
                    "first_choice": self.operators[order[0]], "base_probability": current["base_probability"],
                    "steered_probability": p[operator]})
            return order, p

        policy.choose_order = choose

    def write_generation(self, generation):
        data = {"decisions": self.decisions.pop(generation, []), "recommendations": self.history,
                "active_plans": self.plans, "totals": dict(self.counters), "recommendation_counts": dict(self.recommended),
                "requests": self.calls, "usage": usage(self.output)}
        atomic_json(self.output / f"generation_{generation:04d}.json", data)
        return data
