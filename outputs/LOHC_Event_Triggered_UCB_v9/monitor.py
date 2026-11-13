"""Audit fixed-grid mutation events and ask DeepSeek only when a gate opens."""

from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import lru_cache
import math

import numpy as np
from rdkit import Chem

PROMPT = (
    "你为固定化学网格审查是否继续使用一个局部结构奖励，UCB仍按原生上下文共享经验。"
    "只评价给定触发格。已知性质是代理值，ΔH接受窗口为40–70 kJ/mol H2。"
    "结构惩罚过多时可保留、停用，或改成只在目标结构保留且代理性质改善时给正奖励。"
    "已有记录是观察性证据，不能确认样品生物基来源或真实循环可逆性。"
    "返回严格JSON，顶层只含 cells 数组。每项只含 cell、action、motif、feedback_mode、evidence_ids、reason。"
    "action可为keep、disable或apply。apply必须从输入available_motifs选一个motif，"
    "feedback_mode为mixed或positive_only，其余动作的motif和feedback_mode为null。"
    "cell 必须逐项复制输入中对应整数，不得替换为其他编号。所有触发格恰好出现一次，"
    "证据编号只能从输入选择，reason 用简短中文。"
)


@lru_cache(maxsize=20000)
def smiles(molecule):
    return Chem.MolToSmiles(molecule)


class Monitor:
    def __init__(self, v5, native, protocol, plan, initial, catalog, evidence, llm, output):
        self.v5, self.native, self.protocol = v5, native, protocol
        self.rules = {row["cell"]: {"motif": row["motif"], "mode": "mixed"} for row in plan if row["plan"] == "split"}
        self.initial_rules = dict(self.rules)
        self.reference = self.composition(initial, catalog)
        self.evidence, self.llm, self.output = evidence, llm, output
        self.events, self.attempts, self.children = defaultdict(list), defaultdict(Counter), defaultdict(set)
        self.new_hits = defaultdict(Counter)
        self.requests, self.reviews, self.gates = 0, [], []
        self.attribution = Counter()

    @staticmethod
    def composition(population, catalog):
        counts = defaultdict(Counter)
        for smi in set(population["smi"]):
            cell = catalog.cells.get(smi)
            if cell is not None:
                motifs = catalog.members.get(smi, ())
                counts[cell][motifs[0] if motifs else "untagged"] += 1
        return counts

    @contextmanager
    def capture_attempts(self, ga, catalog):
        original = ga._apply_mutation_operator

        def counted(index, copy_mol, parent, *args, **kwargs):
            child, status = original(index, copy_mol, parent, *args, **kwargs)
            parent_smi = smiles(parent)
            cell = catalog.cells.get(parent_smi)
            if cell is None:
                return child, status
            operator = self.native.MUTATION_OPERATORS[int(index)]
            self.attempts[cell][operator, "attempts"] += 1
            if status == 1:
                child_smi = smiles(child)
                if child_smi != parent_smi:
                    self.attempts[cell][operator, "structure_changed"] += 1
                    if child_smi in self.children[cell] or child_smi in catalog.cells:
                        self.attempts[cell][operator, "duplicate_proposals"] += 1
                    self.children[cell].add(child_smi)
            return child, status

        ga._apply_mutation_operator = counted
        try:
            yield
        finally:
            ga._apply_mutation_operator = original
            smiles.cache_clear()

    def adjust(self, transitions, catalog):
        adjusted = transitions.copy()
        counts = Counter()
        for index, row in transitions.iterrows():
            parent, child = row["parent_smi"], row["child_smi"]
            cell = catalog.cells.get(parent)
            if cell is None:
                continue
            selected = self.rules.get(cell)
            signal = 0
            if selected:
                rule = {"cell": cell, "motif": selected["motif"], "mode": "retain_on_proxy_improvement"}
                signal = self.v5.reward_signal(row, rule, cell, 2.0)
                if selected["mode"] == "positive_only":
                    signal = max(0, signal)
            if signal:
                adjusted.loc[index, "operator_reward"] = np.clip(
                    row["operator_reward"] + self.protocol["feedback_weight"] * signal, -1, 1)
                counts[signal] += 1
                self.attribution[cell, row["experience_context"], row["mutation_operator"], signal] += 1
            if int(row["child_valid"]) != 1:
                continue
            self.events[cell].append({
                "parent": parent, "child": child, "op": row["mutation_operator"], "context": row["experience_context"],
                "motifs": catalog.members.get(parent, ()), "native_reward": float(row["operator_reward"]),
                "delta_h2": finite(row["delta_h2"]), "delta_dh_gain": finite(row["delta_dh_gain"]),
                "hit": int(row["child_target_hit"]), "signal": signal,
            })
        return adjusted, counts

    def reasons(self, cell, rows, current, available):
        p = self.protocol
        parents = {row["parent"] for row in rows}
        if (not current or len(parents) < p["minimum_distinct_parents"]
                or len(rows) < p["minimum_valid_transitions"]):
            return []
        signals = Counter(row["signal"] for row in rows)
        nonzero = signals[-1] + signals[1]
        reasons = []
        if nonzero >= p["minimum_rule_feedback"] and signals[-1] / nonzero >= p["negative_feedback_fraction"]:
            reasons.append("negative_feedback_concentrated")
        if available and (cell not in self.rules or self.rules[cell]["motif"] not in available):
            reasons.append("mature_structure_series")
        positive = [row for row in rows if row["native_reward"] > 0 and set(row["motifs"]) & set(available)]
        conflicts = [row for row in positive if (set(row["motifs"]) & set(available))
                     - set(self.v5.structure(row["child"])[1])]
        if (len(conflicts) >= p["minimum_conflicting_transitions"]
                and len({row["parent"] for row in conflicts}) >= p["minimum_conflicting_parents"]
                and len(conflicts) / len(positive) >= p["minimum_conflicting_fraction"]):
            reasons.append("native_gain_vs_structure_loss")
        previous = self.reference.get(cell, {})
        if previous:
            drift = total_variation(previous, current)
            if drift >= p["parent_composition_total_variation"]:
                reasons.append("active_parent_composition_changed")
        return reasons

    def make_case(self, cell, rows, current, reasons, available):
        per_op = defaultdict(list)
        for row in rows:
            per_op[row["op"]].append(row)
        op_summary = {}
        for op in self.native.MUTATION_OPERATORS:
            records = per_op[op]
            parents = defaultdict(list)
            for row in records:
                parents[row["parent"]].append(row["native_reward"])
            attempts = self.attempts[cell]
            if records or attempts[op, "attempts"]:
                op_summary[op] = {
                    "attempts": attempts[op, "attempts"],
                    "structure_changed": attempts[op, "structure_changed"],
                    "duplicate_proposals": attempts[op, "duplicate_proposals"],
                    "valid_logged": len(records), "distinct_parents": len(parents),
                    "parent_weighted_mean_native_reward": round(float(np.mean(
                        [np.mean(values) for values in parents.values()])), 4) if parents else None,
                    "median_delta_h2": median(records, "delta_h2"),
                    "median_delta_dh_gain": median(records, "delta_dh_gain"),
                    "current_rule_positive": sum(row["signal"] > 0 for row in records),
                    "current_rule_negative": sum(row["signal"] < 0 for row in records),
                }
        strongest = max(rows, key=lambda row: (row["native_reward"], row["parent"], row["child"]))
        weakest = min(rows, key=lambda row: (row["native_reward"], row["parent"], row["child"]))
        return {
            "cell": cell, "triggers": reasons, "active_structure_counts": dict(current),
            "current_rule": self.rules.get(cell), "available_motifs": available,
            "valid_logged": len(rows), "distinct_parents": len({row["parent"] for row in rows}),
            "new_evaluated_since_review": self.new_hits[cell]["evaluated"],
            "new_hits_since_review": self.new_hits[cell]["hits"],
            "operators": op_summary, "representative_pairs": [strongest, weakest],
        }

    def review(self, generation, can_apply, population, trace, catalog):
        for row in trace:
            if row["new_valid_evaluation"] and row["chemical_cell"] is not None:
                counts = self.new_hits[row["chemical_cell"]]
                counts["evaluated"] += 1
                counts["hits"] += int(row["new_native_hit"])
        current = self.composition(population, catalog)
        active_motifs = defaultdict(set)
        for smi in population["smi"]:
            active_motifs[catalog.cells.get(smi)].update(catalog.members.get(smi, ()))
        queue = []
        for cell, rows in self.events.items():
            per_motif = defaultdict(set)
            for row in rows:
                for motif in row["motifs"]:
                    per_motif[motif].add(row["parent"])
            available = sorted(motif for motif, parents in per_motif.items()
                               if len(parents) >= self.protocol["minimum_distinct_parents"] and motif in active_motifs[cell])
            reasons = self.reasons(cell, rows, current.get(cell), available)
            if reasons:
                queue.append(self.make_case(cell, rows, current[cell], reasons, available))
        queue.sort(key=lambda case: (-len(case["triggers"]), -case["distinct_parents"], case["cell"]))
        selected = queue[:self.protocol["maximum_cells_per_request"]] if can_apply else []
        if self.requests >= self.protocol["maximum_requests_per_trial"]:
            selected = []
        gate = {
            "generation": generation, "occupied_cells_monitored": len(catalog.totals),
            "active_cells": len(current), "qualifying_cells": [case["cell"] for case in queue],
            "requested_cells": [case["cell"] for case in selected], "has_next_generation": can_apply,
        }
        self.gates.append(gate)
        if not selected:
            print("gate", gate, flush=True)
            return
        packet = {"generation": generation, "cells": selected, "verified_evidence": self.evidence}
        role = f"event_{generation:04d}"
        self.v5.demo.write_json(self.output / f"{role}_input.json", packet)
        self.requests += 1
        error = None
        try:
            raw = self.llm.run(role, PROMPT, packet)
            decisions = validate(raw, selected, self.evidence)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError) as exc:
            decisions = []
            error = type(exc).__name__
        for result in decisions:
            cell = result["cell"]
            old = self.rules.get(cell)
            if result["action"] == "disable":
                self.rules.pop(cell, None)
            if result["action"] == "apply":
                self.rules[cell] = {"motif": result["motif"], "mode": result["feedback_mode"]}
            self.reference[cell] = current[cell].copy()
            self.events[cell].clear()
            self.attempts[cell].clear()
            self.children[cell].clear()
            self.new_hits[cell].clear()
            result.update(previous_rule=old, active_rule=self.rules.get(cell), changed=old != self.rules.get(cell))
        review = {**gate, "effective_generation": generation + 1, "decisions": decisions, "fallback": error}
        self.reviews.append(review)
        self.v5.demo.write_json(self.output / f"{role}_decision.json", review)
        print("review", generation, [(row["cell"], row["action"], row["changed"]) for row in decisions], error, flush=True)


def finite(value):
    return float(value) if math.isfinite(float(value)) else None


def median(records, key):
    values = [row[key] for row in records if row[key] is not None]
    return round(float(np.median(values)), 4) if values else None


def total_variation(a, b):
    return 0.5 * sum(abs(a.get(key, 0) / sum(a.values()) - b.get(key, 0) / sum(b.values()))
                     for key in set(a) | set(b))


def validate(raw, selected, evidence):
    expected = {row["cell"]: row for row in selected}
    allowed = {row["id"] for row in evidence}
    if set(raw) != {"cells"} or not isinstance(raw["cells"], list) or len(raw["cells"]) != len(expected):
        raise ValueError("Incomplete review")
    if {row["cell"] for row in raw["cells"]} != set(expected):
        raise ValueError("Review contains unexpected grid")
    for row in raw["cells"]:
        if set(row) != {"cell", "action", "motif", "feedback_mode", "evidence_ids", "reason"}:
            raise ValueError("Invalid review keys")
        if not isinstance(row["evidence_ids"], list) or not set(row["evidence_ids"]) <= allowed:
            raise ValueError("Unknown evidence")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("Missing review reason")
        if row["action"] == "apply":
            if (row["motif"] not in expected[row["cell"]]["available_motifs"]
                    or row["feedback_mode"] not in {"mixed", "positive_only"}):
                raise ValueError("Unsupported feedback")
        elif row["action"] not in {"keep", "disable"} or row["motif"] is not None or row["feedback_mode"] is not None:
            raise ValueError("Unsupported review action")
    return raw["cells"]
