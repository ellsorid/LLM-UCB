"""Blind reviews, scaffold feedback, and spatial estimates for a fresh GA run."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import ndtr

from bayesian_audit import rank, summary
from dashboard import ARMS, NX, NY
from extended_audit import HERE, PROMPT, agents, atomic_json, display_bin, read_json, scores, validate

# Display cells overlapping the proxy target region: H₂ >= 4.2 wt%, 40 <= ΔH <= 70.
TARGET = [(x, y) for x in range(7, NX) for y in range(5, 11)]
FIXED_CAPS = (10000, 18000, 26000)
TERMINAL = {"已完成", "已停止", "已立即停止", "运行失败"}
SCAFFOLD_SD, PAIR_SD, SPACE_SD, PRIOR_SD = 15, 15, 14, 25
DRAWS = 4096


def held_out(lean):
    return int(rank("evaluation", lean)[:8], 16) % 3 == 0


def candidate(row):
    scaffold = row.get("scaffold_smi")
    if not isinstance(scaffold, str) or not scaffold:
        mol = row.get("mol")
        if not isinstance(mol, Chem.Mol):
            mol = Chem.MolFromSmiles(str(row["smi"]))
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else ""
    return {"pair": [str(row["smi"]), str(row["sat_smi"])],
            "cell": int(row["chemical_behavior_cell"]),
            "scaffold": scaffold or "(acyclic)",
            "quality": float(row["qd_quality"]),
            "properties": {name: float(row[name]) for name in ("h2_wt_percent", "delta_H_per_H2", "SA score")}}


def key(row):
    return row["cell"], row["scaffold"]


def label(cell):
    return f"{cell[0]},{cell[1]}"


def usage(directory):
    total = Counter()
    for path in directory.glob("*_audit.json"):
        row = read_json(path, {})
        total.update({k: v for k, v in (row.get("usage") or {}).items() if type(v) is int})
    return dict(total)


class Reviews:
    def __init__(self, key_file, output, evidence, request_limit):
        output.mkdir()
        self.output = output
        self.client = agents.DeepSeek(agents.credential(key_file), output)
        self.evidence = [{k: r[k] for k in ("id", "lean_smiles", "rich_smiles", "finding", "doi")} for r in evidence]
        self.evidence_ids = {r["id"] for r in self.evidence}
        self.limit = request_limit
        self.calls = 0
        self.attempted = set()
        self.results = {}
        self.failures = 0
        atomic_json(output / "protocol.json", {"model": "deepseek-flash", "temperature": 0, "max_tokens": 1100,
                                              "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                                              "evidence": self.evidence, "request_limit": request_limit})

    def review(self, rows):
        pending = {tuple(row["pair"]): row for row in rows if tuple(row["pair"]) not in self.attempted}
        rows = list(pending.values())[:max(0, (self.limit - self.calls) * 3)]
        for row in rows:
            self.attempted.add(tuple(row["pair"]))
        packets = []
        for offset in range(0, len(rows), 3):
            batch = rows[offset:offset + 3]
            packets.append(batch)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for _ in range(2):
                count = min(len(packets), self.limit - self.calls)
                jobs = [(self.calls + i + 1, batch) for i, batch in enumerate(packets[:count])]
                self.calls += count
                packets = self.collect(pool, jobs)
                if not packets:
                    break
        self.failures = len(self.attempted - self.results.keys())

    def collect(self, pool, jobs):
        retry = []
        for (_, batch), result in zip(jobs, pool.map(self.ask, jobs)):
            if result is None:
                retry.append(batch)
            else:
                self.results.update(result)
        return retry

    def ask(self, job):
        index, rows = job
        role = f"blind_{index:03d}"
        mapped = {f"B{index:03d}-{i}": row for i, row in enumerate(rows)}
        packet = {"pairs": [{"id": ident, "lean": row["pair"][0], "rich": row["pair"][1],
                             "proxy_properties": {k: v for k, v in row["properties"].items()
                                                  if k != "delta_H_per_H2"}}
                            for ident, row in mapped.items()], "evidence": self.evidence}
        atomic_json(self.output / f"{role}_input.json", packet)
        try:
            result = validate(self.client.run(role, PROMPT, packet), set(mapped), self.evidence_ids)
        except (OSError, ValueError, TypeError, RuntimeError, KeyError, IndexError) as error:
            atomic_json(self.output / f"{role}_failure.json", {"error": type(error).__name__})
            return None
        return {tuple(mapped[r["id"]]["pair"]): r for r in result}


class Feedback:
    def __init__(self, reviews):
        self.reviews = reviews
        self.rows = {}
        self.groups = defaultdict(dict)
        self.grades = defaultdict(dict)
        self.recent = Counter()
        self.last = -2
        self.novel = set()
        self.changed = 0

    def observe(self, rows, novel):
        self.novel = set(novel)
        for row in rows:
            lean = row["pair"][0]
            if lean not in self.rows:
                self.rows[lean] = row
                if not held_out(lean):
                    self.groups[key(row)][lean] = row

    def adjust(self, transitions, generation, operators):
        result = transitions.copy()
        details = []
        used = set()
        for index, tr in transitions.iterrows():
            parent, child = tr["parent_smi"], tr["child_smi"]
            if not tr["child_valid"] or child not in self.novel or child in used:
                continue
            a, b = self.rows.get(parent), self.rows.get(child)
            if a is None or b is None or tr["mutation_operator"] not in operators or held_out(child) or held_out(parent):
                continue
            used.add(child)
            ka, kb = key(a), key(b)
            if ka == kb:
                continue
            self.recent.update((ka, kb))
            if len(self.grades.get(ka, {})) < 2 or len(self.grades.get(kb, {})) < 2:
                continue
            ma, va = self.posterior(ka)
            mb, vb = self.posterior(kb)
            diff = mb - ma
            prob = float(ndtr(diff / math.sqrt(va + vb)))
            if 0.20 < prob < 0.80:
                continue
            before = float(tr["operator_reward"])
            after = float(np.clip(before + np.clip(diff / 100, -.10, .10), -1, 1))
            if before == after:
                continue
            result.loc[index, "operator_reward"] = after
            details.append({"generation": generation, "scored_through_generation": self.last, "parent": parent,
                            "child": child, "operator": tr["mutation_operator"], "context": tr["experience_context"],
                            "parent_group": list(ka), "child_group": list(kb), "before": before, "after": after,
                            "diff": diff, "prob_improvement": prob})
        self.changed += len(details)
        return result, details

    def posterior(self, group):
        values = self.grades[group]
        var = 1 / (1 / SCAFFOLD_SD**2 + len(values) / PAIR_SD**2)
        mean = var * (50 / SCAFFOLD_SD**2 + sum(values.values()) / PAIR_SD**2)
        return mean, var

    def refresh(self, generation):
        if generation - self.last < 2 or self.reviews.calls >= self.reviews.limit:
            return
        picks = []
        groups = sorted(self.recent, key=lambda k: (-self.recent[k], rank(k)))
        for group in groups:
            pool = [r for r in self.groups[group].values() if tuple(r["pair"]) not in self.reviews.attempted]
            need = max(0, 2 - len(self.grades[group]))
            if not need or len(pool) < need:
                continue
            picks.extend(sorted(pool, key=lambda r: rank(r["pair"]))[:need])
            if len(picks) >= 6:
                break
        if len(picks) < 3:
            return
        self.reviews.review(picks)
        self.last = generation
        self.recent.clear()
        for row in picks:
            review = self.reviews.results.get(tuple(row["pair"]))
            grade = scores(row, review)["composite"]
            if grade is not None:
                self.grades[key(row)][row["pair"][0]] = grade
        atomic_json(self.reviews.output / "posterior.json",
                    {"generation": generation, "requests": self.reviews.calls, "failed_pairs": self.reviews.failures,
                     "usage": usage(self.reviews.output),
                     "groups": [{"cell": c, "scaffold": s, "scores": data, "mean": self.posterior((c, s))[0]}
                                for (c, s), data in self.grades.items() if data]})


def spatial_model(groups, reviews, rng):
    counts = {b: Counter(r["scaffold"] for r in rows) for b, rows in groups.items()}
    surface, exact = [], defaultdict(lambda: defaultdict(list))
    for b, rows in groups.items():
        count = counts[b]
        n = len(rows)
        entropy = -sum((v / n) * math.log(v / n) for v in count.values())
        surface.append([*b, round(math.exp(entropy), 5), n, len(count)])
        for row in rows:
            review = reviews.get(tuple(row["pair"]))
            if review:
                grade = scores(row, review)["composite"]
                if grade is not None:
                    exact[b][row["scaffold"]].append(grade)
    cells = [b for b in TARGET if b in counts]
    if not cells or not exact:
        return surface, {}, {}
    index = {b: i for i, b in enumerate(cells)}
    precision = np.eye(len(cells)) / PRIOR_SD**2
    info = np.full(len(cells), 50 / PRIOR_SD**2)
    neighbors = defaultdict(list)
    for b, i in index.items():
        for adj in ((b[0] + 1, b[1]), (b[0], b[1] + 1)):
            if adj not in index:
                continue
            p, q = counts[b], counts[adj]
            overlap = sum(min(p[s] / len(groups[b]), q[s] / len(groups[adj])) for s in p.keys() & q.keys())
            weight = (.25 + .75 * overlap) / SPACE_SD**2
            j = index[adj]
            precision[i, i] += weight
            precision[j, j] += weight
            precision[i, j] -= weight
            precision[j, i] -= weight
            neighbors[b].append(adj)
            neighbors[adj].append(b)
        for grades in exact.get(b, {}).values():
            weight = 1 / (SCAFFOLD_SD**2 + PAIR_SD**2 / len(grades))
            precision[i, i] += weight
            info[i] += weight * np.mean(grades)
    factor, lower = cho_factor(precision, lower=True)
    mean = cho_solve((factor, lower), info)
    draws = np.clip(mean[:, None] + solve_triangular(factor, rng.normal(size=(len(cells), DRAWS)),
                                                     trans="T", lower=True), 0, 100)
    distance = {b: 0 for b in exact}
    queue = deque(distance)
    while queue:
        b = queue.popleft()
        for adj in neighbors[b]:
            if adj not in distance:
                distance[adj] = distance[b] + 1
                queue.append(adj)
    field, supported = {}, {}
    for b, steps in distance.items():
        values = draws[index[b]]
        stat = summary(values)
        n = sum(map(len, exact.get(b, {}).values()))
        sd = float(np.std(values))
        reliability = max(0, min(1, (25 - sd) / 14)) * math.exp(-.28 * steps)
        field[label(b)] = {**stat, "exact": n, "distance": steps, "reliability": reliability}
        if steps <= 2 and sd <= 20:
            supported[b] = values
    return surface, field, supported


def archive(root, cutoffs):
    data = {}
    for arm, cutoff in cutoffs.items():
        bins = defaultdict(list)
        for generation in range(1, cutoff["generation"] + 1):
            path = root / arm / f"spatial_candidates_{generation:04d}.json"
            rows = read_json(path)
            if rows is None:
                raise ValueError(f"Completed generation lacks {path.name}")
            for row in rows:
                p = row["properties"]
                dh = p["delta_H_per_H2"]
                b = display_bin(p["h2_wt_percent"], math.nextafter(dh, -math.inf) if dh == 70 else dh)
                if b is not None and b in TARGET:
                    bins[b].append(row)
        data[arm] = bins
    return data


def allocate(data, reviews, limit):
    shared = data["native"].keys() & data["llm"].keys()
    rng = np.random.default_rng(77)
    _, field_a, _ = spatial_model(data["native"], results_for(reviews, "native"), rng)
    _, field_b, _ = spatial_model(data["llm"], results_for(reviews, "llm"), rng)
    used = set()
    picks = []
    bins = sorted(shared, key=lambda b: (max(field_a.get(label(b), {}).get("reliability", 0),
                                             field_b.get(label(b), {}).get("reliability", 0)), rank(b)))
    for repeat in range(2):
        for b in bins:
            selected = []
            arms = ARMS if int(rank(b)[0], 16) % 2 == 0 else tuple(reversed(ARMS))
            for arm in arms:
                rows = [r for r in data[arm][b] if held_out(r["pair"][0]) and tuple(r["pair"]) not in used
                        and tuple(r["pair"]) not in reviews.attempted
                        and all(r["pair"] != s["pair"] for _, s in selected)]
                if rows:
                    scored = Counter(r["scaffold"] for r in data[arm][b]
                                     if reviews.assignments.get(tuple(r["pair"])) == arm and tuple(r["pair"]) in reviews.results)
                    selected.append((arm, min(rows, key=lambda r: (scored[r["scaffold"]], rank(r["pair"])))))
            if len(selected) == 2:
                used.update(tuple(r["pair"]) for _, r in selected)
                picks.extend(selected)
            if len(picks) >= limit:
                return picks
    return picks


def results_for(reviews, arm):
    return {pair: result for pair, result in reviews.results.items() if reviews.assignments.get(pair) == arm}


def checkpoint(root, cap, cutoffs, reviews, batch_pairs):
    data = archive(root, cutoffs)
    limit = min(batch_pairs, (reviews.limit - reviews.calls) * 3)
    if limit >= 2:
        chosen = allocate(data, reviews, limit - limit % 2)
        reviews.assignments.update({tuple(r["pair"]): arm for arm, r in chosen})
        reviews.review([r for _, r in chosen])
        atomic_json(reviews.output / "assignments.json",
                    [{"pair": pair, "arm": arm} for pair, arm in reviews.assignments.items()])
    rng = np.random.default_rng(11)
    field, supported, surfaces = {}, {}, {}
    for arm in ARMS:
        surfaces[arm], field[arm], supported[arm] = spatial_model(data[arm], results_for(reviews, arm), rng)
    common = sorted(supported["native"].keys() & supported["llm"].keys())
    delta = {}
    for b in common:
        coord = label(b)
        delta[coord] = {**summary(supported["llm"][b] - supported["native"][b], delta=True),
                        "reliability": min(field[arm][coord]["reliability"] for arm in ARMS),
                        "exact": min(field[arm][coord]["exact"] for arm in ARMS)}
    overall = {"target_cells": len(TARGET), "common_cells": len(common),
               "occupied": {arm: len(data[arm]) for arm in ARMS},
               "scored": {arm: sum(r["exact"] > 0 for r in field[arm].values()) for arm in ARMS},
               "supported": {arm: len(supported[arm]) for arm in ARMS}}
    if common:
        sums = {}
        for arm in ARMS:
            values = [supported[arm][b] for b in common]
            sums[arm] = np.mean(values, axis=0)
            overall[arm] = summary(sums[arm])
            overall[arm]["frequency"] = summary(np.average(values, axis=0, weights=[len(data[arm][b]) for b in common]))
        overall["delta"] = summary(sums["llm"] - sums["native"], delta=True)
    return {"cap": cap, "cutoffs": cutoffs, "surfaces": surfaces, "fields": {**field, "delta": delta}, "overall": overall,
            "checkpoint_data": "cumulative unique proxy hits", "blind_requests": reviews.calls,
            "blind_failures": reviews.failures, "usage": usage(reviews.output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--key-file", required=True, type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    evidence = read_json(HERE.parent / "LOHC_Grid_Guided_GA_v4" / "live_run" / "frozen_memory.json")
    reviews = Reviews(args.key_file, root / "independent_review", evidence, 40)
    reviews.assignments = {}
    report = {"stages": {}, "protocol": {"target_cells": [label(b) for b in TARGET], "caps": FIXED_CAPS,
                                       "prior_sd": PRIOR_SD, "neighbor_sd": SPACE_SD, "pair_sd": PAIR_SD,
                                       "scaffold_sd": SCAFFOLD_SD, "draws": DRAWS}}
    while True:
        statuses = {arm: read_json(root / arm / "status.json", {}) for arm in ARMS}
        history = {arm: read_json(root / arm / "progress_history.json", []) for arm in ARMS}
        terminal = all(s.get("phase") in TERMINAL for s in statuses.values())
        if all(history.values()):
            common_budget = min(rows[-1]["gnn_pairs"] for rows in history.values())
            caps = [cap for cap in FIXED_CAPS if cap <= common_budget]
            if terminal and (not caps or common_budget > caps[-1]):
                caps.append(common_budget)
            for cap in caps:
                if str(cap) in report["stages"]:
                    continue
                cutoffs = {arm: max((r for r in rows if r["gnn_pairs"] <= cap),
                                    key=lambda r: r["generation"], default=None) for arm, rows in history.items()}
                if all(cutoffs.values()):
                    stage = checkpoint(root, cap, cutoffs, reviews, batch_pairs=30)
                    report["stages"][str(cap)] = stage
                    atomic_json(root / "spatial_evaluation.json", report)
                    print(cap, stage["overall"], "requests", reviews.calls, flush=True)
        if terminal:
            return
        time.sleep(3)


if __name__ == "__main__":
    main()
