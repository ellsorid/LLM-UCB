"""Compare fixed GNN budget checkpoints using a blinded, scaffold-stratified audit."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
from pathlib import Path
import random

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from extended_audit import ARMS, HERE, PROMPT, agents, atomic_json, compress, display_bin, read_json, scores, validate

CAPS = (15000, 26810)
BIN_LIMIT, SCAFFOLDS, PAIRS = 4, 2, 2
SEED, DRAWS = 11, 20000
MEAN, PRIOR_SD, SCAFFOLD_SD, PAIR_SD = 50, 25, 15, 15


def rank(*parts):
    return hashlib.sha256(repr((SEED, parts)).encode()).hexdigest()


def checkpoint(run, arm, cap):
    cutoff = max((row for row in read_json(run / arm / "progress_history.json")
                  if row["gnn_pairs"] <= cap), key=lambda row: row["generation"])
    bins, seen = defaultdict(lambda: defaultdict(list)), set()
    for generation in range(1, cutoff["generation"] + 1):
        for row in read_json(run / arm / f"candidates_{generation:04d}.json"):
            pair = compress(row["smi"]), compress(row["rich_proxy_smi"])
            props = row["properties"]
            key = display_bin(props["h2_wt_percent"], props["delta_H_per_H2"])
            if key is None or pair in seen:
                continue
            seen.add(pair)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(pair[0])) or "(acyclic)"
            bins[key][scaffold].append({**row, "pair": pair, "generation": generation})
    return cutoff, bins


def plan(run):
    data = {cap: {arm: checkpoint(run, arm, cap) for arm in ARMS} for cap in CAPS}
    pools = [bins for stage in data.values() for _, bins in stage.values()]
    common = set.intersection(*(set(pool) for pool in pools))
    eligible = [key for key in common if all(len(pool[key]) >= SCAFFOLDS for pool in pools)]
    eligible.sort(key=lambda key: (-min(len(pool[key]) for pool in pools), key))
    stages = {}
    for cap, stage in data.items():
        bins = {}
        for key in eligible:
            cell, used = {}, set()
            # Same priority order across budget caps; alternate the first arm by bin.
            arm_order = ARMS if int(rank(key)[0], 16) % 2 == 0 else tuple(reversed(ARMS))
            for arm in arm_order:
                groups = stage[arm][1][key]
                chosen = {}
                for scaffold in sorted(groups, key=lambda name: rank(key, name)):
                    available = [r for r in groups[scaffold] if r["pair"] not in used]
                    if available:
                        chosen[scaffold] = sorted(available, key=lambda r: rank(key, r["pair"]))[:PAIRS]
                        used.update(row["pair"] for row in chosen[scaffold])
                    if len(chosen) == SCAFFOLDS:
                        break
                cell[arm] = {"pool_counts": {s: len(rows) for s, rows in groups.items()}, "sample": chosen}
            bins[key] = cell
        surfaces = {}
        for arm, (_, groups) in stage.items():
            points = []
            for (ix, iy), scaffolds in sorted(groups.items()):
                counts = [len(rows) for rows in scaffolds.values()]
                n = sum(counts)
                effective = math.exp(-sum((count / n) * math.log(count / n) for count in counts))
                points.append([ix, iy, round(effective, 5), n, len(counts)])
            surfaces[arm] = points
        stages[cap] = {"cutoffs": {arm: values[0] for arm, values in stage.items()}, "bins": bins,
                       "surfaces": surfaces}
    keys = [key for key in eligible if all(
        len(stage["bins"][key][arm]["sample"]) == SCAFFOLDS for stage in stages.values() for arm in ARMS
    )][:BIN_LIMIT]
    if not keys:
        raise RuntimeError("No common display bin supports disjoint pairs and scaffold sampling")
    for stage in stages.values():
        stage["bins"] = {f"{key[0]},{key[1]}": stage["bins"][key] for key in keys}
    return stages, len(common), len(eligible)


def summary(draws, delta=False):
    if draws is None:
        return None
    result = {"mean": float(np.mean(draws)), "ci95": np.quantile(draws, [.025, .975]).tolist()}
    if delta:
        result["prob_gt_zero"] = float(np.mean(draws > 0))
    return result


def posterior(cell, rng, pair_sd=PAIR_SD, scaffold_sd=SCAFFOLD_SD):
    y = {s: [row["composite"] for row in rows if row["composite"] is not None]
         for s, rows in cell["sample"].items()}
    y = {s: rows for s, rows in y.items() if rows}
    if not y:
        return None, None
    precisions = {s: 1 / (scaffold_sd**2 + pair_sd**2 / len(rows)) for s, rows in y.items()}
    var = 1 / (1 / PRIOR_SD**2 + sum(precisions.values()))
    mean = var * (MEAN / PRIOR_SD**2 + sum(precisions[s] * np.mean(rows) for s, rows in y.items()))
    mu = rng.normal(mean, np.sqrt(var), DRAWS)
    theta = []
    for rows in y.values():
        local_var = 1 / (1 / scaffold_sd**2 + len(rows) / pair_sd**2)
        local_mean = local_var * (mu / scaffold_sd**2 + sum(rows) / pair_sd**2)
        theta.append(np.clip(rng.normal(local_mean, np.sqrt(local_var)), 0, 100))
    return np.mean(theta, axis=0), np.average(theta, axis=0, weights=[cell["pool_counts"][s] for s in y])


def analyze(stages, reviews):
    rng = np.random.default_rng(SEED)
    for stage in stages.values():
        deltas = defaultdict(list)
        for cell in stage["bins"].values():
            draws = {}
            for arm in ARMS:
                group = cell[arm]
                counts = group["pool_counts"]
                for rows in group["sample"].values():
                    for row in rows:
                        review = reviews.get(tuple(row["pair"]))
                        row.update(scores(row, review), review=review)
                sample = [row for rows in group["sample"].values() for row in rows]
                scored = [row for row in sample if row["composite"] is not None]
                covered = sum(counts[s] for s, rows in group["sample"].items()
                              if any(row["composite"] is not None for row in rows))
                group.update(pool_pairs=sum(counts.values()), pool_scaffolds=len(counts),
                             sampled=len(sample), exact=len(scored),
                             missing_reviews=sum(row["review"] is None for row in sample),
                             priority=sum(bool(row["review"] and row["review"]["recommendation"] == "priority")
                                          for row in sample),
                             scored_scaffold_coverage=covered / sum(counts.values()))
                draws[arm] = posterior(group, rng)
                group.update(equal=summary(draws[arm][0]), frequency=summary(draws[arm][1]))
            a, b = (cell[arm] for arm in ARMS)
            cell["scaffold_tv"] = 0.5 * sum(
                abs(a["pool_counts"].get(s, 0) / a["pool_pairs"] -
                    b["pool_counts"].get(s, 0) / b["pool_pairs"])
                for s in set(a["pool_counts"]) | set(b["pool_counts"]))
            cell["delta"] = {}
            for i, name in enumerate(("equal", "frequency")):
                diff = None if any(draws[arm][i] is None for arm in ARMS) else draws["llm"][i] - draws["native"][i]
                cell["delta"][name] = summary(diff, delta=True)
                if diff is not None:
                    deltas[name].append(diff)
        stage["delta"] = {name: summary(np.mean(items, axis=0), delta=True)
                          for name, items in deltas.items()}
        stage["scored_bins"] = len(deltas["equal"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "live")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--replay", action="store_true", help="Rebuild from recorded responses, without API calls")
    args = parser.parse_args()
    stages, common, eligible = plan(args.run)
    previous = read_json(args.run / "extended_evaluation.json")
    evidence = previous["evidence"]
    reviews = {tuple(row["pair"]): row["review"] for bins in previous["bins"].values() for row in bins.values()
               if row["review"]}
    pairs = {}
    for stage in stages.values():
        for cell in stage["bins"].values():
            for arm in ARMS:
                for rows in cell[arm]["sample"].values():
                    for row in rows:
                        pairs.setdefault(row["pair"], row["properties"])
    missing = sorted(set(pairs) - set(reviews))
    random.Random(SEED).shuffle(missing)
    print(f"Matched bins={common}, eligible={eligible}, sampled unique={len(pairs)}, "
          f"reused={len(pairs)-len(missing)}, new={len(missing)}, max requests={2*((len(missing)+2)//3)}", flush=True)
    if not args.key_file and not args.replay:
        return
    output = args.run / "bayesian_audit"
    if not args.replay:
        output.mkdir(exist_ok=False)
        atomic_json(output / "sample.json", stages)
    llm = None if args.replay else agents.DeepSeek(agents.credential(args.key_file), output)
    ids = {pair: f"B{i:03d}" for i, pair in enumerate(missing, 1)}
    evidence_ids = {row["id"] for row in evidence}

    def ask(job):
        index, group = job
        role = f"blind_{index:02d}"
        packet = {"pairs": [{"id": ids[pair], "lean": pair[0], "rich": pair[1],
                             "proxy_properties": {k: v for k, v in pairs[pair].items() if k != "delta_H_per_H2"}}
                            for pair in group], "evidence": evidence}
        input_path = output / f"{role}_input.json"
        if args.replay:
            if read_json(input_path) != packet:
                raise ValueError(f"Cannot replay mismatched request: {role}")
        else:
            atomic_json(input_path, packet)
        for suffix in ("", "_retry"):
            try:
                name = role + suffix
                raw = read_json(output / f"{name}_response.txt", {}) if args.replay else llm.run(name, PROMPT, packet)
                result = validate(raw, {ids[p] for p in group}, evidence_ids)
                return {p: row for p in group for row in result if row["id"] == ids[p]}
            except (OSError, ValueError, TypeError, RuntimeError, KeyError, IndexError):
                continue
        return {}

    batches = [missing[i:i + 3] for i in range(0, len(missing), 3)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, result in enumerate(executor.map(ask, enumerate(batches, 1)), 1):
            reviews.update(result)
            print(f"Batch {i}/{len(batches)}: {len(result)} validated", flush=True)
    analyze(stages, reviews)
    audits = [read_json(path) for path in output.glob("*_audit.json")]
    usage = Counter()
    for audit in audits:
        usage.update({k: v for k, v in (audit.get("usage") or {}).items() if type(v) is int})
    report = {"seed": SEED, "stages": stages, "common_bins": common, "eligible_bins": eligible,
              "unique_sampled": len(pairs), "reused": len(pairs) - len(missing), "new_requested": len(missing),
              "failed_pairs": len(set(missing) - set(reviews)), "requests": len(audits), "usage": dict(usage),
              "models": sorted({audit["model"] for audit in audits if audit["model"]}),
              "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
              "model_parameters": {"temperature": 0, "thinking": "disabled", "max_tokens": 1100},
              "prior": {"mean": MEAN, "sd": PRIOR_SD, "scaffold_sd": SCAFFOLD_SD, "pair_sd": PAIR_SD, "draws": DRAWS}}
    atomic_json(args.run / "bayesian_evaluation.json", report)
    for cap, stage in stages.items():
        print(cap, stage["delta"], flush=True)
    print(dict(usage), flush=True)


if __name__ == "__main__":
    main()
