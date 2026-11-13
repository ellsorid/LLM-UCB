"""Compare C with a local A baseline at a common GNN budget."""

from bisect import bisect_left, bisect_right
from functools import lru_cache
import json

import numpy as np
from rdkit import Chem


ARMS = ('native', 'llm')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


@lru_cache(maxsize=50000)
def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def collect_evaluated(directory, cap):
    candidates, hard_hits = {}, {}
    for path in sorted(directory.glob('graph_evaluations_*.json')):
        for row in load(path):
            if row['index'] > cap or not row['new_valid_evaluation']:
                continue
            pair = (canonical(row['smi']), canonical(row['rich_proxy_smi']))
            if None not in pair:
                candidates.setdefault(pair, row['matrix_score'])
                if row['new_native_hit']:
                    hard_hits.setdefault(pair, row['index'])
    return candidates, hard_hits


def parents_at(directory, generation):
    return {tuple(canonical(s) for s in row['pair']) for row in load(directory / f'matrix_scores_{generation:04d}.json')}


def distribution(pairs, scores, percentile, high):
    values = [scores[p] for p in pairs if p in scores]
    ranks = [percentile(v) for v in values]
    count = len(pairs & high)
    return {'pairs': len(pairs), 'scored': len(values), 'missing': len(pairs) - len(values),
            'p90_count': count, 'p90_share_scored': count / len(values) if values else None,
            'median_percentile': float(np.median(ranks)) if ranks else None,
            'mean_percentile': float(np.mean(ranks)) if ranks else None,
            'max_score': max(values, default=None)}


def compare(directories):
    configurations = {arm: load(directory / "configuration.json") for arm, directory in directories.items()}
    hashes = {cfg.get("matrix_book_sha256") for cfg in configurations.values()}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("A 和 C 必须使用同一套矩阵评分标尺。")
    if not all(cfg.get("m3os_graph") for cfg in configurations.values()):
        raise ValueError("A 和 C 均需提供按 GNN 评价序号导出的候选记录。")
    logs = {arm: load(directory / "progress_history.json") for arm, directory in directories.items()}
    cap = min(log[-1]["gnn_pairs"] for log in logs.values())
    aligned = {arm: next((row for row in reversed(log) if row["gnn_pairs"] <= cap), {"generation": 0})
               for arm, log in logs.items()}
    data = {arm: collect_evaluated(directory, cap) for arm, directory in directories.items()}
    union = set().union(*(records for records, _ in data.values()))
    scores = {}
    for records, _ in data.values():
        for pair, score in records.items():
            if score is None:
                continue
            if pair in scores and scores[pair] != score:
                raise ValueError("同一分子对在 A 和 C 的矩阵评分不一致。")
            scores[pair] = score
    if not scores:
        raise ValueError("当前共同预算内还没有可用于对照的矩阵分数。")
    ordered = sorted(scores.values())

    def percentile(score):
        return 50 * (bisect_left(ordered, score) + bisect_right(ordered, score)) / len(ordered)

    high = {pair for pair, score in scores.items() if percentile(score) >= 90}
    arms = {}
    for arm, (records, hard_hits) in data.items():
        retained = parents_at(directories[arm], aligned[arm]["generation"])
        arms[arm] = {"generated": distribution(set(records), scores, percentile, high),
                     "hard_hits": distribution(set(hard_hits), scores, percentile, high),
                     "retained": distribution(retained, scores, percentile, high)}
    return {"report": {"gnn_common_cap": cap, "union_pairs": len(union), "union_scored": len(scores), "arms": arms},
            "score_percentile": {str(float(value)): percentile(value) for value in set(ordered)},
            "scores": scores}
