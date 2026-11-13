"""Rank the two completed arms together on the common matrix scoring scale."""

import argparse
from bisect import bisect_left, bisect_right
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


ARMS = ('native', 'llm')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


@lru_cache(maxsize=50000)
def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def collect(directory, generation):
    start = pd.read_csv(directory / 'initial_population.csv.gz', usecols=['smi'])
    starting = {canonical(s) for s in start['smi'].dropna().unique()}
    candidates = {}
    for index in range(1, generation + 1):
        for row in load(directory / f'matrix_candidates_{index:04d}.json'):
            pair = tuple(canonical(s) for s in row['pair'])
            if None in pair or pair[0] in starting:
                continue
            score = row['matrix_score']['score'] if row['matrix_score'] is not None else None
            if pair not in candidates or candidates[pair] is None:
                candidates[pair] = score
            elif score is not None and candidates[pair] != score:
                raise ValueError(f'Inconsistent stored matrix scores for {pair}')
    retained = {tuple(canonical(s) for s in row['pair']) for row in load(directory / f'matrix_scores_{generation:04d}.json')}
    return candidates, retained


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


def compare(run, write=True):
    configurations = [load(run / arm / 'configuration.json') for arm in ARMS]
    if not configurations[0]['matrix_book_sha256'] or configurations[0]['matrix_book_sha256'] != configurations[1]['matrix_book_sha256']:
        raise ValueError('The arms did not use the same matrix evaluation book')
    logs = {arm: load(run / arm / 'progress_history.json') for arm in ARMS}
    generation = min(log[-1]['generation'] for log in logs.values())
    graph_mode = all(c.get('m3os_graph') for c in configurations)
    cap = min(log[-1]['gnn_pairs'] for log in logs.values()) if graph_mode else None
    aligned = {arm: next((r for r in reversed(log) if r['gnn_pairs'] <= cap), {'generation': 0, 'gnn_pairs': 0})
               for arm, log in logs.items()} if graph_mode else {}
    hard_hits = {}
    if graph_mode:
        data = {}
        for arm in ARMS:
            candidates, hard_hits[arm] = collect_evaluated(run / arm, cap)
            data[arm] = candidates, parents_at(run / arm, aligned[arm]['generation'])
    else:
        data = {arm: collect(run / arm, generation) for arm in ARMS}
    membership = {arm: set(records) for arm, (records, _) in data.items()}
    union = set().union(*membership.values())
    scores = {}
    for records, _ in data.values():
        for pair, score in records.items():
            if score is None:
                continue
            if pair in scores and scores[pair] != score:
                raise ValueError(f'Arms disagree on common matrix score for {pair}')
            scores[pair] = score
    if not scores:
        raise ValueError('No common-scale candidate scores')
    ordered = sorted(scores.values())

    def percentile(score):
        return 50 * (bisect_left(ordered, score) + bisect_right(ordered, score)) / len(ordered)

    high = {pair for pair, score in scores.items() if percentile(score) >= 90}
    common = set.intersection(*membership.values())
    report = {'generation': generation, 'union_pairs': len(union), 'union_scored': len(scores),
              'union_missing': len(union) - len(scores), 'shared_pairs': len(common), 'shared_p90': len(high & common),
              'union_p90': len(high), 'p90_lowest_raw_score': min((scores[p] for p in high), default=None),
              'percentile_method': 'midrank across the deduplicated scored union',
              'candidate_population': 'Pairs recorded from the post-screen candidate stage, with starting SMILES removed.',
              'arms': {}}
    if graph_mode:
        report.update(gnn_common_cap=cap, aligned=aligned, score_percentile={str(float(v)): percentile(v) for v in set(ordered)},
                      candidate_population='Unique evaluated non-initial pairs at a common actual GNN budget.')
    for arm, (_, retained) in data.items():
        generated = membership[arm]
        report['arms'][arm] = {
            'gnn_pairs': next(row['gnn_pairs'] for row in logs[arm] if row['generation'] == generation),
            'generated': distribution(generated, scores, percentile, high),
            'exclusive': distribution(generated - common, scores, percentile, high),
            'retained': distribution(retained, scores, percentile, high),
        }
        if graph_mode:
            qualifying = set(hard_hits[arm])
            qualifying_high = qualifying & high
            longest, current = 0, {}
            for stage in range(1, aligned[arm]['generation'] + 1):
                retained_now = parents_at(run / arm, stage) & high
                current = {p: current.get(p, 0) + 1 for p in retained_now}
                longest = max(longest, max(current.values(), default=0))
            rounds = [load(f) for f in sorted((run / arm / 'm3os').glob('round_*.json'))]
            full = configurations[ARMS.index(arm)].get('m3os_full', False)
            branch_summary = {}
            if full and rounds:
                final_parents = parents_at(run / arm, generation) if write else retained
                for name in ('creative', 'rational'):
                    branches = [r.get('branch_results', {}).get(name, {}) for r in rounds]
                    branch_pairs = {tuple(p) for branch in branches for p in branch.get('pairs', [])}
                    branch_summary[name] = {'evaluated_full_run': sum(b.get('evaluated', 0) for b in branches),
                        'feasible_full_run': sum(b.get('feasible', 0) for b in branches),
                        'hard_p90_at_common_cap': len(branch_pairs & qualifying_high),
                        'retained_p90': len(branch_pairs & high & final_parents)}
            report['arms'][arm].update(hard_hits=distribution(qualifying, scores, percentile, high),
                first_p90_hit=min((hard_hits[arm][p] for p in qualifying_high), default=None),
                max_consecutive_p90_retention=longest, full_generation_retained=distribution(
                    parents_at(run / arm, generation), scores, percentile, high) if write else None,
                search={'rounds': len(rounds), 'evaluated': sum(r['m3os_evaluated'] for r in rounds),
                        'feasible': sum(r['m3os_feasible'] for r in rounds),
                        'different_from_top1': sum(r['rank'] != 1 for r in rounds),
                        'fallbacks': sum(r['fallback'] is not None for r in rounds),
                        'calls': rounds[-1]['m3os_calls'] if rounds else 0,
                        'tokens': rounds[-1]['m3os_tokens'] if rounds else 0,
                        'total_calls': logs[arm][-1]['llm_calls'], 'total_tokens': logs[arm][-1]['llm_tokens'],
                        'critic_reviewed': sum(r.get('critic_reviewed', 0) for r in rounds),
                        'critic_expected': sum(r.get('critic_expected', 0) for r in rounds),
                        'critic_applied': sum(r.get('critic_applied', 0) for r in rounds), 'branches': branch_summary})
    if not write:
        return report
    rows = [{'pair': pair, 'arms': [arm for arm in ARMS if pair in membership[arm]], 'score': scores.get(pair),
             'percentile': percentile(scores[pair]) if pair in scores else None} for pair in sorted(union)]
    for name, value in [('pooled_comparison.json', report), ('pooled_molecules.json', rows)]:
        target = run / name
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temp.replace(target)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    print(json.dumps(compare(parser.parse_args().run), ensure_ascii=False, indent=2))
