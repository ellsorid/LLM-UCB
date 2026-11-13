"""Frozen percentile scoring and one reserved position per occupied QD cell."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

from matrix_code_reward import jsonable_features
from rule_steering import atomic_json
from lohc_structure_score_trial import DeepSeek, credential, load
from validate_lohc_matrix_adjustments import cards, features


TAIL = 90.0
OBJECTIVES = ['obj_h2_score', 'obj_dh_score', 'obj_scaffold_novelty']
REVIEW_LIMIT = 4
PROMPT = '''审核已通过硬筛的LOHC Pareto代表。site_facts来自两态矩阵；已有的score是相同量表下的基准。用文献卡校准双向路径或目标加氢选择性。只允许有具体结构事实和可核验引用时给综合分小修正，整数-3至+3。Guaiacol研究只能支撑芳香含氧位点在特定催化条件下的条件性风险。甲基吲哚热化学不直接标定选择性。无法提出适用证据时delta=null。不得由候选排名推断实验性能。返回严格JSON {"reviews":[{"id":"R1","delta":null,"reference":null,"field":null,"reason":"原因"}]}。非零delta时reference必须是文献卡ID，field必须是该候选site_facts中的字段；不得修改输入结构。逐项返回。'''


class EvidenceReview:
    def __init__(self, key, output):
        self.output = output
        output.mkdir()
        self.api = DeepSeek(credential(key), output)
        self.cards = [r for r in cards() if r['id'] in ('NEC-CYCLE', 'NEC-SELECT', 'GUAIACOL-HDO')]
        self.calls = self.tokens = 0
        self.reviewed, self.samples, self.rules = {}, defaultdict(dict), {}
        atomic_json(output / 'protocol.json', {'prompt_sha256': sha256(PROMPT.encode()).hexdigest(),
            'request_limit': REVIEW_LIMIT, 'model': 'deepseek-flash', 'max_delta': 3})

    def review(self, items, generation):
        if not items or self.calls >= REVIEW_LIMIT:
            return
        packet = {'literature': self.cards, 'items': [row['payload'] for row in items]}
        expected = {row['payload']['id']: row for row in items}
        for row in items:
            self.reviewed[row['pair']] = None
        self.calls += 1
        role = f'pareto_{generation:04d}'
        atomic_json(self.output / f'{role}_input.json', packet)
        try:
            reply = self.api.run(role, PROMPT, packet)['reviews']
            if len(reply) != len(expected) or {r['id'] for r in reply} != set(expected):
                raise ValueError('Review IDs do not match the request')
            for row in reply:
                candidate = expected[row['id']]
                delta, ref, field = row['delta'], row['reference'], row['field']
                if not isinstance(row['reason'], str) or not row['reason'].strip():
                    raise ValueError('Missing review reason')
                if delta is not None and (type(delta) is not int or abs(delta) > 3):
                    raise ValueError('Invalid score adjustment')
                if delta and not (ref == 'GUAIACOL-HDO' and delta < 0
                                  and field in ('phenol_neighborhood', 'aryl_ether_neighborhood')
                                  and candidate['payload']['site_facts'].get(field)):
                    row['rejected_delta'] = delta
                    row['delta'] = None
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
            atomic_json(self.output / f'{role}_failure.json', {'error': type(exc).__name__})
            return
        finally:
            audit = self.output / f'{role}_audit.json'
            if audit.exists():
                self.tokens += (load(audit).get('usage') or {}).get('total_tokens', 0)
        for row in reply:
            item = expected[row['id']]
            self.reviewed[item['pair']] = row['delta']
            self.samples[item['signature']][item['scaffold']] = row['delta']
        self.rules = {key: next(iter(deltas.values())) for key, deltas in self.samples.items()
                      if len(deltas) >= 2 and len(set(deltas.values())) == 1 and next(iter(deltas.values()))}
        atomic_json(self.output / f'{role}_accepted.json', {'reviews': reply,
                    'rules': [{'pattern': k[0], 'site_facts': list(k[1:]), 'delta': v} for k, v in self.rules.items()]})

    def adjustment(self, pair, signature):
        return (self.reviewed[pair] or 0) if pair in self.reviewed else self.rules.get(signature, 0)


class PercentileSelection:
    def __init__(self, book, reference, output, reviewer=None):
        self.book, self.output, self.reviewer = book, output, reviewer
        self.reference = sorted(load(reference)['scores'])
        if not self.reference:
            raise ValueError('Frozen percentile reference is empty')
        self.previous = Counter()
        self.ever_generated = set()
        self.ever_retained = set()
        self.max_streak = 0
        self.stats = {}
        self.search_reference = self.reference
        self.recent = deque(maxlen=3)

    @staticmethod
    def rank(reference, score):
        return 50.0 * (bisect_left(reference, score) + bisect_right(reference, score)) / len(reference)

    def percentile(self, score):
        return self.rank(self.reference, score)

    def is_high(self, score):
        return score is not None and self.rank(self.search_reference, score) >= TAIL

    def grade(self, pair):
        result = self.book.evaluate(pair)
        return None if result is None else {**result, 'percentile': self.percentile(result['score'])}

    @lru_cache(maxsize=10000)
    def site(self, pair):
        facts = features(jsonable_features(pair, include_matrix=True)['matrix'], include_neighborhood=True)
        fields = {k: v for k, v in facts.items() if k.endswith('_neighborhood') or k == 'guaiacol_like'}
        return fields, (self.book.evaluate(pair)['pattern'], *fields.values())

    def live_score(self, pair, grade):
        reviewer = self.reviewer
        if reviewer is None:
            return grade['score']
        if pair in reviewer.reviewed:
            delta = reviewer.reviewed[pair] or 0
        elif any(key[0] == grade['pattern'] for key in reviewer.rules):
            delta = reviewer.adjustment(pair, self.site(pair)[1])
        else:
            delta = 0
        return min(100, max(0, grade['score'] + delta))

    def steering_score(self, pair):
        grade = self.book.evaluate(pair)
        if grade is None:
            return None, None, None
        return str(grade['cluster']), grade['pattern'], self.live_score(pair, grade)

    def select(self, generation, ranked, parents, ga):
        columns = [*OBJECTIVES, '_selection_score']
        parents = parents.reset_index(drop=True).copy()
        occupied = set(parents['qd_cell']) - {''}
        ranked = ga.annotate_qd_cells(ranked).reset_index(drop=True)
        sa = pd.to_numeric(ranked['SA score'], errors='coerce')
        eligible = ranked.loc[ranked['qd_cell'].isin(occupied) & ranked['selected_hardfilter'].eq(1) & sa.le(4)].copy()
        eligible['_pair'] = list(zip(eligible['smi'], eligible['sat_smi']))
        grades = {pair: self.grade(pair) for pair in eligible['_pair']}
        eligible['_selection_score'] = [g['score'] if (g := grades[pair]) else np.nan for pair in eligible['_pair']]
        eligible = eligible.dropna(subset=columns)
        def fronts(pool):
            for cell, group in pool.groupby('qd_cell', sort=True):
                front = ga._non_dominated_sort_indices(group[columns].to_numpy(float))[0]
                yield cell, group.iloc[front].sort_values('_selection_score', ascending=False, kind='stable')

        if self.reviewer is not None:
            if generation in (3, 5, 7, 9) and self.reviewer.calls < REVIEW_LIMIT:
                representatives = []
                scaffolds = set()
                for _, front in sorted(fronts(eligible), key=lambda item: -float(item[1]['_selection_score'].iloc[0])):
                    unseen = front.loc[~front['_pair'].isin(self.reviewer.reviewed)]
                    for row in unseen.to_dict('records'):
                        scaffold, pair = row['scaffold_smi'], row['_pair']
                        if scaffold in scaffolds:
                            continue
                        facts, signature = self.site(pair)
                        representatives.append({'pair': pair, 'scaffold': scaffold, 'signature': signature,
                            'payload': {'id': f'R{len(representatives)+1}', 'pair': pair, 'site_facts': facts,
                                        'score': grades[pair]['score']}})
                        scaffolds.add(scaffold)
                        break
                    if len(representatives) >= 3:
                        break
                self.reviewer.review(representatives, generation)
            eligible['_selection_score'] = [self.live_score(pair, grades[pair]) for pair in eligible['_pair']]

        edits = []
        for cell, front in fronts(eligible):
            # Keep the native order as the tie-break within equal structural scores.
            leader = ga._sort_by_pareto_priority(front.loc[front['_selection_score'].eq(front['_selection_score'].iloc[0])]).iloc[0]
            members = parents.loc[parents['qd_cell'].eq(cell)]
            if leader['smi'] in set(parents['smi']):
                continue
            slot = ga._sort_by_pareto_priority(members).index[-1]
            edits.append({'cell': cell, 'added': leader['smi'], 'removed': parents.at[slot, 'smi'],
                          'frozen': grades[leader['_pair']], 'selection_score': float(leader['_selection_score']),
                          'live_percentile': self.rank(self.search_reference, float(leader['_selection_score']))})
            parents.loc[slot] = leader.reindex(parents.columns)
        atomic_json(self.output / f'percentile_elites_{generation:04d}.json', edits)
        if self.reviewer is not None:
            self.recent.append(set(eligible['_pair']))
            active_pairs = set().union(*self.recent)
            if active_pairs:
                self.search_reference = sorted(self.live_score(pair, self.book.evaluate(pair)) for pair in active_pairs)
            atomic_json(self.output / f'live_reference_{generation:04d}.json', {'scores': self.search_reference,
                        'scored_pairs_in_window': len(active_pairs), 'active_rules': len(self.reviewer.rules)})
        return parents, len(edits)

    def observe(self, generation, latest, parents, replacements):
        generated = {tuple((row['smi'], row['rich_proxy_smi'])) for row in latest if row['new_native_hit']}
        retained = {tuple(row['pair']) for row in parents}
        high_generated = {p for p in generated if (g := self.grade(p)) and g['percentile'] >= TAIL}
        high_retained = {p for p in retained if (g := self.grade(p)) and g['percentile'] >= TAIL}
        self.previous = Counter({pair: self.previous[pair] + 1 for pair in high_retained})
        self.ever_generated.update(high_generated)
        self.ever_retained.update(high_retained)
        self.max_streak = max(self.max_streak, max(self.previous.values(), default=0))
        scores = [g['percentile'] for pair in retained if (g := self.grade(pair))]
        self.stats = {'high_generated': len(high_generated), 'high_generated_total': len(self.ever_generated),
                      'high_retained': len(high_retained), 'high_retained_total': len(self.ever_retained),
                      'high_retained_streak': self.max_streak, 'percentile_mean': mean(scores) if scores else None,
                      'elite_replacements': replacements, 'scoring_agent_calls': self.reviewer.calls if self.reviewer else 0,
                      'scoring_agent_tokens': self.reviewer.tokens if self.reviewer else 0,
                      'scoring_agent_rules': len(self.reviewer.rules) if self.reviewer else 0}
        atomic_json(self.output / f'percentile_tail_{generation:04d}.json', {**self.stats,
                    'generated': sorted(high_generated), 'retained': [list(p) + [n] for p, n in sorted(self.previous.items())]})


if __name__ == '__main__':
    import argparse
    from validate_lohc_matrix_adjustments import OUT as ADJUSTMENTS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = ADJUSTMENTS / 'results.json'
    scores = [mean(row['adjusted']) for row in load(source)['candidate_scores'] if row['adjusted'] is not None]
    atomic_json(args.output, {'scores': sorted(scores), 'source_sha256': sha256(source.read_bytes()).hexdigest(),
                             'high_percentile': TAIL, 'tie_method': 'midrank'})
