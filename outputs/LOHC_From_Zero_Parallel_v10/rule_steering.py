"""Apply frozen DeepSeek rules and pass observed, grouped score gains to UCB steering."""

import argparse
from collections import Counter, defaultdict, deque
from functools import lru_cache
import hashlib
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'work'))
from lohc_cluster_rule_experiment import jsonable_features, total
from lohc_structure_score_trial import load, save

from operator_guidance import BONUS, TTL, Steering
from extended_audit import agents, atomic_json


PROMPT = '''你为LOHC分子图编辑GA选择一个已有算子。输入只含图谱簇编号、按算子汇总的父子结构评分变化和原有硬筛命中统计。优先选择有足够成对评分支持、平均分差为正且没有明显牺牲硬筛命中的算子。缺失评分不是零分。仅可选择 allowed 中的算子，证据不足返回 null。只返回 JSON，如 {"cluster":"1","operator":null,"reason":"成对数据不足"}。'''
TAIL_PROMPT = '''你根据LOHC遗传算法的汇总选择一个已有算子。目标是产生可留种的高分子代。优先比较P90子代留种率和产出率，结合样本数、可评分覆盖率、硬筛命中率判断。平均分差只作辅助；缺分保持未知。仅从allowed选择，数据不足返回null。只返回JSON，如 {"cluster":"1","operator":null,"reason":"证据不足"}。'''


def frozen_book(source, output):
    dataset = load(source / 'matrix_patterns.json')
    rule_rows = load(source / 'rules.json')
    all_features = [jsonable_features(row['pair']) for row in dataset['rows']]
    blocks = [np.array([x[name] for x in all_features]) for name in ('global', 'delta')]
    scalers = [StandardScaler().fit(block) for block in blocks]
    space = np.concatenate([s.transform(x)/np.sqrt(x.shape[1]) for s, x in zip(scalers, blocks)], axis=1)
    model = KMeans(n_clusters=4, n_init=20, random_state=11).fit(space)
    assert np.array_equal(model.labels_ + 1, [row['cluster'] for row in dataset['rows']]), 'Frozen cluster IDs changed'
    rules = {}
    for row, info in zip(dataset['rows'], all_features):
        key = tuple([row['cluster'], *info['signature']])
        value = (row['pattern'], total(rule_rows.get(row['pattern'])))
        if key in rules:
            assert rules[key] == value
        rules[key] = value
    save(output, {'centers': model.cluster_centers_.tolist(), 'means': [s.mean_.tolist() for s in scalers],
                  'scales': [s.scale_.tolist() for s in scalers],
                  'rules': [[list(key), pattern, score] for key, (pattern, score) in rules.items()],
                  'source_sha256': hashlib.sha256((source / 'rules.json').read_bytes()).hexdigest()})


class RuleBook:
    def __init__(self, path):
        data = load(path)
        self.rules = {tuple(key): (pattern, score) for key, pattern, score in data['rules']}
        self.centers = np.asarray(data['centers'])
        self.means = [np.asarray(x) for x in data['means']]
        self.scales = [np.asarray(x) for x in data['scales']]

    @lru_cache(maxsize=20000)
    def evaluate(self, pair):
        if not all(isinstance(x, str) and x for x in pair):
            return None, None, None
        try:
            features = jsonable_features(pair)
        except (ValueError, AttributeError, AssertionError, RuntimeError, IndexError):
            return None, None, None
        point = np.concatenate([(np.asarray(features[key])-mu)/sd/np.sqrt(len(mu))
                                for key, mu, sd in zip(('global', 'delta'), self.means, self.scales)])
        cluster = int(np.argmin(((self.centers-point)**2).sum(axis=1)))+1
        pattern, score = self.rules.get(tuple([cluster, *features['signature']]), (None, None))
        return str(cluster), pattern, score

    def annotate(self, parents):
        result = []
        for row in parents:
            cluster, pattern, score = self.evaluate(tuple(row['pair']))
            result.append({**row, 'cluster': cluster, 'rule': pattern, 'score': score})
        return result


class RuleSteering(Steering):
    def __init__(self, key, output, limit, operators, start_generation, book, tail=None):
        output.mkdir()
        self.output = output
        self.score_pair = tail.steering_score if tail else getattr(book, 'steering_score', book.evaluate)
        self.tail = tail
        self.prompt = TAIL_PROMPT if tail else PROMPT
        self.api = agents.DeepSeek(agents.credential(key), output)
        self.limit, self.calls, self.start_generation = limit, 0, start_generation
        self.operators = tuple(operators)
        self.cells, self.plans, self.last_review, self.parents, self.children = {}, {}, {}, {}, {}
        self.history, self.transitions = [], []
        self.events = defaultdict(lambda: deque(maxlen=350))
        self.decisions = defaultdict(list)
        self.counters, self.recommended = Counter(), Counter()
        atomic_json(output / 'protocol.json', {'bonus': BONUS, 'ttl_generations': TTL, 'review_budget': limit,
                    'prompt_sha256': hashlib.sha256(self.prompt.encode()).hexdigest(), 'model': 'deepseek-flash'})

    def parent_frame(self, frame):
        self.parents = {row['smi']: self.score_pair((row['smi'], row.get('sat_smi')))
                        for row in frame.to_dict('records') if row.get('qd_cell')}
        self.cells = {smi: result[0] for smi, result in self.parents.items() if result[2] is not None}

    def observe_children(self, frame):
        self.children = {row['smi']: row.get('sat_smi') for row in frame.to_dict('records')
                         if row.get('origin_type') == 'guided_mutation'}

    def remember(self, transitions):
        if transitions is None or transitions.empty:
            return
        for row in transitions.to_dict('records'):
            parent, child, op = row['parent_smi'], row['child_smi'], row['mutation_operator']
            cluster, _, before = self.parents.get(parent, (None, None, None))
            if cluster is None or op not in self.operators:
                continue
            after = self.score_pair((child, self.children.get(child)))[2] if row['child_valid'] else None
            gain = after-before if before is not None and after is not None else None
            event = {'parent': parent, 'child': child, 'cluster': cluster, 'operator': op,
                     'valid': bool(row['child_valid']), 'hit': bool(row['child_target_hit']), 'score_gain': gain}
            if self.tail is not None:
                event.update(scored=after is not None, high=bool(event['hit'] and self.tail.is_high(after)), retained=False)
            self.events[cluster].append(event)
            self.transitions.append(event)

    def refresh(self, generation, frame, last_generation, qd_reviewer=None):
        self.parent_frame(frame)
        if self.tail is not None:
            survivors = set(frame['smi'])
            for event in self.transitions:
                event['retained'] = event['child'] in survivors
        if not self.start_generation <= generation < last_generation or self.calls >= self.limit:
            return
        ready = []
        for cluster, parent_count in Counter(self.cells.values()).items():
            if parent_count < 3 or generation-self.last_review.get(cluster, -20) < TTL:
                continue
            stats = {}
            for operator in self.operators:
                records = [r for r in self.events[cluster] if r['operator'] == operator]
                gains = [r['score_gain'] for r in records if r['score_gain'] is not None]
                if len(gains) >= 2:
                    stats[operator] = {'records': len(records), 'paired_scores': len(gains),
                        'mean_score_gain': round(float(np.mean(gains)), 4), 'improved': sum(x > 0 for x in gains),
                        'valid_rate': sum(r['valid'] for r in records)/len(records),
                        'hard_hit_rate': sum(r['hit'] for r in records)/len(records)}
                    if self.tail is not None:
                        high = sum(r['high'] for r in records)
                        retained = sum(r['high'] and r['retained'] for r in records)
                        stats[operator].update(high_generated=high, high_retained=retained,
                            high_generation_rate=high / len(records), high_retained_rate=retained / len(records),
                            score_coverage=sum(r['scored'] for r in records) / len(records))
            signal = 'high_generated' if self.tail is not None else 'mean_score_gain'
            if len(stats) >= 2 and max(r[signal] for r in stats.values()) > 0:
                ready.append((self.last_review.get(cluster, -20), -parent_count, cluster, stats))
        if not ready:
            return
        _, _, cluster, stats = min(ready)
        self.last_review[cluster] = generation
        allowed = [k for k, v in stats.items() if v[signal] > 0]
        packet = {'cluster': cluster, 'allowed': allowed, 'operators': stats}
        self.calls += 1
        role = f'operator_{self.calls:03d}'
        atomic_json(self.output / f'{role}_input.json', packet)
        try:
            result = self.api.run(role, self.prompt, packet)
            if (set(result) != {'cluster', 'operator', 'reason'} or result['cluster'] != cluster
                    or result['operator'] not in [None, *allowed] or not isinstance(result['reason'], str)):
                raise ValueError('Invalid steering response')
        except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
            atomic_json(self.output / f'{role}_failure.json', {'error': type(exc).__name__})
            return
        if result['operator'] is not None:
            self.plans[cluster] = {**result, 'from_generation': generation, 'until_generation': generation + TTL}
        self.history.append({'review_generation': generation, **result})

    def write_generation(self, generation):
        data = super().write_generation(generation)
        selected = {(r['parent'], r['recommended']) for r in data['decisions']}
        guided = [r for r in self.transitions if (r['parent'], r['operator']) in selected]
        data['transition_summary'] = {'all': len(self.transitions), 'guided': len(guided),
            'guided_with_scores': sum(r['score_gain'] is not None for r in guided),
            'guided_improved': sum(r['score_gain'] is not None and r['score_gain'] > 0 for r in guided)}
        data['guided_outcomes'] = guided
        self.transitions.clear()
        atomic_json(self.output / f'generation_{generation:04d}.json', data)
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        frozen_book(args.source, args.output)
