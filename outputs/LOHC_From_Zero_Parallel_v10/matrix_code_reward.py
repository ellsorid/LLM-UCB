"""Evaluate frozen matrix rules locally and apply their parent–child score change."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from statistics import mean

import numpy as np
from threadpoolctl import threadpool_limits

from rule_steering import frozen_book, jsonable_features, load, save
from validate_lohc_literature_scale import OUT as LITERATURE, PRIOR, AXES
from validate_lohc_matrix_adjustments import OUT as ADJUSTMENTS, apply_rules, features


def prepare(path):
    graph_path = path.with_name('graph_cluster_book.json')
    with threadpool_limits(limits=1):
        frozen_book(PRIOR, graph_path)
    sources = (graph_path, LITERATURE / 'rules.json', ADJUSTMENTS / 'rules.json')
    graph, baseline, adjustments = (load(source) for source in sources)
    save(path, {'graph': graph, 'baseline': baseline, 'adjustments': adjustments,
                'source_sha256': {str(p): sha256(p.read_bytes()).hexdigest() for p in sources}})


class MatrixBook:
    def __init__(self, path):
        data = load(path)
        graph = data['graph']
        self.rules = {tuple(key): pattern for key, pattern, _ in graph['rules']}
        self.centers = np.asarray(graph['centers'])
        self.means = [np.asarray(values) for values in graph['means']]
        self.scales = [np.asarray(values) for values in graph['scales']]
        self.baseline = data['baseline']
        self.adjustments = data['adjustments']

    @lru_cache(maxsize=20000)
    def evaluate(self, pair):
        if not all(isinstance(smi, str) and smi for smi in pair):
            return None
        try:
            info = jsonable_features(pair, include_matrix=True)
        except (ValueError, AttributeError, AssertionError, RuntimeError, IndexError):
            return None
        point = np.concatenate([(np.asarray(info[key]) - mu) / sd / np.sqrt(len(mu))
                                for key, mu, sd in zip(('global', 'delta'), self.means, self.scales)])
        cluster = int(np.argmin(((self.centers - point)**2).sum(axis=1))) + 1
        pattern = self.rules.get((cluster, *info['signature']))
        axes = AXES[:2]
        rule = self.baseline.get(pattern, {})
        if not all(rule.get(axis) for axis in axes):
            return None
        intervals = {axis: rule[axis][:2] for axis in axes}
        baseline = [mean(intervals[axis][i] for axis in axes) for i in (0, 1)]
        row = {'features': features(info['matrix']), 'axis_intervals': intervals, 'potential': baseline}
        applied = apply_rules(row, self.adjustments[str(cluster)]['rules'])
        return {'score': mean(applied['adjusted']), 'baseline': mean(baseline), 'pattern': pattern,
                'cluster': cluster, 'matched_rules': [hit['id'] for hit in applied['hits']]}

    def annotate(self, parents):
        return [{**row, 'matrix_score': self.evaluate(tuple(row['pair']))} for row in parents]

    def steering_score(self, pair):
        result = self.evaluate(pair)
        if result is None:
            return None, None, None
        return str(result['cluster']), result['pattern'], result['score']


class MatrixReward:
    def __init__(self, book, operators):
        self.book = book
        self.operators = frozenset(operators)
        self.parents = {}
        self.children = {}

    def set_parents(self, frame):
        self.parents = dict(zip(frame['smi'], frame['sat_smi']))

    def set_children(self, frame):
        self.children = dict(zip(frame['smi'], frame['sat_smi']))

    def apply(self, transitions):
        if transitions is None or transitions.empty:
            return transitions, [], 0
        result = transitions.copy()
        details = []
        eligible = 0
        for index, row in transitions.iterrows():
            if not row['child_valid'] or row['mutation_operator'] not in self.operators:
                continue
            before = float(row['operator_reward'])
            if not np.isfinite(before):
                continue
            eligible += 1
            parent, child = row['parent_smi'], row['child_smi']
            parent_grade = self.book.evaluate((parent, self.parents.get(parent)))
            child_grade = self.book.evaluate((child, self.children.get(child)))
            if parent_grade is None or child_grade is None:
                continue
            delta = (child_grade['score'] - parent_grade['score']) / 100
            after = float(np.clip(before + delta, -1, 1))
            result.at[index, 'operator_reward'] = after
            details.append({'parent': parent, 'child': child, 'operator': row['mutation_operator'],
                            'context': row['experience_context'], 'before': before, 'after': after, 'delta': delta,
                            'parent_score': parent_grade, 'child_score': child_grade})
        return result, details, eligible


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('book', type=Path)
    prepare(parser.parse_args().book)
