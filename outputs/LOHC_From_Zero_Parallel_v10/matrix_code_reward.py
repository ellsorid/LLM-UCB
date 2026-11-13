"""Evaluate the shared matrix scoring scale for visualization."""

from functools import lru_cache
from statistics import mean

import numpy as np
from lohc_cluster_rule_experiment import jsonable_features
from lohc_structure_score_trial import load
from validate_lohc_literature_scale import AXES
from validate_lohc_matrix_adjustments import apply_rules, features


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
