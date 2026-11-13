"""Synchronized population surfaces and their difference at a shared GNN cap."""

from collections import Counter
from threading import Lock
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

from dashboard import ARMS, read_json
from structural_projection import StructuralProjection


def values(matrix):
    return [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in matrix.T]


class RuleDashboard:
    def __init__(self, live, replay=None):
        self.live = live
        self.projection = StructuralProjection(live.run)
        self.lock = Lock()
        self.cache = (None, None)
        experiment = (read_json(live.run / 'ac_experiment.json', {}) or {}).get('experiment')
        self.full_m3os = experiment == 'native_vs_m3os_full'
        self.graph_mode = self.full_m3os or experiment == 'm3os_top1_vs_llm_top5'
        self.pooled_cache = (None, None)
        self.replay = replay
        self.replay_indices = self.find_replay_indices() if replay is not None else None
        self.count_cache = (None, None)

    def snapshot(self, generation=None):
        with self.lock:
            return self._snapshot(generation)

    def _snapshot(self, generation):
        history = {arm: read_json(self.live.run / arm / 'progress_history.json', []) for arm in ARMS}
        state = {'status': {arm: self.live.status(arm) for arm in ARMS}, 'history': history, 'surface': None,
                 'graph_mode': self.graph_mode, 'full_m3os': self.full_m3os}
        if self.replay is not None:
            return self.replay_snapshot(state, generation)
        pooled = read_json(self.live.run / 'pooled_comparison.json', None)
        if pooled is not None:
            state['pooled'] = pooled
        if not all(history.values()):
            return state
        cap = min(h[-1]['gnn_pairs'] for h in history.values())
        aligned = {arm: next((h for h in reversed(history[arm]) if h['gnn_pairs'] <= cap), None) for arm in ARMS}
        if not all(aligned.values()):
            return state
        key = tuple(h['generation'] for h in aligned.values())
        if self.graph_mode:
            if pooled is None:
                if self.pooled_cache[0] != cap:
                    from compare_pooled_percentiles import compare
                    try:
                        self.pooled_cache = cap, compare(self.live.run, write=False)
                    except (FileNotFoundError, ValueError):
                        return state
                pooled = self.pooled_cache[1]
            state['pooled'] = pooled
            key = (*key, cap)
        if self.cache[0] != key:
            surface = self.render(aligned, pooled['score_percentile'] if self.graph_mode else None)
            if surface is None:
                return state
            self.cache = key, surface
        state.update(cap=cap, aligned=aligned, surface=self.cache[1])
        if self.graph_mode:
            if self.count_cache[0] != cap:
                self.count_cache = cap, self.count_surface(cap)
            state['count_surface'] = self.count_cache[1]
        return state

    def count_surface(self, cap):
        work = str(Path(__file__).resolve().parents[2] / 'work')
        if work not in sys.path:
            sys.path.insert(0, work)
        from plot_lohc_count_difference import count_frames, first_evaluation
        from compare_pooled_percentiles import collect_evaluated

        seen = {arm: first_evaluation(self.live.run, arm, cap) for arm in ARMS}
        if not all(seen.values()):
            return None
        pool = {}
        for arm in ARMS:
            for pair, score in collect_evaluated(self.live.run / arm, cap)[0].items():
                pool[pair] = {'score': score}
        centers, (frame,) = count_frames(self.projection, pool, seen, [cap])
        z = np.where(frame['supported'], frame['height'], np.nan)
        color = np.where(frame['supported'], frame['color'], np.nan)
        return {'x': centers[0].tolist(), 'y': centers[1].tolist(), 'z': values(z), 'color': values(color),
                'a': len(seen['native']), 'c': len(seen['llm']), 'colored': int(np.isfinite(color).sum()),
                'height_max': float(np.max(np.abs(frame['height']))),
                'color_max': float(np.max(np.abs(color[np.isfinite(color)]))) if np.isfinite(color).any() else 0}

    def find_replay_indices(self):
        from compare_pooled_percentiles import canonical

        expected = {arm: {tuple(r['pair']) for r in self.replay['scores'] if arm in r['arms']} for arm in ARMS}
        lean = {p[0] for pairs in expected.values() for p in pairs}
        indices = {arm: {} for arm in ARMS}
        for arm, pairs in expected.items():
            for path in sorted((self.live.run / arm).glob('graph_evaluations_*.json')):
                for row in read_json(path, []):
                    if not row['new_valid_evaluation'] or row['index'] > self.replay['gnn_common_cap']:
                        continue
                    smi = canonical(row['smi'])
                    if smi in lean:
                        pair = (smi, canonical(row['rich_proxy_smi']))
                        if pair in pairs:
                            indices[arm].setdefault(pair, row['index'])
            if indices[arm].keys() != pairs:
                raise ValueError(f'Missing replay evaluation indices for {arm}')
        return indices

    def replay_snapshot(self, state, generation):
        histories = {arm: {h['generation']: h['gnn_pairs'] for h in state['history'][arm]} for arm in ARMS}
        stages = sorted(set.intersection(*(set(h) for h in histories.values())))
        generation = min(stages[-1], max(stages[0], generation)) if generation is not None else stages[-1]
        cap = min(histories[arm][generation] for arm in ARMS)
        rows = {arm: [r for r in self.replay['scores']
                      if self.replay_indices[arm].get(tuple(r['pair']), float('inf')) <= cap] for arm in ARMS}
        summary = {}
        for arm, items in rows.items():
            scores = [r['score'] for r in items if r['score'] is not None]
            summary[arm] = {'sample': len(items), 'scored': len(scores), 'mean': float(np.mean(scores)) if scores else None}
        if self.cache[0] != cap:
            extent = [r['pair'][0] for r in self.replay['scores']]
            self.cache = cap, self.render_points(rows, scale='absolute_pilot', extent=extent)
        selected = {r['id']: r['score'] for items in rows.values() for r in items}
        state.update(surface=self.cache[1], cap=cap, replay=summary, replay_generation=generation,
                     replay_generations=stages, replay_scores=selected, pilot_pass=self.replay['acceptance_pass'],
                     unique_scored=sum(v is not None for v in selected.values()), unique_sample=len(selected))
        return state

    def render(self, aligned, percentiles=None):
        rows = {}
        for arm, state in aligned.items():
            directory = self.live.run / arm
            generation = state['generation']
            matrix_file = directory / f'matrix_scores_{generation:04d}.json'
            if matrix_file.exists():
                rows[arm] = [{**row, 'score': (row['matrix_score'] or {}).get('score')}
                             for row in read_json(matrix_file, [])]
            else:
                rows[arm] = read_json(directory / f'rule_scores_{generation:04d}.json', [])
        return self.render_points(rows, percentiles=percentiles)

    def render_points(self, rows, percentiles=None, scale='raw', extent=None):
        if not all(rows.values()):
            return None
        self.projection.fit()
        point_sets = {arm: [(self.projection.position(row['pair'][0]), row['score']) for row in entries]
                      for arm, entries in rows.items()}
        if percentiles is not None:
            by_score = {float(score): percentile for score, percentile in percentiles.items()}
            point_sets = {arm: [(xy, by_score.get(float(score)) if score is not None else None)
                                for xy, score in entries] for arm, entries in point_sets.items()}
        points = {arm: [(xy, score) for xy, score in entries if xy is not None] for arm, entries in point_sets.items()}
        if not all(points.values()):
            return None
        if extent is None:
            joint = np.asarray([xy for entries in points.values() for xy, _ in entries])
        else:
            joint = np.asarray([xy for smi in extent if (xy := self.projection.position(smi)) is not None])
        edges = [np.arange(np.floor(a.min()*50)/50-.14, np.ceil(a.max()*50)/50+.16, .02) for a in joint.T]
        grids, planes, counts = {}, {}, {}
        for arm, entries in points.items():
            xy = np.array([p for p, _ in entries])
            scores = np.array([np.nan if s is None else s for _, s in entries])
            valid = np.isfinite(scores)

            def smooth(pos, weights=None):
                return gaussian_filter(np.histogram2d(*pos.T, bins=edges, weights=weights)[0], 2.5, truncate=2.5)

            density = smooth(xy)
            rated = smooth(xy[valid])
            support = (density > .012) & (rated > .012) & (rated >= .25*density)
            grids[arm] = np.divide(smooth(xy[valid], scores[valid]), rated,
                                   out=np.full_like(rated, np.nan), where=support)
            planes[arm] = values(np.where(density > .012, 0, np.nan))
            counts[arm] = dict(sorted(Counter(round(v, 2) for v in scores[valid]).items()))
        delta = grids['llm']-grids['native']
        return {'x': (edges[0][:-1]+.01).tolist(), 'y': (edges[1][:-1]+.01).tolist(),
                'z': {**{arm: values(grid) for arm, grid in grids.items()}, 'delta': values(delta)},
                'ground': planes, 'distributions': counts, 'scale': 'pooled_percentile' if percentiles is not None else scale,
                'delta_nodes': int(np.isfinite(delta).sum()),
                'delta_mean': float(np.nanmean(delta)) if np.isfinite(delta).any() else None}
