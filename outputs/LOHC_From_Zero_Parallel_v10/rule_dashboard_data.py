"""Synchronized population surfaces and their difference at a shared GNN cap."""

from collections import Counter
from pathlib import Path
import sys
from threading import Lock

import numpy as np
from scipy.ndimage import gaussian_filter

from dashboard import ARMS, read_json
from structural_projection import StructuralProjection


def values(matrix):
    return [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in matrix.T]


class RuleDashboard:
    def __init__(self, live):
        self.live = live
        self.projection = StructuralProjection(live.directories["native"])
        self.lock = Lock()
        self.cache = (None, None)

    def status(self):
        return {"status": {arm: self.live.status(arm) for arm in ARMS},
                "baseline_missing": self.live.missing_baseline()}

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    def _snapshot(self):
        from compare_pooled_percentiles import compare

        directories = self.live.directories
        history = {arm: read_json(directory / "progress_history.json", [])
                   for arm, directory in directories.items()}
        state = {**self.status(), "history": history, "surface": None}
        if state["baseline_missing"] or not all(history.values()):
            return state
        cap = min(h[-1]["gnn_pairs"] for h in history.values())
        aligned = {arm: next((h for h in reversed(history[arm]) if h["gnn_pairs"] <= cap), None)
                   for arm in ARMS}
        if not all(aligned.values()):
            return state
        key = (cap, *(h["generation"] for h in aligned.values()))
        if self.cache[0] != key:
            try:
                pooled = compare(directories)
                surface = self.render(aligned, pooled["score_percentile"])
                count_surface = self.count_surface(cap, pooled["scores"])
            except (OSError, ValueError) as error:
                state["comparison_error"] = str(error)
                return state
            self.cache = key, (pooled["report"], surface, count_surface)
        report, surface, count_surface = self.cache[1]
        state.update(cap=cap, aligned=aligned, pooled=report, surface=surface, count_surface=count_surface)
        return state

    def count_surface(self, cap, scores):
        work = str(Path(__file__).resolve().parents[2] / "work")
        if work not in sys.path:
            sys.path.insert(0, work)
        from plot_lohc_count_difference import count_frames, first_evaluation

        seen = {arm: first_evaluation(directory, cap) for arm, directory in self.live.directories.items()}
        if not all(seen.values()):
            return None
        pool = {pair: {"score": scores.get(pair)} for records in seen.values() for pair in records}
        centers, (frame,) = count_frames(self.projection, pool, seen, [cap])
        z = np.where(frame['supported'], frame['height'], np.nan)
        color = np.where(frame['supported'], frame['color'], np.nan)
        return {'x': centers[0].tolist(), 'y': centers[1].tolist(), 'z': values(z), 'color': values(color),
                'a': len(seen['native']), 'c': len(seen['llm']), 'colored': int(np.isfinite(color).sum()),
                'height_max': float(np.max(np.abs(frame['height']))),
                'color_max': float(np.max(np.abs(color[np.isfinite(color)]))) if np.isfinite(color).any() else 0}

    def render(self, aligned, percentiles):
        rows = {}
        for arm, state in aligned.items():
            directory = self.live.directories[arm]
            generation = state["generation"]
            matrix_file = directory / f"matrix_scores_{generation:04d}.json"
            rows[arm] = [{**row, "score": (row["matrix_score"] or {}).get("score")}
                         for row in read_json(matrix_file, [])]
        return self.render_points(rows, percentiles)

    def render_points(self, rows, percentiles):
        if not all(rows.values()):
            return None
        self.projection.fit()
        point_sets = {arm: [(self.projection.position(row['pair'][0]), row['score']) for row in entries]
                      for arm, entries in rows.items()}
        by_score = {float(score): percentile for score, percentile in percentiles.items()}
        points = {arm: [(xy, by_score.get(float(score)) if score is not None else None)
                        for xy, score in entries if xy is not None] for arm, entries in point_sets.items()}
        if not all(points.values()):
            return None
        joint = np.asarray([xy for entries in points.values() for xy, _ in entries])
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
                'ground': planes, 'distributions': counts, 'scale': 'pooled_percentile',
                'delta_nodes': int(np.isfinite(delta).sum()),
                'delta_mean': float(np.nanmean(delta)) if np.isfinite(delta).any() else None}
