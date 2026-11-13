"""Plot cumulative C−A structure counts and visual-only local score differences."""

import numpy as np
from scipy.ndimage import gaussian_filter

from compare_pooled_percentiles import canonical, load


def first_evaluation(directory, cap):
    seen = {}
    for path in sorted(directory.glob('graph_evaluations_*.json')):
        for row in load(path):
            if row['index'] > cap or not row['new_valid_evaluation']:
                continue
            pair = canonical(row['smi']), canonical(row['rich_proxy_smi'])
            if None not in pair:
                seen.setdefault(pair, row['index'])
    return seen


def count_frames(projection, pool, seen, checkpoints):
    projection.fit()
    smiles = sorted({p[0] for p in pool})
    vectors = np.stack([projection.features(smi) for smi in smiles])
    points = dict(zip(smiles, projection.model.transform(vectors)))
    coordinates = np.stack(list(points.values()))
    step = .035
    edges = [np.arange(np.floor(v.min()/step)*step-4*step,
                       np.ceil(v.max()/step)*step+5*step, step) for v in coordinates.T]
    centers = [(axis[:-1] + axis[1:]) / 2 for axis in edges]
    frames = []
    for limit in checkpoints:
        groups = {}
        for arm, records in seen.items():
            selected = [p for p, index in records.items() if index <= limit]
            xy = np.stack([points[p[0]] for p in selected])
            score = np.asarray([pool[p]['score'] if pool[p]['score'] is not None else np.nan for p in selected])
            rated = np.isfinite(score)
            counts = np.histogram2d(*xy.T, bins=edges)[0]
            n_scored = np.histogram2d(*xy[rated].T, bins=edges)[0]
            sum_score = np.histogram2d(*xy[rated].T, bins=edges, weights=score[rated])[0]
            if (counts.sum(), n_scored.sum()) != (len(selected), rated.sum()):
                raise ValueError('Candidates lost during binning')
            smooth = [gaussian_filter(v, sigma=1.2, mode='constant', truncate=3) for v in (counts, n_scored, sum_score)]
            if not np.isclose(smooth[0].sum(), counts.sum()):
                raise ValueError('Smoothed counts are not mass-preserving')
            mean = np.divide(smooth[2], smooth[1], out=np.full_like(counts, np.nan),
                             where=(smooth[1] >= 1) & (smooth[1] >= .25*smooth[0]))
            groups[arm] = (counts, n_scored, smooth[0], mean)
        a, c = groups['native'], groups['llm']
        height = c[2] - a[2]
        if not np.isclose(height.sum(), c[0].sum() - a[0].sum()):
            raise ValueError('Signed difference failed conservation check')
        frames.append({'cap': limit, 'a': a, 'c': c, 'height': height,
                       'supported': (c[2]+a[2]) >= .25, 'color': c[3]-a[3]})
    if len(frames) == 3 and not np.allclose(frames[0]['height'], 0):
        raise ValueError('The common warmup differed unexpectedly')
    return centers, frames
