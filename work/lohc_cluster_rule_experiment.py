"""Replay graph clustering with aggregate-only DeepSeek rules and molecule scores."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from scipy.ndimage import gaussian_filter, label
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from evaluate_lohc_spectral_scores import spectral_features
from fit_lohc_structure_ridge import BASE
from lohc_structure_score_trial import AXES, DeepSeek, credential, load, save
from lohc_typed_reaction_graph import OUT as PILOT, describe
from plot_lohc_coverage_sketch import StructuralProjection


OUT = BASE / 'cluster_rule_experiment_01'
RUN = BASE / 'operator_ac_run_01'
CHECKPOINTS = {'native': 11, 'llm': 12}
MIN_PATTERN_COUNT = 3
BATCH_SIZE = 6
PROMPT = '''你是LOHC结构潜力评审。输入只有一个图谱簇的汇总统计，patterns是互斥的结构模式，每一行至少含3个分子，不是单分子。对每行输出一条可以匹配同类分子的评分规则，只用提供的矩阵统计。两个状态重原子连接相同表示候选图的关系，不能证明实验可逆。三个分项0至4，分数越高越有利：path目标储放氢路径结构合理性；skeleton往返骨架保留潜力；competition目标路径相对副反应的结构优势。0明显不支持，1有重大不利，2有合理依据但存在明显疑点，3结构支持且无明显主导风险，4结构支持很强。以位点类型和环系为依据，不能由模式人数或相对丰度推断高低。未知的分项填null，不能虚构实测物性、催化剂或工况。只返回JSON，逐行使用输入的id且不能添加其他行。示例：{"rules":[{"id":"C1-P01","grades":[3,2,2]}]}。'''


def jsonable_features(pair, include_matrix=False):
    lean, rich = [Chem.MolFromSmiles(s) for s in pair]
    matrix, _, typed, delta = describe(pair)
    rings = [set(r) for r in lean.GetRingInfo().BondRings()]
    fused = any(r & s for i, r in enumerate(rings) for s in rings[i+1:])
    n = sum(v for key, v in typed.items() if key.startswith('motif:lean:aromatic_n:0'))
    oxygen = [name for name in ('furan_o', 'aryl_ether', 'phenol') if
              any(key in typed for key in (f'motif:lean:{name}:0', f'motif:lean:{name}:1'))]
    result = {'global': spectral_features((lean, rich), True).tolist(), 'delta': delta, 'typed': typed,
              'signature': [min(n, 2), '+'.join(oxygen) or 'none', bool(fused)],
              'changed_edges': len(matrix['delta_rich_minus_lean'])}
    if include_matrix:
        result['matrix'] = matrix
    return result


def build():
    pools = {arm: load(RUN / arm / f'parents_{generation:04d}.json') for arm, generation in CHECKPOINTS.items()}
    population = sorted({tuple(row['pair']) for rows in pools.values() for row in rows})
    features = [jsonable_features(pair) for pair in population]
    blocks = [np.asarray([row[key] for row in features]) for key in ('global', 'delta')]
    space = np.concatenate([StandardScaler().fit_transform(block)/np.sqrt(block.shape[1]) for block in blocks], axis=1)
    clusters = KMeans(n_clusters=4, n_init=20, random_state=11).fit_predict(space)
    groups = defaultdict(list)
    for i, (cluster, row) in enumerate(zip(clusters, features)):
        groups[int(cluster), *row['signature']].append(i)
    rule_ids = {}
    packets = defaultdict(list)
    for seq, (key, members) in enumerate(sorted(groups.items()), 1):
        cluster, nitrogen, oxygen, fused = key
        identifier = f'C{cluster+1}-P{seq:02}'
        rule_ids.update({i: identifier for i in members})
        if len(members) < MIN_PATTERN_COUNT:
            continue
        frequency = Counter()
        for i in members:
            frequency.update(features[i]['typed'].keys())
        packets[cluster].append({'id': identifier, 'n': len(members), 'changed_site_aromatic_n_count_capped_at_2': nitrogen,
                                 'oxygen_motifs_at_distance_0_or_1': oxygen, 'fused_rings': fused,
                                 'changed_edge_count_q10_q50_q90': np.quantile([features[i]['changed_edges'] for i in members],
                                                                              [.1,.5,.9]).round(1).tolist(),
                                 'most_frequent_site_labels': [[k, n] for k, n in frequency.most_common(9)]})
    projection = StructuralProjection(RUN)
    projection.fit()
    rows = [{'pair': list(pair), 'pattern': rule_ids[i], 'cluster': int(clusters[i])+1,
             'xy': projection.position(pair[0])} for i, pair in enumerate(population)]
    return {'rows': rows, 'patterns': {str(k): v for k, v in packets.items()}, 'checkpoints': pools,
            'pattern_counts': dict(Counter(rule_ids.values()))}


def validate(response, expected):
    rules = response['rules']
    if set(response) != {'rules'} or len(rules) != len(expected) or {r['id'] for r in rules} != set(expected):
        raise ValueError('Wrong rule IDs')
    for row in rules:
        if set(row) != {'id', 'grades'} or len(row['grades']) != 3 or any(
                g is not None and (type(g) is not int or not 0 <= g <= 4) for g in row['grades']):
            raise ValueError('Invalid grades')
    return {r['id']: r['grades'] for r in rules}


def request_rules(dataset, key_file):
    api = DeepSeek(credential(key_file), OUT)

    def call(job):
        name, payload = job
        save(OUT / f'{name}_input.json', payload)
        try:
            return validate(api.run(name, PROMPT, payload), [x['id'] for x in payload['patterns']])
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            save(OUT / f'{name}_failure.json', {'error': type(exc).__name__})
            return {}

    jobs = [(f'cluster_{cluster}_{i//BATCH_SIZE:02}', {'cluster': int(cluster)+1, 'patterns': patterns[i:i+BATCH_SIZE]})
            for cluster, patterns in dataset['patterns'].items() for i in range(0, len(patterns), BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        outputs = list(pool.map(call, jobs))
        retries = [(f'{name}_retry', payload) for (name, payload), output in zip(jobs, outputs) if not output]
        outputs.extend(pool.map(call, retries))
    return {k: v for result in outputs for k, v in result.items()}


def total(grades):
    return None if grades is None or None in grades else sum(grades)*100/12


def pareto(rows):
    # Offline selection with the three objectives available in the saved parents.
    # All data are structural/physical and independent of the DeepSeek score.
    counts = Counter(row['scaffold'] for row in rows)
    grid = defaultdict(list)
    for row in rows:
        grid[row['cell']].append(row)
    chosen = []
    for members in grid.values():
        vals = np.array([[r['properties']['h2_wt_percent'], -max(40-r['properties']['delta_H_per_H2'], 0,
                        r['properties']['delta_H_per_H2']-70), 1/counts[r['scaffold']]] for r in members])
        for i, row in enumerate(members):
            if not np.any(np.all(vals >= vals[i], axis=1) & np.any(vals > vals[i], axis=1)):
                chosen.append(row)
    return chosen


def surface(arms):
    coords = np.array([r['xy'] for rows in arms.values() for r in rows])
    step = .02
    limits = [(axis.min()-.13, axis.max()+.13) for axis in coords.T]
    edges = [np.arange(lo, hi+step, step) for lo, hi in limits]
    xx, yy = np.meshgrid(*(v[:-1]+step/2 for v in edges), indexing='ij')
    mpl_labels = {'native': 'A  原生 GA', 'llm': 'C  LLM 耦合 UCB'}
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False})
    fig = plt.figure(figsize=(12, 5.8), facecolor='white')
    metadata = {}
    for i, (arm, rows) in enumerate(arms.items(), 1):
        points = np.array([r['xy'] for r in rows])
        scores = np.array([r['score'] if r['score'] is not None else np.nan for r in rows])
        mask = np.isfinite(scores)
        hist = lambda p, w=None: np.histogram2d(*p.T, bins=edges, weights=w)[0]
        density = gaussian_filter(hist(points), 2.5, truncate=2.5)
        available = gaussian_filter(hist(points[mask]), 2.5, truncate=2.5)
        weighted = gaussian_filter(hist(points[mask], scores[mask]), 2.5, truncate=2.5)
        ground = density > .012
        supported = ground & (available > .012) & (available >= .25 * density)
        z = np.divide(weighted, available, out=np.full_like(weighted, np.nan), where=supported)
        ax = fig.add_subplot(1, 2, i, projection='3d')
        ax.plot_surface(xx, yy, np.where(ground, 0, np.nan), color='#dbe4eb', shade=False, rstride=1, cstride=1)
        chart = ax.plot_surface(xx, yy, z, cmap='viridis', vmin=0, vmax=100, rstride=1, cstride=1,
                                linewidth=0, antialiased=False)
        ax.set(xlabel='结构 PC1', ylabel='结构 PC2', zlabel='局部分子平均分', zlim=(0,100), xlim=limits[0], ylim=limits[1])
        ax.set_xticks([-2,-1.5,-1,-.5,0]); ax.set_yticks([-.5,0,.5,1]); ax.set_zticks([0,25,50,75,100])
        ax.tick_params(labelsize=8, pad=0)
        ax.set_box_aspect((1,1,.68)); ax.set_proj_type('ortho'); ax.view_init(elev=31, azim=-56)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
        ax.set_title(f'{mpl_labels[arm]} · {mask.sum()}/{len(rows)} 个分子有评分', fontsize=13, pad=6)
        metadata[arm] = {'surface_nodes': int(supported.sum()), 'connected_regions': int(label(supported)[1])}
    fig.subplots_adjust(left=.045, right=.955, bottom=.14, top=.79, wspace=.1)
    fig.text(.075, .94, '结构覆盖与分子评分', fontsize=23, fontweight='bold', color='#203441')
    fig.text(.075, .88, '底面为候选覆盖，上方为局部平均分。分子分数来自 DeepSeek 结构模式规则。', fontsize=11)
    fig.colorbar(chart, cax=fig.add_axes([.33,.095,.34,.021]), orientation='horizontal', ticks=[0,25,50,75,100])
    fig.text(.5,.026,'同一 PCA、色标和核带宽。局部评分覆盖低于 25% 或缺少分子支撑的位置留空。', ha='center', fontsize=9)
    fig.savefig(OUT / 'continuous_surface.png', dpi=175)
    plt.close(fig)
    return metadata


def analyze(dataset, rules):
    scored = {tuple(row['pair']): {**row, 'score': total(rules.get(row['pattern']))} for row in dataset['rows']}
    assert len(scored) == len(dataset['rows'])
    arms = {arm: [scored[tuple(r['pair'])] for r in pool] for arm, pool in dataset['checkpoints'].items()}
    fine = {arm: pareto(rows) for arm, rows in dataset['checkpoints'].items()}
    reference = load(PILOT / 'ratings.json')
    holdout = []
    for row in reference['records']:
        pair = tuple(row['pair'])
        answer = reference['answers']['typed'].get(row['id'])
        if answer is not None and pair in scored:
            holdout.append({'id': row['id'], 'score': scored[pair]['score'],
                            'reference': total([answer[a][0] for a in AXES])})
    covered = [r for r in holdout if r['score'] is not None]
    a = np.array([r['score'] for r in covered]); b = np.array([r['reference'] for r in covered])
    calls = [load(f) for f in OUT.glob('*_audit.json')]
    usage = {k: sum((r.get('usage') or {}).get(k, 0) for r in calls) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    stats = {'unique_candidates': len(scored), 'unique_scored': sum(r['score'] is not None for r in scored.values()),
             'structural_patterns': len(dataset['pattern_counts']), 'requested_patterns': sum(map(len, dataset['patterns'].values())),
             'complete_rules': sum(total(g) is not None for g in rules.values()), 'request_count': len(calls), 'usage': usage,
             'failed_attempts': len(list(OUT.glob('*_failure.json'))), 'arms': {}, 'surface': surface(arms),
             'validation': {'tested': len(holdout), 'scored': len(covered),
                            'mae': float(np.abs(a-b).mean()) if len(a) else None,
                            'median_baseline_mae_same_subset': float(np.abs(b-np.median(b)).mean()) if len(b) else None,
                            'spearman': float(spearmanr(a,b).statistic) if len(a)>2 and np.ptp(a) and np.ptp(b) else None,
                            'within_one_grade': int((np.abs(a-b) <= 100/12 + 1e-8).sum())}}
    for arm, rows in arms.items():
        vals = [row['score'] for row in rows if row['score'] is not None]
        by_cluster = defaultdict(set)
        for row in rows:
            if row['score'] is not None:
                by_cluster[row['cluster']].add(row['score'])
        stats['arms'][arm] = {'generation': CHECKPOINTS[arm], 'n': len(rows), 'scored': len(vals),
                              'mean': float(np.mean(vals)) if vals else None, 'pareto_fine_candidates': len(fine[arm]),
                              'cluster_distinct_scores': {str(k): len(v) for k,v in by_cluster.items()},
                              'score_counts': {str(round(k,2)): n for k,n in sorted(Counter(vals).items())}}
    save(OUT / 'results.json', {**stats, 'scores': list(scored.values()), 'pareto': fine, 'validation_rows': holdout})
    print(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_path = OUT / 'matrix_patterns.json'
    if dataset_path.exists():
        dataset = load(dataset_path)
    else:
        dataset = build()
        save(dataset_path, dataset)
    save(OUT / 'protocol.json', {'model': 'deepseek-flash', 'max_tokens': 1100, 'temperature': 0,
                                 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                                 'checkpoints': CHECKPOINTS, 'minimum_pattern_count': MIN_PATTERN_COUNT,
                                 'batch_patterns': BATCH_SIZE, 'score_scale': '100*sum(grades)/12'})
    print(f'{len(dataset["rows"])} candidates; {len(dataset["pattern_counts"])} patterns; '
          f'{sum(map(len,dataset["patterns"].values()))} eligible', flush=True)
    rules_path = OUT / 'rules.json'
    if not rules_path.exists():
        save(rules_path, request_rules(dataset, args.key_file))
    analyze(dataset, load(rules_path))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
