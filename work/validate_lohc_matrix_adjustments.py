"""Replay literature-guided, matrix-only score adjustments on 20 held-out pairs."""

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from rdkit import Chem

from lohc_matrix_deepseek_pilot import matrices
from validate_lohc_literature_scale import (
    AXES, BASE, DeepSeek, HISTORICAL, OUT as LITERATURE, PRIOR, credential, load, save,
    source_cards,
)


OUT = BASE / 'matrix_adjustment_validation_02'
SCORE_AXES = AXES[:2]
FEATURES = {
    'phenol_site': '放氢态酚羟基相对变键中心的位置类别',
    'aryl_ether_site': '放氢态非环芳醚相对变键中心的位置类别',
    'methyl_site': '放氢态芳环甲基相对变键中心的位置类别',
    'aryl_amino_site': '放氢态芳环末端氨基相对变键中心的位置类别',
    'phenol_count': '放氢态连接芳香碳的羟基个数',
    'aryl_ether_count': '放氢态非环芳醚氧个数',
    'cn_ring_edit': '放氢态芳香C–N键发生键级变化的数量',
    'fused_aromatic_edit': '放氢态芳香键发生变化且至少一端是稠合位点的数量（芳香度数≥3）',
    'guaiacol_like': '酚羟基与芳醚连接的芳香碳由一条芳香键相连',
}
PROMPT = '''根据LOHC图谱簇统计和文献，选择是否执行 eligible_rules 中的结构修正。代码已经限定可用条件、文献、分项和分数范围。对每条你愿意采用的规则仅返回id、delta、reason；id必须逐字复制，delta必须是该规则允许范围内的整数。不选择某条就省略它；全部证据不足时返回 {"rules":[]}。请给理由并说明催化条件依赖。delta是原有百分制分项上的启发式百分点，禁止当作实验数值。不能仅因位点稀有、数量多或位置不同给分。NEC只用于标准参照；甲基吲哚热化学不能推导目标加氢选择性的优劣。同源规则重叠时执行绝对值最大的一条。只输出JSON，格式为 {"rules":[{"id":"输入中的可选ID","delta":-2,"reason":"说明具体机理和适用限制"}]}。'''


def eligible_rules(histograms):
    # Guaiacol data give a conditional risk sign for exposed aromatic C–O
    # motifs. They do not set a quantitative distance or substituent gradient.
    return {
        f'{feature}={value}': {'feature': feature, 'value': value,
                                'axis': 'hydrogenation_selectivity', 'ref': 'GUAIACOL-HDO',
                                'allowed_delta': [-4, -1], 'aggregate_count': n}
        for feature in ('phenol_count', 'aryl_ether_count', 'guaiacol_like')
        for value, n in histograms[feature].items() if value not in ('0', 'no')
    }


def distances(graph, sources, excluded=None):
    steps = dict.fromkeys(sources, 0)
    queue = deque(steps)
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor != excluded and neighbor not in steps:
                steps[neighbor] = steps[node] + 1
                queue.append(neighbor)
    return steps


def features(matrix, include_neighborhood=False):
    # Every structural predicate uses aligned matrix entries or their atom labels.
    elements = {i: symbol for i, symbol, _, _ in matrix['atoms']}
    hydrogens = {i: h for i, _, h, _ in matrix['atoms']}
    graph = {i: {} for i in elements}
    for i, j, order in matrix['lean_upper']:
        graph[i][j] = graph[j][i] = order
    aromatic_degree = {i: sum(order == 1.5 for order in neighbors.values())
                       for i, neighbors in graph.items()}
    changed = [(i, j) for i, j, _ in matrix['delta_rich_minus_lean']]
    changed_atoms = {i for edge in changed for i in edge}
    cn = [(i, j) for i, j in changed
          if graph[i].get(j) == 1.5 and {elements[i], elements[j]} == {'C', 'N'}]
    cn_dist = distances(graph, {i for edge in cn for i in edge})
    edit_dist = distances(graph, changed_atoms)
    attachments = {key: set() for key in ('phenol', 'aryl_ether', 'methyl', 'aryl_amino')}
    oxygens = {'phenol': set(), 'aryl_ether': set()}
    for i, neighbors in graph.items():
        for j, order in neighbors.items():
            if order != 1 or elements[j] != 'C' or not aromatic_degree[j]:
                continue
            kind = (elements[i], len(neighbors), hydrogens[i])
            if kind == ('O', 1, 1):
                attachments['phenol'].add(j)
                oxygens['phenol'].add(i)
            elif kind == ('O', 2, 0):
                k, = neighbors.keys() - {j}
                if elements[k] == 'C' and neighbors[k] == 1 and k not in distances(graph, [j], excluded=i):
                    attachments['aryl_ether'].add(j)
                    oxygens['aryl_ether'].add(i)
            elif kind == ('C', 1, 3):
                attachments['methyl'].add(j)
            elif kind == ('N', 1, 2):
                attachments['aryl_amino'].add(j)

    def position(sites):
        if not sites:
            return 'none'
        reference, distance = ('CN', cn_dist) if cn else ('edit', edit_dist)
        steps = min(distance.get(site, 3) for site in sites)
        return f'{reference}:{steps if steps < 3 else "3+"}'

    def count_bin(count):
        return str(int(count)) if count < 3 else '3+'

    summary = {
        **{f'{k}_site': position(sites) for k, sites in attachments.items()},
        'phenol_count': count_bin(len(oxygens['phenol'])),
        'aryl_ether_count': count_bin(len(oxygens['aryl_ether'])),
        'cn_ring_edit': count_bin(len(cn)),
        'fused_aromatic_edit': count_bin(sum(graph[i].get(j) == 1.5 and
            max(aromatic_degree[i], aromatic_degree[j]) >= 3 for i, j in changed)),
        'guaiacol_like': 'yes' if any(graph[i].get(j) == 1.5 for i in attachments['phenol']
                                    for j in attachments['aryl_ether']) else 'no',
    }
    if include_neighborhood:
        ring_n = distances(graph, {i for i in graph if elements[i] == 'N' and aromatic_degree[i]})
        fusion = distances(graph, {i for i in graph if aromatic_degree[i] >= 3})

        def band(distances_to_site, site):
            if site not in distances_to_site:
                return '无参照'
            return str(min(distances_to_site[site], 3)).replace('3', '3+')

        for name, sites in attachments.items():
            summary[f'{name}_neighborhood'] = tuple(sorted(
                '/'.join((band(cn_dist or edit_dist, i), band(ring_n, i), band(fusion, i))) for i in sites))
    return summary


def cards():
    return [r for r in source_cards() if r['id'] in ('NEC-CYCLE', 'NEC-SELECT', 'MCH-CYCLE')] + [
        {'id': 'GUAIACOL-HDO', 'reference': 'https://www.sciencedirect.com/science/article/abs/pii/S0021951718304779',
         'excerpt': 'Guaiacol on Ru under H2 can undergo ring hydrogenation or competing hydrogenolysis of phenolic and ether C–O bonds; selectivity depends on the catalytic interface and hydrogen pressure.'},
        {'id': 'INDOLE-METHYL', 'reference': 'https://doi.org/10.3390/ma16072924',
         'excerpt': 'Thermochemical analysis resolves methyl substitution effects in indole-based storage cycles. It does not rank catalytic reversibility or target-product selectivity of the candidates.'},
    ]


def aggregate(rows, held_out):
    populations = defaultdict(list)
    for pair, row in rows.items():
        if pair not in held_out and row['features']:
            populations[row['cluster']].append(row['features'])
    literature = cards()
    packets = {
        cluster: {'cluster': cluster, 'n': len(group), 'feature_definitions': FEATURES,
                  'site_encoding': 'CN:k = 最邻近芳香C–N变键的端点距离k；没有该类变键时用edit:k表示到任一变键端点的距离。距离从取代基所连芳香碳起算，多位点取最短距离，3+为至少三步。',
                  'histograms': {name: {value: n for value, n in sorted(Counter(f[name] for f in group).items())
                                        if n >= 3} for name in FEATURES},
                  'literature': literature}
        for cluster, group in sorted(populations.items())
    }
    for packet in packets.values():
        packet['eligible_rules'] = eligible_rules(packet['histograms'])
    return packets


def validate(reply, packet):
    rules = reply['rules']
    if set(reply) != {'rules'} or not isinstance(rules, list) or len(rules) > 6:
        raise ValueError('Expected at most six adjustment rules')
    validated = {}
    for row in rules:
        if set(row) != {'id', 'delta', 'reason'}:
            raise ValueError('Unexpected adjustment rule fields')
        name, delta, reason = (row[k] for k in ('id', 'delta', 'reason'))
        if not isinstance(name, str) or name in validated or name not in packet['eligible_rules']:
            raise ValueError('Unsupported adjustment ID')
        condition = packet['eligible_rules'][name]
        lo, hi = condition['allowed_delta']
        if type(delta) is not int or not lo <= delta <= hi or not isinstance(reason, str) or not reason.strip():
            raise ValueError('Unsupported adjustment condition, reference, or delta')
        validated[name] = {**condition, **row}
    return list(validated.values())


def ask(api, packet, output):
    cluster = packet['cluster']
    for suffix in ('', '_retry'):
        role = f'cluster_{cluster}{suffix}'
        save(output / f'{role}_input.json', packet)
        try:
            return {'status': 'ok', 'rules': validate(api.run(role, PROMPT, packet), packet)}
        except (OSError, ValueError, RuntimeError, TypeError, KeyError, AttributeError) as exc:
            save(output / f'{role}_failure.json', {'error': type(exc).__name__, 'reason': str(exc)})
    return {'status': 'fallback', 'rules': []}


def apply_rules(row, rules):
    matched = defaultdict(list)
    for rule in rules:
        if row['features'].get(rule['feature']) == rule['value']:
            matched[(rule['axis'], rule['ref'])].append(rule)
    selected = [min(group, key=lambda r: (-abs(r['delta']), r['delta'], r['feature'], r['value']))
                for group in matched.values()]
    result = dict(row)
    if row['potential'] is None:
        return {**result, 'adjusted': None, 'hits': []}
    # Preserve the previous equal weighting of the two literature-anchored axes.
    original = row['axis_intervals']
    clipped = {}
    for axis in SCORE_AXES:
        shift = max(-8, min(8, sum(r['delta'] for r in selected if r['axis'] == axis)))
        clipped[axis] = [max(0, min(100, point + shift)) for point in original[axis]]
    return {**result, 'adjusted': [mean(clipped[a][i] for a in SCORE_AXES) for i in (0, 1)],
            'hits': selected, 'adjusted_axes': clipped}


def comparisons(records, answers, results):
    pairs = defaultdict(list)
    for r in records:
        row = results[tuple(r['pair'])]
        old = answers[r['id']]
        pairs[r['match']].append({
            'id': r['id'], 'member': r['member'], 'pair': row['pair'], 'pattern': row['pattern'],
            'baseline': mean(row['potential']) if row['potential'] else None,
            'new': mean(row['adjusted']) if row['adjusted'] else None,
            'proxy': 100 * (old['path'][0] + old['competition'][0]) / 8,
            'features': row['features'], 'hits': row['hits'],
        })
    details = []
    for group, members in sorted(pairs.items()):
        a, b = sorted(members, key=lambda r: r['member'])
        gaps = {k: b[k] - a[k] if None not in (a[k], b[k]) else None for k in ('baseline', 'new', 'proxy')}
        details.append({'group': group, 'same_pattern': a['pattern'] == b['pattern'],
                        'features_differ': a['features'] != b['features'], 'members': [a, b], 'gaps': gaps})
    return details


def summarize_pairs(pairs):
    comparable = [p for p in pairs if p['gaps']['new'] is not None]
    directional = [p for p in comparable if p['gaps']['proxy']]
    formerly_tied = [p for p in comparable if not p['gaps']['baseline']]
    metrics = {'total': len(pairs), 'comparable': len(comparable), 'proxy_non_ties': len(directional),
               'features_differ': sum(p['features_differ'] for p in pairs),
               'same_pattern': sum(p['same_pattern'] for p in pairs),
               'new_separations_from_baseline_ties': sum(p['gaps']['new'] != 0 for p in formerly_tied),
               'new_separations_from_baseline_ties_same_pattern': sum(
                   p['gaps']['new'] != 0 and p['same_pattern'] for p in formerly_tied)}
    for k in ('baseline', 'new'):
        metrics[k] = {
            'ties': sum(p['gaps'][k] == 0 for p in comparable),
            'direction_correct': sum(p['gaps'][k] * p['gaps']['proxy'] > 0 for p in directional),
            'direction_opposite': sum(p['gaps'][k] * p['gaps']['proxy'] < 0 for p in directional),
            'direction_ties': sum(p['gaps'][k] == 0 for p in directional),
            'separated_proxy_ties': sum(p['gaps'][k] != 0 and p['gaps']['proxy'] == 0 for p in comparable),
        }
    return metrics


def reverse_index(matrix):
    n = len(matrix['atoms'])
    return {**matrix, 'atoms': [[n - 1 - i, e, h1, h2] for i, e, h1, h2 in matrix['atoms']],
            **{key: [[n - 1 - j, n - 1 - i, order] for i, j, order in matrix[key]]
               for key in ('lean_upper', 'rich_upper', 'delta_rich_minus_lean')}}


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    historical = load(HISTORICAL / 'ratings.json')
    held_out = {tuple(r['pair']): r['matrix'] for r in historical['records']}
    baseline = load(LITERATURE / 'results.json')['molecule_intervals']
    absolute_rules = load(LITERATURE / 'rules.json')
    save(out / 'protocol.json', {
        'prompt_sha256': sha256(PROMPT.encode()).hexdigest(), 'model': 'deepseek-flash',
        'temperature': 0, 'max_tokens': 1100, 'attempts_per_cluster': 2,
        'holdout_count': len(held_out), 'aggregation_min_count': 3, 'max_shift_per_axis': 8,
        'eligible_shift': 'A conditional hydrogenolysis risk on target hydrogenation selectivity only, at most four points.',
        'method': 'Existing axis interval plus a clipped signed shift. Average both shifted intervals.',
        'decision': 'For ranking, require fewer ties and more correct than opposite signs on historical non-ties.',
        'target': 'Historical individual DeepSeek structural proxy, not experimental performance.',
    })
    rows, problems = {}, []
    matrix_start = perf_counter()
    for row in baseline:
        pair = tuple(row['pair'])
        try:
            matrix = held_out.get(pair)
            if matrix is None:
                matrix, _ = matrices(tuple(Chem.MolFromSmiles(s) for s in pair))
            observed = features(matrix)
        except (ValueError, RuntimeError) as exc:
            problems.append({'pair': pair, 'reason': str(exc)})
            observed = {}
        axis_intervals = {a: absolute_rules[row['pattern']][a][:2] for a in SCORE_AXES} if row['potential'] else {}
        rows[pair] = {**row, 'features': observed, 'axis_intervals': axis_intervals}
    matrix_seconds = perf_counter() - matrix_start
    assert set(held_out) <= rows.keys()
    packets = aggregate(rows, held_out)
    assert sum(p['n'] for p in packets.values()) == sum(pair not in held_out and bool(row['features'])
                                                       for pair, row in rows.items())
    invariant = sum(features(matrix) == features(reverse_index(matrix)) for matrix in held_out.values())
    assert invariant == len(held_out)
    rule_file = out / 'rules.json'
    if rule_file.exists():
        cluster_rules = load(rule_file)
        for cluster, packet in packets.items():
            rules = cluster_rules[str(cluster)]['rules']
            parsed = validate({'rules': [{k: r[k] for k in ('id', 'delta', 'reason')} for r in rules]}, packet)
            if parsed != rules:
                raise ValueError('Cached rules do not match current conditions')
    else:
        api = DeepSeek(credential(args.key_file), out)
        with ThreadPoolExecutor(max_workers=4) as pool:
            replies = pool.map(lambda packet: ask(api, packet, out), packets.values())
            cluster_rules = {str(cluster): result for cluster, result in zip(packets, replies)}
        save(rule_file, cluster_rules)
    scoring_start = perf_counter()
    scored = {pair: apply_rules(row, cluster_rules[str(row['cluster'])]['rules']) for pair, row in rows.items()}
    scoring_seconds = perf_counter() - scoring_start
    pairs = comparisons(historical['records'], historical['answers']['typed'], scored)
    changes = [mean(r['adjusted']) - mean(r['potential']) for r in scored.values() if r['potential']]
    nonzero = [x for x in changes if x != 0]
    audits = [load(p) for p in out.glob('*_audit.json')]
    usage = {k: sum((r.get('usage') or {}).get(k, 0) for r in audits) for k in
             ('prompt_tokens', 'completion_tokens', 'total_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens')}
    summary = {
        'holdout': summarize_pairs(pairs), 'index_invariance': invariant, 'unique_candidates': len(rows),
        'scored': len(changes), 'aggregate_candidates': sum(p['n'] for p in packets.values()),
        'changed': len(nonzero), 'increased': sum(x > 0 for x in nonzero),
        'decreased': sum(x < 0 for x in nonzero),
        'median_absolute_change_when_changed': median(map(abs, nonzero)) if nonzero else None,
        'maximum_absolute_change': max(map(abs, changes), default=None),
        'clusters': len(packets), 'rules': sum(len(r['rules']) for r in cluster_rules.values()),
        'fallback_clusters': sum(r['status'] == 'fallback' for r in cluster_rules.values()),
        'attempts': len(audits), 'failed_attempts': len(list(out.glob('*_failure.json'))), 'usage': usage,
        'matrix_seconds': matrix_seconds, 'rule_application_seconds': scoring_seconds,
        'matrix_failures': len(problems),
    }
    save(out / 'results.json', {'summary': summary, 'paired_details': pairs,
                                'candidate_scores': list(scored.values()), 'matrix_failures': problems})
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args())
