"""Validate literature anchors and aggregate DeepSeek percentage rules offline."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from lohc_cluster_rule_experiment import BASE, MIN_PATTERN_COUNT, jsonable_features
from lohc_structure_score_trial import DeepSeek, credential, load, save
from lohc_typed_reaction_graph import OUT as HISTORICAL


OUT = BASE / 'literature_scale_validation_01'
PRIOR = BASE / 'cluster_rule_experiment_01'
MEMORY = BASE.parent / 'LOHC_Grid_Guided_GA_v4/live_run/frozen_memory.json'
AXES = ('reversible', 'hydrogenation_selectivity', 'cycling', 'liquid')
EXTRACTION = '''提取文献证据卡中的原文实验数值。百分数按百分数原样返回，温度按卡片字段的单位返回，不做换算。每张卡输出与fields相同顺序的values；不补写额外指标。只输出JSON，如 {"records":[{"id":"R1","values":[1,2]}]}。'''
RUBRIC = '''你负责给LOHC图谱簇的结构模式生成可追溯百分制规则。每条是至少三个分子的统计，无单分子信息。输入已核对的文献参照和簇内联合结构统计。100代表达到对应标准LOHC分项参照的潜力；结构类比是筛选证据，不是实验结果。根据骨架、变键和邻近官能团，估计每项相对参照的结构潜力区间。若同类证据强，允许区间上限到100；若只有类比，必须保留非零区间。相关性不足则null。不得用模式人数当性能。reversible是目标双向储放氢潜力。hydrogenation_selectivity仅是目标加氢产物选择性潜力，引用该维度参照，不把转化率当选择性。矩阵摘要不能提供循环保持率、熔点或相态的证据，cycling和liquid必须返回null。不得估计实测值、虚构文献或把多个催化条件当作同一实验。输出严格JSON，id逐一原样复制，分项用[下界,上界,参照id]或null。例如 {"rules":[{"id":"C1-P01","reversible":[65,90,"NEC-CYCLE"],"hydrogenation_selectivity":null,"cycling":null,"liquid":null}]}。'''


def source_cards():
    memory = {x['id']: x for x in load(MEMORY)}
    return [
        {'id': 'NEC-CYCLE', 'reference': memory['NEC-SIX-CYCLES']['doi'],
         'fields': ['hydrogenation_pct', 'dehydrogenation_pct'],
         'excerpt': 'The integrated N-ethylcarbazole study reported 99.0% hydrogenation and 91.7% dehydrogenation with Pd-based catalysts under its optimized conditions.'},
        {'id': 'NEC-SELECT', 'reference': 'https://www.sciencedirect.com/science/article/pii/S1385894724008891',
         'fields': ['full_hydrogenation_product_selectivity_pct', 'temperature_C', 'pressure_MPa'],
         'excerpt': 'Continuous NEC hydrogenation with 1 wt% Ru/Al2O3 reached 100% conversion, 99.41% selectivity to the fully hydrogenated product, and 5.79 wt% storage, at 80 °C and 4 MPa.'},
        {'id': 'MCH-CYCLE', 'reference': memory['MCH-CYCLE']['doi'],
         'fields': ['cycles', 'initial_dehydrogenation_conversion_pct', 'last_dehydrogenation_conversion_pct'],
         'excerpt': 'The toluene/MCH cycling study ran ten cycles. Toluene hydrogenation conversion exceeded 97% across all cycles. MCH dehydrogenation conversion fell from 90% to 84%. Demethylation and other byproducts occurred.'},
        {'id': 'MCH-PHASE', 'reference': ['https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Mask=4', 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C108872&Mask=4'],
         'fields': ['lean_fusion_K', 'rich_fusion_K'],
         'excerpt': 'NIST phase-change data: toluene fusion temperature 178.1 K; methylcyclohexane fusion measurement at 146.58 K. Both boiling temperatures at ambient pressure are above 298.15 K.'},
        {'id': 'NEC-PHASE', 'reference': '10.1021/acs.jacs.5c18079', 'fields': ['lean_fusion_C'],
         'excerpt': 'The original study reports the melting point of pure N-ethylcarbazole at 71 °C. This record is for the lean endpoint only.'},
    ]


def verify_anchors(raw, cards):
    extracted = {row['id']: row['values'] for row in raw['records']}
    expected = {'NEC-CYCLE': [99, 91.7], 'NEC-SELECT': [99.41, 80, 4], 'MCH-CYCLE': [10, 90, 84],
                'MCH-PHASE': [178.1, 146.58], 'NEC-PHASE': [71]}
    if len(raw['records']) != len(cards) or extracted != expected:
        raise ValueError('Extraction differs from audited primary-source values')
    ratio = lambda x, ref: 100*min(1, x/ref)
    h, d = extracted['NEC-CYCLE']
    sel = extracted['NEC-SELECT'][0]
    _, first, last = extracted['MCH-CYCLE']
    retention = last/first
    anchors = {'NEC-CYCLE': min(ratio(h, 99), ratio(d, 91.7)), 'NEC-SELECT': ratio(sel, 99.41),
               'MCH-CYCLE': ratio(retention, 84/90), 'MCH-PHASE': 100 if max(extracted['MCH-PHASE']) < 293.15 else 0,
               'NEC-PHASE': 0 if extracted['NEC-PHASE'][0] > 25 else None}
    assert anchors == {'NEC-CYCLE': 100, 'NEC-SELECT': 100, 'MCH-CYCLE': 100, 'MCH-PHASE': 100, 'NEC-PHASE': 0}
    assert np.isclose(min(ratio(h, 99), ratio(d*.75, 91.7)), 75) and np.isclose(ratio(sel*.75, 99.41), 75)
    return {'extraction': extracted, 'benchmark_scores': anchors, 'mch_conversion_retention_pct': 100*retention,
            'synthetic_75pct_perturbation_checks': 2}


def patterns_without_holdout(dataset, excluded):
    members = defaultdict(list)
    for row in dataset['rows']:
        pair = tuple(row['pair'])
        if pair not in excluded:
            members[row['pattern']].append(pair)
    result = {}
    for cluster, patterns in dataset['patterns'].items():
        refreshed = []
        for pattern in patterns:
            pairs = members[pattern['id']]
            if len(pairs) < MIN_PATTERN_COUNT:
                continue
            features = [jsonable_features(pair) for pair in pairs]
            frequency = Counter(k for row in features for k in row['typed'])
            refreshed.append({**pattern, 'n': len(pairs),
                              'changed_edge_count_q10_q50_q90': np.quantile([r['changed_edges'] for r in features], [.1,.5,.9]).round(1).tolist(),
                              'most_frequent_site_labels': frequency.most_common(9)})
        result[cluster] = refreshed
    return result


def validate(raw, patterns):
    rows = raw['rules']
    expected = {p['id'] for p in patterns}
    if len(rows) != len(expected) or {r['id'] for r in rows} != expected or set(raw) != {'rules'}:
        raise ValueError('Wrong rule IDs')
    for row in rows:
        if set(row) != {'id', *AXES}:
            raise ValueError('Wrong score dimensions')
        for axis, reference in [('reversible', 'NEC-CYCLE'), ('hydrogenation_selectivity', 'NEC-SELECT')]:
            item = row[axis]
            if item is None:
                continue
            if (not isinstance(item, list) or len(item) != 3 or item[2] != reference or
                any(type(x) not in (int, float) for x in item[:2]) or not 0 <= item[0] < item[1] <= 100):
                raise ValueError(f'Invalid {axis} interval or reference')
        if row['cycling'] is not None or row['liquid'] is not None:
            raise ValueError('Matrix-only rules do not support cycle or phase values')
    return {r['id']: r for r in rows}


def request(api, role, prompt, packet, parse):
    save(OUT / f'{role}_input.json', packet)
    try:
        return parse(api.run(role, prompt, packet))
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        save(OUT / f'{role}_failure.json', {'error': type(exc).__name__, 'reason': str(exc)})
        return None


def experiment(dataset, prior_labels, key_file):
    api = DeepSeek(credential(key_file), OUT)
    cards = source_cards()
    cached = OUT / 'extract_01_response.txt'
    checked = (verify_anchors(load(cached), cards) if cached.exists() else
               request(api, 'extract_01', EXTRACTION, {'cards': cards}, lambda x: verify_anchors(x, cards)))
    if checked is None:
        checked = request(api, 'extract_retry', EXTRACTION, {'cards': cards}, lambda x: verify_anchors(x, cards))
    if checked is None:
        raise RuntimeError('Could not validate the literature extraction')
    save(OUT / 'validated_anchors.json', checked)
    aggregates = patterns_without_holdout(dataset, {tuple(r['pair']) for r in prior_labels['records']})
    baselines = [{'id': r['id'], 'reference': r['reference'], 'scope': r['fields'], 'observed': checked['extraction'][r['id']],
                  'context': r['excerpt']} for r in cards]
    jobs = [(f'cluster_{cluster}_{i//6:02}', {'cluster': int(cluster)+1, 'anchors': baselines, 'patterns': rows[i:i+6]})
            for cluster, rows in aggregates.items() for i in range(0, len(rows), 6)]
    def score_batch(job):
        name, packet = job
        return request(api, name, RUBRIC, packet, lambda x: validate(x, packet['patterns']))
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(score_batch, jobs))
        retries = [(f'{name}_retry', packet) for (name, packet), r in zip(jobs, replies) if r is None]
        replies.extend(pool.map(score_batch, retries))
    rules = {k: v for reply in replies if reply for k, v in reply.items()}
    save(OUT / 'rules.json', rules)
    return checked, rules, len(jobs)


def report(dataset, labels, checked, rules, batches):
    def interval(rule):
        if not rule or any(rule[a] is None for a in AXES[:2]):
            return None
        return [sum(rule[a][i] for a in AXES[:2])/2 for i in (0,1)]
    rows = [{**r, 'potential': interval(rules.get(r['pattern']))} for r in dataset['rows']]
    lookup = {tuple(row['pair']): row for row in rows}
    score = lambda row: float(np.mean(row['potential'])) if row['potential'] else None
    summary = {}
    for arm, population in dataset['checkpoints'].items():
        vals = [score(lookup[tuple(r['pair'])]) for r in population]
        vals = [s for s in vals if s is not None]
        summary[arm] = {'n': len(population), 'scored': len(vals), 'q10_q50_q90': np.quantile(vals,[.1,.5,.9]).tolist() if vals else None,
                        'max': max(vals, default=None), 'mid_ge_80': sum(s >= 80 for s in vals)}
    replay = []
    for r in labels['records']:
        old = labels['answers']['typed'].get(r['id'])
        matched = lookup.get(tuple(r['pair']))
        if old and matched:
            replay.append({'id': r['id'], 'pair_group': r['match'], 'member': r['member'], 'new': score(matched),
                           'old_proxy': 100*(old['path'][0]+old['competition'][0])/8,
                           'new_range': matched['potential']})
    comparable = [r for r in replay if r['new'] is not None]
    by_pair = defaultdict(list)
    for r in comparable:
        by_pair[r['pair_group']].append(r)
    usable = [sorted(v, key=lambda r: r['member']) for v in by_pair.values() if len(v)==2]
    signs = [(np.sign(b['new']-a['new']), np.sign(b['old_proxy']-a['old_proxy'])) for a,b in usable]
    rho = None
    if len(comparable)>2 and all(len({r[key] for r in comparable}) > 1 for key in ('new','old_proxy')):
        rho = float(spearmanr([r['new'] for r in comparable],[r['old_proxy'] for r in comparable]).statistic)
    audits = [load(f) for f in OUT.glob('*_audit.json')]
    tokens = {k:sum((r.get('usage') or {}).get(k,0) for r in audits) for k in ('prompt_tokens','completion_tokens','total_tokens')}
    result = {'anchors': checked, 'planned_rule_batches': batches, 'requests': len(audits), 'usage': tokens,
              'failed_attempts': len(list(OUT.glob('*_failure.json'))), 'rules': len(rules), 'coverage': summary,
              'unique_total': len(rows), 'unique_scored': sum(r['potential'] is not None for r in rows),
              'cycle_and_phase_scored': 0, 'historical_proxy_audit': {'n':len(replay),'scored':len(comparable),'rho':rho,
                  'complete_pairs':len(usable), 'historical_non_ties':sum(bool(b!=0) for a,b in signs),
                  'direction_correct':sum(bool(a==b and b!=0) for a,b in signs),
                  'new_ties_on_historical_non_ties':sum(bool(a==0 and b!=0) for a,b in signs)}}
    save(OUT / 'results.json', {**result, 'molecule_intervals': rows, 'historical_proxy_pairs': replay})
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'protocol.json', {'model':'deepseek-flash','temperature':0,'max_tokens':1100,
              'rule_prompt_sha256':sha256(RUBRIC.encode()).hexdigest(), 'holdout_source': str(HISTORICAL / 'ratings.json'),
              'rules_scale':'mean of two structure-potential intervals; cycle and phase are separate evidence outcomes'})
    data = load(PRIOR / 'matrix_patterns.json')
    labels = load(HISTORICAL / 'ratings.json')
    if (OUT / 'rules.json').exists():
        batches = sum('_retry_' not in p.name for p in OUT.glob('cluster_*_input.json'))
        report(data, labels, load(OUT / 'validated_anchors.json'), load(OUT / 'rules.json'), batches)
    else:
        report(data, labels, *experiment(data, labels, args.key_file))
