"""Blind pilot of typed bond changes on matched LOHC molecular pairs."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import hashlib
from pathlib import Path
import random

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMMPA
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from fit_lohc_structure_ridge import BASE, FINGERPRINT, OUT as RIDGE_OUT, scaffold
from lohc_matrix_deepseek_pilot import MATRIX_PROMPT, grades, matrices, matrix_scores, score, unweighted_graph
from lohc_structure_score_trial import AXES, DeepSeek, PROMPT, credential, load, packet_item, save, validate
from evaluate_lohc_spectral_scores import neighbor_score, spectral_features


OUT = BASE / 'typed_reaction_graph_pilot_01'
SAMPLE = 20
MOTIFS = {name: Chem.MolFromSmarts(smarts) for name, smarts in {
    'phenol': '[c]-[O;H1]', 'aryl_ether': '[c]-[O;X2]-[#6]', 'aromatic_n': '[#7;a]',
    'furan_o': '[o;r5]', 'aryl_exocyclic_cn': '[c]!@[#7]', 'carbonyl': '[#6]=[#8]',
    'aliphatic_amine': '[N;X3;!$(N-C=O)]',
}.items()}
TYPED_PROMPT = PROMPT + '''
输入附有代码对齐的两态反应图摘要。E编号标识具体变键，提供两端原子编号和元素、放氢到储氢的键级以及两态的环系类型。G编号标识局部官能团，dist 是到最近变键端点的图距离，0 表示共享原子，1 表示相邻。T0 标记两态重原子连接关系是否相同。芳香键1.5变单键1属于键级改变；两态的重原子连接相同不能证明实验循环或选择性。原子映射存在多个同等解时，只讨论在这些映射下稳健的环系或官能团类别。三个等级沿用原量表。路径与骨架保留引用T0或相关E编号，竞争反应优先引用相应G或E编号。解释为什么该具体位点支持或削弱这一个评分维度。'''


def fragments(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    for _, text in rdMMPA.FragmentMol(molecule, maxCuts=1, resultsAsMols=False):
        if text.count('.') != 1:
            continue
        x, y = text.split('.')
        n, m = (Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in (x, y))
        if n < m:
            x, y, n, m = y, x, m, n
        if n >= 9 and 1 <= m <= 4:
            yield x, y


def sample_pairs():
    run = BASE / 'operator_ac_run_01'
    known = {tuple(r['pair']) for r in load(RIDGE_OUT / 'labels.json')['rows']}
    pools = set()
    for arm, g in [('native', 11), ('llm', 12)]:
        pools.update(tuple(r['pair']) for r in load(run / arm / f'parents_{g:04d}.json'))
    cuts = defaultdict(dict)
    for p in sorted(pools - known):
        group = scaffold(p)
        for core, frag in fragments(p[0]):
            cuts[group, core].setdefault(frag, p)

    per_scaffold = defaultdict(list)
    for (group, core), members in cuts.items():
        for (left, a), (right, b) in combinations(sorted(members.items()), 2):
            if a == b:
                continue
            fps = [FINGERPRINT.GetFingerprint(Chem.MolFromSmiles(p[0])) for p in (a, b)]
            similarity = DataStructs.TanimotoSimilarity(*fps)
            per_scaffold[group].append((similarity, core, left, right, a, b))
    groups = sorted(per_scaffold)
    selected = []
    chosen_types = Counter()
    for _ in range(min(SAMPLE, len(groups))):
        # Give different single-site substitutions a chance to enter the fixed cohort.
        choices = [(chosen_types[(v[2], v[3])], -v[0], g, v) for g in groups for v in per_scaffold[g]]
        _, _, group, (similarity, core, left, right, a, b) = min(choices)
        groups.remove(group)
        chosen_types[(left, right)] += 1
        selected.append({'scaffold': group, 'core': core, 'fragments': [left, right], 'pairs': [a, b],
                         'lean_morgan_similarity': similarity})
    if len(selected) < 12:
        raise ValueError(f'Only {len(selected)} distinct matched series')
    return selected, len(per_scaffold)


def edge_types(molecule):
    rings = molecule.GetRingInfo().BondRings()
    out = defaultdict(set)
    for ring in rings:
        label = f'{len(ring)}' + ('ar' if all(molecule.GetBondWithIdx(b).GetIsAromatic() for b in ring) else 'nonar')
        for b in ring:
            out[b].add(label)
    return {b.GetIdx(): '/'.join(sorted(out[b.GetIdx()])) or 'nonring' for b in molecule.GetBonds()}


def describe(pair):
    lean, rich = [Chem.MolFromSmiles(s) for s in pair]
    matrix, _ = matrices((lean, rich))
    left = {(i, j): order for i, j, order in matrix['lean_upper']}
    right = {(i, j): order for i, j, order in matrix['rich_upper']}
    lean_types = edge_types(lean)
    # Use the same selected atom mapping to label the rich graph.
    candidates = unweighted_graph(rich).GetSubstructMatches(unweighted_graph(lean), uniquify=False, maxMatches=4096)
    rich_adjacency = Chem.GetAdjacencyMatrix(rich, useBO=True)
    expected = np.zeros_like(rich_adjacency)
    for (i, j), order in right.items():
        expected[i, j] = expected[j, i] = order
    mapping = min(m for m in candidates if np.array_equal(rich_adjacency[np.ix_(m, m)], expected))
    assert all(rich.GetAtomWithIdx(mapping[i]).GetTotalNumHs() == h for i, _, _, h in matrix['atoms'])
    rich_types = edge_types(rich)
    active = {i for i, j, _ in matrix['delta_rich_minus_lean']} | {j for i, j, _ in matrix['delta_rich_minus_lean']}
    distances = Chem.GetDistanceMatrix(lean)
    facts = {'T0': f'两态重原子连接相同；等价候选映射数 {matrix["equally_good_mappings"]}'}
    typed = Counter()
    for k, (i, j, _) in enumerate(matrix['delta_rich_minus_lean'], 1):
        element = '-'.join(sorted([matrix['atoms'][i][1], matrix['atoms'][j][1]]))
        a = lean_types[lean.GetBondBetweenAtoms(i, j).GetIdx()]
        b = rich_types[rich.GetBondBetweenAtoms(mapping[i], mapping[j]).GetIdx()]
        change = f'{left[i,j]:g}>{right[i,j]:g}'
        facts[f'E{k:02}'] = f'{i}-{j} {element} {change} {a}>{b}'
        typed[f'edit:{element}:{change}:{a}>{b}'] += 1
    motif_id = 0
    for phase, mol, lookup in [('lean', lean, dict(enumerate(range(lean.GetNumAtoms())))),
                               ('rich', rich, {index: i for i, index in enumerate(mapping)})]:
        for name, query in MOTIFS.items():
            for atoms in mol.GetSubstructMatches(query):
                indices = sorted(lookup[i] for i in atoms)
                distance = int(min(distances[i, j] for i in indices for j in active))
                motif_id += 1
                facts[f'G{motif_id:02}'] = f'{phase} {name} atoms={indices} dist={distance}'
                typed[f'motif:{phase}:{name}:{min(distance, 3)}'] += 1
    delta = np.zeros_like(expected)
    for i, j, value in matrix['delta_rich_minus_lean']:
        delta[i, j] = delta[j, i] = value
    values = np.linalg.eigvalsh(delta)
    delta_spectrum = np.r_[np.quantile(values, np.linspace(0, 1, 9)),
                           np.mean(abs(values)), np.sqrt(np.mean(values**2))]
    return matrix, facts, dict(typed), delta_spectrum.tolist()


def build_cohort():
    matches, eligible = sample_pairs()
    records = []
    for pair_number, match in enumerate(matches, 1):
        for member, pair in zip(('a', 'b'), match['pairs']):
            matrix, facts, typed, delta = describe(pair)
            records.append({'match': pair_number, 'member': member, 'pair': pair, 'matrix': matrix,
                            'facts': facts, 'typed_features': typed, 'delta_spectrum': delta})
    random.Random(29).shuffle(records)
    for i, record in enumerate(records, 1):
        record['id'] = f'X{i:03}'
    return matches, eligible, records


def collect(key_file, matches, eligible, records):
    key = credential(key_file)
    OUT.mkdir(parents=True)
    save(OUT / 'protocol.json', {'pair_count': len(matches), 'eligible_scaffolds': eligible, 'model': 'deepseek-flash',
                                 'temperature': 0, 'holdout': 'both members of one typed Murcko scaffold',
                                 'neighbors': 3, 'version': {name: hashlib.sha256(prompt.encode()).hexdigest()
                                 for name, prompt in [('raw', MATRIX_PROMPT), ('typed', TYPED_PROMPT)]}})
    api = DeepSeek(key, OUT)

    def request(job):
        arm, role, batch = job
        packet = {'molecules': batch}
        save(OUT / f'{role}_input.json', packet)
        try:
            response = api.run(role, MATRIX_PROMPT if arm == 'raw' else TYPED_PROMPT, packet)
            return (matrix_scores if arm == 'raw' else validate)(response, batch)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            save(OUT / f'{role}_failure.json', {'error': str(error).replace(key, '[redacted]')})
            return {}

    def items_for(arm):
        for row in records:
            item = packet_item(row['id'], row['pair'])
            if arm == 'raw':
                item['features']['M1'] = 'shared atom indices and aligned adjacency matrices'
                item['matrix'] = row['matrix']
            else:
                item['features'].update(row['facts'])
            yield item

    answers = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for arm in ('raw', 'typed'):
            items = list(items_for(arm))
            batches = [(arm, f'{arm}_{i // 2:03}', items[i:i+2]) for i in range(0, len(items), 2)]
            result = {k: v for part in executor.map(request, batches) for k, v in part.items()}
            retries = [(arm, f'{arm}_retry_{i:03}', [item]) for i, item in enumerate(items) if item['id'] not in result]
            result.update({k: v for part in executor.map(request, retries) for k, v in part.items()})
            answers[arm] = result
    save(OUT / 'ratings.json', {'matches': matches, 'records': records, 'answers': answers})


def evaluate(records, labels):
    global_x = np.array([spectral_features(tuple(Chem.MolFromSmiles(s) for s in row['pair']), True) for row in records])
    delta_x = np.array([row['delta_spectrum'] for row in records])
    typed_x = [row['typed_features'] for row in records]
    method_names = ('median', 'global_and_delta', 'global_delta_and_typed')
    output = {name: np.empty(labels.shape) for name in method_names}
    for train, test in LeaveOneGroupOut().split(global_x, groups=[row['match'] for row in records]):
        def distance(features):
            scale = StandardScaler().fit(features[train])
            return cdist(scale.transform(features[test]), scale.transform(features[train])) / np.sqrt(features.shape[1])
        global_delta = np.hypot(distance(global_x), distance(delta_x))
        vectorizer = DictVectorizer().fit([typed_x[i] for i in train])
        typed = vectorizer.transform(typed_x).toarray()
        # A test-only motif differs from every known training motif.
        missing = np.array([sum(typed_x[i].values()) for i in test]) - typed[test].sum(1)
        typed_distance = cdist(typed[test], typed[train], metric='cityblock') + missing[:, None]
        sums = typed[test].sum(1)[:, None] + typed[train].sum(1) + missing[:, None]
        typed_distance = np.divide(typed_distance, sums, out=np.zeros_like(typed_distance), where=sums > 0)
        output['median'][test] = np.median(labels[train], axis=0)
        for name, d in [('global_and_delta', global_delta), ('global_delta_and_typed', np.hypot(global_delta, typed_distance))]:
            output[name][test] = [[neighbor_score(dist, labels[train, i]) for i in range(len(AXES))] for dist in d]
    return output


def directions(actual, predicted, pairs):
    da = actual[pairs[:, 1]] - actual[pairs[:, 0]]
    dp = predicted[pairs[:, 1]] - predicted[pairs[:, 0]]
    clear = np.abs(da) > 1e-8
    signs = np.sign(np.where(np.abs(dp) < 1e-8, 0, dp))
    return {'observed_non_ties': int(clear.sum()), 'direction_correct': int(((signs == np.sign(da)) & clear).sum()),
            'predicted_ties_on_non_ties': int(((signs == 0) & clear).sum()),
            'delta_mae': float(np.abs(da-dp).mean())}


def analyze():
    d = load(OUT / 'ratings.json')
    complete = Counter(row['match'] for row in d['records'] if all(row['id'] in a for a in d['answers'].values()))
    rows = [row for row in d['records'] if complete[row['match']] == 2]
    lookup = {(row['match'], row['member']): i for i, row in enumerate(rows)}
    pair_indexes = np.array([[lookup[i, 'a'], lookup[i, 'b']] for i in sorted(complete) if complete[i] == 2])
    audits = [load(path) for path in OUT.glob('*_audit.json')]
    result = {'n_pairs': len(pair_indexes), 'requests': len(audits), 'failures': len(list(OUT.glob('*_failure.json'))),
              'tokens': {arm: sum((a.get('usage') or {}).get('total_tokens', 0) for a in audits if a['role'].startswith(arm))
                         for arm in ('raw', 'typed')}, 'arms': {}}
    for arm, answers in d['answers'].items():
        actual = np.array([grades(answers[row['id']]) for row in rows])
        stats = {}
        for name, fitted in evaluate(rows, actual).items():
            measured, estimate = score(actual), score(fitted)
            rho = spearmanr(measured, estimate).statistic if np.ptp(measured) and np.ptp(estimate) and name != 'median' else None
            stats[name] = {'total_mae': float(np.abs(measured-estimate).mean()), 'spearman': float(rho) if rho is not None else None,
                           'direction': directions(measured, estimate, pair_indexes),
                           'axis_mae': dict(zip(AXES, np.abs(fitted-actual).mean(0).tolist())),
                           'predictions': dict(zip([row['id'] for row in rows], estimate.tolist()))}
        result['arms'][arm] = {'stats': stats, 'average_score': float(score(actual).mean()),
                               'label_distributions': {axis: dict(Counter(actual[:, i].tolist())) for i, axis in enumerate(AXES)},
                               'typed_references': int(sum(answers[row['id']][axis][1][0] in 'TEG' for row in rows for axis in AXES)),
                               'label_scores': dict(zip([row['id'] for row in rows], score(actual).tolist()))}
    a, b = (np.array([result['arms'][arm]['label_scores'][row['id']] for row in rows]) for arm in ('raw', 'typed'))
    result['raw_to_typed_direction_agreement'] = directions(a, b, pair_indexes)
    save(OUT / 'analysis.json', result)
    for arm, data in result['arms'].items():
        print(arm, {k: {x: v for x, v in y.items() if x != 'predictions'} for k, y in data['stats'].items()},
              data['label_distributions'], data['typed_references'])
    print({k: v for k, v in result.items() if k != 'arms'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if not (OUT / 'ratings.json').exists():
            matches, eligible, records = build_cohort()
            print(f'{len(matches)} pairs from {eligible} eligible scaffolds, {len(records)} inputs', flush=True)
            if args.dry_run:
                raise SystemExit(0)
            collect(args.key_file, matches, eligible, records)
        analyze()
