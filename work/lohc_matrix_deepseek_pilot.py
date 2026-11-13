"""Compare paired DeepSeek scores with and without aligned molecular matrices."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from evaluate_lohc_spectral_scores import neighbor_score, spectral_features
from fit_lohc_structure_ridge import BASE, OUT as RIDGE_OUT, scaffold
from lohc_structure_score_trial import AXES, DeepSeek, PROMPT, credential, load, packet_item, save, validate


OUT = BASE / 'matrix_deepseek_pilot_01'
MATRIX_PROMPT = PROMPT + '''
输入还包含两态共享原子索引的加权邻接矩阵。矩阵用上三角非零三元组表示，每行是[原子i,原子j,键级]，其余项为0，下三角对称。1.5为芳香键。delta_rich_minus_lean 是储氢态减放氢态的键级差矩阵。atoms 每行依次是索引、元素、放氢态氢数、储氢态氢数。M1指向此矩阵包。比较两态非零位置判断连边是否保留，比较差矩阵定位键级改变。单纯键级变化不算重原子连边改变，连边保留不等于实验可逆。评分量表维持上文定义，使用矩阵佐证结构理由，适用时引用M1。每项原样保留，再在每个 assessment 加一个 matrix_check，格式为 {"connectivity_preserved":true,"changed_edge_count":5,"example_edge":[0,1]}。changed_edge_count 只数差矩阵上三角非零边；example_edge 给出其中一条边的两个索引，若无变化则写null。'''


def unweighted_graph(molecule):
    result = Chem.RWMol()
    for atom in molecule.GetAtoms():
        result.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    for bond in molecule.GetBonds():
        result.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return result.GetMol()


def sparse(matrix):
    return [[int(i), int(j), float(matrix[i, j])] for i, j in zip(*np.nonzero(np.triu(matrix, 1)))]


def aligned_bonds(molecules):
    lean, rich = molecules
    if lean.GetNumAtoms() != rich.GetNumAtoms() or lean.GetNumBonds() != rich.GetNumBonds():
        raise ValueError('A shared heavy-atom graph is required for this pilot')
    first, second = (Chem.GetAdjacencyMatrix(mol, useBO=True) for mol in molecules)
    mappings = unweighted_graph(rich).GetSubstructMatches(unweighted_graph(lean), uniquify=False, maxMatches=4096)
    if not mappings or len(mappings) >= 4096:
        raise ValueError('Atom correspondence unavailable or search limit reached')
    alternatives = []
    for match in mappings:
        delta = second[np.ix_(match, match)] - first
        alternatives.append((int(np.count_nonzero(delta)), float(np.abs(delta).sum()), match))
    changed, magnitude, mapping = min(alternatives)
    aligned = second[np.ix_(mapping, mapping)]
    return first, second, aligned, mapping, changed, sum(x[:2] == (changed, magnitude) for x in alternatives)


def matrices(molecules):
    lean, rich = molecules
    first, _, aligned, mapping, changed, ties = aligned_bonds(molecules)
    edits = sparse(aligned - first)
    atoms = [[i, atom.GetSymbol(), atom.GetTotalNumHs(), rich.GetAtomWithIdx(mapping[i]).GetTotalNumHs()]
             for i, atom in enumerate(lean.GetAtoms())]
    assert np.array_equal(first != 0, aligned != 0) and changed == 2 * len(edits)
    input_data = {'shape': [len(atoms), len(atoms)], 'atoms': atoms, 'lean_upper': sparse(first),
                  'rich_upper': sparse(aligned), 'delta_rich_minus_lean': edits,
                  'equally_good_mappings': ties}
    risk_context = np.zeros(10)
    changed_atoms = set()
    for i, j, delta in edits:
        left, right = lean.GetAtomWithIdx(i), lean.GetAtomWithIdx(j)
        bond = lean.GetBondBetweenAtoms(i, j)
        risk_context[0] += 1
        risk_context[1] += abs(delta)
        risk_context[2] += bond.GetIsAromatic()
        risk_context[3] += bond.IsInRing()
        risk_context[4] += any(a.GetAtomicNum() == 7 for a in (left, right))
        risk_context[5] += any(a.GetAtomicNum() == 8 for a in (left, right))
        risk_context[6] += bond.IsInRingSize(5)
        risk_context[7] += bond.IsInRingSize(6)
        changed_atoms.update([i, j])
    risk_context[8] = sum(atoms[i][3] - atoms[i][2] for i in changed_atoms)
    risk_context[9] = len(changed_atoms)
    return input_data, risk_context.tolist()


def matrix_scores(raw, batch):
    checks = {r['id']: r.get('matrix_check') for r in raw['assessments']}
    grades = validate({'assessments': [{k: v for k, v in r.items() if k != 'matrix_check'}
                                       for r in raw['assessments']]}, batch)
    return {identifier: {**row, 'matrix_check': checks[identifier]} for identifier, row in grades.items()}


def matrix_audit(output, matrix):
    check = output.get('matrix_check')
    if not isinstance(check, dict):
        check = {}
    edits = matrix['delta_rich_minus_lean']
    edge = check.get('example_edge')
    return {'connectivity': check.get('connectivity_preserved') is True,
            'count': type(check.get('changed_edge_count')) is int and check['changed_edge_count'] == len(edits),
            'example_edge': (isinstance(edge, list) and len(edge) == 2 and all(type(i) is int for i in edge)
                             and sorted(edge) in [x[:2] for x in edits]) if edits else edge is None}


def collect(key_file):
    frozen = [row for row in load(RIDGE_OUT / 'labels.json')['rows'] if row['split'] == 'test']
    random.Random(23).shuffle(frozen)
    specs = []
    for i, row in enumerate(frozen, 1):
        mols = tuple(Chem.MolFromSmiles(s) for s in row['pair'])
        matrix, local = matrices(mols)
        specs.append({'id': f'M{i:02}', 'source_id': row['id'], 'pair': row['pair'],
                      'matrix': matrix, 'local_features': local})

    key = credential(key_file)
    OUT.mkdir(parents=True)
    save(OUT / 'protocol.json', {'sample': [r['source_id'] for r in specs], 'main_count': len(specs),
                                 'repeat_count_per_arm': 6, 'model': 'deepseek-flash', 'temperature': 0,
                                 'prompts_sha256': {k: hashlib.sha256(v.encode()).hexdigest()
                                                    for k, v in [('control', PROMPT), ('matrix', MATRIX_PROMPT)]}})
    api = DeepSeek(key, OUT)

    def ask(job):
        role, arm, items = job
        payload = {'molecules': items}
        save(OUT / f'{role}_input.json', payload)
        try:
            response = api.run(role, MATRIX_PROMPT if arm == 'matrix' else PROMPT, payload)
            return (matrix_scores if arm == 'matrix' else validate)(response, items)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            save(OUT / f'{role}_failure.json', {'error': str(exc).replace(key, '[redacted]')})
            return {}

    def run_batch(pool, arm, label, rows):
        items = []
        for row in rows:
            item = packet_item(row['id'], row['pair'])
            if arm == 'matrix':
                item['features']['M1'] = 'shared atom indices and aligned adjacency matrices'
                item['matrix'] = row['matrix']
            items.append(item)
        jobs = [(f'{arm}_{label}_{i//2:02}', arm, items[i:i+2]) for i in range(0, len(items), 2)]
        results = {k: v for part in pool.map(ask, jobs) for k, v in part.items()}
        missing = [(f'{arm}_{label}_retry_{i:02}', arm, [item])
                   for i, item in enumerate(items) if item['id'] not in results]
        results.update({k: v for part in pool.map(ask, missing) for k, v in part.items()})
        return results

    repeats = [dict(row, original_id=row['id'], id=f'Q{i:02}') for i, row in enumerate(specs[::4], 1)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        main = {arm: run_batch(pool, arm, 'main', specs) for arm in ('control', 'matrix')}
        repeated = {arm: run_batch(pool, arm, 'repeat', repeats) for arm in ('control', 'matrix')}
    save(OUT / 'ratings.json', {'specs': specs, 'main': main, 'repeats': repeats, 'repeated': repeated})


def grades(row):
    return np.array([row[a][0] for a in AXES])


def score(values):
    return np.asarray(values).sum(axis=-1) * 100 / 12


def evaluate(specs, values):
    global_features = np.array([spectral_features(tuple(Chem.MolFromSmiles(s) for s in r['pair']), True) for r in specs])
    local_features = np.array([r['local_features'] for r in specs])
    groups = [scaffold(r['pair']) for r in specs]
    predictions = {name: np.empty(values.shape) for name in ('median', 'global_graph', 'global_and_local')}
    for train, test in LeaveOneGroupOut().split(global_features, groups=groups):
        distances = []
        for features in (global_features, local_features):
            scale = StandardScaler().fit(features[train])
            distances.append(cdist(scale.transform(features[test]), scale.transform(features[train])) / np.sqrt(features.shape[1]))
        predictions['median'][test] = np.median(values[train], axis=0)
        for name, distance in [('global_graph', distances[0]), ('global_and_local', np.hypot(*distances))]:
            for column in range(len(AXES)):
                predictions[name][test, column] = [neighbor_score(d, values[train, column]) for d in distance]
    result = {}
    for name, predicted in predictions.items():
        measured = score(values)
        fitted = score(predicted)
        rho = spearmanr(measured, fitted).statistic if name != 'median' and np.ptp(measured) and np.ptp(fitted) else None
        result[name] = {'total_mae': float(np.abs(measured - fitted).mean()),
                        'spearman': float(rho) if rho is not None else None,
                        'axis_mae': dict(zip(AXES, np.abs(values - predicted).mean(axis=0).tolist())),
                        'predictions': fitted.tolist()}
    return result


def analyze():
    data = load(OUT / 'ratings.json')
    arms = data['main']
    specs = [row for row in data['specs'] if all(row['id'] in scores for scores in arms.values())]
    values = {arm: np.array([grades(ratings[row['id']]) for row in specs]) for arm, ratings in arms.items()}
    audits = {row['id']: matrix_audit(arms['matrix'][row['id']], row['matrix']) for row in specs}
    repeat = {arm: [score(grades(arms[arm][r['original_id']]) - grades(result[r['id']]))
                    for r in data['repeats'] if r['id'] in result and r['original_id'] in arms[arm]]
              for arm, result in data['repeated'].items()}
    usage = [load(path) for path in OUT.glob('*_audit.json')]
    summary = {'n_paired': len(specs), 'requests': len(usage),
               'tokens': sum((r.get('usage') or {}).get('total_tokens', 0) for r in usage),
               'failures': len(list(OUT.glob('*_failure.json'))),
               'matrix_checks': {k: sum(row[k] for row in audits.values()) for k in ('connectivity', 'count', 'example_edge')},
               'ambiguous_atom_mappings': sum(r['matrix']['equally_good_mappings'] > 1 for r in specs),
               'repeat': {arm: {'n': len(v), 'exact_score': int(np.sum(np.isclose(v, 0))),
                                'mae': float(np.abs(v).mean()) if v else None} for arm, v in repeat.items()},
               'distributions': {arm: {axis: dict(Counter(y[:, i].tolist())) for i, axis in enumerate(AXES)}
                                 for arm, y in values.items()},
               'grade_agreement': dict(zip(AXES, (values['control'] == values['matrix']).sum(axis=0).tolist())),
               'total_score_means': {arm: float(score(y).mean()) for arm, y in values.items()},
               'associations': {arm: evaluate(specs, y) for arm, y in values.items()}}
    save(OUT / 'analysis.json', {**summary, 'order': [r['id'] for r in specs], 'per_candidate_matrix_checks': audits})
    print(json.dumps({**summary, 'associations': {arm: {method: {k: v for k, v in row.items() if k != 'predictions'}
                      for method, row in methods.items()} for arm, methods in summary['associations'].items()}}, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path)
    args = parser.parse_args()
    if not (OUT / 'ratings.json').exists():
        collect(args.key_file)
    analyze()
