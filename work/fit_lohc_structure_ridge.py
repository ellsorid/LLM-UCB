"""Collect blind structure grades and fit a scaffold-held-out ridge surrogate."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import random

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from evaluate_lohc_spectral_scores import evaluate_spectrum, morgan_distances, neighbor_score, spectral_features
from lohc_structure_score_trial import AXES, DeepSeek, PROMPT, ROOT, credential, load, packet_item, save, validate


BASE = ROOT / 'LOHC_From_Zero_Parallel_v10'
ORIGINAL = BASE / 'structure_score_trial_01'
OUT = BASE / 'structure_ridge_experiment_01'
FINGERPRINT = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
MOTIFS = {name: Chem.MolFromSmarts(smarts) for name, smarts in {
    'phenolic_oh': '[c]-[O;H1]', 'aryl_ether': '[c]-[O;X2]-[#6]',
    'aryl_exocyclic_cn': '[c]!@[#7]', 'aromatic_n': '[#7;a]',
    'five_ring_aromatic_o': '[o;r5]', 'aliphatic_amine': '[N;X3;!$(N-C=O)]',
}.items()}
ALPHAS = [0.1, 1, 3, 10, 30, 100, 300]


def scaffold(pair):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(pair[0])) or '(acyclic)'


def diverse_sample(pool, size, seed):
    records = sorted(pool)
    random.Random(seed).shuffle(records)
    fps = tuple(zip(*(tuple(FINGERPRINT.GetFingerprint(Chem.MolFromSmiles(s)) for s in pair) for pair in records)))
    keys = np.array([pool[pair] for pair in records])
    distances = np.ones(len(records))
    selected = []
    for _ in range(size):
        index = int(np.argmax(distances))
        if distances[index] < 0:
            break
        selected.append(records[index])
        similarity = np.minimum(*[DataStructs.BulkTanimotoSimilarity(state[index], state) for state in fps])
        distances = np.minimum(distances, 1 - similarity)
        distances[keys == keys[index]] = -1
    if len(selected) != size:
        raise ValueError('Insufficient distinct scaffolds for the requested sample')
    return selected


def collect(key_file):
    original = load(ORIGINAL / 'results.json')['candidates']
    prior = [dict(id=r['id'], pair=r['pair'], split='train', assessment=r['assessment'])
             for r in original if r['assessment'] is not None]
    old_scaffolds = {scaffold(row['pair']) for row in prior}
    pool = set()
    run = BASE / 'operator_ac_run_01'
    for arm, generation in [('native', 11), ('llm', 12)]:
        pool.update(tuple(row['pair']) for row in load(run / arm / f'parents_{generation:04d}.json'))
    by_pair = {pair: scaffold(pair) for pair in pool}
    novel_scaffolds = sorted(set(by_pair.values()) - old_scaffolds)
    random.Random(19).shuffle(novel_scaffolds)
    holdout = set(novel_scaffolds[:len(novel_scaffolds) // 4])
    split = {'train': diverse_sample({p: s for p, s in by_pair.items() if s not in holdout | old_scaffolds}, 72, 19),
             'test': diverse_sample({p: s for p, s in by_pair.items() if s in holdout}, 24, 29)}
    new = [(partition, pair) for partition, pairs in split.items() for pair in pairs]
    random.Random(31).shuffle(new)
    selection = [dict(id=f'R{i:03d}', pair=pair, split=partition) for i, (partition, pair) in enumerate(new, 1)]
    key = credential(key_file)
    OUT.mkdir(parents=True, exist_ok=True)
    audit_dir = OUT / 'api'
    audit_dir.mkdir(exist_ok=True)
    if list(audit_dir.glob('*_audit.json')):
        raise RuntimeError('API logs already exist. Preserve them and resume from the saved labels.')
    save(OUT / 'protocol.json', {'selection': selection, 'training_prior_ids': [r['id'] for r in prior],
                                 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                                 'model': 'deepseek-flash', 'temperature': 0, 'max_tokens': 1100,
                                 'ridge_alphas': ALPHAS, 'holdout_group': 'lean Murcko scaffold'})
    api = DeepSeek(key, audit_dir)

    def ask(job):
        role, batch = job
        payload = {'molecules': batch}
        save(audit_dir / f'{role}_input.json', payload)
        try:
            return validate(api.run(role, PROMPT, payload), batch)
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            save(audit_dir / f'{role}_failure.json', {'error': str(error).replace(key, '[redacted]')})
            return {}

    requests = [packet_item(row['id'], row['pair']) for row in selection]
    with ThreadPoolExecutor(max_workers=4) as executor:
        batches = [(f'batch_{i // 2 + 1:02}', requests[i:i+2]) for i in range(0, len(requests), 2)]
        scored = {k: v for batch in executor.map(ask, batches) for k, v in batch.items()}
        retries = [(f'retry_{row["id"]}', [row]) for row in requests if row['id'] not in scored]
        scored.update({k: v for batch in executor.map(ask, retries) for k, v in batch.items()})
    labels = prior + [dict(row, assessment=scored[row['id']]) for row in selection if row['id'] in scored]
    audits = [load(path) for path in audit_dir.glob('*_audit.json')]
    save(OUT / 'labels.json', {'rows': labels, 'requests': len(audits),
                              'tokens': sum((a.get('usage') or {}).get('total_tokens', 0) for a in audits),
                              'failures': len(list(audit_dir.glob('*_failure.json'))),
                              'new_scores': len(scored)})


def eigenvalues(mol):
    a = Chem.GetAdjacencyMatrix(mol, useBO=True)
    degrees = a.sum(axis=1)
    scale = np.divide(1, np.sqrt(degrees), out=np.zeros_like(degrees), where=degrees > 0)
    return np.linalg.eigvalsh(np.diag((degrees > 0).astype(float)) - a * np.outer(scale, scale)).clip(0, 2)


def descriptors(pair):
    lean, rich = pair
    a, b = map(eigenvalues, pair)
    heat = lambda values, t: np.exp(-t * values).mean()
    values = {'spectral_gap_lean': a[1], 'spectral_upper_lean': a[-1],
              'heat_1_lean': heat(a, 1), 'heat_4_lean': heat(a, 4),
              'delta_gap': b[1] - a[1], 'delta_heat_1': heat(b, 1) - heat(a, 1),
              'delta_heat_4': heat(b, 4) - heat(a, 4)}
    matrix_count = len(values)
    rings = [set(ring) for ring in lean.GetRingInfo().AtomRings()]
    values.update(heavy_atoms=lean.GetNumHeavyAtoms(),
                  n_atoms=sum(atom.GetAtomicNum() == 7 for atom in lean.GetAtoms()),
                  o_atoms=sum(atom.GetAtomicNum() == 8 for atom in lean.GetAtoms()),
                  rings=len(rings),
                  fused_ring_pairs=sum(len(r & s) >= 2 for i, r in enumerate(rings) for s in rings[i+1:]),
                  aromatic_bonds_lost=sum(bond.GetIsAromatic() for bond in lean.GetBonds()) -
                  sum(bond.GetIsAromatic() for bond in rich.GetBonds()),
                  added_h_per_heavy_atom=(sum(a.GetTotalNumHs() for a in rich.GetAtoms()) -
                                          sum(a.GetTotalNumHs() for a in lean.GetAtoms())) / lean.GetNumHeavyAtoms())
    values.update({name: len((rich if name == 'aliphatic_amine' else lean).GetSubstructMatches(pattern))
                   for name, pattern in MOTIFS.items()})
    return values, matrix_count


def overall(values):
    return np.asarray(values).sum(axis=1) * (100 / 12)


def model(alpha=1):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def fit():
    source = load(OUT / 'labels.json')
    rows = source['rows']
    pairs = [tuple(Chem.MolFromSmiles(s) for s in row['pair']) for row in rows]
    groups = np.array([scaffold(row['pair']) for row in rows])
    train = np.array([row['split'] == 'train' for row in rows])
    test = ~train
    assert not set(groups[train]) & set(groups[test])
    y = np.array([[row['assessment'][axis][0] for axis in AXES] for row in rows], dtype=float)
    features, matrix_count = zip(*(descriptors(pair) for pair in pairs))
    names = list(features[0])
    x = np.array([list(item.values()) for item in features])
    matrix_count = matrix_count[0]
    kinds = {'matrix_ridge': np.arange(matrix_count), 'chemistry_ridge': np.arange(matrix_count, len(names)),
             'combined_ridge': np.arange(len(names))}
    true = overall(y[test])
    predictions = {'train_mean': np.tile(y[train].mean(axis=0), (int(test.sum()), 1))}
    reports = {}
    rng = np.random.default_rng(41)
    grouped_rows = {g: np.flatnonzero(groups[train] == g) for g in np.unique(groups[train])}
    group_ids = list(grouped_rows)
    resamples = [np.concatenate([grouped_rows[g] for g in rng.choice(group_ids, len(group_ids), replace=True)])
                 for _ in range(120)]
    for kind, columns in kinds.items():
        model_x = x[:, columns]
        search = GridSearchCV(model(), {'ridge__alpha': ALPHAS}, cv=GroupKFold(5),
                              scoring=lambda model, values, labels: -np.abs(model.predict(values)-labels).mean())
        search.fit(model_x[train], y[train], groups=groups[train])
        best = search.best_estimator_
        predictions[kind] = best.predict(model_x[test]).clip(0, 4)
        coef = best['ridge'].coef_
        alpha = search.best_params_['ridge__alpha']
        scale = best['standardscaler'].scale_
        boot = [model(alpha).fit(model_x[train][idx], y[train][idx]) for idx in resamples]
        # Express every resampled coefficient per standard deviation of the fixed training cohort.
        draws = np.array([item['ridge'].coef_ / item['standardscaler'].scale_ * scale for item in boot])
        stability = (np.sign(draws) == np.sign(coef)).mean(axis=0)
        effects = coef.sum(axis=0) * 100 / 12
        reports[kind] = {'alpha': alpha, 'train_cv_grade_mae': -search.best_score_,
                         'axis_intercept': dict(zip(AXES, best['ridge'].intercept_.tolist())),
                         'weights': [{'feature': names[j], 'total_points_per_sd': float(effects[i]),
                                       'training_mean': float(best['standardscaler'].mean_[i]),
                                       'training_sd': float(scale[i]),
                                       'axis_weights': dict(zip(AXES, coef[:, i].tolist())),
                                       'axis_sign_stability': dict(zip(AXES, stability[:, i].tolist())),
                                       'total_sign_stability': float((np.sign(draws[:, :, i].sum(axis=1)) ==
                                                                      np.sign(effects[i])).mean())}
                                     for i, j in enumerate(columns)]}

    pair_dist = morgan_distances(pairs)
    spectra = np.array([spectral_features(pair, True) for pair in pairs])
    predictions['morgan_knn'] = np.array([[neighbor_score(distance, y[train, axis]) for axis in range(3)]
                                         for distance in pair_dist[test][:, train]])
    predictions['spectral_cluster_knn'] = np.array([[value for value, _ in evaluate_spectrum(
        spectra[train], spectra[test], y[train, axis], True)] for axis in range(3)]).T
    for key, values in predictions.items():
        estimates = overall(values)
        mae = np.abs(estimates - true)
        rho = float(spearmanr(true, estimates).statistic) if np.ptp(estimates) else None
        reports.setdefault(key, {}).update(total_mae=float(mae.mean()), total_rmse=float(np.sqrt(np.mean(mae**2))),
                                           spearman=rho, within_one_grade=int((mae <= 100/12 + 1e-8).sum()),
                                           axis_mae=dict(zip(AXES, np.abs(values - y[test]).mean(axis=0).tolist())))
    summary = {'n_train': int(train.sum()), 'n_test': int(test.sum()), 'n_train_scaffolds': len(set(groups[train])),
               'n_test_scaffolds': len(set(groups[test])), 'usage': {k: v for k, v in source.items() if k != 'rows'},
               'grade_distribution': {partition: {axis: dict(Counter(y[mask, i].astype(int).tolist()))
                                                  for i, axis in enumerate(AXES)}
                                      for partition, mask in [('train', train), ('test', test)]}}
    save(OUT / 'ridge_results.json', dict(summary, methods=reports, test_predictions=[
        {'id': row['id'], 'grades': y[test][i].tolist(), 'score': true[i],
         'predictions': {key: float(overall(v)[i]) for key, v in predictions.items()}}
        for i, row in enumerate(r for r in rows if r['split'] == 'test')]))
    print(summary)
    for key, report in reports.items():
        print(key, 'MAE', round(report['total_mae'], 3), 'rho', report['spearman'], 'alpha', report.get('alpha'))
        print([(r['feature'], round(r['total_points_per_sd'], 2), round(r['total_sign_stability'], 2))
               for r in sorted(report.get('weights', []), key=lambda x: -abs(x['total_points_per_sd']))[:6]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key-file', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if not (OUT / 'labels.json').exists():
            collect(args.key_file)
        fit()
