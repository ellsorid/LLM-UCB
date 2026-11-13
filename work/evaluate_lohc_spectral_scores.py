"""Evaluate graph spectra against the existing blind LOHC structure scores."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1] / 'outputs' / 'LOHC_From_Zero_Parallel_v10'
TRIAL = ROOT / 'structure_score_trial_01'
SCALES = np.geomspace(.1, 16, 8)
QUANTILES = np.linspace(0, 1, 9)
K = 3
CLUSTERS = 4


def spectrum(molecule, weighted):
    adjacency = Chem.GetAdjacencyMatrix(molecule, useBO=weighted)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.divide(1, np.sqrt(degree), out=np.zeros_like(degree, dtype=float), where=degree > 0)
    laplacian = np.diag((degree > 0).astype(float)) - adjacency * np.outer(inv_sqrt, inv_sqrt)
    values = np.linalg.eigvalsh(laplacian).clip(0, 2)
    # Quantiles and normalized heat traces give fixed-length vectors across graph sizes.
    return np.r_[len(values), np.quantile(values, QUANTILES), np.exp(-np.outer(SCALES, values)).mean(axis=1)]


def spectral_features(pair, weighted):
    lean, rich = (spectrum(molecule, weighted) for molecule in pair)
    return np.r_[lean, rich, rich[1:] - lean[1:]]


def neighbor_score(distances, labels):
    near = np.argsort(distances, kind='stable')[:K]
    distance = distances[near]
    exact = distance <= 1e-10
    if exact.any():
        return float(labels[near[exact]].mean())
    return float(np.average(labels[near], weights=1 / distance))


def morgan_distances(pairs):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    lean, rich = zip(*([generator.GetFingerprint(molecule) for molecule in pair] for pair in pairs))
    return 1 - np.minimum([DataStructs.BulkTanimotoSimilarity(fp, lean) for fp in lean],
                          [DataStructs.BulkTanimotoSimilarity(fp, rich) for fp in rich])


def evaluate_spectrum(train, test, labels, clustered):
    scaler = StandardScaler().fit(train)
    train, test = scaler.transform(train), scaler.transform(test)
    if not clustered:
        return [(neighbor_score(dist, labels), len(labels)) for dist in cdist(test, train)]
    clusters = KMeans(n_clusters=CLUSTERS, n_init=20, random_state=11).fit(train)
    predictions = []
    for point, cluster in zip(test, clusters.predict(test)):
        members = clusters.labels_ == cluster
        dist = cdist(point[None, :], train[members])[0]
        predictions.append((neighbor_score(dist, labels[members]), int(members.sum())))
    return predictions


def metrics(observed, predicted):
    errors = predicted - observed
    rho = spearmanr(observed, predicted).statistic
    decisive = (observed[:, None] - observed[None, :]) > 0
    diffs = (predicted[:, None] - predicted[None, :])[decisive]
    return {'mae': round(float(np.abs(errors).mean()), 3),
            'rmse': round(float(np.sqrt(np.mean(errors ** 2))), 3),
            'spearman': round(float(rho), 3) if np.isfinite(rho) else None,
            'within_one_grade_step': int((np.abs(errors) <= 100 / 12 + .01).sum()),
            'pairwise_order_fraction': round(float(np.mean((diffs > 1e-8) + .5 * (np.abs(diffs) <= 1e-8))), 3)}


def main():
    data = json.loads((TRIAL / 'results.json').read_text(encoding='utf-8-sig'))
    rows = [row for row in data['candidates'] if row['score'] is not None]
    pairs = [tuple(Chem.MolFromSmiles(smiles) for smiles in row['pair']) for row in rows]
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=pair[0]) or '(acyclic)' for pair in pairs]
    y = np.array([row['score'] for row in rows])
    features = {label: np.array([spectral_features(pair, weighted) for pair in pairs])
                for label, weighted in [('connectivity', False), ('bond_weighted', True)]}
    distances = morgan_distances(pairs)
    methods = ['train_mean', 'morgan_knn', 'connectivity_cluster_knn',
               'bond_weighted_knn', 'bond_weighted_cluster_knn']
    predicted = {name: np.full(len(rows), np.nan) for name in methods}
    cluster_sizes = {}
    for train, test in LeaveOneGroupOut().split(y, groups=scaffolds):
        predicted['train_mean'][test] = y[train].mean()
        predicted['morgan_knn'][test] = [neighbor_score(distances[i, train], y[train]) for i in test]
        for label, clustered in [('connectivity', True), ('bond_weighted', False), ('bond_weighted', True)]:
            name = label + ('_cluster_knn' if clustered else '_knn')
            values, sizes = zip(*evaluate_spectrum(features[label][train], features[label][test], y[train], clustered))
            predicted[name][test] = values
            if clustered:
                cluster_sizes.setdefault(name, np.zeros(len(rows), dtype=int))[test] = sizes

    summaries = {name: metrics(y, values) for name, values in predicted.items()}
    # The fold mean is constant within a fold. Across folds its rank is an
    # artifact of which labels were held out, so it has no ranking metric.
    for key in ('spearman', 'pairwise_order_fraction'):
        summaries['train_mean'][key] = None
    for name, sizes in cluster_sizes.items():
        summaries[name]['test_cases_with_fewer_than_three_cluster_neighbors'] = int((sizes < K).sum())
    # This all-sample grouping is descriptive. It is separate from the held-out scoring above.
    classes = KMeans(n_clusters=CLUSTERS, n_init=20, random_state=11).fit_predict(
        StandardScaler().fit_transform(features['bond_weighted']))
    group_summary = [{'class': int(i), 'n': int((classes == i).sum()), 'mean': round(float(y[classes == i].mean()), 2),
                      'score_counts': dict(Counter(y[classes == i].tolist()))} for i in np.unique(classes)]
    report = {'protocol': {'folds': 'leave one lean Murcko scaffold out', 'neighbors': K, 'clusters': CLUSTERS,
                           'label': 'prior structure-only DeepSeek score', 'matrix': 'normalized graph Laplacian',
                           'features': 'size, eigenvalue quantiles and heat traces, both forms and their difference',
                           'one_grade_step': 100 / 12},
              'n': len(rows), 'scaffold_folds': len(set(scaffolds)), 'metrics': summaries,
              'descriptive_classes': group_summary,
              'predictions': [{'id': row['id'], 'scaffold': scaffolds[i], 'score': row['score'],
                               'class_descriptive': int(classes[i]),
                               'predictions': {k: round(float(v[i]), 3) for k, v in predicted.items()}}
                              for i, row in enumerate(rows)]}
    assert all(np.isfinite(p).all() for p in predicted.values())
    out = TRIAL / 'spectral_prediction_experiment.json'
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'predictions'}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
