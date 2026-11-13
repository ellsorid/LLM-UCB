import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors as rdmd
import networkx as nx
import copy
import os
import time
import zipfile
import re
import xml.etree.ElementTree as ET
import subprocess
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
import SA_Score.sascorer as SA
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
# 集成焓变预测
from predict import predict_delta_H, predict_delta_H_with_uncertainty
from enthalpy_window import DH_LOW, DH_TARGET, DH_HIGH, DH_EDGES, dh_quality
from adaptive_ga import (
    MUTATION_OPERATORS,
    ExplorationTracker,
    BIO_PLATFORM_IDS,
    OperatorExperiencePolicy,
    append_csv,
    build_transition_records,
    classify_objective_state,
    context_key,
)

RANK_SELECTION_PRESSURE = 1.05


QD_H2_EDGES = [4.2, 4.8, 5.4, 6.0, np.inf]
QD_DH_EDGES = DH_EDGES
QD_SA_EDGES = [0.0, 2.0, 3.0, 4.000001]


def _qd_finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _qd_bin(value, edges):
    if not _qd_finite(value):
        return -1
    return int(np.digitize([float(value)], edges, right=False)[0] - 1)


def _qd_cell_from_row(row):
    platform_id = str(row.get('platform_id', 'unassigned'))
    if platform_id not in BIO_PLATFORM_IDS:
        return ''
    h2 = row.get('h2_wt_percent', np.nan)
    dh = row.get('delta_H_per_H2', np.nan)
    sa = row.get('SA score', np.nan)
    if not all(_qd_finite(v) for v in (h2, dh, sa)):
        return ''
    h2 = float(h2); dh = float(dh); sa = float(sa)
    if h2 < 4.2 or dh < DH_LOW or dh > DH_HIGH or sa > 4.0:
        return ''
    return '{}|{}'.format(platform_id, _qd_bin(h2, QD_H2_EDGES))


def _qd_quality_from_row(row):
    try:
        h2 = float(row.get('h2_wt_percent', np.nan))
        dh = float(row.get('delta_H_per_H2', np.nan))
        sa = float(row.get('SA score', np.nan))
    except (TypeError, ValueError):
        return np.nan
    if not np.all(np.isfinite([h2, dh, sa])):
        return np.nan
    h2_q = np.clip((h2 - 4.2) / (7.0 - 4.2), 0.0, 1.0)
    dh_q = dh_quality(dh)
    return float((h2_q + dh_q) / 2.0)


def annotate_qd_cells(frame):
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    records = out.to_dict('records')
    out['qd_cell'] = [_qd_cell_from_row(row) for row in records]
    out['qd_quality'] = [_qd_quality_from_row(row) for row in records]
    out['qd_target_hit'] = out['qd_cell'].astype(str).ne('').astype(int)
    return out


def apply_frontier_parent_scores(frame, cfg):
    if frame is None or frame.empty or not bool(cfg.get('full_model_mode', False)):
        return frame
    out = annotate_qd_cells(frame)
    if 'selection_score' not in out.columns:
        return out
    base = pd.to_numeric(out['selection_score'], errors='coerce').fillna(0.0)
    quality = pd.to_numeric(out.get('qd_quality', pd.Series(np.nan, index=out.index)), errors='coerce').fillna(0.0)
    hit = pd.to_numeric(out.get('qd_target_hit', pd.Series(0, index=out.index)), errors='coerce').fillna(0.0)
    cell = out.get('qd_cell', pd.Series('', index=out.index)).fillna('').astype(str)
    counts = cell[cell.ne('')].value_counts()
    rarity = cell.map(lambda v: 1.0 / np.sqrt(float(counts.get(v, 1))) if v else 0.0).astype(float)
    qd_score = 0.45 * hit + 0.35 * rarity + 0.20 * quality
    max_qd = float(qd_score.max()) if qd_score.shape[0] else 0.0
    if max_qd > 0:
        qd_score = qd_score / max_qd
    out['frontier_parent_score'] = qd_score
    combined = qd_score.copy()
    if bool(cfg.get('chemical_qd_mode', False)):
        out = annotate_chemical_qd(out, cfg)
        chem = pd.to_numeric(out.get('chemical_qd_score', 0.0), errors='coerce').fillna(0.0)
        chem_max = float(chem.max()) if chem.shape[0] else 0.0
        if chem_max > 0:
            chem = chem / chem_max
        out['chemical_parent_score'] = chem
        chem_weight = float(cfg.get('chemical_parent_weight', 0.60))
        combined = (1.0 - chem_weight) * combined + chem_weight * chem
    weight = float(cfg.get('frontier_parent_weight', 0.75))
    out['selection_score'] = base * (1.0 + weight * combined)
    return out


def _chemical_behavior_vector_from_mol(mol):
    if mol is None:
        return [np.nan] * 8
    try:
        atoms = list(mol.GetAtoms())
        heavy = max(1, sum(1 for a in atoms if a.GetAtomicNum() > 1))
        hetero = sum(1 for a in atoms if a.GetAtomicNum() not in (1, 6)) / heavy
        n_hetero = sum(1 for a in atoms if a.GetAtomicNum() == 7) / heavy
        frac_sp3 = float(rdmd.CalcFractionCSP3(mol))
        aromatic_rings = float(rdmd.CalcNumAromaticRings(mol))
        rings = float(rdmd.CalcNumRings(mol))
        rot = float(rdmd.CalcNumRotatableBonds(mol))
        hba = float(rdmd.CalcNumHBA(mol))
        hbd = float(rdmd.CalcNumHBD(mol))
        return [
            np.clip(frac_sp3, 0.0, 1.0),
            np.clip(aromatic_rings / 5.0, 0.0, 1.0),
            np.clip(rings / 6.0, 0.0, 1.0),
            np.clip(rot / 8.0, 0.0, 1.0),
            np.clip(hetero / 0.35, 0.0, 1.0),
            np.clip(n_hetero / 0.20, 0.0, 1.0),
            np.clip(hba / 6.0, 0.0, 1.0),
            np.clip(hbd / 3.0, 0.0, 1.0),
        ]
    except Exception:
        return [np.nan] * 8


def _chemical_behavior_centroids(n_cells, dim=8, seed=1729):
    rng = np.random.default_rng(int(seed))
    n_cells = max(4, int(n_cells))
    centroids = rng.random((n_cells, int(dim)))
    anchors = np.array([
        [0.10, 0.85, 0.70, 0.10, 0.05, 0.00, 0.10, 0.05],
        [0.20, 0.55, 0.45, 0.35, 0.20, 0.35, 0.35, 0.15],
        [0.45, 0.35, 0.35, 0.55, 0.25, 0.10, 0.40, 0.20],
        [0.75, 0.10, 0.20, 0.65, 0.18, 0.05, 0.25, 0.10],
    ], dtype=np.float64)
    centroids[: min(n_cells, anchors.shape[0]), :] = anchors[: min(n_cells, anchors.shape[0]), :]
    return centroids


def annotate_chemical_behavior(frame, cfg=None):
    if frame is None or frame.empty:
        return frame
    cfg = cfg or {}
    out = _copy_with_mol(frame)
    vectors = np.asarray([_chemical_behavior_vector_from_mol(mol) for mol in out['mol'].tolist()], dtype=np.float64)
    finite = np.all(np.isfinite(vectors), axis=1)
    vectors_clean = np.where(np.isfinite(vectors), vectors, 0.0)
    n_cells = int(cfg.get('chemical_behavior_cells', 64))
    centroids = _chemical_behavior_centroids(n_cells, dim=vectors_clean.shape[1], seed=int(cfg.get('chemical_behavior_seed', 1729)))
    dist = ((vectors_clean[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(dist, axis=1).astype(int)
    labels[~finite] = -1
    out['chemical_behavior_cell'] = labels
    for i, name in enumerate([
        'beh_fraction_csp3', 'beh_aromatic_ring_density', 'beh_ring_density', 'beh_rotatable_density',
        'beh_heteroatom_density', 'beh_n_density', 'beh_hba_density', 'beh_hbd_density'
    ]):
        out[name] = vectors[:, i]
    return out


def _fingerprint_novelty_scores(mols, k=10, sample_limit=600, seed=1729):
    fps = []
    valid_indices = []
    for idx, mol in enumerate(mols):
        if mol is None:
            continue
        try:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
            valid_indices.append(idx)
        except Exception:
            continue
    scores = np.zeros(len(mols), dtype=np.float64)
    n = len(fps)
    if n <= 1:
        return scores
    rng = np.random.default_rng(int(seed))
    ref_indices = np.arange(n)
    if n > int(sample_limit):
        ref_indices = rng.choice(ref_indices, size=int(sample_limit), replace=False)
    ref_fps = [fps[i] for i in ref_indices]
    k = max(1, int(k))
    for local_idx, fp in enumerate(fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref_fps), dtype=np.float64)
        # Remove exact self when present; nearest-neighbor novelty uses distance to local population.
        dists = 1.0 - sims
        dists = dists[dists > 1e-12]
        if dists.size == 0:
            novelty = 0.0
        else:
            novelty = float(np.mean(np.sort(dists)[: min(k, dists.size)]))
        scores[valid_indices[local_idx]] = novelty
    max_score = float(np.nanmax(scores)) if scores.size else 0.0
    if max_score > 0:
        scores = scores / max_score
    return scores


def annotate_chemical_qd(frame, cfg=None):
    if frame is None or frame.empty:
        return frame
    cfg = cfg or {}
    out = annotate_chemical_behavior(frame, cfg)
    out['chemical_novelty_score'] = _fingerprint_novelty_scores(
        out['mol'].tolist(),
        k=int(cfg.get('chemical_novelty_k', 10)),
        sample_limit=int(cfg.get('chemical_novelty_sample_limit', 600)),
        seed=int(cfg.get('random_seed', cfg.get('chemical_behavior_seed', 1729))),
    )
    cell = pd.to_numeric(out.get('chemical_behavior_cell', -1), errors='coerce').fillna(-1).astype(int)
    counts = cell[cell.ge(0)].value_counts()
    out['chemical_cell_count'] = cell.map(lambda v: int(counts.get(int(v), 0)) if int(v) >= 0 else 0).astype(int)
    out['chemical_cell_rarity'] = cell.map(lambda v: 1.0 / np.sqrt(float(counts.get(int(v), 1))) if int(v) >= 0 else 0.0).astype(float)
    q = pd.to_numeric(out.get('pareto_objective_sum', np.nan), errors='coerce')
    if q.isna().all():
        q = pd.Series([_qd_quality_from_row(row) for row in out.to_dict('records')], index=out.index, dtype=float)
    q = q.fillna(0.0)
    if float(q.max()) > float(q.min()):
        q_norm = (q - float(q.min())) / max(float(q.max()) - float(q.min()), 1e-12)
    else:
        q_norm = pd.Series(0.0, index=out.index, dtype=float)
    out['chemical_quality_score'] = q_norm
    out['chemical_qd_score'] = (
        0.55 * out['chemical_quality_score'].astype(float)
        + 0.30 * out['chemical_novelty_score'].astype(float)
        + 0.15 * out['chemical_cell_rarity'].astype(float)
    )
    return out


def local_pareto_archive(frame, cfg=None, top_k=None):
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[] if frame is None else frame.columns)
    cfg = cfg or {}
    out = annotate_chemical_qd(frame, cfg)
    top_k = int(top_k if top_k is not None else cfg.get('chemical_local_archive_k', 3))
    top_k = max(1, top_k)
    objective_cols = [c for c in ['obj_h2_score', 'obj_dh_score', 'obj_scaffold_novelty'] if c in out.columns]
    rows = []
    for _, grp in out.groupby('chemical_behavior_cell', dropna=False):
        grp = grp.copy()
        grp = grp[pd.to_numeric(grp.get('chemical_behavior_cell', -1), errors='coerce').fillna(-1).astype(int).ge(0)]
        if grp.empty:
            continue
        if objective_cols:
            obj = grp[objective_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float64)
            finite = np.all(np.isfinite(obj), axis=1)
            if np.any(finite):
                finite_grp = grp.iloc[np.where(finite)[0]].copy()
                fronts = _non_dominated_sort_indices(obj[finite])
                rank = np.full(finite_grp.shape[0], np.inf, dtype=np.float64)
                for rank_no, rel in enumerate(fronts, start=1):
                    rank[rel] = rank_no
                finite_grp['_local_pareto_rank'] = rank
                grp = finite_grp
            else:
                grp['_local_pareto_rank'] = 1
        else:
            grp['_local_pareto_rank'] = 1
        grp = grp.sort_values(
            by=['_local_pareto_rank', 'chemical_qd_score', 'selection_score'],
            ascending=[True, False, False]
        ).head(top_k)
        rows.append(grp)
    if not rows:
        return out.head(0).copy()
    archived = pd.concat(rows, ignore_index=True, sort=False)
    archived['chemical_local_archive_elite'] = 1
    return archived.drop(columns=['_local_pareto_rank'], errors='ignore')


def _estimate_prefilter_score(frame):
    out = frame.copy()
    score = pd.Series(0.0, index=out.index, dtype=float)
    origin = out.get('origin_type', pd.Series('', index=out.index)).fillna('').astype(str)
    score += origin.eq('seed_immigrant').astype(float) * 0.55
    score += origin.eq('guided_mutation').astype(float) * 0.45
    score += origin.eq('crossover').astype(float) * 0.30
    score += origin.eq('initial_seed').astype(float) * 0.10
    if 'parent_h2_wt_percent' in out.columns:
        ph2 = pd.to_numeric(out['parent_h2_wt_percent'], errors='coerce')
        pdh = pd.to_numeric(out.get('parent_delta_H_per_H2', np.nan), errors='coerce')
        psa = pd.to_numeric(out.get('parent_SA_score', np.nan), errors='coerce')
        parent_hit = ph2.ge(4.2) & pdh.between(DH_LOW, DH_HIGH) & psa.le(4.0)
        score += parent_hit.fillna(False).astype(float) * 0.30
        score += ph2.fillna(0).between(3.8, 4.8).astype(float) * 0.10
    if 'operator_probability' in out.columns:
        prob = pd.to_numeric(out['operator_probability'], errors='coerce').fillna(0.0)
        score += np.clip(1.0 - prob, 0.0, 1.0) * 0.08
    score += (pd.util.hash_pandas_object(out['smi'].astype(str), index=False).astype('uint64') % 1000) / 1000000.0
    out['pre_gnn_acquisition_score'] = score
    return out


def apply_gnn_budget_prefilter(cached, pending, cfg):
    max_pending = int(cfg.get('max_pending_gnn_per_generation', 0) or 0)
    if pending is None or pending.empty or max_pending <= 0 or pending.shape[0] <= max_pending:
        if pending is not None and not pending.empty and 'pre_gnn_acquisition_score' not in pending.columns:
            pending = _estimate_prefilter_score(pending)
        return pending, 0
    scored = _estimate_prefilter_score(pending)
    keep_parts = []
    used = set()
    origin_quota = max(1, int(max_pending * 0.15))
    if 'origin_type' in scored.columns:
        for _, grp in scored.groupby(scored['origin_type'].fillna('').astype(str)):
            part = grp.sort_values('pre_gnn_acquisition_score', ascending=False).head(origin_quota)
            keep_parts.append(part)
            used.update(part.index.tolist())
    remaining = max_pending - len(used)
    if remaining > 0:
        rest = scored.loc[~scored.index.isin(used)].sort_values('pre_gnn_acquisition_score', ascending=False).head(remaining)
        keep_parts.append(rest)
    kept = pd.concat(keep_parts, ignore_index=False, sort=False) if keep_parts else scored.head(max_pending)
    kept = kept.sort_values('pre_gnn_acquisition_score', ascending=False).head(max_pending).copy()
    return kept, int(scored.shape[0] - kept.shape[0])
DELTA_H_CACHE_RING = {}
DELTA_H_CACHE_FULL = {}
DELTA_H_UNCERTAINTY_CACHE_RING = {}
DELTA_H_UNCERTAINTY_CACHE_FULL = {}
GA_FITNESS_CONFIG = {}
BIO_BASED_FP_LIST = []
BIO_BASED_SMI_SET = set()
BIO_BASED_SIM_CACHE = {}
TARGET_TEMPLATE_CACHE = {}
SAT_SMILES_CACHE_RING = {}
SAT_SMILES_CACHE_FULL = {}
H2_COUNT_CACHE = {}
BIO_BASED_ALLOWED_ATOMIC_NUMS = None
BIO_PLATFORM_CACHE = None
BIO_BASED_DERIVATIZATION_ATOMIC_NUMS = {7}

# Bio-based target motifs used during post-generation scoring.
# These are not "initialization rules"; they are reference chemotypes for
# prioritizing generated molecules after prediction and hard filtering.
_TARGET_TEMPLATE_SMARTS = [
    "c1ccoc1",                # furan core
    "O=CC1=COC=C1",           # furfural-like
    "OCC1=COC=C1",            # HMF-like (hydroxymethyl furan)
    "c1ccc2occc2c1",          # benzofuran
    "c1ccc(cc1)O",            # phenol (lignin platform motif)
    "c1ccc(cc1)OC",           # anisole/guaiacol-like motif
    "c1cc(ccc1O)OC",          # guaiacol-like
    "c1cc(cc(c1O)OC)OC"       # syringol-like
]
_TARGET_TEMPLATE_MOLS = [Chem.MolFromSmarts(s) for s in _TARGET_TEMPLATE_SMARTS]
_FURAN_CORE_MOL = Chem.MolFromSmarts("c1ccoc1")

# Canonical scaffold/platform dictionary for the seed library.
# The workflow starts from preloaded bio-based scaffolds, then explores outward
# by crossover and mutation while trying to retain these platform identities.
BIO_PLATFORM_LIBRARY = [
    {
        'platform_id': 'glycerol_like',
        'platform_name': 'Glycerol-like polyol',
        'anchor_smi': 'OCC(O)CO',
        'smarts': '[OX2H][CX4][CX4]([OX2H])[CX4][OX2H]',
        'retain_similarity_floor': 0.35,
    },
    {
        'platform_id': 'fatty_acid_like',
        'platform_name': 'Fatty-acid-like chain',
        'anchor_smi': 'CCCCCC(=O)O',
        'smarts': '[CX3](=O)[OX2H1]',
        'retain_similarity_floor': 0.45,
    },
    {
        'platform_id': 'levulinic_like',
        'platform_name': 'Levulinic-acid-like',
        'anchor_smi': 'CC(=O)CC(=O)O',
        'smarts': '[CX3](=O)[CX4][CX4][CX3](=O)[OX2H1]',
        'retain_similarity_floor': 0.40,
    },
    {
        'platform_id': 'furan',
        'platform_name': 'Furan core',
        'anchor_smi': 'c1ccoc1',
        'smarts': 'c1ccoc1',
        'retain_similarity_floor': 0.45,
    },
    {
        'platform_id': 'furfural_like',
        'platform_name': 'Furfural-like',
        'anchor_smi': 'O=Cc1ccco1',
        'smarts': '[CX3H1](=O)c1ccco1',
        'retain_similarity_floor': 0.45,
    },
    {
        'platform_id': 'furfuryl_alcohol_like',
        'platform_name': 'Furfuryl-alcohol-like',
        'anchor_smi': 'OCc1ccco1',
        'smarts': '[OX2H][CX4]c1ccco1',
        'retain_similarity_floor': 0.45,
    },
    {
        'platform_id': 'phenol',
        'platform_name': 'Phenol core',
        'anchor_smi': 'Oc1ccccc1',
        'smarts': 'c1ccc(cc1)O',
        'retain_similarity_floor': 0.45,
    },
    {
        'platform_id': 'guaiacol_like',
        'platform_name': 'Guaiacol-like',
        'anchor_smi': 'COc1ccccc1O',
        'smarts': 'c1cc(ccc1O)OC',
        'retain_similarity_floor': 0.45,
    },
    {
        'platform_id': 'syringol_like',
        'platform_name': 'Syringol-like',
        'anchor_smi': 'COc1cc(OC)c(O)cc1',
        'smarts': 'c1cc(cc(c1O)OC)OC',
        'retain_similarity_floor': 0.45,
    },
    {
        'platform_id': 'benzofuran',
        'platform_name': 'Benzofuran core',
        'anchor_smi': 'c1ccc2occc2c1',
        'smarts': 'c1ccc2occc2c1',
        'retain_similarity_floor': 0.50,
    },
]

# Hard-filter and clustering defaults used before Pareto ranking.
SCREENING_DEFAULTS = {
    'sa_max': 6.0,
    'hba_min': 0,
    'hba_max': 5,
    'hbd_min': 0,
    'hbd_max': 2,
    'tpsa_min': 0.0,
    'tpsa_max': 120.0,
    'h2wt_min': 4.2,
    'exclude_n_triple': True,
    'selection_ratio': 0.10,
    'selection_by_ratio': True,
    'n_clusters': 10,
    'representative_mode': 'best',
    'fingerprint_radius': 2,
    'fingerprint_bits': 1024
}

# Pareto ranking defaults applied after the hard filter has reduced the
# generated pool to chemically and property-reasonable candidates.
PARETO_DEFAULTS = {
    'dh_target': DH_TARGET,
    'dh_window_low': DH_LOW,
    'dh_window_high': DH_HIGH,
    'dh_window_penalty': 0.35,
    'h2_objective_min': 4.2,
    'h2_objective_max': 7.0,
    'sa_objective_best': 1.0,
    'sa_objective_max': 4.0,
    'pareto_objectives': [
        'obj_h2_score',
        'obj_dh_score',
        'obj_scaffold_novelty',
    ],
    'pareto_quality_bins': 6,
    'lightweight_screening': True,
    'representative_pareto_front_only': True,
    'representative_require_showcase_reasonable': True,
    'bio_objective_weight': 0.35,
    'template_objective_weight': 0.25,
    'selection_ratio': 0.10,
    'selection_by_ratio': True,
    'n_clusters': 10,
    'representative_mode': 'best',
    'fingerprint_radius': 2,
    'fingerprint_bits': 1024
}

_PAINS_CATALOG = None

def _get_pains_catalog():
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            _PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
        except Exception:
            _PAINS_CATALOG = False
    return _PAINS_CATALOG

def _target_template_features(mol):
    if mol is None:
        return 0.0, 0, 0
    hit_flags = [int(mol.HasSubstructMatch(t)) for t in _TARGET_TEMPLATE_MOLS if t is not None]
    if len(hit_flags) == 0:
        return 0.0, 0, 0
    hit_count = int(sum(hit_flags))
    # Extra emphasis for furan-containing motifs to guide bio-based candidates.
    furan_bonus = 1 if (_FURAN_CORE_MOL is not None and mol.HasSubstructMatch(_FURAN_CORE_MOL)) else 0
    reward = float(hit_count + 2.0 * furan_bonus)
    hit_binary = 1 if hit_count > 0 else 0
    return reward, hit_binary, furan_bonus


def _target_template_features_cached(smi):
    if smi in TARGET_TEMPLATE_CACHE:
        return TARGET_TEMPLATE_CACHE[smi]
    mol = Chem.MolFromSmiles(smi)
    val = _target_template_features(mol)
    TARGET_TEMPLATE_CACHE[smi] = val
    return val

def _get_biobased_allowed_atomic_nums():
    global BIO_BASED_ALLOWED_ATOMIC_NUMS
    if BIO_BASED_ALLOWED_ATOMIC_NUMS is not None:
        return BIO_BASED_ALLOWED_ATOMIC_NUMS
    df = _load_biobased_dataframe()
    allowed = set()
    for smi in df.get('smi', pd.Series(dtype=str)).dropna().astype(str).tolist():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            z = atom.GetAtomicNum()
            if z > 1:
                allowed.add(z)
    if len(allowed) == 0:
        allowed = {6, 8}
    # The bio-based seed/platform library is mostly C/O, but LOHC candidates can
    # reasonably include nitrogen through biomass-derived amination or N-heteroarene
    # derivatization. Keep this as an allowed exploration grammar element rather
    # than a Pareto target so the search is not biased toward a specific motif.
    allowed.update(BIO_BASED_DERIVATIZATION_ATOMIC_NUMS)
    BIO_BASED_ALLOWED_ATOMIC_NUMS = allowed
    return BIO_BASED_ALLOWED_ATOMIC_NUMS

def _contains_only_allowed_elements(mol, allowed_atomic_nums=None):
    if mol is None:
        return False
    allowed = allowed_atomic_nums if allowed_atomic_nums is not None else _get_biobased_allowed_atomic_nums()
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z <= 1:
            continue
        if z not in allowed:
            return False
    return True

def _init_platform_cache():
    global BIO_PLATFORM_CACHE
    if BIO_PLATFORM_CACHE is not None:
        return BIO_PLATFORM_CACHE
    cache = []
    for entry in BIO_PLATFORM_LIBRARY:
        anchor_mol = Chem.MolFromSmiles(entry['anchor_smi'])
        pattern_mol = Chem.MolFromSmarts(entry['smarts'])
        if anchor_mol is None:
            continue
        cache.append({
            **entry,
            'anchor_mol': anchor_mol,
            'pattern_mol': pattern_mol,
            'anchor_fp': AllChem.GetMorganFingerprintAsBitVect(anchor_mol, 2, nBits=1024),
        })
    BIO_PLATFORM_CACHE = cache
    return BIO_PLATFORM_CACHE

def _get_platform_entry(platform_id):
    for entry in _init_platform_cache():
        if entry['platform_id'] == platform_id:
            return entry
    return None

def _assign_platform_family(mol):
    if mol is None:
        return {
            'platform_id': 'unassigned',
            'platform_name': 'Unassigned',
            'platform_anchor_smi': '',
            'platform_match_score': 0.0,
            'platform_core_retained': 0,
        }
    mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    best_entry = None
    best_sim = 0.0
    best_sub_hit = 0
    best_score = -np.inf
    for entry in _init_platform_cache():
        sub_hit = 1 if (entry['pattern_mol'] is not None and mol.HasSubstructMatch(entry['pattern_mol'])) else 0
        sim = float(DataStructs.TanimotoSimilarity(mol_fp, entry['anchor_fp']))
        score = 2.0 * float(sub_hit) + sim
        if score > best_score:
            best_score = score
            best_entry = entry
            best_sim = sim
            best_sub_hit = sub_hit
    if best_entry is None or (best_sub_hit == 0 and best_sim < 0.35):
        return {
            'platform_id': 'unassigned',
            'platform_name': 'Unassigned',
            'platform_anchor_smi': '',
            'platform_match_score': float(best_sim),
            'platform_core_retained': 0,
        }
    return {
        'platform_id': best_entry['platform_id'],
        'platform_name': best_entry['platform_name'],
        'platform_anchor_smi': best_entry['anchor_smi'],
        'platform_match_score': float(best_score),
        'platform_core_retained': int(best_sub_hit),
    }

def _platform_core_is_retained(mol, platform_id):
    if mol is None or platform_id is None or str(platform_id).strip() == '' or str(platform_id) == 'unassigned':
        return True
    entry = _get_platform_entry(str(platform_id))
    if entry is None:
        return True
    if entry['pattern_mol'] is not None and mol.HasSubstructMatch(entry['pattern_mol']):
        return True
    mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    sim = float(DataStructs.TanimotoSimilarity(mol_fp, entry['anchor_fp']))
    return sim >= float(entry.get('retain_similarity_floor', 0.4))

def _safe_murcko_scaffold_smiles(mol):
    if mol is None:
        return ''
    try:
        smi = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return str(smi) if smi is not None else ''
    except Exception:
        return ''

def annotate_platform_families(df_src):
    df = _copy_with_mol(df_src)
    details = [_assign_platform_family(mol) for mol in df['mol'].tolist()]
    df['platform_id'] = [d['platform_id'] for d in details]
    df['platform_name'] = [d['platform_name'] for d in details]
    df['platform_anchor_smi'] = [d['platform_anchor_smi'] for d in details]
    df['platform_match_score'] = [d['platform_match_score'] for d in details]
    df['platform_core_retained'] = [d['platform_core_retained'] for d in details]
    df['platform_family_assigned'] = (df['platform_id'] != 'unassigned').astype(int)
    return df

def sample_platform_balanced(df_src, sample_size, replace=None, random_state=None):
    if df_src is None or df_src.shape[0] == 0 or sample_size <= 0:
        return pd.DataFrame(columns=df_src.columns if df_src is not None else [])
    df = df_src.copy()
    if 'platform_id' not in df.columns:
        df = annotate_platform_families(df)
    valid = df[df['platform_id'].astype(str) != 'unassigned'].copy()
    source = valid if valid.shape[0] > 0 else df
    groups = [grp.copy() for _, grp in source.groupby('platform_id')]
    if len(groups) == 0:
        rep = bool(replace) if replace is not None else (source.shape[0] < int(sample_size))
        return source.sample(n=int(sample_size), replace=rep, random_state=random_state).copy()
    groups = sorted(groups, key=lambda g: str(g['platform_id'].iloc[0]))
    base = int(sample_size) // len(groups)
    rem = int(sample_size) % len(groups)
    draws = []
    used_smis = set()
    total_drawn = 0
    for i, grp in enumerate(groups):
        n_take = base + (1 if i < rem else 0)
        if n_take <= 0:
            continue
        rep = bool(replace) if replace is not None else (grp.shape[0] < n_take)
        if not rep:
            n_take = min(n_take, grp.shape[0])
        if n_take <= 0:
            continue
        sampled_grp = grp.sample(
            n=n_take,
            replace=rep,
            random_state=None if random_state is None else int(random_state) + i
        )
        draws.append(sampled_grp)
        total_drawn += sampled_grp.shape[0]
        if 'smi' in sampled_grp.columns:
            used_smis.update(sampled_grp['smi'].astype(str).tolist())
    if len(draws) == 0:
        rep = bool(replace) if replace is not None else (source.shape[0] < int(sample_size))
        return source.sample(n=int(sample_size), replace=rep, random_state=random_state).copy()
    out = pd.concat(draws, ignore_index=True)
    remaining = int(sample_size) - total_drawn
    if remaining > 0:
        if bool(replace) if replace is not None else False:
            refill_pool = source
            refill_replace = True
        else:
            refill_pool = source[~source['smi'].astype(str).isin(used_smis)].copy() if 'smi' in source.columns else source.copy()
            refill_replace = refill_pool.shape[0] < remaining
        if refill_pool.shape[0] > 0:
            refill = refill_pool.sample(
                n=min(remaining, refill_pool.shape[0]) if not refill_replace else remaining,
                replace=refill_replace,
                random_state=None if random_state is None else int(random_state) + 999
            )
            out = pd.concat([out, refill], ignore_index=True)
    return out

def _predict_delta_H_cached(sat_list, aromatic_list, cache):
    n = len(aromatic_list)
    out = [np.nan] * n
    miss_sat = []
    miss_arom = []
    miss_keys = []
    for sat, arom in zip(sat_list, aromatic_list):
        if sat is None or arom is None:
            continue
        key = (sat, arom)
        if key not in cache:
            miss_keys.append(key)
            miss_sat.append(sat)
            miss_arom.append(arom)
    if len(miss_keys) > 0:
        preds = predict_delta_H(miss_sat, miss_arom)
        for key, val in zip(miss_keys, preds):
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                cache[key] = np.nan
            else:
                cache[key] = float(val)
    for i, (sat, arom) in enumerate(zip(sat_list, aromatic_list)):
        if sat is None or arom is None:
            continue
        out[i] = cache.get((sat, arom), np.nan)
    return out


def _predict_delta_H_uncertainty_cached(sat_list, aromatic_list, cache):
    n = len(aromatic_list)
    mean_out = [np.nan] * n
    std_out = [np.nan] * n
    miss_sat = []
    miss_arom = []
    miss_keys = []
    for sat, arom in zip(sat_list, aromatic_list):
        if sat is None or arom is None:
            continue
        key = (sat, arom)
        if key not in cache:
            miss_keys.append(key)
            miss_sat.append(sat)
            miss_arom.append(arom)
    if len(miss_keys) > 0:
        means, stds = predict_delta_H_with_uncertainty(miss_sat, miss_arom)
        for key, mean, std in zip(miss_keys, means, stds):
            mean = np.nan if mean is None else float(mean)
            std = np.nan if std is None else float(std)
            if not np.isfinite(mean):
                cache[key] = (np.nan, np.nan)
            else:
                cache[key] = (mean, std if np.isfinite(std) else np.nan)
    for i, (sat, arom) in enumerate(zip(sat_list, aromatic_list)):
        if sat is None or arom is None:
            continue
        mean, std = cache.get((sat, arom), (np.nan, np.nan))
        mean_out[i] = mean
        std_out[i] = std
    return mean_out, std_out


def _sigmoid_probability(value, center, width, increasing=True):
    if value is None or not np.isfinite(value):
        return 0.0
    z = (float(value) - float(center)) / max(float(width), 1e-9)
    if not increasing:
        z = -z
    return float(1.0 / (1.0 + np.exp(-z)))


def _window_probability(value, low, high, edge_width):
    if value is None or not np.isfinite(value):
        return 0.0
    left = _sigmoid_probability(value, low, edge_width, increasing=True)
    right = _sigmoid_probability(value, high, edge_width, increasing=False)
    return float(left * right)


def _liquid_proxy_probability(mol):
    if mol is None:
        return 0.0
    try:
        aromatic = float(rdmd.CalcNumAromaticRings(mol))
        rings = float(rdmd.CalcNumRings(mol))
        rotb = float(Descriptors.NumRotatableBonds(mol))
        frac_csp3 = float(rdmd.CalcFractionCSP3(mol))
        heavy = max(1.0, float(mol.GetNumHeavyAtoms()))
        hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in (1, 6)) / heavy
    except Exception:
        return 0.0
    rigid_penalty = _sigmoid_probability(aromatic, 3.0, 0.8, increasing=False)
    ring_penalty = _sigmoid_probability(rings, 5.0, 1.0, increasing=False)
    flexibility = _sigmoid_probability(rotb, 1.0, 1.0, increasing=True)
    saturation = 0.45 + 0.55 * np.clip(frac_csp3, 0.0, 1.0)
    hetero_balance = _window_probability(hetero, 0.03, 0.35, 0.06)
    return float(np.clip(rigid_penalty * ring_penalty * flexibility * saturation * hetero_balance, 0.0, 1.0))


def _lohc_probability_score(mol, dh, dh_uncertainty, h2wt, sa, mw, cfg):
    h2_prob = _sigmoid_probability(h2wt, float(cfg.get('lohc_h2_min', 5.5)), 0.35, increasing=True)
    dh_prob = _window_probability(
        dh,
        float(cfg.get('lohc_dh_low', DH_LOW)),
        float(cfg.get('lohc_dh_high', DH_HIGH)),
        float(cfg.get('lohc_dh_edge_width', 4.0)),
    )
    sa_prob = _sigmoid_probability(sa, float(cfg.get('lohc_sa_max', 4.0)), 0.45, increasing=False)
    mw_prob = _sigmoid_probability(mw, float(cfg.get('lohc_mw_max', 260.0)), 25.0, increasing=False)
    liquid_prob = _liquid_proxy_probability(mol)
    if bool(cfg.get('uncertainty_penalty_enabled', False)):
        uncertainty_prob = _sigmoid_probability(
            dh_uncertainty,
            float(cfg.get('uncertainty_threshold', 20.0)),
            float(cfg.get('uncertainty_soft_width', 5.0)),
            increasing=False,
        )
    else:
        uncertainty_prob = 1.0
    probs = [h2_prob, dh_prob, sa_prob, mw_prob, liquid_prob, uncertainty_prob]
    return float(np.prod(np.clip(probs, 0.0, 1.0)))

def _copy_with_mol(df_src):
    df = df_src.copy()
    if 'smi' not in df.columns:
        if 'mol' not in df.columns:
            df['mol'] = None
        return df
    def _coerce_mol(row):
        m = row.get('mol', None)
        if isinstance(m, Chem.Mol):
            return m
        mol = None
        if isinstance(m, str) and m and not m.startswith('<rdkit.'):
            mol = Chem.MolFromSmiles(m)
        if mol is None:
            smi = row.get('smi', '')
            mol = Chem.MolFromSmiles(str(smi)) if str(smi) else None
        return mol
    df['mol'] = [_coerce_mol(row) for row in df.to_dict('records')]
    return df

def _passes_generation_limits(sa_val, mol_wt, SA_max=None, max_mol_weight=None):
    if SA_max is not None and np.isfinite(sa_val) and sa_val > float(SA_max):
        return False
    if max_mol_weight is not None and np.isfinite(mol_wt) and mol_wt > float(max_mol_weight):
        return False
    return True

def _is_unreasonable_molecule(mol):
    if mol is None:
        return True
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return True
    if not _contains_only_allowed_elements(mol):
        return True
    if len(Chem.GetMolFrags(mol)) != 1:
        return True
    if mol.GetNumHeavyAtoms() < 4:
        return True
    try:
        if any(len(ring) <= 4 for ring in mol.GetRingInfo().AtomRings()):
            return True
    except Exception:
        return True
    forbidden_patterns = [
        Chem.MolFromSmarts("OO"),                  # peroxide-like
        Chem.MolFromSmarts("[CX3](=O)[Cl,Br,I,F]") # acyl halide
    ]
    if any(p is not None and mol.HasSubstructMatch(p) for p in forbidden_patterns):
        return True
    return False

def _is_showcase_reasonable(mol):
    if _is_unreasonable_molecule(mol):
        return False
    if mol is None:
        return False
    try:
        ring_info = mol.GetRingInfo()
        if any(len(ring) <= 4 for ring in ring_info.AtomRings()):
            return False
    except Exception:
        return False
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            return False
        if atom.GetNumRadicalElectrons() != 0:
            return False
        if atom.GetAtomicNum() == 0:
            return False
    forbidden_patterns = [
        Chem.MolFromSmarts("[CX2]=[CX2]=[OX1]"),   # ketene-like
        Chem.MolFromSmarts("[NX2]=[CX2]=[OX1]"),   # isocyanate-like
        Chem.MolFromSmarts("[NX2]=[CX2]=[SX1]"),   # isothiocyanate-like
        Chem.MolFromSmarts("[N]=[N+]=[N-]"),       # azide-like
        Chem.MolFromSmarts("[C-]"),
        Chem.MolFromSmarts("[N-]"),
        Chem.MolFromSmarts("[O-]"),
    ]
    if any(p is not None and mol.HasSubstructMatch(p) for p in forbidden_patterns):
        return False
    return True

def annotate_structure_reasonableness(df_src):
    df = _copy_with_mol(df_src)
    small_ring_flags = []
    min_ring_sizes = []
    num_rings = []
    num_aromatic_rings = []
    formal_charge_sums = []
    radical_flags = []
    showcase_flags = []
    ring_penalties = []

    for mol in df['mol'].tolist():
        if mol is None:
            small_ring_flags.append(np.nan)
            min_ring_sizes.append(np.nan)
            num_rings.append(np.nan)
            num_aromatic_rings.append(np.nan)
            formal_charge_sums.append(np.nan)
            radical_flags.append(np.nan)
            showcase_flags.append(0)
            ring_penalties.append(np.nan)
            continue
        try:
            atom_rings = list(mol.GetRingInfo().AtomRings())
        except Exception:
            atom_rings = []
        ring_sizes = [len(r) for r in atom_rings]
        min_ring = min(ring_sizes) if len(ring_sizes) > 0 else np.nan
        has_small_ring = any(s <= 4 for s in ring_sizes)
        try:
            aromatic_ring_count = int(rdmd.CalcNumAromaticRings(mol))
        except Exception:
            aromatic_ring_count = np.nan
        charge_sum = int(sum(abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()))
        has_radical = any(atom.GetNumRadicalElectrons() != 0 for atom in mol.GetAtoms())

        small_ring_flags.append(int(has_small_ring))
        min_ring_sizes.append(min_ring)
        num_rings.append(len(ring_sizes))
        num_aromatic_rings.append(aromatic_ring_count)
        formal_charge_sums.append(charge_sum)
        radical_flags.append(int(has_radical))
        showcase_flags.append(1 if _is_showcase_reasonable(mol) else 0)
        try:
            ring_penalties.append(calc_ring_penalty(mol))
        except Exception:
            ring_penalties.append(np.nan)

    df['has_small_ring'] = small_ring_flags
    df['min_ring_size'] = min_ring_sizes
    df['num_rings'] = num_rings
    df['num_aromatic_rings'] = num_aromatic_rings
    df['formal_charge_abs_sum'] = formal_charge_sums
    df['has_radical'] = radical_flags
    df['ring_penalty'] = ring_penalties
    df['structure_showcase_pass'] = showcase_flags
    df['structure_reasonable_pass'] = (
        pd.Series(showcase_flags, index=df.index).fillna(0).astype(int)
    )
    return df

def _representative_sort_frame(df_src):
    df = _sort_by_pareto_priority(df_src)
    if 'SA score' in df.columns:
        df = df.sort_values(
            by=['pareto_rank', 'cluster_showcase_pass', 'crowding_distance', 'selection_score', 'fitness value'],
            ascending=[True, False, False, False, False]
        )
    return df

def _pick_cluster_representative(grp, config=None):
    cfg = PARETO_DEFAULTS.copy()
    if config:
        cfg.update(config)
    grp = grp.copy()
    if grp.shape[0] == 0:
        return pd.DataFrame()
    require_showcase = bool(cfg.get('representative_require_showcase_reasonable', True))
    front_only = bool(cfg.get('representative_pareto_front_only', True))

    grp['cluster_showcase_pass'] = [
        1 if _is_showcase_reasonable(mol) else 0 for mol in grp['mol'].tolist()
    ]

    candidate_sets = []
    if front_only and require_showcase:
        candidate_sets.append(('front1_showcase', grp[(pd.to_numeric(grp['pareto_rank'], errors='coerce') == 1) & (grp['cluster_showcase_pass'] == 1)].copy()))
    if require_showcase:
        candidate_sets.append(('best_showcase', grp[grp['cluster_showcase_pass'] == 1].copy()))
    if front_only:
        candidate_sets.append(('front1_any', grp[pd.to_numeric(grp['pareto_rank'], errors='coerce') == 1].copy()))
    candidate_sets.append(('best_any', grp.copy()))

    for source, cand in candidate_sets:
        if cand.shape[0] == 0:
            continue
        rep = _representative_sort_frame(cand).head(1).copy()
        rep['cluster_representative_source'] = source
        rep['cluster_representative'] = 1
        return rep
    return pd.DataFrame()

def annotate_multicriteria_properties(df_src):
    df = _copy_with_mol(df_src)
    pains_catalog = _get_pains_catalog()
    pre_sa = None
    if 'SA score' in df.columns:
        pre_sa = pd.to_numeric(df['SA score'], errors='coerce')

    sa_vals = []
    hba_vals = []
    hbd_vals = []
    tpsa_vals = []
    logp_vals = []
    molwt_vals = []
    pains_flags = []

    for i, mol in enumerate(df['mol'].tolist()):
        if mol is None:
            sa_vals.append(np.nan)
            hba_vals.append(np.nan)
            hbd_vals.append(np.nan)
            tpsa_vals.append(np.nan)
            logp_vals.append(np.nan)
            molwt_vals.append(np.nan)
            pains_flags.append(np.nan)
            continue
        if pre_sa is not None and i < len(pre_sa) and np.isfinite(pre_sa.iloc[i]):
            sa_vals.append(float(pre_sa.iloc[i]))
        else:
            try:
                sa_vals.append(float(SA.calculateScore(mol)))
            except Exception:
                sa_vals.append(np.nan)
        hba_vals.append(float(rdmd.CalcNumHBA(mol)))
        hbd_vals.append(float(rdmd.CalcNumHBD(mol)))
        tpsa_vals.append(float(rdmd.CalcTPSA(mol)))
        logp_vals.append(float(Descriptors.MolLogP(mol)))
        molwt_vals.append(float(Descriptors.MolWt(mol)))
        if pains_catalog is False:
            pains_flags.append(np.nan)
        else:
            try:
                pains_flags.append(bool(pains_catalog.HasMatch(mol)))
            except Exception:
                pains_flags.append(np.nan)

    df['sa_screen'] = sa_vals
    df['hba'] = hba_vals
    df['hbd'] = hbd_vals
    df['tpsa'] = tpsa_vals
    df['logp'] = logp_vals
    df['molwt_screen'] = molwt_vals
    df['pains_flag'] = pains_flags
    return df

def apply_multicriteria_filter(df_src, config=None):
    cfg = SCREENING_DEFAULTS.copy()
    if config:
        cfg.update(config)

    if bool(cfg.get('lightweight_screening', False)):
        df = _copy_with_mol(df_src)
        if 'SA score' in df.columns:
            df['sa_screen'] = pd.to_numeric(df['SA score'], errors='coerce')
        else:
            df['sa_screen'] = np.nan
        if 'molwt_screen' not in df.columns:
            df['molwt_screen'] = df['mol'].apply(lambda m: float(Descriptors.MolWt(m)) if m is not None else np.nan)
        for col in ['hba', 'hbd', 'tpsa', 'logp']:
            if col not in df.columns:
                df[col] = np.nan
        if 'pains_flag' not in df.columns:
            df['pains_flag'] = np.nan
    else:
        df = annotate_multicriteria_properties(df_src)

    pass_sa = pd.to_numeric(df['sa_screen'], errors='coerce') <= float(cfg['sa_max'])
    pains_raw = df['pains_flag']
    pass_pains = pains_raw.isna() | (~pains_raw.astype(bool))
    pass_hba = pd.to_numeric(df['hba'], errors='coerce').between(float(cfg['hba_min']), float(cfg['hba_max']), inclusive='both')
    pass_hbd = pd.to_numeric(df['hbd'], errors='coerce').between(float(cfg['hbd_min']), float(cfg['hbd_max']), inclusive='both')
    pass_tpsa = pd.to_numeric(df['tpsa'], errors='coerce').between(float(cfg['tpsa_min']), float(cfg['tpsa_max']), inclusive='both')

    if 'delta_H_per_H2' in df.columns:
        x = pd.to_numeric(df['delta_H_per_H2'], errors='coerce').to_numpy(dtype=np.float64)
        pass_enthalpy = pd.Series(np.isfinite(x) & (x >= DH_LOW) & (x <= DH_HIGH), index=df.index)
    else:
        pass_enthalpy = pd.Series(False, index=df.index)

    if 'h2_wt_percent' in df.columns:
        pass_h2wt = pd.to_numeric(df['h2_wt_percent'], errors='coerce') >= float(cfg['h2wt_min'])
    else:
        pass_h2wt = pd.Series(False, index=df.index)

    if bool(cfg.get('exclude_n_triple', False)):
        patt_n_triple = Chem.MolFromSmarts("[#7]#*")
        pass_n_triple = pd.Series(
            [False if (m is None or patt_n_triple is None) else (not m.HasSubstructMatch(patt_n_triple)) for m in df['mol'].tolist()],
            index=df.index
        )
    else:
        pass_n_triple = pd.Series(True, index=df.index)

    df['pass_sa'] = pass_sa.fillna(False).astype(int)
    df['pass_pains'] = pass_pains.fillna(False).astype(int)
    df['pass_hba'] = pass_hba.fillna(False).astype(int)
    df['pass_hbd'] = pass_hbd.fillna(False).astype(int)
    df['pass_tpsa'] = pass_tpsa.fillna(False).astype(int)
    df['pass_enthalpy_window'] = pass_enthalpy.fillna(False).astype(int)
    df['pass_h2wt'] = pass_h2wt.fillna(False).astype(int)
    df['pass_n_triple'] = pass_n_triple.fillna(False).astype(int)

    pass_reasonable = pd.Series(
        [not _is_unreasonable_molecule(m) for m in df['mol'].tolist()],
        index=df.index
    )
    hard_selected = pass_reasonable.fillna(False)

    # Multi-criteria score for ratio-based selection.
    score = (
        df['pass_sa'].astype(float)
        + df['pass_pains'].astype(float)
        + df['pass_hba'].astype(float)
        + df['pass_hbd'].astype(float)
        + df['pass_tpsa'].astype(float)
        + df['pass_enthalpy_window'].astype(float)
        + df['pass_h2wt'].astype(float)
        + df['pass_n_triple'].astype(float)
    )
    if 'fitness value' in df.columns:
        fit = pd.to_numeric(df['fitness value'], errors='coerce').fillna(0.0)
        # Fitness acts as tie-breaker instead of dominating multi-criteria score.
        score = score + 0.001 * fit
    df['multicriteria_score'] = score

    selected = hard_selected.copy()
    if bool(cfg.get('selection_by_ratio', True)):
        n_total = int(df.shape[0])
        ratio = float(cfg.get('selection_ratio', 0.10))
        ratio = min(max(ratio, 0.0), 1.0)
        k = int(np.ceil(ratio * n_total))
        k = max(1, min(k, n_total))
        top_idx = df.sort_values(by=['multicriteria_score'], ascending=False).head(k).index
        selected = pd.Series(False, index=df.index)
        selected.loc[top_idx] = True

    df['selected_hardfilter'] = hard_selected.astype(int)
    df['selected_multifilter'] = selected.astype(int)
    df['pass_reasonable'] = pass_reasonable.fillna(False).astype(int)
    return df

def _multifilter_summary(df):
    if df is None or df.shape[0] == 0:
        return pd.DataFrame(columns=['stage', 'passed', 'retention_ratio'])

    total = int(df.shape[0])
    stages = [
        ('total', np.ones(total, dtype=bool)),
        ('sa', df['pass_sa'].to_numpy(dtype=np.int64) == 1),
        ('pains', (df['pass_sa'].to_numpy(dtype=np.int64) == 1) & (df['pass_pains'].to_numpy(dtype=np.int64) == 1)),
        ('hba', (df['pass_sa'].to_numpy(dtype=np.int64) == 1) & (df['pass_pains'].to_numpy(dtype=np.int64) == 1) & (df['pass_hba'].to_numpy(dtype=np.int64) == 1)),
        ('hbd', (df['pass_sa'].to_numpy(dtype=np.int64) == 1) & (df['pass_pains'].to_numpy(dtype=np.int64) == 1) & (df['pass_hba'].to_numpy(dtype=np.int64) == 1) & (df['pass_hbd'].to_numpy(dtype=np.int64) == 1)),
        ('tpsa', (df['pass_sa'].to_numpy(dtype=np.int64) == 1) & (df['pass_pains'].to_numpy(dtype=np.int64) == 1) & (df['pass_hba'].to_numpy(dtype=np.int64) == 1) & (df['pass_hbd'].to_numpy(dtype=np.int64) == 1) & (df['pass_tpsa'].to_numpy(dtype=np.int64) == 1)),
        ('enthalpy_window', (df['pass_sa'].to_numpy(dtype=np.int64) == 1) & (df['pass_pains'].to_numpy(dtype=np.int64) == 1) & (df['pass_hba'].to_numpy(dtype=np.int64) == 1) & (df['pass_hbd'].to_numpy(dtype=np.int64) == 1) & (df['pass_tpsa'].to_numpy(dtype=np.int64) == 1) & (df['pass_enthalpy_window'].to_numpy(dtype=np.int64) == 1)),
        ('h2wt', df['selected_multifilter'].to_numpy(dtype=np.int64) == 1),
    ]
    rows = []
    for name, mask in stages:
        passed = int(np.sum(mask))
        rows.append({'stage': name, 'passed': passed, 'retention_ratio': passed / max(total, 1)})
    return pd.DataFrame(rows)

def cluster_multifilter_candidates(df_src, config=None):
    cfg = SCREENING_DEFAULTS.copy()
    if config:
        cfg.update(config)

    df = df_src.copy()
    if 'selected_multifilter' not in df.columns:
        df = apply_multicriteria_filter(df, cfg)

    df['cluster_id'] = -1
    df['cluster_representative'] = 0
    selected = df[df['selected_multifilter'] == 1].copy()
    if selected.shape[0] == 0:
        return df, pd.DataFrame()

    selected = _copy_with_mol(selected)
    valid = selected['mol'].notna()
    if int(valid.sum()) == 0:
        return df, pd.DataFrame()
    selected = selected.loc[valid].copy()

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            int(cfg['fingerprint_radius']),
            nBits=int(cfg['fingerprint_bits'])
        )
        for mol in selected['mol'].tolist()
    ]
    n = len(fps)
    n_clusters = max(1, min(int(cfg['n_clusters']), n))
    if n == 1:
        labels = np.array([0], dtype=int)
    else:
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            dist[i, :] = 1.0 - np.asarray(sims, dtype=np.float64)
        try:
            model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        except TypeError:
            model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
        labels = model.fit_predict(dist)

    selected['cluster_id'] = labels
    rep_rows = []
    rep_mode = str(cfg.get('representative_mode', 'best')).lower()
    for cluster_id, grp in selected.groupby('cluster_id'):
        grp = grp.copy()
        if rep_mode == 'random':
            rep = grp.sample(n=1, random_state=cluster_id + 17)
        else:
            sort_cols = [c for c in ['fitness value', 'h2_wt_percent', 'delta_H_per_H2'] if c in grp.columns]
            if len(sort_cols) > 0:
                ascending = [False] * len(sort_cols)
                rep = grp.sort_values(by=sort_cols, ascending=ascending).head(1)
            else:
                rep = grp.head(1)
        rep = rep.copy()
        rep['cluster_representative'] = 1
        rep_rows.append(rep)

    reps = pd.concat(rep_rows, ignore_index=True) if len(rep_rows) > 0 else pd.DataFrame()
    selected_indexed = selected[['smi', 'cluster_id']].drop_duplicates(subset=['smi'], keep='first').copy()
    rep_smis = set(reps['smi'].tolist()) if reps.shape[0] > 0 and 'smi' in reps.columns else set()
    selected_indexed['cluster_representative'] = selected_indexed['smi'].apply(lambda smi: 1 if smi in rep_smis else 0)

    df = df.merge(selected_indexed, on='smi', how='left', suffixes=('', '_new'))
    if 'cluster_id_new' in df.columns:
        df['cluster_id'] = df['cluster_id_new'].fillna(df['cluster_id']).astype(int)
        df = df.drop(columns=['cluster_id_new'])
    if 'cluster_representative_new' in df.columns:
        df['cluster_representative'] = df['cluster_representative_new'].fillna(df['cluster_representative']).astype(int)
        df = df.drop(columns=['cluster_representative_new'])

    return df, reps

def _non_dominated_sort_indices(objective_matrix, vectorized_limit=7000, block_size=256):
    arr = np.asarray(objective_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    valid_idx = np.where(np.all(np.isfinite(arr), axis=1))[0]
    if valid_idx.size == 0:
        return []
    objs = arr[valid_idx]
    n = objs.shape[0]

    # A boolean dominance matrix is much faster than nested Python loops for
    # normal GA populations. Blocking limits temporary M x N x objective arrays.
    if n <= int(vectorized_limit):
        dominates = np.zeros((n, n), dtype=bool)
        for start in range(0, n, int(block_size)):
            end = min(n, start + int(block_size))
            block = objs[start:end, None, :]
            ge = block >= objs[None, :, :]
            gt = block > objs[None, :, :]
            dominates[start:end] = np.all(ge, axis=2) & np.any(gt, axis=2)

        domination_count = dominates.sum(axis=0, dtype=np.int32)
        current = np.where(domination_count == 0)[0]
        fronts = []
        while current.size > 0:
            fronts.append(valid_idx[current])
            decrement = dominates[current].sum(axis=0, dtype=np.int32)
            domination_count = domination_count - decrement
            domination_count[current] = -1
            current = np.where(domination_count == 0)[0]
        return fronts

    # Memory-safe fallback for unusually large archive-target batches.
    remaining = np.arange(n, dtype=int)
    fronts = []
    while remaining.size > 0:
        rel_front = _pareto_front_indices_nd(objs[remaining])
        if rel_front.size == 0:
            break
        front = remaining[rel_front]
        fronts.append(valid_idx[front])
        keep = np.ones(remaining.shape[0], dtype=bool)
        keep[rel_front] = False
        remaining = remaining[keep]
    return fronts

def _crowding_distance(objective_matrix):
    arr = np.asarray(objective_matrix, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)
    if n <= 2:
        return np.full(n, np.inf, dtype=np.float64)
    dist = np.zeros(n, dtype=np.float64)
    num_obj = arr.shape[1]
    for j in range(num_obj):
        vals = arr[:, j]
        order = np.argsort(vals)
        dist[order[0]] = np.inf
        dist[order[-1]] = np.inf
        span = vals[order[-1]] - vals[order[0]]
        if not np.isfinite(span) or span <= 1e-12:
            continue
        for i in range(1, n - 1):
            if np.isinf(dist[order[i]]):
                continue
            dist[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / span
    return dist

def _pareto_sort_columns(df_src):
    df = df_src.copy()
    if 'pareto_rank' not in df.columns:
        df['pareto_rank'] = np.inf
    if 'crowding_distance' not in df.columns:
        df['crowding_distance'] = -np.inf
    if 'selection_score' not in df.columns:
        df['selection_score'] = 0.0
    if 'fitness value' not in df.columns:
        df['fitness value'] = 0.0
    return df

def _sort_by_pareto_priority(df_src):
    df = _pareto_sort_columns(df_src)
    return df.sort_values(
        by=['pareto_rank', 'crowding_distance', 'selection_score', 'fitness value'],
        ascending=[True, False, False, False]
    )


def _pareto_clip01(values):
    arr = np.asarray(values, dtype=np.float64)
    return np.clip(arr, 0.0, 1.0)


def _pareto_normalized_scores(df, cfg):
    h2 = pd.to_numeric(df.get('h2_wt_percent', np.nan), errors='coerce').to_numpy(dtype=np.float64)
    dh = pd.to_numeric(df.get('delta_H_per_H2', np.nan), errors='coerce').to_numpy(dtype=np.float64)
    sa = pd.to_numeric(df.get('SA score', np.nan), errors='coerce').to_numpy(dtype=np.float64)

    h2_min = float(cfg.get('h2_objective_min', cfg.get('h2wt_min', 4.2)))
    h2_max = float(cfg.get('h2_objective_max', 7.0))
    h2_span = max(h2_max - h2_min, 1e-9)
    h2_score = _pareto_clip01((h2 - h2_min) / h2_span)

    dh_target = float(cfg.get('dh_target', DH_TARGET))
    dh_low = float(cfg.get('dh_window_low', DH_LOW))
    dh_high = float(cfg.get('dh_window_high', DH_HIGH))
    dh_score = dh_quality(dh, dh_low, dh_target, dh_high)

    sa_best = float(cfg.get('sa_objective_best', 1.0))
    sa_max = float(cfg.get('sa_objective_max', 4.0))
    sa_span = max(sa_max - sa_best, 1e-9)
    sa_score = np.where(np.isfinite(sa) & (sa <= 4.0), 1.0, np.nan)
    return h2_score, dh_score, sa_score


def _add_pareto_diversity_and_domain_scores(df, cfg):
    out = df.copy()
    out = _copy_with_mol(out)
    if 'scaffold_smi' not in out.columns:
        out['scaffold_smi'] = [_safe_murcko_scaffold_smiles(mol) for mol in out['mol'].tolist()]
    scaffold = out['scaffold_smi'].fillna('').astype(str)
    counts = scaffold.map(scaffold.value_counts()).replace(0, np.nan).astype(float)
    novelty = 1.0 / np.sqrt(counts)
    novelty[scaffold == ''] = np.nan
    out['scaffold_frequency'] = counts
    out['obj_scaffold_novelty'] = pd.to_numeric(novelty, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)

    domain_parts = []
    for col in ('bio_sim', 'target_template_score', 'platform_match_score'):
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
            domain_parts.append(_pareto_clip01(vals))
    if domain_parts:
        out['applicability_domain_score'] = np.max(np.vstack(domain_parts), axis=0)
    else:
        out['applicability_domain_score'] = np.nan
    out = out.drop(columns=['mol'], errors='ignore')
    return out


def _pareto_hypervolume_grid(objectives, bins=6):
    arr = np.asarray(objectives, dtype=np.float64)
    arr = arr[np.all(np.isfinite(arr), axis=1)]
    if arr.shape[0] == 0:
        return 0.0
    arr = np.clip(arr, 0.0, 1.0)
    bins = max(3, int(bins))
    coords = [(np.arange(bins, dtype=np.float64) + 0.5) / bins for _ in range(arr.shape[1])]
    mesh = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1).reshape(-1, arr.shape[1])
    dominated = np.any(np.all(arr[:, None, :] >= mesh[None, :, :], axis=2), axis=0)
    return float(np.mean(dominated))


def summarize_pareto_quality(df_src, config=None):
    cfg = SCREENING_DEFAULTS.copy()
    cfg.update(PARETO_DEFAULTS)
    if config:
        cfg.update(config)
    if df_src is None or df_src.shape[0] == 0:
        return pd.DataFrame(columns=['metric', 'value'])
    df = df_src.copy()
    objectives = [c for c in cfg.get('pareto_objectives', []) if c in df.columns]
    if not objectives:
        objectives = [c for c in ['obj_h2_score', 'obj_dh_score'] if c in df.columns]
    feasible = pd.to_numeric(df.get('pareto_feasible', 0), errors='coerce').fillna(0).astype(int) == 1
    front1 = pd.to_numeric(df.get('pareto_rank', np.inf), errors='coerce') == 1
    selected = pd.to_numeric(df.get('selected_pareto', 0), errors='coerce').fillna(0).astype(int) == 1
    obj = df.loc[feasible, objectives].to_numpy(dtype=np.float64) if objectives else np.empty((0, 0))
    front_obj = df.loc[front1, objectives].to_numpy(dtype=np.float64) if objectives else np.empty((0, 0))
    scaffold = df.get('scaffold_smi', pd.Series('', index=df.index)).fillna('').astype(str)
    rows = [
        {'metric': 'total_molecules', 'value': float(df.shape[0])},
        {'metric': 'pareto_feasible_molecules', 'value': float(feasible.sum())},
        {'metric': 'pareto_front1_molecules', 'value': float(front1.sum())},
        {'metric': 'pareto_selected_molecules', 'value': float(selected.sum())},
        {'metric': 'front1_scaffolds', 'value': float(scaffold[front1].replace('', np.nan).dropna().nunique())},
        {'metric': 'selected_scaffolds', 'value': float(scaffold[selected].replace('', np.nan).dropna().nunique())},
        {'metric': 'normalized_hypervolume_feasible', 'value': _pareto_hypervolume_grid(obj, cfg.get('pareto_quality_bins', 6))},
        {'metric': 'normalized_hypervolume_front1', 'value': _pareto_hypervolume_grid(front_obj, cfg.get('pareto_quality_bins', 6))},
    ]
    for col in objectives:
        vals = pd.to_numeric(df.loc[selected, col], errors='coerce')
        rows.append({'metric': f'selected_mean_{col}', 'value': float(vals.mean()) if vals.notna().any() else np.nan})
    return pd.DataFrame(rows)

def annotate_pareto_population(df_src, config=None):
    cfg = SCREENING_DEFAULTS.copy()
    cfg.update(PARETO_DEFAULTS)
    if config:
        cfg.update(config)

    # Reset to a clean RangeIndex because the GA loop concatenates frames and can
    # produce duplicate labels; label-based assignment with duplicate indices can
    # expand the target selection and break vector-length alignment.
    df = apply_multicriteria_filter(df_src, cfg).reset_index(drop=True)
    df = _copy_with_mol(df)

    num_rows = df.shape[0]
    df['obj_h2_capacity'] = pd.to_numeric(df.get('h2_wt_percent', np.nan), errors='coerce')
    dh = pd.to_numeric(df.get('delta_H_per_H2', np.nan), errors='coerce')
    sa = pd.to_numeric(df.get('SA score', np.nan), errors='coerce')

    h2_score, dh_score, sa_score = _pareto_normalized_scores(df, cfg)
    df['obj_h2_score'] = h2_score
    df['obj_dh_score'] = dh_score
    df['obj_sa_score'] = sa_score
    df['obj_dh_window'] = dh_score
    df['obj_sa'] = sa_score
    df['obj_bio'] = pd.to_numeric(df.get('bio_sim', 0.0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    df['obj_template'] = pd.to_numeric(df.get('target_template_score', 0.0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)

    diversity_domain = _add_pareto_diversity_and_domain_scores(df, cfg)
    for col in ['scaffold_smi', 'scaffold_frequency', 'obj_scaffold_novelty', 'applicability_domain_score']:
        if col in diversity_domain.columns:
            df[col] = diversity_domain[col].to_numpy()

    # Pareto sorting uses normalized design-satisfaction objectives so each
    # objective contributes on a comparable 0-1 scale.
    objective_cols = [c for c in cfg.get('pareto_objectives', []) if c in df.columns]
    if not objective_cols:
        objective_cols = ['obj_h2_score', 'obj_dh_score']
    obj_mat = df[objective_cols].to_numpy(dtype=np.float64)
    feasible_mask = (df['selected_hardfilter'].astype(int) == 1).to_numpy(dtype=bool)
    feasible_mask &= np.all(np.isfinite(obj_mat), axis=1)
    if not np.any(feasible_mask):
        feasible_mask = np.all(np.isfinite(obj_mat), axis=1)

    df['pareto_feasible'] = feasible_mask.astype(int)
    df['pareto_rank'] = np.inf
    df['is_pareto'] = 0
    df['crowding_distance'] = np.nan
    df['pareto_objective_sum'] = np.nan
    df['selection_score'] = 0.0
    df['selected_pareto'] = 0
    df['cluster_id'] = -1
    df['cluster_representative'] = 0

    feasible_idx = np.where(feasible_mask)[0]
    if feasible_idx.size > 0:
        feasible_objs = obj_mat[feasible_idx]
        fronts = _non_dominated_sort_indices(feasible_objs)
        crowding = np.full(feasible_idx.shape[0], np.nan, dtype=np.float64)
        rank_vals = np.full(feasible_idx.shape[0], np.inf, dtype=np.float64)
        obj_sum = np.nansum(feasible_objs, axis=1)
        for rank_no, front_rel in enumerate(fronts, start=1):
            rank_vals[front_rel] = float(rank_no)
            front_cd = _crowding_distance(feasible_objs[front_rel])
            crowding[front_rel] = front_cd
        df.iloc[feasible_idx, df.columns.get_loc('pareto_rank')] = rank_vals
        df.iloc[feasible_idx, df.columns.get_loc('crowding_distance')] = crowding
        df.iloc[feasible_idx, df.columns.get_loc('pareto_objective_sum')] = obj_sum
        rank1_idx = feasible_idx[rank_vals == 1]
        if rank1_idx.size > 0:
            df.iloc[rank1_idx, df.columns.get_loc('is_pareto')] = 1

        crowd_sort = crowding.copy()
        finite_cd = np.isfinite(crowd_sort)
        if np.any(finite_cd):
            max_cd = np.nanmax(crowd_sort[finite_cd])
            crowd_sort[~finite_cd] = max_cd + 1.0
        else:
            crowd_sort[:] = 1.0
        if np.nanmax(crowd_sort) > np.nanmin(crowd_sort):
            crowd_norm = (crowd_sort - np.nanmin(crowd_sort)) / (np.nanmax(crowd_sort) - np.nanmin(crowd_sort))
        else:
            crowd_norm = np.ones_like(crowd_sort)
        if np.nanmax(obj_sum) > np.nanmin(obj_sum):
            obj_norm = (obj_sum - np.nanmin(obj_sum)) / (np.nanmax(obj_sum) - np.nanmin(obj_sum))
        else:
            obj_norm = np.ones_like(obj_sum)
        max_rank = np.nanmax(rank_vals[np.isfinite(rank_vals)])
        rank_component = (max_rank - rank_vals + 1.0)
        sel_score = 1000.0 * rank_component + 10.0 * crowd_norm + obj_norm
        df.iloc[feasible_idx, df.columns.get_loc('selection_score')] = sel_score

        selected_mask = np.zeros(feasible_idx.shape[0], dtype=bool)
        if bool(cfg.get('selection_by_ratio', True)):
            ratio = float(cfg.get('selection_ratio', 0.10))
            ratio = min(max(ratio, 0.0), 1.0)
            k = int(np.ceil(ratio * max(num_rows, 1)))
            k = max(1, min(k, feasible_idx.shape[0]))
        else:
            k = feasible_idx.shape[0]
        order = np.lexsort((-obj_sum, -crowd_sort, rank_vals))
        selected_mask[order[:k]] = True
        selected_idx = feasible_idx[selected_mask]
        if selected_idx.size > 0:
            df.iloc[selected_idx, df.columns.get_loc('selected_pareto')] = 1

    return df

def cluster_pareto_candidates(df_src, config=None):
    cfg = SCREENING_DEFAULTS.copy()
    cfg.update(PARETO_DEFAULTS)
    if config:
        cfg.update(config)

    if 'selected_pareto' not in df_src.columns or 'pareto_rank' not in df_src.columns:
        df = annotate_pareto_population(df_src, cfg)
    else:
        df = df_src.copy()
    df = _copy_with_mol(df)
    df['cluster_id'] = -1
    df['cluster_representative'] = 0
    df['cluster_showcase_pass'] = 0
    df['cluster_representative_source'] = ''

    selected = df[df['selected_pareto'] == 1].copy()
    selected = selected[selected['mol'].notna()].copy()
    if selected.shape[0] == 0:
        return df, pd.DataFrame()

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            int(cfg.get('fingerprint_radius', 2)),
            nBits=int(cfg.get('fingerprint_bits', 1024))
        )
        for mol in selected['mol'].tolist()
    ]
    n = len(fps)
    n_clusters = max(1, min(int(cfg.get('n_clusters', 10)), n))
    if n == 1:
        labels = np.array([0], dtype=int)
    else:
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            dist[i, :] = 1.0 - np.asarray(sims, dtype=np.float64)
        try:
            model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        except TypeError:
            model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
        labels = model.fit_predict(dist)
    selected['cluster_id'] = labels

    rep_rows = []
    rep_mode = str(cfg.get('representative_mode', 'best')).lower()
    for cluster_id, grp in selected.groupby('cluster_id'):
        grp = grp.copy()
        if rep_mode == 'random':
            rep = grp.sample(n=1, random_state=int(cluster_id) + 31)
            rep['cluster_showcase_pass'] = [1 if _is_showcase_reasonable(m) else 0 for m in rep['mol'].tolist()]
            rep['cluster_representative_source'] = 'random'
        else:
            rep = _pick_cluster_representative(grp, config=cfg)
        if rep is None or rep.shape[0] == 0:
            continue
        rep = rep.copy()
        rep_rows.append(rep)

    reps = pd.concat(rep_rows, ignore_index=True) if len(rep_rows) > 0 else pd.DataFrame()
    selected['cluster_showcase_pass'] = [1 if _is_showcase_reasonable(m) else 0 for m in selected['mol'].tolist()]
    selected_indexed = selected[['smi', 'cluster_id', 'cluster_showcase_pass']].drop_duplicates(subset=['smi'], keep='first').copy()
    rep_smis = set(reps['smi'].tolist()) if reps.shape[0] > 0 and 'smi' in reps.columns else set()
    selected_indexed['cluster_representative'] = selected_indexed['smi'].apply(lambda smi: 1 if smi in rep_smis else 0)
    if reps.shape[0] > 0 and 'cluster_representative_source' in reps.columns:
        rep_source_map = dict(zip(reps['smi'].tolist(), reps['cluster_representative_source'].tolist()))
    else:
        rep_source_map = {}
    selected_indexed['cluster_representative_source'] = selected_indexed['smi'].apply(lambda smi: rep_source_map.get(smi, ''))

    df = df.merge(selected_indexed, on='smi', how='left', suffixes=('', '_new'))
    if 'cluster_id_new' in df.columns:
        df['cluster_id'] = df['cluster_id_new'].fillna(df['cluster_id']).astype(int)
        df = df.drop(columns=['cluster_id_new'])
    if 'cluster_showcase_pass_new' in df.columns:
        df['cluster_showcase_pass'] = df['cluster_showcase_pass_new'].fillna(df['cluster_showcase_pass']).astype(int)
        df = df.drop(columns=['cluster_showcase_pass_new'])
    if 'cluster_representative_new' in df.columns:
        df['cluster_representative'] = df['cluster_representative_new'].fillna(df['cluster_representative']).astype(int)
        df = df.drop(columns=['cluster_representative_new'])
    if 'cluster_representative_source_new' in df.columns:
        df['cluster_representative_source'] = df['cluster_representative_source_new'].fillna(df['cluster_representative_source']).astype(str)
        df = df.drop(columns=['cluster_representative_source_new'])

    return df, reps

def _project_fingerprint_space(df_src, config=None):
    cfg = PARETO_DEFAULTS.copy()
    if config:
        cfg.update(config)
    df = _copy_with_mol(df_src)
    df = df[df['mol'].notna()].copy()
    if df.shape[0] == 0:
        return df, np.zeros((0, 2), dtype=np.float64)
    bits = int(cfg.get('fingerprint_bits', 1024))
    fps = []
    for mol in df['mol'].tolist():
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, int(cfg.get('fingerprint_radius', 2)), nBits=bits)
        arr = np.zeros((bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    X = np.asarray(fps, dtype=np.float32)
    if X.shape[0] == 1:
        return df, np.zeros((1, 2), dtype=np.float64)
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(X)
    return df, xy

def dedupe_and_sort_population(df_src):
    if df_src is None or df_src.shape[0] == 0:
        return df_src.copy()
    df = df_src.drop_duplicates(subset=['smi'], keep='first').copy()
    return _sort_by_pareto_priority(df)


ARCHIVE_SELECTION_COLS = [
    'pass_sa', 'pass_pains', 'pass_hba', 'pass_hbd', 'pass_tpsa',
    'pass_enthalpy_window', 'pass_h2wt', 'pass_n_triple',
    'multicriteria_score', 'selected_hardfilter', 'selected_multifilter',
    'pass_reasonable', 'pareto_feasible', 'pareto_rank', 'is_pareto',
    'crowding_distance', 'pareto_objective_sum', 'selection_score',
    'selected_pareto', 'cluster_id', 'cluster_representative',
    'cluster_showcase_pass', 'cluster_representative_source',
    'structure_showcase_pass', 'structure_reasonable_pass',
    'archive_included_for_export', 'projection_available'
]


def _strip_archive_selection_columns(df_src):
    if df_src is None or df_src.shape[0] == 0:
        return df_src.copy() if df_src is not None else pd.DataFrame()
    return df_src.drop(columns=[c for c in ARCHIVE_SELECTION_COLS if c in df_src.columns], errors='ignore').copy()


def _prepare_archive_batch(df_candidates, df_ranked=None):
    if df_candidates is None or df_candidates.shape[0] == 0:
        return pd.DataFrame()
    archive_batch = _strip_archive_selection_columns(df_candidates).drop_duplicates(subset=['smi'], keep='first').reset_index(drop=True)
    archive_batch = _copy_with_mol(archive_batch)
    if df_ranked is None or df_ranked.shape[0] == 0:
        return archive_batch

    ranked_clean = _strip_archive_selection_columns(df_ranked).drop_duplicates(subset=['smi'], keep='first').reset_index(drop=True)
    ranked_clean = ranked_clean.drop(columns=['mol'], errors='ignore')
    merge_cols = ['smi'] + [c for c in ranked_clean.columns if c != 'smi']
    archive_batch = archive_batch.drop(columns=[c for c in ranked_clean.columns if c not in ['smi', 'mol'] and c in archive_batch.columns], errors='ignore')
    archive_batch = archive_batch.merge(ranked_clean[merge_cols], on='smi', how='left')
    archive_batch = _copy_with_mol(archive_batch)
    return archive_batch

def update_molecule_archive(df_archive, df_new):
    if df_new is None or df_new.shape[0] == 0:
        return df_archive.copy() if df_archive is not None else pd.DataFrame()
    if df_archive is None or df_archive.shape[0] == 0:
        return _strip_archive_selection_columns(df_new).drop_duplicates(subset=['smi'], keep='first').reset_index(drop=True)
    merged = pd.concat([_strip_archive_selection_columns(df_archive), _strip_archive_selection_columns(df_new)], ignore_index=True, sort=False)
    return merged.drop_duplicates(subset=['smi'], keep='first').reset_index(drop=True)


def _export_candidate_enrichment_tables(df_archive, pareto_candidates, out_dir, config=None):
    if df_archive is None or df_archive.shape[0] == 0:
        return pd.DataFrame()

    cfg = SCREENING_DEFAULTS.copy()
    cfg.update(PARETO_DEFAULTS)
    if config:
        cfg.update(config)

    df = _strip_archive_selection_columns(df_archive).copy()
    if 'smi' not in df.columns:
        return pd.DataFrame()

    gen_col = None
    for candidate in ('discovery_generation', 'generation'):
        if candidate in df.columns:
            gen_col = candidate
            break
    if gen_col is None:
        df['discovery_generation'] = 1
        gen_col = 'discovery_generation'

    df[gen_col] = pd.to_numeric(df[gen_col], errors='coerce')
    max_gen = df[gen_col].dropna().max()
    if not np.isfinite(max_gen):
        max_gen = 1
    df[gen_col] = df[gen_col].fillna(max_gen).astype(int)
    df = df.sort_values([gen_col]).drop_duplicates(subset=['smi'], keep='first').copy()

    sa = pd.to_numeric(df.get('SA score', np.nan), errors='coerce')
    h2 = pd.to_numeric(df.get('h2_wt_percent', np.nan), errors='coerce')
    dh = pd.to_numeric(df.get('delta_H_per_H2', np.nan), errors='coerce')
    fast = pd.to_numeric(df.get('pass_fast_prefilter', 1), errors='coerce').fillna(1)
    sa_max = float(cfg.get('sa_max', cfg.get('sa_max_generation', 6.0)))
    h2_min = float(cfg.get('h2wt_min', 4.2))
    dh_low = float(cfg.get('dh_window_low', DH_LOW))
    dh_high = float(cfg.get('dh_window_high', DH_HIGH))
    df['_property_feasible'] = (
        (fast > 0)
        & np.isfinite(sa) & (sa <= sa_max)
        & np.isfinite(h2) & (h2 >= h2_min)
        & np.isfinite(dh) & (dh >= dh_low) & (dh <= dh_high)
    )

    pareto_smis = set()
    if pareto_candidates is not None and pareto_candidates.shape[0] > 0 and 'smi' in pareto_candidates.columns:
        pareto_smis = set(pareto_candidates['smi'].astype(str).tolist())
    elif 'pareto_rank' in df.columns:
        ranks = pd.to_numeric(df['pareto_rank'], errors='coerce')
        pareto_smis = set(df.loc[np.isfinite(ranks) & (ranks <= 1), 'smi'].astype(str).tolist())
    df['_pareto_selected'] = df['smi'].astype(str).isin(pareto_smis)

    rows = []
    for generation in sorted(df[gen_col].dropna().astype(int).unique().tolist()):
        cur = df[df[gen_col] <= generation]
        rows.append({
            'generation': int(generation),
            'cumulative_unique': int(cur['smi'].nunique()),
            'property_feasible': int(cur['_property_feasible'].sum()),
            'pareto_selected': int(cur['_pareto_selected'].sum()),
        })
    df_summary = pd.DataFrame(rows)
    if df_summary.shape[0] == 0:
        return df_summary

    _safe_to_csv(df_summary, os.path.join(out_dir, 'candidate_enrichment_origin.csv'), index=False)
    df_long = df_summary.melt(
        id_vars=['generation'],
        value_vars=['cumulative_unique', 'property_feasible', 'pareto_selected'],
        var_name='series',
        value_name='count',
    )
    _safe_to_csv(df_long, os.path.join(out_dir, 'candidate_enrichment_long.csv'), index=False)
    return df_summary

def _cluster_molecule_subset(df_src, config=None):
    cfg = SCREENING_DEFAULTS.copy()
    cfg.update(PARETO_DEFAULTS)
    if config:
        cfg.update(config)

    df = _copy_with_mol(df_src)
    df = df[df['mol'].notna()].copy()
    if df.shape[0] == 0:
        if 'cluster_id' not in df.columns:
            df['cluster_id'] = pd.Series(dtype=int)
        if 'cluster_representative' not in df.columns:
            df['cluster_representative'] = pd.Series(dtype=int)
        return df, pd.DataFrame()

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            int(cfg.get('fingerprint_radius', 2)),
            nBits=int(cfg.get('fingerprint_bits', 1024))
        )
        for mol in df['mol'].tolist()
    ]
    n = len(fps)
    n_clusters = max(1, min(int(cfg.get('n_clusters', 10)), n))
    if n == 1:
        labels = np.array([0], dtype=int)
    else:
        dist = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            dist[i, :] = 1.0 - np.asarray(sims, dtype=np.float64)
        try:
            model = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
        except TypeError:
            model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
        labels = model.fit_predict(dist)
    df['cluster_id'] = labels

    rep_rows = []
    for _, grp in df.groupby('cluster_id'):
        rep = _sort_by_pareto_priority(grp).head(1).copy()
        rep['cluster_representative'] = 1
        rep_rows.append(rep)
    reps = pd.concat(rep_rows, ignore_index=True) if len(rep_rows) > 0 else pd.DataFrame(columns=df.columns)
    rep_smis = set(reps['smi'].tolist()) if reps.shape[0] > 0 else set()
    df['cluster_representative'] = df['smi'].isin(rep_smis).astype(int)
    return df, reps

def _export_cluster_subset(df_src, out_dir, prefix):
    export_cols = [
        'smi', 'cluster_id', 'cluster_representative',
        'fitness value', 'selection_score', 'pareto_rank', 'crowding_distance',
        'selected_pareto', 'structure_reasonable_pass',
        'h2_wt_percent', 'delta_H_per_H2', 'SA score',
        'platform_id', 'platform_name', 'x', 'y'
    ]
    export_cols = [c for c in export_cols if c in df_src.columns]
    if df_src is None or df_src.shape[0] == 0:
        _safe_to_csv(pd.DataFrame(columns=export_cols), os.path.join(out_dir, f'{prefix}_scatter.csv'), index=False)
        _safe_to_csv(pd.DataFrame(columns=['cluster_id', 'count', 'center_x', 'center_y', 'max_fitness']), os.path.join(out_dir, f'{prefix}_summary.csv'), index=False)
        _safe_to_csv(pd.DataFrame(columns=export_cols), os.path.join(out_dir, f'{prefix}_representatives.csv'), index=False)
        return pd.DataFrame(columns=export_cols)
    _safe_to_csv(df_src[export_cols].copy(), os.path.join(out_dir, f'{prefix}_scatter.csv'), index=False)

    summary = df_src.groupby('cluster_id', dropna=False).agg(
        count=('smi', 'count'),
        center_x=('x', 'mean'),
        center_y=('y', 'mean'),
        max_fitness=('fitness value', 'max')
    ).reset_index()
    _safe_to_csv(summary, os.path.join(out_dir, f'{prefix}_summary.csv'), index=False)

    reps = df_src[df_src['cluster_representative'] == 1].copy() if 'cluster_representative' in df_src.columns else pd.DataFrame()
    if reps.shape[0] == 0:
        reps = pd.DataFrame(columns=export_cols)
    else:
        reps = reps[export_cols].copy()
    _safe_to_csv(reps, os.path.join(out_dir, f'{prefix}_representatives.csv'), index=False)
    return reps

def _plot_cluster_subset(df_src, reps, fig_path, title):
    if df_src is None or df_src.shape[0] == 0:
        return
    fig = plt.figure(figsize=(9.6, 7.8))
    ax = fig.add_subplot(1, 1, 1)
    labels = pd.to_numeric(df_src['cluster_id'], errors='coerce').fillna(-1).to_numpy(dtype=np.int64)
    ax.scatter(df_src['x'], df_src['y'], c=labels, cmap='tab20', s=20, alpha=0.82, edgecolors='none')

    if reps is not None and reps.shape[0] > 0:
        ax.scatter(
            reps['x'], reps['y'],
            marker='*', s=170, c='black', edgecolors='white', linewidths=0.8,
            label='Representative'
        )
        for _, row in reps.iterrows():
            ax.text(row['x'], row['y'], f"C{int(row['cluster_id'])}", fontsize=8)
        ax.legend(loc='best', fontsize=8)

    ax.set_xlabel('Fingerprint PCA-1')
    ax.set_ylabel('Fingerprint PCA-2')
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

def _export_cluster_subset_with_projection(df_src, out_dir, prefix, config=None, fig_dir=None, title=None):
    cfg = PARETO_DEFAULTS.copy()
    if config:
        cfg.update(config)
    if df_src is None or df_src.shape[0] == 0:
        empty = pd.DataFrame()
        _export_cluster_subset(empty, out_dir, prefix)
        return empty, pd.DataFrame()

    projected, xy = _project_fingerprint_space(df_src, config=cfg)
    if projected.shape[0] == 0:
        empty = pd.DataFrame(columns=df_src.columns)
        _export_cluster_subset(empty, out_dir, prefix)
        return empty, pd.DataFrame()
    projected = projected.copy()
    projected['x'] = xy[:, 0]
    projected['y'] = xy[:, 1]
    clustered, reps = _cluster_molecule_subset(projected, config=cfg)
    reps = _export_cluster_subset(clustered, out_dir, prefix)
    if fig_dir is not None and title is not None:
        _plot_cluster_subset(
            clustered,
            reps,
            os.path.join(fig_dir, f'{prefix}_map.png'),
            title
        )
    return clustered, reps

def export_selection_workflow(df_archive, df_selected, out_dir, fig_dir=None, config=None, max_points=None):
    cfg = SCREENING_DEFAULTS.copy()
    cfg.update(PARETO_DEFAULTS)
    if config:
        cfg.update(config)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    base_dir = os.path.dirname(os.path.normpath(out_dir))
    pareto_dir = os.path.join(base_dir, 'pareto') if base_dir else os.path.join(out_dir, 'pareto')
    if not os.path.exists(pareto_dir):
        os.makedirs(pareto_dir)
    if fig_dir is not None and not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    if df_archive is None or df_archive.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()

    archived_all = dedupe_and_sort_population(_strip_archive_selection_columns(df_archive))
    if max_points is not None and archived_all.shape[0] > int(max_points):
        archived_all = archived_all.head(int(max_points)).copy()
    archived_all = _copy_with_mol(archived_all)
    archived_all['archive_included_for_export'] = 1
    archived_all['projection_available'] = archived_all['mol'].notna().astype(int)
    _safe_to_csv(
        archived_all.drop(columns=['mol'], errors='ignore'),
        os.path.join(out_dir, 'all_generated_molecules.csv'),
        index=False
    )

    all_clustered, all_reps = _export_cluster_subset_with_projection(
        df_src=archived_all,
        out_dir=out_dir,
        prefix='all_generated_cluster',
        config=cfg,
        fig_dir=fig_dir,
        title='All Generated Molecule Cluster Map'
    )

    archived_ranked = annotate_pareto_population(_strip_archive_selection_columns(archived_all), cfg)
    if bool(cfg.get('chemical_qd_mode', False)):
        archived_ranked = annotate_chemical_qd(archived_ranked, cfg)
    archived_ranked = dedupe_and_sort_population(archived_ranked)
    pareto_candidates = archived_ranked[
        pd.to_numeric(archived_ranked['selected_pareto'], errors='coerce').fillna(0).astype(int) == 1
    ].copy()
    _export_candidate_enrichment_tables(
        df_archive=archived_all,
        pareto_candidates=pareto_candidates,
        out_dir=out_dir,
        config=cfg,
    )
    _export_candidate_enrichment_tables(
        df_archive=archived_all,
        pareto_candidates=pareto_candidates,
        out_dir=pareto_dir,
        config=cfg,
    )

    pareto_export_cols = [
        'smi', 'fitness value', 'selection_score', 'pareto_rank', 'crowding_distance',
        'pareto_feasible', 'is_pareto', 'selected_pareto', 'selected_hardfilter',
        'selected_multifilter', 'structure_reasonable_pass', 'structure_showcase_pass',
        'has_small_ring', 'min_ring_size', 'num_rings', 'num_aromatic_rings',
        'ring_penalty', 'formal_charge_abs_sum', 'has_radical',
        'platform_id', 'platform_name', 'platform_anchor_smi', 'platform_match_score',
        'platform_core_retained', 'h2_wt_percent', 'delta_H_per_H2', 'delta_H_diff',
        'SA score', 'bio_sim', 'target_template_hit', 'target_template_score',
        'obj_h2_capacity', 'obj_h2_score', 'obj_dh_score', 'obj_sa_score',
        'obj_dh_window', 'obj_sa', 'obj_bio', 'obj_template',
        'obj_scaffold_novelty', 'applicability_domain_score', 'scaffold_smi',
        'scaffold_frequency', 'pareto_objective_sum'
    ]
    pareto_export_cols = [c for c in pareto_export_cols if c in pareto_candidates.columns]
    _safe_to_csv(
        archived_ranked.drop(columns=['mol'], errors='ignore'),
        os.path.join(pareto_dir, 'pareto_all_ranked.csv'),
        index=False
    )
    _safe_to_csv(
        pareto_candidates[pareto_export_cols].drop(columns=['mol'], errors='ignore'),
        os.path.join(out_dir, 'pareto_candidates.csv'),
        index=False
    )
    _safe_to_csv(
        pareto_candidates[pareto_export_cols].drop(columns=['mol'], errors='ignore'),
        os.path.join(pareto_dir, 'pareto_candidates.csv'),
        index=False
    )
    _safe_to_csv(
        summarize_pareto_quality(archived_ranked, config=cfg),
        os.path.join(pareto_dir, 'pareto_quality_metrics.csv'),
        index=False
    )
    if 'chemical_behavior_cell' in archived_ranked.columns:
        chem_summary = archived_ranked.groupby('chemical_behavior_cell', dropna=False).agg(
            molecules=('smi', 'count'),
            unique_scaffolds=('scaffold_smi', pd.Series.nunique),
            mean_quality=('chemical_quality_score', 'mean'),
            mean_novelty=('chemical_novelty_score', 'mean'),
            max_qd=('chemical_qd_score', 'max'),
        ).reset_index()
        _safe_to_csv(chem_summary, os.path.join(pareto_dir, 'chemical_qd_cells.csv'), index=False)
        _safe_to_csv(
            local_pareto_archive(archived_ranked, cfg=cfg, top_k=int(cfg.get('chemical_local_archive_k', 3))).drop(columns=['mol'], errors='ignore'),
            os.path.join(pareto_dir, 'chemical_local_pareto_archive.csv'),
            index=False
        )

    pareto_clustered, pareto_reps = _export_cluster_subset_with_projection(
        df_src=pareto_candidates,
        out_dir=out_dir,
        prefix='pareto_candidates_cluster',
        config=cfg,
        fig_dir=fig_dir,
        title='Pareto Candidate Cluster Map'
    )
    if pareto_clustered is not None and pareto_clustered.shape[0] > 0:
        _export_cluster_subset(pareto_clustered, pareto_dir, 'pareto_candidates_cluster')

    if pareto_reps is None or pareto_reps.shape[0] == 0:
        empty_reps = pd.DataFrame(columns=pareto_export_cols)
        _safe_to_csv(empty_reps, os.path.join(out_dir, 'pareto_cluster_representatives.csv'), index=False)
        _safe_to_csv(empty_reps, os.path.join(pareto_dir, 'pareto_cluster_representatives.csv'), index=False)
    else:
        rep_cols = [c for c in pareto_export_cols if c in pareto_reps.columns]
        _safe_to_csv(pareto_reps[rep_cols], os.path.join(out_dir, 'pareto_cluster_representatives.csv'), index=False)
        _safe_to_csv(pareto_reps[rep_cols], os.path.join(pareto_dir, 'pareto_cluster_representatives.csv'), index=False)

    return all_clustered, pareto_clustered

def sample_seed_immigrants(seed_pool, immigrant_count, platform_balanced=False, random_state=None, blocked_smis=None):
    if seed_pool is None or seed_pool.shape[0] == 0 or immigrant_count <= 0:
        return pd.DataFrame(columns=['smi', 'mol'])
    blocked = set(str(s) for s in blocked_smis) if blocked_smis is not None else set()
    novel_pool = seed_pool[~seed_pool['smi'].astype(str).isin(blocked)].copy() if len(blocked) > 0 else seed_pool.copy()
    seen_pool = seed_pool[seed_pool['smi'].astype(str).isin(blocked)].copy() if len(blocked) > 0 else seed_pool.head(0).copy()

    def _sample_from_pool(pool, count, seed_offset):
        if pool is None or pool.shape[0] == 0 or count <= 0:
            return pool.head(0).copy() if pool is not None else pd.DataFrame(columns=['smi', 'mol'])
        if platform_balanced:
            return sample_platform_balanced(
                pool,
                sample_size=int(count),
                replace=(pool.shape[0] < int(count)),
                random_state=None if random_state is None else int(random_state) + seed_offset
            )
        return pool.sample(
            n=min(int(count), pool.shape[0]),
            replace=(pool.shape[0] < int(count)),
            random_state=None if random_state is None else int(random_state) + seed_offset
        )

    take_novel = min(int(immigrant_count), novel_pool.shape[0])
    parts = [_sample_from_pool(novel_pool, take_novel, 0)]
    remaining = int(immigrant_count) - sum(p.shape[0] for p in parts)
    if remaining > 0:
        parts.append(_sample_from_pool(seen_pool, remaining, 1000))
    sampled = pd.concat(parts, ignore_index=True) if len(parts) > 0 else seed_pool.head(0).copy()
    df_imm = sampled.copy()
    if 'smi' not in df_imm.columns:
        df_imm = pd.DataFrame({'smi': sampled['smi'].tolist()})
    df_imm['mol'] = df_imm['smi'].apply(Chem.MolFromSmiles)
    df_imm = df_imm.dropna(subset=['mol'])
    return df_imm

def generate_candidate_pool(
    df_pop,
    seed_pool,
    cfg,
    generation,
    stagnation_count,
    archive_smis=None,
    target_candidate_pool_size=None,
    operator_policy=None,
):
    pop_size = int(cfg['pop_size'])
    base_num_children = int(cfg['num_parents']) * 2

    elite_size = max(1, int(pop_size * cfg['elite_fraction']))
    df_elite = dedupe_and_sort_population(df_pop).head(elite_size).copy()
    df_parents = df_pop.copy()
    parent_smis = set(df_parents['smi'].astype(str).tolist())

    target_mode = target_candidate_pool_size is not None and int(target_candidate_pool_size) > 0
    if target_mode:
        target_pool = max(pop_size, int(target_candidate_pool_size))
        immigrant_count = max(1, int(target_pool * float(cfg.get('target_immigrant_fraction', 0.15))))
        child_target = max(base_num_children, target_pool - df_parents.shape[0] - immigrant_count)
        child_oversample = float(cfg.get('target_child_oversample_factor', cfg.get('child_oversample_factor', 3.0)))
        immigrant_oversample = float(cfg.get('target_immigrant_oversample_factor', cfg.get('immigrant_oversample_factor', 4.0)))
    else:
        child_target = base_num_children
        curr_immigrant_fraction = float(cfg['immigrant_fraction'])
        if stagnation_count >= int(cfg['stagnation_patience']):
            curr_immigrant_fraction = min(0.70, curr_immigrant_fraction + 0.15)
        immigrant_count = max(1, int(pop_size * curr_immigrant_fraction))
        child_oversample = float(cfg.get('child_oversample_factor', 3.0))
        immigrant_oversample = float(cfg.get('immigrant_oversample_factor', 4.0))

    curr_mutation_rate = max(
        float(cfg['min_mutation_rate']),
        float(cfg['mutation_rate']) * (1.0 - generation / max(1, int(cfg['generations']) - 1))
    )
    if stagnation_count >= int(cfg['stagnation_patience']):
        curr_mutation_rate = min(
            float(cfg['max_mutation_rate']),
            curr_mutation_rate + float(cfg['stagnation_mutation_boost'])
        )

    raw_child_target = max(child_target, int(np.ceil(child_target * child_oversample)))
    base_rate = max(float(cfg.get('mutation_rate', 0.22)), 1e-6)
    guided_fraction = float(cfg.get('guided_mutant_fraction', 0.25))
    guided_fraction *= curr_mutation_rate / base_rate
    guided_fraction = float(np.clip(guided_fraction, 0.10, 0.50))
    guided_target = max(1, int(np.ceil(raw_child_target * guided_fraction)))
    crossover_target = max(1, raw_child_target - guided_target)

    df_crossed = cross_over(
        df_parents.copy(),
        int(cfg['num_parents']),
        crossover_target,
        float(cfg['max_mol_weight']),
        None,
        float(cfg['sa_max_generation']),
        same_platform_only=bool(cfg['same_platform_crossover']),
        retain_parent_scaffold=bool(cfg['retain_parent_scaffold_in_crossover']),
        max_attempts=min(
            int(cfg.get('max_crossover_attempts', 200000)),
            max(100, int(np.ceil(crossover_target * float(cfg.get('crossover_attempt_factor', 20)))))
        ),
    )
    df_cross_children = df_crossed.drop_duplicates(subset=['smi'], keep='first').copy()
    df_cross_children = df_cross_children[
        ~df_cross_children['smi'].astype(str).isin(parent_smis)
    ].copy()
    df_cross_children['origin_type'] = 'crossover'
    df_cross_children['mutation_operator'] = 'crossover'
    df_cross_children['operation_generation'] = int(generation) + 1

    mutant_attempts = max(
        guided_target,
        int(np.ceil(guided_target * float(cfg.get('guided_mutant_oversample_factor', 2.0))))
    )
    df_mutants = generate_guided_mutation_offspring(
        df_pop=df_parents,
        num_offspring=mutant_attempts,
        max_mol_weight=float(cfg['max_mol_weight']),
        SA_max=float(cfg['sa_max_generation']),
        preserve_platform=bool(cfg['preserve_platform_in_mutation']),
        operator_policy=operator_policy,
        generation=int(generation),
        screening_config=cfg.get('survivor_screening', {}),
    )
    if df_mutants.shape[0] > guided_target:
        df_mutants = df_mutants.head(guided_target).copy()
    if df_mutants.shape[0] > 0:
        df_mutants['operation_generation'] = int(generation) + 1

    df_children = pd.concat([df_cross_children, df_mutants], ignore_index=True, sort=False)
    df_children = df_children.drop_duplicates(subset=['smi'], keep='first').copy()
    df_children = df_children[~df_children['smi'].astype(str).isin(parent_smis)].copy()

    raw_immigrant_count = max(immigrant_count, int(np.ceil(immigrant_count * immigrant_oversample)))
    blocked_for_immigrants = set(parent_smis).union(df_children['smi'].astype(str).tolist())
    if archive_smis is not None:
        blocked_for_immigrants.update(str(s) for s in archive_smis)
    df_imm = sample_seed_immigrants(
        seed_pool,
        raw_immigrant_count,
        platform_balanced=bool(cfg['platform_balanced_immigrants']),
        random_state=int(generation) + 101,
        blocked_smis=blocked_for_immigrants,
    )
    blocked_smis = parent_smis.union(df_children['smi'].astype(str).tolist())
    df_imm = df_imm.drop_duplicates(subset=['smi'], keep='first').copy()
    df_imm = df_imm[~df_imm['smi'].astype(str).isin(blocked_smis)].copy()
    if df_imm.shape[0] > immigrant_count:
        df_imm = df_imm.head(immigrant_count).copy()
    if df_imm.shape[0] > 0:
        df_imm['origin_type'] = 'seed_immigrant'
        df_imm['operation_generation'] = int(generation) + 1

    df_candidates = pd.concat(
        [df_parents, df_children, df_imm],
        ignore_index=True,
        sort=False,
    )
    df_candidates = df_candidates.drop_duplicates(subset=['smi'], keep='first').reset_index(drop=True)
    if 'discovery_generation' not in df_candidates.columns:
        df_candidates['discovery_generation'] = np.nan
    df_candidates['discovery_generation'] = pd.to_numeric(
        df_candidates['discovery_generation'],
        errors='coerce'
    ).fillna(int(generation) + 1).astype(int)
    return df_candidates, df_elite, curr_mutation_rate

def constrain_and_rank_candidates(
    df_candidates,
    df_elite,
    screening_config,
    compute_full_hydrogenation=False,
    cfg=None,
):
    candidates = df_candidates.copy().reset_index(drop=True)
    if 'fitness_evaluated' in candidates.columns:
        evaluated = (
            pd.to_numeric(candidates['fitness_evaluated'], errors='coerce')
            .fillna(0)
            .astype(int)
            .eq(1)
        )
    else:
        evaluated = pd.Series(False, index=candidates.index)

    cached = candidates[evaluated].copy()
    pending = candidates[~evaluated].copy()
    cfg = cfg or {}
    pending_before_prefilter = int(pending.shape[0])
    if bool(cfg.get('full_model_mode', False)):
        pending, prefilter_dropped = apply_gnn_budget_prefilter(cached, pending, cfg)
    else:
        prefilter_dropped = 0
    if pending.shape[0] > 0:
        pending = get_fitness_vals(
            pending,
            compute_full_hydrogenation=compute_full_hydrogenation,
            scoring_config=cfg,
        )
    df_ranked = pd.concat([cached, pending], ignore_index=True, sort=False)
    df_ranked = df_ranked.drop_duplicates(subset=['smi'], keep='first')
    if 'raw_fitness' in df_ranked.columns:
        raw = pd.to_numeric(df_ranked['raw_fitness'], errors='coerce')
        valid_raw = raw[np.isfinite(raw) & (raw > -1e5)]
        shift = 1.0
        if valid_raw.shape[0] > 0 and float(valid_raw.min()) <= 0.0:
            shift = 1.0 - float(valid_raw.min())
        df_ranked['fitness value'] = np.where(
            np.isfinite(raw) & (raw > -1e5),
            raw + shift,
            0.0,
        )
    df_ranked = annotate_pareto_population(df_ranked, screening_config)
    df_ranked = annotate_qd_cells(df_ranked)
    df_ranked.attrs['pending_before_prefilter'] = pending_before_prefilter
    df_ranked.attrs['pending_after_prefilter'] = int(pending.shape[0])
    df_ranked.attrs['prefilter_dropped'] = int(prefilter_dropped)
    return df_ranked

def refresh_population_if_needed(df_pop, seed_pool, cfg, screening_config, stagnation_count, generation):
    if stagnation_count < 3 * int(cfg['stagnation_patience']):
        return df_pop, stagnation_count, False

    pop_size = int(cfg['pop_size'])
    refresh_count = max(1, int(pop_size * float(cfg['refresh_fraction'])))
    keep_count = max(1, pop_size - refresh_count)
    df_keep = df_pop.head(keep_count).copy()
    df_ref = sample_seed_immigrants(
        seed_pool,
        refresh_count,
        platform_balanced=bool(cfg['platform_balanced_immigrants']),
        random_state=int(generation) + 501,
    )
    df_pop = pd.concat([df_keep, df_ref], ignore_index=True)
    df_pop = get_fitness_vals(df_pop)
    df_pop = annotate_pareto_population(df_pop, screening_config)
    df_pop = dedupe_and_sort_population(df_pop).head(pop_size)
    return df_pop, 0, True

def select_next_population(
    df_ranked,
    pop_size,
    screening_config,
    keep_core_ratio,
    keep_cluster_ratio,
    keep_platform_ratio,
    keep_target_ratio,
    keep_bio_ratio,
    random_state=None,
    cfg=None
):
    sa = pd.to_numeric(df_ranked['SA score'], errors='coerce')
    df_ranked = df_ranked.loc[np.isfinite(sa) & (sa <= 4.0)].copy()
    if {'pareto_rank', 'selection_score', 'selected_hardfilter'}.issubset(set(df_ranked.columns)):
        df_screen = df_ranked.copy()
    else:
        df_screen = annotate_pareto_population(df_ranked, config=screening_config)
    df_screen, df_reps = cluster_pareto_candidates(df_screen, config=screening_config)
    df_selected = df_screen[df_screen['selected_pareto'] == 1].copy()
    if df_selected.shape[0] == 0:
        df_selected = df_screen.copy()

    keep_core = int(pop_size * keep_core_ratio)
    keep_cluster = int(pop_size * keep_cluster_ratio)
    keep_platform = int(pop_size * keep_platform_ratio)
    keep_target = int(pop_size * keep_target_ratio)
    cfg = cfg or {}
    keep_chemical = int(pop_size * float(cfg.get('keep_chemical_qd_ratio', 0.0))) if bool(cfg.get('chemical_qd_mode', False)) else 0
    keep_random = pop_size - keep_core - keep_cluster - keep_platform - keep_target - keep_chemical
    keep_random = max(0, keep_random)

    df_core = _sort_by_pareto_priority(df_selected).head(keep_core).copy()
    chosen = set(df_core['smi'].tolist())

    if df_reps is not None and df_reps.shape[0] > 0:
        df_cluster_pool = _sort_by_pareto_priority(df_reps)
    else:
        df_cluster_pool = _sort_by_pareto_priority(df_selected)
    df_cluster = df_cluster_pool[~df_cluster_pool['smi'].isin(chosen)].head(max(0, keep_cluster)).copy()
    chosen.update(df_cluster['smi'].tolist())

    if 'platform_id' in df_selected.columns and keep_platform > 0:
        df_platform_pool = df_selected[~df_selected['smi'].isin(chosen)].copy()
        df_platform_pool = df_platform_pool[df_platform_pool['platform_id'].astype(str) != 'unassigned'].copy()
        platform_rows = []
        for _, grp in df_platform_pool.groupby('platform_id'):
            platform_rows.append(_sort_by_pareto_priority(grp).head(1))
        if len(platform_rows) > 0:
            df_platform = pd.concat(platform_rows, ignore_index=True)
            df_platform = _sort_by_pareto_priority(df_platform).head(max(0, keep_platform)).copy()
        else:
            df_platform = pd.DataFrame(columns=df_selected.columns)
    else:
        df_platform = pd.DataFrame(columns=df_selected.columns)
    chosen.update(df_platform['smi'].tolist())

    if keep_target > 0:
        df_qd_pool = annotate_qd_cells(df_ranked[~df_ranked['smi'].isin(chosen)].copy())
        df_qd_pool = df_qd_pool[df_qd_pool['qd_cell'].astype(str).ne('')].copy()
        qd_rows = []
        if df_qd_pool.shape[0] > 0:
            for _, grp in df_qd_pool.groupby('qd_cell'):
                qd_rows.append(_sort_by_pareto_priority(grp).head(1))
        if len(qd_rows) > 0:
            df_target = pd.concat(qd_rows, ignore_index=True, sort=False)
            df_target = _sort_by_pareto_priority(df_target).head(max(0, keep_target)).copy()
        else:
            df_target = pd.DataFrame(columns=df_ranked.columns)
    else:
        df_target = pd.DataFrame(columns=df_ranked.columns)
    chosen.update(df_target['smi'].tolist())

    if keep_chemical > 0:
        df_chem_pool = df_ranked[~df_ranked['smi'].isin(chosen)].copy()
        df_chem_archive = local_pareto_archive(df_chem_pool, cfg=cfg, top_k=int(cfg.get('chemical_local_archive_k', 3)))
        if df_chem_archive.shape[0] > 0:
            df_chemical = df_chem_archive.sort_values(
                by=['chemical_qd_score', 'selection_score'], ascending=[False, False]
            ).head(max(0, keep_chemical)).copy()
        else:
            df_chemical = pd.DataFrame(columns=df_ranked.columns)
    else:
        df_chemical = pd.DataFrame(columns=df_ranked.columns)
    chosen.update(df_chemical['smi'].tolist())

    df_rand_pool = df_ranked[~df_ranked['smi'].isin(chosen)].copy()
    if df_rand_pool.shape[0] > 0:
        df_rand = df_rand_pool.sample(
            n=min(max(0, keep_random), df_rand_pool.shape[0]),
            replace=False,
            random_state=random_state
        )
    else:
        df_rand = df_rand_pool
    chosen.update(df_rand['smi'].tolist())

    df_next = pd.concat([df_core, df_cluster, df_platform, df_target, df_chemical, df_rand], ignore_index=True, sort=False)
    if df_next.shape[0] < pop_size:
        need = pop_size - df_next.shape[0]
        remainder = df_ranked[~df_ranked['smi'].isin(chosen)].head(need).copy()
        df_next = pd.concat([df_next, remainder], ignore_index=True)
    df_next = dedupe_and_sort_population(df_next).head(pop_size)
    return df_next, df_screen, df_reps

def _load_biobased_dataframe(path='bio-based_start_pop.csv'):
    if not os.path.exists(path):
        return pd.DataFrame(columns=['smi'])
    try:
        if zipfile.is_zipfile(path):
            try:
                df = pd.read_excel(path)
            except Exception:
                smiles = _read_xlsx_smiles_without_openpyxl(path)
                if len(smiles) == 0:
                    return pd.DataFrame(columns=['smi'])
                return pd.DataFrame({'smi': smiles}).drop_duplicates(subset=['smi'])
        else:
            df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=['smi'])
    smi_col = None
    for cand in ['smi', 'SMILES', 'smiles']:
        if cand in df.columns:
            smi_col = cand
            break
    if smi_col is None:
        return pd.DataFrame(columns=['smi'])
    out = pd.DataFrame({'smi': df[smi_col].astype(str)})
    out = out.dropna(subset=['smi']).drop_duplicates(subset=['smi'])
    return out

def _read_xlsx_smiles_without_openpyxl(path):
    ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            shared = []
            if 'xl/sharedStrings.xml' in zf.namelist():
                sroot = ET.fromstring(zf.read('xl/sharedStrings.xml'))
                for si in sroot.findall('.//x:si', ns):
                    texts = [t.text or '' for t in si.findall('.//x:t', ns)]
                    shared.append(''.join(texts))
            sheet_name = 'xl/worksheets/sheet1.xml'
            if sheet_name not in zf.namelist():
                return []
            root = ET.fromstring(zf.read(sheet_name))
    except Exception:
        return []

    rows = []
    for row in root.findall('.//x:sheetData/x:row', ns):
        vals = {}
        for c in row.findall('x:c', ns):
            ref = c.attrib.get('r', '')
            m = re.match(r'([A-Z]+)(\d+)', ref)
            if not m:
                continue
            col = m.group(1)
            t = c.attrib.get('t', '')
            v = c.find('x:v', ns)
            if v is None or v.text is None:
                continue
            raw = v.text
            if t == 's':
                try:
                    val = shared[int(raw)]
                except Exception:
                    val = ''
            else:
                val = raw
            vals[col] = val
        if len(vals) > 0:
            rows.append(vals)
    if len(rows) == 0:
        return []

    first = rows[0]
    header_col = None
    for col, val in first.items():
        low = str(val).strip().lower()
        if low in ('smi', 'smiles'):
            header_col = col
            break
    if header_col is not None:
        start_idx = 1
    else:
        ordered_cols = sorted(first.keys(), key=lambda x: (len(x), x))
        header_col = ordered_cols[0]
        start_idx = 0

    smis = []
    for r in rows[start_idx:]:
        v = str(r.get(header_col, '')).strip()
        if v != '' and v.lower() != 'nan':
            smis.append(v)
    return smis

def get_biobased_seed_pool(SA_max, max_mol_weight):
    # Load the pre-stored bio-based scaffold library and keep only scaffolds
    # that are valid generation seeds under the current coarse limits.
    df = _load_biobased_dataframe()
    if df.shape[0] == 0:
        return pd.DataFrame(columns=['smi'])
    df['mol'] = df['smi'].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=['mol']).copy()
    df['SA score'] = df['mol'].apply(SA.calculateScore)
    df['Mol wt'] = df['mol'].apply(Descriptors.MolWt)
    if SA_max is not None:
        df = df[df['SA score'] <= SA_max]
    if max_mol_weight is not None:
        df = df[df['Mol wt'] <= max_mol_weight]
    df = annotate_platform_families(df)
    keep_cols = [
        'smi', 'platform_id', 'platform_name', 'platform_anchor_smi',
        'platform_match_score', 'platform_core_retained', 'platform_family_assigned'
    ]
    return df[keep_cols].drop_duplicates(subset=['smi']).reset_index(drop=True)

def load_biobased_seed_population(pop_size, SA_max, max_mol_weight, platform_guided=False):
    # This is the workflow entry population, but it is not a random
    # initialization step. It is a sampled working set drawn from the
    # preloaded bio-based scaffold seed library.
    df_seed = get_biobased_seed_pool(SA_max, max_mol_weight)
    if df_seed.shape[0] == 0:
        return generate_random_start_pop(pop_size, SA_max, max_mol_weight)
    if platform_guided:
        df_pop = sample_platform_balanced(
            df_seed,
            sample_size=pop_size,
            replace=(df_seed.shape[0] < pop_size),
            random_state=42
        ).copy()
    elif df_seed.shape[0] >= pop_size:
        df_pop = df_seed.sample(pop_size, replace=False).copy()
    else:
        df_pop = df_seed.sample(pop_size, replace=True).copy()
    df_pop['mol'] = df_pop['smi'].apply(Chem.MolFromSmiles)
    df_pop['fitness value'] = ''
    df_pop['prop'] = df_pop['mol'].apply(calc_prop)
    df_pop = get_fitness_vals(df_pop)
    return df_pop

def generate_biobased_start_pop(pop_size, SA_max, max_mol_weight, platform_guided=False):
    # Backward-compatible wrapper. The actual role of this step is to load
    # a seed population from the bio-based scaffold library.
    return load_biobased_seed_population(
        pop_size=pop_size,
        SA_max=SA_max,
        max_mol_weight=max_mol_weight,
        platform_guided=platform_guided
    )

def _init_biobased_similarity():
    global BIO_BASED_FP_LIST, BIO_BASED_SMI_SET
    if len(BIO_BASED_FP_LIST) > 0:
        return
    df = _load_biobased_dataframe()
    if df.shape[0] == 0:
        return
    fps = []
    smis = set()
    for smi in df['smi'].tolist():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
        smis.add(smi)
    BIO_BASED_FP_LIST = fps
    BIO_BASED_SMI_SET = smis

def _max_biobased_similarity(smi):
    if smi in BIO_BASED_SIM_CACHE:
        return BIO_BASED_SIM_CACHE[smi]
    _init_biobased_similarity()
    if len(BIO_BASED_FP_LIST) == 0:
        BIO_BASED_SIM_CACHE[smi] = 0.0
        return 0.0
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        BIO_BASED_SIM_CACHE[smi] = 0.0
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    sims = DataStructs.BulkTanimotoSimilarity(fp, BIO_BASED_FP_LIST)
    val = float(max(sims)) if len(sims) > 0 else 0.0
    BIO_BASED_SIM_CACHE[smi] = val
    return val

# 与GA一致的氢化处理
def aromatic_to_saturated_smiles(aromatic_smi):
    """仅氢化原分子中芳香环键（论文口径更接近的氢化对定义）。"""
    if aromatic_smi in SAT_SMILES_CACHE_RING:
        return SAT_SMILES_CACHE_RING[aromatic_smi]
    mol = Chem.MolFromSmiles(aromatic_smi)
    if mol is None:
        SAT_SMILES_CACHE_RING[aromatic_smi] = None
        return None
    # Track aromatic ring bonds on the original aromatic representation.
    aromatic_ring_bond_idxs = set()
    for b in mol.GetBonds():
        if b.GetIsAromatic() and b.IsInRing():
            aromatic_ring_bond_idxs.add(b.GetIdx())
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        SAT_SMILES_CACHE_RING[aromatic_smi] = None
        return None

    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        if bond.GetIdx() in aromatic_ring_bond_idxs:
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetIsAromatic(False)

    sat = rw.GetMol()
    sat.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(sat)
    except Exception:
        SAT_SMILES_CACHE_RING[aromatic_smi] = None
        return None
    sat = Chem.AddHs(sat)
    val = Chem.MolToSmiles(sat, kekuleSmiles=True)
    SAT_SMILES_CACHE_RING[aromatic_smi] = val
    return val

def fully_saturated_smiles(aromatic_smi):
    """将全分子所有不饱和键氢化为单键。"""
    if aromatic_smi in SAT_SMILES_CACHE_FULL:
        return SAT_SMILES_CACHE_FULL[aromatic_smi]
    mol = Chem.MolFromSmiles(aromatic_smi)
    if mol is None:
        SAT_SMILES_CACHE_FULL[aromatic_smi] = None
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        SAT_SMILES_CACHE_FULL[aromatic_smi] = None
        return None

    rw = Chem.RWMol(mol)
    for bond in rw.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)

    sat = rw.GetMol()
    sat.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(sat)
    except Exception:
        SAT_SMILES_CACHE_FULL[aromatic_smi] = None
        return None
    sat = Chem.AddHs(sat)
    val = Chem.MolToSmiles(sat, kekuleSmiles=True)
    SAT_SMILES_CACHE_FULL[aromatic_smi] = val
    return val

def count_h2_consumed(aromatic_smi, sat_smi=None):
    """
    Count H2 consumed by hydrogenating only ring unsaturated bonds.
    This follows the same ring-only rule as aromatic_to_saturated_smiles.
    """
    mol_arom = Chem.MolFromSmiles(aromatic_smi)
    if mol_arom is None:
        return None
    if sat_smi is None:
        sat_smi = aromatic_to_saturated_smiles(aromatic_smi)
    mol_sat = Chem.MolFromSmiles(sat_smi) if sat_smi else None
    if mol_sat is None:
        return None

    mol_arom_h = Chem.AddHs(mol_arom)
    mol_sat_h = Chem.AddHs(mol_sat)
    nH_arom = sum(1 for atom in mol_arom_h.GetAtoms() if atom.GetSymbol() == 'H')
    nH_sat = sum(1 for atom in mol_sat_h.GetAtoms() if atom.GetSymbol() == 'H')
    h2_mol = (nH_sat - nH_arom) / 2.0
    if h2_mol <= 0:
        return None
    return h2_mol


def _count_h2_consumed_cached(aromatic_smi, sat_smi=None):
    key = (aromatic_smi, sat_smi)
    if key in H2_COUNT_CACHE:
        return H2_COUNT_CACHE[key]
    val = count_h2_consumed(aromatic_smi, sat_smi=sat_smi)
    H2_COUNT_CACHE[key] = val
    return val

def read_frags_from_csv():
    frag_path = './fragments.csv'

    # for windows
    df_frag = pd.read_csv(frag_path, encoding='cp1252', sep=';')
    # for linux
    # df_frag = pd.read_csv(frag_path, encoding='utf-8', sep=';')

    mol_list = []
    name_list = []
    smi_list = df_frag['SMILES'].to_list()
    name_list = df_frag['frag name'].to_list()
    for num in range(len(smi_list)):
        smi = smi_list[num]
        name = name_list[num]
        mol = Chem.MolFromSmiles(smi)

        if mol is not None:
            mol_list.append(mol)
            name_list.append(name)
        else:
            print('@read_frags_from_csv: mol is none for %s' %(name))
    return mol_list

def generate_start_pop_hexane(pop_size):
    smi = 'CCCCCC'
    start_pop_list = []
    mol = Chem.MolFromSmiles(smi)
    smi_list = []

    for i in range(pop_size):
        start_pop_list.append(mol)
        smi_list.append(smi)

    data = {'mol': start_pop_list, 'smi': smi_list}
    df_pop = pd.DataFrame(data=data)
    df_pop['fitness value'] = ''
    df_pop['SA score'] = ''

    df_pop['prop'] = df_pop['mol'].apply(calc_prop)

    df_pop['SA score'] = df_pop['mol'].apply(SA.calculateScore)

    # get fitness vals
    df_pop = get_fitness_vals(df_pop)

    return df_pop

def generate_random_start_pop(pop_size, SA_max, max_mol_weight):

    if not os.path.exists('random_start_pop.csv'):
        df_pop = pd.read_csv('mcule_preselection.csv', sep=',')
        df_pop['mol'] = df_pop['smi'].apply(Chem.MolFromSmiles)
        df_pop['SA score'] = df_pop['mol'].apply(SA.calculateScore)
        df_pop['Mol wt'] = df_pop['mol'].apply(Descriptors.MolWt)
        if SA_max is not None:
            drop_idxs = df_pop.index[df_pop['SA score'] > SA_max]
            df_pop = df_pop.drop(drop_idxs)
        if max_mol_weight is not None:
            drop_idxs = df_pop.index[df_pop['Mol wt'] > max_mol_weight]
            df_pop = df_pop.drop(drop_idxs)
        df_pop = df_pop.sample(pop_size)
        df_pop['fitness value'] = ''
        df_pop['prop'] = df_pop['mol'].apply(calc_prop)

        # get fitness vals
        df_pop = get_fitness_vals(df_pop)
        df_pop.to_csv('random_start_pop.csv')

    else:
        df_pop = pd.read_csv('random_start_pop.csv')
        df_pop['mol'] = df_pop['smi'].apply(Chem.MolFromSmiles)
        df_pop = df_pop.dropna(subset=['mol']).copy()
        # Re-apply latest constraints even when cached start population exists.
        df_pop['SA score'] = df_pop['mol'].apply(SA.calculateScore)
        df_pop['Mol wt'] = df_pop['mol'].apply(Descriptors.MolWt)
        if SA_max is not None:
            df_pop = df_pop[df_pop['SA score'] <= SA_max]
        if max_mol_weight is not None:
            df_pop = df_pop[df_pop['Mol wt'] <= max_mol_weight]
        df_pop = df_pop.drop_duplicates(subset=['smi'], keep='first')
        if df_pop.shape[0] >= pop_size:
            df_pop = df_pop.sample(pop_size, replace=False)
        elif df_pop.shape[0] > 0:
            df_pop = df_pop.sample(pop_size, replace=True)
        df_pop['fitness value'] = ''
        df_pop['prop'] = df_pop['mol'].apply(calc_prop)
        df_pop = get_fitness_vals(df_pop)

    return df_pop


def get_valences(mol):
    # implicit valences
    num_atoms = mol.GetNumAtoms()
    atom_idx_list = []
    atom_symbol_list = []
    valence_list = []

    for i in range(num_atoms):
        valence = mol.GetAtomWithIdx(i).GetImplicitValence()
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        atom_idx_list.append(i)
        atom_symbol_list.append(symbol)
        valence_list.append(valence)
    return atom_idx_list, atom_symbol_list, valence_list

def get_total_valences(mol):
    # implicit valences
    num_atoms = mol.GetNumAtoms()
    atom_idx_list = []
    atom_symbol_list = []
    valence_list = []

    for i in range(num_atoms):
        valence = mol.GetAtomWithIdx(i).GetTotalValence()
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        atom_idx_list.append(i)
        atom_symbol_list.append(symbol)
        valence_list.append(valence)
    return atom_idx_list, atom_symbol_list, valence_list

def get_explicit_valences(mol):
    # implicit valences
    num_atoms = mol.GetNumAtoms()
    atom_idx_list = []
    atom_symbol_list = []
    valence_list = []

    for i in range(num_atoms):
        valence = mol.GetAtomWithIdx(i).GetExplicitValence()
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        atom_idx_list.append(i)
        atom_symbol_list.append(symbol)
        valence_list.append(valence)
    return atom_idx_list, atom_symbol_list, valence_list

def deletion(mol):
    """
    performs atom deletion on rdkit mol object

    checks if number of atoms is > 1 to ensure that no empty molecule results

    the networkx package returns the atom indices which - if deleted - result in a splitted molecule
    these atoms are not open for deletion. any other atom can be deleted (randomly chosen)

    returns rdkit mol object

    author: laura koenig-mattern

    """

    flag_successful = 0

    start_mol = copy.deepcopy(mol)

    if mol.GetNumAtoms() > 1:  #note for older implementation: it is possible to set mol weight as limit cause otherwise graphs get too large and nx.all_cut_sets needs hours

        atom_idx_set = set()
        for atom in mol.GetAtoms():
            atom_idx_set.add(atom.GetIdx())


        adj_matrix = Chem.GetAdjacencyMatrix(mol)

        # get sets of nodes (-> atoms) that, if deleted, split the molecule into two or more fragments
        # articulation points
        graph = nx.Graph(adj_matrix)
        mol_idx_set = set(list(nx.articulation_points(graph)))
        removable_atom_idx = list(atom_idx_set.difference(mol_idx_set))

        if len(removable_atom_idx) == 0:
            flag_successful = 0
            del_mol = start_mol
            del_mol.UpdatePropertyCache()
            Chem.SanitizeMol(del_mol)
            return del_mol, flag_successful

        rand_int = np.random.randint(len(removable_atom_idx))


        rm_idx = removable_atom_idx[rand_int]

        Chem.Kekulize(mol, clearAromaticFlags=True)

        rw_mol = Chem.RWMol(mol)
        rw_mol.RemoveAtom(rm_idx)

        del_mol = rw_mol.GetMol()
    else:
        del_mol = start_mol
        del_mol.UpdatePropertyCache()
        Chem.SanitizeMol(del_mol)


    if start_mol.GetNumAtoms() > del_mol.GetNumAtoms():
        flag_successful = 1

    if del_mol is not None:
        del_mol.UpdatePropertyCache()
        Chem.SanitizeMol(del_mol)


    return del_mol, flag_successful

def fragmentor(mol):
    """
    splits a given molecule into fragments based on bridge connections

    :param mol: rdkit mol object to be fragmented
    :return: frag_mol: fragmented molecule, flag_successful: 0 - operation not successful, 1 - operation successful

    author: laura koenig-mattern
    """

    flag_successful = 0
    num_bonds = mol.GetNumBonds()

    if num_bonds >= 1:
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        graph = nx.Graph(adj_matrix)

        # returns bridges of the graph as tuple of node indices stored in list
        bridges = [i for i in nx.bridges(graph)]

        if len(bridges) >= 1:
            # get bond object of the bridges
            rand_int = np.random.randint(len(bridges))
            tup = bridges[rand_int]
            atom_idx_1 = tup[0]
            atom_idx_2 = tup[1]

            Chem.Kekulize(mol, clearAromaticFlags=True)

            rw_mol = Chem.RWMol(mol)
            rw_mol.RemoveBond(atom_idx_1, atom_idx_2)

            frag_mol = rw_mol.GetMol()

            if len(Chem.GetMolFrags(frag_mol)) > 1:
                flag_successful = 1

        else:
            # split ring to get fragmented mol
            frag_mol, flag_successful = split_ring(mol)

    else:
        frag_mol = mol
        flag_successful = 0

    frag_mol.UpdatePropertyCache()
    Chem.SanitizeMol(frag_mol)

    return frag_mol, flag_successful

def combinator(frag_mol):
    """
    randomly combines n given molecular fragments until all frags are connected

    :param frag_mol: rdkit mol object containing n fragments
    :return: comb_mol: rdkit mol object containing the connected mol, flag_successful: 0 - not successful, 1 - successful

    author: laura koenig-mattern
    """

    flag_successful = 0
    start_mol = copy.deepcopy(frag_mol)
    idx_tup = Chem.GetMolFrags(frag_mol)
    num_frags = len(idx_tup)

    counter = 0

    while num_frags > 1:
        Chem.Kekulize(frag_mol, clearAromaticFlags=True)
        frag_mol.UpdatePropertyCache()

        atom_idx_list, atom_symbol_list, valence_list = get_valences(frag_mol)

        val_tup = []
        for tup in idx_tup:
            v_tup = tuple([valence_list[element] for element in tup])
            val_tup.append(v_tup)

        # eligable atom idices (eligable means valence is larger than 1 -> eligable for bonding)
        eligable_idx = []

        count_non_eligable_frags = 0

        for num, tup in enumerate(val_tup):
            arr = np.asarray(tup)
            nonzero_idx = np.flatnonzero(arr)
            atm_idxs_elig = np.array(idx_tup[num])[list(nonzero_idx)]
            eligable_idx.append(atm_idxs_elig.tolist())
            if len(list(nonzero_idx)) == 0:
                count_non_eligable_frags = count_non_eligable_frags + 1

        if count_non_eligable_frags > 0:
            #print('combination fails. Number of eligable fragments < number of fragments')
            comb_mol = frag_mol
            return comb_mol, flag_successful

        rng = np.random.default_rng()
        frag_idx = rng.choice(num_frags, 2, replace=False)

        rand_int_1 = np.random.randint(len(eligable_idx[frag_idx[0]]))
        atm_idx1 = eligable_idx[frag_idx[0]][rand_int_1]
        val_atm1 = valence_list[atm_idx1]
        rand_int_2 = np.random.randint(len(eligable_idx[frag_idx[1]]))
        atm_idx2 = eligable_idx[frag_idx[1]][rand_int_2]
        val_atm2 = valence_list[atm_idx2]


        rw_mol = Chem.RWMol(frag_mol)


        if val_atm1 <= 1 or val_atm2 <= 1:
            rw_mol.AddBond(atm_idx1, atm_idx2, order=Chem.BondType.SINGLE)
            #print('bond added')
        if val_atm1 > 1 and val_atm2 > 1:
            rand_int = np.random.randint(2)
            if rand_int == 0:
                rw_mol.AddBond(atm_idx1, atm_idx2, order=Chem.BondType.SINGLE)
                #print('bond added')
            if rand_int == 1:
                rw_mol.AddBond(atm_idx1, atm_idx2, order=Chem.BondType.DOUBLE)
                #print('bond added')

        frag_mol = rw_mol.GetMol()
        frag_mol.UpdatePropertyCache()
        idx_tup = Chem.GetMolFrags(frag_mol)
        num_frags = len(idx_tup)

        counter = counter + 1

        if counter > 100:
            break

    comb_mol = frag_mol

    if num_frags == 1:
        flag_successful = 1

    if num_frags > 1:
        flag_successful = 0
        comb_mol = start_mol

    if comb_mol is not None:
        comb_mol.UpdatePropertyCache()
        Chem.SanitizeMol(comb_mol)
    return comb_mol, flag_successful

def addition(mol1, mol2):
    '''
    connects two molecular fragments to obtain one mol
    :param mol1: rdkit mol obj of fragment 1
    :param mol2: rdkit mol obj of fragment 2
    :return: add_mol: rdkit mol obj of connected mol, flag successful: 0 - not successful, 1 - successful
    '''

    flag_successful = 0
    new_mol = Chem.CombineMols(mol1, mol2)
    add_mol, flag_successful_comb = combinator(new_mol)

    if len(Chem.GetMolFrags(add_mol)) == 1:
        flag_successful = 1

    add_mol.UpdatePropertyCache()
    Chem.SanitizeMol(add_mol)

    return add_mol, flag_successful

def relocation(mol):
    """
    fragments a given molecule into two or more fragments. randomly recombines these fragments
    to obtain a new molecules
    :param mol: rdkit mol obj containing the molecule for relocation
    :return: rel_mol: rdkit mol obj containing the relocated molecule, flagsuccessful: 0 - not successful, 1 - successful

    author: laura koenig-mattern
    """

    flag_successful = 0
    frag_mol, flag_successful_frag = fragmentor(mol)
    rel_mol, flag_successful_comb = combinator(frag_mol)

    if flag_successful_comb == 1 and flag_successful_frag == 1:
        flag_successful = 1

    if flag_successful == 1:
        rel_mol.UpdatePropertyCache()
        Chem.SanitizeMol(rel_mol)


    return rel_mol, flag_successful

def insertion(mol1, mol2):
    """
    inserts one molecule into another molecule. the molecule to be inserted remains the same.
    the molecule to which it should be inserted, is fragmented. the fragments + the mol to be inserted are combined.
    the molecule to be fragmented/inserted is chosen randomly. before the operation is executed, the possible combinations are determined

    :param mol1: rdkit mol obj of one molecule
    :param mol2: rdkit mol obj of the other  molecule
    :return: ins_mol: rdkit obj of inserted mol, flag_successful: 0 - operation not successful, 1 - operation successful

    author: laura könig-mattern
    """

    frag_mol1, flag_successful1 = fragmentor(mol1)
    frag_mol2, flag_successful2 = fragmentor(mol2)


    atom_idx_list1, atom_symbol_list1, valence_list1 = get_valences(mol1)
    atom_idx_list2, atom_symbol_list2, valence_list2 = get_valences(mol2)

    inserted_frag_bool_1 = False
    inserted_frag_bool_2 = False
    insertion_possible_bool_1 = False
    insertion_possible_bool_2 = False

    if np.any(valence_list1):
        inserted_frag_bool_1 = True
    if np.any(valence_list2):
        inserted_frag_bool_2 = True
    if flag_successful1 == 1:
        insertion_possible_bool_1 = True
    if flag_successful2 == 1:
        insertion_possible_bool_2 = True


    flag_successful_ins = 0
    flag_successful_frag = 0
    flag_successful = 0

    ins_mol = None

    if inserted_frag_bool_1 and not inserted_frag_bool_2 and not insertion_possible_bool_1 and insertion_possible_bool_2:
        # insert frag one into frag two
        frag_mol = Chem.CombineMols(mol1, frag_mol2)
        ins_mol, flag_successful_ins = combinator(frag_mol)
        flag_successful_frag = 1

    # 0 1 1 0
    if not inserted_frag_bool_1 and inserted_frag_bool_2 and insertion_possible_bool_1 and not insertion_possible_bool_2:
        # insert frag two into frag one
        frag_mol = Chem.CombineMols(frag_mol1, mol2)
        ins_mol, flag_successful_ins = combinator(frag_mol)
        flag_successful_frag = 1
    # 1 0 1 1
    if inserted_frag_bool_1 and not inserted_frag_bool_2 and insertion_possible_bool_1 and insertion_possible_bool_2:
        # insert frag 1 in frag 2
        frag_mol = Chem.CombineMols(mol1, frag_mol2)
        ins_mol, flag_successful_ins = combinator(frag_mol)
        flag_successful_frag = 1
    # 0 1 1 1
    if not inserted_frag_bool_1 and inserted_frag_bool_2 and insertion_possible_bool_1 and insertion_possible_bool_2:
        # insert frag 2 in frag 1
        frag_mol = Chem.CombineMols(frag_mol1, mol2)
        ins_mol, flag_successful_ins = combinator(frag_mol)
        flag_successful_frag = 1
    # 1 1 1 0
    if inserted_frag_bool_1 and inserted_frag_bool_2 and insertion_possible_bool_1 and not insertion_possible_bool_2:
        # insert mol 2 into mol 1
        frag_mol = Chem.CombineMols(frag_mol1, mol2)
        ins_mol, flag_successful_ins = combinator(frag_mol)
        flag_successful_frag = 1
    # 1 1 0 1
    if inserted_frag_bool_1 and inserted_frag_bool_2 and not  insertion_possible_bool_1 and insertion_possible_bool_2:
        # insert mol 1 into mol 2
        frag_mol = Chem.CombineMols(mol1, frag_mol2)
        ins_mol, flag_successful_ins = combinator(frag_mol)
        flag_successful_frag = 1
    # 1 1 1 1
    if inserted_frag_bool_1 and inserted_frag_bool_2 and insertion_possible_bool_1 and insertion_possible_bool_2:
        # choose both randomly
        rand_ins_int = np.random.randint(2)
        if rand_ins_int == 0:
            frag_mol = Chem.CombineMols(frag_mol1, mol2)
            ins_mol, flag_successful_ins = combinator(frag_mol)
            flag_successful_frag = 1
        if rand_ins_int == 1:
            frag_mol = Chem.CombineMols(mol1, frag_mol2)
            ins_mol, flag_successful_ins = combinator(frag_mol)
            flag_successful_frag = 1

    if flag_successful_frag == 1 and flag_successful_ins == 1 and ins_mol is not None:
        flag_successful = 1

    if ins_mol is not None:
        ins_mol.UpdatePropertyCache()
        Chem.SanitizeMol(ins_mol)

    return ins_mol, flag_successful

def bond_mutation(mol):
    """
    mutates bond type: single to double or double to single bond. bond is randomly chosen.
    if single bond: check valences if mutation to double bond is possible

    :param mol: rdkit mol obj containing the mol to be mutated
    :return: mol: rdkit mol obj containing mutated mol, flag_successful: 0 - not successful, 1 - successful

    author: laura koenig-mattern
    """
    flag_successful = 0

    ini_mol = copy.deepcopy(mol)

    bonds = [bond for bond in mol.GetBonds()]
    atom_idx_list, atom_symbol_list, valence_list = get_valences(mol)
    num_bonds = mol.GetNumBonds()
    rand_bond_idxs = np.random.choice(num_bonds, num_bonds, replace=False)

    for num in range(num_bonds):
        rand_bond_idx = rand_bond_idxs[num]
        bond = bonds[rand_bond_idx]
        type = bond.GetBondType()


        atm_idx1 = bond.GetBeginAtomIdx()
        atm_idx2 = bond.GetEndAtomIdx()
        val1 = valence_list[atm_idx1]
        val2 = valence_list[atm_idx2]

        if str(type) == 'SINGLE':
            if val1 >= 1 and val2 >= 1:
                bond.SetBondType(Chem.BondType.DOUBLE)
                flag_successful = 1
                break

        if str(type) == 'DOUBLE':
            bond.SetBondType(Chem.BondType.SINGLE)
            flag_successful = 1
            break

        if str(type) == 'AROMATIC':
            copy_mol = copy.deepcopy(mol)
            Chem.Kekulize(copy_mol, clearAromaticFlags=True)
            copy_bonds = [bond for bond in copy_mol.GetBonds()]
            copy_bond = copy_bonds[rand_bond_idx]
            copy_type = copy_bond.GetBondType()

            if str(copy_type) == 'DOUBLE':
                Chem.Kekulize(mol, clearAromaticFlags=True)
                bond.SetBondType(Chem.BondType.SINGLE)
                flag_successful = 1
                break

            if str(copy_type) == 'SINGLE':
                Chem.SanitizeMol(mol)
                mol = ini_mol

    if mol is not None:
        try:
            mol.UpdatePropertyCache()
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except:
            #import ipdb; ipdb.set_trace()
            mol.UpdatePropertyCache()
            Chem.Kekulize(mol, clearAromaticFlags=True)
        try:
            mol.UpdatePropertyCache()
            Chem.SanitizeMol(mol)
        except:
            import ipdb; ipdb.set_trace()

    return mol, flag_successful

def selection(df_pop, num_parents):
    """
    selects num_parents fittest molecules based on roulette wheel selection for cross-over

    :param df_pop: dataframe of the population containing rdkit mol obj, MP, BP, property predictions, fitnes vals
    :param num_parents: number of mols to be chosen for cross over using roulette wheel selection
    :return: selected_iloc_idxs: iloc-indices of the selected mols from df_pop

    author: laura koenig-mattern
    """

    if 'selection_score' in df_pop.columns:
        fitness_vals = pd.to_numeric(df_pop['selection_score'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    else:
        fitness_vals = df_pop['fitness value'].to_numpy().reshape(df_pop['fitness value'].shape[0]).astype(np.float64)

    # rank-based selection keeps pressure smoother and preserves diversity longer
    order = np.argsort(fitness_vals)  # ascending
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(fitness_vals) + 1, dtype=np.float64)
    probs = np.power(ranks, RANK_SELECTION_PRESSURE)
    probs = probs / np.sum(probs)
    replace_flag = num_parents > len(fitness_vals)
    selected_iloc_idxs = np.random.choice(len(fitness_vals), num_parents, p=probs, replace=replace_flag)

    return selected_iloc_idxs

def cross_over(
    df_pop,
    num_parents,
    num_children,
    max_mol_weight,
    group_constraints,
    SA_max,
    same_platform_only=False,
    retain_parent_scaffold=False,
    max_attempts=None
):
    """
    performs cross-over. num_parents selected parent mols are fragmented, these fragments are the "mating pool"
    randomly choose fragments from the mating pool, to produce num_children child molecules
    :param df_pop: pandas dataframe of population
    :param num_parents: number of parent mols
    :param num_children: number of child mols
    :param max_mol_weight: maximum mol weight
    :param group_constraints: constraints on predefined functional groups
    :param SA_max: max synthetic accessibility
    :return: df_pop: pandas dataframe with pop + child mols
    """
    # select individuals based on roulette wheel selection
    selected_iloc_idxs = selection(df_pop, num_parents)

    # fragment the chosen molecules and add fragments to pool
    frag_pool = []
    frag_pool_by_platform = {}
    parent_scaffolds_by_platform = {}
    mol_col_idx = df_pop.columns.get_loc('mol')
    platform_col_idx = df_pop.columns.get_loc('platform_id') if 'platform_id' in df_pop.columns else None
    for iloc_idx in selected_iloc_idxs:
        mol = df_pop.iloc[iloc_idx, mol_col_idx]
        platform_id = None if platform_col_idx is None else df_pop.iloc[iloc_idx, platform_col_idx]
        platform_key = None if platform_id is None else str(platform_id)
        copy_mol = copy.deepcopy(mol)
        if platform_key is not None and platform_key != 'unassigned':
            parent_scaffold = _safe_murcko_scaffold_smiles(copy_mol)
            if parent_scaffold != '':
                parent_scaffolds_by_platform.setdefault(platform_key, set()).add(parent_scaffold)

        frag_mol, flag_successful = fragmentor(copy_mol)

        if flag_successful == 1:
            frags = Chem.GetMolFrags(frag_mol, asMols=True)

            for frag in frags:
                frag_pool.append(frag)
                if same_platform_only and platform_key is not None and platform_key != 'unassigned':
                    frag_pool_by_platform.setdefault(platform_key, []).append(frag)

        # if flag_successful == 0:
        #    import ipdb; ipdb.set_trace()

    # form desired number of children
    num_children_formed = 0
    smi_children = []
    child_platform_ids = []
    counter = 0
    if max_attempts is None:
        max_attempts = max(100, int(num_children) * 20)
    max_attempts = max(1, int(max_attempts))
    while num_children_formed < num_children:
        chosen_platform = None
        active_pool = frag_pool
        if same_platform_only:
            eligible_platforms = [pid for pid, pool in frag_pool_by_platform.items() if len(pool) >= 2]
            if len(eligible_platforms) > 0:
                chosen_platform = str(np.random.choice(eligible_platforms))
                active_pool = frag_pool_by_platform[chosen_platform]
        if len(active_pool) < 2:
            break
        # randomly choose two fragments from pool
        frag_idxs = np.random.choice(len(active_pool), 2, replace=False)
        idx1 = frag_idxs[0]
        idx2= frag_idxs[1]
        frag1 = active_pool[idx1]
        frag2 = active_pool[idx2]
        new_mol = Chem.CombineMols(frag1, frag2)

        # combine
        comb_mol, flag_comb_successful = combinator(new_mol)
        if flag_comb_successful == 1:
            try:
                comb_mol.UpdatePropertyCache()
                Chem.SanitizeMol(comb_mol)

            except:
                import ipdb; ipdb.set_trace()
            if Chem.Descriptors.MolWt(comb_mol) > max_mol_weight:
                flag_comb_successful = 0
            if SA_max:
                SA_score = SA.calculateScore(comb_mol)
                if SA_score > SA_max:
                    flag_comb_successful = 0

        counter = counter + 1

        if flag_comb_successful == 1 and _contains_only_allowed_elements(comb_mol):
            if _is_unreasonable_molecule(comb_mol):
                flag_comb_successful = 0
        if flag_comb_successful == 1 and _contains_only_allowed_elements(comb_mol):
            if same_platform_only and chosen_platform is not None and not _platform_core_is_retained(comb_mol, chosen_platform):
                flag_comb_successful = 0
        if flag_comb_successful == 1 and same_platform_only and retain_parent_scaffold and chosen_platform is not None:
            parent_scaffolds = parent_scaffolds_by_platform.get(chosen_platform, set())
            if len(parent_scaffolds) > 0:
                child_scaffold = _safe_murcko_scaffold_smiles(comb_mol)
                if child_scaffold == '' or child_scaffold not in parent_scaffolds:
                    flag_comb_successful = 0

        if flag_comb_successful == 1 and _contains_only_allowed_elements(comb_mol):
            num_children_formed = num_children_formed + 1
            smi_children.append(Chem.MolToSmiles(comb_mol))
            child_platform_ids.append(chosen_platform)

        if counter >= max_attempts:
            break

    # get the smiles from children which are differnt from each other (make sure to add no duplets)
    seen_children = set()
    smi_children_diff = []
    child_platform_ids_diff = []
    for smi, pid in zip(smi_children, child_platform_ids):
        if smi in seen_children:
            continue
        seen_children.add(smi)
        smi_children_diff.append(smi)
        child_platform_ids_diff.append(pid)

    # only add to frame if not already in population
    residual_smi = smi_children_diff
    for i in range(len(smi_children_diff)):
        smi = smi_children_diff[i]
        child_platform_id = child_platform_ids_diff[i]
        #import ipdb; ipdb.set_trace()
        if not smi in df_pop['smi'].values:
            loc_ind = df_pop.shape[0]
            df_pop.loc[loc_ind,'smi'] = smi
            mol = Chem.MolFromSmiles(smi)  #automatically removes all Hs
            mol.UpdatePropertyCache()
            Chem.SanitizeMol(mol)
            df_pop.loc[loc_ind, 'mol'] = mol
            if same_platform_only and child_platform_id is not None:
                entry = _get_platform_entry(child_platform_id)
                df_pop.loc[loc_ind, 'platform_id'] = child_platform_id
                df_pop.loc[loc_ind, 'platform_name'] = entry['platform_name'] if entry is not None else child_platform_id

    return df_pop
def get_SA_scores(df_pop):
    """
    get synthetic accessibility score (SAS, see Ertl et al.)

    :param df_pop: pandas dataframe of the population
    :return: df_pop: pandas dataframe of the population updates with SAS

    author: laura könig-mattern
    """
    if 'SA score' not in df_pop.columns:
        df_pop['SA score'] = np.nan
    sa_series = pd.to_numeric(df_pop['SA score'], errors='coerce')
    miss_mask = ~np.isfinite(sa_series.to_numpy(dtype=np.float64))
    if np.any(miss_mask):
        miss_idx = df_pop.index[miss_mask]
        df_pop.loc[miss_idx, 'SA score'] = df_pop.loc[miss_idx, 'mol'].apply(SA.calculateScore)
    return df_pop

def calc_prop(mol):
    """
    property prediction
    as an example, we use rdkit's logP calculation
    :param mol: rdkit mol obj
    :return: prop: value of property of interest (logP)

    author: laura koenig-mattern
    """
    prop = Chem.Descriptors.MolLogP(mol)
    #prop = Chem.Crippen.MolLogP(mol)
    return prop

def calc_ring_penalty(mol):
    """
    calculates ring penalty of a molecule. the ring penalty s defined as numbers of rings with more than 6 atoms
    :param mol: rdkit mol ob
    :return: ring penalty

    author: laura koenig-mattern
    """

    ri = mol.GetRingInfo()

    if len(ri.AtomRings()) == 0:
        ring_penalty = 0
    else:
        penalised_rings = []
        for ring in ri.AtomRings():
            penalty_single = len(ring) - 6
            if penalty_single < 0:
                penalty_single = 0
            penalised_rings.append(penalty_single)
            ring_penalty = sum(penalised_rings)


    return ring_penalty

def fast_prefilter_candidates(df_src, max_mol_weight=None):
    df = _copy_with_mol(df_src).reset_index(drop=True)
    pass_fast = []
    molwts = []
    for mol in df['mol'].tolist():
        if mol is None:
            pass_fast.append(False)
            molwts.append(np.nan)
            continue
        try:
            mw = float(Descriptors.MolWt(mol))
        except Exception:
            mw = np.nan
        molwts.append(mw)
        ok = not _is_unreasonable_molecule(mol)
        if ok and max_mol_weight is not None and np.isfinite(mw):
            ok = mw <= float(max_mol_weight)
        pass_fast.append(bool(ok))
    df['molwt_screen'] = molwts
    df['pass_fast_prefilter'] = np.asarray(pass_fast, dtype=np.int64)
    return df

def get_fitness_vals(df_pop, compute_full_hydrogenation=True, scoring_config=None):
    """
    define the fitness_function 
    """
    cfg_score = dict(GA_FITNESS_CONFIG)
    if scoring_config:
        cfg_score.update(scoring_config)
    uncertainty_enabled = bool(cfg_score.get('uncertainty_penalty_enabled', False))
    uncertainty_lambda = float(cfg_score.get('uncertainty_penalty_lambda', 0.05))
    uncertainty_threshold = float(cfg_score.get('uncertainty_threshold', 20.0))
    uncertainty_hard_penalty = float(cfg_score.get('uncertainty_hard_penalty', 0.0))
    lohc_probability_enabled = bool(cfg_score.get('lohc_probability_scoring', False))
    lohc_probability_weight = float(cfg_score.get('lohc_probability_weight', 8.0))

    df_pop = fast_prefilter_candidates(df_pop, max_mol_weight=260.0)

    if 'SA score' not in df_pop.columns:
        df_pop['SA score'] = np.nan
    sa_series = pd.to_numeric(df_pop['SA score'], errors='coerce')
    pre_idx = df_pop.index[df_pop['pass_fast_prefilter'].astype(int) == 1]
    miss_sa_idx = [idx for idx in pre_idx if not np.isfinite(sa_series.loc[idx])]
    if len(miss_sa_idx) > 0:
        df_pop.loc[miss_sa_idx, 'SA score'] = df_pop.loc[miss_sa_idx, 'mol'].apply(SA.calculateScore)

    aromatic_list = df_pop['smi'].astype(str).tolist()
    sat_smiles = [None] * df_pop.shape[0]
    full_sat_smiles = [None] * df_pop.shape[0]
    delta_H_list = [np.nan] * df_pop.shape[0]
    delta_H_uncertainty_list = [np.nan] * df_pop.shape[0]
    delta_H_full_list = [np.nan] * df_pop.shape[0]
    delta_H_full_uncertainty_list = [np.nan] * df_pop.shape[0]

    valid_idx = df_pop.index[df_pop['pass_fast_prefilter'].astype(int) == 1].tolist()
    if len(valid_idx) > 0:
        valid_aromatic = [aromatic_list[i] for i in valid_idx]
        valid_sat = [aromatic_to_saturated_smiles(smi) for smi in valid_aromatic]
        for pos, idx in enumerate(valid_idx):
            sat_smiles[idx] = valid_sat[pos]
        if uncertainty_enabled:
            pred_ring, pred_ring_std = _predict_delta_H_uncertainty_cached(
                valid_sat, valid_aromatic, DELTA_H_UNCERTAINTY_CACHE_RING
            )
            for pos, idx in enumerate(valid_idx):
                delta_H_list[idx] = pred_ring[pos]
                delta_H_uncertainty_list[idx] = pred_ring_std[pos]
        else:
            pred_ring = _predict_delta_H_cached(valid_sat, valid_aromatic, DELTA_H_CACHE_RING)
            for pos, idx in enumerate(valid_idx):
                delta_H_list[idx] = pred_ring[pos]

        if compute_full_hydrogenation:
            valid_full_sat = [fully_saturated_smiles(smi) for smi in valid_aromatic]
            for pos, idx in enumerate(valid_idx):
                full_sat_smiles[idx] = valid_full_sat[pos]
            if uncertainty_enabled:
                pred_full, pred_full_std = _predict_delta_H_uncertainty_cached(
                    valid_full_sat, valid_aromatic, DELTA_H_UNCERTAINTY_CACHE_FULL
                )
                for pos, idx in enumerate(valid_idx):
                    delta_H_full_list[idx] = pred_full[pos]
                    delta_H_full_uncertainty_list[idx] = pred_full_std[pos]
            else:
                pred_full = _predict_delta_H_cached(
                    valid_full_sat,
                    valid_aromatic,
                    DELTA_H_CACHE_FULL,
                )
                for pos, idx in enumerate(valid_idx):
                    delta_H_full_list[idx] = pred_full[pos]

    df_pop['sat_smi'] = sat_smiles
    df_pop['full_sat_smi'] = full_sat_smiles
    df_pop['delta_H'] = delta_H_list
    df_pop['delta_H_uncertainty'] = delta_H_uncertainty_list
    df_pop['delta_H_full'] = delta_H_full_list
    df_pop['delta_H_full_uncertainty'] = delta_H_full_uncertainty_list

    # 全脱氢与仅环脱氢的焓变差
    delta_H_diff_list = []
    for dh_ring, dh_full in zip(delta_H_list, delta_H_full_list):
        if dh_ring is None or dh_full is None or np.isnan(dh_ring) or np.isnan(dh_full):
            delta_H_diff_list.append(np.nan)
        else:
            delta_H_diff_list.append(abs(dh_full - dh_ring))
    df_pop['delta_H_diff'] = delta_H_diff_list

    # 模型已直接输出每 mol H2 的焓变：不再做除以 h2_mol 的换算
    delta_H_per_H2_list = []
    h2_list = []
    h2_wt_percent_list = []
    for arom, sat, dh in zip(aromatic_list, sat_smiles, delta_H_list):
        h2_mol = _count_h2_consumed_cached(arom, sat)
        h2_list.append(h2_mol)
        sat_mol = Chem.MolFromSmiles(sat) if sat else None
        sat_mw = Descriptors.MolWt(sat_mol) if sat_mol else np.nan

        # 直接使用模型预测值（单位：kJ/mol-H2）
        if dh is not None and not np.isnan(dh):
            delta_H_per_H2_list.append(dh)
        else:
            delta_H_per_H2_list.append(np.nan)

        if h2_mol is not None and sat_mw and sat_mw > 0:
            h2_wt_percent_list.append(h2_mol * 2.01588 / sat_mw * 100.0)
        else:
            h2_wt_percent_list.append(np.nan)
    df_pop['h2_mol'] = h2_list
    df_pop['delta_H_per_H2'] = delta_H_per_H2_list
    df_pop['h2_wt_percent'] = h2_wt_percent_list

    # Soft scoring only: remove clearly unreasonable molecules, but do not hard-screen
    # candidates by SA / h2wt / ring-count style heuristics before Pareto analysis.
    fitness_list = []
    uncertainty_penalty_list = []
    lohc_probability_list = []
    bio_sim_list = []
    target_hit_list = []
    template_reward_list = []
    gamma_h2 = 1.0
    gamma_sa = 1.2
    gamma_enthalpy = 0.55
    gamma_window_bonus = 2.0
    mw_penalty = 0.02

    sa_vals = pd.to_numeric(df_pop['SA score'], errors='coerce').to_list()

    molwt_series = pd.to_numeric(df_pop.get('molwt_screen', pd.Series(np.nan, index=df_pop.index)), errors='coerce')

    for smi, mol, x, x_uncertainty, h2wt, sa, molwt in zip(
        df_pop['smi'],
        df_pop['mol'],
        delta_H_per_H2_list,
        delta_H_uncertainty_list,
        h2_wt_percent_list,
        sa_vals,
        molwt_series.tolist()
    ):
        fit = -1e6
        bio_sim = np.nan
        template_reward = 0.0
        target_hit = 0
        if mol is None or x is None or np.isnan(x):
            fitness_list.append(fit)
            uncertainty_penalty_list.append(np.nan)
            lohc_probability_list.append(0.0)
            bio_sim_list.append(bio_sim)
            target_hit_list.append(target_hit)
            template_reward_list.append(template_reward)
            continue

        if _is_unreasonable_molecule(mol):
            fitness_list.append(fit)
            uncertainty_penalty_list.append(np.nan)
            lohc_probability_list.append(0.0)
            bio_sim_list.append(bio_sim)
            target_hit_list.append(target_hit)
            template_reward_list.append(template_reward)
            continue

        bio_sim = _max_biobased_similarity(smi)
        template_reward, target_hit, furan_hit = _target_template_features_cached(smi)
        if furan_hit:
            x_low, x_high, x_ref = DH_LOW, DH_HIGH, 55.0
        else:
            x_low, x_high, x_ref = DH_LOW, DH_HIGH, 61.0

        fit = 0.0
        if h2wt is not None and np.isfinite(h2wt):
            fit += gamma_h2 * float(h2wt)
        if sa is None or not np.isfinite(sa) or sa > 4.0:
            fit = -1e6
        fit += gamma_window_bonus * float(dh_quality(x))
        mw = molwt if np.isfinite(molwt) else Descriptors.MolWt(mol)
        if np.isfinite(mw) and mw > 260.0:
            fit -= mw_penalty * (mw - 260.0)

        uncertainty_penalty = 0.0
        if uncertainty_enabled:
            if x_uncertainty is None or not np.isfinite(x_uncertainty):
                uncertainty_penalty = uncertainty_lambda * uncertainty_threshold
            else:
                uncertainty_penalty = uncertainty_lambda * float(x_uncertainty)
                if float(x_uncertainty) > uncertainty_threshold:
                    uncertainty_penalty += uncertainty_hard_penalty
            fit -= uncertainty_penalty

        lohc_probability = _lohc_probability_score(mol, x, x_uncertainty, h2wt, sa, mw, cfg_score)
        if lohc_probability_enabled:
            fit += lohc_probability_weight * lohc_probability

        fitness_list.append(fit)
        uncertainty_penalty_list.append(uncertainty_penalty)
        lohc_probability_list.append(lohc_probability)
        bio_sim_list.append(bio_sim)
        target_hit_list.append(target_hit)
        template_reward_list.append(template_reward)

    # Keep original scoring logic but remap to positive range.
    # Invalid/rejected molecules stay at 0.
    valid_fits = [f for f in fitness_list if f > -1e5 and np.isfinite(f)]
    if len(valid_fits) > 0:
        valid_min = min(valid_fits)
        shift = 1.0 - valid_min if valid_min <= 0 else 1.0
    else:
        shift = 1.0

    fitness_pos = []
    for fit in fitness_list:
        if fit <= -1e5 or not np.isfinite(fit):
            fitness_pos.append(0.0)
        else:
            fitness_pos.append(fit + shift)

    df_pop['raw_fitness'] = fitness_list
    df_pop['bayesian_fitness'] = fitness_list
    df_pop['uncertainty_penalty'] = uncertainty_penalty_list
    df_pop['lohc_probability_score'] = lohc_probability_list
    df_pop['fitness value'] = fitness_pos
    df_pop['bio_sim'] = bio_sim_list
    df_pop['target_template_hit'] = target_hit_list
    df_pop['target_template_score'] = template_reward_list
    df_pop = annotate_platform_families(df_pop)
    df_pop['fitness_evaluated'] = 1
    return df_pop
def get_random_frag():
    """
    draw a random fragment from a pool of fragments
    :return: frag: randomly drawn fragment

    author: laura koenig-mattern
    """
    frag_path = './fragments.csv'

    # for windows
    df_frag = pd.read_csv(frag_path, encoding='cp1252', sep=';')
    # for linux
    # df_frag = pd.read_csv(frag_path, encoding='utf-8', sep=';')

    rand_iloc = np.random.randint(0,df_frag.shape[0])
    rand_loc = df_frag.index[rand_iloc]

    smi = df_frag.loc[rand_loc, 'SMILES']
    frag = Chem.MolFromSmiles(smi)

    return frag

def get_random_frag_ins():
    """
    draw a random atom for atom insertion
    :return: frag: randomly drawn atom
    """

    frag_path = './fragments_insert_atom.csv'

    # for windows
    df_frag = pd.read_csv(frag_path, encoding='cp1252', sep=';')
    # for linux
    # df_frag = pd.read_csv(frag_path, encoding='utf-8', sep=';')

    rand_iloc = np.random.randint(0,df_frag.shape[0])
    rand_loc = df_frag.index[rand_iloc]

    smi = df_frag.loc[rand_loc, 'SMILES']
    frag = Chem.MolFromSmiles(smi)

    return frag

def get_random_frag_from_class():
    """
    randomly draws a class, and then, randomly draws a fragment from the class

    :return: frag: randomly drawn fragment

    author: laura koenig-mattern
    """

    frag_path = './fragments_group_wise.csv'
    # for windows
    df_frag = pd.read_csv(frag_path, encoding='cp1252', sep=';')
    # for linux
    # df_frag = pd.read_csv(frag_path, encoding='utf-8', sep=';')

    rand_col = np.random.randint(0, df_frag.shape[1])

    df_col = df_frag.iloc[:,rand_col].dropna()

    rand_iloc = np.random.randint(0,df_col.shape[0])
    rand_loc = df_col.index[rand_iloc]

    smi = df_col.loc[rand_loc]
    frag = Chem.MolFromSmiles(smi)

    return frag

def bond_addition(mol):
    """
    adds a new bond between randomly chosen atoms

    :param mol: rdkit mol obk of mol to be mutated
    :return: add_bond_mol: rdkit mol obj of mutated mol, flag_successful: 0 - not successful, 1 - successful

    author: laura koenig-mattern
    """
    flag_successful = 0
    atom_idx_list, atom_symbol_list, valence_list = get_valences(mol)

    # molecule must have at least two atoms to be able to create a bond
    if len(atom_idx_list) <= 1:
        flag_successful = 0
        add_mol_bond = mol
        return add_mol_bond, flag_successful

    # check valences to see which atoms are able to create bonds.
    eligable_atom_idx_list = []
    eligable_valence_list = []
    is_aromatic_list = []
    for num, valence in enumerate(valence_list):
        if mol.GetAtomWithIdx(atom_idx_list[num]).GetIsAromatic():
            is_aromatic_list.append(atom_idx_list[num])
        if valence >= 1:
            eligable_atom_idx_list.append(atom_idx_list[num])
            eligable_valence_list.append(valence)

    adj_matrix = Chem.GetAdjacencyMatrix(mol)

    # fill values in adj matrix where both atoms are aromatic with other value than zero or one
    # use is aromatic list
    for idx in is_aromatic_list:
        adj_matrix[idx][is_aromatic_list] = 2

    # fill diagonal values of adjacency matrix with other value than zero or one
    np.fill_diagonal(adj_matrix, 2)
    eligable_adj_matrix = adj_matrix[eligable_atom_idx_list,:][:,eligable_atom_idx_list]


    # find all zero arguments
    eligable_cons = np.argwhere(eligable_adj_matrix == 0)
    if eligable_cons.shape[0] == 0:
        add_mol_bond = mol
        flag_successful = 0
        return add_mol_bond, flag_successful

    # draw a random number to choose which bond to create
    rand_num = np.random.randint(eligable_cons.shape[0])
    con = eligable_cons[rand_num]
    atm_idx_0 = eligable_atom_idx_list[con[0]]
    val_0 = valence_list[atm_idx_0]
    atm_idx_1 = eligable_atom_idx_list[con[1]]
    val_1 = valence_list[atm_idx_1]

    # choose type of bond and add bond
    rw_mol = Chem.RWMol(mol)
    try:
        Chem.Kekulize(rw_mol, clearAromaticFlags=True)
    except Exception:
        add_mol_bond = mol
        flag_successful = 0
        return add_mol_bond, flag_successful

    if val_0 == 1 or val_1 == 1:
        rw_mol.AddBond(atm_idx_0, atm_idx_1, order=Chem.BondType.SINGLE)
    if val_0 > 1 and val_1 > 1:
        rand_bond_int = np.random.randint(2)
        if rand_bond_int == 0:
            rw_mol.AddBond(atm_idx_0, atm_idx_1, order=Chem.BondType.SINGLE)
        if rand_bond_int == 1:
            rw_mol.AddBond(atm_idx_0, atm_idx_1, order=Chem.BondType.DOUBLE)

    add_bond_mol = rw_mol.GetMol()
    try:
        add_bond_mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(add_bond_mol)
        flag_successful = 1
    except Exception:
        # Invalid valence/bonding candidate: reject this mutation and keep original.
        add_mol_bond = mol
        flag_successful = 0
        return add_mol_bond, flag_successful

    return add_bond_mol, flag_successful

def bond_deletion(mol):
    """
    delete existing bond without splitting the molecule

    :param mol: rdkit mol object of mol for mutation
    :return: del_bond_mol: rdkit mol object of mutated mol, flag_successful: 0 - not successful, 1 - successful

    author: laura koenig-mattern
    """
    flag_successful = 0

    num_bonds = mol.GetNumBonds()
    if num_bonds < 1:
        del_bond_mol = mol
        flag_successful = 0
        return del_bond_mol, flag_successful

    # get bonds
    bonds = mol.GetBonds()

    # get bridges
    adj_matrix = Chem.GetAdjacencyMatrix(mol)
    graph = nx.Graph(adj_matrix)
    bridge_atoms = [i for i in nx.bridges(graph)]

    bridges = []
    for i in bridge_atoms:
        atm_idx_0 = i[0]
        atm_idx_1 = i[1]
        bond_idx_curr = mol.GetBondBetweenAtoms(atm_idx_0, atm_idx_1).GetIdx()
        bridges.append(bond_idx_curr)


    erasable_bonds = []
    for bond_idx in range(num_bonds):
        if bond_idx not in bridges:
            erasable_bonds.append(bond_idx)

    if len(erasable_bonds) == 0:
        del_bond_mol = mol
        flag_successful = 0
        return del_bond_mol, flag_successful

    # randomly choose bond to delete
    rand_num = np.random.randint(len(erasable_bonds))
    erase_bond_idx = erasable_bonds[rand_num]
    atm_idx_0 = bonds[erase_bond_idx].GetBeginAtomIdx()
    atm_idx_1 = bonds[erase_bond_idx].GetEndAtomIdx()

    rw_mol = Chem.RWMol(mol)
    Chem.Kekulize(rw_mol, clearAromaticFlags=True)

    rw_mol.RemoveBond(atm_idx_0, atm_idx_1)

    del_bond_mol = rw_mol.GetMol()
    del_bond_mol.UpdatePropertyCache()
    Chem.SanitizeMol(del_bond_mol)

    flag_successful = 1

    return del_bond_mol, flag_successful

def  substitution(mol):
    """
    substitute atom by other atom

    :param mol: rdkit mol obj of mol for mutation
    :return: subs_mol: rdkit mol obj of mutated mol, flag_successful: 0 - not successful, 1 - successful

    author: laura koenig-mattern
    """
    copy_mol = copy.deepcopy(mol)
    flag_successful = 0

    allowed_atomic_nums = sorted(_get_biobased_allowed_atomic_nums())
    base_subs = [('C', 4, 6), ('O', 2, 8), ('N', 3, 7), ('S', 6, 16), ('P', 5, 15)]
    filtered_subs = [row for row in base_subs if row[2] in allowed_atomic_nums]
    if len(filtered_subs) == 0:
        return mol, 0
    symbol_subs = [row[0] for row in filtered_subs]
    subs_valences = [row[1] for row in filtered_subs]
    atomic_nums = [row[2] for row in filtered_subs]

    # get atoms and explicit valences
    atoms = copy_mol.GetAtoms()
    atom_idx_list, atom_symbol_list, valence_list = get_explicit_valences(copy_mol)


    # try maximum 10 times
    count = 0
    for tries in range(10):
        count = count + 1
        # pick random atom from mol to be substituted
        rand_num = np.random.randint(len(atom_symbol_list))

        rand_idx = atom_idx_list[rand_num]
        atom = atoms[rand_idx]
        symbol = atom_symbol_list[rand_idx]
        valence = valence_list[rand_idx]

        # get possible subs
        possible_subs_idx = []
        for i in range(len(symbol_subs)):
            if symbol_subs[i] != symbol and subs_valences[i] >= valence:
                possible_subs_idx.append(i)

        if len(possible_subs_idx) == 0:
            continue

        # chose randomly from possible subs
        rand_int = np.random.randint(len(possible_subs_idx))
        subs_idx = possible_subs_idx[rand_int]
        atomic_num = atomic_nums[subs_idx]

        Chem.Kekulize(copy_mol, clearAromaticFlags=True)

        atom.SetAtomicNum(atomic_num)
        subs_mol = copy_mol

        try:
            subs_mol.UpdatePropertyCache()
            Chem.SanitizeMol(subs_mol)
            flag_successful = 1
        except:
            flag_successful = 0
            #import ipdb; ipdb.set_trace()
            break

        if flag_successful == 1:
            break

    if flag_successful == 0:
        subs_mol = mol

    return subs_mol, flag_successful

def get_neighbor_bonds(mol, atm_idx):
    """
    get bond idxs of neighboring bonds of an atom

    :param mol: rdkit mol obj
    :param atm_idx: atom idx of mol
    :return: bond_idxs: neighbor bond indices

    author: laura koenig-mattern
    """
    atom = mol.GetAtomWithIdx(atm_idx)
    bond_idxs = []
    for nbr in atom.GetNeighbors():
        bond_idxs.append(mol.GetBondBetweenAtoms(atm_idx, nbr.GetIdx()).GetIdx())
    return bond_idxs

def split_ring(mol):
    """
    delete a randomly chosen bond to split a ring.
    if several rings: randomly chose ring. predetermine deletable bonds.


    :param mol: rdkit mol obj of mol with ring to be split
    :return: frag_mol: rdkit mol obj with split ring

    author: laura koenig-mattern
    """
    flag_successful = 0

    ri = mol.GetRingInfo()
    bds_ring = []

    # get rings
    for ring in ri.BondRings():
        bds = set(ring)
        bds_ring.append(bds)

    # chose one ring
    rand_int = np.random.randint(len(bds_ring))
    this_ring = bds_ring[rand_int]

    intersect_bonds = []
    for i, ring in enumerate(bds_ring):
        if i != rand_int:
            intersect_bonds.append(this_ring.intersection(set(ring)))

    # bonds that are shared by multiple rings do not split the graph if deleted. therefore exclude
    eligable_bonds = list(this_ring)

    if len(intersect_bonds) > 0:
        for idx in this_ring:
            for bnd_set in intersect_bonds:
                if idx in bnd_set:
                    if idx in eligable_bonds:
                        eligable_bonds.remove(idx)

    if len(eligable_bonds) > 0:
        # randomly choose first bond
        bond_del_idx_0 = eligable_bonds[np.random.randint(len(eligable_bonds))]
        atm_idx_0 = mol.GetBondWithIdx(bond_del_idx_0).GetBeginAtomIdx()
        atm_idx_1 = mol.GetBondWithIdx(bond_del_idx_0).GetEndAtomIdx()

        # deleting neighbor bonds does not cut the ring into frags. Gets also non-ring-neighbors,
        # but that doesnt matter since we dont care about them when we want to split rings
        neighbor_bonds_0 = get_neighbor_bonds(mol, atm_idx_0)
        neighbor_bonds_1 = get_neighbor_bonds(mol, atm_idx_1)

        non_deletable = set(neighbor_bonds_0 + neighbor_bonds_1)

        eligable_bonds = list(set(eligable_bonds).difference(non_deletable))

    else:
        frag_mol = mol
        flag_successful = 0

    if len(eligable_bonds) > 0:
        bond_del_idx_1 = eligable_bonds[np.random.randint(len(eligable_bonds))]
        atm_idx_2 = mol.GetBondWithIdx(bond_del_idx_1).GetBeginAtomIdx()
        atm_idx_3 = mol.GetBondWithIdx(bond_del_idx_1).GetEndAtomIdx()

        copy_mol = copy.deepcopy(mol)
        Chem.Kekulize(copy_mol, clearAromaticFlags=True)

        rw_newmol = Chem.RWMol(copy_mol)
        rw_newmol.RemoveBond(atm_idx_0, atm_idx_1)
        rw_newmol.RemoveBond(atm_idx_2, atm_idx_3)

        frag_mol = rw_newmol.GetMol()

        if len(Chem.GetMolFrags(frag_mol)) > 1:
            flag_successful = 1
            frag_mol.UpdatePropertyCache()
            Chem.SanitizeMol(frag_mol)


        else:
            flag_successful = 0
            frag_mol = mol
    else:
        frag_mol = mol
        flag_successful = 0

    return frag_mol, flag_successful

def mutation_mol_constrainer(group_constraints, max_mol_weight, flag_successful, mol, mut_mol, SA_max, required_platform_id=None, preserve_platform=False):
    if flag_successful == 1:
        if _is_unreasonable_molecule(mol):
            flag_successful = 0
            mol = mut_mol
        if not _contains_only_allowed_elements(mol):
            flag_successful = 0
            mol = mut_mol
        if preserve_platform and not _platform_core_is_retained(mol, required_platform_id):
            flag_successful = 0
            mol = mut_mol
        mol_wt = Chem.Descriptors.MolWt(mol)
        if mol_wt > max_mol_weight:
            flag_successful = 0
            mol = mut_mol
        if SA_max:
            SA_score = SA.calculateScore(mol)
            if SA_score > SA_max:
                flag_successful = 0
                mol = mut_mol

    return flag_successful, mol

def mol_cleaner(mol):
    smi = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smi)
    mol.UpdatePropertyCache()
    Chem.SanitizeMol(mol)
    return mol

def _apply_mutation_operator(
    operator_idx,
    mut_mol_copy,
    mut_mol,
    group_constraints,
    max_mol_weight,
    SA_max,
    required_platform_id,
    preserve_platform,
):
    if operator_idx == 0:
        mol, flag_successful = addition(mut_mol_copy, get_random_frag_from_class())
    elif operator_idx == 1:
        mol, flag_successful = relocation(mut_mol_copy)
    elif operator_idx == 2:
        mol, flag_successful = insertion(mut_mol_copy, get_random_frag_ins())
    elif operator_idx == 3:
        mol, flag_successful = deletion(mut_mol_copy)
    elif operator_idx == 4:
        mol, flag_successful = bond_mutation(mut_mol_copy)
    elif operator_idx == 5:
        mol, flag_successful = bond_addition(mut_mol_copy)
    elif operator_idx == 6:
        mol, flag_successful = bond_deletion(mut_mol_copy)
    elif operator_idx == 7:
        mol, flag_successful = substitution(mut_mol_copy)
    else:
        return mut_mol, 0

    flag_successful, mol = mutation_mol_constrainer(
        group_constraints,
        max_mol_weight,
        flag_successful,
        mol,
        mut_mol,
        SA_max,
        required_platform_id,
        preserve_platform,
    )
    return mol, flag_successful


def mutation(
    df_pop,
    mutation_rate,
    max_mol_weight,
    group_constraints,
    SA_max,
    preserve_platform=False,
    operator_policy=None,
    generation=0,
    force_mutation=False,
    screening_config=None,
):
    """Apply one valid molecular edit and retain operator-level provenance."""
    if df_pop is None or df_pop.shape[0] == 0:
        return df_pop

    metadata_defaults = {
        'mutation_operator': '',
        'operator_probability': np.nan,
        'experience_context': '',
        'origin_type': '',
    }
    for column, default in metadata_defaults.items():
        if column not in df_pop.columns:
            df_pop[column] = default

    smi_col_idx = df_pop.columns.get_loc('smi')
    mol_col_idx = df_pop.columns.get_loc('mol')
    platform_col_idx = df_pop.columns.get_loc('platform_id') if 'platform_id' in df_pop.columns else None
    cfg = screening_config or {}

    for iloc_idx in range(df_pop.shape[0]):
        if not force_mutation and np.random.random_sample() > mutation_rate:
            continue

        mut_mol = df_pop.iloc[iloc_idx, mol_col_idx]
        if mut_mol is None:
            continue
        parent_smi = str(df_pop.iloc[iloc_idx, smi_col_idx])
        required_platform_id = (
            'unassigned' if platform_col_idx is None else str(df_pop.iloc[iloc_idx, platform_col_idx])
        )
        h2_state, dh_state, sa_state = classify_objective_state(df_pop.iloc[iloc_idx], cfg)
        exp_context = context_key(required_platform_id, h2_state, dh_state, sa_state)

        if operator_policy is None:
            operator_order = np.random.permutation(len(MUTATION_OPERATORS)).tolist()
            probabilities = {name: 1.0 / len(MUTATION_OPERATORS) for name in MUTATION_OPERATORS}
        else:
            operator_order, probabilities = operator_policy.choose_order(exp_context, int(generation), parent_smi=parent_smi)

        for operator_idx in operator_order:
            mut_mol_copy = copy.deepcopy(mut_mol)
            mol, flag_successful = _apply_mutation_operator(
                int(operator_idx),
                mut_mol_copy,
                mut_mol,
                group_constraints,
                max_mol_weight,
                SA_max,
                required_platform_id,
                preserve_platform,
            )
            if flag_successful != 1:
                continue
            smi = Chem.MolToSmiles(mol)
            if smi == parent_smi:
                continue
            clean_mol = Chem.MolFromSmiles(smi)
            if clean_mol is None:
                continue
            operator_name = MUTATION_OPERATORS[int(operator_idx)]
            df_pop.iat[iloc_idx, smi_col_idx] = smi
            df_pop.iat[iloc_idx, mol_col_idx] = clean_mol
            df_pop.at[df_pop.index[iloc_idx], 'mutation_operator'] = operator_name
            df_pop.at[df_pop.index[iloc_idx], 'operator_probability'] = probabilities[operator_name]
            df_pop.at[df_pop.index[iloc_idx], 'experience_context'] = exp_context
            df_pop.at[df_pop.index[iloc_idx], 'origin_type'] = 'guided_mutation'
            break

    return df_pop


_CHILD_PROPERTY_COLUMNS = [
    'SA score', 'sat_smi', 'full_sat_smi', 'delta_H', 'delta_H_uncertainty',
    'delta_H_full', 'delta_H_full_uncertainty', 'delta_H_diff',
    'delta_H_per_H2', 'h2_mol', 'h2_wt_percent',
    'fitness value', 'bayesian_fitness', 'uncertainty_penalty',
    'lohc_probability_score', 'bio_sim', 'target_template_hit', 'target_template_score',
    'molwt_screen', 'pass_fast_prefilter', 'sa_screen', 'hba', 'hbd',
    'tpsa', 'logp', 'pains_flag', 'obj_h2_capacity', 'obj_dh_window',
    'obj_sa', 'obj_bio', 'obj_template', 'pareto_rank',
    'crowding_distance', 'selection_score', 'selected_pareto',
    'selected_hardfilter', 'selected_multifilter', 'pareto_feasible',
    'pareto_objective_sum', 'is_pareto', 'fitness_evaluated', 'raw_fitness', 'scaffold_smi',
]


def generate_guided_mutation_offspring(
    df_pop,
    num_offspring,
    max_mol_weight,
    SA_max,
    preserve_platform,
    operator_policy,
    generation,
    screening_config,
):
    """Generate directly attributable parent-child mutation pairs."""
    if df_pop is None or df_pop.empty or int(num_offspring) <= 0:
        return pd.DataFrame(columns=df_pop.columns if df_pop is not None else [])

    parent_indices = selection(df_pop, int(num_offspring))
    mutants = df_pop.iloc[parent_indices].copy().reset_index(drop=True)
    mutants['parent_smi'] = mutants['smi'].astype(str)
    mutants['parent_platform_id'] = mutants.get(
        'platform_id', pd.Series('unassigned', index=mutants.index)
    ).astype(str)
    mutants['parent_h2_wt_percent'] = pd.to_numeric(
        mutants.get('h2_wt_percent', np.nan), errors='coerce'
    )
    mutants['parent_delta_H_per_H2'] = pd.to_numeric(
        mutants.get('delta_H_per_H2', np.nan), errors='coerce'
    )
    mutants['parent_SA_score'] = pd.to_numeric(
        mutants.get('SA score', np.nan), errors='coerce'
    )
    parent_selected = mutants['selected_pareto'] if 'selected_pareto' in mutants.columns else pd.Series(0, index=mutants.index)
    mutants['parent_selected_pareto'] = pd.to_numeric(
        parent_selected, errors='coerce'
    ).fillna(0).astype(int)

    mutants = mutation(
        mutants,
        mutation_rate=1.0,
        max_mol_weight=max_mol_weight,
        group_constraints=None,
        SA_max=SA_max,
        preserve_platform=preserve_platform,
        operator_policy=operator_policy,
        generation=generation,
        force_mutation=True,
        screening_config=screening_config,
    )
    mutants = mutants[
        mutants['mutation_operator'].astype(str).ne('')
        & mutants['smi'].astype(str).ne(mutants['parent_smi'].astype(str))
    ].copy()
    mutants = mutants.drop_duplicates(subset=['smi'], keep='first').reset_index(drop=True)

    # Parent properties must never leak into prediction/scoring of the edited child.
    for column in _CHILD_PROPERTY_COLUMNS:
        if column in mutants.columns:
            mutants[column] = np.nan
    return mutants

def get_all_time_fittest(df_fittest, df_pop, num_fittest):
    df_fittest = pd.concat([df_fittest, df_pop], ignore_index=True)
    df_fittest = df_fittest.drop_duplicates(subset=['smi'])
    df_fittest = df_fittest.sort_values(by=['fitness value'], ascending=False)
    df_fittest = df_fittest.head(num_fittest)
    return df_fittest

def _compute_diversity_metrics(df_pop, max_pairs=4000):
    n = max(1, df_pop.shape[0])
    smi_series = df_pop['smi'].astype(str) if 'smi' in df_pop.columns else pd.Series([], dtype=str)
    unique_ratio = smi_series.nunique() / float(n)

    mols = df_pop['mol'].tolist() if 'mol' in df_pop.columns else []
    valid_mols = [m for m in mols if m is not None]

    # Morgan fingerprint based average pairwise Tanimoto distance
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in valid_mols]
    avg_tanimoto_dist = np.nan
    if len(fps) >= 2:
        num_fp = len(fps)
        max_pairs_all = num_fp * (num_fp - 1) // 2
        sampled_pairs = min(max_pairs, max_pairs_all)
        if sampled_pairs == max_pairs_all:
            dists = []
            for i in range(num_fp - 1):
                sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
                dists.extend([1.0 - s for s in sims])
            avg_tanimoto_dist = float(np.mean(dists)) if len(dists) > 0 else np.nan
        else:
            rng = np.random.default_rng(42)
            dists = []
            for _ in range(sampled_pairs):
                i, j = rng.choice(num_fp, size=2, replace=False)
                dists.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
            avg_tanimoto_dist = float(np.mean(dists)) if len(dists) > 0 else np.nan

    # Bemis-Murcko scaffold diversity
    scaffold_div = np.nan
    if len(valid_mols) > 0:
        scaffolds = []
        for m in valid_mols:
            try:
                scaffolds.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
            except Exception:
                continue
        if len(scaffolds) > 0:
            scaffold_div = len(set(scaffolds)) / float(len(scaffolds))

    # Descriptor variance as genotype/chemotype diversity proxy
    desc_var = np.nan
    if len(valid_mols) > 1:
        desc_mat = np.array([
            [
                Descriptors.MolWt(m),
                Descriptors.MolLogP(m),
                Descriptors.TPSA(m),
                Descriptors.NumHDonors(m),
                Descriptors.NumHAcceptors(m)
            ]
            for m in valid_mols
        ], dtype=np.float64)
        if desc_mat.shape[0] > 1:
            std = np.std(desc_mat, axis=0)
            mean_abs = np.mean(np.abs(desc_mat), axis=0) + 1e-8
            norm_std = std / mean_abs
            desc_var = float(np.mean(norm_std))

    return {
        'unique ratio': unique_ratio,
        'avg tanimoto dist': avg_tanimoto_dist,
        'scaffold diversity': scaffold_div,
        'descriptor variance': desc_var
    }

def _append_property_snapshot(save_path, df_pop, generation):
    snapshot_cols = ['h2_wt_percent', 'delta_H_per_H2', 'delta_H_uncertainty', 'delta_H_diff', 'SA score', 'lohc_probability_score', 'fitness value']
    available_cols = [c for c in snapshot_cols if c in df_pop.columns]
    if len(available_cols) == 0:
        return
    df_snap = df_pop[available_cols].copy()
    df_snap['generation'] = generation
    out_path = save_path + 'property_snapshots.csv'
    if os.path.exists(out_path):
        df_old = pd.read_csv(out_path)
        df_new = pd.concat([df_old, df_snap], ignore_index=True)
        _safe_to_csv(df_new, out_path, index=False)
    else:
        _safe_to_csv(df_snap, out_path, index=False)

def _safe_to_csv(df, out_path, **kwargs):
    try:
        df.to_csv(out_path, **kwargs)
        return out_path
    except PermissionError:
        root, ext = os.path.splitext(out_path)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{root}_locked_{stamp}{ext}"
        df.to_csv(backup_path, **kwargs)
        print(f"[WARN] File is locked: {out_path}. Wrote backup to: {backup_path}")
        return backup_path


def run_select_redraw(
    input_all='./results/plot/all_generated_molecules.csv',
    input_pareto='./results/plot/pareto_candidates.csv',
    pareto_reps='./results/plot/pareto_cluster_representatives.csv',
    outdir='select',
):
    script_path = os.path.join('select', 'select_molecules.py')
    if not os.path.exists(script_path):
        return
    cmd = [
        sys.executable,
        script_path,
        '--input-all', input_all,
        '--input-pareto', input_pareto,
        '--pareto-representatives', pareto_reps,
        '--outdir', outdir,
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception as exc:
        print(f"[WARN] select redraw failed: {exc}")

def _pareto_front_indices_nd(objective_matrix):
    arr = np.asarray(objective_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.array([], dtype=int)
    valid = np.all(np.isfinite(arr), axis=1)
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return np.array([], dtype=int)
    objs = arr[idx]
    front = []
    for i in range(objs.shape[0]):
        dominated = False
        for j in range(objs.shape[0]):
            if i == j:
                continue
            if np.all(objs[j] >= objs[i]) and np.any(objs[j] > objs[i]):
                dominated = True
                break
        if not dominated:
            front.append(idx[i])
    return np.array(front, dtype=int)

def ini_statistics_frame():
    other_params = [
        'mean fitness',
        'best fitness',
        'prop mean',
        'unique ratio',
        'avg tanimoto dist',
        'scaffold diversity',
        'descriptor variance'
    ]
    col_names = other_params

    df_stats = pd.DataFrame(columns=col_names)

    return df_stats

def ini_group_frame():
    rd_groups = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH',
                 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1',
                 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
                 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
                 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
                 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan',
                 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
                 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
                 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
                 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
                 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
                 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene',
                 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

    col_names = rd_groups

    df_groups = pd.DataFrame(columns=col_names)

    return df_groups

def get_stats(df_stats, df_pop):

    stat_dict = dict.fromkeys([
        'mean fitness',
        'best fitness',
        'prop mean',
        'unique ratio',
        'avg tanimoto dist',
        'scaffold diversity',
        'descriptor variance'
    ])

    fit_col = 'selection_score' if 'selection_score' in df_pop.columns else 'fitness value'
    mean_fit = pd.to_numeric(df_pop[fit_col], errors='coerce').mean()
    best_fit_curr = pd.to_numeric(df_pop[fit_col], errors='coerce').max()
    if df_stats.shape[0] > 0 and 'best fitness' in df_stats.columns:
        prev_best = pd.to_numeric(df_stats['best fitness'], errors='coerce').max()
        if np.isfinite(prev_best):
            best_fit = max(best_fit_curr, prev_best)
        else:
            best_fit = best_fit_curr
    else:
        best_fit = best_fit_curr
    mean_prop = df_pop['prop'].mean()
    diversity = _compute_diversity_metrics(df_pop)

    stat_dict['mean fitness'] = [mean_fit]
    stat_dict['best fitness'] = [best_fit]
    stat_dict['prop mean'] = [mean_prop]
    stat_dict['unique ratio'] = [diversity['unique ratio']]
    stat_dict['avg tanimoto dist'] = [diversity['avg tanimoto dist']]
    stat_dict['scaffold diversity'] = [diversity['scaffold diversity']]
    stat_dict['descriptor variance'] = [diversity['descriptor variance']]

    df_app = pd.DataFrame.from_dict(stat_dict)

    df_stats = pd.concat([df_stats, df_app], ignore_index=True)

    return df_stats

def export_realtime_diversity_data(df_stats, frame_path='./results/frames/', plot_data_path='./results/plot/'):
    if df_stats is None or df_stats.shape[0] == 0:
        return
    if not os.path.exists(frame_path):
        os.makedirs(frame_path)
    if not os.path.exists(plot_data_path):
        os.makedirs(plot_data_path)

    _safe_to_csv(df_stats, frame_path + 'stats.csv', index=False)

    needed = ['unique ratio', 'avg tanimoto dist', 'scaffold diversity', 'descriptor variance']
    if all(c in df_stats.columns for c in needed):
        gen_num = range(1, df_stats.shape[0] + 1)
        df_div = pd.DataFrame({
            'generation': list(gen_num),
            'unique ratio': pd.to_numeric(df_stats['unique ratio'], errors='coerce').to_list(),
            'avg tanimoto dist': pd.to_numeric(df_stats['avg tanimoto dist'], errors='coerce').to_list(),
            'scaffold diversity': pd.to_numeric(df_stats['scaffold diversity'], errors='coerce').to_list(),
            'descriptor variance': pd.to_numeric(df_stats['descriptor variance'], errors='coerce').to_list()
        })
        _safe_to_csv(df_div, plot_data_path + 'diversity_metrics.csv', index=False)

def get_group_counts(df_groups, df_pop):
    rd_groups = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH',
                 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1',
                 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
                 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
                 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
                 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan',
                 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
                 'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
                 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
                 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
                 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd',
                 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene',
                 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

    group_dict = dict.fromkeys(rd_groups)
    for i in range(df_pop.shape[0]):
        locInd = df_pop.index[i]
        mol = df_pop.loc[locInd, 'mol']
        smi = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smi)  # makes sure that Mol is sanitized

        for ii in rd_groups:
            func = getattr(Chem.Fragments, ii)
            num_groups_curr = func(mol)
            num_groups = group_dict[ii]

            if num_groups is None:
                num_groups = 0
            if isinstance(num_groups, list):
                num_groups = num_groups[0]


            tot_groups = num_groups + num_groups_curr


            group_dict[ii] = [tot_groups]

    df_app = pd.DataFrame.from_dict(group_dict)

    df_groups = pd.concat([df_groups, df_app], ignore_index=True)
    return df_groups

def post_process(
    df_pop,
    df_fittest,
    df_archive,
    df_stats,
    df_groups,
    save_all_pops_flag,
    gen,
    screening_config=None,
    pareto_export_rows=80000,
    output_root='./results',
    run_selection_redraw=True,
):

    ##### save frames #####
    save_path = os.path.join(output_root, 'frames') + os.sep
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if save_all_pops_flag == 0:
        _safe_to_csv(df_pop, save_path + 'last_pop.csv')
    if save_all_pops_flag == 1:
        _safe_to_csv(df_pop, save_path + 'last_pop_' + str(gen) + '.csv')
    _safe_to_csv(df_fittest, save_path + 'fittest.csv')
    _safe_to_csv(dedupe_and_sort_population(df_archive), save_path + 'all_generated_molecules.csv')
    _safe_to_csv(df_stats, save_path + 'stats.csv')
    _safe_to_csv(df_groups, save_path + 'group_counts.csv')
    _append_property_snapshot(save_path, df_pop, generation=(gen + 1))

    # Only keep diversity-vs-generation plotting.
    # All other plotting outputs are disabled; their plot data are exported to ./results/plot/.
    main_fig_path = os.path.join(output_root, 'figures') + os.sep
    if not os.path.exists(main_fig_path):
        os.makedirs(main_fig_path)
    plot_data_path = os.path.join(output_root, 'plot') + os.sep
    if not os.path.exists(plot_data_path):
        os.makedirs(plot_data_path)

    gen_num = range(1,df_stats.shape[0]+1)
    mean_fitness = df_stats['mean fitness'].to_list()
    best_fitness = df_stats['best fitness'].to_list()
    mean_prop = df_stats['prop mean'].to_list()

    # Save generation-wise history for post-analysis (no figure output).
    df_fitness = pd.DataFrame({
        'generation': list(gen_num),
        'mean fitness': mean_fitness,
        'best fitness': best_fitness,
        'mean prop': mean_prop
    })
    _safe_to_csv(df_fitness, plot_data_path + 'fitness.csv', index=False)

    # Export diversity data as tables only.
    if all(c in df_stats.columns for c in ['unique ratio', 'avg tanimoto dist', 'scaffold diversity', 'descriptor variance']):
        df_div = pd.DataFrame({
            'generation': list(gen_num),
            'unique ratio': pd.to_numeric(df_stats['unique ratio'], errors='coerce').to_list(),
            'avg tanimoto dist': pd.to_numeric(df_stats['avg tanimoto dist'], errors='coerce').to_list(),
            'scaffold diversity': pd.to_numeric(df_stats['scaffold diversity'], errors='coerce').to_list(),
            'descriptor variance': pd.to_numeric(df_stats['descriptor variance'], errors='coerce').to_list()
        })
        _safe_to_csv(df_div, plot_data_path + 'diversity_metrics.csv', index=False)

    # Export generation snapshots (for external plotting).
    snap_src = save_path + 'property_snapshots.csv'
    if os.path.exists(snap_src):
        try:
            df_snap = pd.read_csv(snap_src)
            _safe_to_csv(df_snap, plot_data_path + 'property_snapshots.csv', index=False)
        except Exception:
            pass

    try:
        export_selection_workflow(
            df_archive=df_archive,
            df_selected=df_pop,
            out_dir=plot_data_path,
            fig_dir=main_fig_path,
            config=screening_config,
            max_points=pareto_export_rows
        )
    except Exception:
        pass

    if run_selection_redraw:
        run_select_redraw(
            input_all=os.path.join(plot_data_path, 'all_generated_molecules.csv'),
            input_pareto=os.path.join(plot_data_path, 'pareto_candidates.csv'),
            pareto_reps=os.path.join(plot_data_path, 'pareto_cluster_representatives.csv'),
            outdir=os.path.join(output_root, 'select'),
        )

    # Export functional-group counts table only (no frequency figures).
    _safe_to_csv(df_groups, plot_data_path + 'group_counts.csv', index=False)
    return

def run_simplified_workflow(cfg):
    start_time = time.time()
    global GA_FITNESS_CONFIG
    GA_FITNESS_CONFIG = dict(cfg)
    seed = int(cfg.get('random_seed', 42))
    np.random.seed(seed)

    pop_size = int(cfg['pop_size'])
    screening_config = cfg['survivor_screening']
    runtime_postprocess = bool(cfg.get('runtime_postprocess', False))
    runtime_stats_export = bool(cfg.get('runtime_stats_export', False))
    output_root = os.path.abspath(str(cfg.get('adaptive_output_dir', 'results/adaptive_ga/default')))
    os.makedirs(output_root, exist_ok=True)
    transition_path = os.path.join(output_root, 'mutation_transitions.csv')
    metrics_path = os.path.join(output_root, 'exploration_metrics.csv')
    history_path = os.path.join(output_root, 'operator_experience_history.csv')
    timings_path = os.path.join(output_root, 'stage_timings.csv')
    if any(os.path.exists(path) for path in (transition_path, metrics_path, history_path)):
        raise FileExistsError(
            f'Adaptive GA run directory already contains source data: {output_root}. '
            'Use a new --run-label to preserve existing results.'
        )

    operator_policy = OperatorExperiencePolicy(
        mode=str(cfg.get('adaptive_mode', 'ucb')),
        beta=float(cfg.get('operator_ucb_beta', 0.8)),
        random_fraction=float(cfg.get('operator_random_fraction', 0.10)),
        warmup_generations=int(cfg.get('operator_warmup_generations', 5)),
        temperature=float(cfg.get('operator_temperature', 0.35)),
        seed=seed,
    )
    tracker = ExplorationTracker(cfg)

    df_pop = generate_random_start_pop(
        pop_size=pop_size,
        SA_max=float(cfg['sa_max_generation']),
        max_mol_weight=float(cfg['max_mol_weight']),
    )
    df_pop['discovery_generation'] = 0
    df_pop['origin_type'] = 'initial_seed'
    df_pop['operation_generation'] = 0
    df_pop = annotate_pareto_population(df_pop, screening_config)
    df_pop = apply_frontier_parent_scores(df_pop, cfg)
    df_pop['scaffold_smi'] = [_safe_murcko_scaffold_smiles(mol) for mol in df_pop['mol'].tolist()]
    initial_metrics = tracker.update(df_pop, generation=0)
    append_csv(pd.DataFrame([initial_metrics]), metrics_path)

    df_fittest = df_pop.copy()
    df_archive = _prepare_archive_batch(df_pop, df_pop)
    df_stats = ini_statistics_frame()
    df_groups = ini_group_frame()

    seed_pool = get_biobased_seed_pool(float(cfg['sa_max_generation']), float(cfg['max_mol_weight']))
    if seed_pool.shape[0] == 0:
        seed_pool = df_pop[['smi']].drop_duplicates().copy()

    best_so_far = -np.inf
    stagnation_count = 0
    max_generations = int(cfg['generations'])
    target_archive_size = int(cfg.get('target_archive_size', 0))
    switch_archive_size = int(cfg.get('switch_archive_size', 10000))
    min_candidate_pool_size = int(cfg.get('min_candidate_pool_size', 20000))
    max_candidate_pool_size = int(cfg.get('max_candidate_pool_size', 100000))
    archive_oversample_factor = float(cfg.get('archive_oversample_factor', 2.0))
    saturation_reached = False

    gen = 0
    while gen < max_generations:
        archive_count = int(df_archive['smi'].nunique()) if df_archive is not None and 'smi' in df_archive.columns else 0
        if (
            bool(cfg.get('stop_at_archive_size', False))
            and target_archive_size > 0
            and archive_count >= target_archive_size
        ):
            print(f'target archive size reached: {archive_count}')
            break
        if (
            target_archive_size > 0
            and switch_archive_size <= archive_count < target_archive_size
        ):
            remaining = max(0, target_archive_size - archive_count)
            target_candidate_pool_size = int(np.ceil(remaining * archive_oversample_factor))
            target_candidate_pool_size = max(min_candidate_pool_size, target_candidate_pool_size)
            target_candidate_pool_size = min(max_candidate_pool_size, target_candidate_pool_size)
            ga_mode = 'archive-target'
        else:
            target_candidate_pool_size = None
            ga_mode = 'generation'

        generation_start = time.perf_counter()
        print(f'------ generation {gen + 1} is running ------')
        print(f'GA mode: {ga_mode}; archive unique: {archive_count}; target candidate pool: {target_candidate_pool_size or "default"}')
        print('generate candidate pool')
        candidate_start = time.perf_counter()
        df_candidates, df_elite, curr_mutation_rate = generate_candidate_pool(
            df_pop=df_pop,
            seed_pool=seed_pool,
            cfg=cfg,
            generation=gen,
            stagnation_count=stagnation_count,
            archive_smis=set(df_archive['smi'].astype(str).tolist()) if df_archive is not None and 'smi' in df_archive.columns else None,
            target_candidate_pool_size=target_candidate_pool_size,
            operator_policy=operator_policy,
        )
        candidate_unique = int(df_candidates['smi'].nunique()) if 'smi' in df_candidates.columns else df_candidates.shape[0]
        print(f'candidate unique: {candidate_unique}')
        print(f'current mutation rate: {curr_mutation_rate:.4f}')
        candidate_seconds = time.perf_counter() - candidate_start

        print('constrain and rank candidates')
        score_start = time.perf_counter()
        df_ranked = constrain_and_rank_candidates(
            df_candidates=df_candidates,
            df_elite=df_elite,
            screening_config=screening_config,
            compute_full_hydrogenation=bool(
                cfg.get('compute_full_hydrogenation', False)
            ),
            cfg=cfg,
        )
        if 'scaffold_smi' not in df_ranked.columns:
            df_ranked['scaffold_smi'] = ''
        scaffold_values = df_ranked['scaffold_smi'].fillna('').astype(str)
        missing_scaffold = scaffold_values.eq('')
        if missing_scaffold.any():
            df_ranked.loc[missing_scaffold, 'scaffold_smi'] = [
                _safe_murcko_scaffold_smiles(mol)
                for mol in df_ranked.loc[missing_scaffold, 'mol'].tolist()
            ]
        score_seconds = time.perf_counter() - score_start
        feedback_start = time.perf_counter()

        transitions = build_transition_records(
            df_ranked,
            generation=gen + 1,
            config=screening_config,
        )
        operator_policy.update(transitions)
        append_csv(transitions, transition_path)
        operator_summary = operator_policy.summary_frame()
        if not operator_summary.empty:
            operator_summary.to_csv(
                os.path.join(output_root, 'operator_experience.csv'),
                index=False,
            )
            history = operator_summary.copy()
            history.insert(0, 'generation', gen + 1)
            append_csv(history, history_path)
        operator_policy.save(output_root)

        generation_metrics = tracker.update(df_ranked, generation=gen + 1)
        append_csv(pd.DataFrame([generation_metrics]), metrics_path)
        print(
            'coverage={:.4f}; QD={:.4f}; target hits={}; new scaffolds={}'.format(
                generation_metrics.get('target_coverage', 0.0),
                generation_metrics.get('qd_score', 0.0),
                generation_metrics.get('generation_target_hits', 0),
                generation_metrics.get('new_unique_scaffolds', 0),
            )
        )

        feedback_seconds = time.perf_counter() - feedback_start
        df_stats_source = df_ranked.copy()
        archive_start = time.perf_counter()
        df_archive_batch = _prepare_archive_batch(df_candidates, df_ranked)
        df_archive = update_molecule_archive(df_archive, df_archive_batch)
        archive_seconds = time.perf_counter() - archive_start

        print('pareto survivor selection')
        survivor_start = time.perf_counter()
        df_ranked = dedupe_and_sort_population(df_ranked)
        df_ranked = apply_frontier_parent_scores(df_ranked, cfg)
        df_pop, _, _ = select_next_population(
            df_ranked=df_ranked,
            pop_size=pop_size,
            screening_config=screening_config,
            keep_core_ratio=float(cfg['keep_core_ratio']),
            keep_cluster_ratio=float(cfg['keep_cluster_ratio']),
            keep_platform_ratio=float(cfg['keep_platform_ratio']),
            keep_target_ratio=float(cfg.get('keep_qd_ratio', 0.0)) if bool(cfg.get('full_model_mode', False)) else 0.0,
            keep_bio_ratio=0.0,
            random_state=gen + 1,
            cfg=cfg,
        )
        survivor_seconds = time.perf_counter() - survivor_start

        df_pop = apply_frontier_parent_scores(df_pop, cfg)
        curr_best = pd.to_numeric(df_pop['selection_score'], errors='coerce').max()
        if curr_best > best_so_far + 1e-9:
            best_so_far = curr_best
            stagnation_count = 0
        else:
            stagnation_count += 1
        print(f'stagnation count: {stagnation_count}')

        df_pop, stagnation_count, did_refresh = refresh_population_if_needed(
            df_pop=df_pop,
            seed_pool=seed_pool,
            cfg=cfg,
            screening_config=screening_config,
            stagnation_count=stagnation_count,
            generation=gen,
        )
        if did_refresh:
            print('stagnation refresh applied')

        print('save fittest')
        df_fittest = get_all_time_fittest(df_fittest, df_pop, int(cfg['num_fittest']))

        if (gen + 1) % int(cfg['stats_every']) == 0:
            print('get stats')
            df_stats = get_stats(df_stats, df_stats_source)
            if runtime_stats_export:
                export_realtime_diversity_data(
                    df_stats,
                    frame_path=os.path.join(output_root, 'frames') + os.sep,
                    plot_data_path=os.path.join(output_root, 'plot') + os.sep,
                )

        if runtime_postprocess and (gen + 1) % int(cfg['postprocess_every']) == 0:
            print('cluster export')
            post_process(
                df_pop=df_pop,
                df_fittest=df_fittest,
                df_archive=df_archive,
                df_stats=df_stats,
                df_groups=df_groups,
                save_all_pops_flag=0,
                gen=gen,
                screening_config=screening_config,
                pareto_export_rows=int(cfg['pareto_export_rows']),
                output_root=output_root,
                run_selection_redraw=bool(cfg.get('run_selection_redraw', False)),
            )

        total_seconds = time.perf_counter() - generation_start
        timing_row = {
            'generation': gen + 1,
            'candidate_pool_seconds': candidate_seconds,
            'score_rank_seconds': score_seconds,
            'feedback_seconds': feedback_seconds,
            'archive_seconds': archive_seconds,
            'survivor_seconds': survivor_seconds,
            'total_seconds': total_seconds,
            'candidate_unique': candidate_unique,
            'pending_before_prefilter': int(df_ranked.attrs.get('pending_before_prefilter', 0)),
            'pending_after_prefilter': int(df_ranked.attrs.get('pending_after_prefilter', 0)),
            'prefilter_dropped': int(df_ranked.attrs.get('prefilter_dropped', 0)),
        }
        append_csv(pd.DataFrame([timing_row]), timings_path)
        print(
            'timing(s): candidates={:.1f}; score+GNN={:.1f}; feedback={:.1f}; '
            'archive={:.1f}; survivor={:.1f}; total={:.1f}'.format(
                candidate_seconds,
                score_seconds,
                feedback_seconds,
                archive_seconds,
                survivor_seconds,
                total_seconds,
            )
        )

        gen += 1
        if bool(cfg.get('saturation_enabled', True)) and tracker.is_saturated():
            saturation_reached = True
            print('target property-space convergence reached; stopping GA')
            break

    print('final export')
    post_process(
        df_pop=df_pop,
        df_fittest=df_fittest,
        df_archive=df_archive,
        df_stats=df_stats,
        df_groups=df_groups,
        save_all_pops_flag=0,
        gen=max(0, gen - 1),
        screening_config=screening_config,
        pareto_export_rows=int(cfg['pareto_export_rows']),
        output_root=output_root,
        run_selection_redraw=bool(cfg.get('run_selection_redraw', False)),
    )
    operator_policy.save(output_root)

    qd_rows = [
        {
            'platform_id': cell[0],
            'h2_bin': cell[1],
            'best_cell_quality': quality,
        }
        for cell, quality in tracker.target_cells.items()
    ]
    pd.DataFrame(qd_rows).to_csv(os.path.join(output_root, 'qd_archive.csv'), index=False)

    mins_passed = (time.time() - start_time) / 60.0
    final_metrics = dict(tracker.metrics[-1]) if tracker.metrics else {}
    final_metrics.update({
        'adaptive_mode': str(cfg.get('adaptive_mode', 'ucb')),
        'random_seed': seed,
        'generations_completed': gen,
        'saturation_reached': int(saturation_reached),
        'runtime_minutes': mins_passed,
    })
    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(output_root, 'run_summary.csv'),
        index=False,
    )
    return mins_passed