"""Blind, read-only verification of DeepSeek's AR/AL/P interpretation."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import random

from rdkit import Chem, RDLogger

from lohc_matrix_memory_trial import graph_identity
from lohc_matrix_protocol import BLOCKS, RULES, VERSION
from lohc_structure_score_trial import DeepSeek, credential, load, save


PROMPT = '''Read the supplied LOHC three-matrix record. All indices are zero-based. AR is rich, AL is lean. In AR and AL, atoms rows are [id,atomic_number,formal_charge,H_count,isotope,aromatic_flag]; bonds rows are [i,j,bond_order], i<j, with 1.5 aromatic. P.correspondence is one [lean_id,rich_id] per lean atom, ordered by lean_id. P.delta is the upper triangle of rich-minus-lean bond order after alignment, indexed by lean atom IDs. P.properties is [theoretical_H2_wt_percent,dehydrogenation_kJ_per_mol_H2,mapping_count]. Null is unknown. Do not calculate or round properties. No stereochemistry is supplied.
Return strict JSON only: {"lean_smiles":"...","rich_smiles":"...","properties":[0,0,0],"correspondence":[[0,0]],"delta":[[0,1,-0.5]]}. Reconstruct both complete molecular graphs as valid SMILES including substituent connectivity, charge and isotope if present. Copy all correspondence and delta rows exactly, retaining order and sign. Include every required field. The matrices are the only source; no literature, names or reference answers are supplied.'''
PROBE = ''' Also return "cross_matrix" as rows [lean_i,lean_j,rich_i,rich_j,lean_bond_order,rich_bond_order,delta] for the first, middle (index floor(length/2)), and last P.delta rows, in that order with repeated selections omitted. Keep lean_i<lean_j and rich_i<rich_j. Compute and verify the delta using the two state tables. The example in the protocol describes a different molecule.'''


def canonical(smi):
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def packet(entry, identifier, permuted):
    n = len(entry['AL'])
    order = list(range(n))
    if permuted:
        random.Random(f'LOHC:{identifier}').shuffle(order)
    index = {old: new for new, old in enumerate(order)}

    def state(matrix, atom_order):
        return {'atoms': [[i, *map(int, matrix[old][n:])] for i, old in enumerate(atom_order)],
                'bonds': [[i, j, matrix[a][b]] for i, a in enumerate(atom_order)
                          for j, b in enumerate(atom_order) if i < j and matrix[a][b]]}

    p = entry['P']
    correspondence = [[i, index[j]] for i in range(n) for j in range(n) if p[i][j] == 1]
    if len(correspondence) != n or len({j for _, j in correspondence}) != n:
        raise ValueError('Invalid correspondence block')
    return {'AR': state(entry['AR'], order), 'AL': state(entry['AL'], list(range(n))),
            'P': {'correspondence': correspondence, 'delta': [[i, j, p[i][n+j]] for i in range(n)
                  for j in range(i+1, n) if p[i][n+j]], 'properties': p[-1][-3:]}}


def cross_matrix(data):
    mapping = dict(data['P']['correspondence'])
    left, right = ({(i, j): order for i, j, order in data[k]['bonds']} for k in ('AL', 'AR'))
    edges = data['P']['delta']
    result = []
    for k in sorted({0, len(edges)//2, len(edges)-1}) if edges else []:
        i, j, d = edges[k]
        a, b = sorted((mapping[i], mapping[j]))
        l, r = left.get((i, j), 0), right.get((a, b), 0)
        if r - l != d:
            raise ValueError('Bond change inconsistent with both state tables')
        result.append([i, j, a, b, l, r, d])
    return result


def checks(reply, expected):
    values = reply.get('properties')
    values = values if isinstance(values, list) and len(values) == 3 else [False] * 3

    def number(a, b):
        return a is None if b is None else type(a) in (int, float) and math.isclose(a, b, abs_tol=1e-5, rel_tol=0)

    result = {'lean_identity': canonical(reply.get('lean_smiles')) == expected['lean_smiles'],
            'rich_identity': canonical(reply.get('rich_smiles')) == expected['rich_smiles'],
            'capacity': number(values[0], expected['properties'][0]),
            'enthalpy': number(values[1], expected['properties'][1]),
            'mapping_count': number(values[2], expected['properties'][2]),
            'correspondence': reply.get('correspondence') == expected['correspondence'],
            'delta': reply.get('delta') == expected['delta']}
    if 'cross_matrix' in expected:
        result['cross_matrix'] = reply.get('cross_matrix') == expected['cross_matrix']
    return result


def run(run_dir, out, key_file, jobs, use_protocol=False):
    RDLogger.DisableLog('rdApp.error')
    entries = load(run_dir / 'three_matrices_prebuild_01/matrices.json')
    source = {r['id']: r['pair'] for r in load(run_dir / 'absolute_structure_replay_01/results.json')['scores']}
    if len(entries) != 60 or set(entries) != set(source):
        raise ValueError('Expected the full 60-pair matrix cohort')
    cases = {}
    for identifier, entry in sorted(entries.items()):
        lean, rich = map(canonical, source[identifier])
        if (canonical(graph_identity(entry['AL'])), canonical(graph_identity(entry['AR']))) != (lean, rich):
            raise ValueError(f'Matrix and source graph disagree for {identifier}')
        for mode in ('original', 'permuted'):
            data = packet(entry, identifier, mode == 'permuted')
            expected = {'lean_smiles': lean, 'rich_smiles': rich, **data['P']}
            if use_protocol:
                expected['cross_matrix'] = cross_matrix(data)
            cases[f'{identifier}_{mode}'] = (data, expected)
    out.mkdir(parents=True, exist_ok=False)
    prompt = '\n'.join((RULES, BLOCKS, PROMPT, PROBE)) if use_protocol else PROMPT
    (out / 'prompt.txt').write_text(prompt, encoding='utf-8')
    save(out / 'protocol.json', {'source': str(run_dir), 'cases': len(cases), 'molecules': len(entries),
        'temperature': 0, 'retry_limit_per_case': 1, 'stereochemistry': 'not encoded; graph identity ignores stereo',
        'retry_policy': 'supply only failing field names, no ground-truth values; replace only passing retry fields',
        'float_abs_tolerance': 1e-5, 'matrix_protocol': VERSION if use_protocol else None})
    api = DeepSeek(credential(key_file), out)

    def one(name):
        data, expected = cases[name]
        save(out / f'{name}_input.json', data)
        first = final = None
        attempts = []
        failing = []
        for attempt in range(2):
            note = (' Re-read these failed check categories; return the full JSON: ' + ', '.join(failing)) if failing else ''
            try:
                reply = api.run(f'{name}_{attempt}', prompt + note, data)
                if not isinstance(reply, dict):
                    raise ValueError('Response must be an object')
                result = checks(reply, expected)
                attempts.append({'checks': result, 'response': reply})
            except (OSError, TypeError, ValueError, KeyError, IndexError, RuntimeError) as error:
                result = checks({}, expected)
                attempts.append({'checks': result, 'error_type': type(error).__name__})
            if attempt == 0:
                first = result
            final = {k: first[k] or result[k] for k in result}
            failing = [k for k, ok in final.items() if not ok]
            if not failing:
                break
        return name, {'expected': expected, 'attempts': attempts, 'first': first, 'after_retry': final}

    results = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for name, result in pool.map(one, cases):
            results[name] = result
            if len(results) % 20 == 0:
                print(f'{len(results)}/{len(cases)} readbacks completed', flush=True)
    summary = {}
    for mode in ('original', 'permuted'):
        subset = [r for name, r in results.items() if name.endswith('_'+mode)]
        summary[mode] = {'n': len(subset), **{stage: {'all_correct': sum(all(r[stage].values()) for r in subset),
            'identity_pair': sum(r[stage]['lean_identity'] and r[stage]['rich_identity'] for r in subset),
            'field_correct': dict(Counter(k for r in subset for k, ok in r[stage].items() if ok))}
            for stage in ('first', 'after_retry')}}
    audits = [load(p) for p in out.glob('*_audit.json')]
    report = {'summary': summary, 'requests': len(audits),
              'tokens': sum((a.get('usage') or {}).get('total_tokens', 0) for a in audits), 'cases': results}
    save(out / 'results.json', report)
    print({k: v for k, v in report.items() if k != 'cases'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--protocol', action='store_true')
    args = parser.parse_args()
    run(args.run, args.out, args.key_file, args.jobs, args.protocol)
