"""Structure-only DeepSeek pilot on the existing A/C blind samples."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, median
import sys

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1] / 'outputs'
sys.path.insert(0, str(ROOT / 'LOHC_Agents_v2'))
from agents import DeepSeek, credential


AXES = ('path', 'skeleton', 'competition')
MOTIFS = {
    '芳香氮': '[nH,nX2,nX3]', '非芳香胺氮': '[N;X3;!$(N-C=O)]',
    '芳醚': '[c]-[O;X2]-[#6]', '脂肪醚': '[C;X4]-[O;X2]-[C;X4]',
    '芳基碳氮键': '[c]-[#7]', '酚羟基': '[c]-[O;H1]',
    '羰基': '[#6]=[#8]', '非芳香碳碳双键': '[C]=[C]',
    '碳氮双键': '[#6]=[#7]', '碳氮三键': '[#6]#[#7]',
    '非芳香三元或四元环原子': '[A;r3,r4]',
}
QUERIES = {label: Chem.MolFromSmarts(smarts) for label, smarts in MOTIFS.items()}
PROMPT = '''你为液态有机储氢载体（LOHC）做结构初筛。每条含同一候选的放氢态和储氢态、骨架以及RDKit提取的结构事实。只依据给定结构和通用化学原理，禁用同分子文献、记忆中的实验数值、催化剂或工况假设。编号与批内其他候选不得影响等级。输出的是结构层面值得验证的程度。
三项各给整数0–4，高分均有利。path 评价两态目标储放氢路径的结构合理性：0结构明显不支持，1有重大障碍，2可解释但有明显疑点，3常见合理的成键变化且有明确驱动力线索，4目标路径简单明确且有强结构支持。skeleton 评价能否在往返中保留并恢复原骨架：0需要破坏骨架，1有强烈结构损失警讯，2存在明确的骨架敏感位点，3主要为键级变化且骨架相对稳健，4两态骨架一致且结构上缺少明显的断裂或重排诱因。competition 评价目标路径相对于可能副反应的结构优势：0强烈不利，1有显著竞争通道，2有明确但未必主导的竞争通道，3只有有限竞争线索，4缺少明显竞争位点。同一风险若涉及两项，分别说清影响。不能把骨架图相同直接当作已证实可逆。条件依赖不等于低分，仍给基于结构的暂定等级。
每项给一个真实 feature ID 和一条简短理由，理由指出环系或键及影响，最多45字。confidence 仅表示结构推断的把握，用 low 或 medium；不因缺文献拒绝评分，也不声称已知实验转化率、循环、相态、来源。只返回严格 JSON，格式示例：{"assessments":[{"id":"S01","path":[3,"B1","芳香环恢复为目标脱氢提供结构依据"],"skeleton":[3,"B1","两态保持环骨架，主要是键级变化"],"competition":[2,"L2","芳醚位点提示可能竞争氢解"],"confidence":"low"}]}。逐项给出全部候选，不添加字段。'''


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def packet_item(identifier, pair):
    facts = {'B1': f'放氢态 {pair[0]}；储氢态 {pair[1]}'}
    for prefix, smiles in zip(('L', 'R'), pair):
        molecule = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        rings = ['aromatic' if all(molecule.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                 else 'nonaromatic' for ring in molecule.GetRingInfo().AtomRings()]
        facts[f'{prefix}1'] = f'骨架 {scaffold or "无环"}；环 {", ".join(rings) or "无"}'
        matches = [label for label, query in QUERIES.items() if molecule.HasSubstructMatch(query)]
        facts.update({f'{prefix}{i}': label for i, label in enumerate(matches, 2)})
    return {'id': identifier, 'features': facts}


def validate(response, batch):
    expected = {x['id']: x['features'] for x in batch}
    rows = response['assessments']
    if set(response) != {'assessments'} or len(rows) != len(expected) or {r['id'] for r in rows} != set(expected):
        raise ValueError('Assessment IDs do not match the request')
    for row in rows:
        if set(row) != {'id', 'confidence', *AXES} or row['confidence'] not in {'low', 'medium'}:
            raise ValueError('Invalid assessment fields')
        for axis in AXES:
            grade, feature, reason = row[axis]
            if type(grade) is not int or not 0 <= grade <= 4 or feature not in expected[row['id']]:
                raise ValueError(f'Invalid grade or feature in {axis}')
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f'Empty reason in {axis}')
    return {row['id']: row for row in rows}


def run(out, key_file):
    source = ROOT / 'LOHC_From_Zero_Parallel_v10/operator_ac_run_01/ac_evaluation.json'
    stages = load(source)['stages']
    cutoff = max(stages, key=int)
    stage = stages[cutoff]
    populations = {arm: [tuple(x['candidate']['pair']) for x in data['sample']]
                   for arm, data in stage['arms'].items()}
    pairs = sorted(set().union(*map(set, populations.values())))
    random.Random(11).shuffle(pairs)
    items = [packet_item(f'S{i:02}', pair) for i, pair in enumerate(pairs, 1)]
    lookup = dict(zip(pairs, (item['id'] for item in items)))
    membership = {arm: [lookup[pair] for pair in group] for arm, group in populations.items()}
    key = credential(key_file)
    out.mkdir(parents=True)
    (out / 'prompt.txt').write_text(PROMPT, encoding='utf-8')
    save(out / 'protocol.json', {'source': str(source), 'cutoff': cutoff, 'cutoffs': stage['cutoffs'],
                                 'membership': membership, 'model': 'deepseek-flash', 'temperature': 0,
                                 'max_tokens': 1100, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                                 'score': '100 * (path + skeleton + competition) / 12',
                                 'retry': 'at most one single-candidate retry on initial failure',
                                 'repeat': 'four prespecified IDs rescored individually under new IDs'})
    api = DeepSeek(key, out)

    def ask(job):
        role, batch = job
        packet = {'molecules': batch}
        save(out / f'{role}_input.json', packet)
        try:
            return validate(api.run(role, PROMPT, packet), batch)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            save(out / f'{role}_failure.json', {'error': str(error).replace(key, '[redacted]')})
            return {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [(f'batch_{i // 2 + 1:02}', items[i:i+2]) for i in range(0, len(items), 2)]
        results = {identifier: row for part in pool.map(ask, jobs) for identifier, row in part.items()}
        retry_jobs = [(f'retry_{x["id"]}', [x]) for x in items if x['id'] not in results]
        results.update({identifier: row for part in pool.map(ask, retry_jobs) for identifier, row in part.items()})
        repeat_items = items[::6][:4]
        repeats = {f'R{i:02}': item for i, item in enumerate(repeat_items, 1)}
        repeat_jobs = [(f'repeat_{identifier}', [{**item, 'id': identifier}]) for identifier, item in repeats.items()]
        repeated = {identifier: row for part in pool.map(ask, repeat_jobs) for identifier, row in part.items()}

    def score(row):
        return round(100 * sum(row[axis][0] for axis in AXES) / 12, 2)

    groups = {}
    for arm, identifiers in membership.items():
        rows = [results[i] for i in identifiers if i in results]
        values = [score(row) for row in rows]
        groups[arm] = {'n': len(identifiers), 'scored': len(rows), 'scores': sorted(values),
                       'mean': round(mean(values), 2) if values else None,
                       'median': median(values) if values else None,
                       'bins': {'<50': sum(v < 50 for v in values), '50–<70': sum(50 <= v < 70 for v in values),
                                '70–<85': sum(70 <= v < 85 for v in values), '85–100': sum(v >= 85 for v in values)},
                       'axis_means': {axis: round(mean(r[axis][0] for r in rows), 2) for axis in AXES} if rows else {},
                       'confidence': Counter(r['confidence'] for r in rows)}
    audits = [load(path) for path in out.glob('*_audit.json')]
    checks = [{'id': item['id'], 'first': score(results[item['id']]), 'repeat': score(repeated[i]),
               'axis_deltas': {axis: repeated[i][axis][0] - results[item['id']][axis][0] for axis in AXES}}
              for i, item in repeats.items() if i in repeated and item['id'] in results]
    report = {'groups': groups, 'unique_candidates': len(items), 'unique_scored': len(results),
              'checks': checks, 'requests': len(audits), 'failed_attempts': len(list(out.glob('*_failure.json'))),
              'tokens': sum((audit.get('usage') or {}).get('total_tokens', 0) for audit in audits),
              'candidates': [{**item, 'pair': pair, 'assessment': results.get(item['id']),
                              'score': score(results[item['id']]) if item['id'] in results else None}
                             for item, pair in zip(items, pairs)], 'membership': membership}
    save(out / 'results.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in {'candidates', 'membership'}}, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--key-file', required=True, type=Path)
    args = parser.parse_args()
    run(args.out, args.key_file)
