"""Budgeted three-matrix reaction Critic for M3OS graph rewards."""

from hashlib import sha256
from http.client import HTTPException
import json
from pathlib import Path
from urllib.parse import quote

from rdkit import Chem, DataStructs

from lohc_matrix_protocol import BLOCKS, DISPATCH, RULES
from lohc_matrix_memory_trial import matrix_pair
from lohc_matrix_readback import cross_matrix, packet
from lohc_matrix_memory_trial import request_json
from lohc_three_matrices import encode
from m3os_full import FullMoleculeSearch
from m3os_search import canonical, fingerprint, load
from rdkit.Chem.Scaffolds import MurckoScaffold


WEIGHT = .25
PROMPT = '''Evaluate LOHC two-state structural reaction potential. Assess whether the recorded bond changes support the intended hydrogenation and reverse transformation, skeleton retention and local competitive reaction risks. Use admitted SP evidence where present, otherwise explicitly infer from structure only. Capacity and enthalpy are separately scored by code; never score them here. Give an absolute heuristic score 0–100: 0 no structural support, 25 major structural obstacle, 50 plausible with substantial uncertainty, 75 clear structural support with limited concerns, 100 strong evidence covering selectivity and repeated cycling. Do not infer experimental reversibility from the mapping. Return JSON {"reviews":[{"id":"X1","SP":50,"basis":"structure_only","evidence_ids":[],"site":[1,2],"reason":"specific and short"}]}. Basis must match the supplied basis. Use one lean-coordinate changed bond from the candidate. Return all IDs exactly once.'''
INSTRUCTION = '\n'.join((RULES, BLOCKS, DISPATCH, PROMPT))


class MatrixSPMoleculeSearch(FullMoleculeSearch):
    def __init__(self, output, book, key_file, save):
        super().__init__(output, book, key_file, save)
        source = Path(__file__).parent / 'seed_data' / 'literature_memory.json'
        self.sp_evidence = load(source)
        self.sp_evidence_pairs = {r['id']: matrix_pair(r['matrices']) for r in self.sp_evidence}
        self.sp_cache = {}
        self.base_scores = {}
        self.online = False
        self.round_ledger = []
        self.active_evidence = {}
        self.save(self.output / 'sp_protocol.json', {'lambda': WEIGHT, 'max_candidates_per_round': 8,
            'max_critic_batches_per_round': 2, 'prompt_sha256': sha256(INSTRUCTION.encode()).hexdigest(),
            'evidence_sha256': sha256(source.read_bytes()).hexdigest(),
            'unscored': 'base graph score', 'evidence': 'frozen exact SP evidence or structure_only'})

    def enable_online(self, generations):
        self.online = True
        self.parent_uses_matrix_book = False
        self.round_limit = max(0, generations - 4)
        self.agent_limit = 1 + self.round_limit * 6
        self.memory_path = self.output / 'literature_memory.json'
        self.save(self.memory_path, self.sp_evidence)
        self.save(self.output / 'online_protocol.json', {'rounds': self.round_limit, 'max_llm_requests': self.agent_limit,
            'search_per_round': 1, 'memory_path': self.memory_path.name, 'lambda': WEIGHT,
            'analogue': 'same nonempty lean scaffold and both-state Morgan similarity >= 0.4, qualitative only',
            'discovery': 'Crossref records queued for review, admitted evidence reread every round'})
        self.save(self.output / 'protocol.json', {**load(self.output / 'protocol.json'),
                                                  'rounds': self.round_limit, 'parent_matrix_book': False})
        self.save(self.output / 'full_protocol.json', {**load(self.output / 'full_protocol.json'),
                                                       'max_calls': self.agent_limit})

    def evidence_for(self, pair):
        exact, similar = [], []
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=pair[0]) if self.online else None
        for row in self.sp_evidence:
            if row['endpoint'] != 'SP':
                continue
            other = self.sp_evidence_pairs[row['id']]
            info = {k: v for k, v in row.items() if k != 'matrices'}
            if other == pair:
                exact.append(info)
            elif self.online and scaffold and MurckoScaffold.MurckoScaffoldSmiles(smiles=other[0]) == scaffold:
                similarity = min(DataStructs.TanimotoSimilarity(fingerprint(a), fingerprint(b)) for a, b in zip(pair, other))
                if similarity >= .4:
                    similar.append({**info, 'source_pair': other, 'similarity': similarity,
                                    'use': 'qualitative analogue; do not transfer observed properties'})
        if exact:
            return 'direct', exact
        if similar:
            return 'analogue', sorted(similar, key=lambda r: (-r['similarity'], r['id']))[:3]
        return 'structure_only', []

    def queue_discovery(self, generation, pair, blocks):
        label = f'{generation:04d}'
        try:
            reply = self.request('evidence_search', label,
                RULES + ' Given the program-derived structure and matrices, propose one focused English scholarly '
                'query for this LOHC conversion or a closely related scaffold, focusing on selectivity and '
                'cycling. Return JSON {"query":"..."}. Do not supply DOI.', {'pair': pair, 'matrices': blocks})
            query = reply.get('query')
            if not isinstance(query, str) or not 5 <= len(query) <= 280:
                raise ValueError('invalid query')
            rows = request_json('https://api.crossref.org/works', {'query.bibliographic': query, 'rows': 8,
                                                                'select': 'DOI,title,URL'})['message']['items']
            self.save(self.output / f'evidence_queue_{label}.json', {'pair': pair, 'query': query,
                'status': 'awaiting_source_verification', 'records': rows})
        except (OSError, HTTPException, RuntimeError, ValueError, TypeError, KeyError) as exc:
            self.save(self.output / f'evidence_queue_{label}.json', {'pair': pair, 'status': 'search_failed',
                                                                     'error': type(exc).__name__})

    def review_evaluated(self, generation, evaluated):
        if self.online:
            current = load(self.memory_path)
            if current != self.sp_evidence:
                pairs = {r['id']: matrix_pair(r['matrices']) for r in current}
                self.sp_evidence, self.sp_evidence_pairs = current, pairs
                self.save(self.output / f'evidence_update_{generation:04d}.json', {'records': len(current),
                    'sha256': sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()})
        branches = {canonical(s): b for s, b in self.active['branches'].items()}
        stats, candidates = {}, []
        for name in ('creative', 'rational'):
            nodes = [self.nodes[p] for p in sorted(evaluated) if p in self.nodes and branches[p[0]] == name]
            feasible = sorted((node for node in nodes if node.feasible), key=lambda n: (-n.score, n.pair))
            stats[name] = {'evaluated': len(nodes), 'feasible': len(feasible), 'pairs': [n.pair for n in nodes]}
            candidates.extend(feasible[:4])
        if self.online:
            changed = [self.nodes[p] for p, signature in self.active_evidence.items()
                       if p in self.nodes and self.evidence_for(p) != signature]
            candidates = list({n.pair: n for n in changed[:4] + candidates}.values())[:8]
        reviews, pending = {}, []
        for node in candidates:
            try:
                matrices = encode(node.pair, float(node.row['delta_H_per_H2']))
                blocks = packet(matrices, '', False)
                sites = [row[:2] for row in blocks['P']['delta']]
                if not sites:
                    continue
                basis, evidence = self.evidence_for(node.pair)
                key = sha256(json.dumps([matrices, evidence, 'deepseek-flash', INSTRUCTION], sort_keys=True).encode()).hexdigest()
                details = {'matrices': matrices, 'key': key, 'basis': basis, 'evidence': evidence,
                           'evidence_ids': {r['id'] for r in evidence}, 'sites': sites}
                if key in self.sp_cache:
                    reviews[node.pair] = (self.sp_cache[key], details)
                else:
                    # Hide scalar objectives from the model, keeping the full matrix in scoring memory.
                    blocks['P']['properties'][:2] = [None, None]
                    pending.append((node.pair, details, {'matrices': blocks, 'basis': basis, 'evidence': evidence,
                                                       'verified_bonds': cross_matrix(blocks)}))
            except (ValueError, RuntimeError, IndexError, AssertionError):
                continue
        parent = self.active['selected']
        try:
            parent_block = packet(encode(parent), '', False)
            parent_block['P']['properties'][:2] = [None, None]
        except (ValueError, RuntimeError, IndexError, AssertionError):
            parent_block = None
        if self.online:
            missing = next((item for item in pending if item[1]['basis'] != 'direct'), None)
            if missing:
                self.queue_discovery(generation, missing[0], missing[2]['matrices'])
        for start in range(0, len(pending), 4):
            batch = {f'X{i+1}': item for i, item in enumerate(pending[start:start+4])}
            try:
                response = self.request('critic', f'{generation:04d}_{start//4+1}', INSTRUCTION,
                                        {'parent_matrices': parent_block,
                                         'candidates': [{'id': k, **v[2]} for k, v in batch.items()]})
            except HTTPException:
                response = {}
            rows = response.get('reviews', [])
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict) or not isinstance(row.get('id'), str) or row['id'] not in batch:
                    continue
                pair, details, _ = batch[row['id']]
                cites = row.get('evidence_ids')
                if (pair not in reviews and type(row.get('SP')) is int and 0 <= row['SP'] <= 100
                        and row.get('basis') == details['basis'] and row.get('site') in details['sites']
                        and isinstance(cites, list) and all(isinstance(v, str) for v in cites)
                        and set(cites) <= details['evidence_ids'] and bool(cites) == bool(details['evidence_ids'])
                        and isinstance(row.get('reason'), str) and row['reason'].strip()):
                    grade = {k: v for k, v in row.items() if k != 'id'}
                    reviews[pair] = (grade, details)
                    self.sp_cache[details['key']] = grade
        records = []
        for pair, (grade, details) in reviews.items():
            node = self.nodes[pair]
            similarity = DataStructs.TanimotoSimilarity(fingerprint(pair[0]), fingerprint(node.root[0]))
            before = node.score
            original = self.base_scores.setdefault(pair, before)
            base_task = min(1, max(0, (original - .2 * similarity) / .8))
            node.score = .8 * ((1 - WEIGHT) * base_task + WEIGHT * grade['SP'] / 100) + .2 * similarity
            prior_delta = node.score - before if self.online or node.visits == 0 else 0.0
            node.reward += prior_delta
            self.active_evidence[pair] = (details['basis'], details['evidence'])
            records.append({'pair': pair, 'score_before': before, 'score_after': node.score, 'prior_delta': prior_delta,
                            'matrices': details['matrices'], 'key': details['key'], 'assessment': grade})
        self.active.update(branch_results=stats, critic_reviewed=len(records), critic_expected=len(candidates),
                           critic_applied=sum(abs(r['score_after'] - r['score_before']) > 1e-12 for r in records))
        self.save(self.output / f'scoring_memory_{generation:04d}.json', records)
        if self.online:
            revisions = self.revise_rewards({tuple(r['pair']) for r in records})
            if revisions:
                self.save(self.output / f'reward_revisions_{generation:04d}.json', revisions)
        return records

    def finish(self, generation, frame, evaluated_pairs, ga):
        super().finish(generation, frame, evaluated_pairs, ga)
        if self.online and self.active is not None:
            record = load(self.output / f'round_{generation:04d}.json')
            self.round_ledger.append({'generation': generation, 'selected': tuple(record['selected']),
                'candidates': [tuple(p) for p in record['evaluated_pairs'] if tuple(p) in self.nodes and self.nodes[tuple(p)].feasible],
                'best': tuple(record['best']) if record['best'] else None, 'reward': record['reward']})

    def revise_rewards(self, rescored):
        revisions = []
        for record in self.round_ledger:
            if not rescored.intersection(record['candidates']):
                continue
            best = min(record['candidates'], key=lambda p: (-self.nodes[p].score, p))
            reward = self.nodes[best].score
            if best == record['best'] and abs(reward - record['reward']) < 1e-12:
                continue
            path = set()
            cursor = record['selected']
            while cursor is not None:
                path.add(cursor)
                cursor = self.nodes[cursor].origin
            old_path, new_path = path | {record['best']}, path | {best}
            for pair in old_path | new_path:
                node = self.nodes[pair]
                node.visits += int(pair in new_path) - int(pair in old_path)
                node.reward += (reward if pair in new_path else 0) - (record['reward'] if pair in old_path else 0)
            revisions.append({'generation': record['generation'], 'old': record['reward'], 'new': reward,
                              'old_best': record['best'], 'best': best})
            record.update(best=best, reward=reward)
        return revisions
