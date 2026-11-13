"""M3OS-style molecule selection and path rewards around the native GA."""

from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from math import isfinite, log, sqrt
import random

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from lohc_structure_score_trial import DeepSeek, credential, load


FIRST_ROUND = 5
ROUND_LIMIT = 8
SLOTS = 32
EXPLORATION = 0.01
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
PROMPT = ('Choose one LOHC parent ID from the shortlist for a batch of existing GA edits. '
          'Use the two molecular states, constraint margins, code scores when present and prior offspring outcomes. '
          'Look for a parent with room to produce a better feasible child. Unknown evidence stays unknown. '
          'Return only JSON with id and reason. Do not propose edits or invent measurements.')


@lru_cache(maxsize=40000)
def canonical(smiles):
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


@lru_cache(maxsize=40000)
def fingerprint(smiles):
    return MORGAN.GetFingerprint(Chem.MolFromSmiles(smiles))


@dataclass
class Node:
    pair: tuple
    row: dict
    origin: tuple | None
    root: tuple
    score: float
    feasible: bool
    reward: float
    visits: int = 0
    parents: set = field(default_factory=set)
    children: set = field(default_factory=set)


class MoleculeSearch:
    def __init__(self, output, book, key_file, save):
        self.output, self.book, self.save = output / 'm3os', book, save
        self.output.mkdir()
        self.api = DeepSeek(credential(key_file), self.output) if key_file else None
        self.nodes, self.by_lean = {}, {}
        self.calls = self.tokens = 0
        self.active = None
        self.stats = {}
        self.round_limit = ROUND_LIMIT
        self.parent_uses_matrix_book = True
        save(self.output / 'protocol.json', {'first_round': FIRST_ROUND, 'rounds': ROUND_LIMIT,
            'reserved_slots': SLOTS, 'exploration': EXPLORATION, 'shortlist_size': 5,
            'task': '0.5 * h2 + 0.5 * enthalpy suitability', 'task_weight': 0.8,
            'prompt_sha256': sha256(PROMPT.encode()).hexdigest(), 'paths': 'first-discovery lineage plus current edge'})

    @property
    def total_calls(self):
        return self.calls

    @property
    def total_tokens(self):
        return self.tokens

    def parent_packet(self, generation, items):
        return {'constraints': {'h2_min': 4.2, 'dh': [40, 70], 'sa_max': 4}, 'parents': items}

    def ingest(self, frame, generation, ga):
        known = self.by_lean.copy()
        for row in frame.to_dict('records'):
            lean, rich = canonical(row.get('smi')), canonical(row.get('sat_smi'))
            h2, dh, sa = (row.get(k, np.nan) for k in ('h2_wt_percent', 'delta_H_per_H2', 'SA score'))
            if lean is None or rich is None or not all(isinstance(v, (int, float, np.number)) and isfinite(v)
                                                       for v in (h2, dh, sa)):
                continue
            pair = (lean, rich)
            current_edit = row.get('operation_generation') == generation and generation > 0
            parent = known.get(canonical(row.get('parent_smi'))) if current_edit else None
            if parent == pair:
                parent = None
            if pair not in self.nodes:
                root = self.nodes[parent].root if parent else pair
                feasible = bool(row.get('qd_target_hit') == 1 and sa <= 4)
                h2_score = min(1.0, max(0.0, (h2 - 4.2) / 2.8))
                task = 0.5 * (h2_score + float(ga.dh_quality(dh)))
                similarity = DataStructs.TanimotoSimilarity(fingerprint(lean), fingerprint(root[0]))
                score = 0.8 * task + 0.2 * similarity
                self.nodes[pair] = Node(pair, row, parent, root, score, feasible, score if feasible else 0.0)
            self.by_lean[lean] = pair
            if parent:
                self.nodes[pair].parents.add(parent)
                self.nodes[parent].children.add(pair)
                self.record_edge(parent, pair, generation, row)

    def record_edge(self, parent, child, generation, row):
        return None

    def ucb(self, node):
        parent_visits = sum(self.nodes[p].visits for p in node.parents) if node.parents else node.visits
        return node.reward / max(1, node.visits) + EXPLORATION * sqrt(log(max(1, parent_visits)) / max(1, node.visits))

    def shortlist(self):
        # Near-feasible states may still produce a useful child, while SA remains a gate.
        eligible = [node for node in self.nodes.values() if node.row['SA score'] <= 4]
        return sorted(eligible, key=lambda node: (-self.ucb(node), -node.score, node.pair))[:5]

    def choose(self, generation, shortlist):
        if not self.api or self.calls >= self.round_limit or len(shortlist) < 2:
            return 0, None, None
        items = []
        for i, node in enumerate(shortlist):
            grade = self.book.evaluate(node.pair) if self.parent_uses_matrix_book else None
            children = [self.nodes[p] for p in node.children]
            items.append({'id': i + 1, 'pair': node.pair, 'ucb': self.ucb(node), 'visits': node.visits,
                          'h2': float(node.row['h2_wt_percent']), 'dh': float(node.row['delta_H_per_H2']),
                          'sa': float(node.row['SA score']), 'matrix_score': grade['score'] if grade else None,
                          'feasible': node.feasible, 'observed_children': len(children),
                          'feasible_children': sum(ch.feasible for ch in children),
                          'best_feasible_child': max((ch.score for ch in children if ch.feasible), default=None)})
            if not self.parent_uses_matrix_book:
                items[-1].pop('matrix_score')
        packet = self.parent_packet(generation, items)
        self.calls += 1
        role = f'select_{generation:04d}'
        self.save(self.output / f'{role}_input.json', packet)
        try:
            reply = self.api.run(role, PROMPT, packet)
            selected = reply.get('id')
            if type(selected) is not int or not 1 <= selected <= len(shortlist) or not isinstance(reply.get('reason'), str):
                raise ValueError('Invalid shortlist ID or reason')
            return selected - 1, reply['reason'], None
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
            return 0, None, type(exc).__name__
        finally:
            audit = self.output / f'{role}_audit.json'
            if audit.exists():
                self.tokens += (load(audit).get('usage') or {}).get('total_tokens', 0)

    def proposals(self, generation, ordinary, ga, cfg):
        self.active = None
        if not FIRST_ROUND <= generation < FIRST_ROUND + self.round_limit:
            return ordinary
        shortlist = self.shortlist()
        if not shortlist:
            return ordinary
        chosen, reason, fallback = self.choose(generation, shortlist)
        node = shortlist[chosen]
        self.active = {'generation': generation, 'shortlist': [{'pair': n.pair, 'ucb': self.ucb(n)} for n in shortlist],
                       'selected': node.pair, 'rank': chosen + 1, 'reason': reason, 'fallback': fallback}
        np_state, random_state = np.random.get_state(), random.getstate()
        np.random.seed(11 + generation * 1009)
        random.seed(11 + generation * 1009)
        try:
            children = ga.generate_guided_mutation_offspring(pd.DataFrame([node.row]), SLOTS * 4,
                float(cfg['max_mol_weight']), float(cfg['sa_max_generation']),
                bool(cfg['preserve_platform_in_mutation']), None, generation - 1, cfg['survivor_screening'])
        finally:
            np.random.set_state(np_state)
            random.setstate(random_state)
        blocked = set(self.by_lean) | {canonical(s) for s in ordinary['smi']}
        keys = children['smi'].map(canonical)
        children = children.loc[keys.notna() & ~keys.isin(blocked) & ~keys.duplicated()].copy()
        children = self.branch_children(generation, node, children, blocked, ga, cfg)
        children['operation_generation'] = generation
        children['discovery_generation'] = generation
        children['fitness_evaluated'] = 0
        children['m3os_round'] = generation
        self.active['proposed'] = children['smi'].tolist()
        return pd.concat([ordinary, children], ignore_index=True, sort=False) if not children.empty else ordinary

    def branch_children(self, generation, node, children, blocked, ga, cfg):
        return children.head(SLOTS).copy()

    def review_evaluated(self, generation, evaluated):
        return None

    def finish(self, generation, frame, evaluated_pairs, ga):
        self.ingest(frame, generation, ga)
        if self.active is None:
            return
        selected = self.active['selected']
        proposed = {canonical(s) for s in self.active['proposed']}
        evaluated = {(canonical(lean), canonical(rich)) for lean, rich in evaluated_pairs if canonical(lean) in proposed}
        if len(evaluated) > SLOTS:
            raise ValueError('M3OS candidate evaluation reservation exceeded')
        feasible = [self.nodes[p] for p in evaluated if p in self.nodes and self.nodes[p].feasible]
        base_best = min(feasible, key=lambda node: (-node.score, node.pair)) if feasible else None
        base_reward = base_best.score if base_best else 0.0
        old_ucb = self.ucb(self.nodes[selected])
        self.review_evaluated(generation, evaluated)
        best = min(feasible, key=lambda node: (-node.score, node.pair)) if feasible else None
        reward = best.score if best else 0.0
        path = []
        cursor = selected
        while cursor is not None:
            path.append(cursor)
            cursor = self.nodes[cursor].origin
        if best and best.pair not in path:
            path.insert(0, best.pair)
        updates = []
        for pair in reversed(path):
            node = self.nodes[pair]
            node.visits += 1
            node.reward += reward
            updates.append({'pair': pair, 'visits': node.visits, 'reward_sum': node.reward})
        self.stats = {'m3os_calls': self.calls, 'm3os_tokens': self.tokens, 'm3os_evaluated': len(evaluated),
                      'm3os_feasible': len(feasible), 'm3os_parent_rank': self.active['rank'], 'm3os_graph_nodes': len(self.nodes)}
        self.save(self.output / f'round_{generation:04d}.json', {**self.active, **self.stats, 'reward': reward,
            'best': best.pair if best else None, 'evaluated_pairs': sorted(evaluated), 'path_updates': updates,
            'base_reward': base_reward, 'base_best': base_best.pair if base_best else None,
            'selected_ucb_before': old_ucb, 'selected_ucb_after': self.ucb(self.nodes[selected]),
            'next_shortlist': [{'pair': n.pair, 'ucb': self.ucb(n)} for n in self.shortlist()]})

    def export(self):
        self.save(self.output / 'graph.json', [{'pair': n.pair, 'origin': n.origin, 'root': n.root,
            'parents': sorted(n.parents), 'score': n.score, 'feasible': n.feasible,
            'visits': n.visits, 'reward_sum': n.reward} for n in self.nodes.values()])
