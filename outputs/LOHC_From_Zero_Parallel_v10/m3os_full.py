"""M3OS-style LOHC tool orchestration around a shared molecular graph."""

from collections import Counter
from hashlib import sha256

import numpy as np
import pandas as pd
from rdkit import Chem

from m3os_search import MoleculeSearch, canonical, load
from validate_lohc_matrix_adjustments import cards


CREATIVE_SLOTS, RATIONAL_SLOTS = 24, 8
AGENT_LIMIT = 41
ARCHITECT = 'Choose one search focus from capacity, retention, diversity for this LOHC contract. Return JSON {"focus":"...","reason":"..."}. Constraints are fixed.'
RETRIEVAL = 'Choose up to three relevant evidence IDs from this sourced LOHC library for the current parent. Return JSON {"ids":[...]}. Select none if no source applies.'
RATIONAL = ('Select up to eight distinct edits from the provided LOHC parent menu. Each edit is an executable site and group change. '
            'Consider recorded success and failures and preserve the ring scaffold. Return JSON {"edits":[{"id":"E1","reason":"...","reference":null}]}. '
            'Reference must be an available evidence ID or null. Do not claim unmeasured properties.')
CRITIC = ('Review all listed graph candidates against their GNN values and hard-gate status. Return JSON {"reviews":[["id",delta,"reference or null","short reason"]]}. '
          'Use delta null when evidence cannot justify a change, otherwise use an integer -10 through 10 (hundredths of task score). '
          'Only use evidence IDs present in this request. State the relevant site and catalyst limitation. Do not double-count the code matrix score. '
          'Hard gates are code results, and predicted values are not measurements. Include every ID exactly once.')
SOURCE_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in {
    'GUAIACOL-HDO': '[OH1]-c:c-[OX2]-[CH3]',
    'NEC-CYCLE': 'c1ccc2c(c1)n([#6])c1ccccc12',
    'NEC-SELECT': 'c1ccc2c(c1)n([#6])c1ccccc12',
    'MCH-CYCLE': '[CH3]-c1ccccc1',
}.items()}


class FullMoleculeSearch(MoleculeSearch):
    def __init__(self, output, book, key_file, save):
        super().__init__(output, book, key_file, save)
        self.evidence = {item['id']: item for item in cards()}
        self.aux_calls = Counter()
        self.aux_tokens = 0
        self.completed = []
        self.edge_history = {}
        self.edge_records = {}
        self.focus = 'capacity'
        self.references = []
        self.agent_limit = AGENT_LIMIT
        self.save(self.output / 'full_protocol.json', {'max_calls': AGENT_LIMIT,
            'creative_slots': CREATIVE_SLOTS, 'rational_slots': RATIONAL_SLOTS, 'critic_delta_bounds': [-0.10, 0.10],
            'prompts': {name: sha256(prompt.encode()).hexdigest() for name, prompt in
                        [('architect', ARCHITECT), ('retrieval', RETRIEVAL), ('rational', RATIONAL), ('critic', CRITIC)]}})

    @property
    def total_calls(self):
        return self.calls + sum(self.aux_calls.values())

    @property
    def total_tokens(self):
        return self.tokens + self.aux_tokens

    def request(self, role, label, prompt, packet):
        if not self.api or self.total_calls >= self.agent_limit:
            return {}
        self.aux_calls[role] += 1
        name = f'{role}_{label}'
        self.save(self.output / f'{name}_input.json', packet)
        try:
            answer = self.api.run(name, prompt, packet)
            return answer if isinstance(answer, dict) else {}
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError) as exc:
            self.save(self.output / f'{name}_failure.json', {'error': type(exc).__name__})
            return {}
        finally:
            audit = self.output / f'{name}_audit.json'
            if audit.exists():
                self.aux_tokens += (load(audit).get('usage') or {}).get('total_tokens', 0)

    def parent_packet(self, generation, items):
        packet = super().parent_packet(generation, items)
        if not self.aux_calls['architect']:
            response = self.request('architect', '0000', ARCHITECT, {'constraints': packet['constraints'],
                'goal': 'maximize unique high-scoring feasible LOHC candidates and maintain diversity'})
            if response.get('focus') in ('capacity', 'retention', 'diversity'):
                self.focus = response['focus']
            self.save(self.output / 'task_contract.json', {'focus': self.focus, **packet['constraints']})
        counts = Counter(str(n.row.get('qd_cell') or 'unassigned') for n in self.nodes.values())
        for item in items:
            node = self.nodes[tuple(item['pair'])]
            cell = str(node.row.get('qd_cell') or 'unassigned')
            item['cell'], item['cell_population'] = cell, counts[cell]
            item['lineage'] = self.lineage(node)
        packet.update(focus=self.focus, recent_rounds=self.completed[-3:], graph_nodes=len(self.nodes))
        return packet

    def lineage(self, node):
        children = [self.nodes[p] for p in node.children]
        best = sorted(children, key=lambda n: (-n.feasible, -n.score, n.pair))[:2]
        failed = sorted((n for n in children if not n.feasible), key=lambda n: (-n.score, n.pair))[:1]
        return [{'pair': n.pair, 'edit': n.row.get('mutation_operator', ''),
                 'h2': float(n.row['h2_wt_percent']), 'dh': float(n.row['delta_H_per_H2']),
                 'sa': float(n.row['SA score']), 'feasible': n.feasible,
                 'retained': self.edge_history.get((node.pair, n.pair))} for n in best + failed]

    def branch_children(self, generation, node, children, blocked, ga, cfg):
        label = f'{generation:04d}'
        creative = children.head(CREATIVE_SLOTS).copy()
        creative['m3os_branch'] = 'creative'
        reserved = blocked | {canonical(s) for s in creative['smi']}
        retrieval = self.request('retrieval', label, RETRIEVAL, {'parent': node.pair, 'focus': self.focus,
            'lineage': self.lineage(node), 'sources': list(self.evidence.values())})
        ids = retrieval.get('ids', [])
        self.references = [self.evidence[k] for k in ids if isinstance(k, str) and k in self.evidence][:3] if isinstance(ids, list) else []
        menu = self.edit_menu(node, reserved, ga, cfg)
        selected = self.request('rational', label, RATIONAL, {'parent': node.pair, 'focus': self.focus,
            'menu': [{'id': k, **edit} for k, (edit, _) in menu.items()], 'lineage': self.lineage(node),
            'literature': self.references}) if menu else {}
        rows, accepted, seen = [], [], set()
        edits = selected.get('edits', [])
        refs = {r['id'] for r in self.references}
        for item in edits if isinstance(edits, list) else []:
            if not isinstance(item, dict) or not isinstance(item.get('id'), str):
                continue
            key = item['id']
            reference = item.get('reference')
            if (key in seen or key not in menu or not isinstance(item.get('reason'), str)
                    or (reference is not None and (not isinstance(reference, str) or reference not in refs))):
                continue
            seen.add(key)
            edit, mol = menu[key]
            rows.append({**node.row, **dict.fromkeys(ga._CHILD_PROPERTY_COLUMNS, np.nan),
                'smi': Chem.MolToSmiles(mol), 'mol': mol, 'parent_smi': node.row['smi'],
                'mutation_operator': f"m3os_{edit['action']}", 'origin_type': 'm3os_rational', 'm3os_branch': 'rational'})
            accepted.append({**item, **edit, 'child': rows[-1]['smi']})
            if len(rows) == RATIONAL_SLOTS:
                break
        self.active['branches'] = {str(row.smi): 'creative' for row in creative.itertuples()}
        self.active['branches'].update({row['smi']: 'rational' for row in rows})
        self.active['rational_edits'] = accepted
        self.save(self.output / f'edits_{label}.json', {'menu_size': len(menu), 'accepted': accepted,
                  'creative_proposed': len(creative), 'rational_proposed': len(rows)})
        return pd.concat([creative, pd.DataFrame(rows)], ignore_index=True, sort=False) if rows else creative

    @staticmethod
    def edit_menu(node, blocked, ga, cfg):
        parent = Chem.MolFromSmiles(node.pair[0])
        menu, seen = {}, set(blocked)
        for atom in parent.GetAtoms():
            if atom.GetAtomicNum() != 6 or not atom.GetIsAromatic() or atom.GetTotalNumHs() == 0:
                continue
            for name, atoms in [('methyl', (6,)), ('hydroxyl', (8,)), ('methoxy', (8, 6)), ('amino', (7,))]:
                edited = Chem.RWMol(parent)
                site = atom.GetIdx()
                for element in atoms:
                    new = edited.AddAtom(Chem.Atom(element))
                    edited.AddBond(site, new, Chem.BondType.SINGLE)
                    site = new
                try:
                    mol = edited.GetMol()
                    Chem.SanitizeMol(mol)
                    ok, mol = ga.mutation_mol_constrainer(None, float(cfg['max_mol_weight']), 1, mol, parent,
                        float(cfg['sa_max_generation']), node.row.get('platform_id'), bool(cfg['preserve_platform_in_mutation']))
                    smi = Chem.MolToSmiles(mol)
                    if ok and smi not in seen:
                        seen.add(smi)
                        menu[f'E{len(menu)+1}'] = ({'action': 'add_group', 'site': atom.GetIdx(), 'group': name}, mol)
                except (ValueError, RuntimeError):
                    continue
                if len(menu) >= 48:
                    return menu
        return menu

    def review_evaluated(self, generation, evaluated):
        parent = self.active['selected']
        branches = {canonical(smi): branch for smi, branch in self.active['branches'].items()}
        reviews = {}
        valid = [self.nodes[p] for p in sorted(evaluated) if p in self.nodes]
        reference_ids = {r['id'] for r in self.references}
        rational_edits = {canonical(row['child']): {k: row[k] for k in ('site', 'group')}
                          for row in self.active['rational_edits']}
        for i in range(0, len(valid), 16):
            batch = valid[i:i+16]
            lookup = {f'X{i+j+1}': node for j, node in enumerate(batch)}
            items = []
            for key, node in lookup.items():
                grade = self.book.evaluate(node.pair)
                items.append({'id': key, 'pair': node.pair, 'branch': branches[node.pair[0]], 'feasible': node.feasible,
                    'h2': float(node.row['h2_wt_percent']), 'dh': float(node.row['delta_H_per_H2']),
                    'sa': float(node.row['SA score']), 'edit': node.row.get('mutation_operator'),
                    'edit_site': rational_edits.get(node.pair[0]), 'matrix_score': grade['score'] if grade else None})
            answer = self.request('critic', f'{generation:04d}_{i//16+1}', CRITIC,
                {'parent': parent, 'candidates': items, 'literature': self.references})
            proposed = answer.get('reviews', [])
            if not isinstance(proposed, list):
                continue
            for review in proposed:
                if not isinstance(review, list) or len(review) != 4 or not isinstance(review[0], str):
                    continue
                key, delta, reference, reason = review
                if key not in lookup or key in reviews or not isinstance(reason, str):
                    continue
                node = lookup[key]
                pattern = SOURCE_PATTERNS.get(reference) if isinstance(reference, str) else None
                has_evidence = (pattern is not None and reference in reference_ids
                                and Chem.MolFromSmiles(node.pair[0]).HasSubstructMatch(pattern))
                if reference in ('GUAIACOL-HDO', 'MCH-CYCLE') and type(delta) is int and delta > 0:
                    has_evidence = False
                accepted = type(delta) is int and abs(delta) <= 10 and has_evidence and node.feasible
                if accepted:
                    new_score = min(1.0, max(0.0, node.score + 0.8 * delta / 100))
                    if node.visits == 0:
                        node.reward += new_score - node.score
                    node.score = new_score
                reviews[key] = {'pair': node.pair, 'reason': reason, 'reference': reference,
                                'proposed_delta': delta, 'applied_delta': delta if accepted else None}
        stats = {}
        for name in ('creative', 'rational'):
            selected = [p for p in evaluated if branches[p[0]] == name]
            stats[name] = {'evaluated': len(selected), 'feasible': sum(p in self.nodes and self.nodes[p].feasible for p in selected),
                           'pairs': sorted(selected)}
        self.active.update(branch_results=stats, critic_reviewed=len(reviews), critic_expected=len(valid),
                           critic_applied=sum(r['applied_delta'] not in (None, 0) for r in reviews.values()))
        self.save(self.output / f'critic_{generation:04d}_validated.json', {'reviews': list(reviews.values())})

    def observe_survivors(self, generation, frame):
        if self.active is None:
            return
        retained = {canonical(s) for s in frame['smi']}
        summary = {'generation': generation, 'selected': self.active['selected'], 'branches': {}}
        for name, branch in self.active['branch_results'].items():
            for pair in branch['pairs']:
                self.edge_history[(self.active['selected'], pair)] = pair[0] in retained
            summary['branches'][name] = {k: branch[k] for k in ('evaluated', 'feasible')}
            summary['branches'][name]['retained'] = sum(p[0] in retained for p in branch['pairs'])
        self.completed.append(summary)
        self.save(self.output / f'outcomes_{generation:04d}.json', summary)

    def record_edge(self, parent, child, generation, row):
        key = (parent, child, generation)
        self.edge_records[key] = {'parent': parent, 'child': child, 'generation': generation,
                                  'edit': row.get('mutation_operator') if isinstance(row.get('mutation_operator'), str) else None,
                                  'branch': row.get('m3os_branch') if isinstance(row.get('m3os_branch'), str) else 'ordinary_ga'}

    def export(self):
        super().export()
        self.save(self.output / 'graph_evidence.json', {
            'nodes': [{'pair': n.pair, 'h2': float(n.row['h2_wt_percent']), 'dh': float(n.row['delta_H_per_H2']),
                       'sa': float(n.row['SA score']), 'feasible': n.feasible, 'matrix': self.book.evaluate(n.pair)}
                      for n in self.nodes.values()], 'edges': list(self.edge_records.values()),
            'rounds': self.completed, 'literature': list(self.evidence.values())})
