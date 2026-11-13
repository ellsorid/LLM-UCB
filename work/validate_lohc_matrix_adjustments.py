"""Extract matrix features and apply the shared score adjustments."""
from collections import defaultdict, deque
from statistics import mean

from validate_lohc_literature_scale import AXES


SCORE_AXES = AXES[:2]


def distances(graph, sources, excluded=None):
    steps = dict.fromkeys(sources, 0)
    queue = deque(steps)
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor != excluded and neighbor not in steps:
                steps[neighbor] = steps[node] + 1
                queue.append(neighbor)
    return steps


def features(matrix, include_neighborhood=False):
    # Every structural predicate uses aligned matrix entries or their atom labels.
    elements = {i: symbol for i, symbol, _, _ in matrix['atoms']}
    hydrogens = {i: h for i, _, h, _ in matrix['atoms']}
    graph = {i: {} for i in elements}
    for i, j, order in matrix['lean_upper']:
        graph[i][j] = graph[j][i] = order
    aromatic_degree = {i: sum(order == 1.5 for order in neighbors.values())
                       for i, neighbors in graph.items()}
    changed = [(i, j) for i, j, _ in matrix['delta_rich_minus_lean']]
    changed_atoms = {i for edge in changed for i in edge}
    cn = [(i, j) for i, j in changed
          if graph[i].get(j) == 1.5 and {elements[i], elements[j]} == {'C', 'N'}]
    cn_dist = distances(graph, {i for edge in cn for i in edge})
    edit_dist = distances(graph, changed_atoms)
    attachments = {key: set() for key in ('phenol', 'aryl_ether', 'methyl', 'aryl_amino')}
    oxygens = {'phenol': set(), 'aryl_ether': set()}
    for i, neighbors in graph.items():
        for j, order in neighbors.items():
            if order != 1 or elements[j] != 'C' or not aromatic_degree[j]:
                continue
            kind = (elements[i], len(neighbors), hydrogens[i])
            if kind == ('O', 1, 1):
                attachments['phenol'].add(j)
                oxygens['phenol'].add(i)
            elif kind == ('O', 2, 0):
                k, = neighbors.keys() - {j}
                if elements[k] == 'C' and neighbors[k] == 1 and k not in distances(graph, [j], excluded=i):
                    attachments['aryl_ether'].add(j)
                    oxygens['aryl_ether'].add(i)
            elif kind == ('C', 1, 3):
                attachments['methyl'].add(j)
            elif kind == ('N', 1, 2):
                attachments['aryl_amino'].add(j)

    def position(sites):
        if not sites:
            return 'none'
        reference, distance = ('CN', cn_dist) if cn else ('edit', edit_dist)
        steps = min(distance.get(site, 3) for site in sites)
        return f'{reference}:{steps if steps < 3 else "3+"}'

    def count_bin(count):
        return str(int(count)) if count < 3 else '3+'

    summary = {
        **{f'{k}_site': position(sites) for k, sites in attachments.items()},
        'phenol_count': count_bin(len(oxygens['phenol'])),
        'aryl_ether_count': count_bin(len(oxygens['aryl_ether'])),
        'cn_ring_edit': count_bin(len(cn)),
        'fused_aromatic_edit': count_bin(sum(graph[i].get(j) == 1.5 and
            max(aromatic_degree[i], aromatic_degree[j]) >= 3 for i, j in changed)),
        'guaiacol_like': 'yes' if any(graph[i].get(j) == 1.5 for i in attachments['phenol']
                                    for j in attachments['aryl_ether']) else 'no',
    }
    if include_neighborhood:
        ring_n = distances(graph, {i for i in graph if elements[i] == 'N' and aromatic_degree[i]})
        fusion = distances(graph, {i for i in graph if aromatic_degree[i] >= 3})

        def band(distances_to_site, site):
            if site not in distances_to_site:
                return '无参照'
            return str(min(distances_to_site[site], 3)).replace('3', '3+')

        for name, sites in attachments.items():
            summary[f'{name}_neighborhood'] = tuple(sorted(
                '/'.join((band(cn_dist or edit_dist, i), band(ring_n, i), band(fusion, i))) for i in sites))
    return summary


def apply_rules(row, rules):
    matched = defaultdict(list)
    for rule in rules:
        if row['features'].get(rule['feature']) == rule['value']:
            matched[(rule['axis'], rule['ref'])].append(rule)
    selected = [min(group, key=lambda r: (-abs(r['delta']), r['delta'], r['feature'], r['value']))
                for group in matched.values()]
    result = dict(row)
    if row['potential'] is None:
        return {**result, 'adjusted': None, 'hits': []}
    # Preserve the previous equal weighting of the two literature-anchored axes.
    original = row['axis_intervals']
    clipped = {}
    for axis in SCORE_AXES:
        shift = max(-8, min(8, sum(r['delta'] for r in selected if r['axis'] == axis)))
        clipped[axis] = [max(0, min(100, point + shift)) for point in original[axis]]
    return {**result, 'adjusted': [mean(clipped[a][i] for a in SCORE_AXES) for i in (0, 1)],
            'hits': selected, 'adjusted_axes': clipped}
