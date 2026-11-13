"""Build the three-matrix wire format and verify its bond changes."""
import random


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
