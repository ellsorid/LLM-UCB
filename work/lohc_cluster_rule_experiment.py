"""Graph features used by the published matrix score scale."""
from rdkit import Chem

from evaluate_lohc_spectral_scores import spectral_features
from lohc_typed_reaction_graph import describe


def jsonable_features(pair, include_matrix=False):
    lean, rich = [Chem.MolFromSmiles(s) for s in pair]
    matrix, _, typed, delta = describe(pair)
    rings = [set(r) for r in lean.GetRingInfo().BondRings()]
    fused = any(r & s for i, r in enumerate(rings) for s in rings[i+1:])
    n = sum(v for key, v in typed.items() if key.startswith('motif:lean:aromatic_n:0'))
    oxygen = [name for name in ('furan_o', 'aryl_ether', 'phenol') if
              any(key in typed for key in (f'motif:lean:{name}:0', f'motif:lean:{name}:1'))]
    result = {'global': spectral_features((lean, rich), True).tolist(), 'delta': delta, 'typed': typed,
              'signature': [min(n, 2), '+'.join(oxygen) or 'none', bool(fused)],
              'changed_edges': len(matrix['delta_rich_minus_lean'])}
    if include_matrix:
        result['matrix'] = matrix
    return result
