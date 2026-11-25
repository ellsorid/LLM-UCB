"""Deterministic structural projections of the original 12-channel matrix.

Extraction uses numerical graph rules, not RDKit perception or LLM output.
RDKit is reserved for the independent offline audit in audit_matrix_structure.
"""
from collections import Counter
from itertools import combinations
import hashlib,json
import networkx as nx
import numpy as np
from .compat.pair_core import ATOM_FIELDS,BOND_FIELDS,dense_from_state,state_from_dense,validate_mapping

VERSION='MATRIX_STRUCTURE_V1'
SYMBOLS=('X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()
RULES={
 'nitrile':'C triple-bonded to N; members C,N',
 'carbonyl':'C double-bonded to O; members C,O; includes amide carbonyl',
 'amide':'C(=O)-N; members C,O,N for each match',
 'amine':'neutral non-aromatic N; all incident heavy bonds single to C; at least one C; no neighbor carbonyl C; members N and all immediate heavy C neighbors',
 'hydroxyl':'O with original attached H>=1, single-bonded to C; members O,C; structural pattern, not a reactivity label',
}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_matrix(matrix):
    m=np.asarray(matrix,dtype=float);n=len(m)
    if m.shape!=(n,n,12) or n==0 or not np.isfinite(m).all():raise ValueError('Expected finite nonempty NxNx12')
    if not np.array_equal(m,m.transpose(1,0,2)):raise ValueError('Asymmetric matrix')
    for i in range(n):
        a=m[i,i]
        if np.any(a[7:]) or not np.equal(a[:7],np.rint(a[:7])).all():raise ValueError('Invalid diagonal layout/integer attributes')
        if not 1<=a[0]<len(SYMBOLS) or min(a[2],a[4],a[5])<0 or a[3] not in [0,1] or a[6] not in range(6):raise ValueError('Invalid atom attribute')
        for j in range(i+1,n):
            b=m[i,j]
            if np.any(b[:7]) or b[7]<0 or (b[7]==0 and np.any(b[8:])):raise ValueError('Invalid off-diagonal layout')
            if any(x not in [0,1] for x in b[8:11]) or b[11] not in range(7):raise ValueError('Invalid bond flags/stereo')
    # Existing serializer must reproduce ALL channels, including original stereo codes.
    if not np.array_equal(dense_from_state(state_from_dense(m)),m):raise ValueError('Serialization loss')
    return m


def graph(matrix):
    m=validate_matrix(matrix);g=nx.Graph()
    for i in range(len(m)):g.add_node(i,values=m[i,i].tolist())
    for i in range(len(m)):
        for j in range(i+1,len(m)):
            if m[i,j,7]>0:g.add_edge(i,j,values=m[i,j].tolist())
    return g


def cell(matrix,i,j):
    i,j=sorted([int(i),int(j)])
    return dict(parent_row=i,parent_col=j,v=matrix[i,j].tolist())


def support(matrix,ids,bonds):
    return [cell(matrix,i,i) for i in sorted(set(ids))]+[cell(matrix,*e) for e in sorted({tuple(sorted(e)) for e in bonds})]


def canonical_cycle(cycle):
    c=list(cycle);variants=[]
    for seq in [c,list(reversed(c))]:
        for i in range(len(seq)):variants.append(seq[i:]+seq[:i])
    return min(variants)


def functional_groups(matrix,g):
    m=matrix;z=lambda i:int(m[i,i,0]);bo=lambda i,j:m[i,j,7]
    carbonyl={i:[j for j in g[i] if z(j)==8 and bo(i,j)==2] for i in g if z(i)==6}
    hits=set()
    def add(kind,ids):hits.add((kind,tuple(sorted(ids))))
    for c,oxygens in carbonyl.items():
        for o in oxygens:
            add('carbonyl',[c,o])
            for n in g[c]:
                if z(n)==7 and bo(c,n)==1:add('amide',[c,o,n])
    for i in g:
        if z(i)==6:
            for j in g[i]:
                if z(j)==7 and bo(i,j)==3:add('nitrile',[i,j])
        if z(i)==8 and m[i,i,2]>=1:
            for j in g[i]:
                if z(j)==6 and bo(i,j)==1:add('hydroxyl',[i,j])
        if z(i)==7 and m[i,i,1]==0 and not m[i,i,3] and len(g[i]) and all(z(j)==6 and bo(i,j)==1 for j in g[i]) and not any(carbonyl.get(j) for j in g[i]):
            add('amine',[i,*g[i]])
    result=[]
    for k,(kind,members) in enumerate(sorted(hits)):
        ids=set(members)
        # Include one-hop evidence needed for exclusion rules (e.g. amide vs amine).
        context=ids|{j for i in ids for j in g[i]}
        edges=[(a,b) for a,b in g.edges if a in context and b in context]
        result.append(dict(group_id=f'G{k}',kind=kind,parent_ids=list(members),
            member_elements=[SYMBOLS[int(m[i,i,0])] for i in members],rule=RULES[kind],
            provenance='program_matrix_pattern',evidence=support(m,context,edges)))
    return result


def skeleton_label(zs,aromatic,saturated):
    if sorted(zs)==[6,6,6,6,8]:return 'furan' if aromatic else 'tetrahydrofuran' if saturated else 'other'
    if sorted(zs)==[6,6,6,6,6,7]:return 'pyridine' if aromatic else 'piperidine' if saturated else 'other'
    return 'other'


def extract_state(matrix,*,state,molecule_id):
    if state not in ['lean','rich']:raise ValueError('Explicit lean/rich state required')
    m=validate_matrix(matrix);g=graph(m);n=len(m)
    bridge_set={tuple(sorted(e)) for e in nx.bridges(g)}
    cyclic_edges=sorted(tuple(sorted(e)) for e in g.edges if tuple(sorted(e)) not in bridge_set)
    ring_atoms=sorted({i for e in cyclic_edges for i in e});rg=nx.Graph();rg.add_edges_from(cyclic_edges)
    systems=[]
    for k,component in enumerate(sorted([sorted(c) for c in nx.connected_components(rg)])):
        edges=[list(e) for e in cyclic_edges if e[0] in component and e[1] in component]
        systems.append(dict(system_id=f'R{k}',parent_ids=component,parent_bonds=edges,cycle_rank=len(edges)-len(component)+1))
    rings=[]
    for k,sequence in enumerate(sorted(canonical_cycle(c) for c in nx.cycle_basis(g))):
        edges=sorted({tuple(sorted([sequence[i],sequence[(i+1)%len(sequence)]])) for i in range(len(sequence))})
        ids=sorted(sequence);zs=[int(m[i,i,0]) for i in ids]
        aromatic=all(m[i,i,3]==1 for i in ids) and all(m[a,b,8]==1 for a,b in edges)
        saturated=all(m[a,b,7]==1 for a,b in edges)
        rings.append(dict(ring_id=f'C{k}',parent_ids=ids,closed_sequence=sequence+[sequence[0]],parent_bonds=[list(e) for e in edges],
            atomic_numbers=zs,aromatic_from_input_flags=aromatic,saturated_from_bond_orders=saturated,
            local_skeleton=skeleton_label(zs,aromatic,saturated),evidence=support(m,ids,edges)))
    relations=[]
    for a,b in combinations(rings,2):
        nodes=sorted(set(a['parent_ids'])&set(b['parent_ids']));edges=sorted(set(map(tuple,a['parent_bonds']))&set(map(tuple,b['parent_bonds'])))
        kind='disjoint'
        if len(nodes)==1:kind='spiro_junction'
        elif len(nodes)==2 and len(edges)==1:kind='shared_edge_junction'
        elif nodes:kind='shared_path_or_complex_junction'
        relations.append(dict(cycle_a=a['ring_id'],cycle_b=b['ring_id'],shared_parent_ids=nodes,shared_parent_bonds=[list(e) for e in edges],basis_relation=kind))
    nonring=sorted([sorted(c) for c in nx.connected_components(g.subgraph(set(g)-set(ring_atoms)))])
    units=[dict(unit_id=r['system_id'],kind='ring_system',parent_ids=r['parent_ids']) for r in systems]
    units += [dict(unit_id=f'F{k}',kind='nonring_component',parent_ids=ids) for k,ids in enumerate(nonring)]
    owner={i:u['unit_id'] for u in units for i in u['parent_ids']}
    links=[dict(unit_a=owner[a],unit_b=owner[b],parent_bond=[a,b],evidence=cell(m,a,b)) for a,b in sorted(tuple(sorted(e)) for e in g.edges) if owner[a]!=owner[b]]
    groups=functional_groups(m,g);paths=dict(nx.all_pairs_shortest_path_length(g));group_relations=[]
    for a,b in combinations(groups,2):
        distances=[paths[i][j] for i in a['parent_ids'] for j in b['parent_ids'] if j in paths[i]]
        group_relations.append(dict(group_a=a['group_id'],group_b=b['group_id'],minimum_bond_distance=min(distances) if distances else None,disconnected=not bool(distances)))
    counts=Counter()
    for i in g:counts[SYMBOLS[int(m[i,i,0])]]+=1;counts['H']+=int(m[i,i,2])
    order=(['C','H']+sorted(set(counts)-{'C','H'})) if counts['C'] else sorted(counts)
    formula=''.join(s+(str(counts[s]) if counts[s]!=1 else '') for s in order if counts[s])
    expected_ringflags={e for e in cyclic_edges};actual_ringflags={(a,b) for a,b in g.edges if m[a,b,10]}
    atoms=[dict(parent_id=i,element=SYMBOLS[int(m[i,i,0])],**dict(zip(ATOM_FIELDS,map(int,m[i,i,:7]))),evidence=cell(m,i,i)) for i in g]
    bonds=[dict(parent_ids=[a,b],**dict(zip(BOND_FIELDS[2:],m[a,b,7:].tolist())),evidence=cell(m,a,b)) for a,b in sorted(tuple(sorted(e)) for e in g.edges)]
    partition=[i for u in units for i in u['parent_ids']]
    return dict(version=VERSION,molecule_id=molecule_id,state=state,coordinates='parent_aligned',matrix_sha256=digest(m.tolist()),
        provenance=dict(extractor='program_only; numpy/networkx; no LLM',aromaticity='original channel flags read, not reperceived',
            cycles='networkx.cycle_basis; independent fundamental cycles, not all rings or a unique chemical ring decomposition',
            ring_systems='connected components of all non-bridge edges',stereochemistry='original atom/bond codes preserved; no new assignment'),
        atoms=atoms,bonds=bonds,cycles=rings,ring_systems=systems,basis_cycle_relations=relations,functional_groups=groups,
        whole=dict(formula_from_elements_and_attached_H=formula,net_formal_charge=int(sum(m[i,i,1] for i in g)),connected_components=[sorted(c) for c in nx.connected_components(g)],
            cycle_rank=g.number_of_edges()-n+nx.number_connected_components(g),ring_parent_ids=ring_atoms,ring_parent_bonds=[list(e) for e in cyclic_edges],
            units=units,unit_connections=links,group_relations=group_relations),
        checks=dict(atom_partition_exact=sorted(partition)==list(range(n)) and len(set(partition))==n,
            original_ring_flags_agree_with_topology=actual_ringflags==expected_ringflags,
            ring_flag_symmetric_difference=[list(e) for e in sorted(actual_ringflags^expected_ringflags)]),
        limits=['Five explicitly defined functional patterns only; overlapping carbonyl/amide retained.',
            'Formula is an element/attached-H count; isotope mass labels and charge are retained separately.',
            'Basis junction labels describe intersections of selected cycles, not universal fused/bridged nomenclature.',
            'No molecule naming, energy, stability, reversibility, or evidence-derived property score.'])


def group_key(group):return group['kind'],tuple(group['parent_ids'])


def extract_pair(record,*,coordinate_mode):
    if coordinate_mode not in ['aligned','native']:raise ValueError('Declare aligned or native coordinates')
    source=record.get('whole_structure',record.get('states'))
    if source is None:raise ValueError('Missing states')
    lean=validate_matrix(dense_from_state(source['lean']));native_rich=validate_matrix(dense_from_state(source['rich']))
    mapping=record.get('mapping',{}).get('lean_to_rich')
    if mapping is None:raise ValueError('Explicit complete state mapping required')
    if not isinstance(mapping,list) or len(mapping)!=len(lean) or any(type(i) is not int for i in mapping) or sorted(mapping)!=list(range(len(lean))):raise ValueError('Invalid mapping')
    used=mapping if coordinate_mode=='native' else list(range(len(lean)))
    rich=validate_mapping(lean,native_rich,used)
    identity=record.get('record_id','unnamed')
    states={s:extract_state(m,state=s,molecule_id=identity) for s,m in [('lean',lean),('rich',rich)]}
    events=[]
    for i in range(len(lean)):
        for j in range(i,len(lean)):
            for c in np.flatnonzero(lean[i,j]!=rich[i,j]):
                events.append(dict(parent_row=i,parent_col=j,channel=int(c),before=float(lean[i,j,c]),after=float(rich[i,j,c]),delta=float(rich[i,j,c]-lean[i,j,c])))
    lg={group_key(g) for g in states['lean']['functional_groups']};rg={group_key(g) for g in states['rich']['functional_groups']}
    def group_rows(keys):return [dict(kind=k,parent_ids=list(ids)) for k,ids in sorted(keys)]
    return dict(version=VERSION,molecule_id=identity,input_hash=digest(record),input_coordinate_mode=coordinate_mode,
        alignment=dict(declared_lean_to_rich=mapping,applied_lean_to_rich=used,parent_to_source_rows=dict(lean=list(range(len(lean))),rich=used),
            scope='Explicit mapped shared-skeleton pairs only; mapping uniqueness not asserted; aligned inputs are not aligned again.'),
        states=states,changes=dict(convention='rich_aligned_minus_lean',events=events,functional_groups_unchanged=group_rows(lg&rg),functional_groups_lost=group_rows(lg-rg),functional_groups_gained=group_rows(rg-lg),
            interpretation='Structural differences and pattern membership changes only; categorical code subtraction is bookkeeping, not energy or mechanism.'))


def main():
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path,help='Existing JSON record with whole_structure/states and mapping')
    p.add_argument('--coordinate-mode',required=True,choices=['aligned','native'])
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Refuse overwrite')
    record=json.loads(a.input.read_text(encoding='utf-8'));record=record.get('record',record)
    result=extract_pair(record,coordinate_mode=a.coordinate_mode)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(a.output),states=list(result['states']),API_calls=0)),flush=True)


if __name__=='__main__':main()
