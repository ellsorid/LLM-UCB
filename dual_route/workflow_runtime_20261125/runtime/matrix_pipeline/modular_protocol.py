"""Matrix projections and selective protocol composition; no provider/API dependencies."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np

from .compat.pair_core import dense_from_state, state_from_dense, validate_mapping
from .local_structure import local_packet, recover_local, distances, role_distribution, entropy, ROLE_NAMES

VERSION = 'MATRIX_MODULES_V2'
PROTOCOL_DIR = Path(__file__).parent / 'protocols' / 'modular_v2'
MODULES = {
    'atoms': ('01_atoms.md', []),
    'bonds': ('02_bonds.md', ['atoms']),
    'regions': ('03_regions.md', ['atoms', 'bonds']),
    'local': ('04_local.md', ['atoms', 'bonds']),
    'changes': ('05_changes.md', ['atoms', 'bonds', 'local']),
    'statistics': ('06_statistics.md', []),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cells(matrix, ids, diagonal=True, bonds=True):
    """Indexed original tensor cells; parent coordinates only, upper triangle once."""
    ids = sorted(set(map(int, ids)))
    out = []
    for i in ids:
        if diagonal:
            out.append(dict(parent_row=i, parent_col=i, v=matrix[i, i].tolist()))
        if bonds:
            for j in ids:
                if j > i and matrix[i, j, 7] != 0:
                    out.append(dict(parent_row=i, parent_col=j, v=matrix[i, j].tolist()))
    return out


def view(matrix, ids):
    ids = sorted(set(map(int, ids))); inside = set(ids)
    crossing = [dict(inside_parent=i, outside_parent=int(j), v=matrix[i, j].tolist())
                for i in ids for j in np.flatnonzero(matrix[i, :, 7]) if int(j) not in inside]
    return dict(parent_ids=ids, cells=cells(matrix, ids), boundary_cells=crossing,
                parent_degree=(matrix[:, :, 7] != 0).sum(1)[ids].astype(int).tolist(),
                parent_ring=(matrix[:, :, 10].sum(1)[ids] > 0).astype(int).tolist())


def changes(pair, ids):
    out = []
    for i in sorted(ids):
        for j in sorted(ids):
            if j < i:
                continue
            for c in (range(7) if i == j else range(7, 12)):
                a, b = float(pair[0, i, j, c]), float(pair[1, i, j, c])
                if a != b:
                    out.append(dict(parent_row=i, parent_col=j, channel=c, before=a, after=b, delta=b-a))
    return out


def canonical_cycle(ids):
    ids = list(map(int, ids)); k = ids.index(min(ids)); forward = ids[k:] + ids[:k]
    return min(forward, [forward[0]] + forward[:0:-1])


def validate_local(packet, pair, centers):
    ids = packet['atom_ids']; radius = packet['radius']; n = len(pair[0])
    require(radius in [1, 2, 3], 'Unsupported local radius')
    require(len(ids) == len(set(ids)) and all(type(i) is int and 0 <= i < n for i in ids), 'Invalid local parent IDs')
    require(packet['center_parent_atom_ids'] == centers, 'Local center differs from selected centers')
    require(all(type(k) is int and 0 <= k < len(ids) for k in packet['center_local_atom_ids']), 'Bad local center index')
    require([ids[k] for k in packet['center_local_atom_ids']] == centers, 'Local center mapping mismatch')
    d = distances(pair[0], centers)
    require(set(ids) == set(np.flatnonzero(d <= radius)), 'Local radius membership mismatch')
    require(packet['hop_distance'] == d[ids].astype(int).tolist(), 'Local hop distances mismatch')
    require(np.array_equal(recover_local(packet), pair[:, ids][:, :, ids]), 'Local matrix differs from parent matrix')
    require(packet['delta_convention'] == 'rich_aligned_minus_lean', 'Unsupported local delta convention')
    observed_events = []
    channel = dict(charge=1, H_count=2, aromatic_atom=3, radicals=5, chirality=6,
                   bond_order=7, aromatic_bond=8, conjugation=9, ring_bond=10, bond_stereo=11)
    for event in packet['change_events']:
        p = [ids[k] for k in event['atoms']]
        i, j = (p[0], p[0]) if len(p) == 1 else sorted(p)
        observed_events.append((i, j, channel[event['kind']], float(event['before']), float(event['after'])))
    expected_events = [(e['parent_row'], e['parent_col'], e['channel'], e['before'], e['after']) for e in changes(pair, ids)]
    require(sorted(observed_events) == sorted(expected_events), 'Local change events mismatch')
    require(packet['boundary_mask'] == [any(int(j) not in ids for j in np.flatnonzero(pair[0, i, :, 7])) for i in ids], 'Local boundary mask mismatch')
    for k, state in enumerate(['lean', 'rich']):
        m = pair[k]; context = packet['parent_context'][state]
        require(context['degree'] == (m[:, :, 7] != 0).sum(1)[ids].astype(int).tolist(), 'Local parent degree mismatch')
        require(context['ring_atom'] == (m[:, :, 10].sum(1)[ids] > 0).astype(int).tolist(), 'Local parent ring mismatch')
        boundaries = sorted((ids[b['local_atom']], b['outside_parent_atom'], tuple(b['bond_fields'])) for b in packet['boundary_bonds'][state])
        expected = sorted((i, int(j), tuple(m[i, j, 7:].tolist())) for i in ids for j in np.flatnonzero(m[i, :, 7]) if int(j) not in ids)
        require(boundaries == expected, 'Local boundary bonds mismatch')
        counts, p = role_distribution(m, ids); saved = packet['roles'][state]
        require(saved['counts'] == counts.tolist() and np.allclose(saved['probabilities'], p, rtol=0, atol=1e-12), 'Local role distribution mismatch')
        require(np.isclose(saved['shannon_nats'], entropy(p), rtol=0, atol=1e-12), 'Local entropy mismatch')
    pl = np.asarray(packet['roles']['lean']['probabilities']); pr = np.asarray(packet['roles']['rich']['probabilities'])
    js = max(0., entropy((pl + pr) / 2) - (entropy(pl) + entropy(pr)) / 2)
    require(np.isclose(packet['role_JS_nats'], js, rtol=0, atol=1e-12), 'Local JS mismatch')


def decompose(record, *, coordinate_mode, centers=None):
    """Existing paired record -> six block kinds; input is never modified."""
    require(coordinate_mode in ['aligned', 'native'], 'Explicit aligned/native coordinate mode required')
    source = record.get('whole_structure', record.get('states'))
    require(source is not None, 'Missing whole_structure/states')
    raw = np.stack([dense_from_state(source[s]) for s in ['lean', 'rich']]); n = len(raw[0])
    require(n > 0, 'Empty molecule')
    mapping = record.get('mapping', {}).get('lean_to_rich')
    require(mapping is not None, 'Missing state mapping')
    require(len(mapping)==n and all(type(i) is int for i in mapping) and sorted(mapping)==list(range(n)), 'State map must be an integer bijection')
    if coordinate_mode == 'native':
        pair = np.stack([raw[0], validate_mapping(raw[0], raw[1], mapping)])
        source_rows = dict(lean=list(range(n)), rich=list(map(int, mapping)))
    else:
        validate_mapping(raw[0], raw[1], list(range(n)))
        pair = raw.copy(); source_rows = dict(lean=list(range(n)), rich=list(range(n)))
    for m in pair:
        require(np.array_equal(dense_from_state(state_from_dense(m)), m), 'Matrix serialization loss')
        require(np.array_equal(m, m.transpose(1, 0, 2)), 'Asymmetric input')
        require(all(m[i, j, 7] > 0 for i in range(n) for j in range(i+1, n) if np.any(m[i, j, 7:])), 'Bond flags without bond order')
    if centers is None:
        centers = record.get('center_parent_atom_ids')
        if centers is None:
            require('center_parent_atom_id' in record, 'Explicit center(s) required')
            centers = [record['center_parent_atom_id']]
    centers = sorted(set(centers))
    require(centers and all(type(i) is int and 0 <= i < n for i in centers), 'Invalid center(s)')
    saved_local = record.get('multiscale')
    local = copy.deepcopy(saved_local) if saved_local is not None else [local_packet(pair, centers, r) for r in [1, 2, 3]]
    require(sorted(p['radius'] for p in local) == [1, 2, 3], 'Exactly one local packet per radius required')
    for p in local:
        validate_local(p, pair, centers)
    local.sort(key=lambda p: p['radius'])
    molecule_id = str(record.get('record_id', record.get('reaction_id', digest(source))))
    bundle = dict(version=VERSION, molecule_id=molecule_id, source_hash=digest(record), coordinate_mode='parent_aligned',
                  input_coordinate_mode=coordinate_mode, parent_to_source_rows=source_rows,
                  input_mapping=copy.deepcopy(record['mapping']), parent_atom_count=n,
                  local_origin='existing_validated_packets' if saved_local is not None else 'existing_local_packet_function', blocks=[])
    def emit(kind, state, suffix, data):
        block_id=f'{kind}:{state}:{suffix}'
        block=dict(block_id=block_id, module=kind, molecule_id=molecule_id, state=state,
                   coordinates='parent_aligned', source_hash=bundle['source_hash'], data=data)
        block['block_hash'] = digest(block); bundle['blocks'].append(block)
        return block_id
    all_ids = list(range(n))
    for k, state in enumerate(['lean', 'rich']):
        m = pair[k]
        emit('atoms', state, 'all', dict(parent_ids=all_ids, diagonal_cells=cells(m, all_ids, bonds=False)))
        emit('bonds', state, 'all', dict(parent_ids=all_ids, bond_cells=cells(m, all_ids, diagonal=False), atom_context=cells(m, all_ids, bonds=False)))
        for region_kind, c in [('ring', 10), ('aromatic', 8), ('conjugated', 9)]:
            g = nx.Graph(); g.add_edges_from((i, j) for i in range(n) for j in range(i+1, n) if m[i, j, c] != 0)
            for index, component in enumerate(sorted(nx.connected_components(g), key=lambda z: sorted(z))):
                core = sorted(component); context = sorted(set(core) | {int(j) for i in core for j in np.flatnonzero(m[i, :, 7])})
                edges = sorted([sorted(e) for e in g.subgraph(core).edges()])
                cycles = sorted(canonical_cycle(r) for r in nx.cycle_basis(g.subgraph(core))) if region_kind == 'ring' else []
                emit('regions', state, f'{region_kind}{index}', dict(region_kind=region_kind, feature_channel=c,
                     core_parent_ids=core, feature_parent_bonds=edges, program_cycle_basis=cycles,
                     cycle_method='networkx.cycle_basis; independent basis, not all chemical rings', matrix_view=view(m, context)))
    for p in local:
        ids=p['atom_ids']; radius=p['radius']; local_ids={str(i): pid for i,pid in enumerate(ids)}
        for k,state in enumerate(['lean','rich']):
            emit('local', state, f'r{radius}', dict(radius=radius, center_parent_ids=centers,
                 source_local_to_parent=local_ids, parent_hop_distances={str(i):int(d) for i,d in zip(ids,p['hop_distance'])},
                 matrix_view=view(pair[k],ids), uncapped=True))
        emit('statistics', 'pair', f'r{radius}', dict(radius=radius, parent_ids=sorted(ids),
             linked_local_blocks=[f'local:lean:r{radius}',f'local:rich:r{radius}'],role_names=ROLE_NAMES,
             roles=copy.deepcopy(p['roles']), role_JS_nats=p['role_JS_nats'],log_base='e',quantity_origin='program_computed_structure_distribution'))
    events=changes(pair,all_ids)
    changed_ids=sorted({e[key] for e in events for key in ['parent_row','parent_col']})
    context=sorted(set(changed_ids)|{int(j) for i in changed_ids for j in np.flatnonzero(pair[0,i,:,7])})
    emit('changes','pair','all',dict(convention='rich_aligned_minus_lean',events=events,
         changed_parent_ids=changed_ids, context={s:view(pair[k],context) for k,s in enumerate(['lean','rich'])},
         local_change_events=[dict(radius=p['radius'],events=changes(pair,p['atom_ids'])) for p in local]))
    require(np.array_equal(reconstruct(bundle, native=True),raw), 'Decomposition cannot reconstruct native matrices exactly')
    return bundle


def reconstruct(bundle, *, native=False):
    n=bundle['parent_atom_count'];pair=np.zeros((2,n,n,12),dtype=np.float32)
    for k,state in enumerate(['lean','rich']):
        for kind,key in [('atoms','diagonal_cells'),('bonds','bond_cells')]:
            candidates=[b for b in bundle['blocks'] if b['module']==kind and b['state']==state]
            require(len(candidates)==1, 'Missing/duplicate reconstruction block')
            for cell in candidates[0]['data'][key]:
                i,j=cell['parent_row'],cell['parent_col'];pair[k,i,j]=cell['v'];pair[k,j,i]=cell['v']
        if native:
            inverse=np.argsort(bundle['parent_to_source_rows'][state]);pair[k]=pair[k,inverse][:,inverse]
    return pair


def compose(bundle, block_ids, *, child='read_selected'):
    """Load only selected payloads and their protocol definitions, not all data."""
    require(block_ids and len(block_ids)==len(set(block_ids)), 'Select unique nonempty block IDs')
    by_id={b['block_id']:b for b in bundle['blocks']}
    require(set(block_ids)<=set(by_id), 'Unknown block ID')
    selected=[by_id[k] for k in block_ids]; modules=[]
    def visit(kind):
        for dep in MODULES[kind][1]:visit(dep)
        if kind not in modules:modules.append(kind)
    for b in selected:
        require(digest({k:v for k,v in b.items() if k!='block_hash'})==b['block_hash'], 'Block integrity mismatch')
        visit(b['module'])
    registry=json.loads((PROTOCOL_DIR/'children.json').read_text(encoding='utf-8'))
    require(child in registry,'Unknown child protocol')
    task=registry[child]
    require(set(b['module'] for b in selected)<=set(task['allowed_modules']), 'Child cannot consume selected module')
    paths=[PROTOCOL_DIR/'00_mother.md']+[PROTOCOL_DIR/MODULES[k][0] for k in modules]
    text='\n\n'.join(p.read_text(encoding='utf-8') for p in paths)+'\n\nCHILD TASK:\n'+task['instruction']
    return dict(system=text,user=dict(protocol_version=VERSION,molecule_id=bundle['molecule_id'],blocks=copy.deepcopy(selected)),
                protocol_modules=modules,child=child,source_hash=bundle['source_hash'],
                protocol_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def pointer(value,path):
    require(isinstance(path,str) and path.startswith('/'),'Evidence path must be JSON pointer')
    for part in path[1:].split('/'):
        part=part.replace('~1','/').replace('~0','~')
        if isinstance(value,list):
            require(part.isdigit() and (part=='0' or not part.startswith('0')),'Invalid array index in evidence path')
            value=value[int(part)]
        else:value=value[part]
    return value


def same_value(a,b):
    if type(a) is bool or type(b) is bool:return type(a) is type(b) and a==b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return np.isfinite(a) and np.isfinite(b) and a==b
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(same_value(x,y) for x,y in zip(a,b))
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(same_value(a[k],b[k]) for k in a)
    return type(a) is type(b) and a==b


def validate_readout(result, request):
    """Validate linkage/quoted data only; cannot certify chemical interpretation."""
    errors=[];by_id={b['block_id']:b for b in request['user']['blocks']}
    if not isinstance(result,dict) or set(result)!={'observations','unresolved'}:return ['Expected observations and unresolved']
    if not isinstance(result['observations'],list) or not isinstance(result['unresolved'],list):return ['Expected output lists']
    if not all(isinstance(x,str) for x in result['unresolved']):errors.append('Unresolved entries must be strings')
    for index,row in enumerate(result['observations']):
        try:
            require(set(row)=={'block_id','state','parent_atom_ids','support','interpretation'},'Observation fields mismatch')
            block=by_id[row['block_id']]; require(row['state']==block['state'],'Observation state mismatch')
            require(isinstance(row['interpretation'],str) and bool(row['interpretation'].strip()),'Missing interpretation')
            require(isinstance(row['parent_atom_ids'],list) and len(row['parent_atom_ids'])==len(set(row['parent_atom_ids'])),'Invalid site list')
            # Parent IDs must actually occur in this payload, not merely be valid globally.
            allowed=set()
            def scan(x):
                if isinstance(x,dict):
                    for key,v in x.items():
                        if key in ['parent_row','parent_col','inside_parent','outside_parent'] and type(v) is int:allowed.add(v)
                        elif key in ['parent_ids','core_parent_ids','center_parent_ids','changed_parent_ids'] and isinstance(v,list):allowed.update(v)
                        scan(v)
                elif isinstance(x,list):
                    for y in x:scan(y)
            scan(block['data'])
            require(all(type(i) is int and i in allowed for i in row['parent_atom_ids']),'Site outside block scope')
            require(isinstance(row['support'],list) and bool(row['support']),'Exact input support required')
            for cite in row['support']:
                require(set(cite)=={'path','value'},'Evidence fields mismatch')
                actual=pointer(block['data'],cite['path'])
                require(same_value(actual,cite['value']),'Evidence value differs from matrix block')
        except (ValueError,KeyError,TypeError,IndexError) as e:errors.append(f'observations[{index}]: {e}')
    return errors


def main():
    p=argparse.ArgumentParser();p.add_argument('--record',required=True,type=Path);p.add_argument('--coordinates',required=True,choices=['aligned','native']);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();require(not a.output.exists(),'Refuse to overwrite decomposition output')
    bundle=decompose(json.loads(a.record.read_text(encoding='utf-8')),coordinate_mode=a.coordinates)
    a.output.mkdir(parents=True);(a.output/'bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for b in bundle['blocks']:
        name=b['block_id'].replace(':','_')+'.json'
        (a.output/name).write_text(json.dumps(compose(bundle,[b['block_id']]),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(blocks=len(bundle['blocks']),modules=sorted({b['module'] for b in bundle['blocks']}),API_calls=0)))


if __name__=='__main__':main()
