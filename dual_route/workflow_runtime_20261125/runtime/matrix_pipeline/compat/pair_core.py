"""Experimental shared pair core and explicit legacy views; no target values in structure."""
import copy
import json
import numpy as np

VERSION='LOHC_PAIR_CORE_DRAFT_0.1'
ATOM_FIELDS=['atomic_number','formal_charge','total_hydrogens','aromatic_atom','isotope_label','radical_electrons','atom_chirality']
BOND_FIELDS=['i','j','bond_order','aromatic_bond','conjugated_bond','ring_bond','bond_stereochemistry']


def state_from_dense(matrix):
    matrix=np.asarray(matrix)
    n=len(matrix)
    if matrix.shape!=(n,n,12) or not np.isfinite(matrix).all():raise ValueError('Invalid dense state')
    diag=np.arange(n)
    atoms=matrix[diag,diag,:7]
    if not np.equal(atoms,np.rint(atoms)).all():raise ValueError('Non-integral atom feature')
    edges=[]
    for i,j in zip(*np.triu_indices(n,1)):
        if np.any(matrix[i,j,7:]!=0):
            edges.append([int(i),int(j),float(matrix[i,j,7]),*map(int,matrix[i,j,8:])])
    result={'atom_fields':ATOM_FIELDS.copy(),'atoms':atoms.astype(np.int64).tolist(),
            'bond_fields':BOND_FIELDS.copy(),'edges':edges}
    if not np.array_equal(dense_from_state(result),matrix):raise ValueError('Unexpected layout or asymmetry')
    return result


def dense_from_state(state):
    if state['atom_fields']!=ATOM_FIELDS or state['bond_fields']!=BOND_FIELDS:raise ValueError('Field mismatch')
    atoms=np.asarray(state['atoms'],dtype=np.float32)
    n=len(atoms)
    if atoms.shape!=(n,7):raise ValueError('Invalid atoms')
    result=np.zeros((n,n,12),dtype=np.float32)
    result[np.arange(n),np.arange(n),:7]=atoms
    seen=set()
    for row in state['edges']:
        i,j=map(int,row[:2])
        if not 0<=i<j<n or (i,j) in seen or len(row)!=7:raise ValueError('Invalid edge')
        seen.add((i,j));result[i,j,7:]=row[2:];result[j,i,7:]=row[2:]
    return result


def validate_mapping(lean,rich,mapping):
    p=np.asarray(mapping,dtype=int);n=len(lean)
    if len(rich)!=n or p.shape!=(n,) or not np.array_equal(np.sort(p),np.arange(n)):
        raise ValueError('Map must be a complete bijection')
    aligned=rich[p[:,None],p[None,:]]
    if not np.array_equal(np.diag(lean[:,:,0]),np.diag(aligned[:,:,0])):raise ValueError('Mapped element mismatch')
    if not np.array_equal(lean[:,:,7]!=0,aligned[:,:,7]!=0):raise ValueError('Mapped skeleton mismatch')
    return aligned


def legacy_gnn_view(core):
    matrix=dense_from_state(core['states']['lean'])
    for i,j,c,value in core.get('extensions',{}).get('legacy_lean_patch',[]):
        matrix[int(i),int(j),int(c)]=value
    return matrix


def delta_view(core):
    lean=dense_from_state(core['states']['lean']);rich=dense_from_state(core['states']['rich'])
    return validate_mapping(lean,rich,core['mapping']['lean_to_rich'])-lean


def legacy_llm_view(core,properties):
    """Reproduce original canonical-order AR/AL/P with its explicitly stored mapping policy."""
    matrices={k:dense_from_state(core['states'][k]) for k in ('lean','rich')}
    orders=core['provenance']['canonical_orders']
    lorder=np.asarray(orders['lean'],dtype=int);rorder=np.asarray(orders['rich'],dtype=int)
    n=len(lorder);invr=np.argsort(rorder)
    native_map=np.asarray(core['extensions']['legacy_llm_mapping']['lean_to_rich'],dtype=int)
    mapping=invr[native_map[lorder]]
    lean=matrices['lean'][lorder[:,None],lorder[None,:]]
    rich=matrices['rich'][rorder[:,None],rorder[None,:]]
    delta=validate_mapping(lean,rich,mapping)[:,:,7]-lean[:,:,7]
    transform=np.zeros((n,n));transform[np.arange(n),mapping]=1
    def state(matrix):
        atoms=matrix[np.arange(n),np.arange(n),:7]
        return np.column_stack((matrix[:,:,7],atoms[:,[0,1,2,4,3]])).tolist()
    p=np.column_stack((transform,delta,np.zeros((n,3)))).tolist()
    p.append([None]*(2*n)+list(properties))
    return {'AR':state(rich),'AL':state(lean),'P':p}


def invariant_features(core):
    """Audit features only; not a learned QD cell or a stability score."""
    delta=delta_view(core);lean=dense_from_state(core['states']['lean'])
    values=[]
    for c in range(12):
        m=delta[:,:,c].astype(np.float64)
        values.extend([float(np.trace(m)),float(np.sum(m*m))])
    for c in (7,8,9,10):
        eigen=np.linalg.eigvalsh(lean[:,:,c].astype(np.float64))
        values.extend(eigen.tolist())
    values.append(float(np.trace(lean[:,:,8]@lean[:,:,9])))
    return np.asarray(values)


def permute_core(core,lean_order,rich_order):
    result=copy.deepcopy(core)
    orders={'lean':np.asarray(lean_order),'rich':np.asarray(rich_order)}
    for state,order in orders.items():
        m=dense_from_state(core['states'][state])
        result['states'][state]=state_from_dense(m[order[:,None],order[None,:]])
    inverse_r=np.argsort(orders['rich'])
    result['mapping']['lean_to_rich']=inverse_r[np.asarray(core['mapping']['lean_to_rich'])[orders['lean']]].tolist()
    # These provenance caches/legacy patches refer to native indices. The
    # invariant-feature test consumes only the updated structural facts.
    result['extensions']={}
    result['provenance']={**result['provenance'],'permuted_for_audit':True}
    return result


def serialize(core):
    return json.dumps(core,ensure_ascii=False,separators=(',',':'),allow_nan=False)
