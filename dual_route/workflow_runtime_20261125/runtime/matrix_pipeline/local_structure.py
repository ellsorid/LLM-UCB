"""Lossless indexed local views of the existing LOHC 12-channel pair core.

Numeric descriptors are separate from graph facts and from property labels.
Local induced graphs retain original H counts; they are not capped molecules.
"""
from collections import Counter, deque
import hashlib
import json

import numpy as np
from rdkit import Chem

from .compat.pair_core import state_from_dense, dense_from_state

VERSION = 'LOHC_LOCAL_VIEW_0.1'
RADII = (1, 2, 3)
ROLE_NAMES = [f'{z}_ring{r}_arom{a}' for z in ['C','N','O','F','other'] for r in range(2) for a in range(2)]
SMARTS = {
    'N_with_H': '[#7;!H0]',
    'N_H0_degree3': '[#7;H0;D3]',
    'aromatic_N': '[n]',
    'carbonyl_C': '[C;X3]=[O;X1]',
    'nitrile_C': '[C]#[N]',
    'amide_N': '[N]-[C](=[O])',
    'alpha_carbonyl_C': '[C;X4]-[C;X3](=[O;X1])',
    'allylic_C': '[C;X4]-[C]=[C]',
}
PATTERNS = {k:Chem.MolFromSmarts(v) for k,v in SMARTS.items()}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def distances(matrix, centers):
    adj=matrix[:,:,7]!=0
    out=np.full(len(matrix),np.inf);queue=deque()
    for i in centers:out[i]=0;queue.append(int(i))
    while queue:
        i=queue.popleft()
        for j in np.flatnonzero(adj[i]):
            if np.isinf(out[j]):out[j]=out[i]+1;queue.append(int(j))
    return out


def entropy(p):
    p=np.asarray(p,dtype=float);v=p[p>0]
    return float(-np.sum(v*np.log(v)))


def role_distribution(matrix, ids):
    z=np.diag(matrix[:,:,0])[ids];arom=np.diag(matrix[:,:,3])[ids].astype(int)
    ring=(matrix[:,:,10].sum(1)[ids]>0).astype(int)
    element=np.full(len(ids),4)
    for i,value in enumerate((6,7,8,9)):element[z==value]=i
    counts=np.bincount(4*element+2*ring+arom,minlength=20)
    p=counts/max(len(ids),1)
    return counts,p


def local_packet(pair, centers, radius, source_row=None, mapping_status=None):
    pair=np.asarray(pair);n=pair.shape[1]
    if pair.shape!=(2,n,n,12):raise ValueError('Expected aligned [2,n,n,12]')
    if not np.array_equal(pair[0,:,:,7]!=0,pair[1,:,:,7]!=0):raise ValueError('Current LOHC view requires shared heavy-atom topology')
    if radius not in RADII:raise ValueError('Unsupported radius')
    centers=sorted(set(map(int,centers)))
    if not centers or any(i<0 or i>=n for i in centers):raise ValueError('Invalid centers')
    d=distances(pair[0],centers);ids=np.flatnonzero(d<=radius)
    inside=set(map(int,ids));lookup={int(j):i for i,j in enumerate(ids)}
    local=pair[:,ids][:,:,ids]
    states={};outside={};parent_context={};roles={}
    for s,name in enumerate(('lean','rich')):
        states[name]=state_from_dense(local[s])
        outside[name]=[dict(local_atom=lookup[i],outside_parent_atom=int(j),bond_fields=pair[s,i,j,7:].tolist())
                       for i in ids for j in np.flatnonzero(pair[s,i,:,7]) if int(j) not in inside]
        parent_context[name]=dict(degree=(pair[s,:,:,7]!=0).sum(1)[ids].astype(int).tolist(),
                                  ring_atom=(pair[s,:,:,10].sum(1)[ids]>0).astype(int).tolist())
        c,p=role_distribution(pair[s],ids)
        roles[name]=dict(counts=c.tolist(),probabilities=p.tolist(),shannon_nats=entropy(p))
    dl=pair[1]-pair[0]
    # Delta labels are events; category IDs themselves are never numeric energies.
    events=[]
    for i in ids:
        for c,name in [(1,'charge'),(2,'H_count'),(3,'aromatic_atom'),(5,'radicals'),(6,'chirality')]:
            if dl[i,i,c]:events.append(dict(kind=name,atoms=[lookup[int(i)]],before=float(pair[0,i,i,c]),after=float(pair[1,i,i,c])))
        for j in ids:
            if j<=i:continue
            for c,name in [(7,'bond_order'),(8,'aromatic_bond'),(9,'conjugation'),(10,'ring_bond'),(11,'bond_stereo')]:
                if dl[i,j,c]:events.append(dict(kind=name,atoms=[lookup[int(i)],lookup[int(j)]],before=float(pair[0,i,j,c]),after=float(pair[1,i,j,c])))
    pl=np.asarray(roles['lean']['probabilities']);pr=np.asarray(roles['rich']['probabilities'])
    return dict(schema_version=VERSION,source_row=source_row,parent_atom_count=n,radius=radius,
                atom_ids=ids.tolist(),center_parent_atom_ids=centers,
                center_local_atom_ids=[lookup[i] for i in centers],hop_distance=d[ids].astype(int).tolist(),
                states=states,parent_context=parent_context,boundary_bonds=outside,
                boundary_mask=[any(int(j) not in inside for j in np.flatnonzero(pair[0,i,:,7])) for i in ids],
                mapping_status=mapping_status,delta_convention='rich_aligned_minus_lean',change_events=events,
                roles=roles,role_JS_nats=max(0.,entropy((pl+pr)/2)-(entropy(pl)+entropy(pr))/2),
                semantics='Uncapped induced subgraph; parent atom H counts and parent ring/degree context retained; not an isolated molecule.')


def recover_local(packet):
    return np.stack([dense_from_state(packet['states'][name]) for name in ('lean','rich')])


def walk_counts(matrix, centers, max_length=3):
    """Exact type-masked matrix-product walk counts, with backtracking allowed."""
    n=len(matrix);atoms=[tuple(map(float,matrix[i,i,:7])) for i in range(n)]
    counts=Counter()
    def visit(i,depth,seq):
        counts[(depth,seq)]+=1
        if depth==max_length:return
        for j in np.flatnonzero(matrix[i,:,7]):
            visit(int(j),depth+1,seq+(tuple(map(float,matrix[i,j,7:])),atoms[j]))
    for c in centers:visit(int(c),0,(atoms[c],))
    return counts


def walk_signature(counts,radius):
    values=[(str(k),int(v)) for k,v in counts.items() if k[0]<=radius]
    return digest(sorted(values))


def matrix_roles(matrix):
    """Explicit structural predicates, independently checked against SMARTS.

    These describe local environments; they do not assert degradation mechanisms.
    """
    n=len(matrix);z=np.diag(matrix[:,:,0]);h=np.diag(matrix[:,:,2]);ar=np.diag(matrix[:,:,3])
    b=matrix[:,:,7];degree=(b!=0).sum(1);x=degree+h
    carbonyl=np.array([z[i]==6 and not ar[i] and x[i]==3 and any(z[j]==8 and not ar[j] and x[j]==1 and b[i,j]==2 for j in range(n)) for i in range(n)])
    result=[]
    for i in range(n):
        nbr=np.flatnonzero(b[i])
        result.append(dict(
            N_with_H=bool(z[i]==7 and h[i]>0),
            N_H0_degree3=bool(z[i]==7 and h[i]==0 and degree[i]==3),
            aromatic_N=bool(z[i]==7 and ar[i]),
            carbonyl_C=bool(carbonyl[i]),
            nitrile_C=bool(z[i]==6 and not ar[i] and any(z[j]==7 and not ar[j] and b[i,j]==3 for j in nbr)),
            amide_N=bool(z[i]==7 and not ar[i] and any(z[j]==6 and not ar[j] and b[i,j]==1 and any(z[k]==8 and not ar[k] and b[j,k]==2 for k in range(n)) for j in nbr)),
            alpha_carbonyl_C=bool(z[i]==6 and not ar[i] and x[i]==4 and any(b[i,j]==1 and carbonyl[j] for j in nbr)),
            allylic_C=bool(z[i]==6 and not ar[i] and x[i]==4 and any(z[j]==6 and not ar[j] and b[i,j]==1 and any(z[k]==6 and not ar[k] and b[j,k]==2 for k in range(n)) for j in nbr)),
        ))
    return result


def rdkit_roles(mol):
    result=[{name:False for name in SMARTS} for _ in mol.GetAtoms()]
    for name,pattern in PATTERNS.items():
        for match in mol.GetSubstructMatches(pattern):result[match[0]][name]=True
    return result


def rooted_fragment_key(mol,center,radius):
    # Canonical text is an audit index only; no property value is encoded.
    d=Chem.GetDistanceMatrix(mol)[center]
    ids=np.flatnonzero(d<=radius).tolist()
    bonds=[b.GetIdx() for b in mol.GetBonds() if b.GetBeginAtomIdx() in ids and b.GetEndAtomIdx() in ids]
    return Chem.MolFragmentToSmiles(mol,atomsToUse=ids,bondsToUse=bonds,rootedAtAtom=int(center),
                                  canonical=True,allHsExplicit=True,allBondsExplicit=True)


def explicit_h_view(matrix):
    """Optional H-node view with parent heavy-atom linkage; no stereospecific H claim."""
    from .core import to_mol,from_mol
    mol=Chem.AddHs(to_mol(matrix));view=from_mol(mol)
    parent=[]
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()!=1:parent.append(atom.GetIdx())
        else:
            nbr=list(atom.GetNeighbors());parent.append(nbr[0].GetIdx() if len(nbr)==1 else None)
    return dict(matrix_shape=list(view.shape),state=state_from_dense(view),heavy_atom_count=len(matrix),
                parent_heavy_atom_ids=parent,scope='Equivalent H groups only; no pro-R/pro-S assignment without validated stereo.')
