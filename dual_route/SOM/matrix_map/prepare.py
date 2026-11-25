"""Prepare variable-size paired matrices; identities/labels never enter the encoder."""
import os, sys, json, hashlib, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse.csgraph import shortest_path
from sklearn.model_selection import GroupShuffleSplit
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
ROOT = (Path(__file__).resolve().parents[2] / 'workflow_runtime_20261125' / 'runtime')
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from matrix_pipeline.core import load_cohort, from_mol, to_mol
from matrix_pipeline.local_structure import matrix_roles, rooted_fragment_key

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()

def scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    sc = MurckoScaffold.GetScaffoldForMol(mol)
    # Acyclic series are grouped by generic full heavy-atom topology.
    if sc.GetNumAtoms() == 0: sc = mol
    return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(sc))

def graph(mol):
    g = nx.Graph()
    for a in mol.GetAtoms(): g.add_node(a.GetIdx(), z=a.GetAtomicNum())
    for b in mol.GetBonds(): g.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    return g

def align_generated(lean, rich):
    if lean.GetNumAtoms() != rich.GetNumAtoms(): raise ValueError('heavy_atom_count')
    matcher = nx.algorithms.isomorphism.GraphMatcher(graph(lean), graph(rich), node_match=lambda a,b:a['z']==b['z'])
    it = matcher.isomorphisms_iter(); mapping = next(it, None)
    if mapping is None: raise ValueError('heavy_topology')
    ambiguous = next(it, None) is not None
    order = [mapping[i] for i in range(lean.GetNumAtoms())]
    a,b = from_mol(lean),from_mol(rich)
    return np.stack([a,b[order][:,order]]), ambiguous

def state_stats(m):
    n=len(m); b=m[...,7]; adj=b!=0; z=np.diag(m[...,0]); deg=adj.sum(1)
    dist=shortest_path(np.ascontiguousarray(adj,dtype=float), directed=False, unweighted=True)
    cycle=adj.sum()/2-n+nx.number_connected_components(nx.from_numpy_array(adj))
    vals=[n,*[(z==a).sum() for a in (6,7,8,9)],(z!=6).sum(),adj.sum()/2,cycle,
          *[(deg==d).sum() for d in range(5)],
          *[(np.triu(b)==v).sum() for v in (1.,1.5,2.,3.)],
          *[m[...,ch].sum()/2 for ch in (8,9,10)],
          *[(np.triu(dist,1)==d).sum() for d in range(1,9)],
          float(np.where(np.isfinite(dist),dist,0).max())]
    return np.asarray(vals,np.float32),dist

def features(pair):
    """WL count summaries supplement, rather than replace, the raw local matrices."""
    local=np.zeros((2,512),np.float32); delta=np.zeros(256,np.float32)
    globals_=[]; roles=[]; lengths=[]; ego=np.zeros((2,3,len(pair[0]),10),np.float32)
    nodes=[]
    for s,m in enumerate(pair):
        stats,dist=state_stats(m); globals_.append(stats); lengths.append(dist)
        atoms=[tuple(m[i,i,:7].tolist()) for i in range(len(m))]
        labels=[digest(a) for a in atoms]
        for r in range(1,4):
            labels=[digest((labels[i],sorted((tuple(m[i,j,7:].tolist()),labels[j]) for j in np.flatnonzero(m[i,:,7])))) for i in range(len(m))]
            for key in labels: local[s,int(key[:12],16)%512]+=1
            for i in range(len(m)):
                ids=np.flatnonzero(dist[i]<=r); sub=m[ids][:,ids]
                zz=np.diag(sub[...,0]);edges=(sub[...,7]!=0).sum()/2
                # atom count, C/N/O/F, hetero, edges, cycle rank, aromatic/conjugated edges
                ego[s,r-1,i]=[len(ids),*[(zz==a).sum() for a in (6,7,8,9)],(zz!=6).sum(),edges,edges-len(ids)+1,sub[...,8].sum()/2,sub[...,9].sum()/2]
        roles.append(np.asarray([[int(v) for v in row.values()] for row in matrix_roles(m)],np.float32))
        nodes.append(np.asarray(atoms))
    for i in range(len(pair[0])):
        before,after=nodes[0][i],nodes[1][i]
        if not np.array_equal(before,after): delta[int(digest(('a',tuple(before),tuple(after)))[:12],16)%256]+=1
        for j in range(i+1,len(pair[0])):
            a,b=pair[0,i,j,7:],pair[1,i,j,7:]
            if not np.array_equal(a,b):
                key=('e',tuple(sorted([int(before[0]),int(nodes[0][j][0])])),tuple(a),tuple(b))
                delta[int(digest(key)[:12],16)%256]+=1
    g0,g1=globals_; change=np.r_[g1-g0, np.abs(pair[1,...,7]-pair[0,...,7]).sum()/2,
        (np.abs(pair[1,...,7]-pair[0,...,7]).sum(1)>0).sum()]
    return np.r_[g0,g1],np.sqrt(local).ravel(),np.sqrt(delta),change,ego,np.stack(roles),lengths[0]

def audit_fragments(pair):
    result=Counter()
    for s,m in enumerate(pair):
        mol=to_mol(m)
        for r in (1,2,3):
            for i in range(len(m)): result[(s,r,rooted_fragment_key(mol,i,r))]+=1
    return result

def prepare(out, generated=2048, seed=20261122):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    cache=out/'dataset.npz'
    if cache.exists(): return
    RDLogger.DisableLog('rdApp.warning'); RDLogger.DisableLog('rdApp.error')
    started=time.perf_counter(); records=[];pairs=[]; seen=set(); rejects=Counter()
    for m in load_cohort():
        smi,rich=m.identity; key=(smi,rich)
        if key in seen: continue
        seen.add(key);pairs.append(m.pair)
        records.append(dict(id=m.uid,lean_smiles=smi,rich_smiles=rich,parent_smiles='',source='frozen_cohort',source_row=m.source_row,
                            scaffold=scaffold(smi),mapping_ambiguous=False))
    path=Path(os.environ.get('LOHC_GENERATED_LIBRARY','all_generated_molecules.csv'))
    if generated and path.exists():
        frame=pd.read_csv(path,usecols=['smi','sat_smi','parent_smi','discovery_generation'])
        # Random unscored structural sample: no fitness/property filtering.
        order=np.random.default_rng(seed).permutation(len(frame));added=0
        for ix in order:
            row=frame.iloc[ix]
            try:
                lean=Chem.MolFromSmiles(str(row.smi));rich=Chem.MolFromSmiles(str(row.sat_smi))
                if lean is None or rich is None: raise ValueError('parse')
                lean=Chem.RemoveHs(lean);rich=Chem.RemoveHs(rich)
                if lean.GetNumAtoms()>24 or lean.GetNumAtoms()<2: raise ValueError('size_limit24')
                a,b=Chem.MolToSmiles(lean),Chem.MolToSmiles(rich)
                if (a,b) in seen: rejects['duplicate']+=1;continue
                pair,ambiguous=align_generated(lean,rich)
                seen.add((a,b));pairs.append(pair)
                records.append(dict(id=digest((a,b)),lean_smiles=a,rich_smiles=b,parent_smiles=str(row.parent_smi),source='generated_library',
                                    source_row=int(ix),scaffold=scaffold(a),mapping_ambiguous=ambiguous))
                added+=1
                if added>=generated:break
            except Exception as e: rejects[str(e)[:80]]+=1
    frame=pd.DataFrame(records)
    # Union scaffold families with recorded parent families, including unsampled parents.
    parent={}
    def find(a):
        parent.setdefault(a,a)
        while parent[a]!=a: parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b): parent[find(a)]=find(b)
    for row in records:
        find(row['scaffold'])
        if row['parent_smiles'] not in ('','nan','None'):
            try: union(row['scaffold'],scaffold(row['parent_smiles']))
            except Exception: rejects['parent_parse']+=1
    groups=np.asarray([find(s) for s in frame.scaffold]); ids=np.arange(len(frame))
    dev,test=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=1319).split(ids,groups=groups))
    it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.18,random_state=1320).split(dev,groups=groups[dev]))
    train,val=dev[it],dev[iv]
    for a,b in ((train,val),(train,test),(val,test)): assert not(set(groups[a])&set(groups[b]))
    frame['group']=groups;frame['split']='train';frame.loc[val,'split']='validation';frame.loc[test,'split']='test'
    frame['heavy_atoms']=[len(p[0]) for p in pairs];frame.to_csv(out/'data_split.csv',index=False)
    print('Prepared identities',len(frame),'split',{k:len(v) for k,v in [('train',train),('val',val),('test',test)]},flush=True)
    nmax=max(len(p[0]) for p in pairs); N=len(pairs)
    x=np.zeros((N,2,nmax,nmax,12),np.float16);mask=np.zeros((N,nmax),bool)
    dist=np.full((N,nmax,nmax),255,np.uint8);ego=np.zeros((N,2,3,nmax,10),np.float32);role=np.zeros((N,2,nmax,8),np.float32)
    gs=[];ls=[];ds=[];cs=[];fps=[];maccs=[]
    generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    for i,p in enumerate(pairs):
        n=len(p[0]);x[i,:,:n,:n]=p;mask[i,:n]=True
        g,l,d,c,e,r,di=features(p);gs.append(g);ls.append(l);ds.append(d);cs.append(c);ego[i,:,:,:n]=e;role[i,:,:n]=r
        dist[i,:n,:n]=np.where(np.isfinite(di),di,255).astype(np.uint8)
        mols=[to_mol(m) for m in p]
        fps.append(np.concatenate([generator.GetFingerprintAsNumPy(m) for m in mols]))
        maccs.append(np.concatenate([np.asarray(MACCSkeys.GenMACCSKeys(m),np.float32) for m in mols]))
        if i%1500==0:print('Matrix/local features',i,'/',N,flush=True)
    np.savez_compressed(cache,x=x,mask=mask,distance=dist,global_target=np.asarray(gs),local_target=np.asarray(ls),
        delta_target=np.asarray(ds),change_target=np.asarray(cs),ego=ego,roles=role,morgan=np.asarray(fps,np.uint8),
        maccs=np.asarray(maccs,np.uint8),train=train,val=val,test=test)
    rng=np.random.default_rng(999);audit_ids=np.r_[rng.choice(train,min(1024,len(train)),replace=False),rng.choice(val,min(192,len(val)),replace=False),rng.choice(test,min(192,len(test)),replace=False)]
    fragments={str(i):[[list(k),v] for k,v in audit_fragments(pairs[i]).items()] for i in audit_ids}
    dump(out/'audit_fragments.json',fragments)
    dump(out/'data_audit.json',dict(n=N,nmax=nmax,source_counts=frame.source.value_counts().to_dict(),
        split_counts=frame.split.value_counts().to_dict(),group_count=len(set(groups)),
        mapping_ambiguous=int(frame.mapping_ambiguous.sum()),rejections=dict(rejects),seconds=time.perf_counter()-started,
        matrix_input='Aligned lean/rich [2,n,n,12]; all local centers and radii1/2/3; uncapped induced local topology.',
        generated_sampling='Unscored random sample, element/topology isomorphism checked; first feasible map, symmetry ambiguity recorded, not uniqueness verified.',
        split_policy='Generic scaffold + sampled recorded-parent family union, GroupShuffleSplit1319/1320. Multi-generation unsampled ancestry not available; no full lineage-disjoint claim.',
        historical_exposure='Original cohort previously used for manager diagnosis; this is a development study, not an untouched external benchmark.',
        local_summary='SHA256-bucketed WL counts512 per state; not lossless or exclusive to matrix representation. Raw induced local matrices are separate network inputs.',
        properties='No enthalpy, fitness, LLM scores or SMILES features are structure training inputs.'))
    print('Data ready',time.perf_counter()-started,flush=True)
