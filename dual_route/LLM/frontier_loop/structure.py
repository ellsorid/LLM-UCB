"""Read-only bridge to the existing SOM and validated paired-matrix operators."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np


class StructureBridge:
    def __init__(self, archive):
        self.archive=Path(archive).resolve()
        self.runtime=self.archive/'workflow_runtime_20261125/runtime'
        os.environ.update(LOHC_PROJECT_ROOT=str(self.runtime),LOHC_BUNDLED_ONLY='1',PYTHONDONTWRITEBYTECODE='1')
        sys.path.insert(0,str(self.runtime))
        import torch
        torch.set_num_threads(2)
        self.torch=torch
        self.model_path=self.runtime/'outputs/matrix_som_v2/model.pt'
        self.model=torch.load(self.model_path,map_location='cpu',weights_only=False)
        self.model_hash=hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        self.centers=self.model['centers'].numpy()
        self.bandwidth=max(float(np.median(np.sum((self.centers[:,None]-self.centers[None,:])**2,axis=2))),1e-6)

    def describe(self,molecules):
        from matrix_map.simple_som import encode_pairs,classify
        z=encode_pairs([m.pair for m in molecules],self.model['transform'],self.model['weights'],device='cpu')
        cls,dist,out=classify(z,self.model)
        arr=z.numpy().astype(float)
        sq=((arr[:,None]-self.centers[None,:])**2).sum(2)
        response=np.exp(-sq/(2*self.bandwidth))
        response/=np.maximum(np.linalg.norm(response,axis=1,keepdims=True),1e-12)
        # No new encoder, no projection: full original 1338 structural dimensions
        # plus a nonlinear readout on the 128 frozen SOM prototypes.
        psi=np.c_[np.ones(len(arr)),arr,.25*response]
        return [dict(uid=m.uid,molecule=m,z=arr[i],psi=psi[i],class_id=int(cls[i]),
            prototype_distance=float(dist[i]),outlier=bool(out[i]),scored=False) for i,m in enumerate(molecules)]

    def seeds(self,count=24):
        from matrix_pipeline.core import Molecule,to_mol,identity
        from rdkit import Chem
        molecules=[];seen=set()
        with np.load(self.runtime/'outputs/matrix_som_v2/matrix_inputs.npz') as r:
            for i in range(min(256,len(r['x']))):
                n=int(r['mask'][i].sum());p=r['x'][i,:,:n,:n].copy()
                try:
                    names=tuple(Chem.MolToSmiles(to_mol(a),isomericSmiles=True) for a in p)
                except Exception:
                    continue
                uid=identity(names)
                if uid in seen:continue
                seen.add(uid);molecules.append(Molecule(uid,p,names,uid,source_row=i))
                if len(molecules)==count:break
        if len(molecules)<count:raise ValueError('Too few valid seed molecules')
        return self.describe(molecules)

    def children(self,parent,operator,max_sites=2):
        from matrix_pipeline.operators import menu,apply
        from matrix_pipeline.tensors import pack
        from matrix_pipeline.core import validate_child
        molecule=parent['molecule'];t,mask=pack([molecule],'cpu')
        actions=[a for a in menu(t,mask,max_atoms=20) if a['kind']==operator][:max_sites]
        if not actions:return [],[]
        batch,sizes=apply(t,mask,actions);valid=[];errors=[]
        for i,a in enumerate(actions):
            try:
                p=batch[i,:,:int(sizes[i]),:int(sizes[i])].numpy()
                child=validate_child(p,molecule,a)
                valid.append(child)
            except Exception as exc:
                errors.append(dict(operator=operator,site=a['site'],error=type(exc).__name__+': '+str(exc)))
        records=self.describe(valid) if valid else []
        for r in records:
            r.update(parent_uid=parent['uid'],operator=operator,edit=r['molecule'].action)
        return records,errors

    def differential_packet(self,parent,child):
        from matrix_map.simple_protocol import difference
        from matrix_pipeline.local_structure import local_packet
        p=parent['molecule'].pair;c=child['molecule'].pair;n=max(p.shape[1],c.shape[1])
        a=np.zeros((2,n,n,12));b=a.copy()
        a[:,:p.shape[1],:p.shape[1]]=p;b[:,:c.shape[1],:c.shape[1]]=c
        delta=b-a;idx=np.argwhere(np.abs(delta)>1e-6)
        site=int(child['edit']['site'])
        return dict(parent_uid=parent['uid'],child_uid=child['uid'],operation=child['edit'],
            matrix_delta_scope='child-minus-parent separately for lean/rich, NOT rich-minus-lean',
            retained_parent_atom_ids=list(range(p.shape[1])),
            added_atom_ids=list(range(p.shape[1],c.shape[1])),
            sparse_delta=[dict(state=['lean','rich'][int(s)],i=int(i),j=int(j),channel=int(k),delta=float(delta[s,i,j,k])) for s,i,j,k in idx],
            structure_difference=difference(p,c),
            parent_local=local_packet(p,[site],2),child_local=local_packet(c,[site],2),
            differential_method='Existing SOM re-encoding of validated finite matrix edits; no atomic gradient inversion')

    def structural_direction(self,parent,occupancy):
        cls=parent['class_id'];top=self.model['topology'][cls].numpy()
        neighbors=[i for i in range(len(top)) if 0<top[i]<=1]
        target=min(neighbors,key=lambda i:(occupancy.get(i,0),top[i],i)) if neighbors else cls
        delta=self.centers[target]-parent['z']
        return delta/(np.linalg.norm(delta)+1e-12),target
