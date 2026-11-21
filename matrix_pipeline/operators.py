"""Aligned sidechain replacements with explicit hydrogen-release stoichiometry."""
import hashlib,json
from collections import defaultdict
import torch

OPERATORS={
 'methyl':{'atoms':[6],'lean_H':[3],'rich_H':[3],'orders':[0,0],'delta_nH2':0},
 'amino':{'atoms':[7],'lean_H':[2],'rich_H':[2],'orders':[0,0],'delta_nH2':0},
 'hydroxyl':{'atoms':[8],'lean_H':[1],'rich_H':[1],'orders':[0,0],'delta_nH2':0},
 'vinyl_ethyl':{'atoms':[6,6],'lean_H':[1,2],'rich_H':[2,3],'orders':[2,1],'delta_nH2':1},
 'ethynyl_ethyl':{'atoms':[6,6],'lean_H':[0,1],'rich_H':[2,3],'orders':[3,1],'delta_nH2':2},
}

def action(parent,site,kind):
 spec=OPERATORS[kind]
 return {'parent_index':parent,'site':site,'kind':kind,'delta_nH2':spec['delta_nH2']}

def menu(pair,mask,max_atoms=14):
 from .tensors import diagonal
 d=diagonal(pair);size=mask.sum(1)
 legal=(d[:,:,:,6]==0).all((1,2))&(pair[:,:,:,:,11]==0).all((1,2,3))
 sites=(d[:,0,:,0]==6)&(d[:,:,:,2]>=1).all(1)&mask&legal[:,None]
 positions=torch.nonzero(sites).cpu().tolist();sizes=size.cpu().tolist()
 return [action(pi,site,kind) for pi,site in positions for kind,spec in OPERATORS.items() if sizes[pi]+len(spec['atoms'])<=max_atoms]

def apply(pair,mask,actions):
 n=pair.shape[2];extra=max(len(OPERATORS[a['kind']]['atoms']) for a in actions)
 out=torch.zeros((len(actions),2,n+extra,n+extra,12),device=pair.device,dtype=pair.dtype)
 pi=torch.tensor([a['parent_index'] for a in actions],device=pair.device);ii=torch.arange(len(actions),device=pair.device)
 start=mask[pi].sum(1).long();site=torch.tensor([a['site'] for a in actions],device=pair.device)
 out[:,:,:n,:n]=pair[pi];sizes=start.clone()
 for kind,spec in OPERATORS.items():
  indices=[i for i,a in enumerate(actions) if a['kind']==kind]
  if not indices:continue
  rows=ii[indices];first=start[indices];anchor=site[indices];sizes[indices]+=len(spec['atoms'])
  for state,H in enumerate([spec['lean_H'],spec['rich_H']]):
   out[rows,state,anchor,anchor,2]-=1
   for k,z in enumerate(spec['atoms']):
    atom=first+k;out[rows,state,atom,atom,0]=z;out[rows,state,atom,atom,2]=H[k]
   out[rows,state,anchor,first,7]=1;out[rows,state,first,anchor,7]=1
   if len(spec['atoms'])==2:
    out[rows,state,first,first+1,7]=spec['orders'][state];out[rows,state,first+1,first,7]=spec['orders'][state]
 return out,sizes

def balanced_actions(actions,parents,budget,generation,seed=17):
 """Equal-turn parent/operator groups; invariant to incoming enumeration order."""
 groups=defaultdict(list)
 def digest(value):return hashlib.sha256(f'{seed}:{generation}:{value}'.encode()).hexdigest()
 for a in actions:groups[parents[a['parent_index']].uid,a['kind']].append(a)
 keys=sorted(groups,key=lambda k:digest(str(k)))
 for key in keys:groups[key].sort(key=lambda a:digest(json.dumps(a,sort_keys=True)))
 result=[];level=0
 while len(result)<min(budget,len(actions)):
  for key in keys:
   if level<len(groups[key]):result.append(groups[key][level])
   if len(result)==min(budget,len(actions)):break
  level+=1
 return result

def balanced_parents(active,visits,uids,budget):
 """Least-visited parents, stable UID-hash ties; no cell-number prefix."""
 keys=torch.tensor([int(hashlib.sha256(u.encode()).hexdigest()[:15],16) for u in uids],device=active.device)
 ordered=active[torch.argsort(keys[active],stable=True)]
 ordered=ordered[torch.argsort(visits[ordered],stable=True)][:budget]
 visits[ordered]+=1
 return ordered
