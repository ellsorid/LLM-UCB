"""GPU/CPU-equivalent features, matrix edits, CVT archive, and linear UCB."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
import torch
from .core import PROJECT

def pack(molecules,device='cpu',legacy=False):
 n=max(len(m.pair[0]) for m in molecules);x=np.zeros((len(molecules),2,n,n,12),np.float32);mask=np.zeros((len(molecules),n),bool)
 for i,m in enumerate(molecules):
  k=len(m.pair[0]);x[i,:,:k,:k]=m.pair;mask[i,:k]=True
  if legacy:
   for a,b,c,v in m.gnn_patch or []:x[i,0,int(a),int(b),int(c)]=v
 return torch.tensor(x,device=device),torch.tensor(mask,device=device)

def diagonal(m):return m.diagonal(dim1=-3,dim2=-2).transpose(-1,-2)

def qd_features(pair,mask):
 """18 typed-ring coordinates plus six mixed paths and six change/size fields."""
 p=pair.double();z=diagonal(p)[...,0];n=mask.sum(-1).double().clamp_min(1)
 basics=[];mixed=[]
 for s in range(2):
  r=p[:,s,:,:,10];c=p[:,s,:,:,9];ring2=(r*r.transpose(-1,-2)).sum(-1)
  hetero=((z[:,s]!=1)&(z[:,s]!=6)&mask).double();rr=[];mm=[]
  for atom in [7,8,9]:
   v=(z[:,s]==atom).double();rr.extend([v.sum(-1),(v*ring2).sum(-1)/n])
   mm.append(torch.einsum('bi,bij,bjk,bk->b',v,r,c,hetero)/n)
  basics.append(torch.stack(rr,-1));mixed.append(torch.stack(mm,-1))
 delta=p[:,1,:,:,7]-p[:,0,:,:,7];w=delta.abs().sum(-1)
 h=diagonal(p)[...,2];nh2=(h[:,1].sum(-1)-h[:,0].sum(-1))/2
 extra=torch.stack([nh2,torch.linalg.vector_norm(delta,dim=(-2,-1))/n,
  (p[:,1,:,:,8]-p[:,0,:,:,8]).sum((-2,-1))/(2*n),
  (p[:,1,:,:,9]-p[:,0,:,:,9]).sum((-2,-1))/(2*n),((w>0)&mask).sum(-1)/n,n/20],-1)
 return torch.cat([basics[0],basics[1],basics[1]-basics[0],*mixed,extra],-1).float()

def subspace45(pair,mask):
 """Exact P04 joint Krylov mask/rank/projection fields; float64 rank decisions."""
 p=pair.double();b=p[...,7];c=p[...,9];z=diagonal(p)[...,0][:,0]
 b=b/torch.linalg.matrix_norm(b,ord=2).clamp_min(1e-12)[...,None,None]
 c=c/torch.linalg.matrix_norm(c,ord=2).clamp_min(1e-12)[...,None,None]
 w=(p[:,1,:,:,7]-p[:,0,:,:,7]).abs().sum(-1)
 seeds=torch.stack([(z==7).double(),(z==8).double(),(z==9).double(),((z!=1)&(z!=6)&mask).double(),w],1)
 seeds=seeds/torch.linalg.vector_norm(seeds,dim=-1,keepdim=True).clamp_min(1e-30)
 v=seeds[:,None].expand(-1,2,-1,-1)
 bv=torch.einsum('bsij,bskj->bski',b,v);cv=torch.einsum('bsij,bskj->bski',c,v)
 bcv=torch.einsum('bsij,bskj->bski',b,cv)
 cols=torch.stack([v,bv,cv,bcv],-1)
 u,s,_=torch.linalg.svd(cols,full_matrices=False)
 keep=s>torch.maximum(s[...,:1]*1e-8,torch.full_like(s[...,:1],1e-12))
 rank=keep.sum(-1).double();u=u*keep.unsqueeze(-2)
 overlap=(u[:,0].transpose(-1,-2)@u[:,1]).square().sum((-2,-1))
 l,r=rank[:,0],rank[:,1];dist=((l+r-2*overlap)/2).clamp_min(0)
 fields=torch.stack([(l>0).double(),(r>0).double(),l,r,r-l,((l>0)&(r>0)).double(),dist,
  2*dist/(l+r).clamp_min(1),overlap/torch.minimum(l,r).clamp_min(1)],-1)
 return fields.flatten(1).float()

def local_entropy48(pair,mask):
 """P04-B1 reaction-seeded diffusion/role H and JS, computed in batches."""
 p=pair.double();d=diagonal(p);z=d[:,:, :,0];n=mask.shape[1]
 w=(p[:,1,:,:,7]-p[:,0,:,:,7]).abs().sum(-1);changed=(w>0)&mask
 adjacency=(p[:,0,:,:,7]!=0);region=changed.clone();result=[]
 element=torch.full_like(z,4,dtype=torch.long)
 for i,atomic in enumerate([6,7,8,9]):element=torch.where(z==atomic,i,element)
 ring=(p[...,10].sum(-1)>0).long();arom=d[:,:,:,3].long();code=4*element+2*ring+arom
 for radius in [1,2]:
  region=region|torch.any(adjacency&region[:,None,:],dim=-1)
  p0=w*region;p0=p0/p0.sum(-1,keepdim=True).clamp_min(1e-30)
  for channel in [7,9]:
   a=(p[...,channel]!=0).double()*region[:,None,:,None]*region[:,None,None,:]
   lap=torch.diag_embed(a.sum(-1))-a;eig,u=torch.linalg.eigh(lap)
   coeff=(u.transpose(-1,-2)@p0[:,None,:,None]).squeeze(-1)
   for t in [.25,1.,4.]:
    mass=(u@(torch.exp(-t*eig)*coeff)[...,None]).squeeze(-1).clamp_min(0)
    mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-30);q=torch.zeros((*mass.shape[:2],20),device=p.device,dtype=p.dtype)
    q.scatter_add_(2,code,mass)
    h=-(q*q.clamp_min(1e-30).log()).sum(-1);mix=q.mean(1);hm=-(mix*mix.clamp_min(1e-30).log()).sum(-1)
    result.append(torch.stack([h[:,0],h[:,1],h[:,1]-h[:,0],(hm-h.mean(1)).clamp_min(0)],-1))
 return torch.cat(result,-1).float()

def edit_menu(molecules,max_atoms=14):
 actions=[]
 for pi,m in enumerate(molecules):
  n=len(m.pair[0])
  if n>=max_atoms:continue
  d=np.diagonal(m.pair,axis1=1,axis2=2).transpose(0,2,1)
  if np.any(d[:,:,6]) or np.any(m.pair[:,:,:,11]):continue
  for site in range(n):
   # Carbon-bound H substitution keeps the existing scaffold in both states.
   if d[0,site,0]!=6 or min(d[:,site,2])<1:continue
   for atom in [6,7,8]:actions.append({'parent_index':pi,'site':site,'element':atom,'new_hydrogens':{6:3,7:2,8:1}[atom],'kind':'append_single_bond_substituent'})
 return actions

def apply_edits(parents,mask,actions):
 n=parents.shape[2];out=torch.zeros((len(actions),2,n+1,n+1,12),device=parents.device,dtype=parents.dtype)
 pi=torch.tensor([a['parent_index'] for a in actions],device=parents.device);site=torch.tensor([a['site'] for a in actions],device=parents.device)
 size=mask[pi].sum(-1).long();ii=torch.arange(len(actions),device=parents.device)
 out[:,:,:n,:n]=parents[pi]
 for s in [0,1]:
  out[ii,s,size,size,0]=torch.tensor([a['element'] for a in actions],device=parents.device,dtype=parents.dtype)
  out[ii,s,size,size,2]=torch.tensor([a['new_hydrogens'] for a in actions],device=parents.device,dtype=parents.dtype)
  out[ii,s,site,site,2]-=1
  out[ii,s,site,size,7]=1;out[ii,s,size,site,7]=1
 return out,size+1

class PopulationBank:
 """Resident paired matrices; Python sidecars carry only chemistry/trace labels."""
 def __init__(self,device,capacity=1024,max_atoms=14):
  self.pairs=torch.zeros((capacity,2,max_atoms,max_atoms,12),device=device);self.mask=torch.zeros((capacity,max_atoms),device=device,dtype=torch.bool);self.count=0
 def append(self,pair,mask):
  b,n=len(pair),pair.shape[2]
  if self.count+b>len(self.pairs) or n>self.pairs.shape[2]:raise ValueError('Resident population capacity exceeded')
  sl=slice(self.count,self.count+b);self.pairs[sl,:,:n,:n]=pair;self.mask[sl,:n]=mask;self.count+=b
 def gather(self,ids):
  masks=self.mask[ids];n=int(masks.sum(1).max());return self.pairs[ids,:,:n,:n],masks[:,:n]

def resident_edit_menu(pair,mask,max_atoms=14):
 d=diagonal(pair);legal=((d[:,:,:,6]==0).all((1,2))&(pair[:,:,:,:,11]==0).all((1,2,3))&(mask.sum(1)<max_atoms))
 sites=(d[:,0,:,0]==6)&(d[:,:,:,2]>=1).all(1)&mask&legal[:,None]
 positions=torch.nonzero(sites).cpu().tolist()
 return [{'parent_index':pi,'site':site,'element':atom,'new_hydrogens':{6:3,7:2,8:1}[atom],'kind':'append_single_bond_substituent'} for pi,site in positions for atom in [6,7,8]]

class Atlas:
 def __init__(self,mean,scale,centers):self.mean,self.scale,self.centers=mean,scale,centers
 @classmethod
 def fit(cls,features,cells=64,iterations=12):
  mean=features.mean(0);scale=features.std(0,unbiased=False).clamp_min(1e-6);x=(features-mean)/scale
  centers=[x[0]];best=torch.full((len(x),),float('inf'),device=x.device)
  for _ in range(1,min(cells,len(x))):
   best=torch.minimum(best,(x-centers[-1]).square().sum(1));centers.append(x[best.argmax()])
  centers=torch.stack(centers)
  for _ in range(iterations):
   ids=torch.cdist(x,centers).argmin(1);sums=torch.zeros_like(centers);sums.index_add_(0,ids,x)
   count=torch.bincount(ids,minlength=len(centers));centers=torch.where(count[:,None]>0,sums/count.clamp_min(1)[:,None],centers)
  return cls(mean,scale,centers)
 def locate(self,features):
  x=((features-self.mean)/self.scale).double();centers=self.centers.double()
  # Freeze numerical tie policy for exact same-molecule cell assignment.
  distance=(x[:,None]-centers[None]).square().sum(-1)
  return torch.round(distance/1e-8).argmin(1)

class LinUCB:
 def __init__(self,mean,scale,alpha=.5):
  self.mean,self.scale,self.alpha=mean,scale,alpha;self.inv=torch.eye(len(mean)+1,device=mean.device,dtype=torch.float64);self.b=torch.zeros(len(mean)+1,device=mean.device,dtype=torch.float64)
 def context(self,x):
  x=((x-self.mean)/self.scale).double();x=x/torch.linalg.vector_norm(x,dim=1,keepdim=True).clamp_min(1)
  return torch.cat([torch.ones((len(x),1),device=x.device,dtype=x.dtype),x],1)
 def score(self,x):
  phi=self.context(x);return phi@(self.inv@self.b)+self.alpha*(torch.einsum('bi,ij,bj->b',phi,self.inv,phi).clamp_min(0)).sqrt()
 def update(self,x,reward):
  phi=self.context(x);a=self.inv@phi.T
  self.inv-=a@torch.linalg.solve(torch.eye(len(phi),device=x.device,dtype=torch.float64)+phi@a,a.T)
  self.inv=(self.inv+self.inv.T)/2;self.b+=phi.T@reward.double()

class StateUCB(LinUCB):
 def __init__(self,mean,scale,alpha=.5):
  self.structural_dimensions=len(mean)
  super().__init__(torch.cat([mean,torch.zeros(3,device=mean.device)]),torch.cat([scale,torch.ones(3,device=mean.device)]),alpha)
 def context(self,x):
  structural=((x[:,:45]-self.mean[:45])/self.scale[:45]).double()
  structural/=torch.linalg.vector_norm(structural,dim=1,keepdim=True).clamp_min(1)
  return torch.cat([torch.ones((len(x),1),device=x.device,dtype=torch.float64),structural,x[:,45:].double()],1)
 def with_state(self,x,cells,archive):
  occupied=torch.isfinite(archive.quality[cells]);quality=torch.where(occupied,archive.quality[cells],torch.zeros_like(archive.quality[cells]))
  fraction=torch.isfinite(archive.quality).float().mean().expand(len(x))
  return torch.cat([x,(~occupied).float()[:,None],quality[:,None],fraction[:,None]],1)

class Archive:
 def __init__(self,cells,device):
  self.quality=torch.full((cells,),-float('inf'),device=device);self.member=torch.full((cells,),-1,device=device,dtype=torch.long)
 def update(self,cells,quality,member):
  before=self.quality[cells].clone();best=torch.full_like(self.quality,-float('inf'));best.scatter_reduce_(0,cells,quality,reduce='amax',include_self=True)
  # Stable minimum member ID resolves equal-quality ties within this batch.
  chosen=torch.full_like(self.member,torch.iinfo(torch.long).max)
  candidate=torch.where(quality==best[cells],member,torch.full_like(member,torch.iinfo(torch.long).max))
  chosen.scatter_reduce_(0,cells,candidate,reduce='amin',include_self=True)
  replace=best>self.quality;self.quality[replace]=best[replace];self.member[replace]=chosen[replace]
  admitted=replace[cells]&(self.member[cells]==member)
  reward=torch.where(admitted,torch.where(torch.isfinite(before),torch.clamp(quality-before,min=0,max=1),torch.ones_like(quality)),torch.zeros_like(quality))
  return admitted,reward

class FrozenGNN:
 def __init__(self,device):
  root=Path(__file__).resolve().parents[1]/'models/lean_matrix'
  if root.exists():
   from models.lean_matrix.model import MatrixGIN
   self.path=root/'checkpoints/fold_1_best.pth'
  else:
   sys.path.insert(0,str(PROJECT/'newgnn'));from model import MatrixGIN
   self.path=PROJECT/'newgnn/matrix_gnn_12ch_results/checkpoints/fold_1_best.pth'
  ckpt=torch.load(self.path,map_location='cpu',weights_only=True);self.mean,self.scale=ckpt['target_mean'],ckpt['target_std']
  self.model=MatrixGIN(hidden_dim=ckpt['hidden_dim'],num_layers=ckpt['num_gnn_layers']).to(device).eval();self.model.load_state_dict(ckpt['model_state_dict'])
 @torch.inference_mode()
 def predict(self,pair,mask,batch=64):
  return torch.cat([self.model(pair[i:i+batch,0],mask[i:i+batch].float())*self.scale+self.mean for i in range(0,len(pair),batch)])
