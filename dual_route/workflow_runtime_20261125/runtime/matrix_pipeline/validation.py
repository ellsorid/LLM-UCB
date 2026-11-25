"""Semantic, CPU/GPU and throughput checks on real unique molecular pairs."""
import sys
sys.dont_write_bytecode=True
import copy,json,csv,time
import numpy as np
import torch
from scipy import sparse
from .core import MANAGEMENT,validate_child,properties
from .tensors import pack,qd_features,subspace45,local_entropy48,Atlas,LinUCB,Archive,edit_menu,apply_edits

def validate_and_benchmark(cohort,x,mask,cached,atlas,ucb,gnn,device,out):
 from .runtime import dump,csvout,sync
 rng=np.random.default_rng(171121);ids=rng.choice(len(cohort),128,replace=False);subset=[cohort[i] for i in ids]
 xp,mp=pack(subset,device);xc,mc=pack(subset,'cpu');g=subspace45(xp,mp);c=subspace45(xc,mc)
 legacy=sparse.load_npz(MANAGEMENT/'runs/P04/20261121_typed_krylov_v1/typed_basic.npz').toarray()[ids]
 entropy=sparse.load_npz(MANAGEMENT/'runs/P04/20261121_local_role_entropy_v1/entropy.npz').toarray()[ids]
 errors={
  'subspace45_vs_cached_max':float((g-cached[ids]).abs().max()),
  'subspace_CPU_GPU_max':float((g.cpu()-c).abs().max()),
  'typed18_vs_cached_max':float(np.max(np.abs(qd_features(xp,mp).cpu().numpy()[:,:18]-legacy))),
  'entropy48_vs_cached_max':float(np.max(np.abs(local_entropy48(xp,mp).cpu().numpy()-entropy)))}
 if any(v>1e-5 for v in errors.values()):raise AssertionError(json.dumps(errors))
 # Random shared permutations retain cell and context despite changing atom IDs.
 perm=xp.clone()
 for i,m in enumerate(subset):
  n=len(m.pair[0]);p=torch.tensor(rng.permutation(n),device=device);perm[i,:,:n,:n]=xp[i,:,p[:,None],p[None,:]]
 errors['permutation_context_max']=float((subspace45(perm,mp)-g).abs().max())
 if errors['permutation_context_max']>1e-5:raise AssertionError('Permutation drift')
 aq=qd_features(xp,mp);cpu_atlas=Atlas(atlas.mean.cpu(),atlas.scale.cpu(),atlas.centers.cpu())
 cells=atlas.locate(aq);cpu_cells=cpu_atlas.locate(aq.cpu())
 if not torch.equal(cells.cpu(),cpu_cells):raise AssertionError('CPU/GPU cell mismatch')
 if not torch.equal(atlas.locate(qd_features(perm,mp)),cells):raise AssertionError('Permutation cell mismatch')
 # LinUCB posterior matches direct ridge solve after batched observations.
 gu=LinUCB(ucb.mean[:45].clone(),ucb.scale[:45].clone());cu=LinUCB(ucb.mean[:45].cpu(),ucb.scale[:45].cpu());feedback=torch.linspace(0,1,16,device=device)
 gu.update(g[:16],feedback);cu.update(c[:16],feedback.cpu())
 errors['UCB_CPU_GPU_score_max']=float((gu.score(g).cpu()-cu.score(c)).abs().max())
 phi=gu.context(g[:16]);direct=torch.linalg.solve(torch.eye(phi.shape[1],device=device,dtype=torch.float64)+phi.T@phi,phi.T@feedback.double())
 errors['UCB_Woodbury_direct_max']=float((gu.inv@gu.b-direct).abs().max())
 if errors['UCB_CPU_GPU_score_max']>1e-7 or errors['UCB_Woodbury_direct_max']>1e-9:raise AssertionError('UCB posterior mismatch')
 # Admission/tie behaviour agrees across devices, including multiple same-cell rows.
 a=Archive(5,device);b=Archive(5,'cpu');cs=torch.tensor([0,0,1,3,3],device=device);qs=torch.tensor([.5,.7,.8,.9,.9],device=device);members=torch.tensor([5,4,3,2,1],device=device)
 ag,rg=a.update(cs,qs,members);ac,rc=b.update(cs.cpu(),qs.cpu(),members.cpu())
 if not torch.equal(a.member.cpu(),b.member) or not torch.equal(ag.cpu(),ac) or not torch.equal(rg.cpu(),rc):raise AssertionError('Archive mismatch')
 # Independent edit orders must lead to the same canonical pair and QD cell.
 parent=next(m for m in cohort if len({a['site'] for a in edit_menu([m])})>=2)
 sites=sorted({a['site'] for a in edit_menu([parent])})[:2]
 def apply_one(m,site,element):
  pp,pm=pack([m],device);act={'parent_index':0,'site':site,'element':element,'new_hydrogens':{6:3,7:2,8:1}[element],'kind':'append_single_bond_substituent'}
  child,nn=apply_edits(pp,pm,[act]);return validate_child(child[0,:,:int(nn[0]),:int(nn[0])].cpu().numpy(),m,act)
 route_a=apply_one(apply_one(parent,sites[0],6),sites[1],7)
 route_b=apply_one(apply_one(parent,sites[1],7),sites[0],6)
 if route_a.uid!=route_b.uid:raise AssertionError('Path-dependent molecular identity')
 routes,rm=pack([route_a,route_b],device)
 if int(atlas.locate(qd_features(routes,rm))[0])!=int(atlas.locate(qd_features(routes,rm))[1]):raise AssertionError('Path-dependent QD cell')
 if not np.isclose(properties(parent.pair)[0],properties(route_a.pair)[0]):raise AssertionError('Substitution changed hydrogen stoichiometry')
 # Frozen GNN compatibility and inference equivalence, including archived patches.
 xp,mp=pack(subset,device,legacy=True);xc,mc=pack(subset,'cpu',legacy=True)
 cpu_model=copy.deepcopy(gnn.model).cpu().eval()
 with torch.inference_mode():cp=cpu_model(xc[:,0],mc.float())*gnn.scale+gnn.mean
 gp=gnn.predict(xp,mp);errors['GNN_CPU_GPU_max']=float((gp.cpu()-cp).abs().max())
 with (MANAGEMENT/'runs/P00/20261121_pair_core_audit_v1/gnn_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
  reference={int(r['source_row']):float(r['reference_prediction']) for r in csv.DictReader(f)}
 errors['GNN_cached_prediction_max']=float(max(abs(float(v)-reference[m.source_row]) for v,m in zip(gp.cpu(),subset)))
 if errors['GNN_CPU_GPU_max']>.001 or errors['GNN_cached_prediction_max']>.001:raise AssertionError('GNN compatibility failure')
 validation={'passed':True,'real_unique_pairs':len(cohort),'feature_cases':128,'errors':errors,'same_pair_different_edit_paths_same_cell':True,
  'same_frozen_checkpoint':True,'no_GNN_training':True,'resident_numerical_kernels_only_on_GPU':True,'chemistry_and_API_boundaries_on_CPU':True}
 dump(out/'validation.json',validation);print(json.dumps({'validation':validation}),flush=True)
 benchmark=[]
 for count in [256,2048,len(cohort)]:
  for kind in ['decision_kernels','frozen_GNN']:
   for target in ['cpu',device]:
    tp=x[:count].to(target);tm=mask[:count].to(target)
    at=Atlas(atlas.mean.to(target),atlas.scale.to(target),atlas.centers.to(target));uu=LinUCB(ucb.mean[:45].to(target),ucb.scale[:45].to(target))
    def compute():
     if kind=='decision_kernels':
      q=qd_features(tp,tm);cx=subspace45(tp,tm);cell=at.locate(q);score=uu.score(cx)
      ar=Archive(64,target);ad,rew=ar.update(cell,score.float(),torch.arange(count,device=target));uu.update(cx[:8],rew[:8]);return ad
     if target=='cpu':
      with torch.inference_mode():return torch.cat([cpu_model(tp[i:i+64,0],tm[i:i+64].float()) for i in range(0,len(tp),64)])*gnn.scale+gnn.mean
     return gnn.predict(tp,tm,batch=64)
    compute();sync(target);times=[]
    for _ in range(3):
     start=time.perf_counter();compute();sync(target);times.append(time.perf_counter()-start)
    median=float(np.median(times));benchmark.append({'kind':kind,'device':target,'unique_molecules':count,'repeats':3,'median_seconds':median,'molecules_per_second':count/median,
     'scope':'resident tensors; excludes chemistry, transfers and API; frozen_GNN batch64; decision includes feature extraction/QD/archive/UCB'})
    print(json.dumps(benchmark[-1]),flush=True)
 csvout(out/'throughput.csv',benchmark)
 dump(out/'benchmark.json',{'hardware':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU','torch':torch.__version__,
  'CPU_threads':torch.get_num_threads(),'unique_pairs':len(cohort),'rows':benchmark,'interpretation':'Measured device kernels; total-loop speed is separately reported, never inferred from kernel speedup.'})
