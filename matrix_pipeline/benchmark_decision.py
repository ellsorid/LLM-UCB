"""Follow-up timing to separate FP64 geometry extraction from resident decisions."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy import sparse
from .core import load_cohort,MANAGEMENT
from .tensors import pack,qd_features,subspace45,Atlas,LinUCB,Archive
from .runtime import dump,csvout,sync

def main():
 p=argparse.ArgumentParser();p.add_argument('--calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;cohort=load_cohort();x,mask=pack(cohort,'cpu');q=qd_features(x,mask)
 if len(cohort)!=9812:raise ValueError('Historical throughput benchmark requires the external9812-pair cohort')
 state=torch.load(a.calibration,map_location='cpu',weights_only=True)
 columns=state.get('context_columns',[i for i in range(55) if i%11 not in [9,10]])
 x45=torch.tensor(sparse.load_npz(MANAGEMENT/'runs/P04/20261121_typed_krylov_v1/krylov_joint.npz').toarray()[:,columns])
 result=[]
 for count in [256,2048,len(cohort)]:
  for device in ['cpu','cuda']:
   qt=q[:count].to(device);xx=x45[:count].to(device);tensor=x[:count].to(device);mm=mask[:count].to(device)
   at=Atlas(state['descriptor_mean'].to(device),state['descriptor_scale'].to(device),state['centers'].to(device));ucb=LinUCB(state['context_mean'].to(device),state['context_scale'].to(device))
   for kind in ['feature_extraction','resident_QD_UCB_archive']:
    def run():
     if kind=='feature_extraction':return qd_features(tensor,mm),subspace45(tensor,mm)
     cell=at.locate(qt);score=ucb.score(xx);archive=Archive(64,device);archive.update(cell,score.float(),torch.arange(count,device=device));ucb.update(xx[:8],torch.linspace(0,1,8,device=device))
    run();sync(device);times=[]
    for _ in range(5):
     start=time.perf_counter();run();sync(device);times.append(time.perf_counter()-start)
    median=float(np.median(times));row={'kind':kind,'device':device,'unique_molecules':count,'repeats':5,'median_seconds':median,'molecules_per_second':count/median}
    result.append(row);print(json.dumps(row),flush=True)
 csvout(a.output/'decision_breakdown.csv',result)
 dump(a.output/'decision_breakdown.json',{'status':'completed','role':'follow-up after observing aggregate geometry extraction bottleneck',
  'real_unique_pairs':len(cohort),'cached_feature_semantics':'same verified45 components','rows':result,'scope':'resident numerical stages; no chemistry, transfer or API; not full loop speedup'})

if __name__=='__main__':main()
