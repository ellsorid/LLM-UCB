"""Bounded executable for the audited paired-matrix generation workflow."""
import sys
sys.dont_write_bytecode=True
import argparse,csv,hashlib,json,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from scipy import sparse
from .core import MANAGEMENT,load_cohort
from .tensors import pack,qd_features,subspace45,Atlas,StateUCB,FrozenGNN
from .api import Client

def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def csvout(path,rows):
 if not rows:return
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8-sig',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
  writer.writeheader();writer.writerows(rows)

def sync(device):
 if str(device).startswith('cuda'):torch.cuda.synchronize()

def stage(timing,name,device,fn):
 sync(device);start=time.perf_counter();result=fn();sync(device)
 timing[name]=timing.get(name,0)+time.perf_counter()-start
 return result

def calibration(cohort,device,out,provided=None):
 pair,mask=pack(cohort,device)
 old=MANAGEMENT/'runs/P04/20261121_typed_krylov_v1'
 bundled=Path(__file__).parent/'data/calibration.pt'
 use_bundle=provided or os.environ.get('LOHC_BUNDLED_ONLY')=='1' or not (old/'krylov_joint.npz').exists()
 if use_bundle:
  state=torch.load(provided or bundled,map_location=device,weights_only=True)
  atlas=Atlas(state['descriptor_mean'],state['descriptor_scale'],state['centers'])
  ucb=StateUCB(state['context_mean'][:45],state['context_scale'][:45])
  context=subspace45(pair,mask)
 else:
  descriptor=qd_features(pair,mask);atlas=Atlas.fit(descriptor,cells=64)
  names=json.loads((old/'feature_manifest.json').read_text(encoding='utf-8'))['krylov_joint']
  columns=[i for i,n in enumerate(names) if not n.endswith('_mean_angle') and not n.endswith('_max_angle')]
  context=torch.tensor(sparse.load_npz(old/'krylov_joint.npz').toarray()[:,columns],device=device)
  if len(context)!=len(cohort):raise ValueError('Calibration cohort does not match cached features')
  ucb=StateUCB(context.mean(0),context.std(0,unbiased=False).clamp_min(1e-6))
 state={'descriptor_mean':atlas.mean.cpu(),'descriptor_scale':atlas.scale.cpu(),'centers':atlas.centers.cpu(),
  'context_mean':ucb.mean[:45].cpu(),'context_scale':ucb.scale[:45].cpu(),
  'fit_scope':'frozen9812-pair development atlas; not heldout evaluation','used_bundled_calibration':bool(use_bundle)}
 torch.save(state,out/'calibration.pt')
 return atlas,ucb,pair,mask,context

def memory_records():
 path=Path(__file__).parent/'data/literature_memory.json'
 return json.loads(path.read_text(encoding='utf-8')),hashlib.sha256(path.read_bytes()).hexdigest()

def evaluate_features(pair,mask):return qd_features(pair,mask),subspace45(pair,mask)

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
 parser.add_argument('--rounds',type=int,default=3)
 parser.add_argument('--key-file',type=Path)
 parser.add_argument('--live',action='store_true')
 parser.add_argument('--bundled-only',action='store_true')
 parser.add_argument('--calibration',type=Path)
 parser.add_argument('--bench',action='store_true')
 parser.add_argument('--concurrency-probe',action='store_true')
 parser.add_argument('--skip-concurrency-probe',action='store_true',help=argparse.SUPPRESS)
 args=parser.parse_args()
 if not 1<=args.rounds<=3:raise ValueError('Bounded validation runner supports1–3 rounds')
 if args.bundled_only:os.environ['LOHC_BUNDLED_ONLY']='1'
 torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
 out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 max_calls=2+2*args.rounds+(2 if args.concurrency_probe and not args.skip_concurrency_probe else 0)
 dump(out/'config.json',{'version':'matrix-loop-v2','device':args.device,'rounds':args.rounds,'live':args.live,'max_API_calls':max_calls,
  'GNN_training':False,'same_seed_and_child_critic':True,'parent_selection':'least-visited with stable identity ties',
  'action_budget':'balanced parent/operator groups96','UCB':'45 structural +3 pre-update archive-state fields +intercept',
  'operators':['methyl','amino','hydroxyl','vinyl_ethyl','ethynyl_ethyl'],'feasibility':{'H2_wt_percent_min':4.2,'enthalpy_window':[40,70],'SA_max':4}})
 cohort=load_cohort();atlas,ucb,pair,mask,context=calibration(cohort,args.device,out,args.calibration)
 gnn=FrozenGNN(args.device)
 dump(out/'inputs.json',{'pairs_loaded':len(cohort),'checkpoint':str(gnn.path),'checkpoint_sha256':hashlib.sha256(gnn.path.read_bytes()).hexdigest(),
  'seeds_sha256':hashlib.sha256((Path(__file__).parent/'data/seeds.jsonl').read_bytes()).hexdigest()})
 if args.bench:
  if len(cohort)!=9812:raise ValueError('Full benchmark requires the external9812-pair audit data')
  from .validation import validate_and_benchmark
  validate_and_benchmark(cohort,pair,mask,context,atlas,ucb,gnn,args.device,out)
 client=Client(args.key_file,out/'api',max_calls=max_calls) if args.live else None
 from .fair_loop import run,CRITIC
 result,_=run(cohort,atlas,ucb,gnn,args.device,out,args.rounds,client)
 if client and args.concurrency_probe and not args.skip_concurrency_probe:
  reviews=sorted(out.glob('review_*.json'));request=json.loads(reviews[-1].read_text(encoding='utf-8'))['request']
  tick=time.perf_counter()
  with ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(client.run,f'parallel_probe_{i}',CRITIC,request,1800) for i in range(2)]
   replies=[f.result() for f in futures]
  dump(out/'api_concurrency.json',{'concurrency':2,'requests':2,'seconds':time.perf_counter()-tick,'JSON_valid':all('evaluations' in r for r in replies)})
 result.update(API_calls=client.calls if client else 0,total_wall_seconds=time.perf_counter()-start)
 dump(out/'loop_metrics.json',result)
 print(json.dumps({'completed':True,'output':str(out),'API_calls':result['API_calls'],'wall_seconds':result['total_wall_seconds']}),flush=True)
