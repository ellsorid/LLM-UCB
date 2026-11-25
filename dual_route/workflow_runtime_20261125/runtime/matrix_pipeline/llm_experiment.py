"""P10 model/configuration pilot and short end-to-end matrix-loop debugging."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,random,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import torch
from .gateway import Gateway,MODELS,CONFIGS,ModelClient,parameters
from .runtime import dump,calibration
from .fair_loop import CRITIC,run
from .core import MANAGEMENT,load_cohort
from .tensors import FrozenGNN

def check(parsed,packet):
 if not isinstance(parsed,dict) or not isinstance(parsed.get('evaluations'),list):return False
 result=parsed['evaluations']
 if any(not isinstance(r,dict) for r in result):return False
 ids=[r.get('id') for r in result]
 if any(not isinstance(i,str) for i in ids):return False
 if len(ids)!=len(set(ids)) or set(ids)!={r['id'] for r in packet['candidates']}:return False
 for r in result:
  score=r.get('sp_score')
  if isinstance(score,bool) or not isinstance(score,(int,float)) or not 0<=score<=100:return False
  refs=r.get('evidence_ids',[])
  if not isinstance(refs,list) or any(e not in {v['id'] for v in packet['candidate_evidence'][r['id']]} for e in refs):return False
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--key-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--phase',choices=['probe','fixed','flow','all'],default='all');p.add_argument('--models',nargs='+',default=MODELS);p.add_argument('--retry-unavailable',action='store_true');a=p.parse_args()
 root=a.output;root.mkdir(parents=True,exist_ok=True)
 source=MANAGEMENT/'runs/P11/20261121_matrix_loop_v2'
 packets=sorted(source.glob('review_[123]_*.json'))
 if len(packets)!=3:raise ValueError('Expected three frozen v2 child-review packets')
 frozen=[json.loads(f.read_text(encoding='utf-8'))['request'] for f in packets]
 manifest=[{'packet':i,'source_name':f.name,'sha256':hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest(),'target_ids':[r['id'] for r in v['candidates']]} for i,(f,v) in enumerate(zip(packets,frozen))]
 config={'models':MODELS,'configurations':{m:{c:parameters(m,c) for c in CONFIGS} for m in MODELS},'frozen_packets':manifest,'repeats':3,
  'fixed_calls_max':81,'global_requests_max':144,'frozen_framework':'matrix-loop-v2; no GNN training/matrix ablations',
  'scope':'development robustness pilot; no independent chemical truth labels','thought_text_saved':False,'max_generated_tokens_per_call':8192}
 if (root/'config.json').exists():
  old=json.loads((root/'config.json').read_text(encoding='utf-8'))
  if old!=config:raise ValueError('Frozen experiment configuration changed')
 else:dump(root/'config.json',config)
 for i,v in enumerate(frozen):dump(root/f'packet_{i}.json',v)
 gateway=Gateway(a.key_file,root/'api',144)
 availability=json.loads((root/'availability.json').read_text(encoding='utf-8')) if (root/'availability.json').exists() else {}
 if a.phase in ['probe','all']:
  for model in a.models:
   previous=availability.get(model,{})
   if previous and not a.retry_unavailable:continue
   availability[model]={}
   for setting in CONFIGS:
    parsed,audit=gateway.call(model,setting,f'probe_{model}_{setting}','Return JSON only: {"ok":true}.',{'task':'API compatibility and mode-metadata probe'},2048)
    audit['probe_JSON_valid']=isinstance(parsed,dict) and parsed.get('ok') is True
    availability[model][setting]=audit;dump(root/'availability.json',availability)
    print(json.dumps({'phase':'probe','model':model,'setting':setting,'status':audit['status'],'http':audit['http_status'],'reasoning':audit.get('thinking_observation'),'error':audit.get('error_excerpt')}),flush=True)
    if audit.get('error_code')=='insufficient_user_quota':
     dump(root/'execution_status.json',{'experiment_status':'blocked_gateway_quota','requests_used':gateway.calls,'formal_successful_calls':0,'resume':'--retry-unavailable after quota/channel restoration'})
     return
    if audit.get('http_status') in [401,403,404,503]:break
 if a.phase in ['fixed','all']:
  jobs=[(m,c,i,r) for m in a.models for c in CONFIGS if availability.get(m,{}).get(c,{}).get('probe_JSON_valid') for i in range(3) for r in range(3)]
  random.Random(1711).shuffle(jobs)
  def execute(job):
   model,setting,packet,repeat=job;name=f'{model}_{setting}_p{packet}_r{repeat}';path=root/'fixed'/f'{name}.json'
   if path.exists():return None
   result,audit=gateway.call(model,setting,name,CRITIC,frozen[packet],8192)
   valid=check(result,frozen[packet]);row={'model':model,'setting':setting,'packet':packet,'repeat':repeat,'schema_valid':valid,'result':result,'audit':audit}
   dump(path,row);return {'phase':'fixed','model':model,'setting':setting,'packet':packet,'repeat':repeat,'status':audit['status'],'schema':valid,'seconds':round(audit['seconds'],2),'calls':gateway.calls}
  with ThreadPoolExecutor(max_workers=2) as pool:
   for future in as_completed([pool.submit(execute,j) for j in jobs]):
    value=future.result()
    if value:print(json.dumps(value),flush=True)
 if a.phase in ['flow','all']:
  import os
  os.environ['LOHC_BUNDLED_ONLY']='1';torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False
  cohort=load_cohort();device='cuda' if torch.cuda.is_available() else 'cpu';gnn=FrozenGNN(device)
  for model in a.models:
   for setting in CONFIGS:
    if not availability.get(model,{}).get(setting,{}).get('probe_JSON_valid'):continue
    fixed=list((root/'fixed').glob(f'{model}_{setting}_*.json'))
    successful=sum(json.loads(f.read_text(encoding='utf-8'))['schema_valid'] for f in fixed)
    if successful<6:continue
    out=root/'flow'/f'{model}_{setting}'
    if (out/'loop_metrics.json').exists() or (out/'failure.json').exists():continue
    out.mkdir(parents=True,exist_ok=True);client=ModelClient(gateway,model,setting,'flow_'+model+'_'+setting)
    try:
     atlas,ucb,_,_,_=calibration(cohort,device,out,Path(__file__).parent/'data/calibration.pt')
     metrics,_=run(cohort,atlas,ucb,gnn,device,out,2,client)
     print(json.dumps({'phase':'flow_complete','model':model,'setting':setting,'rounds':metrics['rounds'],'occupied':metrics['final_archive_cells'],'calls':client.calls}),flush=True)
    except Exception as exc:
     dump(out/'failure.json',{'error_type':type(exc).__name__,'message':gateway.clean(exc),'API_calls':client.calls})
     print(json.dumps({'phase':'flow_failed','model':model,'setting':setting,'error':type(exc).__name__}),flush=True)
 dump(root/'execution_status.json',{'requests_used':gateway.calls,'limit':144,'phase':a.phase,'requested_models':a.models})
 print(json.dumps({'phase':'finished','requests_used':gateway.calls}),flush=True)

if __name__=='__main__':main()
