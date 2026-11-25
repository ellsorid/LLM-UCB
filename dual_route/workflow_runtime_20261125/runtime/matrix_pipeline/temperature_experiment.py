"""Frozen official-API temperature pilot, 48 completion requests maximum."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,http.client,json,os,random,threading
from pathlib import Path
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor,as_completed
import torch
from .api import credential
from .gateway import Gateway,ModelClient
from .llm_experiment import check
from .runtime import dump,calibration
from .fair_loop import CRITIC,run
from .core import MANAGEMENT,load_cohort
from .tensors import FrozenGNN

TEMPERATURES={'t0':0.0,'t05':0.5,'t10':1.0}
MODEL='deepseek-flash'

class OfficialTemperatureGateway(Gateway):
 def __init__(self,key_file,output,limit=48):
  self.key=credential(Path(key_file));self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
  self.url=urlsplit('https://api.deepseek.com');self.base='';self.lock=threading.Lock()
  self.calls=len(list(self.output.glob('*_audit.json')));self.limit=limit;self.quota_blocked=False
 def request_parameters(self,model,config):
  if model!=MODEL:raise ValueError('Frozen model mismatch')
  return {'temperature':TEMPERATURES[config],'thinking':{'type':'disabled'}}

class TemperatureClient(ModelClient):
 def run(self,*args,**kwargs):
  result=super().run(*args,**kwargs)
  return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--key-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--phase',choices=['probe','fixed','flow','all'],default='all');a=p.parse_args()
 root=a.output;root.mkdir(parents=True,exist_ok=True)
 packets=sorted((MANAGEMENT/'runs/P11/20261121_matrix_loop_v2').glob('review_[123]_*.json'))
 if len(packets)!=3:raise ValueError('Expected three frozen review packets')
 frozen=[json.loads(f.read_text(encoding='utf-8'))['request'] for f in packets]
 manifest=[{'packet':i,'source_name':f.name,'sha256':hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest(),'target_ids':[r['id'] for r in v['candidates']]} for i,(f,v) in enumerate(zip(packets,frozen))]
 config={'provider':'DeepSeek official','model':MODEL,'temperatures':TEMPERATURES,'thinking':'disabled','packets':manifest,'repeats':3,
  'fixed_calls':27,'flow_rounds':2,'flow_calls_max':18,'probe_calls':3,'completion_calls_max':48,'output_token_limit':8192,
  'job_order_seed':1711,'GPU_seed':17,'concurrency':2,'scope':'development robustness pilot; no independent chemical truth labels',
  'critic_sha256':hashlib.sha256(CRITIC.encode()).hexdigest(),'training':False}
 if (root/'config.json').exists():
  if json.loads((root/'config.json').read_text(encoding='utf-8'))!=config:raise ValueError('Frozen configuration changed')
 else:dump(root/'config.json',config)
 for i,v in enumerate(frozen):dump(root/f'packet_{i}.json',v)
 g=OfficialTemperatureGateway(a.key_file,root/'api')
 availability=json.loads((root/'availability.json').read_text(encoding='utf-8')) if (root/'availability.json').exists() else {}
 def save_status(stage):
  rows=[json.loads(f.read_text(encoding='utf-8')) for f in (root/'fixed').glob('*.json')]
  dump(root/'execution_status.json',{'phase':stage,'requests_used':g.calls,'limit':48,'fixed_attempts':len(rows),
   'fixed_schema_valid':sum(r['schema_valid'] for r in rows),'completed_flows':len(list((root/'flow').glob('*/loop_metrics.json'))),
   'failed_flows':len(list((root/'flow').glob('*/failure.json'))),'quota_blocked':g.quota_blocked})
 if a.phase in ['probe','all']:
  for setting in TEMPERATURES:
   if setting in availability:continue
   parsed,audit=g.call(MODEL,setting,'probe_'+setting,'Return JSON only: {"ok":true}.',{'task':'temperature compatibility probe'},2048)
   audit['probe_valid']=isinstance(parsed,dict) and parsed.get('ok') is True and not audit.get('requested_off_but_reasoning_observed',False)
   availability[setting]=audit;dump(root/'availability.json',availability);save_status('probe')
   print(json.dumps({'phase':'probe','setting':setting,'status':audit['status'],'http':audit['http_status'],'valid':audit['probe_valid'],'returned_model':audit.get('returned_model')}),flush=True)
   if audit.get('http_status') in [401,402,403] or g.quota_blocked:return
 if a.phase in ['fixed','all']:
  jobs=[(c,i,r) for c in TEMPERATURES if availability.get(c,{}).get('probe_valid') for i in range(3) for r in range(3)]
  random.Random(1711).shuffle(jobs)
  def execute(job):
   setting,packet,repeat=job;name=f'{setting}_p{packet}_r{repeat}';path=root/'fixed'/f'{name}.json'
   if path.exists():return None
   result,audit=g.call(MODEL,setting,name,CRITIC,frozen[packet],8192)
   valid=check(result,frozen[packet])
   row={'model':MODEL,'setting':setting,'temperature':TEMPERATURES[setting],'packet':packet,'repeat':repeat,'schema_valid':valid,
    'mode_valid':not audit.get('requested_off_but_reasoning_observed',False),'result':result,'audit':audit}
   dump(path,row)
   return {'phase':'fixed','setting':setting,'packet':packet,'repeat':repeat,'status':audit['status'],'schema':valid,'seconds':round(audit.get('seconds',0),2),'requests_used':g.calls}
  with ThreadPoolExecutor(max_workers=2) as pool:
   for future in as_completed([pool.submit(execute,j) for j in jobs]):
    value=future.result()
    if value:print(json.dumps(value),flush=True);save_status('fixed')
 if a.phase in ['flow','all'] and not g.quota_blocked:
  os.environ['LOHC_BUNDLED_ONLY']='1';torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
  cohort=load_cohort();device='cuda' if torch.cuda.is_available() else 'cpu';gnn=FrozenGNN(device)
  for setting in TEMPERATURES:
   rows=[json.loads(f.read_text(encoding='utf-8')) for f in (root/'fixed').glob(f'{setting}_*.json')]
   if sum(r['schema_valid'] and r['mode_valid'] for r in rows)<6:continue
   if g.calls+6>48:break
   out=root/'flow'/setting
   if (out/'loop_metrics.json').exists() or (out/'failure.json').exists():continue
   torch.manual_seed(17);out.mkdir(parents=True,exist_ok=True)
   client=TemperatureClient(g,MODEL,setting,'flow_'+setting)
   try:
    atlas,ucb,_,_,_=calibration(cohort,device,out,Path(__file__).parent/'data/calibration.pt')
    metrics,_=run(cohort,atlas,ucb,gnn,device,out,2,client)
    print(json.dumps({'phase':'flow_complete','setting':setting,'device':device,'occupied':metrics['final_archive_cells'],'calls':client.calls}),flush=True)
   except Exception as exc:
    dump(out/'failure.json',{'error_type':type(exc).__name__,'message':g.clean(exc),'API_calls':client.calls})
    print(json.dumps({'phase':'flow_failed','setting':setting,'error_type':type(exc).__name__}),flush=True)
   save_status('flow')
   if g.quota_blocked:break
 save_status('finished')
 print(json.dumps({'phase':'finished','requests_used':g.calls}),flush=True)

if __name__=='__main__':main()
