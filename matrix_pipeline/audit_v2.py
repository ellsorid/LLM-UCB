"""Audit the corrected live run and immutable pretrained checkpoint manifest."""
import argparse,hashlib,json,unittest
from pathlib import Path
import numpy as np
import torch
from .runtime import dump
from .test_v2 import V2Tests
from .operators import OPERATORS

def rows(path):
 import csv
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--portable',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(V2Tests);result=unittest.TextTestRunner(verbosity=1).run(suite)
 if not result.wasSuccessful():raise AssertionError('Regression tests failed')
 metrics=json.loads((args.run/'loop_metrics.json').read_text(encoding='utf-8'));seeds=rows(args.run/'seed_evaluations.csv');children=rows(args.run/'selected_candidates.csv');events=rows(args.run/'validation_events.csv')
 protocols={r['protocol_sha256'] for r in seeds+children}
 if len(protocols)!=1 or protocols!={metrics['critic_protocol_sha256']}:raise AssertionError('Seed/child policy mismatch')
 audits=[json.loads(p.read_text(encoding='utf-8')) for p in (args.run/'api').glob('*audit.json')]
 if len(audits)!=8 or any(r['http_status']!=200 for r in audits):raise AssertionError('Live API acceptance failed')
 increments={k:[] for k in OPERATORS};capacity_gain={k:0 for k in OPERATORS}
 for r in events:
  if r['status']!='valid':continue
  increments[r['kind']].append(float(r['delta_nH2']))
  capacity_gain[r['kind']]+=float(r['delta_capacity'])>0
 for kind,values in increments.items():
  if not values or not np.allclose(values,OPERATORS[kind]['delta_nH2']):raise AssertionError('Invalid stoichiometry '+kind)
 if capacity_gain['ethynyl_ethyl']==0:raise AssertionError('Capacity cannot improve')
 state=torch.load(args.run/'final_state.pt',map_location='cpu',weights_only=True)
 if state['ucb_inverse'].shape!=(49,49) or not torch.isfinite(state['ucb_inverse']).all():raise AssertionError('State-aware UCB dimensions')
 portable=json.loads((args.portable/'loop_metrics.json').read_text(encoding='utf-8'))
 if portable['API_calls']!=0 or portable['rounds']!=2:raise AssertionError('Portable smoke failed')
 root=Path(__file__).resolve().parents[1];manifest=json.loads((root/'models/lean_matrix/checkpoint_manifest.json').read_text())
 for r in manifest['checkpoints']:
  p=root/'models/lean_matrix'/r['file']
  if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise AssertionError('Checkpoint changed')
 report={'passed':True,'version':'matrix-loop-v2','regression_tests':result.testsRun,'live_rounds':metrics['rounds'],'live_API_requests':len(audits),
  'API_HTTP_statuses':[r['http_status'] for r in audits],'tokens':sum(r['usage']['total_tokens'] for r in audits),
  'initial_archive_cells':metrics['initial_archive_cells'],'final_archive_cells':metrics['final_archive_cells'],
  'bank_pairs':metrics['bank_size'],'seed_evaluations':len(seeds),'child_evaluations':len(children),'same_critic_protocol':True,
  'valid_operator_cases':{k:len(v) for k,v in increments.items()},'capacity_improving_cases':capacity_gain,
  'H2_stoichiometry_checked':True,'fair_parent_and_operator_tests':True,'UCB_coefficient_dimensions':49,
  'CPU_GPU_regression_included':torch.cuda.is_available(),'portable_CPU_rounds':2,'portable_requires_external_project':False,
  'checkpoints':len(manifest['checkpoints']),'original_weight_bytes':manifest['total_bytes'],'unchanged_original_weights':True,
  'GNN_training':False,'scope':'functional validation, not independent chemical accuracy or legacy-policy superiority'}
 dump(args.output,report);print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
