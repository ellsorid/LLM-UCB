"""Regression tests for storage-site edits, fair menus, archive state and checkpoints."""
import sys
sys.dont_write_bytecode=True
import hashlib,json,unittest
from pathlib import Path
import numpy as np
import torch
from .core import compiler,Molecule,identity,validate_child,properties
from .tensors import pack,StateUCB,Archive
from .operators import menu,apply,balanced_actions,balanced_parents,OPERATORS

def benzene():
 p=np.stack([compiler.smiles_to_matrix('c1ccccc1'),compiler.smiles_to_matrix('C1CCCCC1')]);uid=identity(('c1ccccc1','C1CCCCC1'))
 return Molecule(uid,p,('c1ccccc1','C1CCCCC1'),uid)

class V2Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def test_storage_sites_preserve_scaffold_and_can_increase_capacity(self):
  m=benzene();x,mask=pack([m]);actions=menu(x,mask);y,n=apply(x,mask,actions);seen=set();improved=False
  for i,a in enumerate(actions):
   if a['kind'] in seen:continue
   seen.add(a['kind']);child=validate_child(y[i,:,:int(n[i]),:int(n[i])].numpy(),m,a)
   delta=properties(child.pair)[0]-properties(m.pair)[0]
   self.assertAlmostEqual(delta,OPERATORS[a['kind']]['delta_nH2'])
   np.testing.assert_array_equal(child.pair[:,:6,:6,7],m.pair[:,:,:,7])
   if a['kind']=='ethynyl_ethyl':improved=properties(child.pair)[1]>properties(m.pair)[1]
  self.assertEqual(seen,set(OPERATORS));self.assertTrue(improved)
 def test_menu_reordering_does_not_change_budget(self):
  parents=[benzene() for _ in range(4)]
  for i,m in enumerate(parents):m.uid=f'parent{i}'
  x,mask=pack(parents);actions=menu(x,mask)
  a=balanced_actions(actions,parents,40,1);b=balanced_actions(list(reversed(actions)),parents,40,1)
  self.assertEqual(a,b)
  counts={(p,k):sum(x['parent_index']==p and x['kind']==k for x in a) for p in range(4) for k in OPERATORS}
  self.assertEqual(set(counts.values()),{2})
 def test_every_parent_is_considered_without_cell_prefix(self):
  uids=[f'uid{i}' for i in range(32)];visits=torch.zeros(32,dtype=torch.long);visited=set()
  for _ in range(3):visited.update(balanced_parents(torch.arange(32),visits,uids,12).tolist())
  self.assertEqual(len(visited),32)
  a=balanced_parents(torch.arange(32),torch.zeros(32,dtype=torch.long),uids,12)
  b=balanced_parents(torch.arange(31,-1,-1),torch.zeros(32,dtype=torch.long),uids,12)
  self.assertTrue(torch.equal(a,b))
 def test_archive_state_enters_context_and_posterior(self):
  model=StateUCB(torch.zeros(45),torch.ones(45));archive=Archive(4,'cpu');x=torch.ones((2,45));cells=torch.tensor([0,1])
  before=model.with_state(x,cells,archive).clone();archive.update(torch.tensor([0]),torch.tensor([.7]),torch.tensor([0]))
  after=model.with_state(x,cells,archive)
  self.assertNotEqual(before[0,45].item(),after[0,45].item());self.assertAlmostEqual(after[0,46].item(),.7,places=6)
  self.assertEqual(model.context(after).shape,(2,49));model.update(after,torch.tensor([0.,1.]));self.assertTrue(torch.isfinite(model.score(after)).all())
 def test_five_frozen_models_load_and_denormalize(self):
  from models.lean_matrix.model import MatrixGIN
  root=Path(__file__).resolve().parents[1]/'models/lean_matrix';m=benzene();x,mask=pack([m]);seen=0
  for p in sorted((root/'checkpoints').glob('*.pth')):
   ckpt=torch.load(p,map_location='cpu',weights_only=True);model=MatrixGIN(hidden_dim=ckpt['hidden_dim'],num_layers=ckpt['num_gnn_layers']);model.load_state_dict(ckpt['model_state_dict']);model.eval()
   with torch.inference_mode():value=model(x[:,0],mask.float())*ckpt['target_std']+ckpt['target_mean']
   self.assertTrue(torch.isfinite(value).all());self.assertGreater(ckpt['target_std'],0);seen+=1
  self.assertEqual(seen,5)
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
 def test_paired_edits_and_state_UCB_match_GPU(self):
  m=benzene();a,ma=pack([m]);g,mg=pack([m],'cuda');actions=menu(a,ma);ca,na=apply(a,ma,actions);cg,ng=apply(g,mg,actions)
  np.testing.assert_array_equal(ca.numpy(),cg.cpu().numpy());np.testing.assert_array_equal(na.numpy(),ng.cpu().numpy())
  c=StateUCB(torch.zeros(45),torch.ones(45));g=StateUCB(torch.zeros(45,device='cuda'),torch.ones(45,device='cuda'))
  xx=torch.arange(96,dtype=torch.float32).reshape(2,48)/100;reward=torch.tensor([.2,.8]);c.update(xx,reward);g.update(xx.cuda(),reward.cuda())
  np.testing.assert_allclose(c.score(xx).numpy(),g.score(xx.cuda()).cpu().numpy(),atol=1e-9)

if __name__=='__main__':unittest.main()
