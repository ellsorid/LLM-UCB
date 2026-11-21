"""The CPU chemistry boundary; every numerical consumer uses paired matrices."""
import sys
sys.dont_write_bytecode=True
import csv,gzip,json,hashlib,importlib.util,os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from rdkit import Chem,RDConfig
from rdkit.Chem import rdMolDescriptors,rdCIPLabeler
sys.path.insert(0,str(Path(RDConfig.RDContribDir)/'SA_Score'))
import sascorer

REPOSITORY=Path(__file__).resolve().parents[1]
PROJECT=Path(os.environ.get('LOHC_PROJECT_ROOT',r'D:\GNN STAFF'))
MANAGEMENT=PROJECT/'experiment_management'
from .compat.pair_core import dense_from_state,validate_mapping,state_from_dense,VERSION as CORE_VERSION
from .compat.p04_features import paired_krylov
_spec=importlib.util.spec_from_file_location('matrix_pipeline_compiler',Path(__file__).parent/'compat/compile_lohc_multichannel.py')
compiler=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(compiler)

@dataclass
class Molecule:
 uid:str
 pair:np.ndarray # [2,n,n,12], common atom coordinates
 identity:tuple
 root:str
 parent:str|None=None
 action:dict|None=None
 source_row:int|None=None
 gnn_patch:list|None=None

def identity(pair):
 return hashlib.sha256(json.dumps(pair,separators=(',',':')).encode()).hexdigest()

def load_cohort():
 path=MANAGEMENT/'runs/P02/20261121_structural_space_v1/task_labels.csv'
 if not path.exists() or os.environ.get('LOHC_BUNDLED_ONLY')=='1':
  rows=[json.loads(s) for s in (Path(__file__).parent/'data/seeds.jsonl').read_text(encoding='utf-8').splitlines()]
  return [Molecule(r['reaction_id'],np.stack([dense_from_state(r['states'][s]) for s in ['lean','rich']]),('', ''),r['reaction_id']) for r in rows]
 with path.open(encoding='utf-8-sig',newline='') as f:labels=list(csv.DictReader(f))
 with (MANAGEMENT/'runs/P00/20261121_pair_core_audit_v1/canonical_structure_keys.csv').open(encoding='utf-8-sig',newline='') as f:
  keys={int(r['source_row']):(r['lean_canonical'],r['rich_canonical']) for r in csv.DictReader(f)}
 wanted={int(r['source_row']):i for i,r in enumerate(labels)};result=[None]*len(labels)
 with gzip.open(MANAGEMENT/'runs/P00/20261121_pair_core_audit_v1/shared_pairs.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   core=json.loads(line);row=int(core['provenance']['source_row'])
   if row not in wanted:continue
   lean=dense_from_state(core['states']['lean']);rich=validate_mapping(lean,dense_from_state(core['states']['rich']),core['mapping']['lean_to_rich'])
   # Identity strings are boundary labels, never features or model inputs.
   pair=keys[row]
   uid=identity(pair);result[wanted[row]]=Molecule(uid,np.stack([lean,rich]),pair,uid,source_row=row,gnn_patch=core['extensions'].get('legacy_lean_patch',[]))
 if any(v is None for v in result):raise ValueError('Incomplete frozen cohort')
 if len({r.uid for r in result})!=len(result):raise ValueError('Identity duplicates in frozen cohort')
 return result

def to_mol(matrix):
 """Construct chemistry from atom/bond facts. Generated edits reject stereo below."""
 m=np.asarray(matrix);n=len(m);rw=Chem.RWMol()
 for i in range(n):
  z,q,h,arom,iso,rad,stereo=m[i,i,:7]
  a=Chem.Atom(int(z));a.SetFormalCharge(int(q));a.SetNumExplicitHs(int(h));a.SetNoImplicit(True)
  a.SetIsAromatic(bool(arom));a.SetIsotope(int(iso));a.SetNumRadicalElectrons(int(rad));rw.AddAtom(a)
 for i in range(n):
  for j in range(i+1,n):
   order=float(m[i,j,7])
   if order:
    rw.AddBond(i,j,{1.:Chem.BondType.SINGLE,1.5:Chem.BondType.AROMATIC,2.:Chem.BondType.DOUBLE,3.:Chem.BondType.TRIPLE}[order])
    rw.GetBondBetweenAtoms(i,j).SetIsAromatic(bool(m[i,j,8]))
 mol=rw.GetMol();Chem.SanitizeMol(mol);Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
 # Restore specified atom CIP labels by trying tetrahedral orientation.
 for i in range(n):
  target={1:'R',2:'S'}.get(int(m[i,i,6]))
  if target:
   a=mol.GetAtomWithIdx(i);a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
   Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
   if not a.HasProp('_CIPCode') or a.GetProp('_CIPCode')!=target:a.InvertChirality()
 Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
 # Current generation excludes bond-stereo cases; load identity uses original keys.
 return mol

def from_mol(mol):
 Chem.SanitizeMol(mol);Chem.AssignStereochemistry(mol,cleanIt=True,force=True);rdCIPLabeler.AssignCIPLabels(mol)
 n=mol.GetNumAtoms();m=np.zeros((n,n,12),np.float32)
 for i,a in enumerate(mol.GetAtoms()):
  m[i,i,:7]=[a.GetAtomicNum(),a.GetFormalCharge(),a.GetTotalNumHs(),a.GetIsAromatic(),a.GetIsotope(),a.GetNumRadicalElectrons(),compiler.get_atom_chirality(a)]
 for b in mol.GetBonds():
  i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();m[i,j,7:]=[compiler.get_bond_order(b),b.GetIsAromatic(),b.GetIsConjugated(),b.IsInRing(),compiler.get_bond_stereochemistry(b)];m[j,i,7:]=m[i,j,7:]
 return m

def validate_child(pair,parent,action):
 if np.any(parent.pair[:,:,:,11]) or np.any(np.diagonal(parent.pair[:,:,:,:7],axis1=1,axis2=2)[:,6]):
  raise ValueError('stereochemical edits not supported by append-only operator v1')
 state=[to_mol(m) for m in pair];normalized=np.stack([from_mol(m) for m in state])
 n=len(parent.pair[0])
 # Check retained scaffold/order/H counts; conjugation flags may legitimately change.
 for s in range(2):
  if not np.array_equal(normalized[s,:n,:n,7],parent.pair[s,:,:,7]):raise ValueError('Parent scaffold changed')
 if not np.array_equal(normalized[0,:,:,7]!=0,normalized[1,:,:,7]!=0):raise ValueError('Two-state topology differs')
 expected=float(action.get('delta_nH2',0))
 if not np.isclose(properties(normalized)[0]-properties(parent.pair)[0],expected):raise ValueError('Unexpected H2 stoichiometry')
 p=tuple(Chem.MolToSmiles(m,isomericSmiles=True) for m in state)
 uid=identity(p)
 return Molecule(uid,normalized,p,parent.root,parent.uid,action)

def properties(pair):
 # Matrix-derived hydrogen release and mass; no target delta_H is loaded.
 diag=np.diagonal(pair,axis1=1,axis2=2).transpose(0,2,1)
 nh2=float((diag[1,:,2].sum()-diag[0,:,2].sum())/2)
 periodic=Chem.GetPeriodicTable()
 mass=sum((periodic.GetMassForIsotope(int(a[0]),int(a[4])) if a[4] else periodic.GetAtomicWeight(int(a[0])))+float(a[2])*periodic.GetAtomicWeight(1) for a in diag[1])
 return nh2,100*nh2*2*periodic.GetAtomicWeight(1)/mass

def synthesis_score(pair):
 # Frozen legacy SA heuristic at the chemistry boundary, from the shared matrix.
 return float(sascorer.calculateScore(to_mol(pair[0])))

def packet(molecule):
 # Transport envelope: shared-core state tables, not the full provenance schema.
 return {'version':CORE_VERSION,'id':molecule.uid,'states':{'lean':state_from_dense(molecule.pair[0]),'rich':state_from_dense(molecule.pair[1])},
  'mapping':{'lean_to_rich':list(range(len(molecule.pair[0]))),'status':'aligned'},'parent':molecule.parent,'root':molecule.root}

def shared_core(molecule):
 return {'schema_version':CORE_VERSION,'reaction_id':molecule.uid,
  'states':{'lean':state_from_dense(molecule.pair[0]),'rich':state_from_dense(molecule.pair[1])},
  'mapping':{'lean_to_rich':list(range(len(molecule.pair[0]))),'status':'unverified','method_version':'explicit_aligned_runtime_map_v1','alternatives':[]},
  'provenance':{'source_row':molecule.source_row,'normalizer':'current matrix compiler; RDKit validated',
   'mapping_note':'bijection and element/topology consistency checked; uniqueness across automorphisms not asserted'},
  'extensions':{'matrix_loop_v1':{'parent':molecule.parent,'root':molecule.root,'action':molecule.action}}}

def legacy_evidence_molecule(record):
 states=[]
 for name in ['AL','AR']:
  raw=np.asarray(record['matrices'][name],dtype=float);n=len(raw);m=np.zeros((n,n,12),np.float32)
  m[:,:,7]=raw[:,:n];m[:,:,8]=(m[:,:,7]==1.5)
  for i in range(n):
   z,q,h,isotope,arom=raw[i,n:];m[i,i,:7]=[z,q,h,arom,isotope,0,0]
  states.append(from_mol(to_mol(m)))
 n=len(states[0]);mapping=np.argmax(np.asarray(record['matrices']['P'])[:n,:n],axis=1)
 rich=validate_mapping(states[0],states[1],mapping);pair=np.stack([states[0],rich])
 return Molecule('memory_'+record['id'],pair,('', ''),'literature')
