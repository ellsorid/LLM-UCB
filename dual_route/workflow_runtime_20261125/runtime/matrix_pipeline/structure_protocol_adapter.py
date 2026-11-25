"""Business-call protocol injection. No API calls, scoring, or output-schema replacement."""
import copy,hashlib,json,re
from pathlib import Path
import numpy as np
from .modular_protocol import PROTOCOL_DIR,MODULES,decompose,digest,cells
from .compat.pair_core import dense_from_state

MARKER='STRUCTURE_BUSINESS_ADAPTER_V1'
PROFILES={
 'evidence_live':([], 'Use program-precomputed structures and retrieved primary texts for scoped evidence assessment. Similarity retrieves candidates; source conditions and applicability still require checking.'),
 'precomputed':([], 'Interpret program-precomputed structure for ONE state. This is supplied-structure comprehension and synthesis, not independent motif discovery. Preserve original parent IDs and complete connectivity.'),
 'whole':(['atoms','bonds','regions'],'Interpret only the requested state-specific scope. Exact cell copying is numerical readback; a selected ring is local interpretation. Whole-molecule interpretation requires all connectivity and substituents, not a ring label.'),
 'local':(['local'],'Read local structure. Radius1 reads center, connections, boundaries and raw ring flags; ring interpretation requires the relevant region context.'),
 'relation':(['local','regions','changes'],'Match local prerequisites to the whole scaffold/state; distinguish evidence from inferred applicability.'),
 'review':(['local','regions','changes'],'Property/evidence evaluation or audit only. State labels imply no aromaticity/saturation. Separate matrix facts, local motifs, complete-state interpretation and evidence-backed endpoint claims; obey the existing endpoint, source and score contract.'),
 'edit':(['local','regions','changes'],'Choose edits using actual allowed parent sites, protected structure and supplied feedback. Cross-molecule local_contrast/target_minus_parent is NOT rich-minus-lean and NOT a SOM prototype gradient.'),
 'selection':(['atoms','bonds'],'Compare supplied candidate structure only as requested; preserve the existing allowed-ID and numerical selection contract.'),
 'fact_read':(['atoms','bonds','local'],'Recover the exact program-provided facts only. This is fact readback, not independent chemical inference.'),
 'recall':(['atoms','bonds','local'],'Recover saved structure/evidence identifiers exactly; do not generate new chemistry or evidence.'),
}


def route(role,packet,agent=None):
    name=role.lower()
    if name.startswith('evidence_live_'):return 'evidence_live'
    if name.startswith('whole_precomputed_'):return 'precomputed'
    if name.startswith('whole_'):return 'whole'
    if any(x in name for x in ['probe','protocol_','matrix_original','matrix_permuted']):return None
    if name.startswith(('recall_','memory_readback_')):return 'recall'
    if name.startswith(('generate','generation','context_generate','numeric_','mapped_','edit_','planner_')):
        return 'selection' if name.startswith('planner_') and 'candidates' in packet else 'edit'
    if name.startswith('plan_'):return 'relation'
    if name.startswith('read_'):
        if agent=='global':return 'whole'
        if agent=='local':return 'local'
        if agent=='coupling':return 'relation'
        return 'fact_read'
    if name.startswith('extract_'):
        return {'global':'whole','local':'local','coupling':'relation'}.get(agent,'relation')
    if name.startswith(('score','evaluate','critic','review','evidence_audit')):return 'review'
    if 'parent' in packet and ('target_examples' in packet or 'available_attachment_sites' in packet):return 'edit'
    return None


def matrix_contexts(packet,allowed):
    """Normalize existing full/local matrices to parent cells; do not create new local scopes."""
    result=[]
    def emit(kind,state,path,data,node,owner):
        result.append(dict(block_id=path+':'+kind+':'+state,module=kind,state=state,coordinates='parent_aligned',
            molecule_id='source:'+owner,source_hash=digest(node),data=data))
    def walk(node,path='$',depth=0,owner=None):
        if depth>20:raise ValueError('Structure nesting too deep')
        if isinstance(node,list):
            for i,x in enumerate(node):walk(x,f'{path}[{i}]',depth+1,owner)
            return
        if not isinstance(node,dict):return
        for identity_key in ['record_id','id','case_id']:
            if isinstance(node.get(identity_key),(str,int)):
                owner='id:'+str(node[identity_key]);break
        # Source nodes retain local arrays; normalize scientifically used coordinates here.
        local_states=node.get('states') if 'atom_ids' in node else None
        if isinstance(local_states,dict) and {'lean','rich'}<=set(local_states):
            ids=node['atom_ids'];require_ids(ids)
            if 'local' in allowed:
                for state in ['lean','rich']:
                    m=dense_from_state(local_states[state]);local_cells=[]
                    for i in range(len(ids)):
                        local_cells.append(dict(parent_row=ids[i],parent_col=ids[i],v=m[i,i].tolist()))
                        for j in range(i+1,len(ids)):
                            if m[i,j,7]:
                                a,b=sorted([ids[i],ids[j]]);local_cells.append(dict(parent_row=a,parent_col=b,v=m[i,j].tolist()))
                    boundary=[dict(inside_parent=ids[b['local_atom']],outside_parent=b['outside_parent_atom'],v=[0.]*7+b['bond_fields']) for b in node['boundary_bonds'][state]]
                    order=np.argsort(ids);context=node['parent_context'][state]
                    emit('local',state,path,dict(radius=node['radius'],center_parent_ids=node['center_parent_atom_ids'],
                         matrix_view=dict(parent_ids=sorted(ids),cells=sorted(local_cells,key=lambda c:(c['parent_row'],c['parent_col'])),boundary_cells=boundary,
                         parent_degree=np.asarray(context['degree'])[order].tolist(),parent_ring=np.asarray(context['ring_atom'])[order].tolist()),
                         parent_hop_distances={str(i):d for i,d in zip(ids,node['hop_distance'])},uncapped=True),node,owner or path)
            return
        pair=None
        for key in ['whole_states','whole_structure','states']:
            value=node.get(key)
            if isinstance(value,dict) and {'lean','rich'}<=set(value) and all(isinstance(value[s],dict) and 'atom_fields' in value[s] for s in ['lean','rich']):pair=value;break
        if pair is not None and allowed & {'atoms','bonds','regions','changes'}:
            owner=owner or path
            n=len(pair['lean']['atoms']);mapping=node.get('mapping',dict(lean_to_rich=list(range(n))))
            mode='native' if node.get('coordinates')=='native' else 'aligned'
            record=dict(whole_structure=pair,mapping=mapping,center_parent_atom_id=0)
            # Generated auxiliary local packets are never sent: only existing local scopes are normalized above.
            bundle=decompose(record,coordinate_mode=mode)
            for b in bundle['blocks']:
                if b['module'] in allowed and b['module'] not in ['local','statistics']:
                    emit(b['module'],b['state'],path+':'+b['block_id'],b['data'],node,owner)
        # Strict memory views already have parent IDs, not local indices: retain their declared format.
        for k,v in node.items():
            if k not in ['atoms','edges','matrix_cells','cells','indexed_matrix_rows','whole_states','whole_structure','states']:
                walk(v,path+'.'+k,depth+1,owner)
    walk(packet)
    return result


def require_ids(ids):
    if not isinstance(ids,list) or len(ids)!=len(set(ids)) or not all(type(i) is int and i>=0 for i in ids):raise ValueError('Invalid source local parent map')


def inject(role,instruction,packet,agent=None):
    profile=route(role,packet,agent)
    if profile is None:return instruction,packet
    if isinstance(packet.get('_structure_protocol'),dict) and MARKER in instruction:
        previous=packet['_structure_protocol']
        current=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if previous.get('adapter_sha256')!=current or previous.get('profile')!=profile or any(hashlib.sha256((PROTOCOL_DIR/n).read_bytes()).hexdigest()!=h for n,h in previous.get('protocol_hashes',{}).items()):
            raise ValueError('Stale/different protocol binding; prepare a new request from original input')
        return instruction,packet
    modules=[]
    def add(kind):
        for dep in MODULES[kind][1]:add(dep)
        if kind not in modules:modules.append(kind)
    for kind in PROFILES[profile][0]:add(kind)
    paths=[PROTOCOL_DIR/'00_mother.md']+[PROTOCOL_DIR/MODULES[k][0] for k in modules]
    if profile=='precomputed':paths.append(PROTOCOL_DIR/'09_precomputed_structure.md')
    if profile=='evidence_live':paths.append(PROTOCOL_DIR/'10_evidence_loop.md')
    # Business output is the existing caller's schema, not the generic module-readout envelope.
    definitions=[paths[0].read_text(encoding='utf-8').split('The common readout envelope')[0]]
    definitions += [p.read_text(encoding='utf-8') for p in paths[1:]]
    context=[] if profile in ['fact_read','recall'] else matrix_contexts(packet,set(modules))
    metadata=dict(version=MARKER,profile=profile,modules=modules,
        adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output_schema='original_business_task_unchanged',
        layout='Normalized _matrix_blocks use parent coordinates. Other input arrays retain their original declared layout; local atom_ids/atom_parent_ids translate local indices. memory_view already uses parent IDs.',
        protocol_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    sent=copy.deepcopy(packet);sent['_structure_protocol']=metadata
    if context:sent['_matrix_blocks']=context
    if profile=='review':
        metadata['program_change_checks']=[dict(block_id=b['block_id'],molecule_id=b['molecule_id'],
            changed_channels=sorted({e['channel'] for e in b['data']['events']}),ring_bond_flags_changed=any(e['channel']==10 for e in b['data']['events']))
            for b in context if b['module']=='changes']
    bridge=f'''{MARKER}
The following inherited definitions explain input only. Parent-cell declarations apply
to normalized _matrix_blocks. Original structures are audit/context inputs in their declared
layout: do not mistake original local row IDs for parent IDs or reapply maps to normalized cells.
For chemical matrix inspection prefer normalized blocks when present. For an exact read/recall,
copy the authorized program facts/memory view; do not substitute newly inferred facts.
Existing task instructions and exact OUTPUT schema below remain authoritative. Do not emit
the generic observations envelope unless the existing task explicitly requests it. Preserve
fact_catalog keys, candidate IDs, evidence IDs and all task-specific site constraints.
Profile {profile}: {PROFILES[profile][1]}
For review tasks, program_change_checks is authoritative about which channels changed.
Do not claim ring flags or any other unchanged channel changed. Ground transformation prose
in those program checks, not chemical expectations or the presence of other changes.
'''
    return bridge+'\n\n'+'\n\n'.join(definitions)+'\n\nORIGINAL BUSINESS TASK:\n'+instruction,sent


def call_with_structure(gateway,model,setting,role,instruction,packet,cap):
    instruction,packet=inject(role,instruction,packet)
    if hasattr(gateway,'output'):
        folder=Path(gateway.output)/'structure_requests';folder.mkdir(parents=True,exist_ok=True)
        name=re.sub(r'[^A-Za-z0-9_.-]','_',role)+'_'+digest([instruction,packet])[:12]+'.json'
        (folder/name).write_text(json.dumps(dict(system=instruction,user=packet),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return gateway.call(model,setting,role,instruction,packet,cap)


def require_current_cached_result(path, api_output):
    """Legacy result files without an actual bound request must not silently resume."""
    row=json.loads(Path(path).read_text(encoding='utf-8'));role=row.get('audit',{}).get('role')
    if not isinstance(role,str):raise ValueError('Unbound historical result; use a new output directory')
    name=re.sub(r'[^A-Za-z0-9_.-]','_',role)
    for candidate in (Path(api_output)/'structure_requests').glob(name+'_*.json'):
        saved=json.loads(candidate.read_text(encoding='utf-8'))
        if MARKER not in saved.get('system',''):continue
        inject(role,saved['system'],saved['user'])  # validates current protocol/adapter hashes
        return
    raise ValueError('Unbound historical result; use a new output directory')
