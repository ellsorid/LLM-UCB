"""Typed graph operators and paired Krylov geometry in shared atom coordinates."""
import numpy as np


def typed_state(m):
    m=np.asarray(m,dtype=float);n=len(m);z=np.diag(m[:,:,0]);ring=m[:,:,10];conj=m[:,:,9];adj=(m[:,:,7]!=0).astype(float)
    vectors={x:(z==x).astype(float) for x in (7,8,9)};hetero=(~np.isin(z,[1,6])).astype(float)
    rp={k:np.linalg.matrix_power(ring,k) for k in (2,3,4)}
    cp={k:np.linalg.matrix_power(conj,k) for k in (2,3,4)}
    ap={k:np.linalg.matrix_power(adj,k) for k in (1,2,3)}
    basic=[];bn=[];higher=[];hn=[]
    for x,v in vectors.items():
        basic.extend([v.sum(),v@np.diag(rp[2])/n]);bn.extend([f'Z{x}_count',f'Z{x}_ring_trace_k2_per_n'])
        for k in (3,4):higher.append(v@np.diag(rp[k])/n);hn.append(f'Z{x}_ring_trace_k{k}_per_n')
        for k in (2,3,4):higher.append(v@np.diag(cp[k])/n);hn.append(f'Z{x}_conj_trace_k{k}_per_n')
        for k in (1,2,3):higher.append(v@ap[k]@v/n);hn.append(f'Z{x}_self_walk_k{k}_per_n')
        higher.append(v@ring@conj@hetero/n);hn.append(f'Z{x}_ring_conj_hetero_walk_per_n')
    for k in (1,2,3):higher.append(vectors[7]@ap[k]@vectors[8]/n);hn.append(f'N_O_walk_k{k}_per_n')
    return np.asarray(basic),bn,np.asarray(higher),hn


def paired_typed(lean,rich):
    lb,bn,lh,hn=typed_state(lean);rb,_,rh,_=typed_state(rich)
    higher=list(np.r_[lh,rh,rh-lh]);higher_names=[s+'_'+name for s in ['L','R','R_minus_L'] for name in hn]
    weights=np.abs(rich[:,:,7]-lean[:,:,7]).sum(1);denominator=max(weights.sum(),1.)
    for state,m in [('L',lean),('R',rich)]:
        z=np.diag(m[:,:,0]);hetero=(~np.isin(z,[1,6])).astype(float);ring=m[:,:,10].astype(float);conj=m[:,:,9].astype(float)
        for x in (7,8,9):
            d=np.diag((z==x).astype(float))
            higher.extend([weights@ring@d@conj@hetero/denominator,weights@conj@d@ring@hetero/denominator])
            higher_names.extend([f'{state}_changed_ring_Z{x}_conj_hetero',f'{state}_changed_conj_Z{x}_ring_hetero'])
    return np.r_[lb,rb,rb-lb],[s+'_'+name for s in ['L','R','R_minus_L'] for name in bn],np.asarray(higher),higher_names


def basis(b,c,seed,joint):
    length=np.linalg.norm(seed)
    if length==0:return np.zeros((len(seed),0))
    seed=seed/length
    if joint:columns=np.column_stack([seed,b@seed,c@seed,b@c@seed])
    else:columns=np.column_stack([seed,b@seed,b@b@seed,b@b@b@seed])
    u,s,_=np.linalg.svd(columns,full_matrices=False)
    rank=np.count_nonzero(s>max(s[0]*1e-8,1e-12))
    return u[:,:rank]


def geometry(q1,q2):
    r1,r2=q1.shape[1],q2.shape[1]
    valid=bool(r1 and r2)
    if valid:
        singular=np.linalg.svd(q1.T@q2,compute_uv=False)
        singular=np.clip(singular,0,1);singular[np.abs(1-singular)<1e-12]=1
        angles=np.arccos(singular);overlap=float(np.sum(singular**2))
    else:angles=np.array([]);overlap=0.
    distance=max(0.,(r1+r2-2*overlap)/2)
    return [int(r1>0),int(r2>0),r1,r2,r2-r1,int(valid),distance,
            2*distance/max(r1+r2,1),overlap/max(min(r1,r2),1),float(angles.mean()) if valid else 0.,float(angles.max()) if valid else 0.]


def paired_krylov(lean,rich,joint=False):
    z=np.diag(lean[:,:,0]);weights=np.abs(rich[:,:,7]-lean[:,:,7]).sum(1)
    vectors={f'Z{x}':(z==x).astype(float) for x in (7,8,9)}
    vectors['hetero']=(~np.isin(z,[1,6])).astype(float);vectors['changed_site']=weights
    ops=[]
    for m in (lean,rich):
        b=m[:,:,7].astype(float);c=m[:,:,9].astype(float)
        ops.append((b/max(np.linalg.norm(b,2),1e-12),c/max(np.linalg.norm(c,2),1e-12)))
    keys=['present_L','present_R','rank_L','rank_R','rank_delta','angles_valid','chordal2','rank_normalized_chordal2','overlap_fraction','mean_angle','max_angle']
    result=[];names=[];diagnostic={}
    for name,v in vectors.items():
        left=basis(*ops[0],v,joint);right=basis(*ops[1],v,joint)
        values=geometry(left,right);result.extend(values);names.extend(name+'_'+k for k in keys)
        diagnostic[name]={'rank_L':left.shape[1],'rank_R':right.shape[1],
            'chordal2':values[6],'angles_valid':bool(values[5]),'mean_angle':values[9],
            'rank_saturated':left.shape[1]==len(z) or right.shape[1]==len(z)}
    return np.asarray(result),names,diagnostic


def all_features(lean,rich):
    basic,bn,higher,hn=paired_typed(lean,rich)
    bond,kn,bd=paired_krylov(lean,rich,False);joint,jn,jd=paired_krylov(lean,rich,True)
    return {'typed_basic':basic,'typed_higher':higher,'krylov_bond':bond,'krylov_joint':joint},\
        {'typed_basic':bn,'typed_higher':hn,'krylov_bond':kn,'krylov_joint':jn},{'bond':bd,'joint':jd}
