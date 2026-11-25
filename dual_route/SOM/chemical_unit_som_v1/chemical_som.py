"""GPU SOM using whole matrix attributes and chemistry-defined local units."""
import argparse
from collections import Counter
from hashlib import blake2b
import json
from pathlib import Path
import sys
import time

import numpy as np

from chemical_units import KINDS, extract


WIDTH = 256
VERSION = "whole_attributes_chemical_units_som_v1"


def encode_token(token):
    return blake2b(repr(token).encode(),digest_size=12).hexdigest()


def unit_tokens(unit, matrix):
    """Keep contextual connectivity and anchors without encoding atom numbers."""
    atoms = {i:(z,q,h,a) for i,z,q,h,a in unit["atom_labels"]}
    neighbors = {i:[] for i in atoms}
    boundary = {i:[] for i in atoms}
    for i,j,order in unit["internal_bonds"]:
        neighbors[i].append((j,order))
        neighbors[j].append((i,order))
    for i,j,order in unit["boundary_bonds"]:
        boundary[i].append((order,int(matrix[j,j,0]),bool(matrix[j,j,3])))
    anchors = set(unit["anchor_ids"])
    incident_changes = {i:[] for i in atoms}
    for change in unit["paired_changes"]:
        for i in set(change["atoms"]) & atoms.keys():
            incident_changes[i].append((change["channel"],change["lean"],change["rich"]))
    labels = {i:encode_token((data,i in anchors,sorted(boundary[i]),sorted(incident_changes[i])))
              for i,data in atoms.items()}
    for _ in range(2):
        labels = {i:encode_token((labels[i],sorted((order,labels[j]) for j,order in neighbors[i])))
                  for i in atoms}
    prefix = (unit["state"],unit["name"])
    return [(prefix,"unit"),(prefix,"context_graph",tuple(sorted(labels.values()))),
            *((prefix,"atom_context",label) for label in labels.values())]


def raw_features(pair, base=None):
    from matrix_map.prepare import state_stats
    from rdkit import Chem

    units = extract(pair,"feature_input")["units"]
    table = Chem.GetPeriodicTable()
    values = []
    for matrix in pair:
        n = len(matrix)
        z = np.diag(matrix[:,:,0])
        hydrogens = float(np.trace(matrix[:,:,2]))
        mass = sum(table.GetAtomicWeight(int(atom)) for atom in z) + hydrogens*table.GetAtomicWeight(1)
        values.append([mass,hydrogens,float(np.trace(matrix[:,:,1])),float(np.trace(matrix[:,:,3]))/n])
    extra = np.array([*values[0],*values[1],(values[1][1]-values[0][1])/2,
                      (values[1][1]-values[0][1])*table.GetAtomicWeight(1)/max(values[1][0],1)*100],np.float32)
    if base is None:
        base = np.concatenate([state_stats(m)[0] for m in pair])
    blocks = np.zeros((len(KINDS),WIDTH),np.float32)
    for k,kind in enumerate(KINDS):
        for unit in units[kind]:
            matrix = pair[int(unit["state"]=="rich")]
            for token in unit_tokens(unit,matrix):
                blocks[k,int(encode_token(token),16)%WIDTH] += 1
    return np.r_[base,extra],blocks


def fit_transform(whole, local, train):
    mean, std = whole[train].mean(0),whole[train].std(0)
    config = {"mean":mean,"std":np.where(std>1e-6,std,1),"spread":[]}
    blocks = base_blocks(whole,local,config)
    rng = np.random.default_rng(42)
    first,second = rng.choice(train,(2,8192))
    for block in blocks:
        sampled = np.sum((block[first]-block[second])**2,axis=1)
        positive = sampled[sampled>1e-8]
        config["spread"].append(float(np.median(positive)) if len(positive) else 1.)
    return config


def base_blocks(whole, local, transform):
    whole = np.clip((whole-transform["mean"])/transform["std"],-6,6)
    local = np.log1p(local)
    local /= np.maximum(np.linalg.norm(local,axis=2,keepdims=True),1)
    return [whole,*local.transpose(1,0,2)]


def transform_arrays(whole, local, transform):
    g = transform.get("global_weight",.5)
    weights = (g,*([(1-g)/4]*4))
    return np.concatenate([a*np.sqrt(w/spread) for a,w,spread in
                           zip(base_blocks(whole,local,transform),weights,transform["spread"])],axis=1)


def encode_pairs(pairs, transform, device="cpu"):
    import torch
    data = [raw_features(pair) for pair in pairs]
    whole,local = (np.stack(x) for x in zip(*data))
    return torch.as_tensor(transform_arrays(whole,local,transform),dtype=torch.float32,device=device)


def prepare(dataset, out):
    with np.load(dataset,allow_pickle=False) as r:
        matrices, masks, global_base = r["x"],r["mask"],r["global_target"]
        split = {k:r[k] for k in ("train","val","test")}
    whole = np.zeros((len(matrices),global_base.shape[1]+10),np.float32)
    local = np.zeros((len(matrices),len(KINDS),WIDTH),np.float32)
    failures = []
    for n,(pair,mask,base) in enumerate(zip(matrices,masks,global_base)):
        ids = np.flatnonzero(mask)
        pair = pair[:,ids][:,:,ids]
        try:
            whole[n],local[n] = raw_features(pair,base)
            if n < 12:
                order = np.arange(len(ids))[::-1]
                g,l = raw_features(pair[:,order][:,:,order])
                np.testing.assert_allclose(g,whole[n],atol=1e-5,rtol=1e-6)
                np.testing.assert_array_equal(l,local[n])
        except (ValueError,RuntimeError) as error:
            failures.append({"row":n,"error":str(error)})
        if (n+1)%1000 == 0:
            print(f"Prepared {n+1}/{len(matrices)} rows; excluded {len(failures)}",flush=True)
    eligible = np.ones(len(matrices),dtype=bool)
    eligible[[f["row"] for f in failures]] = False
    split = {k:v[eligible[v]] for k,v in split.items()}
    np.savez_compressed(out/"chemical_som_dataset.npz",whole=whole,local=local,eligible=eligible,**split)
    (out/"DATASET_AUDIT.json").write_text(json.dumps({
        "version":VERSION,"source":str(dataset),"rows":len(matrices),
        "split":{k:len(v) for k,v in split.items()},"excluded":failures,
        "global_columns":whole.shape[1],"local_kinds":KINDS,"local_width_per_kind":WIDTH,
        "permutation_checks":12,"training_labels":None,
    },indent=2),encoding="utf-8")


def metrics(z, model, test, scaffold, morgan):
    import torch
    from matrix_map.selection import distances

    d,node = distances(z[test],model["centers"]).topk(2,largest=False)
    topological = model["topology"][node[:,0],node[:,1]]
    cells = node[:,0].cpu().numpy()
    i,j = np.triu_indices(len(cells),1)
    peers = cells[i]==cells[j]
    both = morgan[test].float()
    overlap = both@both.T
    count = both.sum(1)
    tanimoto = (overlap/(count[:,None]+count[None,:]-overlap).clamp_min(1)).cpu().numpy()
    corpus_distance = distances(z[test],z[test])
    scale = corpus_distance.triu(1)
    nonzero = scale[scale>1e-8]
    return {"test_rows":len(test),"occupied_test_nodes":int(len(set(cells))),
            "quantization_mean":float(d[:,0].sqrt().mean()),
            "quantization_relative_to_test_pair_rms":float(d[:,0].mean().sqrt()/nonzero.mean().sqrt()),
            "topographic_error":float((topological>1).float().mean()),
            "same_cell_pairs":int(peers.sum()),
            "same_cell_scaffold_consistency":float(np.mean(scaffold[test][i[peers]]==scaffold[test][j[peers]])) if peers.any() else None,
            "same_cell_morgan_tanimoto":float(tanimoto[i[peers],j[peers]].mean()) if peers.any() else None}


def train(dataset, baseline, frame_path, out, epochs, prepared=None, tune_balance=False):
    import pandas as pd
    import torch
    from matrix_map.selection import distances, graph_som
    from matrix_map.simple_som import represent

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    raw = np.load(prepared or out/"chemical_som_dataset.npz",allow_pickle=False)
    train_ids,val,test = (raw[k] for k in ("train","val","test"))
    whole,local = raw["whole"],raw["local"]
    transform = fit_transform(whole,local,train_ids)
    old = torch.load(baseline,map_location="cuda",weights_only=False)
    scaffold = pd.read_csv(frame_path)["scaffold"].fillna("").to_numpy()
    with np.load(dataset,allow_pickle=False) as original:
        old_z = represent(original["global_target"],original["local_target"],original["delta_target"],
                          old["transform"],old["weights"])
        fp = torch.as_tensor(original["morgan"],device="cuda")
    choices = [.25,.4,.5,.65,.75] if tune_balance else [.5]
    search = []
    runs = []
    best = None
    for weight in choices:
        transform["global_weight"] = weight
        z = torch.as_tensor(transform_arrays(whole,local,transform),device="cuda",dtype=torch.float32)
        for seed in ((42,) if tune_balance else (42,43)):
            model = graph_som(z[train_ids],k=len(old["centers"]),epochs=epochs,seed=seed)
            error = float(distances(z[val],model["centers"]).amin(dim=1).mean())
            runs.append({"seed":seed,"global_weight":weight,"validation_quantization_squared":error})
            if tune_balance:
                v = metrics(z,model,val,scaffold,fp)
                # Unsupervised fits. Independent scaffold and fingerprint are validation probes.
                criterion = (v["same_cell_scaffold_consistency"]+v["same_cell_morgan_tanimoto"])/2
                search.append({"weight":weight,"criterion":criterion,"validation":v})
            else:
                criterion = -error
            if best is None or criterion > best[0]:
                best = (criterion,model,seed,weight)
            print(f"GPU SOM seed {seed}, global weight {weight:.2f}, validation criterion {criterion:.6f}",flush=True)
    _,model,seed,weight = best
    transform["global_weight"] = weight
    z = torch.as_tensor(transform_arrays(whole,local,transform),device="cuda",dtype=torch.float32)
    d,bmu = distances(z,model["centers"]).min(dim=1)
    radius = d.sqrt()
    threshold = torch.quantile(radius[train_ids],.95)
    model["node_distance_q95"] = torch.stack([
        torch.quantile(members,.95) if len(members)>=5 else threshold
        for k in range(len(model["centers"])) for members in [radius[train_ids][bmu[train_ids]==k]]])
    model.update(version=VERSION,transform=transform,weights=[weight]+[(1-weight)/4]*4,kind="chemical_unit_graph_som",
                 dimension=z.shape[1],seed=seed)
    torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in model.items()},out/"model.pt")
    np.savez_compressed(out/"assignments.npz",node=bmu.cpu().numpy(),distance=radius.cpu().numpy(),
                        eligible=raw["eligible"],train=train_ids,val=val,test=test)
    report = {"version":VERSION,"device":torch.cuda.get_device_name(),"epochs":epochs,
              "nodes":len(model["centers"]),"feature_dimension":z.shape[1],"seed_runs":runs,
              "selected_global_weight":weight,"validation_weight_search":search,
              "old_som":metrics(old_z,old,test,scaffold,fp),"new_som":metrics(z,model,test,scaffold,fp)}
    (out/"METRICS.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    for name in ("archive","out"):
        p.add_argument(f"--{name}",required=True,type=Path)
    p.add_argument("--epochs",type=int,default=60)
    p.add_argument("--prepared",type=Path)
    p.add_argument("--tune-balance",action="store_true")
    args = p.parse_args()
    runtime = args.archive/"workflow_runtime_20261125/runtime"
    sys.path.insert(0,str(runtime))
    old = runtime/"outputs/matrix_som_v2"
    previous = runtime/"outputs/matrix_map_v1/data_split.csv"
    args.out.mkdir(exist_ok=True)
    if (args.out/"model.pt").exists():
        raise SystemExit("Use a fresh output directory for a new training run")
    start = time.perf_counter()
    if args.prepared is None and not (args.out/"chemical_som_dataset.npz").exists():
        prepare(old/"matrix_inputs.npz",args.out)
    train(old/"matrix_inputs.npz",old/"model.pt",previous,args.out,args.epochs,args.prepared,args.tune_balance)
    print(f"Finished in {time.perf_counter()-start:.1f} seconds",flush=True)
