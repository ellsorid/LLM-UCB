"""Verify model reload and placement of historical aligned matrix edits."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from chemical_som import encode_pairs
from chemical_units import load_pair


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for option in ("archive","training","replay"):
        p.add_argument(f"--{option}",type=Path,required=True)
    args = p.parse_args()
    rt = args.archive/"workflow_runtime_20261125/runtime"
    sys.path.insert(0,str(rt))
    from matrix_map.selection import distances
    from matrix_map.simple_som import encode_pairs as old_encode

    new = torch.load(args.training/"model.pt",map_location="cpu",weights_only=False)
    old = torch.load(rt/"outputs/matrix_som_v2/model.pt",map_location="cpu",weights_only=False)
    archive_input = rt/"outputs/matrix_som_v2/matrix_inputs.npz"
    with np.load(archive_input,allow_pickle=False) as raw:
        mats,masks,ids = raw["x"],raw["mask"],raw["test"][:12]
        batch = [mats[i][:,masks[i]][:,:,masks[i]] for i in ids]
    with np.load(args.training/"assignments.npz",allow_pickle=False) as match:
        assigned = match["node"][ids]
    encoded = encode_pairs(batch,new["transform"])
    reloaded = distances(encoded,new["centers"]).argmin(1).numpy()
    np.testing.assert_array_equal(reloaded,assigned)
    reversed_batch = [x[:,::-1,::-1].copy() for x in batch]
    torch.testing.assert_close(encoded,encode_pairs(reversed_batch,new["transform"]),rtol=0,atol=1e-6)

    replay = args.replay/"family_1/frontier"
    source = args.archive/"runs/frontier_small_001/evaluations/warm_start/evaluation/target_record.json"
    refs = [read(source)]
    pairs = []
    for step in read(replay/"rounds.json"):
        for i,child in enumerate(step["admissions"],1):
            rec = read(replay/"evaluations"/f"g{step['generation']}_slot{i}"/"target_record.json")
            refs.append(rec)
            pairs.append((child["parent_uid"],child["uid"],child["operator"]))
    indices = {r["record_id"]:i for i,r in enumerate(refs)}
    structures = [load_pair(r) for r in refs]
    a = old_encode(structures,old["transform"],old["weights"],device="cpu")
    b = encode_pairs(structures,new["transform"])
    a_nodes = distances(a,old["centers"]).argmin(1)
    b_nodes = distances(b,new["centers"]).argmin(1)
    report = []
    for parent,child,edit in pairs:
        i,j = indices[parent],indices[child]
        report.append({"parent":parent,"child":child,"edit":edit,
            "old_cells":[int(a_nodes[i]),int(a_nodes[j])],"new_cells":[int(b_nodes[i]),int(b_nodes[j])],
            "new_whole_distance":float(torch.linalg.vector_norm((b[i]-b[j])[:68])),
            "new_local_distance":float(torch.linalg.vector_norm((b[i]-b[j])[68:]))})
    result = {"model_reload_assignments_match":len(ids),"permutation_preserved":len(ids),
              "recorded_edit_pairs":report}
    (args.training/"MODEL_CHECKS.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result),flush=True)
