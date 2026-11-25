"""Use frozen old GA operators and validate their products as paired matrices."""
import os
from pathlib import Path
import random
import sys
import types

import numpy as np
import pandas as pd

from population_ucb import PopulationUCB, quotas


def generate(legacy_root, archive_runtime, parents, known, budget=12, policy=None, generation=1):
    from matrix_map.prepare import align_generated
    from matrix_pipeline.core import Molecule, identity, properties, synthesis_score, to_mol
    from rdkit import Chem, RDConfig, RDLogger

    # Legacy utils imports its predictor even when only its generators are used.
    # This bridge hands off paired matrices to the new scorer.
    def new_pipeline_only(*args, **kwargs):
        raise RuntimeError("Use the new paired-matrix evaluation entrypoint")
    previous_predict = sys.modules.get("predict")
    previous_utils = sys.modules.get("utils")
    stub = types.ModuleType("predict")
    stub.predict_delta_H = stub.predict_delta_H_with_uncertainty = new_pipeline_only
    sys.modules["predict"] = stub
    sys.modules.pop("utils",None)
    legacy_root = Path(legacy_root).resolve()
    sys.path.insert(0,str(legacy_root))
    cwd = Path.cwd()
    try:
        import utils as ga
        RDLogger.DisableLog("rdApp.warning")
        ga.SA.readFragmentScores(str(Path(RDConfig.RDContribDir)/"SA_Score/fpscores"))
        os.chdir(legacy_root)
        np.random.seed(41+generation)
        random.seed(41+generation)
        smiles_parents = {m.identity[0]:m.uid for m in parents}
        frame = pd.DataFrame([{"smi":m.identity[0],"mol":to_mol(m.pair[0]),"selection_score":1.,
                               "SA score":synthesis_score(m.pair)} for m in parents])
        frame = ga.annotate_platform_families(frame)
        if policy is None:
            policy,_ = PopulationUCB().choose({"whole":1.,"local":1.})
        limits = quotas(policy,budget)
        cross = ga.cross_over(frame.copy(),len(frame),limits["crossover"],260,None,4.,
            same_platform_only=policy.same_platform,retain_parent_scaffold=policy.preserve_scaffold,max_attempts=350)
        cross = cross.loc[~cross["smi"].isin(smiles_parents)]
        mutants = ga.generate_guided_mutation_offspring(frame,limits["mutation"]*3,260,4.,
            policy.preserve_scaffold,None,generation,{})
        immigrants = pd.read_csv(archive_runtime/"outputs/matrix_map_v1/data_split.csv").head(512)
        immigrants = pd.DataFrame({"smi":immigrants["lean_smiles"]})
        immigrants["mol"] = immigrants["smi"].map(Chem.MolFromSmiles)
        immigrants = immigrants[immigrants["mol"].map(
            lambda m:m is not None and any(b.GetIsAromatic() and b.IsInRing() for b in m.GetBonds()))]
        immigrants = ga.sample_seed_immigrants(immigrants,limits["immigration"],random_state=42,
                                               blocked_smis=smiles_parents)
        valid,rejected = [],[]
        seen = set(known)
        for route,rows in (("crossover",cross),("mutation",mutants),("immigration",immigrants)):
            kept = 0
            for row in rows.drop_duplicates("smi").to_dict("records"):
                if kept>=limits[route]:
                    break
                try:
                    lean = Chem.MolFromSmiles(row["smi"])
                    rich_name = ga.aromatic_to_saturated_smiles(row["smi"])
                    rich = Chem.MolFromSmiles(rich_name) if rich_name else None
                    if lean is None or rich is None or lean.GetNumAtoms()>20:
                        raise ValueError("Cannot form supported two-state pair")
                    pair,ambiguous = align_generated(lean,rich)
                    if properties(pair)[0]<=0 or synthesis_score(pair)>4:
                        raise ValueError("Failed prospective pair filters")
                    names = tuple(Chem.MolToSmiles(to_mol(x),isomericSmiles=True) for x in pair)
                    uid = identity(names)
                    if uid in seen:
                        continue
                    parent = smiles_parents.get(row.get("parent_smi")) if route=="mutation" else None
                    valid.append((Molecule(uid,pair,names,parent or uid,parent),
                                  {"operation":route,"parent_uid":parent,
                                   "crossover_parent_pool":list(smiles_parents.values()) if route=="crossover" else [],
                                   "mapping_ambiguous":bool(ambiguous)}))
                    kept += 1
                    seen.add(uid)
                except (ValueError,RuntimeError) as error:
                    rejected.append({"operation":route,"reason":str(error)})
        return {"policy":policy.name,"requested":limits,"valid":valid,"rejected":rejected}
    finally:
        os.chdir(cwd)
        sys.path.remove(str(legacy_root))
        for name,mod in (("predict",previous_predict),("utils",previous_utils)):
            if mod is None:
                sys.modules.pop(name,None)
            else:
                sys.modules[name] = mod
