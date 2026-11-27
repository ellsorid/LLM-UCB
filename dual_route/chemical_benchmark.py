"""Dual-layer external chemical benchmark (classic LOHCs + broadened hydrogen-storage motifs & functional groups)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from rdkit import Chem

ARCHIVE_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = ARCHIVE_ROOT / "workflow_runtime_20261125" / "runtime"
for _p in (
    ARCHIVE_ROOT,
    ARCHIVE_ROOT / "SOM" / "chemical_unit_som_v1",
    ARCHIVE_ROOT / "LLM" / "overall_local_v1",
    ARCHIVE_ROOT / "LLM" / "ucb_chemical_units_v1",
    ARCHIVE_ROOT / "LLM" / "frontier_loop",
    RUNTIME_ROOT,
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from chemical_som import encode_pairs  # noqa: E402
from chemical_units import KINDS, extract  # noqa: E402
from mapping import KalmanMap, som_manifold_features  # noqa: E402
from matrix_pipeline.core import Molecule, from_mol, identity, properties, synthesis_score, to_mol  # noqa: E402
from memory_store import MemoryStore  # noqa: E402
import torch  # noqa: E402


BENCHMARK_MOLECULES: dict[str, dict[str, str]] = {
    # Layer A: Classic experimentally characterized LOHC systems
    "nec": {
        "name": "N-Ethylcarbazole (NEC)",
        "layer": "classic_lohc",
        "lean_smiles": "CCn1c2ccccc2c2ccccc12",
        "rich_smiles": "CCN1C2CCCCC2C2CCCCC12",
        "chemical_role": "Benchmark heteroaromatic LOHC with lowered dehydrogenation enthalpy (~50-53 kJ/mol-H2)",
    },
    "biphenyl": {
        "name": "Biphenyl",
        "layer": "classic_lohc",
        "lean_smiles": "c1ccc(-c2ccccc2)cc1",
        "rich_smiles": "C1CCC(C2CCCCC2)CC1",
        "chemical_role": "All-carbocyclic aromatic reference requiring higher dehydrogenation enthalpy (~65 kJ/mol-H2)",
    },
    "2_methylindole": {
        "name": "2-Methylindole",
        "layer": "classic_lohc",
        "lean_smiles": "Cc1cc2ccccc2[nH]1",
        "rich_smiles": "CC1CC2CCCCC2N1",
        "chemical_role": "C2-methylated indole with lower melting point and suppressed oligomerization",
    },
    "indole": {
        "name": "Indole",
        "layer": "classic_lohc",
        "lean_smiles": "c1ccc2[nH]ccc2c1",
        "rich_smiles": "C1CCC2NCCC2C1",
        "chemical_role": "Unsubstituted indole with high melting point and higher side-reaction tendency",
    },
    "dbt_24": {
        "name": "2,4-Dibenzyltoluene (DBT isomer)",
        "layer": "classic_lohc",
        "lean_smiles": "Cc1ccc(Cc2ccccc2)cc1Cc1ccccc1",
        "rich_smiles": "CC1CCC(CC2CCCCC2)CC1CC1CCCCC1",
        "chemical_role": "Industrial non-volatile thermal-stable LOHC benchmark",
    },
    "toluene": {
        "name": "Toluene",
        "layer": "classic_lohc",
        "lean_smiles": "Cc1ccccc1",
        "rich_smiles": "CC1CCCCC1",
        "chemical_role": "Light volatile monoaromatic LOHC (MCH/toluene) with high dehydrogenation enthalpy and boil-off loss",
    },
    "quinaldine": {
        "name": "2-Methylquinoline (Quinaldine)",
        "layer": "classic_lohc",
        "lean_smiles": "Cc1ccc2ccccc2n1",
        "rich_smiles": "CC1CCC2CCCCC2N1",
        "chemical_role": "Sterically shielded bicyclic N-heteroaromatic LOHC with mitigated N coordination poisoning",
    },
    "pyridine": {
        "name": "Pyridine",
        "layer": "classic_lohc",
        "lean_smiles": "c1ccncc1",
        "rich_smiles": "C1CCNCC1",
        "chemical_role": "Unshielded basic sp2-N heterocycle causing strong catalyst poisoning/adsorption inhibition",
    },
    # Layer B: Broadened hydrogen-storage motifs & functional-group contrast molecules
    "1_2_dimethylindole": {
        "name": "1,2-Dimethylindole",
        "layer": "motif_and_functional_group",
        "lean_smiles": "Cc1cc2ccccc2n1C",
        "rich_smiles": "CC1CC2CCCCC2N1C",
        "chemical_role": "Short N-methyl + C2 steric block preserves reversible cycling and resists C-N dealkylation",
    },
    "n_butylindole": {
        "name": "N-Butylindole",
        "layer": "motif_and_functional_group",
        "lean_smiles": "CCCCn1ccc2ccccc21",
        "rich_smiles": "CCCCN1CCC2CCCCC21",
        "chemical_role": "Long N-alkyl chain prone to thermal/catalytic C-N bond cleavage and dealkylation",
    },
    "quinoline": {
        "name": "Quinoline",
        "layer": "motif_and_functional_group",
        "lean_smiles": "c1ccc2ncccc2c1",
        "rich_smiles": "C1CCC2NCCCC2C1",
        "chemical_role": "Ring-N incorporation lowers thermodynamic dehydrogenation barrier vs pure carbocycle",
    },
    "naphthalene": {
        "name": "Naphthalene",
        "layer": "motif_and_functional_group",
        "lean_smiles": "c1ccc2ccccc2c1",
        "rich_smiles": "C1CCC2CCCCC2C1",
        "chemical_role": "Pure bicyclic carbocycle baseline requiring higher dehydrogenation temperature",
    },
    "2_methylbenzimidazole": {
        "name": "2-Methylbenzimidazole",
        "layer": "motif_and_functional_group",
        "lean_smiles": "Cc1nc2ccccc2[nH]1",
        "rich_smiles": "CC1NC2CCCCC2N1",
        "chemical_role": "Dual ring-N heteroaromatic with C2-methyl protection",
    },
    "ethylbenzene": {
        "name": "Ethylbenzene",
        "layer": "motif_and_functional_group",
        "lean_smiles": "CCc1ccccc1",
        "rich_smiles": "CCC1CCCCC1",
        "chemical_role": "Saturated alkylbenzene without reactive terminal unsaturated polymerizable sidechain",
    },
    "phenylacetylene": {
        "name": "Phenylacetylene",
        "layer": "motif_and_functional_group",
        "lean_smiles": "C#Cc1ccccc1",
        "rich_smiles": "CCC1CCCCC1",
        "chemical_role": "Terminal alkyne undergoes facile thermal polymerization/oligomerization and catalyst coking",
    },
    "styrene": {
        "name": "Styrene",
        "layer": "motif_and_functional_group",
        "lean_smiles": "C=Cc1ccccc1",
        "rich_smiles": "CCC1CCCCC1",
        "chemical_role": "Conjugated vinyl substituent undergoes thermally induced polymerization and oligomer fouling",
    },
    "propyl_amino_pyrazole": {
        "name": "1-Propyl-5-amino-pyrazole analogue",
        "layer": "motif_and_functional_group",
        "lean_smiles": "CCCc1cnn(C2CCCC2)c1N",
        "rich_smiles": "CCCC1CNN(C2CCCC2)C1N",
        "chemical_role": "Unknown-region pyrazole scaffold with saturated alkyl sidechain avoiding alkyne coking",
    },
    "propargyl_amino_pyrazole": {
        "name": "1-Propargyl-5-amino-pyrazole (live run root)",
        "layer": "motif_and_functional_group",
        "lean_smiles": "C#CCc1cnn(C2CCCC2)c1N",
        "rich_smiles": "CCCC1CNN(C2CCCC2)C1N",
        "chemical_role": "Terminal-alkyne pyrazole with elevated polymerization/coking risk at dehydrogenation T",
    },
    "acetophenone": {
        "name": "Acetophenone / 1-Phenylethanol",
        "layer": "motif_and_functional_group",
        "lean_smiles": "CC(=O)c1ccccc1",
        "rich_smiles": "CC(O)c1ccccc1",
        "chemical_role": "Reversible ketone <-> secondary-alcohol hydrogen-storage motif with low enthalpy",
    },
    "furan": {
        "name": "Furan",
        "layer": "motif_and_functional_group",
        "lean_smiles": "c1ccoc1",
        "rich_smiles": "C1CCOC1",
        "chemical_role": "Oxygen heteroaromatic vulnerable to irreversible C-O ring-opening hydrogenolysis",
    },
    "benzophenone": {
        "name": "Benzophenone / Diphenylmethanol",
        "layer": "motif_and_functional_group",
        "lean_smiles": "O=C(c1ccccc1)c1ccccc1",
        "rich_smiles": "OC(c1ccccc1)c1ccccc1",
        "chemical_role": "Resonance-stabilized diaryl ketone/alcohol storage motif",
    },
    "benzyl_methyl_ether": {
        "name": "Benzyl methyl ether",
        "layer": "motif_and_functional_group",
        "lean_smiles": "COCc1ccccc1",
        "rich_smiles": "COCC1CCCCC1",
        "chemical_role": "Benzylic C-O ether bond highly susceptible to catalytic hydrogenolysis cleavage",
    },
    "benzonitrile": {
        "name": "Benzonitrile / Benzylamine",
        "layer": "motif_and_functional_group",
        "lean_smiles": "N#Cc1ccccc1",
        "rich_smiles": "NCc1ccccc1",
        "chemical_role": "Reversible nitrile <-> primary amine hydrogen-transfer motif",
    },
    "2_methylfuran": {
        "name": "2-Methylfuran",
        "layer": "motif_and_functional_group",
        "lean_smiles": "Cc1ccco1",
        "rich_smiles": "CC1CCCO1",
        "chemical_role": "Furan ring prone to C-O ring opening and hydrodeoxygenation under cycling",
    },
    "phenazine": {
        "name": "Phenazine",
        "layer": "motif_and_functional_group",
        "lean_smiles": "c1ccc2nc3ccccc3nc2c1",
        "rich_smiles": "C1CCC2NC3CCCCC3NC2C1",
        "chemical_role": "Diazine-bridged tricyclic N-heterocycle with reversible hydrogenation without C-O scission",
    },
    "dibenzofuran": {
        "name": "Dibenzofuran",
        "layer": "motif_and_functional_group",
        "lean_smiles": "c1ccc2oc3ccccc3c2c1",
        "rich_smiles": "C1CCC2OC3CCCCC3C2C1",
        "chemical_role": "O-bridged tricycle susceptible to C-O hydrogenolysis and high-temperature ring opening",
    },
    "2_6_lutidine": {
        "name": "2,6-Dimethylpyridine (Lutidine)",
        "layer": "motif_and_functional_group",
        "lean_smiles": "Cc1cccc(C)n1",
        "rich_smiles": "CC1CCCC(C)N1",
        "chemical_role": "Ortho-dimethyl steric shielding weakens destructive N lone-pair catalyst poisoning",
    },
    "4_methylacetophenone": {
        "name": "4-Methylacetophenone",
        "layer": "motif_and_functional_group",
        "lean_smiles": "CC(=O)c1ccc(C)cc1",
        "rich_smiles": "CC(O)c1ccc(C)cc1",
        "chemical_role": "Clean ketone/alcohol pair free of reactive nucleophilic amine condensation partner",
    },
    "4_aminoacetophenone": {
        "name": "4-Aminoacetophenone",
        "layer": "motif_and_functional_group",
        "lean_smiles": "CC(=O)c1ccc(N)cc1",
        "rich_smiles": "CC(O)c1ccc(N)cc1",
        "chemical_role": "Antagonistic amine + ketone coexistence triggers intermolecular Schiff-base oligomerization",
    },
}


BENCHMARK_CONTRAST_PAIRS: list[dict[str, str]] = [
    # Layer A: Classic LOHC benchmarks
    {
        "id": "A1_nec_vs_biphenyl",
        "layer": "classic_lohc",
        "mechanism": "ring_heteroatom_barrier_lowering",
        "preferred": "nec",
        "disfavored": "biphenyl",
        "rationale": "Ring-N in NEC lowers dehydrogenation enthalpy vs all-carbon biphenyl",
    },
    {
        "id": "A2_2me_indole_vs_indole",
        "layer": "classic_lohc",
        "mechanism": "c2_steric_stabilization",
        "preferred": "2_methylindole",
        "disfavored": "indole",
        "rationale": "C2-methyl substituent reduces melting point and suppresses indole oligomerization",
    },
    {
        "id": "A3_dbt_vs_toluene",
        "layer": "classic_lohc",
        "mechanism": "multiring_thermal_stability",
        "preferred": "dbt_24",
        "disfavored": "toluene",
        "rationale": "DBT offers superior thermal stability and low vapor pressure vs volatile toluene",
    },
    {
        "id": "A4_quinaldine_vs_pyridine",
        "layer": "classic_lohc",
        "mechanism": "ortho_shielded_bicyclic_vs_poisoning_monocycle",
        "preferred": "quinaldine",
        "disfavored": "pyridine",
        "rationale": "C2-shielded bicyclic quinaldine avoids severe catalyst N-poisoning of free pyridine",
    },
    # Layer B: Broadened hydrogen-storage motifs & functional groups
    {
        "id": "B1_12dmi_vs_nbutylindole",
        "layer": "motif_and_functional_group",
        "mechanism": "heteroatom_vs_cn_cleavage",
        "preferred": "1_2_dimethylindole",
        "disfavored": "n_butylindole",
        "rationale": "Protected N-methyl avoids facile C-N dealkylation of long N-butyl chain",
    },
    {
        "id": "B2_quinoline_vs_naphthalene",
        "layer": "motif_and_functional_group",
        "mechanism": "heteroatom_vs_cn_cleavage",
        "preferred": "quinoline",
        "disfavored": "naphthalene",
        "rationale": "Endocyclic N facilitates reversible catalytic dehydrogenation vs pure naphthalene",
    },
    {
        "id": "B3_2mebenzimidazole_vs_naphthalene",
        "layer": "motif_and_functional_group",
        "mechanism": "heteroatom_vs_cn_cleavage",
        "preferred": "2_methylbenzimidazole",
        "disfavored": "naphthalene",
        "rationale": "Protected diazole lowers thermodynamic dehydrogenation barrier vs carbocycle",
    },
    {
        "id": "B4_ethylbenzene_vs_phenylacetylene",
        "layer": "motif_and_functional_group",
        "mechanism": "unsaturated_polymerization_trap",
        "preferred": "ethylbenzene",
        "disfavored": "phenylacetylene",
        "rationale": "Terminal alkyne induces severe thermal polymerization and coking",
    },
    {
        "id": "B5_ethylbenzene_vs_styrene",
        "layer": "motif_and_functional_group",
        "mechanism": "unsaturated_polymerization_trap",
        "preferred": "ethylbenzene",
        "disfavored": "styrene",
        "rationale": "Unprotected conjugated vinyl substituent polymerizes under thermal storage/cycling",
    },
    {
        "id": "B6_propyl_vs_propargyl_pyrazole",
        "layer": "motif_and_functional_group",
        "mechanism": "unsaturated_polymerization_trap",
        "preferred": "propyl_amino_pyrazole",
        "disfavored": "propargyl_amino_pyrazole",
        "rationale": "Unknown-region scaffold comparison: saturated alkyl avoids terminal-alkyne coking trap",
    },
    {
        "id": "B7_acetophenone_vs_furan",
        "layer": "motif_and_functional_group",
        "mechanism": "reversible_fg_vs_co_cleavage",
        "preferred": "acetophenone",
        "disfavored": "furan",
        "rationale": "Ketone/alcohol cycling is cleanly reversible whereas furan suffers ring-opening hydrogenolysis",
    },
    {
        "id": "B8_benzophenone_vs_benzylether",
        "layer": "motif_and_functional_group",
        "mechanism": "reversible_fg_vs_co_cleavage",
        "preferred": "benzophenone",
        "disfavored": "benzyl_methyl_ether",
        "rationale": "Diaryl ketone/alcohol resists facile benzylic C-O hydrogenolysis cleavage",
    },
    {
        "id": "B9_benzonitrile_vs_2mefuran",
        "layer": "motif_and_functional_group",
        "mechanism": "reversible_fg_vs_co_cleavage",
        "preferred": "benzonitrile",
        "disfavored": "2_methylfuran",
        "rationale": "Reversible nitrile/amine transfer outperforms ring-opening-prone methylfuran",
    },
    {
        "id": "B10_phenazine_vs_dibenzofuran",
        "layer": "motif_and_functional_group",
        "mechanism": "reversible_fg_vs_co_cleavage",
        "preferred": "phenazine",
        "disfavored": "dibenzofuran",
        "rationale": "N-bridged phenazine cycles reversibly vs O-bridged dibenzofuran hydrodeoxygenation risk",
    },
    {
        "id": "B11_lutidine_vs_pyridine",
        "layer": "motif_and_functional_group",
        "mechanism": "multi_unit_steric_electronic_coupling",
        "preferred": "2_6_lutidine",
        "disfavored": "pyridine",
        "rationale": "Ortho-methyl steric shielding mitigates catalyst lone-pair poisoning on pyridine N",
    },
    {
        "id": "B12_methyl_vs_amino_acetophenone",
        "layer": "motif_and_functional_group",
        "mechanism": "multi_unit_steric_electronic_coupling",
        "preferred": "4_methylacetophenone",
        "disfavored": "4_aminoacetophenone",
        "rationale": "Coexistence of nucleophilic -NH2 and electrophilic C=O triggers Schiff-base condensation",
    },
]


def _skeleton_copy(mol: Chem.Mol) -> Chem.Mol:
    """Create an element-only single-bond graph copy for lean<->rich bijection matching."""
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(False)
    for bond in rw.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return rw.GetMol()


def build_aligned_pair_from_smiles(lean_smiles: str, rich_smiles: str, name: str = "") -> Molecule:
    """Build an atom-aligned [2, n, n, 12] paired Molecule from lean & rich SMILES."""
    lean_mol = Chem.MolFromSmiles(lean_smiles)
    rich_mol = Chem.MolFromSmiles(rich_smiles)
    if lean_mol is None or rich_mol is None:
        raise ValueError(f"Invalid SMILES for {name}: lean={lean_smiles}, rich={rich_smiles}")
    if lean_mol.GetNumAtoms() != rich_mol.GetNumAtoms():
        raise ValueError(f"Heavy-atom count mismatch for {name}: {lean_mol.GetNumAtoms()} vs {rich_mol.GetNumAtoms()}")

    lean_skel = _skeleton_copy(lean_mol)
    rich_skel = _skeleton_copy(rich_mol)
    match = rich_skel.GetSubstructMatch(lean_skel)
    if len(match) != lean_mol.GetNumAtoms():
        raise ValueError(f"Failed skeleton bijection between lean and rich for {name}")

    rich_aligned = Chem.RenumberAtoms(rich_mol, list(match))
    lean_mat = from_mol(lean_mol)
    rich_mat = from_mol(rich_aligned)
    pair = np.stack([lean_mat, rich_mat])

    if not np.array_equal(np.diag(pair[0, :, :, 0]), np.diag(pair[1, :, :, 0])):
        raise ValueError(f"Element alignment check failed for {name}")
    if not np.array_equal(pair[0, :, :, 7] != 0, pair[1, :, :, 7] != 0):
        raise ValueError(f"Bond topology differs between lean and rich for {name}")

    nh2, _ = properties(pair)
    if nh2 <= 0:
        raise ValueError(f"Expected positive H2 release stoichiometry for {name}, got {nh2}")

    names = tuple(Chem.MolToSmiles(to_mol(m), isomericSmiles=True) for m in pair)
    uid = identity(names)
    return Molecule(uid=uid, pair=pair, identity=names, root=uid)


def build_benchmark_dataset() -> dict[str, Any]:
    """Compile all benchmark molecules and verify unit extraction & contrast-pair integrity."""
    molecules: dict[str, Molecule] = {}
    records: dict[str, dict[str, Any]] = {}
    for key, spec in BENCHMARK_MOLECULES.items():
        mol = build_aligned_pair_from_smiles(spec["lean_smiles"], spec["rich_smiles"], name=key)
        units = extract(mol.pair, mol.uid)
        nh2, h2_wt = properties(mol.pair)
        sa = float(synthesis_score(mol.pair))
        molecules[key] = mol
        records[key] = {
            "key": key,
            "uid": mol.uid,
            "name": spec["name"],
            "layer": spec["layer"],
            "lean_smiles": mol.identity[0],
            "rich_smiles": mol.identity[1],
            "nH2": float(nh2),
            "h2_wt_percent": round(float(h2_wt), 3),
            "sa_score": round(sa, 3),
            "chemical_role": spec["chemical_role"],
            "unit_counts": {k: len(units["units"][k]) for k in KINDS},
        }

    for pair_spec in BENCHMARK_CONTRAST_PAIRS:
        if pair_spec["preferred"] not in records or pair_spec["disfavored"] not in records:
            raise KeyError(f"Contrast pair {pair_spec['id']} references unknown molecule key")

    return {
        "version": "chemical_benchmark_v1",
        "molecule_count": len(records),
        "pair_count": len(BENCHMARK_CONTRAST_PAIRS),
        "molecules": records,
        "contrast_pairs": BENCHMARK_CONTRAST_PAIRS,
        "_mol_objects": molecules,
    }


def export_benchmark_json(out_path: Path) -> dict[str, Any]:
    """Serialize benchmark metadata (excluding numpy arrays) to JSON."""
    data = build_benchmark_dataset()
    serializable = {k: v for k, v in data.items() if k != "_mol_objects"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    return serializable


def evaluate_run_on_benchmark(
    run_dir: Path,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """Measure a completed run's SOM/Kalman/Memory coverage and contrast-pair discrimination on the benchmark."""
    mpath = model_path or (ARCHIVE_ROOT / "SOM" / "chemical_unit_som_v1" / "artifacts" / "model.pt")
    model = torch.load(mpath, map_location="cpu", weights_only=False)
    centers = model["centers"].numpy()
    bandwidth = max(float(np.median(((centers[:, None] - centers[None, :]) ** 2).sum(2))), 1e-6)

    mapping_files = sorted(run_dir.glob("G*_MAPPING.npz"), key=lambda p: int(p.stem.split("_")[0][1:]))
    if not mapping_files:
        raise FileNotFoundError(f"No G*_MAPPING.npz found in {run_dir}")
    km = KalmanMap.load(mapping_files[-1])

    traj = json.loads((run_dir / "TRAJECTORY.json").read_text(encoding="utf-8"))
    occupied_cells = {int(row["cell"]) for row in traj}

    mem_raw = json.loads((run_dir / "MEMORY_LIVE.json").read_text(encoding="utf-8"))
    store = MemoryStore(
        mem_raw.get("local_memory", mem_raw),
        mem_raw.get("global_records", []),
        protocol_hash="benchmark_eval",
    )

    rep = json.loads((run_dir / "RUN_REPORT.json").read_text(encoding="utf-8"))
    cfg = rep.get("experiment_config", {})
    sa_w = float(cfg.get("sa_penalty_weight", 0.0))
    sa_t = float(cfg.get("sa_threshold", 3.5))

    dataset = build_benchmark_dataset()
    mol_objs: dict[str, Molecule] = dataset["_mol_objects"]
    mol_evals: dict[str, dict[str, Any]] = {}
    for key, mol in mol_objs.items():
        z = encode_pairs([mol.pair], model["transform"])[0].numpy().astype(float)
        cell, _, psi = som_manifold_features(z, centers, bandwidth)
        mean_arr, var_arr = km.predict(psi)
        kalman_score = float(np.clip(mean_arr[0] * 100.0, 0.0, 100.0))
        kalman_sd = float(np.sqrt(max(float(var_arr[0]), 0.0)) * 100.0)
        units = extract(mol.pair, mol.uid)
        tags, packet = store.prepare_local(units)
        local_tags = [t for t in tags if t.get("stream", "local") == "local"]
        coupling_tags = [t for t in tags if t.get("stream") == "coupling"]
        covered_kinds = {
            t["kind"] for t in tags if t.get("used_for_judgement") and float(t.get("similarity", 0.0)) >= 0.5
        }
        sa_val = float(dataset["molecules"][key]["sa_score"])
        sa_pen = min(12.0, sa_w * max(0.0, sa_val - sa_t)) if sa_w > 0 else 0.0
        mol_evals[key] = {
            **dataset["molecules"][key],
            "som_cell": int(cell),
            "in_unknown_cell": int(cell) not in occupied_cells,
            "kalman_pred_score": round(kalman_score, 3),
            "kalman_pred_sd": round(kalman_sd, 3),
            "kalman_minus_sa_pen": round(kalman_score - sa_pen, 3),
            "matched_local_claims": len(local_tags),
            "matched_coupling_claims": len(coupling_tags),
            "matched_packet_size": len(packet),
            "evidence_gap": round(1.0 - len(covered_kinds) / len(KINDS), 3),
        }

    pair_results = []
    for p in BENCHMARK_CONTRAST_PAIRS:
        pref = mol_evals[p["preferred"]]
        disf = mol_evals[p["disfavored"]]
        margin_raw = round(pref["kalman_pred_score"] - disf["kalman_pred_score"], 3)
        margin_net = round(pref["kalman_minus_sa_pen"] - disf["kalman_minus_sa_pen"], 3)
        pair_results.append({
            **p,
            "preferred_score": pref["kalman_pred_score"],
            "disfavored_score": disf["kalman_pred_score"],
            "margin_kalman": margin_raw,
            "pass_kalman": margin_raw > 0,
            "margin_with_sa": margin_net,
            "pass_with_sa": margin_net > 0,
        })

    def _acc(subset: list[dict[str, Any]], field: str) -> float:
        return round(sum(1 for r in subset if r[field]) / max(len(subset), 1), 3)

    layer_a = [r for r in pair_results if r["layer"] == "classic_lohc"]
    layer_b = [r for r in pair_results if r["layer"] == "motif_and_functional_group"]

    hybrid_trajectory: list[dict[str, Any]] = []
    hybrid_final: dict[str, Any] | None = None
    rule_files = sorted(run_dir.glob("G*_CHEMICAL_RULES.json"), key=lambda p: int(p.stem.split("_")[0][1:]))
    if rule_files:
        from chemical_rule_surrogate import ChemicalRuleBank, HybridChemicalSurrogate
        from mapping import som_laplacian_prior

        prior_cov = som_laplacian_prior(model["topology"])
        # Precompute SOM psi and aligned pair for each observed trajectory molecule
        traj_cache: list[tuple[int, dict[str, Any]]] = []
        for row in traj:
            gen_i = int(row["generation"])
            mol_i = build_aligned_pair_from_smiles(row["lean_smiles"], row["rich_smiles"], name=str(row["uid"])[:10])
            z_i = encode_pairs([mol_i.pair], model["transform"])[0].numpy().astype(float)
            _, _, psi_i = som_manifold_features(z_i, centers, bandwidth)
            traj_cache.append((
                gen_i,
                {
                    "uid": row["uid"],
                    "psi": psi_i,
                    "pair": mol_i.pair,
                    "score_01": float(row["score"]) / 100.0,
                    "noise_var": 0.04 if row.get("evidence_status") == "full" else 0.18,
                },
            ))

        # Precompute benchmark SOM psi and aligned pairs
        bench_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for key, mol in mol_objs.items():
            z_b = encode_pairs([mol.pair], model["transform"])[0].numpy().astype(float)
            _, _, psi_b = som_manifold_features(z_b, centers, bandwidth)
            bench_cache[key] = (psi_b, mol.pair)

        for rf in rule_files:
            gen_g = int(rf.stem.split("_")[0][1:])
            rule_payload = json.loads(rf.read_text(encoding="utf-8"))
            bank_g = ChemicalRuleBank.from_dict(rule_payload)
            obs_g = [obs for g_obs, obs in traj_cache if g_obs <= gen_g]

            # 1) SOM-only surrogate at generation g (empty rule bank)
            som_surr_g = HybridChemicalSurrogate(prior_cov, context="som_only")
            som_surr_g.fit(obs_g, ChemicalRuleBank(), auto_falsify=False)

            # 2) Hybrid SOM + LLM discovered rules surrogate at generation g
            hyb_surr_g = HybridChemicalSurrogate(prior_cov, context="hybrid")
            hyb_surr_g.fit(obs_g, bank_g, auto_falsify=False)

            som_scores_g = {k: som_surr_g.predict(*bench_cache[k])[0] * 100.0 for k in mol_objs}
            hyb_scores_g = {k: hyb_surr_g.predict(*bench_cache[k])[0] * 100.0 for k in mol_objs}

            def _pair_metrics(scores_map: dict[str, float]) -> dict[str, float]:
                margins_all = [scores_map[p["preferred"]] - scores_map[p["disfavored"]] for p in BENCHMARK_CONTRAST_PAIRS]
                margins_a = [scores_map[p["preferred"]] - scores_map[p["disfavored"]] for p in BENCHMARK_CONTRAST_PAIRS if p["layer"] == "classic_lohc"]
                margins_b = [scores_map[p["preferred"]] - scores_map[p["disfavored"]] for p in BENCHMARK_CONTRAST_PAIRS if p["layer"] == "motif_and_functional_group"]
                return {
                    "overall_acc": round(sum(m > 0 for m in margins_all) / len(margins_all), 3),
                    "layer_a_acc": round(sum(m > 0 for m in margins_a) / len(margins_a), 3),
                    "layer_b_acc": round(sum(m > 0 for m in margins_b) / len(margins_b), 3),
                    "mean_margin_all": round(float(np.mean(margins_all)), 3),
                    "mean_margin_layer_b": round(float(np.mean(margins_b)), 3),
                }

            step_eval = {
                "generation": gen_g,
                "active_rules": len(bank_g.active_rules()),
                "supported_rules": sum(1 for r in bank_g.rules.values() if r.status in ("supported", "revised")),
                "falsified_rules": sum(1 for r in bank_g.rules.values() if r.status == "falsified"),
                "som_only": _pair_metrics(som_scores_g),
                "hybrid_rule_surrogate": _pair_metrics(hyb_scores_g),
            }
            hybrid_trajectory.append(step_eval)
        if hybrid_trajectory:
            hybrid_final = hybrid_trajectory[-1]

    return {
        "run_dir": str(run_dir),
        "occupied_cells_in_run": len(occupied_cells),
        "benchmark_molecules": len(mol_evals),
        "unknown_cell_molecules": sum(1 for v in mol_evals.values() if v["in_unknown_cell"]),
        "zero_local_match_molecules": sum(1 for v in mol_evals.values() if v["matched_local_claims"] == 0),
        "zero_coupling_match_molecules": sum(1 for v in mol_evals.values() if v["matched_coupling_claims"] == 0),
        "mean_evidence_gap": round(float(np.mean([v["evidence_gap"] for v in mol_evals.values()])), 3),
        "accuracy": {
            "overall_kalman": _acc(pair_results, "pass_kalman"),
            "overall_with_sa": _acc(pair_results, "pass_with_sa"),
            "layer_a_classic_lohc_kalman": _acc(layer_a, "pass_kalman"),
            "layer_b_motifs_fg_kalman": _acc(layer_b, "pass_kalman"),
            "layer_a_classic_lohc_with_sa": _acc(layer_a, "pass_with_sa"),
            "layer_b_motifs_fg_with_sa": _acc(layer_b, "pass_with_sa"),
        },
        "hybrid_rule_surrogate_final": hybrid_final,
        "hybrid_rule_surrogate_trajectory": hybrid_trajectory,
        "pair_results": pair_results,
        "molecule_evals": mol_evals,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-json", type=Path, default=ARCHIVE_ROOT / "benchmarks" / "chemical_benchmark_v1.json")
    parser.add_argument("--eval-run", type=Path, action="append", default=[])
    args = parser.parse_args()

    exported = export_benchmark_json(args.export_json)
    summary: dict[str, Any] = {
        "exported": str(args.export_json),
        "molecule_count": exported["molecule_count"],
        "pair_count": exported["pair_count"],
    }
    if args.eval_run:
        summary["evaluations"] = []
        for rdir in args.eval_run:
            ev = evaluate_run_on_benchmark(rdir)
            summary["evaluations"].append({k: v for k, v in ev.items() if k not in ("pair_results", "molecule_evals")})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
