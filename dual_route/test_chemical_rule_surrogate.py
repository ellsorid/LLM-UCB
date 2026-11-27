"""Offline unit and end-to-end verification tests for MVP-2 chemical rule discovery & hybrid surrogate."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = REPO_ROOT / "dual_route"
RUNTIME_ROOT = ARCHIVE_ROOT / "workflow_runtime_20261125" / "runtime"
for p in (
    ARCHIVE_ROOT,
    ARCHIVE_ROOT / "SOM" / "chemical_unit_som_v1",
    ARCHIVE_ROOT / "LLM" / "overall_local_v1",
    ARCHIVE_ROOT / "LLM" / "ucb_chemical_units_v1",
    ARCHIVE_ROOT / "LLM" / "frontier_loop",
    RUNTIME_ROOT,
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import chemical_benchmark  # noqa: E402
from chemical_rule_surrogate import (  # noqa: E402
    ChemicalRuleBank,
    HybridChemicalSurrogate,
    validate_rule_proposal,
)
from chemical_units import KINDS, extract  # noqa: E402
from evidence_finder import _detect_motifs  # noqa: E402
from mapping import som_laplacian_prior, som_manifold_features  # noqa: E402
from matrix_pipeline.compat.pair_core import dense_from_state, state_from_dense  # noqa: E402
from matrix_pipeline.core import Molecule, identity, to_mol  # noqa: E402
from rdkit import Chem  # noqa: E402
import run_dual_ten_generation as runner  # noqa: E402


def _load_bundled_seed_molecules():
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "matrix_pipeline" / "data" / "seeds.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    molecules = []
    for i, r in enumerate(rows):
        pair = np.stack([dense_from_state(r["states"][s]) for s in ("lean", "rich")])
        names = tuple(Chem.MolToSmiles(to_mol(m), isomericSmiles=True) for m in pair)
        uid = identity(names)
        molecules.append(Molecule(uid, pair, names, uid, source_row=i))
    return molecules


class MockRuleGateway:
    """Offline mock handling design, local scoring, and incremental rule induction."""

    def __init__(self, limit=200):
        self.limit = limit
        self.calls = 0
        self.quota_blocked = False
        self.roles = []

    def call(self, model, config, role, task, payload, max_tokens):
        self.calls += 1
        self.roles.append(role)
        usage = {"prompt_tokens": 110, "completion_tokens": 35, "total_tokens": 145, "prompt_cache_hit_tokens": 0}
        if role.endswith("_design"):
            first = payload["allowed_edit_paths"][0]
            return {"edit_ids": [first["id"]], "reason": "Mock heuristic design step"}, {
                "status": "ok",
                "http_status": 200,
                "usage": usage,
            }
        if role.endswith("_rule_induction"):
            ref = payload["allowed_refs"][0]
            if payload["generation"] == 1:
                ans = {
                    "new_rules": [
                        {
                            "name": "terminal_alkyne_polymerization_risk",
                            "chemical_statement": "Terminal alkynes undergo thermal polymerization/coking during dehydrogenation.",
                            "predicate_type": "smarts_lean",
                            "smarts_lean": "[CX2;H1]#C",
                            "smarts_rich": "",
                            "partner_smarts": "",
                            "expected_direction": "unfavorable",
                            "scope_tier": 3,
                            "endpoint": "storage_stability",
                            "boundary_conditions": "Elevated dehydrogenation temperatures",
                            "supporting_dois": [ref],
                        },
                        {
                            "name": "ketone_alcohol_reversibility",
                            "chemical_statement": "Carbonyl/secondary alcohol groups provide reversible low-enthalpy hydrogen cycling.",
                            "predicate_type": "smarts_lean",
                            "smarts_lean": "[#6]C(=O)[#6]",
                            "smarts_rich": "",
                            "partner_smarts": "",
                            "expected_direction": "favorable",
                            "scope_tier": 2,
                            "endpoint": "cycle_reversibility",
                            "boundary_conditions": "Supported Pt/Ru hydrogenation catalysts",
                            "supporting_dois": [ref],
                        },
                    ],
                    "revise_rules": [],
                    "falsify_rule_ids": [],
                }
            else:
                ans = {
                    "new_rules": [
                        {
                            "name": "ortho_shielded_pyridine_stability",
                            "chemical_statement": "Ortho-substituted ring-N heterocycles suppress destructive catalyst lone-pair poisoning.",
                            "predicate_type": "coupling_smarts",
                            "smarts_lean": "c1nc(C)ccc1",
                            "smarts_rich": "",
                            "partner_smarts": "n1ccccc1",
                            "expected_direction": "coupling_synergy",
                            "scope_tier": 3,
                            "endpoint": "cycle_reversibility",
                            "boundary_conditions": "Sterically hindered N-heterocycles",
                            "supporting_dois": [ref],
                        }
                    ],
                    "revise_rules": [],
                    "falsify_rule_ids": [],
                }
            return ans, {"status": "ok", "http_status": 200, "usage": usage}
        if "observations" in payload:
            return {
                "judgements": [
                    {
                        "id": obs["id"],
                        "direction": "favorable",
                        "reason": "Consistent with reversible hydrogen storage motif",
                    }
                    for obs in payload["observations"]
                ]
            }, {"status": "ok", "http_status": 200, "usage": usage}
        raise ValueError(f"Unexpected payload for role {role}")


class ChemicalRuleSurrogateTests(unittest.TestCase):
    def test_motif_detection_prioritizes_distinguishing_functional_groups(self):
        ds = chemical_benchmark.build_benchmark_dataset()
        mols = ds["_mol_objects"]
        pa = mols["phenylacetylene"]
        bme = mols["benzyl_methyl_ether"]
        pa_motifs = _detect_motifs(pa, extract(pa.pair, pa.uid))
        bme_motifs = _detect_motifs(bme, extract(bme.pair, bme.uid))
        self.assertEqual(pa_motifs[0], "terminal alkyne")
        self.assertEqual(bme_motifs[0], "benzyl ether")

    def test_rule_validation_blocks_bad_smarts_wildcards_and_fake_dois(self):
        ok_bad_smarts, _, _ = validate_rule_proposal(
            {
                "name": "bad",
                "chemical_statement": "Invalid SMARTS pattern test",
                "predicate_type": "smarts_lean",
                "smarts_lean": "C(((",
                "expected_direction": "favorable",
                "scope_tier": 2,
                "supporting_dois": ["10.1021/valid.1"],
            },
            {"10.1021/valid.1"},
        )
        self.assertFalse(ok_bad_smarts)

        ok_trivial, _, _ = validate_rule_proposal(
            {
                "name": "trivial",
                "chemical_statement": "Single atom wildcard pattern",
                "predicate_type": "smarts_lean",
                "smarts_lean": "[#6]",
                "expected_direction": "favorable",
                "scope_tier": 2,
                "supporting_dois": ["10.1021/valid.1"],
            },
            {"10.1021/valid.1"},
        )
        self.assertFalse(ok_trivial)

        ok_fake_doi, _, _ = validate_rule_proposal(
            {
                "name": "fake_doi",
                "chemical_statement": "Terminal alkyne undergoes polymerization risk",
                "predicate_type": "smarts_lean",
                "smarts_lean": "[CX2;H1]#C",
                "expected_direction": "unfavorable",
                "scope_tier": 3,
                "supporting_dois": ["10.9999/hallucinated"],
            },
            {"10.1021/valid.1"},
        )
        self.assertFalse(ok_fake_doi)

        ok_good, rule_obj, _ = validate_rule_proposal(
            {
                "name": "amine_ketone_condensation",
                "chemical_statement": "Coexisting amine and ketone undergo condensation oligomerization",
                "predicate_type": "coupling_smarts",
                "smarts_lean": "[#6]C(=O)[#6]",
                "partner_smarts": "[NX3;H2]",
                "expected_direction": "coupling_antagonism",
                "scope_tier": 3,
                "supporting_dois": ["10.1021/valid.1"],
            },
            {"10.1021/valid.1"},
        )
        self.assertTrue(ok_good)
        self.assertIsNotNone(rule_obj)

    def test_empirical_falsification_flips_and_prunes_wrong_rule(self):
        model_path = ARCHIVE_ROOT / "SOM" / "chemical_unit_som_v1" / "artifacts" / "model.pt"
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        centers = model["centers"].numpy()
        prior_cov = som_laplacian_prior(model["topology"])

        ds = chemical_benchmark.build_benchmark_dataset()
        mols = ds["_mol_objects"]

        bank = ChemicalRuleBank()
        bank.ingest_proposals(
            {
                "new_rules": [
                    {
                        "name": "wrong_alkyne_favorable",
                        "chemical_statement": "Terminal alkyne mistakenly hypothesized as favorable.",
                        "predicate_type": "smarts_lean",
                        "smarts_lean": "[CX2;H1]#C",
                        "expected_direction": "favorable",
                        "scope_tier": 3,
                        "endpoint": "storage_stability",
                        "supporting_dois": ["10.1021/mock.test"],
                    }
                ]
            },
            generation=1,
            allowed_refs={"10.1021/mock.test"},
        )
        wrong_id = next(iter(bank.rules))

        # Construct empirical observations where alkyne molecules score 20 pts LOWER than saturated analogues
        obs = []
        alkyne_keys = ["phenylacetylene", "propargyl_amino_pyrazole"]
        sat_keys = ["ethylbenzene", "propyl_amino_pyrazole", "toluene", "biphenyl"]
        # Duplicate observations with distinct uids to simulate 6 alkyne observations vs 8 non-alkyne observations
        for rep_idx in range(3):
            for k in alkyne_keys:
                m = mols[k]
                obs.append({
                    "uid": f"{k}_{rep_idx}",
                    "psi": np.r_[1.0, np.ones(len(centers)) / len(centers)],
                    "pair": m.pair,
                    "score_01": 0.42,
                    "noise_var": 0.02,
                })
        for rep_idx in range(2):
            for k in sat_keys:
                m = mols[k]
                obs.append({
                    "uid": f"{k}_{rep_idx}",
                    "psi": np.r_[1.0, np.ones(len(centers)) / len(centers)],
                    "pair": m.pair,
                    "score_01": 0.68,
                    "noise_var": 0.02,
                })

        surr = HybridChemicalSurrogate(prior_cov, "falsify_test")
        summary = surr.fit(obs, bank, auto_falsify=True)
        self.assertIn(wrong_id, summary["newly_falsified"])
        self.assertEqual(bank.rules[wrong_id].status, "falsified")
        self.assertLess(bank.rules[wrong_id].beta_points, 0.0)
        self.assertEqual(len(bank.active_rules()), 0)

    def test_unknown_region_extrapolation_and_additive_explanation(self):
        model_path = ARCHIVE_ROOT / "SOM" / "chemical_unit_som_v1" / "artifacts" / "model.pt"
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        centers = model["centers"].numpy()
        prior_cov = som_laplacian_prior(model["topology"])

        ds = chemical_benchmark.build_benchmark_dataset()
        mols = ds["_mol_objects"]
        bank = ChemicalRuleBank()
        bank.ingest_proposals(
            {
                "new_rules": [
                    {
                        "name": "alkyne_coking_penalty",
                        "chemical_statement": "Terminal alkyne undergoes thermal coking and polymerization.",
                        "predicate_type": "smarts_lean",
                        "smarts_lean": "[CX2;H1]#C",
                        "expected_direction": "unfavorable",
                        "scope_tier": 3,
                        "endpoint": "storage_stability",
                        "supporting_dois": ["10.1021/coking.ref1", "10.1021/coking.ref2"],
                    }
                ]
            },
            generation=1,
            allowed_refs={"10.1021/coking.ref1", "10.1021/coking.ref2"},
        )

        # Fit only on flat identical uniform SOM observations (zero signal on unknown pyrazole pair)
        surr = HybridChemicalSurrogate(prior_cov, "extrap_test")
        surr.fit([], bank, auto_falsify=False)

        uniform_psi = np.r_[1.0, np.ones(len(centers)) / len(centers)]
        exp_propyl = surr.explain(uniform_psi, mols["propyl_amino_pyrazole"].pair)
        exp_propargyl = surr.explain(uniform_psi, mols["propargyl_amino_pyrazole"].pair)

        self.assertAlmostEqual(
            exp_propargyl["som_base_points"] + exp_propargyl["rule_delta_points"],
            exp_propargyl["total_points"],
            places=3,
        )
        self.assertGreater(exp_propyl["total_points"], exp_propargyl["total_points"])
        self.assertEqual(len(exp_propargyl["matched_rules"]), 1)

    def test_end_to_end_chemical_rule_induction_with_benchmark_eval(self):
        seeds = _load_bundled_seed_molecules()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inputs = tmp_path / "inputs"
            (inputs / "LLM" / "overall_local_v1" / "artifacts").mkdir(parents=True)
            (inputs / "runs" / "frontier_small_001" / "memory_store" / "memory").mkdir(parents=True)
            (inputs / "workflow_runtime_20261125" / "runtime" / "outputs" / "matrix_map_v1").mkdir(parents=True)

            unit_index = {k: [] for k in KINDS}
            for idx in range(2):
                m = seeds[idx]
                ext = extract(m.pair, m.uid)
                for kind, units in ext["units"].items():
                    for u in units[:1]:
                        unit_index[kind].append({
                            "source_molecule": m.uid,
                            "reference": f"doi:10.1021/seed.{idx}/{kind}",
                            "source_id": f"10.1021/seed.{idx}",
                            "stream": "local",
                            "evidence": {
                                "claim_id": f"c_{idx}_{u['unit_id']}",
                                "source_id": f"10.1021/seed.{idx}",
                                "endpoint": "cycle_reversibility",
                                "target_state": u["state"],
                                "conditions": "450 K",
                                "catalyst": "Pt/Al2O3",
                                "quote": "Reversible dehydrogenation proceeded cleanly without cleavage.",
                                "observation": "Reversible dehydrogenation proceeded cleanly without cleavage.",
                                "attribution": "molecular_structure",
                            },
                            "source_unit": u,
                        })
            (inputs / "LLM" / "overall_local_v1" / "artifacts" / "MEMORY_V2.json").write_text(
                json.dumps({"unit_index": unit_index, "whole_evidence": [], "unresolved_local_evidence": [], "source_molecules": [s.uid for s in seeds[:2]]}),
                encoding="utf-8",
            )
            global_records = [
                {
                    "case": s.uid,
                    "input": {"structure": {"whole_states": {"lean": state_from_dense(s.pair[0]), "rich": state_from_dense(s.pair[1])}}},
                    "evaluation": {"source": "LIVE_LLM", "score_context": "test", "critic": {"accept": True}, "score": {"molecule_id": s.uid, "overall": {"score": 72.0 + idx * 2.0}}},
                }
                for idx, s in enumerate(seeds)
            ]
            (inputs / "runs" / "frontier_small_001" / "memory_store" / "memory" / "global.json").write_text(
                json.dumps({"records": global_records}), encoding="utf-8"
            )
            with (inputs / "workflow_runtime_20261125" / "runtime" / "outputs" / "matrix_map_v1" / "data_split.csv").open(
                "w", encoding="utf-8", newline=""
            ) as f:
                writer = csv.DictWriter(f, fieldnames=["lean_smiles"])
                writer.writeheader()
                for s in seeds:
                    writer.writerow({"lean_smiles": s.identity[0]})

            out_dir = tmp_path / "run_mvp2"
            gw = MockRuleGateway(limit=200)
            report = runner.run(
                ARCHIVE_ROOT,
                out_dir,
                key=None,
                inputs=inputs,
                max_generations=2,
                strict_local_gate=False,
                gateway_override=gw,
                initial_pool_override=seeds,
                enable_chemical_rules=True,
            )

            self.assertEqual(report["actual_generations"], 2)
            self.assertTrue((out_dir / "G1_CHEMICAL_RULES.json").is_file())
            self.assertTrue((out_dir / "G2_CHEMICAL_RULES.json").is_file())
            g2_design = json.loads((out_dir / "G2_DESIGN_INPUT.json").read_text(encoding="utf-8"))
            self.assertIn("discovered_chemical_rules", g2_design)
            self.assertGreaterEqual(len(g2_design["discovered_chemical_rules"]), 2)

            bench_ev = chemical_benchmark.evaluate_run_on_benchmark(out_dir)
            self.assertIsNotNone(bench_ev["hybrid_rule_surrogate_final"])
            self.assertEqual(len(bench_ev["hybrid_rule_surrogate_trajectory"]), 2)
            self.assertGreaterEqual(bench_ev["hybrid_rule_surrogate_final"]["active_rules"], 3)


if __name__ == "__main__":
    unittest.main()
