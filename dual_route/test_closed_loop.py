"""Offline verification tests for the repaired SOM-Kalman-Memory-UCB closed loop."""
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

from chemical_units import KINDS, extract  # noqa: E402
from mapping import KalmanMap, som_laplacian_prior, som_manifold_features  # noqa: E402
from matrix_pipeline.compat.pair_core import dense_from_state, state_from_dense  # noqa: E402
from matrix_pipeline.core import Molecule, identity, to_mol  # noqa: E402
from memory_store import MemoryStore  # noqa: E402
from population_ucb import PopulationUCB  # noqa: E402
from property_feedback import (  # noqa: E402
    EditCandidate,
    EditEvidence,
    Evaluation,
    PopulationState,
    commit_batch,
    generation_quotas,
    select_next_parents,
)
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


class MockGateway:
    """Deterministic offline mock for OfficialTemperatureGateway."""

    def __init__(self, limit=100):
        self.limit = limit
        self.calls = 0
        self.quota_blocked = False
        self.roles = []

    def call(self, model, config, role, task, payload, max_tokens):
        self.calls += 1
        self.roles.append(role)
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_cache_hit_tokens": 0,
        }
        if role.endswith("_design"):
            first = payload["allowed_edit_paths"][0]
            answer = {"edit_ids": [first["id"]], "reason": "Test directed step"}
        elif "observations" in payload:
            answer = {
                "judgements": [
                    {
                        "id": obs["id"],
                        "direction": "favorable",
                        "reason": "Consistent unit stabilization",
                    }
                    for obs in payload["observations"]
                ]
            }
        else:
            raise ValueError(f"Unexpected payload for role {role}")
        return answer, {"status": "ok", "http_status": 200, "usage": usage}


class ClosedLoopTests(unittest.TestCase):
    def test_som_laplacian_kalman_propagation(self):
        model_path = ARCHIVE_ROOT / "SOM" / "chemical_unit_som_v1" / "artifacts" / "model.pt"
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        topo = model["topology"].numpy()
        k = len(topo)
        prior_cov = som_laplacian_prior(topo)
        self.assertEqual(prior_cov.shape, (k + 1, k + 1))
        np.testing.assert_allclose(prior_cov, prior_cov.T, atol=1e-10)
        np.linalg.cholesky(prior_cov)

        km = KalmanMap(k + 1, "som_test", prior_cov=prior_cov)
        means0, sds0 = km.cell_surface()
        self.assertEqual(means0.shape, (k,))
        self.assertEqual(sds0.shape, (k,))

        c0 = 0
        near_cells = np.flatnonzero(topo[c0] == 1)
        far_cells = np.flatnonzero(topo[c0] >= 4)
        self.assertGreater(len(near_cells), 0)
        self.assertGreater(len(far_cells), 0)

        onehot = np.zeros(k, dtype=float)
        onehot[c0] = 1.0
        km.update("obs_0", np.r_[1.0, onehot], 0.95, 0.02, "som_test")
        means1, sds1 = km.cell_surface()

        near_shift = np.mean(np.abs(means1[near_cells] - means0[near_cells]))
        far_shift = np.mean(np.abs(means1[far_cells] - means0[far_cells]))
        near_shrink = np.mean(sds0[near_cells] - sds1[near_cells])
        far_shrink = np.mean(sds0[far_cells] - sds1[far_cells])
        self.assertGreater(near_shift, far_shift)
        self.assertGreater(near_shrink, far_shrink)

        draw = km.sample_weights(np.random.default_rng(7))
        self.assertEqual(draw.shape, (k + 1,))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "km.npz"
            km.save(path)
            loaded = KalmanMap.load(path)
            m_loaded, s_loaded = loaded.cell_surface()
            np.testing.assert_allclose(m_loaded, means1, atol=1e-10)
            np.testing.assert_allclose(s_loaded, sds1, atol=1e-10)

        centers = model["centers"].numpy()
        cell, simplex, psi = som_manifold_features(centers[5], centers, 1.0)
        self.assertEqual(cell, 5)
        self.assertAlmostEqual(float(simplex.sum()), 1.0, places=6)
        self.assertEqual(psi.shape, (k + 1,))

    def test_memory_store_growth_and_cache(self):
        seeds = _load_bundled_seed_molecules()
        m0 = seeds[0]
        units0 = extract(m0.pair, m0.uid)
        kind, unit = next((k, grp[0]) for k, grp in units0["units"].items() if grp)

        store = MemoryStore({"unit_index": {k: [] for k in KINDS}}, [], protocol_hash="proto_v1")
        added = store.add_local_claim(
            kind,
            unit,
            {
                "source_id": "OPENALEX_2026_CUSTOM",
                "reference": "doi:10.1000/test-1",
                "endpoint": "storage_stability",
                "state": unit["state"],
                "observation": "Substituted ring shows clean reversible dehydrogenation.",
                "catalyst": "Pt/Al2O3",
            },
        )
        self.assertTrue(added)
        self.assertFalse(
            store.add_local_claim(
                kind,
                unit,
                {
                    "source_id": "OPENALEX_2026_CUSTOM",
                    "reference": "doi:10.1000/test-1",
                    "endpoint": "storage_stability",
                    "state": unit["state"],
                    "observation": "Substituted ring shows clean reversible dehydrogenation.",
                },
            )
        )

        # Verify non-whitelisted source is retrieved via prepare_local
        tags, packet = store.prepare_local(extract(m0.pair, "other_mol"))
        self.assertEqual(len(packet), 1)
        self.assertEqual(tags[0]["source_id"], "OPENALEX_2026_CUSTOM")

        cached, uncached = store.lookup_cached(packet, tags)
        self.assertEqual(len(cached), 0)
        self.assertEqual(len(uncached), 1)
        store.store_cached(
            uncached,
            tags,
            [{"id": uncached[0]["id"], "direction": "favorable", "reason": "Reversible"}],
        )
        cached2, uncached2 = store.lookup_cached(packet, tags)
        self.assertEqual(len(cached2), 1)
        self.assertEqual(len(uncached2), 0)
        self.assertEqual(store.cache_hits, 1)

        self.assertTrue(
            store.add_global_record(m0, 82.5, 80.0, 85.0, "ctx_v1", {"doi:10.1000/test-1"}, 1)
        )
        prior = store.whole_prior(m0.pair, "novel_query_uid")
        self.assertIsNotNone(prior["score"])
        self.assertAlmostEqual(prior["score"], 82.5, places=5)

    def test_closed_operator_ucb_and_soft_gate(self):
        km = KalmanMap(5, "ctx")
        p_psi = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        c_psi = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
        p_cand = EditCandidate("p1", "p1", 0, 0, p_psi, np.array([1.0, 0.0, 0.0, 0.0]), ("f", "k", "bootstrap"), frozenset({"u1"}), 0.2, frozenset({"ref1"}))
        c_cand = EditCandidate("c1", "p1", 1, 1, c_psi, np.array([0.0, 1.0, 0.0, 0.0]), ("f", "k", "vinyl_ethyl"), frozenset({"u2"}), 0.3, frozenset({"ref1"}))
        km.update("p1", p_psi, 0.6, 0.05, "ctx")
        state = PopulationState(
            km,
            {"p1": p_cand},
            {"p1": Evaluation("p1", "ctx", 60.0, 0.8, frozenset({"ref1"}), True, "full")},
        )
        ev_c = Evaluation("c1", "ctx", 78.0, 0.8, frozenset({"ref1"}), True, "full")
        link = EditEvidence("p1", "c1", 1, "ctx", frozenset({"ref1"}), True)
        updated, gate = commit_batch(state, [c_cand], {"c1": ev_c}, {"c1": link}, strict_local_gate=False)
        self.assertTrue(gate["committed"])
        self.assertEqual(len(gate["operator_updates"]), 1)
        self.assertGreater(updated.posts[c_cand.arm]["mean"], 0.0)

        # Novel candidate with no local evidence: accepted under soft gate, rejected under strict gate
        c_novel = EditCandidate("c2", "p1", 2, 2, np.array([1.0, 0.0, 0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0]), ("f", "k2", "methyl"), frozenset({"u3"}), 1.0, frozenset())
        ev_novel = Evaluation("c2", "ctx", 62.0, 0.2, frozenset(), True, "overall_only_novel")
        _, strict_gate = commit_batch(updated, [c_novel], {"c2": ev_novel}, {}, strict_local_gate=True)
        self.assertFalse(strict_gate["committed"])
        updated2, soft_gate = commit_batch(updated, [c_novel], {"c2": ev_novel}, {}, strict_local_gate=False)
        self.assertTrue(soft_gate["committed"])
        self.assertEqual(soft_gate["open_hypotheses"], ["c2"])
        self.assertIn(c_novel.uid, updated2.evaluations)

        # Verify generation_quotas can emit a directed route now that operator post mean > 0
        c_next = EditCandidate("c3", "c1", 1, 1, c_psi, np.array([0.0, 1.0, 0.0, 0.0]), ("f", "k", "vinyl_ethyl"), frozenset({"u4"}), 0.2)
        plan = generation_quotas(updated, [c_next], budget=2, directed_budget=1)
        self.assertEqual(len(plan["routes"]), 1)

    def test_population_ucb_fine_policy_and_elitism(self):
        ucb = PopulationUCB()
        ucb.observe(
            "G1",
            [{"whole": 5.0, "local": 2.0, "evidence_weight_change": 0.0}],
            novel_cell_fraction=0.5,
            novel_unit_fraction=0.2,
            policy="whole_crossover",
            legal_yield=0.8,
        )
        self.assertEqual(ucb.fine["whole_crossover"].count, 1)
        policy, audit = ucb.choose({"whole": 0.7, "local": 0.3})
        self.assertIn(policy.name, ucb.fine)
        self.assertIn("coarse_ucb", audit)

        km = KalmanMap(4, "ctx")
        mols = {}
        members = {}
        evals = {}
        for i in range(10):
            uid = f"m{i}"
            cell = i % 3
            onehot = np.zeros(3)
            onehot[cell] = 1.0
            psi = np.r_[1.0, onehot]
            km.update(uid, psi, (90 - i * 5) / 100.0, 0.05, "ctx")
            members[uid] = EditCandidate(uid, "m0", 0, cell, psi, onehot, ("f", "k", "op"), frozenset(), 0.1 * (i % 5))
            evals[uid] = Evaluation(uid, "ctx", float(90 - i * 5), 0.8, frozenset({"r"}), True, "full")
            mols[uid] = uid
        selected, p_audit = select_next_parents(PopulationState(km, members, evals), mols, max_parents=5, elite_quota=3)
        self.assertEqual(len(selected), 5)
        self.assertIn("m0", selected[:3])
        self.assertEqual({row["role"] for row in p_audit}, {"elite", "frontier"})

        selected_prot, p_audit_prot = select_next_parents(
            PopulationState(km, members, evals), mols, max_parents=5, elite_quota=3, protected_uids=["m9"]
        )
        self.assertEqual(selected_prot[0], "m9")
        self.assertEqual(p_audit_prot[0]["role"], "directed_heuristic")
        self.assertEqual(len(selected_prot), 5)

    def test_end_to_end_three_generations_offline(self):
        seeds = _load_bundled_seed_molecules()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inputs = tmp_path / "inputs"
            (inputs / "LLM" / "overall_local_v1" / "artifacts").mkdir(parents=True)
            (inputs / "runs" / "frontier_small_001" / "memory_store" / "memory").mkdir(parents=True)
            (inputs / "workflow_runtime_20261125" / "runtime" / "outputs" / "matrix_map_v1").mkdir(parents=True)

            # Build local memory fixture indexed from 2 bundled seeds using custom non-whitelisted DOIs
            unit_index = {k: [] for k in KINDS}
            for idx in range(2):
                m = seeds[idx]
                ext = extract(m.pair, m.uid)
                for kind, units in ext["units"].items():
                    for u in units[:2]:
                        unit_index[kind].append({
                            "source_molecule": m.uid,
                            "reference": f"custom/seed_{idx}/{kind}/{u['unit_id']}",
                            "source_id": f"OPENALEX_SEED_{idx}",
                            "stream": "local",
                            "evidence": {
                                "claim_id": f"c_{idx}_{u['unit_id']}",
                                "source_id": f"OPENALEX_SEED_{idx}",
                                "endpoint": "cycle_reversibility",
                                "target_state": u["state"],
                                "conditions": "450 K, Pt/Al2O3",
                                "catalyst": "Pt/Al2O3",
                                "quote": "Reversible hydrogenation observed without ring cleavage.",
                                "observation": "Reversible hydrogenation observed without ring cleavage.",
                                "attribution": "molecular_structure",
                            },
                            "source_unit": u,
                        })
            (inputs / "LLM" / "overall_local_v1" / "artifacts" / "MEMORY_V2.json").write_text(
                json.dumps({"unit_index": unit_index, "whole_evidence": [], "unresolved_local_evidence": [], "source_molecules": [s.uid for s in seeds[:2]]}),
                encoding="utf-8",
            )

            # Build global memory fixture from bundled seeds
            global_records = [
                {
                    "case": s.uid,
                    "input": {
                        "structure": {
                            "whole_states": {
                                "lean": state_from_dense(s.pair[0]),
                                "rich": state_from_dense(s.pair[1]),
                            }
                        }
                    },
                    "evaluation": {
                        "source": "LIVE_LLM",
                        "score_context": "test",
                        "critic": {"accept": True},
                        "score": {"molecule_id": s.uid, "overall": {"score": 70.0 + idx * 3.0}},
                    },
                }
                for idx, s in enumerate(seeds)
            ]
            (inputs / "runs" / "frontier_small_001" / "memory_store" / "memory" / "global.json").write_text(
                json.dumps({"records": global_records}), encoding="utf-8"
            )

            # Build immigrant pool CSV
            with (inputs / "workflow_runtime_20261125" / "runtime" / "outputs" / "matrix_map_v1" / "data_split.csv").open(
                "w", encoding="utf-8", newline=""
            ) as f:
                writer = csv.DictWriter(f, fieldnames=["lean_smiles"])
                writer.writeheader()
                for s in seeds:
                    writer.writerow({"lean_smiles": s.identity[0]})

            out_dir = tmp_path / "run_out"
            gateway = MockGateway(limit=100)
            report = runner.run(
                ARCHIVE_ROOT,
                out_dir,
                key=None,
                inputs=inputs,
                max_generations=3,
                strict_local_gate=False,
                gateway_override=gateway,
                initial_pool_override=seeds,
            )

            self.assertEqual(report["actual_generations"], 3)
            self.assertEqual(report["status"], "3_generations_committed")
            self.assertGreater(report["final_memory"]["added_global_records"], 0)
            self.assertGreater(len(report["ucb_operator_posts"]), 0)
            self.assertTrue((out_dir / "G3_SOM_SURFACE.json").is_file())
            self.assertTrue((out_dir / "MEMORY_LIVE.json").is_file())
            self.assertTrue((out_dir / "TRAJECTORY.json").is_file())

            surface = json.loads((out_dir / "G3_SOM_SURFACE.json").read_text(encoding="utf-8"))
            self.assertEqual(len(surface["cells"]), 128)
            traj = json.loads((out_dir / "TRAJECTORY.json").read_text(encoding="utf-8"))
            self.assertGreater(len(traj), len(seeds))
            proposed_ops = {r["arm"][-1] for rnd in report["rounds"] for r in rnd["selected"]}
            design_inputs = [
                json.loads((out_dir / f"G{g}_DESIGN_INPUT.json").read_text(encoding="utf-8"))
                for g in range(1, 4)
            ]
            menu_ops = {
                item["operator"]
                for pkt in design_inputs
                for item in pkt["allowed_edit_paths"]
            } | {
                cont["operator"]
                for pkt in design_inputs
                for item in pkt["allowed_edit_paths"]
                for cont in item["continuations"]
            }
            self.assertTrue(bool(menu_ops & {"vinyl_ethyl", "ethynyl_ethyl"}) or bool(proposed_ops))
            self.assertIn("root_molecule", design_inputs[0])
            self.assertIn("lean_smiles", design_inputs[0]["root_molecule"])
            self.assertIn("sa_score", design_inputs[0]["root_molecule"])
            first_cand = design_inputs[0]["allowed_edit_paths"][0]
            self.assertIn("lean_smiles", first_cand)
            self.assertIn("sa_score", first_cand)
            self.assertIn("sa_penalty_points", first_cand)
            self.assertIn("sa_penalty", traj[-1])
            self.assertIn("directed_lineage", report)
            self.assertEqual(report["directed_lineage"]["directed_in_parents_generations"], 3)


if __name__ == "__main__":
    unittest.main()
