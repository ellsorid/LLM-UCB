"""Offline verification tests for the Autonomous Literature Evidence-Finding Agent."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

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
from evidence_finder import (  # noqa: E402
    AutonomousEvidenceFinder,
    LiteratureSearchClient,
    build_gap_queries,
    normalize_doi,
    quote_in_abstract,
    reconstruct_openalex_abstract,
    strip_xml_tags,
    validate_extracted_claims,
)
from matrix_pipeline.compat.pair_core import dense_from_state, state_from_dense  # noqa: E402
from matrix_pipeline.core import Molecule, identity, to_mol  # noqa: E402
from memory_store import MemoryStore  # noqa: E402
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


class MockEvidenceGateway:
    """Offline mock handling design, evidence extraction, and local judgement calls."""

    def __init__(self, limit=200):
        self.limit = limit
        self.calls = 0
        self.quota_blocked = False
        self.roles = []

    def call(self, model, config, role, task, payload, max_tokens):
        self.calls += 1
        self.roles.append(role)
        usage = {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "prompt_cache_hit_tokens": 0,
        }
        if role.endswith("_design"):
            first = payload["allowed_edit_paths"][0]
            answer = {"edit_ids": [first["id"]], "reason": "Select top evidence-backed route"}
        elif role.startswith("evidence_extract_"):
            paper = payload["papers"][0]
            claims = []
            for unit in payload["candidate_units"][:2]:
                claims.append({
                    "doi": paper["doi"],
                    "unit_id": unit["unit_id"],
                    "endpoint": "cycle_reversibility",
                    "state": unit["state"],
                    "catalyst": "Pt/Al2O3",
                    "conditions": "450 K, 1 bar",
                    "attribution": "molecular_structure",
                    "observation": "Reversible dehydrogenation-hydrogenation cycling proceeds cleanly without ring cleavage.",
                    "quote": "Reversible hydrogenation and dehydrogenation cycling occurred cleanly without ring cleavage",
                })
            answer = {"claims": claims}
        elif "observations" in payload:
            answer = {
                "judgements": [
                    {
                        "id": obs["id"],
                        "direction": "favorable",
                        "reason": "Extracted structural unit supports reversible cycling",
                    }
                    for obs in payload["observations"]
                ]
            }
        else:
            raise ValueError(f"Unexpected payload for role {role}")
        return answer, {"status": "ok", "http_status": 200, "usage": usage}


class EvidenceFinderTests(unittest.TestCase):
    def test_reconstruct_openalex_abstract_and_crossref_strip(self):
        inv = {
            "Reversible": [0],
            "dehydrogenation": [1, 5],
            "of": [2],
            "heterocyclic": [3],
            "LOHC": [4],
            "occurred": [6],
            "cleanly.": [7],
        }
        text = reconstruct_openalex_abstract(inv)
        self.assertEqual(text, "Reversible dehydrogenation of heterocyclic LOHC dehydrogenation occurred cleanly.")

        jats = "<jats:p>Experimental <jats:italic>thermal stability</jats:italic> &amp; reversible hydrogenation of furan rings.</jats:p>"
        cleaned = strip_xml_tags(jats)
        self.assertEqual(cleaned, "Experimental thermal stability & reversible hydrogenation of furan rings.")

        self.assertEqual(normalize_doi("https://doi.org/10.1021/ACS.Test.1"), "10.1021/acs.test.1")
        self.assertTrue(quote_in_abstract("reversible hydrogenation of furan rings", cleaned))
        self.assertFalse(quote_in_abstract("invented sentence not in abstract", cleaned))

    def test_build_gap_queries_from_real_molecules(self):
        seeds = _load_bundled_seed_molecules()
        m0 = seeds[0]
        units0 = extract(m0.pair, m0.uid)
        queries = build_gap_queries(m0, units0, memory_store=None, max_queries=2)
        self.assertEqual(len(queries), 2)
        self.assertEqual({q["endpoint"] for q in queries}, {"cycle_reversibility", "storage_stability"})
        self.assertTrue(all(q["target_units"] for q in queries))

        # Cover one kind in MemoryStore and verify gap queries still target remaining gaps
        store = MemoryStore({"unit_index": {k: [] for k in KINDS}}, [], protocol_hash="p")
        first_kind, first_unit = next((k, grp[0]) for k, grp in units0["units"].items() if grp)
        store.add_local_claim(
            first_kind,
            first_unit,
            {
                "source_id": "10.1000/covered",
                "reference": "doi:10.1000/covered",
                "endpoint": "cycle_reversibility",
                "state": first_unit["state"],
                "observation": "Covered unit.",
            },
        )
        queries2 = build_gap_queries(m0, units0, memory_store=store, max_queries=2)
        self.assertEqual(len(queries2), 2)
        self.assertFalse(any(uid.startswith(first_kind + "/") for uid in queries2[0]["target_units"]))

    def test_validate_extracted_claims_blocks_hallucinations_and_catalyst_interventions(self):
        seeds = _load_bundled_seed_molecules()
        m0 = seeds[0]
        units0 = extract(m0.pair, m0.uid)
        unit_lookup = {u["unit_id"]: (k, u) for k in KINDS for u in units0["units"][k]}
        sample_uid, (_, sample_unit) = next(iter(unit_lookup.items()))
        wrong_state = "rich" if sample_unit["state"] == "lean" else "lean"

        papers = [
            {
                "doi": "10.1021/real.2025.01",
                "url": "https://doi.org/10.1021/real.2025.01",
                "title": "LOHC stability",
                "year": 2025,
                "abstract": "In this study, reversible dehydrogenation of the aromatic ring occurred cleanly, whereas adding Na promoter suppressed cracking.",
            }
        ]
        response = {
            "claims": [
                {
                    "doi": "10.9999/fake.doi",
                    "unit_id": sample_uid,
                    "endpoint": "cycle_reversibility",
                    "state": sample_unit["state"],
                    "catalyst": "Pt",
                    "conditions": "450 K",
                    "attribution": "molecular_structure",
                    "observation": "Fake DOI.",
                    "quote": "reversible dehydrogenation of the aromatic ring occurred cleanly",
                },
                {
                    "doi": "10.1021/real.2025.01",
                    "unit_id": sample_uid,
                    "endpoint": "cycle_reversibility",
                    "state": sample_unit["state"],
                    "catalyst": "Pt",
                    "conditions": "450 K",
                    "attribution": "molecular_structure",
                    "observation": "Fabricated quote.",
                    "quote": "completely fabricated quote that is absent from paper",
                },
                {
                    "doi": "10.1021/real.2025.01",
                    "unit_id": sample_uid,
                    "endpoint": "cycle_reversibility",
                    "state": wrong_state,
                    "catalyst": "Pt",
                    "conditions": "450 K",
                    "attribution": "molecular_structure",
                    "observation": "Wrong state.",
                    "quote": "reversible dehydrogenation of the aromatic ring occurred cleanly",
                },
                {
                    "doi": "10.1021/real.2025.01",
                    "unit_id": sample_uid,
                    "endpoint": "storage_stability",
                    "state": sample_unit["state"],
                    "catalyst": "Na-Pt",
                    "conditions": "450 K",
                    "attribution": "external_intervention",
                    "observation": "Na promoter suppressed cracking.",
                    "quote": "adding Na promoter suppressed cracking",
                },
                {
                    "doi": "10.1021/real.2025.01",
                    "unit_id": sample_uid,
                    "endpoint": "cycle_reversibility",
                    "state": sample_unit["state"],
                    "catalyst": "Pt",
                    "conditions": "450 K",
                    "attribution": "molecular_structure",
                    "observation": "Aromatic ring undergoes clean reversible dehydrogenation.",
                    "quote": "reversible dehydrogenation of the aromatic ring occurred cleanly",
                },
            ]
        }
        accepted, contextual, rejected = validate_extracted_claims(response, papers, unit_lookup)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(contextual), 1)
        self.assertEqual(len(rejected), 3)
        self.assertEqual(accepted[0]["claim_row"]["source_id"], "10.1021/real.2025.01")

    def test_autonomous_evidence_finder_caches_queries_and_ingests_memory(self):
        seeds = _load_bundled_seed_molecules()
        m0 = seeds[0]
        units0 = extract(m0.pair, m0.uid)
        backend_calls = {"n": 0}

        def fake_backend(query, limit):
            backend_calls["n"] += 1
            return [
                {
                    "doi": "https://doi.org/10.1016/j.mock.2025.100",
                    "title": "LOHC Reversible Cycling",
                    "year": 2025,
                    "abstract": "Reversible hydrogenation and dehydrogenation cycling occurred cleanly without ring cleavage across repeated cycles.",
                }
            ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_file = tmp_path / "search_cache.json"
            client = LiteratureSearchClient(max_requests=10, cache_path=cache_file, backend=fake_backend)
            gw = MockEvidenceGateway()
            finder = AutonomousEvidenceFinder(gw, out_dir=tmp_path / "audit", search_client=client)
            store = MemoryStore({"unit_index": {k: [] for k in KINDS}}, [], protocol_hash="p")

            added1 = finder(m0, units0, store)
            self.assertGreater(added1, 0)
            self.assertEqual(backend_calls["n"], 2)
            self.assertTrue((tmp_path / "audit" / "SEARCH_AUDIT.json").is_file())
            self.assertTrue(cache_file.is_file())

            # Second invocation with fresh client reading same cache_file should hit disk cache
            client2 = LiteratureSearchClient(max_requests=10, cache_path=cache_file, backend=fake_backend)
            finder2 = AutonomousEvidenceFinder(gw, out_dir=tmp_path / "audit2", search_client=client2)
            finder2(m0, units0, store)
            self.assertEqual(backend_calls["n"], 2)
            self.assertGreater(client2.cache_hits, 0)

    def test_end_to_end_empty_local_memory_unblocked_by_evidence_finder(self):
        seeds = _load_bundled_seed_molecules()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inputs = tmp_path / "inputs"
            (inputs / "LLM" / "overall_local_v1" / "artifacts").mkdir(parents=True)
            (inputs / "runs" / "frontier_small_001" / "memory_store" / "memory").mkdir(parents=True)
            (inputs / "workflow_runtime_20261125" / "runtime" / "outputs" / "matrix_map_v1").mkdir(parents=True)

            # Start with completely EMPTY local memory!
            (inputs / "LLM" / "overall_local_v1" / "artifacts" / "MEMORY_V2.json").write_text(
                json.dumps({
                    "unit_index": {k: [] for k in KINDS},
                    "whole_evidence": [],
                    "unresolved_local_evidence": [],
                    "source_molecules": [],
                }),
                encoding="utf-8",
            )

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
                        "score": {"molecule_id": s.uid, "overall": {"score": 72.0 + idx * 2.0}},
                    },
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

            def offline_search(query, limit):
                return [
                    {
                        "doi": "10.1021/lohc.auto.2026",
                        "title": f"Carrier study for {query[:40]}",
                        "year": 2026,
                        "abstract": "Reversible hydrogenation and dehydrogenation cycling occurred cleanly without ring cleavage in bench tests.",
                    }
                ]

            gw = MockEvidenceGateway(limit=200)
            client = LiteratureSearchClient(max_requests=40, backend=offline_search)
            out_dir = tmp_path / "run_out"
            finder = AutonomousEvidenceFinder(
                gateway=gw,
                out_dir=out_dir / "evidence_search",
                search_client=client,
                max_llm_extract_calls=40,
            )

            report = runner.run(
                ARCHIVE_ROOT,
                out_dir,
                key=None,
                inputs=inputs,
                max_generations=2,
                strict_local_gate=True,
                gateway_override=gw,
                evidence_finder=finder,
                initial_pool_override=seeds,
            )

            self.assertEqual(report["actual_generations"], 2)
            self.assertEqual(report["status"], "2_generations_committed")
            self.assertGreater(report["final_memory"]["added_local_claims"], 0)
            self.assertGreater(report["evidence_search"]["claims_added_to_memory"], 0)
            self.assertGreater(report["search_calls"], 0)
            self.assertTrue((out_dir / "evidence_search" / "SEARCH_AUDIT.json").is_file())

    def test_high_evidence_gap_triggers_search_and_deduplicates_extraction(self):
        seeds = _load_bundled_seed_molecules()
        m0 = seeds[0]
        units0 = extract(m0.pair, m0.uid)

        def offline_search(query, limit):
            return [
                {
                    "doi": "10.1021/lohc.dedup.2026",
                    "title": "LOHC reversible cycling",
                    "year": 2026,
                    "abstract": "Reversible hydrogenation and dehydrogenation cycling occurred cleanly without ring cleavage across cycles.",
                }
            ]

        gw = MockEvidenceGateway(limit=100)
        client = LiteratureSearchClient(max_requests=20, backend=offline_search)
        finder = AutonomousEvidenceFinder(gw, search_client=client)
        store = MemoryStore({"unit_index": {k: [] for k in KINDS}}, [], protocol_hash="p")

        finder(m0, units0, store)
        self.assertEqual(finder.audit_log[-1]["llm_status"], "ok")
        calls_after_first = gw.calls
        finder(m0, units0, store)
        self.assertEqual(finder.audit_log[-1]["llm_status"], "skipped_already_extracted")
        self.assertEqual(gw.calls, calls_after_first)

        # Now test end-to-end with non-empty packet where only 1 of 4 unit kinds is covered (gap = 0.75)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inputs = tmp_path / "inputs"
            (inputs / "LLM" / "overall_local_v1" / "artifacts").mkdir(parents=True)
            (inputs / "runs" / "frontier_small_001" / "memory_store" / "memory").mkdir(parents=True)
            (inputs / "workflow_runtime_20261125" / "runtime" / "outputs" / "matrix_map_v1").mkdir(parents=True)

            unit_index = {k: [] for k in KINDS}
            for s in seeds:
                ext = extract(s.pair, s.uid)
                if ext["units"]["ring_or_conjugation"]:
                    u = ext["units"]["ring_or_conjugation"][0]
                    unit_index["ring_or_conjugation"].append({
                        "source_molecule": s.uid,
                        "reference": f"seed/{s.uid}/ring",
                        "source_id": "EXISTING_RING_SOURCE",
                        "stream": "local",
                        "evidence": {
                            "claim_id": f"c_{s.uid}",
                            "source_id": "EXISTING_RING_SOURCE",
                            "endpoint": "cycle_reversibility",
                            "target_state": u["state"],
                            "conditions": "450 K",
                            "catalyst": "Pt",
                            "quote": "Ring system supports reversible dehydrogenation.",
                            "observation": "Ring system supports reversible dehydrogenation.",
                            "attribution": "molecular_structure",
                        },
                        "source_unit": u,
                    })
            (inputs / "LLM" / "overall_local_v1" / "artifacts" / "MEMORY_V2.json").write_text(
                json.dumps({"unit_index": unit_index, "whole_evidence": [], "unresolved_local_evidence": [], "source_molecules": []}),
                encoding="utf-8",
            )
            global_records = [
                {
                    "case": s.uid,
                    "input": {"structure": {"whole_states": {"lean": state_from_dense(s.pair[0]), "rich": state_from_dense(s.pair[1])}}},
                    "evaluation": {"source": "LIVE_LLM", "score_context": "test", "critic": {"accept": True}, "score": {"molecule_id": s.uid, "overall": {"score": 75.0}}},
                }
                for s in seeds
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

            out_dir = tmp_path / "run_out"
            gw2 = MockEvidenceGateway(limit=200)
            client2 = LiteratureSearchClient(max_requests=40, backend=offline_search)
            finder2 = AutonomousEvidenceFinder(gw2, out_dir=out_dir / "evidence_search", search_client=client2, max_llm_extract_calls=40)
            report = runner.run(
                ARCHIVE_ROOT,
                out_dir,
                key=None,
                inputs=inputs,
                max_generations=1,
                strict_local_gate=False,
                gateway_override=gw2,
                evidence_finder=finder2,
                initial_pool_override=seeds,
                search_gap_threshold=0.75,
            )
            self.assertEqual(report["actual_generations"], 1)
            self.assertGreater(report["evidence_search"]["invocations"], 0)
            self.assertGreater(report["final_memory"]["added_local_claims"], 0)

    def test_radical_molecules_rejected_by_native_ga_filter(self):
        seeds = _load_bundled_seed_molecules()
        base = seeds[0]
        dirty_pair = base.pair.copy()
        dirty_pair[0, 0, 0, 5] = 1.0  # Inject unpaired radical electron on atom 0
        dirty_mol = Molecule("dirty_radical_uid", dirty_pair, ("[CH]1ccccc1", "C1CCCCC1"), "dirty_radical_uid")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inputs = tmp_path / "inputs"
            (inputs / "LLM" / "overall_local_v1" / "artifacts").mkdir(parents=True)
            (inputs / "runs" / "frontier_small_001" / "memory_store" / "memory").mkdir(parents=True)
            (inputs / "workflow_runtime_20261125" / "runtime" / "outputs" / "matrix_map_v1").mkdir(parents=True)
            (inputs / "LLM" / "overall_local_v1" / "artifacts" / "MEMORY_V2.json").write_text(
                json.dumps({"unit_index": {k: [] for k in KINDS}, "whole_evidence": [], "unresolved_local_evidence": [], "source_molecules": []}),
                encoding="utf-8",
            )
            (inputs / "runs" / "frontier_small_001" / "memory_store" / "memory" / "global.json").write_text(
                json.dumps({"records": []}), encoding="utf-8"
            )
            with (inputs / "workflow_runtime_20261125" / "runtime" / "outputs" / "matrix_map_v1" / "data_split.csv").open(
                "w", encoding="utf-8", newline=""
            ) as f:
                csv.DictWriter(f, fieldnames=["lean_smiles"]).writeheader()

            out_dir = tmp_path / "coverage_only"
            report = runner.run(
                ARCHIVE_ROOT,
                out_dir,
                key=None,
                inputs=inputs,
                initial_pool_override=[dirty_mol, *seeds[:2]],
            )
            initial_uids = {row["uid"] for row in json.loads((out_dir / "INITIAL_POOL.json").read_text(encoding="utf-8"))}
            self.assertNotIn("dirty_radical_uid", initial_uids)
            self.assertEqual(report["initial_pool"]["size"], 2)

    def test_broadened_tier_queries_vague_filter_and_coupling_stream(self):
        from overall_local_probe import local_result as probe_local_result

        seeds = _load_bundled_seed_molecules()
        m0 = seeds[0]
        units0 = extract(m0.pair, m0.uid)
        q3 = build_gap_queries(m0, units0, memory_store=None, max_queries=3)
        self.assertEqual(len(q3), 3)
        self.assertTrue(any(q.get("query_mode") == "failure_mode_counterexample" for q in q3))
        self.assertEqual({q["tier"] for q in q3}.issubset({1, 2, 3}), True)

        unit_lookup = {u["unit_id"]: (k, u) for k in KINDS for u in units0["units"][k]}
        uid_str, (kind, unit_obj) = next(iter(unit_lookup.items()))
        papers = [
            {
                "doi": "10.1021/coupling.test.2026",
                "url": "https://doi.org/10.1021/coupling.test.2026",
                "title": "Steric shielding suppresses C-N cleavage in alkylated heterocycles",
                "year": 2026,
                "abstract": (
                    "Thermodynamic properties of hydrogenation reactions in liquid phase were derived. "
                    "Ortho-methyl steric shielding suppressed catalyst poisoning and prevented C-N cleavage during reversible cycling."
                ),
            }
        ]
        response = {
            "claims": [
                {
                    "doi": "10.1021/coupling.test.2026",
                    "unit_id": uid_str,
                    "endpoint": "storage_stability",
                    "state": unit_obj["state"],
                    "catalyst": "Pt/Al2O3",
                    "conditions": "450 K",
                    "attribution": "molecular_structure",
                    "observation": "Thermodynamic properties of hydrogenation reactions in liquid phase were derived.",
                    "quote": "Thermodynamic properties of hydrogenation reactions in liquid phase were derived",
                },
                {
                    "doi": "10.1021/coupling.test.2026",
                    "unit_id": uid_str,
                    "endpoint": "cycle_reversibility",
                    "state": unit_obj["state"],
                    "catalyst": "Pt/Al2O3",
                    "conditions": "450 K",
                    "attribution": "molecular_structure",
                    "stream": "coupling",
                    "scope_tier": 3,
                    "polarity": "favorable",
                    "observation": "Ortho-methyl steric shielding suppresses catalyst poisoning and prevents C-N cleavage.",
                    "quote": "Ortho-methyl steric shielding suppressed catalyst poisoning and prevented C-N cleavage",
                },
                {
                    "doi": "10.1021/coupling.test.2026",
                    "unit_id": uid_str,
                    "endpoint": "storage_stability",
                    "state": unit_obj["state"],
                    "catalyst": "Pt/Al2O3",
                    "conditions": "450 K",
                    "attribution": "molecular_structure",
                    "stream": "local",
                    "scope_tier": 2,
                    "observation": "Steric shielding prevents C-N cleavage during reversible cycling.",
                    "quote": "suppressed catalyst poisoning and prevented C-N cleavage during reversible cycling",
                },
            ]
        }
        accepted, contextual, rejected = validate_extracted_claims(response, papers, unit_lookup)
        self.assertEqual(len(contextual), 0)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["error"], "Vague generic observation lacks mechanistic effect")
        self.assertEqual(len(accepted), 2)

        store = MemoryStore({"unit_index": {k: [] for k in KINDS}}, [], protocol_hash="p")
        for item in accepted:
            self.assertTrue(store.add_local_claim(item["kind"], item["unit"], item["claim_row"]))
        tags, packet = store.prepare_local(units0)
        self.assertEqual({t["stream"] for t in tags}, {"local", "coupling"})
        res = probe_local_result(
            packet,
            {"judgements": [{"id": r["id"], "direction": "favorable", "reason": "Synergistic protection"} for r in packet]},
            tags,
        )
        self.assertEqual(res["coupling_status"], "supported")
        self.assertGreater(res["coupling_multiplier"], 1.0)

    def test_dual_layer_chemical_benchmark_integrity(self):
        import chemical_benchmark

        ds = chemical_benchmark.build_benchmark_dataset()
        self.assertGreaterEqual(ds["molecule_count"], 24)
        self.assertEqual(ds["pair_count"], 16)
        layers = {p["layer"] for p in ds["contrast_pairs"]}
        self.assertEqual(layers, {"classic_lohc", "motif_and_functional_group"})
        for key, rec in ds["molecules"].items():
            self.assertGreater(rec["nH2"], 0.0, msg=f"{key} nH2 <= 0")
            self.assertGreater(sum(rec["unit_counts"].values()), 0, msg=f"{key} has 0 extracted units")


if __name__ == "__main__":
    unittest.main()
