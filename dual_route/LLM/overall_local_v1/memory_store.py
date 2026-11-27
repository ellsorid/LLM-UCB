"""Dynamic growing evidence memory store and LLM judgement cache."""
import copy
from hashlib import sha256
import json
from pathlib import Path
import threading

from chemical_units import KINDS
from overall_local_probe import prepare_local as _prepare_local
from overall_local_probe import whole_prior as _whole_prior


class MemoryStore:
    """In-run dynamic memory holding unit-indexed local claims, global priors, and caches."""

    def __init__(self, local_memory, global_records, protocol_hash="", observations_override=None):
        self.local_memory = copy.deepcopy(local_memory) if local_memory else {}
        unit_index = self.local_memory.setdefault("unit_index", {})
        for kind in KINDS:
            unit_index.setdefault(kind, [])
        self.local_memory.setdefault("whole_evidence", [])
        self.local_memory.setdefault("unresolved_local_evidence", [])
        self.local_memory.setdefault("source_molecules", [])
        self.global_records = copy.deepcopy(list(global_records)) if global_records else []
        self.protocol_hash = str(protocol_hash)
        self.observations_override = observations_override
        self.judgement_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.open_hypotheses = []
        self._open_uids = set()
        self.added_local_claims = 0
        self.added_global_records = 0
        self._lock = threading.RLock()
        self._local_dedup = {
            (
                row.get("reference"),
                row.get("source_id"),
                row.get("source_unit", {}).get("name"),
                row.get("source_unit", {}).get("state"),
                tuple(row.get("source_unit", {}).get("parent_ids", ())),
            )
            for group in unit_index.values()
            for row in group
        }
        self._global_cases = {r.get("case") for r in self.global_records if r.get("case")}

    def prepare_local(self, target_units):
        with self._lock:
            return _prepare_local(
                target_units,
                self.local_memory,
                observations_override=self.observations_override,
                allow_general_claims=True,
            )

    def whole_prior(self, pair, uid):
        with self._lock:
            return _whole_prior(pair, uid, self.global_records)

    def add_local_claim(self, kind, unit, claim_row):
        """Validate, deduplicate, and append a chemical-unit-indexed local claim."""
        if kind not in KINDS:
            raise ValueError(f"Unsupported chemical unit kind: {kind}")
        if not isinstance(unit, dict) or not isinstance(claim_row, dict):
            raise ValueError("unit and claim_row must be dicts")
        required_unit = ("name", "state", "parent_ids", "atom_labels", "internal_bonds", "boundary_bonds")
        if any(field not in unit for field in required_unit):
            raise ValueError("Source unit is missing required structural fields")
        source_id = str(claim_row.get("source_id", "")).strip()
        reference = str(claim_row.get("reference", "")).strip()
        endpoint = str(claim_row.get("endpoint", "")).strip()
        state = str(claim_row.get("state", "")).strip()
        text = str(claim_row.get("observation") or claim_row.get("quote") or "").strip()
        if not source_id or not reference or not endpoint or state not in ("lean", "rich") or not text:
            raise ValueError("Claim row must specify source_id, reference, endpoint, state, and observation/quote")
        if unit["state"] != state:
            raise ValueError("Source unit state must match claim state")
        dedup_key = (
            reference,
            source_id,
            unit["name"],
            unit["state"],
            tuple(unit["parent_ids"]),
            str(claim_row.get("stream", "local")),
        )
        with self._lock:
            if dedup_key in self._local_dedup:
                return False
            stream = str(claim_row.get("stream", "local")).strip() or "local"
            scope_tier = claim_row.get("scope_tier")
            row = {
                "source_molecule": str(claim_row.get("source_molecule", "external_literature")),
                "reference": reference,
                "source_id": source_id,
                "stream": stream,
                "evidence": {
                    "claim_id": str(claim_row.get("claim_id", reference)),
                    "source_id": source_id,
                    "endpoint": endpoint,
                    "target_state": state,
                    "scope_tier": scope_tier,
                    "polarity": claim_row.get("polarity"),
                    "partner_unit_id": claim_row.get("partner_unit_id"),
                    "conditions": str(claim_row.get("conditions", "unspecified")),
                    "catalyst": str(claim_row.get("catalyst", claim_row.get("conditions", "unspecified"))),
                    "quote": str(claim_row.get("quote") or text),
                    "observation": text,
                    "attribution": str(claim_row.get("attribution", "molecular_structure")),
                },
                "source_unit": copy.deepcopy(unit),
            }
            self.local_memory["unit_index"][kind].append(row)
            self._local_dedup.add(dedup_key)
            self.added_local_claims += 1
            return True

    def add_global_record(
        self,
        molecule,
        combined_score,
        overall_score,
        local_score,
        score_context,
        evidence_refs,
        generation,
    ):
        """Serialize a committed paired molecule and append it to global analogy memory."""
        from matrix_pipeline.compat.pair_core import state_from_dense

        with self._lock:
            if molecule.uid in self._global_cases:
                return False
            record = {
                "case": molecule.uid,
                "generation": int(generation),
                "input": {
                    "structure": {
                        "whole_states": {
                            "lean": state_from_dense(molecule.pair[0]),
                            "rich": state_from_dense(molecule.pair[1]),
                        }
                    }
                },
                "evaluation": {
                    "source": "CLOSED_LOOP_COMMITTED",
                    "score_context": str(score_context),
                    "critic": {"accept": True},
                    "score": {
                        "molecule_id": molecule.uid,
                        "overall": {"score": float(combined_score)},
                        "prior_overall": float(overall_score) if overall_score is not None else None,
                        "local": float(local_score) if local_score is not None else None,
                    },
                    "evidence_refs": sorted(evidence_refs),
                },
            }
            self.global_records.append(record)
            self._global_cases.add(molecule.uid)
            self.added_global_records += 1
            return True

    def cache_key(self, obs_item, tag):
        payload = [
            self.protocol_hash,
            str(tag.get("source_id", "")),
            str(tag.get("kind", "")),
            str(tag.get("state", "")),
            str(obs_item.get("unit_type", "")),
            str(obs_item.get("observation", "")),
        ]
        return sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()

    def lookup_cached(self, packet, tags):
        """Split observation packet into cached LLM judgements and uncached observations."""
        tag_by_id = {row["id"]: row for row in tags}
        cached_judgements = []
        uncached_packet = []
        with self._lock:
            for obs in packet:
                tag = tag_by_id.get(obs["id"], {})
                key = self.cache_key(obs, tag)
                if key in self.judgement_cache:
                    cached = self.judgement_cache[key]
                    cached_judgements.append({
                        "id": obs["id"],
                        "direction": cached["direction"],
                        "reason": cached["reason"],
                    })
                    self.cache_hits += 1
                else:
                    uncached_packet.append(obs)
                    self.cache_misses += 1
        return cached_judgements, uncached_packet

    def store_cached(self, packet, tags, judgements):
        obs_by_id = {row["id"]: row for row in packet}
        tag_by_id = {row["id"]: row for row in tags}
        with self._lock:
            for j in judgements:
                obs = obs_by_id.get(j["id"])
                tag = tag_by_id.get(j["id"])
                if obs is None or tag is None:
                    continue
                key = self.cache_key(obs, tag)
                self.judgement_cache[key] = {
                    "direction": j["direction"],
                    "reason": j["reason"],
                }

    def record_open_hypothesis(self, uid, cell, smiles, missing_unit_kinds, generation, reason):
        with self._lock:
            if uid in self._open_uids:
                return
            self._open_uids.add(uid)
            self.open_hypotheses.append({
                "uid": uid,
                "cell": int(cell),
                "smiles": list(smiles),
                "missing_unit_kinds": sorted(missing_unit_kinds),
                "generation": int(generation),
                "reason": str(reason),
            })

    def snapshot(self):
        with self._lock:
            counts = {kind: len(rows) for kind, rows in self.local_memory["unit_index"].items()}
            return {
                "local_claims_by_kind": counts,
                "total_local_claims": sum(counts.values()),
                "added_local_claims": self.added_local_claims,
                "total_global_records": len(self.global_records),
                "added_global_records": self.added_global_records,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_size": len(self.judgement_cache),
                "open_hypotheses_count": len(self.open_hypotheses),
                "open_hypotheses": list(self.open_hypotheses),
            }

    def save(self, path):
        with self._lock:
            summary = self.snapshot()
            payload = summary | {
                "local_memory": copy.deepcopy(self.local_memory),
                "global_records_meta": [
                    {
                        "case": r.get("case"),
                        "generation": r.get("generation"),
                        "source": r.get("evaluation", {}).get("source"),
                        "score": r.get("evaluation", {}).get("score", {}).get("overall", {}).get("score"),
                    }
                    for r in self.global_records
                ],
            }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
