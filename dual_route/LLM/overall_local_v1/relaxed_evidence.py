"""Experimental matrix retrieval and versioned similarity-weighted scoring."""
from collections import Counter


VERSION = "SIMILARITY_COUPLED_V1"
FLOORS = {"global": 0.15, "local": 0.15, "coupling": 0.10}


def similarity(left, right):
    """Weighted Jaccard on atom/bond feature counts."""
    total = sum((left | right).values())
    return sum((left & right).values()) / total if total else 0.0


def features(matrix, center=None, radius=3):
    size = len(matrix)
    nodes = set(range(size)) if center is None else {center}
    if center is not None:
        for _ in range(radius):
            nodes |= {j for i in nodes for j in range(size) if i != j and matrix[i, j, 7]}
    counts = Counter()
    for i in nodes:
        z = int(matrix[i, i, 0])
        counts["element", z] += 1
        counts["atom", z, tuple(matrix[i, i, 1:4])] += 1
        if i == center:
            counts["root", z, tuple(matrix[i, i, 1:4])] += 3
        for j in nodes:
            order = matrix[i, j, 7]
            if j > i and order:
                counts["bond", tuple(sorted((z, int(matrix[j, j, 0])))), float(order)] += 1
    return counts


def match(target, source, site, source_sites):
    """Read both states and compare local environments at three radii."""
    whole = sum(similarity(features(a), features(b)) for a, b in zip(target, source)) / 2
    if site is None:
        return whole, None, []
    scores = []
    for i in source_sites:
        if not 0 <= i < len(source[0]):
            continue
        if any(a[site, site, 0] != b[i, i, 0] for a, b in zip(target, source)):
            continue
        score = sum(similarity(features(a, site, r), features(b, i, r))
                    for a, b in zip(target, source) for r in (1, 2, 3)) / 6
        scores.append((score, i))
    peak = max((x[0] for x in scores), default=0)
    return whole, peak, [i for s, i in scores if s == peak and peak > 0]


def retrieve(target, site, records, endpoint, state, conditions=None, *, strict=False, limit=3):
    by_stream = {stream: [] for stream in FLOORS}
    for record, source in records:
        stream = record["agent"]
        if stream not in by_stream:
            continue
        for claim in record["evidence"]["claims"]:
            if claim["endpoint"] != endpoint or claim.get("target_state") != state:
                continue
            source_conditions = " ".join(claim.get("conditions", "").casefold().split())
            known = conditions is not None and source_conditions in conditions
            if conditions is not None and not known:
                continue
            whole, local, anchors = match(target, source, None if stream == "global" else site,
                                          claim.get("site_parent_ids", []))
            if stream != "global" and not anchors:
                continue
            if stream == "coupling" and not claim.get("basis_refs"):
                continue
            score = {"global": whole, "local": local, "coupling": min(whole, local or 0)}[stream]
            if score < (1.0 if strict else FLOORS[stream]) or (strict and not known):
                continue
            by_stream[stream].append({
                "reference": record["record_id"] + "/" + claim["claim_id"],
                "source_id": claim["source_id"], "similarity": score,
                "whole_similarity": whole, "local_similarity": local,
                "state": state, "endpoint": endpoint,
                "applicability": (1.0 if known else 0.5) / max(1, len(anchors)),
                "source_conditions": claim.get("conditions"), "source_sites": anchors,
                "target_site": site if stream != "global" else None,
                "quote": claim.get("quote"), "observation": claim.get("observation"),
                "evidence_level": claim.get("evidence_level"),
            })
    for stream, hits in by_stream.items():
        unique = {}
        for hit in sorted(hits, key=lambda h: (-h["similarity"], h["reference"])):
            unique.setdefault(hit["source_id"], hit)
        by_stream[stream] = list(unique.values())[:limit]
    return by_stream


def aggregate(items, power=2.0):
    """Absolute weighting prevents a lone weak source from gaining full weight."""
    strongest = {}
    for item in items:
        strength = item["similarity"] ** power * item["quality"] * item["applicability"]
        key = item["source_id"]
        if key not in strongest or strength > strongest[key][0]:
            strongest[key] = strength, item["direction"]
    total = sum(w for w, _ in strongest.values())
    signed = sum(w * direction for w, direction in strongest.values()) / max(1, total)
    return signed, min(1, total)


def score(global_items, local_items, coupling_items, lam=0.2):
    global_signal, global_coverage = aggregate(global_items)
    local_signal, local_coverage = aggregate(local_items)
    coupling, coupling_coverage = aggregate(coupling_items)
    present = [signal for signal, coverage in ((global_signal, global_coverage),
              (local_signal, local_coverage)) if coverage > 0]
    base = 50 + 50 * sum(present) / len(present) if present else None
    multiplier = 1 + lam * coupling
    return {"version": VERSION, "base": base, "coupling_multiplier": multiplier,
            "score": min(100, max(0, base * multiplier)) if base is not None else None,
            "coverage": {"global": global_coverage, "local": local_coverage, "coupling": coupling_coverage},
            "coupling_status": "supported" if coupling_coverage else "unknown"}


def query_tiers(target_name, scaffold, motif, endpoint_terms, mechanism_terms):
    """Use OR within each tier and release the exact-name constraint first."""
    endpoint = " OR ".join(f'"{term}"' for term in endpoint_terms)
    return [
        f'"{target_name}" AND ({endpoint})',
        f'("{scaffold}" OR "{motif}") AND ({endpoint})',
        f'"{motif}" AND (' + " OR ".join(f'"{term}"' for term in mechanism_terms) + ")",
    ]
