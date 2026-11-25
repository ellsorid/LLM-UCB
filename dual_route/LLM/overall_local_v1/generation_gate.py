"""Program-directed local evidence and all-or-nothing generation handoff."""
from collections import deque
from math import isfinite


def matrix_sites(pair, edit_site):
    """State-specific targets through three bonds of the edited atom."""
    sites = {}
    for state, matrix in zip(("lean", "rich"), pair):
        if not 0 <= edit_site < len(matrix):
            raise ValueError("Edit site is missing from the mapped state")
        queue = deque([(edit_site, 0)])
        depth = {edit_site: 0}
        rows = []
        while queue:
            i, distance = queue.popleft()
            neighbors = [(j, float(matrix[i, j, 7])) for j in range(len(matrix))
                         if j != i and matrix[i, j, 7]]
            rows.append({"site_parent_id": i, "distance_from_edit": distance,
                         "atomic_number": int(matrix[i, i, 0]),
                         "aromatic": bool(matrix[i, i, 3]),
                         "bonds": [{"parent_id": j, "order": order,
                                    "atomic_number": int(matrix[j, j, 0])}
                                   for j, order in neighbors]})
            if distance == 3:
                continue
            for j, _ in neighbors:
                if j not in depth:
                    depth[j] = distance + 1
                    queue.append((j, distance + 1))
        sites[state] = rows
    return sites


def local_gap_queries(claims, sites):
    """Generate a fallback mechanism query only for an uncovered property."""
    covered = {c["endpoint"] for c in claims}
    atoms = sorted(
        (row for rows in sites.values() for row in rows if row["atomic_number"] in {7, 8, 16}),
        key=lambda row: row["distance_from_edit"],
    )
    if not atoms:
        motif = "organic molecular structure"
    else:
        motif = {7: "nitrogen-containing organic compounds", 8: "oxygen-containing organic compounds",
                 16: "sulfur-containing organic compounds"}[atoms[0]["atomic_number"]]
    terms = {
        "storage_stability": '("storage stability" OR "oxidative degradation")',
        "cycle_reversibility": '("dehydrogenation" OR "hydrogenation cycling")',
    }
    return [f'"{motif}" AND {keywords}' for endpoint, keywords in terms.items() if endpoint not in covered]


def retain_local_claims(result, sites, validator, prior=()):
    """Keep valid source-backed claims and record why individual rows failed."""
    if not isinstance(result, dict) or not isinstance(result.get("claims"), list):
        return result, []
    empty = result | {"claims": []}
    if validator(empty):
        return result, []
    allowed = {state: {row["site_parent_id"] for row in rows} for state, rows in sites.items()}
    valid, rejected = [], []
    ids, signatures = set(), set()
    for claim in [*prior, *result["claims"]]:
        if not isinstance(claim, dict):
            rejected.append({"error": "Invalid local evidence row"})
            continue
        state = claim.get("target_state")
        target_ids = claim.get("site_parent_ids")
        is_located = isinstance(target_ids, list) and bool(target_ids) and all(
            type(i) is int and i in allowed.get(state, ()) for i in target_ids
        )
        try:
            errors = validator(empty | {"claims": [claim]}) if is_located else ["No allowed target site in that state"]
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            errors = [str(error)]
        if errors:
            rejected.append({"claim_id": claim.get("claim_id"), "source_id": claim.get("source_id"),
                             "endpoint": claim.get("endpoint"), "errors": errors})
            continue
        signature = (state, claim["endpoint"], claim["source_id"], claim["quote_ref"], tuple(sorted(target_ids)))
        if signature in signatures:
            continue
        signatures.add(signature)
        claim_id = claim["claim_id"]
        if claim_id in ids:
            suffix = 1
            while f"{claim_id}_R{suffix}" in ids:
                suffix += 1
            claim = claim | {"claim_id": f"{claim_id}_R{suffix}"}
        ids.add(claim["claim_id"])
        valid.append(claim)
    return result | {"claims": valid}, rejected


def generation_missing(queue, staged):
    """A queued molecule needs its own accepted score before the generation can commit."""
    missing = []
    for item in queue:
        uid = item["uid"]
        outcome = staged.get(uid)
        if not outcome or outcome.get("critic", {}).get("accept") is not True:
            missing.append(uid)
            continue
        score = outcome.get("score", {})
        value = score.get("overall", {}).get("score")
        if score.get("molecule_id") != uid or type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 100:
            missing.append(uid)
    return missing
