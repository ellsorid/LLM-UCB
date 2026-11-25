"""Property-to-structure selection and score-gated population updates."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite, sqrt

import numpy as np


@dataclass(frozen=True)
class EditCandidate:
    uid: str
    parent_uid: str
    parent_site: int
    cell: int
    psi: np.ndarray
    response: np.ndarray
    arm: tuple[str, str, str]
    units: frozenset[str]
    evidence_gap: float
    evidence_refs: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Evaluation:
    uid: str
    context: str
    score: float               # 0–100, from the program's overall/local aggregation
    evidence_confidence: float # 0–1, separate from the score
    evidence_refs: frozenset[str]
    validation_passed: bool


@dataclass(frozen=True)
class EditEvidence:
    parent_uid: str
    child_uid: str
    parent_site: int
    context: str
    evidence_refs: frozenset[str]
    comparable_and_attributable: bool


@dataclass
class PopulationState:
    mapping: object
    members: dict
    evaluations: dict = field(default_factory=dict)
    posts: dict = field(default_factory=dict)
    committed: set = field(default_factory=set)


def _fraction(value):
    return isfinite(value) and 0 <= value <= 1


def _edit_readout(state, child):
    parent = state.members[child.parent_uid]
    if (child.response.shape != parent.response.shape or not 0 <= child.cell < len(child.response)
        or not np.isfinite(child.response).all() or not np.isfinite(parent.response).all()):
        raise ValueError("Incompatible SOM responses")
    if not _fraction(child.evidence_gap):
        raise ValueError("Evidence gap must be a fraction")
    mean, variance = state.mapping.predict(child.psi)
    delta, delta_var = state.mapping.delta(parent.psi, child.psi)
    response_change = float(child.response[child.cell]-parent.response[child.cell])
    return {"expected_score":float(np.clip(mean[0],0,1)),
            "predicted_score_delta":delta, "delta_sd":sqrt(delta_var),
            "prediction_sd":sqrt(float(variance[0])),
            "destination_response_delta":response_change}


def select_batch(state, candidates, budget):
    """Project property goals back onto reachable cells, then select real molecules."""
    if budget < 0:
        raise ValueError("Evaluation budget must be nonnegative")
    rows = list(candidates)
    if len({r.uid for r in rows}) != len(rows):
        raise ValueError("Duplicate candidate")
    rows = [r for r in rows if r.uid not in state.members and r.parent_uid in state.members]
    evaluated_per_cell = Counter(state.members[uid].cell for uid in state.evaluations)
    covered = set().union(*(state.members[uid].units for uid in state.evaluations))
    by_cell = defaultdict(list)
    audit = {}
    for row in rows:
        r = _edit_readout(state,row)
        unseen = len(row.units-covered)/max(len(row.units),1)
        underexplored = 1/(1+evaluated_per_cell[row.cell])
        r["evaluated_in_cell"] = evaluated_per_cell[row.cell]
        r["unseen_units_fraction"] = unseen
        r["cell_priority"] = (
            .35*r["expected_score"]+.20*np.clip(r["predicted_score_delta"],0,1)
            +.15*min(r["prediction_sd"],1)+.10*max(r["destination_response_delta"],0)
            +.20*underexplored
        )
        r["contrast_priority"] = (
            .35*min(r["delta_sd"],1)+.25*unseen+.20*row.evidence_gap
            +.20*min(abs(r["predicted_score_delta"]),1)
        )
        audit[row.uid] = r
        by_cell[row.cell].append(row)
    representatives = []
    for cell, members in by_cell.items():
        priorities = [audit[r.uid]["cell_priority"] for r in members]
        cutoff = np.median(priorities)
        eligible = [r for r in members if audit[r.uid]["cell_priority"]>=cutoff]
        vectors = np.stack([r.psi for r in eligible])
        distances = np.linalg.norm(vectors-vectors.mean(0),axis=1)
        best = min(range(len(eligible)),key=lambda i:(distances[i],eligible[i].uid))
        representatives.append((max(priorities),cell,eligible[best]))
    representatives.sort(key=lambda item:(-item[0],item[1],item[2].uid))
    selection = []
    if representatives and budget:
        selection.append((representatives[0][2],"regional_representative"))
    if selection and budget >= 2:
        first = selection[0][0].uid
        contrasts = [r for r in rows if r.uid!=first and r.parent_uid in state.evaluations]
        if contrasts:
            chosen = min(contrasts,key=lambda r:(-audit[r.uid]["contrast_priority"],r.uid))
            selection.append((chosen,"local_contrast"))
    already = {r.uid for r,_ in selection}
    selection.extend((r,"regional_representative") for _,_,r in representatives if r.uid not in already)
    return selection[:budget], audit


def commit_batch(state, batch, evaluations, edit_evidence):
    """Return a new state only after every queued prediction is program-validated."""
    uids = {r.uid for r in batch}
    if len(uids)!=len(batch) or uids & state.members.keys():
        raise ValueError("Duplicate or already committed molecule")
    validated = {}
    problems = {}
    for row in batch:
        e = evaluations.get(row.uid)
        if (e is None or e.uid!=row.uid or e.validation_passed is not True or e.context!=state.mapping.context
            or not isfinite(e.score) or not 0<=e.score<=100 or not _fraction(e.evidence_confidence)
            or not e.evidence_refs or row.parent_uid not in state.members):
            problems[row.uid] = "Missing, invalid or incompatible score"
        else:
            validated[row.uid] = e
    if problems or not batch:
        return state, {"committed":False,"problems":problems,"operator_updates":[]}

    updated = deepcopy(state)
    rewards = []
    for row in batch:
        e = validated[row.uid]
        variance = .02+.18*(1-e.evidence_confidence)**2
        updated.mapping.update(e.uid,row.psi,e.score/100,variance,e.context)
        linked = edit_evidence.get(row.uid)
        parent = state.evaluations.get(row.parent_uid)
        if (parent and parent.context==e.context and linked and linked.comparable_and_attributable
            and linked.context==e.context and linked.parent_uid==parent.uid and linked.child_uid==e.uid
            and linked.parent_site==row.parent_site
            and linked.evidence_refs & parent.evidence_refs & e.evidence_refs):
            prior = updated.posts.get(row.arm,{"mean":0.,"variance":.04})
            reward = (e.score-parent.score)/100
            noise = .04+.12*(1-min(e.evidence_confidence,parent.evidence_confidence))**2
            gain = prior["variance"]/(prior["variance"]+noise)
            updated.posts[row.arm] = {"mean":prior["mean"]+gain*(reward-prior["mean"]),
                                      "variance":prior["variance"]*(1-gain)}
            rewards.append({"uid":row.uid,"arm":row.arm,"reward":reward})
        updated.members[row.uid] = row
        updated.evaluations[row.uid] = e
        updated.committed.add(row.uid)
    return updated, {"committed":True,"problems":{},"operator_updates":rewards}


def generation_quotas(state, candidates, budget, directed_budget=1):
    """Separate property-directed generation from evidence-gap exploration."""
    if budget < 0 or directed_budget < 0:
        raise ValueError("Generation budgets must be nonnegative")
    rows = [r for r in candidates if r.parent_uid in state.evaluations and r.uid not in state.members]
    if len({r.uid for r in rows}) != len(rows):
        raise ValueError("Duplicate candidate in quota allocation")
    records = []
    for row in rows:
        pred = _edit_readout(state,row)
        post = state.posts.get(row.arm,{"mean":0.,"variance":.04})
        mean = post["mean"]
        beta = sqrt(post["variance"]*row.evidence_gap)
        directed = min(max(pred["predicted_score_delta"],0),1)*max(pred["destination_response_delta"],0) if mean>0 else 0.
        records.append((row,mean+beta,directed))
    directed = sorted((r for r in records if r[2]>0),key=lambda r:(-r[2],r[0].uid))[:min(budget,directed_budget)]
    used = {r[0].uid for r in directed}
    exploration = sorted((r for r in records if r[0].uid not in used),key=lambda r:(-r[1],r[0].uid))
    chosen = [*(("directed",r) for r in directed),
              *(("exploration",r) for r in exploration[:budget-len(directed)])]
    return {"routes":[{"uid":r.uid,"route":route,"arm":r.arm,"exploration_utility":u,"directed_utility":d}
                      for route,(r,u,d) in chosen],
            "quota_by_arm":dict(Counter("|".join(r.arm) for _,(r,_,_) in chosen))}
