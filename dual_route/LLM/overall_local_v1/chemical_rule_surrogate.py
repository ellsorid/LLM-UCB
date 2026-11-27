"""LLM chemical rule induction, empirical falsification, and SOM+rule grey-box Bayesian surrogate."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import math
from typing import Any

import numpy as np
from rdkit import Chem


VALID_PREDICATE_TYPES = {"smarts_lean", "smarts_rich", "smarts_pair", "coupling_smarts"}
VALID_EXPECTED_DIRECTIONS = {"favorable", "unfavorable", "coupling_synergy", "coupling_antagonism"}
VALID_RULE_STATUS = {"tentative", "supported", "revised", "falsified"}
TIER_PRIOR_MAGNITUDE = {1: 0.025, 2: 0.018, 3: 0.014}

RULE_INDUCTION_TASK = """You are a physical-organic and catalysis chemist inducing explicit, testable structure-property rules during iterative hydrogen-carrier discovery.
Inspect ONLY the supplied run observations:
  1. Newly ingested verified literature claims (with DOI source_id, state, endpoint, stream local/coupling);
  2. High-residual surprises where the current SOM+rule surrogate erred on real scores/penalties;
  3. Parent-to-child Matched Molecular Pairs showing how structural edits changed score, SA, and H2 capacity;
  4. Existing active chemical rules with their empirical Bayesian weights (beta_points, sd_points, z_score, status).
Synthesize generalizable, machine-executable RDKit SMARTS rules (e.g., terminal alkyne thermal polymerization risk, C-N dealkylation of long N-alkyl chains, ortho-methyl steric shielding of ring N, ketone/alcohol reversibility, furan C-O ring-opening risk, antagonistic amine+ketone condensation).
Rules MUST satisfy:
  - `predicate_type`: `"smarts_lean"` | `"smarts_rich"` | `"smarts_pair"` | `"coupling_smarts"`
  - `expected_direction`: `"favorable"` | `"unfavorable"` | `"coupling_synergy"` | `"coupling_antagonism"`
  - `scope_tier`: 1 (LOHC), 2 (hydrogen-storage motif reversibility), or 3 (functional-group stability/failure/coupling)
  - Valid non-trivial RDKit SMARTS strings (`>= 2` heavy/SMARTS atoms; supply `smarts_lean` for lean/pair/coupling, `smarts_rich` for rich/pair, `partner_smarts` for coupling_smarts)
  - `supporting_dois` MUST use ONLY exact DOI or reference IDs provided in `allowed_refs`. Never invent DOIs!
  - Propose at most 2–3 highest-signal `new_rules` per generation with concise 1-sentence `chemical_statement` values so the JSON finishes cleanly without truncation.
Return JSON exactly:
{"new_rules":[{"name":"short_snake_or_title","chemical_statement":"concise mechanistic cause and effect","predicate_type":"smarts_lean|smarts_rich|smarts_pair|coupling_smarts","smarts_lean":"SMARTS_or_empty","smarts_rich":"SMARTS_or_empty","partner_smarts":"SMARTS_or_empty","expected_direction":"favorable|unfavorable|coupling_synergy|coupling_antagonism","scope_tier":2,"endpoint":"cycle_reversibility|storage_stability","boundary_conditions":"catalyst/temperature/scaffold scope","supporting_dois":["exact ref from allowed_refs"]}],"revise_rules":[{"rule_id":"existing_id","chemical_statement":"updated mechanism","boundary_conditions":"updated scope","supporting_dois":["exact ref"]}],"falsify_rule_ids":["existing_id_contradicted_by_data"]}
If no new rule is warranted, return {"new_rules":[],"revise_rules":[],"falsify_rule_ids":[]}."""


@dataclass
class ChemicalRule:
    rule_id: str
    name: str
    chemical_statement: str
    predicate_type: str
    smarts_lean: str = ""
    smarts_rich: str = ""
    partner_smarts: str = ""
    expected_direction: str = "favorable"
    scope_tier: int = 2
    endpoint: str = "cycle_reversibility"
    boundary_conditions: str = "unspecified"
    supporting_dois: list[str] = field(default_factory=list)
    contradicting_dois: list[str] = field(default_factory=list)
    generation_created: int = 0
    status: str = "tentative"
    beta_points: float = 0.0
    sd_points: float = 0.0
    z_score: float = 0.0
    n_matched_observed: int = 0

    def expected_sign(self) -> float:
        return 1.0 if self.expected_direction in ("favorable", "coupling_synergy") else -1.0

    def signature(self) -> tuple[str, str, str, str]:
        return (
            self.predicate_type,
            self.smarts_lean.strip(),
            self.smarts_rich.strip(),
            self.partner_smarts.strip(),
        )

    def prior_mean_01(self) -> float:
        base = TIER_PRIOR_MAGNITUDE.get(int(self.scope_tier), 0.014)
        support_factor = min(1.5, 0.7 + 0.3 * max(1, len(self.supporting_dois)))
        return float(self.expected_sign() * base * support_factor)


_SMARTS_CACHE: dict[str, Chem.Mol | None] = {}


def _compile_smarts(smarts: str) -> Chem.Mol | None:
    s = str(smarts or "").strip()
    if not s:
        return None
    if s not in _SMARTS_CACHE:
        try:
            _SMARTS_CACHE[s] = Chem.MolFromSmarts(s)
        except Exception:
            _SMARTS_CACHE[s] = None
    return _SMARTS_CACHE[s]


def _pair_to_mols(pair: np.ndarray) -> tuple[Chem.Mol, Chem.Mol]:
    from matrix_pipeline.core import to_mol

    return to_mol(pair[0]), to_mol(pair[1])


def evaluate_rule(rule: ChemicalRule, pair: np.ndarray) -> float:
    """Evaluate a compiled RDKit SMARTS chemical rule predicate on a [2, n, n, 12] pair."""
    try:
        lean_mol, rich_mol = _pair_to_mols(pair)
    except Exception:
        return 0.0

    ptype = rule.predicate_type
    if ptype == "smarts_lean":
        pat = _compile_smarts(rule.smarts_lean)
        return 1.0 if (pat is not None and lean_mol.HasSubstructMatch(pat)) else 0.0
    if ptype == "smarts_rich":
        pat = _compile_smarts(rule.smarts_rich)
        return 1.0 if (pat is not None and rich_mol.HasSubstructMatch(pat)) else 0.0
    if ptype == "smarts_pair":
        p_lean = _compile_smarts(rule.smarts_lean)
        p_rich = _compile_smarts(rule.smarts_rich)
        if p_lean is None or p_rich is None:
            return 0.0
        return 1.0 if (lean_mol.HasSubstructMatch(p_lean) and rich_mol.HasSubstructMatch(p_rich)) else 0.0
    if ptype == "coupling_smarts":
        p_main = _compile_smarts(rule.smarts_lean)
        p_partner = _compile_smarts(rule.partner_smarts)
        if p_main is None or p_partner is None:
            return 0.0
        has_main = lean_mol.HasSubstructMatch(p_main) or rich_mol.HasSubstructMatch(p_main)
        has_partner = lean_mol.HasSubstructMatch(p_partner) or rich_mol.HasSubstructMatch(p_partner)
        return 1.0 if (has_main and has_partner) else 0.0
    return 0.0


def _normalize_ref(ref: str) -> str:
    value = str(ref or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.lower().startswith(prefix):
            value = value[len(prefix):]
            break
    return value.strip().lower()


def validate_rule_proposal(
    proposal: dict[str, Any],
    allowed_refs: set[str] | list[str] | tuple[str, ...],
    sample_pairs: list[np.ndarray] | None = None,
) -> tuple[bool, ChemicalRule | None, str]:
    """Strictly validate an LLM rule proposal against SMARTS compilation, nontriviality, and allowed refs."""
    if not isinstance(proposal, dict):
        return False, None, "Proposal must be a dict"

    name = str(proposal.get("name") or "").strip()
    stmt = str(proposal.get("chemical_statement") or "").strip()
    ptype = str(proposal.get("predicate_type") or "").strip()
    edirection = str(proposal.get("expected_direction") or "").strip()
    endpoint = str(proposal.get("endpoint") or "cycle_reversibility").strip()
    boundary = str(proposal.get("boundary_conditions") or "unspecified").strip() or "unspecified"
    try:
        scope_tier = int(proposal.get("scope_tier", 2))
    except (TypeError, ValueError):
        return False, None, "Invalid scope_tier"

    if not name or len(stmt) < 10:
        return False, None, "Missing rule name or chemical_statement"
    if ptype not in VALID_PREDICATE_TYPES:
        return False, None, f"Unsupported predicate_type: {ptype}"
    if edirection not in VALID_EXPECTED_DIRECTIONS:
        return False, None, f"Unsupported expected_direction: {edirection}"
    if scope_tier not in (1, 2, 3):
        return False, None, f"Unsupported scope_tier: {scope_tier}"
    if endpoint not in ("cycle_reversibility", "storage_stability"):
        return False, None, f"Unsupported endpoint: {endpoint}"

    s_lean = str(proposal.get("smarts_lean") or "").strip()
    s_rich = str(proposal.get("smarts_rich") or "").strip()
    s_part = str(proposal.get("partner_smarts") or "").strip()

    required_smarts: list[str] = []
    if ptype == "smarts_lean":
        required_smarts = [s_lean]
    elif ptype == "smarts_rich":
        required_smarts = [s_rich]
    elif ptype == "smarts_pair":
        required_smarts = [s_lean, s_rich]
    elif ptype == "coupling_smarts":
        required_smarts = [s_lean, s_part]

    _TRIVIAL_1ATOM_SMARTS = {"*", "[*]", "C", "[C]", "[#6]", "[c]", "A", "[A]"}
    for idx_s, s in enumerate(required_smarts):
        if not s:
            return False, None, f"Missing required SMARTS for {ptype}"
        pat = _compile_smarts(s)
        if pat is None:
            return False, None, f"Invalid SMARTS pattern: {s}"
        is_partner = (ptype == "coupling_smarts" and idx_s == 1)
        if pat.GetNumAtoms() < 2:
            if not is_partner or s in _TRIVIAL_1ATOM_SMARTS:
                return False, None, f"Trivial <2-atom SMARTS rejected: {s}"

    raw_refs = proposal.get("supporting_dois")
    if not isinstance(raw_refs, list) or not raw_refs:
        return False, None, "supporting_dois must be a non-empty list"
    norm_allowed = {_normalize_ref(r) for r in allowed_refs if _normalize_ref(r)}
    norm_allowed_raw = {str(r).strip().lower() for r in allowed_refs if str(r).strip()}
    cleaned_refs: list[str] = []
    for ref in raw_refs:
        nr = _normalize_ref(str(ref))
        rr = str(ref).strip().lower()
        if nr not in norm_allowed and rr not in norm_allowed_raw:
            return False, None, f"Hallucinated or unauthorized supporting ref: {ref}"
        if nr and nr not in cleaned_refs:
            cleaned_refs.append(nr)

    rule_sig_hash = sha256(f"{ptype}|{s_lean}|{s_rich}|{s_part}".encode("utf-8")).hexdigest()[:10]
    rule = ChemicalRule(
        rule_id=f"R_{rule_sig_hash}",
        name=name[:80],
        chemical_statement=stmt[:320],
        predicate_type=ptype,
        smarts_lean=s_lean,
        smarts_rich=s_rich,
        partner_smarts=s_part,
        expected_direction=edirection,
        scope_tier=scope_tier,
        endpoint=endpoint,
        boundary_conditions=boundary[:180],
        supporting_dois=cleaned_refs,
    )

    if sample_pairs and len(sample_pairs) >= 6:
        hits = sum(1 for pr in sample_pairs if evaluate_rule(rule, pr) > 0.5)
        if hits / len(sample_pairs) >= 0.95:
            return False, None, f"Overly broad wildcard predicate matched {hits}/{len(sample_pairs)} samples"

    return True, rule, "ok"


class ChemicalRuleBank:
    """Versioned repository of executable, falsifiable LLM-discovered chemical rules."""

    def __init__(self, rules: list[ChemicalRule] | None = None):
        self.rules: dict[str, ChemicalRule] = {}
        self._sig_to_id: dict[tuple[str, str, str, str], str] = {}
        self.history: list[dict[str, Any]] = []
        for r in rules or []:
            self.rules[r.rule_id] = r
            self._sig_to_id[r.signature()] = r.rule_id

    def active_rules(self) -> list[ChemicalRule]:
        return [r for r in self.rules.values() if r.status != "falsified"]

    def evaluate_vector(self, pair: np.ndarray, rules: list[ChemicalRule] | None = None) -> np.ndarray:
        use_rules = self.active_rules() if rules is None else rules
        if not use_rules:
            return np.zeros(0, dtype=float)
        return np.asarray([evaluate_rule(r, pair) for r in use_rules], dtype=float)

    def ingest_proposals(
        self,
        answer: dict[str, Any],
        generation: int,
        allowed_refs: set[str] | list[str] | tuple[str, ...],
        sample_pairs: list[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        accepted_ids: list[str] = []
        revised_ids: list[str] = []
        falsified_ids: list[str] = []
        rejected: list[dict[str, str]] = []

        if not isinstance(answer, dict):
            return {"accepted": [], "revised": [], "falsified": [], "rejected": [{"error": "Invalid rule answer envelope"}]}

        for prop in answer.get("new_rules") or []:
            ok, rule, reason = validate_rule_proposal(prop, allowed_refs, sample_pairs=sample_pairs)
            if not ok or rule is None:
                rejected.append({"name": str((prop or {}).get("name", "")), "error": reason})
                continue
            sig = rule.signature()
            if sig in self._sig_to_id:
                existing = self.rules[self._sig_to_id[sig]]
                if existing.status == "falsified":
                    rejected.append({"name": rule.name, "error": "Matches previously falsified predicate"})
                    continue
                for doi in rule.supporting_dois:
                    if doi not in existing.supporting_dois:
                        existing.supporting_dois.append(doi)
                existing.chemical_statement = rule.chemical_statement
                existing.boundary_conditions = rule.boundary_conditions
                existing.status = "revised"
                revised_ids.append(existing.rule_id)
            else:
                rule.generation_created = int(generation)
                self.rules[rule.rule_id] = rule
                self._sig_to_id[sig] = rule.rule_id
                accepted_ids.append(rule.rule_id)

        norm_allowed = {_normalize_ref(r) for r in allowed_refs if _normalize_ref(r)}
        for rev in answer.get("revise_rules") or []:
            if not isinstance(rev, dict):
                continue
            rid = str(rev.get("rule_id") or "").strip()
            target = self.rules.get(rid)
            if target is None or target.status == "falsified":
                continue
            if str(rev.get("chemical_statement") or "").strip():
                target.chemical_statement = str(rev["chemical_statement"]).strip()[:320]
            if str(rev.get("boundary_conditions") or "").strip():
                target.boundary_conditions = str(rev["boundary_conditions"]).strip()[:180]
            for doi in rev.get("supporting_dois") or []:
                nr = _normalize_ref(str(doi))
                if nr in norm_allowed and nr not in target.supporting_dois:
                    target.supporting_dois.append(nr)
            target.status = "revised"
            revised_ids.append(rid)

        for fid in answer.get("falsify_rule_ids") or []:
            rid = str(fid or "").strip()
            target = self.rules.get(rid)
            if target is not None and target.status != "falsified":
                target.status = "falsified"
                falsified_ids.append(rid)

        event = {
            "generation": int(generation),
            "accepted": accepted_ids,
            "revised": revised_ids,
            "falsified": falsified_ids,
            "rejected": rejected,
        }
        self.history.append(event)
        return event

    def summary_for_prompt(self, limit: int = 8) -> list[dict[str, Any]]:
        rows = []
        for r in self.rules.values():
            rows.append({
                "rule_id": r.rule_id,
                "name": r.name,
                "chemical_statement": r.chemical_statement,
                "predicate_type": r.predicate_type,
                "expected_direction": r.expected_direction,
                "scope_tier": r.scope_tier,
                "status": r.status,
                "beta_points": round(float(r.beta_points), 3),
                "sd_points": round(float(r.sd_points), 3),
                "z_score": round(float(r.z_score), 2),
                "n_matched": int(r.n_matched_observed),
            })
        return rows[-limit:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": [asdict(r) for r in self.rules.values()],
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChemicalRuleBank":
        rules = [ChemicalRule(**r) for r in (payload or {}).get("rules", [])]
        bank = cls(rules)
        bank.history = list((payload or {}).get("history", []))
        return bank


class HybridChemicalSurrogate:
    """Joint Bayesian linear surrogate over [SOM_simplex_basis, executable_chemical_rule_basis]."""

    def __init__(
        self,
        som_prior_cov: np.ndarray,
        context: str,
        rule_prior_var: float = 0.04,
    ):
        self.som_prior_cov = np.asarray(som_prior_cov, dtype=float)
        if self.som_prior_cov.ndim != 2 or self.som_prior_cov.shape[0] != self.som_prior_cov.shape[1]:
            raise ValueError("som_prior_cov must be square")
        self.context = str(context)
        self.som_dim = int(self.som_prior_cov.shape[0])
        self.rule_prior_var = float(rule_prior_var)

        self.active_rules: list[ChemicalRule] = []
        self.mean = np.zeros(self.som_dim, dtype=float)
        self.mean[0] = 0.5
        self.cov = self.som_prior_cov.copy()

    def _build_joint_prior(self, rules: list[ChemicalRule]) -> tuple[np.ndarray, np.ndarray]:
        k_rules = len(rules)
        dim = self.som_dim + k_rules
        mu0 = np.zeros(dim, dtype=float)
        mu0[0] = 0.5
        cov0 = np.zeros((dim, dim), dtype=float)
        cov0[: self.som_dim, : self.som_dim] = self.som_prior_cov
        for idx, rule in enumerate(rules):
            mu0[self.som_dim + idx] = rule.prior_mean_01()
            cov0[self.som_dim + idx, self.som_dim + idx] = self.rule_prior_var
        return mu0, cov0

    def _solve_posterior(
        self,
        observations: list[dict[str, Any]],
        rules: list[ChemicalRule],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mu0, cov0 = self._build_joint_prior(rules)
        dim = len(mu0)
        cov0_jit = cov0 + 1e-10 * np.eye(dim, dtype=float)
        prec0 = np.linalg.inv(cov0_jit)
        prec0 = 0.5 * (prec0 + prec0.T)

        if not observations:
            return mu0, cov0, np.zeros((0, len(rules)), dtype=float)

        rows_phi = []
        ys = []
        inv_vars = []
        rule_cols = []
        for obs in observations:
            psi = np.asarray(obs["psi"], dtype=float).ravel()
            r_vec = (
                np.asarray([evaluate_rule(r, obs["pair"]) for r in rules], dtype=float)
                if rules
                else np.zeros(0, dtype=float)
            )
            rule_cols.append(r_vec)
            rows_phi.append(np.concatenate([psi, r_vec]))
            ys.append(float(obs["score_01"]))
            inv_vars.append(1.0 / max(float(obs.get("noise_var", 0.04)), 1e-4))

        phi = np.vstack(rows_phi)
        y_vec = np.asarray(ys, dtype=float)
        w_vec = np.asarray(inv_vars, dtype=float)

        # Exact Bayesian posterior: (Sigma0^-1 + Phi^T W Phi) mu = Sigma0^-1 mu0 + Phi^T W y
        weighted_phi = phi * w_vec[:, None]
        prec_post = prec0 + phi.T @ weighted_phi
        prec_post = 0.5 * (prec_post + prec_post.T)
        rhs = prec0 @ mu0 + weighted_phi.T @ y_vec
        mean_post = np.linalg.solve(prec_post, rhs)
        cov_post = np.linalg.inv(prec_post)
        cov_post = 0.5 * (cov_post + cov_post.T)
        rule_matrix = np.vstack(rule_cols) if rules else np.zeros((len(observations), 0), dtype=float)
        return mean_post, cov_post, rule_matrix

    def fit(
        self,
        observations: list[dict[str, Any]],
        rule_bank: ChemicalRuleBank,
        auto_falsify: bool = True,
    ) -> dict[str, Any]:
        """Fit joint Bayesian posterior and automatically confirm or falsify contradicted rules."""
        rules = rule_bank.active_rules()
        mean_post, cov_post, rule_mat = self._solve_posterior(observations, rules)

        newly_falsified: list[str] = []
        for idx, rule in enumerate(rules):
            w_idx = self.som_dim + idx
            beta_pts = float(mean_post[w_idx] * 100.0)
            sd_pts = float(math.sqrt(max(float(cov_post[w_idx, w_idx]), 0.0)) * 100.0)
            z = float(abs(beta_pts) / max(sd_pts, 1e-6))
            n_hit = int(np.sum(rule_mat[:, idx] > 0.5)) if rule_mat.shape[0] else 0
            rule.beta_points = round(beta_pts, 4)
            rule.sd_points = round(sd_pts, 4)
            rule.z_score = round(z, 3)
            rule.n_matched_observed = n_hit

            if auto_falsify and n_hit >= 4 and (beta_pts * rule.expected_sign() < 0.0) and z >= 1.5:
                rule.status = "falsified"
                newly_falsified.append(rule.rule_id)
            elif n_hit >= 2 and (beta_pts * rule.expected_sign() > 0.0) and rule.status == "tentative":
                rule.status = "supported"

        if newly_falsified:
            rules = rule_bank.active_rules()
            mean_post, cov_post, rule_mat = self._solve_posterior(observations, rules)
            for idx, rule in enumerate(rules):
                w_idx = self.som_dim + idx
                beta_pts = float(mean_post[w_idx] * 100.0)
                sd_pts = float(math.sqrt(max(float(cov_post[w_idx, w_idx]), 0.0)) * 100.0)
                rule.beta_points = round(beta_pts, 4)
                rule.sd_points = round(sd_pts, 4)
                rule.z_score = round(float(abs(beta_pts) / max(sd_pts, 1e-6)), 3)
                rule.n_matched_observed = int(np.sum(rule_mat[:, idx] > 0.5)) if rule_mat.shape[0] else 0

        self.active_rules = list(rules)
        self.mean = mean_post
        self.cov = cov_post
        return {
            "n_observations": len(observations),
            "active_rules": len(self.active_rules),
            "newly_falsified": newly_falsified,
        }

    def _feature_vector(self, psi_som: np.ndarray, pair: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi = np.asarray(psi_som, dtype=float).ravel()
        r_vec = (
            np.asarray([evaluate_rule(r, pair) for r in self.active_rules], dtype=float)
            if self.active_rules
            else np.zeros(0, dtype=float)
        )
        return np.concatenate([psi, r_vec]), r_vec

    def predict(self, psi_som: np.ndarray, pair: np.ndarray) -> tuple[float, float]:
        feat, _ = self._feature_vector(psi_som, pair)
        mean_val = float(feat @ self.mean)
        var_val = max(float(feat @ self.cov @ feat), 0.0)
        return mean_val, var_val

    def explain(self, psi_som: np.ndarray, pair: np.ndarray) -> dict[str, Any]:
        """Return exact additive decomposition: total_points == som_base_points + rule_delta_points."""
        psi = np.asarray(psi_som, dtype=float).ravel()
        feat, r_vec = self._feature_vector(psi, pair)
        som_base_pts = float((psi @ self.mean[: self.som_dim]) * 100.0)
        matched = []
        rule_delta_pts = 0.0
        for idx, rule in enumerate(self.active_rules):
            if r_vec[idx] > 0.5:
                contrib = float(self.mean[self.som_dim + idx] * 100.0)
                rule_delta_pts += contrib
                matched.append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "scope_tier": rule.scope_tier,
                    "status": rule.status,
                    "contribution_points": round(contrib, 4),
                    "chemical_statement": rule.chemical_statement,
                })
        total_pts = som_base_pts + rule_delta_pts
        total_sd_pts = float(math.sqrt(max(float(feat @ self.cov @ feat), 0.0)) * 100.0)
        return {
            "som_base_points": round(som_base_pts, 4),
            "rule_delta_points": round(rule_delta_pts, 4),
            "total_points": round(total_pts, 4),
            "total_sd_points": round(total_sd_pts, 4),
            "matched_rules": matched,
        }


def build_rule_induction_digest(
    generation: int,
    observations: list[dict[str, Any]],
    surrogate: HybridChemicalSurrogate,
    rule_bank: ChemicalRuleBank,
    new_claims: list[dict[str, Any]],
    mmp_pairs: list[dict[str, Any]],
) -> tuple[dict[str, Any], set[str]]:
    """Construct leakage-free run digest and authorized reference set for LLM rule induction."""
    allowed_refs: set[str] = set()
    claim_digest = []
    for row in new_claims[-8:]:
        sid = _normalize_ref(str(row.get("source_id") or row.get("reference") or ""))
        if sid:
            allowed_refs.add(sid)
        claim_digest.append({
            "ref": sid,
            "unit_name": row.get("unit_name", ""),
            "kind": row.get("kind", ""),
            "state": row.get("state", ""),
            "stream": row.get("stream", "local"),
            "scope_tier": row.get("scope_tier"),
            "observation": row.get("observation", ""),
        })

    residuals = []
    for obs in observations:
        pred01, _ = surrogate.predict(obs["psi"], obs["pair"])
        actual_pts = float(obs["score_01"] * 100.0)
        pred_pts = float(pred01 * 100.0)
        residuals.append({
            "uid": obs["uid"],
            "lean_smiles": obs.get("lean_smiles", ""),
            "actual_score": round(actual_pts, 3),
            "pred_score": round(pred_pts, 3),
            "residual_points": round(actual_pts - pred_pts, 3),
            "sa_score": obs.get("sa_score"),
            "h2_wt_percent": obs.get("h2_wt_percent"),
        })
        allowed_refs.add(f"obs:{str(obs['uid'])[:12]}")
    residuals.sort(key=lambda r: -abs(r["residual_points"]))

    for mp in mmp_pairs[:5]:
        if mp.get("child_uid"):
            allowed_refs.add(f"obs:{str(mp['child_uid'])[:12]}")

    digest = {
        "generation": int(generation),
        "allowed_refs": sorted(allowed_refs),
        "new_literature_claims": claim_digest,
        "high_residual_surprises": residuals[:5],
        "matched_molecular_pairs": mmp_pairs[:5],
        "existing_rules": rule_bank.summary_for_prompt(limit=8),
    }
    return digest, allowed_refs


def induce_rules_from_generation(
    gateway,
    stage: str,
    generation: int,
    observations: list[dict[str, Any]],
    surrogate: HybridChemicalSurrogate,
    rule_bank: ChemicalRuleBank,
    new_claims: list[dict[str, Any]],
    mmp_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Call LLM once per generation to induce, revise, or falsify explicit chemical rules."""
    if gateway is None:
        return {"status": "skipped_no_gateway", "usage": {}, "event": {}}
    if getattr(gateway, "quota_blocked", False) or getattr(gateway, "calls", 0) >= getattr(gateway, "limit", 10**9):
        return {"status": "skipped_budget", "usage": {}, "event": {}}

    digest, allowed_refs = build_rule_induction_digest(
        generation=generation,
        observations=observations,
        surrogate=surrogate,
        rule_bank=rule_bank,
        new_claims=new_claims,
        mmp_pairs=mmp_pairs,
    )
    if not allowed_refs:
        return {"status": "skipped_no_refs", "usage": {}, "event": {}}

    answer, audit = gateway.call(
        "deepseek-flash",
        "t05",
        f"{stage}_rule_induction",
        RULE_INDUCTION_TASK,
        digest,
        2400,
    )
    status = audit.get("status", "error")
    event = {}
    if status == "ok":
        sample_pairs = [obs["pair"] for obs in observations]
        event = rule_bank.ingest_proposals(
            answer=answer,
            generation=generation,
            allowed_refs=allowed_refs,
            sample_pairs=sample_pairs,
        )
    return {
        "status": status,
        "usage": audit.get("usage") or {},
        "raw_answer": answer if status == "ok" else None,
        "event": event,
    }
