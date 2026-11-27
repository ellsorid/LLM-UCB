# Dual-Route ChemMech-LLM Workflow (`dual_route/`)

This module implements the **SOM + Hierarchical UCB + Unsupervised Literature-Grounded LLM Chemical Mechanism Discovery** closed loop.

## Key Components

| Module | Path | Role |
| :--- | :--- | :--- |
| Main closed-loop runner | `run_dual_ten_generation.py` | Coordinates bootstrap, hierarchical PopulationUCB, SA penalty, heuristic parent protection, literature search, and chemical-rule induction |
| Chemical rule bank & hybrid surrogate | `LLM/overall_local_v1/chemical_rule_surrogate.py` | Induces executable RDKit SMARTS rules from residuals/MMPs/literature claims, solves joint SOM+Rule Bayesian posteriors, and auto-falsifies contradicted hypotheses |
| Tier 1–3 evidence finder | `LLM/overall_local_v1/evidence_finder.py` | Autonomous OpenAlex/CrossRef retrieval for LOHC, broad hydrogenation/dehydrogenation motifs, failure modes, and multi-unit coupling |
| Local & coupling scoring probe | `LLM/overall_local_v1/overall_local_probe.py`, `relaxed_evidence.py` | Grounds directional LLM judgements and applies `scope_tier` weights + `coupling_multiplier` |
| External chemical benchmark | `chemical_benchmark.py`, `benchmarks/chemical_benchmark_v1.json` | Zero-shot 29-molecule / 16-pair dual-layer evaluation (`classic_lohc` + `motif_and_functional_group`) |
| Population UCB & parent selection | `LLM/ucb_chemical_units_v1/population_ucb.py`, `property_feedback.py` | Coarse-to-fine operator allocation and protected heuristic seed retention (`select_next_parents(..., protected_uids=...)`) |
| Frozen SOM & Kalman map | `SOM/chemical_unit_som_v1/`, `LLM/frontier_loop/mapping.py` | 128-prototype chemical-unit SOM manifold and graph-Laplacian Bayesian prior |

## Key CLI Flags (`run_dual_ten_generation.py`)

- `--enable-evidence-search`: Enables online Tier 1–3 OpenAlex/CrossRef evidence retrieval during evaluation.
- `--enable-chemical-rules`: Activates per-generation LLM SMARTS rule induction (`ChemicalRuleBank`), joint `HybridChemicalSurrogate` fitting, automatic falsification, and `G*_CHEMICAL_RULES.json` logging.
- `--sa-penalty-weight 2.0 --sa-threshold 3.5`: Penalizes high-SA poly-alkyne/complex molecules above SA threshold.
- `--root-cooldown 4.0`: Rotates Track-2 surgical root molecules across generations.
- `--no-protect-directed-parent`: Disables guaranteed next-generation parent retention of the LLM heuristic seed (enabled by default).

## Offline Verification Suites

```bash
python dual_route/test_chemical_rule_surrogate.py
python dual_route/test_evidence_finder.py
python dual_route/test_closed_loop.py
```
