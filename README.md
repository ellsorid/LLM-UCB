# ChemMech-LLM

**ChemMech-LLM: An Unsupervised LLM Framework for Autonomous Chemical Mechanism Discovery and Grey-Box Surrogate Modeling**

ChemMech-LLM is a closed-loop chemical discovery framework that integrates an unsupervised literature-grounded scoring agent, a falsifiable chemical-rule induction engine, and a Self-Organizing Map (SOM) Bayesian surrogate with hierarchical UCB + LLM directed molecular editing. Rather than fitting black-box models only on fixed pre-labeled datasets, ChemMech-LLM actively retrieves real literature abstracts for unseen chemical motifs and functional groups, infers executable RDKit SMARTS structure–property rules from residuals and matched molecular pairs, and continuously validates or falsifies those hypotheses on empirical closed-loop observations.

---

## Core Architecture

1. **Dual-State Matrix & Chemistry-Defined Functional Units (`dual_route/`)**
   - Represents every candidate molecule as an atom-aligned lean/rich pair (`[2, n, n, 12]` tensor) and decomposes each molecule into four chemical unit classes: `functional_group`, `ring_or_conjugation`, `attachment`, and `reaction_change` (`net_hydrogenation_region`).
   - Enforces chemical validity, neutrality, hydrogen-release stoichiometry (`nH2`, `H2 wt%`), and RDKit synthetic accessibility (`SA_Score`) at the chemistry boundary.

2. **Unsupervised Tier 1–3 Literature Retrieval & Coupling Activation (`dual_route/LLM/overall_local_v1/evidence_finder.py`, `overall_local_probe.py`)**
   - Addresses high-uncertainty endpoints lacking dense pre-labeled data (`cycle_reversibility` and `storage_stability`) using a three-tier retrieval scope:
     - **Tier 1**: Liquid organic hydrogen carrier (LOHC) cycling and stability;
     - **Tier 2**: Broad reversible catalytic hydrogenation/dehydrogenation of corresponding rings and functional groups (e.g., ketones/alcohols, nitriles/amines, N/O-heterocycles);
     - **Tier 3**: Failure modes, side reactions (C–N/C–O cleavage, ring opening, terminal-alkyne/vinyl polymerization, coking), and multi-unit steric/electronic coupling (`coupling` stream activating `coupling_multiplier`).
   - Enforces verbatim abstract quote grounding, DOI verification, and deterministic causal filtering against generic boilerplate summaries.

3. **Falsifiable Chemical Rule Bank & Grey-Box Surrogate (`dual_route/LLM/overall_local_v1/chemical_rule_surrogate.py`)**
   - Models molecular property scores jointly over a frozen 128-prototype SOM Laplacian manifold basis and incrementally discovered executable SMARTS predicates:
     $$\hat{y}_t(m) = \mathbf{w}_{\text{SOM}}^\top \psi_{\text{SOM}}(m) + \sum_{k=1}^{K_t} \beta_k \cdot r_k(m)$$
   - Solves exact Bayesian linear posteriors for $\beta_k \pm \sigma_k$ and `z_score`, automatically promoting empirically confirmed rules (`supported`) and pruning hypotheses contradicted by generation feedback (`falsified`).

4. **Dual-Route Population Evolution (`dual_route/run_dual_ten_generation.py`)**
   - **Track 1 (Hierarchical `PopulationUCB`)**: Coarse-to-fine policy allocation across mutation, crossover, immigration, and balanced exploration.
   - **Track 2 (Molecule-Vision LLM Directed Editing)**: Evaluates 1- and 2-step legal molecular edits with `lean_smiles`, `sa_score`, `delta_sa`, `H2 wt%`, and SOM cell shifts, retains the heuristic parent into next-generation `PARENTS`, and feeds UCB refinement descendants.

---

## Verified Quantitative Results (6-Generation Controlled Ablation)

All three controlled runs start from the same 256-candidate pool (`bootstrap=24`, `6 × 20 = 120` offspring, `144` total molecules):

| Metric | `v1` Baseline | `v2` SA + Vision + Protected Parent | **`v3` Full ChemMech-LLM (Tier 1–3 + Coupling + Rules)** |
| :--- | :---: | :---: | :---: |
| Committed Generations / API Calls | `6` / `48` | `6` / `43` | `6` / **`60`** |
| Auto-Mined Literature Claims Added | `13` | `6` | **`42`** (`44` structural + `56` contextual) |
| Coupling-Supported Evaluations | `0` | `0` | **`116`** (`coupling_status="supported"`) |
| Heuristic Parent Retention / Offspring Lineage | `0/6` · `0` / `0` | `6/6` · `12` / `15` | **`6/6` · `15` direct / `25` descendants** |
| Unique Track-2 `design_root` Scaffolds | `4` | `5` | **`6`** |
| Mean `SA Score` (G1–G6) | `4.293` | `3.902` | **`3.673`** (`-0.620` vs `v1`) |
| Mean `H2 wt%` (G1–G6) | `4.586%` | `4.513%` | **`5.035%`** (`+0.449 wt%` vs `v1`) |
| Mean / Max Unpenalized Chemical Score (G1–G6) | `56.766` / `62.470` | `56.988` / `63.933` | **`78.728` / `85.968`** |

### Chemical Mechanism Discovery & Unseen-Region Benchmark (`chemical_benchmark_v1`)

Evaluated zero-shot on an external 29-molecule / 16-pair dual-layer chemical contrast benchmark ([dual_route/benchmarks/chemical_benchmark_v1.json](dual_route/benchmarks/chemical_benchmark_v1.json)), where **`24/29` benchmark molecules map to SOM cells never occupied by the `v3` run**:

- **`v3` Rule Bank Evolution (G1 → G6)**: LLM induced **`16` executable RDKit SMARTS rules** (`14` active, `9` empirically `supported` / `revised`, **`2` false hypotheses automatically `falsified`** by empirical Bayesian evidence including `n_adjacent_ch_activation_kinetic_bias` with `beta = -18.04 ± 5.23 pts, z = 3.45` across `n = 14` observations).
- **Zero-Shot Contrast Accuracy**:
  - **SOM-only Surrogate (G1–G6)**: plateaus at `81.2%` overall (`100%` Layer A classic LOHCs, `75.0%` Layer B motifs & functional groups, Layer B mean margin `+0.468` at G6).
  - **Hybrid SOM + Discovered Chemical Rules Surrogate**: climbs at G2 and holds **`93.8%` (`15/16`) overall** and **`91.7%` (`11/12`) on unseen Layer B motif/functional-group pairs** (Layer B mean margin `+0.818` at G6; overall margin `+1.678`), notably overturning SOM misclassifications on ortho-shielded N-heterocycles vs long-chain C–N dealkylation pairs.

---

## Repository Contents & Attached Experimental Data

- [`dual_route/`](dual_route/README.md): Core SOM–UCB–LLM implementation, autonomous evidence finder, chemical rule surrogate, external chemical benchmark (`dual_route/chemical_benchmark.py`), and offline test suites.
- [`experiments/`](experiments/): Clean, secret-free telemetry archives for all three ablation runs (`live_large_pop_v1/`, `live_large_pop_v2_sa_directed/`, `live_large_pop_v3_chem_rules/`) plus [`experiments/SUMMARY_V1_V2_V3.json`](experiments/SUMMARY_V1_V2_V3.json). Note: raw API logs (`api/`) are omitted for credential safety.
- [`visualization/`](visualization/): Publication-grade vector/raster scientific plotting scripts and outputs (`Fig2a_composed`, `Fig2_som_migration_relief`, `Fig2_score_gradient`, `Fig2_directed_saddle`, `Fig2b_ucb_cube`, `Fig2b_llm_directed`).
- [`matrix_pipeline/`](matrix_pipeline/) & [`models/`](models/): Paired-matrix chemistry compiler, bundled seed fixtures, and frozen lean-state GNN checkpoints.

---

## Verification & Usage

Run all 19 offline tests across the three closed-loop suites:

```bash
python dual_route/test_chemical_rule_surrogate.py
python dual_route/test_evidence_finder.py
python dual_route/test_closed_loop.py
```

Evaluate an archived experiment on the external dual-layer chemical benchmark:

```bash
python dual_route/chemical_benchmark.py --eval-run experiments/live_large_pop_v3_chem_rules
```

Run a full live 6-generation ChemMech-LLM experiment (requires local external data archive `--inputs` and DeepSeek API key file `--key`):

```bash
python dual_route/run_dual_ten_generation.py ^
  --archive dual_route ^
  --inputs "<local-data-root>" ^
  --out runs/chemmech_live ^
  --key "<path-to-api-key-file>" ^
  --max-generations 6 ^
  --bootstrap-size 24 ^
  --batch-size 20 ^
  --max-parents 16 ^
  --elite-quota 10 ^
  --native-ga-budget 32 ^
  --shortlist-size 48 ^
  --enable-evidence-search ^
  --enable-chemical-rules ^
  --sa-penalty-weight 2.0 ^
  --sa-threshold 3.5 ^
  --root-cooldown 4.0
```
