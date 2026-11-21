# Matrix loop v2

The executable path uses a shared paired-matrix core, matrix QD, archive-state contextual UCB, and the existing frozen single-lean-state MatrixGIN. The original GA entry remains available. This is a bounded research validation runner with a defined operator library, not a migration of every legacy GA operator.

## Run from a clean clone

Install `requirements-matrix.txt`; select a PyTorch build suitable for the machine. CPU is supported. The repository contains eight example seed pairs, a frozen development atlas, entropy scales, six existing literature records and all five GNN checkpoints. No local `D:` project is required for the bundled mode.

```bash
python -m unittest matrix_pipeline.test_v2 -v
python -m matrix_pipeline --bundled-only --device cpu --rounds 2 --output runs/offline_example
python -m matrix_pipeline --bundled-only --rounds 3 --live --key-file '<credential file>' --output runs/live_example
```

Without `--live`, both seeds and offspring use the same explicitly neutral offline score; no LLM response is fabricated. With `--live`, every seed is evaluated by the same critic/prompt/evidence policy as offspring before competing in the archive. A three-round run normally uses two seed-review requests and six loop requests; `--concurrency-probe` adds two bounded requests. Every output directory must be new. Provider/JSON failures stop the run and are recorded; there is no silent offline fallback.

If the full audited development project is available, set `LOHC_PROJECT_ROOT` to its directory. The full-data mode reuses9812 paired matrices and verified cached features. `--calibration <saved calibration.pt>` freezes identical atlas/context scaling across runs. `--bench` requires the full external audit data; the bundled examples are not a9812-sample benchmark. Historical throughput helpers measure their declared kernels and do not establish v2 whole-system speedup.

## Representation

- Shared core: seven atom and five bond fields, two aligned states and an explicit atom map.
- GPU population: `[population,2,N,N,12]` plus `[population,N]` mask; archive members are tensor indices.
- GNN: only the lean state, with the checkpoint's own target normalization. No GNN training or enthalpy correction occurs.
- QD: frozen64-cell CVT on30 typed-ring, cross-channel-path and reaction-change coordinates. Its calibration came from9812 development pairs; it is not heldout evidence.
- UCB:45 structural mask/rank/projection fields plus three pre-update archive-state fields: empty-cell indicator, incumbent quality and occupied-cell fraction; an intercept makes49 coefficients. Structural entropy is not an epistemic uncertainty bonus.
- Memory:48 local role-entropy/JS fields plus structure30, filtered to the requested SP endpoint. LLM must still assess applicability. Existing memory is a six-record working library, not an independently validated evidence benchmark.

## Matrix edits and fairness corrections

Methyl/amino/hydroxyl substitutions are applied to both states. Paired vinyl/ethyl and ethynyl/ethyl sidechains add respectively1 and2 formal H2 equivalents while preserving the parent heavy-atom scaffold. Every edit is checked for topology, valence and expected H2 change. These are formal storage stoichiometries, not proof of kinetic reversibility or stability.

Parents are selected from all occupied cells by least-visited counts with stable identity-based ties. Candidate budget is balanced across parent/operator groups. Neither selection uses a cell-number prefix or the first96 enumerated actions. Under a fixed parent pool the regression test verifies all32 parents are considered within three12-parent rounds.

H2 capacity≥4.2 wt%, predicted enthalpy40–70 kJ/mol H2 and legacy SA≤4 gate elite admission. Seed and child SP scores use one critic protocol and frozen evidence hash. UCB uses actual admission/quality-improvement feedback and the context saved before archive mutation. It remains an empirical contextual selection rule; a regret bound or independent chemical-quality advantage is not claimed.

## Boundaries and artifacts

Chemistry validation, canonical identity and the frozen legacy SA heuristic stay at the CPU boundary; SA retains its original fingerprint calculation. Provider inference is external. Matrix edits, descriptors, GNN inference, population storage, QD and UCB numerical updates run on the selected device.

Artifacts include configuration, input hashes, calibration, seed evaluations, candidate validation, reviews, selected candidates/rewards, full shared-core molecules, archive/population/UCB tensors and API status/token audits. Credentials are read at runtime and not stored. Raw API logs and the original dataset are excluded from Git. Fixed-budget legacy comparisons and independent stability/mechanism validation remain separate research steps.
