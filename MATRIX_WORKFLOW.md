# Matrix generation and pretrained lean-state GNN

The new executable entry is `python -m matrix_pipeline`. Start with the [workflow and commands](matrix_pipeline/WORKFLOW.md), then the [pretrained model card](models/lean_matrix/MODEL_CARD.md).

The branch includes all five frozen single-lean-state12-channel MatrixGIN checkpoints, their normalization and hashes, training/data-loading sources, portable example seed matrices, frozen QD calibration and evidence scales. No model was retrained. Existing legacy workflow files remain unchanged.

V2 fixes four issues found in the initial integration: paired operators can change H2 stoichiometry; parent/operator budgets are balanced; seed and child LLM scores share one evaluation protocol; UCB includes pre-update archive state. See [validation results](reports/matrix_loop_v2_validation.json) and the [verification report](reports/MATRIX_V2_VERIFICATION.md).
