# Frozen single-lean-state 12-channel MatrixGIN

This directory contains the previously trained model and all five original checkpoints. It does not contain the older GNN2/PyG model. No checkpoint was retrained or edited for this release.

Input: float tensor `M[B,N,N,12]` and atom mask `[B,N]`. Channels0–6 are diagonal atom properties: atomic number, formal charge, total H, aromatic atom, isotope label, radical electrons, CIP category. Channels7–11 are symmetric bond properties: bond order, aromatic, conjugated, ring, stereo. Padding is zero with mask=0. The model consumes only the hydrogen-lean matrix; rich-state information is not passed to the network.

Architecture: hidden dimension128, three dense GIN layers. The output is normalized during training and must be denormalized using the selected checkpoint's own `target_std` and `target_mean`. The runtime defaults to fold1, not an ensemble. All five folds are provided for reproducibility and optional independent deployment choices.

```python
from pathlib import Path
import torch
from models.lean_matrix.model import MatrixGIN

checkpoint = torch.load(
    Path('models/lean_matrix/checkpoints/fold_1_best.pth'),
    map_location='cpu', weights_only=True,
)
model = MatrixGIN(hidden_dim=checkpoint['hidden_dim'],
                  num_layers=checkpoint['num_gnn_layers'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
with torch.inference_mode():
    delta_H = model(M, mask) * checkpoint['target_std'] + checkpoint['target_mean']
```

Target convention: dehydrogenation enthalpy, source dataset kJ/mol H2. The checkpoint produces an estimate, not an experimentally measured value. Generated molecules outside the original size/chemistry distribution require independent validation.

`training_summary.json` is the original record:9841 samples, five-fold grouped validation, reported OOF R²0.9689/MAE2.7174/RMSE4.8995. This split is not a strict scaffold extrapolation benchmark, and the validation fold also informed epoch selection. These values are preserved as historical training results, not a new heldout performance claim.

`checkpoint_manifest.json` records file sizes, SHA256, architecture and target normalization. Each checkpoint is896742 bytes; total4483710 bytes. Training/data-loading sources are preserved; the only model-source adaptation is package-relative configuration import with the original script-import fallback. The original training dataset is not bundled. Run `python models/lean_matrix/train_matrix_gnn.py --help` to inspect the original training CLI; this release did not execute it for training.
