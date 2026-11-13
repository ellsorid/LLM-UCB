"""Fixed shared structure coordinates for independently assessed GA parents."""

import csv
import gzip
from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA


class StructuralProjection:
    def __init__(self, baseline):
        self.baseline = baseline
        self.model = None
        self.reference_size = 0
        self.fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    @lru_cache(maxsize=4096)
    def features(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        bits = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(self.fingerprint.GetFingerprint(mol), bits)
        return bits

    def fit(self):
        if self.model is not None:
            return
        with gzip.open(self.baseline / "initial_population.csv.gz", "rt", encoding="utf-8") as stream:
            reference = sorted({r["smi"] for r in csv.DictReader(stream)})
        features = [bits for smi in reference if (bits := self.features(smi)) is not None]
        if len(features) < 2:
            raise ValueError("The initial population does not support a two-axis projection")
        self.reference_size = len(features)
        self.model = PCA(n_components=2, svd_solver="randomized", random_state=11).fit(np.asarray(features))

    @lru_cache(maxsize=4096)
    def position(self, smiles):
        bits = self.features(smiles)
        if bits is None:
            return None
        return self.model.transform(bits.reshape(1, -1))[0].tolist()
