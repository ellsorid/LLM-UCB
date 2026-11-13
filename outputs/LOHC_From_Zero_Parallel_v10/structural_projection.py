"""Fixed shared structure coordinates for independently assessed GA parents."""

import csv
import gzip
from functools import lru_cache
from threading import Lock

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA

from dashboard import ARMS, read_json


class StructuralProjection:
    def __init__(self, run):
        self.run = run
        self.lock = Lock()
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
        with gzip.open(self.run / "native" / "initial_population.csv.gz", "rt", encoding="utf-8") as stream:
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

    def snapshot(self):
        evaluation = read_json(self.run / "ac_evaluation.json", {}) or {}
        stages = evaluation.get("stages", {})
        status = {arm: read_json(self.run / arm / "status.json", {}) for arm in ARMS}
        if not stages:
            return {"stages": {}, "statuses": status}
        with self.lock:
            self.fit()
            result = {}
            for cap, stage in stages.items():
                groups = {}
                for arm, group in stage["arms"].items():
                    samples = []
                    for entry in group["sample"]:
                        candidate = entry["candidate"]
                        coords = self.position(candidate["pair"][0])
                        if coords is None:
                            continue
                        value = entry["score"]
                        physical = 25 * candidate["quality"]
                        interval = [value["low"], value["high"]] if value else [physical, physical + 75]
                        samples.append({"smiles": candidate["pair"][0], "scaffold": candidate["scaffold"],
                                        "cell": candidate["cell"], "xy": coords, "bounds": interval,
                                        "value": value["mid"] if value and value["usable"] else None,
                                        "high": bool(value and value["high_f"]), "reviewed": value is not None})
                    groups[arm] = {**{k: v for k, v in group.items() if k != "sample"}, "points": samples}
                result[cap] = {"arms": groups, "cutoffs": stage["cutoffs"]}
            return {"stages": result, "statuses": status, "reference_size": self.reference_size,
                    "explained": self.model.explained_variance_ratio_.tolist(), "high_f": 60}
