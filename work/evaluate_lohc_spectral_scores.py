"""Molecular graph spectra used by the matrix score scale."""
import numpy as np
from rdkit import Chem


SCALES = np.geomspace(.1, 16, 8)


QUANTILES = np.linspace(0, 1, 9)


def spectrum(molecule, weighted):
    adjacency = Chem.GetAdjacencyMatrix(molecule, useBO=weighted)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.divide(1, np.sqrt(degree), out=np.zeros_like(degree, dtype=float), where=degree > 0)
    laplacian = np.diag((degree > 0).astype(float)) - adjacency * np.outer(inv_sqrt, inv_sqrt)
    values = np.linalg.eigvalsh(laplacian).clip(0, 2)
    # Quantiles and normalized heat traces give fixed-length vectors across graph sizes.
    return np.r_[len(values), np.quantile(values, QUANTILES), np.exp(-np.outer(SCALES, values)).mean(axis=1)]


def spectral_features(pair, weighted):
    lean, rich = (spectrum(molecule, weighted) for molecule in pair)
    return np.r_[lean, rich, rich[1:] - lean[1:]]
