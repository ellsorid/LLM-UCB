import numpy as np

DH_LOW, DH_TARGET, DH_HIGH = 40.0, 55.0, 70.0
DH_EDGES = (40., 47.5, 55., 62.5, np.nextafter(70., np.inf))

def dh_quality(value, low=DH_LOW, target=DH_TARGET, high=DH_HIGH):
    """FINAL SI S23; target is retained for legacy callers and has no scoring role."""
    value = np.asarray(value, dtype=float)
    return np.clip(1 - (np.maximum(low-value, 0)/20)**2 - (np.maximum(value-high, 0)/35)**2, 0, 1)
