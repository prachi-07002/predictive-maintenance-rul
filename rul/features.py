"""Feature engineering: turn a trailing sensor window into one vector.

For each sensor in the window we compute six statistics -- last value,
mean, standard deviation, linear-trend slope, maximum, and change
(last - first) -- plus the current cycle number. The slope / change
features are what let a model see *how fast* a machine is degrading,
not just how degraded it already is; a machine at vib 2.0 and rising
fast has far less RUL left than one at vib 2.0 and flat.
"""

import numpy as np

from .data import N_SENSORS, SENSOR_NAMES

# 6 stats per sensor + 1 cycle feature.
N_FEATURES = N_SENSORS * 6 + 1

FEATURE_NAMES = []
for name in SENSOR_NAMES:
    FEATURE_NAMES += [
        f"{name}_last",
        f"{name}_mean",
        f"{name}_std",
        f"{name}_slope",
        f"{name}_max",
        f"{name}_delta",
    ]
FEATURE_NAMES.append("cycle")


def _slope(values):
    """Least-squares slope of values against 0..len-1 (per cycle)."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    denom = float(x @ x)
    if denom == 0:
        return 0.0
    return float(x @ (values - values.mean()) / denom)


def window_features(window, cycle):
    """Feature vector for one (window, N_SENSORS) history ending at ``cycle``."""
    window = np.asarray(window, dtype=float)
    feats = np.empty(N_FEATURES, dtype=float)
    for j in range(N_SENSORS):
        col = window[:, j]
        base = j * 6
        feats[base + 0] = col[-1]
        feats[base + 1] = col.mean()
        feats[base + 2] = col.std()
        feats[base + 3] = _slope(col)
        feats[base + 4] = col.max()
        feats[base + 5] = col[-1] - col[0]
    feats[-1] = float(cycle)
    return feats


def snapshots_to_xy(snapshots):
    """Stack snapshots into a feature matrix X and target vector y (RUL)."""
    X = np.stack(
        [window_features(s["window"], s["cycle"]) for s in snapshots]
    )
    y = np.array([s["rul"] for s in snapshots], dtype=float)
    return X, y
