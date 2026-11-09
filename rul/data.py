"""Synthetic run-to-failure sensor data.

Each unit (machine) runs from cycle 0 until it fails at cycle ``lifetime``.
Five sensors are recorded every cycle. All of them degrade monotonically
in expectation as the unit approaches failure -- vibration and temperature
rise (convexly, as real wear accelerates near end of life), motor current
and kurtosis rise, oil pressure falls -- with per-unit calibration offsets
and measurement noise on top, so no two units look identical.

The ground-truth remaining useful life at cycle ``t`` is ``lifetime - t``.
"""

import numpy as np

SENSOR_NAMES = ["vib_rms", "temp_c", "current_a", "kurtosis", "oil_pressure"]
N_SENSORS = len(SENSOR_NAMES)

# Failure threshold used by the linear-trend baseline: the vibration level
# a healthy-to-failed unit crosses shortly before failure (see data.py).
VIB_FAILURE_THRESHOLD = 3.0


def _unit_trajectory(rng, lifetime):
    """One unit's full sensor history, shape (lifetime + 1, N_SENSORS)."""
    t = np.arange(lifetime + 1, dtype=float)
    p = t / lifetime  # fraction of life consumed, 0 .. 1

    # Per-unit calibration offsets (same machine, slightly different sensors).
    vib0 = rng.uniform(0.40, 0.70)
    temp0 = rng.uniform(36.0, 42.0)
    cur0 = rng.uniform(4.6, 5.4)
    kurt0 = rng.uniform(2.8, 3.2)
    press0 = rng.uniform(96.0, 104.0)

    # Degradation is convex in p: slow early wear, accelerating near failure.
    vib = vib0 + 0.4 * p + 2.6 * p ** 3 + rng.normal(0, 0.08, size=t.shape)
    temp = temp0 + 4.0 * p + 18.0 * p ** 3 + rng.normal(0, 0.7, size=t.shape)
    current = cur0 + 1.2 * p ** 2 + rng.normal(0, 0.15, size=t.shape)
    kurt = kurt0 + 1.5 * p ** 2 + rng.normal(0, 0.20, size=t.shape)
    pressure = press0 - 25.0 * p ** 2 + rng.normal(0, 1.0, size=t.shape)

    # Occasional vibration spikes (transient shocks, not degradation).
    spikes = rng.random(size=t.shape) < 0.01
    vib = vib + spikes * rng.uniform(0.3, 0.8, size=t.shape)

    return np.column_stack([vib, temp, current, kurt, pressure])


def generate_dataset(n_units=60, seed=42, min_life=180, max_life=420):
    """Generate run-to-failure histories for ``n_units`` machines.

    Returns a list of dicts, one per unit::
        {"unit_id": int, "lifetime": int,
         "cycles": np.ndarray (lifetime + 1,),
         "sensors": np.ndarray (lifetime + 1, N_SENSORS)}
    Lifetimes are uniform in [min_life, max_life]. Same seed -> same data.
    """
    rng = np.random.default_rng(seed)
    units = []
    for unit_id in range(n_units):
        lifetime = int(rng.integers(min_life, max_life + 1))
        units.append(
            {
                "unit_id": unit_id,
                "lifetime": lifetime,
                "cycles": np.arange(lifetime + 1, dtype=float),
                "sensors": _unit_trajectory(rng, lifetime),
            }
        )
    return units


def split_by_unit(units, test_fraction=0.3, seed=42):
    """Split units into train/test by unit id -- never by row.

    Splitting by row would leak: snapshots from the same machine would
    appear in both train and test, and the model could just memorise that
    machine. Splitting by unit tests whether the model generalises to a
    machine it has never seen, which is the real deployment question.
    """
    rng = np.random.default_rng(seed)
    ids = np.array([u["unit_id"] for u in units])
    perm = rng.permutation(ids)
    n_test = max(1, int(round(len(units) * test_fraction)))
    test_ids = set(perm[:n_test].tolist())
    train = [u for u in units if u["unit_id"] not in test_ids]
    test = [u for u in units if u["unit_id"] in test_ids]
    return train, test


def make_snapshots(units, window=20, step=10, min_cycle=20, max_rul=200):
    """Turn full histories into (features-window, RUL) prediction snapshots.

    For each unit, take a snapshot every ``step`` cycles starting at
    ``min_cycle`` (so a full ``window`` of history exists), while the true
    RUL is at most ``max_rul`` cycles -- far-from-failure snapshots carry
    almost no degradation signal and would dominate the training set.
    Each snapshot is ``{"unit_id", "cycle", "window", "rul"}`` where
    ``window`` is the (window, N_SENSORS) sensor history ending at ``cycle``.
    """
    snapshots = []
    for unit in units:
        lifetime = unit["lifetime"]
        sensors = unit["sensors"]
        for cycle in range(min_cycle, lifetime, step):
            rul = lifetime - cycle
            if rul > max_rul or rul <= 0:
                continue
            snapshots.append(
                {
                    "unit_id": unit["unit_id"],
                    "cycle": cycle,
                    "window": sensors[cycle - window + 1 : cycle + 1].copy(),
                    "rul": float(rul),
                }
            )
    return snapshots
