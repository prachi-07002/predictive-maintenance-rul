"""End-to-end experiment: generate data, train, and compare RUL models.

Split is BY UNIT (42 train / 18 test units): the test machines are ones
the models have never seen. Three predictors are compared on the same
test snapshots:

* Baseline -- linear extrapolation of vibration RMS to the failure
  threshold (what a hand calculation would do).
* Ridge -- ridge regression on engineered window features.
* GBM -- gradient-boosted regression trees (from scratch) on the same
  features.

Run:  python3 experiment.py
"""

import time

import numpy as np

from rul.data import generate_dataset, split_by_unit, make_snapshots
from rul.features import snapshots_to_xy
from rul.models import BaselinePredictor, RidgeRUL, GradientBoostingRUL
from rul.evaluate import evaluate


def run(n_units=60, seed=42, verbose=True):
    t0 = time.time()
    units = generate_dataset(n_units=n_units, seed=seed)
    train_units, test_units = split_by_unit(units, test_fraction=0.3, seed=seed)
    train_snaps = make_snapshots(train_units)
    test_snaps = make_snapshots(test_units)
    X_train, y_train = snapshots_to_xy(train_snaps)
    X_test, y_test = snapshots_to_xy(test_snaps)

    results = {}

    # --- Baseline: needs raw windows, not feature vectors. ---
    baseline = BaselinePredictor().fit(train_snaps)
    results["Baseline (linear trend)"] = evaluate(
        y_test, baseline.predict(test_snaps)
    )

    # --- Ridge regression. ---
    ridge = RidgeRUL(alpha=10.0).fit(X_train, y_train)
    results["Ridge regression"] = evaluate(y_test, ridge.predict(X_test))

    # --- Gradient-boosted trees. ---
    gbm = GradientBoostingRUL(
        n_estimators=80, learning_rate=0.1, max_depth=3, min_samples_leaf=10
    ).fit(X_train, y_train)
    results["Gradient boosting (from scratch)"] = evaluate(
        y_test, gbm.predict(X_test)
    )

    if verbose:
        lifetimes = [u["lifetime"] for u in units]
        print("Predictive maintenance: remaining-useful-life prediction")
        print(f"  units: {len(units)} ({len(train_units)} train / "
              f"{len(test_units)} test, split BY UNIT)")
        print(f"  lifetimes: {min(lifetimes)}-{max(lifetimes)} cycles "
              f"(mean {np.mean(lifetimes):.0f})")
        print(f"  snapshots: {len(train_snaps)} train / {len(test_snaps)} test "
              f"(window=20 cycles, step=10, RUL <= 200)")
        print(f"  features: {X_train.shape[1]} per snapshot")
        print()
        print(f"  {'model':<34} {'RMSE':>7} {'MAE':>7} {'R^2':>6} "
              f"{'within +/-10':>12} {'within +/-20':>12}")
        for name, m in results.items():
            print(f"  {name:<34} {m['rmse']:7.2f} {m['mae']:7.2f} "
                  f"{m['r2']:6.3f} {m['within_10']:12.1%} {m['within_20']:12.1%}")
        print()
        base_rmse = results["Baseline (linear trend)"]["rmse"]
        for name in ("Ridge regression", "Gradient boosting (from scratch)"):
            cut = 100 * (1 - results[name]["rmse"] / base_rmse)
            print(f"  {name} cuts baseline RMSE by {cut:.1f}%")
        print()
        # One held-out machine, traced through its life.
        unit = test_units[0]
        snaps = make_snapshots([unit])
        X_u, y_u = snapshots_to_xy(snaps)
        pred_u = gbm.predict(X_u)
        print(f"  Example held-out unit #{unit['unit_id']} "
              f"(lifetime {unit['lifetime']} cycles), GBM predictions:")
        for s, yt, yp in list(zip(snaps, y_u, pred_u))[::4]:
            print(f"    cycle {s['cycle']:>3}: true RUL {yt:5.0f}  "
                  f"predicted {yp:6.1f}  (error {yp - yt:+6.1f})")
        print(f"\n  elapsed: {time.time() - t0:.1f}s")
    return results


if __name__ == "__main__":
    run()
