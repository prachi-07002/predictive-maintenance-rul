# Predictive Maintenance — Remaining Useful Life (RUL)

Predicting how many cycles a machine has left before failure, from its
sensor history — built **from scratch** in pure Python + NumPy, no
sklearn, no scipy, no network calls. The setup mirrors real predictive
maintenance for industrial robots / smart manufacturing: synthetic
run-to-failure data (vibration, temperature, motor current, kurtosis,
oil pressure) that degrades convexly — slow early wear, accelerating
near failure — with per-unit calibration offsets and measurement noise,
so the model has to generalise to machines it has never seen.

## What's inside

| Module | What it implements |
|---|---|
| `rul/data.py` | Synthetic run-to-failure generator (60 units, lifetimes 181–416 cycles), **split by unit** (never by row — see below), snapshot builder (20-cycle trailing window, every 10 cycles, RUL ≤ 200) |
| `rul/features.py` | Window feature engineering: per sensor — last, mean, std, linear slope, max, delta — plus current cycle (31 features). Slope/delta features capture *how fast* a machine is degrading, not just how degraded it is |
| `rul/models.py` | **Baseline:** linear extrapolation of vibration RMS to the failure threshold (3.0). **RidgeRUL:** ridge regression, closed-form NumPy solve, train-only standardisation. **GradientBoostingRUL:** gradient-boosted regression trees from scratch — shallow CART trees with the exact best split found via sorted cumulative sums, fitted sequentially to residuals |
| `rul/evaluate.py` | RMSE, MAE, R², fraction within ±10 / ±20 cycles |
| `experiment.py` | End-to-end run: generate → split by unit → train all three → compare, plus a trace of one held-out machine through its life |

## Quickstart

```bash
pip install -r requirements.txt   # numpy only
python3 experiment.py             # full experiment, ~6 s
python3 -m unittest discover -s tests -v
```

## Results (measured, not estimated)

Setup: 60 units, 42 train / 18 test **split by unit**, 826 train / 360
test snapshots, 31 features, measured on this machine with
`python3 experiment.py`.

| model | RMSE (cycles) | MAE (cycles) | R² | within ±10 | within ±20 |
|---|---:|---:|---:|---:|---:|
| Baseline (linear trend) | 71.86 | 51.97 | −0.550 | 18.3% | 33.3% |
| Ridge regression | **18.47** | 14.11 | **0.898** | 44.4% | 75.3% |
| Gradient boosting (from scratch) | 18.65 | **13.39** | 0.896 | **53.1%** | 74.4% |

- **Learned models cut baseline RMSE by ~74%** (71.86 → 18.47 cycles).
- The baseline is worse than predicting the mean (R² = −0.55): wear is
  convex, so a straight line fitted to recent wear systematically
  mis-times the failure threshold, and early in life the slope is near
  zero so the extrapolation is wild.
- Ridge and GBM land in the same place (RMSE ≈ 18.5) — the signal is
  mostly in the engineered window features, and the remaining error is
  dominated by sensor noise and per-unit variation, not model capacity.
  GBM's edge is in the tails: 53% of its predictions land within ±10
  cycles vs 44% for ridge.
- Example held-out unit (#5, lifetime 380 cycles), GBM: at cycle 260
  true RUL 120 → predicted 120.7; at cycle 340 true RUL 40 →
  predicted 36.1. Predictions tighten as failure approaches, which is
  exactly when the decision matters.

## Why split by unit?

Splitting snapshots randomly would leak: snapshots from the same machine
would appear in both train and test, and the model could just memorise
that machine's calibration offsets. Splitting by unit tests whether the
model generalises to a machine it has never seen — the real deployment
question for a factory floor with new robots coming online.

## Resume bullets

- Built a remaining-useful-life (RUL) prediction system from scratch in Python + NumPy for synthetic run-to-failure sensor data (60 machines, 5 degrading sensors), with no ML libraries
- Engineered 31 trailing-window features (per-sensor level, trend slope, and change) and evaluated with a strict split-by-unit protocol (42 train / 18 unseen test machines) to prevent leakage
- Cut baseline RMSE by 74% (71.9 → 18.5 cycles, R² 0.90) with ridge regression, and matched it with a gradient-boosted regression-tree ensemble implemented from scratch (exact best-split search via cumulative sums)
- Showed the hand-calculation baseline (linear vibration-trend extrapolation) is worse than predicting the mean (R² −0.55) because real wear accelerates convexly near failure

## Design notes

- **GBM split search respects the leaf minimum:** the first version
  picked the highest-gain split, which isolated a single extreme
  vibration spike — huge apparent gain, useless split — and the tree
  then refused to grow past depth 1. Enforcing `min_samples_leaf` in
  the split search itself fixed it.
- **Everything is text:** pure Python + NumPy, no binary assets, so it
  pushes to GitHub cleanly.

## Tests

13 unit tests (`tests/test_rul.py`, stdlib `unittest` only) covering
data reproducibility and degradation trends, no-overlap unit splits,
correct RUL labels, feature values on a known linear ramp, baseline
exactness on a linear ramp and its flat-trend fallback, ridge fitting
a known linear function, GBM beating a mean predictor, end-to-end
ridge and GBM both beating the baseline on held-out units, and metric
values on hand-computed examples:

```bash
python3 -m unittest discover -s tests -v   # 13/13 pass
```
