"""Unit tests for predictive-maintenance-rul (stdlib unittest + NumPy)."""

import unittest

import numpy as np

from rul.data import generate_dataset, split_by_unit, make_snapshots
from rul.features import window_features, snapshots_to_xy, N_FEATURES
from rul.models import BaselinePredictor, RidgeRUL, GradientBoostingRUL
from rul.evaluate import rmse, mae, r2_score, within_tolerance, evaluate


class TestData(unittest.TestCase):
    def test_reproducible_and_shaped(self):
        a = generate_dataset(n_units=5, seed=7)
        b = generate_dataset(n_units=5, seed=7)
        self.assertEqual(len(a), 5)
        for u, v in zip(a, b):
            self.assertEqual(u["lifetime"], v["lifetime"])
            np.testing.assert_array_equal(u["sensors"], v["sensors"])
        self.assertEqual(a[0]["sensors"].shape, (a[0]["lifetime"] + 1, 5))

    def test_lifetimes_in_range_and_sensors_degrade(self):
        units = generate_dataset(n_units=20, seed=1)
        for u in units:
            self.assertTrue(180 <= u["lifetime"] <= 420)
        # Vibration: mean of last 20 cycles well above mean of first 20.
        for u in units:
            vib = u["sensors"][:, 0]
            self.assertGreater(vib[-20:].mean(), vib[:20].mean() + 1.0)
            # Temperature rises too.
            self.assertGreater(u["sensors"][-20:, 1].mean(),
                               u["sensors"][:20, 1].mean() + 5.0)

    def test_split_by_unit_has_no_overlap(self):
        units = generate_dataset(n_units=20, seed=3)
        train, test = split_by_unit(units, test_fraction=0.3, seed=3)
        train_ids = {u["unit_id"] for u in train}
        test_ids = {u["unit_id"] for u in test}
        self.assertEqual(len(train), 14)
        self.assertEqual(len(test), 6)
        self.assertFalse(train_ids & test_ids)
        self.assertEqual(train_ids | test_ids,
                         {u["unit_id"] for u in units})

    def test_snapshots_rul_is_correct(self):
        units = generate_dataset(n_units=3, seed=5)
        snaps = make_snapshots(units, window=20, step=10)
        self.assertTrue(len(snaps) > 0)
        for s in snaps:
            unit = units[s["unit_id"]]
            self.assertEqual(s["rul"], unit["lifetime"] - s["cycle"])
            self.assertEqual(s["window"].shape, (20, 5))
            self.assertTrue(0 < s["rul"] <= 200)


class TestFeatures(unittest.TestCase):
    def test_window_features_shape_and_values(self):
        # Linear ramp in every sensor: slope known, last/mean known.
        window = np.tile(np.arange(20, dtype=float)[:, None], (1, 5))
        feats = window_features(window, cycle=100)
        self.assertEqual(len(feats), N_FEATURES)
        # vib_rms (col 0): last=19, mean=9.5, slope=1, max=19, delta=19.
        np.testing.assert_allclose(feats[0:6], [19, 9.5, 5.766, 1.0, 19, 19],
                                   atol=0.01)
        self.assertEqual(feats[-1], 100.0)

    def test_snapshots_to_xy_shapes(self):
        units = generate_dataset(n_units=4, seed=9)
        snaps = make_snapshots(units)
        X, y = snapshots_to_xy(snaps)
        self.assertEqual(X.shape, (len(snaps), N_FEATURES))
        self.assertEqual(y.shape, (len(snaps),))
        self.assertTrue(np.all(y > 0))


class TestModels(unittest.TestCase):
    def _small_split(self, n_units=30, seed=11):
        units = generate_dataset(n_units=n_units, seed=seed)
        tr, te = split_by_unit(units, test_fraction=0.3, seed=seed)
        tr_s, te_s = make_snapshots(tr), make_snapshots(te)
        return tr_s, te_s, snapshots_to_xy(tr_s), snapshots_to_xy(te_s)

    def test_baseline_linear_ramp(self):
        # Perfectly linear vibration crossing threshold 3.0 at cycle 30:
        # at cycle 20 (window 0..20, vib 0..2) it should predict RUL 10.
        window = np.tile(np.linspace(0, 2, 21)[:, None], (1, 5))
        snap = {"window": window, "cycle": 20, "rul": 10.0}
        pred = BaselinePredictor(threshold=3.0).predict([snap])[0]
        self.assertAlmostEqual(pred, 10.0, delta=1.0)

    def test_baseline_flat_trend_falls_back_finite(self):
        window = np.ones((20, 5))  # no trend at all
        snap = {"window": window, "cycle": 50, "rul": 80.0}
        pred = BaselinePredictor().fit([snap]).predict([snap])[0]
        self.assertTrue(np.isfinite(pred))
        self.assertGreaterEqual(pred, 0.0)

    def test_ridge_fits_linear_data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 4))
        # Shifted +15 so every target is positive: RidgeRUL clips RUL
        # predictions at 0 (a negative remaining life is meaningless),
        # and that clip would otherwise distort this generic fit check.
        y = 3 * X[:, 0] - 2 * X[:, 1] + 15
        model = RidgeRUL(alpha=0.1).fit(X, y)
        self.assertLess(rmse(y, model.predict(X)), 0.5)

    def test_gbm_beats_mean_predictor(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 3))
        y = 10 * (X[:, 0] > 0) + 5 * X[:, 1] ** 2 + rng.normal(0, 0.5, 300)
        model = GradientBoostingRUL(n_estimators=30, learning_rate=0.2,
                                    max_depth=3).fit(X, y)
        mean_rmse = rmse(y, np.full_like(y, y.mean()))
        self.assertLess(rmse(y, model.predict(X)), 0.5 * mean_rmse)

    def test_end_to_end_ridge_beats_baseline(self):
        tr_s, te_s, (Xtr, ytr), (Xte, yte) = self._small_split()
        base = BaselinePredictor().fit(tr_s).predict(te_s)
        ridge = RidgeRUL(alpha=10.0).fit(Xtr, ytr).predict(Xte)
        self.assertLess(rmse(yte, ridge), rmse(yte, base))
        self.assertLess(rmse(yte, ridge), 25.0)

    def test_end_to_end_gbm_beats_baseline(self):
        tr_s, te_s, (Xtr, ytr), (Xte, yte) = self._small_split()
        base = BaselinePredictor().fit(tr_s).predict(te_s)
        gbm = GradientBoostingRUL(n_estimators=50, learning_rate=0.1,
                                  max_depth=3).fit(Xtr, ytr).predict(Xte)
        self.assertLess(rmse(yte, gbm), rmse(yte, base))


class TestEvaluate(unittest.TestCase):
    def test_metrics_on_known_values(self):
        y = np.array([100.0, 50.0, 0.0])
        p = np.array([90.0, 50.0, 10.0])
        self.assertAlmostEqual(rmse(y, p), np.sqrt(200 / 3), places=6)
        self.assertAlmostEqual(mae(y, p), 20 / 3, places=6)
        self.assertAlmostEqual(r2_score(y, y), 1.0)
        self.assertAlmostEqual(within_tolerance(y, p, tol=10.0), 1.0)
        self.assertAlmostEqual(within_tolerance(y, p, tol=5.0), 1 / 3)
        m = evaluate(y, p)
        self.assertEqual(set(m), {"rmse", "mae", "r2", "within_10", "within_20"})


if __name__ == "__main__":
    unittest.main()
