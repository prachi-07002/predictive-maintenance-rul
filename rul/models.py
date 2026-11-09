"""RUL models: a linear-trend baseline and two learned models.

Baseline -- ``BaselinePredictor``: fit a straight line to vibration RMS
over the trailing window and extrapolate it to the failure threshold.
This is the classic "trend extrapolation" a maintenance engineer would
do by hand. It fails systematically here because real wear is convex:
a straight line fitted to recent (accelerating) wear underestimates how
fast the threshold will be reached... in the other direction, early in
life the slope is near zero and the extrapolation is wild, so the
baseline needs a fallback for flat/negative slopes.

Learned models -- both fit on engineered window features (see features.py):

* ``RidgeRUL``: ridge regression solved in closed form with NumPy
  (standardisation fitted on train only, intercept unpenalised).
* ``GradientBoostingRUL``: gradient-boosted regression trees built from
  scratch -- shallow CART trees (best split found exactly via sorted
  cumulative sums, squared-error loss), fitted sequentially to residuals
  with a learning rate. No sklearn anywhere.
"""

import numpy as np

from .data import VIB_FAILURE_THRESHOLD

# Fallback RUL (cycles) when the baseline trend is flat or falling: the
# mean RUL of the training snapshots is substituted at predict time by
# the caller via ``fit``; this constant is only the pre-fit default.
_DEFAULT_FALLBACK_RUL = 100.0


class BaselinePredictor:
    """Linear extrapolation of vibration RMS to the failure threshold."""

    def __init__(self, threshold=VIB_FAILURE_THRESHOLD):
        self.threshold = threshold
        self.fallback_rul_ = _DEFAULT_FALLBACK_RUL

    def fit(self, snapshots):
        # The fallback is the mean RUL seen in training -- the best
        # constant guess when the trend gives no information.
        if snapshots:
            self.fallback_rul_ = float(
                np.mean([s["rul"] for s in snapshots])
            )
        return self

    def predict_one(self, snapshot):
        window = snapshot["window"]
        vib = window[:, 0]  # vib_rms is sensor column 0
        n = len(vib)
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, vib, 1)
        if slope <= 1e-4:
            return self.fallback_rul_
        # Cycles (within the window) until vib hits the threshold.
        steps_to_fail = (self.threshold - intercept) / slope
        rul = steps_to_fail - (n - 1)
        return float(np.clip(rul, 0.0, 300.0))

    def predict(self, snapshots):
        return np.array([self.predict_one(s) for s in snapshots], dtype=float)


class RidgeRUL:
    """Ridge regression (closed form) on standardised window features."""

    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Xs = (X - self.mean_) / self.scale_
        n, d = Xs.shape
        # Augment with an intercept column, unpenalised (penalty[0] = 0).
        Xa = np.column_stack([np.ones(n), Xs])
        penalty = np.eye(d + 1) * self.alpha
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(Xa.T @ Xa + penalty, Xa.T @ y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.mean_) / self.scale_
        Xa = np.column_stack([np.ones(len(Xs)), Xs])
        return np.clip(Xa @ self.coef_, 0.0, None)


# ---------------------------------------------------------------------------
# Gradient-boosted regression trees, from scratch.
# ---------------------------------------------------------------------------

class _TreeNode:
    __slots__ = ("value", "feature", "threshold", "left", "right")

    def __init__(self, value, feature=None, threshold=None, left=None, right=None):
        self.value = value  # mean target in this node (leaf prediction)
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

    @property
    def is_leaf(self):
        return self.feature is None


def _best_split(X, y, min_samples_leaf=1):
    """Best (feature, threshold) minimising squared error, exactly.

    For each feature, sort by that feature and use cumulative sums of y
    and y**2 to score every split in O(n log n) instead of rescanning.
    Only splits leaving at least ``min_samples_leaf`` points on each
    side are considered -- without this, the search happily isolates a
    single extreme point (huge apparent gain, useless split) and the
    tree then refuses to grow at all. Returns (feature, threshold, gain)
    or None if no split helps.
    """
    n = len(y)
    total_sum = float(y.sum())
    total_sq = float((y * y).sum())
    total_sse = total_sq - total_sum * total_sum / n
    best = None  # (sse, feature, threshold)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="mergesort")
        xs = X[order, j]
        ys = y[order]
        # Only split between distinct values, with enough points each side.
        distinct = xs[1:] != xs[:-1]
        if not np.any(distinct):
            continue
        csum = np.cumsum(ys)
        csq = np.cumsum(ys * ys)
        # Split after index i (left = 0..i): consider only distinct cuts.
        idx = np.nonzero(distinct)[0]
        n_left = idx + 1
        n_right = n - n_left
        ok = (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
        if not np.any(ok):
            continue
        idx = idx[ok]
        n_left = n_left[ok]
        n_right = n_right[ok]
        s_left = csum[idx]
        sq_left = csq[idx]
        sse_left = sq_left - s_left * s_left / n_left
        s_right = total_sum - s_left
        sq_right = total_sq - sq_left
        sse_right = sq_right - s_right * s_right / n_right
        sse = sse_left + sse_right
        k = int(np.argmin(sse))
        if best is None or sse[k] < best[0]:
            thr = 0.5 * (xs[idx[k]] + xs[idx[k] + 1])
            best = (float(sse[k]), j, float(thr))
    if best is None or best[0] >= total_sse - 1e-9:
        return None
    return best[1], best[2], total_sse - best[0]


def _build_tree(X, y, depth, max_depth, min_samples_leaf):
    value = float(y.mean())
    if depth >= max_depth or len(y) < 2 * min_samples_leaf:
        return _TreeNode(value)
    split = _best_split(X, y, min_samples_leaf=min_samples_leaf)
    if split is None:
        return _TreeNode(value)
    feature, threshold, _gain = split
    mask = X[:, feature] <= threshold
    if mask.sum() < min_samples_leaf or (~mask).sum() < min_samples_leaf:
        return _TreeNode(value)  # safety net; _best_split already enforces this
    return _TreeNode(
        value,
        feature=feature,
        threshold=threshold,
        left=_build_tree(X[mask], y[mask], depth + 1, max_depth, min_samples_leaf),
        right=_build_tree(X[~mask], y[~mask], depth + 1, max_depth, min_samples_leaf),
    )


def _tree_predict(node, X):
    out = np.empty(len(X), dtype=float)
    # Iterative traversal, one node at a time over the whole batch.
    stack = [(node, np.arange(len(X)))]
    while stack:
        nd, idx = stack.pop()
        if nd.is_leaf or len(idx) == 0:
            out[idx] = nd.value
        else:
            mask = X[idx, nd.feature] <= nd.threshold
            stack.append((nd.left, idx[mask]))
            stack.append((nd.right, idx[~mask]))
    return out


class GradientBoostingRUL:
    """Gradient boosting with shallow regression trees (squared loss).

    Starts from the mean RUL, then each tree fits the current residuals
    and is added with ``learning_rate`` shrinkage. Deterministic: no
    subsampling, no randomness.
    """

    def __init__(self, n_estimators=80, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.init_ = float(y.mean())
        pred = np.full(len(y), self.init_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            residual = y - pred
            if float(residual @ residual) < 1e-9:
                break
            tree = _build_tree(
                X, residual, 0, self.max_depth, self.min_samples_leaf
            )
            update = _tree_predict(tree, X)
            if float(update @ update) < 1e-12:
                break
            pred = pred + self.learning_rate * update
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        pred = np.full(len(X), self.init_)
        for tree in self.trees_:
            pred = pred + self.learning_rate * _tree_predict(tree, X)
        return np.clip(pred, 0.0, None)
