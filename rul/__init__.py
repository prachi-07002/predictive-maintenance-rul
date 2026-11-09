"""Predictive maintenance: remaining-useful-life (RUL) prediction.

Synthetic run-to-failure sensor data for industrial machines, plus a
linear-trend baseline and two learned models (ridge regression and a
from-scratch gradient-boosted regression-tree ensemble), all in NumPy.
"""

from .data import generate_dataset, split_by_unit, make_snapshots
from .features import window_features, SENSOR_NAMES, N_SENSORS
from .models import BaselinePredictor, RidgeRUL, GradientBoostingRUL
from .evaluate import rmse, mae, r2_score, within_tolerance, evaluate

__all__ = [
    "generate_dataset",
    "split_by_unit",
    "make_snapshots",
    "window_features",
    "SENSOR_NAMES",
    "N_SENSORS",
    "BaselinePredictor",
    "RidgeRUL",
    "GradientBoostingRUL",
    "rmse",
    "mae",
    "r2_score",
    "within_tolerance",
    "evaluate",
]
