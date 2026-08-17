"""Geometry utilities."""

import numpy as np


def transform_point(x, y, M):
    """Apply a 2x3 affine matrix M to a point (x, y)."""
    vec = np.array([x, y, 1.0])
    tx = M[0, 0] * vec[0] + M[0, 1] * vec[1] + M[0, 2]
    ty = M[1, 0] * vec[0] + M[1, 1] * vec[1] + M[1, 2]
    return float(tx), float(ty)


def euclidean_error(pred_x, pred_y, gt_x, gt_y):
    """Euclidean distance between predicted and ground truth."""
    return np.hypot(pred_x - gt_x, pred_y - gt_y)