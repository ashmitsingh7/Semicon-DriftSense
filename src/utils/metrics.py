"""Metrics utilities for evaluation."""

import numpy as np


def summarize_errors(errors):
    """Compute summary statistics for error array."""
    arr = np.array(errors)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def threshold_metrics(errors, thresholds=(1, 2, 3, 5, 10, 20)):
    """Fraction of errors within each threshold."""
    arr = np.array(errors)
    return {f"within_{t}px": float((arr <= t).mean()) for t in thresholds}