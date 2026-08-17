"""
Dataset Generation Module
-------------------------
Synthetic DRAM/FinFET/mixed_logic image pair generation with tracked GT.
"""

from .pattern_synth import (
    synth_canvas,
    apply_edge_brightening,
    apply_sensor_noise,
    apply_geometric_degradation,
    transform_point,
)

from .build_dataset import main, make_pair

__all__ = [
    "synth_canvas",
    "apply_edge_brightening",
    "apply_sensor_noise",
    "apply_geometric_degradation",
    "transform_point",
    "main",
    "make_pair",
]