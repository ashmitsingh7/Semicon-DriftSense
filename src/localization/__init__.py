"""
Localization Module
-------------------
Coarse-to-fine NCC localization with gated phase reranker.
"""

from .v1_localize import localize, _subpixel_peak, _peak_sharpness
from .rerankers.gated_reranker import (
    gated_rerank,
    localize_with_gated_rerank,
    phase_rerank_candidates,
    native_rerank_candidates,
    should_rerank,
)

__all__ = [
    "localize",
    "_subpixel_peak",
    "_peak_sharpness",
    "gated_rerank",
    "localize_with_gated_rerank",
    "phase_rerank_candidates",
    "native_rerank_candidates",
    "should_rerank",
]