"""
Rerankers Module
----------------
Gated rerankers for FinFET ambiguity resolution.
"""

from .gated_reranker import (
    gated_rerank,
    localize_with_gated_rerank,
    phase_rerank_candidates,
    native_rerank_candidates,
    should_rerank,
    build_nominal_template,
    prep_search,
)

# phase_reranker exports
from .phase_reranker import (
    phase_reranker_gated,
    should_rerank as phase_should_rerank,
    build_nominal_template as phase_build_nominal_template,
    prep_search as phase_prep_search,
)

# native_verifier exports
from .native_verifier import (
    native_reranker_gated,
    should_rerank as native_should_rerank,
)

__all__ = [
    "gated_rerank",
    "localize_with_gated_rerank",
    "phase_rerank_candidates",
    "native_rerank_candidates",
    "should_rerank",
    "build_nominal_template",
    "prep_search",
    "phase_reranker_gated",
    "phase_should_rerank",
    "phase_build_nominal_template",
    "phase_prep_search",
    "native_reranker_gated",
    "native_should_rerank",
]