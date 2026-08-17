"""Gated rerankers for Drift-Sense localization."""

from .phase_reranker import phase_reranker_gated, should_rerank as phase_should_rerank

# Native verifier has optional dependency on pattern_synth (for synthetic self-eval only)
# Import lazily to avoid import errors when pattern_synth is not available
try:
    from .native_verifier import native_reranker_gated, should_rerank as native_should_rerank
except ImportError:
    native_reranker_gated = None
    native_should_rerank = None

__all__ = [
    "phase_reranker_gated",
    "phase_should_rerank",
]
if native_reranker_gated is not None:
    __all__.extend(["native_reranker_gated", "native_should_rerank"])