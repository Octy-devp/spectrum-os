"""Quantum causality layer (PLAN-23 §6) — role-vector tomography,
decoherence watching, and 6D state vector operations over DCA multigraphs.

numpy-only computation, zero API. Lives outside ``kernel/`` for now; wiring
``osc.quantum`` into the kernel OSC facade is a deliberate follow-up (task F
scope: additive only, kernel untouched).
"""

from .multigraph import (
    ROLES,
    RoleAssignment,
    RoleMultigraph,
    load_knowledge_edges,
    load_warfare_tree,
)
from .tomography import kl_divergence, role_entropy, tomography
from .decoherence import decoherence_watch
from .entangle import entangle, intervene
from .vector import (
    StateVector,
    project_ternary,
    vector_from_roles,
)
from .dca_grammar import (
    DCA_GRAMMAR,
    mutate_alternative,
    validate_alternative,
    validate_alternative_chain,
)
from .quarantine import (
    TEMPORAL_BLACKLIST,
    inspector_gate,
    mechanical_filter,
    quarantine_check,
)
from .markov import (
    absorption_probabilities,
    count_transitions,
    count_transitions_soft,
    entropy_rate,
    estimate_rate_matrix,
    hitting_times,
    sample_rate_matrix,
    smoothed_role_posteriors,
    spectral_gap,
    stationary_distribution,
    summarize_samples,
)

# Re-export ANCHORS_SYSTEM_PROMPT_V25 from synth layer — it is conceptually
# a synth-layer constant (LLM prompt), but the quantum UX surface (osc.quantum)
# is the primary entry point for role-transition operations that consume it.
from ..synth.anchors import ANCHORS_SYSTEM_PROMPT_V25

__all__ = [
    "ROLES",
    "RoleAssignment",
    "RoleMultigraph",
    "load_knowledge_edges",
    "load_warfare_tree",
    "kl_divergence",
    "role_entropy",
    "tomography",
    "decoherence_watch",
    "entangle",
    "intervene",
    "StateVector",
    "project_ternary",
    "vector_from_roles",
    "DCA_GRAMMAR",
    "validate_alternative",
    "validate_alternative_chain",
    "mutate_alternative",
    "ANCHORS_SYSTEM_PROMPT_V25",
    "TEMPORAL_BLACKLIST",
    "mechanical_filter",
    "inspector_gate",
    "quarantine_check",
    "count_transitions",
    "count_transitions_soft",
    "estimate_rate_matrix",
    "sample_rate_matrix",
    "hitting_times",
    "absorption_probabilities",
    "smoothed_role_posteriors",
    "stationary_distribution",
    "entropy_rate",
    "spectral_gap",
    "summarize_samples",
]

