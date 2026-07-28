"""Quantum causality layer (PLAN-23 §6) — role-vector tomography and
decoherence watching over DCA multigraphs.

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
]
