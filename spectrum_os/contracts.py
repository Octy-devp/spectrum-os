"""Spectrum OS Universal Exchange Dataclasses & Validation Contracts (PLAN-23 §6.1).

Defines core schema contracts for role trajectories, DCA substrates, and CLAD corpora.
All core modules remain clean of domain-specific paths or ECC references.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spectrum_os.quantum.multigraph import ROLES, RoleMultigraph


@dataclass
class RoleTrajectory:
    """Per-thread time-ordered role sequence (PLAN-23 §6.1)."""

    thread_id: str
    points: list[tuple[float | str, str]]  # list of (time, role)
    source_id: str = ""
    bias_flags: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("RoleTrajectory points sequence cannot be empty")

        prev_t: float | str | None = None
        for pt in self.points:
            if not isinstance(pt, (tuple, list)) or len(pt) != 2:
                raise ValueError(f"Invalid point entry {pt!r}, expected (time, role) tuple")
            t, role = pt[0], pt[1]
            if role not in ROLES:
                raise ValueError(f"Invalid role {role!r} in trajectory; allowed: {ROLES}")

            if prev_t is not None:
                # Check monotonic non-decreasing order
                try:
                    if t < prev_t:
                        raise ValueError(f"Non-monotonic time sequence: {t!r} < {prev_t!r}")
                except TypeError:
                    if str(t) < str(prev_t):
                        raise ValueError(f"Non-monotonic time sequence: {t!r} < {prev_t!r}")
            prev_t = t

    def role_sequence(self) -> list[str]:
        """Return the sequence of role strings in time order."""
        return [role for _, role in self.points]


@dataclass
class DCASubstrate:
    """State nodes + role-labeled edges + provenance multigraph substrate (PLAN-23 §6.1)."""

    nodes: dict[str, dict] = field(default_factory=dict)  # state_id -> {"mass": float, "meta": dict}
    edges: list[dict] = field(default_factory=list)      # list of edge dicts
    bias_flags: list[dict] = field(default_factory=list)
    source_id: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate relations and annotate orphan edges without deleting them
        for edge in self.edges:
            relation = edge.get("relation", "")
            if "->" not in relation:
                raise ValueError(f"Invalid relation format {relation!r}, expected 'X->Y'")
            try:
                x, y = relation.split("->", 1)
            except ValueError:
                raise ValueError(f"Invalid relation format {relation!r}")
            if x not in ROLES or y not in ROLES:
                raise ValueError(f"Unknown role in edge relation {relation!r}; allowed: {ROLES}")

            frm, to = edge.get("from"), edge.get("to")
            is_orphan = edge.get("orphan", False)
            if not frm or not to or frm not in self.nodes or to not in self.nodes:
                edge["orphan"] = True
            else:
                edge["orphan"] = bool(is_orphan)

    def to_multigraph(self) -> RoleMultigraph:
        """Convert substrate to a RoleMultigraph object."""
        graph = RoleMultigraph()
        for edge in self.edges:
            frm = str(edge.get("from", ""))
            to = str(edge.get("to", ""))
            relation = str(edge.get("relation", ""))
            weight = float(edge.get("strength", 1.0))
            source = str(edge.get("provenance", self.source_id))
            graph.add_edge(frm, to, relation, weight=weight, source=source)
        return graph

    def transition_type_counts(self) -> dict[str, int]:
        """Count edges by transition relation type."""
        counts: dict[str, int] = {}
        for edge in self.edges:
            rel = str(edge.get("relation", ""))
            counts[rel] = counts.get(rel, 0) + 1
        return counts


@dataclass
class CLADCorpus:
    """Pre-structured DCA content pool (seed corpus for generator gate)."""

    entries: list[dict] = field(default_factory=list)
    source_id: str = ""
    bias_flags: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen_ids: set[str] = set()
        required_roles = {"crisis", "lag", "alternative", "direction"}

        for idx, entry in enumerate(self.entries):
            entry_id = entry.get("id")
            if not entry_id:
                raise ValueError(f"Entry at index {idx} missing required 'id' field")
            if entry_id in seen_ids:
                raise ValueError(f"Duplicate entry id {entry_id!r} in CLADCorpus")
            seen_ids.add(entry_id)

            clad = entry.get("clad")
            if not isinstance(clad, dict):
                raise ValueError(f"Entry {entry_id!r} missing 'clad' dictionary")

            missing = required_roles - set(clad.keys())
            if missing:
                raise ValueError(f"Entry {entry_id!r} clad missing required roles: {missing}")
