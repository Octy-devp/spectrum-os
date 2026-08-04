"""Spectrum OS Universal Exchange Dataclasses & Validation Contracts (PLAN-23 §6.1).

Defines core schema contracts for role trajectories, DCA substrates, CLAD corpora,
and world-interface payloads (field specs, actor cards, situations, field logs).
All core modules remain clean of domain-specific paths or world-specific references.
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


@dataclass
class FieldSpec:
    """Field folder contract: declares which fields a worldline must populate.

    World-agnostic: 'worldline' is any registered world (alpha/beta/gamma...), and
    'actors' lists the actor ids present in the field, not any specific world's cast.
    """

    field_id: str
    worldline: str
    temporal_scope: dict
    spatial_scope: dict
    actors: list[str]
    params: dict = field(default_factory=dict)
    triggers: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("FieldSpec field_id cannot be empty")
        if not self.worldline:
            raise ValueError("FieldSpec worldline cannot be empty")
        if not self.actors:
            raise ValueError("FieldSpec actors cannot be empty")

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-safe) for pipeline/world consumption."""
        return {
            "field_id": self.field_id,
            "worldline": self.worldline,
            "temporal_scope": self.temporal_scope,
            "spatial_scope": self.spatial_scope,
            "actors": self.actors,
            "params": self.params,
            "triggers": self.triggers,
            "meta": self.meta,
        }


@dataclass
class ActorCard:
    """Actor card contract: internal tensions seed multiple storylines per actor.

    Carries inherited conditions and field coordinates so the actor is situated
    inside the field; internal_tensions must be non-empty to break single-thread
    generation (the N=5 width lock). World-agnostic by design.
    """

    actor_id: str
    name: str
    inherited_conditions: dict
    field_coordinates: dict
    internal_tensions: list[dict]
    temporal_states: dict
    # Optional R/C initial force ratio seeds (force-model-social-dynamics.md §7.2).
    # None = unset (backward compatible); when set they must be non-negative.
    revolutionary_force: float | None = None
    conservative_force: float | None = None
    # Optional latent (potential) revolutionary force P_R,₀ seed (§2.2b reservoir).
    # None = unset ≡ 0 (backward compatible); when set must be non-negative.
    latent_force: float | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.actor_id:
            raise ValueError("ActorCard actor_id cannot be empty")
        if not self.internal_tensions:
            raise ValueError("ActorCard internal_tensions cannot be empty")
        for label, val in (
            ("revolutionary_force", self.revolutionary_force),
            ("conservative_force", self.conservative_force),
            ("latent_force", self.latent_force),
        ):
            if val is not None and val < 0:
                raise ValueError(f"ActorCard {label} must be non-negative, got {val}")

    def to_dict(self) -> dict:
        """Serialize to a plain dict; nested values (internal_tensions etc.) pass through."""
        return {
            "actor_id": self.actor_id,
            "name": self.name,
            "inherited_conditions": self.inherited_conditions,
            "field_coordinates": self.field_coordinates,
            "internal_tensions": self.internal_tensions,
            "temporal_states": self.temporal_states,
            "revolutionary_force": self.revolutionary_force,
            "conservative_force": self.conservative_force,
            "latent_force": self.latent_force,
            "meta": self.meta,
        }


@dataclass
class SituationSpec:
    """Situation payload contract: digest + local texture + optional 6D vector + actors.

    'digest' is the world-supplied situation summary; 'local_texture' carries the
    world-specific situation detail; 'actors' maps actor_id -> ActorCard.
    vector_6d is optional. World-agnostic by design.
    """

    digest: str
    local_texture: dict
    vector_6d: dict | None = None
    actors: dict[str, ActorCard] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.digest:
            raise ValueError("SituationSpec digest cannot be empty")

    def _mapped_vector_6d(self) -> dict | None:
        """Map ``vector_6d`` to the pipeline's ``6d_vector`` shape (dict with d1..d6).

        Mapping rule (probe.py ``_extract_situation_vector`` 為準——它要求
        ``situation["6d_vector"]`` 為含 ``d1``/``d2``/``d3`` 的 dict)：
        - dict（已含 d1.. 鍵）→ 直接透傳（管線原生形狀）。
        - 6 元素 list ``[v0..v5]`` → ``{"d1": v0, ..., "d6": v5}``。
        - None → None（管線 ``_extract_situation_vector`` 回 None，行為不變）。
        - 其他形狀 → 原樣透傳（管線不消費，不會崩潰）。
        """
        vec = self.vector_6d
        if vec is None:
            return None
        if isinstance(vec, dict):
            return vec
        if isinstance(vec, (list, tuple)) and len(vec) == 6:
            return {
                "d1": vec[0],
                "d2": vec[1],
                "d3": vec[2],
                "d4": vec[3],
                "d5": vec[4],
                "d6": vec[5],
            }
        return vec

    def to_dict(self) -> dict:
        """Serialize to the pipeline-consumable plain dict (probe.py 期待形狀).

        Output keys: ``digest`` / ``local_texture`` / ``6d_vector``（鍵名對映自
        ``vector_6d``）/ ``actors``（{aid: ActorCard.to_dict()}）。可直接被
        ``_root_label`` / ``_extract_situation_vector`` / ``_extract_situation_labels``
        消費，且 ``json.dumps`` 安全（無 dataclass 崩潰）。
        """
        return {
            "digest": self.digest,
            "local_texture": self.local_texture,
            "6d_vector": self._mapped_vector_6d(),
            "actors": {aid: card.to_dict() for aid, card in self.actors.items()},
        }


@dataclass
class FieldLog:
    """Field history log: append-only record of each generation layer."""

    field_id: str
    entries: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("FieldLog field_id cannot be empty")

    def add_entry(self, layer: int, selected: dict, unselected: list[dict]) -> None:
        """Append a generation record to the log."""
        self.entries.append({"layer": layer, "selected": selected, "unselected": unselected})

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-safe)."""
        return {
            "field_id": self.field_id,
            "entries": self.entries,
            "meta": self.meta,
        }
