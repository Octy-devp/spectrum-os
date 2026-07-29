"""Role multigraph — the shared-node, role-labeled-edge DCA substrate (PLAN-23 §6.1).

A state has no intrinsic DCA type. Its role is relational: the same state S can
crystallize as Direction in one thread, act as Crisis in a second, persist as
Lag in a third, and remain an uncollapsed Alternative in a fourth. The ontology
of a DCA node is therefore a *role vector* ``<thread: role>``, and the DCA
structure is a multigraph — states are shared nodes, roles are edge labels —
not a single-typed tree.

This module provides the two things the rest of ``spectrum_os.quantum`` needs:

* :class:`RoleMultigraph` — a flat store of :class:`RoleAssignment` records
  (state, thread, role, weight), the mechanical ground truth extracted from
  real data sources.
* Loaders for the two real sources surveyed for PLAN-23 task F:

  1. ``knowledge-dca-edges.yaml`` (α₂, ECC ``index/data/``) — a true typed-edge
     multigraph. Each edge ``(A --[X->Y]--> B)`` assigns role ``X`` to the
     source state and role ``Y`` to the target state *within that edge's
     relational context*. The edge is the finest "thread" the data provides.
  2. ``dca-branch-tree.yaml`` (β₁, ECC ``index/warfare/data/fields/*/``) —
     per-faction time series of DCA quads (free-text crisis / lag /
     alternative / direction fields per round, with multiple branch entries
     per round = superposed candidate measurements). No typed edges; the
     "threads" within a (faction, round) state are the branch entries.

Honest limits of the extraction (see module functions for details):

* The knowledge edge file carries only 3 of the 12 possible ordered role
  pairs (alternative->direction, direction->crisis, crisis->lag), so the
  measurable role vocabulary is position-asymmetric: ``lag`` is severely
  under-measured (7 of 350 edges at survey time).
* The warfare quads map ``chosen_path`` -> ``alternative`` and
  ``strategic_intent`` -> ``direction``. This is an interpretation declared
  here, not a fact of the data.
* Placeholder values (``"(crisis empty)"``, blanks, ``"延續上回合 …"``
  carry-overs) are treated as *unmeasured* (weight 0, ``stale=True``), not
  as active roles — a carried-over field is the absence of a new measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ROLES: tuple[str, ...] = ("crisis", "lag", "alternative", "direction")

# Warfare dca-branch-tree field -> DCA role. Declared interpretation:
# chosen_path is the alternative-thread a faction actually enacted;
# strategic_intent is the direction it crystallized toward.
_WARFARE_KEY_TO_ROLE: dict[str, str] = {
    "crisis": "crisis",
    "lag": "lag",
    "alternative": "alternative",
    "chosen_path": "alternative",
    "direction": "direction",
    "strategic_intent": "direction",
    # "doctrines" is emergent doctrine, not a DCA role — deliberately unmapped.
}

# Placeholder markers observed in the three warfare trees: explicit "(X empty)"
# sentinels, and "延續上回合 …" ("carried over from previous round") which marks
# a field that was not re-measured this round.
_PLACEHOLDER_EXACT = {
    "(crisis empty)",
    "(lag empty)",
    "(alternative empty)",
    "(direction empty)",
    "(chosen_path empty)",
    "(strategic_intent empty)",
    "(doctrines empty)",
}
_CARRY_OVER_PREFIX = "延續上回合"


def field_mass(value: object) -> int:
    """Return the informative character mass of a warfare quad field.

    0 for absent / blank / placeholder / carried-over values.
    """
    if value is None:
        return 0
    text = str(value).strip()
    if not text or text in _PLACEHOLDER_EXACT or text.startswith(_CARRY_OVER_PREFIX):
        return 0
    return len(text)


def _text_key(value: object) -> str:
    """Whitespace-normalized text key for duplicate detection (§6.5)."""
    return " ".join(str(value).split())


def _entry_form(keys: set[str]) -> str:
    """Template form of a warfare quad entry from its populated keys.

    ``5key`` = chosen_path/strategic_intent family; ``4key`` = alternative/
    direction family. A round mixing forms is flagged ``mixed`` by the caller.
    """
    if keys & {"chosen_path", "strategic_intent"}:
        return "5key"
    if keys & {"alternative", "direction"}:
        return "4key"
    return "other"


@dataclass
class RoleAssignment:
    """One role observation of a state within one measurement context (thread)."""

    state_id: str
    thread: str          # measurement context: edge label or branch entry
    role: str            # one of ROLES
    weight: float = 1.0  # presence weight (0 if stale/unmeasured)
    mass: int = 0        # informative character mass (warfare fields; 0 elsewhere)
    stale: bool = False  # carried-over placeholder — recorded but unmeasured
    source: str = ""     # provenance tag (loader-defined)
    note: str = ""       # provenance flag, e.g. "dup_text:direction" — measured
                         # but a verbatim duplicate of another role field in the
                         # same entry (one measurement, not two; see §6.5)


@dataclass
class RoleMultigraph:
    """Flat store of role assignments over shared states.

    ``assignments`` maps state_id -> list of RoleAssignment. ``state_times``
    optionally maps state_id -> an orderable time coordinate (warfare rounds);
    it stays empty for time-less graphs such as the knowledge multigraph, in
    which case decoherence_watch honestly returns UNKNOWN.
    """

    assignments: dict[str, list[RoleAssignment]] = field(default_factory=dict)
    state_times: dict[str, float] = field(default_factory=dict)
    state_meta: dict[str, dict] = field(default_factory=dict)
    """Per-state provenance metadata (warfare trees): ``template`` (4key/5key/
    mixed/other key-form of the round's entries) and ``template_switch`` (True
    when the form changes vs the previous round of the same faction). Markers
    only — interpretation belongs to the API gate layer (PLAN-23 §6.5)."""

    def add(self, assignment: RoleAssignment) -> None:
        self.assignments.setdefault(assignment.state_id, []).append(assignment)

    def add_edge(self, from_id: str, to_id: str, relation: str,
                 weight: float = 1.0, source: str = "") -> None:
        """Add a typed edge: source plays role X, target plays role Y (``X->Y``)."""
        try:
            x, y = relation.split("->", 1)
        except ValueError:
            raise ValueError(f"relation must be 'X->Y', got {relation!r}")
        for role in (x, y):
            if role not in ROLES:
                raise ValueError(f"unknown role {role!r} in relation {relation!r}")
        thread = f"edge:{from_id}--{relation}-->{to_id}"
        self.add(RoleAssignment(from_id, thread, x, weight=weight, source=source))
        self.add(RoleAssignment(to_id, thread, y, weight=weight, source=source))

    def states(self) -> list[str]:
        return sorted(self.assignments)

    def state_assignments(self, state_id: str) -> list[RoleAssignment]:
        return self.assignments.get(state_id, [])

    # ------------------------------------------------------------------
    # Constructors from real data shapes (pre-parsed dicts — no I/O here)
    # ------------------------------------------------------------------

    @classmethod
    def from_knowledge_edges(cls, data: dict, source: str = "knowledge-dca-edges") -> "RoleMultigraph":
        """Build from the parsed content of ``knowledge-dca-edges.yaml``.

        Expected shape: ``{"edges": [{"from", "to", "relation", "strength", ...}]}``.
        Edges with malformed relations are skipped (counted in the returned
        graph's ``source``-tagged assignments absence), not fatal.
        """
        graph = cls()
        for edge in data.get("edges", []):
            frm, to = edge.get("from"), edge.get("to")
            relation = edge.get("relation", "")
            if not frm or not to or "->" not in relation:
                continue
            graph.add_edge(frm, to, relation,
                           weight=float(edge.get("strength", 1.0)),
                           source=source)
        return graph

    @classmethod
    def from_warfare_tree(cls, data: dict, system_id: str = "warfare") -> "RoleMultigraph":
        """Build from the parsed content of a warfare ``dca-branch-tree.yaml``.

        Expected shape: ``{"factions": {name: [{"round": int, "dca": {...}}]}}``.

        Produces one state per (faction, round) — ``"{faction}/r{round:02d}"`` —
        whose threads are the branch entries within that round (multiple
        entries per round = superposed candidate quads). Each populated quad
        field contributes one role observation; placeholders and carry-overs
        are recorded as stale (weight 0). ``state_times`` is filled with the
        round number, giving decoherence_watch its time axis.
        """
        graph = cls()
        for faction, entries in (data.get("factions") or {}).items():
            branch_counts: dict[int, int] = {}
            round_forms: dict[int, set[str]] = {}
            prev_form: str | None = None
            for entry in entries:
                rnd = entry.get("round")
                if rnd is None:
                    continue
                branch_idx = branch_counts.get(rnd, 0)
                branch_counts[rnd] = branch_idx + 1
                state_id = f"{faction}/r{int(rnd):02d}"
                thread = f"{state_id}/b{branch_idx}"
                graph.state_times[state_id] = float(rnd)

                seen_texts: dict[str, str] = {}  # per-entry: text_key -> first role
                populated: set[str] = set()
                for key, value in (entry.get("dca") or {}).items():
                    role = _WARFARE_KEY_TO_ROLE.get(key)
                    if role is None:
                        continue
                    mass = field_mass(value)
                    note = ""
                    if mass > 0:
                        populated.add(key)
                        tk = _text_key(value)
                        other = seen_texts.get(tk)
                        if other is not None and other != role:
                            # §6.5 artifact filter: two role fields with verbatim
                            # identical text are ONE measurement, not two roles.
                            mass = 0
                            note = f"dup_text:{other}"
                        else:
                            seen_texts.setdefault(tk, role)
                    graph.add(RoleAssignment(
                        state_id, thread, role,
                        weight=1.0 if mass > 0 else 0.0,
                        mass=mass,
                        stale=(mass == 0 and not note),
                        note=note,
                        source=system_id,
                    ))
                if populated:
                    round_forms.setdefault(int(rnd), set()).add(_entry_form(populated))

            # Template-form markers (mechanical flags; interpretation is API-gate work)
            for rnd in sorted(round_forms):
                forms = round_forms[rnd]
                form = forms.pop() if len(forms) == 1 else "mixed"
                state_id = f"{faction}/r{rnd:02d}"
                meta = graph.state_meta.setdefault(state_id, {})
                meta["template"] = form
                if prev_form is not None and form != prev_form:
                    meta["template_switch"] = True
                prev_form = form
        return graph


def load_knowledge_edges(path: str) -> RoleMultigraph:
    """Load a ``knowledge-dca-edges.yaml`` file (requires PyYAML, I/O only)."""
    import yaml  # local import: computation core stays dependency-free

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return RoleMultigraph.from_knowledge_edges(data)


def load_warfare_tree(path: str, system_id: str | None = None) -> RoleMultigraph:
    """Load a warfare ``dca-branch-tree.yaml`` file (requires PyYAML, I/O only)."""
    import yaml

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if system_id is None:
        # e.g. ".../fields/experiment-track0/dca-branch-tree.yaml" -> "experiment-track0"
        system_id = path.rstrip("/").split("/")[-2] if "/" in path else path
    return RoleMultigraph.from_warfare_tree(data, system_id=system_id)
