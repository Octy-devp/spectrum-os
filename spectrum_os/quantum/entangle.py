"""State entanglement and intervention (PLAN-23 §6.3).

Two operations built on the :class:`~.multigraph.RoleMultigraph`:

* :func:`entangle` — find all threads (measurement contexts) that share a
  given state, the prerequisite for understanding which role assignments
  must cohere.
* :func:`intervene` — reassign a state's role in one thread and propagate
  the change to all entangled threads, collapsing the role vector to a
  single ``ASSERTED`` verdict.

A state has no intrinsic DCA type. Its role is defined relationally across
threads in the multigraph. When a human (or an API gate layer) intervenes
to assert a role in one thread, the entanglement network propagates that
assertion to every thread that shares the state — this is the operational
meaning of "quantum causality" in PLAN-23: the multigraph is the substrate
and intervention is the measurement.
"""

from __future__ import annotations

from .multigraph import RoleMultigraph


def entangle(state_id: str, multigraph: RoleMultigraph | None = None) -> list[str]:
    """Find all threads (measurement contexts) that share a given state.

    Args:
        state_id: The state identifier to search for.
        multigraph: A :class:`RoleMultigraph` to query. If ``None``, a
            fresh empty multigraph is created — ``entangle`` always returns
            ``[state_id]`` in that case (a state always entangles with at
            least itself).

    Returns:
        Sorted list of thread identifiers that contain this state. If no
        threads share the state (including when no multigraph is provided),
        returns ``[state_id]`` — a state always entangles with at least
        itself.
    """
    if multigraph is None:
        return [state_id]

    thread_index = multigraph.threads()
    # Collect all threads whose state-set contains state_id
    matching: set[str] = set()
    for thread_name, states in thread_index.items():
        if state_id in states:
            matching.add(thread_name)

    if not matching:
        return [state_id]
    return sorted(matching)


def intervene(state_id: str, thread: str, new_role: str,
              multigraph: RoleMultigraph | None = None) -> dict:
    """Reassign a state's role in a specific thread and propagate to entangled threads.

    The intervention collapses the state's role vector: after the
    reassignment, the state has the asserted role in *all* threads that
    share it (via entanglement). This is the write-side counterpart of
    ``entangle`` — measurement that forces coherence across the multigraph.

    Args:
        state_id: The state to reassign.
        thread: The thread in which the role is being reassigned (the
            intervention entry point).
        new_role: The new role to assign (one of ``crisis``, ``lag``,
            ``alternative``, ``direction``).
        multigraph: The :class:`RoleMultigraph` to operate on. If ``None``,
            a fresh empty multigraph is created.

    Returns:
        A dict with keys ``intervened`` (state_id), ``thread`` (entry
        thread), ``new_role``, ``propagated_to`` (list of ``(thread_id,
        old_role)`` tuples for each thread that previously held a different
        role for this state), and ``verdict`` (always ``"ASSERTED"``).
    """
    if multigraph is None:
        multigraph = RoleMultigraph()

    # 1. Find entangled threads and record current roles
    entangled = entangle(state_id, multigraph)
    propagated: list[tuple[str, str]] = []
    for t in entangled:
        for a in multigraph.state_assignments(state_id):
            if a.thread == t and a.role != new_role:
                propagated.append((t, a.role))

    # 2. Reassign the role in the specified thread
    multigraph.assign_role(thread, state_id, new_role)

    # 3. Propagate to all entangled threads (ensuring coherence)
    for t in entangled:
        if t != thread:
            multigraph.assign_role(t, state_id, new_role)

    return {
        "intervened": state_id,
        "thread": thread,
        "new_role": new_role,
        "propagated_to": propagated,
        "verdict": "ASSERTED",
    }
