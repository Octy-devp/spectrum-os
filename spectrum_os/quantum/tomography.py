"""quantum.tomography — role distribution of a state across threads (PLAN-23 §6.3).

Enumerates every role a state plays across all measurement contexts (threads)
and collapses the role vector into an aggregate distribution:

* ``per_thread`` — the raw role vector ``<thread: role>`` (the ontology).
* ``role_distribution`` — ``{role: weight}``, computed with **equal thread
  weighting**: each thread's weights are normalised to sum 1, then averaged
  across threads. One measurement context = one vote; edge strength / field
  mass only apportions role weight *within* a thread. For knowledge edges
  (single-role threads) this reduces to a plain role-count share — the
  near-constant edge strength (0.9) in the current data would carry no
  extra information anyway.
* ``entropy`` / ``entropy_norm`` — Shannon entropy of the aggregate
  distribution, in nats and normalised by ``ln(len(ROLES))``.
* ``styx_shadow`` — PLAN-23 §6.3 Styx-shadow predicate: the Alternative role
  is alive (weight > 0) while the Direction role is zero. A suppressed
  possibility is not dead; it is "Direction has stayed zero".

Ternary verdict (data sufficiency — the honesty line of §6.4):

* UNKNOWN   — state absent from the graph, or all observations stale.
* CONTESTED — exactly one active measurement context: the role vector is
  collapsed by construction and superposition is unobservable.
* ASSERTED  — two or more active contexts measured.
"""

from __future__ import annotations

import math

import numpy as np

from ..kernel._types import SpectrumResult, Verdict
from .multigraph import ROLES, RoleAssignment, RoleMultigraph

_N_ROLE = len(ROLES)


def kl_divergence(p: dict[str, float], q: dict[str, float],
                  roles: tuple[str, ...] = ROLES, eps: float = 1e-12) -> float:
    """KL(p || q) in nats over the shared role support.

    Hook for FALSIFY-005 (tomography distribution vs reference, fail if
    KL > 0.5 and uncalibratable). Zero-probability entries are floored at
    ``eps`` so the divergence stays finite.
    """
    pv = np.array([max(p.get(r, 0.0), eps) for r in roles], dtype=np.float64)
    qv = np.array([max(q.get(r, 0.0), eps) for r in roles], dtype=np.float64)
    pv /= pv.sum()
    qv /= qv.sum()
    return float(np.sum(pv * np.log(pv / qv)))


def distribution_shift(p: dict[str, float], q: dict[str, float],
                       eps: float = 1e-12) -> dict:
    """Per-component shift contributions between two arbitrary distributions.

    Role-agnostic counterpart of :func:`kl_divergence` (which is typed over
    the DCA ``ROLES`` support and silently mismatches arbitrary keys — keys
    outside ``roles`` are floored to ``eps``, yielding a spurious near-zero
    divergence).  Accepts any key→weight mappings and reports the
    direction-aware breakdown of a distributional migration: for each key,
    ``p·ln(p/q)`` with the new state measured against the old baseline.

    Returns
    -------
    dict with keys ``contributions`` (per-key signed p·ln(p/q)),
    ``total_forward`` (KL(p‖q) in nats), ``total_reverse`` (KL(q‖p)), and
    ``jeffreys`` (forward + reverse — the symmetric migration magnitude).
    Zero-probability entries are floored at ``eps``.
    """
    keys = sorted(set(p) | set(q))
    contributions = {}
    total_reverse = 0.0
    for k in keys:
        pv = max(float(p.get(k, 0.0)), eps)
        qv = max(float(q.get(k, 0.0)), eps)
        contributions[k] = float(pv * np.log(pv / qv))
        total_reverse += float(qv * np.log(qv / pv))
    total_forward = float(sum(contributions.values()))
    return {"contributions": contributions,
            "total_forward": total_forward,
            "total_reverse": total_reverse,
            "jeffreys": total_forward + total_reverse}


def role_entropy(distribution: dict[str, float]) -> tuple[float, float]:
    """Shannon entropy (nats) of a role distribution, and its [0, 1] normalisation."""
    p = np.array([distribution.get(r, 0.0) for r in ROLES], dtype=np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0, 0.0
    p = p / total
    nonzero = p[p > 0]
    h = float(-np.sum(nonzero * np.log(nonzero)))
    return h, h / math.log(_N_ROLE)


def _aggregate(assignments: list[RoleAssignment], weight_mode: str) -> dict[str, float]:
    """Equal-thread-weighted aggregate distribution over ROLES."""
    by_thread: dict[str, list[RoleAssignment]] = {}
    for a in assignments:
        by_thread.setdefault(a.thread, []).append(a)

    totals = dict.fromkeys(ROLES, 0.0)
    n_threads = 0
    for thread_assignments in by_thread.values():
        weights = {r: 0.0 for r in ROLES}
        for a in thread_assignments:
            w = a.weight if weight_mode == "presence" else float(a.mass)
            weights[a.role] = weights.get(a.role, 0.0) + w
        thread_total = sum(weights.values())
        if thread_total <= 0:
            continue  # stale / empty thread casts no vote
        n_threads += 1
        for r in ROLES:
            totals[r] += weights[r] / thread_total
    if n_threads == 0:
        return dict.fromkeys(ROLES, 0.0)
    return {r: totals[r] / n_threads for r in ROLES}


def tomography(graph: RoleMultigraph, state_id: str,
               weight_mode: str = "presence") -> SpectrumResult:
    """Role-vector scan: enumerate a state's roles across all threads.

    Parameters
    ----------
    graph
        A RoleMultigraph (knowledge edges or warfare tree).
    state_id
        Knowledge entry id (e.g. ``"230-spielrein-destruction-instinct"``) or
        warfare state id (e.g. ``"rus/r05"``).
    weight_mode
        ``"presence"`` — each active field/edge votes with weight 1.
        ``"mass"`` — votes with informative character mass (warfare fields;
        a crude verbosity proxy, reported for sensitivity only).
    """
    if weight_mode not in ("presence", "mass"):
        raise ValueError(f"weight_mode must be 'presence' or 'mass', got {weight_mode!r}")

    assignments = graph.state_assignments(state_id)
    if not assignments:
        return SpectrumResult(
            values={"state_id": state_id, "n_threads": 0},
            verdict=Verdict.UNKNOWN,
            confidence_reason="state not present in multigraph — no measurement contexts",
        )

    active = [a for a in assignments if (a.weight if weight_mode == "presence" else a.mass) > 0]
    stale_threads = sorted({a.thread for a in assignments if a.stale})
    if not active:
        return SpectrumResult(
            values={
                "state_id": state_id,
                "n_threads": 0,
                "stale_threads": stale_threads,
            },
            verdict=Verdict.UNKNOWN,
            confidence_reason="all observations are stale placeholders/carry-overs — unmeasured",
        )

    per_thread: dict[str, dict[str, float]] = {}
    for a in active:
        w = a.weight if weight_mode == "presence" else float(a.mass)
        per_thread.setdefault(a.thread, {})[a.role] = (
            per_thread.setdefault(a.thread, {}).get(a.role, 0.0) + w
        )

    distribution = _aggregate(active, weight_mode)
    entropy, entropy_norm = role_entropy(distribution)
    dominant_role = max(ROLES, key=lambda r: (distribution[r], -ROLES.index(r)))
    dominance = distribution[dominant_role]
    styx_shadow = distribution["alternative"] > 0.0 and distribution["direction"] == 0.0

    threads = sorted(per_thread)
    n_threads = len(threads)
    if n_threads == 1:
        verdict = Verdict.CONTESTED
        reason = ("single active measurement context — role vector collapsed by "
                  "construction, superposition unobservable")
    else:
        verdict = Verdict.ASSERTED
        reason = f"{n_threads} active measurement contexts"

    return SpectrumResult(
        values={
            "state_id": state_id,
            "n_threads": n_threads,
            "per_thread": {t: per_thread[t] for t in threads},
            "role_distribution": distribution,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
            "dominant_role": dominant_role,
            "dominance": dominance,
            "styx_shadow": styx_shadow,
            "stale_threads": stale_threads,
            "weight_mode": weight_mode,
        },
        verdict=verdict,
        confidence_reason=reason,
    )
