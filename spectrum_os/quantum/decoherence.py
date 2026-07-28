"""quantum.decoherence_watch — role-entropy time series (PLAN-23 §6.3).

Monitors a system's role-distribution entropy over time. High superposition
(multiple roles active) collapsing into single-role dominance — an entropy
cliff — is the phase-transition alarm of §6.3: measurement has happened, the
role vector has decohered.

Also reports dominant-role flips between consecutive time points; a
``direction -> crisis`` flip is DCA rule #1 (what crystallized becomes the
source of the next rupture) and is flagged as ``rule1_flip``.

Time axis
---------
Only warfare trees carry a real time coordinate (round numbers, stored in
``RoleMultigraph.state_times``). The knowledge multigraph is a single
timeless snapshot; watching it yields an honest UNKNOWN rather than a
fabricated series. Note the alarm is *detection* only — interpreting whether
a drop is genuine crystallization or a reporting/template artifact belongs
to the API gate layer (PLAN-23 §8 division of labor).

Ternary verdict:

* UNKNOWN   — fewer than 2 time points (no series to watch).
* CONTESTED — 2–3 time points (series too thin for a reliable alarm).
* ASSERTED  — 4 or more time points.
"""

from __future__ import annotations

import numpy as np

from ..kernel._types import SpectrumResult, Verdict
from .multigraph import ROLES, RoleMultigraph
from .tomography import role_entropy, tomography


def decoherence_watch(graph: RoleMultigraph, system_id: str,
                      weight_mode: str = "presence",
                      drop_fraction: float = 0.5,
                      min_drop: float = 0.1) -> SpectrumResult:
    """Entropy-watch one system (e.g. a warfare faction).

    Parameters
    ----------
    graph
        A RoleMultigraph. States belonging to the system are selected by the
        id prefix ``"{system_id}/"`` and must have a recorded time in
        ``graph.state_times``.
    system_id
        System prefix — for warfare trees the faction name (``"rus"``).
    weight_mode
        Passed through to :func:`tomography`.
    drop_fraction
        Alarm when a consecutive entropy drop reaches this fraction of the
        running peak (and exceeds ``min_drop`` nats).
    min_drop
        Absolute floor (nats) for an alarm, guarding against noise at low
        entropy.
    """
    points = sorted(
        ((t, sid) for sid, t in graph.state_times.items()
         if sid.startswith(f"{system_id}/")),
        key=lambda item: item[0],
    )

    if len(points) < 2:
        return SpectrumResult(
            values={
                "system_id": system_id,
                "n_points": len(points),
                "series": [],
            },
            verdict=Verdict.UNKNOWN,
            confidence_reason=(
                "no time axis or fewer than 2 time points — decoherence "
                "unwatchable (knowledge multigraph is a timeless snapshot; "
                "warfare factions need >= 2 rounds)"
            ),
        )

    series: list[dict] = []
    for t, sid in points:
        scan = tomography(graph, sid, weight_mode=weight_mode)
        distribution = scan.values.get("role_distribution", dict.fromkeys(ROLES, 0.0))
        entropy, entropy_norm = role_entropy(distribution)
        series.append({
            "t": t,
            "state_id": sid,
            "entropy": entropy,
            "entropy_norm": entropy_norm,
            "distribution": distribution,
            "dominant_role": scan.values.get("dominant_role"),
        })

    entropies = np.array([p["entropy"] for p in series], dtype=np.float64)
    running_peak = np.maximum.accumulate(entropies)

    alarms: list[dict] = []
    for i in range(1, len(series)):
        drop = float(entropies[i - 1] - entropies[i])
        threshold = max(min_drop, drop_fraction * float(running_peak[i - 1]))
        if drop >= threshold:
            alarms.append({
                "t": series[i]["t"],
                "from_t": series[i - 1]["t"],
                "from_h": float(entropies[i - 1]),
                "to_h": float(entropies[i]),
                "drop": drop,
            })

    flips: list[dict] = []
    for i in range(1, len(series)):
        prev_role = series[i - 1]["dominant_role"]
        curr_role = series[i]["dominant_role"]
        if prev_role and curr_role and prev_role != curr_role:
            flips.append({
                "t": series[i]["t"],
                "from_role": prev_role,
                "to_role": curr_role,
                "rule1_flip": prev_role == "direction" and curr_role == "crisis",
            })

    max_drop = float(np.max(entropies[:-1] - entropies[1:])) if len(series) > 1 else 0.0

    verdict = Verdict.ASSERTED if len(series) >= 4 else Verdict.CONTESTED
    reason = (f"{len(series)} time points, {len(alarms)} alarm(s), "
              f"max consecutive drop {max_drop:.3f} nats")
    if len(series) < 4:
        reason += " — series too thin for a reliable alarm"

    return SpectrumResult(
        values={
            "system_id": system_id,
            "n_points": len(series),
            "series": series,
            "alarms": alarms,
            "dominant_flips": flips,
            "max_drop": max_drop,
            "alarm_detected": len(alarms) > 0,
            "drop_fraction": drop_fraction,
            "min_drop": min_drop,
        },
        verdict=verdict,
        confidence_reason=reason,
    )
