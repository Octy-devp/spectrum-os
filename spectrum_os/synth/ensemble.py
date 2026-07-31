"""synth.ensemble — Branch Ensemble Engine (pay-as-you-go).

PLAN-23 §5 + §6.6 implementation:
- Mechanical Markov trajectory generation (zero LLM for state transitions).
- Action reweighting injection at step `t` (Bayesian intervention approximation).
- Gate call budget control (`gate_every` & `max_gates`).
- Seed reproducibility.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spectrum_os.contracts import DCASubstrate
from spectrum_os.quantum.multigraph import ROLES
from spectrum_os.quantum.markov import estimate_rate_matrix


def run_ensemble(
    initial_vector: dict | str,
    substrate: DCASubstrate | None = None,
    rate_matrix: np.ndarray | dict | None = None,
    action: dict | list[dict] | None = None,
    n_branches: int = 10,
    horizon: int = 24,
    gate_fn: Callable | None = None,
    gate_every: int = 4,
    max_gates: int = 12,
    seed: int = 42,
    situation: dict | None = None,
    residual_trigger: Callable | Mapping | None = None,
    residual_key_fn: Callable | None = None,
    enumerate_gate_fn: Callable | None = None,
    **gate_kwargs: Any,
) -> dict:
    """Run pay-as-you-go branch ensemble simulation with Batch-by-Phase scheduling.

    Phase 1: Pure mechanical simulation (zero API calls, draw Markov state transitions).
    Phase 2: Batch candidate generation calls ([task:generate] or gate_fn.stage1).
    Phase 3: Batch adversarial review calls ([task:adversary] or gate_fn.stage2).
    Backfill: Populate tags_per_step for each branch.

    Args:
        initial_vector: Dict containing 'role' or string initial role name (e.g. "crisis").
        substrate: Optional DCASubstrate context.
        rate_matrix: 4x4 NumPy transition matrix or dict containing "matrix" key.
            If None, defaults to uniform stochastic matrix over legal transitions.
        action: Optional intervention dict or list of dicts:
            ``{"t": int, "reweight": {"role": str, "factor": float}}``.
        n_branches: Number of branches to simulate (default 10).
        horizon: Simulation step length (default 24).
        gate_fn: Optional gate function (e.g. alt_gate) called at branch points.
        gate_every: Step interval for calling gate_fn (default 4).
        max_gates: Maximum gate calls per branch (default 12).
        seed: Random seed for reproducible trajectory sampling.
        situation: Optional situation context payload dict containing vector_6d, local_texture, digest.
        residual_trigger: Optional Mode-B trigger. Either a callable
            ``(t, context) -> bool`` (context is the gate_input dict) or a
            residual table Mapping (auto-wrapped via
            ``residual_trigger.make_residual_trigger`` with
            ``key_fn=residual_key_fn``). When it returns True at a gate point,
            the engine switches to the enumerate path (Mode B — spectrum
            failure) instead of the reflect path. None (default) keeps the
            pre-N4 behaviour exactly.
        residual_key_fn: Optional key mapping for a Mapping ``residual_trigger``
            — ``(t, gate_input) -> lookup key``, passed through to
            ``make_residual_trigger``. ⚠️ A date-keyed table WITHOUT it is
            silently dead: the identity int-step lookup never matches
            ``"YYYY-MM-DD"`` keys, so Mode B stays off with no error. For a
            date-keyed table whose date lives in the situation payload, use
            ``residual_key_fn=lambda t, ctx: ctx["situation"].get("date")``.
            Ignored when ``residual_trigger`` is already a callable.
        enumerate_gate_fn: Mode-B gate called at triggered gate points.
            Defaults to ``alt_gate_enumerate``. It returns
            ``{"accidents": [...]}`` (NOT ``{"candidates": [...]}``) — results
            are recorded in each branch's ``accidents_by_step`` and never flow
            into ``tags_per_step`` / accumulated candidate tags (F10).
        **gate_kwargs: Additional keyword arguments passed to gate_fn.

    Returns:
        Dict containing branches data and meta summary.
    """
    rng = np.random.default_rng(seed)

    # Resolve initial role
    if isinstance(initial_vector, str):
        initial_role = initial_vector
    elif isinstance(initial_vector, dict):
        initial_role = initial_vector.get("role", "crisis")
    else:
        initial_role = "crisis"

    if initial_role not in ROLES:
        raise ValueError(f"Invalid initial_role {initial_role!r}; allowed: {ROLES}")

    # Resolve rate matrix
    if rate_matrix is None:
        P = estimate_rate_matrix(np.zeros((len(ROLES), len(ROLES))))["matrix"]
    elif isinstance(rate_matrix, dict):
        P = np.asarray(rate_matrix["matrix"], dtype=np.float64)
    else:
        P = np.asarray(rate_matrix, dtype=np.float64)

    if P.shape != (len(ROLES), len(ROLES)):
        raise ValueError(f"Expected rate_matrix of shape (4, 4), got {P.shape}")

    # Resolve actions
    actions_list: list[dict] = []
    if isinstance(action, dict):
        actions_list = [action]
    elif isinstance(action, list):
        actions_list = [a for a in action if isinstance(a, dict)]

    # --- Phase 1: Pure Mechanical Simulation (Zero API calls) ---
    branch_records: list[dict] = []
    gate_points: list[dict] = []

    for b_idx in range(n_branches):
        branch_gate_calls = 0
        current_role = initial_role
        branch_roles = [current_role]
        interventions: list[dict] = []

        for t in range(horizon - 1):
            r_idx = ROLES.index(current_role)
            probs = np.copy(P[r_idx])

            matching_actions = [act for act in actions_list if act.get("t") == t]
            for act in matching_actions:
                rw = act.get("reweight", {})
                target_role = rw.get("role")
                factor = float(rw.get("factor", 1.0))
                if target_role in ROLES:
                    t_idx = ROLES.index(target_role)
                    probs[t_idx] *= factor
                    interventions.append({"t": t, "role": target_role, "factor": factor})

            sum_p = np.sum(probs)
            if sum_p > 0:
                probs /= sum_p
            else:
                probs = np.full(len(ROLES), 1.0 / len(ROLES), dtype=np.float64)

            next_role = str(rng.choice(ROLES, p=probs))
            branch_roles.append(next_role)

            if (
                gate_fn is not None
                and (t + 1) % gate_every == 0
                and branch_gate_calls < max_gates
            ):
                branch_gate_calls += 1
                gate_points.append(
                    {
                        "branch_id": b_idx,
                        "t": t,
                        "current_role": current_role,
                        "next_role": next_role,
                        "thread_history": branch_roles[:],
                    }
                )

            current_role = next_role

        branch_records.append(
            {
                "branch_id": b_idx,
                "roles": branch_roles,
                "interventions": interventions,
                "tags_per_step": {},
                "accidents_by_step": {},
            }
        )

    total_gate_calls = len(gate_points)

    # --- N4 (Mode B) trigger setup ---
    # residual_trigger accepts either a callable ``(t, context) -> bool`` or a
    # residual table Mapping (auto-wrapped). enumerate_gate_fn defaults to the
    # historical-accident enumeration gate (alt_gate_enumerate).
    if isinstance(residual_trigger, Mapping):
        from spectrum_os.synth.residual_trigger import make_residual_trigger

        residual_trigger = make_residual_trigger(
            residual_trigger, key_fn=residual_key_fn
        )
    if enumerate_gate_fn is None:
        from spectrum_os.synth.alt_gate import alt_gate_enumerate

        enumerate_gate_fn = alt_gate_enumerate

    if gate_fn is None or not gate_points:
        return {
            "branches": branch_records,
            "meta": {
                "n_branches": n_branches,
                "horizon": horizon,
                "total_gate_calls": 0,
                "total_enumerate_calls": 0,
                "seed": seed,
            },
        }

    # Check if gate_fn supports 2-stage execution via .stage1 and .stage2
    has_stages = hasattr(gate_fn, "stage1") and hasattr(gate_fn, "stage2")

    # --- Phase 2: Batch Generation Phase ([task:generate] or Stage 1 / gate_fn) ---
    branch_gate_map: dict[int, list[dict]] = {}
    for gp in gate_points:
        branch_gate_map.setdefault(gp["branch_id"], []).append(gp)

    total_enumerate_calls = 0

    for b_idx, gps in branch_gate_map.items():
        accumulated_tags: list[str] = []
        for gp in gps:
            if situation is not None:
                sit_payload = dict(situation)
                sit_payload["t"] = gp["t"]
                sit_payload["thread_history"] = gp["thread_history"]
            else:
                sit_payload = None

            gate_input = {
                "state_vector": {
                    "current_role": gp["current_role"],
                    "next_role": gp["next_role"],
                    "t": gp["t"],
                },
                "thread_history": gp["thread_history"],
                "existing_labels": accumulated_tags[:],
                "situation": sit_payload,
            }

            # --- N4: two-phase state switching ---
            # Subcritical (trigger False) -> Mode A reflect path (gate_fn /
            # stage1-stage2). Critical (trigger True) -> Mode B enumerate path:
            # the enumerate gate is called DIRECTLY (no stage1/stage2) and its
            # ``accidents`` are stored separately in ``accidents_by_step`` —
            # never merged into accumulated tags (F10).
            use_enumerate = bool(
                residual_trigger is not None and residual_trigger(gp["t"], gate_input)
            )
            if use_enumerate:
                gp["mode"] = "enumerate"
                enum_res = enumerate_gate_fn(gate_input, **gate_kwargs)
                gp["enumerate_res"] = enum_res
                gp["n_accidents"] = len(enum_res.get("accidents", []))
                total_enumerate_calls += 1
                continue

            gp["mode"] = "reflect"
            if has_stages:
                stage1_res = gate_fn.stage1(gate_input, **gate_kwargs)
                gp["stage1_res"] = stage1_res
                step_tags = []
                for cand in stage1_res.get("aggregated_candidates", []):
                    step_tags.extend(cand.get("concept_tags", []))
                accumulated_tags.extend(step_tags)
            else:
                gate_res = gate_fn(gate_input, **gate_kwargs)
                gp["gate_res"] = gate_res
                step_tags = []
                for cand in gate_res.get("candidates", []):
                    step_tags.extend(cand.get("concept_tags", []))
                accumulated_tags.extend(step_tags)

    # --- Phase 3: Batch Adversarial Phase ([task:adversary] or Stage 2) ---
    for gp in gate_points:
        b_idx = gp["branch_id"]
        t = gp["t"]
        if gp.get("mode") == "enumerate":
            # Mode B points record accidents ONLY in ``accidents_by_step``.
            # ``tags_per_step[t]`` is set to an empty list so the per-gate-point
            # presence invariant holds without ever leaking accident data into
            # tag logic.
            branch_records[b_idx]["accidents_by_step"][t] = gp["enumerate_res"].get(
                "accidents", []
            )
            branch_records[b_idx]["tags_per_step"][t] = []
            continue

        if has_stages and "stage1_res" in gp:
            final_res = gate_fn.stage2(gp["stage1_res"], **gate_kwargs)
        else:
            final_res = gp.get("gate_res", {})

        step_tags: list[str] = []
        for cand in final_res.get("candidates", []):
            step_tags.extend(cand.get("concept_tags", []))
        branch_records[b_idx]["tags_per_step"][t] = step_tags

    return {
        "branches": branch_records,
        "meta": {
            "n_branches": n_branches,
            "horizon": horizon,
            "total_gate_calls": total_gate_calls,
            "total_enumerate_calls": total_enumerate_calls,
            "seed": seed,
        },
    }

