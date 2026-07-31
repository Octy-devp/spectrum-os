"""synth.ensemble — Branch Ensemble Engine (pay-as-you-go).

PLAN-23 §5 + §6.6 implementation:
- Mechanical Markov trajectory generation (zero LLM for state transitions).
- Action reweighting injection at step `t` (Bayesian intervention approximation).
- Gate call budget control (`gate_every` & `max_gates`).
- Seed reproducibility.
"""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

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
) -> dict:
    """Run pay-as-you-go branch ensemble simulation.

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
        # Default: uniform over LEGAL exits only — DCA grammar structural zeros
        # must never be sampleable (forbidden pairs stay exactly 0.0).
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

    total_gate_calls = 0
    branches: list[dict] = []

    for b_idx in range(n_branches):
        branch_gate_calls = 0
        current_role = initial_role
        branch_roles = [current_role]
        tags_per_step: dict[int, list[str]] = {}
        interventions: list[dict] = []
        accumulated_tags: list[str] = []

        for t in range(horizon - 1):
            r_idx = ROLES.index(current_role)
            probs = np.copy(P[r_idx])

            # Apply action reweighting at step t if specified
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

            # Mechanical Markov sample
            next_role = str(rng.choice(ROLES, p=probs))
            branch_roles.append(next_role)

            # Check gate call budget (per-branch): a global cap would starve
            # later branches of any gate calls, making standing_wave "nodes"
            # vacuously empty by construction.
            if (
                gate_fn is not None
                and (t + 1) % gate_every == 0
                and branch_gate_calls < max_gates
            ):
                branch_gate_calls += 1
                total_gate_calls += 1
                gate_input = {
                    "state_vector": {"current_role": current_role, "next_role": next_role, "t": t},
                    "thread_history": branch_roles[:],
                    "existing_labels": accumulated_tags[:],
                }
                gate_res = gate_fn(gate_input)
                step_tags: list[str] = []
                for cand in gate_res.get("candidates", []):
                    step_tags.extend(cand.get("concept_tags", []))
                tags_per_step[t] = step_tags
                accumulated_tags.extend(step_tags)

            current_role = next_role

        branches.append(
            {
                "branch_id": b_idx,
                "roles": branch_roles,
                "tags_per_step": tags_per_step,
                "interventions": interventions,
            }
        )

    return {
        "branches": branches,
        "meta": {
            "n_branches": n_branches,
            "horizon": horizon,
            "total_gate_calls": total_gate_calls,
            "seed": seed,
        },
    }
