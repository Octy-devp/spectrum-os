"""Quantum Markov Dynamics Core — Transition Matrix Estimation & Posterior Sampling.

Provides transition count aggregation, constrained rate matrix estimation (with
Dirichlet prior smoothing over allowed DCA transitions), and posterior sampling
for stochastic state transition matrices.
"""

from __future__ import annotations

import numpy as np

from spectrum_os.quantum.dca_grammar import DCA_GRAMMAR
from spectrum_os.quantum.multigraph import ROLES


def count_transitions(
    role_sequences: list[list[str]],
) -> tuple[np.ndarray, list[dict]]:
    """Compute 4x4 transition counts from role sequences and flag anomalies.

    Args:
        role_sequences: A list of sequences, where each sequence is a list of role
            names (e.g. ``["crisis", "lag", "alternative"]``).

    Returns:
        A tuple ``(counts, anomalies)`` where:
        - ``counts`` is a 4x4 NumPy float64 array containing transition frequencies.
        - ``anomalies`` is a list of dicts describing forbidden transitions found:
          ``{"from": src, "to": dst, "position": k, "sequence": seq}``.
          Note that forbidden transitions are counted in ``counts`` and flagged in
          ``anomalies`` without being swallowed.
    """
    role_to_idx = {r: i for i, r in enumerate(ROLES)}
    forbidden_set = set(tuple(pair) for pair in DCA_GRAMMAR.get("forbidden", []))

    counts = np.zeros((len(ROLES), len(ROLES)), dtype=np.float64)
    anomalies: list[dict] = []

    for seq in role_sequences:
        for k in range(len(seq) - 1):
            src, dst = seq[k], seq[k + 1]
            if src not in role_to_idx or dst not in role_to_idx:
                raise ValueError(f"Unknown role in sequence transition: {src!r} -> {dst!r}")

            src_idx = role_to_idx[src]
            dst_idx = role_to_idx[dst]
            counts[src_idx, dst_idx] += 1.0

            if (src, dst) in forbidden_set:
                anomalies.append(
                    {
                        "from": src,
                        "to": dst,
                        "position": k,
                        "sequence": seq,
                    }
                )

    return counts, anomalies


def estimate_rate_matrix(
    counts: np.ndarray,
    alpha: float = 1.0,
    min_total_count: float = 10.0,
) -> dict:
    """Estimate stochastic transition rate matrix with Dirichlet prior smoothing.

    Forbidden cells are enforced to be identically 0.0. The denominator for row i
    is sum_{j in A_i} C_{i,j} + alpha * |A_i|, where A_i is the set of allowed target
    roles from role i.

    Args:
        counts: 4x4 array of transition counts.
        alpha: Dirichlet prior parameter (pseudo-counts). Default 1.0.
        min_total_count: Threshold for total_count to mark verdict as ASSERTED vs UNKNOWN.

    Returns:
        Dict containing:
        - ``"matrix"``: 4x4 stochastic transition matrix.
        - ``"verdict"``: ``"ASSERTED"`` if total_count >= min_total_count else ``"UNKNOWN"``.
        - ``"counts"``: The input counts array (as float64 ndarray).
        - ``"total_count"``: Total transition count sum (float).
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    num_roles = len(ROLES)
    matrix = np.zeros((num_roles, num_roles), dtype=np.float64)
    forbidden_set = set(tuple(pair) for pair in DCA_GRAMMAR.get("forbidden", []))

    for i, src in enumerate(ROLES):
        allowed_indices = [
            j for j, dst in enumerate(ROLES) if (src, dst) not in forbidden_set
        ]
        num_allowed = len(allowed_indices)
        denom = float(np.sum(counts_arr[i, allowed_indices]) + alpha * num_allowed)
        for j in allowed_indices:
            matrix[i, j] = (counts_arr[i, j] + alpha) / denom

    total_count = float(np.sum(counts_arr))
    verdict = "ASSERTED" if total_count >= min_total_count else "UNKNOWN"

    return {
        "matrix": matrix,
        "verdict": verdict,
        "counts": counts_arr,
        "total_count": total_count,
    }


def sample_rate_matrix(
    counts: np.ndarray,
    alpha: float = 1.0,
    n: int = 200,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Draw n stochastic rate matrices from Dirichlet posterior.

    Forbidden cells remain 0.0 in all generated samples.

    Args:
        counts: 4x4 array of transition counts.
        alpha: Dirichlet prior parameter.
        n: Number of samples to draw.
        seed: Optional random seed for reproducible sampling.

    Returns:
        List of n NumPy arrays of shape (4, 4), each representing a valid stochastic matrix.
    """
    counts_arr = np.asarray(counts, dtype=np.float64)
    rng = np.random.default_rng(seed)
    num_roles = len(ROLES)
    forbidden_set = set(tuple(pair) for pair in DCA_GRAMMAR.get("forbidden", []))

    samples = [np.zeros((num_roles, num_roles), dtype=np.float64) for _ in range(n)]

    for i, src in enumerate(ROLES):
        allowed_indices = [
            j for j, dst in enumerate(ROLES) if (src, dst) not in forbidden_set
        ]
        alpha_params = [counts_arr[i, j] + alpha for j in allowed_indices]
        dirichlet_samples = rng.dirichlet(alpha_params, size=n)

        for k in range(n):
            for idx_in_allowed, j in enumerate(allowed_indices):
                samples[k][i, j] = dirichlet_samples[k, idx_in_allowed]

    return samples


def _resolve_roles(P: np.ndarray, roles: list[str] | None = None) -> list[str]:
    """Helper to resolve role names for matrix dimension."""
    if roles is not None:
        return list(roles)
    n = len(P)
    if n == len(ROLES):
        return list(ROLES)
    elif n <= len(ROLES):
        return list(ROLES[:n])
    else:
        return [f"role_{i}" for i in range(n)]


def hitting_times(
    P: np.ndarray,
    target: str,
    roles: list[str] | None = None,
) -> dict[str, float]:
    """Compute expected steps to reach target role for all roles using (I - Q)^(-1) 1.

    For unreachable states (e.g. disconnected absorbing components), the hitting
    time is set to float('inf').

    Args:
        P: Stochastic transition matrix.
        target: Target role name.
        roles: Optional list of role names matching P dimensions.

    Returns:
        Dict mapping role name to expected hitting time (target role maps to 0.0).
    """
    P_arr = np.asarray(P, dtype=np.float64)
    role_list = _resolve_roles(P_arr, roles)
    if target not in role_list:
        raise ValueError(f"Target role {target!r} not in roles {role_list!r}")

    target_idx = role_list.index(target)
    n = len(role_list)

    # Reachability analysis: find all states that can reach target_idx via positive probability transitions
    reachable_to_target = {target_idx}
    queue = [target_idx]
    while queue:
        curr = queue.pop(0)
        preds = np.where(P_arr[:, curr] > 0)[0]
        for p in preds:
            if p not in reachable_to_target:
                reachable_to_target.add(p)
                queue.append(p)

    res = {}
    for idx, r in enumerate(role_list):
        if r == target:
            res[r] = 0.0
        elif idx not in reachable_to_target:
            res[r] = float("inf")

    reachable_transient = [i for i in range(n) if i != target_idx and i in reachable_to_target]
    if reachable_transient:
        Q_sub = P_arr[np.ix_(reachable_transient, reachable_transient)]
        I_Q = np.eye(len(reachable_transient)) - Q_sub
        ones = np.ones(len(reachable_transient), dtype=np.float64)

        try:
            t_sub = np.linalg.solve(I_Q, ones)
        except np.linalg.LinAlgError:
            t_sub = np.linalg.lstsq(I_Q, ones, rcond=None)[0]

        for k, idx in enumerate(reachable_transient):
            res[role_list[idx]] = float(max(t_sub[k], 0.0))

    return res


def absorption_probabilities(
    P: np.ndarray,
    absorbing: list[str],
    roles: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute absorption probability matrix B = (I - Q)^(-1) R for transient states into absorbing states.

    Args:
        P: Stochastic transition matrix.
        absorbing: List of absorbing role names.
        roles: Optional list of role names matching P dimensions.

    Returns:
        Nested dict: res[transient_role][absorbing_role] = probability.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    role_list = _resolve_roles(P_arr, roles)

    absorbing_set = set(absorbing)
    for r in absorbing_set:
        if r not in role_list:
            raise ValueError(f"Absorbing role {r!r} not in roles {role_list!r}")

    transient_indices = [i for i, r in enumerate(role_list) if r not in absorbing_set]
    absorbing_indices = [i for i, r in enumerate(role_list) if r in absorbing_set]

    if not transient_indices or not absorbing_indices:
        return {}

    Q = P_arr[np.ix_(transient_indices, transient_indices)]
    R = P_arr[np.ix_(transient_indices, absorbing_indices)]
    I_Q = np.eye(len(transient_indices)) - Q

    try:
        B = np.linalg.solve(I_Q, R)
    except np.linalg.LinAlgError:
        B = np.linalg.lstsq(I_Q, R, rcond=None)[0]

    res = {}
    for i, t_idx in enumerate(transient_indices):
        t_role = role_list[t_idx]
        res[t_role] = {}
        for j, a_idx in enumerate(absorbing_indices):
            a_role = role_list[a_idx]
            res[t_role][a_role] = float(max(min(B[i, j], 1.0), 0.0))

    return res


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Compute stationary distribution pi satisfying pi P = pi, sum(pi) = 1.

    Args:
        P: Stochastic transition matrix.

    Returns:
        1D NumPy array of length equal to dim(P).
    """
    P_arr = np.asarray(P, dtype=np.float64)
    n = len(P_arr)
    if n == 0:
        return np.array([], dtype=np.float64)
    if np.allclose(P_arr, 0.0):
        return np.ones(n, dtype=np.float64) / float(n)
    eigenvalues, eigenvectors = np.linalg.eig(P_arr.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    if np.sum(pi) < 0:
        pi = -pi
    pi = np.maximum(pi, 0.0)
    s = np.sum(pi)
    if s > 0:
        pi = pi / s
    else:
        pi = np.ones(n, dtype=np.float64) / float(n)
    return pi


def entropy_rate(P: np.ndarray) -> float:
    """Compute entropy rate -sum_i pi_i sum_j P_ij ln(P_ij) with 0 ln 0 = 0.

    Args:
        P: Stochastic transition matrix.

    Returns:
        Entropy rate float value >= 0.0.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    pi = stationary_distribution(P_arr)

    rate = 0.0
    for i in range(len(P_arr)):
        row = P_arr[i]
        positive_mask = row > 0
        if np.any(positive_mask):
            row_entropy = -np.sum(row[positive_mask] * np.log(row[positive_mask]))
            rate += pi[i] * row_entropy

    return float(max(rate, 0.0))


def spectral_gap(P: np.ndarray) -> float:
    """Compute spectral gap 1 - |lambda_2| for transition matrix P.

    Args:
        P: Stochastic transition matrix.

    Returns:
        Spectral gap float value between 0.0 and 1.0.
    """
    P_arr = np.asarray(P, dtype=np.float64)
    eigenvalues = np.linalg.eigvals(P_arr)
    mags = np.sort(np.abs(eigenvalues))[::-1]
    if len(mags) < 2:
        return 0.0
    gap = 1.0 - mags[1]
    return float(max(gap, 0.0))


def summarize_samples(values: list[float]) -> dict[str, float]:
    """Summarize posterior sample metrics with mean and 95% confidence interval.

    Args:
        values: List of float metric samples.

    Returns:
        Dict with keys 'mean', 'ci95_low', 'ci95_high'.
    """
    arr = np.asarray(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    return {
        "mean": mean_val,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }

