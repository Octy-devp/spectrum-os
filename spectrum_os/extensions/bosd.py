"""Bayesian Online Segmentation with Duration (BOSD) — extensions layer.

Online filtering recursion (Eq. 5) of Agudelo-España et al. 2019
(arXiv:1902.04524), generalized to the joint latent state
(run length, segment length, regime label).

Layer declaration (spectrum-os MODIFICATION-LAW rule 7): **extensions
layer**. Pure mathematics, world-agnostic: no world-specific vocabulary,
no I/O, no randomness. Two runs over identical inputs are bit-identical.

Model
-----
Latent state at time t is the triple (r_t, d_t, z_t):

* ``r_t`` -- run length (Def. 1): number of observations since the last
  changepoint; the age of the segment containing t.
* ``d_t`` -- segment length (Def. 3): total number of steps of the segment
  containing t. The observation model may condition on ``d_t`` (temporal
  scaling) through duration-conditional sufficient statistics.
* ``l_t`` -- residual time (Def. 2): ``l_t = d_t - r_t``; marginalised as
  ``p(l_t | Y) = sum_{r_t} p(l_t | r_t) p(r_t | Y)``.
* ``z_t`` -- discrete regime label in ``{0, ..., K-1}``.

Filtering recursion (Eq. 5), carried in log space with log-sum-exp
accumulation to prevent underflow::

    gamma_t(r, d, z) = p(r_t, d_t, z_t, Y_{1:t})
                     = sum_{r', d', z'} p(y_t | r, d, z, Y^{r})
                                       * p(r, d, z | r', d', z')
                                       * gamma_{t-1}(r', d', z')

    p(r, d, z | Y_{1:t}) = gamma_t(r, d, z) / sum_{r,d,z} gamma_t(r, d, z)

Transition ``p(r, d, z | r', d', z')`` has exactly two branches:

* survival: ``(r'+1, d', z')`` with probability ``1 - h(r', d')``;
* changepoint: ``(1, d, z)`` with probability
  ``h(r', d') * g(d | z) * pi(z | z')``,

where ``h`` is the hazard (a plain constant by default), ``g`` the
segment-duration prior and ``pi`` the regime transition matrix. A
hypothesis whose run has reached its hypothesised length (``r' = d'``)
cannot survive -- its duration claim would be falsified -- so the deadline
forces ``h(r', d') = 1`` there. This guarantees at least one live branch
per live hypothesis, so the evidence never strands.

Complexity per step on the truncated grid ``1 <= r <= d <= D``, ``z < K``
with caller-supplied sufficient statistics (no stored observation
history):

* ``O(D^2 * K)`` predictive-density evaluations and stats updates,
* ``O(K^2)`` regime mixing,
* ``O(D^2)`` hazard bookkeeping (precomputed once at init),

matching the exponential-family bound ``O(K^2 + D^2 * K)`` stated in the
paper.

UPM protocol (caller-supplied unknown-parameter model)
------------------------------------------------------
The filter is agnostic to the observation model. The caller supplies a
:class:`UPM` (or any object with the same three attributes):

* ``initial_stats`` -- the empty / prior sufficient statistics, **or** a
  deterministic factory. A factory taking ``(d, z)`` yields duration- and
  regime-conditional statistics (temporal scaling, Def. 3); a
  zero-argument factory yields plain statistics. Factories must be pure.
* ``predictive_loglik(stats, y)`` -- ``log p(y_t | stats)``, the one-step
  posterior predictive density.
* ``update(stats, y)`` -- the new sufficient statistics after absorbing
  ``y``; must be pure (return a new value, never mutate).

Sufficient statistics live per hypothesis ``(r, d, z)`` inside the
:class:`BOSDState`, so observation history is summarised, never stored.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

__all__ = [
    "UPM",
    "BOSDConfig",
    "BOSDState",
    "BOSDPosterior",
    "constant_hazard",
    "uniform_duration_logprior",
    "bosd_init",
    "bosd_update",
]


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def _logsumexp(a: np.ndarray, axis=None):
    """Robust log-sum-exp. All-``-inf`` slices reduce to ``-inf``, not NaN."""
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return -math.inf
    m = np.max(a, axis=axis, keepdims=True)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)) + m_safe
    if axis is None:
        return float(out.reshape(()))
    axes = np.atleast_1d(axis)
    axes = np.where(axes < 0, axes + a.ndim, axes)
    keep = [i for i in range(a.ndim) if i not in set(axes.tolist())]
    return out.reshape(tuple(a.shape[i] for i in keep))


# ---------------------------------------------------------------------------
# UPM protocol
# ---------------------------------------------------------------------------


@dataclass
class UPM:
    """Unknown-parameter model: caller-supplied conjugate statistics.

    Attributes:
        initial_stats: sufficient-statistics object, or a pure factory.
            Factories taking ``(d, z)`` enable duration/state-conditional
            priors (temporal scaling, Def. 3); zero-argument factories are
            also accepted.
        predictive_loglik: ``(stats, y) -> log p(y | stats)``.
        update: ``(stats, y) -> new stats``; pure, never mutates its input.
    """

    initial_stats: Union[Any, Callable[..., Any]]
    predictive_loglik: Callable[[Any, Any], float]
    update: Callable[[Any, Any], Any]


def _as_upm(upm: Any) -> UPM:
    """Accept either a :class:`UPM` or a duck-typed equivalent."""
    if isinstance(upm, UPM):
        return upm
    return UPM(
        initial_stats=upm.initial_stats,
        predictive_loglik=upm.predictive_loglik,
        update=upm.update,
    )


def _resolve_initial_stats(initial_stats: Any, d: int, z: int) -> Any:
    """Resolve ``initial_stats`` for hypothesis (d, z).

    Non-callables pass through; callables are invoked as ``(d, z)`` when
    their signature accepts two positional arguments, else with no
    arguments. Deterministic by contract.
    """
    if not callable(initial_stats):
        return initial_stats
    try:
        inspect.signature(initial_stats).bind(d, z)
        return initial_stats(d, z)
    except TypeError:
        return initial_stats()


def constant_hazard(h: float) -> Callable[[int, int], float]:
    """Constant hazard ``h(r, d) = h`` — the paper's baseline parameterisation."""
    h = float(h)
    if not 0.0 <= h <= 1.0:
        raise ValueError(f"hazard must lie in [0, 1], got {h}")

    def _h(r: int, d: int) -> float:
        return h

    return _h


def uniform_duration_logprior(max_duration: int) -> Callable[[int, int], float]:
    """Log prior ``log g(d | z) = -log D`` on ``{1, ..., D}``, independent of z."""
    log_uniform = -math.log(float(max_duration))

    def _g(d: int, z: int) -> float:
        return log_uniform

    return _g


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BOSDConfig:
    """Explicit filter parameters (no hidden defaults inside the recursion).

    Attributes:
        k_states: ``K`` — number of discrete regime labels.
        max_duration: ``D`` — cap on segment length ``d`` (and therefore on
            run length ``r <= d``). The truncated grid is what bounds the
            per-step cost at ``O(K^2 + D^2 K)``.
        hazard: constant in ``[0, 1]``, or a callable ``h(r, d) -> float``
            in human units (``r, d >= 1``). The deadline rule forces the
            effective hazard to 1 whenever ``r >= d``.
        duration_logprior: optional callable ``log g(d | z)`` in human units
            (``d >= 1``, ``z`` 0-based). Defaults to uniform on ``{1..D}``.
        regime_transition: optional ``(K, K)`` row-stochastic matrix
            ``pi[z_new, z_prev]``. Defaults to uniform.
        initial_state_probs: optional ``(K,)`` distribution over the first
            regime. Defaults to uniform.
    """

    k_states: int
    max_duration: int
    hazard: Union[float, Callable[[int, int], float]] = 0.1
    duration_logprior: Optional[Callable[[int, int], float]] = None
    regime_transition: Optional[np.ndarray] = None
    initial_state_probs: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.k_states < 1:
            raise ValueError("k_states must be >= 1")
        if self.max_duration < 1:
            raise ValueError("max_duration must be >= 1")
        if isinstance(self.hazard, float) and not 0.0 <= self.hazard <= 1.0:
            raise ValueError("constant hazard must lie in [0, 1]")
        if self.regime_transition is not None:
            pi = np.asarray(self.regime_transition, dtype=np.float64)
            if pi.shape != (self.k_states, self.k_states):
                raise ValueError("regime_transition must be (K, K)")
            if not np.allclose(pi.sum(axis=1), 1.0, atol=1e-9):
                raise ValueError("regime_transition rows must sum to 1")
        if self.initial_state_probs is not None:
            p0 = np.asarray(self.initial_state_probs, dtype=np.float64)
            if p0.shape != (self.k_states,):
                raise ValueError("initial_state_probs must be (K,)")
            if not math.isclose(float(p0.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("initial_state_probs must sum to 1")


# ---------------------------------------------------------------------------
# State and posterior summaries
# ---------------------------------------------------------------------------


@dataclass
class BOSDPosterior:
    """Filtered summaries at one time step (all distributions normalised).

    Array indexing conventions:
        ``run_length_posterior[i]``     = ``P(r_t = i + 1)``      (shape D)
        ``residual_time_posterior[i]``  = ``P(l_t = i)``          (shape D)
        ``segment_length_posterior[i]`` = ``P(d_t = i + 1)``      (shape D)
        ``state_posterior[i]``          = ``P(z_t = i)``          (shape K)
        ``joint_posterior[r-1, d-1, z]``= ``P(r_t, d_t, z_t)``    (D, D, K)
    """

    t: int
    run_length_posterior: np.ndarray
    residual_time_posterior: np.ndarray
    segment_length_posterior: np.ndarray
    state_posterior: np.ndarray
    joint_posterior: np.ndarray
    map_run_length: int
    map_residual_time: int
    map_segment_length: int
    map_state: int
    log_evidence: float


@dataclass
class BOSDState:
    """Immutable-by-convention filter state; ``bosd_update`` returns a new one.

    Attributes:
        log_gamma: normalised log filtering distribution over the grid
            ``[r-1, d-1, z]``; entries with ``r > d`` are ``-inf``.
        loglik: cumulative log evidence ``sum_t log p(y_t | Y_{1:t-1})``.
        stats: sufficient statistics per hypothesis, keyed ``(r-1, d-1, z)``.
        init_stats: resolved prior statistics per ``(d-1, z)`` (cached so a
            factory is evaluated once per hypothesis, deterministically).
        log_haz / log_surv: precomputed ``(D, D)`` log hazard / log survival
            tables in human units ``[r-1, d-1]`` (deadline folded in).
        log_g_grid: ``(D, K)`` table of ``log g(d | z)``.
    """

    config: BOSDConfig
    t: int
    log_gamma: np.ndarray
    loglik: float
    stats: Dict[Tuple[int, int, int], Any]
    init_stats: Dict[Tuple[int, int], Any]
    posterior: Optional[BOSDPosterior]
    log_haz: np.ndarray = field(repr=False)
    log_surv: np.ndarray = field(repr=False)
    log_g_grid: np.ndarray = field(repr=False)
    log_pi: np.ndarray = field(repr=False)
    log_p0: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def _hazard_tables(config: BOSDConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Materialise the (D, D) hazard tables in human units, deadline folded in.

    ``log_haz[i, j] = log h_eff(r=i+1, d=j+1)`` and
    ``log_surv[i, j] = log(1 - h_eff(...))``, with ``h_eff = 1`` whenever
    ``r >= d`` (a run cannot outlive its hypothesised segment length).
    """
    D = config.max_duration
    haz = np.ones((D, D), dtype=np.float64)
    if isinstance(config.hazard, float):
        haz[:, :] = config.hazard
    else:
        for i in range(D):
            for j in range(D):
                haz[i, j] = float(config.hazard(i + 1, j + 1))
    # Deadline: at r >= d the segment must end.
    deadline = np.arange(D)[:, None] >= np.arange(D)[None, :]
    haz = np.where(deadline, 1.0, haz)
    haz = np.clip(haz, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        log_haz = np.log(haz)
        log_surv = np.log(1.0 - haz)
    return log_haz, log_surv


def bosd_init(config: BOSDConfig, upm: Any) -> BOSDState:
    """Create the empty filter state (before the first observation)."""
    K = config.k_states
    D = config.max_duration
    if config.duration_logprior is not None:
        g = config.duration_logprior
    else:
        g = uniform_duration_logprior(D)
    log_g_grid = np.empty((D, K), dtype=np.float64)
    for j in range(D):
        for z in range(K):
            log_g_grid[j, z] = float(g(j + 1, z))
    pi = (
        np.asarray(config.regime_transition, dtype=np.float64)
        if config.regime_transition is not None
        else np.full((K, K), 1.0 / K)
    )
    p0 = (
        np.asarray(config.initial_state_probs, dtype=np.float64)
        if config.initial_state_probs is not None
        else np.full(K, 1.0 / K)
    )
    with np.errstate(divide="ignore"):
        log_pi = np.log(pi)
        log_p0 = np.log(p0)
    log_haz, log_surv = _hazard_tables(config)
    return BOSDState(
        config=config,
        t=0,
        log_gamma=np.full((D, D, K), -math.inf),
        loglik=0.0,
        stats={},
        init_stats={},
        posterior=None,
        log_haz=log_haz,
        log_surv=log_surv,
        log_g_grid=log_g_grid,
        log_pi=log_pi,
        log_p0=log_p0,
    )


# ---------------------------------------------------------------------------
# Eq. (5) recursion
# ---------------------------------------------------------------------------


def _emission_tables(
    state: BOSDState, y: Any, upm: UPM
) -> Tuple[Dict[Tuple[int, int], Any], np.ndarray]:
    """One-step predictive densities ``p(y | r, d, z, Y^{r})`` on the grid.

    Emission for target ``(r=1, d, z)`` uses the prior statistics; for
    ``r >= 2`` it uses the predecessor's statistics (the run so far).
    Returns the (cached) resolved prior stats and the ``(D, D, K)`` table.
    """
    D = state.config.max_duration
    K = state.config.k_states
    init_stats = dict(state.init_stats)
    emission = np.full((D, D, K), -math.inf)
    for j in range(D):
        for z in range(K):
            if (j, z) not in init_stats:
                init_stats[(j, z)] = _resolve_initial_stats(upm.initial_stats, j + 1, z)
            emission[0, j, z] = float(upm.predictive_loglik(init_stats[(j, z)], y))
    if state.t > 0:  # runs longer than state.t cannot exist yet
        for i in range(1, min(D, state.t + 1)):
            for j in range(i, D):
                for z in range(K):
                    prev = state.stats[(i - 1, j, z)]
                    emission[i, j, z] = float(upm.predictive_loglik(prev, y))
    return init_stats, emission


def _advance_stats(
    state: BOSDState,
    y: Any,
    upm: UPM,
    init_stats: Dict[Tuple[int, int], Any],
) -> Dict[Tuple[int, int, int], Any]:
    """Absorb ``y`` into every valid hypothesis' sufficient statistics."""
    D = state.config.max_duration
    K = state.config.k_states
    new_stats: Dict[Tuple[int, int, int], Any] = {}
    for j in range(D):
        for z in range(K):
            new_stats[(0, j, z)] = upm.update(init_stats[(j, z)], y)
    if state.t > 0:  # runs longer than state.t cannot exist yet
        for i in range(1, min(D, state.t + 1)):
            for j in range(i, D):
                for z in range(K):
                    new_stats[(i, j, z)] = upm.update(state.stats[(i - 1, j, z)], y)
    return new_stats


def _summarize(log_gamma: np.ndarray, log_evidence: float, t: int) -> BOSDPosterior:
    """Marginals, residual-time distribution and MAP summaries from gamma_t."""
    joint = np.exp(log_gamma)
    D, _, K = joint.shape
    run_length = joint.sum(axis=(1, 2))
    segment_length = joint.sum(axis=(0, 2))
    state_post = joint.sum(axis=(0, 1))
    # Def. 2: p(l | Y) = sum over hypotheses with d - r = l.
    residual = np.zeros(D, dtype=np.float64)
    for i in range(D):
        residual[: D - i] += joint[i, i:, :].sum(axis=1)
    return BOSDPosterior(
        t=t,
        run_length_posterior=run_length,
        residual_time_posterior=residual,
        segment_length_posterior=segment_length,
        state_posterior=state_post,
        joint_posterior=joint,
        map_run_length=int(np.argmax(run_length)) + 1,
        map_residual_time=int(np.argmax(residual)),
        map_segment_length=int(np.argmax(segment_length)) + 1,
        map_state=int(np.argmax(state_post)),
        log_evidence=float(log_evidence),
    )


def bosd_update(state: BOSDState, y_t: Any, upm: Any) -> BOSDState:
    """One step of the Eq. (5) filtering recursion.

    Computes ``gamma_t(r, d, z) = p(r_t, d_t, z_t, Y_{1:t})`` from
    ``gamma_{t-1}`` by summing over predecessor hypotheses, mixing a
    survival branch and a changepoint branch, weighting each by its
    one-step predictive density, then normalising (the normaliser is the
    log-evidence increment). All accumulation runs through log-sum-exp so
    long sequences cannot underflow.

    Args:
        state: previous filter state (from :func:`bosd_init` or a prior
            :func:`bosd_update`); never mutated.
        y_t: the new observation, passed to the UPM verbatim (scalar or
            vector).
        upm: a :class:`UPM` or duck-typed equivalent.

    Returns:
        A new :class:`BOSDState` with ``posterior`` populated
        (run length, residual time, joint (r, d, z) summary, MAP segment
        length among them). Deterministic: identical inputs give
        bit-identical states.

    Raises:
        ValueError: if the total evidence underflows to ``-inf`` (only
            possible with a degenerate caller-supplied prior).
    """
    upm = _as_upm(upm)
    cfg = state.config
    D = cfg.max_duration
    K = cfg.k_states
    log_w = np.full((D, D, K), -math.inf)

    if state.t == 0:
        # Seed: the first observation starts a run, r = 1, d ~ g(.|z), z ~ p0.
        init_stats, emission = _emission_tables(state, y_t, upm)
        log_w[0] = state.log_p0[None, :] + state.log_g_grid + emission[0]
        new_stats = _advance_stats(state, y_t, upm, init_stats)
    else:
        gamma_prev = state.log_gamma
        init_stats, emission = _emission_tables(state, y_t, upm)
        # Survival branch: (r'+1, d', z') with prob 1 - h(r', d').
        # log_surv is -inf at the deadline (r' = d'), so falsified duration
        # claims drop out here.
        log_w[1:] = (
            gamma_prev[:-1] + state.log_surv[:-1, :, None] + emission[1:]
        )
        # Changepoint branch: (1, d, z) with prob h(r', d') g(d|z) pi(z|z').
        # The O(K^2) regime mix collapses the predecessor sum.
        weighted = gamma_prev + state.log_haz[..., None]
        inflow_by_prev_z = _logsumexp(weighted, axis=(0, 1))  # (K,)
        inflow_by_new_z = _logsumexp(state.log_pi + inflow_by_prev_z[None, :], axis=1)
        log_w[0] = state.log_g_grid + emission[0] + inflow_by_new_z[None, :]
        new_stats = _advance_stats(state, y_t, upm, init_stats)

    log_norm = _logsumexp(log_w)
    if not math.isfinite(log_norm):
        raise ValueError(
            "BOSD evidence underflowed to -inf; check hazard / duration_logprior "
            "so that at least one hypothesis has finite prior mass"
        )
    log_gamma = log_w - log_norm
    return BOSDState(
        config=cfg,
        t=state.t + 1,
        log_gamma=log_gamma,
        loglik=state.loglik + log_norm,
        stats=new_stats,
        init_stats=init_stats,
        posterior=_summarize(log_gamma, log_norm, state.t + 1),
        log_haz=state.log_haz,
        log_surv=state.log_surv,
        log_g_grid=state.log_g_grid,
        log_pi=state.log_pi,
        log_p0=state.log_p0,
    )
