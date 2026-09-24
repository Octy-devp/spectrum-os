"""Tests for spectrum_os.extensions.bosd — Eq. (5) online filtering recursion.

Fixtures exercise the UPM protocol with a conjugate known-variance Gaussian
model (stats = (n, sum_x)), so every expectation below is closed-form or
seeded-deterministic. No randomness inside the filter itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from spectrum_os.extensions.bosd import (
    BOSDConfig,
    bosd_init,
    bosd_update,
)


class KnownVarianceGaussianUPM:
    """Conjugate normal model with known variance.

    Sufficient statistics: (n, sum_x). The one-step predictive is a normal
    density centred on the posterior mean with inflated variance.
    """

    def __init__(self, variance=1.0, prior_mean=0.0, prior_strength=1.0):
        self.variance = float(variance)
        self.prior_mean = float(prior_mean)
        self.prior_strength = float(prior_strength)

    def initial_stats(self):
        return (0, 0.0)

    def predictive_loglik(self, stats, y):
        n, s = stats
        post_prec = self.prior_strength + n
        post_mean = (self.prior_strength * self.prior_mean + s) / post_prec
        var = self.variance * (1.0 + 1.0 / post_prec)
        dev = (y - post_mean) / math.sqrt(var)
        return -0.5 * (math.log(2.0 * math.pi * var) + dev * dev)

    def update(self, stats, y):
        n, s = stats
        return (n + 1, s + float(y))


class CountingUPM:
    """Protocol-conformant wrapper counting UPM calls (complexity probe).

    Uses the two-argument ``initial_stats(d, z)`` factory form so the
    temporal-scaling hook (Def. 3) is exercised: duration and regime ride
    along inside the stats tuple.
    """

    def __init__(self, inner):
        self.inner = inner
        self.factory_calls = []
        self.predictive_calls = 0
        self.update_calls = 0

    def initial_stats(self, d, z):
        self.factory_calls.append((d, z))
        base = self.inner.initial_stats()
        return (base[0], base[1], d, z)

    def predictive_loglik(self, stats, y):
        self.predictive_calls += 1
        return self.inner.predictive_loglik(stats[:2], y)

    def update(self, stats, y):
        self.update_calls += 1
        n, s = self.inner.update(stats[:2], y)
        return (n, s, stats[2], stats[3])


def _run_series(config, upm, ys):
    state = bosd_init(config, upm)
    posteriors = []
    for y in ys:
        state = bosd_update(state, y, upm)
        posteriors.append(state.posterior)
    return state, posteriors


def test_changepoint_localization():
    """Posterior peak lands on the known changepoint (within tolerance)."""
    rng = np.random.default_rng(20260923)
    n1, n2 = 40, 40
    ys = np.concatenate(
        [rng.normal(0.0, 1.0, n1), rng.normal(3.0, 1.0, n2)]
    ).tolist()
    config = BOSDConfig(
        k_states=1,
        max_duration=50,
        hazard=0.005,
        # Long-segment duration prior: degenerate at d = D, so the deadline
        # rule cannot force a reset inside the 80-step window and the only
        # run-length resets are evidence-driven (the true changepoint).
        duration_logprior=lambda d, z: 0.0 if d == 50 else -math.inf,
    )
    upm = KnownVarianceGaussianUPM(variance=1.0, prior_mean=0.0, prior_strength=1.0)
    state, posts = _run_series(config, upm, ys)

    # Posteriors are normalised at every step and evidence stays finite.
    for p in posts:
        assert p.run_length_posterior.sum() == pytest.approx(1.0, abs=1e-9)
        assert p.joint_posterior.sum() == pytest.approx(1.0, abs=1e-9)
        assert (p.joint_posterior >= 0.0).all()
    assert math.isfinite(state.loglik)

    # After the true changepoint (first obs of the new level is t = n1 + 1),
    # the MAP run length tracks t - n1 within a 2-step detection tolerance.
    for t in range(n1 + 3, n1 + n2 + 1):
        map_r = posts[t - 1].map_run_length
        assert abs(map_r - (t - n1)) <= 2, (t, map_r, t - n1)

    # The pre-change run was tracked as well (before any hazard reset).
    for t in range(10, n1 - 1):
        assert posts[t - 1].map_run_length == t


def test_residual_time_decreases_toward_segment_end():
    """Def. 2: with h = 0 and a degenerate duration prior at d = D, the run
    grows deterministically and the residual-time mean falls as (D - t),
    until the deadline forces the segment to end (h_eff(r, d) = 1)."""
    D = 20
    config = BOSDConfig(
        k_states=1,
        max_duration=D,
        hazard=0.0,
        duration_logprior=lambda d, z: 0.0 if d == D else -math.inf,
    )
    upm = KnownVarianceGaussianUPM()
    state = bosd_init(config, upm)
    means = []
    for t in range(1, D + 1):
        state = bosd_update(state, 0.0, upm)
        p = state.posterior
        mean_res = float((np.arange(D) * p.residual_time_posterior).sum())
        assert p.map_run_length == t
        assert p.map_residual_time == D - t
        assert mean_res == pytest.approx(float(D - t), abs=1e-9)
        means.append(mean_res)
    assert all(a > b for a, b in zip(means, means[1:]))

    # At the deadline the segment must end even with h = 0.
    state = bosd_update(state, 0.0, upm)
    p = state.posterior
    assert p.map_run_length == 1
    assert p.map_residual_time == D - 1
    assert math.isfinite(state.loglik)


def test_deterministic_bitwise_reproducibility():
    """Two runs over identical inputs are bit-identical (pure recursion)."""
    rng = np.random.default_rng(7)
    ys = np.concatenate(
        [rng.normal(0.0, 1.0, 15), rng.normal(2.0, 1.0, 15)]
    ).tolist()
    config = BOSDConfig(k_states=2, max_duration=12, hazard=0.05)
    upm = KnownVarianceGaussianUPM()

    def run():
        st = bosd_init(config, upm)
        snaps = []
        for y in ys:
            st = bosd_update(st, y, upm)
            snaps.append(
                (
                    st.t,
                    st.log_gamma.tobytes(),
                    st.posterior.joint_posterior.tobytes(),
                    st.posterior.run_length_posterior.tobytes(),
                    st.posterior.residual_time_posterior.tobytes(),
                    st.loglik,
                    tuple(sorted(st.stats.items())),
                )
            )
        return snaps

    assert run() == run()


def test_small_grid_k4_d2_complexity_and_protocol():
    """K=4, D=2: shapes, normalisation, UPM call counts within the
    O(K^2 + D^2 K) per-step bound, and the (d, z) factory hook."""
    K, D, T = 4, 2, 30
    rng = np.random.default_rng(11)
    means = [-1.5, 0.0, 1.5, 3.0]
    ys = [means[(t // 5) % K] + float(rng.normal(0.0, 0.5)) for t in range(T)]

    config = BOSDConfig(k_states=K, max_duration=D, hazard=0.2)
    upm = CountingUPM(KnownVarianceGaussianUPM(variance=0.25))
    state = bosd_init(config, upm)
    for y in ys:
        state = bosd_update(state, y, upm)

    # Per step: O(D^2 K) emissions + O(D K) prior-stats emissions; same order
    # for stats updates. The counts must respect that bound.
    bound = T * (D * D * K + D * K)
    assert 0 < upm.predictive_calls <= bound
    assert 0 < upm.update_calls <= bound

    # Temporal-scaling hook: the factory saw the full (d, z) grid.
    assert set(upm.factory_calls) == {
        (d, z) for d in range(1, D + 1) for z in range(K)
    }

    p = state.posterior
    assert p.run_length_posterior.shape == (D,)
    assert p.residual_time_posterior.shape == (D,)
    assert p.segment_length_posterior.shape == (D,)
    assert p.state_posterior.shape == (K,)
    assert p.joint_posterior.shape == (D, D, K)
    assert p.joint_posterior.sum() == pytest.approx(1.0, abs=1e-9)
    assert p.state_posterior.sum() == pytest.approx(1.0, abs=1e-9)
    assert (p.joint_posterior >= 0.0).all()
    assert (p.residual_time_posterior >= 0.0).all()
    assert 1 <= p.map_segment_length <= D
    assert 0 <= p.map_state < K
    assert math.isfinite(state.loglik)
    # Entries with r > d carry no mass on the truncated grid.
    assert p.joint_posterior[1, 0, :].sum() == pytest.approx(0.0, abs=1e-15)
