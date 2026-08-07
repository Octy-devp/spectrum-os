"""Tests for the v1.6 kinetic-mathematics corrections.

Methodology: ``docs/method/force-model-social-dynamics.md`` v1.6 — §2.2b
(βC → P_R compression channel: suppression *stores* R as latent potential
rather than destroying it ⇒ conservation of T = R + P_R under pressure,
revival possible), §2.3b (S → a, c bandwagon/avalanche feedback) and §2.4b
(tension → λ release gating).

These are world-agnostic: the mechanisms are expressed purely in the universal
observables (R, C, P_R, S, T); the substrate supplies only the *form* of the
feedback functions and the parameter values.

Covers:
1. Reservoir-active detection (βC compression only when reservoir in use).
2. Conservation: dT = a·R − ρ·P under pressure (no −βC destruction term).
3. Backward compat: reservoir off ⇒ P inert (v1.0 behaviour).
4. P accumulates under pressure (vs v1.5 pure-destruction collapse).
5. No deadlock: with the reservoir on, high-β suppression does not collapse
   S to 0 (contrast with v1.5 pure destruction).
6. Revival (1905→1917 shape): high-pressure suppression accumulates P_R;
   release phase revives R (S rises).
7. Growth feedback a(S,T) — bandwagon avalanche.
8. Release feedback λ(T) — tension-gated reservoir opening.
9. S ∈ [0, 1] invariant holds with feedbacks active.
10. Feedback shape validation (wrong length raises).
"""

import numpy as np
import pytest

from spectrum_os.kernel.forces import ForceFieldDynamics


def _china_reservoir(**kw):
    """Single-node China-like engine with the v1.6 reservoir active."""
    d = dict(
        revolutionary_force=2.0,
        conservative_force=7.0,
        growth_r=0.001,
        growth_c=0.0005,
        alpha=0.002,
        beta=0.005,
        mu=0.004,
        lam=0.0001,
        rho=0.0005,
        latent_force=5.5,
    )
    d.update(kw)
    return ForceFieldDynamics(**d)


class TestReservoirCompression:
    def test_reservoir_active_flag(self):
        assert ForceFieldDynamics(5.0, 5.0)._reservoir_active is False
        assert _china_reservoir()._reservoir_active is True
        assert ForceFieldDynamics(5.0, 5.0, mu=0.004)._reservoir_active is True
        assert ForceFieldDynamics(5.0, 5.0, latent_force=1.0)._reservoir_active is True

    def test_betaC_compression_conserves_T(self):
        """dT = aR − ρP exactly: βC transfers R→P_R, not destruction."""
        eng = _china_reservoir()
        R, C, P = 2.0, 7.0, 5.5
        eng._R[:] = R
        eng._C[:] = C
        eng._P[:] = P
        dR, dC, dP = eng.derivatives_full(0.0)
        dT = dR + dP
        aR = 0.001 * R
        rhoP = 0.0005 * P
        assert abs(dT - (aR - rhoP)) < 1e-12
        # P grows: βC + μR exceeds λP + ρP
        assert dP > 0

    def test_reservoir_off_P_inert(self):
        """No mu/lam/rho/latent_force ⇒ dP = 0 (v1.0 behaviour)."""
        eng = ForceFieldDynamics(
            revolutionary_force=2.0,
            conservative_force=7.0,
            growth_r=0.001,
            growth_c=0.0005,
            alpha=0.002,
            beta=0.005,
        )
        dR, dC, dP = eng.derivatives_full(0.0)
        assert dP == 0.0
        assert eng._P == 0.0

    def test_P_accumulates_under_pressure(self):
        """With reservoir on + high β, P_R accumulates instead of being destroyed."""
        eng = _china_reservoir(beta=0.010)
        traj = eng.run(60.0)
        assert traj.P[-1, 0] > 5.5  # reservoir grew under pressure
        # contrast: reservoir off ⇒ P stays 0, R collapses
        eng2 = ForceFieldDynamics(
            revolutionary_force=2.0,
            conservative_force=7.0,
            growth_r=0.001,
            growth_c=0.0005,
            alpha=0.002,
            beta=0.010,
        )
        traj2 = eng2.run(60.0)
        assert traj2.P[-1, 0] == 0.0
        assert traj2.R[-1, 0] < 0.5  # pure destruction collapses R

    def test_no_deadlock_with_reservoir(self):
        """High-β over the full window: reservoir on *conserves capacity* T = R + P_R
        (stored as latent potential ⇒ revival channel); reservoir off destroys T
        (βC pure destruction ⇒ true deadlock). Kinetic R may still collapse under
        sustained pressure — that is the 'compression' — but the capacity lives on."""
        eng = _china_reservoir(beta=0.010, mod_beta=[lambda t: 1.5])
        traj = eng.run(240.0)
        assert traj.T[-1, 0] > 3.0  # capacity conserved (T ≈ R₀+P₀ − ρ decay)
        # v1.5-style (reservoir off, βC destroys) → T collapses to ~0
        eng2 = ForceFieldDynamics(
            revolutionary_force=2.0,
            conservative_force=7.0,
            growth_r=0.001,
            growth_c=0.0005,
            alpha=0.002,
            beta=0.010,
            mod_beta=[lambda t: 1.5],
        )
        traj2 = eng2.run(240.0)
        assert traj2.T[-1, 0] < 0.1  # deadlock: capacity destroyed

    def test_revival_after_release(self):
        """1905→1917 shape: crush (P accumulates) → release (R revives, S rises)."""
        # crush phase: high β, low λ (accumulate)
        eng = _china_reservoir(beta=0.005, mod_beta=[lambda t: 1.5])
        crush = eng.run(60.0)
        S_crush, P_crush = crush.S[-1, 0], crush.P[-1, 0]
        assert P_crush > 5.5
        # release phase: external field lifts α, λ opens
        eng2 = ForceFieldDynamics(
            revolutionary_force=crush.R[-1, 0],
            conservative_force=crush.C[-1, 0],
            growth_r=0.001,
            growth_c=0.0005,
            alpha=0.002,
            beta=0.005,
            mu=0.004,
            lam=0.020,
            rho=0.0005,
            latent_force=P_crush,
            mod_alpha=[lambda t: 2.0],
        )
        rel = eng2.run(60.0)
        assert rel.S[-1, 0] > S_crush + 0.05  # R revived, S rose


class TestGrowthFeedback:
    def test_bandwagon_a_fn(self):
        """a(S,T) rising in S ⇒ avalanche (tipping): steep feedback pushes S to
        revolution content, while constant-a stays low."""
        base = _china_reservoir(lam=0.02)
        fb = _china_reservoir(lam=0.02, growth_r_fn=lambda S, T: 1.0 + 100.0 * S)
        tb = base.run(120.0)
        tf = fb.run(120.0)
        assert tf.S[-1, 0] > 0.7  # avalanche to revolution content
        assert tf.S[-1, 0] > tb.S[-1, 0] + 0.3  # far above constant-a

    def test_s_invariant_with_feedback(self):
        eng = _china_reservoir(
            growth_r_fn=lambda S, T: 1.0 + 5.0 * S,
            growth_c_fn=lambda S, T: 1.0 - 2.0 * S,
            release_fn=lambda T: 0.0,
        )
        traj = eng.run(120.0)
        assert np.all(traj.S >= 0.0) and np.all(traj.S <= 1.0)

    def test_feedback_wrong_shape_raises(self):
        eng = _china_reservoir(growth_r_fn=lambda S, T: np.zeros(3))  # n=1, len-3
        with pytest.raises(ValueError):
            eng.run(1.0)


class TestReleaseFeedback:
    def test_tension_gated_holds_reservoir(self):
        """λ(T) gate closed ⇒ P_R held (no decay via λ); constant λ drains it."""
        const = _china_reservoir(lam=0.02)
        gated = _china_reservoir(lam=0.02, release_fn=lambda T: 0.0)  # gate closed
        tc = const.run(20.0)
        tg = gated.run(20.0)
        assert tg.P[-1, 0] > tc.P[-1, 0]

    def test_release_fn_receives_tension(self):
        """λ(T) is wired to the tension observable (universal, per-node array)."""
        seen = {}

        def gate(T):
            seen["T"] = np.asarray(T).copy()
            return 0.0

        eng = _china_reservoir(lam=0.02, release_fn=gate)
        eng.run(5.0)
        assert seen.get("T") is not None and seen["T"].shape == (1,)
