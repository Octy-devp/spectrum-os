"""Tests for spectrum_os/kernel/forces.py — R/C counterforce field engine.

Covers the social-dynamics first theorem engine (methodology
``docs/method/force-model-social-dynamics.md`` v1.0, §2.2–§2.6):

1. ODE convergence / non-negativity / no explosion.
2. S = R/(R+C) ∈ [0, 1] invariant (incl. degenerate R+C → 0).
3. Hungary-1919 reproduction: R initially dominant → external field raises β /
   lowers α → S flips from R-win to C-win.
4. External field modulation actually alters the rates.
5. Multi-node Richardson coupling basic behaviour.

Plus the ActorCard contract extension (revolutionary_force / conservative_force)
and the ``from_actor_cards`` / ``osc.forces`` wiring.
"""

import numpy as np
import pytest

from spectrum_os.contracts import ActorCard
from spectrum_os.kernel.forces import ForceFieldDynamics, ForceTrajectory
from spectrum_os.kernel import osc


def _engine(**kwargs):
    """Default well-behaved single-node engine with R dominant."""
    defaults = dict(
        revolutionary_force=10.0,
        conservative_force=5.0,
        growth_r=0.03,
        growth_c=0.01,
        alpha=0.012,
        beta=0.010,
    )
    defaults.update(kwargs)
    return ForceFieldDynamics(**defaults)


# ---------------------------------------------------------------------------
# 1. ODE convergence / non-negativity / no explosion
# ---------------------------------------------------------------------------

class TestOdeBehaviour:
    def _no_clamp_regime(self, method, dt):
        """Smooth regime (no clamping, forces stay positive) for convergence checks."""
        return ForceFieldDynamics(
            revolutionary_force=5.0,
            conservative_force=3.0,
            growth_r=0.02,
            growth_c=0.02,
            alpha=0.003,
            beta=0.003,
            method=method,
            dt=dt,
        ).run(5.0)

    def test_rk4_converges_under_refinement(self):
        """RK4 is 4th order: halving dt collapses the trajectory difference."""
        coarse = self._no_clamp_regime("rk4", 0.2)
        fine = self._no_clamp_regime("rk4", 0.1)
        fine_s = np.interp(coarse.t, fine.t, fine.S[:, 0])
        assert np.max(np.abs(fine_s - coarse.S[:, 0])) < 1e-9

        finest = self._no_clamp_regime("rk4", 0.05)
        finest_s = np.interp(coarse.t, finest.t, finest.S[:, 0])
        assert np.max(np.abs(finest_s - coarse.S[:, 0])) < 1e-9

    def test_euler_converges_under_refinement(self):
        """Euler is 1st order: halves with dt (smaller tolerance)."""
        coarse = self._no_clamp_regime("euler", 0.02)
        fine = self._no_clamp_regime("euler", 0.01)
        fine_s = np.interp(coarse.t, fine.t, fine.S[:, 0])
        assert np.max(np.abs(fine_s - coarse.S[:, 0])) < 1e-5

    def test_rk4_and_euler_agree_for_small_dt(self):
        """RK4 (fine dt) and Euler (very fine dt) land on the same trajectory."""
        rk4 = self._no_clamp_regime("rk4", 0.05)
        euler = self._no_clamp_regime("euler", 0.01)
        euler_s = np.interp(rk4.t, euler.t, euler.S[:, 0])
        assert np.allclose(rk4.S[:, 0], euler_s, atol=1e-4)

    def test_pure_conflict_is_bounded_no_explosion(self):
        """Lanchester-like pure conflict (no growth): bounded, no NaN/inf."""
        eng = ForceFieldDynamics(
            revolutionary_force=1.0,
            conservative_force=1.0,
            growth_r=0.0,
            growth_c=0.0,
            alpha=0.1,
            beta=0.1,
            dt=0.01,
        )
        traj = eng.run(10.0)
        assert np.all(np.isfinite(traj.R))
        assert np.all(np.isfinite(traj.C))
        assert np.all(traj.R >= -1e-12)
        assert np.all(traj.C >= -1e-12)
        # No growth terms → magnitude cannot exceed the initial envelope.
        assert np.max(traj.R) <= 1.0 + 1e-6
        assert np.max(traj.C) <= 1.0 + 1e-6

    def test_mixed_regime_stays_finite_and_nonneg(self):
        """Growth + suppression over a long run: finite, non-negative, S sane."""
        eng = _engine(dt=0.1)
        traj = eng.run(50.0)
        assert np.all(np.isfinite(traj.R))
        assert np.all(np.isfinite(traj.C))
        assert np.all(traj.R >= -1e-12)
        assert np.all(traj.C >= -1e-12)
        assert np.all((traj.S >= 0.0) & (traj.S <= 1.0))

    def test_invalid_method_rejected(self):
        with pytest.raises(ValueError, match="method must be one of"):
            _engine(method="midpoint")

    def test_negative_initial_force_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            ForceFieldDynamics(revolutionary_force=-1.0, conservative_force=1.0)

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="equal length"):
            ForceFieldDynamics(
                revolutionary_force=[1.0, 2.0], conservative_force=[1.0]
            )


# ---------------------------------------------------------------------------
# 2. S ∈ [0, 1] invariant (incl. degeneracy)
# ---------------------------------------------------------------------------

class TestSInvariant:
    def test_degenerate_zero_zero_pins_to_half(self):
        """R + C → 0 must yield 0.5 (undecided content), never NaN."""
        eng = ForceFieldDynamics(revolutionary_force=0.0, conservative_force=0.0)
        assert eng.s() == pytest.approx(0.5)

    def test_extreme_limits(self):
        eng = ForceFieldDynamics(revolutionary_force=1.0, conservative_force=1.0)
        assert eng.s(R=0.0, C=5.0) == pytest.approx(0.0)
        assert eng.s(R=5.0, C=0.0) == pytest.approx(1.0)
        assert eng.s(R=3.0, C=2.0) == pytest.approx(0.6)

    def test_invariant_across_parameter_sweep(self):
        """S stays in [0, 1] for a broad sweep of rates and initial conditions."""
        rng = np.random.default_rng(7)
        for _ in range(40):
            R0 = float(rng.uniform(0.0, 20.0))
            C0 = float(rng.uniform(0.0, 20.0))
            eng = ForceFieldDynamics(
                revolutionary_force=R0,
                conservative_force=C0,
                growth_r=float(rng.uniform(0.0, 0.1)),
                growth_c=float(rng.uniform(0.0, 0.1)),
                alpha=float(rng.uniform(0.0, 0.1)),
                beta=float(rng.uniform(0.0, 0.1)),
                dt=0.2,
            )
            traj = eng.run(20.0)
            assert np.all(traj.S >= -1e-12)
            assert np.all(traj.S <= 1.0 + 1e-12)

    def test_invariant_holds_after_repeated_step(self):
        eng = _engine(dt=0.1)
        for _ in range(50):
            snap = eng.step()
            assert 0.0 <= float(snap["S"][0]) <= 1.0


# ---------------------------------------------------------------------------
# 3. Hungary-1919 reproduction (core validation)
# ---------------------------------------------------------------------------

class TestHungaryFlip:
    """R initially seizes power (S high) → external field raises β / lowers α
    (Entente blockade, Romanian invasion strengthen the counter-revolution) →
    C counterattacks → S flips from R-win to C-win (capitalist content)."""

    @staticmethod
    def _hungary_engine():
        # Phase 1 (t < T1): R dominant, C survives (both grow at parity).
        def mod_alpha(t):
            return 0.1 if t >= 15.0 else 1.0   # R's suppression of C collapses

        def mod_beta(t):
            return 20.0 if t >= 15.0 else 1.0  # C's suppression of R skyrockets

        return ForceFieldDynamics(
            revolutionary_force=10.0,
            conservative_force=5.0,
            growth_r=0.02,
            growth_c=0.02,
            alpha=0.003,   # α⁰ — R suppresses C (weak)
            beta=0.003,    # β⁰ — C suppresses R (weak)
            mod_alpha=mod_alpha,
            mod_beta=mod_beta,
            dt=0.5,
            node_ids=["hungary-1919"],
            meta={"case": "hungary-1919"},
        )

    def test_s_flips_from_r_win_to_c_win(self):
        eng = self._hungary_engine()
        traj = eng.run(150.0)

        s0 = float(traj.S[0, 0])
        s_peak = float(np.max(traj.S[:, 0]))
        s_end = float(traj.S[-1, 0])

        # R starts and stays dominant before the field lands (t < 15).
        pre = traj.t < 15.0
        assert s0 > 0.6
        assert float(np.min(traj.S[pre, 0])) > 0.65

        # The resultant actually flips across the 0.5 boundary.
        crossed = np.where((traj.S[:, 0] - 0.5) * (traj.S[0, 0] - 0.5) < 0)[0]
        assert crossed.size > 0

        # And ends clearly on the C side: revolutionary content is NOT the
        # final content — the counter-revolution's victory was mis-attributed.
        assert s_end < 0.2
        assert s_peak > 0.67

    def test_wrong_sign_does_not_flip(self):
        """Guards the sign convention: α↑/β↓ favours R — S must NOT flip.

        The §2.2 equations are the SSOT: β is C's suppression of R (raising it
        makes R lose), α is R's suppression of C (lowering it makes C win).
        A field doing the opposite (α↑/β↓) is mathematically R-favouring, so
        the flip must not occur — otherwise the sign convention is broken.
        """

        def mod_alpha(t):
            return 3.0 if t >= 15.0 else 1.0  # R crushes C harder

        def mod_beta(t):
            return 0.2 if t >= 15.0 else 1.0  # C stops suppressing R

        eng = ForceFieldDynamics(
            revolutionary_force=10.0,
            conservative_force=5.0,
            growth_r=0.02,
            growth_c=0.02,
            alpha=0.003,
            beta=0.003,
            mod_alpha=mod_alpha,
            mod_beta=mod_beta,
            dt=0.5,
        )
        traj = eng.run(150.0)
        assert float(traj.S[0, 0]) > 0.6
        assert float(traj.S[-1, 0]) > 0.6  # still R-dominant

    def test_no_field_no_flip(self):
        """Baseline: identical setup with no external field — R keeps winning."""
        eng = ForceFieldDynamics(
            revolutionary_force=10.0,
            conservative_force=5.0,
            growth_r=0.02,
            growth_c=0.02,
            alpha=0.003,
            beta=0.003,
            dt=0.5,
        )
        traj = eng.run(150.0)
        assert float(traj.S[0, 0]) > 0.6
        assert float(traj.S[-1, 0]) > 0.6


# ---------------------------------------------------------------------------
# 4. External field modulation
# ---------------------------------------------------------------------------

class TestExternalField:
    def test_effective_rates_apply_modifiers(self):
        eng = _engine(alpha=0.1, beta=0.05, mod_alpha=lambda t: 2.0, mod_beta=lambda t: 3.0)
        a_eff, b_eff = eng.effective_rates(t=5.0)
        assert a_eff == pytest.approx([0.2])
        assert b_eff == pytest.approx([0.15])

    def test_effective_rates_default_identity(self):
        eng = _engine(alpha=0.1, beta=0.05)
        a_eff, b_eff = eng.effective_rates()
        assert a_eff == pytest.approx([0.1])
        assert b_eff == pytest.approx([0.05])

    def test_time_dependent_field_changes_trajectory(self):
        """A stronger β-modifier must push the final S further toward C."""
        def run_with(mod_beta):
            return ForceFieldDynamics(
                revolutionary_force=10.0,
                conservative_force=8.0,
                growth_r=0.02,
                growth_c=0.02,
                alpha=0.005,
                beta=0.005,
                mod_beta=mod_beta,
                dt=0.5,
            ).run(60.0)

        mild = run_with(lambda t: 1.0)
        strong = run_with(lambda t: 2.0)
        # Stronger counter-revolution suppression ⇒ R loses more ground
        # (verified: mild S_end ≈ 0.68, strong S_end ≈ 0.42).
        assert float(strong.S[-1, 0]) < float(mild.S[-1, 0])

    def test_per_node_modifier_list(self):
        eng = ForceFieldDynamics(
            revolutionary_force=[10.0, 10.0],
            conservative_force=[5.0, 5.0],
            alpha=[0.01, 0.01],
            beta=[0.01, 0.01],
            mod_beta=[lambda t: 1.0, lambda t: 10.0],
            dt=0.5,
        )
        a_eff, b_eff = eng.effective_rates()
        assert b_eff == pytest.approx([0.01, 0.1])
        traj = eng.run(60.0)
        # Node 1 (strong field) must end with lower S than node 0.
        assert float(traj.S[-1, 1]) < float(traj.S[-1, 0])


# ---------------------------------------------------------------------------
# 5. Multi-node coupling (Richardson)
# ---------------------------------------------------------------------------

class TestCoupling:
    def test_coupling_equalises_forces(self):
        """With symmetric coupling and no growth/suppression, forces converge
        toward the mean (κ·(R_j − R_i) pulls weaker nodes up, strong ones down)."""
        eng = ForceFieldDynamics(
            revolutionary_force=[10.0, 2.0],
            conservative_force=[1.0, 1.0],
            growth_r=0.0,
            growth_c=0.0,
            alpha=0.0,
            beta=0.0,
            kappa=0.5,
            dt=0.1,
            node_ids=["strong", "weak"],
        )
        traj = eng.run(30.0)
        r_strong = traj.R[:, 0]
        r_weak = traj.R[:, 1]
        mean = (10.0 + 2.0) / 2.0
        # The gap narrows monotonically toward the mean.
        gap0 = abs(r_strong[0] - r_weak[0])
        gap_end = abs(r_strong[-1] - r_weak[-1])
        assert gap_end < gap0 * 1e-3
        assert abs(float(r_strong[-1]) - mean) < 1e-3
        assert abs(float(r_weak[-1]) - mean) < 1e-3

    def test_coupling_boosts_weak_node(self):
        """A weak node's R rises when coupled to a strong one vs uncoupled."""
        base = dict(
            revolutionary_force=[10.0, 1.0],
            conservative_force=[1.0, 1.0],
            growth_r=0.0,
            growth_c=0.0,
            alpha=0.0,
            beta=0.0,
            dt=0.2,
        )
        uncoupled = ForceFieldDynamics(**base).run(10.0)
        coupled = ForceFieldDynamics(kappa=0.3, **base).run(10.0)
        assert float(coupled.R[-1, 1]) > float(uncoupled.R[-1, 1])

    def test_kappa_shape_validation(self):
        with pytest.raises(ValueError, match="kappa.R must have shape"):
            ForceFieldDynamics(
                revolutionary_force=[1.0, 1.0],
                conservative_force=[1.0, 1.0],
                kappa={"R": np.ones((3, 3))},
            )

    def test_kappa_dict_unknown_key_rejected(self):
        with pytest.raises(ValueError, match="kappa dict keys"):
            ForceFieldDynamics(
                revolutionary_force=[1.0, 1.0],
                conservative_force=[1.0, 1.0],
                kappa={"X": np.ones((2, 2))},
            )

    def test_force_specific_coupling_matrices(self):
        """R and C coupling can differ (κ_R vs κ_C)."""
        kR = np.array([[0.0, 0.4], [0.4, 0.0]])
        kC = np.array([[0.0, 0.0], [0.0, 0.0]])
        eng = ForceFieldDynamics(
            revolutionary_force=[10.0, 1.0],
            conservative_force=[1.0, 1.0],
            growth_r=0.0,
            growth_c=0.0,
            alpha=0.0,
            beta=0.0,
            kappa={"R": kR, "C": kC},
            dt=0.2,
        )
        traj = eng.run(15.0)
        # Only R couples → R equalises (gap ≈ 9·e^{−0.8·15} < 1e-3), C stays flat.
        assert abs(float(traj.R[-1, 0]) - float(traj.R[-1, 1])) < 1e-3
        assert float(traj.C[-1, 0]) == pytest.approx(1.0)
        assert float(traj.C[-1, 1]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ActorCard extension + wiring
# ---------------------------------------------------------------------------

def _card(actor_id, r=None, c=None):
    return ActorCard(
        actor_id=actor_id,
        name=f"{actor_id} name",
        inherited_conditions={"rank": "pioneer"},
        field_coordinates={"x": 1, "y": 2},
        internal_tensions=[{"axis_A": "a", "axis_B": "b", "note": "t"}],
        temporal_states={"t0": "idle"},
        revolutionary_force=r,
        conservative_force=c,
    )


class TestActorCardForces:
    def test_new_fields_default_to_none(self):
        """Backward compatibility: omitting the fields is still valid."""
        card = _card("alpha")
        assert card.revolutionary_force is None
        assert card.conservative_force is None

    def test_fields_roundtrip_through_to_dict(self):
        card = _card("alpha", r=7.5, c=2.5)
        d = card.to_dict()
        assert d["revolutionary_force"] == 7.5
        assert d["conservative_force"] == 2.5
        assert card.revolutionary_force == 7.5

    def test_negative_force_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            _card("alpha", r=-1.0)
        with pytest.raises(ValueError, match="non-negative"):
            _card("alpha", c=-0.5)

    def test_from_actor_cards_builds_engine(self):
        cards = [_card("state", r=10.0, c=5.0), _card("party", r=4.0, c=None)]
        eng = ForceFieldDynamics.from_actor_cards(
            cards, growth_r=0.02, growth_c=0.01, alpha=0.01, beta=0.01
        )
        assert eng.n_nodes == 2
        assert eng.node_ids == ["state", "party"]
        R, C = eng.state
        assert R.tolist() == [10.0, 4.0]
        assert C.tolist() == [5.0, 0.0]  # missing → 0.0 fallback

    def test_from_actor_cards_rejects_non_cards(self):
        with pytest.raises(ValueError, match="expected ActorCard"):
            ForceFieldDynamics.from_actor_cards([{"not": "a card"}])


class TestWiring:
    def test_osc_forces_exposed(self):
        """The kernel OSC entry point must expose the forces module."""
        assert hasattr(osc, "forces")
        assert osc.forces.ForceFieldDynamics is ForceFieldDynamics

    def test_trajectory_to_dict_is_json_safe(self):
        import json

        eng = _engine(dt=0.5)
        traj = eng.run(10.0)
        payload = traj.to_dict()
        json.dumps(payload)  # must not raise
        assert len(payload["t"]) == len(payload["R"])
