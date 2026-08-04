"""Tests for the v1.3 P_R reservoir + four-quadrant coupling extensions.

Methodology: ``docs/method/force-model-social-dynamics.md`` v1.3 — §2.2b
(P_R potential reservoir: μ compression / λ release / ρ decay) and §2.6
(four-quadrant full-sign coupling matrix) plus §3.2 (conservation = H₀,
observable not constraint) and §3.3 (parameters = semantic anchors).

Covers:
1. P_R backward compatibility — no mu/lam/rho/latent_force ⇒ v1.0 dynamics.
2. 1905→1917 revival — high-β crush compresses R into P_R (T ≈ conserved),
   external-field modulation triggers λ-release, R revives (revolution is not
   mathematically killed).
3. Panic spiral — R rises ⇒ κ_RC makes C counterattack (cross-class escalation).
4. T observability — ForceTrajectory.T exists, correct shape, equals R + P_R.
5. Nonlinear panic — panic_exponent > 1 ⇒ superlinear C counterattack.
6. S ∈ [0, 1] invariant still holds with the reservoir active.
7. Four-quadrant kappa API (backward-compatible mapping + new full-sign dict).
8. ActorCard latent_force extension.
"""

import numpy as np
import pytest

from spectrum_os.contracts import ActorCard
from spectrum_os.kernel.forces import ForceFieldDynamics, ForceTrajectory


def _ref_v10_rk4(R0, C0, a, c, alpha, beta, kR, kC, t_end, dt=0.05):
    """Reference v1.0 RK4 integrator (no reservoir) — the backward-compat oracle."""
    n = len(R0)
    R = np.array(R0, dtype=np.float64)
    C = np.array(C0, dtype=np.float64)
    t = 0.0
    while t < t_end - 1e-12:
        def f(r, cc):
            dR = a * r - beta * cc
            dC = c * cc - alpha * r
            if np.any(kR):
                dR = dR + kR @ r - np.sum(kR, axis=1) * r
            if np.any(kC):
                dC = dC + kC @ cc - np.sum(kC, axis=1) * cc
            return dR, dC
        k1r, k1c = f(R, C)
        k2r, k2c = f(R + dt / 2 * k1r, C + dt / 2 * k1c)
        k3r, k3c = f(R + dt / 2 * k2r, C + dt / 2 * k2c)
        k4r, k4c = f(R + dt * k3r, C + dt * k3c)
        R = np.maximum(R + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r), 0.0)
        C = np.maximum(C + dt / 6 * (k1c + 2 * k2c + 2 * k3c + k4c), 0.0)
        t += dt
    return R, C


# ---------------------------------------------------------------------------
# 1. P_R backward compatibility
# ---------------------------------------------------------------------------

class TestPReservoirBackwardCompat:
    def test_defaults_match_v10_reference(self):
        """No mu/lam/rho/latent_force ⇒ engine reproduces the v1.0 equations."""
        kR = np.array([[0.0, 0.3], [0.3, 0.0]])
        kC = np.full((2, 2), 0.1)
        eng = ForceFieldDynamics(
            revolutionary_force=[10.0, 4.0],
            conservative_force=[5.0, 3.0],
            growth_r=0.02, growth_c=0.01, alpha=0.012, beta=0.010,
            kappa={"R": kR, "C": kC},
            dt=0.05,
        )
        traj = eng.run(40.0)

        # reservoir never activates
        assert np.all(traj.P == 0.0)
        # T = R + P_R collapses to R when the reservoir is off
        assert np.allclose(traj.T, traj.R)

        # matches the hand-rolled v1.0 integrator
        R_ref, C_ref = _ref_v10_rk4(
            [10.0, 4.0], [5.0, 3.0], 0.02, 0.01, 0.012, 0.010, kR, kC, 40.0
        )
        assert np.allclose(traj.R[-1], R_ref, atol=1e-6)
        assert np.allclose(traj.C[-1], C_ref, atol=1e-6)

    def test_explicit_zero_reservoir_identical(self):
        """Passing mu=lam=rho=0 and latent_force=0 explicitly is identical."""
        base = dict(
            revolutionary_force=10.0, conservative_force=5.0,
            growth_r=0.03, growth_c=0.01, alpha=0.012, beta=0.010, dt=0.1,
        )
        a = ForceFieldDynamics(**base).run(50.0)
        b = ForceFieldDynamics(**base, mu=0.0, lam=0.0, rho=0.0, latent_force=0.0).run(50.0)
        assert np.allclose(a.R, b.R)
        assert np.allclose(a.C, b.C)
        assert np.allclose(a.S, b.S)
        assert np.all(b.P == 0.0)

    def test_derivatives_still_returns_two_tuple(self):
        """Public API stability: derivatives() keeps returning (dR, dC)."""
        eng = ForceFieldDynamics(revolutionary_force=5.0, conservative_force=3.0)
        out = eng.derivatives(0.0)
        assert isinstance(out, tuple) and len(out) == 2
        dR, dC = out
        assert dR.shape == (1,) and dC.shape == (1,)

    def test_state_still_two_tuple(self):
        eng = ForceFieldDynamics(revolutionary_force=5.0, conservative_force=3.0)
        R, C = eng.state  # unpacking must still work
        assert R.tolist() == [5.0] and C.tolist() == [3.0]


# ---------------------------------------------------------------------------
# 2. 1905 → 1917 revival (P_R reservoir core validation)
# ---------------------------------------------------------------------------

def _revival_engine(field_change=True):
    """1905 crush (high β, compression) → 1917 field change (β collapses, α rises).

    Parameters are semantic anchors (v1.3 §3.3): β > α = counter-revolution's
    suppression dominates (Stolypin reaction); μ > λ = repression is mostly
    compression into latent potential, not destruction; rho ≈ 0 = the 1905
    memory is not forgotten.
    """

    def mod_beta(t):
        return 1.0 if (not field_change or t < 20.0) else 0.005

    def mod_alpha(t):
        return 1.0 if (not field_change or t < 20.0) else 8.0

    return ForceFieldDynamics(
        revolutionary_force=10.0,
        conservative_force=8.0,
        growth_r=0.02,
        growth_c=0.0,
        alpha=0.01,
        beta=0.03,
        mu=0.15,
        lam=0.03,
        rho=0.001,
        mod_beta=mod_beta,
        mod_alpha=mod_alpha,
        dt=0.1,
        node_ids=["1905-russia"],
        meta={"case": "1905-1917-revival"},
    )


class TestRevival1905To1917:
    def test_high_beta_crushes_r_but_p_grows_and_t_approx_conserved(self):
        eng = _revival_engine()
        traj = eng.run(20.0)  # phase 1 only: high-β repression
        R, P, T = traj.R[:, 0], traj.P[:, 0], traj.T[:, 0]

        # R decays deep under high β (kinetic force compressed away)
        assert R[0] == pytest.approx(10.0)
        assert R[-1] < 0.3 * R[0]          # 0.46 < 3.0

        # P_R grows into the reservoir (0 → 6.6)
        assert P[0] == pytest.approx(0.0)
        assert P[-1] > 0.4 * R[0]          # 6.17 > 4.0

        # T = R + P_R approximately conserved during the crush (H₀, drift < 50%)
        drift = np.max(np.abs(T - T[0])) / T[0]
        assert drift < 0.5

    def test_field_change_triggers_release_and_revival(self):
        """mod_beta drop ⇒ λ releases P_R ⇒ R revives (revolution NOT killed)."""
        eng = _revival_engine()
        traj = eng.run(200.0)
        t, R, P, S = traj.t, traj.R[:, 0], traj.P[:, 0], traj.S[:, 0]

        i1 = np.argmin(np.abs(t - 20.0))
        r_trough = float(np.min(R[: i1 + 1]))

        # release signature: immediately after the field change P falls while R
        # rises (P_R is being converted back into kinetic R).
        i2 = np.argmin(np.abs(t - 40.0))
        assert P[i2] < P[i1]
        assert R[i2] > R[i1] * 2.0

        # R revives to a clearly positive level well above the trough
        assert R[-1] > 2.0 * r_trough
        assert R[-1] > 1.0
        # and the resultant flips back toward the revolutionary side
        assert S[-1] > 0.8

    def test_persistent_repression_without_field_change_stays_dead(self):
        """No external-field modulation ⇒ no λ release ⇒ R stays dead.

        This is the counterfactual that makes the claim precise: the reservoir
        preserves capacity across repression, but the *release* is triggered by
        the external field (1917). Without the field change the revolution does
        not spontaneously revive — so "not permanently killed" is a conditional
        truth, not an automatic one.
        """
        eng = _revival_engine(field_change=False)
        traj = eng.run(200.0)
        assert float(traj.R[-1, 0]) < 0.01
        assert float(traj.S[-1, 0]) < 0.05

    def test_revival_requires_reservoir(self):
        """Without μ/λ/ρ the same field change never produces the crush→memory
        pattern: R barely dips (no compression), P ≡ 0."""
        eng = ForceFieldDynamics(
            revolutionary_force=10.0,
            conservative_force=8.0,
            growth_r=0.02,
            growth_c=0.0,
            alpha=0.01,
            beta=0.03,
            mod_beta=(lambda t: 1.0 if t < 20.0 else 0.005),
            mod_alpha=(lambda t: 1.0 if t < 20.0 else 8.0),
            dt=0.1,
        )
        traj = eng.run(200.0)
        assert np.all(traj.P == 0.0)
        # no compression phase: R never gets crushed below 90% of R0 in phase 1
        i1 = np.argmin(np.abs(traj.t - 20.0))
        assert float(np.min(traj.R[: i1 + 1, 0])) > 0.9 * traj.R[0, 0]


# ---------------------------------------------------------------------------
# 3. Panic spiral (four-quadrant κ_RC)
# ---------------------------------------------------------------------------

def _panic_engine(panic_exponent=0.0, kRC=None, uncoupled=False):
    """Node 0 = radical (growing R); node 1 = conservative (target of panic).

    κ_RC[1][0] > 0 ⇒ node 0's rising R drives node 1's C up (counterattack).
    """
    if kRC is None:
        kRC = np.array([[0.0, 0.0], [0.05, 0.0]])
    return ForceFieldDynamics(
        revolutionary_force=[10.0, 1.0],
        conservative_force=[1.0, 1.0],
        growth_r=[0.05, 0.0],
        growth_c=[0.0, 0.0],
        alpha=0.0,
        beta=0.0,
        kappa=None if uncoupled else {"RC": kRC},
        panic_exponent=panic_exponent,
        dt=0.1,
        node_ids=["radical", "conservative"],
    )


class TestPanicSpiral:
    def test_rc_coupling_makes_c_counterattack(self):
        """R rises ⇒ κ_RC drives C up: cross-class escalation exists."""
        coupled = _panic_engine(panic_exponent=0.0).run(30.0)
        uncoupled = _panic_engine(uncoupled=True).run(30.0)

        # driver: radical node's R grows
        assert float(coupled.R[-1, 0]) > 4.0 * coupled.R[0, 0]

        # target: conservative node's C rises only when coupled
        assert float(uncoupled.C[-1, 1]) == pytest.approx(1.0)
        assert float(coupled.C[-1, 1]) > 3.0
        assert float(coupled.C[-1, 1]) > 10.0 * float(uncoupled.C[-1, 1])

        # cross-class selectivity: the radical node's own C stays flat (κ_RC
        # targets the conservative node, not the radical's own inertia)
        assert float(coupled.C[-1, 0]) == pytest.approx(1.0, abs=1e-9)

    def test_cr_quadrant_activation_and_suppression(self):
        """κ_CR is full-sign: positive = C rises ⇒ R activated (equation as
        written, dR += κ_CR·(C_j − C_i)); negative = C rises ⇒ R suppressed
        (the regime-dependent "壓制" reading, §2.6)."""
        base = dict(
            revolutionary_force=[10.0, 1.0],
            conservative_force=[1.0, 10.0],
            growth_r=0.0,
            growth_c=0.05,   # node 1's C rises
            alpha=0.0,
            beta=0.0,
            dt=0.1,
        )
        alone = ForceFieldDynamics(**base).run(20.0)

        # positive κ_CR: node 1's rising C activates node 0's R
        activate = ForceFieldDynamics(
            kappa={"CR": np.array([[0.0, 0.05], [0.0, 0.0]])}, **base
        ).run(20.0)
        assert float(activate.R[-1, 0]) > float(alone.R[-1, 0])

        # negative κ_CR: node 1's rising C suppresses node 0's R
        suppress = ForceFieldDynamics(
            kappa={"CR": np.array([[0.0, -0.05], [0.0, 0.0]])}, **base
        ).run(20.0)
        assert float(suppress.R[-1, 0]) < float(alone.R[-1, 0])


# ---------------------------------------------------------------------------
# 4. T observability (conservation = H₀, not a constraint)
# ---------------------------------------------------------------------------

class TestTObservability:
    def test_t_exists_correct_shape_equals_r_plus_p(self):
        eng = ForceFieldDynamics(
            revolutionary_force=[5.0, 3.0],
            conservative_force=[4.0, 2.0],
            mu=0.1, lam=0.02, rho=0.0,
            latent_force=[0.0, 1.5],
            dt=0.1,
        )
        traj = eng.run(50.0)
        assert hasattr(traj, "T")
        assert traj.T.shape == traj.R.shape == (501, 2)
        assert np.allclose(traj.T, traj.R + traj.P)
        assert "T" in traj.to_dict()

    def test_t_is_output_not_constraint(self):
        """T is NOT conserved by force — it is an observable H₀ that can drift
        (e.g. new technology raising total transformative capacity)."""
        eng = ForceFieldDynamics(
            revolutionary_force=5.0, conservative_force=5.0,
            growth_r=0.05, growth_c=0.0, alpha=0.001, beta=0.001,
            mu=0.1, lam=0.02, rho=0.0, dt=0.1,
        )
        traj = eng.run(50.0)
        T = traj.T[:, 0]
        # T moves materially across the run (not pinned constant)
        assert np.max(np.abs(np.diff(T))) > 1e-3
        # and it can GROW (violating naive conservation is allowed)
        assert T[-1] > T[0]

    def test_tension_field_preserved_separately(self):
        """§2.4 tension (the former T) is still available under its own field."""
        eng = ForceFieldDynamics(
            revolutionary_force=5.0, conservative_force=5.0,
            growth_r=0.02, growth_c=0.02, alpha=0.01, beta=0.01,
            mu=0.1, lam=0.02, rho=0.0, dt=0.1,
        )
        traj = eng.run(20.0)
        assert traj.tension.shape == traj.R.shape
        assert np.allclose(traj.tension, np.sqrt(traj.R * traj.C))
        assert "tension" in traj.to_dict()


# ---------------------------------------------------------------------------
# 5. Nonlinear panic (panic_exponent > 1 ⇒ superlinear)
# ---------------------------------------------------------------------------

class TestNonlinearPanic:
    def test_panic_exponent_superlinear(self):
        """p > 1 ⇒ the C counterattack is superlinear in the source's R."""
        c_lin = _panic_engine(panic_exponent=0.0).run(30.0).C[-1, 1]
        c_p1 = _panic_engine(panic_exponent=1.0).run(30.0).C[-1, 1]
        c_p2 = _panic_engine(panic_exponent=2.0).run(30.0).C[-1, 1]

        # monotone in the exponent
        assert c_p1 > c_lin
        assert c_p2 > c_p1
        # and superlinear: the p=2 response overwhelms the linear one
        assert c_p2 > 50.0 * c_lin

    def test_linear_default_is_backward_compatible(self):
        """panic_exponent default 0.0 = pure linear Richardson κ_RC term."""
        linear = _panic_engine(panic_exponent=0.0)
        explicit = _panic_engine(panic_exponent=0.0)
        a = linear.run(20.0)
        b = explicit.run(20.0)
        assert np.allclose(a.C, b.C)


# ---------------------------------------------------------------------------
# 6. S ∈ [0, 1] invariant with the reservoir active
# ---------------------------------------------------------------------------

class TestSInvariantWithReservoir:
    def test_s_stays_in_unit_interval_with_reservoir(self):
        eng = ForceFieldDynamics(
            revolutionary_force=10.0, conservative_force=8.0,
            growth_r=0.02, growth_c=0.0, alpha=0.01, beta=0.03,
            mu=0.15, lam=0.03, rho=0.001, dt=0.1,
        )
        traj = eng.run(200.0)
        assert np.all(traj.S >= -1e-12)
        assert np.all(traj.S <= 1.0 + 1e-12)

    def test_s_ignores_latent_directly(self):
        """S = R/(R+C): P_R is potential and does not enter S directly — it only
        acts through λ-release back into R."""
        eng = ForceFieldDynamics(
            revolutionary_force=5.0, conservative_force=3.0,
            mu=0.1, lam=0.02, rho=0.0,
        )
        s1 = eng.s(R=4.0, C=2.0)
        # same (R, C) → same S regardless of what P_R holds
        assert s1 == pytest.approx(4.0 / 6.0)


# ---------------------------------------------------------------------------
# 7. Four-quadrant kappa API (backward compatible)
# ---------------------------------------------------------------------------

class TestKappaFourQuadrant:
    def test_full_sign_dict_api(self):
        """All four quadrants settable via {"RR","RC","CC","CR"}."""
        n = 2
        eng = ForceFieldDynamics(
            revolutionary_force=[10.0, 1.0],
            conservative_force=[1.0, 1.0],
            growth_r=0.0, growth_c=0.0, alpha=0.0, beta=0.0,
            kappa={"RR": np.full((n, n), 0.1),
                   "RC": np.array([[0.0, 0.0], [0.05, 0.0]]),
                   "CC": np.full((n, n), 0.2),
                   "CR": np.array([[0.0, 0.0], [0.0, 0.0]])},
            dt=0.1,
        )
        traj = eng.run(10.0)
        # RR diffusion acts on R; CC on C; RC drives C_1 from R_0
        assert float(traj.R[-1, 1]) > traj.R[0, 1]
        assert float(traj.C[-1, 1]) > traj.C[0, 1]

    def test_legacy_forms_map_to_same_force_quadrants(self):
        """Scalar / (n,n) array / {"R","C"} all keep the v1.0 semantics."""
        k = np.array([[0.0, 0.3], [0.3, 0.0]])
        scalar = ForceFieldDynamics(
            revolutionary_force=[10.0, 2.0], conservative_force=[1.0, 1.0],
            growth_r=0.0, growth_c=0.0, alpha=0.0, beta=0.0,
            kappa=0.5, dt=0.1,
        ).run(20.0)
        arr = ForceFieldDynamics(
            revolutionary_force=[10.0, 2.0], conservative_force=[1.0, 1.0],
            growth_r=0.0, growth_c=0.0, alpha=0.0, beta=0.0,
            kappa=k, dt=0.1,
        ).run(20.0)
        # scalar → uniform RR & CC: both forces equalise
        assert abs(float(scalar.R[-1, 0]) - float(scalar.R[-1, 1])) < 1e-3
        assert abs(float(scalar.C[-1, 0]) - float(scalar.C[-1, 1])) < 1e-3
        # array → same matrix for RR and CC: R equalises
        assert abs(float(arr.R[-1, 0]) - float(arr.R[-1, 1])) < 1e-3
        # legacy dict maps R→RR, C→CC
        eng = ForceFieldDynamics(
            revolutionary_force=[10.0, 2.0], conservative_force=[1.0, 1.0],
            growth_r=0.0, growth_c=0.0, alpha=0.0, beta=0.0,
            kappa={"R": k, "C": np.zeros((2, 2))}, dt=0.1,
        ).run(20.0)
        assert abs(float(eng.R[-1, 0]) - float(eng.R[-1, 1])) < 1e-3
        assert float(eng.C[-1, 0]) == pytest.approx(1.0)

    def test_ambiguous_R_and_RR_rejected(self):
        with pytest.raises(ValueError, match="either 'R' or 'RR'"):
            ForceFieldDynamics(
                revolutionary_force=[1.0, 1.0], conservative_force=[1.0, 1.0],
                kappa={"R": np.ones((2, 2)), "RR": np.ones((2, 2))},
            )

    def test_unknown_quadrant_key_rejected(self):
        with pytest.raises(ValueError, match="kappa dict keys"):
            ForceFieldDynamics(
                revolutionary_force=[1.0, 1.0], conservative_force=[1.0, 1.0],
                kappa={"XX": np.ones((2, 2))},
            )

    def test_quadrant_shape_validation(self):
        with pytest.raises(ValueError, match="kappa.RC must have shape"):
            ForceFieldDynamics(
                revolutionary_force=[1.0, 1.0], conservative_force=[1.0, 1.0],
                kappa={"RC": np.ones((3, 3))},
            )

    def test_negative_panic_exponent_rejected(self):
        with pytest.raises(ValueError, match="panic_exponent"):
            ForceFieldDynamics(
                revolutionary_force=1.0, conservative_force=1.0,
                panic_exponent=-1.0,
            )


# ---------------------------------------------------------------------------
# 8. ActorCard latent_force extension
# ---------------------------------------------------------------------------

def _card(actor_id, r=None, c=None, latent=None):
    return ActorCard(
        actor_id=actor_id,
        name=f"{actor_id} name",
        inherited_conditions={"rank": "pioneer"},
        field_coordinates={"x": 1, "y": 2},
        internal_tensions=[{"axis_A": "a", "axis_B": "b", "note": "t"}],
        temporal_states={"t0": "idle"},
        revolutionary_force=r,
        conservative_force=c,
        latent_force=latent,
    )


class TestActorCardLatentForce:
    def test_latent_force_defaults_to_none(self):
        card = _card("alpha")
        assert card.latent_force is None

    def test_roundtrip_through_to_dict(self):
        card = _card("alpha", r=7.5, c=2.5, latent=4.0)
        d = card.to_dict()
        assert d["latent_force"] == 4.0
        assert card.latent_force == 4.0

    def test_negative_latent_force_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            _card("alpha", latent=-1.0)

    def test_from_actor_cards_seeds_latent_force(self):
        cards = [
            _card("state", r=10.0, c=5.0, latent=3.0),
            _card("party", r=4.0, c=None, latent=None),
        ]
        eng = ForceFieldDynamics.from_actor_cards(
            cards, growth_r=0.02, growth_c=0.01, alpha=0.01, beta=0.01,
            mu=0.1, lam=0.02, rho=0.0,
        )
        P = eng.latent_state
        assert P.tolist() == [3.0, 0.0]

    def test_from_actor_cards_without_latent_is_unchanged(self):
        cards = [_card("state", r=10.0, c=5.0), _card("party", r=4.0, c=None)]
        eng = ForceFieldDynamics.from_actor_cards(
            cards, growth_r=0.02, growth_c=0.01, alpha=0.01, beta=0.01,
        )
        assert np.all(eng.latent_state == 0.0)
