"""Tests for the opt-in 5-D fast/slow state vector (Φ, K) + double-speed ODE.

Source of truth: ECC ``index/locations/data/almanac/THEORY-LEDGER.md``
§「完整動態方程組」 (2026-09-08 agy 裁定):

    dR/dt   = aR − δR + Φ·μ_F·R + (1−η)λP_R − βC
    dC/dt   = κ_C(1−Φ) − γ_C·σ_soviet·C
    dP_R/dt = S_in·R − λP_R − ημ_E·P_R
    dΦ/dt   = κ_Φ·CoopShare·(1−Φ) − ζ_Φ·DebtStress·Φ
    dK/dt   = ημ_E·P_R − γ_K·K

plus mixed operator splitting (春耕鉗 / 秋收兌現) and the ledger's three
stability predicates, plus the THEORY-LEDGER minimal patch (2026-09-10 agy
裁定) restoring the force-model v1.6 "R suppresses C" channel in the C
equation as the bilinear contact-surface term:

    dC/dt += −α_rc·R·C        (fast-channel mode only; α_rc [UNCALIBRATED])

Covers:
1. Backward compatibility — legacy path byte-for-byte identical (frozen golden
   digests captured from the pre-change code).
2. Φ saturation / decay (CoopShare drive vs DebtStress back-pressure) and the
   Φ ∈ [0, 1] invariant.
3. The three stability predicates (convergence / supercritical bifurcation /
   oscillatory divergence), each with a triggerable case.
4. K accumulation and depreciation.
5. Seasonal operator splitting (spring clamp immunity at Φ→1, autumn pay-out).
6. The 5-D RHS against a hand-computed evaluation.
7. The contact-surface suppression patch −α_rc·R·C (sign/magnitude, internal
   steady state, legacy invariance, α_rc=0 degeneration).
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from spectrum_os.kernel.forces import (
    FastChannelConfig,
    ForceFieldDynamics,
    StabilityRegime,
)

# Frozen SHA-256 digests over (t, R, C, P, S, T, tension) of the legacy
# engine, captured from the pre-change code (v1.6). These must never change:
# the opt-in fast channel may not perturb the 2/3-D path.
_GOLDEN_A = "c69f6b4f454a8478c0a0e5da5d165a9682a2a213ada42fc726d5e227b5344990"
_GOLDEN_B = "69fda23c838c4f22d3f2f869e88c1ce012c76369f74b3542255ec4fcfad6b9e6"
_GOLDEN_B_FINAL_S = 0.852642141900274


def _digest(traj) -> str:
    h = hashlib.sha256()
    for arr in (traj.t, traj.R, traj.C, traj.P, traj.S, traj.T, traj.tension):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _legacy_rich() -> ForceFieldDynamics:
    return ForceFieldDynamics(
        [1.2, 0.8], [0.5, 0.9],
        growth_r=[0.02, 0.03], growth_c=[0.01, 0.02],
        alpha=[0.05, 0.04], beta=[0.03, 0.02],
        mu=[0.02, 0.01], lam=0.15, rho=0.005, latent_force=[0.1, 0.0],
        kappa={"RR": [[0.0, 0.3], [0.3, 0.0]], "CC": [[0.0, 0.1], [0.1, 0.0]],
               "RC": [[0.0, 0.2], [0.2, 0.0]], "CR": [[0.0, 0.15], [0.15, 0.0]]},
        panic_exponent=0.5, method="rk4", dt=0.25,
    )


def _legacy_simple() -> ForceFieldDynamics:
    return ForceFieldDynamics(
        1.5, 0.4, growth_r=0.02, growth_c=0.01, alpha=0.05, beta=0.03,
        mu=0.02, lam=0.15, method="euler", dt=0.25,
    )


def _fast(**kw) -> ForceFieldDynamics:
    """A single-node engine in the opt-in 5-D mode (sane defaults)."""
    base = dict(
        phi0=0.65, phi_drive=0.65, phi_suppress=0.5,
        kappa_phi=1.5, zeta_phi=1.0, delta=0.0, mu_f=0.12, eta=0.0, lam=0.15,
        s_in=0.02, mu_e=0.0, kappa_c=0.05, gamma_c=0.10, sigma_admin=1.0,
        k0=0.0, gamma_k=0.0, clamp_spring=0.0, autumn_yield=0.0,
        reflow_purity=1.0, surplus_gain=0.0,
    )
    base.update(kw)
    return ForceFieldDynamics(
        1.0, 0.5, growth_r=0.02, growth_c=0.01, alpha=0.05, beta=0.03,
        fast_channel=FastChannelConfig(**base), dt=0.25,
    )


class TestBackwardCompatibility:
    """The opt-in mode must not perturb the legacy 2/3-D path at all."""

    def test_legacy_golden_digest_rich(self):
        assert _digest(_legacy_rich().run(3.0)) == _GOLDEN_A

    def test_legacy_golden_digest_simple(self):
        traj = _legacy_simple().run(2.0)
        assert _digest(traj) == _GOLDEN_B
        assert traj.final_s() == _GOLDEN_B_FINAL_S

    def test_fast_disabled_bit_identical(self):
        """fast_channel=None ≡ default; output arrays bit-identical."""
        a = _legacy_rich().run(2.0)
        eng = ForceFieldDynamics(
            [1.2, 0.8], [0.5, 0.9],
            growth_r=[0.02, 0.03], growth_c=[0.01, 0.02],
            alpha=[0.05, 0.04], beta=[0.03, 0.02],
            mu=[0.02, 0.01], lam=0.15, rho=0.005, latent_force=[0.1, 0.0],
            kappa={"RR": [[0.0, 0.3], [0.3, 0.0]],
                   "CC": [[0.0, 0.1], [0.1, 0.0]],
                   "RC": [[0.0, 0.2], [0.2, 0.0]],
                   "CR": [[0.0, 0.15], [0.15, 0.0]]},
            panic_exponent=0.5, method="rk4", dt=0.25, fast_channel=None,
        )
        b = eng.run(2.0)
        for field in ("t", "R", "C", "P", "S", "T", "tension"):
            assert np.array_equal(getattr(a, field), getattr(b, field))

    def test_legacy_trajectory_has_no_fast_keys(self):
        traj = _legacy_simple().run(1.0)
        assert traj.phi is None
        assert traj.k_pool is None
        assert traj.final_phi() is None
        payload = traj.to_dict()
        assert "phi" not in payload
        assert "k_pool" not in payload
        assert "fast_channel" not in payload["meta"]
        json.dumps(payload)  # JSON-safe

    def test_fast_disabled_state_and_derivatives(self):
        eng = ForceFieldDynamics(1.0, 0.5)
        assert eng.fast_channel_enabled is False
        assert eng.phi is None
        assert eng.k_pool is None
        assert eng.fast_state() is None
        assert eng.five_dim_state() is None
        assert eng.fast_config is None
        with pytest.raises(RuntimeError):
            eng.derivatives_5d(0.0)
        with pytest.raises(RuntimeError):
            eng.assess_stability()


class TestOperatorSplitAndState:
    def test_phi_initial_and_bounds_hold(self):
        eng = _fast(phi0=0.99, phi_drive=2.0, phi_suppress=0.0)
        assert eng.phi is not None and eng.phi[0] == pytest.approx(0.99)
        traj = eng.run(3.0)
        assert np.all(traj.phi >= 0.0) and np.all(traj.phi <= 1.0)
        # Φ saturates upward (drive >> back-pressure)
        assert traj.final_phi() > 0.999

    def test_to_dict_and_meta_json_safe(self):
        traj = _fast().run(1.0)
        payload = traj.to_dict()
        assert "phi" in payload and "k_pool" in payload
        json.dumps(payload)
        assert payload["meta"]["fast_channel"]["enabled"] is True
        prov = payload["meta"]["fast_channel"]["provenance"]
        # every declared config field carries a provenance tag
        for f in FastChannelConfig.__dataclass_fields__:
            assert f in prov, f"missing provenance tag for {f}"

    def test_reset_restores_phi_and_k(self):
        eng = _fast(phi0=0.4, eta=0.5, mu_e=0.3, s_in=0.05, gamma_k=0.0)
        eng.run(2.0)
        assert float(eng.phi[0]) != pytest.approx(0.4)
        eng.reset()
        phi, k = eng.fast_state()
        assert phi[0] == pytest.approx(0.4)
        assert k[0] == 0.0

    def test_spring_clamp_immunity_at_phi_one(self):
        """Φ→1 ⇒ the spring clamp is fully immunised (ledger)."""
        common = dict(phi0=1.0, phi_drive=1.0, phi_suppress=0.0, mu_f=0.0,
                      s_in=0.0, kappa_c=0.0, gamma_c=0.0, autumn_yield=0.0,
                      surplus_gain=0.0, clamp_spring=1.0)
        eng_clamped = _fast(**common)
        eng_free = _fast(**{**common, "clamp_spring": 0.0})
        t_clamped = eng_clamped.run(1.0)
        t_free = eng_free.run(1.0)
        assert np.array_equal(t_clamped.R, t_free.R)

    def test_spring_clamp_zeroes_r_at_phi_zero(self):
        """Φ=0 + Clamp_spring=1 ⇒ R is annihilated at the spring boundary."""
        eng = _fast(phi0=0.0, phi_drive=0.0, phi_suppress=0.0, clamp_spring=1.0)
        eng.run(0.25)  # lands exactly on the spring event
        assert float(eng.state[0][0]) == pytest.approx(0.0)

    def test_autumn_payout_adds_r_and_p(self):
        # Φ held exactly at its equilibrium 0.65 (phi_suppress=0.525) so the
        # autumn injection is analytically known.
        base = dict(phi0=0.65, phi_drive=0.65, phi_suppress=0.525,
                    mu_f=0.0, s_in=0.0, kappa_c=0.0, gamma_c=0.0,
                    clamp_spring=0.0, surplus_gain=0.0)
        eng_none = _fast(**{**base, "autumn_yield": 0.0})
        eng_pay = _fast(**{**base, "autumn_yield": 0.05, "surplus_gain": 0.02})
        t_none = eng_none.run(0.75)
        t_pay = eng_pay.run(0.75)
        # Continuous flow is identical; the only delta is the autumn operator.
        assert t_pay.R[-1, 0] == pytest.approx(t_none.R[-1, 0] + 0.05 * 0.65)
        assert (float(t_pay.P[-1, 0]) - float(t_none.P[-1, 0])) == pytest.approx(0.02)


class TestPhiDynamics:
    def test_phi_saturates_under_high_drive(self):
        eng = _fast(phi0=0.65, phi_drive=1.0, phi_suppress=0.0)
        traj = eng.run(5.0)
        assert traj.final_phi() > 0.999

    def test_phi_decays_under_high_suppress(self):
        eng = _fast(phi0=0.65, phi_drive=0.0, phi_suppress=1.0, zeta_phi=1.0)
        traj = eng.run(3.0)
        assert traj.final_phi() < 0.05
        # strictly decreasing
        assert np.all(np.diff(traj.phi[:, 0]) < 0.0)

    def test_phi_equilibrium_matches_logistic(self):
        cfg = FastChannelConfig(phi0=0.1, phi_drive=0.7, phi_suppress=0.2,
                                kappa_phi=1.5, zeta_phi=1.0)
        eng = ForceFieldDynamics(1.0, 0.5, fast_channel=cfg, dt=0.25)
        phi_star = 1.5 * 0.7 / (1.5 * 0.7 + 1.0 * 0.2)
        final = eng.run(20.0).final_phi()
        assert final == pytest.approx(phi_star, abs=1e-6)

    def test_substrate_driver_callables(self):
        """Φ drivers may be world-supplied callables (v1.6 substrate pattern)."""
        seen: list[tuple[float, dict]] = []

        def drive_fn(t, ctx):
            seen.append((t, ctx))
            # drive depends on the world observable S — legal, kernel-agnostic
            return 1.0 * ctx["S"]

        def suppress_fn(t, ctx):
            return 0.25 * (1.0 - ctx["S"])

        cfg = FastChannelConfig(phi0=0.5, phi_drive_fn=drive_fn,
                                phi_suppress_fn=suppress_fn, kappa_phi=1.0,
                                zeta_phi=1.0)
        eng = ForceFieldDynamics(1.0, 0.5, fast_channel=cfg, dt=0.25)
        traj = eng.run(1.0)
        assert seen, "driver callables were never evaluated"
        # context carries exactly the universal observables
        assert set(seen[0][1]) == {"R", "C", "P", "Phi", "K", "S", "T"}
        assert np.all(traj.phi >= 0.0) and np.all(traj.phi <= 1.0)

    def test_scalar_and_callable_driver_agree(self):
        """A constant callable ≡ the scalar slot (same law, two spellings)."""
        const = 0.4
        a = ForceFieldDynamics(
            1.0, 0.5, dt=0.25,
            fast_channel=FastChannelConfig(phi0=0.3, phi_drive=const,
                                           phi_suppress=0.2))
        b = ForceFieldDynamics(
            1.0, 0.5, dt=0.25,
            fast_channel=FastChannelConfig(
                phi0=0.3, phi_drive_fn=lambda t, ctx: const,
                phi_suppress_fn=lambda t, ctx: 0.2))
        assert np.allclose(a.run(3.0).phi, b.run(3.0).phi, rtol=0, atol=0)


class TestStabilityCriteria:
    def test_convergent_regime(self):
        """βC* + δ dominates the growth/fast channel → convergent, not escape."""
        cfg = FastChannelConfig(phi_drive=0.0, phi_suppress=1.0, kappa_c=1.0,
                                gamma_c=1.0, sigma_admin=1.0, delta=0.1,
                                mu_f=0.0)
        eng = ForceFieldDynamics(1.0, 0.5, growth_r=0.0, beta=0.5,
                                 fast_channel=cfg)
        v = eng.assess_stability()
        assert isinstance(v, StabilityRegime)
        assert v.convergent is True
        assert v.escape_poverty_trap is False
        assert np.all(v.convergence_margin > 0)

    def test_supercritical_bifurcation_regime(self):
        """Φ*·μ_F + a − δ > βC* with λ > 0 → escape the poverty trap."""
        cfg = FastChannelConfig(phi_drive=1.0, phi_suppress=0.0, mu_f=0.5,
                                lam=0.15, kappa_c=0.0, gamma_c=0.0)
        eng = ForceFieldDynamics(1.0, 0.5, growth_r=0.05, beta=0.0,
                                 fast_channel=cfg)
        v = eng.assess_stability()
        assert v.escape_poverty_trap is True
        assert v.convergent is False
        assert np.all(v.bifurcation_margin > 0)

    def test_bifurcation_requires_lambda(self):
        """Zero λ still allows the margin but not the reservoir-release escape."""
        cfg = FastChannelConfig(phi_drive=1.0, phi_suppress=0.0, mu_f=0.5,
                                lam=0.0, kappa_c=0.0, gamma_c=0.0)
        eng = ForceFieldDynamics(1.0, 0.5, growth_r=0.05, beta=0.0,
                                 fast_channel=cfg)
        v = eng.assess_stability()
        assert np.all(v.bifurcation_margin > 0)
        assert v.escape_poverty_trap is False

    def test_oscillatory_divergence_triggered(self):
        cfg = FastChannelConfig(phi_drive=0.65, phi_suppress=0.5, mu_f=0.12,
                                clamp_spring=0.5, autumn_yield=2.0,
                                reflow_purity=1.0)
        eng = ForceFieldDynamics(1.0, 0.5, growth_r=0.02, beta=0.0,
                                 fast_channel=cfg)
        v = eng.assess_stability()
        assert v.oscillatory_divergence is True
        assert np.all(v.seasonal_loop_gain > 1.0)

    def test_oscillation_needs_both_clamp_and_yield(self):
        base = dict(phi_drive=0.65, phi_suppress=0.5, mu_f=0.12)
        no_clamp = FastChannelConfig(**base, clamp_spring=0.0, autumn_yield=2.0)
        no_yield = FastChannelConfig(**base, clamp_spring=0.5, autumn_yield=0.0)
        for cfg in (no_clamp, no_yield):
            eng = ForceFieldDynamics(1.0, 0.5, growth_r=0.02, beta=0.0,
                                     fast_channel=cfg)
            assert eng.assess_stability().oscillatory_divergence is False


class TestAlienatedCapitalPool:
    def test_k_accumulates_when_eta_positive(self):
        cfg = FastChannelConfig(eta=0.5, mu_e=0.2, s_in=0.05, lam=0.15,
                                gamma_k=0.0, phi_drive=0.65, phi_suppress=0.5)
        eng = ForceFieldDynamics(1.0, 0.5, growth_r=0.02, beta=0.03,
                                 fast_channel=cfg, dt=0.25)
        traj = eng.run(3.0)
        assert traj.k_pool[-1, 0] > 0.0
        assert np.all(np.diff(traj.k_pool[:, 0]) > 0.0)

    def test_k_depreciates_when_inflow_zero(self):
        cfg = FastChannelConfig(k0=1.0, gamma_k=0.5, eta=0.0, mu_e=0.0,
                                s_in=0.0)
        eng = ForceFieldDynamics(1.0, 0.5, fast_channel=cfg)
        traj = eng.run(3.0)
        assert traj.k_pool[-1, 0] == pytest.approx(np.exp(-0.5 * 3.0), abs=1e-3)
        assert np.all(np.diff(traj.k_pool[:, 0]) < 0.0)


class TestCompressionChannel:
    """§2.2b/v1.6: repression COMPRESSES force into latent potential — it does
    not destroy it. βC is one transfer (R → P_R), so T = R + P_R is conserved
    against it.

    Evidence is α history, and the compression channel exists to represent it:
    the Canton Commune (Dec 1927) was crushed within days, yet the class forces
    it laid bare re-emerged in 1927–49; the 1905 defeat (Stolypin reaction) fed
    1917. Without this channel the high-β case decays to a dead-lock attractor
    and revival is unrepresentable.
    """

    def _pressed(self) -> ForceFieldDynamics:
        """5-D engine with only the suppression/reservoir channels live."""
        return ForceFieldDynamics(
            1.0, 1.0, growth_r=0.0, alpha=0.0, beta=0.40, dt=0.25,
            fast_channel=FastChannelConfig(
                phi0=0.65, phi_drive=0.0, phi_suppress=0.0,
                mu_f=0.0, s_in=0.0, kappa_c=0.0, gamma_c=0.0,
                alpha_rc=0.0, eta=0.0, mu_e=0.0, gamma_k=0.0,
                delta=0.0, lam=0.15, k_dissolve=0.0,
            ),
        )

    def test_bc_is_one_transfer_not_a_loss(self):
        """dR + dP_R == 0 against the βC term alone (exact conservation)."""
        eng = self._pressed()
        # P=0 isolates βC: the λ·P release term (also a transfer) vanishes.
        dR, _dC, dP, _dPhi, _dK = eng.derivatives_5d(
            0.0, R=1.0, C=0.9, P=0.0, Phi=0.65, K=0.0)
        assert float(dR[0]) == pytest.approx(-0.40 * 0.9)
        assert float(dP[0]) == pytest.approx(+0.40 * 0.9)
        assert float(dR[0] + dP[0]) == pytest.approx(0.0, abs=1e-12)

    def test_repression_credits_the_reservoir(self):
        eng = self._pressed()
        traj = eng.run(4.0)
        assert traj.R[-1, 0] < traj.R[0, 0]          # kinetic force crushed
        assert float(eng._P_pool[0]) > 0.0           # ...into latent potential

    def test_suppressed_force_revives_when_pressure_lifts(self):
        """The 1905→1917 / Canton shape: compress, then release."""
        eng = self._pressed()
        eng.run(4.0)
        R_pressed = float(eng.state[0][0])
        P_pressed = float(eng._P_pool[0])
        T_pressed = R_pressed + P_pressed

        eng.set_modulation(mod_beta=lambda t: 0.0)   # pressure lifts
        eng.run(16.0)
        R_revived = float(eng.state[0][0])
        P_left = float(eng._P_pool[0])

        assert R_revived > R_pressed, "latent force must revive"
        assert P_left < P_pressed, "the reservoir is the source of the revival"
        # Total capacity survives the whole cycle (a=0, ρ=0, μ_F=0).
        assert R_revived + P_left == pytest.approx(T_pressed, rel=1e-9)

    def test_compression_is_not_lanchester_destruction(self):
        """A dead-lock attractor would leave nothing to revive from."""
        eng = self._pressed()
        eng.run(4.0)
        assert float(eng.state[0][0]) == pytest.approx(0.0, abs=1e-9)
        assert float(eng._P_pool[0]) > 1.0, "capacity is latent, not destroyed"


    def test_spontaneous_compression_mu_and_forgetting_rho(self):
        """§2.2b: μ (self-organisation into latent form) and ρ (genuinely
        forgotten history) must act in the 5-D path, not only in 3-D.

        μ is one transfer (R → P_pool) so T stays conserved; ρ is the only
        genuinely dissipative channel. ρ is declared to drain the centralized
        reserve only — P_dist is living capacity, not idle stock.
        """
        def build(mu=0.0, rho=0.0):
            return ForceFieldDynamics(
                1.0, 0.5, growth_r=0.0, alpha=0.0, beta=0.0, mu=mu, rho=rho,
                dt=0.25,
                fast_channel=FastChannelConfig(
                    phi0=0.65, phi_drive=0.0, phi_suppress=0.0, mu_f=0.0,
                    s_in=0.0, kappa_c=0.0, gamma_c=0.0, alpha_rc=0.0,
                    eta=0.0, mu_e=0.0, gamma_k=0.0, delta=0.0, lam=0.0,
                    k_dissolve=0.0,
                ),
            )

        # μ alone: R compresses into P_pool, T conserved
        # (derivatives_morphology addresses P_pool/P_dist directly)
        eng = build(mu=0.05)
        dR, _dC, dPp, _dPd, _dPhi, _dK, _dM = eng.derivatives_morphology(
            0.0, R=1.0, C=0.5, P_pool=0.0, P_dist=0.0, Phi=0.65, K=0.0, M=0.0)
        assert float(dR[0]) == pytest.approx(-0.05)
        assert float(dPp[0]) == pytest.approx(+0.05)
        assert float(dR[0] + dPp[0]) == pytest.approx(0.0, abs=1e-12)

        # ρ alone: the pool decays (the one true loss)
        eng = build(rho=0.10)
        _dR, _dC, dPp, _dPd, _dPhi, _dK, _dM = eng.derivatives_morphology(
            0.0, R=0.0, C=0.0, P_pool=0.8, P_dist=0.5, Phi=0.65, K=0.0, M=0.0)
        assert float(dPp[0]) == pytest.approx(-0.10 * 0.8)
        assert float(_dPd[0]) == pytest.approx(0.0), "P_dist is not decayed"

        # No ρ, no other action ⇒ a pool built by μ does NOT evaporate
        eng = build(mu=0.05, rho=0.0)
        eng.run(4.0)
        assert float(eng._P_pool[0]) > 0.0
        assert float(eng.state[0][0]) < 1.0


class TestCSelfReproduction:
    """GENERAL-PRINCIPLES §5 (constitutional): "兩者各有內部增長" — BOTH forces
    grow internally. C is institutional reproduction, so it must reproduce
    itself; it is not merely sourced from blocked circulation. §2.3b hands the
    *form* of that growth to the substrate (``growth_c_fn``).

    The ledger's κ_C·(1−Φ) is kept as an ADDITIVE blockage source: friction does
    not vanish merely because circulation was repaired while institutions keep
    reproducing themselves.
    """

    def _engine(self, *, growth_c=0.05, growth_c_fn=None, c_source_fn=None,
                kappa_c=0.0, phi0=0.65, alpha_rc=0.0, gamma_c=0.0):
        """Isolate the C channel: no suppression, no dissipation."""
        return ForceFieldDynamics(
            1.0, 0.5, growth_r=0.0, growth_c=growth_c, alpha=0.0, beta=0.0,
            growth_c_fn=growth_c_fn, dt=0.25,
            fast_channel=FastChannelConfig(
                phi0=phi0, phi_drive=0.0, phi_suppress=0.0,
                kappa_c=kappa_c, gamma_c=gamma_c, alpha_rc=alpha_rc,
                sigma_admin=1.0, mu_f=0.0, s_in=0.0, lam=0.0,
                eta=0.0, mu_e=0.0, gamma_k=0.0, delta=0.0, k_dissolve=0.0,
                c_source_fn=c_source_fn,
            ),
        )

    def test_c_grows_without_any_blockage(self):
        """κ_C=0 ⇒ no blockage source, yet C must still grow (constitutional)."""
        eng = self._engine(growth_c=0.05, kappa_c=0.0)
        traj = eng.run(3.0)
        assert traj.C[-1, 0] > traj.C[0, 0]
        # Nothing else acts on C ⇒ pure exponential: C(3) = 0.5·e^{0.15}
        assert traj.C[-1, 0] == pytest.approx(0.5 * np.exp(0.05 * 3.0), rel=1e-6)

    def test_growth_c_fn_attaches_in_5d(self):
        """§2.3b avalanche/desertion feedback must reach the 5-D C equation."""
        seen: list[tuple[np.ndarray, np.ndarray]] = []

        def avalanche(S, T):
            seen.append((S, T))
            return 2.0          # constant multiplier ⇒ c_eff = 2·growth_c

        eng = self._engine(growth_c=0.05, growth_c_fn=avalanche)
        eng.run(1.0)
        assert seen, "growth_c_fn was never evaluated in the 5-D path"
        assert seen[0][0].shape == (1,)     # receives per-node S
        assert float(eng.state[1][0]) == pytest.approx(0.5 * np.exp(0.1), rel=1e-6)

    def test_blockage_source_is_additive_not_a_replacement(self):
        """Repairing circulation must not switch institutional growth off."""
        clean = self._engine(growth_c=0.05, kappa_c=0.0, phi0=0.5)
        blocked = self._engine(growth_c=0.05, kappa_c=0.4, phi0=0.5)
        c_clean = clean.run(2.0).C[-1, 0]
        c_blocked = blocked.run(2.0).C[-1, 0]
        assert c_blocked > c_clean, "blockage must add to self-reproduction"
        # ...and self-reproduction alone still grows (Φ fixed, no repair path)
        assert c_clean > 0.5

    def test_c_source_fn_replaces_the_whole_law(self):
        """A world may declare its own institutional-reproduction law."""
        def constant_source(t, ctx):
            return np.full_like(ctx["C"], 0.4)

        eng = self._engine(growth_c=0.05, c_source_fn=constant_source)
        eng.run(1.0)
        # Pure constant influx, self-reproduction suppressed ⇒ C = 0.5 + 0.4t
        assert float(eng.state[1][0]) == pytest.approx(0.9, rel=1e-6)

    def test_c_source_fn_context_is_the_universal_observables(self):
        seen: list[dict] = []

        def probe(t, ctx):
            seen.append(dict(ctx))
            return np.full_like(ctx["C"], 0.1)

        self._engine(c_source_fn=probe).run(0.5)
        assert seen
        assert set(seen[0]) == {"R", "C", "P", "Phi", "K", "S", "T"}


class TestFiveDimRHS:
    def test_rhs_matches_hand_computation(self):
        cfg = FastChannelConfig(
            phi0=0.6, phi_drive=0.7, phi_suppress=0.2, kappa_phi=1.5,
            zeta_phi=1.0, delta=0.01, mu_f=0.12, eta=0.2, lam=0.15,
            s_in=0.05, mu_e=0.3, kappa_c=0.05, gamma_c=0.1,
            alpha_rc=0.05, sigma_admin=1.2, gamma_k=0.4,
        )
        eng = ForceFieldDynamics(1.0, 0.5, growth_r=0.02, beta=0.03,
                                 fast_channel=cfg)
        dR, dC, dP, dPhi, dK = eng.derivatives_5d(
            0.0, R=1.0, C=0.5, P=0.2, Phi=0.6, K=0.1)
        # dR = aR − δR + Φμ_F R + (1−η)λP − βC
        assert float(dR[0]) == pytest.approx(
            0.02 - 0.01 + 0.6 * 0.12 + (1 - 0.2) * 0.15 * 0.2 - 0.03 * 0.5)
        # dC = κ_C(1−Φ) − α_rc·R·C − γ_C·σ·C  (contact-surface patch included)
        assert float(dC[0]) == pytest.approx(
            0.05 * (1 - 0.6) - 0.05 * 1.0 * 0.5 - 0.1 * 1.2 * 0.5)
        assert float(dP[0]) == pytest.approx(
            0.05 - 0.15 * 0.2 - 0.2 * 0.3 * 0.2 + 0.03 * 0.5)  # incl. βC → P_R
        assert float(dPhi[0]) == pytest.approx(
            1.5 * 0.7 * (1 - 0.6) - 1.0 * 0.2 * 0.6)
        assert float(dK[0]) == pytest.approx(0.2 * 0.3 * 0.2 - 0.4 * 0.1)

    def test_fast_channel_accepts_true_and_dict(self):
        e_true = ForceFieldDynamics(1.0, 0.5, fast_channel=True)
        assert e_true.fast_channel_enabled is True
        e_dict = ForceFieldDynamics(1.0, 0.5, fast_channel={"phi0": 0.3})
        assert e_dict.phi[0] == pytest.approx(0.3)

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            FastChannelConfig(year_period=0.0)
        with pytest.raises(ValueError):
            FastChannelConfig(spring_phase=0.5, autumn_phase=0.5)
        with pytest.raises(ValueError):
            ForceFieldDynamics(1.0, 0.5,
                               fast_channel=FastChannelConfig(eta=1.5))
        with pytest.raises(ValueError):
            ForceFieldDynamics(1.0, 0.5,
                               fast_channel=FastChannelConfig(phi0=1.5))
        with pytest.raises(ValueError):
            ForceFieldDynamics(1.0, 0.5, fast_channel=123)


class TestHarnessSmoke:
    """The standalone ΔNSPV harness must import and run (synthetic scenario)."""

    @staticmethod
    def _load():
        path = (Path(__file__).resolve().parents[1]
                / "scripts" / "fast_channel_dnspv_harness.py")
        spec = importlib.util.spec_from_file_location("_fc_harness", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_harness_measure_runs(self):
        mod = self._load()
        d = mod._synthetic()
        m = mod.measure(d)
        assert m["legacy"]["delta_NSPV"] > 0
        assert m["fast"]["delta_NSPV"] >= m["legacy"]["delta_NSPV"]
        assert 0.0 <= m["fast"]["phi_iv_final"] <= 1.0
        # provenance is declared for the ledger-anchored parameters
        assert mod.PROVENANCE["mu_f"].startswith("EXPERIMENTAL")
        assert mod.PROVENANCE["lam"].startswith("CALIBRATED")


class TestContactSurfaceSuppression:
    """THEORY-LEDGER minimal patch (2026-09-10): dC/dt += −α_rc·R·C.

    Restores the force-model v1.6 "R suppresses C" channel (dC/dt = cC − αR)
    inside the fast-channel C equation as a bilinear contact term. Four
    assertions:
    ① sign + magnitude — higher R ⇒ strictly faster C decay, at the analytic
      rate γ_C·σ + α_rc·R (R frozen);
    ② C has an internal steady state C* = κ_C(1−Φ*)/(γ_C·σ + α_rc·R) —
      bounded, non-negative, no divergence;
    ③ legacy invariance — fast_channel=None never sees the term (dC
      independent of R; golden digests byte-identical);
    ④ α_rc = 0 degenerates exactly to the pre-patch ledger equation
      (frozen pre-patch digests), and the term vanishes at R = 0.
    """

    # Frozen SHA-256 digests captured from the PRE-PATCH code (before the
    # −α_rc·R·C term existed). α_rc=0 / R=0 remove the term exactly, so these
    # digests must reproduce bit-for-bit forever.
    _PREPATCH_ALPHA0 = "e445ffdb3b83a4eb125af436121d4916c1b768a24dc3d8563ccb4112aeaa2096"
    _PREPATCH_R0 = "21fe51073c3bd4dc6928736599ef98192ff51b78778f974a266b640fa2cb3305"

    @staticmethod
    def _frozen_r_engine(r0, alpha_rc=0.1, dt=0.05, **cfg_kw):
        """Engine with R frozen (a=δ=μ_F=β=0, no P_R inflow) so dC is analytic."""
        base = dict(kappa_c=0.0, gamma_c=0.10, sigma_admin=1.0, mu_f=0.0,
                    s_in=0.0, lam=0.15, clamp_spring=0.0, autumn_yield=0.0)
        base.update(cfg_kw)
        return ForceFieldDynamics(
            r0, 1.0, growth_r=0.0, growth_c=0.0, alpha=0.0, beta=0.0,
            fast_channel=FastChannelConfig(alpha_rc=alpha_rc, **base),
            method="rk4", dt=dt)

    # ── ① sign and magnitude ─────────────────────────────────────────────
    def test_higher_r_decays_c_faster(self):
        lo = self._frozen_r_engine(0.5).run(2.0)
        hi = self._frozen_r_engine(2.0).run(2.0)
        assert np.all(np.diff(lo.C[:, 0]) < 0.0)          # sign: C decays
        assert np.all(hi.C[1:, 0] < lo.C[1:, 0])          # higher R ⇒ faster
        # magnitude: exact exponential rate γ_C·σ + α_rc·R (R frozen, κ_C=0)
        for traj, r0 in ((lo, 0.5), (hi, 2.0)):
            rate = 0.10 + 0.1 * r0
            assert np.allclose(traj.C[:, 0], np.exp(-rate * traj.t), rtol=2e-4)

    def test_rhs_contact_term_is_bilinear(self):
        """∂(dC)/∂C = −(γ_C·σ + α_rc·R) and the term scales with R·C."""
        eng = self._frozen_r_engine(1.0)
        dC_lowC = eng.derivatives_5d(0.0, R=1.0, C=0.1)[1]
        dC_highC = eng.derivatives_5d(0.0, R=1.0, C=1.0)[1]
        assert float(dC_highC[0]) == pytest.approx(10.0 * float(dC_lowC[0]))
        dC_hi_r = eng.derivatives_5d(0.0, R=3.0, C=0.5)[1]
        dC_lo_r = eng.derivatives_5d(0.0, R=1.0, C=0.5)[1]
        assert float(dC_hi_r[0]) < float(dC_lo_r[0]) < 0.0

    # ── ② internal steady state ──────────────────────────────────────────
    def test_c_converges_to_internal_steady_state(self):
        """C* = κ_C(1−Φ*)/(γ_C·σ + α_rc·R) with R frozen; bounded, no divergence."""
        r0 = 1.0
        eng = self._frozen_r_engine(
            r0, kappa_c=0.05, phi_drive=0.65, phi_suppress=0.5,
            kappa_phi=1.5, zeta_phi=1.0)
        traj = eng.run(60.0)
        phi_star = 1.5 * 0.65 / (1.5 * 0.65 + 1.0 * 0.5)
        c_star = 0.05 * (1.0 - phi_star) / (0.10 + 0.1 * r0)
        assert traj.final_phi() == pytest.approx(phi_star, abs=1e-6)
        assert traj.C[-1, 0] == pytest.approx(c_star, abs=1e-4)
        assert np.all(np.isfinite(traj.C))
        assert np.all(traj.C >= 0.0)                       # C ≥ 0 floor holds
        assert np.all(traj.C <= 1.0 + 1e-9)                # bounded (C0 = 1)

    def test_steady_state_lower_with_higher_alpha(self):
        """Stronger contact suppression ⇒ lower C* (monotone in α_rc)."""
        finals = []
        for a in (0.05, 0.1, 0.2):
            eng = self._frozen_r_engine(1.0, alpha_rc=a, kappa_c=0.05)
            finals.append(eng.run(60.0).C[-1, 0])
        assert finals[0] > finals[1] > finals[2] > 0.0

    # ── ③ legacy invariance ──────────────────────────────────────────────
    def test_legacy_dC_is_exactly_v16_linear(self):
        """fast_channel=None: dC is exactly the v1.6 cC − α_eff·R — the
        contact-surface −α_rc·R·C term exists ONLY in the 5-D mode (legacy
        keeps the linear suppression; no extra R·C product may appear)."""
        for r0 in (0.1, 2.0):
            eng = ForceFieldDynamics(r0, 1.0, growth_c=0.01, alpha=0.05,
                                     beta=0.03)
            dC = eng.derivatives(0.0)[1]
            assert float(dC[0]) == pytest.approx(0.01 * 1.0 - 0.05 * r0)

    def test_legacy_golden_digests_unchanged(self):
        """The pre-change frozen legacy digests still hold (byte-identity)."""
        assert _digest(_legacy_rich().run(3.0)) == _GOLDEN_A
        assert _digest(_legacy_simple().run(2.0)) == _GOLDEN_B

    # ── ④ α_rc = 0 degeneration ──────────────────────────────────────────
    def test_alpha_zero_reproduces_pre_patch_digest(self):
        eng = self._frozen_r_engine(1.0, alpha_rc=0.0, kappa_c=0.05, dt=0.25)
        # RHS degenerates exactly to the ledger equation without the patch
        # (state queried before any run: R=C=1, Φ=0.65, κ_C(1−Φ) − γ_C·σ·C)
        dC = eng.derivatives_5d(0.0)[1]
        assert float(dC[0]) == pytest.approx(0.05 * (1.0 - float(eng.phi[0]))
                                             - 0.10 * 1.0 * 1.0)
        traj = eng.run(4.0)
        assert _digest(traj) == self._PREPATCH_ALPHA0

    def test_contact_term_vanishes_at_r_zero(self):
        """R = 0 ⇒ no contact surface ⇒ bit-identical to the pre-patch engine."""
        eng = self._frozen_r_engine(0.0, alpha_rc=0.2, kappa_c=0.05, dt=0.25)
        traj = eng.run(4.0)
        assert _digest(traj) == self._PREPATCH_R0

    def test_negative_alpha_rc_rejected(self):
        with pytest.raises(ValueError):
            ForceFieldDynamics(
                1.0, 1.0, fast_channel=FastChannelConfig(alpha_rc=-0.1))

    def test_per_node_alpha_rc_broadcast(self):
        eng = ForceFieldDynamics(
            [1.0, 1.0], [1.0, 1.0],
            fast_channel=FastChannelConfig(kappa_c=0.0, gamma_c=0.10,
                                           mu_f=0.0, s_in=0.0,
                                           alpha_rc=[0.0, 0.5]),
            method="euler", dt=0.1)
        dC = eng.derivatives_5d(0.0)[1]
        assert float(dC[0]) == pytest.approx(-0.10)   # α_rc=0 ⇒ γ_C·σ only
        assert float(dC[1]) == pytest.approx(-0.10 - 0.5 * 1.0 * 1.0)

    # ── ⑤ two-mode digestion coupling (2026-09-10) ───────────────────────
    def test_two_mode_digestion_conservation(self):
        cfg = FastChannelConfig(mu_cef=0.5, mu_sri=0.3)
        assert cfg.rho_impl == pytest.approx(0.2)
        assert cfg.mu_cef + cfg.mu_sri + cfg.rho_impl == pytest.approx(1.0)

        # Explicit valid rho
        cfg_explicit = FastChannelConfig(mu_cef=0.5, mu_sri=0.3, rho=0.2)
        assert cfg_explicit.rho_impl == pytest.approx(0.2)

        # Invalid sums raise ValueError
        with pytest.raises(ValueError):
            FastChannelConfig(mu_cef=0.8, mu_sri=0.3)
        with pytest.raises(ValueError):
            FastChannelConfig(mu_cef=0.5, mu_sri=0.3, rho=0.3)
        with pytest.raises(ValueError):
            FastChannelConfig(mu_cef=-0.1)

    def test_two_mode_digestion_rhs_injection(self):
        # kappa_c=0.0, alpha_rc=0.1, gamma_c=0.0 -> dC = -0.1*1*1 = -0.1
        # [-C_dot]_+ = 0.1
        cfg = FastChannelConfig(
            mu_cef=0.5, mu_sri=0.3,
            kappa_c=0.0, gamma_c=0.0, alpha_rc=0.1,
            kappa_phi=0.0, zeta_phi=0.0,
            s_in=0.0, lam=0.0, mu_e=0.0, gamma_k=0.0,
        )
        eng = ForceFieldDynamics(1.0, 1.0, fast_channel=cfg)
        dR, dC, dP, dPhi, dK = eng.derivatives_5d(0.0, R=1.0, C=1.0, P=0.0, Phi=0.5, K=0.0)
        assert float(dC[0]) == pytest.approx(-0.1)
        # dPhi received T_c_phi * 0.5 * 0.1 = 0.05
        assert float(dPhi[0]) == pytest.approx(0.05)
        # dP received T_c_p * 0.3 * 0.1 = 0.03
        assert float(dP[0]) == pytest.approx(0.03)
        # dK received T_c_k * 0.2 * 0.1 = 0.02
        assert float(dK[0]) == pytest.approx(0.02)

    def test_two_mode_exclusion_mode_dissipates(self):
        # mu=0 -> exclusion mode, decay does not enter pools
        cfg = FastChannelConfig(
            mu_cef=0.0, mu_sri=0.0,
            kappa_c=0.0, gamma_c=0.0, alpha_rc=0.1,
            kappa_phi=0.0, zeta_phi=0.0,
            s_in=0.0, lam=0.0, mu_e=0.0, gamma_k=0.0,
        )
        eng = ForceFieldDynamics(1.0, 1.0, fast_channel=cfg)
        dR, dC, dP, dPhi, dK = eng.derivatives_5d(0.0, R=1.0, C=1.0, P=0.0, Phi=0.5, K=0.0)
        assert float(dC[0]) == pytest.approx(-0.1)
        assert float(dPhi[0]) == pytest.approx(0.0)
        assert float(dP[0]) == pytest.approx(0.0)
        assert float(dK[0]) == pytest.approx(0.0)

    def test_two_mode_positive_rectification(self):
        # Friction grows: kappa_c*(1-Phi) = 0.5 > dC_sink = 0.1 -> dC = +0.4 > 0
        # [-C_dot]_+ = 0 -> no release injected into Phi, P_R, K
        cfg = FastChannelConfig(
            mu_cef=0.5, mu_sri=0.3,
            kappa_c=1.0, gamma_c=0.0, alpha_rc=0.1,
            kappa_phi=0.0, zeta_phi=0.0,
            s_in=0.0, lam=0.0, mu_e=0.0, gamma_k=0.0,
        )
        eng = ForceFieldDynamics(1.0, 1.0, fast_channel=cfg)
        dR, dC, dP, dPhi, dK = eng.derivatives_5d(0.0, R=1.0, C=1.0, P=0.0, Phi=0.5, K=0.0)
        assert float(dC[0]) == pytest.approx(0.4)
        assert float(dPhi[0]) == pytest.approx(0.0)
        assert float(dP[0]) == pytest.approx(0.0)
        assert float(dK[0]) == pytest.approx(0.0)

    # ── ⑥ P_R morphology split & dissolution (2026-09-10) ────────────────
    def test_two_mode_morphology_conservation(self):
        """Conservation: dP_total = dP_pool + dP_dist. Dissolution terms cancel out exactly."""
        cfg = FastChannelConfig(
            mu_cef=0.2, mu_sri=0.4,
            k_dissolve=0.5, m_half=1.0, m0=1.0,
            p_pool0=2.0, p_dist0=1.0,
            kappa_c=0.0, gamma_c=0.0, alpha_rc=0.1,
            s_in=0.3, lam=0.2, mu_e=0.1, eta=0.5,
        )
        eng = ForceFieldDynamics(1.0, 1.0, fast_channel=cfg)
        # At m0=1.0, m_half=1.0: g(M) = 1.0 / (1.0 + 1.0) = 0.5
        # j_dissolve = 0.5 * 0.5 * 2.0 = 0.5
        # dC = -0.1*1*1 = -0.1 -> [-C_dot]_+ = 0.1
        # j_sri_total = 1.0 * 0.4 * 0.1 = 0.04 -> j_sri_dist = 0.04, j_sri_pool = 0.0
        dR, dC, dPp, dPd, dPhi, dK, dM = eng.derivatives_morphology(0.0)
        # dP_total = dPp + dPd
        dP_total = float(dPp[0]) + float(dPd[0])
        # Compare with derivatives_5d
        _r, _c, dP_5d, _phi, _k = eng.derivatives_5d(0.0)
        assert float(dP_5d[0]) == pytest.approx(dP_total)
        # Check cancellation of j_dissolve:
        # Base dPp = s_in*R - lam*Pp - eta*mu_e*Pp - j_dissolve = 0.3*1 - 0.2*2 - 0.5*0.1*2 - 0.5 = 0.3 - 0.4 - 0.1 - 0.5 = -0.7
        # dPd = j_dissolve + j_sri_dist = 0.5 + 0.04 = 0.54
        # Total dP = -0.7 + 0.54 = -0.16
        # Base total dP without dissolution = 0.3 - 0.4 - 0.1 + 0.04 = -0.16
        assert float(dPp[0]) == pytest.approx(-0.70)
        assert float(dPd[0]) == pytest.approx(0.54)
        assert dP_total == pytest.approx(-0.16)

    def test_two_mode_g_saturation_monotonicity(self):
        """g(M) is saturating and monotone: g(0)=0, g(m_half)=0.5, g(inf)->1, dM/dt >= 0."""
        cfg = FastChannelConfig(
            mu_cef=0.5, mu_sri=0.5, mu_sri_to_dist=1.0,
            m_half=1.5, m0=0.0,
            kappa_c=0.0, gamma_c=0.0, alpha_rc=0.2,
        )
        eng = ForceFieldDynamics(1.0, 1.0, fast_channel=cfg)
        assert float(eng.g_sat[0]) == pytest.approx(0.0)
        # Step forward under positive friction decay -> C decays, SRI flows into M
        traj = eng.run(t_end=5.0, dt=0.1)
        assert traj.m_cum is not None
        assert traj.g_sat is not None
        m_arr = traj.m_cum[:, 0]
        g_arr = traj.g_sat[:, 0]
        # Monotonically non-decreasing
        assert np.all(np.diff(m_arr) >= -1e-9)
        assert np.all(np.diff(g_arr) >= -1e-9)
        assert m_arr[-1] > m_arr[0]
        assert 0.0 < g_arr[-1] < 1.0

    def test_two_mode_mu_sri_routes_to_dist(self):
        """mu_sri_to_dist=1.0 routes all SRI into P_dist and M; S_in routes into P_pool."""
        cfg = FastChannelConfig(
            mu_cef=0.0, mu_sri=0.6, mu_sri_to_dist=1.0,
            k_dissolve=0.0, m_half=1.0, m0=0.0,
            p_pool0=0.0, p_dist0=0.0,
            s_in=0.5, lam=0.0, mu_e=0.0,
            kappa_c=0.0, gamma_c=0.0, alpha_rc=0.1,
            kappa_phi=0.0, zeta_phi=0.0,
        )
        eng = ForceFieldDynamics(1.0, 1.0, fast_channel=cfg)
        # dC = -0.1 -> [-C_dot]_+ = 0.1
        # SRI flow = 0.6 * 0.1 = 0.06 -> all enters P_dist
        # S_in flow = 0.5 * 1.0 = 0.5 -> enters P_pool
        dR, dC, dPp, dPd, dPhi, dK, dM = eng.derivatives_morphology(0.0)
        assert float(dPp[0]) == pytest.approx(0.5)
        assert float(dPd[0]) == pytest.approx(0.06)
        assert float(dM[0]) == pytest.approx(0.06)

    def test_two_mode_morphology_legacy_compatibility(self):
        """When digestion is off (mu_sri=0, k_dissolve=0), morphology leaves legacy dynamics invariant."""
        cfg = FastChannelConfig(mu_cef=0.0, mu_sri=0.0, k_dissolve=0.0)
        eng = ForceFieldDynamics(1.0, 1.0, latent_force=0.5, fast_channel=cfg)
        # Initial: p_dist0=0, p_pool0=0.5
        assert float(eng.p_dist[0]) == pytest.approx(0.0)
        assert float(eng.p_pool[0]) == pytest.approx(0.5)
        assert float(eng.pool_share[0]) == pytest.approx(1.0)
        traj = eng.run(t_end=2.0, dt=0.05)
        # P_dist remains 0 throughout
        assert np.all(traj.p_dist == 0.0)
        # P_pool == P throughout
        assert np.allclose(traj.p_pool, traj.P)


