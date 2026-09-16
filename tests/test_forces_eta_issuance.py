"""SIXTH STATE round (2026-09-16): η value-level state + issuance channel
+ R logistic capacity + spatial damping + bounded-issuance invariant.

Covers the four-point targeted engine thaw:

1. η as the SIXTH STATE variable (alongside R/C/P_R/Φ/K): canonical seed
   1.42, erosion law ``dη/dt = −κ_eta·max(0, j_issue − j_verified)/v_stock``,
   real/nominal S columns (``S_real = R·η/(R·η + C)`` vs ``S = R/(R+C)``).
2. ``issuance_fn`` — a legal injection channel writing a mechanical
   cumulative-issuance counter (post-step, never touching R/C).
3. ``capacity`` (K_cap) — logistic carrying ceiling ``a_eff·R·(1 − R/K_cap)``.
4. ``spatial_latency`` / ``tau_spatial`` — ``a_eff(i) = a₀·exp(−latency/τ)``.
5. ``verify.check_issuance_bound`` — flow + magnitude bounded invariants.

World-agnostic throughout: the kernel knows ``issuance`` / ``capacity`` /
``eta_level`` as magnitudes, never their economic semantics.
"""

import json

import numpy as np
import pytest

from spectrum_os.kernel.forces import (
    FastChannelConfig,
    ForceFieldDynamics,
)
from spectrum_os.kernel.verify import check_issuance_bound


def _fast(**kw) -> ForceFieldDynamics:
    dt = kw.pop("dt", 0.25)
    eta_level = kw.pop("eta_level", 1.42)  # engine-level seed, not a config field
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
        fast_channel=FastChannelConfig(**base), dt=dt, eta_level=eta_level,
    )


def _const(value: float):
    def _fn(t, ctx):
        return np.full_like(np.asarray(ctx["R"], dtype=np.float64), value)
    return _fn


class TestEtaStateVariable:
    """η — the SIXTH STATE variable (seed 1.42, erosion law, real/real S)."""

    def test_default_seed_is_canonical(self):
        eng = _fast()
        assert eng.eta_level is not None
        assert eng.eta_level[0] == pytest.approx(1.42)

    def test_custom_seed_and_validation(self):
        assert ForceFieldDynamics(
            1.0, 0.5, fast_channel=True, eta_level=0.9
        ).eta_level[0] == pytest.approx(0.9)
        with pytest.raises(ValueError):
            ForceFieldDynamics(1.0, 0.5, fast_channel=True, eta_level=-0.1)
        # legacy path accepts and validates the seed but stays η-free
        legacy = ForceFieldDynamics(1.0, 0.5, eta_level=2.0)
        assert legacy.eta_level is None
        assert legacy.issued_cumulative is None
        assert legacy.six_dim_state() is None
        with pytest.raises(RuntimeError):
            legacy.s_real()

    def test_erosion_law_hand_computation(self):
        """Constant flows ⇒ RK4 is exact: Δη = −κ·max(0, j−v)/stock·Δt."""
        kappa, j_issue, j_ver, stock = 0.1, 0.5, 0.1, 2.0
        eng = _fast(kappa_eta=kappa, verified_stock=stock,
                    issuance_fn=_const(j_issue),
                    verified_growth_fn=_const(j_ver))
        traj = eng.run(1.0)
        expected = 1.42 - kappa * max(0.0, j_issue - j_ver) / stock * 1.0
        assert traj.eta[-1, 0] == pytest.approx(expected)

    def test_eta_inert_by_default_even_with_issuance(self):
        """κ_eta = 0 (default) ⇒ η frozen at the seed; issuance only counts."""
        eng = _fast(issuance_fn=_const(0.5))
        traj = eng.run(1.0)
        assert np.all(traj.eta == pytest.approx(1.42))
        # 4 steps of dt=0.25, each accumulating the constant per-step amount
        assert traj.issued_cumulative[-1, 0] == pytest.approx(4 * 0.5)

    def test_verified_growth_gating(self):
        """verified ≥ issuance ⇒ max(0, ·) = 0 ⇒ η constant."""
        eng = _fast(kappa_eta=0.5, issuance_fn=_const(0.3),
                    verified_growth_fn=_const(0.3))
        traj = eng.run(1.0)
        assert traj.eta[-1, 0] == pytest.approx(1.42)

    def test_verified_stock_denominator(self):
        """Doubling the verified stock halves the erosion."""
        common = dict(kappa_eta=0.2, issuance_fn=_const(0.4),
                      verified_growth_fn=_const(0.0))
        t_small = _fast(verified_stock=1.0, **common).run(1.0)
        t_big = _fast(verified_stock=2.0, **common).run(1.0)
        drop_small = 1.42 - t_small.eta[-1, 0]
        drop_big = 1.42 - t_big.eta[-1, 0]
        assert drop_small > 0
        assert drop_big == pytest.approx(drop_small / 2.0)

    def test_eta_monotone_non_increasing_and_floored(self):
        eng = _fast(kappa_eta=50.0, issuance_fn=_const(1.0))
        traj = eng.run(3.0)
        assert np.all(np.diff(traj.eta[:, 0]) <= 1e-12)
        assert np.all(traj.eta >= 0.0)

    def test_negative_kappa_eta_rejected(self):
        with pytest.raises(ValueError):
            FastChannelConfig(kappa_eta=-0.1)
        with pytest.raises(ValueError):
            FastChannelConfig(verified_stock=-1.0)

    def test_reset_restores_eta_and_zeroes_counter(self):
        eng = _fast(kappa_eta=0.2, issuance_fn=_const(0.5))
        eng.run(1.0)
        assert float(eng.eta_level[0]) != pytest.approx(1.42)
        assert float(eng.issued_cumulative[0]) > 0.0
        eng.reset()
        assert float(eng.eta_level[0]) == pytest.approx(1.42)
        assert float(eng.issued_cumulative[0]) == 0.0


class TestRealAndNominalColumns:
    """S_nominal = R/(R+C) vs S_real = R·η/(R·η + C) — two columns, one state."""

    def test_s_real_hand_formula(self):
        eng = _fast()
        s_real = eng.s_real(R=np.array([1.0]), C=np.array([0.5]),
                            eta=np.array([1.42]))
        assert s_real[0] == pytest.approx(1.0 * 1.42 / (1.0 * 1.42 + 0.5))

    def test_eta_one_identity_s_real_equals_nominal(self):
        eng = _fast(eta_level=1.0)
        traj = eng.run(1.0)
        assert np.allclose(traj.s_real, traj.S)

    def test_eta_above_one_lifts_real_share(self):
        eng = _fast()
        traj = eng.run(0.5)
        assert np.all(traj.s_real >= traj.S - 1e-12)

    def test_degenerate_state_pins_half(self):
        eng = _fast()
        assert eng.s_real(R=0.0, C=0.0)[0] == pytest.approx(0.5)

    def test_snapshot_and_dict_columns(self):
        eng = _fast()
        snap = eng.step()
        assert {"Eta", "S_real", "issued_cumulative"} <= set(snap)
        payload = eng.run(0.5).to_dict()
        assert {"eta", "s_real", "issued_cumulative"} <= set(payload)
        json.dumps(payload)  # JSON-safe

    def test_legacy_snapshot_has_no_eta_keys(self):
        eng = ForceFieldDynamics(1.0, 0.5)
        snap = eng.step()
        assert "Eta" not in snap and "S_real" not in snap
        assert "eta" not in ForceFieldDynamics(1.0, 0.5).run(0.5).to_dict()

    def test_six_dim_state(self):
        eng = _fast()
        state = eng.six_dim_state()
        assert len(state) == 6
        r, c, p, phi, k, eta = state
        assert eta[0] == pytest.approx(1.42)

    def test_provenance_tags_declared(self):
        prov = FastChannelConfig().provenance()
        for f in ("issuance_fn", "verified_growth_fn",
                  "verified_stock", "kappa_eta"):
            assert f in prov


class TestIssuanceChannel:
    """The legal injection channel: a mechanical accumulator, R/C untouched."""

    def test_accumulates_once_per_step(self):
        eng = _fast(issuance_fn=_const(0.5), dt=1.0)
        for _ in range(3):
            snap = eng.step()
        assert float(snap["issued_cumulative"][0]) == pytest.approx(1.5)

    def test_no_substrate_no_accumulation(self):
        eng = _fast(dt=1.0)
        eng.run(2.0)
        assert np.all(eng.issued_cumulative == 0.0)
        traj = eng.run  # noqa: F841  (property exists on trajectory too)
        traj2 = ForceFieldDynamics(
            1.0, 0.5, growth_r=0.02, beta=0.03,
            fast_channel=FastChannelConfig(), dt=1.0).run(1.0)
        assert np.all(traj2.issued_cumulative == 0.0)

    def test_history_is_cumulative(self):
        eng = _fast(issuance_fn=_const(0.25), dt=1.0)
        traj = eng.run(3.0)
        assert np.allclose(traj.issued_cumulative[:, 0],
                           [0.0, 0.25, 0.5, 0.75])

    def test_issuance_does_not_touch_r_or_c(self):
        """The kernel only counts — the dynamics are blind to issuance."""
        eng_plain = _fast(dt=1.0)
        eng_issue = _fast(issuance_fn=_const(10.0), dt=1.0)
        t_plain = eng_plain.run(2.0)
        t_issue = eng_issue.run(2.0)
        for field in ("t", "R", "C", "P", "S", "T", "tension", "phi", "k_pool"):
            assert np.array_equal(getattr(t_plain, field),
                                  getattr(t_issue, field))

    def test_substrate_context_is_universal_observables(self):
        seen: list[dict] = []

        def probe(t, ctx):
            seen.append(dict(ctx))
            return 0.1

        eng = _fast(issuance_fn=probe, dt=1.0)
        eng.run(1.0)
        assert seen
        assert set(seen[0]) == {"R", "C", "P", "Phi", "K", "S", "T"}


class TestRCapacity:
    """Logistic carrying ceiling: a_eff·R·(1 − R/K_cap); None ≡ uncapped."""

    def test_legacy_hand_computation(self):
        eng = ForceFieldDynamics(1.0, 0.5, growth_r=0.05, beta=0.03,
                                 alpha=0.0, capacity=2.0)
        dR, dC, _dP = eng.derivatives_full(0.0, R=1.0, C=0.5)
        assert float(dR[0]) == pytest.approx(0.05 * 1.0 * (1 - 0.5) - 0.03 * 0.5)

    def test_default_is_bit_identical_uncapped(self):
        a = ForceFieldDynamics(1.0, 0.5, growth_r=0.05, beta=0.03).run(2.0)
        b = ForceFieldDynamics(1.0, 0.5, growth_r=0.05, beta=0.03,
                               capacity=None).run(2.0)
        for field in ("t", "R", "C", "P", "S", "T", "tension"):
            assert np.array_equal(getattr(a, field), getattr(b, field))

    def test_five_dim_path_includes_capacity(self):
        """Stripped config ⇒ dR is exactly a·R·(1 − R/K_cap) on the 5-D path."""
        stripped = FastChannelConfig(phi0=0.65, phi_drive=0.65,
                                     phi_suppress=0.5, mu_f=0.0, lam=0.0,
                                     s_in=0.0)
        dR_cap = ForceFieldDynamics(
            1.0, 0.5, growth_r=0.02, beta=0.0, capacity=4.0,
            fast_channel=stripped).derivatives_5d(
            0.0, R=1.0, C=0.5, P=0.2, Phi=0.6, K=0.1)[0]
        assert float(dR_cap[0]) == pytest.approx(0.02 * 1.0 * (1 - 0.25))

    def test_r_saturates_below_capacity(self):
        eng = ForceFieldDynamics(1.0, 0.0, growth_r=0.5, capacity=3.0,
                                 beta=0.0, method="euler", dt=0.05)
        traj = eng.run(20.0)
        assert traj.R[-1, 0] < 3.0
        assert traj.R[-1, 0] == pytest.approx(3.0, rel=0.05)

    def test_invalid_capacity_rejected(self):
        with pytest.raises(ValueError):
            ForceFieldDynamics(1.0, 0.5, capacity=0.0)
        with pytest.raises(ValueError):
            ForceFieldDynamics(1.0, 0.5, capacity=-1.0)


class TestSpatialDamping:
    """a_eff(i) = a₀·exp(−latency_i/τ_spatial); both None ≡ disabled."""

    def test_hand_computation_per_node(self):
        import math
        eng = ForceFieldDynamics(
            [1.0, 1.0], [0.0, 0.0], growth_r=0.05,
            spatial_latency=[0.0, 2.0], tau_spatial=1.0)
        dR, _dC, _dP = eng.derivatives_full(0.0)
        assert np.allclose(dR, [0.05 * math.exp(0.0), 0.05 * math.exp(-2.0)])

    def test_disabled_by_default_bit_identical(self):
        a = ForceFieldDynamics(1.0, 0.5, growth_r=0.05, beta=0.03).run(1.0)
        b = ForceFieldDynamics(1.0, 0.5, growth_r=0.05, beta=0.03,
                               spatial_latency=None, tau_spatial=None).run(1.0)
        assert np.array_equal(a.R, b.R)

    def test_five_dim_path_includes_damping(self):
        """Stripped config ⇒ dR is exactly a·R·exp(−lat/τ) on the 5-D path."""
        import math
        stripped = FastChannelConfig(phi0=0.65, phi_drive=0.65,
                                     phi_suppress=0.5, mu_f=0.0, lam=0.0,
                                     s_in=0.0)
        dR_damped = ForceFieldDynamics(
            1.0, 0.5, growth_r=0.02, beta=0.0, spatial_latency=1.0,
            tau_spatial=2.0, fast_channel=stripped).derivatives_5d(
            0.0, R=1.0, C=0.5, P=0.2, Phi=0.6, K=0.1)[0]
        assert float(dR_damped[0]) == pytest.approx(0.02 * math.exp(-0.5))

    def test_invalid_arguments_rejected(self):
        with pytest.raises(ValueError):
            ForceFieldDynamics(1.0, 0.5, spatial_latency=1.0)  # τ missing
        with pytest.raises(ValueError):
            ForceFieldDynamics(1.0, 0.5, spatial_latency=1.0, tau_spatial=0.0)
        with pytest.raises(ValueError):
            ForceFieldDynamics(1.0, 0.5, spatial_latency=-1.0, tau_spatial=1.0)


class TestIssuanceBound:
    """verify.check_issuance_bound — flow + magnitude bounded invariants."""

    def test_flow_bound_pass_and_fail(self):
        ok = check_issuance_bound([1.0, 2.0], max_single_year=1.0,
                                  years_elapsed=3.0)
        assert ok["flow_bound_ok"] is True
        bad = check_issuance_bound([3.5], max_single_year=1.0,
                                   years_elapsed=3.0)
        assert bad["flow_bound_ok"] is False
        assert bad["detail"]["flow_bound"]["violator_nodes"] == [0]

    def test_magnitude_bound_pass_and_fail(self):
        ok = check_issuance_bound([1.0], r_initial=[1.0], r_final=[1e5])
        assert ok["magnitude_bound_ok"] is True
        bad = check_issuance_bound([1.0], r_initial=[1.0], r_final=[2e6])
        assert bad["magnitude_bound_ok"] is False
        assert bad["detail"]["magnitude_bound"]["violator_nodes"] == [0]

    def test_none_means_not_evaluated(self):
        res = check_issuance_bound([1.0])
        assert res["flow_bound_ok"] is None
        assert res["magnitude_bound_ok"] is None

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            check_issuance_bound([-1.0])
        with pytest.raises(ValueError):
            check_issuance_bound([1.0], max_single_year=1.0, years_elapsed=0.0)
        with pytest.raises(ValueError):
            check_issuance_bound([1.0], r_initial=[1.0, 2.0], r_final=[1.0])
        with pytest.raises(ValueError):
            check_issuance_bound([1.0], r_initial=[1.0], r_final=[1.0],
                                 magnitude_cap=0.0)

    def test_end_to_end_with_trajectory(self):
        eng = _fast(issuance_fn=_const(2.0), dt=1.0)
        traj = eng.run(3.0)
        res = check_issuance_bound(traj.issued_cumulative[-1],
                                   max_single_year=2.0, years_elapsed=3.0,
                                   r_initial=traj.R[0], r_final=traj.R[-1])
        assert res["flow_bound_ok"] is True
        assert res["magnitude_bound_ok"] is True
