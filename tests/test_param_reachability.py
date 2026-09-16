"""Parameter reachability guard — no configuration field may be silently ignored.

## Why this exists

The 5-D fast-channel RHS dropped several constructor arguments without saying so:
`mu` and `rho` (§2.2b spontaneous compression / forgetting) were accepted by
`ForceFieldDynamics.__init__` and then never read by `derivatives_morphology`;
`growth_c_fn` (§2.3b avalanche feedback) had nowhere to attach. ECC passed
`mu=0.02` and got a model that silently ignored it — wrong physics, exit code 0,
no warning. This is the same failure shape as the "移動的裂縫" the project warns
about: nothing crashes, the numbers are just quietly not what they claim.

## The check

For every field of :class:`FastChannelConfig`, perturb it and assert the
integrated trajectory changes; for every constructor argument of
:class:`ForceFieldDynamics`, likewise. A field that can be set and not matter is
either dead code or a missing wire.

The check is deliberately end-to-end (run a short horizon crossing both seasonal
events) rather than per-derivative, so it also covers initial conditions
(`phi0`, `k0`, `m0`, `p_pool0`, `p_dist0`) and the seasonal operator split
(`clamp_spring`, `autumn_yield`, …) that never appear in a single RHS call.

If a field is genuinely inert, add it to ``DECLARED_INERT`` **with a reason** —
and the test will then hold that entry to account in both directions (an entry
that *does* matter is a stale declaration and also fails).
"""

from __future__ import annotations

from dataclasses import fields as dc_fields
from typing import Any

import numpy as np
import pytest

from spectrum_os.kernel.forces import FastChannelConfig, ForceFieldDynamics

DT = 0.25
HORIZON = 1.30   # crosses spring (0.25), autumn (0.75) and the next spring (1.25)

# A base configuration in which every channel is live (drives non-zero, digestion
# active, morphology active, seasonal operators active) so that "inert" cannot be
# an artefact of a switched-off subsystem.
BASE_CFG: dict[str, Any] = {
    "phi0": 0.60, "phi_drive": 0.50, "phi_suppress": 0.30,
    "kappa_phi": 1.5, "zeta_phi": 0.6,
    "delta": 0.02, "mu_f": 0.12, "eta": 0.20, "lam": 0.15,
    "s_in": 0.03, "mu_e": 0.25, "k0": 0.10, "gamma_k": 0.30,
    "kappa_c": 0.08, "gamma_c": 0.12, "alpha_rc": 0.10, "sigma_admin": 1.10,
    "year_period": 1.0, "spring_phase": 0.25, "autumn_phase": 0.75,
    "clamp_spring": 0.30, "autumn_yield": 0.40, "reflow_purity": 0.90,
    "surplus_gain": 0.05,
    "mu_cef": 0.30, "mu_sri": 0.30, "rho": 0.40,
    "t_c_phi": 1.0, "t_c_p": 1.0, "t_c_k": 1.0,
    "m_half": 0.50, "mu_sri_to_dist": 0.60, "k_dissolve": 0.40, "m0": 0.20,
    "p_pool0": 0.30, "p_dist0": 0.10,
    # η value-level channel (SIXTH STATE 2026-09-16): kept LIVE so κ_eta /
    # verified_stock reachability is measurable — without an issuance flow the
    # erosion law is vacuously inert and the guard would mislabel wired params.
    "issuance_fn": lambda t, ctx: np.full_like(ctx["C"], 0.40),
    "kappa_eta": 0.15,
    "verified_stock": 1.5,
}

BASE_ENGINE: dict[str, Any] = {
    "growth_r": 0.03, "growth_c": 0.02, "alpha": 0.05, "beta": 0.05,
    "mu": 0.02, "lam": 0.15, "rho": 0.01, "latent_force": 0.20,
}

# The digestion triple is coupled by the conservation law
# mu_cef + mu_sri + rho == 1, so it must be perturbed as a group.
DIGESTION_TRIPLE = ("mu_cef", "mu_sri", "rho")

DECLARED_INERT: dict[str, str] = {
    # (field name) -> reason. Empty is the strong state: nothing is ignored.
}


def _trajectory(cfg_kwargs: dict[str, Any],
                engine_kwargs: dict[str, Any]) -> np.ndarray:
    eng = ForceFieldDynamics(
        [1.2, 0.9], [0.6, 0.5],
        fast_channel=FastChannelConfig(**cfg_kwargs), dt=DT, method="rk4",
        **engine_kwargs,
    )
    traj = eng.run(HORIZON)
    parts = [traj.R.ravel(), traj.C.ravel(), traj.P.ravel(), traj.phi.ravel(),
             traj.k_pool.ravel(), np.atleast_1d(traj.final_s())]
    if traj.eta is not None:
        # SIXTH STATE (η) is an emitted observable too — a parameter that only
        # moves η (e.g. κ_eta) must still count as reachable.
        parts.append(traj.eta.ravel())
    return np.concatenate(parts)


def _perturb(name: str, value: Any) -> Any:
    """A deliberately different value of the same kind, respecting validation.

    Constraint-aware on purpose: a perturbation that trips ``__post_init__``
    would make the guard measure validation instead of reachability.
    """
    if callable(value):
        return lambda t, ctx: np.full_like(ctx["C"], 0.42)
    if value is None:
        return 0.77
    if isinstance(value, bool):
        return not value
    if name.endswith("_phase"):          # seasonal phases live in [0, 1)
        return 0.15 if float(value) > 0.5 else 0.65
    if name in ("mu_sri_to_dist", "eta", "sigma_admin"):  # bounded channels
        return 0.85 if float(value) < 0.5 else 0.35
    return float(value) * 1.37 + 0.11


def _differs(a: np.ndarray, b: np.ndarray) -> bool:
    return not np.allclose(a, b, rtol=0.0, atol=1e-12)


def _inert_fields() -> list[str]:
    base = _trajectory(BASE_CFG, BASE_ENGINE)
    inert: list[str] = []
    for f in dc_fields(FastChannelConfig):
        name = f.name
        if name in DIGESTION_TRIPLE:
            continue          # checked as a group below
        current = BASE_CFG.get(name, f.default)
        if current is None or callable(current):
            continue          # callable slots are exercised separately
        kwargs = dict(BASE_CFG)
        kwargs[name] = _perturb(name, current)
        engine = dict(BASE_ENGINE)
        if name == "lam":
            # `lam` exists on both sides and a disagreement is deliberately a
            # loud error, so clear the engine copy to let the config's govern
            # (otherwise the guard would measure the conflict check instead of
            # reachability).
            engine["lam"] = 0.0
        if not _differs(base, _trajectory(kwargs, engine)):
            inert.append(name)
    return inert


def test_digestion_triple_reaches_the_trajectory():
    """The conservation-coupled triple must be perturbable as a group.

    Perturbing one member alone trips ``mu_cef + mu_sri + rho == 1`` (correctly),
    so reachability is tested by re-splitting the same total differently —
    which changes *which* route friction takes without changing the budget.
    """
    base = _trajectory(BASE_CFG, BASE_ENGINE)
    kwargs = dict(BASE_CFG)
    kwargs["mu_cef"], kwargs["mu_sri"], kwargs["rho"] = 0.60, 0.10, 0.30
    assert _differs(base, _trajectory(kwargs, BASE_ENGINE)), (
        "the digestion split (mu_cef / mu_sri / rho) does not reach the "
        "trajectory — friction's routing is not actually configurable"
    )


def test_conservation_is_enforced_not_assumed():
    """An unbalanced triple must be rejected loudly, not silently normalised."""
    kwargs = dict(BASE_CFG)
    kwargs["mu_cef"], kwargs["mu_sri"], kwargs["rho"] = 0.5, 0.5, 0.5
    with pytest.raises(ValueError):
        _trajectory(kwargs, BASE_ENGINE)


def test_no_fast_channel_field_is_silently_ignored():
    found = set(_inert_fields())
    undeclared = sorted(found - set(DECLARED_INERT))
    assert not undeclared, (
        "FastChannelConfig field(s) can be set and have no effect — dead code or "
        "a missing wire:\n  " + "\n  ".join(undeclared)
        + "\n\nFix the RHS so the parameter reaches it, or declare it inert with "
        "a reason in DECLARED_INERT."
    )


def test_declared_inert_entries_are_still_inert():
    """A stale exemption is as dangerous as a missing one."""
    found = set(_inert_fields())
    stale = sorted(set(DECLARED_INERT) - found)
    assert not stale, (
        "DECLARED_INERT lists field(s) that now affect the trajectory — the "
        "declaration is stale, remove it:\n  " + "\n  ".join(stale)
    )


def test_substrate_callables_override_their_scalar_slots():
    """A callable slot must take precedence — the whole point of the substrate."""
    scalar = _trajectory(BASE_CFG, BASE_ENGINE)
    for slot, fn_kwarg, cite in (
        ("phi_drive", "phi_drive_fn", "Phi logistic drive"),
        ("phi_suppress", "phi_suppress_fn", "Phi back-pressure"),
    ):
        kwargs = dict(BASE_CFG)
        kwargs[fn_kwarg] = lambda t, ctx: np.full_like(ctx["C"], 1.31)
        assert _differs(scalar, _trajectory(kwargs, BASE_ENGINE)), (
            f"{fn_kwarg} did not override {slot} ({cite})"
        )


@pytest.mark.parametrize("engine_arg", ["mu", "rho"])
def test_engine_reservoir_args_reach_the_5d_rhs(engine_arg: str):
    """§2.2b channels must act in the 5-D path, not only in the legacy 3-D path.

    μ and ρ were exactly the arguments ECC passed and the 5-D RHS dropped: the
    model ran, the exit code was 0, and the physics was not what the caller
    configured.
    """
    base = _trajectory(BASE_CFG, BASE_ENGINE)
    kwargs = dict(BASE_ENGINE)
    kwargs[engine_arg] = float(BASE_ENGINE[engine_arg]) * 3.0 + 0.05
    assert _differs(base, _trajectory(BASE_CFG, kwargs)), (
        f"engine argument {engine_arg!r} is accepted but never reaches the 5-D "
        "RHS — the caller would believe it was applied"
    )


def test_engine_lambda_conflict_is_loud_not_silent():
    """λ exists on both the engine and the config — silence is the bug.

    The 5-D RHS reads ``fast_channel.lam``, so an engine-level ``lam`` that
    disagrees would be accepted and ignored. The contract is: **either the value
    matters, or setting it raises.** Never silently accepted. (ECC's normal case
    is engine lam == config lam == 0.15.)
    """
    conflicting = dict(BASE_ENGINE)
    conflicting["lam"] = float(BASE_CFG["lam"]) * 5.0 + 1.0
    with pytest.raises(ValueError, match="ambiguous lambda"):
        _trajectory(BASE_CFG, conflicting)

    # agreement is fine, and so is "engine unset" (config governs alone)
    agreeing = dict(BASE_ENGINE, lam=float(BASE_CFG["lam"]))
    _trajectory(BASE_CFG, agreeing)
    unset = dict(BASE_ENGINE, lam=0.0)
    _trajectory(BASE_CFG, unset)


def test_compression_channel_conserves_total_capacity():
    """The one physics assertion that must never regress (§2.2b/v1.6).

    Evidence is α history: the Canton Commune (1927) was crushed in days yet the
    forces it disclosed re-emerged; 1905 fed 1917. Repression compresses, it does
    not destroy — so βC must appear as −βC in dR and +βC in dP, and the two must
    cancel exactly.
    """
    eng = ForceFieldDynamics(
        [1.0], [0.9], growth_r=0.0, alpha=0.0, beta=0.40, dt=DT,
        fast_channel=FastChannelConfig(
            phi0=0.65, phi_drive=0.0, phi_suppress=0.0, mu_f=0.0, s_in=0.0,
            kappa_c=0.0, gamma_c=0.0, alpha_rc=0.0, eta=0.0, mu_e=0.0,
            gamma_k=0.0, delta=0.0, lam=0.0, k_dissolve=0.0,
        ),
    )
    dR, _dC, dP, _dPhi, _dK = eng.derivatives_5d(
        0.0, R=1.0, C=0.9, P=0.0, Phi=0.65, K=0.0)
    assert float(dR[0]) == pytest.approx(-0.40 * 0.9)
    assert float(dP[0]) == pytest.approx(+0.40 * 0.9)
    assert float(dR[0] + dP[0]) == pytest.approx(0.0, abs=1e-12)
