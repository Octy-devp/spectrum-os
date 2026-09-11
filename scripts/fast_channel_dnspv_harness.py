#!/usr/bin/env python3
"""ΔNSPV harness — legacy 2/3-D vs the 5-D double-speed (fast channel Φ).

Standalone, deterministic, numpy-only. Reproduces the ECC urban-almanac
1917→1920 intervention projection under two engines:

  * ``legacy`` — the existing :class:`ForceFieldDynamics` R/C/P_R engine with the
    almanac's calibrated parameters (``growth_r=0.02, growth_c=0.01, alpha=0.05,
    beta=0.03, mu=0.02, lam=0.15, rk4, dt=0.25``; the recipe behind
    ``index/locations/data/almanac/1917/intervention/dynamics.json``).
  * ``fast`` — the opt-in 5-D mode (Φ, K) from THEORY-LEDGER
    §「完整動態方程組」, with the ledger's stated parameters
    (λ=0.15 [CALIBRATED], μ_F≈0.12, κ_Φ≈1.5 [EXPERIMENTAL], Φ 0.65→0.90).

Both arms use the SAME measured intervention layer (RLO boost from
``S-trajectory-1920.json`` + the AE cut recipe), so the comparison isolates the
model: does adding the fast channel raise ΔNSPV from the legacy +0.009…+0.011
into the ledger's predicted +0.08…+0.14 band?

Data are read READ-ONLY from the ECC almanac (``--ecc-root``, default
``/home/octy/projects/ECC``). If the data are unavailable the harness falls back
to a small, explicitly documented synthetic calibration scenario
(``--scenario synthetic``) — it never invents measurements silently.

NSPV projection (same convention as ``project-almanac-dynamics.py`` /
``calc-nspv.py``): each node's 1920 NSPV is its measured 1917 NSPV scaled by the
dynamics ratio ``S_1920/S_1917``; the aggregate is the sample mean. ΔNSPV =
mean(intervention) − mean(counterfactual).

No parameter is tuned to hit the band: μ_F/κ_Φ are the ledger's EXPERIMENTAL
targets; unspecified coefficients get a documented choice and a sensitivity
table. See the report for the verdict.

Usage:
    python scripts/fast_channel_dnspv_harness.py [--ecc-root PATH] [--out FILE]
                                                 [--scenario auto|ecc|synthetic]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np

# Make ``import spectrum_os`` work when run from the repo root or directly.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spectrum_os.kernel.forces import FastChannelConfig, ForceFieldDynamics  # noqa: E402

DEFAULT_ECC_ROOT = "/home/octy/projects/ECC"
HORIZON = 3.0          # 1917 → 1920
DT = 0.25              # quarterly (matches the almanac projection)

# Legacy almanac parameters (SSOT: project-almanac-dynamics.py PARAMS).
LEGACY_PARAMS = dict(
    growth_r=0.02, growth_c=0.01, alpha=0.05, beta=0.03, mu=0.02, lam=0.15,
)

# AE-cut recipe (SSOT: calc-nspv.py — 3.5-day lower-bound cut, RSFSR cities).
AE_CUT_DAYS = 3.5
RU_CITIES = ("moscow", "petersburg", "voronezh", "kharkov", "kiev")

# ── fast-channel scenario parameters + provenance ────────────────────────────
# THEORY-LEDGER §「完整動態方程組」 (2026-09-08 agy) pins λ, μ_F, κ_Φ and the
# Φ jump; κ_C / γ_C / σ_soviet / δ / S_in / η are NOT pinned and are documented
# scenario/proxy choices (crack report).
FAST_PARAMS = dict(
    phi0=0.65,            # [SCENARIO] ledger's initial circulation conductivity
    kappa_phi=1.5,        # [EXPERIMENTAL] ledger κ_Φ≈1.5
    zeta_phi=1.0,         # [DERIVED_PROXY] ζ_Φ=1.0
    delta=0.0,            # [DERIVED_PROXY] no autonomous R decay declared
    mu_f=0.12,            # [EXPERIMENTAL] ledger μ_F≈0.12
    eta=0.0,              # [SCENARIO] NSPV closed loop → η=0 by construction
    lam=0.15,             # [CALIBRATED] ledger λ=0.15
    s_in=0.02,            # [DERIVED_PROXY] R→P_R synthesis (mirrors legacy μ)
    mu_e=0.0,             # [DERIVED_PROXY] inert at η=0
    gamma_k=0.0,          # [DERIVED_PROXY] inert at η=0
    gamma_c=0.10,         # [DERIVED_PROXY] friction dissipation
    alpha_rc=0.1,         # [UNCALIBRATED] THEORY-LEDGER minimal patch 2026-09-10
    sigma_soviet=1.0,     # [SCENARIO] full soviet effectiveness
    clamp_spring=0.0,     # [EXPERIMENTAL] unspecified → 0 in the spec-anchored core
    autumn_yield=0.0,     # [EXPERIMENTAL] unspecified → 0 in the spec-anchored core
    reflow_purity=1.0,    # [DERIVED_PROXY]
    surplus_gain=0.0,     # [DERIVED_PROXY]
)
# Φ scenario: baseline Φ*≈0.65 (measured mean rlo_share as CoopShare); cooperative
# entry drives the intervention Φ*→0.9066 (ledger target 0.90).
CO_SHARE_BASE = 0.297     # [MEASURED] mean rlo_share (dkk-aggregate-1917.json)
DEBT_STRESS_BASE = 0.24   # [DERIVED_PROXY] calibrated so Φ*≈0.65
CO_SHARE_IV = 0.647       # [SCENARIO] cooperative entry
DEBT_STRESS_IV = 0.10     # [DERIVED_PROXY] usury back-pressure collapses

PROVENANCE = {
    "legacy_params": "PARAMS of project-almanac-dynamics.py [CALIBRATED]",
    "lam": "CALIBRATED (THEORY-LEDGER λ=0.15)",
    "mu_f": "EXPERIMENTAL (THEORY-LEDGER μ_F≈0.12)",
    "kappa_phi": "EXPERIMENTAL (THEORY-LEDGER κ_Φ≈1.5)",
    "zeta_phi": "DERIVED_PROXY (ζ_Φ=1.0)",
    "delta": "DERIVED_PROXY (δ=0)",
    "eta": "SCENARIO (η=0, NSPV closed loop)",
    "s_in": "DERIVED_PROXY (S_in=0.02)",
    "gamma_c": "DERIVED_PROXY (γ_C=0.10)",
    "alpha_rc": "UNCALIBRATED (THEORY-LEDGER minimal patch 2026-09-10; α_rc=0.1, O(0.1) argument)",
    "kappa_c": "DERIVED_PROXY (per-node κ_C=C0·γ_C/(1−Φ0), friction-neutral)",
    "sigma_soviet": "SCENARIO (σ_soviet=1.0)",
    "phi0": "SCENARIO (Φ0=0.65)",
    "co_share_base": "MEASURED (mean rlo_share=0.297)",
    "debt_stress_base": "DERIVED_PROXY (0.24 → Φ*=0.65)",
    "co_share_iv": "SCENARIO (0.647 → Φ*=0.9066, ledger target 0.90)",
    "debt_stress_iv": "DERIVED_PROXY (0.10)",
    "clamp_spring": "EXPERIMENTAL (unspecified; 0 in spec-anchored core)",
    "autumn_yield": "EXPERIMENTAL (unspecified; 0 in spec-anchored core)",
}


def _nspv_project(nspv0: np.ndarray, s0: np.ndarray, R: np.ndarray,
                  C: np.ndarray) -> float:
    """Aggregate NSPV projection: mean(NSPV_1917 · S_1920/S_1917)."""
    s = R / (R + C)
    return float(statistics.mean(nspv0 * s / s0))


def _load_ecc(ecc_root: str) -> dict | None:
    root = Path(ecc_root)
    nspv_path = root / "index/locations/data/nspv-index.json"
    traj_path = root / "index/locations/data/almanac/1917/intervention/S-trajectory-1920.json"
    if not (nspv_path.exists() and traj_path.exists()):
        return None
    nspv = json.load(open(nspv_path, encoding="utf-8"))
    base = nspv.get("1917", {}).get("villages")
    if not base:
        return None
    names = list(base.keys())
    R0 = np.array([base[n]["NMP"] + base[n]["SRI_proxy"] + base[n]["CEF_proxy"]
                   for n in names], dtype=np.float64)
    C0 = np.array([base[n]["AE"] + base[n]["governance_risk"]
                   for n in names], dtype=np.float64)
    nspv0 = np.array([base[n]["NSPV"] for n in names], dtype=np.float64)
    traj = json.load(open(traj_path, encoding="utf-8"))
    boost_map = dict(zip(traj.get("nodes", []), traj.get("boost_per_node", [])))
    boost = np.array([float(boost_map.get(n, 0.0)) for n in names], dtype=np.float64)
    ae_cut = np.array(
        [0.5 * min(AE_CUT_DAYS / 30.0, 1.0)
         if any(c in n.lower() for c in RU_CITIES) else 0.0 for n in names],
        dtype=np.float64,
    )
    return {
        "source": "ecc", "names": names, "R0": R0, "C0": C0,
        "nspv0": nspv0, "s0": R0 / (R0 + C0), "boost": boost, "ae_cut": ae_cut,
        "provenance": {
            "nspv_index": str(nspv_path) + " [MEASURED, read-only]",
            "boost": str(traj_path) + " [MEASURED RPI-weighted, read-only]",
            "ae_cut": "calc-nspv.py recipe (3.5d, [DERIVED_PROXY])",
        },
    }


def _synthetic() -> dict:
    """Documented fallback calibration scenario (only if ECC data unavailable)."""
    rng = np.random.default_rng(20260908)
    n = 90
    R0 = np.clip(0.85 + 0.35 * rng.standard_normal(n), 0.05, None)
    C0 = np.clip(0.42 + 0.15 * rng.standard_normal(n), 0.05, None)
    nspv0 = 0.60 * (R0 / (R0 + C0))
    boost = np.clip(0.02 * rng.random(n) + 0.01, 0.0, None)
    ae_cut = (0.5 * min(AE_CUT_DAYS / 30.0, 1.0)
              * (rng.random(n) < 0.30)).astype(np.float64)
    return {
        "source": "synthetic", "names": [f"node-{i}" for i in range(n)],
        "R0": R0, "C0": C0, "nspv0": nspv0, "s0": R0 / (R0 + C0),
        "boost": boost, "ae_cut": ae_cut,
        "provenance": {"note": "documented fallback scenario [SCENARIO]; ECC data not found"},
    }


def _run_legacy(R0: np.ndarray, C0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eng = ForceFieldDynamics(R0.copy(), C0.copy(), **LEGACY_PARAMS,
                             method="rk4", dt=DT)
    for _ in range(int(round(HORIZON / DT))):
        eng.step(DT)
    return eng._R.copy(), eng._C.copy()


def _run_fast(R0: np.ndarray, C0: np.ndarray, co_share: float,
              debt_stress: float, *, kappa_c: np.ndarray,
              clamp_spring: float = 0.0, autumn_yield: float = 0.0,
              mu_f: float = FAST_PARAMS["mu_f"]) -> tuple[np.ndarray, np.ndarray, float, float]:
    cfg = FastChannelConfig(
        phi0=FAST_PARAMS["phi0"], co_share=co_share, debt_stress=debt_stress,
        kappa_phi=FAST_PARAMS["kappa_phi"], zeta_phi=FAST_PARAMS["zeta_phi"],
        delta=FAST_PARAMS["delta"], mu_f=mu_f, eta=FAST_PARAMS["eta"],
        lam=FAST_PARAMS["lam"], s_in=FAST_PARAMS["s_in"],
        mu_e=FAST_PARAMS["mu_e"], kappa_c=kappa_c,
        gamma_c=FAST_PARAMS["gamma_c"], alpha_rc=FAST_PARAMS["alpha_rc"],
        sigma_soviet=FAST_PARAMS["sigma_soviet"],
        gamma_k=FAST_PARAMS["gamma_k"], clamp_spring=clamp_spring,
        autumn_yield=autumn_yield, reflow_purity=FAST_PARAMS["reflow_purity"],
        surplus_gain=FAST_PARAMS["surplus_gain"],
    )
    eng = ForceFieldDynamics(R0.copy(), C0.copy(), growth_r=LEGACY_PARAMS["growth_r"],
                             growth_c=LEGACY_PARAMS["growth_c"],
                             alpha=LEGACY_PARAMS["alpha"], beta=LEGACY_PARAMS["beta"],
                             fast_channel=cfg, method="rk4", dt=DT)
    phi_hist = [float(eng.phi[0])]
    for _ in range(int(round(HORIZON / DT))):
        eng.step(DT)
        phi_hist.append(float(eng.phi[0]))
    return eng._R.copy(), eng._C.copy(), phi_hist[-1], float(np.mean(phi_hist))


def _delta(d: dict, R_iv: np.ndarray, C_iv: np.ndarray,
           R_cf: np.ndarray, C_cf: np.ndarray) -> dict:
    agg_cf = _nspv_project(d["nspv0"], d["s0"], R_cf, C_cf)
    agg_iv = _nspv_project(d["nspv0"], d["s0"], R_iv, C_iv)
    return {
        "aggregate_NSPV_cf": round(agg_cf, 4),
        "aggregate_NSPV_iv": round(agg_iv, 4),
        "delta_NSPV": round(agg_iv - agg_cf, 4),
        "delta_NSPV_rel": round((agg_iv - agg_cf) / agg_cf, 4) if agg_cf else None,
    }


def measure(d: dict, *, fscale: float = 1.0, clamp_spring: float = 0.0,
            autumn_yield: float = 0.0, mu_f: float = FAST_PARAMS["mu_f"]) -> dict:
    """Run both arms and return the ΔNSPV comparison.

    ``fscale`` scales the per-node κ_C above the friction-neutral calibration
    (κ_C_i = C0_i·γ_C/(1−Φ0) keeps the counterfactual C stationary at the
    measured 1917 level; fscale>1 inflates the friction stock).
    """
    R0, C0, boost, ae_cut = d["R0"], d["C0"], d["boost"], d["ae_cut"]
    # Intervention initial condition (RLO boost + AE cut); C has a physical floor.
    R_iv0 = R0 + boost + ae_cut
    C_iv0 = np.maximum(C0 - ae_cut, 0.0)

    # legacy arm
    Rc, Cc = _run_legacy(R0, C0)
    Ri, Ci = _run_legacy(R_iv0, C_iv0)
    legacy = _delta(d, Ri, Ci, Rc, Cc)

    # fast arm (friction-neutral κ_C per node)
    kappa_c = fscale * C0 * FAST_PARAMS["gamma_c"] / (1.0 - FAST_PARAMS["phi0"])
    Rc2, Cc2, phi_cf_final, phi_cf_mean = _run_fast(
        R0, C0, CO_SHARE_BASE, DEBT_STRESS_BASE, kappa_c=kappa_c,
        clamp_spring=clamp_spring, autumn_yield=autumn_yield, mu_f=mu_f)
    Ri2, Ci2, phi_iv_final, phi_iv_mean = _run_fast(
        R_iv0, C_iv0, CO_SHARE_IV, DEBT_STRESS_IV,
        kappa_c=kappa_c, clamp_spring=clamp_spring, autumn_yield=autumn_yield,
        mu_f=mu_f)
    fast = _delta(d, Ri2, Ci2, Rc2, Cc2)

    # fast channel alone (same initial conditions, no boost/AE cut)
    R3, C3, _, _ = _run_fast(R0, C0, CO_SHARE_IV, DEBT_STRESS_IV,
                             kappa_c=kappa_c, clamp_spring=clamp_spring,
                             autumn_yield=autumn_yield, mu_f=mu_f)
    fast_only_delta = _nspv_project(d["nspv0"], d["s0"], R3, C3) - fast["aggregate_NSPV_cf"]

    fast.update({
        "phi_cf_final": round(phi_cf_final, 4),
        "phi_cf_mean": round(phi_cf_mean, 4),
        "phi_iv_final": round(phi_iv_final, 4),
        "phi_iv_mean": round(phi_iv_mean, 4),
        "delta_NSPV_fast_only": round(float(fast_only_delta), 4),
        "mean_C_cf": round(float(Cc2.mean()), 4),
        "mean_C_iv": round(float(Ci2.mean()), 4),
        "mean_R_cf": round(float(Rc2.mean()), 4),
        "mean_R_iv": round(float(Ri2.mean()), 4),
    })
    return {
        "legacy": legacy,
        "fast": fast,
        "params": {
            "fscale_kappa_c": fscale, "clamp_spring": clamp_spring,
            "autumn_yield": autumn_yield, "mu_f": mu_f,
            "co_share_base": CO_SHARE_BASE, "debt_stress_base": DEBT_STRESS_BASE,
            "co_share_iv": CO_SHARE_IV, "debt_stress_iv": DEBT_STRESS_IV,
            **FAST_PARAMS,
        },
    }


def in_band(delta: float, lo: float = 0.08, hi: float = 0.14) -> str:
    return "IN BAND" if lo <= delta <= hi else "BELOW BAND" if delta < lo else "ABOVE BAND"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ecc-root", default=os.environ.get("ECC_ROOT", DEFAULT_ECC_ROOT))
    ap.add_argument("--scenario", choices=("auto", "ecc", "synthetic"), default="auto")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args(argv)

    d = None
    if args.scenario in ("auto", "ecc"):
        d = _load_ecc(args.ecc_root)
    if d is None:
        if args.scenario == "ecc":
            print(f"ERROR: ECC data not found under {args.ecc_root}", file=sys.stderr)
            return 2
        d = _synthetic()
        reason = ("explicit --scenario synthetic" if args.scenario == "synthetic"
                  else f"ECC data unavailable ({args.ecc_root})")
        print(f"[data] synthetic fallback: {reason}")
    n = len(d["names"])
    print(f"[data] source={d['source']} nodes={n} horizon={HORIZON}y dt={DT} "
          f"(horizon 1917→1920)")

    primary = measure(d)
    print("\n=== PRIMARY SCENARIO (spec-anchored: γ_C=0.10, friction-neutral κ_C, "
          "clamp_spring=0, autumn_yield=0) ===")
    print(f"  provenance: {json.dumps(PROVENANCE, ensure_ascii=False)}")
    lg, fa = primary["legacy"], primary["fast"]
    print(f"  legacy : cf={lg['aggregate_NSPV_cf']:.4f} iv={lg['aggregate_NSPV_iv']:.4f} "
          f"ΔNSPV={lg['delta_NSPV']:+.4f} ({lg['delta_NSPV_rel']*100:+.2f}%)")
    print(f"  5D fast: cf={fa['aggregate_NSPV_cf']:.4f} iv={fa['aggregate_NSPV_iv']:.4f} "
          f"ΔNSPV={fa['delta_NSPV']:+.4f} ({fa['delta_NSPV_rel']*100:+.2f}%)")
    print(f"  5D fast channel alone (no boost/AE cut): ΔNSPV={fa['delta_NSPV_fast_only']:+.4f}")
    print(f"  Φ: cf {fa['phi_cf_final']:.4f} → iv {fa['phi_iv_final']:.4f} "
          f"(mean iv {fa['phi_iv_mean']:.4f})")
    print(f"  VERDICT: {in_band(fa['delta_NSPV'])} (target +0.08…+0.14)")

    print("\n=== SENSITIVITY (pre-declared grid; unspecified coefficients) ===")
    print(f"  {'fscale':>6} {'clamp':>6} {'mu_F':>5} {'ΔNSPV':>9} {'rel%':>8}  band")
    rows = []
    for fscale in (1.0, 2.0, 3.0, 5.0):
        for clamp_spring in (0.0, 0.2):
            for mu_f in (0.12, 0.30):
                m = measure(d, fscale=fscale, clamp_spring=clamp_spring, mu_f=mu_f)
                dv = m["fast"]["delta_NSPV"]
                rel = m["fast"]["delta_NSPV_rel"] * 100
                rows.append({"fscale": fscale, "clamp_spring": clamp_spring,
                             "mu_f": mu_f, "delta_NSPV": dv,
                             "delta_NSPV_rel": round(rel, 2),
                             "band": in_band(dv)})
                print(f"  {fscale:>6.1f} {clamp_spring:>6.1f} {mu_f:>5.2f} "
                      f"{dv:>+9.4f} {rel:>+7.2f}%  {in_band(dv)}")

    result = {
        "schema": "fast-channel-dnspv-v1",
        "ecc_root": args.ecc_root,
        "data_source": d["source"],
        "nodes": n,
        "horizon_years": HORIZON,
        "dt": DT,
        "legacy_params": LEGACY_PARAMS,
        "fast_params": FAST_PARAMS,
        "provenance": PROVENANCE,
        "data_provenance": d["provenance"],
        "primary": primary,
        "sensitivity": rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"\n[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
