#!/usr/bin/env python3
"""α_rc contact-surface suppression probe — 90-node ECC 1917 base period.

THEORY-LEDGER minimal patch (2026-09-10, UNCALIBRATED): the fast-channel C
equation gains the bilinear term −α_rc·R·C. This probe sweeps
α_rc ∈ {0, 0.05, 0.1, 0.2} on the measured ECC urban-almanac base period
(``index/locations/data/nspv-index.json`` 1917 villages, READ-ONLY) and
reports the C(t) trajectory's end state and stability:

  * counterfactual base arm (no intervention): R0 = NMP+SRI+CEF, C0 = AE+gov-risk,
    friction-neutral κ_C,i = C0_i·γ_C/(1−Φ0) (C stationary at 1917 when α_rc=0),
    phi_drive/phi_suppress = harness base (Φ* ≈ 0.65), RK4, dt=0.25, horizon 3y
    (1917→1920, identical recipe to fast_channel_dnspv_harness.py);
  * per sweep: C(1y/2y/3y) means, analytic frozen-R quasi-steady state
    C* = κ_C·(1−Φ*)/(γ_C·σ + α_rc·R), min/max C, negative/NaN counts.

Data are only read, never written. No commit, no ECC writes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_channel_dnspv_harness as h  # noqa: E402

ALPHA_GRID = (0.0, 0.05, 0.1, 0.2)


def run_arm(r0, c0, alpha_rc, dt, horizon):
    """One base-period run; returns (t, C, R, phi_final) histories."""
    fp = h.FAST_PARAMS
    kappa_c = c0 * fp["gamma_c"] / (1.0 - fp["phi0"])
    cfg = h.FastChannelConfig(
        phi0=fp["phi0"], phi_drive=h.PHI_DRIVE_BASE,
        phi_suppress=h.PHI_SUPPRESS_BASE, kappa_phi=fp["kappa_phi"],
        zeta_phi=fp["zeta_phi"], delta=fp["delta"], mu_f=fp["mu_f"],
        eta=fp["eta"], lam=fp["lam"], s_in=fp["s_in"], mu_e=fp["mu_e"],
        kappa_c=kappa_c, gamma_c=fp["gamma_c"], alpha_rc=alpha_rc,
        sigma_admin=fp["sigma_admin"], gamma_k=fp["gamma_k"],
        clamp_spring=fp["clamp_spring"], autumn_yield=fp["autumn_yield"],
        reflow_purity=fp["reflow_purity"], surplus_gain=fp["surplus_gain"],
    )
    eng = h.ForceFieldDynamics(
        r0.copy(), c0.copy(),
        growth_r=h.LEGACY_PARAMS["growth_r"], growth_c=h.LEGACY_PARAMS["growth_c"],
        alpha=h.LEGACY_PARAMS["alpha"], beta=h.LEGACY_PARAMS["beta"],
        fast_channel=cfg, method="rk4", dt=dt)
    steps = int(round(horizon / dt))
    t_hist = [0.0]
    C_hist = [eng._C.copy()]
    R_hist = [eng._R.copy()]
    for k in range(1, steps + 1):
        eng.step(dt)
        t_hist.append(eng.t)
        C_hist.append(eng._C.copy())
        R_hist.append(eng._R.copy())
    return (np.array(t_hist), np.array(C_hist), np.array(R_hist),
            kappa_c, float(eng.phi[0]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ecc-root", default=h.DEFAULT_ECC_ROOT)
    ap.add_argument("--horizon", type=float, default=h.HORIZON)
    ap.add_argument("--dt", type=float, default=h.DT)
    args = ap.parse_args(argv)

    d = h._load_ecc(args.ecc_root)
    if d is None:
        print(f"ERROR: ECC data not found under {args.ecc_root}", file=sys.stderr)
        return 2
    R0, C0 = d["R0"], d["C0"]
    fp = h.FAST_PARAMS
    drive = fp["kappa_phi"] * h.PHI_DRIVE_BASE
    decay = fp["zeta_phi"] * h.PHI_SUPPRESS_BASE
    phi_star = drive / (drive + decay)

    print(f"[data] source={d['source']} nodes={len(d['names'])} "
          f"horizon={args.horizon}y dt={args.dt} (base period 1917→1920, no intervention)")
    print(f"[params] γ_C={fp['gamma_c']} σ={fp['sigma_admin']} Φ0={fp['phi0']} "
          f"Φ*={phi_star:.4f} phi_drive={h.PHI_DRIVE_BASE} phi_suppress={h.PHI_SUPPRESS_BASE}")
    print(f"[base] mean R0={R0.mean():.4f} mean C0={C0.mean():.4f} "
          f"(friction-neutral κ_C ⇒ C stationary at α_rc=0)")
    hdr = (f"  {'α_rc':>5} {'C(1y)':>8} {'C(2y)':>8} {'C(3y)':>8} {'C*analytic':>10} "
           f"{'min C':>8} {'max C':>8} {'neg':>4} {'NaN':>4} {'stability':>12}")
    print(hdr)
    results = []
    c_ref = None
    for a_rc in ALPHA_GRID:
        t, C, R, kappa_c, phi_final = run_arm(R0, C0, a_rc, args.dt, args.horizon)
        idx = {1.0: int(round(1.0 / args.dt)), 2.0: int(round(2.0 / args.dt)),
               3.0: len(t) - 1}
        c_star = float(np.mean(kappa_c * (1.0 - phi_star)
                               / (fp["gamma_c"] * fp["sigma_admin"]
                                  + a_rc * R[-1])))
        neg = int(np.sum(C < 0.0))
        nan = int(np.sum(~np.isfinite(C)))
        c1, c2, c3 = float(C[idx[1.0]].mean()), float(C[idx[2.0]].mean()), float(C[idx[3.0]].mean())
        diverged = bool(np.any(C > 10.0 * C[0].mean() + 1.0))
        stable = "DIVERGED" if diverged else "bounded"
        flag = f"{stable}{' +NEG' if neg else ''}{' +NaN' if nan else ''}"
        if a_rc == 0.0:
            c_ref = c3
        rel = "" if c_ref is None else f" ({(c3 / c_ref - 1) * 100:+.1f}% vs α=0)"
        print(f"  {a_rc:>5.2f} {c1:>8.4f} {c2:>8.4f} {c3:>8.4f} {c_star:>10.4f} "
              f"{float(C.min()):>8.4f} {float(C.max()):>8.4f} {neg:>4d} {nan:>4d} "
              f"{flag:>12}{rel}")
        results.append({"alpha_rc": a_rc, "C_1y": c1, "C_2y": c2, "C_3y": c3,
                        "C_star_analytic": c_star, "min_C": float(C.min()),
                        "max_C": float(C.max()), "negatives": neg, "nan": nan,
                        "phi_final": phi_final, "stable": not diverged and not neg and not nan})
    print("\n[note] C* analytic = κ_C·(1−Φ*)/(γ_C·σ + α_rc·R) with frozen final R —")
    print("       the −α_rc·R·C term is a dissipation channel: higher α_rc ⇒ lower C*.")
    print("[note] α_rc is [THEORY-LEDGER minimal patch 2026-09-10, UNCALIBRATED];")
    print("       this probe documents sensitivity, it does not calibrate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
