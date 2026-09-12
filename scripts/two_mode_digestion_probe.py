#!/usr/bin/env python3
"""Two-Mode Digestion of AE Discriminative Probe — 45 Rural Panel Units (1917→1920).

Operationalises the two political modes of friction (AE/C) resolution:
  1. Digestion Mode (RLO / β₁ path):
     AE is actively eliminated and channelled into:
       - Fast channel: CEF (circulation conductivity Φ) via T_{C→Φ}·μ_CEF·[-Ċ]_+
       - Slow channel: SRI (potential transformative reservoir P_R) via T_{C→P}·μ_SRI·[-Ċ]_+
       - Granulation pool: K (irreversible friction loss / scar tissue) via T_{C→K}·ρ·[-Ċ]_+
     Parameters: μ_CEF=0.5, μ_SRI=0.3, ρ=0.2 (conservation sum = 1.0).

  2. Exclusion Mode (War Communism / Stolypin failure / Historical α₁):
     AE / C is purged along with carrier institutions. Attenuation purely dissipates
     out of the system ("衰減不入池"), so μ_CEF=0, μ_SRI=0, ρ=None.

  3. Transition Mode (1917 Switch / Scab Phase):
     Decree announces digestion, but logistics collapse and self-preservation scar:
     Phase 1 (1917.5→1918.5, 1.5y): ρ=0.6, μ_CEF=0.25, μ_SRI=0.15 (scab dominates).
     Phase 2 (1918.5→1920.0, 1.5y): ρ=0.2, μ_CEF=0.50, μ_SRI=0.30 (digestion matures).

  4. Failure Reference (Rigid Inertia / α_rc=0):
     Friction wall does not crack (Ċ=0), demonstrating that without friction decay,
     digestion mechanisms remain dormant (no free energy to harvest).

Data are read READ-ONLY from ECC nspv-index.json and 1917 rural observation volumes.
Zero writes to ECC. No git commit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure spectrum_os importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spectrum_os.kernel.forces import FastChannelConfig, ForceFieldDynamics  # noqa: E402

DEFAULT_ECC_ROOT = "/home/octy/projects/ECC"
HORIZON = 3.0   # 1917 -> 1920
DT = 0.25       # Quarterly integration

CORE_15 = [
    "berlin", "brunn", "budapest", "dresden", "graz", "hanover",
    "kharkov", "kiev", "linz", "moscow", "odessa", "salzburg",
    "st-petersburg", "voronezh", "warsaw"
]
RINGS = ["periurban", "primary", "secondary"]

SAMPLE_NODES = [
    "berlin-periurban",
    "budapest-secondary",
    "moscow-primary",
    "voronezh-secondary",
]


def load_45_rural_nodes(ecc_root: str | Path) -> list[dict]:
    root = Path(ecc_root)
    nspv_path = root / "index/locations/data/nspv-index.json"
    if not nspv_path.exists():
        raise FileNotFoundError(f"Missing {nspv_path}")

    with open(nspv_path, encoding="utf-8") as f:
        nspv_data = json.load(f)

    bpu = nspv_data.get("1917", {}).get("by_panel_unit", {})
    obs_dir = root / "index/locations/data/almanac/1917/observation"

    nodes = []
    for city in CORE_15:
        for ring in RINGS:
            unit = f"{city}-{ring}"
            if unit not in bpu:
                raise KeyError(f"Unit {unit} missing in 1917 by_panel_unit")
            d = bpu[unit]

            rlo_share = 0.297
            regime = "UNKNOWN"
            obs_file = obs_dir / f"{unit}-1917.json"
            if obs_file.exists():
                try:
                    with open(obs_file, encoding="utf-8") as of:
                        od = json.load(of)
                        gc = od.get("metrics", {}).get("grain_channel", {})
                        if "rlo_share" in gc and gc["rlo_share"] is not None:
                            rlo_share = float(gc["rlo_share"])
                        gov = od.get("metrics", {}).get("governance", {})
                        if "regime" in gov and gov["regime"]:
                            regime = str(gov["regime"])
                except Exception:
                    pass

            nodes.append({
                "unit": unit,
                "city": city,
                "ring": ring,
                "R0": float(d["NMP"] + d["SRI_proxy"] + d["CEF_proxy"]),
                "C0": float(d["AE"] + d["governance_risk"]),
                "rlo_share": rlo_share,
                "regime": regime,
            })
    return nodes


def run_scenario(nodes: list[dict], mode: str, dt: float = DT, horizon: float = HORIZON) -> dict:
    R0 = np.array([n["R0"] for n in nodes], dtype=np.float64)
    C0 = np.array([n["C0"] for n in nodes], dtype=np.float64)
    rlo = np.array([n["rlo_share"] for n in nodes], dtype=np.float64)
    n_nodes = len(nodes)

    phi0 = 0.65
    gamma_c = 0.10
    kappa_c = C0 * gamma_c / (1.0 - phi0)
    phi_suppress = 0.24
    alpha_rc = 0.10 if mode != "failure" else 0.0

    steps = int(round(horizon / dt))

    if mode == "digestion":
        cfg = FastChannelConfig(
            phi0=phi0, phi_drive=rlo, phi_suppress=phi_suppress,
            kappa_phi=1.5, zeta_phi=1.0, delta=0.0, mu_f=0.12,
            eta=0.0, lam=0.15, s_in=0.02, mu_e=0.0, gamma_k=0.0,
            kappa_c=kappa_c, gamma_c=gamma_c, alpha_rc=alpha_rc, sigma_admin=1.0,
            mu_cef=0.5, mu_sri=0.3, rho=0.2,
            t_c_phi=1.0, t_c_p=1.0, t_c_k=1.0,
        )
        eng = ForceFieldDynamics(
            R0.copy(), C0.copy(), growth_r=0.02, growth_c=0.01,
            alpha=0.05, beta=0.03, fast_channel=cfg, method="rk4", dt=dt
        )
        history = {
            "t": [0.0],
            "C": [eng._C.copy()],
            "Phi": [eng.phi.copy()],
            "P": [eng._P.copy()],
            "K": [eng.k_pool.copy()],
            "R": [eng._R.copy()],
        }
        for _ in range(steps):
            eng.step(dt)
            history["t"].append(eng.t)
            history["C"].append(eng._C.copy())
            history["Phi"].append(eng.phi.copy())
            history["P"].append(eng._P.copy())
            history["K"].append(eng.k_pool.copy())
            history["R"].append(eng._R.copy())

    elif mode == "exclusion":
        # Pure dissipation: mu_cef=0, mu_sri=0, rho=None
        cfg = FastChannelConfig(
            phi0=phi0, phi_drive=rlo, phi_suppress=phi_suppress,
            kappa_phi=1.5, zeta_phi=1.0, delta=0.0, mu_f=0.12,
            eta=0.0, lam=0.15, s_in=0.02, mu_e=0.0, gamma_k=0.0,
            kappa_c=kappa_c, gamma_c=gamma_c, alpha_rc=alpha_rc, sigma_admin=1.0,
            mu_cef=0.0, mu_sri=0.0, rho=None,
        )
        eng = ForceFieldDynamics(
            R0.copy(), C0.copy(), growth_r=0.02, growth_c=0.01,
            alpha=0.05, beta=0.03, fast_channel=cfg, method="rk4", dt=dt
        )
        history = {
            "t": [0.0],
            "C": [eng._C.copy()],
            "Phi": [eng.phi.copy()],
            "P": [eng._P.copy()],
            "K": [eng.k_pool.copy()],
            "R": [eng._R.copy()],
        }
        for _ in range(steps):
            eng.step(dt)
            history["t"].append(eng.t)
            history["C"].append(eng._C.copy())
            history["Phi"].append(eng.phi.copy())
            history["P"].append(eng._P.copy())
            history["K"].append(eng.k_pool.copy())
            history["R"].append(eng._R.copy())

    elif mode == "transition":
        # 2-stage transition: 0->1.5y scab phase, 1.5->3.0y mature digestion
        mid_steps = int(round(1.5 / dt))
        rem_steps = steps - mid_steps

        cfg1 = FastChannelConfig(
            phi0=phi0, phi_drive=rlo, phi_suppress=phi_suppress,
            kappa_phi=1.5, zeta_phi=1.0, delta=0.0, mu_f=0.12,
            eta=0.0, lam=0.15, s_in=0.02, mu_e=0.0, gamma_k=0.0,
            kappa_c=kappa_c, gamma_c=gamma_c, alpha_rc=alpha_rc, sigma_admin=1.0,
            mu_cef=0.25, mu_sri=0.15, rho=0.60,
        )
        eng1 = ForceFieldDynamics(
            R0.copy(), C0.copy(), growth_r=0.02, growth_c=0.01,
            alpha=0.05, beta=0.03, fast_channel=cfg1, method="rk4", dt=dt
        )
        history = {
            "t": [0.0],
            "C": [eng1._C.copy()],
            "Phi": [eng1.phi.copy()],
            "P": [eng1._P.copy()],
            "K": [eng1.k_pool.copy()],
            "R": [eng1._R.copy()],
        }
        for _ in range(mid_steps):
            eng1.step(dt)
            history["t"].append(eng1.t)
            history["C"].append(eng1._C.copy())
            history["Phi"].append(eng1.phi.copy())
            history["P"].append(eng1._P.copy())
            history["K"].append(eng1.k_pool.copy())
            history["R"].append(eng1._R.copy())

        cfg2 = FastChannelConfig(
            phi0=eng1.phi, k0=eng1.k_pool, phi_drive=rlo, phi_suppress=phi_suppress,
            kappa_phi=1.5, zeta_phi=1.0, delta=0.0, mu_f=0.12,
            eta=0.0, lam=0.15, s_in=0.02, mu_e=0.0, gamma_k=0.0,
            kappa_c=kappa_c, gamma_c=gamma_c, alpha_rc=alpha_rc, sigma_admin=1.0,
            mu_cef=0.50, mu_sri=0.30, rho=0.20,
        )
        eng2 = ForceFieldDynamics(
            eng1._R.copy(), eng1._C.copy(), latent_force=eng1._P.copy(),
            growth_r=0.02, growth_c=0.01, alpha=0.05, beta=0.03,
            fast_channel=cfg2, method="rk4", dt=dt
        )
        for _ in range(rem_steps):
            eng2.step(dt)
            history["t"].append(eng2.t + 1.5)
            history["C"].append(eng2._C.copy())
            history["Phi"].append(eng2.phi.copy())
            history["P"].append(eng2._P.copy())
            history["K"].append(eng2.k_pool.copy())
            history["R"].append(eng2._R.copy())

    elif mode == "failure":
        # alpha_rc=0, friction-neutral base, [-C_dot]_+ = 0
        cfg = FastChannelConfig(
            phi0=phi0, phi_drive=rlo, phi_suppress=phi_suppress,
            kappa_phi=1.5, zeta_phi=1.0, delta=0.0, mu_f=0.12,
            eta=0.0, lam=0.15, s_in=0.02, mu_e=0.0, gamma_k=0.0,
            kappa_c=kappa_c, gamma_c=gamma_c, alpha_rc=0.0, sigma_admin=1.0,
            mu_cef=0.5, mu_sri=0.3, rho=0.2,
        )
        eng = ForceFieldDynamics(
            R0.copy(), C0.copy(), growth_r=0.02, growth_c=0.01,
            alpha=0.05, beta=0.03, fast_channel=cfg, method="rk4", dt=dt
        )
        history = {
            "t": [0.0],
            "C": [eng._C.copy()],
            "Phi": [eng.phi.copy()],
            "P": [eng._P.copy()],
            "K": [eng.k_pool.copy()],
            "R": [eng._R.copy()],
        }
        for _ in range(steps):
            eng.step(dt)
            history["t"].append(eng.t)
            history["C"].append(eng._C.copy())
            history["Phi"].append(eng.phi.copy())
            history["P"].append(eng._P.copy())
            history["K"].append(eng.k_pool.copy())
            history["R"].append(eng._R.copy())
    else:
        raise ValueError(f"Unknown mode {mode}")

    # Convert histories to numpy arrays: shape (n_steps+1, n_nodes)
    for k in ("C", "Phi", "P", "K", "R"):
        history[k] = np.array(history[k])
    history["t"] = np.array(history["t"])
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ecc-root", default=DEFAULT_ECC_ROOT)
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--horizon", type=float, default=HORIZON)
    args = ap.parse_args()

    nodes = load_45_rural_nodes(args.ecc_root)
    print(f"Loaded {len(nodes)} rural panel units from {args.ecc_root}.")

    scenarios = {
        "Digestion": run_scenario(nodes, "digestion", args.dt, args.horizon),
        "Exclusion": run_scenario(nodes, "exclusion", args.dt, args.horizon),
        "Transition": run_scenario(nodes, "transition", args.dt, args.horizon),
        "Failure": run_scenario(nodes, "failure", args.dt, args.horizon),
    }

    # Summary table across time steps
    print("\n" + "=" * 80)
    print("45-VILLAGE AGGREGATE TRAJECTORY COMPARISON (MEAN ACROSS 45 UNITS)")
    print("=" * 80)
    time_indices = [0, 4, 8, 12]  # t = 0.0, 1.0, 2.0, 3.0
    times = [0.0, 1.0, 2.0, 3.0]

    header = f"{'Scenario':<12} | {'Year':<6} | {'C':<7} | {'Phi':<7} | {'P_R':<7} | {'K_pool':<7} | {'R':<7} | {'S':<7}"
    print(header)
    print("-" * len(header))

    for sname, hist in scenarios.items():
        for tidx, yr in zip(time_indices, times):
            c_mean = float(hist["C"][tidx].mean())
            phi_mean = float(hist["Phi"][tidx].mean())
            p_mean = float(hist["P"][tidx].mean())
            k_mean = float(hist["K"][tidx].mean())
            r_mean = float(hist["R"][tidx].mean())
            s_mean = float((hist["R"][tidx] / (hist["R"][tidx] + hist["C"][tidx])).mean())
            print(f"{sname:<12} | {yr:<6.1f} | {c_mean:<7.4f} | {phi_mean:<7.4f} | {p_mean:<7.4f} | {k_mean:<7.4f} | {r_mean:<7.4f} | {s_mean:<7.4f}")
        print("-" * len(header))

    # Sample villages at 1920 (t=3.0)
    print("\n" + "=" * 80)
    print("SAMPLE VILLAGES AT 1920 (t=3.0)")
    print("=" * 80)
    sample_hdr = f"{'Village':<20} | {'Regime':<6} | {'RLO':<5} | {'Scenario':<11} | {'C(3y)':<7} | {'Phi(3y)':<7} | {'P_R(3y)':<7} | {'K(3y)':<7}"
    print(sample_hdr)
    print("-" * len(sample_hdr))

    sample_id_map = {n["unit"]: i for i, n in enumerate(nodes)}
    for s_id in SAMPLE_NODES:
        idx = sample_id_map[s_id]
        n_info = nodes[idx]
        for sname, hist in scenarios.items():
            c_val = hist["C"][-1, idx]
            phi_val = hist["Phi"][-1, idx]
            p_val = hist["P"][-1, idx]
            k_val = hist["K"][-1, idx]
            print(f"{s_id:<20} | {n_info['regime']:<6} | {n_info['rlo_share']:<5.2f} | {sname:<11} | {c_val:<7.4f} | {phi_val:<7.4f} | {p_val:<7.4f} | {k_val:<7.4f}")
        print("-" * len(sample_hdr))

    # Regime aggregation of K_pool and P_R at 1920
    print("\n" + "=" * 80)
    print("REGIME AGGREGATION AT 1920 (t=3.0): GRANULATION K & POTENTIAL P_R")
    print("=" * 80)
    reg_hdr = f"{'Regime':<8} | {'N':<3} | {'RLO mean':<8} | {'K(Dig)':<8} | {'K(Trans)':<8} | {'P_R(Dig)':<8} | {'P_R(Trans)':<9} | {'P_R(Excl)':<9}"
    print(reg_hdr)
    print("-" * len(reg_hdr))

    regimes = sorted(set(n["regime"] for n in nodes))
    for reg in regimes:
        reg_idxs = [i for i, n in enumerate(nodes) if n["regime"] == reg]
        rlo_m = float(np.mean([nodes[i]["rlo_share"] for i in reg_idxs]))
        k_dig = float(scenarios["Digestion"]["K"][-1, reg_idxs].mean())
        k_tr = float(scenarios["Transition"]["K"][-1, reg_idxs].mean())
        pr_dig = float(scenarios["Digestion"]["P"][-1, reg_idxs].mean())
        pr_tr = float(scenarios["Transition"]["P"][-1, reg_idxs].mean())
        pr_ex = float(scenarios["Exclusion"]["P"][-1, reg_idxs].mean())
        print(f"{reg:<8} | {len(reg_idxs):<3} | {rlo_m:<8.3f} | {k_dig:<8.4f} | {k_tr:<8.4f} | {pr_dig:<8.4f} | {pr_tr:<9.4f} | {pr_ex:<9.4f}")
    print("-" * len(reg_hdr))


if __name__ == "__main__":
    main()
