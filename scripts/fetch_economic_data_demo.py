#!/usr/bin/env python3
"""World Economic Data 40-Year Spectrum & Markov Dynamics Experiment (1984-2024).

Fetches official World Bank macro economic time series (World, US, China, Germany
GDP growth & World Inflation), computes FFT amplitude/period spectra via
spectrum_os.extras.fft_spectrum, discretizes economic regimes into DCA roles,
and runs the markov.py kernel to compute transition rates, hitting times,
entropy rate, and spectral gap.
"""

import json
import urllib.request
from pathlib import Path
import numpy as np

from spectrum_os.extras.fft_spectrum import fft_decompose
from spectrum_os.quantum.markov import (
    count_transitions, count_transitions_soft, estimate_rate_matrix, sample_rate_matrix,
    hitting_times, absorption_probabilities, stationary_distribution,
    entropy_rate, spectral_gap, summarize_samples, smoothed_role_posteriors
)
from spectrum_os.quantum.multigraph import ROLES

INDICATORS = {
    "world_gdp_growth": ("WLD", "NY.GDP.MKTP.KD.ZG", "World GDP Growth (Annual %)"),
    "us_gdp_growth": ("USA", "NY.GDP.MKTP.KD.ZG", "US GDP Growth (Annual %)"),
    "china_gdp_growth": ("CHN", "NY.GDP.MKTP.KD.ZG", "China GDP Growth (Annual %)"),
    "germany_gdp_growth": ("DEU", "NY.GDP.MKTP.KD.ZG", "Germany GDP Growth (Annual %)"),
    "world_inflation": ("WLD", "FP.CPI.TOTL.ZG", "World Inflation CPI (Annual %)"),
}


def fetch_worldbank_series(country: str, indicator: str, start_year: int = 1984, end_year: int = 2024) -> list[tuple[int, float]]:
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start_year}:{end_year}&format=json&per_page=100"
    try:
        req = urllib.request.urlopen(url)
        data = json.loads(req.read().decode("utf-8"))
        if len(data) > 1 and data[1]:
            records = [
                (int(item["date"]), float(item["value"]))
                for item in data[1]
                if item["value"] is not None
            ]
            return sorted(records, key=lambda x: x[0])
    except Exception as e:
        print(f"Error fetching {country} {indicator}: {e}")
    return []


def discretize_gdp_to_dca_role(val: float) -> str:
    """Discretize economic growth rate into DCA role:
    - Crisis: < 1.0% (recession / growth collapse)
    - Lag: 1.0% <= val < 3.0% (stagnant / low growth recovery)
    - Alternative: 3.0% <= val < 4.5% (moderate expansion)
    - Direction: >= 4.5% (boom / high expansion peak)
    """
    if val < 1.0:
        return "crisis"
    elif val < 3.0:
        return "lag"
    elif val < 4.5:
        return "alternative"
    else:
        return "direction"


def compute_gdp_emission_loglik(val: float, sigma: float = 1.0) -> list[float]:
    """Compute log P(obs_t = val | role_k) using Gaussian emission likelihood.

    Programmer semantic choice (docstring explicit):
    Role centers:
    - Crisis (0): 0.0%
    - Lag (1): 2.0%
    - Alternative (2): 3.75%
    - Direction (3): 5.5%
    Log likelihood is unnormalized Gaussian: -0.5 * ((val - center) / sigma)^2.
    """
    centers = [0.0, 2.0, 3.75, 5.5]
    return [-0.5 * ((val - c) / sigma) ** 2 for c in centers]


def run_experiment(out_path: str = "data/world_economic_spectrum_report.json") -> dict:
    print("Fetching 40-year world economic data from World Bank (1984-2024)...")
    
    indicator_data = {}
    spectral_analysis = {}
    role_sequences = []
    gdp_series_list = []

    for key, (country, ind_code, label) in INDICATORS.items():
        records = fetch_worldbank_series(country, ind_code)
        if not records:
            continue

        years = [r[0] for r in records]
        values = [r[1] for r in records]
        indicator_data[key] = {
            "label": label,
            "years": years,
            "values": values,
            "min_val": min(values),
            "max_val": max(values),
            "mean_val": float(np.mean(values)),
        }

        # 1. Experimental FFT Spectrum Analysis
        fft_res = fft_decompose(values, top_k=5)
        top_peaks = []
        for p in fft_res.values.get("peaks", []):
            freq = p["freq_per_month"]  # per step
            period_years = 1.0 / freq if freq > 0 else float("inf")
            top_peaks.append({
                "period_years": round(period_years, 2) if np.isfinite(period_years) else "inf",
                "amplitude": round(p["amplitude"], 4),
                "amplitude_share": round(p["amplitude_share"], 4),
            })
        
        spectral_analysis[key] = {
            "label": label,
            "verdict": fft_res.verdict.name,
            "confidence_reason": fft_res.confidence_reason,
            "top_cycle_peaks": top_peaks,
        }

        # 2. Convert to DCA Role Sequence for Markov Dynamics
        if "gdp" in key:
            seq = [discretize_gdp_to_dca_role(v) for v in values]
            role_sequences.append(seq)
            gdp_series_list.append(values)

    # 3. Markov Dynamics Kernel on Economic Sequences (Hard vs Soft)
    counts_hard, anomalies = count_transitions(role_sequences)
    rate_res_hard = estimate_rate_matrix(counts_hard, alpha=1.0)
    P_hard = rate_res_hard["matrix"]

    # HMM Soft Reconstruction
    total_soft_counts = np.zeros((4, 4), dtype=np.float64)
    for vals in gdp_series_list:
        emission_log = np.array([compute_gdp_emission_loglik(v) for v in vals])
        hmm_res = smoothed_role_posteriors(emission_log, P_hard)
        xi = hmm_res["xi"]
        total_soft_counts += count_transitions_soft(xi)

    rate_res_soft = estimate_rate_matrix(total_soft_counts, alpha=1.0)
    P_soft = rate_res_soft["matrix"]

    samples = sample_rate_matrix(total_soft_counts, alpha=1.0, n=200, seed=42)

    ht_point = hitting_times(P_soft, target="direction")
    abs_point = absorption_probabilities(P_soft, absorbing=["direction", "crisis"])
    pi_point = stationary_distribution(P_soft)
    h_point = entropy_rate(P_soft)
    sg_point = spectral_gap(P_soft)

    h_samples = [entropy_rate(P) for P in samples]
    sg_samples = [spectral_gap(P) for P in samples]

    report = {
        "title": "World Macroeconomic Time Series 40-Year Spectrum & Markov Dynamics (1984-2024)",
        "source": "World Bank Open Data API",
        "num_indicators": len(indicator_data),
        "indicators_summary": {k: v["label"] for k, v in indicator_data.items()},
        "spectral_analysis": spectral_analysis,
        "markov_dynamics": {
            "num_economic_sequences": len(role_sequences),
            "total_transitions_hard": int(counts_hard.sum()),
            "total_transitions_soft": float(total_soft_counts.sum()),
            "verdict": rate_res_soft["verdict"],
            "anomalies": anomalies,
            "transition_counts_hard": counts_hard.tolist(),
            "transition_counts_soft": total_soft_counts.tolist(),
            "rate_matrix_hard": P_hard.tolist(),
            "rate_matrix_soft": P_soft.tolist(),
            "stationary_distribution_soft": {r: round(float(pi_point[i]), 4) for i, r in enumerate(ROLES)},
            "entropy_rate_soft": {
                "point": round(h_point, 4),
                "interval_95ci": summarize_samples(h_samples)
            },
            "spectral_gap_soft": {
                "point": round(sg_point, 4),
                "interval_95ci": summarize_samples(sg_samples)
            },
            "hitting_times_to_peak_boom": {r: round(v, 2) for r, v in ht_point.items()},
            "absorption_probabilities_to_direction_vs_crisis": abs_point,
        }
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report written to {out_file}")
    return report


if __name__ == "__main__":
    run_experiment()

