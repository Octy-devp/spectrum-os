#!/usr/bin/env python3
"""Liveness Diagnostic Script for Rate Gate (PLAN-23 §6.6.2 & Task 3.8).

Evaluates whether the rate gate prompt is responsive to state variations ("live water drop")
or outputting state-independent static distributions ("collapsed prompt").

Computes pairwise KL divergence of rate distributions across N>=5 distinct states.
Default mode is dry-run (mocked API). Pass --live to execute real API calls.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from spectrum_os.quantum import kl_divergence
from spectrum_os.synth import rate_gate

# N=5 distinct input states
SAMPLE_STATES = [
    {
        "name": "State 1 — Crisis Disruption",
        "input": {
            "state_vector": {"d1": -1, "d2": 1, "d3": -1, "d4": 12.0, "d5": 0.3, "d6": 0.9},
            "thread_history": {"months_in_current_role": 18, "recent_transitions": ["crisis", "crisis", "lag"]},
            "local_texture": {"steel_output_tons": 5000000, "railway_delays_days": 14},
            "legal_exits": ["crisis", "lag", "alternative"],
            "realized_counts": {"crisis": 12, "lag": 3, "alternative": 0},
        },
        "mock_rates": {"crisis": 0.70, "lag": 0.25, "alternative": 0.05},
    },
    {
        "name": "State 2 — High Lag Stagnation",
        "input": {
            "state_vector": {"d1": 0, "d2": 1, "d3": 0, "d4": 24.0, "d5": -0.1, "d6": 0.4},
            "thread_history": {"months_in_current_role": 36, "recent_transitions": ["lag", "lag", "lag"]},
            "local_texture": {"steel_output_tons": 12000000, "railway_delays_days": 2},
            "legal_exits": ["crisis", "lag", "alternative"],
            "realized_counts": {"crisis": 1, "lag": 20, "alternative": 2},
        },
        "mock_rates": {"crisis": 0.10, "lag": 0.75, "alternative": 0.15},
    },
    {
        "name": "State 3 — Alternative Expansion",
        "input": {
            "state_vector": {"d1": 1, "d2": -1, "d3": 1, "d4": 6.0, "d5": 0.8, "d6": 0.7},
            "thread_history": {"months_in_current_role": 8, "recent_transitions": ["lag", "alternative", "alternative"]},
            "local_texture": {"steel_output_tons": 22000000, "railway_delays_days": 0},
            "legal_exits": ["lag", "alternative", "direction"],
            "realized_counts": {"lag": 2, "alternative": 15, "direction": 5},
        },
        "mock_rates": {"lag": 0.15, "alternative": 0.65, "direction": 0.20},
    },
    {
        "name": "State 4 — Direction Boom Peak",
        "input": {
            "state_vector": {"d1": 1, "d2": -1, "d3": 1, "d4": 12.0, "d5": 1.2, "d6": 0.95},
            "thread_history": {"months_in_current_role": 12, "recent_transitions": ["alternative", "direction", "direction"]},
            "local_texture": {"steel_output_tons": 35000000, "railway_delays_days": 0},
            "legal_exits": ["alternative", "direction", "crisis"],
            "realized_counts": {"alternative": 3, "direction": 18, "crisis": 0},
        },
        "mock_rates": {"alternative": 0.15, "direction": 0.80, "crisis": 0.05},
    },
    {
        "name": "State 5 — Balanced Transition Threshold",
        "input": {
            "state_vector": {"d1": 0, "d2": 0, "d3": 0, "d4": 12.0, "d5": 0.0, "d6": 0.5},
            "thread_history": {"months_in_current_role": 4, "recent_transitions": ["crisis", "lag", "alternative"]},
            "local_texture": {"steel_output_tons": 16000000, "railway_delays_days": 4},
            "legal_exits": ["crisis", "lag", "alternative"],
            "realized_counts": {"crisis": 4, "lag": 4, "alternative": 4},
        },
        "mock_rates": {"crisis": 0.33, "lag": 0.34, "alternative": 0.33},
    },
]


def run_liveness_diagnostic(live: bool = False, n_samples: int = 5) -> dict:
    """Run rate gate liveness diagnostic over N=5 states.

    Args:
        live: If True, call actual API; if False, run dry-run with mock API.
        n_samples: Number of sampling draws per state.

    Returns:
        Diagnostic report dictionary.
    """
    state_results = []
    mean_rate_dicts = []

    for item in SAMPLE_STATES:
        state_name = item["name"]
        inp = item["input"]
        mock_r = item["mock_rates"]

        if not live:
            # Mock API function
            def mock_api_fn(prompt, api_key, **kwargs):
                return json.dumps({
                    "walk": {"crisis": "處境限制", "lag": "進度微滯", "alive_exits": inp["legal_exits"][:2]},
                    "rates": mock_r,
                    "evidence": ["指標數據穩定", "當前月數對齊"],
                })

            res = rate_gate(
                inp,
                call_api_fn=mock_api_fn,
                api_key="mock-key",
                n_samples=n_samples,
                w_prior=5.0,
            )
        else:
            res = rate_gate(inp, n_samples=n_samples, w_prior=5.0)

        mean_rates = res["fused_rates"]
        state_results.append({
            "name": state_name,
            "legal_exits": inp["legal_exits"],
            "fused_rates": mean_rates,
            "confidence": res["confidence"],
        })
        mean_rate_dicts.append(mean_rates)

    # Compute pairwise KL divergence matrix
    N = len(SAMPLE_STATES)
    kl_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i != j:
                kl_matrix[i, j] = kl_divergence(mean_rate_dicts[i], mean_rate_dicts[j])

    # Off-diagonal average KL divergence
    off_diag_kl = [kl_matrix[i, j] for i in range(N) for j in range(N) if i != j]
    mean_kl = float(np.mean(off_diag_kl)) if off_diag_kl else 0.0

    verdict = "ALIVE" if mean_kl > 0.05 else "COLLAPSED"

    report = {
        "mode": "live" if live else "dry-run",
        "n_states": N,
        "n_samples_per_state": n_samples,
        "mean_pairwise_kl": round(mean_kl, 4),
        "verdict": verdict,
        "kl_matrix": np.round(kl_matrix, 4).tolist(),
        "state_results": state_results,
    }

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rate Gate Liveness Diagnostic")
    parser.add_argument("--live", action="store_true", help="Run with real API calls")
    args = parser.parse_args()

    report = run_liveness_diagnostic(live=args.live)
    print("=" * 60)
    print(f"RATE GATE LIVENESS DIAGNOSTIC REPORT ({report['mode'].upper()})")
    print("=" * 60)
    print(f"N States: {report['n_states']} | Samples/State: {report['n_samples_per_state']}")
    print(f"Mean Pairwise KL Divergence: {report['mean_pairwise_kl']:.4f}")
    print(f"Verdict: {report['verdict']}")
    print("-" * 60)
    for res in report["state_results"]:
        rates_str = ", ".join(f"{k}: {v:.3f}" for k, v in res["fused_rates"].items())
        print(f"[{res['name']}] -> {rates_str}")
    print("=" * 60)
