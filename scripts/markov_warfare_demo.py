"""Markov Warfare Demo Script (PLAN-23 Task 3 & 4).

Loads Track 0 DCA branch tree, extracts role sequences along threads,
estimates rate matrix, performs Dirichlet posterior sampling, and computes
Markov dynamics metrics (entropy rate, spectral gap, hitting times, absorption).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import numpy as np

from spectrum_os.quantum import (
    absorption_probabilities,
    count_transitions,
    entropy_rate,
    estimate_rate_matrix,
    hitting_times,
    load_warfare_tree,
    sample_rate_matrix,
    spectral_gap,
    summarize_samples,
)

TRACK0_PATH = "/home/octy/projects/ECC/index/warfare/data/fields/experiment-track0/dca-branch-tree.yaml"
DEFAULT_OUTPUT_PATH = "data/markov_demo_report.json"


def run_demo(
    tree_path: str = TRACK0_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    n_samples: int = 200,
    seed: int = 42,
) -> dict:
    """Run the Markov warfare demo on a DCA tree and write the report to JSON.

    Args:
        tree_path: Path to DCA branch tree YAML file.
        output_path: Path where JSON report will be saved.
        n_samples: Number of posterior rate matrix samples.
        seed: Random seed for posterior sampling.

    Returns:
        Report dictionary.
    """
    if not os.path.exists(tree_path):
        raise FileNotFoundError(f"DCA warfare tree not found: {tree_path}")

    graph = load_warfare_tree(tree_path)

    # Extract role assignments grouped by thread
    thread_assignments: dict[str, list] = {}
    for _sid, assign_list in graph.assignments.items():
        for a in assign_list:
            thread_assignments.setdefault(a.thread, []).append(a)

    # Extract role sequences from non-stale assignments along each thread
    role_sequences = []
    for _thread_name, assignments in thread_assignments.items():
        seq = [a.role for a in assignments if not a.stale]
        if seq:
            role_sequences.append(seq)

    counts, anomalies = count_transitions(role_sequences)
    rate_est = estimate_rate_matrix(counts, alpha=1.0, min_total_count=10.0)
    matrix = rate_est["matrix"]

    samples = sample_rate_matrix(counts, alpha=1.0, n=n_samples, seed=seed)

    pt_entropy = entropy_rate(matrix)
    pt_gap = spectral_gap(matrix)

    sample_entropies = [entropy_rate(s) for s in samples]
    sample_gaps = [spectral_gap(s) for s in samples]

    entropy_summary = summarize_samples(sample_entropies)
    gap_summary = summarize_samples(sample_gaps)

    ht = hitting_times(matrix, target="direction")
    abs_prob = absorption_probabilities(matrix, absorbing=["direction", "crisis"])

    report = {
        "track0_path": tree_path,
        "total_sequences": len(role_sequences),
        "transition_counts": counts.tolist(),
        "anomalies": anomalies,
        "rate_matrix": matrix.tolist(),
        "verdict": rate_est["verdict"],
        "total_count": float(rate_est["total_count"]),
        "point_estimates": {
            "entropy_rate": float(pt_entropy),
            "spectral_gap": float(pt_gap),
        },
        "posterior_samples": {
            "entropy_rate": entropy_summary,
            "spectral_gap": gap_summary,
        },
        "hitting_times_to_direction": ht,
        "absorption_probabilities": abs_prob,
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return report


if __name__ == "__main__":
    report = run_demo()
    print(f"Report successfully written to {DEFAULT_OUTPUT_PATH}")
    print(f"Total sequences processed: {report['total_sequences']}")
    print(f"Verdict: {report['verdict']} (total transitions: {report['total_count']})")
    print(f"Point Entropy Rate: {report['point_estimates']['entropy_rate']:.4f}")
    print(f"Point Spectral Gap: {report['point_estimates']['spectral_gap']:.4f}")
