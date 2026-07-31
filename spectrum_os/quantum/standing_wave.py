"""quantum.standing_wave — Concept-level Standing Wave Decomposition (PLAN-23 §5).

Pure NumPy standing wave decomposition for branch ensembles:
- nodes (inevitable): Concept tags present in ALL branches (prevalence == 1.0).
- antinodes (contingent windows): Concept tags with lowest non-zero prevalence across branches.
- divergence_curve: Role-level distribution entropy per time step across branches.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from spectrum_os.quantum.multigraph import ROLES


def standing_wave(branches: list[dict]) -> dict:
    """Decompose branch ensemble trajectories into concept-level standing waves.

    Args:
        branches: List of branch dicts containing 'roles' and 'tags_per_step'.

    Returns:
        Dict containing nodes, antinodes, divergence_curve, and meta summary.
    """
    n_branches = len(branches)
    if n_branches == 0:
        return {
            "nodes": [],
            "antinodes": [],
            "divergence_curve": [],
            "meta": {
                "total_branches": 0,
                "horizon": 0,
                "total_unique_tags": 0,
            },
        }

    horizon = max((len(b.get("roles", [])) for b in branches), default=0)

    # 1. Divergence curve (role distribution entropy per step)
    divergence_curve: list[float] = []
    for t in range(horizon):
        roles_at_t = [
            b["roles"][t] for b in branches if t < len(b.get("roles", []))
        ]
        if not roles_at_t:
            divergence_curve.append(0.0)
            continue

        counts = [roles_at_t.count(r) for r in ROLES]
        total = float(sum(counts))
        if total == 0.0:
            divergence_curve.append(0.0)
        else:
            probs = np.array([c / total for c in counts if c > 0], dtype=np.float64)
            # Shannon entropy H(t) in bits (log2)
            entropy = float(-np.sum(probs * np.log2(probs)))
            divergence_curve.append(round(entropy, 4))

    # 2. Nodes & Antinodes computation
    branch_tag_sets: list[set[str]] = []
    branch_first_steps: list[dict[str, int]] = []

    for b in branches:
        b_tags: set[str] = set()
        first_steps: dict[str, int] = {}

        tags_per_step = b.get("tags_per_step", {})
        for step_key, tag_list in tags_per_step.items():
            try:
                step_idx = int(step_key)
            except (ValueError, TypeError):
                continue

            for tag in tag_list:
                if not isinstance(tag, str) or not tag.strip():
                    continue
                tag_clean = tag.strip()
                b_tags.add(tag_clean)
                if tag_clean not in first_steps or step_idx < first_steps[tag_clean]:
                    first_steps[tag_clean] = step_idx

        branch_tag_sets.append(b_tags)
        branch_first_steps.append(first_steps)

    all_tags: set[str] = set()
    for b_tags in branch_tag_sets:
        all_tags.update(b_tags)

    tag_stats: list[dict] = []
    for tag in all_tags:
        count = sum(1 for b_tags in branch_tag_sets if tag in b_tags)
        prevalence = count / n_branches
        min_first_step = min(
            f_steps[tag] for f_steps in branch_first_steps if tag in f_steps
        )
        tag_stats.append(
            {
                "tag": tag,
                "prevalence": round(prevalence, 4),
                "first_step": min_first_step,
            }
        )

    # Nodes: prevalence == 1.0
    nodes = [
        {"tag": item["tag"], "prevalence": 1.0}
        for item in tag_stats
        if item["prevalence"] == 1.0
    ]
    nodes.sort(key=lambda x: x["tag"])

    # Antinodes: 0 < prevalence < 1.0
    contingent = [item for item in tag_stats if 0.0 < item["prevalence"] < 1.0]
    if contingent:
        min_prev = min(item["prevalence"] for item in contingent)
        # Sort contingent antinodes by prevalence ascending, then first_step ascending
        contingent.sort(key=lambda x: (x["prevalence"], x["first_step"], x["tag"]))
        antinodes = contingent
    else:
        antinodes = []

    return {
        "nodes": nodes,
        "antinodes": antinodes,
        "divergence_curve": divergence_curve,
        "meta": {
            "total_branches": n_branches,
            "horizon": horizon,
            "total_unique_tags": len(all_tags),
        },
    }
