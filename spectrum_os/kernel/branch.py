"""Monte Carlo branch simulation — explore what-if adjustments to sector targets.

Each branch applies fractional adjustments to one or more sector targets,
recomputes the deviation spectrum, and adds Gaussian noise proportional to
the historical deviation variance.
"""

import numpy as np
from . import sector as sector_store
from . import wave

# ---------------------------------------------------------------------------
# Threshold constants (adjustable)
# ---------------------------------------------------------------------------
_DEFAULT_NOISE_SCALE_FACTOR = 1.0


def simulate(adjustments: dict[str, float], n: int = 10,
             noise_scale: float | None = None) -> dict:
    """Monte Carlo branch simulation.

    Parameters
    ----------
    adjustments
        Mapping ``{sector_id: fractional_change}``, e.g. ``{"coal": -0.05}``.
    n
        Number of branches to simulate (≥ 1).
    noise_scale
        Noise scaling factor.  If ``None`` it is estimated from each sector's
        historical deviation variance.

    Returns
    -------
    dict with keys:

    * ``branches`` — list of dicts, each with ``sector_id``,
      ``original_deviation``, ``adjusted_deviation``, ``delta``.
    * ``ensemble_stats`` — dict with ``mean_delta``, ``std_delta``,
      ``n_positive``, ``n_negative``.
    * ``nodes`` — list of dicts representing features stable across all
      branches (low variance).
    * ``antinodes`` — list of dicts representing features with highest
      variance across branches.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    # Validate all sectors exist and have targets
    sector_data: dict[str, dict] = {}
    for sid, frac in adjustments.items():
        sec = sector_store.get(sid)
        if sec is None:
            raise ValueError(f"Sector '{sid}' not found")
        if sec.targets is None or len(sec.targets) == 0:
            raise ValueError(f"Sector '{sid}' has no targets — cannot simulate")

        # Store original data
        arr = np.array(sec.timeseries, dtype=np.float64)
        plan = np.array(sec.targets, dtype=np.float64)
        orig_result = wave.decompose(arr, plan)

        sector_data[sid] = {
            "timeseries": arr,
            "original_targets": plan,
            "original_deviation": orig_result["deviation"],
            "original_valid_range": orig_result["valid_range"],
        }

    # ---- Noise estimation per sector ----
    noise_sigmas: dict[str, float] = {}
    for sid, sd in sector_data.items():
        dev = sd["original_deviation"]
        if dev is not None:
            vr = sd["original_valid_range"]
            valid_dev = dev[vr[0]:vr[1] + 1]
            finite = valid_dev[np.isfinite(valid_dev)]
            if noise_scale is not None:
                noise_sigmas[sid] = noise_scale
            elif len(finite) > 1:
                noise_sigmas[sid] = float(np.std(finite)) * _DEFAULT_NOISE_SCALE_FACTOR
            else:
                noise_sigmas[sid] = 0.01
        else:
            noise_sigmas[sid] = 0.01

    # ---- Branches ----
    branches: list[dict] = []
    all_deviations: dict[str, list[np.ndarray]] = {sid: [] for sid in adjustments}

    rng = np.random.default_rng(42)

    for _ in range(n):
        for sid, frac in adjustments.items():
            sd = sector_data[sid]
            adjusted_targets = sd["original_targets"] * (1.0 + frac)

            # Recompute deviation
            adj_result = wave.decompose(sd["timeseries"], adjusted_targets)
            adj_dev = adj_result["deviation"]
            orig_dev = sd["original_deviation"]

            # Add Gaussian noise
            sigma = noise_sigmas[sid]
            if adj_dev is not None and sigma > 0:
                noise = rng.normal(0, sigma, size=len(adj_dev))
                adj_dev = adj_dev + noise

            # Compute delta (mean signed fractional change in deviation)
            vr = sd["original_valid_range"]
            delta = 0.0
            if adj_dev is not None and orig_dev is not None:
                a_seg = adj_dev[vr[0]:vr[1] + 1]
                o_seg = orig_dev[vr[0]:vr[1] + 1]
                finite_mask = np.isfinite(a_seg) & np.isfinite(o_seg)
                if np.any(finite_mask):
                    delta = float(np.mean(a_seg[finite_mask] - o_seg[finite_mask]))

            branches.append({
                "sector_id": sid,
                "original_deviation": orig_dev,
                "adjusted_deviation": adj_dev,
                "delta": delta,
            })
            all_deviations[sid].append(adj_dev)

    # ---- Ensemble stats ----
    deltas = [b["delta"] for b in branches]
    ensemble_stats = {
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "std_delta": float(np.std(deltas)) if len(deltas) > 1 else 0.0,
        "n_positive": int(np.sum(np.array(deltas) > 0)),
        "n_negative": int(np.sum(np.array(deltas) < 0)),
    }

    # ---- Nodes & antinodes (cross-branch variance) ----
    all_variances: dict[str, list[float]] = {}
    for sid, dev_list in all_deviations.items():
        # Reshape: find min length across branches
        lengths = [len(d) for d in dev_list if d is not None]
        if not lengths:
            continue
        min_len = min(lengths)
        # Stack deviations into a matrix
        stacked = np.array([d[:min_len] for d in dev_list if d is not None])
        if stacked.shape[0] < 2:
            continue
        pointwise_var = np.var(stacked, axis=0)
        all_variances[sid] = pointwise_var.tolist()

    # Nodes: features with consistently low variance (< 50th percentile)
    # Antinodes: features with highest variance (> 90th percentile)
    nodes: list[dict] = []
    antinodes: list[dict] = []
    for sid, variances in all_variances.items():
        if not variances:
            continue
        var_arr = np.array(variances)
        if len(var_arr) == 0:
            continue
        low_thresh = np.percentile(var_arr, 50) if len(var_arr) > 1 else 0.0
        high_thresh = np.percentile(var_arr, 90) if len(var_arr) > 1 else float('inf')

        low_indices = np.where(var_arr <= low_thresh)[0]
        high_indices = np.where(var_arr >= high_thresh)[0]

        for idx in low_indices:
            nodes.append({"sector_id": sid, "index": int(idx), "variance": float(var_arr[idx])})
        for idx in high_indices:
            antinodes.append({"sector_id": sid, "index": int(idx), "variance": float(var_arr[idx])})

    # Sort by variance
    nodes.sort(key=lambda x: x["variance"])
    antinodes.sort(key=lambda x: x["variance"], reverse=True)

    return {
        "branches": branches,
        "ensemble_stats": ensemble_stats,
        "nodes": nodes,
        "antinodes": antinodes,
    }
