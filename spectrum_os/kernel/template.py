"""Pattern matching — empirical template library for deviation signatures.

Each template captures the characteristic spectral fingerprint of a sector
at a particular historical period: its dominant periods, their amplitudes,
and the mean deviation.
"""

import numpy as np
from . import sector as sector_store
from . import wave

# ---------------------------------------------------------------------------
# Threshold constants (adjustable)
# ---------------------------------------------------------------------------
_DEFAULT_SIMILARITY_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# In-memory template library
# ---------------------------------------------------------------------------
#   library_name -> list[dict]
#   Each template dict:
#       {"name": str, "periods": list[int], "amplitudes": list[float],
#        "mean_deviation": float}
_templates: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _signature(sector_id: str) -> dict:
    """Extract deviation signature from a sector.

    Returns
    -------
    dict with keys ``periods``, ``amplitudes``, ``mean_deviation``.
    """
    sec = sector_store.get(sector_id)
    if sec is None:
        raise ValueError(f"Sector '{sector_id}' not found")

    arr = np.array(sec.timeseries, dtype=np.float64)
    targets = np.array(sec.targets, dtype=np.float64) if sec.targets is not None else None

    result = wave.decompose(arr, targets)
    deviation = result["deviation"]
    periods = result["dominant_periods"]

    if deviation is not None:
        valid = deviation[result["valid_range"][0]:result["valid_range"][1] + 1]
        finite = valid[np.isfinite(valid)]
        mean_dev = float(np.mean(finite)) if len(finite) > 0 else 0.0
    else:
        mean_dev = 0.0

    # Amplitudes: standard deviation of each dominant period's contribution
    # Use autocorrelation values as amplitude proxies
    amplitudes = []
    for p in periods:
        # Find autocorrelation at this lag → amplitude estimate
        centered = arr - np.mean(arr)
        variance = np.var(arr)
        if variance > 0 and p < len(arr):
            r = float(np.sum(centered[:len(arr) - p] * centered[p:]) / (len(arr) * variance))
            amplitudes.append(abs(r))
        else:
            amplitudes.append(0.0)

    return {
        "periods": sorted(periods),
        "amplitudes": amplitudes,
        "mean_deviation": mean_dev,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register(name: str, pattern: dict, library: str = "default"):
    """Register an empirical pattern template.

    Parameters
    ----------
    name
        Unique name for this template (e.g. ``"pre-war-boom"``).
    pattern
        Dict with keys ``periods`` (list[int]), ``amplitudes`` (list[float]),
        and ``mean_deviation`` (float).
    library
        Library name (default ``"default"``).
    """
    if library not in _templates:
        _templates[library] = []

    # Check for duplicate name
    for t in _templates[library]:
        if t["name"] == name:
            raise ValueError(f"Template '{name}' already exists in library '{library}'")

    _templates[library].append({
        "name": name,
        "periods": sorted(pattern.get("periods", [])),
        "amplitudes": pattern.get("amplitudes", []),
        "mean_deviation": pattern.get("mean_deviation", 0.0),
    })


def match(sector_id: str, library: str = "default") -> dict:
    """Match a sector's deviation signature against all templates in a library.

    Similarity is computed as a weighted combination:

    * 50 % — Jaccard index on period sets
    * 30 % — inverse normalised MSE on amplitude vectors
    * 20 % — inverse absolute difference on mean deviation

    Returns
    -------
    dict with keys ``best_match`` (str | None), ``similarity`` (float),
    ``all_scores`` (dict[str, float]).
    """
    if library not in _templates or not _templates[library]:
        return {"best_match": None, "similarity": 0.0, "all_scores": {}}

    sig = _signature(sector_id)
    sig_periods = set(sig["periods"])
    sig_amps = np.array(sig["amplitudes"], dtype=np.float64)
    sig_mean_dev = sig["mean_deviation"]

    scores: dict[str, float] = {}

    for tmpl in _templates[library]:
        # --- Jaccard on periods (0.5 weight) ---
        tmpl_periods = set(tmpl["periods"])
        if len(sig_periods) == 0 and len(tmpl_periods) == 0:
            jaccard = 1.0
        elif len(sig_periods) == 0 or len(tmpl_periods) == 0:
            jaccard = 0.0
        else:
            intersection = sig_periods & tmpl_periods
            union = sig_periods | tmpl_periods
            jaccard = len(intersection) / len(union)

        # --- Inverse MSE on amplitudes (0.3 weight) ---
        tmpl_amps = np.array(tmpl["amplitudes"], dtype=np.float64)
        min_len = min(len(sig_amps), len(tmpl_amps))
        if min_len > 0:
            mse = float(np.mean((sig_amps[:min_len] - tmpl_amps[:min_len]) ** 2))
            amp_score = 1.0 / (1.0 + mse)
        else:
            amp_score = 1.0 if len(sig_amps) == len(tmpl_amps) else 0.0

        # --- Inverse abs diff on mean deviation (0.2 weight) ---
        dev_diff = abs(sig_mean_dev - tmpl["mean_deviation"])
        dev_score = 1.0 / (1.0 + dev_diff)

        similarity = 0.5 * jaccard + 0.3 * amp_score + 0.2 * dev_score
        scores[tmpl["name"]] = similarity

    if not scores:
        return {"best_match": None, "similarity": 0.0, "all_scores": {}}

    best_name = max(scores, key=scores.get)
    best_sim = scores[best_name]

    # If below threshold, treat as no match
    if best_sim < _DEFAULT_SIMILARITY_THRESHOLD:
        best_name = None

    return {"best_match": best_name, "similarity": best_sim, "all_scores": scores}


def list_templates(library: str = "default") -> list[dict]:
    """List all templates in a library."""
    return _templates.get(library, [])
