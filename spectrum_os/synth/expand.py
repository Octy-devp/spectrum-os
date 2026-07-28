"""synth.expand — pure-numpy expansion of anchors into monthly series.

PLAN-23 §7.3: 分段階梯 + 事件衝擊 + AR 噪聲，固定 seed 可重現，
🔴 禁用線性內插 (Stage 0 教訓 — 線性內插製造週期偽影, FALSIFY-001)。

Semantics
---------
- ``plan``   = piecewise-CONSTANT step function from ``turning_points``
  (level at month_index holds until the next turning point; ``baseline``
  holds before the first one).  No shocks, no noise — the plan does not
  know the future.
- ``actual`` = plan + event shocks + AR(1) noise.

Deviation ``(actual − plan)/plan`` is therefore driven entirely by shocks
and noise — the NSPV plan/actual story.
"""

import numpy as np

from ..kernel import sector as sector_store
from .anchors import assert_contract

_DEFAULT_AR_RHO = 0.3
_DEFAULT_NOISE_FRACTION = 0.02  # of baseline, when noise_sigma not given


def _step_series(anchors: dict, n_months: int) -> np.ndarray:
    """Piecewise-constant step series from turning points.  No interpolation."""
    mags = anchors["magnitudes"]
    base = np.full(n_months, float(mags["baseline"]), dtype=np.float64)
    tps = sorted(anchors["turning_points"], key=lambda t: t["month_index"])
    for tp in tps:
        m = tp["month_index"]
        if 0 <= m < n_months:
            base[m:] = float(tp["level"])
    return base


def expand(anchors: dict, n_months: int, seed: int, *,
           include_shocks: bool = True,
           include_noise: bool = True) -> np.ndarray:
    """Expand anchors into a monthly series.

    Parameters
    ----------
    anchors
        Contract-checked anchors dict (re-validated here — the consumer
        side of the mechanical contract).
    n_months
        Length of the output series.
    seed
        RNG seed for the AR(1) noise.  Same anchors + same seed →
        bit-identical output.
    include_shocks
        Apply ``event_shocks`` (signed steps; ``duration_months == 0``
        means permanent).
    include_noise
        Add AR(1) noise with ``magnitudes.noise_sigma`` /
        ``magnitudes.ar_rho`` (defaults: 2% of baseline, rho 0.3).
    """
    if n_months < 1:
        raise ValueError("n_months must be >= 1")
    assert_contract(anchors, n_months=n_months)

    series = _step_series(anchors, n_months)

    if include_shocks:
        for sh in anchors["event_shocks"]:
            m = sh["month_index"]
            if not (0 <= m < n_months):
                continue
            delta = float(sh["delta"])
            dur = int(sh["duration_months"])
            if dur <= 0:
                series[m:] += delta
            else:
                series[m:m + dur] += delta

    if include_noise:
        mags = anchors["magnitudes"]
        baseline = float(mags["baseline"])
        sigma = float(mags.get("noise_sigma")
                      or _DEFAULT_NOISE_FRACTION * baseline)
        rho = float(mags.get("ar_rho", _DEFAULT_AR_RHO))
        rng = np.random.default_rng(seed)
        eps = rng.normal(0.0, sigma, n_months)
        noise = np.empty(n_months, dtype=np.float64)
        acc = 0.0
        for t in range(n_months):
            acc = rho * acc + eps[t]
            noise[t] = acc
        series = series + noise

    return series


def expand_plan(anchors: dict, n_months: int) -> np.ndarray:
    """The plan series: turning-point steps only — no shocks, no noise."""
    return expand(anchors, n_months, seed=0,
                  include_shocks=False, include_noise=False)


def expand_pair(anchors: dict, n_months: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(actual, plan)`` for an anchors dict."""
    actual = expand(anchors, n_months, seed,
                    include_shocks=True, include_noise=True)
    plan = expand_plan(anchors, n_months)
    return actual, plan


def register_synthetic_sector(name: str, anchors: dict, n_months: int,
                              seed: int,
                              generated_by: str = "synth.expand"):
    """Expand an (actual, plan) pair and register it as a synthetic sector.

    The kernel enforces full provenance on synthetic registration
    (``meta.synthetic=True`` + anchors + generated_by + seed).
    """
    actual, plan = expand_pair(anchors, n_months, seed)
    meta = {
        "synthetic": True,
        "anchors": anchors,
        "generated_by": generated_by,
        "seed": seed,
    }
    return sector_store.create(name, actual.tolist(), plan.tolist(), meta=meta)
