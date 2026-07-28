"""Convenience wrappers — sector-ID-based lookups for wave operations."""

import numpy as np
from . import sector as sector_store
from . import wave
from ._types import SpectrumResult, Verdict


def decompose(sector_id: str, long_window: int = 36, mid_window: int = 6) -> SpectrumResult:
    """Look up sector by ID, then call :func:`wave.decompose`.

    Parameters
    ----------
    sector_id
        The registered sector identifier.
    long_window, mid_window
        Passed through to :func:`wave.decompose`.

    Returns
    -------
    :class:`SpectrumResult` with *values* containing ``longwave``, ``midwave``,
    ``shortwave``, and ``deviation`` arrays.
    """
    sec = sector_store.get(sector_id)
    if sec is None:
        raise ValueError(f"Sector '{sector_id}' not found")

    synthetic = bool(sec.meta and sec.meta.get("synthetic") is True)

    arr = np.array(sec.timeseries, dtype=np.float64)
    targets = np.array(sec.targets, dtype=np.float64) if sec.targets is not None else None

    result = wave.decompose(arr, targets, long_window, mid_window,
                            synthetic=synthetic)

    return SpectrumResult(
        values={
            "longwave": result["longwave"],
            "midwave": result["midwave"],
            "shortwave": result["shortwave"],
            "deviation": result["deviation"],
            "synthetic_input": result["synthetic_input"],
        },
        verdict=result["verdict"],
        confidence_reason=result["confidence_reason"],
        dominant_periods=result["dominant_periods"],
        valid_range=result["valid_range"],
        boundary_note=("synthetic input — verdict capped at CONTESTED"
                       if synthetic else None),
    )


def extrapolate(sector_id: str, horizon: int = 12, long_window: int = 36,
                n_simulations: int = 200, seed: int | None = None) -> dict:
    """Look up sector by ID, then call :func:`wave.extrapolate`.

    Parameters
    ----------
    sector_id
        The registered sector identifier.
    horizon
        Number of future steps to project.
    long_window
        Window for the moving-average trend.
    n_simulations
        Monte Carlo resamples for confidence bands.
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict with ``forecast``, ``lower``, ``upper``, ``verdict``, etc.
    """
    sec = sector_store.get(sector_id)
    if sec is None:
        raise ValueError(f"Sector '{sector_id}' not found")

    arr = np.array(sec.timeseries, dtype=np.float64)
    targets = np.array(sec.targets, dtype=np.float64) if sec.targets is not None else None

    result = wave.extrapolate(
        arr, targets=targets, horizon=horizon, long_window=long_window,
        n_simulations=n_simulations, seed=seed,
    )

    # Synthetic ceiling
    if sec.meta and sec.meta.get("synthetic") is True:
        if result["verdict"] == Verdict.ASSERTED:
            result["verdict"] = Verdict.CONTESTED
        result["confidence_reason"] += " [synthetic ceiling applied]"
        result["synthetic_input"] = True
    else:
        result["synthetic_input"] = False

    return result


def correlate(a_id: str, b_id: str, max_lag: int = 24) -> dict:
    """Look up two sectors by ID, then call :func:`wave.correlate`.

    Parameters
    ----------
    a_id, b_id
        Registered sector identifiers.
    max_lag
        Maximum lag (passed to :func:`wave.correlate`).

    Returns
    -------
    dict with keys ``r``, ``lag``, ``equiv_lags``, and ``synthetic_input``
    (True when either sector is registered as synthetic).
    """
    sec_a = sector_store.get(a_id)
    if sec_a is None:
        raise ValueError(f"Sector '{a_id}' not found")
    sec_b = sector_store.get(b_id)
    if sec_b is None:
        raise ValueError(f"Sector '{b_id}' not found")

    synthetic = any(
        bool(s.meta and s.meta.get("synthetic") is True) for s in (sec_a, sec_b)
    )

    arr_a = np.array(sec_a.timeseries, dtype=np.float64)
    arr_b = np.array(sec_b.timeseries, dtype=np.float64)

    return wave.correlate(arr_a, arr_b, max_lag, synthetic=synthetic)
