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

    arr = np.array(sec.timeseries, dtype=np.float64)
    targets = np.array(sec.targets, dtype=np.float64) if sec.targets is not None else None

    result = wave.decompose(arr, targets, long_window, mid_window)

    return SpectrumResult(
        values={
            "longwave": result["longwave"],
            "midwave": result["midwave"],
            "shortwave": result["shortwave"],
            "deviation": result["deviation"],
        },
        verdict=result["verdict"],
        confidence_reason=result["confidence_reason"],
        dominant_periods=result["dominant_periods"],
        valid_range=result["valid_range"],
    )


def correlate(a_id: str, b_id: str, max_lag: int = 24) -> dict:
    """Look up two sectors by ID, then call :func:`wave.correlate`.

    Parameters
    ----------
    a_id, b_id
        Registered sector identifiers.
    max_lag
        Maximum lag in months (passed to :func:`wave.correlate`).

    Returns
    -------
    dict with keys ``r`` and ``lag_months``.
    """
    sec_a = sector_store.get(a_id)
    if sec_a is None:
        raise ValueError(f"Sector '{a_id}' not found")
    sec_b = sector_store.get(b_id)
    if sec_b is None:
        raise ValueError(f"Sector '{b_id}' not found")

    arr_a = np.array(sec_a.timeseries, dtype=np.float64)
    arr_b = np.array(sec_b.timeseries, dtype=np.float64)

    return wave.correlate(arr_a, arr_b, max_lag)
