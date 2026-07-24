"""
Spectrum OS Kernel — Wave decomposition engine.

Decomposes β₁ economic time series into long/mid/short waves and deviation
spectra.  numpy-only — no scipy, no sklearn.
"""

import numpy as np
from ._types import Verdict

# ---------------------------------------------------------------------------
# Verdict thresholds (adjustable)
# ---------------------------------------------------------------------------
_ASSERTED_MIN_LEN = 24
_ASSERTED_MIN_PEAK = 0.4
_CONTESTED_MIN_LEN = 12
_CONTESTED_MIN_PEAK = 0.2
_KNOWN_MAX_NAN = 2


def _as_float_array(data):
    """Convert input to float64 ndarray, handling all array-likes."""
    return np.asarray(data, dtype=np.float64)


# ---------------------------------------------------------------------------
# Moving average
# ---------------------------------------------------------------------------

def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Moving average with 'same' output length.

    Leading/trailing edges are set to NaN where the window extends beyond
    the data.  Uses prefix-sum for O(n) computation.
    """
    arr = _as_float_array(data)
    if window <= 0 or len(arr) == 0:
        return np.full_like(arr, np.nan)
    if window == 1:
        return arr.copy()
    n = len(arr)
    if window > n:
        return np.full_like(arr, np.nan)

    half = window // 2
    result = np.full(n, np.nan, dtype=np.float64)

    # Prefix-sum for O(n) computation
    cum = np.empty(n + 1, dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(arr, out=cum[1:])

    if window % 2 == 1:
        start = half
        end = n - half
        left = np.arange(start, end) - half
        right = np.arange(start, end) + half + 1
    else:
        start = half
        end = n - half + 1
        left = np.arange(start, end) - half
        right = np.arange(start, end) + half

    if start < end:
        sums = cum[right] - cum[left]
        result[start:end] = sums / window

    return result


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def _find_peaks(autocorr: np.ndarray, threshold: float = 0.3) -> list[int]:
    """Manual peak detection with non-maximum suppression for plateaus.

    A point *i* is a strict peak if ``autocorr[i] > autocorr[i-1]`` AND
    ``autocorr[i] > autocorr[i+1]``.

    For plateaus (consecutive equal values) only the midpoint of the
    plateau is emitted if the plateau value exceeds both neighbours.
    Only peaks above *threshold* are returned.
    """
    arr = _as_float_array(autocorr)
    if len(arr) < 3:
        return []

    peaks: list[int] = []
    i = 1
    while i < len(arr) - 1:
        # ---- detect plateau ------------------------------------------------
        if arr[i] == arr[i + 1]:
            j = i + 1
            while j < len(arr) - 1 and arr[j] == arr[j + 1]:
                j += 1
            # arr[i] … arr[j] is a plateau
            # If plateau reaches the last element, it has no right neighbor → not a peak
            if j < len(arr) - 1 and arr[i] > arr[i - 1] and arr[j] > arr[j + 1]:
                midpoint = (i + j) // 2
                if arr[midpoint] > threshold:
                    peaks.append(midpoint)
            i = j + 1
        # ---- strict peak ---------------------------------------------------
        elif arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            if arr[i] > threshold:
                peaks.append(i)
            i += 1
        else:
            i += 1

    return peaks


# ---------------------------------------------------------------------------
# Dominant periods via autocorrelation
# ---------------------------------------------------------------------------

def dominant_periods(timeseries: np.ndarray, max_lag: int = 36,
                     threshold: float = 0.3) -> list[int]:
    """Detect dominant periods via normalised autocorrelation.

    Autocorrelation at lag *k* is computed as

        r(k) = Σ_t (x[t] - μ)(x[t+k] - μ)  /  (n · Var(x))

    where the sum runs over the *n − k* overlapping pairs.  Peak detection
    is manual via :func:`_find_peaks`.

    Default ``max_lag=36`` ensures periods up to 36 months can be detected
    (the autocorrelation peak is at lag=period, and the array index must
    have both left and right neighbours for peak detection).
    """
    arr = _as_float_array(timeseries)
    n = len(arr)
    if n < 3:
        return []

    # leave at least 2 elements to correlate
    max_lag = min(max_lag, n - 2)
    if max_lag < 1:
        return []

    centered = arr - np.mean(arr)
    variance = np.sum(centered ** 2) / n
    if variance == 0.0:
        return []

    autocorr = np.empty(max_lag + 1, dtype=np.float64)
    for k in range(max_lag + 1):
        autocorr[k] = np.sum(centered[:n - k] * centered[k:]) / (n * variance)

    return _find_peaks(autocorr, threshold)


def _max_autocorr_peak(timeseries: np.ndarray, max_lag: int = 36) -> float:
    """Maximum autocorrelation value (excluding lag 0).

    Default ``max_lag=36`` matches :func:`dominant_periods`.
    """
    arr = _as_float_array(timeseries)
    n = len(arr)
    if n < 2:
        return 0.0

    max_lag = min(max_lag, n - 1)
    if max_lag < 1:
        return 0.0

    centered = arr - np.mean(arr)
    variance = np.sum(centered ** 2) / n
    if variance == 0.0:
        return 0.0

    max_val = 0.0
    for k in range(1, max_lag + 1):
        r = np.sum(centered[:n - k] * centered[k:]) / (n * variance)
        if r > max_val:
            max_val = r

    return max_val


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def correlate(a: np.ndarray, b: np.ndarray, max_lag: int = 24) -> dict:
    """Lagged cross-correlation between two time series.

    Returns
    -------
    dict with keys ``r`` (max correlation, Pearson-like) and ``lag_months``.
    Only lags in ``[-max_lag, +max_lag]`` are considered.
    """
    x = _as_float_array(a)
    y = _as_float_array(b)
    n = min(len(x), len(y))

    if n < 2:
        return {"r": 0.0, "lag_months": 0}

    # Trim to equal length
    x = x[:n]
    y = y[:n]

    # Mean-centre
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)

    max_lag = min(max_lag, n - 1)

    best_r = -np.inf
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            seg_a = x_c[-lag:]
            seg_b = y_c[:lag]
        elif lag == 0:
            seg_a = x_c
            seg_b = y_c
        else:
            seg_a = x_c[:-lag]
            seg_b = y_c[lag:]

        r_num = np.sum(seg_a * seg_b)
        r_den = np.sqrt(np.sum(seg_a ** 2) * np.sum(seg_b ** 2))

        if r_den == 0.0:
            continue

        r = r_num / r_den

        if r > best_r:
            best_r = r
            best_lag = lag

    return {"r": float(best_r), "lag_months": best_lag}


# ---------------------------------------------------------------------------
# Decompose — top-level entry point
# ---------------------------------------------------------------------------

def decompose(timeseries: np.ndarray, targets: np.ndarray | None = None,
              long_window: int = 36, mid_window: int = 6) -> dict:
    """Decompose a time series into long/mid/short waves and deviation.

    Parameters
    ----------
    timeseries
        Actual observed values.
    targets
        Planned values (optional).  If provided the deviation spectrum
        ``(actual - plan) / (|plan| + 1e-8)`` is computed.
    long_window
        Moving-average window for the long wave (months).  Default 36.
    mid_window
        Moving-average window for the mid wave (months).  Default 6.

    Returns
    -------
    dict with keys ``longwave``, ``midwave``, ``shortwave``, ``deviation``,
    ``valid_range``, ``verdict``, ``dominant_periods``, ``confidence_reason``.
    """
    arr = _as_float_array(timeseries)
    n = len(arr)

    # ---- wave decomposition -----------------------------------------------
    longwave = moving_average(arr, long_window)
    mid_avg = moving_average(arr, mid_window)
    midwave = mid_avg - longwave
    shortwave = arr - longwave - midwave

    # Propagate NaN from longwave into mid/short
    nan_mask = np.isnan(longwave)
    midwave = np.where(nan_mask, np.nan, midwave)
    shortwave = np.where(nan_mask, np.nan, shortwave)

    # ---- deviation --------------------------------------------------------
    if targets is not None:
        plan = _as_float_array(targets)
        deviation = (arr - plan) / (np.abs(plan) + 1e-8)
    else:
        deviation = None

    # ---- valid range ------------------------------------------------------
    valid_mask = ~nan_mask
    if np.any(valid_mask):
        valid_idx = np.where(valid_mask)[0]
        valid_range = (int(valid_idx[0]), int(valid_idx[-1]))
    else:
        valid_range = (0, 0)

    # ---- interior NaN count -----------------------------------------------
    clean = ~np.isnan(arr)
    if np.any(clean):
        first = int(np.where(clean)[0][0])
        last = int(np.where(clean)[0][-1])
        interior_nan = int(np.sum(np.isnan(arr[first:last + 1])))
    else:
        interior_nan = n

    effective_len = n - interior_nan

    # ---- autocorrelation / periods ----------------------------------------
    periods = dominant_periods(arr)
    max_peak = _max_autocorr_peak(arr)

    # ---- verdict ----------------------------------------------------------
    if (effective_len >= _ASSERTED_MIN_LEN
            and max_peak >= _ASSERTED_MIN_PEAK
            and interior_nan == 0):
        verdict = Verdict.ASSERTED
        reason = (
            f"Data length {n} ≥ {_ASSERTED_MIN_LEN}, "
            f"peak {max_peak:.3f} ≥ {_ASSERTED_MIN_PEAK}, "
            f"NaN mid-sequence = {interior_nan}"
        )
    elif (effective_len >= _CONTESTED_MIN_LEN
          and (_CONTESTED_MIN_PEAK <= max_peak < _ASSERTED_MIN_PEAK
               or 0 < interior_nan <= _KNOWN_MAX_NAN)):
        verdict = Verdict.CONTESTED
        reason = (
            f"Data length {n} ≥ {_CONTESTED_MIN_LEN}, "
            f"peak {max_peak:.3f} in [{_CONTESTED_MIN_PEAK}, {_ASSERTED_MIN_PEAK}) "
            f"or NaN {interior_nan} in [1, {_KNOWN_MAX_NAN}]"
        )
    else:
        verdict = Verdict.UNKNOWN
        reason = (
            f"Insufficient data or weak pattern "
            f"(len={effective_len}, peak={max_peak:.3f}, NaN={interior_nan})"
        )

    return {
        "longwave": longwave,
        "midwave": midwave,
        "shortwave": shortwave,
        "deviation": deviation,
        "valid_range": valid_range,
        "verdict": verdict,
        "dominant_periods": periods,
        "confidence_reason": reason,
    }
