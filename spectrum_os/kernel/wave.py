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

def correlate(a: np.ndarray, b: np.ndarray, max_lag: int = 24,
              synthetic: bool = False) -> dict:
    """Lagged cross-correlation between two time series.

    Detects the lag that maximises Pearson-like correlation within
    ``[-max_lag, +max_lag]``.  When the data contains a strong periodic
    component (dominant period *p*), the reported lag is aliased — lag *L*
    and *L ± p* are equivalent.  The return dict includes ``equiv_lags``
    listing all equivalent lags within the search window.

    Parameters
    ----------
    synthetic
        True when either input series originates from the synth gate layer
        (PLAN-23 §7.3).  The result is stamped ``synthetic_input: True`` so
        downstream consumers know the correlation rests on synthetic data.

    Returns
    -------
    dict with keys:
      ``r``         — max correlation (Pearson-like, [-1, +1])
      ``lag``       — lag at max correlation (positive = *b* lags behind *a*)
      ``equiv_lags``— list of (lag, r) for all equivalent peaks
      ``synthetic_input`` — True when any input is synthetic
    """
    x = _as_float_array(a)
    y = _as_float_array(b)
    n = min(len(x), len(y))

    if n < 2:
        return {"r": 0.0, "lag": 0, "equiv_lags": [(0, 0.0)],
                "synthetic_input": bool(synthetic)}

    x = x[:n]
    y = y[:n]

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

    # ── detect aliasing: find dominant periods in both series ──────────
    dominant = dominant_periods(x, max_lag=max_lag // 2) + \
               dominant_periods(y, max_lag=max_lag // 2)
    strong_periods = sorted(set(p for p in dominant if p > 1))

    equiv = [(best_lag, float(best_r))]
    if strong_periods:
        for period in strong_periods:
            for k in range(-max_lag // period - 1, max_lag // period + 2):
                candidate = best_lag + k * period
                if abs(candidate) <= max_lag and candidate != best_lag:
                    # Recompute r at this lag
                    lag = candidate
                    if lag < 0:
                        seg_a = x_c[-lag:]
                        seg_b = y_c[:lag]
                    elif lag == 0:
                        seg_a = x_c
                        seg_b = y_c
                    else:
                        seg_a = x_c[:-lag]
                        seg_b = y_c[lag:]
                    r = np.sum(seg_a * seg_b) / \
                        (np.sqrt(np.sum(seg_a ** 2) * np.sum(seg_b ** 2)) + 1e-15)
                    equiv.append((lag, float(r)))
        # Deduplicate by lag
        seen = set()
        unique_equiv = []
        for lag, val in equiv:
            if lag not in seen:
                seen.add(lag)
                unique_equiv.append((lag, val))
        equiv = sorted(unique_equiv, key=lambda x: -abs(x[1]))
    else:
        equiv = [(best_lag, float(best_r))]

    return {"r": float(best_r), "lag": best_lag, "equiv_lags": equiv,
            "synthetic_input": bool(synthetic)}


# ---------------------------------------------------------------------------
# Decompose — top-level entry point
# ---------------------------------------------------------------------------

def decompose(timeseries: np.ndarray, targets: np.ndarray | None = None,
              long_window: int = 36, mid_window: int = 6,
              synthetic: bool = False) -> dict:
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
    synthetic
        True when the series originates from the synth gate layer
        (PLAN-23 §7.3 guardrail).  The verdict is then capped at
        CONTESTED and the result is stamped ``synthetic_input: True``.

    Returns
    -------
    dict with keys ``longwave``, ``midwave``, ``shortwave``, ``deviation``,
    ``valid_range``, ``verdict``, ``dominant_periods``, ``confidence_reason``,
    ``synthetic_input``.
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

    # ---- synthetic verdict ceiling (PLAN-23 §7.3) --------------------------
    # Synthetic input may never produce an ASSERTED verdict: the anchors come
    # from an LLM gate, so confidence is capped at CONTESTED regardless of
    # how clean the expanded series looks.
    if synthetic and verdict == Verdict.ASSERTED:
        verdict = Verdict.CONTESTED
        reason += " | verdict capped at CONTESTED: synthetic input"

    return {
        "longwave": longwave,
        "midwave": midwave,
        "shortwave": shortwave,
        "deviation": deviation,
        "valid_range": valid_range,
        "verdict": verdict,
        "dominant_periods": periods,
        "confidence_reason": reason,
        "synthetic_input": bool(synthetic),
    }


# ---------------------------------------------------------------------------
# Extrapolation — forward projection with confidence bands
# ---------------------------------------------------------------------------

def extrapolate(timeseries: np.ndarray, targets: np.ndarray | None = None,
                horizon: int = 12, long_window: int = 36,
                n_simulations: int = 200, seed: int | None = None) -> dict:
    """Extrapolate a time series forward by *horizon* steps.

    Method: decompose into longwave trend + dominant-period sinusoids,
    extend the fit forward, and resample residuals for confidence bands.

    Parameters
    ----------
    timeseries
        Historical data (length N).
    targets
        Optional plan/target values (same length).  If provided, the
        deviation from targets is used to compute residuals.
    horizon
        Number of steps to project forward.
    long_window
        Window for the moving-average trend (passed to :func:`moving_average`).
    n_simulations
        Number of Monte Carlo residual resamples for confidence bands.
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:

    * ``forecast`` — ``np.ndarray`` of length *horizon*: point forecast
    * ``lower`` / ``upper`` — 80% confidence band (P10 / P90)
    * ``ci_lower`` / ``ci_upper`` — 95% confidence band (P2.5 / P97.5)
    * ``trend_slope`` — slope of the longwave trend (per step)
    * ``dominant_periods`` — periods used for sinusoidal extension
    * ``residual_std`` — std of in-sample residuals
    * ``verdict`` — ternary confidence
    * ``confidence_reason`` — human-readable explanation
    """
    arr = _as_float_array(timeseries)
    n = len(arr)
    if n < max(horizon, 6):
        return {
            "forecast": np.full(horizon, np.nan),
            "lower": np.full(horizon, np.nan),
            "upper": np.full(horizon, np.nan),
            "ci_lower": np.full(horizon, np.nan),
            "ci_upper": np.full(horizon, np.nan),
            "trend_slope": 0.0,
            "dominant_periods": [],
            "residual_std": 0.0,
            "verdict": Verdict.UNKNOWN,
            "confidence_reason": f"Data length {n} < horizon {horizon}, cannot extrapolate",
        }

    # --- Step 1: Longwave trend ---
    longwave = moving_average(arr, long_window)
    # Fill NaN edges with nearest valid value
    first_valid = 0
    while first_valid < n and np.isnan(longwave[first_valid]):
        first_valid += 1
    last_valid = n - 1
    while last_valid >= 0 and np.isnan(longwave[last_valid]):
        last_valid -= 1

    if first_valid >= last_valid:
        # Not enough valid longwave points — fall back to linear trend
        x_all = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x_all, arr, 1)
        trend_slope = coeffs[0]
    else:
        filled = longwave.copy()
        filled[:first_valid] = filled[first_valid]
        filled[last_valid + 1:] = filled[last_valid]
        # Linear fit on filled longwave
        x_all = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x_all, filled, 1)
        trend_slope = coeffs[0]

    # --- Step 2: Dominant periods → sinusoidal fit ---
    periods = dominant_periods(arr, max_lag=min(36, n // 2))

    # Build sinusoidal components from detected periods
    # Fit: arr ≈ trend(t) + Σ A_p sin(2π t/p + φ_p) + residuals
    x_all = np.arange(n, dtype=np.float64)
    trend_vals = np.polyval(coeffs, x_all)

    if periods:
        # Design matrix: trend + sinusoids
        n_cols = 1 + 2 * len(periods)  # intercept + slopes + A_sin + A_cos per period
        X = np.column_stack([np.ones(n)])
        for p in periods:
            omega = 2 * np.pi / p
            X = np.column_stack([X, np.sin(omega * x_all), np.cos(omega * x_all)])

        # OLS fit
        try:
            beta, _, _, _ = np.linalg.lstsq(X, arr, rcond=None)
            fitted = X @ beta
        except np.linalg.LinAlgError:
            fitted = trend_vals
            periods_used = []
        else:
            periods_used = periods
    else:
        fitted = trend_vals
        periods_used = []

    # --- Step 3: Residuals ---
    residuals = arr - fitted
    valid_residuals = residuals[~np.isnan(residuals)]
    residual_std = float(np.std(valid_residuals)) if len(valid_residuals) > 1 else 0.0

    # --- Step 4: Forward extrapolation ---
    x_future = np.arange(n, n + horizon, dtype=np.float64)

    # Extend trend
    trend_future = np.polyval(coeffs, x_future)

    # Extend sinusoids
    if periods_used:
        X_future = np.column_stack([np.ones(horizon)])
        for p in periods_used:
            omega = 2 * np.pi / p
            X_future = np.column_stack([X_future, np.sin(omega * x_future), np.cos(omega * x_future)])
        point_forecast = X_future @ beta[:X_future.shape[1]]
    else:
        point_forecast = trend_future

    # --- Step 5: Monte Carlo residual resampling ---
    rng = np.random.default_rng(seed)
    ensemble = np.zeros((n_simulations, horizon))
    for sim in range(n_simulations):
        sampled = rng.choice(valid_residuals, size=horizon, replace=True)
        # Dampen residuals over time (uncertainty grows → but residuals should decay toward zero for stability)
        decay = np.linspace(1.0, 0.5, horizon)
        ensemble[sim] = point_forecast + sampled * decay

    # Confidence bands
    lower = np.percentile(ensemble, 10, axis=0)
    upper = np.percentile(ensemble, 90, axis=0)
    ci_lower = np.percentile(ensemble, 2.5, axis=0)
    ci_upper = np.percentile(ensemble, 97.5, axis=0)

    # --- Step 6: Verdict ---
    if n >= _ASSERTED_MIN_LEN and residual_std > 0:
        # Check R² of fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((arr - np.mean(arr)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if r_squared >= 0.3 and len(periods_used) > 0:
            verdict = Verdict.ASSERTED
            reason = f"R²={r_squared:.3f}, {len(periods_used)} periods fitted, residual_std={residual_std:.4f}"
        elif r_squared >= 0.1:
            verdict = Verdict.CONTESTED
            reason = f"R²={r_squared:.3f} (weak fit), residual_std={residual_std:.4f}"
        else:
            verdict = Verdict.UNKNOWN
            reason = f"R²={r_squared:.3f} (poor fit), extrapolation unreliable"
    elif n >= _CONTESTED_MIN_LEN:
        verdict = Verdict.CONTESTED
        reason = f"Short data ({n} pts), trend-only extrapolation"
    else:
        verdict = Verdict.UNKNOWN
        reason = f"Insufficient data ({n} pts)"

    # Synthetic ceiling
    # (Caller should check meta.synthetic and downgrade if needed)

    return {
        "forecast": point_forecast,
        "lower": lower,
        "upper": upper,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "trend_slope": float(trend_slope),
        "dominant_periods": periods_used,
        "residual_std": residual_std,
        "verdict": verdict,
        "confidence_reason": reason,
    }
