"""Prediction calibration — compare predicted vs realised values.

Stores a bounded in-memory state log for audit and drift tracking.
"""

import numpy as np
from ._types import Verdict

# ---------------------------------------------------------------------------
# Threshold constants (adjustable)
# ---------------------------------------------------------------------------
_VERIFY_MAPE_THRESHOLD = 0.15       # 15 % MAPE triggers re-calibrate
_MAX_STATE_LOG_ENTRIES = 10000

# ---------------------------------------------------------------------------
# In-memory state log
# ---------------------------------------------------------------------------
_state_log: list[dict] = []


def _mape(realized: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error, avoiding division by zero."""
    denom = np.abs(realized)
    mask = denom > 1e-12
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((realized[mask] - predicted[mask]) / denom[mask])))


def _mae(realized: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(realized - predicted)))


def verify(prediction_id: str, realized: list[float],
           predicted: list[float]) -> dict:
    """Compare predicted vs realised values.

    Parameters
    ----------
    prediction_id
        Unique identifier for the prediction being verified.
    realized
        Actual observed values.
    predicted
        Forecast / predicted values.

    Returns
    -------
    dict with keys ``prediction_id``, ``mae``, ``mape``, ``verdict``,
    ``re_calibrate``, ``state_log_entry``.

    Verdict logic:

    * ``ASSERTED`` — MAPE < ``_VERIFY_MAPE_THRESHOLD / 2``
    * ``CONTESTED`` — MAPE < ``_VERIFY_MAPE_THRESHOLD``
    * ``UNKNOWN`` — MAPE >= ``_VERIFY_MAPE_THRESHOLD``

    ``re_calibrate`` is ``True`` when MAPE >= ``_VERIFY_MAPE_THRESHOLD``.
    """
    realized_arr = np.array(realized, dtype=np.float64)
    predicted_arr = np.array(predicted, dtype=np.float64)

    # Edge case: empty arrays
    if len(realized_arr) == 0 or len(predicted_arr) == 0:
        mape_val = 0.0
        mae_val = 0.0
    else:
        mape_val = _mape(realized_arr, predicted_arr)
        mae_val = _mae(realized_arr, predicted_arr)

    # Verdict
    half_threshold = _VERIFY_MAPE_THRESHOLD / 2.0
    if mape_val < half_threshold:
        verdict = Verdict.ASSERTED
    elif mape_val < _VERIFY_MAPE_THRESHOLD:
        verdict = Verdict.CONTESTED
    else:
        verdict = Verdict.UNKNOWN

    re_calibrate = mape_val >= _VERIFY_MAPE_THRESHOLD

    entry = {
        "prediction_id": prediction_id,
        "mae": mae_val,
        "mape": mape_val,
        "verdict": verdict.value,
        "re_calibrate": re_calibrate,
        "realized": realized,
        "predicted": predicted,
    }

    # Append to state log, trimming if over limit
    _state_log.append(entry)
    if len(_state_log) > _MAX_STATE_LOG_ENTRIES:
        _state_log.pop(0)

    return {
        "prediction_id": prediction_id,
        "mae": mae_val,
        "mape": mape_val,
        "verdict": verdict,
        "re_calibrate": re_calibrate,
        "state_log_entry": entry,
    }


def get_state_log(n: int = 100) -> list[dict]:
    """Return the last *n* entries from the state log."""
    if n <= 0:
        return []
    return _state_log[-n:]


def clear_state_log():
    """Clear all entries from the state log."""
    _state_log.clear()
