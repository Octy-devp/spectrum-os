"""Prediction calibration — compare predicted vs realised values.

Stores a bounded in-memory state log for audit and drift tracking.
Optionally persists every entry to a JSONL file (``init_log``) so the
log survives restarts; ``query`` / ``summary`` read that file back.
Persistence is opt-in: without ``init_log`` the module behaves exactly
as before (memory only).
"""

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from ._types import Verdict

# ---------------------------------------------------------------------------
# Threshold constants (adjustable)
# ---------------------------------------------------------------------------
_VERIFY_MAPE_THRESHOLD = 0.15       # 15 % MAPE triggers re-calibrate
_MAX_STATE_LOG_ENTRIES = 10000

# ---------------------------------------------------------------------------
# In-memory state log + optional JSONL persistence
# ---------------------------------------------------------------------------
_state_log: list[dict] = []
_log_path: Path | None = None


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

    # Guard: unequal-length arrays — truncate to common length
    if len(realized_arr) != len(predicted_arr):
        min_len = min(len(realized_arr), len(predicted_arr))
        realized_arr = realized_arr[:min_len]
        predicted_arr = predicted_arr[:min_len]

    # Edge case: empty arrays
    if len(realized_arr) == 0 or len(predicted_arr) == 0:
        mape_val = 0.0
        mae_val = 0.0
    else:
        mape_val = _mape(realized_arr, predicted_arr)
        mae_val = _mae(realized_arr, predicted_arr)

    # Verdict
    # Guard: when all realised values are zero (MAPE undefined / returns 0),
    # fall back to MAE so we don't silently label large errors as ASSERTED.
    _mape_undefined = (mape_val == 0.0 and mae_val > 0.0
                       and np.all(np.abs(realized_arr) < 1e-12))
    if _mape_undefined:
        # MAE-only verdict: scale threshold by typical magnitude ~1
        _effective_mae_threshold = 0.5  # half a unit off → CONTESTED
        if mae_val < _effective_mae_threshold:
            verdict = Verdict.ASSERTED
        elif mae_val < _effective_mae_threshold * 2:
            verdict = Verdict.CONTESTED
        else:
            verdict = Verdict.UNKNOWN
    else:
        half_threshold = _VERIFY_MAPE_THRESHOLD / 2.0
        if mape_val < half_threshold:
            verdict = Verdict.ASSERTED
        elif mape_val < _VERIFY_MAPE_THRESHOLD:
            verdict = Verdict.CONTESTED
        else:
            verdict = Verdict.UNKNOWN

    re_calibrate = (verdict == Verdict.UNKNOWN)

    ts = datetime.now(timezone.utc).isoformat()
    entry = {
        "ts": ts,
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

    # Persist to JSONL if init_log() was called
    if _log_path is not None:
        _append_jsonl(_log_path, {
            "ts": ts,
            "prediction_id": prediction_id,
            "mae": mae_val,
            "mape": mape_val,
            "verdict": verdict.value,
            "re_calibrate": re_calibrate,
        })

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


# PLAN-23 API naming: the verify op is called ``verify.evaluate`` in the
# plan; ``verify()`` above is the implementation. Alias keeps both usable.
evaluate = verify


# ---------------------------------------------------------------------------
# JSONL persistence (opt-in via init_log)
# ---------------------------------------------------------------------------

def init_log(path: str | Path | None = "data/state_log.jsonl") -> Path | None:
    """Enable JSONL persistence of the verify state log.

    After calling this, every ``verify()`` / ``evaluate()`` call appends
    one JSON line to *path* with keys ``ts`` (ISO 8601, UTC),
    ``prediction_id``, ``mae``, ``mape``, ``verdict``, ``re_calibrate``
    (in addition to the unchanged in-memory buffer). An existing file is
    kept — re-initialising after a restart continues the same log.

    If *path* is not None and persistence was not previously enabled,
    any existing in-memory ``_state_log`` entries are flushed to the
    file before switching to file-backed mode.

    Parameters
    ----------
    path
        Target JSONL file. Parent directories are created as needed.
        Pass ``None`` to disable persistence again.

    Returns
    -------
    The resolved :class:`~pathlib.Path`, or ``None`` when disabling.
    """
    global _log_path
    if path is None:
        _log_path = None
        return None
    p = Path(path)
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(exist_ok=True)
    # Flush in-memory buffer to file when transitioning from memory-only
    # to file-backed mode.
    if _log_path is None and _state_log:
        for entry in _state_log:
            # Preserve the FULL entry: gate records carry custom keys
            # (gate_type / observation / rates / confidence_score …).
            # Cherry-picking core keys here silently destroyed them
            # (S4 reporter observations were lost this way).
            full = {
                "ts": entry.get("ts", ""),
                "prediction_id": entry.get("prediction_id", ""),
                "mae": entry.get("mae", 0.0),
                "mape": entry.get("mape", 0.0),
                "verdict": entry.get("verdict", ""),
                "re_calibrate": entry.get("re_calibrate", False),
            }
            for k, v in entry.items():
                if k not in full:
                    full[k] = v
            _append_jsonl(p, full)
    _log_path = p
    return p


def _append_jsonl(path: Path, record: dict) -> None:
    """Append one JSON line to *path* (open/append/close per call)."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_log_path(path: str | Path | None) -> Path:
    """Return the explicit *path* or the one set by ``init_log``."""
    p = Path(path) if path is not None else _log_path
    if p is None:
        raise RuntimeError(
            "state log not persisted: call init_log(path) first "
            "or pass path= explicitly"
        )
    return p


def _parse_ts(value) -> datetime:
    """Parse an ISO 8601 timestamp; naive values are assumed UTC."""
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _iter_log_records(p: Path):
    """Yield parsed JSONL records, skipping blank/corrupt lines.

    Caller can access ``_skipped_lines`` post-iteration for a count
    of lines that were corrupt/non-dict and silently discarded.
    """
    _iter_log_records._skipped_lines = 0  # type: ignore[attr-defined]
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                _iter_log_records._skipped_lines += 1  # type: ignore[attr-defined]
                continue
            if not isinstance(rec, dict):
                _iter_log_records._skipped_lines += 1  # type: ignore[attr-defined]
                continue
            yield rec


def query(prediction_id: str | None = None,
          verdict=None,
          ts_from=None,
          ts_to=None,
          limit: int | None = None,
          path: str | Path | None = None) -> list[dict]:
    """Query the persisted JSONL state log.

    Parameters
    ----------
    prediction_id
        Keep only entries with this exact prediction id.
    verdict
        Keep only entries with this verdict — a :class:`Verdict` member
        or its string value (case-insensitive).
    ts_from, ts_to
        Inclusive ISO 8601 bounds on the entry ``ts`` (naive datetimes
        or date-only strings are treated as UTC).
    limit
        If given, return at most this many of the *most recent* matches.
    path
        Read this file instead of the one set by ``init_log``.

    Returns
    -------
    list of record dicts in file (chronological) order.
    """
    p = _resolve_log_path(path)
    if isinstance(verdict, Verdict):
        v_filter = verdict.value
    elif verdict is not None:
        v_filter = str(verdict).lower()
    else:
        v_filter = None
    from_dt = _parse_ts(ts_from) if ts_from is not None else None
    to_dt = _parse_ts(ts_to) if ts_to is not None else None

    matches: list[dict] = []
    for rec in _iter_log_records(p):
        if prediction_id is not None and rec.get("prediction_id") != prediction_id:
            continue
        if v_filter is not None and rec.get("verdict") != v_filter:
            continue
        if from_dt is not None or to_dt is not None:
            rec_ts_raw = rec.get("ts")
            if not rec_ts_raw:
                continue
            try:
                rec_ts = _parse_ts(rec_ts_raw)
            except ValueError:
                continue
            if from_dt is not None and rec_ts < from_dt:
                continue
            if to_dt is not None and rec_ts > to_dt:
                continue
        matches.append(rec)

    if limit is not None:
        if limit <= 0:
            return []
        matches = matches[-limit:]
    return matches


def summary(n: int = 100, path: str | Path | None = None) -> dict:
    """Aggregate statistics over the persisted JSONL state log.

    Parameters
    ----------
    n
        Number of most recent entries used for ``recent_mape_mean``.
    path
        Read this file instead of the one set by ``init_log``.

    Returns
    -------
    dict with keys ``total``, ``verdicts`` (count per verdict value),
    ``re_calibrate_rate``, ``recent_n``, ``recent_mape_mean``,
    ``skipped_lines`` (corrupt JSONL lines silently discarded).
    """
    p = _resolve_log_path(path)
    total = 0
    re_cal = 0
    verdicts: dict[str, int] = {}
    recent_mapes: deque[float] = deque(maxlen=max(n, 1))
    for rec in _iter_log_records(p):
        total += 1
        v = rec.get("verdict")
        verdicts[v] = verdicts.get(v, 0) + 1
        if rec.get("re_calibrate"):
            re_cal += 1
        m = rec.get("mape")
        if isinstance(m, (int, float)):
            recent_mapes.append(float(m))
    return {
        "total": total,
        "verdicts": verdicts,
        "re_calibrate_rate": (re_cal / total) if total else 0.0,
        "recent_n": len(recent_mapes),
        "recent_mape_mean": (sum(recent_mapes) / len(recent_mapes)) if recent_mapes else 0.0,
        "skipped_lines": getattr(_iter_log_records, "_skipped_lines", 0),
    }


# ---------------------------------------------------------------------------
# Phase 2C: query_state_log / summarize_state_log
# ---------------------------------------------------------------------------

def query_state_log(sector_id: str | None = None, n: int = 10) -> list[dict]:
    """Read the persisted JSONL state log, optionally filter by *sector_id*,
    and return the most recent *n* entries.

    Parameters
    ----------
    sector_id
        If provided, only return entries whose ``sector_id`` field matches.
    n
        Maximum number of most recent entries to return.

    Returns
    -------
    List of record dicts, or ``[]`` if the log file does not exist.
    """
    log_file = Path("data/state_log.jsonl")
    if not log_file.exists():
        return []
    entries: list[dict] = []
    for rec in _iter_log_records(log_file):
        if sector_id is not None:
            if rec.get("sector_id") != sector_id:
                continue
        entries.append(rec)
    if n <= 0:
        return []
    return entries[-n:]


def summarize_state_log(sector_id: str | None = None) -> dict:
    """Aggregate statistics over the persisted JSONL state log,
    optionally filtered by *sector_id*.

    Parameters
    ----------
    sector_id
        If provided, only consider entries whose ``sector_id`` matches.

    Returns
    -------
    dict with keys ``total_entries``, ``avg_mape``,
    ``verdict_distribution`` (``{ASSERTED: N, CONTESTED: N, UNKNOWN: N}``),
    ``calibration_trend`` (list of ``(entry_index, mape)`` for the last 20
    entries), and ``last_entry`` (most recent entry dict or ``None``).
    """
    log_file = Path("data/state_log.jsonl")
    entries: list[dict] = []
    for rec in _iter_log_records(log_file):
        if sector_id is not None:
            if rec.get("sector_id") != sector_id:
                continue
        entries.append(rec)

    if not entries:
        return {
            "total_entries": 0,
            "avg_mape": 0.0,
            "verdict_distribution": {"ASSERTED": 0, "CONTESTED": 0, "UNKNOWN": 0},
            "calibration_trend": [],
            "last_entry": None,
        }

    total = len(entries)
    mape_values = [
        float(e["mape"])
        for e in entries
        if isinstance(e.get("mape"), (int, float))
    ]
    avg_mape = sum(mape_values) / len(mape_values) if mape_values else 0.0

    verdict_dist: dict[str, int] = {"ASSERTED": 0, "CONTESTED": 0, "UNKNOWN": 0}
    for e in entries:
        v = str(e.get("verdict", "")).upper()
        if v in verdict_dist:
            verdict_dist[v] += 1

    # calibration_trend: last 20 entries as (entry_index, mape)
    recent = entries[-20:]
    calibration_trend = [
        (total - len(recent) + i, e.get("mape", 0.0))
        for i, e in enumerate(recent)
    ]

    return {
        "total_entries": total,
        "avg_mape": avg_mape,
        "verdict_distribution": verdict_dist,
        "calibration_trend": calibration_trend,
        "last_entry": entries[-1],
    }
