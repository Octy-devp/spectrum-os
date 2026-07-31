"""synth.residual_trigger — Spectrum-failure residual diagnosis & Mode-B switching.

PLAN-23 v2.9.5–7 (版本 B：歷史的意外). Two-phase state switching:

- Subcritical (Mode A / reflect): the spectrum can explain — the gate reflects
  the situation through ``alt_gate``.
- Critical (Mode B / enumerate): the spectrum fails — the gate enumerates
  accidents through ``alt_gate_enumerate``. Residual diagnosis is the
  phase-state thermometer.

Residual criteria (v2.9.7, Round-3 fix):
  ① ``saturation``        — L1 field strength saturates (>= 0.9), no resolution.
  ② ``correlation_flip``  — arc correlation turns positive (r > 0), gravity
                            collapse.
  ③ ``decoupling``        — zg/ha decouple (pct_divergence > 40, ha_pct high
                            vs zg_pct low).
  ④ ``gap``               — character gap hit (institutional vacuum / affective
                            lag). **Amplifier only**: never fires by itself —
                            rides on a sharp criterion ①–③ in the same entry.
  ⑤ language sensitivity  — the same situation under n language perturbations
                            yields high output divergence. LLM-backed, so it is
                            a pluggable callback; default None = disabled.

This module is deliberately pure — numpy-free, no LLM dependency (⑤ is supplied
by the caller as a callback).
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Mapping

# Criteria ①–③ are table-driven SHARP triggers; ④ ``gap`` is table-driven but
# amplifier-only (never ignites by itself); ⑤ is callback-driven.
SHARP_CRITERIA = ("saturation", "correlation_flip", "decoupling")
GAP_CRITERION = "gap"
CRITERIA_KEYS = SHARP_CRITERIA + (GAP_CRITERION,)


def _entry_hits(entry: Any) -> bool:
    """True when a residual-table entry carries a SHARP trigger criterion.

    Safe degradation: non-dict entry, missing/empty/non-list ``criteria`` or an
    all-unknown criteria list all mean "no hit" — never crashes.

    ``gap`` is deliberately excluded from firing. Gap coverage is character-arc
    granularity and saturates the whole timeline; letting a pure-``gap`` entry
    fire would degenerate Mode B from "spectrum failure" into "the entire
    timeline", destroying the two-phase meaning. ``gap`` only rides along when
    a sharp criterion ①–③ sits in the same entry (it amplifies, never ignites).
    """
    if not isinstance(entry, dict):
        return False
    criteria = entry.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        return False
    return any(c in SHARP_CRITERIA for c in criteria)


def load_residual_table(source: str | os.PathLike | Mapping | None) -> dict:
    """Load a residual table from a dict or a JSON file path.

    Accepted shapes:
    - ``{key: {"criteria": [...]}}`` (bare table)
    - ``{"_meta": {...}, "residuals": {key: {...}}}`` (generator output —
      ``_meta`` is stripped, ``residuals`` unwrapped)

    Missing / broken sources degrade to an empty table (never raises for IO or
    parse errors) so callers can run safely without data.
    """
    if source is None:
        return {}
    if isinstance(source, Mapping):
        data = dict(source)
    elif isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError, TypeError):
            # IO / parse / fspath surprises all degrade to an empty table.
            return {}
    else:
        # Neither a mapping nor a path (e.g. an int) — never raises, degrades.
        return {}
    if not isinstance(data, dict):
        return {}
    if "residuals" in data and isinstance(data["residuals"], dict):
        return dict(data["residuals"])
    if "_meta" in data:
        return {k: v for k, v in data.items() if k != "_meta"}
    return data


def should_enumerate(
    residual_table: Mapping | None,
    t: Any,
    *,
    sensitivity_cb: Callable | None = None,
    sensitivity_threshold: float = 0.5,
    context: Any = None,
) -> bool:
    """Decide whether the gate point at ``t`` should switch to Mode B.

    Args:
        residual_table: Mapping of ``{date|t: {"criteria": [...]}}`` or None.
            The table may mix day keys (``"YYYY-MM-DD"``, saturation /
            decoupling) with month keys (``"YYYY-MM"``, correlation_flip /
            gap); a day-key miss falls back to the month prefix ``t[:7]``.
        t: Lookup key — an ensemble step (int) or a date string
            (``"YYYY-MM-DD"`` / ``"YYYY-MM"``).
        sensitivity_cb: Optional callable (⑤). Invoked as
            ``sensitivity_cb(t, context=context)``; a returned score at/above
            ``sensitivity_threshold`` also triggers Mode B. Default None
            (disabled — no LLM calls).
        sensitivity_threshold: Threshold for ``sensitivity_cb`` (default 0.5).
        context: Opaque context passed through to ``sensitivity_cb``.

    Returns:
        True when the residual table hits any SHARP criterion ①–③ at ``t``
        (day key, month-prefix fallback, or int round-trip), or when
        ``sensitivity_cb`` reports a score at/above the threshold. ``gap`` (④)
        is amplifier-only: a pure-``gap`` entry (e.g. ``criteria == ["gap"]``)
        never fires by itself. False when there is no table, no sharp hit, or
        no (or below-threshold) callback.
    """
    if residual_table is not None:
        entry = residual_table.get(t)
        if _entry_hits(entry):
            return True
        # JSON round-trip: integer keys serialise to strings — be lenient so
        # tables loaded from files still hit int ``t`` lookups.
        if not isinstance(t, str):
            entry = residual_table.get(str(t))
            if _entry_hits(entry):
                return True
        # Key-granularity fallback: the table mixes day keys (saturation /
        # decoupling) and month keys (correlation_flip / gap). A day-key miss
        # narrows to the month prefix "YYYY-MM-DD" -> "YYYY-MM" so month-key
        # entries still hit; there is intentionally no month -> day reverse.
        if isinstance(t, str) and len(t) == 10 and t[4] == "-" and t[7] == "-":
            entry = residual_table.get(t[:7])
            if _entry_hits(entry):
                return True
    if sensitivity_cb is not None:
        try:
            score = sensitivity_cb(t, context=context)
        except Exception:
            # ⑤ is a pluggable probe — a failing probe must not crash the run.
            score = 0.0
        if isinstance(score, (int, float)) and score >= sensitivity_threshold:
            return True
    return False


def make_residual_trigger(
    residual_table: Mapping | None = None,
    *,
    sensitivity_cb: Callable | None = None,
    sensitivity_threshold: float = 0.5,
    key_fn: Callable | None = None,
) -> Callable[[Any, Any], bool]:
    """Build a ``(t, context) -> bool`` trigger for ``run_ensemble``.

    ``key_fn`` (optional) maps ``(t, context) -> lookup key`` — use it when the
    table is date-keyed and the context carries the date, e.g.::

        key_fn=lambda t, ctx: ctx["situation"].get("date")

    Default: identity on ``t`` (int-step-keyed tables).

    ⚠️ A date-keyed table WITHOUT a ``key_fn`` is silently dead: the identity
    lookup on int steps never matches ``"YYYY-MM-DD"`` keys, so Mode B stays
    off without any error. Always supply ``key_fn`` for date-keyed tables.
    """
    table = load_residual_table(residual_table)

    def trigger(t: Any, context: Any = None) -> bool:
        key = key_fn(t, context) if key_fn is not None else t
        return should_enumerate(
            table,
            key,
            sensitivity_cb=sensitivity_cb,
            sensitivity_threshold=sensitivity_threshold,
            context=context,
        )

    return trigger
