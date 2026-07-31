#!/usr/bin/env python3
"""Build a residual table from spectrum data files (N4 / S7-5).

Converts spectrum engine outputs into the residual table consumed by
``spectrum_os.synth.residual_trigger.should_enumerate`` /
``make_residual_trigger``. spectrum-os is a general engine — ALL input paths
are injected via CLI, never hardcoded to ECC paths.

Inputs (each optional; a missing file skips its criterion):
  --anchor-field       anchor-field.json
                       {"_meta", "daily_field": [{"date", "field_strength"}],
                        "arc_correlations": {"arc": {"pearson_r", ...}}}
  --narrative-spectrum narrative-spectrum.json
                       {"_meta", "entries": {"file": {"date",
                        "coordinates": {"zg_pct", "ha_pct",
                                        "pct_divergence"}}}}
  --gap-spectrum       gap-spectrum-signatures.json
                       {"_meta", "results": {"char": {
                        "classification": {"primary_type"},
                        "gap_info": {"arcs", "has_gap"}}}}
  --output             residual-table.json (default: residual-table.json)

Criteria (PLAN-23 v2.9.7):
  ① saturation        daily date key; field_strength >= --saturation-threshold
  ② correlation_flip  monthly keys ("YYYY-MM") covering arcs with pearson_r
                       > --correlation-threshold
  ③ decoupling        daily date key; pct_divergence > --divergence-threshold
                       AND ha_pct >= --ha-pct-min AND zg_pct <= --zg-pct-max
  ④ gap               monthly keys covering arcs of characters whose
                       primary_type is 制度真空/情感滯後 (arc->month range
                       mapping, best-effort — see ARC_MONTH_RANGES)

Amplifier semantics (Round-3 fix): ④ ``gap`` is OUTPUT to ``detail`` for the
prompt-layer amplifier wiring, but it never fires by itself —
``should_enumerate`` ignores a pure-``gap`` entry (sharp triggers are only
① saturation / ② correlation_flip / ③ decoupling). A pure-gap month in this
table therefore does not switch Mode B; it only enriches the enumerate prompt
when a sharp signal in the same entry fires.

Output:
  {"_meta": {...}, "residuals": {"<date|month>": {
     "criteria": ["saturation", ...],
     "detail": {"saturation": {"field_strength": 0.9}, ...}}}}
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

# Arc name -> (start_month, end_month) inclusive. Gap data uses "德國篇" as an
# alias of "德意志篇"; unknown arcs are skipped with a warning.
ARC_MONTH_RANGES: dict[str, tuple[str, str]] = {
    "香港成長篇": ("1900-01", "1908-12"),
    "德意志篇": ("1908-01", "1909-12"),
    "德國篇": ("1908-01", "1909-12"),
    "沙俄篇": ("1909-01", "1913-12"),
    "過度篇": ("1913-06", "1914-01"),
    "重返歐洲篇": ("1914-01", "1920-12"),
}

GAP_PRIMARY_TYPES = {"制度真空", "情感滯後"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor-field", default=None, help="anchor-field.json path")
    p.add_argument("--narrative-spectrum", default=None, help="narrative-spectrum JSON path")
    p.add_argument("--gap-spectrum", default=None, help="gap-spectrum-signatures JSON path")
    p.add_argument("--output", default="residual-table.json", help="output JSON path")
    p.add_argument("--saturation-threshold", type=float, default=0.9, help="① L1 field strength threshold")
    p.add_argument("--correlation-threshold", type=float, default=0.0, help="② arc pearson_r threshold (flip when above)")
    p.add_argument("--divergence-threshold", type=float, default=40.0, help="③ pct_divergence threshold")
    p.add_argument("--ha-pct-min", type=float, default=90.0, help="③ ha_pct lower bound (high)")
    p.add_argument("--zg-pct-max", type=float, default=40.0, help="③ zg_pct upper bound (low)")
    return p.parse_args()


def load_json(path: str | None) -> dict:
    """Load a JSON file; return {} on missing/broken input (safe degradation)."""
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        print(f"[warn] cannot load {path}: {e}", file=sys.stderr)
        return {}
    return data if isinstance(data, dict) else {}


def parse_date_key(s: object) -> str | None:
    """Parse 'YYYY-MM-DD' -> normalized key; None for 'YYYY-??-??' / garbage."""
    if not isinstance(s, str):
        return None
    parts = s.split("-")
    if len(parts) != 3:
        return None
    y, m, d = parts
    if not (y.isdigit() and m.isdigit() and d.isdigit()):
        return None
    return f"{y}-{m}-{d}"


def months_between(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' keys between two month keys."""
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out: list[str] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def add_hit(table: dict, key: str, criterion: str, detail: dict) -> None:
    """Merge a criterion hit into the table entry for ``key``.

    ``detail[criterion]`` accumulates every hit as a list (multiple characters
    or scenes can hit the same month/date — nothing is dropped).
    """
    entry = table.setdefault(key, {"criteria": [], "detail": {}})
    if criterion not in entry["criteria"]:
        entry["criteria"].append(criterion)
    entry["detail"].setdefault(criterion, []).append(detail)


def build_saturation(anchor: dict, threshold: float, table: dict) -> dict:
    """① L1 field-strength saturation -> daily date keys."""
    stats = {"n_days": 0, "min_strength": None, "max_strength": None}
    for e in anchor.get("daily_field", []):
        if not isinstance(e, dict):
            continue
        fs = e.get("field_strength")
        if not isinstance(fs, (int, float)):
            continue
        fs = float(fs)
        if stats["min_strength"] is None or fs < stats["min_strength"]:
            stats["min_strength"] = fs
        if stats["max_strength"] is None or fs > stats["max_strength"]:
            stats["max_strength"] = fs
        if fs >= threshold:
            key = parse_date_key(e.get("date"))
            if key:
                add_hit(table, key, "saturation", {"field_strength": round(fs, 4)})
                stats["n_days"] += 1
    return stats


def build_correlation_flip(anchor: dict, threshold: float, table: dict, warnings: list) -> dict:
    """② arc correlation turns positive -> monthly keys over the arc's range."""
    stats = {"n_arcs": 0, "unknown_arcs": []}
    for arc, info in (anchor.get("arc_correlations") or {}).items():
        r = info.get("pearson_r") if isinstance(info, dict) else None
        if not isinstance(r, (int, float)):
            continue
        r = float(r)
        if r > threshold:
            rng = ARC_MONTH_RANGES.get(arc)
            if rng:
                for m in months_between(*rng):
                    add_hit(table, m, "correlation_flip", {"arc": arc, "pearson_r": round(r, 4)})
                stats["n_arcs"] += 1
            else:
                stats["unknown_arcs"].append(arc)
    if stats["unknown_arcs"]:
        warnings.append(f"correlation_flip: unknown arcs skipped: {stats['unknown_arcs']}")
    return stats


def build_decoupling(
    narrative: dict, div_thr: float, ha_min: float, zg_max: float, table: dict
) -> dict:
    """③ zg/ha decoupling -> daily date keys of divergent scenes."""
    stats = {"n_scenes": 0, "divergent": 0, "hits": 0}
    for ent in (narrative.get("entries") or {}).values():
        if not isinstance(ent, dict):
            continue
        coords = ent.get("coordinates")
        if not isinstance(coords, dict):
            continue
        pd = coords.get("pct_divergence")
        hap = coords.get("ha_pct")
        zgp = coords.get("zg_pct")
        if not all(isinstance(v, (int, float)) for v in (pd, hap, zgp)):
            continue
        pd, hap, zgp = float(pd), float(hap), float(zgp)
        stats["n_scenes"] += 1
        if pd > div_thr and hap >= ha_min and zgp <= zg_max:
            stats["divergent"] += 1
            key = parse_date_key(ent.get("date"))
            if key:
                add_hit(
                    table,
                    key,
                    "decoupling",
                    {
                        "pct_divergence": pd,
                        "ha_pct": hap,
                        "zg_pct": zgp,
                        "scene_id": ent.get("scene_id", ""),
                    },
                )
                stats["hits"] += 1
    return stats


def build_gap(gap_spectrum: dict, table: dict, warnings: list) -> dict:
    """④ character gap hit -> monthly keys over the character's arcs.

    Best-effort: gap data is per-character; without per-scene dates the arcs'
    month ranges are the finest available granularity.
    """
    stats: dict = {"n_characters": 0, "gap_characters": 0, "unknown_arcs": []}
    unknown_arcs: set = set()
    for cid, c in (gap_spectrum.get("results") or {}).items():
        if not isinstance(c, dict):
            continue
        cls = c.get("classification")
        gi = c.get("gap_info")
        if not isinstance(cls, dict) or not isinstance(gi, dict):
            continue
        primary_type = cls.get("primary_type")
        if primary_type not in GAP_PRIMARY_TYPES or not gi.get("has_gap", False):
            continue
        stats["n_characters"] += 1
        stats["gap_characters"] += 1
        for arc in gi.get("arcs", []):
            rng = ARC_MONTH_RANGES.get(arc)
            if rng:
                for m in months_between(*rng):
                    add_hit(
                        table,
                        m,
                        "gap",
                        {"character": cid, "primary_type": primary_type, "arc": arc},
                    )
            else:
                unknown_arcs.add(arc)
    stats["unknown_arcs"] = sorted(unknown_arcs)
    if unknown_arcs:
        warnings.append(f"gap: unknown arcs skipped: {sorted(unknown_arcs)}")
    return stats


def main() -> int:
    args = parse_args()

    anchor = load_json(args.anchor_field)
    narrative = load_json(args.narrative_spectrum)
    gap_spectrum = load_json(args.gap_spectrum)

    table: dict = {}
    warnings: list[str] = []
    criteria_stats: dict = {}

    if args.anchor_field:
        criteria_stats["saturation"] = build_saturation(
            anchor, args.saturation_threshold, table
        )
        criteria_stats["correlation_flip"] = build_correlation_flip(
            anchor, args.correlation_threshold, table, warnings
        )
    if args.narrative_spectrum:
        criteria_stats["decoupling"] = build_decoupling(
            narrative, args.divergence_threshold, args.ha_pct_min, args.zg_pct_max, table
        )
    if args.gap_spectrum:
        criteria_stats["gap"] = build_gap(gap_spectrum, table, warnings)

    if not table:
        warnings.append("empty residual table — no input files produced any hit")

    out = {
        "_meta": {
            "generator": "spectrum_os/scripts/build_residual_table.py",
            "date": datetime.now(timezone.utc).isoformat(),
            "thresholds": {
                "saturation": args.saturation_threshold,
                "correlation": args.correlation_threshold,
                "divergence": args.divergence_threshold,
                "ha_pct_min": args.ha_pct_min,
                "zg_pct_max": args.zg_pct_max,
            },
            "criteria_stats": criteria_stats,
            "warnings": warnings,
        },
        "residuals": table,
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    n_keys = len(table)
    n_hits = sum(len(e["criteria"]) for e in table.values())
    print(f"residual table -> {args.output}")
    print(f"  {n_keys} keys, {n_hits} criterion hits")
    for crit, st in criteria_stats.items():
        print(f"  {crit}: {st}")
    for w in warnings:
        print(f"  [warn] {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
