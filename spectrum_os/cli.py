"""tio-sh — Spectrum OS command-line interface (PLAN-23 §四-C).

Subcommands: decompose / correlate / cluster / verify.
Exit codes: 0 = ok, 1 = error, 2 = verdict UNKNOWN (scriptable).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .kernel import osc
from .kernel._types import Verdict

DEFAULT_STORE = "data/sectors/sectors.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_store(path: str):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"error: sector store not found: {path}")
    osc.sectors.load(str(p))


def _plain(obj):
    """Recursively convert numpy/Verdict values to JSON-safe types."""
    if isinstance(obj, Verdict):
        return obj.value
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _emit(payload: dict, as_json: bool, summary: str):
    if as_json:
        print(json.dumps(_plain(payload), ensure_ascii=False, indent=2))
    else:
        print(summary)


def _verdict_of(payload) -> str:
    v = getattr(payload, "verdict", None)
    if isinstance(payload, dict):
        v = payload.get("verdict", v)
    if isinstance(v, Verdict):
        return v.value
    return str(v or "unknown")


def _exit_code(verdict: str) -> int:
    return 2 if verdict == Verdict.UNKNOWN.value else 0


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def _cmd_decompose(args) -> int:
    _load_store(args.store)
    try:
        result = osc.wave.decompose(args.sector_id)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    values = result["values"] if isinstance(result, dict) else result.values
    verdict = _verdict_of(result)
    periods = values.get("dominant_periods", [])
    payload = {"sector": args.sector_id, "verdict": verdict,
               "dominant_periods": periods,
               "synthetic_input": values.get("synthetic_input", False)}
    synth_note = " [synthetic]" if payload["synthetic_input"] else ""
    _emit(payload, args.json,
          f"{args.sector_id}: verdict={verdict}{synth_note}  periods={list(periods)}")
    return _exit_code(verdict)


def _cmd_correlate(args) -> int:
    _load_store(args.store)
    try:
        result = osc.wave.correlate(args.a, args.b, max_lag=args.max_lag)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    payload = dict(result)
    payload.setdefault("pair", [args.a, args.b])
    r = result.get("r", result.get("correlation", "?"))
    lag = result.get("lag", result.get("lag_months", "?"))
    equiv = result.get("equiv_lags")
    lag_txt = f"{lag} (≡ {equiv})" if equiv else str(lag)
    _emit(payload, args.json, f"{args.a} × {args.b}: r={r}  lag={lag_txt}")
    return 0


def _cmd_cluster(args) -> int:
    _load_store(args.store)
    sectors = osc.sectors.list_sectors()
    if not sectors:
        raise SystemExit("error: no sectors registered in store")
    signatures = []
    for s in sectors:
        try:
            res = osc.wave.decompose(s.id)
        except ValueError:
            continue
        values = res["values"] if isinstance(res, dict) else res.values
        dev = np.asarray(values.get("deviation", []), dtype=np.float64)
        signatures.append({
            "sector": s.id,
            "periods": list(values.get("dominant_periods", [])),
            "amplitudes": [],
            "mean_deviation": float(np.mean(np.abs(dev))) if dev.size else 0.0,
        })
    if len(signatures) < args.n:
        raise SystemExit(
            f"error: need ≥{args.n} sectors with usable data, got {len(signatures)}")
    result = osc.cluster.run(signatures, n_clusters=args.n)
    payload = result if isinstance(result, dict) else {"clusters": result}
    _emit(payload, args.json,
          f"{len(signatures)} signatures → {args.n} clusters: {payload.get('assignments', payload)}")
    return 0


def _cmd_verify(args) -> int:
    realized = json.loads(Path(args.realized).read_text())
    predicted = json.loads(Path(args.predicted).read_text())
    result = osc.verify.evaluate(args.prediction_id, realized, predicted)
    verdict = _verdict_of(result)
    payload = dict(result)
    _emit(payload, args.json,
          f"{args.prediction_id}: mae={result.get('mae'):.4f}  mape={result.get('mape'):.4f}"
          f"  verdict={verdict}  re_calibrate={result.get('re_calibrate')}")
    return _exit_code(verdict)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tio-sh", description="Spectrum OS CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decompose")
    d.add_argument("sector_id")
    d.add_argument("--method", default="moving_avg")
    d.add_argument("--store", default=DEFAULT_STORE)
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=_cmd_decompose)

    c = sub.add_parser("correlate")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--max-lag", type=int, default=24)
    c.add_argument("--store", default=DEFAULT_STORE)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=_cmd_correlate)

    k = sub.add_parser("cluster")
    k.add_argument("--n", type=int, default=5)
    k.add_argument("--store", default=DEFAULT_STORE)
    k.add_argument("--json", action="store_true")
    k.set_defaults(func=_cmd_cluster)

    v = sub.add_parser("verify")
    v.add_argument("prediction_id")
    v.add_argument("--realized", required=True)
    v.add_argument("--predicted", required=True)
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=_cmd_verify)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
