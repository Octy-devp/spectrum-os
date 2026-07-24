# Spectrum OS

> **v0.1.0** — A numpy-only kernel for time-series spectrum analysis in the β₁ worldline.
>
> Part of the [ECC project](https://github.com/Octy-devp/ECC) (Erstes Central Committee) —
> a two-chamber Soviet alternate-history engine that maintains dual worldlines (α = real history, β₁ = fictional narrative)
> and compares them via spectrum analysis.

## What is this?

Spectrum OS is a **spectrum analysis engine** designed for comparing time-series data across narrative, historical, and economic domains. It decomposes signals into moving-average components, detects dominant periods via autocorrelation, cross-correlates time series with lag detection, runs Monte Carlo branch simulations, clusters trajectory signatures, and verifies predictions against ground truth.

All math is **numpy-only** — no FFT, no SciPy, no external ML libraries.

It was originally built to answer the question: *"Does β₁'s narrative mood follow its geo-financial infrastructure, or vice versa?"* (Answer: gt_fin leads mood by ~16 months.)

## Quick start

```python
import numpy as np
from spectrum_os import osc

# 1. Register a sector (time series with metadata)
osc.sectors.create("hk_mood", values=[0.2, 0.5, -0.1, ...], meta={"chapter": "HK"})

# 2. Decompose into moving-average components
result = osc.wave.decompose("hk_mood")
# → {dominant_periods: [...], valid_range: [...], verdict: ASSERTED|CONTESTED|UNKNOWN}

# 3. Cross-correlate two sectors
r = osc.wave.correlate("gt_fin", "monthly_mood")
# → {r: 0.498, lag: 16, verdict: ASSERTED, ...}

# 4. Register templates and match patterns
osc.template.register("my_pattern", source_values=[...])
osc.template.match("hk_mood")
# → [{template_id: "my_pattern", jaccard: 0.72, mse: 0.03, verdict: ...}]

# 5. Branch simulation (Monte Carlo)
result = osc.branch.simulate("hk_mood", n_simulations=1000, n_steps=30)
# → {mean_delta: 0.023, ci_lower: -0.12, ci_upper: 0.17, ...}

# 6. Cluster trajectory signatures
clusters = osc.cluster.run(sector_ids=["hk", "germany", "russia", ...], n_clusters=3)
# → {labels: [0, 0, 1, ...], stability: 0.87, ...}

# 7. Verify predictions
v = osc.verify.evaluate(actual=[...], predicted=[...])
# → {mape: 12.3, verdict: ASSERTED, state_log: [...]}
```

## Architecture

```
spectrum_os/
├── kernel/
│   ├── sector.py     — In-memory sector store (CRUD + JSON serialization)
│   ├── wave.py       — Core engine: moving_average(), decompose(), dominant_periods(), correlate()
│   ├── wave_ops.py   — Convenience wrappers (lookup sector by ID, then call wave.py)
│   ├── template.py   — Pattern matching (Jaccard + MSE similarity)
│   ├── branch.py     — Monte Carlo simulation with ensemble stats
│   ├── cluster.py    — Manual k-means (numpy-only, k-means++ init, Lloyd iteration)
│   ├── verify.py     — MAPE-based verification with ternary verdict (ASSERTED/CONTESTED/UNKNOWN)
│   └── _types.py     — Verdict enum, SpectrumResult dataclass, Sector dataclass
├── __init__.py        — Exports `osc` (OSC singleton)
```

### Key design decisions

| Decision | Rationale |
|----------|-----------|
| **numpy-only kernel** | Zero external dependencies for core analysis. Reproducible, auditable, and deterministic. |
| **Moving-average decomposition** (not FFT) | Signals are short (7–252 data points) and non-stationary. FFT assumes stationarity and infinite periodic signals. Moving averages preserve temporal locality. |
| **Ternary verdict** (ASSERTED/CONTESTED/UNKNOWN) | Inspired by ECC's T-SALC spectrum framework. Not all results are meaningful — the verdict prevents false confidence on weak data. |
| **Max lag = 36 by default** | Autocorrelation at lags > N/3 becomes unreliable for small-N series. The default is conservative. |
| **In-memory store** | Sectors live in a dict. JSON `save/load` for persistence. No SQLite dependency in kernel. |

## Function reference

All operations (pure Python, no network) accessed via the `osc` singleton:

| Function | Signature | Returns |
|----------|-----------|---------|
| Create sector | `osc.sectors.create(id, values, meta)` | `Sector` |
| List sectors | `osc.sectors.list()` | `dict[str, Sector]` |
| Get sector | `osc.sectors.get(id)` | `Sector` |
| Delete sector | `osc.sectors.delete(id)` | — |
| Save to JSON | `osc.sectors.save(path)` | — |
| Load from JSON | `osc.sectors.load(path)` | — |
| Decompose | `osc.wave.decompose(id)` | `SpectrumResult` |
| Correlate | `osc.wave.correlate(id_a, id_b)` | `SpectrumResult` |
| Register template | `osc.template.register(id, source_values, ...)` | — |
| Match template | `osc.template.match(target_id)` | `list[MatchResult]` |
| List templates | `osc.template.list_templates()` | `dict` |
| Simulate branch | `osc.branch.simulate(id, n_simulations, n_steps)` | `BranchResult` |
| Cluster sectors | `osc.cluster.run(sector_ids, n_clusters)` | `ClusterResult` |
| Verify | `osc.verify.evaluate(actual, predicted)` | `VerifyResult` |

## Data feeds

The `feeds/` directory contains optional data-fetching scripts:

- **weather-query.py** — Open-Meteo API wrapper (free, no key). Fetches forecast/historical weather, converts to sectors, analyzes typhoon signatures. Located in the ECC monorepo (`scripts/weather-query.py`).

## Testing

```
pytest tests/ -v
```

53 tests covering all kernel modules. Run from repo root.

## Dependencies

- **Runtime:** `numpy>=2.0,<3.0`
- **Dev:** `pytest`

## License

Part of the ECC project. See ECC repository for license information.
