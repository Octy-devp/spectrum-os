# Spectrum OS

> **v2.7** — A numpy-only kernel + quantum superposition layer for time-series spectrum analysis.
>
> Part of the [ECC project](https://github.com/Octy-devp/ECC) — an alternate-history engine that maintains dual worldlines
> (α = real history, β₁ = fictional narrative) and compares them via spectrum analysis.

## What is this?

Spectrum OS decomposes time series into moving-average components, detects dominant periods via autocorrelation, cross-correlates with lag detection, runs Monte Carlo branch simulations, clusters trajectory signatures, and verifies predictions with ternary verdict. The **quantum layer** adds 6D state vectors (ternary direction × multi-spectrum content), DCA grammar gating, tomography, entanglement, decoherence monitoring, and Markov dynamics — all numpy-only.

## Quick start

```python
import numpy as np
from spectrum_os import osc

# ── Classical kernel ──
osc.sectors.create("annual_sine", np.sin(2*np.pi*np.arange(120)/12).tolist())
r = osc.wave.decompose("annual_sine")
# → dominant_periods=[12, 24], verdict=ASSERTED

r2 = osc.wave.correlate("annual_sine", "annual_sine")
# → r=1.0, lag=0, equiv_lags=[(0, 1.0), (±12, 1.0), ...]

# ── Quantum layer ──
sv = osc.quantum.StateVector(d1=1, d2=0, d3=-1, d4=12.0, d5=0.0, d6=0.7)
t = osc.quantum.project_ternary(sv)

# ── Markov dynamics ──
from spectrum_os.quantum.markov import count_transitions, estimate_rate_matrix
counts, anomalies = count_transitions([["crisis","lag","alternative","direction"]])
res = estimate_rate_matrix(counts)  # → {matrix: 4×4, verdict: "ASSERTED", ...}
```

## Architecture

```
spectrum_os/
├── kernel/              # Classical spectrum engine (1,958 lines)
│   ├── sector.py        — In-memory sector store (CRUD + JSON serialization)
│   ├── wave.py          — moving_average(), decompose(), dominant_periods(), correlate()
│   ├── wave_ops.py      — Convenience wrappers for sector-ID-based wave operations
│   ├── template.py      — Pattern matching (Jaccard + MSE similarity)
│   ├── branch.py        — Monte Carlo simulation with ensemble stats
│   ├── cluster.py       — Manual k-means (numpy-only, k-means++ init)
│   ├── verify.py        — MAPE verification + ternary verdict + state_log JSONL
│   └── _types.py        — Verdict enum, SpectrumResult, Sector dataclass
│
├── quantum/             # Quantum superposition layer (1,762 lines)
│   ├── markov.py        — Markov dynamics spine: transition counts, Dirichlet posterior,
│   │                       horizon 4 quantities (hitting time / absorption / stationary / entropy rate)
│   ├── multigraph.py    — Role multigraph: states as shared nodes, roles as edge labels
│   ├── vector.py        — 6D StateVector: D1–D3 ternary direction, D4–D6 spectrum content
│   ├── tomography.py    — Quantum tomography: enumerate state roles across threads
│   ├── decoherence.py   — Decoherence watch: entropy monitoring for phase transition detection
│   ├── quarantine.py    — Anti-contamination: mechanical blacklist + semantic inspection
│   ├── dca_grammar.py   — DCA recursion grammar (crisis→lag→alternative→direction rules)
│   └── entangle.py      — Entangle: find threads sharing a state; Intervene: reassign and propagate
│
├── synth/               # LLM gate layer (994 lines)
│   ├── anchors.py       — "Living mathematics" prompt: 6D vector computation
│   ├── rate_gate.py     — G2 rate estimation: Bayesian fusion + mechanical veto layers
│   └── expand.py        — Expansion operations
│
├── extras/              # α₁ analysis tools (166 lines)
│   ├── fft_spectrum.py  — FFT spectrum (numpy.fft)
│   └── pca_decomp.py    — PCA decomposition (numpy.linalg.svd)
│
├── cli.py               — tio-sh command line (decompose/correlate/cluster/verify/quantum)
├── feeds/               — Data feeds (ecc_feeds, cache)
└── scripts/             — Experiments (sarajevo branch, danube steel, falsify009 rupture, etc.)
```

## Operations

### Classical kernel

| Operation | Returns |
|-----------|---------|
| `osc.sectors.create(id, values)` | `Sector` |
| `osc.sectors.list()` / `get(id)` / `delete(id)` | `dict` / `Sector` / — |
| `osc.sectors.save(path)` / `load(path)` | — / — |
| `osc.wave.decompose(id)` | `SpectrumResult` (long/mid/short waves + deviation) |
| `osc.wave.correlate(a, b, max_lag)` | dict (`r`, `lag`, `equiv_lags`) |
| `osc.template.register(id, pattern)` | — |
| `osc.template.match(target_id)` | `list[MatchResult]` |
| `osc.branch.simulate(id, n, steps)` | `BranchResult` |
| `osc.cluster.run(sector_ids, n_clusters)` | `ClusterResult` |
| `osc.verify.evaluate(actual, predicted)` | `VerifyResult` (+ state_log) |
| `osc.verify.init_log(path)` / `query()` / `summary()` | — / filtered entries / stats |

### Quantum layer

| Operation | Returns |
|-----------|---------|
| `osc.quantum.StateVector(d1..d6)` | 6D state vector |
| `osc.quantum.tomography(state_id)` | Role distribution across threads |
| `osc.quantum.entangle(state_id)` | Threads sharing this state |
| `osc.quantum.decoherence_watch(system_id)` | Entropy monitoring |
| `osc.quantum.intervene(state, thread, role)` | Role reassign + propagate |
| `osc.quantum.project_ternary(sv)` | D1–D3 ternary verdict |
| `osc.quantum.quarantine_check(text, year)` | Anachronism detection |
| `osc.quantum.dca_grammar.validate_alternative(from, to)` | Transition legality |
| `osc.quantum.dca_grammar.mutate_alternative(role)` | Valid next roles |

### Markov dynamics (`osc.quantum.markov.*`)

| Operation | Returns |
|-----------|---------|
| `count_transitions(sequences)` | 4×4 counts + anomalies |
| `estimate_rate_matrix(counts)` | Dirichlet posterior mean 4×4 P matrix |
| `sample_rate_matrix(counts, n=200)` | n posterior samples |
| `hitting_times(P, target)` | Expected steps to target role |
| `absorption_probabilities(P, absorbing)` | Absorbing probabilities |
| `stationary_distribution(P)` | π vector |
| `entropy_rate(P)` | Markov entropy rate |
| `spectral_gap(P)` | 1 − |λ₂| |
| `summarize_samples(values)` | Mean + 95% CI |

### Extras

| Operation | Returns |
|-----------|---------|
| `osc.wave.fft_spectrum(id)` | FFT dominant periods |
| `osc.wave.pca_decompose([id1, id2, ...])` | PCA components |

## Testing

```
pytest tests/ -v
```

**241 tests** across 14 test files (2,912 lines). All kernel, quantum, synth, extras, feeds, and Markov modules covered.

## Data feeds

- **ECC feeds**: `feeds/ecc_feeds.py` pulls narrative, historical atmosphere, and geo-tech-fin data into sectors
- **World Bank data**: `scripts/fetch_economic_data_demo.py` fetches 40-year GDP/inflation from World Bank API, maps to DCA roles, runs Markov dynamics
- **Weather data**: `ECC/scripts/weather-query.py` wraps Open-Meteo API for real-time weather data

## Dependencies

- **Runtime:** `numpy>=2.0,<3.0`
- **Dev:** `pytest`

## License

Part of the ECC project. See ECC repository.
