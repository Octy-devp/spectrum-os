# Spectrum OS

> **v3.4** — A world-agnostic spectrum computer: a numpy-only kernel + quantum superposition layer + LLM-gated synthesis, running any worldline (α, β, γ, …) in parallel.
>
> Spectrum OS is an **independent operating system**, not a part of any single project. The [ECC project](https://github.com/Octy-devp/ECC) is its **first substrate** — one worldline (β₁) plugged into the engine — but the OS itself is world-agnostic: worlds differ only in what they feed in (rate-matrix content), never in the mechanism.

## What is this?

Spectrum OS decomposes time series into moving-average components, detects dominant periods via autocorrelation, cross-correlates with lag detection, runs Monte Carlo branch simulations, clusters trajectory signatures, and verifies predictions with a ternary verdict. The **quantum layer** adds 6D state vectors (ternary direction × multi-spectrum content), DCA grammar gating, tomography, entanglement, decoherence monitoring, and Markov dynamics — all numpy-only. The **synth layer** is the only place an LLM touches data, and it does so strictly as a *gate* at branch points: the LLM produces structured anchors only; numpy does all dynamics.

The core triad — **spectrum (constraint) → residual (failure) → accident (eruption)** — is domain-agnostic. Every trigger is critical slowing-down; domains differ only in what the rate matrix encodes:

| Domain | Rate matrix encodes | Eruption shape |
|:---|:---|:---|
| Social / political | institutional matrix | **revolution** |
| Economic | production / financial structure | **black swan** |
| Weather | atmospheric physics | **typhoon turn** |
| Military | forces / logistics / geography | **battle reversal** |

The same critical-slowing physics everywhere: small accidents are amplified into phase turns at critical points — **the accident itself is unpredictable (it is fluctuation), but residuals mark the regions where accidents can erupt.**

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
# → D1–D3 ternary direction (will/resistance/relation), D4 period_months, D5 phase, D6 amplitude

t = osc.quantum.project_ternary({"direction": 0.4, "crisis": 0.3, "lag": 0.2, "alternative": 0.1})
# → (1, -1, -1): project a 4-role distribution onto D1–D3

sv2 = osc.quantum.vector_from_roles(
    {"direction": 0.4, "crisis": 0.3, "lag": 0.2, "alternative": 0.1},
    {"period_months": 12.0, "phase": 0.0, "amplitude": 0.7},
)  # → build a full 6D StateVector from roles + spectrum data

# ── Markov dynamics ──
from spectrum_os.quantum.markov import count_transitions, estimate_rate_matrix
counts, anomalies = count_transitions([["crisis","lag","alternative","direction"]])
res = estimate_rate_matrix(counts)
# → {matrix: 4×4 (roles × roles), counts, total_count, bias_notes, verdict}
#   verdict depends on data volume: sparse counts → UNKNOWN, richer → CONTESTED/ASSERTED
```

## World interface: plugin contracts

A world plugs into the OS by providing **seven plugins**. The OS-side contracts live in `spectrum_os/contracts.py` (four dataclasses with `__post_init__` validation + JSON-safe `to_dict()`); the rest already exist or are in progress:

| # | Plugin | Contract | Status |
|:---:|:---|:---|:---|
| ① | Field folder (the world's working directory for one field) | `FieldSpec` | `contracts.py` |
| ② | Actor cards (one structured profile per actor) | `ActorCard` | `contracts.py` |
| ③ | Situation payload (digest + local texture + 6D vector) | `SituationSpec` | `contracts.py` |
| ④ | Source registry (file / http_api / llm_knowledge drivers) | `SourceSpec` | `spectrum_os/sources.py` |
| ⑤ | Constraint field (residual table, saturation, rate matrix) | — | existing probe input |
| ⑥ | World system prompt | — | T16 (partially landed) |
| ⑦ | Field history log (append-only generation record) | `FieldLog` | `contracts.py` |

```python
from spectrum_os.contracts import FieldSpec, ActorCard, SituationSpec, FieldLog

# ① Field folder — declares what a worldline must populate (world-agnostic)
field = FieldSpec(
    field_id="july-crisis-demo",
    worldline="beta",                                     # any world: alpha/beta/gamma...
    temporal_scope={"start": "1914-06", "end": "1914-08"},
    spatial_scope={"primary": "europe", "nodes": ["vienna", "belgrade"]},
    actors=["austria-hungary", "serbia"],
    params={"mobilization_cap": 0.8},
)

# ② Actor card — internal tensions seed multiple storylines per actor
card = ActorCard(
    actor_id="austria-hungary",
    name="Austria-Hungary",
    inherited_conditions={"institutions": "dual monarchy"},
    field_coordinates={"capital": "vienna"},
    internal_tensions=[
        {"axis": "hawks-vs-doves", "forces": ["military", "civilian"],
         "stakes": "mobilization"},
    ],
    temporal_states={"1914-06": {"faction": "hawk-leaning"}},
)

# ③ Situation payload — digest + local texture + optional 6D vector + actors
sit = SituationSpec(
    digest="mobilization gridlock, central delay, external mediation.",
    local_texture={"region": "north", "season": "summer"},
    actors={"austria-hungary": card},
)

# ⑦ Field history log — append-only; selected + unselected kept together (superposition)
log = FieldLog(field_id="july-crisis-demo")
log.add_entry(layer=1, selected={"label": "route-a"}, unselected=[{"label": "route-b"}])

# Every contract serializes JSON-safe for pipeline consumption:
sit.to_dict()          # → {"digest", "local_texture", "6d_vector", "actors"}
card.to_dict()         # → {"actor_id", "name", "inherited_conditions", ...}
field.to_dict()        # → {"field_id", "worldline", "temporal_scope", ...}
log.to_dict()          # → {"field_id", "entries", "meta"}
```

`SituationSpec.to_dict()` maps `vector_6d` (optional) to the pipeline's native `6d_vector` key so the probe pipeline can consume it directly. `contracts.py` also carries the OS-1 universal currency dataclasses (`RoleTrajectory`, `DCASubstrate`, `CLADCorpus`) from PLAN-23 §6.1.

## Human–machine convergence: manual layer trigger

A **probe is a field where the human picks the direction**. The automatic `probe_tree` grows every layer from all passed branches; the manual machinery pauses after each layer so a human selects one branch, and only that branch becomes the next layer's `reflect_on` — breaking the automatic 1:1 continuation lock (one parent can now expand into many children). This is the OS-side mirror of substrate field runs: any world drives the same machinery, and the human acts as the selector at each layer.

```python
from spectrum_os.synth.probe import probe_expand_layer, probe_select, probe_tree_manual
from spectrum_os.synth.convergence import convergence_view, probe_converge_round
from spectrum_os.contracts import FieldLog

situation = {
    "digest": "mobilization gridlock, central delay, external mediation.",
    "local_texture": {"region": "north", "season": "summer"},
}
```

**Option A — one convergence round** (`probe_converge_round`): expand one layer → (optionally) select → (optionally) write back to the field log.

```python
field_log = FieldLog(field_id="demo-field")

r = probe_converge_round(
    situation=situation, n_branch=3, depth=2,
    gate_fn=mock_gate,                    # pass a mock or a real LLM gate
    selected_label="route-a",             # None → expand only, no collapse
    field_log=field_log,
)
print(r["view"])            # text tree; selected marked （已選）, siblings （未選）
r["next_state"]             # carry into the next round
r["next_reflect_on"]        # → [the chosen branch] — the next layer's reflection
field_log.entries           # one entry: {layer, selected, unselected[]}
```

**Option B — the single-layer primitive** (`probe_expand_layer` + `probe_select`): full manual control, `state` carried across calls.

```python
# Layer 1 — expand from the situation
exp1 = probe_expand_layer(situation, n_branch=3, depth=2, gate_fn=mock_gate)
print(convergence_view(exp1["tree"]))     # human reads the tree, picks "route-a"

# Human picks → tree collapses; unselected branches stay (marked, not deleted)
collapsed = probe_select(exp1["result"], selected_label="route-a")
assert collapsed["collapse"]["selected"] == "route-a"

# Layer 2 — reflect on exactly the one branch the human picked
chosen = next(b for b in exp1["passed"] if b["label"] == "route-a")
exp2 = probe_expand_layer(
    situation, state=exp1["state"], reflect_on=[chosen], gate_fn=mock_gate,
)
assert exp2["layer_entry"]["layer"] == 2
```

**Option C — callback-driven** (`probe_tree_manual`): the generator pauses after each layer and calls `on_layer(layer_entry, ctx)`; return a chosen branch to continue, or `None` to stop the tree at that layer.

```python
def on_layer(layer_entry, ctx):
    print(ctx["k"], ctx["passed"])        # layer index + passed branches + echo notes
    return layer_entry["branches"][0]     # pick one; None → stop

res = probe_tree_manual(situation, n_branch=3, depth=2, gate_fn=mock_gate, on_layer=on_layer)
res["meta"]["manual"]          # → True
res["meta"]["stopped_at_layer"]  # set only if on_layer returned None
```

`convergence_view` renders the working tree in `text` / `markdown` / `json` formats (pure function, zero API, zero UI dependencies). Note: `spectrum_os.synth.convergence_view` (re-exported by the `synth` package) is the **programmatic data view** of a probe result; the **render function** lives at `spectrum_os.synth.convergence.convergence_view` — import it from the submodule to avoid shadowing.

## Architecture

```
spectrum_os/
├── kernel/              # Classical spectrum engine (~2.0k lines)
│   ├── sector.py        — In-memory sector store (CRUD + JSON serialization)
│   ├── wave.py          — moving_average(), decompose(), dominant_periods(), correlate()
│   ├── wave_ops.py      — Convenience wrappers for sector-ID-based wave operations
│   ├── template.py      — Pattern matching (Jaccard + MSE similarity)
│   ├── branch.py        — Monte Carlo simulation with ensemble stats
│   ├── cluster.py       — Manual k-means (numpy-only, k-means++ init)
│   ├── verify.py        — MAPE verification + ternary verdict + state_log JSONL
│   └── _types.py        — Verdict enum, SpectrumResult, Sector dataclass
│
├── quantum/             # Quantum superposition layer (~1.9k lines)
│   ├── markov.py        — Markov dynamics spine: transition counts, Dirichlet posterior,
│   │                       horizon 4 quantities (hitting time / absorption / stationary / entropy rate)
│   ├── multigraph.py    — Role multigraph: states as shared nodes, roles as edge labels
│   ├── vector.py        — 6D StateVector: D1–D3 ternary direction, D4–D6 spectrum content
│   ├── tomography.py    — Quantum tomography: enumerate state roles across threads
│   ├── decoherence.py   — Decoherence watch: entropy monitoring for phase transition detection
│   ├── quarantine.py    — Anti-contamination: mechanical blacklist + semantic inference
│   ├── dca_grammar.py   — DCA recursion grammar (crisis→lag→alternative→direction rules)
│   ├── entangle.py      — Entangle: find threads sharing a state; Intervene: reassign and propagate
│   └── standing_wave.py — Standing-wave decomposition: shared = necessary, most divergent = risk window
│
├── synth/               # LLM gate layer (~6.6k lines) — the only place an LLM touches data
│   ├── probe.py         — Decision-tree probe: layer-by-layer tree generation + rigidity measurement,
│   │                       manual layer trigger (probe_tree_manual / probe_expand_layer), probe_select
│   ├── convergence.py   — Human–machine convergence UX: convergence_view (text/markdown/json render)
│   │                       + probe_converge_round (expand → select → field-log write-back)
│   ├── alt_gate.py      — DCA alternative gate: alternative enumeration + adversary situation awareness
│   ├── anchors.py       — "Living mathematics" prompt: 6D vector computation
│   ├── rate_gate.py     — G2 rate estimation: Bayesian fusion + mechanical veto layers
│   ├── gate_prompts.py  — Unified world-agnostic gate prompt + routing-leak/schema checks
│   ├── ensemble.py      — Branch ensemble runner
│   ├── residual_trigger.py — Residual-trigger table + Mode A→B phase switch (accident enumeration)
│   └── expand.py        — Expansion operations
│
├── contracts.py         — World-interface plugin contracts: FieldSpec / ActorCard / SituationSpec /
│                           FieldLog (+ OS-1 universal currency: RoleTrajectory / DCASubstrate / CLADCorpus)
├── sources.py           — OS-1 universal base: SourceSpec registry + file/http_api/llm_knowledge drivers
│
├── extras/              # α₁ analysis tools (~170 lines)
│   ├── fft_spectrum.py  — FFT spectrum (numpy.fft)
│   └── pca_decomp.py    — PCA decomposition (numpy.linalg.svd)
│
├── cli.py               — tio-sh command line (decompose/correlate/cluster/verify/quantum)
├── feeds/               — Data feeds (ecc_feeds, cache)
└── scripts/             — Experiments (sarajevo branch, danube steel, falsify009 rupture, etc.)
```

Total: **36 modules / ~11.4k lines**, 24 test files.

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

### Probe & convergence (`spectrum_os.synth.probe` / `spectrum_os.synth.convergence`)

| Operation | Returns |
|-----------|---------|
| `probe_tree(situation, ...)` | Automatic tree (all passed branches continue) |
| `probe_tree_manual(situation, ..., on_layer=...)` | Tree, pausing per layer for human picks |
| `probe_expand_layer(situation, *, state, reflect_on, ...)` | `{layer_entry, passed, state, tree, result, calls}` |
| `probe_select(result, *, selected_label, ...)` | Collapsed tree (`collapse` + `state_log_entry`) |
| `convergence_view(tree_or_result, *, format, layer, include, path, field_log)` | text / markdown / json render |
| `probe_converge_round(*, situation, selected_label, field_log, ...)` | `{view, expanded, collapse, next_state, next_reflect_on, ...}` |

### Extras

| Operation | Returns |
|-----------|---------|
| `osc.wave.fft_spectrum(id)` | FFT dominant periods |
| `osc.wave.pca_decompose([id1, id2, ...])` | PCA components |

## Testing

```
pytest tests/ -v
```

**631 tests** across 24 test files (all green, ~4.9s). Kernel, quantum, synth, contracts, convergence, extras, feeds, and Markov modules covered.

## Data feeds

- **ECC feeds**: `feeds/ecc_feeds.py` pulls narrative, historical atmosphere, and geo-tech-fin data into sectors (ECC-substrate specific)
- **World Bank data**: `scripts/fetch_economic_data_demo.py` fetches 40-year GDP/inflation from World Bank API, maps to DCA roles, runs Markov dynamics
- **Weather data**: `ECC/scripts/weather-query.py` wraps Open-Meteo API for real-time weather data (ECC-substrate specific)

## Dependencies

- **Runtime:** `numpy>=2.0,<3.0`
- **Dev:** `pytest`

## License

Independent world-agnostic engine. The ECC project (its first substrate) lives in its own repository.
