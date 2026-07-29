# Spectrum OS v2.5 Vector Contract + Quantum Layer — Implementation Plan

> **For agentic workers:** REQUIRED: Use the `subagent-driven-development` agent (recommended) or `executing-plans` agent to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the v2.5 vector contract (6D state vector), quantum entangle/intervene operations, synth LLM prompt upgrade, state_log persistence, DCA grammar gate, and anti-pollution double-layer quarantine.

**Architecture:** Phase 1 lays the 6D vector foundation (`spectrum_os/quantum/vector.py`). Phase 2 runs four independent tracks in parallel — quantum ops, LLM gate upgrade, state_log persistence, and DCA grammar gate. Phase 3 integrates anti-pollution quarantine with end-to-end tests and verification.

**Tech Stack:** Python 3.10+, numpy ≥2.0, DeepSeek v4-flash (LLM gate), ECC api_utils.py (shared API module)

---

## File Map

| File | Action | Responsibility |
|:---|:---|:---|
| `spectrum_os/quantum/vector.py` | **Create** | 6D state vector dataclass, D1-D3 ternary projection, D4-D6 multi-spectrum content, 4-role→6D mapping |
| `spectrum_os/quantum/__init__.py` | Modify | Export `StateVector`, `project_ternary`, `vector_from_roles` |
| `spectrum_os/quantum/entangle.py` | **Create** | `entangle()` — all threads sharing a state; `intervene()` — role reassign + propagate |
| `spectrum_os/synth/anchors.py` | Modify | Upgrade LLM prompt from "結構化驗證器" to "living mathematics" — 6D vector computation output |
| `spectrum_os/synth/__init__.py` | Modify | Export new prompt constants if needed |
| `spectrum_os/kernel/verify.py` | Modify | Add `query()` and `summary()` for state_log JSONL |
| `spectrum_os/kernel/branch.py` | Modify | Add DCA grammar validation step in `simulate()` |
| `spectrum_os/quantum/quarantine.py` | **Create** | Double-layer anti-pollution: mechanical blacklist + second LLM inspector |
| `spectrum_os/quantum/dca_grammar.py` | **Create** | DCA grammar rules (Crisis→Lag→Alternative→Direction recursion), validate alternative against grammar |
| `tests/test_vector.py` | **Create** | Unit tests for 6D vector, ternary projection, role→vector mapping |
| `tests/test_entangle.py` | **Create** | Unit tests for entangle() and intervene() |
| `tests/test_quarantine.py` | **Create** | Unit tests for double-layer quarantine |
| `tests/test_dca_grammar.py` | **Create** | Unit tests for DCA grammar validation |

---

## Phase 1: Foundation — 6D State Vector (sequential, no parallelism)

### Task 1.1: Create the 6D StateVector dataclass

**Files:**
- Create: `spectrum_os/quantum/vector.py`

**Description:** Define the `StateVector` dataclass with 6 dimensions: D1-D3 ternary direction ({-1,0,+1}), D4-D6 multi-spectrum content (float). Include `__repr__`, `to_dict()`, `from_dict()`, and `norm()`.

**Can run parallel:** NO — foundation for all downstream tasks

**Effort:** 30 min

- [ ] **Step 1: Create `spectrum_os/quantum/vector.py`**

```python
"""6D state vector — v2.5 vector contract (PLAN-23 §〇.五).

D1–D3: ternary direction projection — {−1, 0, +1}
  - D1 = will (意志) — Direction role weight minus Crisis role weight
  - D2 = resistance (阻力) — Lag role weight
  - D3 = relation (關係) — Alternative role weight minus (Crisis + Lag + Direction)/3

D4–D6: multi-spectrum content — endogenous tendency of spectrum axes
  - D4 = frequency (dominant period in months, or 0)
  - D5 = phase (phase angle in radians, wrapped to [-π, π])
  - D6 = amplitude (normalized deviation amplitude)

The quantum superposition is the PARALLEL coexistence of D4–D6 content —
the ternary (D1–D3) is a direction projection, NOT a cage for the state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# Role order for consistent indexing
_ROLE_ORDER: tuple[str, ...] = ("direction", "crisis", "lag", "alternative")


@dataclass
class StateVector:
    """6D state vector: [D1 will, D2 resistance, D3 relation,
                         D4 frequency, D5 phase, D6 amplitude]."""

    d1: float  # will: direction - crisis, range [-1, +1]
    d2: float  # resistance: lag, range [0, 1]
    d3: float  # relation: alternative - mean(other roles), range [-1, +1]
    d4: float  # frequency: dominant period in months (0 = no period)
    d5: float  # phase: phase angle in radians, range [-π, +π]
    d6: float  # amplitude: normalized deviation amplitude, range [0, +∞)
    meta: dict[str, Any] | None = None

    @property
    def ternary(self) -> np.ndarray:
        """D1–D3 ternary direction as numpy array."""
        return np.array([self.d1, self.d2, self.d3], dtype=np.float64)

    @property
    def spectrum(self) -> np.ndarray:
        """D4–D6 multi-spectrum content as numpy array."""
        return np.array([self.d4, self.d5, self.d6], dtype=np.float64)

    @property
    def full(self) -> np.ndarray:
        """Full 6D vector as numpy array."""
        return np.array([self.d1, self.d2, self.d3,
                         self.d4, self.d5, self.d6], dtype=np.float64)

    def norm(self) -> float:
        """Euclidean norm of the full 6D vector."""
        return float(np.linalg.norm(self.full))

    def ternary_norm(self) -> float:
        """Euclidean norm of the ternary sub-vector (D1–D3)."""
        return float(np.linalg.norm(self.ternary))

    def spectrum_norm(self) -> float:
        """Euclidean norm of the spectrum sub-vector (D4–D6)."""
        return float(np.linalg.norm(self.spectrum))

    def to_dict(self) -> dict[str, Any]:
        return {
            "d1": self.d1, "d2": self.d2, "d3": self.d3,
            "d4": self.d4, "d5": self.d5, "d6": self.d6,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StateVector:
        return cls(
            d1=float(d["d1"]), d2=float(d["d2"]), d3=float(d["d3"]),
            d4=float(d["d4"]), d5=float(d["d5"]), d6=float(d["d6"]),
            meta=d.get("meta"),
        )

    def __repr__(self) -> str:
        return (f"StateVector(D1={self.d1:+.2f}, D2={self.d2:.2f}, "
                f"D3={self.d3:+.2f}, D4={self.d4:.1f}, "
                f"D5={self.d5:+.2f}, D6={self.d6:.3f})")


def project_ternary(role_distribution: dict[str, float]) -> tuple[float, float, float]:
    """Project a 4-role distribution onto D1–D3 ternary direction.

    Parameters
    ----------
    role_distribution
        Dict mapping role names to weights, e.g.
        ``{"direction": 0.3, "crisis": 0.4, "lag": 0.1, "alternative": 0.2}``.

    Returns
    -------
    (d1, d2, d3) where each is in the appropriate range.
    """
    d = role_distribution.get("direction", 0.0)
    c = role_distribution.get("crisis", 0.0)
    l = role_distribution.get("lag", 0.0)
    a = role_distribution.get("alternative", 0.0)

    # Normalize to sum 1
    total = d + c + l + a
    if total <= 0:
        return (0.0, 0.0, 0.0)
    d, c, l, a = d / total, c / total, l / total, a / total

    d1 = d - c                     # will: direction minus crisis, [-1, +1]
    d2 = l                          # resistance: lag weight, [0, 1]
    d3 = a - (d + c + l) / 3.0     # relation: alternative minus mean of others, [-1, +1]

    return (d1, d2, d3)


def project_spectrum(decompose_result: dict | None = None,
                     dominant_period: float = 0.0,
                     phase_angle: float = 0.0,
                     amplitude: float = 0.0) -> tuple[float, float, float]:
    """Extract D4–D6 multi-spectrum content from a decompose result or raw values.

    Parameters
    ----------
    decompose_result
        The dict returned by ``wave.decompose()``. If provided, D4-D6
        are extracted from ``dominant_periods`` and ``deviation``.
    dominant_period
        Override: dominant period in months.
    phase_angle
        Override: phase angle in radians.
    amplitude
        Override: normalized deviation amplitude.

    Returns
    -------
    (d4, d5, d6)
    """
    if decompose_result is not None:
        values = decompose_result.get("values", decompose_result)
        periods = values.get("dominant_periods", [])
        d4 = float(periods[0]) if periods else 0.0
        deviation = values.get("deviation")
        if deviation is not None and len(deviation) > 0:
            d6 = float(np.std(deviation))
            # Phase from first peak in deviation
            peak_idx = int(np.argmax(np.abs(deviation)))
            d5 = 2.0 * math.pi * peak_idx / max(len(deviation), 1)
            d5 = (d5 + math.pi) % (2 * math.pi) - math.pi  # wrap to [-π, π]
        else:
            d5, d6 = 0.0, 0.0
        return (d4, d5, d6)

    return (float(dominant_period), float(phase_angle), float(amplitude))


def vector_from_roles(role_distribution: dict[str, float],
                      decompose_result: dict | None = None,
                      dominant_period: float = 0.0,
                      phase_angle: float = 0.0,
                      amplitude: float = 0.0) -> StateVector:
    """Build a full 6D StateVector from a 4-role distribution and spectrum data.

    This is the primary entry point for the vector transformation pipeline:
    4-role labels → 6D [ternary direction, multi-spectrum content].

    Parameters
    ----------
    role_distribution
        From ``quantum.tomography()`` -> ``values["role_distribution"]``.
    decompose_result
        Optional ``wave.decompose()`` result for D4-D6 extraction.
    dominant_period, phase_angle, amplitude
        Manual overrides for D4-D6 (used when decompose_result is None).
    """
    d1, d2, d3 = project_ternary(role_distribution)
    if decompose_result is not None:
        d4, d5, d6 = project_spectrum(decompose_result=decompose_result)
    else:
        d4, d5, d6 = project_spectrum(
            dominant_period=dominant_period,
            phase_angle=phase_angle,
            amplitude=amplitude)
    return StateVector(d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, d6=d6)
```

- [ ] **Step 2: Export from `spectrum_os/quantum/__init__.py`** — add `StateVector`, `project_ternary`, `project_spectrum`, `vector_from_roles` to the module's `__all__` and imports.

Modify `spectrum_os/quantum/__init__.py` — add after the existing `from .decoherence import decoherence_watch`:

```python
from .vector import (
    StateVector,
    project_ternary,
    project_spectrum,
    vector_from_roles,
)
```

And extend `__all__`:

```python
__all__ = [
    ...
    "StateVector",
    "project_ternary",
    "project_spectrum",
    "vector_from_roles",
]
```

- [ ] **Step 3: Verify imports work**

Run: `python -c "from spectrum_os.quantum import StateVector, project_ternary, vector_from_roles; print('OK')"`
Expected: `OK`

---

## Phase 2: Quantum Completion + LLM Gate + State Log + Grammar Gate (parallel)

All four tracks in Phase 2 are independent — they touch different files and share no mutable state. They CAN and SHOULD run in parallel.

---

### Track 2A: quantum.entangle + quantum.intervene

### Task 2A.1: Implement `quantum.entangle()`

**Files:**
- Create: `spectrum_os/quantum/entangle.py`

**Description:** `entangle(state_id)` finds all threads (measurement contexts) that share a given state node in the multigraph. Entanglement = shared-node topology. Returns the set of thread IDs plus per-thread role assignments.

**Can run parallel:** YES (with Tracks 2B, 2C, 2D)

**Effort:** 45 min

- [ ] **Step 1: Create `spectrum_os/quantum/entangle.py`**

```python
"""quantum.entangle / intervene — shared-node entanglement and role reassignment
(PLAN-23 §6.3).

- entangle(state_id)  → all threads sharing a state (entanglement = shared node)
- intervene(state_id, thread, new_role) → role reassign + cascade to shared threads
"""

from __future__ import annotations

from ..kernel._types import SpectrumResult, Verdict
from .multigraph import ROLES, RoleAssignment, RoleMultigraph
from .tomography import tomography


def entangle(graph: RoleMultigraph, state_id: str) -> SpectrumResult:
    """Find all threads that share *state_id* in the multigraph.

    Entanglement is the mechanical fact that one node participates in
    multiple measurement contexts. This function enumerates those threads
    and the role the state plays in each.

    Parameters
    ----------
    graph
        The role multigraph to query.
    state_id
        The shared node to inspect.

    Returns
    -------
    SpectrumResult with:
    - ``state_id``
    - ``n_threads`` — number of entangled threads
    - ``threads`` — list of {thread, role, weight, mass, stale}
    - ``roles_seen`` — set of unique roles this state plays
    - ``entangled`` — True when >= 2 threads (the state is superposed)
    """
    assignments = graph.get_state_assignments(state_id)

    if not assignments:
        return SpectrumResult(
            values={
                "state_id": state_id,
                "n_threads": 0,
                "threads": [],
                "roles_seen": [],
                "entangled": False,
            },
            verdict=Verdict.UNKNOWN,
            confidence_reason=f"state '{state_id}' not found in graph",
        )

    threads = []
    roles_seen: set[str] = set()
    for a in assignments:
        threads.append({
            "thread": a.thread,
            "role": a.role,
            "weight": a.weight,
            "mass": a.mass,
            "stale": a.stale,
        })
        if not a.stale:
            roles_seen.add(a.role)

    n_threads = len(threads)
    entangled = n_threads >= 2

    if n_threads == 1:
        verdict = Verdict.CONTESTED
        reason = "single thread — superposition unobservable"
    elif any(a.stale for a in assignments):
        verdict = Verdict.CONTESTED
        reason = f"{n_threads} threads but some stale — entanglement may be partial"
    else:
        verdict = Verdict.ASSERTED
        reason = f"{n_threads} active threads — state is superposed"

    return SpectrumResult(
        values={
            "state_id": state_id,
            "n_threads": n_threads,
            "threads": threads,
            "roles_seen": sorted(roles_seen),
            "entangled": entangled,
        },
        verdict=verdict,
        confidence_reason=reason,
    )


def intervene(graph: RoleMultigraph, state_id: str, thread: str,
              new_role: str, *, note: str = "") -> SpectrumResult:
    """Reassign a state's role in one thread and propagate to shared threads.

    This is the "action ripple" operation of PLAN-23 §6.3: changing a role
    in one measurement context may shift superposition balance across all
    threads that share the same state.

    Parameters
    ----------
    graph
        The role multigraph to mutate. This is an IN-PLACE operation.
    state_id
        The node whose role is being reassigned.
    thread
        The specific measurement context (thread) to modify.
    new_role
        Must be one of ROLES.
    note
        Provenance note attached to the modified assignment.

    Returns
    -------
    SpectrumResult with:
    - ``state_id``, ``thread``, ``old_role``, ``new_role``
    - ``before`` — tomography snapshot before intervention
    - ``after`` — tomography snapshot after intervention
    - ``affected_threads`` — list of threads whose role changed
    """
    if new_role not in ROLES:
        raise ValueError(
            f"new_role must be one of {ROLES}, got {new_role!r}")

    # Snapshot before
    before = tomography(graph, state_id)

    # Find the assignment to modify
    assignments = graph.get_state_assignments(state_id)
    target = None
    for a in assignments:
        if a.thread == thread:
            target = a
            break

    if target is None:
        return SpectrumResult(
            values={
                "state_id": state_id,
                "thread": thread,
                "old_role": None,
                "new_role": new_role,
                "before": before.values,
                "after": before.values,
                "affected_threads": [],
            },
            verdict=Verdict.UNKNOWN,
            confidence_reason=(
                f"state '{state_id}' not found in thread '{thread}'"),
        )

    old_role = target.role
    if old_role == new_role:
        return SpectrumResult(
            values={
                "state_id": state_id,
                "thread": thread,
                "old_role": old_role,
                "new_role": new_role,
                "before": before.values,
                "after": before.values,
                "affected_threads": [],
            },
            verdict=Verdict.ASSERTED,
            confidence_reason="new_role identical to old_role — no change",
        )

    # Mutate
    target.role = new_role
    if note:
        target.note = (target.note + "; " if target.note else "") + note

    # Snapshot after
    after = tomography(graph, state_id)

    affected = [thread]
    return SpectrumResult(
        values={
            "state_id": state_id,
            "thread": thread,
            "old_role": old_role,
            "new_role": new_role,
            "before": before.values,
            "after": after.values,
            "affected_threads": affected,
        },
        verdict=Verdict.ASSERTED,
        confidence_reason=(
            f"role reassigned: {old_role} → {new_role} in thread '{thread}'"),
    )
```

- [ ] **Step 2: Ensure `RoleMultigraph.get_state_assignments()` exists**

Check `spectrum_os/quantum/multigraph.py` — the `RoleMultigraph` class needs a `get_state_assignments(state_id)` method. If not present, add:

```python
def get_state_assignments(self, state_id: str) -> list[RoleAssignment]:
    """Return all role assignments involving *state_id* (as source or target)."""
    result: list[RoleAssignment] = []
    for a in self.assignments:
        if a.state_id == state_id:
            result.append(a)
    # Also check if state_id appears as target in any edge
    for a in self.assignments:
        if a.thread.endswith(f"->{state_id}") and a.state_id != state_id:
            # This is the SOURCE of an edge where state_id is target —
            # we need the TARGET's role. Reconstruct from the edge label.
            parts = a.thread.split("->")
            if len(parts) == 2 and parts[1] == state_id:
                target_role = parts[0].split("-")[-1] if "-" in parts[0] else ""
                if target_role in ROLES:
                    result.append(RoleAssignment(
                        state_id=state_id,
                        thread=a.thread,
                        role=target_role,
                        weight=a.weight,
                        mass=a.mass,
                        stale=a.stale,
                        source=a.source,
                    ))
    return result
```

- [ ] **Step 3: Export from `spectrum_os/quantum/__init__.py`**

Add:
```python
from .entangle import entangle, intervene
```

And add `"entangle"`, `"intervene"` to `__all__`.

- [ ] **Step 4: Verify**

Run: `python -c "from spectrum_os.quantum import entangle, intervene; print('OK')"`
Expected: `OK`

---

### Track 2B: Synth LLM Prompt Upgrade to "Formal 6D Computation"

### Task 2B.1: Upgrade synth.anchors system prompt

**Files:**
- Modify: `spectrum_os/synth/anchors.py` (the `_SYSTEM_MESSAGE` and `_SCHEMA_RULES` constants)

**Description:** Replace the "結構化數據驗證器" prompt with a "living mathematics" prompt. The LLM now operates on 6D vectors — it does formal computation on [D1-D3 ternary, D4-D6 spectrum], not narrative interpretation. Input is vector structure → output is vector computation.

**Can run parallel:** YES (with Tracks 2A, 2C, 2D)

**Effort:** 45 min

- [ ] **Step 1: Replace `_SYSTEM_MESSAGE` in `spectrum_os/synth/anchors.py`**

Find and replace the `_SYSTEM_MESSAGE` constant (around line 208):

Old:
```python
_SYSTEM_MESSAGE = (
    "你是一個結構化數據驗證器。唯一任務：把輸入的校準數字與事件清單，"
    "轉換為指定 schema 的錨點 JSON。你只做參數提取，不做敘事、不做評論、"
    "不做角色扮演。你只輸出 JSON——直接輸出，無開場白。"
)
```

New:
```python
_SYSTEM_MESSAGE = (
    "你是活的數學（living mathematics）——一台在 6D 狀態向量上執行形式計算的機器。"
    "你的唯一任務：接收向量結構（三進制方向 [D1-D3] + 多頻譜內容 [D4-D6]），"
    "產出向量計算結果。你理解 DCA 文法（Crisis→Lag→Alternative→Direction 遞歸），"
    "但你不做世界大事的詮釋、不寫敘事、不解釋意義。詮釋層歸人。"
    "你的語言是形式約束與數字轉換，不是散文。"
    "你只輸出 JSON——直接輸出，無開場白。"
)
```

- [ ] **Step 2: Add 6D vector schema to `_SCHEMA_RULES`**

After the existing `_SCHEMA_RULES` opening line `【輸出 schema——機械契約，鍵名不可更改】`, add a 6D vector section before the existing JSON template:

```python
_VECTOR_RULES = """\
【6D 向量計算——v2.5 向量契約】
每個 turning_point / event_shock 可附帶選填的 ``vector_6d`` 欄位：
{
  "d1": <float, [-1,+1], 意志: direction − crisis>,
  "d2": <float, [0,1], 阻力: lag weight>,
  "d3": <float, [-1,+1], 關係: alternative − mean(others)>,
  "d4": <float, >=0, 頻率: dominant period months (0=無週期)>,
  "d5": <float, [-π,+π], 相位: phase angle radians>,
  "d6": <float, >=0, 振幅: normalized deviation amplitude>
}

向量計算規則：
- D1–D3 從角色分佈投影：角色權重 → 三進制方向（project_ternary）
- D4–D6 從校準數字推導：數量級變化率 → 頻率；轉折點間距 → 相位；波動幅度 → 振幅
- ``vector_6d`` 是選填的——若校準數據不足以推導 D4-D6，則省略該欄位
- 🔴 禁令：不得憑空發明 D4-D6 值（沒有校準數據 → 不填 vector_6d）
"""
```

Then insert `_VECTOR_RULES` into `_build_prompt()` — add it after the `lines.append(_SCHEMA_RULES.replace(...))` line:

```python
    lines.append(_SCHEMA_RULES.replace("__N_MONTHS__", str(period["n_months"])))
    lines.append("")
    lines.append(_VECTOR_RULES)
```

- [ ] **Step 3: Update `assert_contract()` to accept optional `vector_6d`**

In `assert_contract()`, after the turning_points validation loop, add:

```python
    # ---- optional vector_6d on turning_points -------------------------------
    for i, tp in enumerate(tps):
        v6 = tp.get("vector_6d")
        if v6 is None:
            continue
        if not isinstance(v6, dict):
            raise AnchorContractError(
                f"turning_points[{i}].vector_6d must be a dict or absent, "
                f"got {type(v6).__name__}")
        for dim in ("d1", "d2", "d3", "d4", "d5", "d6"):
            if dim in v6 and not _is_number(v6[dim]):
                raise AnchorContractError(
                    f"turning_points[{i}].vector_6d.{dim} must be numeric "
                    f"or absent")
```

- [ ] **Step 4: Verify prompt builds without error**

Run: `python -c "
from spectrum_os.synth.anchors import _build_prompt
spec = {'name':'test','description':'test sector','period':{'start':'1910','end':'1911','n_months':12},'calibration':[],'beta1_events':[]}
p = _build_prompt(spec)
print('VECTOR_RULES' in p)
"`
Expected: `True`

---

### Track 2C: State Log JSONL Query & Summary

### Task 2C.1: Implement `verify.query()` and `verify.summary()`

**Files:**
- Modify: `spectrum_os/kernel/verify.py`

**Description:** Add `query()` for filtered reads from the JSONL state log and `summary()` for aggregate statistics (mean MAPE, verdict distribution, re-calibrate rate). Both operate on the persisted JSONL file when available, falling back to the in-memory buffer.

**Can run parallel:** YES (with Tracks 2A, 2B, 2D)

**Effort:** 45 min

- [ ] **Step 1: Add `query()` function to `spectrum_os/kernel/verify.py`**

Add after the `_resolve_log_path` function:

```python
def query(*,
          path: str | Path | None = None,
          prediction_id: str | None = None,
          verdict: str | None = None,
          re_calibrate: bool | None = None,
          ts_after: str | None = None,
          ts_before: str | None = None,
          limit: int = 1000) -> list[dict]:
    """Query the state log with optional filters.

    Reads from the persisted JSONL file when available (via *path* or
    ``init_log``), falling back to the in-memory buffer.

    Parameters
    ----------
    path
        JSONL file path. Defaults to the one set by ``init_log()``.
    prediction_id
        Exact match filter.
    verdict
        One of ``"asserted"``, ``"contested"``, ``"unknown"``.
    re_calibrate
        Filter by ``re_calibrate`` flag.
    ts_after
        ISO 8601 timestamp — return entries after this time.
    ts_before
        ISO 8601 timestamp — return entries before this time.
    limit
        Max entries to return.

    Returns
    -------
    List of state log dicts matching all filters.
    """
    p = _resolve_log_path(path) if (path is not None or _log_path is not None) else None

    if p is not None and p.exists():
        # Read from persisted JSONL
        rows: list[dict] = []
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _match_filters(rec, prediction_id, verdict,
                                  re_calibrate, ts_after, ts_before):
                    rows.append(rec)
                    if len(rows) >= limit:
                        break
        return rows

    # Fallback to in-memory buffer
    rows = []
    for rec in reversed(_state_log):
        if _match_filters(rec, prediction_id, verdict,
                          re_calibrate, ts_after, ts_before):
            rows.insert(0, rec)
            if len(rows) >= limit:
                break
    return rows


def _match_filters(rec: dict,
                   prediction_id: str | None,
                   verdict: str | None,
                   re_calibrate: bool | None,
                   ts_after: str | None,
                   ts_before: str | None) -> bool:
    if prediction_id is not None and rec.get("prediction_id") != prediction_id:
        return False
    if verdict is not None and rec.get("verdict") != verdict:
        return False
    if re_calibrate is not None and rec.get("re_calibrate") != re_calibrate:
        return False
    ts = rec.get("ts", "")
    if ts_after is not None and ts < ts_after:
        return False
    if ts_before is not None and ts > ts_before:
        return False
    return True


def summary(*,
            path: str | Path | None = None,
            ts_after: str | None = None,
            ts_before: str | None = None) -> dict:
    """Aggregate statistics over the state log.

    Parameters
    ----------
    path, ts_after, ts_before
        Same as :func:`query`.

    Returns
    -------
    dict with:
    - ``n_total`` — total matching entries
    - ``mean_mape``, ``median_mape``, ``std_mape``
    - ``mean_mae``, ``median_mae``
    - ``verdict_counts`` — {asserted: N, contested: N, unknown: N}
    - ``re_calibrate_rate`` — fraction where re_calibrate is True
    - ``min_ts``, ``max_ts`` — time range
    """
    rows = query(path=path, ts_after=ts_after, ts_before=ts_before, limit=100000)
    if not rows:
        return {
            "n_total": 0,
            "mean_mape": 0.0, "median_mape": 0.0, "std_mape": 0.0,
            "mean_mae": 0.0, "median_mae": 0.0,
            "verdict_counts": {"asserted": 0, "contested": 0, "unknown": 0},
            "re_calibrate_rate": 0.0,
            "min_ts": None, "max_ts": None,
        }

    mapes = [r["mape"] for r in rows]
    maes = [r["mae"] for r in rows]
    verdicts = [r.get("verdict", "unknown") for r in rows]
    recal = [r.get("re_calibrate", False) for r in rows]
    timestamps = [r.get("ts", "") for r in rows]

    return {
        "n_total": len(rows),
        "mean_mape": float(np.mean(mapes)),
        "median_mape": float(np.median(mapes)),
        "std_mape": float(np.std(mapes)),
        "mean_mae": float(np.mean(maes)),
        "median_mae": float(np.median(maes)),
        "verdict_counts": {
            "asserted": verdicts.count("asserted"),
            "contested": verdicts.count("contested"),
            "unknown": verdicts.count("unknown"),
        },
        "re_calibrate_rate": sum(recal) / len(recal) if recal else 0.0,
        "min_ts": min(timestamps) if timestamps else None,
        "max_ts": max(timestamps) if timestamps else None,
    }
```

- [ ] **Step 2: Verify query and summary work**

Run:
```bash
python -c "
from spectrum_os.kernel import verify
import tempfile, os

# Set up temp log
tmp = os.path.join(tempfile.mkdtemp(), 'log.jsonl')
verify.init_log(tmp)

# Add test entries
verify.verify('p1', [1,2,3], [1,2,3])
verify.verify('p2', [1,2,3], [10,20,30])

# Query
r = verify.query(verdict='asserted')
print('query asserted:', len(r))

# Summary
s = verify.summary()
print('summary n_total:', s['n_total'])
print('summary verdicts:', s['verdict_counts'])

verify.init_log(None)
print('OK')
"
```
Expected: `query asserted: 1`, `summary n_total: 2`, verdicts with asserted + unknown, `OK`

---

### Track 2D: DCA Grammar Gate for branch.simulate

### Task 2D.1: Create DCA grammar module

**Files:**
- Create: `spectrum_os/quantum/dca_grammar.py`

**Description:** Define DCA grammar rules (Crisis→Lag→Alternative→Direction recursion, edge type constraints) and a `validate_alternative()` function that checks whether a proposed alternative conforms to the grammar before it enters branch simulation.

**Can run parallel:** YES (with Tracks 2A, 2B, 2C)

**Effort:** 1 hr

- [ ] **Step 1: Create `spectrum_os/quantum/dca_grammar.py`**

```python
"""DCA grammar gate — validate alternatives against DCA recursion rules
(PLAN-23 §五, DCA 文法閘).

The DCA grammar (Crisis→Lag→Alternative→Direction) constrains what
alternatives can be generated at a given branch point. This module
provides the mechanical half of the grammar gate: rule definitions
and structural validation. The LLM gate (synth layer) provides the
generative half: grammar-aware alternative proposal.

Grammar Rules
-------------
1. A Crisis node must have at least one outgoing Alternative edge.
2. A Lag node must have exactly one outgoing edge (it carries inertia).
3. An Alternative node may branch into multiple Direction nodes.
4. A Direction node is terminal (no required outgoing edges).
5. Self-loops are forbidden.
6. Cycles of length < 3 are forbidden (immediate back-and-forth).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..kernel._types import SpectrumResult, Verdict

# Valid DCA edge relation types (source_role->target_role)
VALID_RELATIONS: set[str] = {
    "crisis->lag", "crisis->alternative",
    "lag->alternative", "lag->direction",
    "alternative->direction", "alternative->crisis",
    "direction->crisis",
}

# Role constraints per node type
NODE_CONSTRAINTS: dict[str, dict[str, int | None]] = {
    "crisis":     {"min_outgoing": 1, "max_outgoing": None, "terminal": False},
    "lag":        {"min_outgoing": 1, "max_outgoing": 1,    "terminal": False},
    "alternative": {"min_outgoing": 1, "max_outgoing": None, "terminal": False},
    "direction":  {"min_outgoing": 0, "max_outgoing": None, "terminal": True},
}


@dataclass
class GrammarViolation:
    """A single grammar rule violation."""
    rule: str
    message: str
    nodes: list[str] = field(default_factory=list)


def validate_relation(relation: str) -> GrammarViolation | None:
    """Check whether a relation string is a valid DCA edge type."""
    if relation not in VALID_RELATIONS:
        return GrammarViolation(
            rule="invalid_relation",
            message=f"'{relation}' is not a valid DCA edge relation. "
                    f"Valid relations: {sorted(VALID_RELATIONS)}",
        )
    return None


def validate_alternative(proposed_nodes: list[dict],
                         existing_edges: list[tuple[str, str, str]] | None = None
                         ) -> SpectrumResult:
    """Validate a proposed alternative against DCA grammar rules.

    Parameters
    ----------
    proposed_nodes
        List of dicts, each with ``node_id``, ``role``, and optionally
        ``outgoing`` (list of target node_ids).
    existing_edges
        Optional list of (source_id, relation, target_id) for cycle checking.

    Returns
    -------
    SpectrumResult with:
    - ``valid`` — True if grammar is satisfied
    - ``violations`` — list of GrammarViolation dicts
    - ``warnings`` — non-fatal concerns (e.g., terminal node has outgoing)
    """
    violations: list[dict] = []
    warnings: list[dict] = []

    node_ids = {n["node_id"] for n in proposed_nodes}
    node_roles: dict[str, str] = {}
    outgoing: dict[str, list[str]] = {}

    for n in proposed_nodes:
        nid = n["node_id"]
        role = n.get("role", "")
        node_roles[nid] = role
        outgoing[nid] = n.get("outgoing", [])

        # Check role is valid
        if role not in NODE_CONSTRAINTS:
            violations.append({
                "rule": "invalid_role",
                "message": f"node '{nid}' has invalid role '{role}'. "
                           f"Valid: {sorted(NODE_CONSTRAINTS)}",
                "nodes": [nid],
            })
            continue

        constraints = NODE_CONSTRAINTS[role]
        n_out = len(outgoing[nid])

        # Min outgoing
        min_out = constraints["min_outgoing"]
        if min_out is not None and n_out < min_out:
            violations.append({
                "rule": "insufficient_outgoing",
                "message": f"node '{nid}' (role={role}) has {n_out} outgoing "
                           f"edges but needs at least {min_out}",
                "nodes": [nid],
            })

        # Max outgoing
        max_out = constraints["max_outgoing"]
        if max_out is not None and n_out > max_out:
            violations.append({
                "rule": "excess_outgoing",
                "message": f"node '{nid}' (role={role}) has {n_out} outgoing "
                           f"edges but allows at most {max_out}",
                "nodes": [nid],
            })

        # Terminal node with outgoing — warning, not violation
        if constraints["terminal"] and n_out > 0:
            warnings.append({
                "rule": "terminal_has_outgoing",
                "message": f"node '{nid}' is a Direction (terminal) but has "
                           f"{n_out} outgoing edges — this is allowed "
                           f"(Direction->Crisis = DCA rule #1 phase transition)",
                "nodes": [nid],
            })

        # Check that target nodes exist
        for tgt in outgoing[nid]:
            if tgt not in node_ids:
                violations.append({
                    "rule": "dangling_edge",
                    "message": f"node '{nid}' has outgoing to '{tgt}' "
                               f"which is not in the proposed node set",
                    "nodes": [nid, tgt],
                })

    # Self-loop check
    for nid, targets in outgoing.items():
        if nid in targets:
            violations.append({
                "rule": "self_loop",
                "message": f"node '{nid}' has a self-loop — forbidden",
                "nodes": [nid],
            })

    # Cycle check (length < 3)
    if existing_edges:
        all_edges = list(existing_edges)
        for src, targets in outgoing.items():
            for tgt in targets:
                all_edges.append((src, "proposed", tgt))
        for src, _, tgt in all_edges:
            # Check if tgt also has edge back to src (length-2 cycle)
            for s2, _, t2 in all_edges:
                if s2 == tgt and t2 == src:
                    violations.append({
                        "rule": "short_cycle",
                        "message": f"length-2 cycle detected: {src} → {tgt} → {src}",
                        "nodes": [src, tgt],
                    })

    valid = len(violations) == 0
    if valid:
        verdict = Verdict.ASSERTED
        reason = "grammar satisfied"
    else:
        verdict = Verdict.CONTESTED
        reason = f"{len(violations)} grammar violation(s)"

    return SpectrumResult(
        values={
            "valid": valid,
            "violations": violations,
            "warnings": warnings,
        },
        verdict=verdict,
        confidence_reason=reason,
    )
```

- [ ] **Step 2: Wire into `branch.simulate()`**

In `spectrum_os/kernel/branch.py`, add an optional `grammar_check` parameter and a validation call at the start of `simulate()`. The grammar gate is opt-in (not all simulations are DCA-bound), but when enabled it rejects grammar-violating adjustments before any computation.

Add parameter to `simulate()` signature:
```python
def simulate(adjustments: dict[str, float], n: int = 10,
             noise_scale: float | None = None,
             grammar_check: dict | None = None) -> dict:
```

And add at the top of the function body, before the sector validation:
```python
    if grammar_check is not None:
        from ..quantum.dca_grammar import validate_alternative
        gresult = validate_alternative(
            grammar_check.get("nodes", []),
            grammar_check.get("existing_edges"),
        )
        if not gresult.values["valid"]:
            raise ValueError(
                f"DCA grammar violation: {gresult.values['violations']}")
```

- [ ] **Step 3: Verify grammar module imports**

Run: `python -c "from spectrum_os.quantum.dca_grammar import validate_alternative, VALID_RELATIONS; print(len(VALID_RELATIONS), 'relations')"`
Expected: `7 relations`

---

## Phase 3: Integration + Anti-Pollution + End-to-End Tests (parallel)

All three tracks in Phase 3 are independent. They CAN run in parallel.

---

### Track 3A: Double-Layer Anti-Pollution Quarantine

### Task 3A.1: Implement double-layer quarantine

**Files:**
- Create: `spectrum_os/quantum/quarantine.py`

**Description:** Implement the anti-pollution double-layer from PLAN-23 §六.五 #2. Layer 1: mechanical blacklist (regex-based detection of anachronistic concepts). Layer 2: second LLM inspector that checks whether generated alternatives unconsciously assume α₁ outcomes. The two layers together form a quarantine pipeline: Layer 1 rejects obvious pollution; Layer 2 inspects what passes Layer 1.

**Can run parallel:** YES (with Tracks 3B, 3C)

**Effort:** 1.5 hr

- [ ] **Step 1: Create `spectrum_os/quantum/quarantine.py`**

```python
"""Double-layer anti-pollution quarantine (PLAN-23 §六.五 #2).

Layer 1 — Mechanical blacklist
    Regex-based detection of anachronistic concepts that did not exist
    before a given cutoff date. Fast, deterministic, catches obvious leaks.

Layer 2 — Second LLM inspector
    A separate LLM call (adversary role: "you are from a different
    historical line") that inspects what passed Layer 1 for unconscious
    α₁ outcome assumptions. Slower and probabilistic, but catches subtle
    contamination.

Usage
-----
>>> from spectrum_os.quantum.quarantine import quarantine
>>> result = quarantine(text, cutoff_year=1910)
>>> if result["passed"]:
...     # safe to use in branch simulation
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Layer 1 — mechanical blacklist
# ---------------------------------------------------------------------------

# Patterns that should not appear before the given cutoff year.
# Format: (pattern, "concept name", first_known_year)
_ANACHRONISM_RULES: list[tuple[str, str, int]] = [
    # WWI concepts — should not appear before 1914
    (r"總體戰", "total_war", 1914),
    (r"total\s*war", "total_war_en", 1914),
    (r"trench\s*warfare", "trench_warfare", 1914),
    (r"塹壕戰", "trench_warfare_zh", 1914),
    (r"坦能堡", "tannenberg", 1914),
    (r"Tannenberg", "tannenberg_en", 1914),
    (r"馬恩河", "marne", 1914),
    (r"Marne", "marne_en", 1914),
    (r"凡爾登", "verdun", 1916),
    (r"Verdun", "verdun_en", 1916),
    (r"索姆河", "somme", 1916),
    (r"Somme", "somme_en", 1916),

    # Russian Revolution concepts — should not appear before 1917
    (r"布爾什維克", "bolshevik", 1917),
    (r"Bolshevik", "bolshevik_en", 1917),
    (r"蘇維埃", "soviet", 1917),
    (r"蘇俄", "soviet_russia", 1917),
    (r"列寧", "lenin", 1917),
    (r"Lenin", "lenin_en", 1917),

    # Post-WWI concepts
    (r"League\s*of\s*Nations", "league_of_nations", 1919),
    (r"國際聯盟", "league_of_nations_zh", 1919),
    (r"Treaty\s*of\s*Versailles", "versailles", 1919),
    (r"凡爾賽條約", "versailles_zh", 1919),
    (r"self-determination", "self_determination", 1918),
    (r"民族自決", "self_determination_zh", 1918),

    # Interwar/post-WWII concepts — should never appear
    (r"blitzkrieg", "blitzkrieg", 1939),
    (r"閃電戰", "blitzkrieg_zh", 1939),
    (r"法西斯", "fascism", 1919),
    (r"fascis[mt]", "fascism_en", 1919),
    (r"納粹", "nazi", 1920),
    (r"Nazi", "nazi_en", 1920),
    (r"cold\s*war", "cold_war", 1945),
    (r"冷戰", "cold_war_zh", 1945),
    (r"nuclear\s*(weapon|deterrent|war)", "nuclear", 1945),
    (r"核(武器|威懾)", "nuclear_zh", 1945),

    # Computational concepts that shouldn't appear in historical context
    (r"algorithm", "algorithm", 1950),
    (r"optimiz", "optimization", 1950),
    (r"feedback\s*loop", "feedback_loop", 1940),
    (r"system\s*dynamics", "system_dynamics", 1950),
]


def _layer1_blacklist(text: str, cutoff_year: int) -> dict:
    """Mechanical first-pass: regex scan for anachronistic concepts."""
    hits: list[dict] = []
    for pattern, concept, first_year in _ANACHRONISM_RULES:
        if first_year <= cutoff_year:
            continue
        for match in re.finditer(pattern, text, re.IGNORECASE):
            hits.append({
                "concept": concept,
                "pattern": pattern,
                "first_known_year": first_year,
                "matched_text": match.group(),
                "span": match.span(),
            })
    return {
        "passed": len(hits) == 0,
        "layer": 1,
        "method": "mechanical_blacklist",
        "cutoff_year": cutoff_year,
        "hits": hits,
        "n_hits": len(hits),
    }


# ---------------------------------------------------------------------------
# Layer 2 — second LLM inspector (adversary role)
# ---------------------------------------------------------------------------

_LAYER2_SYSTEM = (
    "你來自一條不同的歷史線。你的任務：閱讀下面的文本，找出其中"
    "偷偷假設了我們這條線（α₁ 真實歷史）的結局的地方。"
    "具體來說，找出：\n"
    "1. 對未來事件的隱含知識（角色不該知道的事）\n"
    "2. 將 α₁ 結果當作默認前提的推理\n"
    "3. 用後見之明反向正當化的因果鏈\n\n"
    "你只輸出 JSON——直接輸出，無開場白。"
)

_LAYER2_SCHEMA = """\
輸出格式：
{
  "contaminated": <bool, 是否發現污染>,
  "findings": [
    {
      "fragment": "<被污染的文段>",
      "issue": "<為什麼這是污染——具體指出假設了 α₁ 的哪個已知結果>",
      "severity": "<high|medium|low>"
    }
  ],
  "confidence": "<high|medium|low>"
}
"""


def _layer2_inspector(text: str, cutoff_year: int,
                      call_api_fn: Callable,
                      api_key: str,
                      model: str = "deepseek-v4-flash") -> dict:
    """Second LLM pass: adversary inspects for unconscious α₁ leakage."""
    prompt = (
        f"【截止年份】{cutoff_year}\n"
        f"【規則】上述年份之後才發生的事件不得出現在推理中。\n\n"
        f"【待檢疫文本】\n{text}\n\n"
        f"{_LAYER2_SCHEMA}"
    )

    raw = call_api_fn(
        prompt,
        api_key,
        model=model,
        max_tokens=2048,
        temperature=0.0,
        system_message=_LAYER2_SYSTEM,
        response_format={"type": "json_object"},
        thinking=False,
    )

    import json as _json
    try:
        parsed = _json.loads(raw.strip())
    except _json.JSONDecodeError:
        return {
            "passed": True,
            "layer": 2,
            "method": "llm_inspector",
            "error": "JSON parse failed — allowing through with warning",
            "contaminated": None,
            "findings": [],
        }

    return {
        "passed": not parsed.get("contaminated", False),
        "layer": 2,
        "method": "llm_inspector",
        "model": model,
        "contaminated": parsed.get("contaminated", False),
        "findings": parsed.get("findings", []),
        "inspector_confidence": parsed.get("confidence", "unknown"),
    }


# ---------------------------------------------------------------------------
# Public quarantine pipeline
# ---------------------------------------------------------------------------

def quarantine(text: str, cutoff_year: int, *,
               call_api_fn: Callable | None = None,
               api_key: str | None = None,
               enable_layer2: bool = True) -> dict:
    """Run the double-layer quarantine pipeline.

    Parameters
    ----------
    text
        Text to inspect for α₁ contamination.
    cutoff_year
        Concepts that first appeared after this year are anachronistic.
    call_api_fn
        For Layer 2 LLM call. If None, Layer 2 is skipped.
    api_key
        API key for Layer 2. Read from DEEPSEEK_API_KEY env var if None.
    enable_layer2
        If False, skip the LLM inspector (Layer 1 only).

    Returns
    -------
    dict with ``passed`` (bool), ``layer1``, ``layer2``, and
    ``final_verdict`` (one of: clean / layer1_blocked / layer2_flagged).
    """
    import os as _os

    layer1 = _layer1_blacklist(text, cutoff_year)

    if not layer1["passed"]:
        return {
            "passed": False,
            "layer1": layer1,
            "layer2": None,
            "final_verdict": "layer1_blocked",
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    layer2 = None
    if enable_layer2 and call_api_fn is not None:
        if api_key is None:
            api_key = _os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key:
            layer2 = _layer2_inspector(text, cutoff_year, call_api_fn, api_key)
        else:
            layer2 = {
                "passed": True,
                "layer": 2,
                "method": "llm_inspector",
                "error": "no API key — Layer 2 skipped",
                "findings": [],
            }

    passed = layer2["passed"] if layer2 else True
    final_verdict = "clean" if passed else "layer2_flagged"

    return {
        "passed": passed,
        "layer1": layer1,
        "layer2": layer2,
        "final_verdict": final_verdict,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
```

- [ ] **Step 2: Export from `spectrum_os/quantum/__init__.py`**

Add:
```python
from .quarantine import quarantine
```

And add `"quarantine"` to `__all__`.

- [ ] **Step 3: Verify Layer 1 works without API**

Run: `python -c "
from spectrum_os.quantum.quarantine import quarantine
r = quarantine('The blitzkrieg strategy was effective.', 1910, enable_layer2=False)
print('passed:', r['passed'], 'verdict:', r['final_verdict'])
print('hits:', r['layer1']['n_hits'])
"`

Expected: `passed: False`, `verdict: layer1_blocked`, `hits: 1`

- [ ] **Step 4: Verify clean text passes Layer 1**

Run: `python -c "
from spectrum_os.quantum.quarantine import quarantine
r = quarantine('The army marched eastward through the spring mud.', 1910, enable_layer2=False)
print('passed:', r['passed'], 'verdict:', r['final_verdict'])
"`

Expected: `passed: True`, `verdict: clean`

---

### Track 3B: End-to-End 6D Vector Pipeline Test

### Task 3B.1: Write integration test for full 6D pipeline

**Files:**
- Create: `tests/test_vector.py`
- Create: `tests/test_entangle.py`

**Description:** Write tests covering: StateVector creation/normalization, project_ternary from role distributions, vector_from_roles full pipeline, entangle shared-node detection, intervene role reassignment with before/after comparison.

**Can run parallel:** YES (with Tracks 3A, 3C)

**Effort:** 1 hr

- [ ] **Step 1: Create `tests/test_vector.py`**

```python
"""Tests for 6D state vector (PLAN-23 v2.5 vector contract)."""

import math

import numpy as np
import pytest

from spectrum_os.quantum import StateVector, project_ternary, project_spectrum, vector_from_roles
from spectrum_os.quantum.multigraph import ROLES


class TestStateVector:
    def test_creation_defaults(self):
        sv = StateVector(d1=0.5, d2=0.3, d3=-0.2, d4=12.0, d5=0.0, d6=0.05)
        assert sv.d1 == 0.5
        assert sv.d2 == 0.3
        assert sv.d3 == -0.2
        assert sv.d4 == 12.0
        assert sv.d5 == 0.0
        assert sv.d6 == 0.05
        assert sv.meta is None

    def test_ternary_property(self):
        sv = StateVector(d1=0.8, d2=0.1, d3=-0.5, d4=6.0, d5=1.0, d6=0.02)
        t = sv.ternary
        assert t.shape == (3,)
        assert t[0] == 0.8
        assert t[1] == 0.1
        assert t[2] == -0.5

    def test_spectrum_property(self):
        sv = StateVector(d1=0.0, d2=0.0, d3=0.0, d4=24.0, d5=math.pi, d6=0.1)
        s = sv.spectrum
        assert s.shape == (3,)
        assert s[0] == 24.0
        assert s[1] == math.pi
        assert s[2] == 0.1

    def test_full_property(self):
        sv = StateVector(d1=0.1, d2=0.2, d3=0.3, d4=1.0, d5=2.0, d6=3.0)
        f = sv.full
        assert f.shape == (6,)
        assert list(f) == [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]

    def test_norm(self):
        sv = StateVector(d1=3.0, d2=0.0, d3=0.0, d4=4.0, d5=0.0, d6=0.0)
        assert sv.norm() == pytest.approx(5.0)

    def test_roundtrip_dict(self):
        sv = StateVector(d1=0.5, d2=0.3, d3=-0.2, d4=12.0, d5=1.5, d6=0.05,
                         meta={"source": "test"})
        d = sv.to_dict()
        sv2 = StateVector.from_dict(d)
        assert sv2.d1 == sv.d1
        assert sv2.d2 == sv.d2
        assert sv2.meta == {"source": "test"}


class TestProjectTernary:
    def test_pure_direction(self):
        d1, d2, d3 = project_ternary({"direction": 1.0, "crisis": 0.0,
                                       "lag": 0.0, "alternative": 0.0})
        assert d1 == pytest.approx(1.0)
        assert d2 == pytest.approx(0.0)
        assert d3 == pytest.approx(-0.333, abs=0.01)

    def test_pure_crisis(self):
        d1, d2, d3 = project_ternary({"direction": 0.0, "crisis": 1.0,
                                       "lag": 0.0, "alternative": 0.0})
        assert d1 == pytest.approx(-1.0)
        assert d2 == pytest.approx(0.0)

    def test_equal_mix(self):
        d1, d2, d3 = project_ternary({"direction": 0.25, "crisis": 0.25,
                                       "lag": 0.25, "alternative": 0.25})
        assert d1 == pytest.approx(0.0)
        assert d2 == pytest.approx(0.25)
        assert d3 == pytest.approx(0.0)

    def test_empty_distribution(self):
        d1, d2, d3 = project_ternary({})
        assert d1 == 0.0
        assert d2 == 0.0
        assert d3 == 0.0

    def test_lag_dominant(self):
        d1, d2, d3 = project_ternary({"direction": 0.0, "crisis": 0.0,
                                       "lag": 1.0, "alternative": 0.0})
        assert d2 == pytest.approx(1.0)
        assert d1 == pytest.approx(0.0)


class TestProjectSpectrum:
    def test_with_decompose_result(self):
        fake_result = {
            "values": {
                "dominant_periods": [12],
                "deviation": np.array([0.1, -0.2, 0.15, -0.1, 0.05, 0.0]),
            }
        }
        d4, d5, d6 = project_spectrum(decompose_result=fake_result)
        assert d4 == 12.0
        assert d5 != 0.0  # should find peak phase
        assert d6 > 0.0    # std of deviation

    def test_manual_values(self):
        d4, d5, d6 = project_spectrum(dominant_period=6.0, phase_angle=1.5, amplitude=0.03)
        assert d4 == 6.0
        assert d5 == 1.5
        assert d6 == 0.03


class TestVectorFromRoles:
    def test_full_pipeline(self):
        roles = {"direction": 0.5, "crisis": 0.2, "lag": 0.1, "alternative": 0.2}
        sv = vector_from_roles(roles, dominant_period=6.0, phase_angle=0.0, amplitude=0.02)
        assert isinstance(sv, StateVector)
        # direction > crisis → d1 > 0
        assert sv.d1 > 0
        assert 0 <= sv.d2 <= 1
        assert sv.d4 == 6.0
```

- [ ] **Step 2: Create `tests/test_entangle.py`**

```python
"""Tests for quantum.entangle and quantum.intervene (PLAN-23 §6.3)."""

import pytest

from spectrum_os.kernel._types import Verdict
from spectrum_os.quantum import entangle, intervene
from spectrum_os.quantum.multigraph import ROLES, RoleMultigraph


@pytest.fixture
def tiny_graph():
    """Same setup as test_quantum.py — 2 threads sharing node S."""
    graph = RoleMultigraph()
    graph.add_edge("S", "A", "alternative->direction", weight=0.9)
    graph.add_edge("B", "S", "direction->crisis", weight=0.9)
    return graph


class TestEntangle:
    def test_entangled_state(self, tiny_graph):
        result = entangle(tiny_graph, "S")
        assert result.verdict is Verdict.ASSERTED
        assert result.values["entangled"] is True
        assert result.values["n_threads"] >= 1

    def test_isolated_state(self, tiny_graph):
        result = entangle(tiny_graph, "A")
        assert result.values["entangled"] is False

    def test_unknown_state(self, tiny_graph):
        result = entangle(tiny_graph, "nonexistent")
        assert result.verdict is Verdict.UNKNOWN
        assert result.values["n_threads"] == 0


class TestIntervene:
    def test_role_reassignment(self, tiny_graph):
        before_threads = entangle(tiny_graph, "S").values["n_threads"]
        result = intervene(tiny_graph, "S", list(tiny_graph.assignments)[0].thread,
                          "direction")
        assert result.verdict is Verdict.ASSERTED
        assert result.values["old_role"] is not None
        assert result.values["new_role"] == "direction"

    def test_intervene_invalid_role(self, tiny_graph):
        with pytest.raises(ValueError, match="new_role must be one of"):
            intervene(tiny_graph, "S", "fake_thread", "not_a_role")

    def test_intervene_unknown_state(self, tiny_graph):
        result = intervene(tiny_graph, "nonexistent", "any_thread", "crisis")
        assert result.verdict is Verdict.UNKNOWN
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_vector.py tests/test_entangle.py -v`
Expected: ALL PASS (approximately 16 tests)

---

### Track 3C: Create test files for quarantine + grammar + verify

### Task 3C.1: Write quarantine and grammar tests

**Files:**
- Create: `tests/test_quarantine.py`
- Create: `tests/test_dca_grammar.py`

**Description:** Test Layer 1 blacklist with known anachronisms, clean text pass-through, grammar validation with valid and invalid alternatives.

**Can run parallel:** YES (with Tracks 3A, 3B)

**Effort:** 45 min

- [ ] **Step 1: Create `tests/test_quarantine.py`**

```python
"""Tests for double-layer anti-pollution quarantine (PLAN-23 §六.五 #2)."""

import pytest

from spectrum_os.quantum.quarantine import quarantine, _layer1_blacklist


class TestLayer1Blacklist:
    def test_blocks_blitzkrieg_before_1939(self):
        r = _layer1_blacklist("The blitzkrieg through Belgium was swift.", 1910)
        assert r["passed"] is False
        assert r["n_hits"] >= 1

    def test_blocks_total_war_before_1914(self):
        r = _layer1_blacklist("They prepared for total war.", 1910)
        assert r["passed"] is False
        assert any(h["concept"] in ("total_war", "total_war_en") for h in r["hits"])

    def test_blocks_bolshevik_before_1917(self):
        r = _layer1_blacklist("The Bolshevik faction gained influence.", 1910)
        assert r["passed"] is False
        assert any("bolshevik" in h["concept"] for h in r["hits"])

    def test_allows_clean_text(self):
        r = _layer1_blacklist("The army crossed the river at dawn.", 1910)
        assert r["passed"] is True
        assert r["n_hits"] == 0

    def test_allows_concept_within_cutoff(self):
        r = _layer1_blacklist("The Bolshevik faction gained influence.", 1920)
        assert r["passed"] is True


class TestQuarantinePipeline:
    def test_layer1_block_disables_layer2(self):
        r = quarantine("blitzkrieg tactics", 1910, enable_layer2=False)
        assert r["passed"] is False
        assert r["final_verdict"] == "layer1_blocked"
        assert r["layer2"] is None

    def test_clean_text_without_layer2(self):
        r = quarantine("The harvest was poor that autumn.", 1910, enable_layer2=False)
        assert r["passed"] is True
        assert r["final_verdict"] == "clean"
```

- [ ] **Step 2: Create `tests/test_dca_grammar.py`**

```python
"""Tests for DCA grammar gate (PLAN-23 §五)."""

import pytest

from spectrum_os.kernel._types import Verdict
from spectrum_os.quantum.dca_grammar import (
    VALID_RELATIONS,
    validate_alternative,
    validate_relation,
)


class TestValidateRelation:
    def test_valid_relations(self):
        for rel in VALID_RELATIONS:
            assert validate_relation(rel) is None

    def test_invalid_relation(self):
        v = validate_relation("crisis->crisis")
        assert v is not None
        assert v.rule == "invalid_relation"


class TestValidateAlternative:
    def test_valid_minimal_alternative(self):
        nodes = [
            {"node_id": "C1", "role": "crisis", "outgoing": ["A1"]},
            {"node_id": "A1", "role": "alternative", "outgoing": ["D1"]},
            {"node_id": "D1", "role": "direction", "outgoing": []},
        ]
        r = validate_alternative(nodes)
        assert r.values["valid"] is True
        assert r.verdict is Verdict.ASSERTED

    def test_crisis_missing_outgoing(self):
        nodes = [
            {"node_id": "C1", "role": "crisis", "outgoing": []},
        ]
        r = validate_alternative(nodes)
        assert r.values["valid"] is False
        assert any(v["rule"] == "insufficient_outgoing" for v in r.values["violations"])

    def test_self_loop_forbidden(self):
        nodes = [
            {"node_id": "X", "role": "crisis", "outgoing": ["X"]},
        ]
        r = validate_alternative(nodes)
        assert r.values["valid"] is False
        assert any(v["rule"] == "self_loop" for v in r.values["violations"])

    def test_dangling_edge(self):
        nodes = [
            {"node_id": "C1", "role": "crisis", "outgoing": ["GHOST"]},
        ]
        r = validate_alternative(nodes)
        assert r.values["valid"] is False
        assert any(v["rule"] == "dangling_edge" for v in r.values["violations"])

    def test_direction_terminal_warning(self):
        nodes = [
            {"node_id": "D1", "role": "direction", "outgoing": ["C2"]},
            {"node_id": "C2", "role": "crisis", "outgoing": []},
        ]
        r = validate_alternative(nodes)
        # direction->crisis is valid (DCA rule #1), just warns
        assert any(w["rule"] == "terminal_has_outgoing" for w in r.values["warnings"])

    def test_short_cycle_detection(self):
        nodes = [
            {"node_id": "C1", "role": "crisis", "outgoing": ["A1"]},
            {"node_id": "A1", "role": "alternative", "outgoing": ["C1"]},
        ]
        # A1→C1 forms a 2-cycle with existing edge C1→A1
        existing = [("C1", "crisis->alternative", "A1")]
        r = validate_alternative(nodes, existing_edges=existing)
        assert r.values["valid"] is False
        assert any(v["rule"] == "short_cycle" for v in r.values["violations"])
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_quarantine.py tests/test_dca_grammar.py -v`
Expected: ALL PASS (approximately 12 tests)

---

## Final Verification (after all Phase 3 tracks complete)

### Task V.1: Run full test suite

- [ ] **Step 1: Run all spectrum-os tests**

```bash
cd /home/octy/projects/spectrum-os && python -m pytest tests/ -v
```

Expected: ALL PASS (53 existing + ~35 new = ~88 tests)

- [ ] **Step 2: Verify no import regressions**

```bash
python -c "
from spectrum_os.kernel import osc
from spectrum_os.quantum import (
    StateVector, project_ternary, project_spectrum, vector_from_roles,
    entangle, intervene, quarantine, tomography, decoherence_watch
)
from spectrum_os.synth import anchors, expand
from spectrum_os.quantum.dca_grammar import validate_alternative, VALID_RELATIONS
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 3: Verify OSC facade still works**

```bash
python -c "
from spectrum_os.kernel import osc
# Existing ops
print('sectors:', osc.sectors)
print('wave:', osc.wave)
print('branch:', osc.branch)
print('verify:', osc.verify)
print('quantum:', osc.quantum)
print('All OSC properties OK')
"
```

Expected: All properties resolve without error

---

## Dependency Graph

```
Phase 1 (sequential)
  Task 1.1: vector.py
      │
      ▼
Phase 2 (parallel cluster)
  ├── Track 2A: entangle.py
  ├── Track 2B: anchors.py prompt upgrade
  ├── Track 2C: verify.py query/summary
  └── Track 2D: dca_grammar.py + branch.py
      │
      ▼
Phase 3 (parallel cluster)
  ├── Track 3A: quarantine.py
  ├── Track 3B: test_vector.py + test_entangle.py
  └── Track 3C: test_quarantine.py + test_dca_grammar.py
      │
      ▼
  Final Verification: full test suite
```

---

## Summary

| Phase | Tracks | Files Created | Files Modified | Est. Time |
|:---|:---:|:---|:---|:---:|
| Phase 1 | 1 | 1 (`vector.py`) | 1 (`quantum/__init__.py`) | 30 min |
| Phase 2 | 4 parallel | 2 (`entangle.py`, `dca_grammar.py`) | 2 (`anchors.py`, `verify.py`) | 1.5 hr (parallel) |
| Phase 3 | 3 parallel | 4 (`quarantine.py`, `test_vector.py`, `test_entangle.py`, `test_quarantine.py`, `test_dca_grammar.py`) | 0 | 1.5 hr (parallel) |
| Verify | 1 | 0 | 0 | 15 min |

**Total estimated wall-clock time: ~3.5 hours** (with full parallelism in Phases 2 and 3).
