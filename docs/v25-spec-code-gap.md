# Spectrum OS v2.5 — PLAN-23 spec vs code gap

> **Date**: 2026-07-29  
> **Status**: documented, no action required for v2.5  
> **Source**: PLAN-23 v2.5 ("向量契約入憲") vs `/home/octy/projects/spectrum-os/` implementation

## 1. D1–D3: continuous vs discrete ternary

| Aspect | Plan | Code |
|:---|---:|:---|
| D1 formula | `d - c` (float) | `sign(direction - crisis)` (int) |
| D2 formula | `l` (float) | `sign(lag - 0.25)` (int) |
| D3 formula | `a - mean(d,c,l)/3` (float) | `sign(alternative - mean(...))` (int) |
| Range | `[-1, +1]` continuous | `{-1, 0, +1}` discrete |

**Decision**: Discrete is *more honest* for a "ternary direction" concept — continuous values here imply a precision that the 4-role projection doesn't warrant. The code is internally consistent (docstring matches behavior). **No change needed.** Plan should update to match code.

## 2. Missing `project_spectrum()` standalone function

Plan specified a standalone `project_spectrum(decompose_result) -> (d4,d5,d6)` to extract spectrum content from wave decomposition results. Code handles this inline in `vector_from_roles()` via a simpler `dict[str, float]` interface (`freq`, `phase`, `amplitude` keys).

**Decision**: The dict interface is simpler and avoids coupling `vector.py` to `kernel/wave.py`. **No change needed.**

## 3. Missing `StateVector.ternary` / `.spectrum` / `.full` numpy-array properties

Plan specified properties returning `np.array([d1,d2,d3])`, `np.array([d4,d5,d6])`, `np.array([...all 6...])`. Code omits them — they'd add a numpy dependency to `vector.py` which was designed to stay numpy-free (only `math`). Callers can use `to_dict()` and convert externally.

**Decision**: v2.5 stays numpy-free in `vector.py`. If a future version needs these, add them in `kernel/` or a bridge module. **No change for now.**

## 4. Branch DCA grammar validation not wired

Plan specified: "Add DCA grammar validation step in `simulate()` — each adjustment mutation must pass `validate_alternative()` before being accepted." Code in `branch.py` has no DCA grammar import or validation. This is a real gap: branch simulations can produce role transitions that violate DCA grammar rules.

**Decision**: Deferred to v2.6 — requires non-trivial refactoring of `branch.py`'s simulation loop to inject grammar checks. **Fix in v2.6.**

## 5. "Second LLM inspector" is mechanical fallback only

Plan described `inspector_gate()` as a second LLM call that cross-validates the first LLM's output. Code is honest: `inspector_gate` is a pure mechanical filter (TEMPORAL_BLACKLIST + emergence-year check), and the code labels it "mechanical fallback (Phase 3 scope)" in its docstring.

**Decision**: Full LLM cross-validation is a v3.0 feature — requires a second API call with different prompt, cost implications, and latency concerns. Mechanical fallback is adequate for v2.5. **No change for now.**

---

## Summary

| Gap | Severity | Action |
|:---|:---:|:---|
| D1–D3 discrete vs continuous | Low | Update plan to match code (or vice versa) |
| Missing `project_spectrum()` | None | Covered by `vector_from_roles` dict interface |
| Missing numpy array properties | Low | v3.0 if needed; stay numpy-free for now |
| Branch DCA validation not wired | **Medium** | v2.6 |
| Second LLM inspector not implemented | Low | v3.0 |

All five gaps are **conscious, documented divergences** — not bugs. The implementation is internally consistent with its own docstrings.
