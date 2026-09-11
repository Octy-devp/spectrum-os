"""Spectrum OS Kernel — R/C counterforce field engine (social dynamics first theorem).

Methodology: ECC ``docs/method/force-model-social-dynamics.md`` (v1.3, 2026-08-04).
Constitutional-level, worldline-agnostic spec: any society's "content" is not a
statistically pre-measurable label — it is the resultant vector S(t) of its
internal transformative force R (revolution) vs inertial force C (counter-
revolution), run to its deterministic end.

Core equations (§2.2 + §2.2b — the equations are the SSOT, see sign convention):

    dR_i/dt = a_i·R_i − β_i·C_i + λ_i·P_R,i − μ_i·R_i   (+ coupling, §2.6)
    dP_R,i/dt = β_i·C_i + μ_i·R_i − λ_i·P_R,i − ρ_i·P_R,i
    dC_i/dt = c_i·C_i − α_i·R_i                         (+ coupling, §2.6)

    S_i(t) = R_i / (R_i + C_i) ∈ [0, 1]   (social content = resultant, §2.3)
    T_i(t) = R_i + P_R,i                  (total transformative capacity, §2.2b)
    tension_i(t) = f(R_i · C_i)           (conflict intensity, §2.4)

P_R potential reservoir (§2.2b): high-pressure repression does NOT annihilate R —
it compresses kinetic R into latent potential P_R (μ), which an external-field
modulation or C's internal cracks can release back into R (λ); ρ is what is
truly forgotten. This is why revolution is never mathematically killed
(1905 crushed → 1917 erupts): the Lanchester "conservation" applies to
T = R + P_R, not to R alone.

External field injection (§2.5 — the world-value law is a dynamic participant,
not a boundary condition):

    α_i(t) = α_i⁰ · mod_α(t)     β_i(t) = β_i⁰ · mod_β(t)

Four-quadrant full-sign coupling (§2.6 — κ must emerge from actor cards'
``field_coordinates``, never hard-fixed):

    dR_i/dt += Σ_{j≠i} [ κ^RR_ij·(R_j − R_i) + κ^CR_ij·(C_j − C_i) ]
    dC_i/dt += Σ_{j≠i} [ κ^CC_ij·(C_j − C_i) + κ^RC_ij·(R_j − R_i) ]

    κ^RR — solidarity diffusion (R_j rises ⇒ R_i rises)
    κ^CC — coordination diffusion (C_j rises ⇒ C_i rises)
    κ^RC — heterogeneous panic (R_j rises ⇒ C_i counterattacks — the §2.6
           "panicky exponential counter-reaction"; optionally nonlinear via
           ``panic_exponent``: the term is weighted by the source R_j^p)
    κ^CR — C_j rises ⇒ R_i suppressed/activated (regime-dependent sign)

    NOTE on cross-label naming: the spec's equation block transposes the two
    cross terms relative to its own semantics bullets. This module follows the
    semantics + executable test contract (κ^RC drives C from R — "R rise → C
    counterattack"), which is the consistent reading across the §2.6 text, the
    implementation brief, and the panic-spiral test.

Sign convention (read off §2.2 equations — this module follows them as written):
    β_i — C's suppression efficiency against R   (raising β ⇒ R loses)
    α_i — R's suppression efficiency against C   (raising α ⇒ C loses)

Consequence for the Hungary-1919 reproduction: R initially seizes power (S high);
the external field (Entente blockade, Romanian invasion) strengthens the counter-
revolution's hand ⇒ the field must RAISE β and LOWER α for S to flip to C. The
naive opposite modulation (α↑/β↓) mathematically favours R and cannot flip —
guarded by a dedicated test.

Epistemology (v1.3 §3.2–§3.3):
- Parameters (α, β, a, c, κ, μ, λ, ρ) are SEMANTIC ANCHORS, not physical
  constants: sign and relative magnitude carry the judgement; absolute values
  are neither available nor needed. The engine never fits parameters.
- LLM = data source, not judge: this core is a deterministic numpy-only ODE
  engine with zero LLM dependency — conclusions come from the mathematical
  structure of the equations, not from any model's verdict.
- Conservation = H₀ for testing: T = R + P_R is emitted as an observable and
  never enforced as a constraint (§3.2). Analysts falsify conservation on data.

Deterministic numpy-only core — no LLM, no scipy. RK4 default, Euler optional.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Degenerate guard for S = R/(R+C): when both forces vanish the "content" is
# undecided (a dead/zero-tension state), pinned to 0.5 rather than 0/0 → NaN.
_S_EPS = 1e-12

_SUPPORTED_METHODS = ("rk4", "euler")

# Fast-channel invariants / tolerances.
_FAST_T_EPS = 1e-9       # time tolerance for seasonal operator-split boundaries
_PHI_MIN = 0.0           # Φ is a conductance share ∈ [0, 1]
_PHI_MAX = 1.0

# Provenance tags for every fast-channel parameter (v5.0 ECC discipline — a
# parameter without a tag is a fabricated number). See THEORY-LEDGER §「完整
# 動態方程組」 (2026-09-08 agy):
#   CALIBRATED       — anchored to measured / canon data
#   EXPERIMENTAL     — declared target value, awaiting falsification
#   DERIVED_PROXY    — derived from other quantities; no direct measurement
#   SCENARIO         — a documented scenario choice, not a measurement
_FAST_PROVENANCE: dict[str, str] = {
    "phi0": "SCENARIO",
    "co_share": "MEASURED",
    "debt_stress": "DERIVED_PROXY",
    "kappa_phi": "EXPERIMENTAL",
    "zeta_phi": "DERIVED_PROXY",
    "delta": "DERIVED_PROXY",
    "mu_f": "EXPERIMENTAL",
    "eta": "SCENARIO",
    "lam": "CALIBRATED",
    "s_in": "DERIVED_PROXY",
    "mu_e": "DERIVED_PROXY",
    "kappa_c": "DERIVED_PROXY",
    "gamma_c": "DERIVED_PROXY",
    "alpha_rc": "[THEORY-LEDGER minimal patch 2026-09-10, UNCALIBRATED]",
    "sigma_soviet": "SCENARIO",
    "k0": "SCENARIO",
    "gamma_k": "DERIVED_PROXY",
    "clamp_spring": "EXPERIMENTAL",
    "autumn_yield": "EXPERIMENTAL",
    "reflow_purity": "DERIVED_PROXY",
    "surplus_gain": "DERIVED_PROXY",
    "year_period": "SCENARIO",
    "spring_phase": "DERIVED_PROXY",
    "autumn_phase": "DERIVED_PROXY",
    "mu_cef": "[TWO-MODE DIGESTION 2026-09-10, SCENARIO]",
    "mu_sri": "[TWO-MODE DIGESTION 2026-09-10, SCENARIO]",
    "rho": "[TWO-MODE DIGESTION 2026-09-10, SCENARIO]",
    "t_c_phi": "[DIMENSIONAL_OPERATOR 2026-09-10, CALIBRATED]",
    "t_c_p": "[DIMENSIONAL_OPERATOR 2026-09-10, CALIBRATED]",
    "t_c_k": "[DIMENSIONAL_OPERATOR 2026-09-10, CALIBRATED]",
    "m_half": "[MORPHOLOGY 2026-09-10, CALIBRATED]",
    "mu_sri_to_dist": "[MORPHOLOGY 2026-09-10, SCENARIO]",
    "k_dissolve": "[MORPHOLOGY 2026-09-10, SCENARIO]",
    "m0": "[MORPHOLOGY 2026-09-10, SCENARIO]",
    "p_pool0": "[MORPHOLOGY 2026-09-10, SCENARIO]",
    "p_dist0": "[MORPHOLOGY 2026-09-10, SCENARIO]",
}


def _as_float_array(value: Any, name: str) -> np.ndarray:
    """Coerce a scalar/array-like into a float64 1-D array (never scalar)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or 1-D, got shape {arr.shape}")
    return arr


def _broadcast(value: Any, n: int, name: str) -> np.ndarray:
    """Broadcast a scalar or length-n array-like to a length-n float array."""
    arr = _as_float_array(value, name)
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=np.float64)
    if arr.size != n:
        raise ValueError(f"{name} length {arr.size} != n_nodes {n}")
    return arr


def _apply_multiplier(fn: Callable | None, *args: np.ndarray) -> np.ndarray:
    """Per-node non-negative feedback multiplier (``None`` ≡ constant 1).

    Mirrors the legacy inline ``_mult`` closure; kept module-level so the opt-in
    5-D RHS can share it without touching the legacy code path.
    """
    if fn is None:
        return np.ones_like(args[0])
    m = np.asarray(fn(*args), dtype=np.float64)
    if m.ndim == 0:
        m = np.full(args[0].shape, float(m))
    return np.maximum(np.broadcast_to(m, args[0].shape), 0.0)


@dataclass
class ForceTrajectory:
    """Result of an R/C field integration run.

    Arrays are indexed ``[step, node]``: ``R``/``C``/``P``/``S``/``T``/``tension``
    have shape ``(n_steps, n_nodes)``; ``t`` has shape ``(n_steps,)``.

    Field semantics (v1.3 — see module docstring):
    - ``R`` / ``C`` / ``P`` — kinetic revolutionary force, kinetic inertial
      force, latent (potential) revolutionary force P_R (§2.2b).
    - ``S`` — resultant S = R/(R+C) ∈ [0, 1]; P_R does not enter directly
      (it acts only through λ-release back into R).
    - ``T`` — total transformative capacity T = R + P_R: the §3.2 conservation
      observable (H₀), NOT a hard constraint. T may grow or decay across the
      run (e.g. new technology raising total transformative capacity) — the
      engine only makes it observable so analysts can falsify conservation.
    - ``tension`` — §2.4 conflict intensity f(R·C) (the v1.0 ``T`` field,
      renamed for clarity).
    """

    t: np.ndarray
    R: np.ndarray
    C: np.ndarray
    P: np.ndarray
    S: np.ndarray
    T: np.ndarray
    tension: np.ndarray
    meta: dict = field(default_factory=dict)
    # Opt-in 5-D fast-channel history. ``None`` on the legacy 2/3-D path so the
    # legacy trajectory is byte-for-byte unchanged (no extra keys in to_dict).
    phi: np.ndarray | None = None
    k_pool: np.ndarray | None = None
    # Opt-in P_R morphology breakdown & cumulative socialization (THEORY-LEDGER 2026-09-10).
    p_pool: np.ndarray | None = None
    p_dist: np.ndarray | None = None
    m_cum: np.ndarray | None = None
    g_sat: np.ndarray | None = None

    def to_dict(self) -> dict:
        """Serialize to plain JSON-safe dict (numpy arrays → nested lists).

        The ``phi`` / ``k_pool`` / morphology keys are emitted ONLY when the
        fast-channel mode produced them — legacy serialization stays bit-identical.
        """
        payload = {
            "t": self.t.tolist(),
            "R": self.R.tolist(),
            "C": self.C.tolist(),
            "P": self.P.tolist(),
            "S": self.S.tolist(),
            "T": self.T.tolist(),
            "tension": self.tension.tolist(),
            "meta": self.meta,
        }
        if self.phi is not None:
            payload["phi"] = self.phi.tolist()
        if self.k_pool is not None:
            payload["k_pool"] = self.k_pool.tolist()
        if self.p_pool is not None:
            payload["p_pool"] = self.p_pool.tolist()
        if self.p_dist is not None:
            payload["p_dist"] = self.p_dist.tolist()
        if self.m_cum is not None:
            payload["m_cum"] = self.m_cum.tolist()
        if self.g_sat is not None:
            payload["g_sat"] = self.g_sat.tolist()
        return payload

    def final_s(self) -> float:
        """Resultant S at the last step (single-node convenience: ``float``)."""
        if self.S.ndim == 1:
            return float(self.S[-1])
        return float(self.S[-1, 0])

    def final_phi(self) -> float | None:
        """Φ at the last step, or ``None`` on the legacy path."""
        if self.phi is None:
            return None
        if self.phi.ndim == 1:
            return float(self.phi[-1])
        return float(self.phi[-1, 0])


@dataclass(frozen=True)
class FastChannelConfig:
    """Opt-in parameters for the 5-D fast/slow state vector.

    Source of truth: ECC THEORY-LEDGER §「完整動態方程組」 (2026-09-08 agy 裁定). This is NOT a redesign — it is the
    ledger's ODE system, operationalised. The state vector becomes

        x(t) = (R, C, P_R, Φ, K) ∈ ℝ⁵

    with the ledger's ODE block (2/3-D is exactly the legacy engine; this block
    is only integrated when :class:`ForceFieldDynamics` is built with
    ``fast_channel=<this>``):

        dR/dt   = a·R − δ·R + Φ·μ_F·R + (1−η)·λ·P_R − β·C
        dC/dt   = κ_C·(1−Φ) − α_rc·R·C − γ_C·σ_soviet·C
        dP_R/dt = S_in·R − λ·P_R − η·μ_E·P_R
        dΦ/dt   = κ_Φ·CoopShare·(1−Φ) − ζ_Φ·DebtStress·Φ
        dK/dt   = η·μ_E·P_R − γ_K·K

    Notes / interpretations (declared, not silent):
    - ``a`` and ``β`` are taken from the base engine (``growth_r`` / ``beta``),
      so the §2.5 external-field modulation still applies to β. ``β``'s friction
      term is ``β·C`` (the ledger's ``βCR`` read through the module's §2.2 sign
      convention and the ledger's own stability criterion ``βC*``).
    - Only the fast/slow channels above are integrated in this mode; the legacy
      4-quadrant κ coupling / substrate feedback fns are a legacy-path feature.
    - Φ is a conductance share ∈ [0, 1] and is clamped to that range.

    Seasonal operator splitting (ledger's 混合算子分裂): the continuous ODE is
    interrupted at ``year_period·spring_phase`` (春耕鉗) and
    ``year_period·autumn_phase`` (秋收兌現):

        spring: R ← R · (1 − (1−Φ)·Clamp_spring)
        autumn: R ← R + Yield·Φ·ReflowPurity ;  P_R ← P_R + SurplusGain

    Every parameter below is tagged with a provenance label (see
    :meth:`provenance`); untagged numbers are forbidden by ECC discipline.
    """

    # ── Φ fast variable ────────────────────────────────────────────────
    phi0: Any = 0.65                 # [SCENARIO] initial circulation conductivity
    co_share: Any = 0.65             # [MEASURED] CoopShare driver (rlo/coop share)
    debt_stress: Any = 0.5           # [DERIVED_PROXY] DebtStress back-pressure
    kappa_phi: Any = 1.5             # [EXPERIMENTAL] logistic drive κ_Φ (ledger)
    zeta_phi: Any = 1.0              # [DERIVED_PROXY] usury back-pressure ζ_Φ
    # ── R equation ─────────────────────────────────────────────────────
    delta: Any = 0.0                 # [DERIVED_PROXY] δ — autonomous R decay
    mu_f: Any = 0.12                 # [EXPERIMENTAL] Φ monetisation μ_F (ledger)
    eta: Any = 0.0                   # [SCENARIO] alienation extraction η ∈ [0,1]
    lam: Any = 0.15                  # [CALIBRATED] λ — P_R release/decay (ledger)
    # ── P_R / K equations ──────────────────────────────────────────────
    s_in: Any = 0.0                  # [DERIVED_PROXY] S_in — R → P_R synthesis
    mu_e: Any = 0.0                  # [DERIVED_PROXY] μ_E — P_R → K extraction
    k0: Any = 0.0                    # [SCENARIO] initial alienated-capital pool
    gamma_k: Any = 0.0               # [DERIVED_PROXY] γ_K — K depreciation
    # ── C equation ─────────────────────────────────────────────────────
    kappa_c: Any = 0.0               # [DERIVED_PROXY] κ_C — friction source (1−Φ)
    gamma_c: Any = 0.0               # [DERIVED_PROXY] γ_C — friction dissipation
    # α_rc — contact-surface suppression (R vs C): the minimal patch restores
    # the force-model v1.6 "R suppresses C" channel (dC/dt = cC − αR) inside the
    # ledger's C equation as the bilinear −α_rc·R·C. Bilinear (not linear −αR)
    # keeps C ≥ 0 a repelling boundary of the flow (no constant forcing from R),
    # and matches the "RLO = the organ that eliminates AE" adjudication. Default
    # 0.1: (i) at the engine's operating point R ~ O(1) the added dissipation
    # α_rc·R ≈ γ_C·σ_soviet = 0.10 — a same-order, non-dominant correction
    # (perturbative until calibrated); (ii) matching v1.6's linear −α·R at the
    # friction-neutral reference C_ref ≈ 0.4 with the almanac α = 0.05 gives
    # α_rc = α/C_ref ≈ 0.12 — the same O(0.1). UNCALIBRATED: calibrate against
    # RLO-elimination events before treating the value as measured.
    alpha_rc: Any = 0.1              # [UNCALIBRATED] α_rc — R·C contact-surface suppression
    sigma_soviet: Any = 1.0          # [SCENARIO] σ_soviet — soviet effectiveness
    # ── mixed operator splitting (seasonal jumps) ──────────────────────
    year_period: float = 1.0         # [SCENARIO] seasonal period (years)
    spring_phase: float = 0.25       # [DERIVED_PROXY] 春耕鉗 phase within the year
    autumn_phase: float = 0.75       # [DERIVED_PROXY] 秋收兌現 phase
    clamp_spring: Any = 0.0          # [EXPERIMENTAL] Clamp_spring coefficient
    autumn_yield: Any = 0.0          # [EXPERIMENTAL] Yield (autumn injection)
    reflow_purity: Any = 1.0         # [DERIVED_PROXY] ReflowPurity multiplier
    surplus_gain: Any = 0.0          # [DERIVED_PROXY] SurplusGain → P_R
    # ── two-mode digestion coupling (THEORY-LEDGER 2026-09-10) ────────
    # Partition of C's net dissipation flow [-C_dot]_+ = max(0, -dC/dt):
    #   dPhi/dt += T_{C->Phi} * mu_cef * [-C_dot]_+   (fast channel)
    #   dP_R/dt += T_{C->P}   * mu_sri * [-C_dot]_+   (slow channel)
    #   dK/dt   += T_{C->K}   * rho    * [-C_dot]_+   (granulation / alienated sediment)
    # Conservation: mu_cef + mu_sri + rho == 1.0.
    # Dimensional operators:
    #   T_{C->Phi}: [Phi]/[C] = 1/[C] (dimension: 1/capacity; converts eliminated friction to circulation conductivity)
    #   T_{C->P}  : [P_R]/[C] = 1     (dimensionless; capacity-to-capacity potential transfer)
    #   T_{C->K}  : [K]/[C]   = 1     (dimensionless; capacity-to-alienation loss)
    mu_cef: Any = 0.0                # [TWO-MODE DIGESTION 2026-09-10, SCENARIO] fast channel share
    mu_sri: Any = 0.0                # [TWO-MODE DIGESTION 2026-09-10, SCENARIO] slow channel share
    rho: Any = None                  # [TWO-MODE DIGESTION 2026-09-10, SCENARIO] granulation loss share (default: 1 - mu_cef - mu_sri)
    t_c_phi: Any = 1.0               # [DIMENSIONAL_OPERATOR 2026-09-10, CALIBRATED] unit: 1/[C]
    t_c_p: Any = 1.0                 # [DIMENSIONAL_OPERATOR 2026-09-10, CALIBRATED] unit: [P]/[C] = 1
    t_c_k: Any = 1.0                 # [DIMENSIONAL_OPERATOR 2026-09-10, CALIBRATED] unit: [K]/[C] = 1
    # ── P_R morphology breakdown (THEORY-LEDGER 2026-09-10) ───────────
    # P_R = P_pool + P_dist:
    #   P_pool: centralized/concentrated reserve, subject to administrative mobilization (lambda)
    #           and bureaucratic extraction (eta * mu_e).
    #   P_dist: distributed capacity (literacy, mutual aid), immune to capture/mobilization,
    #           works directly to maintain social reproduction.
    #   g(M) = M / (M + m_half): saturating morphology transition factor driven by cumulative M(t).
    #   J_dissolve = k_dissolve * g(M) * P_pool: autonomous dissolution of pool into distributed form.
    #   J_sri = T_{C->P} * mu_sri * [-C_dot]_+ is partitioned:
    #     J_sri_dist = J_sri * mu_sri_to_dist  (default 1.0: SRI routes to distributed capacity)
    #     J_sri_pool = J_sri * (1 - mu_sri_to_dist)
    #   dM/dt = J_sri_dist >= 0 (cumulative socialization capacity).
    m_half: Any = 1.0                # [MORPHOLOGY 2026-09-10, CALIBRATED] saturation half-point for g(M)
    mu_sri_to_dist: Any = 1.0        # [MORPHOLOGY 2026-09-10, SCENARIO] fraction of SRI flow routing to P_dist
    k_dissolve: Any = 0.0            # [MORPHOLOGY 2026-09-10, SCENARIO] base pool dissolution rate
    m0: Any = 0.0                    # [MORPHOLOGY 2026-09-10, SCENARIO] initial cumulative socialization M(0)
    p_pool0: Any = None              # [MORPHOLOGY 2026-09-10, SCENARIO] initial P_pool (defaults to P0 if None)
    p_dist0: Any = 0.0               # [MORPHOLOGY 2026-09-10, SCENARIO] initial P_dist

    @property
    def rho_impl(self) -> float:
        """Conservation assertion: rho_impl = 1 - mu_cef - mu_sri."""
        if self.rho is not None:
            return float(self.rho)
        return max(0.0, 1.0 - float(self.mu_cef) - float(self.mu_sri))

    def __post_init__(self) -> None:
        for name, val in (("year_period", self.year_period),
                          ("spring_phase", self.spring_phase),
                          ("autumn_phase", self.autumn_phase)):
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"{name} must be a float, got {type(val).__name__} "
                    "(seasonal phases cannot be per-node in this implementation)"
                )
        if self.year_period <= 0:
            raise ValueError(f"year_period must be positive, got {self.year_period}")
        if not (0.0 <= self.spring_phase < 1.0):
            raise ValueError(f"spring_phase must be in [0, 1), got {self.spring_phase}")
        if not (0.0 <= self.autumn_phase < 1.0):
            raise ValueError(f"autumn_phase must be in [0, 1), got {self.autumn_phase}")
        if abs(self.spring_phase - self.autumn_phase) < _FAST_T_EPS:
            raise ValueError("spring_phase and autumn_phase must differ")
        for name, val in (("mu_cef", self.mu_cef), ("mu_sri", self.mu_sri),
                          ("t_c_phi", self.t_c_phi), ("t_c_p", self.t_c_p),
                          ("t_c_k", self.t_c_k),
                          ("k_dissolve", self.k_dissolve), ("m0", self.m0),
                          ("p_dist0", self.p_dist0)):
            if isinstance(val, (int, float)) and val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")
        if isinstance(self.m_half, (int, float)) and self.m_half <= 0:
            raise ValueError(f"m_half must be strictly positive, got {self.m_half}")
        if isinstance(self.mu_sri_to_dist, (int, float)) and not (0.0 <= self.mu_sri_to_dist <= 1.0):
            raise ValueError(f"mu_sri_to_dist must be in [0, 1], got {self.mu_sri_to_dist}")
        if self.p_pool0 is not None and isinstance(self.p_pool0, (int, float)) and self.p_pool0 < 0:
            raise ValueError(f"p_pool0 must be non-negative, got {self.p_pool0}")
        if isinstance(self.mu_cef, (int, float)) and isinstance(self.mu_sri, (int, float)):
            if self.mu_cef + self.mu_sri > 1.0 + 1e-9:
                raise ValueError(
                    f"mu_cef + mu_sri cannot exceed 1.0; got {self.mu_cef + self.mu_sri}"
                )
        if self.rho is not None:
            if isinstance(self.rho, (int, float)) and self.rho < 0:
                raise ValueError(f"rho must be non-negative, got {self.rho}")
            if isinstance(self.mu_cef, (int, float)) and isinstance(self.mu_sri, (int, float)) and isinstance(self.rho, (int, float)):
                tot = self.mu_cef + self.mu_sri + self.rho
                if abs(tot - 1.0) > 1e-5:
                    raise ValueError(
                        f"Conservation violation: mu_cef + mu_sri + rho must sum to 1.0, got {tot}"
                    )

    def provenance(self) -> dict[str, str]:
        """Return the parameter → provenance-tag map (read-only copy)."""
        return dict(_FAST_PROVENANCE)


@dataclass
class StabilityRegime:
    """Result of :meth:`ForceFieldDynamics.assess_stability` (ledger §穩定性判據).

    Three assertable predicates:
    - ``convergent`` — 收斂域: ``βC* + δ > a + Φ*·μ_F`` (the suppression/decay
      channel dominates the growth/fast-monetisation channel → forces converge).
    - ``escape_poverty_trap`` — 超臨界分叉: ``Φ*·μ_F + a − δ > βC*`` AND ``λ > 0``
      (fast channel + slow reservoir release overcome suppression → the poverty
      trap attractor is escaped).
    - ``oscillatory_divergence`` — 振盪發散: autumn jump gain + spring clamp form
      a seasonal loop whose gain exceeds 1 (cobweb / limit cycle). Operational
      proxy ``[DERIVED_PROXY]``: ``(1−clamp_loss)·(1+autumn_frac)·exp(r_cont·T)``.

    Arrays are per-node; the booleans are the per-node ``all()``/``any()``
    reductions documented in ``detail``.
    """

    convergent: bool
    escape_poverty_trap: bool
    oscillatory_divergence: bool
    phi_star: np.ndarray
    c_star: np.ndarray
    convergence_margin: np.ndarray   # (βC*+δ) − (a+Φ*μ_F)
    bifurcation_margin: np.ndarray   # (Φ*μ_F+a−δ) − βC*
    seasonal_loop_gain: np.ndarray
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "convergent": bool(self.convergent),
            "escape_poverty_trap": bool(self.escape_poverty_trap),
            "oscillatory_divergence": bool(self.oscillatory_divergence),
            "phi_star": self.phi_star.tolist(),
            "c_star": self.c_star.tolist(),
            "convergence_margin": self.convergence_margin.tolist(),
            "bifurcation_margin": self.bifurcation_margin.tolist(),
            "seasonal_loop_gain": self.seasonal_loop_gain.tolist(),
            "detail": self.detail,
        }


class ForceFieldDynamics:
    """Deterministic R/C counterforce field engine (no LLM dependency).

    Integrates the §2.2 conflict ODEs for ``n`` coupled nodes with the §2.2b P_R
    potential reservoir, §2.5 external field modulation and §2.6 four-quadrant
    full-sign coupling. ``S = R/(R+C)`` is the social content resultant;
    ``T = R + P_R`` is the total transformative capacity (conservation H₀);
    ``tension = f(R·C)`` is the conflict intensity.

    v1.6: C's suppression compresses R into the reservoir (βC → P_R) rather
    than destroying it — total capacity T = R + P_R is conserved under pressure,
    so a suppressed transformative force can revive (1905→1917 shape). Growth
    and release rates may be substrate-supplied feedbacks of the universal
    observables: ``a(S,T)`` / ``c(S,T)`` (bandwagon/desertion + fatigue,
    §2.3b/§2.4b) and ``λ(T)`` (tension-gated reservoir opening, §2.4b). All
    world-agnostic — the *form* is supplied by the substrate, the structure is
    universal.

    Parameters
    ----------
    revolutionary_force, conservative_force
        Initial force magnitudes R₀, C₀ — scalar or length-``n`` array-like.
        Non-negative (validated).
    growth_r, growth_c
        Growth rates ``a_i`` (organisation expansion / class formation) and
        ``c_i`` (institutional reproduction). Scalar or length-``n``.
    alpha, beta
        Base suppression efficiencies α⁰ᵢ (R against C) and β⁰ᵢ (C against R).
        Non-negative. These are the values the external field modulates.
    mu, lam, rho
        P_R reservoir rates (§2.2b) — compression μ (spontaneous kinetic R →
        latent P_R), release λ (P_R → R under external-field modulation /
        C-cracks / high tension), decay ρ (genuinely forgotten history).
        Scalar or length-``n``; default 0 (reservoir disabled ⇒ v1.0 behaviour,
        backward compatible). When the reservoir is active (any of μ/λ/ρ or
        latent_force non-zero), C's suppression also compresses R into P_R
        (βC → P_R, v1.6) — suppression transfers rather than destroys, so
        total capacity T = R + P_R is conserved under pressure.
    latent_force
        Initial latent potential P_R,₀ — scalar or length-``n``; ``None`` ≡ 0.
        Default 0 (reservoir off).
    kappa
        Four-quadrant coupling. None (default, no coupling), a scalar (uniform
        κ for the same-force quadrants RR and CC, as in v1.0), an ``(n, n)``
        array (same matrix for RR and CC), the v1.0 dict ``{"R": (n,n),
        "C": (n,n)}`` (R→RR, C→CC), or the full-sign dict ``{"RR": (n,n),
        "RC": (n,n), "CC": (n,n), "CR": (n,n)}``. Missing quadrants → zero;
        giving both ``R`` and ``RR`` (or ``C`` and ``CC``) is ambiguous → error.
        Diagonal entries are inert by construction (``X_j − X_i = 0`` when
        ``j == i``).
    panic_exponent
        Exponent ``p`` on the source magnitude R_j in the κ^RC cross term
        (heterogeneous panic): ``dC_i += Σ κ^RC_ij · (R_j − R_i) · R_j^p``.
        Default 0.0 = linear Richardson term (backward compatible); ``p > 1``
        gives the "panicky exponential counter-reaction" (§2.6).
    mod_alpha, mod_beta
        External field modifiers — a callable ``t -> float`` (applied to every
        node) or a length-``n`` list of callables. Default None ≡ identity
        (``mod ≡ 1``). Effective rates are ``α⁰·mod_α(t)``, ``β⁰·mod_beta(t)``,
        clamped to non-negative (a suppression efficiency cannot go negative).
    growth_r_fn, growth_c_fn
        Optional substrate-supplied growth-rate feedbacks (§2.3b/§2.4b):
        ``fn(S, T) -> per-node multiplier`` on the base rates a, c — the
        bandwagon/desertion and fatigue channels. World-agnostic: the *form*
        (monotonic in S, etc.) is a historical proposition supplied by the
        substrate; None (default) ≡ constant base rate. Multipliers clamped
        non-negative.
    release_fn
        Optional substrate-supplied release-rate feedback (§2.4b):
        ``fn(T) -> per-node multiplier`` on the base release rate λ — the
        tension-gated opening of the P_R reservoir (high tension → latent force
        activates). None (default) ≡ constant λ.
    tension_fn
        ``f`` for §2.4 ``tension = f(R·C)``; default ``numpy.sqrt``
        (geometric-mean intensity, monotonic in the product). Must map
        non-negative → non-negative.
    method
        Integrator: ``"rk4"`` (default, classical 4th-order Runge–Kutta) or
        ``"euler"`` (explicit forward Euler). RK4 handles the time-dependent
        modulation naturally via intermediate stage evaluations.
    dt
        Default integration step for :meth:`step` / :meth:`run`.
    clamp_nonneg
        When True (default), forces (incl. P_R) are clamped to ``>= 0`` after
        every step. Forces are magnitudes — a negative force is unphysical, and
        the ``S ∈ [0, 1]`` invariant holds exactly under this clamp.
    node_ids
        Optional per-node labels (e.g. actor ids); carried into trajectory meta.
    meta
        Free-form provenance dict, carried into trajectory meta.
    fast_channel
        Opt-in 5-D fast/slow mode (THEORY-LEDGER §完整動態方程組). ``None``
        (default) or ``False`` ⇒ legacy 2/3-D path, byte-for-byte unchanged.
        ``True`` ⇒ default :class:`FastChannelConfig`; a dict is expanded into
        one; a :class:`FastChannelConfig` is used as-is. When enabled the state
        vector is ``(R, C, P_R, Φ, K)`` with the Φ fast channel (流通電導率) and
        the K alienated-capital pool; :meth:`derivatives_5d`, :meth:`assess_stability`
        and seasonal operator splitting become available. All parameters live on
        the config, each with a provenance tag (:meth:`FastChannelConfig.provenance`).
    """

    def __init__(
        self,
        revolutionary_force: Any,
        conservative_force: Any,
        *,
        growth_r: Any = 0.0,
        growth_c: Any = 0.0,
        alpha: Any = 0.0,
        beta: Any = 0.0,
        mu: Any = 0.0,
        lam: Any = 0.0,
        rho: Any = 0.0,
        latent_force: Any = 0.0,
        kappa: Any = None,
        panic_exponent: float = 0.0,
        mod_alpha: Callable | list[Callable] | None = None,
        mod_beta: Callable | list[Callable] | None = None,
        growth_r_fn: Callable | None = None,
        growth_c_fn: Callable | None = None,
        release_fn: Callable | None = None,
        tension_fn: Callable | None = None,
        method: str = "rk4",
        dt: float = 1.0,
        clamp_nonneg: bool = True,
        node_ids: list[str] | None = None,
        meta: dict | None = None,
        fast_channel: "FastChannelConfig | bool | dict | None" = None,
    ) -> None:
        R0 = _as_float_array(revolutionary_force, "revolutionary_force")
        C0 = _as_float_array(conservative_force, "conservative_force")
        if R0.size != C0.size:
            raise ValueError(
                f"revolutionary_force ({R0.size}) and conservative_force "
                f"({C0.size}) must have equal length"
            )
        n = int(R0.size)
        if n == 0:
            raise ValueError("force field requires at least one node")

        for name, arr in (("revolutionary_force", R0), ("conservative_force", C0)):
            if np.any(arr < 0):
                raise ValueError(f"{name} must be non-negative, got {arr.tolist()}")

        self._n = n
        self._R0 = R0.astype(np.float64, copy=True)
        self._C0 = C0.astype(np.float64, copy=True)

        self._a = _broadcast(growth_r, n, "growth_r")
        self._c = _broadcast(growth_c, n, "growth_c")
        self._a_fn = growth_r_fn
        self._c_fn = growth_c_fn
        self._alpha0 = _broadcast(alpha, n, "alpha")
        self._beta0 = _broadcast(beta, n, "beta")

        for name, arr in (("alpha", self._alpha0), ("beta", self._beta0)):
            if np.any(arr < 0):
                raise ValueError(f"{name} must be non-negative, got {arr.tolist()}")

        # P_R reservoir (§2.2b): compression μ, release λ, decay ρ — all optional
        # and defaulting to 0 (reservoir disabled ⇒ v1.0 behaviour).
        self._mu = _broadcast(mu, n, "mu")
        self._lam = _broadcast(lam, n, "lam")
        self._rho = _broadcast(rho, n, "rho")
        for name, arr in (("mu", self._mu), ("lam", self._lam), ("rho", self._rho)):
            if np.any(arr < 0):
                raise ValueError(f"{name} must be non-negative, got {arr.tolist()}")

        P0 = 0.0 if latent_force is None else latent_force
        self._P0 = _broadcast(P0, n, "latent_force")
        if np.any(self._P0 < 0):
            raise ValueError(
                f"latent_force must be non-negative, got {self._P0.tolist()}"
            )
        self._lam_fn = release_fn
        # v1.6: the βC → P_R compression channel is active only when the
        # reservoir is in use (any compression/release/decay rate or initial
        # latent force set). Reservoir off ⇒ v1.0 behaviour (P inert).
        self._reservoir_active = bool(
            np.any(self._mu) or np.any(self._lam) or np.any(self._rho)
            or np.any(self._P0)
        )

        kappas = self._resolve_kappa(kappa, n)
        self._kappa_RR = kappas["RR"]
        self._kappa_RC = kappas["RC"]
        self._kappa_CC = kappas["CC"]
        self._kappa_CR = kappas["CR"]

        if panic_exponent < 0:
            raise ValueError(
                f"panic_exponent must be non-negative, got {panic_exponent}"
            )
        self._panic_exp = float(panic_exponent)

        self._mod_alpha = self._resolve_mods(mod_alpha, n, "mod_alpha")
        self._mod_beta = self._resolve_mods(mod_beta, n, "mod_beta")

        if method not in _SUPPORTED_METHODS:
            raise ValueError(
                f"method must be one of {_SUPPORTED_METHODS}, got {method!r}"
            )
        self._method = method

        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._dt = float(dt)
        self._clamp_nonneg = bool(clamp_nonneg)

        if node_ids is not None and len(node_ids) != n:
            raise ValueError(f"node_ids length {len(node_ids)} != n_nodes {n}")
        self._node_ids = list(node_ids) if node_ids is not None else [f"node-{i}" for i in range(n)]

        self._tension_fn = tension_fn or (lambda x: np.sqrt(x))
        self._meta = dict(meta or {})

        # Runtime state
        self._t = 0.0
        self._R = self._R0.copy()
        self._C = self._C0.copy()
        self._P = self._P0.copy()

        # Opt-in 5-D fast-channel state (Φ, K) — see FastChannelConfig. When
        # disabled (None/False) every attribute below stays ``None`` and the
        # legacy code path is executed byte-for-byte unchanged.
        self._fast = self._build_fast_config(fast_channel)
        self._Phi0: np.ndarray | None = None
        self._K0: np.ndarray | None = None
        self._Phi: np.ndarray | None = None
        self._K: np.ndarray | None = None
        self._fc: dict[str, np.ndarray] = {}
        if self._fast is not None:
            self._initialise_fast_state(n)

    # ------------------------------------------------------------------
    # Fast-channel setup (opt-in 5-D mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fast_config(
        fast_channel: "FastChannelConfig | bool | dict | None",
    ) -> "FastChannelConfig | None":
        """Normalise the opt-in ``fast_channel`` argument.

        ``None``/``False`` → legacy 2/3-D path (default). ``True`` → default
        :class:`FastChannelConfig`. A dict is expanded into the config. An
        existing config passes through unchanged.
        """
        if fast_channel is None or fast_channel is False:
            return None
        if fast_channel is True:
            return FastChannelConfig()
        if isinstance(fast_channel, FastChannelConfig):
            return fast_channel
        if isinstance(fast_channel, dict):
            return FastChannelConfig(**fast_channel)
        raise ValueError(
            "fast_channel must be None/False (legacy), True (defaults), a "
            f"FastChannelConfig, or a dict; got {type(fast_channel).__name__}"
        )

    def _initialise_fast_state(self, n: int) -> None:
        """Broadcast + validate the 5-D parameters, then seed (Φ₀, K₀)."""
        cfg = self._fast
        assert cfg is not None  # guarded by caller

        def _b(name: str, value: Any) -> np.ndarray:
            return _broadcast(value, n, f"fast_channel.{name}")

        self._mu_f = _b("mu_f", cfg.mu_f)
        self._lam5 = _b("lam", cfg.lam)
        self._mu_cef = _b("mu_cef", cfg.mu_cef)
        self._mu_sri = _b("mu_sri", cfg.mu_sri)
        self._rho_val = _b("rho", cfg.rho if cfg.rho is not None else cfg.rho_impl)
        self._t_c_phi = _b("t_c_phi", cfg.t_c_phi)
        self._t_c_p = _b("t_c_p", cfg.t_c_p)
        self._t_c_k = _b("t_c_k", cfg.t_c_k)
        self._fc = {
            "co_share": _b("co_share", cfg.co_share),
            "debt_stress": _b("debt_stress", cfg.debt_stress),
            "kappa_phi": _b("kappa_phi", cfg.kappa_phi),
            "zeta_phi": _b("zeta_phi", cfg.zeta_phi),
            "delta": _b("delta", cfg.delta),
            "eta": _b("eta", cfg.eta),
            "mu_f": self._mu_f,
            "lam": self._lam5,
            "s_in": _b("s_in", cfg.s_in),
            "mu_e": _b("mu_e", cfg.mu_e),
            "kappa_c": _b("kappa_c", cfg.kappa_c),
            "gamma_c": _b("gamma_c", cfg.gamma_c),
            "alpha_rc": _b("alpha_rc", cfg.alpha_rc),
            "sigma_soviet": _b("sigma_soviet", cfg.sigma_soviet),
            "gamma_k": _b("gamma_k", cfg.gamma_k),
            "clamp_spring": _b("clamp_spring", cfg.clamp_spring),
            "autumn_yield": _b("autumn_yield", cfg.autumn_yield),
            "reflow_purity": _b("reflow_purity", cfg.reflow_purity),
            "surplus_gain": _b("surplus_gain", cfg.surplus_gain),
            "mu_cef": self._mu_cef,
            "mu_sri": self._mu_sri,
            "rho": self._rho_val,
            "t_c_phi": self._t_c_phi,
            "t_c_p": self._t_c_p,
            "t_c_k": self._t_c_k,
        }

        nonneg = ("co_share", "debt_stress", "kappa_phi", "zeta_phi", "delta",
                  "mu_f", "lam", "s_in", "mu_e", "kappa_c", "gamma_c",
                  "alpha_rc",
                  "gamma_k", "clamp_spring", "autumn_yield", "reflow_purity",
                  "surplus_gain", "mu_cef", "mu_sri", "rho", "t_c_phi", "t_c_p", "t_c_k")
        for name in nonneg:
            if np.any(self._fc[name] < 0):
                raise ValueError(
                    f"fast_channel.{name} must be non-negative, "
                    f"got {self._fc[name].tolist()}"
                )
        eta = self._fc["eta"]
        if np.any(eta < 0) or np.any(eta > 1):
            raise ValueError(
                f"fast_channel.eta must be in [0, 1], got {eta.tolist()}"
            )
        if np.any(self._fc["sigma_soviet"] <= 0):
            raise ValueError(
                "fast_channel.sigma_soviet must be positive, "
                f"got {self._fc['sigma_soviet'].tolist()}"
            )

        phi0 = _b("phi0", cfg.phi0)
        if np.any(phi0 < 0) or np.any(phi0 > 1):
            raise ValueError(
                f"fast_channel.phi0 must be in [0, 1], got {phi0.tolist()}"
            )
        k0 = _b("k0", cfg.k0)
        if np.any(k0 < 0):
            raise ValueError(f"fast_channel.k0 must be non-negative, got {k0.tolist()}")

        # Morphology broadcast & validation
        self._m_half = _b("m_half", cfg.m_half)
        if np.any(self._m_half <= 0):
            raise ValueError(f"fast_channel.m_half must be strictly positive, got {self._m_half.tolist()}")
        self._mu_sri_to_dist = _b("mu_sri_to_dist", cfg.mu_sri_to_dist)
        if np.any(self._mu_sri_to_dist < 0) or np.any(self._mu_sri_to_dist > 1.0):
            raise ValueError(f"fast_channel.mu_sri_to_dist must be in [0, 1], got {self._mu_sri_to_dist.tolist()}")
        self._k_dissolve = _b("k_dissolve", cfg.k_dissolve)
        if np.any(self._k_dissolve < 0):
            raise ValueError(f"fast_channel.k_dissolve must be non-negative, got {self._k_dissolve.tolist()}")
        self._m0 = _b("m0", cfg.m0)
        if np.any(self._m0 < 0):
            raise ValueError(f"fast_channel.m0 must be non-negative, got {self._m0.tolist()}")
        self._p_dist0 = _b("p_dist0", cfg.p_dist0)
        if np.any(self._p_dist0 < 0):
            raise ValueError(f"fast_channel.p_dist0 must be non-negative, got {self._p_dist0.tolist()}")
        if cfg.p_pool0 is None:
            self._p_pool0 = np.maximum(self._P0 - self._p_dist0, 0.0)
        else:
            self._p_pool0 = _b("p_pool0", cfg.p_pool0)
            if np.any(self._p_pool0 < 0):
                raise ValueError(f"fast_channel.p_pool0 must be non-negative, got {self._p_pool0.tolist()}")

        self._fc.update({
            "m_half": self._m_half,
            "mu_sri_to_dist": self._mu_sri_to_dist,
            "k_dissolve": self._k_dissolve,
            "m0": self._m0,
            "p_pool0": self._p_pool0,
            "p_dist0": self._p_dist0,
        })

        self._Phi0 = np.clip(phi0, _PHI_MIN, _PHI_MAX).astype(np.float64)
        self._K0 = k0.astype(np.float64)
        self._Phi = self._Phi0.copy()
        self._K = self._K0.copy()
        self._P_pool0 = self._p_pool0.copy().astype(np.float64)
        self._P_dist0 = self._p_dist0.copy().astype(np.float64)
        self._P_pool = self._P_pool0.copy()
        self._P_dist = self._P_dist0.copy()
        self._M0 = self._m0.copy().astype(np.float64)
        self._M = self._M0.copy()
        # Synchronize total P with P_pool + P_dist
        self._P = self._P_pool + self._P_dist
        self._P0 = self._P.copy()

        self._digestion_active = bool(
            np.any(self._mu_cef > 0.0) or np.any(self._mu_sri > 0.0) or (cfg.rho is not None)
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_kappa(kappa: Any, n: int) -> dict[str, np.ndarray]:
        """Normalise ``kappa`` into the four-quadrant coupling matrices.

        Accepts (all backward compatible):
        - None → every quadrant zero (no coupling).
        - scalar → uniform κ for the same-force quadrants RR and CC (v1.0
          semantics); the cross quadrants RC/CR stay zero.
        - ``(n, n)`` array → the same matrix for RR and CC (v1.0 semantics).
        - dict with keys ``R``/``C`` (v1.0 form; R→RR, C→CC) and/or the
          four-quadrant keys ``RR``/``RC``/``CC``/``CR``. Missing quadrants →
          zero. Giving both ``R`` and ``RR`` (or ``C`` and ``CC``) is
          ambiguous → error.

        Quadrant semantics (§2.6): RR = solidarity diffusion; CC = coordination
        diffusion; RC = heterogeneous panic (R_j rises ⇒ C_i counterattacks);
        CR = C_j rises ⇒ R_i suppressed/activated (regime-dependent sign).
        """
        zeros = np.zeros((n, n), dtype=np.float64)
        if kappa is None:
            return {"RR": zeros.copy(), "RC": zeros.copy(),
                    "CC": zeros.copy(), "CR": zeros.copy()}

        def _mat(key: str, value: Any) -> np.ndarray:
            mat = np.asarray(value, dtype=np.float64)
            if mat.ndim == 0:
                # Scalar κ → uniform coupling across every ordered pair.
                mat = np.full((n, n), float(mat), dtype=np.float64)
            if mat.ndim != 2 or mat.shape != (n, n):
                raise ValueError(f"kappa.{key} must have shape ({n}, {n}), got {mat.shape}")
            return mat

        if isinstance(kappa, dict):
            allowed = {"R", "C", "RR", "RC", "CC", "CR"}
            unknown = set(kappa) - allowed
            if unknown:
                raise ValueError(
                    f"kappa dict keys must be one of {sorted(allowed)}, "
                    f"got {sorted(unknown)}"
                )
            if "R" in kappa and "RR" in kappa:
                raise ValueError("kappa dict: give either 'R' or 'RR', not both")
            if "C" in kappa and "CC" in kappa:
                raise ValueError("kappa dict: give either 'C' or 'CC', not both")
            out = {"RR": zeros.copy(), "RC": zeros.copy(),
                   "CC": zeros.copy(), "CR": zeros.copy()}
            for key in ("RR", "RC", "CC", "CR"):
                if key in kappa:
                    out[key] = _mat(key, kappa[key])
            if "R" in kappa:
                out["RR"] = _mat("R", kappa["R"])
            if "C" in kappa:
                out["CC"] = _mat("C", kappa["C"])
            return out

        # scalar or (n, n) array → the same matrix for RR and CC (v1.0 semantics)
        mat = _mat("RR", kappa)
        return {"RR": mat.copy(), "RC": zeros.copy(),
                "CC": mat.copy(), "CR": zeros.copy()}

    @staticmethod
    def _resolve_mods(
        mods: Callable | list[Callable] | None, n: int, name: str
    ) -> list[Callable]:
        """Normalise a modifier into a length-``n`` list of callables."""
        if mods is None:
            return [lambda t: 1.0] * n
        if callable(mods):
            return [mods] * n
        if isinstance(mods, (list, tuple)):
            if len(mods) != n:
                raise ValueError(f"{name} length {len(mods)} != n_nodes {n}")
            if not all(callable(m) for m in mods):
                raise ValueError(f"{name} entries must be callables")
            return list(mods)
        raise ValueError(f"{name} must be None, a callable, or a list of callables")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Number of field nodes."""
        return self._n

    @property
    def node_ids(self) -> list[str]:
        """Per-node labels."""
        return list(self._node_ids)

    @property
    def t(self) -> float:
        """Current integration time."""
        return float(self._t)

    @property
    def state(self) -> tuple[np.ndarray, np.ndarray]:
        """Current (R, C) magnitudes (read-only copies)."""
        return self._R.copy(), self._C.copy()

    @property
    def latent_state(self) -> np.ndarray:
        """Current latent potential P_R magnitudes (read-only copy, shape ``(n,)``)."""
        return self._P.copy()

    @property
    def fast_channel_enabled(self) -> bool:
        """True when the opt-in 5-D (Φ, K) mode is active."""
        return self._fast is not None

    @property
    def fast_config(self) -> "FastChannelConfig | None":
        """The active :class:`FastChannelConfig`, or ``None`` on the legacy path."""
        return self._fast

    @property
    def phi(self) -> np.ndarray | None:
        """Current circulation conductivity Φ (read-only copy), or ``None``."""
        return None if self._Phi is None else self._Phi.copy()

    @property
    def k_pool(self) -> np.ndarray | None:
        """Current alienated-capital pool K (read-only copy), or ``None``."""
        return None if self._K is None else self._K.copy()

    @property
    def p_pool(self) -> np.ndarray | None:
        """Current centralized reserve pool P_pool (read-only copy), or ``None``."""
        return None if self._P_pool is None else self._P_pool.copy()

    @property
    def p_dist(self) -> np.ndarray | None:
        """Current distributed capacity P_dist (read-only copy), or ``None``."""
        return None if self._P_dist is None else self._P_dist.copy()

    @property
    def m_cum(self) -> np.ndarray | None:
        """Current cumulative socialization capacity M (read-only copy), or ``None``."""
        return None if self._M is None else self._M.copy()

    @property
    def g_sat(self) -> np.ndarray | None:
        """Current morphology transition factor g(M) = M / (M + m_half), or ``None``."""
        if self._M is None or self._m_half is None:
            return None
        denom = np.maximum(self._M + self._m_half, 1e-12)
        return np.maximum(self._M, 0.0) / denom

    @property
    def pool_share(self) -> np.ndarray | None:
        """Current pool share s_pool = P_pool / (P_pool + P_dist), or ``None``."""
        if self._P_pool is None or self._P_dist is None:
            return None
        tot = self._P_pool + self._P_dist
        denom = np.where(tot > 1e-12, tot, 1.0)
        return np.where(tot > 1e-12, self._P_pool / denom, 1.0)

    def fast_state(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Current ``(Φ, K)`` read-only copies, or ``None`` on the legacy path."""
        if self._Phi is None or self._K is None:
            return None
        return self._Phi.copy(), self._K.copy()

    def morphology_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Current ``(P_pool, P_dist, M, g_sat)`` read-only copies, or ``None``."""
        if self._P_pool is None or self._P_dist is None:
            return None
        return self._P_pool.copy(), self._P_dist.copy(), self._M.copy(), self.g_sat

    def five_dim_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray] | None:
        """Current ``(R, C, P_R, Φ, K)`` read-only copies, or ``None`` (legacy)."""
        if self._Phi is None or self._K is None:
            return None
        return (self._R.copy(), self._C.copy(), self._P.copy(),
                self._Phi.copy(), self._K.copy())

    def effective_rates(self, t: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return the external-field-modulated rates (α_eff, β_eff) at time ``t``.

        α_eff[i] = α⁰ᵢ · max(mod_αᵢ(t), 0), β_eff[i] = β⁰ᵢ · max(mod_βᵢ(t), 0).
        Default ``t`` = current engine time.
        """
        tt = self._t if t is None else float(t)
        mod_a = np.array([max(float(fn(tt)), 0.0) for fn in self._mod_alpha], dtype=np.float64)
        mod_b = np.array([max(float(fn(tt)), 0.0) for fn in self._mod_beta], dtype=np.float64)
        return self._alpha0 * mod_a, self._beta0 * mod_b

    def set_modulation(
        self,
        mod_alpha: Callable | list[Callable] | None = None,
        mod_beta: Callable | list[Callable] | None = None,
    ) -> "ForceFieldDynamics":
        """公開設定外部場調製器（取代現行 mod）；None 保持現值。

        契約同建構子：``callable(t)`` 或 length-``n`` callable 清單。供編排器
        （如 ``synth.round_loop``）經公開接口注入人選決策——不需訪問私有屬性。
        α_eff = α⁰·mod_α(t)、β_eff = β⁰·mod_β(t)。回傳 self 以便鏈式呼叫。
        """
        if mod_alpha is not None:
            self._mod_alpha = self._resolve_mods(mod_alpha, self._n, "mod_alpha")
        if mod_beta is not None:
            self._mod_beta = self._resolve_mods(mod_beta, self._n, "mod_beta")
        return self

    def current_modulation(self) -> tuple[list[Callable], list[Callable]]:
        """回傳現行外部場調製器（length-``n`` callable 清單副本，供查詢/編排）。"""
        return list(self._mod_alpha), list(self._mod_beta)

    # ------------------------------------------------------------------
    # Core observables
    # ------------------------------------------------------------------

    def s(self, R: Any = None, C: Any = None) -> np.ndarray:
        """Resultant S = R/(R+C) ∈ [0, 1] (§2.3).

        Defaults to the current engine state. Degenerate guard: when R+C → 0
        (both forces vanish) S is pinned to 0.5 — the content is undecided,
        not NaN. Returns shape ``(n,)``.
        """
        R_arr = self._R if R is None else _as_float_array(R, "R")
        C_arr = self._C if C is None else _as_float_array(C, "C")
        total = R_arr + C_arr
        out = np.full_like(R_arr, 0.5, dtype=np.float64)
        mask = total > _S_EPS
        out[mask] = R_arr[mask] / total[mask]
        return np.clip(out, 0.0, 1.0)

    def tension(self, R: Any = None, C: Any = None) -> np.ndarray:
        """Tension f(R·C) ≥ 0 (§2.4) — conflict intensity, not force ratio.

        Defaults to the current engine state. Returns shape ``(n,)``.
        """
        R_arr = self._R if R is None else _as_float_array(R, "R")
        C_arr = self._C if C is None else _as_float_array(C, "C")
        product = np.maximum(R_arr, 0.0) * np.maximum(C_arr, 0.0)
        return np.asarray(self._tension_fn(product), dtype=np.float64)

    def total_capacity(self, R: Any = None, P: Any = None) -> np.ndarray:
        """Total transformative capacity T = R + P_R (§2.2b conservation observable).

        This is the H₀ to test on historical data — NOT a hard constraint. The
        engine never enforces T constancy; it only emits T as an observable so
        analysts can falsify the conservation hypothesis (§3.2). Defaults to the
        current engine state. Returns shape ``(n,)``.
        """
        R_arr = self._R if R is None else _as_float_array(R, "R")
        P_arr = self._P if P is None else _as_float_array(P, "P")
        return R_arr + P_arr

    # ------------------------------------------------------------------
    # ODE right-hand side
    # ------------------------------------------------------------------

    def derivatives(self, t: float, R: Any = None, C: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """ODE right-hand side for (R, C): growth − suppression + coupling + reservoir.

        Returns ``(dR, dC)`` — the raw time derivatives (§2.2 + §2.2b + §2.6),
        with the external field (§2.5) folded into α/β at time ``t``. Defaults
        to the current engine state for R/C/P_R. The P_R derivative is available
        via :meth:`derivatives_full`.
        """
        dR, dC, _dP = self.derivatives_full(t, R, C)
        return dR, dC

    def derivatives_full(
        self, t: float, R: Any = None, C: Any = None, P: Any = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full ODE right-hand side incl. the P_R reservoir: ``(dR, dC, dP)``.

        Equations (§2.2 + §2.2b + §2.6), with the external field (§2.5) folded
        into α/β at time ``t``:

            dR = a·R − β_eff·C + λ·P − μ·R + κ_RR(R_j−R_i) + κ_CR(C_j−C_i)
            dP = β_eff·C + μ·R − λ·P − ρ·P
            dC = c·C − α_eff·R + κ_CC(C_j−C_i) + κ_RC(R_j−R_i)·R_j^p

        where ``p = panic_exponent`` (0 ⇒ the κ_RC term is linear). Defaults to
        the current engine state for R/C/P_R.
        """
        R_arr = self._R if R is None else _as_float_array(R, "R")
        C_arr = self._C if C is None else _as_float_array(C, "C")
        P_arr = self._P if P is None else _as_float_array(P, "P")
        alpha_eff, beta_eff = self.effective_rates(t)

        # v1.6 substrate-supplied feedbacks (§2.3b/§2.4b), expressed purely in
        # the universal observables: a(S,T), c(S,T) (bandwagon/desertion +
        # fatigue) and λ(T) (tension-gated reservoir opening). None ≡ constant.
        S_arr = self.s(R_arr, C_arr)
        T_arr = self.tension(R_arr, C_arr)

        def _mult(fn: Callable | None, *args: np.ndarray) -> np.ndarray:
            if fn is None:
                return np.ones_like(args[0])
            m = np.asarray(fn(*args), dtype=np.float64)
            if m.ndim == 0:
                m = np.full(args[0].shape, float(m))
            return np.maximum(np.broadcast_to(m, args[0].shape), 0.0)

        a_eff = self._a * _mult(self._a_fn, S_arr, T_arr)
        c_eff = self._c * _mult(self._c_fn, S_arr, T_arr)
        lam_eff = self._lam * _mult(self._lam_fn, T_arr)

        dR = a_eff * R_arr - beta_eff * C_arr + lam_eff * P_arr - self._mu * R_arr
        dC = c_eff * C_arr - alpha_eff * R_arr
        # v1.6: C's suppression compresses R into P_R (βC → P_R) instead of
        # destroying it — total capacity T = R + P_R conserved under pressure
        # (§2.2b). Applies only when the reservoir is active.
        dP = self._mu * R_arr - (lam_eff + self._rho) * P_arr
        if self._reservoir_active:
            dP = dP + beta_eff * C_arr

        # Same-force diffusion: solidarity (RR) and coordination (CC).
        if np.any(self._kappa_RR):
            dR = dR + self._kappa_RR @ R_arr - np.sum(self._kappa_RR, axis=1) * R_arr
        if np.any(self._kappa_CC):
            dC = dC + self._kappa_CC @ C_arr - np.sum(self._kappa_CC, axis=1) * C_arr
        # Cross-quadrant: C_j rises ⇒ R_i responds (κ_CR, regime-dependent).
        if np.any(self._kappa_CR):
            dR = dR + self._kappa_CR @ C_arr - np.sum(self._kappa_CR, axis=1) * C_arr
        # Cross-quadrant: R_j rises ⇒ C_i counterattacks (κ_RC, heterogeneous
        # panic). Nonlinear option: the term is weighted by the source R_j^p.
        if np.any(self._kappa_RC):
            if self._panic_exp != 0.0:
                # dC_i += Σ_j κ_RC_ij · (R_j − R_i) · R_j^p
                Rw = np.power(np.maximum(R_arr, 0.0), self._panic_exp)
                dC = dC + self._kappa_RC @ (R_arr * Rw) \
                    - R_arr * (np.sum(self._kappa_RC, axis=1) * Rw)
            else:
                dC = dC + self._kappa_RC @ R_arr - np.sum(self._kappa_RC, axis=1) * R_arr

        return dR, dC, dP

    # ------------------------------------------------------------------
    # Opt-in 5-D / 7-component fast-channel RHS (THEORY-LEDGER 2026-09-10)
    # ------------------------------------------------------------------

    def derivatives_morphology(
        self,
        t: float,
        R: Any = None,
        C: Any = None,
        P_pool: Any = None,
        P_dist: Any = None,
        Phi: Any = None,
        K: Any = None,
        M: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """7-component RHS tracking P_R morphology: ``(dR, dC, dP_pool, dP_dist, dΦ, dK, dM)``.

        Operationalises THEORY-LEDGER (2026-09-10) P_R morphology split:
            P_R = P_pool + P_dist
            P_pool: centralized/concentrated reserve subject to administrative mobilization (lambda)
                    and bureaucratic extraction (eta * mu_e).
            P_dist: distributed capacity (literacy, mutual aid), immune to capture, does direct work.
            g(M) = M / (M + m_half): saturating morphology transition factor.
            J_dissolve = k_dissolve * g(M) * P_pool: autonomous dissolution of pool into distributed form.
            dM/dt = J_sri_dist >= 0: cumulative socialization capacity.
        """
        if self._fast is None:
            raise RuntimeError(
                "derivatives_morphology requires the opt-in fast channel; construct "
                "ForceFieldDynamics(..., fast_channel=...) to enable it"
            )
        R_arr = self._R if R is None else _as_float_array(R, "R")
        C_arr = self._C if C is None else _as_float_array(C, "C")
        Pp_arr = self._P_pool if P_pool is None else _as_float_array(P_pool, "P_pool")
        Pd_arr = self._P_dist if P_dist is None else _as_float_array(P_dist, "P_dist")
        Phi_arr = self._Phi if Phi is None else _as_float_array(Phi, "Phi")
        K_arr = self._K if K is None else _as_float_array(K, "K")
        M_arr = self._M if M is None else _as_float_array(M, "M")

        fc = self._fc
        _alpha_eff, beta_eff = self.effective_rates(t)
        S_arr = self.s(R_arr, C_arr)
        T_arr = self.tension(R_arr, C_arr)
        a_eff = self._a * _apply_multiplier(self._a_fn, S_arr, T_arr)

        # Central command mobilization lambda and bureaucratic alienation eta*mu_e
        # draw strictly from P_pool (centralized morphology).
        dR = (a_eff * R_arr - fc["delta"] * R_arr
              + Phi_arr * fc["mu_f"] * R_arr
              + (1.0 - fc["eta"]) * fc["lam"] * Pp_arr
              - beta_eff * C_arr)

        dC_source = fc["kappa_c"] * (1.0 - Phi_arr)
        dC_sink = (fc["alpha_rc"] * R_arr * C_arr
                   + fc["gamma_c"] * fc["sigma_soviet"] * C_arr)
        dC = dC_source - dC_sink

        dPhi = (fc["kappa_phi"] * fc["co_share"] * (1.0 - Phi_arr)
                - fc["zeta_phi"] * fc["debt_stress"] * Phi_arr)
        dK = fc["eta"] * fc["mu_e"] * Pp_arr - fc["gamma_k"] * K_arr

        # Morphology dissolution: J_dissolve = k_dissolve * g(M) * P_pool
        # g(M) = M / (M + m_half)
        m_denom = np.maximum(M_arr + fc["m_half"], 1e-12)
        g_sat = np.maximum(M_arr, 0.0) / m_denom
        j_dissolve = fc["k_dissolve"] * g_sat * Pp_arr

        # Base pool evolution before digestion flows
        dPp = fc["s_in"] * R_arr - fc["lam"] * Pp_arr - fc["eta"] * fc["mu_e"] * Pp_arr - j_dissolve
        dPd = j_dissolve
        dM = np.zeros_like(M_arr)

        if self._digestion_active:
            # Positive-rectified net decay flow of C: [-C_dot]_+ = max(0, -dC)
            c_decay_pos = np.maximum(0.0, -dC)
            dPhi = dPhi + fc["t_c_phi"] * fc["mu_cef"] * c_decay_pos
            j_sri_total = fc["t_c_p"] * fc["mu_sri"] * c_decay_pos
            dK = dK + fc["t_c_k"] * fc["rho"] * c_decay_pos

            # SRI routing into distributed vs pool morphology
            j_sri_dist = j_sri_total * fc["mu_sri_to_dist"]
            j_sri_pool = j_sri_total * (1.0 - fc["mu_sri_to_dist"])
            dPp = dPp + j_sri_pool
            dPd = dPd + j_sri_dist
            dM = dM + j_sri_dist

        return dR, dC, dPp, dPd, dPhi, dK, dM

    def derivatives_5d(
        self,
        t: float,
        R: Any = None,
        C: Any = None,
        P: Any = None,
        Phi: Any = None,
        K: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """5-D fast/slow RHS: ``(dR, dC, dP, dΦ, dK)``.

        Delegates to :meth:`derivatives_morphology` by decomposing P into
        P_pool and P_dist using current pool share (or internal state), and returns
        dP = dP_pool + dP_dist, preserving exact 5-D dynamics.
        """
        if self._fast is None:
            raise RuntimeError(
                "derivatives_5d requires the opt-in fast channel; construct "
                "ForceFieldDynamics(..., fast_channel=...) to enable it"
            )
        if P is None:
            Pp_arr = self._P_pool
            Pd_arr = self._P_dist
        else:
            P_arr = _as_float_array(P, "P")
            cur_tot = self._P_pool + self._P_dist
            safe_tot = np.where(cur_tot > 1e-12, cur_tot, 1.0)
            share_pool = np.where(cur_tot > 1e-12, self._P_pool / safe_tot, 1.0)
            Pp_arr = P_arr * share_pool
            Pd_arr = P_arr * (1.0 - share_pool)

        dR, dC, dPp, dPd, dPhi, dK, _dM = self.derivatives_morphology(
            t, R=R, C=C, P_pool=Pp_arr, P_dist=Pd_arr, Phi=Phi, K=K, M=self._M
        )
        dP = dPp + dPd
        return dR, dC, dP, dPhi, dK

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def _stage_derivatives(self, t: float, R: np.ndarray, C: np.ndarray, P: np.ndarray,
                           dR: np.ndarray, dC: np.ndarray, dP: np.ndarray,
                           dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return full derivatives at a mid/last stage for RK4 (incl. P_R)."""
        return self.derivatives_full(t, R + dt * dR, C + dt * dC, P + dt * dP)

    # ------------------------------------------------------------------
    # Opt-in 5-D / morphology integration + seasonal operator splitting
    # ------------------------------------------------------------------

    def _integrate_5d(self, dt: float) -> None:
        """Continuous 7-component step (no seasonal operator) using ``method``."""
        if dt <= 0:
            return
        y = (self._R, self._C, self._P_pool, self._P_dist, self._Phi, self._K, self._M)
        if self._method == "rk4":
            t = self._t
            k1 = self.derivatives_morphology(t, *y)

            def _stage(frac: float, ks) -> tuple:
                return tuple(b + frac * dt * k for b, k in zip(y, ks))

            k2 = self.derivatives_morphology(t + 0.5 * dt, *_stage(0.5, k1))
            k3 = self.derivatives_morphology(t + 0.5 * dt, *_stage(0.5, k2))
            k4 = self.derivatives_morphology(t + dt, *_stage(1.0, k3))
            inc = tuple((k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * (dt / 6.0)
                        for i in range(7))
        else:  # euler
            inc = tuple(v * dt for v in self.derivatives_morphology(self._t, *y))

        new_R = self._R + inc[0]
        new_C = self._C + inc[1]
        new_Pp = self._P_pool + inc[2]
        new_Pd = self._P_dist + inc[3]
        new_Phi = np.clip(self._Phi + inc[4], _PHI_MIN, _PHI_MAX)
        new_K = self._K + inc[5]
        new_M = self._M + inc[6]

        if self._clamp_nonneg:
            new_R = np.maximum(new_R, 0.0)
            new_C = np.maximum(new_C, 0.0)
            new_Pp = np.maximum(new_Pp, 0.0)
            new_Pd = np.maximum(new_Pd, 0.0)
            new_K = np.maximum(new_K, 0.0)
            new_M = np.maximum(new_M, 0.0)

        self._t += dt
        self._R = new_R
        self._C = new_C
        self._P_pool = new_Pp
        self._P_dist = new_Pd
        self._P = self._P_pool + self._P_dist
        self._Phi = new_Phi
        self._K = new_K
        self._M = new_M

    def _season_events(self, t0: float, t1: float) -> list[tuple[float, str]]:
        """Seasonal operator-split events in the half-open interval ``(t0, t1]``.

        Events recur every ``year_period`` at ``spring_phase`` (春耕鉗) and
        ``autumn_phase`` (秋收兌現); the strict lower bound prevents re-applying
        an event already consumed at the current time.
        """
        cfg = self._fast
        assert cfg is not None
        period = cfg.year_period
        events: list[tuple[float, str]] = []
        for phase, kind in ((cfg.spring_phase, "spring"),
                            (cfg.autumn_phase, "autumn")):
            base = phase * period
            k = int(math.floor((t0 - base) / period))
            for kk in (k, k + 1, k + 2):
                te = kk * period + base
                if t0 + _FAST_T_EPS < te <= t1 + _FAST_T_EPS:
                    events.append((te, kind))
        events.sort()
        return events

    def _apply_season_event(self, kind: str) -> None:
        """Apply one seasonal jump operator (spring clamp / autumn pay-out)."""
        fc = self._fc
        if kind == "spring":
            # R ← R · (1 − (1−Φ)·Clamp_spring); Φ→1 ⇒ fully immune.
            factor = np.maximum(1.0 - (1.0 - self._Phi) * fc["clamp_spring"], 0.0)
            self._R = self._R * factor
        elif kind == "autumn":
            # R ← R + Yield·Φ·ReflowPurity ;  P_R ← P_R + SurplusGain.
            # Surplus gain from agricultural production enters centralized reserve pool (P_pool).
            self._R = self._R + fc["autumn_yield"] * self._Phi * fc["reflow_purity"]
            self._P_pool = self._P_pool + fc["surplus_gain"]
            self._P = self._P_pool + self._P_dist
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"unknown season event {kind!r}")
        if self._clamp_nonneg:
            self._R = np.maximum(self._R, 0.0)
            self._P_pool = np.maximum(self._P_pool, 0.0)
            self._P_dist = np.maximum(self._P_dist, 0.0)
            self._P = self._P_pool + self._P_dist
        self._Phi = np.clip(self._Phi, _PHI_MIN, _PHI_MAX)

    def _advance_5d(self, dt: float) -> None:
        """Advance one ``dt`` with the ledger's mixed operator splitting.

        The continuous 5-D ODE is integrated up to each seasonal boundary, the
        discrete jump operator is applied there, and integration resumes — so
        the seasonal jumps are embedded in (not added to) the continuous flow.
        """
        if dt <= 0:
            return
        cfg = self._fast
        assert cfg is not None
        remaining = float(dt)
        guard = 0
        max_iter = 4096
        while remaining > _FAST_T_EPS:
            guard += 1
            if guard > max_iter:  # pragma: no cover - defensive
                raise RuntimeError("seasonal operator-split loop did not terminate")
            t1 = self._t + remaining
            events = self._season_events(self._t, t1)
            if not events:
                self._integrate_5d(remaining)
                return
            te, kind = events[0]
            sub = te - self._t
            if sub > _FAST_T_EPS:
                self._integrate_5d(sub)
                remaining -= sub
            else:
                # Already at the boundary (floating-point residue): drop the
                # sub-epsilon remainder and consume the event below.
                remaining = 0.0
            self._apply_season_event(kind)

    def _advance(self, dt: float) -> None:
        """Advance the internal state by one step of ``dt`` using ``method``.

        Opt-in 5-D mode dispatches to :meth:`_advance_5d` (continuous ODE +
        seasonal operator splitting); the legacy path below is untouched.
        """
        if self._fast is not None:
            self._advance_5d(dt)
            return
        if self._method == "rk4":
            t, R, C, P = self._t, self._R, self._C, self._P
            k1 = self.derivatives_full(t, R, C, P)
            k2 = self._stage_derivatives(t, R, C, P, k1[0], k1[1], k1[2], dt / 2.0)
            k3 = self._stage_derivatives(t, R, C, P, k2[0], k2[1], k2[2], dt / 2.0)
            k4 = self._stage_derivatives(t, R, C, P, k3[0], k3[1], k3[2], dt)
            dR = dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
            dC = dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
            dP = dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        else:  # euler
            dR, dC, dP = self.derivatives_full(self._t, self._R, self._C, self._P)
            dR = dR * dt
            dC = dC * dt
            dP = dP * dt

        new_R = self._R + dR
        new_C = self._C + dC
        new_P = self._P + dP
        if self._clamp_nonneg:
            new_R = np.maximum(new_R, 0.0)
            new_C = np.maximum(new_C, 0.0)
            new_P = np.maximum(new_P, 0.0)

        self._t += dt
        self._R = new_R
        self._C = new_C
        self._P = new_P

    def step(self, dt: float | None = None) -> dict:
        """Advance one step (default ``self._dt``) and return a state snapshot.

        Snapshot keys: ``t``, ``R``, ``C``, ``S``, ``T`` (numpy arrays). In the
        opt-in 5-D mode, ``Phi`` and ``K`` are added; the legacy snapshot shape
        is unchanged.
        """
        self._advance(self._dt if dt is None else float(dt))
        snap = {
            "t": self._t,
            "R": self._R.copy(),
            "C": self._C.copy(),
            "P": self._P.copy(),
            "S": self.s(),
            "T": self.total_capacity(),
            "tension": self.tension(),
        }
        if self._fast is not None:
            snap["Phi"] = self._Phi.copy()
            snap["K"] = self._K.copy()
        return snap

    def run(self, t_end: float, dt: float | None = None) -> ForceTrajectory:
        """Integrate from the current state to ``t_end`` (inclusive).

        Steps of ``dt`` (default ``self._dt``); the final step is shrunk so the
        last sample lands exactly on ``t_end``. The engine's internal state is
        left at ``t_end``; call :meth:`reset` to re-run from initial conditions.

        Returns a :class:`ForceTrajectory` whose arrays are ``[step, node]``.
        """
        dt = self._dt if dt is None else float(dt)
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if t_end < self._t:
            raise ValueError(f"t_end {t_end} < current time {self._t}")

        t0 = self._t
        n_steps = int(np.floor((t_end - t0) / dt + 1e-9))

        times = np.empty(n_steps + 1, dtype=np.float64)
        R_hist = np.empty((n_steps + 1, self._n), dtype=np.float64)
        C_hist = np.empty((n_steps + 1, self._n), dtype=np.float64)
        P_hist = np.empty((n_steps + 1, self._n), dtype=np.float64)
        fast = self._fast is not None
        Phi_hist = (np.empty((n_steps + 1, self._n), dtype=np.float64)
                    if fast else None)
        K_hist = (np.empty((n_steps + 1, self._n), dtype=np.float64)
                  if fast else None)
        P_pool_hist = (np.empty((n_steps + 1, self._n), dtype=np.float64)
                       if fast else None)
        P_dist_hist = (np.empty((n_steps + 1, self._n), dtype=np.float64)
                       if fast else None)
        M_hist = (np.empty((n_steps + 1, self._n), dtype=np.float64)
                  if fast else None)
        G_sat_hist = (np.empty((n_steps + 1, self._n), dtype=np.float64)
                      if fast else None)

        times[0] = self._t
        R_hist[0] = self._R
        C_hist[0] = self._C
        P_hist[0] = self._P
        if fast:
            Phi_hist[0] = self._Phi
            K_hist[0] = self._K
            P_pool_hist[0] = self._P_pool
            P_dist_hist[0] = self._P_dist
            M_hist[0] = self._M
            G_sat_hist[0] = self.g_sat

        for k in range(1, n_steps + 1):
            target = t0 + k * dt
            step_dt = dt
            if k == n_steps and target > t_end + 1e-12:
                # Shrink the final step to land exactly on t_end.
                target = t_end
                step_dt = target - self._t
            self._advance(step_dt)
            times[k] = self._t
            R_hist[k] = self._R
            C_hist[k] = self._C
            P_hist[k] = self._P
            if fast:
                Phi_hist[k] = self._Phi
                K_hist[k] = self._K
                P_pool_hist[k] = self._P_pool
                P_dist_hist[k] = self._P_dist
                M_hist[k] = self._M
                G_sat_hist[k] = self.g_sat

        # If a remainder (< dt) still separates us from t_end, take one short step.
        if self._t < t_end - 1e-12:
            self._advance(t_end - self._t)
            times = np.append(times, self._t)
            R_hist = np.vstack([R_hist, self._R])
            C_hist = np.vstack([C_hist, self._C])
            P_hist = np.vstack([P_hist, self._P])
            if fast:
                Phi_hist = np.vstack([Phi_hist, self._Phi])
                K_hist = np.vstack([K_hist, self._K])
                P_pool_hist = np.vstack([P_pool_hist, self._P_pool])
                P_dist_hist = np.vstack([P_dist_hist, self._P_dist])
                M_hist = np.vstack([M_hist, self._M])
                G_sat_hist = np.vstack([G_sat_hist, self.g_sat])

        S_hist = np.empty_like(R_hist)
        T_hist = np.empty_like(R_hist)      # T = R + P_R (conservation observable)
        tens_hist = np.empty_like(R_hist)   # tension = f(R·C) (§2.4)
        for k in range(len(times)):
            S_hist[k] = self.s(R_hist[k], C_hist[k])
            T_hist[k] = R_hist[k] + P_hist[k]
            tens_hist[k] = self.tension(R_hist[k], C_hist[k])

        meta = {
            "method": self._method,
            "dt": self._dt,
            "clamp_nonneg": self._clamp_nonneg,
            "n_nodes": self._n,
            "node_ids": self._node_ids,
            "growth_r": self._a.tolist(),
            "growth_c": self._c.tolist(),
            "alpha0": self._alpha0.tolist(),
            "beta0": self._beta0.tolist(),
            "mu": self._mu.tolist(),
            "lam": self._lam.tolist(),
            "rho": self._rho.tolist(),
            "latent_force0": self._P0.tolist(),
            "panic_exponent": self._panic_exp,
            "kappa_RR": self._kappa_RR.tolist(),
            "kappa_RC": self._kappa_RC.tolist(),
            "kappa_CC": self._kappa_CC.tolist(),
            "kappa_CR": self._kappa_CR.tolist(),
            "meta": self._meta,
        }
        if fast:
            # Fast-channel provenance is appended ONLY in the opt-in mode, so the
            # legacy meta dict is byte-for-byte unchanged.
            meta["fast_channel"] = {
                "enabled": True,
                "provenance": self._fast.provenance(),
                "phi0": self._Phi0.tolist(),
                "k0": self._K0.tolist(),
                "year_period": self._fast.year_period,
                "spring_phase": self._fast.spring_phase,
                "autumn_phase": self._fast.autumn_phase,
                "params": {k: v.tolist() for k, v in self._fc.items()},
            }
        return ForceTrajectory(
            t=times, R=R_hist, C=C_hist, P=P_hist, S=S_hist,
            T=T_hist, tension=tens_hist, meta=meta,
            phi=Phi_hist, k_pool=K_hist,
            p_pool=P_pool_hist, p_dist=P_dist_hist,
            m_cum=M_hist, g_sat=G_sat_hist,
        )

    def reset(self, t0: float = 0.0) -> "ForceFieldDynamics":
        """Restore initial forces (R₀, C₀, P_R,₀), Φ₀, K₀, morphology and time ``t0``. Returns self."""
        self._t = float(t0)
        self._R = self._R0.copy()
        self._C = self._C0.copy()
        self._P = self._P0.copy()
        if self._fast is not None:
            self._Phi = self._Phi0.copy()
            self._K = self._K0.copy()
            self._P_pool = self._P_pool0.copy()
            self._P_dist = self._P_dist0.copy()
            self._M = self._M0.copy()
            self._P = self._P_pool + self._P_dist
        return self

    # ------------------------------------------------------------------
    # Stability criteria (ledger §穩定性判據 — assertable checks)
    # ------------------------------------------------------------------

    def assess_stability(self) -> StabilityRegime:
        """Evaluate the ledger's three stability predicates at the current state.

        - 收斂域 ``βC* + δ > a + Φ*·μ_F``
        - 超臨界分叉 ``Φ*·μ_F + a − δ > βC*`` AND ``λ > 0``
        - 振盪發散 (seasonal loop gain > 1) — an operational ``[DERIVED_PROXY]``
          proxy for the ledger's qualitative "秋收跳躍增益過大＋春耕鉗滯後".

        Equilibria used: Φ* from the Φ logistic (with the current CoopShare /
        DebtStress), C* from the C equation (frozen-R quasi-steady state
        including the −α_rc·R·C contact-surface patch). Only available in the
        opt-in 5-D mode.
        """
        if self._fast is None:
            raise RuntimeError(
                "assess_stability requires the opt-in fast channel; construct "
                "ForceFieldDynamics(..., fast_channel=...) to enable it"
            )
        cfg = self._fast
        fc = self._fc
        _alpha_eff, beta_eff = self.effective_rates(self._t)
        S_arr = self.s()
        T_arr = self.tension()
        a_eff = self._a * _apply_multiplier(self._a_fn, S_arr, T_arr)

        drive = fc["kappa_phi"] * fc["co_share"]
        decay = fc["zeta_phi"] * fc["debt_stress"]
        denom = drive + decay
        phi_star = np.where(denom > 0.0, drive / np.where(denom > 0.0, denom, 1.0), 1.0)
        phi_star = np.clip(phi_star, _PHI_MIN, _PHI_MAX)

        # C-equation quasi-steady state (frozen-R): the contact-surface patch
        # makes C's total dissipation rate γ_C·σ + α_rc·R, so
        # C* = κ_C·(1−Φ*) / (γ_C·σ + α_rc·R_current).
        gc_sig = fc["gamma_c"] * fc["sigma_soviet"] + fc["alpha_rc"] * self._R
        c_star = np.where(
            gc_sig > 0.0,
            fc["kappa_c"] * (1.0 - phi_star) / np.where(gc_sig > 0.0, gc_sig, 1.0),
            0.0,
        )

        conv_margin = beta_eff * c_star + fc["delta"] - (a_eff + phi_star * fc["mu_f"])
        bif_margin = phi_star * fc["mu_f"] + a_eff - fc["delta"] - beta_eff * c_star

        r_cont = a_eff - fc["delta"] + phi_star * fc["mu_f"]
        clamp_loss = np.clip((1.0 - phi_star) * fc["clamp_spring"], 0.0, 1.0)
        autumn_frac = (fc["autumn_yield"] * phi_star * fc["reflow_purity"]
                       / np.maximum(self._R, _S_EPS))
        loop_gain = ((1.0 - clamp_loss) * (1.0 + autumn_frac)
                     * np.exp(r_cont * cfg.year_period))

        seasonal_active = bool(
            np.any(fc["autumn_yield"] * fc["reflow_purity"] > 0.0)
            and np.any(fc["clamp_spring"] > 0.0)
        )
        convergent = bool(np.all(conv_margin > 0.0))
        escape = bool(np.all(bif_margin > 0.0)) and bool(np.any(fc["lam"] > 0.0))
        oscillatory = bool(seasonal_active and np.any(loop_gain > 1.0))

        detail = {
            "a_eff": a_eff.tolist(),
            "beta_eff": beta_eff.tolist(),
            "seasonal_active": seasonal_active,
            "criteria": {
                "convergence": "beta·C* + delta > a + Phi*·mu_F",
                "bifurcation": "Phi*·mu_F + a − delta > beta·C* and lam > 0",
                "oscillation": (
                    "(1−clamp_loss)·(1+autumn_frac)·exp(r_cont·T) > 1 "
                    "[DERIVED_PROXY operationalisation]"
                ),
            },
        }
        return StabilityRegime(
            convergent=convergent,
            escape_poverty_trap=escape,
            oscillatory_divergence=oscillatory,
            phi_star=phi_star,
            c_star=c_star,
            convergence_margin=conv_margin,
            bifurcation_margin=bif_margin,
            seasonal_loop_gain=loop_gain,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # ActorCard integration
    # ------------------------------------------------------------------

    @classmethod
    def from_actor_cards(cls, cards: list[Any], **kwargs: Any) -> "ForceFieldDynamics":
        """Build a field from ActorCards' optional R/C/latent-force fields.

        ``revolutionary_force`` / ``conservative_force`` / ``latent_force``
        (optional contract fields) seed R₀/C₀/P_R,₀; a card missing any falls
        back to 0.0. ``node_ids`` default to the cards' ``actor_id``. All other
        constructor kwargs pass through (growth, alpha/beta, mu/lam/rho, kappa,
        mods, …).
        """
        from spectrum_os.contracts import ActorCard

        cards = list(cards)
        if not cards:
            raise ValueError("from_actor_cards requires at least one ActorCard")
        for card in cards:
            if not isinstance(card, ActorCard):
                raise ValueError(f"expected ActorCard, got {type(card).__name__}")

        R0 = [float(c.revolutionary_force) if c.revolutionary_force is not None else 0.0
              for c in cards]
        C0 = [float(c.conservative_force) if c.conservative_force is not None else 0.0
              for c in cards]
        if any(c.latent_force is not None for c in cards):
            kwargs.setdefault(
                "latent_force",
                [float(c.latent_force) if c.latent_force is not None else 0.0
                 for c in cards],
            )
        kwargs.setdefault("node_ids", [c.actor_id for c in cards])
        return cls(R0, C0, **kwargs)
