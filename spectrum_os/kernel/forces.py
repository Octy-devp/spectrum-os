"""Spectrum OS Kernel — R/C counterforce field engine (social dynamics first theorem).

Methodology: ECC ``docs/method/force-model-social-dynamics.md`` (v1.3, 2026-08-04).
Constitutional-level, worldline-agnostic spec: any society's "content" is not a
statistically pre-measurable label — it is the resultant vector S(t) of its
internal transformative force R (revolution) vs inertial force C (counter-
revolution), run to its deterministic end.

Core equations (§2.2 + §2.2b — the equations are the SSOT, see sign convention):

    dR_i/dt = a_i·R_i − β_i·C_i + λ_i·P_R,i − μ_i·R_i   (+ coupling, §2.6)
    dP_R,i/dt = μ_i·R_i − λ_i·P_R,i − ρ_i·P_R,i
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

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Degenerate guard for S = R/(R+C): when both forces vanish the "content" is
# undecided (a dead/zero-tension state), pinned to 0.5 rather than 0/0 → NaN.
_S_EPS = 1e-12

_SUPPORTED_METHODS = ("rk4", "euler")


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

    def to_dict(self) -> dict:
        """Serialize to plain JSON-safe dict (numpy arrays → nested lists)."""
        return {
            "t": self.t.tolist(),
            "R": self.R.tolist(),
            "C": self.C.tolist(),
            "P": self.P.tolist(),
            "S": self.S.tolist(),
            "T": self.T.tolist(),
            "tension": self.tension.tolist(),
            "meta": self.meta,
        }

    def final_s(self) -> float:
        """Resultant S at the last step (single-node convenience: ``float``)."""
        if self.S.ndim == 1:
            return float(self.S[-1])
        return float(self.S[-1, 0])


class ForceFieldDynamics:
    """Deterministic R/C counterforce field engine (no LLM dependency).

    Integrates the §2.2 conflict ODEs for ``n`` coupled nodes with the §2.2b P_R
    potential reservoir, §2.5 external field modulation and §2.6 four-quadrant
    full-sign coupling. ``S = R/(R+C)`` is the social content resultant;
    ``T = R + P_R`` is the total transformative capacity (conservation H₀);
    ``tension = f(R·C)`` is the conflict intensity.

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
        P_R reservoir rates (§2.2b) — compression μ (kinetic R → latent P_R),
        release λ (P_R → R under external-field modulation / C-cracks), decay ρ
        (genuinely forgotten history). Scalar or length-``n``; default 0
        (reservoir disabled ⇒ v1.0 behaviour, backward compatible).
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
        tension_fn: Callable | None = None,
        method: str = "rk4",
        dt: float = 1.0,
        clamp_nonneg: bool = True,
        node_ids: list[str] | None = None,
        meta: dict | None = None,
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
            dP = μ·R − λ·P − ρ·P
            dC = c·C − α_eff·R + κ_CC(C_j−C_i) + κ_RC(R_j−R_i)·R_j^p

        where ``p = panic_exponent`` (0 ⇒ the κ_RC term is linear). Defaults to
        the current engine state for R/C/P_R.
        """
        R_arr = self._R if R is None else _as_float_array(R, "R")
        C_arr = self._C if C is None else _as_float_array(C, "C")
        P_arr = self._P if P is None else _as_float_array(P, "P")
        alpha_eff, beta_eff = self.effective_rates(t)

        dR = self._a * R_arr - beta_eff * C_arr + self._lam * P_arr - self._mu * R_arr
        dC = self._c * C_arr - alpha_eff * R_arr
        dP = self._mu * R_arr - (self._lam + self._rho) * P_arr

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
    # Integration
    # ------------------------------------------------------------------

    def _stage_derivatives(self, t: float, R: np.ndarray, C: np.ndarray, P: np.ndarray,
                           dR: np.ndarray, dC: np.ndarray, dP: np.ndarray,
                           dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return full derivatives at a mid/last stage for RK4 (incl. P_R)."""
        return self.derivatives_full(t, R + dt * dR, C + dt * dC, P + dt * dP)

    def _advance(self, dt: float) -> None:
        """Advance the internal state by one step of ``dt`` using ``method``."""
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

        Snapshot keys: ``t``, ``R``, ``C``, ``S``, ``T`` (numpy arrays).
        """
        self._advance(self._dt if dt is None else float(dt))
        return {
            "t": self._t,
            "R": self._R.copy(),
            "C": self._C.copy(),
            "P": self._P.copy(),
            "S": self.s(),
            "T": self.total_capacity(),
            "tension": self.tension(),
        }

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

        times[0] = self._t
        R_hist[0] = self._R
        C_hist[0] = self._C
        P_hist[0] = self._P

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

        # If a remainder (< dt) still separates us from t_end, take one short step.
        if self._t < t_end - 1e-12:
            self._advance(t_end - self._t)
            times = np.append(times, self._t)
            R_hist = np.vstack([R_hist, self._R])
            C_hist = np.vstack([C_hist, self._C])
            P_hist = np.vstack([P_hist, self._P])

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
        return ForceTrajectory(
            t=times, R=R_hist, C=C_hist, P=P_hist, S=S_hist,
            T=T_hist, tension=tens_hist, meta=meta,
        )

    def reset(self, t0: float = 0.0) -> "ForceFieldDynamics":
        """Restore initial forces (R₀, C₀, P_R,₀) and time ``t0``. Returns self."""
        self._t = float(t0)
        self._R = self._R0.copy()
        self._C = self._C0.copy()
        self._P = self._P0.copy()
        return self

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
