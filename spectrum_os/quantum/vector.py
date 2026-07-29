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
from dataclasses import dataclass, field
from typing import Any

_ROLE_ORDER: tuple[str, ...] = ("direction", "crisis", "lag", "alternative")


@dataclass
class StateVector:
    """6D state vector: [D1 will, D2 resistance, D3 relation,
                         D4 frequency, D5 phase, D6 amplitude]."""
    d1: int = 0   # will: −1/0/+1
    d2: int = 0   # resistance: −1/0/+1
    d3: int = 0   # relation: −1/0/+1
    d4: float = 0.0  # frequency (dominant period in months, ≥ 0)
    d5: float = 0.0  # phase (radians, [-π, π])
    d6: float = 0.0  # amplitude (normalized, [0, 1])

    def __post_init__(self) -> None:
        """Validate and normalise range contracts after construction.

        D1–D3: strictly −1, 0, or +1 (ternary direction).
        D4: non-negative (frequency).
        D5: auto-wrapped to [-π, π] via atan2 — phase is periodic;
            any real value is accepted and normalised, not rejected.
        D6: [0, 1] (amplitude).
        """
        for attr, name in [(self.d1, "d1"), (self.d2, "d2"), (self.d3, "d3")]:
            if attr not in (-1, 0, 1):
                raise ValueError(f"{name} must be −1, 0, or +1, got {attr}")
        if self.d4 < 0:
            raise ValueError(f"d4 (frequency) must be ≥ 0, got {self.d4}")
        # Phase wrapping: sin/cos preserve equivalence; atan2 normalises to [-π, π]
        self.d5 = math.atan2(math.sin(self.d5), math.cos(self.d5))
        if not (0 <= self.d6 <= 1):
            raise ValueError(f"d6 (amplitude) must be in [0, 1], got {self.d6}")

    def norm(self) -> float:
        """Euclidean norm of the full 6D vector."""
        return math.sqrt(
            self.d1 * self.d1 + self.d2 * self.d2 + self.d3 * self.d3 +
            self.d4 * self.d4 + self.d5 * self.d5 + self.d6 * self.d6
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "d1": self.d1, "d2": self.d2, "d3": self.d3,
            "d4": self.d4, "d5": self.d5, "d6": self.d6,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StateVector:
        """Deserialize from a plain dict (no normalisation — trust the stored value)."""
        return object.__new__(cls).__manual_init__(
            d1=int(d["d1"]), d2=int(d["d2"]), d3=int(d["d3"]),
            d4=float(d["d4"]), d5=float(d["d5"]), d6=float(d["d6"]),
        )

    def __manual_init__(self, d1, d2, d3, d4, d5, d6):
        """Bypass __post_init__ for deserialization."""
        self.d1, self.d2, self.d3 = d1, d2, d3
        self.d4, self.d5, self.d6 = d4, d5, d6
        return self

    def __repr__(self) -> str:
        return (f"StateVector(D1={self.d1:+d}, D2={self.d2:+d}, "
                f"D3={self.d3:+d}, D4={self.d4:.1f}, "
                f"D5={self.d5:+.2f}, D6={self.d6:.3f})")


def project_ternary(role_distribution: dict[str, float]) -> tuple[int, int, int]:
    """Project a 4-role distribution onto D1–D3 ternary direction.

    Parameters
    ----------
    role_distribution
        Dict mapping role names to weights, e.g.
        ``{"direction": 0.4, "crisis": 0.3, "lag": 0.2, "alternative": 0.1}``.
        Weights are normalized to sum 1 internally.

    Returns
    -------
    (d1, d2, d3) where each is −1, 0, or +1.
      - d1 = sign(direction - crisis) → will
      - d2 = sign(lag - 0.25)       → resistance (threshold at 0.25)
      - d3 = sign(alternative - mean(crisis, lag, direction)) → relation
    """
    d = role_distribution.get("direction", 0.0)
    c = role_distribution.get("crisis", 0.0)
    l = role_distribution.get("lag", 0.0)
    a = role_distribution.get("alternative", 0.0)

    # Normalize to sum 1
    total = d + c + l + a
    if total <= 0:
        return (0, 0, 0)
    d, c, l, a = d / total, c / total, l / total, a / total

    d1 = _sign_int(d - c)                           # will: direction minus crisis
    d2 = _sign_int(l - 0.25)                         # resistance: lag vs threshold
    mean_others = (c + l + d) / 3.0
    d3 = _sign_int(a - mean_others)                  # relation: alternative vs mean of others

    return (d1, d2, d3)


def vector_from_roles(role_distribution: dict[str, float],
                      spectrum_data: dict[str, float] | None = None) -> StateVector:
    """Build a full 6D StateVector from a 4-role distribution and optional
    spectrum data.

    Parameters
    ----------
    role_distribution
        Dict mapping role names to weights (``direction``, ``crisis``,
        ``lag``, ``alternative``).
    spectrum_data
        Optional dict with keys ``freq``, ``phase``, ``amplitude``.
        If None, D4–D6 default to 0.0.

    Returns
    -------
    StateVector with D1–D3 from ternary projection and D4–D6 from spectrum.
    """
    d1, d2, d3 = project_ternary(role_distribution)
    if spectrum_data is not None:
        d4 = float(spectrum_data.get("freq", 0.0))
        d5 = float(spectrum_data.get("phase", 0.0))
        d6 = float(spectrum_data.get("amplitude", 0.0))
    else:
        d4, d5, d6 = 0.0, 0.0, 0.0
    return StateVector(d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, d6=d6)


def norm(vec: StateVector) -> float:
    """Euclidean norm of a 6D state vector.

    This is a standalone function; ``StateVector.norm()`` is also available
    as a method.
    """
    return math.sqrt(
        vec.d1 * vec.d1 + vec.d2 * vec.d2 + vec.d3 * vec.d3 +
        vec.d4 * vec.d4 + vec.d5 * vec.d5 + vec.d6 * vec.d6
    )


def to_array(vec: StateVector) -> list[float]:
    """Return [d1..d6] as floats."""
    return [float(vec.d1), float(vec.d2), float(vec.d3),
            vec.d4, vec.d5, vec.d6]


def _sign_int(x: float) -> int:
    """Sign function returning −1, 0, or +1."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0
