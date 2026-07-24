from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional


class Verdict(Enum):
    ASSERTED = "asserted"    # data sufficient, stable pattern
    CONTESTED = "contested"  # pattern inconsistent, needs attention
    UNKNOWN = "unknown"      # insufficient data or cannot determine


@dataclass
class SpectrumResult:
    """Unified return type for all API operations."""

    values: dict[str, Any]            # actual numerical results
    verdict: Verdict                  # ternary confidence
    confidence_reason: str            # human-readable explanation
    dominant_periods: list[int] = field(default_factory=list)
    valid_range: tuple[int, int] | None = None  # (start, end) of valid data
    boundary_note: str | None = None  # near-threshold warning


@dataclass
class Sector:
    id: str
    name: str
    timeseries: list[float]
    targets: list[float] | None
    created: str  # ISO date string
