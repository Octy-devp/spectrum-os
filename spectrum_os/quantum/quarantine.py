"""
Double-layer anti-pollution quarantine system (PLAN-23 §六.五-2).

Layer 1: Mechanical blacklist — blocks α₁ concepts that cannot exist before
their historical emergence date.
Layer 2: Semantic era-consistency check — performed by the adversary gate
(PLAN-23 §六․五-5).  Since PLAN-23 v2.6.1 there is NO separate second-LLM
inspector: gate output is already compressed to structure/numbers, so
concept smuggling survives only in alternative labels, which is exactly
the adversary gate's range.  `inspector_gate()` here remains as the
mechanical fallback (emergence-year check).

This is the output-side quarantine: after generation, before ingestion into
β₁ SSOT or spectrum computation.  Complemented by input-side knowledge
cutoff (§六.五-2).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Layer 1: mechanical blacklist
# ---------------------------------------------------------------------------

TEMPORAL_BLACKLIST: list[str] = [
    # α₁ concepts that cannot exist before their historical emergence.
    # Each entry must have a corresponding entry in EMERGENCE_YEAR.
    "total_war", "nuclear", "blitzkrieg", "fascism", "nazi",
    "holocaust", "atomic", "missile", "cold_war", "nato",
    "united_nations", "decolonization", "superpower",
]

# Earliest plausible year for each concept in α₁.
# These are conservative lower bounds (first recorded use or emergence).
EMERGENCE_YEAR: dict[str, int] = {
    "total_war": 1917,        # Ludendorff's concept, post-1916
    "nuclear": 1938,          # nuclear fission discovery
    "blitzkrieg": 1935,       # first used in military literature
    "fascism": 1919,          # Fasci Italiani di Combattimento
    "nazi": 1920,             # NSDAP founding
    "holocaust": 1941,        # term applied to Shoah
    "atomic": 1938,           # fission
    "missile": 1944,          # V-2
    "cold_war": 1945,         # Orwell 1945, popularised 1947
    "nato": 1949,
    "united_nations": 1942,   # Declaration by United Nations
    "decolonization": 1950,   # entered common usage post-WWII
    "superpower": 1944,       # W.T.R. Fox
}


def mechanical_filter(text: str) -> tuple[bool, list[str]]:
    """Check *text* against the temporal blacklist (case-insensitive).

    Returns:
        (is_clean, matched_terms) — *is_clean* is ``True`` when no
        blacklisted term appears; *matched_terms* lists any that were found.
    """
    lower = text.lower()
    matched: list[str] = []
    for term in TEMPORAL_BLACKLIST:
        if term in lower:
            matched.append(term)
    return (len(matched) == 0, matched)


# ---------------------------------------------------------------------------
# Layer 2: inspector gate (mechanical fallback)
# ---------------------------------------------------------------------------

def inspector_gate(text: str, year: int) -> dict:
    """Second-layer inspector gate — mechanical fallback.

    The full semantic inspector is NOT a separate LLM: since PLAN-23 v2.6.1
    the era-consistency attack is performed by the adversary gate
    (§六․五-5).  What remains here is the mechanical fallback: check each
    blacklisted concept's emergence year against the given *year*.

    Returns:
        ``{"passed": bool, "anachronisms": [str], "confidence": float,
           "reason": str}``
    """
    lower = text.lower()
    anachronisms: list[str] = []
    for term in TEMPORAL_BLACKLIST:
        if term in lower and year < EMERGENCE_YEAR.get(term, 2100):
            anachronisms.append(term)

    if anachronisms:
        return {
            "passed": False,
            "anachronisms": anachronisms,
            "confidence": 0.0,
            "reason": "mechanical_fallback",
        }
    return {
        "passed": True,
        "anachronisms": [],
        "confidence": 0.0,
        "reason": "mechanical_fallback",
    }


# ---------------------------------------------------------------------------
# Full dual-layer quarantine
# ---------------------------------------------------------------------------

def quarantine_check(text: str, year: int) -> dict:
    """Run the full dual-layer quarantine pipeline.

    1. Mechanical blacklist (Layer 1) — auto-fail on any match.
    2. Inspector gate (Layer 2) — emergence-year check.

    Returns:
        ``{"passed": bool, "layer1": (ok, matches),
          "layer2": {...inspector verdict...}}``
    """
    layer1_ok, layer1_matches = mechanical_filter(text)
    layer2 = inspector_gate(text, year)

    return {
        "passed": layer1_ok and layer2["passed"],
        "layer1": (layer1_ok, layer1_matches),
        "layer2": layer2,
    }
