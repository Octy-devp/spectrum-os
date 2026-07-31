"""synth — LLM-gated synthetic sector layer (PLAN-23 §7.3, 方案 A).

Deliberately **outside** ``kernel/``: the kernel stays numpy-only.  This
package is the only place where an LLM touches sector data, and it does so
under the anti-contamination charter (§六․五): the LLM produces structured
anchors only; numpy does all dynamics; every output is marked synthetic and
verdict-capped downstream.
"""

from .anchors import (
    ANCHORS_SYSTEM_PROMPT_V25,
    CONTRACT_TOP_KEYS,
    AnchorContractError,
    anchors,
    assert_contract,
)
from .expand import (
    expand,
    expand_pair,
    expand_plan,
    register_synthetic_sector,
)
from .rate_gate import (
    RATE_GATE_SYSTEM_PROMPT_V1,
    RateGateContractError,
    assert_rate_gate_contract,
    mechanical_confidence_rate_gate,
    rate_gate,
)
from .alt_gate import (
    ALT_GATE_SYSTEM_PROMPT_V1,
    ADVERSARY_SYSTEM_PROMPT_V1,
    AltGateContractError,
    assert_alt_gate_contract,
    mechanical_confidence_alt_gate,
    alt_gate,
)
from .ensemble import run_ensemble

__all__ = [
    "ANCHORS_SYSTEM_PROMPT_V25",
    "CONTRACT_TOP_KEYS",
    "AnchorContractError",
    "anchors",
    "assert_contract",
    "expand",
    "expand_pair",
    "expand_plan",
    "register_synthetic_sector",
    "RATE_GATE_SYSTEM_PROMPT_V1",
    "RateGateContractError",
    "assert_rate_gate_contract",
    "mechanical_confidence_rate_gate",
    "rate_gate",
    "ALT_GATE_SYSTEM_PROMPT_V1",
    "ADVERSARY_SYSTEM_PROMPT_V1",
    "AltGateContractError",
    "assert_alt_gate_contract",
    "mechanical_confidence_alt_gate",
    "alt_gate",
    "run_ensemble",
]

