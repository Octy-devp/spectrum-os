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
]
