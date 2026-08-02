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
    alt_gate_enumerate,
)
from .gate_prompts import (
    GATE_SYSTEM_PROMPT_UNIFIED,
    ROUTING_LEAK_KEYWORDS,
    VERSION_B_SYSTEM_PROMPT,
    check_routing_leak_or_schema,
)
from .ensemble import run_ensemble
from .probe import (
    AXIS_A_VALUES,
    AXIS_B_VALUES,
    NECESSITY_HINT_KEY,
    TreeProbeError,
    assert_tree_gate_contract,
    cluster_archetypes_semantic,
    convergence_view,
    probe_select,
    probe_tree,
    standing_wave_per_layer,
)
from .residual_trigger import (
    load_residual_table,
    make_residual_trigger,
    should_enumerate,
)
__all__ = [
    "GATE_SYSTEM_PROMPT_UNIFIED",
    "VERSION_B_SYSTEM_PROMPT",
    "ROUTING_LEAK_KEYWORDS",
    "check_routing_leak_or_schema",
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
    "alt_gate_enumerate",
    "load_residual_table",
    "make_residual_trigger",
    "should_enumerate",
    "run_ensemble",
    "TREE_GENERATE_SYSTEM_PROMPT",
    "TreeProbeError",
    "assert_tree_gate_contract",
    "standing_wave_per_layer",
    "cluster_archetypes_semantic",
    "convergence_view",
    "probe_tree",
    "probe_select",
]


