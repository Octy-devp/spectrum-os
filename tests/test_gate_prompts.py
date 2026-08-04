"""Tests for synth/gate_prompts.py, unified prompts, and Batch-by-Phase scheduling."""

from __future__ import annotations

import json
import pytest

from spectrum_os.synth.gate_prompts import (
    GATE_SYSTEM_PROMPT_UNIFIED,
    ROUTING_LEAK_KEYWORDS,
    TREE_GENERATE_SYSTEM_PROMPT,
    check_routing_leak_or_schema,
)
from spectrum_os.synth.rate_gate import (
    RateGateContractError,
    rate_gate,
)
from spectrum_os.synth.alt_gate import (
    ADVERSARY_SYSTEM_PROMPT_V1,
    AltGateContractError,
    alt_gate,
)
from spectrum_os.synth.ensemble import run_ensemble


class TestGatePrompts:
    def test_unified_system_prompt_clauses(self):
        """Unified system prompt must contain all three task clauses."""
        assert "When called with [task:rate]" in GATE_SYSTEM_PROMPT_UNIFIED
        assert "When called with [task:generate]" in GATE_SYSTEM_PROMPT_UNIFIED
        assert "When called with [task:adversary]" in GATE_SYSTEM_PROMPT_UNIFIED

    def test_user_message_tag_at_end(self):
        """When unified=True, the routing tag is placed at the end of the user message."""
        calls = []

        def mock_call_api(prompt, api_key, system_message, **kwargs):
            calls.append((prompt, system_message))
            if "[task:rate]" in prompt:
                return json.dumps({
                    "walk": {"crisis": "ok", "lag": "ok", "alive_exits": ["lag"]},
                    "rates": {"lag": 1.0},
                    "evidence": ["ok"]
                })
            elif "[task:generate]" in prompt:
                return json.dumps({
                    "walk": {"crisis": "ok"},
                    "candidates": [{"label": "cand1", "concept_tags": ["t1"], "provenance_hint": "novel", "novel": True}],
                    "evidence": ["ok"]
                })
            elif "[task:adversary]" in prompt:
                return json.dumps({
                    "flagged_labels": [],
                    "reasons": {}
                })
            return "{}"

        sample_rate_input = {
            "legal_exits": ["lag"],
            "realized_counts": {"lag": 1},
        }

        # Rate gate unified
        rate_gate(sample_rate_input, call_api_fn=mock_call_api, api_key="test", n_samples=1, unified=True)
        assert len(calls) == 1
        prompt, sys_msg = calls[0]
        assert sys_msg == GATE_SYSTEM_PROMPT_UNIFIED
        assert prompt.endswith("\n[task:rate]")

        calls.clear()

        sample_alt_input = {"existing_labels": []}
        # Alt gate unified
        alt_gate(sample_alt_input, call_api_fn=mock_call_api, api_key="test", n_samples=1, unified=True)
        assert len(calls) == 2  # generate + adversary
        gen_prompt, gen_sys = calls[0]
        adv_prompt, adv_sys = calls[1]
        assert gen_sys == GATE_SYSTEM_PROMPT_UNIFIED
        assert gen_prompt.endswith("\n[task:generate]")
        assert adv_sys == GATE_SYSTEM_PROMPT_UNIFIED
        assert adv_prompt.endswith("\n[task:adversary]")

    def test_routing_leak_detection_three_cases(self):
        """Routing leak detection test cases: leak keyword, schema mismatch, valid."""
        # Case A: Leak keyword in raw response
        leak_raw = 'Here is the response based on TASK ROUTING: {"walk": {}, "rates": {"lag": 1.0}, "evidence": []}'
        leak_parsed = {"walk": {}, "rates": {"lag": 1.0}, "evidence": []}
        err_a = check_routing_leak_or_schema(leak_raw, leak_parsed, "rate")
        assert err_a is not None
        assert "ROUTING_LEAK_DETECTED" in err_a

        # Case B: Schema mismatch (task:rate receives candidates)
        mismatch_raw = '{"walk": {}, "candidates": [], "evidence": []}'
        mismatch_parsed = {"walk": {}, "candidates": [], "evidence": []}
        err_b = check_routing_leak_or_schema(mismatch_raw, mismatch_parsed, "rate")
        assert err_b is not None
        assert "SCHEMA_MISMATCH" in err_b

        # Case C: Valid response
        valid_raw = '{"walk": {}, "rates": {"lag": 1.0}, "evidence": []}'
        valid_parsed = {"walk": {}, "rates": {"lag": 1.0}, "evidence": []}
        err_c = check_routing_leak_or_schema(valid_raw, valid_parsed, "rate")
        assert err_c is None

    def test_legacy_mode_compatibility(self):
        """When unified=False (default), system prompt and user prompt remain legacy format."""
        calls = []

        def mock_call_api(prompt, api_key, system_message, **kwargs):
            calls.append((prompt, system_message))
            return json.dumps({
                "walk": {"crisis": "ok", "lag": "ok", "alive_exits": ["lag"]},
                "rates": {"lag": 1.0},
                "evidence": ["ok"]
            })

        sample_input = {"legal_exits": ["lag"], "realized_counts": {"lag": 1}}
        rate_gate(sample_input, call_api_fn=mock_call_api, api_key="test", n_samples=1, unified=False)
        assert len(calls) == 1
        prompt, sys_msg = calls[0]
        assert sys_msg != GATE_SYSTEM_PROMPT_UNIFIED
        assert "[task:rate]" not in prompt

    def test_batch_by_phase_execution_order(self):
        """Verify Batch-by-Phase in run_ensemble: all generate calls precede all adversary calls."""
        call_log = []

        def mock_call_api(prompt, api_key, system_message="", **kwargs):
            if "[task:adversary]" in prompt or system_message == ADVERSARY_SYSTEM_PROMPT_V1:
                call_log.append("adversary")
                return json.dumps({"flagged_labels": [], "reasons": {}})
            else:
                call_log.append("generate")
                return json.dumps({
                    "walk": {"crisis": "ok"},
                    "candidates": [{"label": "cand_x", "concept_tags": ["tag_x"], "provenance_hint": "novel", "novel": True}],
                    "evidence": ["ok"]
                })

        # 3 branches, horizon 12, gate_every 4 -> 2 gate points per branch -> 6 gate points total
        res = run_ensemble(
            "crisis",
            n_branches=3,
            horizon=12,
            gate_fn=alt_gate,
            gate_every=4,
            max_gates=5,
            seed=42,
            call_api_fn=mock_call_api,
            api_key="test-key",
            n_samples=1,
            unified=True,
        )

        assert res["meta"]["total_gate_calls"] == 6
        # 6 generate calls followed by 6 adversary calls
        assert len(call_log) == 12
        first_half = call_log[:6]
        second_half = call_log[6:]
        assert all(c == "generate" for c in first_half), f"Expected all generate in Phase 2, got {first_half}"
        assert all(c == "adversary" for c in second_half), f"Expected all adversary in Phase 3, got {second_half}"

    def test_routing_leak_retry_in_gate(self):
        """Rate gate should retry once when leak is detected, and raise RateGateContractError if still leaking."""
        call_count = 0

        def leaky_call_api(prompt, api_key, **kwargs):
            nonlocal call_count
            call_count += 1
            return 'TASK ROUTING: {"walk": {}, "rates": {"lag": 1.0}, "evidence": []}'

        sample_input = {"legal_exits": ["lag"], "realized_counts": {"lag": 1}}
        with pytest.raises(RateGateContractError, match="ROUTING_LEAK_DETECTED"):
            rate_gate(sample_input, call_api_fn=leaky_call_api, api_key="test", n_samples=1, unified=True)

        assert call_count == 2


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2026-08-04：TestTreeGeneratePromptHintSemantics 已刪除——那是調 prompt 過程
# 生成的「文本快照測試」（鎖死 prompt 措辭，不測行為）。prompt 措辭不該被測試
# 綁架；prompt 品質由 ab_prompt_test.py（樹寬/echo/質量）實測。
# 行為測試（routing leak/retry/batch）保留於 TestGatePrompts。
