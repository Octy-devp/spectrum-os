"""Tests for synth/alt_gate.py (PLAN-23 §5)."""

import json
from unittest.mock import MagicMock

import pytest

from spectrum_os.kernel import verify as _verify
from spectrum_os.synth.alt_gate import (
    AltGateContractError,
    alt_gate,
    assert_alt_gate_contract,
    mechanical_confidence_alt_gate,
)


class TestAltGateContract:
    def test_valid_contract(self):
        valid = {
            "walk": {
                "crisis": "補給限制",
                "lag": "命令延後",
                "alive_space": "邊界調撥機制",
            },
            "candidates": [
                {
                    "label": "糧食調撥站",
                    "concept_tags": ["糧食網絡", "備用通道"],
                    "provenance_hint": "novel",
                    "novel": True,
                }
            ],
            "evidence": ["物資流動延遲"],
        }
        assert_alt_gate_contract(valid)

    def test_non_dict_raises(self):
        with pytest.raises(AltGateContractError, match="must be dict"):
            assert_alt_gate_contract(["not", "a", "dict"])

    def test_forbidden_series_key_raises(self):
        invalid = {
            "walk": {"crisis": "ok"},
            "candidates": [],
            "evidence": [],
            "series": [1, 2, 3],
        }
        with pytest.raises(AltGateContractError, match="forbidden key 'series'"):
            assert_alt_gate_contract(invalid)

    def test_placeholder_refusal_raises(self):
        invalid = {
            "walk": {"crisis": "None"},
            "candidates": [],
            "evidence": [],
        }
        with pytest.raises(AltGateContractError, match="placeholder or shell refusal"):
            assert_alt_gate_contract(invalid)

    def test_prose_key_refusal_raises(self):
        invalid = {
            "walk": {"crisis": "ok"},
            "candidates": [
                {
                    "label": "糧食調撥站",
                    "concept_tags": ["標籤"],
                    "provenance_hint": "novel",
                    "novel": True,
                    "narrative": "這裡是一大段敘事文字描述...",
                }
            ],
            "evidence": ["ok"],
        }
        with pytest.raises(AltGateContractError, match="forbidden prose key 'narrative'"):
            assert_alt_gate_contract(invalid)

    def test_quarantine_filter_raises(self):
        invalid = {
            "walk": {"crisis": "cold_war nuclear escalation"},  # Anachronistic phrase caught by quarantine filter
            "candidates": [],
            "evidence": ["ok"],
        }
        with pytest.raises(AltGateContractError, match="failed mechanical filter"):
            assert_alt_gate_contract(invalid)

    def test_invalid_grammar_transition_raises(self):
        invalid = {
            "walk": {"crisis": "ok"},
            "candidates": [
                {
                    "label": "無效跳躍",
                    "concept_tags": ["標籤"],
                    "provenance_hint": "novel",
                    "novel": True,
                    "from_role": "lag",
                    "to_role": "direction",
                }
            ],
            "evidence": ["ok"],
        }
        with pytest.raises(AltGateContractError, match="invalid per DCA grammar"):
            assert_alt_gate_contract(invalid)

    def test_example_echo_rejection_in_contract(self):
        """assert_alt_gate_contract must raise AltGateContractError('example echo') if label or tag matches prompt example."""
        echo_label_dict = {
            "walk": {"crisis": "ok"},
            "candidates": [
                {
                    "label": "物資替換點",  # In PROMPT_EXAMPLE_LABELS
                    "concept_tags": ["自訂標籤"],
                    "provenance_hint": "novel",
                    "novel": True,
                }
            ],
            "evidence": ["ok"],
        }
        with pytest.raises(AltGateContractError, match="example echo"):
            assert_alt_gate_contract(echo_label_dict)

        echo_tag_dict = {
            "walk": {"crisis": "ok"},
            "candidates": [
                {
                    "label": "自訂標籤",
                    "concept_tags": ["物資網絡"],  # In PROMPT_EXAMPLE_LABELS
                    "provenance_hint": "novel",
                    "novel": True,
                }
            ],
            "evidence": ["ok"],
        }
        with pytest.raises(AltGateContractError, match="example echo"):
            assert_alt_gate_contract(echo_tag_dict)


class TestAltGateExecution:
    def test_mechanical_confidence(self):
        samples = [
            [{"label": "A"}, {"label": "B"}],
            [{"label": "A"}, {"label": "B"}],
        ]
        conf = mechanical_confidence_alt_gate(samples)
        assert conf["score"] == 1.0

        samples_diff = [
            [{"label": "A"}],
            [{"label": "B"}],
        ]
        conf_diff = mechanical_confidence_alt_gate(samples_diff)
        assert conf_diff["score"] == 0.0

    def test_alt_gate_2stage_adversary_and_deduplication(self):
        _verify._state_log.clear()

        gen_response = json.dumps({
            "walk": {
                "crisis": "物資滯留",
                "lag": "命令延遲",
                "alive_space": "備用物資點",
            },
            "candidates": [
                {
                    "label": "舊節點",  # Matches existing_labels
                    "concept_tags": ["網絡"],
                    "provenance_hint": "novel",
                    "novel": True,
                },
                {
                    "label": "偷渡節點",  # Will be flagged by adversary
                    "concept_tags": ["偷渡"],
                    "provenance_hint": "novel",
                    "novel": True,
                },
            ],
            "evidence": ["鐵路滯後"],
        })

        adv_response = json.dumps({
            "flagged_labels": ["偷渡節點"],
            "reasons": {"偷渡節點": "偷渡既定結局"},
        })

        mock_api = MagicMock()
        # First n_samples calls return gen_response, final call returns adv_response
        mock_api.side_effect = [gen_response, gen_response, gen_response, adv_response]

        input_data = {
            "state_vector": {"role": "crisis"},
            "thread_history": ["crisis"],
            "existing_labels": ["舊節點"],
        }

        res = alt_gate(
            input_data,
            call_api_fn=mock_api,
            api_key="mock_key",
            n_samples=3,
        )

        assert mock_api.call_count == 4  # 3 gen + 1 adv

        candidates = res["candidates"]
        assert len(candidates) == 2

        # Check deduplication / novel flag against existing_labels
        cand_old = next(c for c in candidates if c["label"] == "舊節點")
        assert cand_old["novel"] is False

        # Check 2-stage adversary flagging and weight halving
        cand_adv = next(c for c in candidates if c["label"] == "偷渡節點")
        assert cand_adv["adversary_flag"] is True
        assert cand_adv["weight"] == 0.5

        # Check state_log recording
        assert len(_verify._state_log) > 0
        last_entry = _verify._state_log[-1]
        assert last_entry["gate_type"] == "alt_gate"

    def test_reporter_two_stage_call_order_and_state_log(self):
        """Reporter mode (reporter=True) must run observe (prose) -> compress (JSON) -> adversary, saving observation in state_log only."""
        _verify._state_log.clear()
        calls = []

        def mock_call_api(prompt, api_key, system_message="", temperature=0.6, **kwargs):
            calls.append((prompt, temperature))
            if "[task:observe]" in prompt:
                return "1914 年 7 月危機中，奧匈長達48小時的最後通牒正在逼近，鐵路運力限制了外交選擇。"
            elif "[task:compress]" in prompt:
                return json.dumps({
                    "walk": {
                        "crisis": "最後通牒逼近",
                        "lag": "鐵路運力限制",
                        "alive_space": "外交溝通管道"
                    },
                    "candidates": [
                        {
                            "label": "邊界溝通線",
                            "concept_tags": ["外交選擇"],
                            "provenance_hint": "novel",
                            "novel": True,
                        }
                    ],
                    "evidence": ["48小時最後通牒"]
                })
            elif "[task:adversary]" in prompt:
                return json.dumps({
                    "flagged_labels": [],
                    "reasons": {}
                })
            return "{}"

        input_data = {
            "state_vector": {"role": "crisis"},
            "situation": {
                "vector_6d": [0, 0, 0, 12, 0, 0.8],
                "local_texture": {"deadline": 48},
                "digest": "July crisis timeline",
            }
        }

        res = alt_gate(
            input_data,
            call_api_fn=mock_call_api,
            api_key="test_key",
            n_samples=1,
            reporter=True,
            unified=True,
        )

        # 3 calls: observe, compress, adversary
        assert len(calls) == 3
        assert "[task:observe]" in calls[0][0]
        assert calls[0][1] == 0.6  # Hot temp for observe
        assert "[task:compress]" in calls[1][0]
        assert calls[1][1] == 0.1  # Cool temp for compress
        assert "[task:adversary]" in calls[2][0]

        # Check returned dict result does NOT have key 'observation'
        assert "observation" not in res
        assert "walk" in res and "candidates" in res

        # Check state_log entry DOES have observation prose
        assert len(_verify._state_log) > 0
        last_entry = _verify._state_log[-1]
        assert "observation" in last_entry
        assert "48小時的最後通牒" in last_entry["observation"]

    def test_situation_payload_arrival_in_ensemble(self):
        """run_ensemble with situation parameter passes situation payload to gate_input."""
        from spectrum_os.synth.ensemble import run_ensemble
        received_inputs = []

        def mock_gate(gate_input, **kwargs):
            received_inputs.append(gate_input)
            return {
                "walk": {"crisis": "ok"},
                "candidates": [
                    {
                        "label": "valid_candidate",
                        "concept_tags": ["tag1"],
                        "provenance_hint": "novel",
                        "novel": True,
                    }
                ],
                "evidence": ["ok"],
            }

        sit = {
            "vector_6d": [0.55, 0.20, 0.15, 12.0, 0.0, 0.85],
            "local_texture": {"ultimatum_deadline_hours": 48},
            "digest": "1914 July Crisis",
        }

        res = run_ensemble(
            "crisis",
            n_branches=2,
            horizon=8,
            gate_fn=mock_gate,
            gate_every=4,
            max_gates=2,
            seed=42,
            situation=sit,
        )

        assert len(received_inputs) > 0
        for gi in received_inputs:
            assert "situation" in gi
            assert gi["situation"] is not None
            assert gi["situation"]["vector_6d"] == sit["vector_6d"]
            assert gi["situation"]["local_texture"]["ultimatum_deadline_hours"] == 48
            assert "t" in gi["situation"]
            assert "thread_history" in gi["situation"]
