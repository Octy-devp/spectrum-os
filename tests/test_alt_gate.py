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
                "crisis": "補給線斷裂",
                "lag": "動員法令未完成",
                "alive_space": "邊界物資替換機制",
            },
            "candidates": [
                {
                    "label": "物資替換點",
                    "concept_tags": ["物資網絡", "替代通道"],
                    "provenance_hint": "novel",
                    "novel": True,
                }
            ],
            "evidence": ["物資流動滯後"],
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
                    "label": "物資替換點",
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
                    "to_role": "direction",  # Forbidden by DCA grammar (lag -> direction)
                }
            ],
            "evidence": ["ok"],
        }
        with pytest.raises(AltGateContractError, match="invalid per DCA grammar"):
            assert_alt_gate_contract(invalid)


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
