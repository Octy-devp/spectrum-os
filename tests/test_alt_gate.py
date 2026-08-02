"""Tests for synth/alt_gate.py (PLAN-23 §5)."""

import json
from unittest.mock import MagicMock

import pytest

from spectrum_os.kernel import verify as _verify
from spectrum_os.synth.alt_gate import (
    ADVERSARY_SYSTEM_PROMPT_V1,
    AltGateContractError,
    alt_gate,
    alt_gate_enumerate,
    alt_gate_generate,
    assert_alt_gate_contract,
    mechanical_confidence_alt_gate,
    validate_accident,
)
from spectrum_os.synth.gate_prompts import (
    GATE_SYSTEM_PROMPT_UNIFIED,
    check_routing_leak_or_schema,
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

    def test_example_echo_nonfatal_recorded_in_contract(self):
        """語義中介取代黑名單（決策 4）：assert_alt_gate_contract 對 example echo 不再 raise——降為 echo_notes 記錄（非致命）。"""
        echo_label_dict = {
            "walk": {"crisis": "ok"},
            "candidates": [
                {
                    "label": "湧現的替代",  # In PROMPT_EXAMPLE_LABELS
                    "concept_tags": ["自訂標籤"],
                    "provenance_hint": "novel",
                    "novel": True,
                }
            ],
            "evidence": ["ok"],
        }
        # 不傳 echo_notes → 靜默通過（向後相容，不 raise）
        assert_alt_gate_contract(echo_label_dict)
        # 傳 echo_notes → 記錄命中（非致命）
        notes: list[str] = []
        assert_alt_gate_contract(echo_label_dict, echo_notes=notes)
        assert notes and any("example echo" in n for n in notes)

        echo_tag_dict = {
            "walk": {"crisis": "ok"},
            "candidates": [
                {
                    "label": "自訂標籤",
                    "concept_tags": ["繼承的詞彙"],  # In PROMPT_EXAMPLE_LABELS
                    "provenance_hint": "novel",
                    "novel": True,
                }
            ],
            "evidence": ["ok"],
        }
        notes2: list[str] = []
        assert_alt_gate_contract(echo_tag_dict, echo_notes=notes2)
        assert notes2 and any("example echo" in n for n in notes2)


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

    def test_example_echo_no_longer_stripped_or_fatal(self):
        """語義中介取代黑名單（決策 2/4）：example echo 不再 fatal、不再靜默剝離——候選保留 + echo_notes 記錄，零 retry。"""
        gen_response = json.dumps({
            "walk": {"crisis": "補給限制", "lag": "命令延後", "alive_space": "邊界調撥"},
            "candidates": [
                {
                    "label": "湧現的替代",  # In PROMPT_EXAMPLE_LABELS
                    "concept_tags": ["自訂標籤"],
                    "provenance_hint": "novel",
                    "novel": True,
                },
                {
                    "label": "真實出口",
                    "concept_tags": ["替代通道"],
                    "provenance_hint": "novel",
                    "novel": True,
                },
            ],
            "evidence": ["補給線滯後"],
        })
        mock_api = MagicMock()
        mock_api.side_effect = [gen_response]  # 單次成功——無 retry、無剝離
        res = alt_gate_generate(
            {"state_vector": {"role": "crisis"}},
            call_api_fn=mock_api,
            api_key="mock_key",
            n_samples=1,
        )
        assert mock_api.call_count == 1  # 鏡射詞不觸發 retry（retry=馬可夫原地踏步）
        labels = [c["label"] for c in res["aggregated_candidates"]]
        assert "湧現的替代" in labels   # 不再剝離候選
        assert "真實出口" in labels
        assert "echo_notes" in res       # 記錄存在
        assert any("example echo" in n for n in res["echo_notes"])

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


class TestN2AdversarySituationAwareness:
    """Round 2 N2: adversary must receive and judge against the situation."""

    def test_stage1_res_carries_situation_and_adversary_prompt_contains_it(self):
        """stage1 result must carry situation (ensemble stage2 path) and adversary prompt must include it."""
        from spectrum_os.synth.alt_gate import alt_gate_adversary, alt_gate_generate

        calls = []

        def mock_call_api(prompt, api_key, system_message="", temperature=0.6, **kwargs):
            calls.append(prompt)
            if "[task:adversary]" in prompt:
                return json.dumps({"flagged_labels": [], "reasons": {}})
            return json.dumps({
                "walk": {"crisis": "物資滯留", "lag": "命令延遲", "alive_space": "備用物資點"},
                "candidates": [
                    {
                        "label": "調撥節點",
                        "concept_tags": ["網絡"],
                        "provenance_hint": "novel",
                        "novel": True,
                    }
                ],
                "evidence": ["鐵路滯後"],
            })

        situation = {
            "vector_6d": [0, 0, 0, 12, 0, 0.85],
            "local_texture": {"mode": "條約制衡"},
            "digest": "1914 年 7 月危機：鐵路運力制約外交選擇",
        }
        input_data = {
            "state_vector": {"role": "crisis"},
            "thread_history": ["crisis"],
            "existing_labels": [],
            "situation": situation,
        }

        stage1_res = alt_gate_generate(
            input_data, call_api_fn=mock_call_api, api_key="k", n_samples=1, unified=True
        )
        assert stage1_res["situation"] == situation

        alt_gate_adversary(stage1_res, call_api_fn=mock_call_api, api_key="k", unified=True)
        adv_prompt = [p for p in calls if "[task:adversary]" in p][-1]
        assert "1914 年 7 月危機" in adv_prompt
        assert "條約制衡" in adv_prompt

    def test_unified_adversary_clause_is_usage_judge(self):
        """[task:adversary] clause must be usage-judge positioned and situation-grounded."""
        assert "用法裁判" in GATE_SYSTEM_PROMPT_UNIFIED
        assert "以處境為據" in GATE_SYSTEM_PROMPT_UNIFIED
        assert "expression" in GATE_SYSTEM_PROMPT_UNIFIED
        assert "substitution" in GATE_SYSTEM_PROMPT_UNIFIED

    def test_legacy_adversary_prompt_is_usage_judge(self):
        """Legacy ADVERSARY_SYSTEM_PROMPT_V1 must also be usage-judge framed."""
        assert "用法裁判" in ADVERSARY_SYSTEM_PROMPT_V1
        assert "以處境為據" in ADVERSARY_SYSTEM_PROMPT_V1


class TestN3SituationEchoRejection:
    """Round 2 N3: echo rejection referenced by situation labels."""

    def test_label_verbatim_in_situation_labels_raises(self):
        sit_labels = ["條約制衡", "鐵路優先權"]
        echo_dict = {
            "walk": {"crisis": "ok"},
            "candidates": [
                {
                    "label": "條約制衡",  # verbatim in situation_labels
                    "concept_tags": ["自訂標籤"],
                    "provenance_hint": "novel",
                    "novel": True,
                }
            ],
            "evidence": ["ok"],
        }
        with pytest.raises(AltGateContractError, match="situation echo"):
            assert_alt_gate_contract(echo_dict, situation_labels=sit_labels)

    def test_concept_tag_material_reference_allowed(self):
        """concept_tags referencing situation material must NOT raise (revival protection)."""
        sit_labels = ["鐵路優先權"]
        revival_dict = {
            "walk": {"crisis": "ok"},
            "candidates": [
                {
                    "label": "鐵路再調配",  # distinct from situation label
                    "concept_tags": ["繼承的鐵路優先權"],  # material reference — legal
                    "provenance_hint": "novel",
                    "novel": True,
                }
            ],
            "evidence": ["ok"],
        }
        assert_alt_gate_contract(revival_dict, situation_labels=sit_labels)

    def test_no_situation_labels_behavior_unchanged(self):
        """Without situation_labels the echo check is skipped (backwards compatible)."""
        valid = {
            "walk": {"crisis": "ok"},
            "candidates": [
                {
                    "label": "條約制衡",  # would echo only if situation_labels were given
                    "concept_tags": ["標籤"],
                    "provenance_hint": "novel",
                    "novel": True,
                }
            ],
            "evidence": ["ok"],
        }
        assert_alt_gate_contract(valid)

    def test_extract_situation_labels_from_input_data(self):
        """_extract_situation_labels pulls digest + local_texture string values."""
        from spectrum_os.synth.alt_gate import _extract_situation_labels

        input_data = {
            "situation": {
                "vector_6d": [0, 0, 0, 12, 0, 0.85],
                "local_texture": {"mode": "條約制衡", "deadline": 48},
                "digest": "鐵路優先權下的七月危機",
            }
        }
        labels = _extract_situation_labels(input_data)
        assert "鐵路優先權下的七月危機" in labels
        assert "條約制衡" in labels
        assert 48 not in labels

        assert _extract_situation_labels({"situation": None}) is None
        assert _extract_situation_labels({}) is None


class TestVersionBEnumerate:
    """Version B: [task:enumerate] accident schema contract (S7)."""

    def _good_accident(self):
        return {
            "pattern": "軍方-右派政變",
            "instances": ["毛奇政變(1914)", "容克政變(β₁柏林)"],
            "domain": "政治",
            "source": "emergent",
            "usage": "expression",
            "grounding": "合理推測",
            "world_development": "戰爭爆發但德國政治孤立",
        }

    def test_validate_accident_valid(self):
        validate_accident(self._good_accident(), 0)

    def test_validate_accident_bad_domain(self):
        bad = self._good_accident()
        bad["domain"] = "天氣突變"
        with pytest.raises(AltGateContractError, match="domain"):
            validate_accident(bad, 0)

    def test_validate_accident_bad_grounding(self):
        bad = self._good_accident()
        bad["grounding"] = "不確定"
        with pytest.raises(AltGateContractError, match="grounding"):
            validate_accident(bad, 0)

    def test_validate_accident_bad_pattern(self):
        bad = self._good_accident()
        bad["pattern"] = ""
        with pytest.raises(AltGateContractError, match="pattern"):
            validate_accident(bad, 0)

    def test_validate_accident_missing_world_development(self):
        bad = self._good_accident()
        bad["world_development"] = ""
        with pytest.raises(AltGateContractError, match="world_development"):
            validate_accident(bad, 0)

    def test_enumerate_schema_check(self):
        assert check_routing_leak_or_schema('{}', {"accidents": []}, "enumerate") is None
        err = check_routing_leak_or_schema('{}', {"candidates": []}, "enumerate")
        assert err is not None and "enumerate" in err
        err2 = check_routing_leak_or_schema('{}', {"rates": {}}, "enumerate")
        assert err2 is not None and "enumerate" in err2

    def test_alt_gate_enumerate_mock(self):
        """alt_gate_enumerate calls API with version B prompt + [task:enumerate], validates schema."""
        from spectrum_os.synth.alt_gate import alt_gate_enumerate

        mock = MagicMock()
        mock.return_value = json.dumps(
            {
                "accidents": [
                    {
                        "pattern": "蒂薩阻止對塞戰爭",
                        "instances": ["蒂薩在匈牙利議會否決戰爭"],
                        "domain": "政治",
                        "source": "inherited",
                        "usage": "expression",
                        "grounding": "文獻偶發",
                        "world_development": "危機降溫，奧匈內部民族矛盾激化",
                    }
                ]
            },
            ensure_ascii=False,
        )
        res = alt_gate_enumerate(
            {"situation": {"digest": "測試處境"}},
            call_api_fn=mock,
            api_key="test-key",
        )
        assert len(res["accidents"]) == 1
        assert res["accidents"][0]["pattern"] == "蒂薩阻止對塞戰爭"
        assert res["generated_by"]["gate"] == "alt_gate_enumerate"
        args = mock.call_args
        assert "[task:enumerate]" in args.args[0]
        assert "歷史的意外" in args.kwargs["system_message"]

    def test_alt_gate_enumerate_contract_reject(self):
        """alt_gate_enumerate raises when schema is wrong."""
        from spectrum_os.synth.alt_gate import alt_gate_enumerate

        mock = MagicMock()
        mock.return_value = json.dumps({"candidates": [{"label": "x"}]}, ensure_ascii=False)
        with pytest.raises(AltGateContractError):
            alt_gate_enumerate({"situation": {}}, call_api_fn=mock, api_key="test-key")

    def test_invalid_json_normalized_to_alt_gate_contract_error(self):
        """Malformed JSON from the API must surface as AltGateContractError (not AnchorContractError)."""
        mock = MagicMock()
        mock.return_value = "{not json"
        with pytest.raises(AltGateContractError):
            alt_gate_enumerate({"situation": {}}, call_api_fn=mock, api_key="test-key")

    def test_retry_after_parse_failure(self):
        """First attempt returns malformed JSON, second returns valid accidents — retry must succeed."""
        good = json.dumps(
            {
                "accidents": [
                    {
                        "pattern": "鐵路罷工蔓延",
                        "instances": ["莫斯科樞紐癱瘓"],
                        "domain": "社會",
                        "source": "emergent",
                        "usage": "expression",
                        "grounding": "合理推測",
                        "world_development": "罷工網由鐵路節點向工業城市擴散",
                    }
                ]
            },
            ensure_ascii=False,
        )
        mock = MagicMock()
        mock.side_effect = ["{not json", good]
        res = alt_gate_enumerate({"situation": {}}, call_api_fn=mock, api_key="test-key")
        assert len(res["accidents"]) == 1
        assert mock.call_count == 2
        temps = [c.kwargs.get("temperature") for c in mock.call_args_list]
        assert temps[1] > temps[0]

    def test_validate_accident_quarantine_placeholder(self):
        """validate_accident rejects pattern='None' via quarantine (placeholder / shell refusal)."""
        bad = self._good_accident()
        bad["pattern"] = "None"
        with pytest.raises(AltGateContractError, match="placeholder|shell"):
            validate_accident(bad, 0)

    def test_validate_accident_empty_instances(self):
        """validate_accident rejects an empty instances list (burst mode requires >= 1 instance)."""
        bad = self._good_accident()
        bad["instances"] = []
        with pytest.raises(AltGateContractError, match="instances"):
            validate_accident(bad, 0)

    def test_enumerate_schema_rejects_flagged_labels(self):
        """[task:enumerate] schema check must reject flagged_labels contamination (mirrors adversary)."""
        err = check_routing_leak_or_schema(
            '{}', {"accidents": [], "flagged_labels": []}, "enumerate"
        )
        assert err is not None and "enumerate" in err


# ---------------------------------------------------------------------------
# §12.13：Mode A → Mode B 依賴（後驗初始化接線，純機械部分）
# ---------------------------------------------------------------------------

class TestModeAPosterior:
    def test_record_and_load_roundtrip(self, tmp_path):
        from spectrum_os.synth.alt_gate import (
            load_mode_a_history,
            record_mode_a_history,
        )
        p = str(tmp_path / "mode_a.jsonl")
        record_mode_a_history(
            p, {"ts": "t1", "mode": "markov", "role_sequence": ["crisis", "lag"]}
        )
        record_mode_a_history(
            p, {"ts": "t2", "mode": "spectrum", "labels": ["動員令凍結"]}
        )
        hist = load_mode_a_history(p)
        assert len(hist) == 2
        assert hist[0]["ts"] == "t1" and hist[1]["ts"] == "t2"

    def test_load_skips_corrupt_lines(self, tmp_path):
        from spectrum_os.synth.alt_gate import load_mode_a_history
        p = tmp_path / "mode_a.jsonl"
        p.write_text('{"ts": "ok"}\nnot-json\n{"ts": "ok2"}\n', encoding="utf-8")
        hist = load_mode_a_history(str(p))
        assert [h["ts"] for h in hist] == ["ok", "ok2"]

    def test_load_missing_file_returns_empty(self, tmp_path):
        from spectrum_os.synth.alt_gate import load_mode_a_history
        assert load_mode_a_history(str(tmp_path / "nope.jsonl")) == []

    def test_posterior_init_dominant_transitions(self):
        from spectrum_os.synth.alt_gate import mode_a_posterior_init
        rate = [
            [0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 0.8, 0.2],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        hist = [{"mode": "markov", "rate_matrix": {"matrix": rate},
                 "role_sequence": ["crisis", "lag"]}]
        posterior = mode_a_posterior_init(hist)
        assert posterior["n_entries"] == 1
        assert posterior["source"] == "mode_a_history"
        assert posterior["rate_matrix"]["matrix"] == rate
        dom = posterior["rate_matrix"]["dominant_transitions"]
        assert dom and dom[0]["rate"] >= 0.8
        # 角色序列 → 馬可夫轉移計數 4x4
        assert posterior["role_statistics"]["transition_counts"] is not None
        assert len(posterior["role_statistics"]["transition_counts"]) == 4

    def test_posterior_init_accumulates_labels(self):
        from spectrum_os.synth.alt_gate import mode_a_posterior_init
        hist = [
            {"labels": ["動員令凍結", "地方調撥自主"]},
            {"labels": ["動員令凍結", "國際調停介入"]},
        ]
        posterior = mode_a_posterior_init(hist)
        assert posterior["accumulated_labels"] == [
            "動員令凍結", "地方調撥自主", "國際調停介入",
        ]
        assert "PROMPT-DEPENDENT" in posterior["annotation"]

    def test_posterior_init_empty(self):
        from spectrum_os.synth.alt_gate import mode_a_posterior_init
        posterior = mode_a_posterior_init([])
        assert posterior["n_entries"] == 0

    def test_enumerate_injects_mode_a_history_data(self):
        from spectrum_os.synth.gate_prompts import VERSION_B_SYSTEM_PROMPT
        mock = MagicMock()
        mock.return_value = json.dumps({
            "accidents": [{
                "pattern": "蒂薩阻止對塞戰爭",
                "instances": ["蒂薩在匈牙利議會否決戰爭"],
                "domain": "政治",
                "source": "inherited",
                "usage": "expression",
                "grounding": "文獻偶發",
                "world_development": "危機降溫",
            }]
        }, ensure_ascii=False)
        hist = [{"mode": "markov",
                 "rate_matrix": [[0.0, 1.0, 0.0, 0.0]] * 4,
                 "labels": ["動員令凍結"]}]
        res = alt_gate_enumerate(
            {"situation": {"digest": "測試處境"}},
            call_api_fn=mock, api_key="test-key", mode_a_history=hist,
        )
        args = mock.call_args
        prompt = args.args[0]
        assert "mode_a_history" in prompt          # 資料層注入（非 prompt 措辭）
        assert "[task:enumerate]" in prompt        # 既有 prompt 字串保留
        assert args.kwargs["system_message"] == VERSION_B_SYSTEM_PROMPT  # 未改 prompt
        assert res["mode_a_posterior"]["n_entries"] == 1

    def test_enumerate_loads_from_log_path(self, tmp_path):
        from spectrum_os.synth.alt_gate import record_mode_a_history
        p = str(tmp_path / "mode_a.jsonl")
        record_mode_a_history(p, {"mode": "spectrum", "labels": ["動員令凍結"]})
        mock = MagicMock()
        mock.return_value = json.dumps({
            "accidents": [{
                "pattern": "意外甲", "instances": ["實例"], "domain": "政治",
                "source": "emergent", "usage": "expression",
                "grounding": "合理推測", "world_development": "發展",
            }]
        }, ensure_ascii=False)
        res = alt_gate_enumerate(
            {"situation": {}}, call_api_fn=mock, api_key="k",
            mode_a_log_path=p,
        )
        assert res["mode_a_posterior"]["accumulated_labels"] == ["動員令凍結"]

    def test_without_mode_a_backward_compat(self):
        mock = MagicMock()
        mock.return_value = json.dumps({
            "accidents": [{
                "pattern": "意外甲", "instances": ["實例"], "domain": "自然",
                "source": "emergent", "usage": "expression",
                "grounding": "弱支持", "world_development": "發展",
            }]
        }, ensure_ascii=False)
        res = alt_gate_enumerate({"situation": {}}, call_api_fn=mock, api_key="k")
        assert "mode_a_history" not in mock.call_args.args[0]
        assert res["mode_a_posterior"] is None
