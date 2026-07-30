"""Tests for the Rate Gate layer (synth/rate_gate.py).

All tests use injected mock API functions — 0 real API calls.
"""

from __future__ import annotations

import json
import numpy as np
import pytest

from spectrum_os.synth import (
    RATE_GATE_SYSTEM_PROMPT_V1,
    RateGateContractError,
    assert_rate_gate_contract,
    mechanical_confidence_rate_gate,
    rate_gate,
)
from scripts.rate_gate_liveness import run_liveness_diagnostic


def _sample_input() -> dict:
    return {
        "state_vector": {"d1": 0, "d2": 1, "d3": -1, "d4": 12.0, "d5": 0.3, "d6": 0.8},
        "thread_history": {"months_in_current_role": 14, "recent_transitions": ["crisis", "lag", "lag"]},
        "local_texture": {"steel_output_tons": 17600000, "railway_delays_days": 3},
        "legal_exits": ["crisis", "alternative", "lag"],
        "realized_counts": {"lag": 5, "alternative": 2, "crisis": 1},
    }


def _sample_llm_response() -> str:
    return json.dumps({
        "walk": {
            "crisis": "補給線斷裂",
            "lag": "動員法令未完成",
            "alive_exits": ["lag", "alternative"],
        },
        "rates": {
            "lag": 0.55,
            "alternative": 0.35,
            "crisis": 0.10,
        },
        "evidence": ["鐵路延遲3日", "角色已持續14月"],
    })


def _mock_api(response_text: str):
    def call_api_fn(prompt, api_key, **kwargs):
        return response_text
    return call_api_fn


class TestRateGate:
    def test_rate_gate_basic_mock(self):
        res = rate_gate(
            _sample_input(),
            call_api_fn=_mock_api(_sample_llm_response()),
            api_key="test-key",
            n_samples=3,
            w_prior=5.0,
        )
        assert "fused_rates" in res
        assert "llm_mean_rates" in res
        assert "confidence" in res
        assert res["generated_by"]["gate"] == "synth.rate_gate"
        assert res["generated_by"]["n_samples"] == 3
        # Sum of fused_rates should be 1.0
        np.testing.assert_allclose(sum(res["fused_rates"].values()), 1.0)

    def test_rate_gate_confidence_overwritten(self):
        """FIX-21: LLM self-rating must be discarded and recomputed from n-sample dispersion."""
        response = json.loads(_sample_llm_response())
        response["confidence"] = {"score": 0.99, "note": "LLM self rating"}
        res = rate_gate(
            _sample_input(),
            call_api_fn=_mock_api(json.dumps(response)),
            api_key="test-key",
            n_samples=3,
        )
        conf = res["confidence"]
        assert "score" in conf
        assert conf["basis"] == "n_sample_dispersion"
        assert conf.get("note") != "LLM self rating"

    def test_rate_gate_forbidden_zeroing(self):
        """Forbidden or non-legal exits must be zeroed out in fused rates."""
        response = json.loads(_sample_llm_response())
        # Add forbidden exit "direction" (legal_exits is crisis, alternative, lag)
        response["rates"]["direction"] = 0.50
        with pytest.raises(RateGateContractError, match="not in legal_exits"):
            rate_gate(
                _sample_input(),
                call_api_fn=_mock_api(json.dumps(response)),
                api_key="test-key",
            )

    def test_rate_gate_placeholder_refusal_raises(self):
        """FIX-14/15: Placeholders and shell refusals must be rejected."""
        response = json.loads(_sample_llm_response())
        response["walk"]["crisis"] = "<placeholder>"
        with pytest.raises(RateGateContractError, match="forbidden placeholder"):
            rate_gate(
                _sample_input(),
                call_api_fn=_mock_api(json.dumps(response)),
                api_key="test-key",
            )

        response2 = json.loads(_sample_llm_response())
        response2["walk"]["crisis"] = "None"
        with pytest.raises(RateGateContractError, match="forbidden placeholder"):
            rate_gate(
                _sample_input(),
                call_api_fn=_mock_api(json.dumps(response2)),
                api_key="test-key",
            )

    def test_rate_gate_forbidden_series_key_raises(self):
        """Pointwise series keys in output must be rejected."""
        response = json.loads(_sample_llm_response())
        response["series"] = [0.1, 0.2, 0.3]
        with pytest.raises(RateGateContractError, match="forbidden key"):
            rate_gate(
                _sample_input(),
                call_api_fn=_mock_api(json.dumps(response)),
                api_key="test-key",
            )

    def test_rate_gate_bayesian_fusion_limits(self):
        inp_zero = _sample_input()
        inp_zero["realized_counts"] = {"lag": 0, "alternative": 0, "crisis": 0}
        res_zero = rate_gate(
            inp_zero,
            call_api_fn=_mock_api(_sample_llm_response()),
            api_key="test-key",
            w_prior=5.0,
        )
        # With 0 realized counts, fused rates must equal llm_mean_rates
        for k in inp_zero["legal_exits"]:
            pytest.approx(res_zero["fused_rates"][k], res_zero["llm_mean_rates"][k])

        # Large realized counts limit (data dominates)
        inp_large = _sample_input()
        inp_large["realized_counts"] = {"lag": 1000, "alternative": 0, "crisis": 0}
        res_large = rate_gate(
            inp_large,
            call_api_fn=_mock_api(_sample_llm_response()),
            api_key="test-key",
            w_prior=5.0,
        )
        assert res_large["fused_rates"]["lag"] > 0.99

    def test_liveness_script_mock(self):
        report = run_liveness_diagnostic(live=False, n_samples=3)
        assert report["mode"] == "dry-run"
        assert report["n_states"] == 5
        assert report["verdict"] in ("ALIVE", "COLLAPSED")
        assert len(report["state_results"]) == 5

    def test_rate_gate_non_string_walk_raises(self):
        response = json.loads(_sample_llm_response())
        response["walk"]["crisis"] = 123
        with pytest.raises(RateGateContractError, match="must be str"):
            assert_rate_gate_contract(response, legal_exits=["crisis", "lag", "alternative"])

    def test_rate_gate_nan_rate_value_raises(self):
        response = json.loads(_sample_llm_response())
        response["rates"]["crisis"] = float("nan")
        with pytest.raises(RateGateContractError, match="must be non-negative finite float"):
            assert_rate_gate_contract(response, legal_exits=["crisis", "lag", "alternative"])

    def test_rate_gate_null_value_raises(self):
        response = json.loads(_sample_llm_response())
        response["walk"]["crisis"] = None
        with pytest.raises(RateGateContractError, match="forbidden placeholder"):
            assert_rate_gate_contract(response, legal_exits=["crisis", "lag", "alternative"])



def test_rate_gate_state_log_written(tmp_path):
    """FALSIFY-010: every gate call must be recorded in the verify state_log."""
    from spectrum_os.kernel import verify as _verify

    _verify.clear_state_log()
    log_file = tmp_path / "state_log.jsonl"
    _verify.init_log(log_file)
    try:
        rate_gate(
            _sample_input(),
            call_api_fn=_mock_api(_sample_llm_response()),
            api_key="test-key",
            n_samples=2,
        )
        # In-memory buffer record
        buf = [e for e in _verify.get_state_log() if e.get("gate_type") == "rate_gate"]
        assert len(buf) == 1
        entry = buf[0]
        assert entry["prediction_id"].startswith("rate_gate-")
        assert "llm_mean_rates" in entry and "fused_rates" in entry
        assert entry["n_samples"] == 2 and entry["w_prior"] == 5.0
        # Persisted JSONL record
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert any(json.loads(l).get("gate_type") == "rate_gate" for l in lines)
    finally:
        _verify.init_log(None)
        _verify.clear_state_log()
