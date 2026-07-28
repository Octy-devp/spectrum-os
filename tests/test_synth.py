"""Tests for the synth gate layer (PLAN-23 §7.3) and synthetic guardrails.

No test in this file touches the real API — ``synth.anchors`` is exercised
through an injected mock ``call_api_fn``.
"""

import json

import numpy as np
import pytest

from spectrum_os.kernel import branch, cluster, sector, wave_ops
from spectrum_os.kernel._types import Verdict
from spectrum_os.synth import (
    AnchorContractError,
    anchors,
    assert_contract,
    expand,
    expand_pair,
    expand_plan,
    register_synthetic_sector,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_sectors():
    sector._sectors.clear()
    yield


N_MONTHS = 48


def _anchors() -> dict:
    """A contract-valid anchors dict (as the LLM gate would produce)."""
    return {
        "turning_points": [
            {"month_index": 0, "level": 200.0, "source": "CEHE vol.7 — AH steel 1913"},
            {"month_index": 12, "level": 240.0, "source": "CEHE vol.7 — wartime peak"},
            {"month_index": 30, "level": 180.0, "source": "β₁ inference"},
        ],
        "magnitudes": {
            "baseline": 200.0,
            "unit": "kt_steel_per_month",
            "noise_sigma": 4.0,
            "ar_rho": 0.3,
        },
        "event_shocks": [
            {"month_index": 6, "delta": -30.0, "duration_months": 3,
             "source": "β₁: 黎明手術 disruption"},
            {"month_index": 18, "delta": 25.0, "duration_months": 0,
             "source": "β₁: ECC 基礎條約"},
        ],
        "confidence": {"llm_self_rating": "very confident — must be discarded"},
        "calibration_sources": ["Cambridge Economic History of Europe vol.7-8"],
    }


def _spec() -> dict:
    return {
        "name": "danube_steel_nspv",
        "description": "DSF steel output under NSPV accounting",
        "period": {"start": "1916-01", "end": "1920-12", "n_months": N_MONTHS},
        "calibration": [
            {"metric": "AH steel 1913", "value": 217, "unit": "kt/month",
             "source": "CEHE vol.7-8"},
        ],
        "beta1_events": [
            {"month_index": 5, "month": "1916-06", "event": "黎明手術 / DSF 成立"},
        ],
    }


# ===================================================================
# expand — determinism, step semantics, shocks
# ===================================================================

class TestExpand:
    def test_deterministic_same_seed(self):
        a = expand(_anchors(), N_MONTHS, seed=23)
        b = expand(_anchors(), N_MONTHS, seed=23)
        np.testing.assert_array_equal(a, b)

    def test_different_seed_same_plan_different_actual(self):
        a = expand(_anchors(), N_MONTHS, seed=23)
        b = expand(_anchors(), N_MONTHS, seed=99)
        assert not np.array_equal(a, b)
        # The plan must be seed-independent
        p1 = expand_plan(_anchors(), N_MONTHS)
        p2 = expand_plan(_anchors(), N_MONTHS)
        np.testing.assert_array_equal(p1, p2)

    def test_step_function_no_interpolation(self):
        """Between turning points the plan is exactly constant — no lerp."""
        plan = expand_plan(_anchors(), N_MONTHS)
        # Segment 1: months 0-11 at 200
        assert np.all(plan[0:12] == 200.0)
        # Segment 2: months 12-29 at 240
        assert np.all(plan[12:30] == 240.0)
        # Segment 3: months 30+ at 180
        assert np.all(plan[30:] == 180.0)
        # A step, not a ramp: the jump happens in one month
        assert plan[12] - plan[11] == 40.0

    def test_shocks_temporary_and_permanent(self):
        a = expand(_anchors(), N_MONTHS, seed=1, include_noise=False)
        plan = expand_plan(_anchors(), N_MONTHS)
        diff = a - plan
        # Temporary shock: months 6-8 only
        assert np.all(diff[6:9] == -30.0)
        assert diff[5] == 0.0
        # Permanent shock from month 18 onward
        assert np.all(diff[18:] == 25.0)
        # After the temporary shock expired, only the permanent one remains
        assert diff[9] == 0.0

    def test_expand_pair_shapes(self):
        actual, plan = expand_pair(_anchors(), N_MONTHS, seed=7)
        assert actual.shape == (N_MONTHS,)
        assert plan.shape == (N_MONTHS,)

    def test_expand_revalidates_contract(self):
        bad = _anchors()
        bad["magnitudes"]["baseline"] = -5.0
        with pytest.raises(AnchorContractError):
            expand(bad, N_MONTHS, seed=1)


# ===================================================================
# assert_contract — mechanical producer/consumer contract
# ===================================================================

class TestContract:
    def test_valid_passes(self):
        assert_contract(_anchors(), n_months=N_MONTHS)

    def test_missing_key(self):
        bad = _anchors()
        del bad["turning_points"]
        with pytest.raises(AnchorContractError, match="missing required keys"):
            assert_contract(bad)

    def test_wrong_key_name_caught(self):
        """FIX-01/06 lesson: producer/consumer key mismatch must not pass."""
        bad = _anchors()
        bad["turning_point"] = bad.pop("turning_points")  # wrong singular key
        with pytest.raises(AnchorContractError, match="missing required keys"):
            assert_contract(bad)

    def test_forbidden_series_key(self):
        """The gate must never write pointwise series values."""
        bad = _anchors()
        bad["series"] = [200.0] * N_MONTHS
        with pytest.raises(AnchorContractError, match="forbidden key"):
            assert_contract(bad)

    def test_bad_types(self):
        bad = _anchors()
        bad["turning_points"][0]["level"] = "high"
        with pytest.raises(AnchorContractError, match="level must be numeric"):
            assert_contract(bad)

    def test_month_index_out_of_range(self):
        bad = _anchors()
        bad["event_shocks"][0]["month_index"] = 48
        with pytest.raises(AnchorContractError, match="out of range"):
            assert_contract(bad, n_months=N_MONTHS)

    def test_negative_duration(self):
        bad = _anchors()
        bad["event_shocks"][0]["duration_months"] = -1
        with pytest.raises(AnchorContractError, match="duration_months"):
            assert_contract(bad)

    def test_duplicate_turning_point_month(self):
        bad = _anchors()
        bad["turning_points"].append(
            {"month_index": 12, "level": 999.0, "source": "x"})
        with pytest.raises(AnchorContractError, match="duplicated"):
            assert_contract(bad)

    def test_empty_calibration_sources(self):
        bad = _anchors()
        bad["calibration_sources"] = []
        with pytest.raises(AnchorContractError, match="calibration_sources"):
            assert_contract(bad)


# ===================================================================
# anchors() — mock API parsing + mechanical confidence backfill
# ===================================================================

class TestAnchorsGate:
    def _mock_api(self, payload: str):
        def call_api_fn(prompt, api_key, **kwargs):
            return payload
        return call_api_fn

    def test_parses_plain_json(self):
        result = anchors(_spec(),
                         call_api_fn=self._mock_api(json.dumps(_anchors())),
                         api_key="test-key")
        assert_contract(result, n_months=N_MONTHS)
        assert result["generated_by"]["gate"] == "synth.anchors"
        assert result["generated_by"]["model"] == "deepseek-v4-flash"

    def test_parses_fenced_json(self):
        payload = "```json\n" + json.dumps(_anchors()) + "\n```"
        result = anchors(_spec(),
                         call_api_fn=self._mock_api(payload),
                         api_key="test-key")
        assert_contract(result, n_months=N_MONTHS)

    def test_bad_json_raises(self):
        with pytest.raises(AnchorContractError, match="not valid JSON"):
            anchors(_spec(),
                    call_api_fn=self._mock_api("this is not json"),
                    api_key="test-key")

    def test_contract_violation_from_llm_caught(self):
        bad = _anchors()
        del bad["event_shocks"]
        with pytest.raises(AnchorContractError, match="missing required keys"):
            anchors(_spec(),
                    call_api_fn=self._mock_api(json.dumps(bad)),
                    api_key="test-key")

    def test_confidence_mechanically_backfilled(self):
        """FIX-21: LLM self-assessment must be discarded and recomputed."""
        result = anchors(_spec(),
                         call_api_fn=self._mock_api(json.dumps(_anchors())),
                         api_key="test-key")
        conf = result["confidence"]
        assert "llm_self_rating" not in conf
        # 5 anchors (3 tp + 2 shocks), all sourced → fraction 1.0
        assert conf["sourced_fraction"] == 1.0
        assert conf["basis"] == "sourced"

    def test_confidence_partial_sourcing(self):
        partial = _anchors()
        partial["turning_points"][2]["source"] = ""
        partial["event_shocks"][1]["source"] = "   "
        result = anchors(_spec(),
                         call_api_fn=self._mock_api(json.dumps(partial)),
                         api_key="test-key")
        # 3 of 5 sourced
        assert result["confidence"]["sourced_fraction"] == 0.6
        assert result["confidence"]["basis"] == "mixed"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            anchors(_spec(), call_api_fn=lambda *a, **k: "{}", api_key=None)


# ===================================================================
# Synthetic guardrails — meta enforcement, verdict cap, markers
# ===================================================================

class TestSyntheticGuardrails:
    def test_meta_enforcement_missing_keys(self):
        with pytest.raises(ValueError, match="missing required keys"):
            sector.create("bad_syn", [1.0, 2.0], meta={"synthetic": True})

    def test_meta_enforcement_partial(self):
        with pytest.raises(ValueError, match="seed"):
            sector.create("bad_syn", [1.0, 2.0],
                          meta={"synthetic": True, "anchors": {},
                                "generated_by": "test"})

    def test_meta_enforcement_ok(self):
        sec = sector.create(
            "good_syn", [1.0, 2.0],
            meta={"synthetic": True, "anchors": {}, "generated_by": "test",
                  "seed": 1})
        assert sec.meta["synthetic"] is True

    def test_non_synthetic_meta_passes(self):
        sec = sector.create("plain", [1.0, 2.0], meta={"source": "wx feed"})
        assert sec.meta == {"source": "wx feed"}

    def test_meta_roundtrip_save_load(self, tmp_path):
        sector.create("rt", [1.0, 2.0], [1.0, 1.0],
                      meta={"synthetic": True, "anchors": {"a": 1},
                            "generated_by": "test", "seed": 5})
        p = str(tmp_path / "sectors.json")
        sector.save(p)
        sector._sectors.clear()
        sector.load(p)
        loaded = sector.get("rt")
        assert loaded.meta["synthetic"] is True
        assert loaded.meta["anchors"] == {"a": 1}
        assert loaded.meta["seed"] == 5

    def test_register_synthetic_sector(self):
        sec = register_synthetic_sector("danube_steel_nspv", _anchors(),
                                        N_MONTHS, seed=23)
        assert sec.meta["synthetic"] is True
        assert sec.meta["seed"] == 23
        assert sec.targets is not None and len(sec.targets) == N_MONTHS
        assert len(sec.timeseries) == N_MONTHS

    def test_verdict_capped_at_contested(self):
        """A clean periodic synthetic series that would be ASSERTED must
        come back CONTESTED, with the marker visible."""
        t = np.arange(60)
        signal = (np.sin(2 * np.pi * t / 12) * 10 + 200.0).tolist()
        plan = np.full(60, 200.0).tolist()
        sector.create("syn_sec", signal, plan,
                      meta={"synthetic": True, "anchors": {},
                            "generated_by": "test", "seed": 1})
        sector.create("real_sec", signal, plan)

        syn = wave_ops.decompose("syn_sec")
        real = wave_ops.decompose("real_sec")

        assert real.verdict == Verdict.ASSERTED  # control: same data asserts
        assert syn.verdict == Verdict.CONTESTED
        assert syn.values["synthetic_input"] is True
        assert real.values["synthetic_input"] is False
        assert "capped" in syn.confidence_reason
        assert syn.boundary_note is not None

    def test_correlate_marks_synthetic(self):
        t = np.arange(60)
        a = (0.01 * t + np.random.default_rng(1).normal(0, 0.1, 60)).tolist()
        b = ([0.0] * 3 + a[:-3])
        sector.create("syn_a", a, meta={"synthetic": True, "anchors": {},
                                        "generated_by": "test", "seed": 1})
        sector.create("real_b", b)
        sector.create("real_c", a)

        marked = wave_ops.correlate("syn_a", "real_b")
        assert marked["synthetic_input"] is True

        unmarked = wave_ops.correlate("real_b", "real_c")
        assert unmarked["synthetic_input"] is False

    def test_cluster_marks_synthetic(self):
        sigs = [
            {"periods": [6, 12], "amplitudes": [0.5, 0.3],
             "mean_deviation": -0.02, "synthetic": True},
            {"periods": [24, 36], "amplitudes": [0.8, 0.6],
             "mean_deviation": 0.05},
        ]
        result = cluster.run(sigs, n_clusters=2)
        assert result["synthetic_input"] is True

        clean = cluster.run([{k: v for k, v in s.items() if k != "synthetic"}
                             for s in sigs], n_clusters=2)
        assert clean["synthetic_input"] is False

    def test_branch_simulate_runs_with_synthetic(self):
        """Offline acceptance: a synthetic sector carries targets, so
        branch.simulate no longer raises 'no targets'."""
        register_synthetic_sector("danube_steel_nspv", _anchors(),
                                  N_MONTHS, seed=23)
        result = branch.simulate({"danube_steel_nspv": 0.10}, n=5)
        assert len(result["branches"]) == 5
        assert all(b["sector_id"] == "danube_steel_nspv"
                   for b in result["branches"])
        assert "ensemble_stats" in result
