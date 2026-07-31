"""Tests for synth/ensemble.py (PLAN-23 §5)."""

import numpy as np
import pytest

from spectrum_os.quantum.multigraph import ROLES
from spectrum_os.synth.ensemble import run_ensemble


class TestEnsembleEngine:
    def test_seed_reproducibility(self):
        res1 = run_ensemble("crisis", n_branches=5, horizon=10, seed=123)
        res2 = run_ensemble("crisis", n_branches=5, horizon=10, seed=123)

        assert len(res1["branches"]) == len(res2["branches"])
        for b1, b2 in zip(res1["branches"], res2["branches"]):
            assert b1["roles"] == b2["roles"]

    def test_action_reweighting_injection(self):
        # Base matrix without action
        rate_matrix = np.full((4, 4), 0.25)

        # Baseline: uniform
        base_res = run_ensemble(
            "crisis",
            rate_matrix=rate_matrix,
            n_branches=50,
            horizon=5,
            seed=42,
        )

        # Action: strongly favor "alternative" at step t=0
        action = {"t": 0, "reweight": {"role": "alternative", "factor": 100.0}}
        act_res = run_ensemble(
            "crisis",
            rate_matrix=rate_matrix,
            action=action,
            n_branches=50,
            horizon=5,
            seed=42,
        )

        # Count roles at t=1 (first transition after t=0 injection)
        roles_t1_base = [b["roles"][1] for b in base_res["branches"]]
        roles_t1_act = [b["roles"][1] for b in act_res["branches"]]

        alt_count_base = roles_t1_base.count("alternative")
        alt_count_act = roles_t1_act.count("alternative")

        assert alt_count_act > alt_count_base
        assert alt_count_act / 50.0 > 0.8

    def test_gate_call_budget(self):
        call_count = 0

        def mock_gate(input_data):
            nonlocal call_count
            call_count += 1
            return {
                "candidates": [
                    {"label": "gated_label", "concept_tags": ["tag_a", "tag_b"]}
                ]
            }

        # Request 10 branches, horizon 20, gate_every 2 (9 gate points per branch).
        # Per-branch budget max_gates=5 -> each branch capped at 5 calls -> 50 total.
        res = run_ensemble(
            "crisis",
            n_branches=10,
            horizon=20,
            gate_fn=mock_gate,
            gate_every=2,
            max_gates=5,
            seed=42,
        )

        assert res["meta"]["total_gate_calls"] == 50
        assert call_count == 50
        # Per-branch budget guarantees every branch receives gate coverage —
        # later branches must not starve (standing_wave nodes depend on it).
        for b in res["branches"]:
            assert b["tags_per_step"], "every branch must receive gate tags"

    def test_mechanical_baseline_without_gate(self):
        res = run_ensemble("crisis", n_branches=3, horizon=8, gate_fn=None, seed=99)
        assert res["meta"]["total_gate_calls"] == 0
        for b in res["branches"]:
            assert b["tags_per_step"] == {}
            assert b["accidents_by_step"] == {}


class TestEnsembleResidualTrigger:
    """N4: two-phase state switching (Mode A reflect <-> Mode B enumerate)."""

    def _reflect_gate(self, input_data, **kwargs):
        return {
            "candidates": [
                {
                    "label": "reflected",
                    "concept_tags": ["tag_reflect"],
                    "provenance_hint": "novel",
                    "novel": True,
                }
            ]
        }

    def _enumerate_gate(self, input_data, **kwargs):
        return {
            "accidents": [
                {
                    "pattern": "意外模式",
                    "instances": ["實例一"],
                    "domain": "政治",
                    "source": "emergent",
                    "usage": "expression",
                    "grounding": "合理推測",
                    "world_development": "導向的世界",
                }
            ]
        }

    def test_trigger_hit_switches_to_enumerate(self):
        calls = {"reflect": 0, "enumerate": 0}

        def reflect(input_data, **kwargs):
            calls["reflect"] += 1
            return self._reflect_gate(input_data, **kwargs)

        def enumer(input_data, **kwargs):
            calls["enumerate"] += 1
            return self._enumerate_gate(input_data, **kwargs)

        res = run_ensemble(
            "crisis",
            n_branches=2,
            horizon=8,
            gate_fn=reflect,
            gate_every=4,
            max_gates=2,
            seed=42,
            residual_trigger=lambda t, context: True,
            enumerate_gate_fn=enumer,
        )

        # horizon 8 -> gate points at t=3 only; 2 branches -> 2 enumerate calls
        assert calls["enumerate"] == 2
        assert calls["reflect"] == 0
        assert res["meta"]["total_enumerate_calls"] == 2
        for b in res["branches"]:
            assert len(b["accidents_by_step"][3]) == 1
            assert b["accidents_by_step"][3][0]["pattern"] == "意外模式"
            # accidents never leak into tags_per_step
            assert b["tags_per_step"][3] == []

    def test_trigger_miss_keeps_reflect(self):
        calls = {"reflect": 0, "enumerate": 0}

        def reflect(input_data, **kwargs):
            calls["reflect"] += 1
            return self._reflect_gate(input_data, **kwargs)

        def enumer(input_data, **kwargs):
            calls["enumerate"] += 1
            return self._enumerate_gate(input_data, **kwargs)

        res = run_ensemble(
            "crisis",
            n_branches=2,
            horizon=8,
            gate_fn=reflect,
            gate_every=4,
            max_gates=2,
            seed=42,
            residual_trigger=lambda t, context: False,
            enumerate_gate_fn=enumer,
        )

        assert calls["reflect"] == 2
        assert calls["enumerate"] == 0
        assert res["meta"]["total_enumerate_calls"] == 0
        for b in res["branches"]:
            assert b["accidents_by_step"] == {}
            assert b["tags_per_step"][3] == ["tag_reflect"]

    def test_mixed_trigger(self):
        # horizon 12, gate_every 4 -> gate points at t=3 and t=7; trigger only at t=3
        calls = {"reflect": 0, "enumerate": 0}

        def reflect(input_data, **kwargs):
            calls["reflect"] += 1
            return self._reflect_gate(input_data, **kwargs)

        def enumer(input_data, **kwargs):
            calls["enumerate"] += 1
            return self._enumerate_gate(input_data, **kwargs)

        res = run_ensemble(
            "crisis",
            n_branches=1,
            horizon=12,
            gate_fn=reflect,
            gate_every=4,
            max_gates=2,
            seed=42,
            residual_trigger=lambda t, context: t == 3,
            enumerate_gate_fn=enumer,
        )

        assert calls["enumerate"] == 1
        assert calls["reflect"] == 1
        b = res["branches"][0]
        assert list(b["accidents_by_step"].keys()) == [3]
        assert b["tags_per_step"][3] == []
        assert b["tags_per_step"][7] == ["tag_reflect"]

    def test_residual_trigger_none_unchanged(self):
        # Backward compatibility: without residual_trigger the engine never
        # enumerates and accidents_by_step stays empty.
        calls = {"reflect": 0, "enumerate": 0}

        def reflect(input_data, **kwargs):
            calls["reflect"] += 1
            return self._reflect_gate(input_data, **kwargs)

        def enumer(input_data, **kwargs):
            calls["enumerate"] += 1
            return self._enumerate_gate(input_data, **kwargs)

        res = run_ensemble(
            "crisis",
            n_branches=2,
            horizon=8,
            gate_fn=reflect,
            gate_every=4,
            max_gates=2,
            seed=42,
            enumerate_gate_fn=enumer,
        )

        assert calls["reflect"] == 2
        assert calls["enumerate"] == 0
        assert res["meta"]["total_enumerate_calls"] == 0
        for b in res["branches"]:
            assert b["accidents_by_step"] == {}
            assert b["tags_per_step"][3] == ["tag_reflect"]

    def test_residual_table_mapping_auto_wrapped(self):
        # A residual table Mapping passed as residual_trigger is auto-wrapped;
        # gate_input is the trigger context (t keys match int steps directly).
        calls = {"reflect": 0, "enumerate": 0}

        def reflect(input_data, **kwargs):
            calls["reflect"] += 1
            return self._reflect_gate(input_data, **kwargs)

        def enumer(input_data, **kwargs):
            calls["enumerate"] += 1
            return self._enumerate_gate(input_data, **kwargs)

        res = run_ensemble(
            "crisis",
            n_branches=1,
            horizon=8,
            gate_fn=reflect,
            gate_every=4,
            max_gates=2,
            seed=42,
            residual_trigger={3: {"criteria": ["saturation"]}},
            enumerate_gate_fn=enumer,
        )

        assert calls["enumerate"] == 1
        assert calls["reflect"] == 0
        b = res["branches"][0]
        assert 3 in b["accidents_by_step"]

    def test_residual_table_mapping_with_key_fn_hits_date_key(self):
        # A date-keyed residual table Mapping requires residual_key_fn to map
        # the gate point to the date carried in the situation payload; with it,
        # the date-keyed table hits and Mode B switches on.
        calls = {"reflect": 0, "enumerate": 0}

        def reflect(input_data, **kwargs):
            calls["reflect"] += 1
            return self._reflect_gate(input_data, **kwargs)

        def enumer(input_data, **kwargs):
            calls["enumerate"] += 1
            return self._enumerate_gate(input_data, **kwargs)

        res = run_ensemble(
            "crisis",
            n_branches=1,
            horizon=8,
            gate_fn=reflect,
            gate_every=4,
            max_gates=2,
            seed=42,
            situation={"date": "1914-08-02"},
            residual_trigger={"1914-08-02": {"criteria": ["saturation"]}},
            residual_key_fn=lambda t, ctx: ctx["situation"].get("date"),
            enumerate_gate_fn=enumer,
        )

        assert calls["enumerate"] == 1
        assert calls["reflect"] == 0
        assert res["meta"]["total_enumerate_calls"] == 1
        b = res["branches"][0]
        assert 3 in b["accidents_by_step"]

    def test_residual_table_mapping_without_key_fn_date_key_misses(self):
        # Guard against the silent-death regression: a date-keyed Mapping
        # WITHOUT residual_key_fn must NOT enumerate (identity int-step lookup
        # never matches the date key), so Mode B stays off. This is documented
        # behaviour — date-keyed tables must always supply residual_key_fn.
        calls = {"reflect": 0, "enumerate": 0}

        def reflect(input_data, **kwargs):
            calls["reflect"] += 1
            return self._reflect_gate(input_data, **kwargs)

        def enumer(input_data, **kwargs):
            calls["enumerate"] += 1
            return self._enumerate_gate(input_data, **kwargs)

        res = run_ensemble(
            "crisis",
            n_branches=1,
            horizon=8,
            gate_fn=reflect,
            gate_every=4,
            max_gates=2,
            seed=42,
            situation={"date": "1914-08-02"},
            residual_trigger={"1914-08-02": {"criteria": ["saturation"]}},
            enumerate_gate_fn=enumer,
        )

        assert calls["enumerate"] == 0
        assert calls["reflect"] == 1
        assert res["meta"]["total_enumerate_calls"] == 0
        assert res["branches"][0]["accidents_by_step"] == {}
