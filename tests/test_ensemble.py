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
