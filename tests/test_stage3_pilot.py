"""Unit and smoke tests for Stage 3 Pilot & Universality Check (PLAN-23 §5)."""

import json
from pathlib import Path
import pytest

from feeds.ecc_feeds import register_ecc_sources
from scripts.stage3_sarajevo_pilot import run_sarajevo_pilot, build_sarajevo_initial_vector
from scripts.stage3_universality_check import run_universality_check, run_pipeline, dry_run_gate
from spectrum_os import sources

register_ecc_sources()
ECC_DATA_EXISTS = Path(sources.get_source("ecc-knowledge-edges").locator).exists()


@pytest.mark.skipif(not ECC_DATA_EXISTS, reason="ECC source data directory not found")
class TestStage3SarajevoPilot:
    def test_build_initial_vector(self):
        vec_data = build_sarajevo_initial_vector()
        assert vec_data["role"] == "crisis"
        assert "vector" in vec_data
        vec = vec_data["vector"]
        assert set(vec.keys()) == {"d1", "d2", "d3", "d4", "d5", "d6"}
        assert vec_data["provenance"]["source_id"] == "ecc-knowledge-edges"

    def test_run_sarajevo_pilot_dry_run(self, tmp_path):
        report_file = tmp_path / "stage3_sarajevo_test_report.json"
        report = run_sarajevo_pilot(
            live=False,
            n_branches=5,
            horizon=12,
            seed=42,
            out_path=str(report_file),
        )

        assert report_file.exists()
        assert report["mode"] == "dry-run"
        assert report["branches_summary"]["n_branches"] == 5
        assert report["branches_summary"]["horizon"] == 12

        # Standing wave schema check
        sw = report["standing_wave"]
        assert "nodes" in sw
        assert "antinodes" in sw
        assert "divergence_curve" in sw

        # Anchor comparison & verify check
        ac = report["anchor_comparison"]
        assert "anchor_branch_ratio" in ac
        assert "closest_branch_id" in ac
        assert "verify_result" in ac
        vr = ac["verify_result"]
        assert vr["prediction_id"] == "3-sarajevo-vs-alpha1"
        assert "mae" in vr
        assert "mape" in vr
        assert "verdict" in vr

        # Gate calls & quarantine
        assert "gate_calls_and_cost" in report
        assert "quarantine_and_adversary_hits" in report


class TestStage3UniversalityCheck:
    def test_run_pipeline_zero_branching(self):
        initial_vec = {
            "role": "crisis",
            "vector": {"d1": -1, "d2": -1, "d3": -1, "d4": 12.0, "d5": 0.0, "d6": 0.8},
            "provenance": {"source_id": "mock_src"},
        }
        action = {"t": 1, "reweight": {"role": "alternative", "factor": 2.0}}

        res = run_pipeline(
            initial_vector=initial_vec,
            substrate=None,
            action=action,
            n_branches=5,
            horizon=12,
            gate_fn=dry_run_gate,
            seed=42,
        )

        assert "ensemble" in res
        assert "standing_wave" in res
        sw = res["standing_wave"]
        assert len(sw["divergence_curve"]) == 12

    @pytest.mark.skipif(not ECC_DATA_EXISTS, reason="ECC source data directory not found")
    def test_run_universality_check_full(self, tmp_path):
        report_file = tmp_path / "stage3_universality_test_report.json"
        report = run_universality_check(
            n_branches=5,
            horizon=12,
            seed=42,
            out_path=str(report_file),
        )

        assert report_file.exists()
        assert report["generality_assertion"]["zero_branching"] is True
        assert "source1_sarajevo_1914" in report
        assert "source2_worldbank_gdp" in report

        sw1 = report["source1_sarajevo_1914"]["standing_wave_summary"]
        sw2 = report["source2_worldbank_gdp"]["standing_wave_summary"]
        assert sw1["meta"]["total_branches"] == 5
        assert sw2["meta"]["total_branches"] == 5
