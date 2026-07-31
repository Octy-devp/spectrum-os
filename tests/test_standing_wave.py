"""Tests for quantum/standing_wave.py (PLAN-23 §5)."""

import math
import pytest

from spectrum_os.quantum.standing_wave import standing_wave


class TestStandingWave:
    def test_handcrafted_nodes_and_antinodes(self):
        branches = [
            {
                "branch_id": 0,
                "roles": ["crisis", "lag", "alternative"],
                "tags_per_step": {0: ["tag_common", "tag_b0"]},
            },
            {
                "branch_id": 1,
                "roles": ["crisis", "crisis", "direction"],
                "tags_per_step": {0: ["tag_common", "tag_b1"]},
            },
            {
                "branch_id": 2,
                "roles": ["crisis", "lag", "lag"],
                "tags_per_step": {1: ["tag_common"]},
            },
        ]

        res = standing_wave(branches)

        # Nodes: tag_common present in 3/3 branches
        nodes = res["nodes"]
        assert len(nodes) == 1
        assert nodes[0]["tag"] == "tag_common"
        assert nodes[0]["prevalence"] == 1.0

        # Antinodes: tag_b0 and tag_b1 present in 1/3 branches (prevalence 0.3333)
        antinodes = res["antinodes"]
        assert len(antinodes) == 2
        tags_anti = [a["tag"] for a in antinodes]
        assert "tag_b0" in tags_anti
        assert "tag_b1" in tags_anti
        for a in antinodes:
            assert a["prevalence"] == pytest.approx(1.0 / 3.0, abs=1e-3)

        # Divergence curve at t=0: all 3 branches are "crisis" -> H(0) = 0.0
        # At t=1: roles are ["lag", "crisis", "lag"] -> p(lag)=2/3, p(crisis)=1/3 -> H(1) > 0
        curve = res["divergence_curve"]
        assert len(curve) == 3
        assert curve[0] == 0.0
        assert curve[1] > 0.0

    def test_single_branch_boundary(self):
        branches = [
            {
                "branch_id": 0,
                "roles": ["crisis", "lag"],
                "tags_per_step": {0: ["tag_a"]},
            }
        ]
        res = standing_wave(branches)
        assert len(res["nodes"]) == 1
        assert res["nodes"][0]["tag"] == "tag_a"
        assert res["antinodes"] == []
        assert res["divergence_curve"] == [0.0, 0.0]

    def test_empty_tags_boundary(self):
        branches = [
            {"branch_id": 0, "roles": ["crisis", "lag"], "tags_per_step": {}},
            {"branch_id": 1, "roles": ["crisis", "alternative"], "tags_per_step": {}},
        ]
        res = standing_wave(branches)
        assert res["nodes"] == []
        assert res["antinodes"] == []
        assert len(res["divergence_curve"]) == 2
        assert res["divergence_curve"][0] == 0.0
        assert res["divergence_curve"][1] == 1.0  # 50% lag, 50% alternative -> log2(2) = 1.0
