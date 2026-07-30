"""Tests for spectrum_os/contracts.py & markov.py bias flag integration (PLAN-23 §6.1, §6.6)."""

import numpy as np
import pytest

from spectrum_os.contracts import CLADCorpus, DCASubstrate, RoleTrajectory
from spectrum_os.quantum.markov import estimate_rate_matrix
from spectrum_os.quantum.multigraph import RoleMultigraph


class TestRoleTrajectory:
    def test_valid_trajectory(self):
        traj = RoleTrajectory(
            thread_id="t1",
            points=[(1.0, "crisis"), (2.0, "lag"), (3.0, "alternative")],
            source_id="src1",
            bias_flags=[{"kind": "under_measurement"}],
        )
        assert traj.thread_id == "t1"
        assert traj.role_sequence() == ["crisis", "lag", "alternative"]
        assert traj.source_id == "src1"
        assert len(traj.bias_flags) == 1

    def test_invalid_role(self):
        with pytest.raises(ValueError, match="Invalid role"):
            RoleTrajectory(thread_id="t1", points=[(1.0, "unknown_role")])

    def test_empty_points(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            RoleTrajectory(thread_id="t1", points=[])

    def test_non_monotonic_time(self):
        with pytest.raises(ValueError, match="Non-monotonic"):
            RoleTrajectory(
                thread_id="t1",
                points=[(2.0, "crisis"), (1.0, "lag")],
            )


class TestDCASubstrate:
    def test_valid_substrate(self):
        sub = DCASubstrate(
            nodes={"s1": {"mass": 10}, "s2": {"mass": 5}},
            edges=[
                {"from": "s1", "to": "s2", "relation": "crisis->lag", "strength": 1.0},
                {"from": "s2", "to": "s1", "relation": "alternative->direction", "strength": 0.8},
            ],
            source_id="dca_src",
        )
        assert len(sub.nodes) == 2
        assert len(sub.edges) == 2
        assert not sub.edges[0]["orphan"]
        assert sub.transition_type_counts() == {
            "crisis->lag": 1,
            "alternative->direction": 1,
        }

    def test_orphan_edge_tagging(self):
        # s3 is missing from nodes -> orphan edge marked True, edge NOT deleted
        sub = DCASubstrate(
            nodes={"s1": {"mass": 10}},
            edges=[
                {"from": "s1", "to": "s3", "relation": "crisis->lag", "strength": 1.0},
            ],
        )
        assert len(sub.edges) == 1
        assert sub.edges[0]["orphan"] is True
        assert sub.edges[0]["from"] == "s1"
        assert sub.edges[0]["to"] == "s3"

    def test_invalid_relation(self):
        with pytest.raises(ValueError, match="Invalid relation"):
            DCASubstrate(
                nodes={"s1": {}, "s2": {}},
                edges=[{"from": "s1", "to": "s2", "relation": "invalid_rel"}],
            )

    def test_unknown_role_in_relation(self):
        with pytest.raises(ValueError, match="Unknown role"):
            DCASubstrate(
                nodes={"s1": {}, "s2": {}},
                edges=[{"from": "s1", "to": "s2", "relation": "foo->bar"}],
            )

    def test_to_multigraph_roundtrip(self):
        sub = DCASubstrate(
            nodes={"s1": {}, "s2": {}},
            edges=[
                {"from": "s1", "to": "s2", "relation": "crisis->lag", "strength": 1.0},
            ],
            source_id="dca_test",
        )
        mg = sub.to_multigraph()
        assert isinstance(mg, RoleMultigraph)
        assert "s1" in mg.assignments
        assert "s2" in mg.assignments


class TestCLADCorpus:
    def test_valid_clad_corpus(self):
        corpus = CLADCorpus(
            entries=[
                {
                    "id": "c1",
                    "clad": {
                        "crisis": "text1",
                        "lag": "text2",
                        "alternative": "text3",
                        "direction": "text4",
                    },
                }
            ],
            source_id="clad_src",
        )
        assert len(corpus.entries) == 1
        assert corpus.entries[0]["id"] == "c1"

    def test_missing_role_in_clad(self):
        with pytest.raises(ValueError, match="missing required roles"):
            CLADCorpus(
                entries=[
                    {
                        "id": "c1",
                        "clad": {"crisis": "text1", "lag": "text2"},
                    }
                ]
            )

    def test_duplicate_id(self):
        with pytest.raises(ValueError, match="Duplicate entry id"):
            CLADCorpus(
                entries=[
                    {
                        "id": "c1",
                        "clad": {
                            "crisis": "a",
                            "lag": "b",
                            "alternative": "c",
                            "direction": "d",
                        },
                    },
                    {
                        "id": "c1",
                        "clad": {
                            "crisis": "a",
                            "lag": "b",
                            "alternative": "c",
                            "direction": "d",
                        },
                    },
                ]
            )


class TestMarkovBiasDowngrade:
    def test_no_flags_unaffected(self):
        counts = np.ones((4, 4)) * 5  # total count = 80 >= 10
        res = estimate_rate_matrix(counts, alpha=1.0, min_total_count=10.0)
        assert res["verdict"] == "ASSERTED"
        assert res["bias_notes"] == []

    def test_lag_under_measurement_downgrades_verdict(self):
        # Create counts where lag (index 1) count is low (e.g. 2.0 < min_total_count 10.0)
        counts = np.ones((4, 4)) * 5.0
        counts[1, :] = 0.5  # row 1 (lag) total count = 2.0 < 10.0

        bias_flags = [
            {
                "kind": "under_measurement",
                "scope": {"roles": ["lag"]},
                "note": "lag role is under-measured in dataset",
            }
        ]

        res = estimate_rate_matrix(counts, alpha=1.0, min_total_count=10.0, bias_flags=bias_flags)
        assert res["verdict"] == "CONTESTED"
        assert len(res["bias_notes"]) > 0
        assert "lag" in res["bias_notes"][0]
        assert "under_measurement" in res["bias_notes"][0]
