"""Tests for spectrum_os/contracts.py & markov.py bias flag integration (PLAN-23 §6.1, §6.6).

Covers the core role/DCA/CLAD contracts plus the four world-interface contracts
(FieldSpec / ActorCard / SituationSpec / FieldLog, T17a) and the world-agnostic
source check (OS layer schema must contain zero world-specific content).
"""

from pathlib import Path

import json

import numpy as np
import pytest

from spectrum_os.contracts import (
    CLADCorpus,
    DCASubstrate,
    RoleTrajectory,
    FieldSpec,
    ActorCard,
    SituationSpec,
    FieldLog,
)
from spectrum_os.quantum.markov import estimate_rate_matrix
from spectrum_os.quantum.multigraph import RoleMultigraph
from spectrum_os.synth.probe import (
    _extract_situation_vector,
    _root_label,
)


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


# ---------------------------------------------------------------------------
# World-interface contracts (T17a plugin contract schema)
# ---------------------------------------------------------------------------


def _make_actor_card(actor_id: str = "actor-alpha", n_tensions: int = 2) -> ActorCard:
    """Build a valid world-agnostic ActorCard for reuse across contract tests."""
    return ActorCard(
        actor_id=actor_id,
        name=f"{actor_id} name",
        inherited_conditions={"rank": "pioneer"},
        field_coordinates={"x": 1, "y": 2},
        internal_tensions=[
            {"axis_A": "inherited", "axis_B": "expression", "note": f"tension-{i}"}
            for i in range(n_tensions)
        ],
        temporal_states={"t0": "idle", "t1": "active"},
    )


class TestFieldSpec:
    """Field folder contract: field_id / worldline / actors are mandatory."""

    def test_valid_field_spec_keeps_fields(self):
        spec = FieldSpec(
            field_id="field-1",
            worldline="alpha",
            temporal_scope={"start": 1900, "end": 1920},
            spatial_scope={"region": "delta"},
            actors=["actor-alpha", "actor-beta"],
            params={"depth": 3},
            triggers={"gate": "N5"},
            meta={"source": "demo"},
        )
        assert spec.field_id == "field-1"
        assert spec.worldline == "alpha"
        assert spec.temporal_scope == {"start": 1900, "end": 1920}
        assert spec.spatial_scope == {"region": "delta"}
        assert spec.actors == ["actor-alpha", "actor-beta"]
        assert spec.params["depth"] == 3
        assert spec.triggers["gate"] == "N5"
        assert spec.meta["source"] == "demo"

    def test_defaults_params_triggers_meta(self):
        spec = FieldSpec(
            field_id="field-1",
            worldline="alpha",
            temporal_scope={},
            spatial_scope={},
            actors=["actor-alpha"],
        )
        assert spec.params == {}
        assert spec.triggers == {}
        assert spec.meta == {}

    def test_empty_field_id_raises(self):
        with pytest.raises(ValueError, match="field_id cannot be empty"):
            FieldSpec(
                field_id="",
                worldline="alpha",
                temporal_scope={},
                spatial_scope={},
                actors=["actor-alpha"],
            )

    def test_empty_worldline_raises(self):
        with pytest.raises(ValueError, match="worldline cannot be empty"):
            FieldSpec(
                field_id="field-1",
                worldline="",
                temporal_scope={},
                spatial_scope={},
                actors=["actor-alpha"],
            )

    def test_empty_actors_raises(self):
        with pytest.raises(ValueError, match="actors cannot be empty"):
            FieldSpec(
                field_id="field-1",
                worldline="alpha",
                temporal_scope={},
                spatial_scope={},
                actors=[],
            )


class TestActorCard:
    """Actor card: internal_tensions must be non-empty (breaks the N=5 width lock)."""

    def test_valid_actor_card_with_multiple_tensions(self):
        card = _make_actor_card(actor_id="actor-alpha", n_tensions=3)
        assert card.actor_id == "actor-alpha"
        assert card.name == "actor-alpha name"
        assert card.inherited_conditions == {"rank": "pioneer"}
        assert card.field_coordinates == {"x": 1, "y": 2}
        assert len(card.internal_tensions) == 3
        assert card.internal_tensions[0]["axis_A"] == "inherited"
        assert card.temporal_states == {"t0": "idle", "t1": "active"}
        assert card.meta == {}

    def test_single_tension_also_valid(self):
        card = _make_actor_card(actor_id="actor-beta", n_tensions=1)
        assert len(card.internal_tensions) == 1

    def test_empty_actor_id_raises(self):
        with pytest.raises(ValueError, match="actor_id cannot be empty"):
            _make_actor_card(actor_id="")

    def test_empty_internal_tensions_raises(self):
        # 打破 N=5 的關鍵：每行動者至少一條內在張力
        with pytest.raises(ValueError, match="internal_tensions cannot be empty"):
            _make_actor_card(n_tensions=0)


class TestSituationSpec:
    """Situation payload: digest mandatory; vector_6d & actors optional."""

    def test_valid_situation_with_actors_and_vector(self):
        card = _make_actor_card(actor_id="actor-alpha")
        sit = SituationSpec(
            digest="demo situation digest",
            local_texture={"region": "delta", "season": "autumn"},
            vector_6d=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            actors={"actor-alpha": card},
        )
        assert sit.digest == "demo situation digest"
        assert sit.local_texture == {"region": "delta", "season": "autumn"}
        assert sit.vector_6d == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        assert sit.actors["actor-alpha"] is card

    def test_vector_6d_defaults_to_none(self):
        sit = SituationSpec(digest="demo digest", local_texture={})
        assert sit.vector_6d is None

    def test_flat_dict_upgrade_without_actors_is_valid(self):
        # 從扁平 dict 升級的相容性：無 actors 也合法
        sit = SituationSpec(digest="demo digest", local_texture={"region": "delta"})
        assert sit.actors == {}
        assert sit.vector_6d is None

    def test_empty_digest_raises(self):
        with pytest.raises(ValueError, match="digest cannot be empty"):
            SituationSpec(digest="", local_texture={})


class TestFieldLog:
    """Field history log: append-only generation-layer records."""

    def test_valid_field_log_starts_empty(self):
        log = FieldLog(field_id="field-1")
        assert log.field_id == "field-1"
        assert log.entries == []
        assert log.meta == {}

    def test_empty_field_id_raises(self):
        with pytest.raises(ValueError, match="field_id cannot be empty"):
            FieldLog(field_id="")

    def test_add_entry_appends_layer_selected_unselected(self):
        log = FieldLog(field_id="field-1")
        log.add_entry(
            layer=1,
            selected={"label": "branch-a", "axis_A": "inherited"},
            unselected=[{"label": "branch-b"}, {"label": "branch-c"}],
        )
        assert len(log.entries) == 1
        entry = log.entries[0]
        assert set(entry.keys()) == {"layer", "selected", "unselected"}
        assert entry["layer"] == 1
        assert entry["selected"] == {"label": "branch-a", "axis_A": "inherited"}
        assert entry["unselected"] == [{"label": "branch-b"}, {"label": "branch-c"}]

    def test_multiple_add_entries_accumulate_in_order(self):
        log = FieldLog(field_id="field-1")
        log.add_entry(layer=1, selected={"label": "l1"}, unselected=[])
        log.add_entry(layer=2, selected={"label": "l2"}, unselected=[{"label": "l1"}])
        log.add_entry(layer=3, selected={"label": "l3"}, unselected=[])
        assert len(log.entries) == 3
        assert [e["layer"] for e in log.entries] == [1, 2, 3]
        assert log.entries[1]["selected"] == {"label": "l2"}
        assert log.entries[1]["unselected"] == [{"label": "l1"}]


class TestContractPipelineIntegration:
    """T17a 契約 to_dict() 產物必須能被現有探針管線消費（N3 整合缺陷修復）。

    SituationSpec 是 dataclass；探針管線吃扁平 dict。to_dict() 必須輸出
    probe.py 期待的鍵形狀（digest / local_texture / 6d_vector / actors），
    否則 _root_label 靜默退化、_extract_situation_vector 遺失參考、
    json.dumps 硬崩潰。
    """

    def _make_situation(self, **kw) -> SituationSpec:
        card = _make_actor_card(actor_id="actor-alpha")
        defaults = dict(
            digest="示例處境 digest 第一行",
            local_texture={"k": "v"},
            vector_6d=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            actors={"actor-alpha": card},
        )
        defaults.update(kw)
        return SituationSpec(**defaults)

    def test_to_dict_consumable_by_root_label(self):
        # _root_label 吃 digest 首行（≤20 字）；非泛化「處境」代表有命中。
        d = self._make_situation().to_dict()
        label = _root_label(d)
        assert label.startswith("示例處境")
        assert label != "處境"

    def test_to_dict_vector_list_maps_to_6d_vector_dict(self):
        # list [v0..v5] → dict {"d1".."d6"}，_extract_situation_vector 只吃 dict。
        d = self._make_situation().to_dict()
        vec = _extract_situation_vector(d)
        assert vec == {
            "d1": 0.1, "d2": 0.2, "d3": 0.3,
            "d4": 0.4, "d5": 0.5, "d6": 0.6,
        }

    def test_to_dict_vector_dict_passes_through(self):
        # 已是管線原生形狀（dict 含 d1/d2/d3）→ 直接透傳。
        sit = self._make_situation(vector_6d={"d1": 1, "d2": 0, "d3": 0})
        d = sit.to_dict()
        assert _extract_situation_vector(d) == {"d1": 1, "d2": 0, "d3": 0}

    def test_to_dict_is_json_dumps_safe(self):
        # json.dumps(payload) 對 dataclass 硬崩潰——to_dict() 產物必須可序列化。
        d = self._make_situation().to_dict()
        assert bool(json.dumps(d, ensure_ascii=False))
        assert json.loads(json.dumps(d, ensure_ascii=False))["digest"] == "示例處境 digest 第一行"

    def test_to_dict_empty_actors_and_none_vector_still_valid(self):
        # 空 actors / vector_6d=None → to_dict() 仍合法、仍可被管線消費。
        sit = self._make_situation(actors={}, vector_6d=None)
        d = sit.to_dict()
        assert d["actors"] == {}
        assert d["6d_vector"] is None
        assert _extract_situation_vector(d) is None
        assert _root_label(d) != "處境"
        assert bool(json.dumps(d, ensure_ascii=False))

    def test_to_dict_actor_cards_become_plain_dicts(self):
        d = self._make_situation().to_dict()
        card_dict = d["actors"]["actor-alpha"]
        assert isinstance(card_dict, dict)
        assert card_dict["actor_id"] == "actor-alpha"
        assert card_dict["internal_tensions"] == [
            {"axis_A": "inherited", "axis_B": "expression", "note": "tension-0"},
            {"axis_A": "inherited", "axis_B": "expression", "note": "tension-1"},
        ]

    def test_all_world_interface_contracts_have_to_dict(self):
        # FieldSpec / ActorCard / SituationSpec / FieldLog 四契約皆有 to_dict。
        assert callable(FieldSpec.to_dict)
        assert callable(ActorCard.to_dict)
        assert callable(SituationSpec.to_dict)
        assert callable(FieldLog.to_dict)


class TestWorldAgnosticContracts:
    """OS 層 schema 零世界特定內容：contracts.py 原始碼不得含世界特定詞。"""

    FORBIDDEN_TERMS = ["/home/octy/projects/ECC", "1914", "巴爾幹", "奧匈"]

    def _contracts_source(self) -> str:
        path = Path(__file__).resolve().parent.parent / "spectrum_os" / "contracts.py"
        return path.read_text(encoding="utf-8")

    @pytest.mark.parametrize("term", FORBIDDEN_TERMS)
    def test_no_world_specific_term_in_contracts_source(self, term):
        source = self._contracts_source()
        assert term not in source, f"contracts.py 含有世界特定詞 {term!r}"
