"""Tests for round_loop Mode B 委派整合（probe_converge_fn / alt_gate 意外枚舉）。

把新元件 round_loop（Mode A 輪界狀態機）與既有 Mode B 探針機制
（probe_expand_layer / probe_select / probe_converge_round / alt_gate_enumerate）
連貫整合——第一輪 alpha 模擬呈現「展開可能性空間 → 人選 → 坍縮」。

覆蓋（force-model-social-dynamics.md §2.8 / §12.13）：
A. 處境組裝器——probe → SituationSpec 形狀（digest / local_texture / 6d_vector /
   actors），含 alpha 場 yaml 素材（load_field_actors）。
B. 委派接線——撕裂後 probe_converge_fn 被呼叫、situation 形狀正確。
C. branch → decision 對映——mod_alpha / mod_beta / state → settle 生效。
D. 統一寫回——給定 probe_converge_fn 時，寫回只走 delegate（round_loop 不重複寫）。
E. 向後相容——未給 probe_converge_fn 時機械 settlement_fn 行為不變。
F. FieldLog 契約——add_entry(layer, selected, unselected)，不可整包塞 decision。
G. §12.13——意外枚舉路徑先 record_mode_a_history → mode_a_posterior_init → 枚舉。

🔴 無真實 LLM——全部用 mock probe_converge_fn / enumerate_fn / record fn。
"""

from __future__ import annotations

import numpy as np
import pytest

from spectrum_os.contracts import ActorCard, FieldLog
from spectrum_os.kernel.forces import ForceFieldDynamics
from spectrum_os.synth.round_loop import (
    RoundLoop,
    assemble_situation,
    load_field_actors,
)

SITUATION_YAML = """\
digest: |
  1900 年 1 月：世紀之交的世界。德國在威廉二世的『世界政策』下迅速工業化
  並擴張海軍，挑戰英國海上霸權。
local_texture:
  great_power_system: "bismarckian-alliance-system"
  naval_race: "德國海軍法案（1898 年通過）"
vector_6d:
  d1: -1
  d2: -1
  d3: -1
  d4: 6
  d5: 0.2
  d6: 0.3
"""

ACTOR_YAML = """\
actor_id: "german-reich"
name: "德意志帝國"
inherited_conditions:
  regime: "君主立憲"
field_coordinates:
  economic_capital: 0.8
internal_tensions:
  - id: "personal-rule"
    label: "威廉二世的個人統治"
    description: "皇帝繞過首相個人主導外交。"
    protagonists: ["威廉二世", "帝國宰相"]
force_balance_1900:
  revolutionary_force: 4.0
  conservative_force: 6.0
  latent_force: 1.0
temporal_states:
  "1900": {status: "世界政策", note: "海軍擴張"}
meta:
  worldline: "alpha"
"""


def _engine(**kw) -> ForceFieldDynamics:
    """單節點測試引擎（參數是語義錨點，非物理常數——v1.3 §3.3）。"""
    defaults = dict(
        revolutionary_force=10.0,
        conservative_force=6.0,
        growth_r=0.02,
        growth_c=0.01,
        alpha=0.012,
        beta=0.010,
        dt=0.5,
    )
    defaults.update(kw)
    return ForceFieldDynamics(**defaults)


def _card(aid: str = "german-reich", **kw) -> ActorCard:
    d = dict(
        actor_id=aid,
        name=aid,
        inherited_conditions={"regime": "x"},
        field_coordinates={"economic_capital": 0.8},
        internal_tensions=[{"id": "t1", "label": "張力一"}],
        temporal_states={},
        revolutionary_force=4.0,
        conservative_force=6.0,
        latent_force=1.0,
    )
    d.update(kw)
    return ActorCard(**d)


def _minimal_probe() -> dict:
    """直接餵 assemble_situation 的最小探針 dict（不需真引擎）。"""
    return {
        "round": 2,
        "status": "tear",
        "tear_info": {
            "triggered": True,
            "at_time": 4.0,
            "round": 2,
            "criteria": ["saturation", "decoupling"],
        },
        "state": {
            "t": 4.0,
            "R": np.array([8.0]),
            "C": np.array([4.0]),
            "P": np.array([2.0]),
            "S": np.array([0.667]),
            "T": np.array([10.0]),
            "tension": np.array([0.8]),
        },
        "trajectory": [],
        "s_series": [np.array([0.6]), np.array([0.667])],
        "context": {"round": 2},
    }


def _make_converge_mock(selected_branch: dict, *, written: dict | None = None):
    """兩段式 probe_converge_round mock：第一呼只展開（selected_label=None），
    第二呼坍縮（selected_label 給定）→ field_log_entry.selected。

    回傳 ``(converge_fn, calls)``——``calls`` 記錄每次呼叫的 kw。
    """
    calls: list[dict] = []

    def converge_fn(situation, constraint_field=None, **kw):
        calls.append({"situation": situation, "constraint_field": constraint_field, **kw})
        if kw.get("selected_label") is None:
            return {
                "expanded": {
                    "layer_entry": {"layer": 1, "branches": [selected_branch]}
                },
                "collapse": None,
                "field_log_entry": None,
                "next_state": None,
                "next_reflect_on": None,
                "path": [],
                "written": {},
                "view": "tree",
            }
        return {
            "expanded": {
                "layer_entry": {"layer": 1, "branches": [selected_branch]}
            },
            "collapse": {
                "selected": kw["selected_label"],
                "layer": 1,
                "unselected": ["未選之路"],
            },
            "field_log_entry": {"selected": selected_branch},
            "next_state": None,
            "next_reflect_on": [selected_branch],
            "path": [kw["selected_label"]],
            "written": written or {},
            "view": "tree",
        }

    return converge_fn, calls


# ---------------------------------------------------------------------------
# A. 處境組裝器（SituationSpec 形狀）
# ---------------------------------------------------------------------------

class TestSituationAssembler:
    def test_assemble_situation_shape_from_probe(self):
        sit = assemble_situation(_minimal_probe(), node_ids=["germany"])
        assert set(sit) == {"digest", "local_texture", "6d_vector", "actors"}
        # digest：撕裂時刻的世界狀態敘述
        assert isinstance(sit["digest"], str) and sit["digest"]
        assert "撕裂" in sit["digest"] and "第 2 輪" in sit["digest"]
        assert "saturation" in sit["digest"]  # 殘差準則進入 digest
        # local_texture：結構化參數
        lt = sit["local_texture"]
        assert lt["tear_node"]["node_id"] == "germany"
        assert lt["tension"] == [0.8]
        assert lt["p_r_release"] == [2.0]
        assert lt["s_final"] == [0.667]
        assert lt["criteria"] == ["saturation", "decoupling"]
        assert lt["status"] == "tear" and lt["round"] == 2
        # 6d_vector：含 d1/d2/d3 的 dict（probe.py _extract_situation_vector 契約）
        vec = sit["6d_vector"]
        assert {"d1", "d2", "d3"} <= set(vec)
        assert vec["d1"] == 1  # tension 0.8 > 0.5 → 危機顯現
        assert vec["d2"] == 1  # S 0.667 > 0.5 → R 主導
        assert vec["d3"] == 1  # tear + criteria → 意外
        assert sit["actors"] == {}

    def test_assemble_situation_with_actor_cards(self):
        card = _card("german-reich")
        sit = assemble_situation(_minimal_probe(), actors={"german-reich": card})
        a = sit["actors"]["german-reich"]
        assert a["actor_id"] == "german-reich"
        assert a["revolutionary_force"] == 4.0
        assert a["conservative_force"] == 6.0
        assert a["latent_force"] == 1.0
        assert a["internal_tensions"][0]["label"] == "張力一"

    def test_assemble_situation_accepts_plain_dicts(self):
        sit = assemble_situation(
            _minimal_probe(),
            actors={"german-reich": {"actor_id": "german-reich", "revolutionary_force": 9.0}},
        )
        assert sit["actors"]["german-reich"]["revolutionary_force"] == 9.0

    def test_assemble_situation_world_digest_and_vector_override(self):
        sit = assemble_situation(
            _minimal_probe(),
            world_digest="1900 年 1 月：世紀之交。",
            vector_6d={"d1": -1, "d2": -1, "d3": -1},
            local_texture_extra={"extra": "x"},
        )
        assert sit["digest"].startswith("1900 年 1 月：世紀之交。")
        assert sit["6d_vector"] == {"d1": -1, "d2": -1, "d3": -1}
        assert sit["local_texture"]["extra"] == "x"

    def test_load_field_actors_from_yaml(self, tmp_path):
        (tmp_path / "situation-1900-01.yaml").write_text(SITUATION_YAML, encoding="utf-8")
        actors_dir = tmp_path / "actors"
        actors_dir.mkdir()
        (actors_dir / "german-reich.yaml").write_text(ACTOR_YAML, encoding="utf-8")

        digest, lt, vec, actors = load_field_actors(tmp_path)
        assert digest and "1900" in digest
        assert lt["great_power_system"] == "bismarckian-alliance-system"
        assert vec["d1"] == -1
        assert "german-reich" in actors
        card = actors["german-reich"]
        assert card.revolutionary_force == 4.0
        assert card.conservative_force == 6.0
        assert card.latent_force == 1.0
        assert card.internal_tensions[0]["id"] == "personal-rule"


# ---------------------------------------------------------------------------
# B. 委派接線（撕裂後 probe_converge_fn 被呼叫、situation 形狀正確）
# ---------------------------------------------------------------------------

class TestModeBDelegation:
    def test_tear_delegates_with_correct_situation_shape(self):
        eng = _engine(dt=0.5)
        converge_fn, calls = _make_converge_mock({"label": "b1", "layer": 1})
        loop = RoundLoop(
            eng,
            lambda t, ctx: t >= 3.0,
            probe_converge_fn=converge_fn,
            mode_b={
                "actors": {"german-reich": _card()},
                "selected_label_fn": lambda result, ctx: None,  # 只展開不選
            },
        )
        result = loop.run_one_round(max_steps=100)
        assert result["status"] == "tear"
        loop.settle()

        assert len(calls) == 1
        sit = calls[0]["situation"]
        assert set(sit) == {"digest", "local_texture", "6d_vector", "actors"}
        assert isinstance(sit["digest"], str) and sit["digest"]
        assert sit["local_texture"]["round"] == 1
        assert sit["local_texture"]["status"] == "tear"
        assert {"d1", "d2", "d3"} <= set(sit["6d_vector"])
        # actors 含三力（revolutionary/conservative/latent force）
        a = sit["actors"]["german-reich"]
        assert a["revolutionary_force"] == 4.0
        assert a["conservative_force"] == 6.0
        assert a["latent_force"] == 1.0
        # 第一呼是展開（selected_label=None、不寫 field_log）
        assert calls[0]["selected_label"] is None
        assert calls[0]["field_log"] is None
        # constraint_field 從探針組裝（非 None）
        assert calls[0]["constraint_field"] is not None
        assert loop.rounds_completed() == 1
        assert loop.settlement_history()[-1]["mode_b"]["delegated"] is True

    def test_expand_only_round_no_selection_no_write(self):
        """selected_label_fn 回 None → 只展開給人看：不坍縮、無決策、不寫回。"""
        eng = _engine(dt=0.5)
        converge_fn, calls = _make_converge_mock({"label": "b1", "layer": 1})
        loop = RoundLoop(
            eng,
            lambda t, ctx: t >= 3.0,
            probe_converge_fn=converge_fn,
            mode_b={"selected_label_fn": lambda result, ctx: None},
        )
        loop.run_one_round(max_steps=100)
        rec = loop.settle()
        assert len(calls) == 1
        assert rec["applied"] == {}
        assert rec["written"] == {}
        assert len(loop.field_log.entries) == 0
        assert loop.rounds_completed() == 1


# ---------------------------------------------------------------------------
# C. branch → decision 對映（mod_alpha / mod_beta / state → settle 生效）
# ---------------------------------------------------------------------------

class TestBranchToDecision:
    def test_branch_mapped_to_decision_and_applied(self):
        eng = _engine(alpha=0.012, beta=0.010, dt=0.5)
        branch = {
            "label": "選定之路",
            "layer": 1,
            "mod_alpha": 2.0,
            "mod_beta": 0.5,
            "state": {"R": 30.0},
        }
        converge_fn, _ = _make_converge_mock(branch)
        loop = RoundLoop(
            eng,
            lambda t, ctx: t >= 3.0,
            probe_converge_fn=converge_fn,
            mode_b={"selected_label_fn": lambda result, ctx: "選定之路"},
        )
        loop.run_one_round(max_steps=100)
        rec = loop.settle()

        # branch → decision 對映：mod_alpha/mod_beta/state
        assert rec["applied"]["mod_alpha"] == [2.0]
        assert rec["applied"]["mod_beta"] == [0.5]
        assert rec["applied"]["state"] == {"R": [30.0]}
        assert rec["decision"]["accepted_path"] == "選定之路"
        assert rec["decision"]["rejected_paths"] == ["未選之路"]  # collapse.unselected
        # settle 生效：effective_rates 反映合成 mod
        a, b = eng.effective_rates(0.0)
        assert a.tolist() == pytest.approx([0.012 * 2.0])
        assert b.tolist() == pytest.approx([0.010 * 0.5])
        # 下一輪從覆寫後的狀態起點
        result2 = loop.run_one_round(max_steps=2)
        assert result2["start_state"]["R"][0] == pytest.approx(30.0)
        assert loop.rounds_completed() == 1

    def test_custom_branch_to_decision_override(self):
        eng = _engine(alpha=0.012, dt=0.5)
        converge_fn, _ = _make_converge_mock({"label": "x", "layer": 1})
        loop = RoundLoop(
            eng,
            lambda t, ctx: t >= 3.0,
            probe_converge_fn=converge_fn,
            mode_b={
                "selected_label_fn": lambda result, ctx: "x",
                "branch_to_decision": lambda branch, result, ctx: {"mod_alpha": 3.0},
            },
        )
        loop.run_one_round(max_steps=100)
        rec = loop.settle()
        assert rec["applied"]["mod_alpha"] == [3.0]
        a, _ = eng.effective_rates(0.0)
        assert a.tolist() == pytest.approx([0.012 * 3.0])


# ---------------------------------------------------------------------------
# D. 統一寫回（給定 probe_converge_fn 時只走 delegate，round_loop 不重複寫）
# ---------------------------------------------------------------------------

class TestUnifiedWriteback:
    def test_writeback_only_via_delegate(self):
        eng = _engine(dt=0.5)
        fl = FieldLog(field_id="mb-test")
        calls = []

        def converge_fn(situation, constraint_field=None, field_log=None, **kw):
            calls.append(
                {
                    "situation": situation,
                    "constraint_field": constraint_field,
                    "field_log": field_log,
                    **kw,
                }
            )
            if kw.get("selected_label") is None:
                return {
                    "expanded": {
                        "layer_entry": {"layer": 1, "branches": [{"label": "路"}]}
                    },
                    "collapse": None,
                    "field_log_entry": None,
                    "written": {},
                    "view": "tree",
                }
            # 模擬 probe_converge_round 的統一寫回：delegate 寫 FieldLog
            field_log.add_entry(1, {"label": kw["selected_label"]}, [{"label": "別條"}])
            return {
                "expanded": {"layer_entry": {"layer": 1, "branches": [{"label": "路"}]}},
                "collapse": {"selected": kw["selected_label"], "layer": 1, "unselected": ["別條"]},
                "field_log_entry": {"selected": {"label": kw["selected_label"]}},
                "written": {"field_log": "delegated"},
                "view": "tree",
            }

        loop = RoundLoop(
            eng,
            lambda t, ctx: t >= 3.0,
            probe_converge_fn=converge_fn,
            field_log=fl,
            mode_b={"selected_label_fn": lambda result, ctx: "路"},
        )
        loop.run_one_round(max_steps=100)
        rec = loop.settle()

        # delegate 被給到同一個 FieldLog（統一寫回的同一條 log）
        assert calls[-1]["field_log"] is fl
        # 只有 delegate 寫了一次——round_loop 不重複寫
        assert len(fl.entries) == 1
        # delegate 的 written 被收錄於結算記錄
        assert rec["written"] == {"field_log": "delegated"}
        assert rec["mode_b"]["delegate_wrote"] is True

    def test_store_paths_forwarded_to_delegate(self, tmp_path):
        eng = _engine(dt=0.5)
        store = {
            "field_log_path": tmp_path / "fl.json",
            "snapshot_dir": tmp_path / "rounds",
            "dashboard_path": tmp_path / "dash.md",
        }
        converge_fn, calls = _make_converge_mock({"label": "路", "layer": 1})
        loop = RoundLoop(
            eng,
            lambda t, ctx: t >= 3.0,
            probe_converge_fn=converge_fn,
            store=store,
            mode_b={"selected_label_fn": lambda result, ctx: "路"},
        )
        loop.run_one_round(max_steps=100)
        loop.settle()
        # 第二呼（坍縮+寫回）收到 store 的落盤路徑（統一寫回）
        second = calls[-1]
        assert second["field_log_path"] == str(tmp_path / "fl.json")
        assert second["snapshot_dir"] == str(tmp_path / "rounds")
        assert second["dashboard_path"] == str(tmp_path / "dash.md")
        # 且 delegate 被給到 loop 的 FieldLog
        assert second["field_log"] is loop.field_log


# ---------------------------------------------------------------------------
# E. 向後相容（未給 probe_converge_fn → 機械 settlement_fn 不變）
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_mechanical_path_unchanged_without_mode_b(self):
        eng = _engine(dt=0.5)
        loop = RoundLoop(
            eng, lambda t, ctx: t >= 4.0, lambda payload: {"accepted_path": "a"}
        )
        loop.run_one_round(max_steps=100)
        rec = loop.settle()
        assert rec["written"] == {}
        assert rec["applied"] == {}
        assert len(loop.field_log.entries) == 1
        assert loop.field_log.entries[0]["selected"]["accepted"] == "a"
        assert "mode_b" not in rec
        assert loop.rounds_completed() == 1

    def test_explicit_decision_overrides_delegation(self):
        """settle(decision=...) 顯式給決策 = 人類覆寫 → 機械路徑（不委派）。"""
        eng = _engine(dt=0.5)
        calls = []

        def converge_fn(situation, **kw):
            calls.append(kw)
            return {}

        loop = RoundLoop(
            eng,
            lambda t, ctx: t >= 3.0,
            probe_converge_fn=converge_fn,
            mode_b={"selected_label_fn": lambda result, ctx: "x"},
        )
        loop.run_one_round(max_steps=100)
        rec = loop.settle({"mod_alpha": 1.5, "accepted_path": "manual"})
        assert calls == []  # 人類覆寫 → 不委派
        assert rec["applied"]["mod_alpha"] == [1.5]
        assert len(loop.field_log.entries) == 1  # 機械路徑寫回
        a, _ = eng.effective_rates(0.0)
        assert a.tolist() == pytest.approx([0.012 * 1.5])


# ---------------------------------------------------------------------------
# F. FieldLog 契約（add_entry(layer, selected, unselected)，不整包塞 decision）
# ---------------------------------------------------------------------------

class TestFieldLogContract:
    def test_field_log_entry_shape_compliant(self):
        eng = _engine(dt=0.5)
        loop = RoundLoop(
            eng,
            lambda t, ctx: t >= 4.0,
            lambda payload: {"accepted_path": "a", "rejected_paths": ["b", "c"]},
        )
        loop.run_one_round(max_steps=100)
        loop.settle()
        e = loop.field_log.entries[0]
        # 契約形狀：{layer, selected, unselected}
        assert set(e) == {"layer", "selected", "unselected"}
        # selected 是結構化條目——不可把 decision 整包塞進 selected
        assert "decision" not in e["selected"]
        assert e["selected"]["accepted"] == "a"
        assert e["selected"]["rejected"] == ["b", "c"]
        assert e["selected"]["round"] == 1
        # unselected 維持疊加（label dict 清單）
        assert e["unselected"] == [{"label": "b"}, {"label": "c"}]


# ---------------------------------------------------------------------------
# G. §12.13 意外枚舉（record_mode_a_history → posterior → enumerate）
# ---------------------------------------------------------------------------

class TestEnumeratePath:
    def test_enumerate_records_mode_a_before_enumerate(self, tmp_path):
        eng = _engine(dt=0.5)
        order: list[tuple] = []
        log_path = tmp_path / "mode_a.jsonl"

        def record_fn(path, entry):
            order.append(("record", str(path), entry))

        def posterior_fn(history):
            order.append(("posterior", len(history)))
            return {"n_entries": len(history), "source": "mode_a_history"}

        def enum_fn(input_data, **kw):
            order.append(("enumerate",))
            return {
                "accidents": [
                    {
                        "pattern": "宮廷政變",
                        "instances": ["威廉二世遇刺"],
                        "domain": "政治",
                        "source": "emergent",
                        "usage": "expression",
                        "grounding": "合理推測",
                        "world_development": "攝政體制取代個人統治",
                    }
                ],
                "mode_a_posterior": input_data.get("mode_a_history"),
            }

        loop = RoundLoop(
            eng,
            lambda t, ctx: t >= 3.0,
            mode_b={
                "style": "enumerate",
                "mode_a_log_path": str(log_path),
                "record_mode_a_history_fn": record_fn,
                "mode_a_posterior_init_fn": posterior_fn,
                "enumerate_fn": enum_fn,
                "select_accident_fn": lambda accidents, ctx: accidents[0],
            },
        )
        loop.run_one_round(max_steps=100)
        rec = loop.settle()

        # §12.13 順序：record → posterior → enumerate（Mode B 不空手起跳）
        assert [o[0] for o in order] == ["record", "posterior", "enumerate"]
        assert order[0][1] == str(log_path)
        # 歷史條目是 ODE 軌跡（純機械）
        entry = order[0][2]
        assert entry["mode"] == "ode"
        assert entry["round"] == 1
        assert entry["spectrum"]["final_S"] is not None
        assert entry["source"] == "spectrum_os.synth.round_loop:mode_b"
        # 後驗初始化收到歷史（n_entries=1）
        assert order[1][1] == 1
        # decision 對映：accepted = accident pattern
        assert rec["decision"]["accepted_path"] == "宮廷政變"
        assert rec["mode_b"]["style"] == "enumerate"
        assert loop.rounds_completed() == 1

    def test_enumerate_input_carries_situation_and_posterior(self):
        eng = _engine(dt=0.5)
        seen: dict = {}

        def record_fn(path, entry):
            pass

        def posterior_fn(history):
            return {"n_entries": len(history), "source": "mode_a_history"}

        def enum_fn(input_data, **kw):
            seen["input"] = input_data
            return {"accidents": []}

        loop = RoundLoop(
            eng,
            lambda t, ctx: t >= 3.0,
            mode_b={
                "style": "enumerate",
                "record_mode_a_history_fn": record_fn,
                "mode_a_posterior_init_fn": posterior_fn,
                "enumerate_fn": enum_fn,
            },
        )
        loop.run_one_round(max_steps=100)
        loop.settle()
        data = seen["input"]
        # SituationSpec 形狀 + §12.13 後驗（資料層注入）
        assert set(data) >= {"digest", "local_texture", "6d_vector", "actors", "mode_a_history", "round"}
        assert data["mode_a_history"] == {"n_entries": 1, "source": "mode_a_history"}
        assert {"d1", "d2", "d3"} <= set(data["6d_vector"])
        assert data["round"] == 1

    def test_style_callable_driven_by_tear_semantics(self):
        """mode_b['style'] 是 callable → 由撕裂語義（tear_info）決定形態。"""
        eng = _engine(dt=0.5)
        seen_tear: list = []

        def record_fn(path, entry):
            pass

        def posterior_fn(history):
            return {"n_entries": len(history)}

        def enum_fn(input_data, **kw):
            return {"accidents": []}

        def style_fn(tear_info, probe):
            seen_tear.append(tear_info)
            return "enumerate"

        loop = RoundLoop(
            eng,
            lambda t, ctx: t >= 3.0,
            mode_b={
                "style": style_fn,
                "record_mode_a_history_fn": record_fn,
                "mode_a_posterior_init_fn": posterior_fn,
                "enumerate_fn": enum_fn,
            },
        )
        loop.run_one_round(max_steps=100)
        rec = loop.settle()
        assert seen_tear and seen_tear[0]["triggered"] is True
        assert rec["mode_b"]["style"] == "enumerate"

    def test_field_dir_actors_flow_into_situation(self, tmp_path):
        """alpha 場素材（situation yaml + actors/*.yaml）經 field_dir 流入處境。"""
        (tmp_path / "situation-1900-01.yaml").write_text(SITUATION_YAML, encoding="utf-8")
        actors_dir = tmp_path / "actors"
        actors_dir.mkdir()
        (actors_dir / "german-reich.yaml").write_text(ACTOR_YAML, encoding="utf-8")

        eng = _engine(dt=0.5)
        seen: dict = {}

        def converge_fn(situation, **kw):
            seen["situation"] = situation
            return {
                "expanded": {"layer_entry": {"layer": 1, "branches": [{"label": "b"}]}},
                "collapse": None,
                "field_log_entry": None,
                "written": {},
                "view": "tree",
            }

        loop = RoundLoop(
            eng,
            lambda t, ctx: t >= 3.0,
            probe_converge_fn=converge_fn,
            mode_b={"field_dir": str(tmp_path), "selected_label_fn": lambda r, c: None},
        )
        loop.run_one_round(max_steps=100)
        loop.settle()
        sit = seen["situation"]
        # world_digest 來自 situation yaml；actors 來自 actors/*.yaml
        assert "1900" in sit["digest"]
        assert "german-reich" in sit["actors"]
        a = sit["actors"]["german-reich"]
        assert a["revolutionary_force"] == 4.0
        assert a["conservative_force"] == 6.0
        assert a["latent_force"] == 1.0
