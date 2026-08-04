"""Tests for synth/convergence.py — T19 N1 收束 UX（convergence_view / probe_converge_round）。

全部 mock gate_fn，零真 API。世界無關檢定：渲染輸出不得含世界特定詞。
"""

from __future__ import annotations

import json

import pytest

from spectrum_os.contracts import FieldLog
from spectrum_os.kernel import verify as _verify
from spectrum_os.synth.convergence import convergence_view, probe_converge_round
from spectrum_os.synth.probe import TreeProbeError

# ---------------------------------------------------------------------------
# 測試資料（世界無關：中性標籤，不帶任何世界/歷史特定詞）
# ---------------------------------------------------------------------------

SITUATION = {
    "digest": "situation: mobilization gridlock, central delay, external mediation.",
    "local_texture": {"region": "north", "season": "summer"},
}

#: 世界無關檢定禁詞（出現即違規）。
WORLD_WORDS = [
    "ECC", "ecc", "巴爾幹", "1914", "沙俄", "德意志", "蘇維埃", "奧匈",
    "塞爾維亞", "薩拉熱窩", "hongkong", "香港",
]


def _branch(label: str, layer: int, rigidity: float = 0.6, **overrides) -> dict:
    """最小合法分支（含 T16 可選欄位 perspective/binding，供渲染測試）。"""
    b = {
        "label": label,
        "grounding": "situation-internal support",
        "axis_A": "emergent",
        "axis_B": "expression",
        "rigidity_prevalence": rigidity,
        "conditions": ["edge condition"],
        "perspective": "from observer eye",
        "binding": "opposing binding",
        "layer": layer,
        "children": [],
    }
    b.update(overrides)
    return b


def _make_tree() -> dict:
    """手動造一棵兩層工作樹（純 dict，零 API）。"""
    root = {
        "label": "situation root",
        "grounding": "處境（root）",
        "layer": 0,
        "parent_label": None,
        "children": [],
    }
    a1 = _branch("route-a", 1, 0.8)
    a2 = _branch("route-b", 1, 0.4)
    a3 = _branch("route-c", 1, 0.5)
    root["children"] = [a1, a2, a3]
    b1 = _branch("route-a1", 2, 0.9)
    a1["children"] = [b1]
    return {
        "root": root,
        "layers": [
            {"layer": 1, "date_ref": "t1", "branches": [a1, a2, a3]},
            {"layer": 2, "date_ref": "t2", "branches": [b1]},
        ],
        "paths": [],
        "prevalence": [],
        "rigidity_map": [],
        "rejected": [],
        "echo_notes": [],
        "meta": {},
    }


def make_mock_gate(layer_responses: list[dict]):
    """回傳 (mock_gate, calls)——mock 依序回傳每層 JSON，零真 API。"""
    calls: list[str] = []
    queue = list(layer_responses)

    def mock_gate(prompt, api_key, **kwargs):
        calls.append(prompt)
        if not queue:
            raise AssertionError("mock gate 被呼叫次數超過提供層數")
        return json.dumps(queue.pop(0), ensure_ascii=False)

    return mock_gate, calls


LAYER_1 = {
    "layer": 1,
    "date_ref": "t1",
    "branches": [
        {
            "label": "route-a", "grounding": "rail mismatch",
            "axis_A": "inherited", "axis_B": "expression",
            "rigidity_prevalence": 0.8, "conditions": ["no river crossing"],
        },
        {
            "label": "route-b", "grounding": "local stores",
            "axis_A": "emergent", "axis_B": "expression",
            "rigidity_prevalence": 0.4, "conditions": ["central delay"],
        },
        {
            "label": "route-c", "grounding": "embassy channels",
            "axis_A": "inherited", "axis_B": "expression",
            "rigidity_prevalence": 0.5, "conditions": ["channel open"],
        },
    ],
}

LAYER_2 = {
    "layer": 2,
    "date_ref": "t2",
    "branches": [
        {
            "label": "route-a1", "grounding": "staff lead",
            "axis_A": "inherited", "axis_B": "expression",
            "rigidity_prevalence": 0.9, "conditions": ["a unfrozen"],
            "parent": "route-a",
        },
        {
            "label": "route-b1", "grounding": "local autonomy",
            "axis_A": "emergent", "axis_B": "expression",
            "rigidity_prevalence": 0.3, "conditions": ["b in effect"],
            "parent": "route-b",
        },
    ],
}


@pytest.fixture(autouse=True)
def _reset_verify_state():
    """每次測試前清空 verify 全域 state log 並停用檔案持久化（測試隔離）。"""
    _verify.clear_state_log()
    _verify.init_log(None)
    yield


# ---------------------------------------------------------------------------
# 1. convergence_view：三種格式
# ---------------------------------------------------------------------------

class TestConvergenceViewFormats:
    def test_text_format_renders_tree(self):
        view = convergence_view(_make_tree())
        assert isinstance(view, str)
        # root 標頭在頂
        assert view.splitlines()[0] == "[root] situation root"
        # 每分支一行：層號 + label + 剛性 + （可選）標記
        assert "L1 route-a" in view
        assert "L2 route-a1" in view
        assert "剛性=0.8" in view
        assert "（可選）" in view
        # 縮排顯示層次（L2 行前面有接頭 │）
        lines = view.splitlines()
        l1_line = next(ln for ln in lines if "L1 route-a" in ln)
        l2_line = next(ln for ln in lines if "L2 route-a1" in ln)
        assert l1_line.index("L1") < l2_line.index("L2")

    def test_markdown_format_renders_tree(self):
        view = convergence_view(_make_tree(), format="markdown")
        assert isinstance(view, str)
        assert view.startswith("# 收束視圖")
        assert "**situation root**" in view
        assert "`L1` route-a" in view
        assert "`L2` route-a1" in view
        assert "（可選）" in view

    def test_json_format_structured(self):
        tree = _make_tree()
        view = convergence_view(tree, format="json")
        assert isinstance(view, dict)
        assert view["format"] == "json"
        assert view["tree"] is tree  # 原樣結構化輸出
        assert view["collapse"] is None
        assert view["selection"] is None
        selectable = {s["label"] for s in view["selectable"]}
        assert selectable == {"route-a", "route-b", "route-c", "route-a1"}

    def test_accepts_result_dict(self):
        tree = _make_tree()
        result = {"trees": [tree], "meta": {}}
        assert convergence_view(result) == convergence_view(tree)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="format"):
            convergence_view(_make_tree(), format="xml")

    def test_invalid_include_raises(self):
        with pytest.raises(ValueError, match="include"):
            convergence_view(_make_tree(), include=("label", "not-a-field"))

    def test_invalid_input_raises(self):
        with pytest.raises(TypeError):
            convergence_view(["not", "a", "dict"])

    def test_empty_tree_raises(self):
        with pytest.raises(ValueError, match="root"):
            convergence_view({"root": {}, "layers": []})


class TestConvergenceViewSelection:
    def test_collapse_marks_selected_and_unselected(self):
        tree = _make_tree()
        collapse = {
            "selected": "route-a",
            "layer": 1,
            "unselected": ["route-b", "route-c"],
        }
        view = convergence_view({"trees": [tree], "collapse": collapse})
        # 被選 / 未選標記出現在該層；未選不刪除（仍渲染）
        assert "route-a" in view
        assert "route-b" in view
        assert "route-c" in view
        lines = {ln.split("L1 ", 1)[-1].split("  ")[0] if "L1 " in ln else "" for ln in view.splitlines()}
        l1_lines = [ln for ln in view.splitlines() if "L1 " in ln]
        assert any("route-a" in ln and "（已選）" in ln for ln in l1_lines)
        assert any("route-b" in ln and "（未選）" in ln for ln in l1_lines)
        assert any("route-c" in ln and "（未選）" in ln for ln in l1_lines)
        # 其他層（L2）不受 collapse 影響，仍可選
        l2_lines = [ln for ln in view.splitlines() if "L2 " in ln]
        assert all("（可選）" in ln for ln in l2_lines)

    def test_json_selection_reflects_collapse(self):
        tree = _make_tree()
        collapse = {
            "selected": "route-a",
            "layer": 1,
            "unselected": ["route-b", "route-c"],
        }
        view = convergence_view({"trees": [tree], "collapse": collapse}, format="json")
        assert view["selection"]["selected"] == "route-a"
        assert set(view["selection"]["unselected"]) == {"route-b", "route-c"}
        selectable = {s["label"] for s in view["selectable"]}
        # 已選/未選分支不再是 selectable；L2 仍可選
        assert "route-a" not in selectable
        assert "route-b" not in selectable
        assert "route-a1" in selectable


class TestConvergenceViewFields:
    def test_include_filters_fields(self):
        tree = _make_tree()
        # 只 label → 無鏡角/束縛/依憑/剛性
        view = convergence_view(tree, include=("label",))
        assert "鏡角=" not in view
        assert "束縛=" not in view
        assert "依憑=" not in view
        assert "剛性=" not in view
        # 加 perspective → 鏡角出現
        view2 = convergence_view(tree, include=("label", "perspective"))
        assert "鏡角=from observer eye" in view2

    def test_missing_fields_rendered_as_dash(self):
        # 缺 perspective/binding 的分支 → 渲染為 —
        root = {
            "label": "situation root", "grounding": "處境（root）", "layer": 0,
            "parent_label": None, "children": [],
        }
        a1 = _branch("route-a", 1, 0.8)
        del a1["perspective"]
        del a1["binding"]
        root["children"] = [a1]
        tree = {
            "root": root,
            "layers": [{"layer": 1, "date_ref": "t1", "branches": [a1]}],
            "paths": [], "prevalence": [], "rigidity_map": [],
            "rejected": [], "echo_notes": [], "meta": {},
        }
        view = convergence_view(tree)
        assert "鏡角=—" in view
        assert "束縛=—" in view

    def test_layer_filter(self):
        tree = _make_tree()
        view = convergence_view(tree, layer=1)
        assert "L1 route-a" in view
        assert "L2 route-a1" not in view

    def test_world_agnostic(self):
        """渲染輸出零世界特定詞（世界無關檢定）。"""
        tree = _make_tree()
        for fmt in ("text", "markdown", "json"):
            view = convergence_view(tree, format=fmt)
            payload = view if isinstance(view, str) else json.dumps(view, ensure_ascii=False)
            for word in WORLD_WORDS:
                assert word not in payload, f"{fmt} 輸出含世界特定詞 {word!r}"


# ---------------------------------------------------------------------------
# 2. probe_converge_round：完整一輪
# ---------------------------------------------------------------------------

class TestProbeConvergeRound:
    def test_round_full_flow_with_field_log(self):
        mock_gate, calls = make_mock_gate([LAYER_1])
        field_log = FieldLog(field_id="f-test")
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a", field_log=field_log,
        )
        # view：收束後人看的文字樹（N2 缺口 1 修復——回傳 view 反映本輪已選/未選）
        assert isinstance(r["view"], str)
        assert "route-a" in r["view"]
        assert "（已選）" in r["view"]
        assert "（未選）" in r["view"]
        # collapse：人選的那條坍縮，未選不刪除
        assert r["collapse"]["selected"] == "route-a"
        assert set(r["collapse"]["unselected"]) == {"route-b", "route-c"}
        # next_state / next_reflect_on：下一輪續接
        assert r["next_state"] is not None
        assert [b["label"] for b in r["next_reflect_on"]] == ["route-a"]
        # FieldLog：selected + unselected 一起寫入（維持疊加）
        assert len(field_log.entries) == 1
        entry = field_log.entries[0]
        assert entry["layer"] == 1
        assert entry["selected"]["label"] == "route-a"
        assert {b["label"] for b in entry["unselected"]} == {"route-b", "route-c"}
        assert r["field_log_entry"] is entry
        assert len(calls) == 1

    def test_round_no_selection_expands_only(self):
        mock_gate, calls = make_mock_gate([LAYER_1])
        field_log = FieldLog(field_id="f-test")
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label=None, field_log=field_log,
        )
        # 只展開給人看：不坍縮、不寫場 log、無 next_reflect_on
        assert r["collapse"] is None
        assert r["next_reflect_on"] is None
        assert r["field_log_entry"] is None
        assert field_log.entries == []
        assert "route-a" in r["view"]
        assert len(calls) == 1

    def test_round_chains_two_layers(self):
        mock_gate, calls = make_mock_gate([LAYER_1, LAYER_2])
        r1 = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a",
        )
        r2 = probe_converge_round(
            situation=SITUATION, state=r1["next_state"],
            reflect_on=r1["next_reflect_on"],
            gate_fn=mock_gate, selected_label="route-a1",
        )
        assert r2["expanded"]["layer_entry"]["layer"] == 2
        assert r2["collapse"]["selected"] == "route-a1"
        # 第二層 payload 的 reflection 只有人選的那 1 個父
        payload2 = json.loads(calls[1].split("\n[task:tree_generate]")[0])
        assert [b["label"] for b in payload2["reflection"]["passed_branches"]] == [
            "route-a"
        ]
        assert [b["label"] for b in r2["next_reflect_on"]] == ["route-a1"]

    def test_round_invalid_selection_raises_and_no_log_write(self):
        mock_gate, calls = make_mock_gate([LAYER_1])
        field_log = FieldLog(field_id="f-test")
        with pytest.raises(TreeProbeError, match="selected_label"):
            probe_converge_round(
                situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
                selected_label="not-a-branch", field_log=field_log,
            )
        # 攔截在收束副作用之前：不寫場 log、verify 無殘留
        assert field_log.entries == []
        assert _verify._state_log == []
        assert len(calls) == 1  # 只展開（gate 呼叫），未坍縮

    def test_round_state_log_persisted(self, tmp_path):
        mock_gate, _ = make_mock_gate([LAYER_1])
        log_path = str(tmp_path / "converge.jsonl")
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a", state_log_path=log_path,
        )
        assert r["collapse"]["selected"] == "route-a"
        with open(log_path, encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        assert any(
            e.get("gate_type") == "probe_tree"
            and e.get("selected_branch") == "route-a"
            for e in lines
        )

    def test_round_view_format_forwarded(self):
        mock_gate, _ = make_mock_gate([LAYER_1])
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a", view_format="json",
        )
        assert isinstance(r["view"], dict)
        assert r["view"]["format"] == "json"

    def test_round_world_agnostic(self):
        mock_gate, _ = make_mock_gate([LAYER_1])
        field_log = FieldLog(field_id="f-test")
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a", field_log=field_log,
        )
        for word in WORLD_WORDS:
            assert word not in r["view"]
            assert word not in json.dumps(field_log.to_dict(), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 3. N2 修復：選後渲染（缺口 1）+ 累積路徑標記（缺口 2）
# ---------------------------------------------------------------------------

class TestPostSelectionRendering:
    def test_round_view_marks_selected_and_unselected_after_select(self):
        """選後 view：本輪 selected 標（已選）、同層未選標（未選）、他層仍（可選）。"""
        mock_gate, _ = make_mock_gate([LAYER_1])
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a",
        )
        lines = r["view"].splitlines()
        l1_lines = [ln for ln in lines if "L1 " in ln]
        assert len(l1_lines) == 3
        assert any("route-a" in ln and "（已選）" in ln for ln in l1_lines)
        assert any("route-b" in ln and "（未選）" in ln for ln in l1_lines)
        assert any("route-c" in ln and "（未選）" in ln for ln in l1_lines)

    def test_round_view_all_selectable_without_selection(self):
        """選前 view：全部（可選），無（已選）/（未選）標記。"""
        mock_gate, _ = make_mock_gate([LAYER_1])
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label=None,
        )
        lines = r["view"].splitlines()
        branch_lines = [ln for ln in lines if "L1 " in ln]
        assert len(branch_lines) == 3
        assert all("（可選）" in ln for ln in branch_lines)
        assert not any("（已選）" in ln for ln in lines)
        assert not any("（未選）" in ln for ln in lines)

    def test_round_view_json_selection_reflects_collapse(self):
        """json 選後 view：selection 反映本輪收束。"""
        mock_gate, _ = make_mock_gate([LAYER_1])
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a", view_format="json",
        )
        assert r["view"]["selection"]["selected"] == "route-a"
        assert set(r["view"]["selection"]["unselected"]) == {"route-b", "route-c"}
        selectable = {s["label"] for s in r["view"]["selectable"]}
        assert "route-a" not in selectable


class TestAccumulatedPathMarking:
    def test_convergence_view_path_marks_walked_labels(self):
        """path 參數：走過的 label 標（已選），其餘仍（可選）。"""
        tree = _make_tree()
        view = convergence_view(tree, path=["route-a"])
        lines = view.splitlines()
        l1_lines = [ln for ln in lines if "L1 " in ln]
        assert any("route-a" in ln and "（已選）" in ln for ln in l1_lines)
        assert any("route-b" in ln and "（可選）" in ln for ln in l1_lines)
        # L2 不受 path 影響，仍可選
        assert all("（可選）" in ln for ln in lines if "L2 " in ln)

    def test_convergence_view_field_log_marks_walked_labels(self):
        """field_log 參數：從場 log 的 selected entry 讀已走路徑。"""
        tree = _make_tree()
        field_log = FieldLog(field_id="f-test")
        field_log.add_entry(
            1,
            _branch("route-a", 1, 0.8),
            [_branch("route-b", 1, 0.4), _branch("route-c", 1, 0.5)],
        )
        view = convergence_view(tree, field_log=field_log)
        lines = view.splitlines()
        l1_lines = [ln for ln in lines if "L1 " in ln]
        assert any("route-a" in ln and "（已選）" in ln for ln in l1_lines)
        assert any("route-b" in ln and "（可選）" in ln for ln in l1_lines)

    def test_collapse_takes_precedence_within_its_layer(self):
        """collapse 層內標記優先於 path——該層未選標（未選）不被 path 覆蓋。"""
        tree = _make_tree()
        # 在 L2 加一條 sibling，讓 collapse.unselected 有真實節點
        b2 = _branch("route-a2", 2, 0.7)
        tree["root"]["children"][0]["children"].append(b2)
        tree["layers"][1]["branches"].append(b2)
        collapse = {
            "selected": "route-a1", "layer": 2, "unselected": ["route-a2"],
        }
        view = convergence_view(
            {"trees": [tree], "collapse": collapse}, path=["route-a1"],
        )
        lines = view.splitlines()
        l2_lines = [ln for ln in lines if "L2 " in ln]
        assert any("route-a1" in ln and "（已選）" in ln for ln in l2_lines)
        assert any("route-a2" in ln and "（未選）" in ln for ln in l2_lines)

    def test_json_selectable_excludes_walked_labels(self):
        """json：path 內的 label 不再是 selectable，且 walked 欄位列出。"""
        tree = _make_tree()
        view = convergence_view(tree, format="json", path=["route-a"])
        selectable = {s["label"] for s in view["selectable"]}
        assert "route-a" not in selectable
        assert "route-b" in selectable
        assert view["walked"] == ["route-a"]

    def test_round_accumulates_path_across_rounds(self):
        """多輪：回傳 path 累積，round2 view 標記前輪已選（L1 route-a（已選））。"""
        mock_gate, calls = make_mock_gate([LAYER_1, LAYER_2])
        r1 = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a",
        )
        assert r1["path"] == ["route-a"]
        r2 = probe_converge_round(
            situation=SITUATION, state=r1["next_state"],
            reflect_on=r1["next_reflect_on"],
            gate_fn=mock_gate, selected_label="route-a1", path=r1["path"],
        )
        assert r2["path"] == ["route-a", "route-a1"]
        lines = r2["view"].splitlines()
        # 前輪已選（L1 route-a）→（已選）；同層未走（route-b/c）仍（可選）
        l1_lines = [ln for ln in lines if "L1 " in ln]
        assert any("route-a" in ln and "（已選）" in ln for ln in l1_lines)
        assert any("route-b" in ln and "（可選）" in ln for ln in l1_lines)
        # 本輪 L2：route-a1（已選）、route-b1（未選）
        l2_lines = [ln for ln in lines if "L2 " in ln]
        assert any("route-a1" in ln and "（已選）" in ln for ln in l2_lines)
        assert any("route-b1" in ln and "（未選）" in ln for ln in l2_lines)
        assert len(calls) == 2

    def test_round_field_log_marks_prior_selection(self):
        """多輪 + 共用 field_log：round2 view 從場 log 讀前輪已選（L1 route-a（已選））。"""
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2])
        field_log = FieldLog(field_id="f-test")
        r1 = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a", field_log=field_log,
        )
        r2 = probe_converge_round(
            situation=SITUATION, state=r1["next_state"],
            reflect_on=r1["next_reflect_on"],
            gate_fn=mock_gate, selected_label="route-a1", field_log=field_log,
        )
        lines = r2["view"].splitlines()
        l1_lines = [ln for ln in lines if "L1 " in ln]
        assert any("route-a" in ln and "（已選）" in ln for ln in l1_lines)
        l2_lines = [ln for ln in lines if "L2 " in ln]
        assert any("route-a1" in ln and "（已選）" in ln for ln in l2_lines)
        assert any("route-b1" in ln and "（未選）" in ln for ln in l2_lines)
        assert len(field_log.entries) == 2

    def test_round_state_log_entry_exposed_on_inmemory_fallback(self):
        """state_log_path=None：回傳 state_log_entry（in-memory fallback 可查）。"""
        mock_gate, _ = make_mock_gate([LAYER_1])
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a",
        )
        assert r["state_log_entry"] is not None
        assert r["state_log_entry"]["selected_branch"] == "route-a"
        assert r["state_log_entry"]["layer"] == 1


# ---------------------------------------------------------------------------
# 4. T19 companion：落盤層接入（指向哪就寫到哪——仿 warfare dashboard）
# ---------------------------------------------------------------------------

class TestPersistenceWiring:
    def test_round_writes_dashboard_fieldlog_snapshot(self, tmp_path):
        """指定三個路徑 → 收束後全部寫檔（.bak 備份 + 快照命名）。"""
        mock_gate, _ = make_mock_gate([LAYER_1])
        field_log = FieldLog(field_id="f-test")
        dash = tmp_path / "rounds" / "round-1-dashboard.md"
        fl_path = tmp_path / "field-log.json"
        snap_dir = tmp_path / "snapshots"
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a", field_log=field_log,
            dashboard_path=dash, field_log_path=fl_path, snapshot_dir=snap_dir,
        )
        # written：三鍵都在，指向正確
        assert set(r["written"].keys()) == {"dashboard", "field_log", "snapshot"}
        assert r["written"]["dashboard"] == dash
        assert r["written"]["field_log"] == fl_path
        assert r["written"]["snapshot"].name == "converge-round-01.json"
        # dashboard 內容 = 本輪 view（文字樹）
        assert dash.read_text(encoding="utf-8") == r["view"]
        assert "route-a" in dash.read_text(encoding="utf-8")
        # FieldLog 落盤：selected + unselected 疊加
        saved = json.loads(fl_path.read_text(encoding="utf-8"))
        assert saved["field_id"] == "f-test"
        assert saved["entries"][0]["selected"]["label"] == "route-a"
        assert {b["label"] for b in saved["entries"][0]["unselected"]} == {
            "route-b", "route-c",
        }
        # 快照：round / collapse / path
        snap = json.loads(
            (snap_dir / "converge-round-01.json").read_text(encoding="utf-8")
        )
        assert snap["round"] == 1
        assert snap["collapse"]["selected"] == "route-a"
        assert snap["path"] == ["route-a"]

    def test_round_no_selection_no_write(self, tmp_path):
        """selected_label=None（只展開給人看）→ 不落盤。"""
        mock_gate, _ = make_mock_gate([LAYER_1])
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label=None,
            dashboard_path=tmp_path / "dash.md",
            field_log_path=tmp_path / "fl.json",
            snapshot_dir=tmp_path / "snaps",
        )
        assert r["written"] == {}
        assert not (tmp_path / "dash.md").exists()

    def test_round_snapshot_round_num_increments_with_field_log(self, tmp_path):
        """快照 round_num = field_log 筆數（跨輪續接不覆蓋）。"""
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2])
        field_log = FieldLog(field_id="f-test")
        snap_dir = tmp_path / "snaps"
        r1 = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a", field_log=field_log, snapshot_dir=snap_dir,
        )
        assert r1["written"]["snapshot"].name == "converge-round-01.json"
        r2 = probe_converge_round(
            situation=SITUATION, state=r1["next_state"],
            reflect_on=r1["next_reflect_on"],
            gate_fn=mock_gate, selected_label="route-a1", field_log=field_log,
            snapshot_dir=snap_dir,
        )
        assert r2["written"]["snapshot"].name == "converge-round-02.json"
        assert len(list(snap_dir.glob("converge-round-*.json"))) == 2

    def test_round_dashboard_backup_on_rewrite(self, tmp_path):
        """同檔 dashboard 重寫 → .bak 保留舊版。"""
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_1])
        dash = tmp_path / "dashboard.md"
        probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a", dashboard_path=dash,
        )
        first = dash.read_text(encoding="utf-8")
        probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a", dashboard_path=dash,
        )
        assert dash.read_text(encoding="utf-8") == first  # 內容相同（同分支）
        assert (tmp_path / "dashboard.md.bak").exists()
        assert (tmp_path / "dashboard.md.bak").read_text(encoding="utf-8") == first

    def test_round_json_view_with_dashboard_path_raises(self, tmp_path):
        """view_format=json 配 dashboard_path → 明確錯誤（dashboard 需文字）。"""
        mock_gate, _ = make_mock_gate([LAYER_1])
        with pytest.raises(TypeError, match="dashboard"):
            probe_converge_round(
                situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
                selected_label="route-a", view_format="json",
                dashboard_path=tmp_path / "dash.json",
            )

    def test_round_written_files_world_agnostic(self, tmp_path):
        """落盤產物零世界特定詞（世界無關檢定）。"""
        mock_gate, _ = make_mock_gate([LAYER_1])
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a",
            dashboard_path=tmp_path / "dash.md",
            field_log_path=tmp_path / "fl.json",
            snapshot_dir=tmp_path / "snaps",
        )
        for key, p in r["written"].items():
            text = p.read_text(encoding="utf-8")
            for word in WORLD_WORDS:
                assert word not in text, f"{key} 落盤含世界特定詞 {word!r}"

    def test_round_fieldlog_path_without_fieldlog_no_write(self, tmp_path):
        """field_log_path 給定但 field_log=None → 不寫 field log（其餘照寫）。"""
        mock_gate, _ = make_mock_gate([LAYER_1])
        r = probe_converge_round(
            situation=SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
            selected_label="route-a", field_log=None,
            field_log_path=tmp_path / "fl.json",
            dashboard_path=tmp_path / "dash.md",
        )
        assert "field_log" not in r["written"]
        assert "dashboard" in r["written"]
        assert not (tmp_path / "fl.json").exists()
