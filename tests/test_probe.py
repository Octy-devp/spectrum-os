"""Tests for synth/probe.py — 決策樹探針（約束剛性測量器，PLAN-23 §十二 v1.2）。

全部 mock gate_fn / dry-run，零真 API。
"""

from __future__ import annotations

import json
import os

import pytest

from spectrum_os.kernel import verify as _verify
from spectrum_os.synth.gate_prompts import (
    GATE_SYSTEM_PROMPT_UNIFIED,
    TREE_GENERATE_SYSTEM_PROMPT,
    check_routing_leak_or_schema,
)
from spectrum_os.synth.probe import (
    NECESSITY_HINT_KEY,
    TreeProbeError,
    assert_tree_gate_contract,
    probe_select,
    probe_tree,
    standing_wave_per_layer,
)

# ---------------------------------------------------------------------------
# 測試資料
# ---------------------------------------------------------------------------

SITUATION = {
    "digest": "1914-07 巴爾幹動員僵局：鐵路時刻表無法銜接、中央命令延遲、外部使館斡旋未斷。",
    "local_texture": {"region": "巴爾幹", "season": "夏"},
    "state_vector": {"role": "crisis", "period_months": 12},
}

LAYER_1 = {
    "layer": 1,
    "date_ref": "1914-07",
    "branches": [
        {
            "label": "動員令凍結",
            "grounding": "鐵路時刻表無法銜接",
            "axis_A": "inherited",
            "axis_B": "expression",
            "rigidity_prevalence": 0.8,
            "conditions": ["軍隊不得跨河"],
        },
        {
            "label": "地方調撥自主",
            "grounding": "州郡糧倉自持",
            "axis_A": "emergent",
            "axis_B": "expression",
            "rigidity_prevalence": 0.4,
            "conditions": ["中央命令延遲"],
        },
        {
            "label": "國際調停介入",
            "grounding": "外部使館斡旋",
            "axis_A": "inherited",
            "axis_B": "expression",
            "rigidity_prevalence": 0.5,
            "conditions": ["外交渠道未斷"],
        },
    ],
}

LAYER_2 = {
    "layer": 2,
    "date_ref": "1914-08",
    "branches": [
        {
            "label": "軍部強行開戰",
            "grounding": "參謀本部主導",
            "axis_A": "inherited",
            "axis_B": "expression",
            "rigidity_prevalence": 0.9,
            "conditions": ["動員令解凍"],
            "parent": "動員令凍結",
        },
        {
            "label": "動員叫停",
            "grounding": "沙皇猶豫",
            "axis_A": "inherited",
            "axis_B": "expression",
            "rigidity_prevalence": 0.3,
            "conditions": ["調撥自主生效"],
            "parent": "地方調撥自主",
        },
        {
            "label": "談判延長",
            "grounding": "使館緩衝",
            "axis_A": "emergent",
            "axis_B": "expression",
            "rigidity_prevalence": 0.5,
            "conditions": ["調停介入"],
            "parent": "國際調停介入",
        },
    ],
}

LAYER_3 = {
    "layer": 3,
    "date_ref": "1914-09",
    "branches": [
        {
            "label": "全國動員",
            "grounding": "戰時內閣成立",
            "axis_A": "emergent",
            "axis_B": "expression",
            "rigidity_prevalence": 0.95,
            "conditions": ["軍部開戰"],
            "parent": "軍部強行開戰",
        },
        {
            "label": "局部戰爭",
            "grounding": "邊境衝突",
            "axis_A": "inherited",
            "axis_B": "expression",
            "rigidity_prevalence": 0.4,
            "conditions": ["動員叫停"],
            "parent": "動員叫停",
        },
        {
            "label": "會談破裂",
            "grounding": "雙方無讓步",
            "axis_A": "emergent",
            "axis_B": "expression",
            "rigidity_prevalence": 0.6,
            "conditions": ["談判延長"],
            "parent": "談判延長",
        },
    ],
}


def make_mock_gate(layer_responses: list[dict]):
    """回傳 (mock_gate, calls) ——mock 依序回傳每層 JSON，零真 API。"""
    calls: list[str] = []
    queue = list(layer_responses)

    def mock_gate(prompt, api_key, **kwargs):
        calls.append(prompt)
        if not queue:
            raise AssertionError("mock gate 被呼叫次數超過提供層數")
        return json.dumps(queue.pop(0), ensure_ascii=False)

    return mock_gate, calls


def valid_branch(label, **overrides):
    b = {
        "label": label,
        "grounding": "處境內支撐",
        "axis_A": "emergent",
        "axis_B": "expression",
        "rigidity_prevalence": 0.6,
        "conditions": ["邊條件"],
    }
    b.update(overrides)
    return b


@pytest.fixture(autouse=True)
def _reset_verify_state():
    """每次測試前清空 verify 全域 state log 並停用檔案持久化（測試隔離）。"""
    _verify.clear_state_log()
    _verify.init_log(None)
    yield


# ---------------------------------------------------------------------------
# 1. 樹生成（mock gate_fn → 樹結構正確）
# ---------------------------------------------------------------------------

class TestProbeTreeGeneration:
    def test_tree_structure_and_prevalence(self):
        mock_gate, calls = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(
            SITUATION,
            n_branch=3,
            n_sample=1,
            depth=3,
            gate_fn=mock_gate,
        )
        assert len(calls) == 3  # 每層 1 call（成本錨定 depth）
        assert result["meta"]["calls"] == 3
        assert result["meta"]["cost_anchor"] == 3

        tree = result["trees"][0]
        assert len(tree["layers"]) == 3
        assert all(len(le["branches"]) == 3 for le in tree["layers"])
        # 路徑 = 3（每條 layer-3 分支一條根到葉世界線）
        assert len(tree["paths"]) == 3
        assert all(len(p) == 4 for p in tree["paths"])  # root + 3 層
        assert tree["meta"]["n_paths"] == 3

        # 每節點均有解析出的 parent（無 dangling）
        for le in tree["layers"][1:]:
            for b in le["branches"]:
                assert b["parent_label"] in [pb["label"] for pb in tree["layers"][le["layer"] - 2]["branches"]]

        # prevalence：layer 1「動員令凍結」通過 1 條路徑（→ 軍部強行開戰 → 全國動員）
        p = {e["label"]: e for e in tree["prevalence"] if e["layer"] == 1}
        assert p["動員令凍結"]["prevalence"] == pytest.approx(1 / 3, abs=1e-3)
        assert p["地方調撥自主"]["prevalence"] == pytest.approx(1 / 3, abs=1e-3)
        assert p["國際調停介入"]["prevalence"] == pytest.approx(1 / 3, abs=1e-3)

        # 全樹無 necessity_hint（機器不輸出「必然」）
        assert NECESSITY_HINT_KEY not in json.dumps(result, ensure_ascii=False)

    def test_rigidity_map_layers(self):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        rmap = result["rigidity_map"]
        assert len(rmap) == 3
        for e in rmap:
            assert 0.0 <= e["rigidity_prevalence"] <= 1.0  # 連續值，不切 hard/soft
            assert set(e) >= {"layer", "date_ref", "rigidity_prevalence", "confidence_band", "tags"}
            assert e["confidence_band"]["lower"] <= e["confidence_band"]["upper"]
        # 高 LLM 剛性的層（0.8/0.9/0.95 主線）→ 整體剛性偏高（機械離散補數混合後仍 > 0）
        assert all(e["rigidity_prevalence"] > 0.0 for e in rmap)

    def test_default_parent_hangs_main_line(self):
        # 無 parent 欄位 → 機械掛主線（LLM 剛性最高者）
        l1 = {"layer": 1, "date_ref": "1914-07", "branches": [
            valid_branch("分支甲", rigidity_prevalence=0.9),
            valid_branch("分支乙", rigidity_prevalence=0.2),
            valid_branch("分支丙", rigidity_prevalence=0.5),
        ]}
        l2 = {"layer": 2, "date_ref": "1914-08", "branches": [
            valid_branch("續甲", rigidity_prevalence=0.8),  # 無 parent → 主線 = 分支甲
            valid_branch("續乙", rigidity_prevalence=0.8),
            valid_branch("續丙", rigidity_prevalence=0.8),
        ]}
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        tree = result["trees"][0]
        l2_branches = tree["layers"][1]["branches"]
        assert all(b["parent_label"] == "分支甲" for b in l2_branches)

    def test_rejected_node_not_propagated(self):
        # 一個 quarantine 命中（nuclear）→ rejected，不進樹、不成反射對象
        l1 = {"layer": 1, "date_ref": "1914-07", "branches": [
            valid_branch("乾淨分支"),
            valid_branch("第二支"),
            valid_branch("nuclear 佈署"),  # 黑名單 → rejected
        ]}
        l2 = {"layer": 2, "date_ref": "1914-08", "branches": [
            valid_branch("續分支", parent="乾淨分支"),
            valid_branch("續二", parent="第二支"),
            valid_branch("續三", parent="第二支"),
        ]}
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        tree = result["trees"][0]
        assert len(tree["layers"][0]["branches"]) == 2
        assert all(b["label"] != "nuclear 佈署" for b in tree["layers"][0]["branches"])
        assert any(r["label"] == "nuclear 佈署" for r in result["rejected"])

    def test_n_sample_aggregation(self):
        # n_sample=2 → 每層 1 call × 2 = 6 calls；prevalence 聚合、band 採樣本變異
        mock_gate, calls = make_mock_gate(
            [LAYER_1, LAYER_2, LAYER_3, LAYER_1, LAYER_2, LAYER_3]
        )
        result = probe_tree(
            SITUATION, n_branch=3, n_sample=2, depth=3, gate_fn=mock_gate
        )
        assert len(calls) == 6
        assert result["meta"]["calls"] == 6
        assert len(result["trees"]) == 2
        p = {e["label"]: e for e in result["prevalence"] if e["layer"] == 1}
        assert p["動員令凍結"]["prevalence"] == pytest.approx(1 / 3, abs=1e-3)
        assert p["動員令凍結"]["confidence_band"]["basis"] == "cross_tree_var"

    def test_single_path_remasks_unknown(self):
        # 每層 1 存活分支（其餘被 quarantine 拒）→ 路徑數 1 < MIN_PATHS → remasking UNKNOWN
        l1 = {"layer": 1, "date_ref": "1914-07", "branches": [
            valid_branch("單線一"),
            valid_branch("nuclear 佈署"),   # 黑名單 → rejected
            valid_branch("cold_war 部署"),  # 黑名單 → rejected
        ]}
        l2 = {"layer": 2, "date_ref": "1914-08", "branches": [
            valid_branch("單線二", parent="單線一"),
            valid_branch("nuclear 佈署"),
            valid_branch("cold_war 部署"),
        ]}
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        tree = result["trees"][0]
        assert tree["meta"]["n_paths"] == 1
        for e in tree["prevalence"]:
            assert e["confidence_band"]["confidence"] == "UNKNOWN"  # 樣本不足 remasking

    def test_dry_run_no_api_key_needed(self):
        # mock gate_fn 注入 → 不需 DEEPSEEK_API_KEY
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        assert result["meta"]["calls"] == 3


# ---------------------------------------------------------------------------
# 2. assert_tree_gate_contract 負面案例
# ---------------------------------------------------------------------------

class TestTreeGateContract:
    def test_valid_contract_passes(self):
        assert_tree_gate_contract(LAYER_1, max_branches=5, max_depth=3)

    def test_missing_branches_raises(self):
        with pytest.raises(TreeProbeError, match="missing required keys"):
            assert_tree_gate_contract({"layer": 1})

    def test_missing_branch_key_raises(self):
        bad = {"layer": 1, "branches": [valid_branch("x")]}
        del bad["branches"][0]["grounding"]
        with pytest.raises(TreeProbeError, match="missing required keys"):
            assert_tree_gate_contract(bad)

    def test_empty_conditions_raises(self):
        bad = {"layer": 1, "branches": [valid_branch("x", conditions=[])]}
        with pytest.raises(TreeProbeError, match="conditions 必須是非空"):
            assert_tree_gate_contract(bad)

    def test_invalid_axis_a_raises(self):
        bad = {"layer": 1, "branches": [valid_branch("x", axis_A="parody")]}
        with pytest.raises(TreeProbeError, match="axis_A"):
            assert_tree_gate_contract(bad)

    def test_invalid_axis_b_raises(self):
        bad = {"layer": 1, "branches": [valid_branch("x", axis_B="gravity")]}
        with pytest.raises(TreeProbeError, match="axis_B"):
            assert_tree_gate_contract(bad)

    def test_rigidity_out_of_range_raises(self):
        bad = {"layer": 1, "branches": [valid_branch("x", rigidity_prevalence=1.5)]}
        with pytest.raises(TreeProbeError, match="rigidity_prevalence"):
            assert_tree_gate_contract(bad)

    def test_necessity_hint_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("x", **{NECESSITY_HINT_KEY: "必然"})]}
        with pytest.raises(TreeProbeError, match="necessity_hint"):
            assert_tree_gate_contract(bad)
        # 巢狀出現也拒
        bad2 = {"layer": 1, "branches": [{"label": "x", "nested": {NECESSITY_HINT_KEY: 1}}]}
        with pytest.raises(TreeProbeError, match="necessity_hint"):
            assert_tree_gate_contract(bad2)

    def test_placeholder_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("None")]}
        with pytest.raises(TreeProbeError, match="placeholder or shell refusal"):
            assert_tree_gate_contract(bad)

    def test_shell_empty_rejected(self):
        bad = {"layer": 1, "branches": [{"label": "?", "grounding": "", "axis_A": "x"}]}
        with pytest.raises(TreeProbeError):
            assert_tree_gate_contract(bad)

    def test_forbidden_series_key_rejected(self):
        bad = {"layer": 1, "branches": [], "series": [1, 2, 3]}
        with pytest.raises(TreeProbeError, match="forbidden key 'series'"):
            assert_tree_gate_contract(bad)

    def test_situation_echo_rejected_when_labels_given(self):
        bad = {"layer": 1, "branches": [valid_branch("巴爾幹")]}
        with pytest.raises(TreeProbeError, match="situation echo"):
            assert_tree_gate_contract(bad, ["巴爾幹"])

    def test_duplicate_label_in_layer_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("同一"), valid_branch("同一")]}
        with pytest.raises(TreeProbeError, match="duplicate label"):
            assert_tree_gate_contract(bad)

    def test_max_branches_exceeded_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch(f"分支{i}") for i in range(4)]}
        with pytest.raises(TreeProbeError, match="max_branches"):
            assert_tree_gate_contract(bad, max_branches=3)

    def test_max_depth_exceeded_rejected(self):
        bad = {"layer": 4, "branches": [valid_branch("x")]}
        with pytest.raises(TreeProbeError, match="max_depth"):
            assert_tree_gate_contract(bad, max_depth=3)

    def test_quarantine_label_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("cold_war 佈署")]}
        with pytest.raises(TreeProbeError, match="mechanical filter"):
            assert_tree_gate_contract(bad)

    def test_role_illegal_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("x", role="not_a_role")]}
        with pytest.raises(TreeProbeError, match="role"):
            assert_tree_gate_contract(bad)

    def test_strength_out_of_range_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("x", strength=3.0)]}
        with pytest.raises(TreeProbeError, match="strength"):
            assert_tree_gate_contract(bad)

    def test_period_months_illegal_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("x", period_months=-3)]}
        with pytest.raises(TreeProbeError, match="period_months"):
            assert_tree_gate_contract(bad)


# ---------------------------------------------------------------------------
# 3. 連續剛性 + 信賴帶 + remasking（standing_wave_per_layer）
# ---------------------------------------------------------------------------

class TestContinuousRigidity:
    def test_standing_wave_per_layer_prevalence(self):
        layers = [
            {"layer": 1, "date_ref": "1914-07", "branches": [
                valid_branch("甲"), valid_branch("乙"), valid_branch("甲"),
            ]},
        ]
        res = standing_wave_per_layer(layers)
        per = res["per_layer"][0]
        prev = {t["label"]: t["prevalence"] for t in per["antinodes"] + per["nodes"]}
        assert prev["甲"] == pytest.approx(2 / 3, abs=1e-3)
        assert prev["乙"] == pytest.approx(1 / 3, abs=1e-3)
        assert res["meta"]["basis"] == "layer_grouped"  # 非跨步 union
        assert 0.0 <= per["layer_rigidity"] <= 1.0

    def test_standing_wave_path_weights(self):
        layers = [
            {"layer": 1, "date_ref": "1914-07", "branches": [valid_branch("甲"), valid_branch("乙")]},
        ]
        # 甲有 4 條路徑、乙有 1 條（path 加權）
        res = standing_wave_per_layer(layers, path_weights={1: [4, 1]})
        per = res["per_layer"][0]
        prev = {t["label"]: t["prevalence"] for t in per["nodes"] + per["antinodes"]}
        assert prev["甲"] == pytest.approx(0.8, abs=1e-3)
        assert prev["乙"] == pytest.approx(0.2, abs=1e-3)

    def test_zero_prevalence_remasks_unknown(self):
        # 分支權重 0 → prevalence==0 → UNKNOWN（remasking，維持叠加交人）
        layers = [
            {"layer": 1, "date_ref": "1914-07", "branches": [
                valid_branch("無路"), valid_branch("有路"),
            ]},
        ]
        res = standing_wave_per_layer(layers, path_weights={1: [0, 1]})
        per = res["per_layer"][0]
        zero = next(t for t in per["remasked"] if t["label"] == "無路")
        assert zero["prevalence"] == 0.0
        assert zero["confidence_band"]["confidence"] == "UNKNOWN"

    def test_band_continuous_no_hard_cut(self):
        # 連續值 + 信賴帶，無 hard/soft 切、無「必然」標記
        layers = [
            {"layer": 1, "date_ref": "1914-07", "branches": [
                valid_branch("甲"), valid_branch("乙"), valid_branch("丙"),
            ]},
        ]
        res = standing_wave_per_layer(layers)
        per = res["per_layer"][0]
        for t in per["nodes"] + per["antinodes"]:
            cb = t["confidence_band"]
            assert 0.0 <= t["prevalence"] <= 1.0
            assert cb["lower"] <= cb["upper"]
            assert "necessity" not in json.dumps(t, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 4. 門節點（剛性低谷 + sharp 命中；saturation 只作 heuristics）
# ---------------------------------------------------------------------------

class TestGateNodes:
    # 剛性低谷層 = layer 2（LLM 剛性明顯偏低，其他層高）——sharp 命中放 layer 2 的 date_ref
    L1_HIGH = {"layer": 1, "date_ref": "1914-07", "branches": [
        valid_branch("動員令凍結", rigidity_prevalence=0.9),
        valid_branch("地方調撥自主", rigidity_prevalence=0.8),
        valid_branch("國際調停介入", rigidity_prevalence=0.85),
    ]}
    L2_LOW = {"layer": 2, "date_ref": "1914-08", "branches": [
        valid_branch("軍部強行開戰", rigidity_prevalence=0.2, parent="動員令凍結"),
        valid_branch("動員叫停", rigidity_prevalence=0.3, parent="地方調撥自主"),
        valid_branch("談判延長", rigidity_prevalence=0.25, parent="國際調停介入"),
    ]}
    L3_HIGH = {"layer": 3, "date_ref": "1914-09", "branches": [
        valid_branch("全國動員", rigidity_prevalence=0.9, parent="軍部強行開戰"),
        valid_branch("局部戰爭", rigidity_prevalence=0.95, parent="動員叫停"),
        valid_branch("會談破裂", rigidity_prevalence=0.9, parent="談判延長"),
    ]}

    def _run(self, residuals=None, saturation=None):
        constraint = {}
        if residuals is not None:
            constraint["residuals"] = residuals
        if saturation is not None:
            constraint["saturation"] = saturation
        mock_gate, _ = make_mock_gate([self.L1_HIGH, self.L2_LOW, self.L3_HIGH])
        return probe_tree(
            SITUATION, constraint_field=constraint, n_branch=3, depth=3, gate_fn=mock_gate
        )

    def test_trough_plus_sharp_marks_gate(self):
        # sharp 命中在 layer 2（剛性低谷）date_ref → 門節點在 layer 2
        residuals = {"1914-08": {"criteria": ["saturation", "decoupling"]}}
        result = self._run(residuals=residuals)
        gates = result["gate_nodes"]
        assert len(gates) >= 1
        g = gates[0]
        assert g["layer"] == 2
        assert any("① 剛性低谷" in c for c in g["conditions"])
        assert any("② 殘差 sharp 命中" in c for c in g["conditions"])

    def test_trough_without_sharp_not_marked(self):
        result = self._run(residuals={})
        assert result["gate_nodes"] == []

    def test_saturation_only_heuristics(self):
        # saturation ≥ 0.99 標註但無 sharp 命中 → 不構成機械門（人保留覆核）
        result = self._run(residuals={}, saturation={"1914-08": 0.99})
        assert result["gate_nodes"] == []
        # sharp 命中 + saturation 0.99 → 卡上標註 heuristics
        result2 = self._run(
            residuals={"1914-08": {"criteria": ["saturation"]}},
            saturation={"1914-08": 0.99},
        )
        g = result2["gate_nodes"][0]
        assert g["layer"] == 2
        assert g["saturation_heuristic"] is True
        assert any("③ 不可延期" in c for c in g["conditions"])


# ---------------------------------------------------------------------------
# 5. Phase E：state_log 回寫（含 re_calibrate）
# ---------------------------------------------------------------------------

class TestProbeSelectStateLog:
    def test_probe_select_records_entry(self, tmp_path):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)

        log_path = str(tmp_path / "state_log.jsonl")
        probe_select(
            result,
            selected_label="動員令凍結",
            verdict="selected",
            re_calibrate=True,
            state_log_path=log_path,
        )

        assert result["collapse"]["selected"] == "動員令凍結"
        assert "地方調撥自主" in result["collapse"]["unselected"]
        assert "國際調停介入" in result["collapse"]["unselected"]

        entry = result["state_log_entry"]
        assert entry["gate_type"] == "probe_tree"
        assert entry["verdict"] == "selected"
        assert entry["re_calibrate"] is True
        assert entry["selected_branch"] == "動員令凍結"
        assert entry["layer"] == 1
        assert entry["n_branch"] == 3
        assert entry["prediction_id"].startswith("probe_tree-")

        # 持久化：JSONL 檔案有該筆
        with open(log_path, encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        assert any(
            e.get("gate_type") == "probe_tree" and e.get("selected_branch") == "動員令凍結"
            for e in lines
        )

    def test_probe_select_unknown_label_raises(self):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        with pytest.raises(TreeProbeError, match="不在樹中"):
            probe_select(result, selected_label="不存在分支")

    def test_probe_select_invalid_verdict_raises(self):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        with pytest.raises(TreeProbeError, match="verdict"):
            probe_select(result, selected_label="動員令凍結", verdict="maybe")


# ---------------------------------------------------------------------------
# 6. 向後相容 + routing / prompt
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_existing_gates_unaffected(self):
        # 既有 unified prompt 段落仍在
        assert "When called with [task:rate]" in GATE_SYSTEM_PROMPT_UNIFIED
        assert "When called with [task:generate]" in GATE_SYSTEM_PROMPT_UNIFIED
        assert "When called with [task:adversary]" in GATE_SYSTEM_PROMPT_UNIFIED
        # 新增 tree_generate 段落
        assert "When called with [task:tree_generate]" in GATE_SYSTEM_PROMPT_UNIFIED

    def test_routing_leak_schema_tree_generate(self):
        # 合法 tree_generate → None
        valid_raw = json.dumps(LAYER_1)
        assert check_routing_leak_or_schema(valid_raw, LAYER_1, "tree_generate") is None
        # 含 candidates → SCHEMA_MISMATCH
        bad = {"layer": 1, "branches": [], "candidates": []}
        err = check_routing_leak_or_schema(json.dumps(bad), bad, "tree_generate")
        assert err is not None and "SCHEMA_MISMATCH" in err
        # leak 關鍵字 → ROUTING_LEAK_DETECTED
        leak = 'TASK ROUTING ' + json.dumps(LAYER_1)
        err2 = check_routing_leak_or_schema(leak, LAYER_1, "tree_generate")
        assert err2 is not None and "ROUTING_LEAK_DETECTED" in err2

    def test_standalone_tree_prompt(self):
        # 獨立 prompt：操作/約束/兩軸判準/成熟/鐵律（§12.3），並禁止輸出 necessity_hint
        for clause in ("【操作】", "【約束】", "【兩軸判準】", "【成熟】", "【鐵律】"):
            assert clause in TREE_GENERATE_SYSTEM_PROMPT
        assert "necessity_hint" in TREE_GENERATE_SYSTEM_PROMPT  # 指令禁止輸出

    def test_probe_does_not_pollute_state_log(self):
        # 純生成（無 state_log_path / probe_select）不寫 verify state_log
        before = len(_verify.get_state_log())
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        assert len(_verify.get_state_log()) == before

    def test_import_cycle_free_and_api(self):
        # probe 可從 synth 包匯入，不破壞既有模組
        import spectrum_os.synth as synth
        assert callable(synth.probe_tree)
        assert callable(synth.assert_tree_gate_contract)
        assert callable(synth.standing_wave_per_layer)


# ---------------------------------------------------------------------------
# 7. F1-F10 修補回歸（systematic-debug 缺陷清單）
# ---------------------------------------------------------------------------

class TestFixes:
    # --- F1：分支多餘鍵未攔截 ---
    def test_f1_prose_key_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("x", narrative="敘事散文")]}
        with pytest.raises(TreeProbeError, match="prose 鍵|未允許鍵"):
            assert_tree_gate_contract(bad)

    def test_f1_extra_key_rejected(self):
        # 非 prose 的未知鍵也拒（鍵集外即拒）
        bad = {"layer": 1, "branches": [valid_branch("x", foo="bar")]}
        with pytest.raises(TreeProbeError, match="未允許鍵"):
            assert_tree_gate_contract(bad)

    def test_f1_prose_key_via_probe_tree(self):
        # 分支含 prose 鍵 → probe_tree 整層契約拒絕（F1 鍵集外即拒）
        l1 = {"layer": 1, "date_ref": "1914-07", "branches": [
            valid_branch("乾淨"),
            valid_branch("第二支"),
            {"label": "洩漏", "grounding": "支撐", "axis_A": "emergent",
             "axis_B": "expression", "rigidity_prevalence": 0.5,
             "conditions": ["邊條件"], "text": "散文洩漏"},
        ]}
        l2 = {"layer": 2, "date_ref": "1914-08", "branches": [
            valid_branch("續一", parent="乾淨"),
            valid_branch("續二", parent="第二支"),
            valid_branch("續三", parent="第二支"),
        ]}
        mock_gate, _ = make_mock_gate([l1, l2])
        with pytest.raises(TreeProbeError, match="prose 鍵|未允許鍵"):
            probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)

    def test_f1_allowed_optional_fields_pass(self):
        # role/strength/period_months 為 F1 回退可選欄位——允許且驗合法性
        ok = {"layer": 1, "branches": [
            valid_branch("x", role="crisis", strength=0.6, period_months=12)
        ]}
        assert_tree_gate_contract(ok)
        bad = {"layer": 1, "branches": [valid_branch("x", role="not_a_role")]}
        with pytest.raises(TreeProbeError, match="role"):
            assert_tree_gate_contract(bad)

    # --- F2：role/period 為輔助顯示（不混入 blend）文件化判定 ---
    def test_f2_role_period_auxiliary_not_in_blend(self):
        from spectrum_os.synth.probe import _layer_rigidity_components
        base = [valid_branch("同一", rigidity_prevalence=0.8) for _ in range(3)]
        with_role = [
            valid_branch("同一", rigidity_prevalence=0.8, role=r, period_months=p)
            for r, p in [("crisis", 3), ("crisis", 5), ("crisis", 9)]
        ]
        comp_base = _layer_rigidity_components(base)
        comp_role = _layer_rigidity_components(with_role)
        # role/period 離散度不改變 blend 結果（label 語義離散為主代理）
        assert comp_role["rigidity_prevalence"] == comp_base["rigidity_prevalence"]
        # 但以輔助顯示進 components
        assert comp_role["components"]["role_dispersion"] == 0.0
        assert comp_role["components"]["period_dispersion"] > 0.0
        assert comp_base["components"]["role_dispersion"] is None

    # --- F3：n_sample>1 坍縮以 trees[tree_index] 為準 ---
    def test_f3_probe_select_multi_tree_collapse(self):
        l1b = {"layer": 1, "date_ref": "1914-07", "branches": [
            valid_branch("甲線"), valid_branch("乙線"), valid_branch("丙線"),
        ]}
        l2b = {"layer": 2, "date_ref": "1914-08", "branches": [
            valid_branch("續甲", parent="甲線"),
            valid_branch("續乙", parent="乙線"),
            valid_branch("續丙", parent="丙線"),
        ]}
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, l1b, l2b])
        result = probe_tree(SITUATION, n_branch=3, n_sample=2, depth=2, gate_fn=mock_gate)
        assert len(result["trees"]) == 2
        # 第一棵（tree_index=0）
        probe_select(result, selected_label="動員令凍結", tree_index=0)
        assert result["collapse"]["tree_index"] == 0
        assert set(result["collapse"]["unselected"]) == {"地方調撥自主", "國際調停介入"}
        # 第二棵（tree_index=1）——unselected 反映第二棵同層其餘分支
        probe_select(result, selected_label="甲線", tree_index=1)
        assert result["collapse"]["tree_index"] == 1
        assert set(result["collapse"]["unselected"]) == {"乙線", "丙線"}

    def test_f3_tree_index_out_of_range_raises(self):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        with pytest.raises(TreeProbeError, match="tree_index"):
            probe_select(result, selected_label="動員令凍結", tree_index=5)

    # --- F4：saturation 兼容殘差條目內形狀 ---
    def test_f4_saturation_inside_residual_entry(self):
        # saturation 放殘差條目內 {date_ref: {criteria:[...], saturation:0.99}} → ③ 標註
        constraint = {"residuals": {
            "1914-08": {"criteria": ["saturation"], "saturation": 0.99}
        }}
        mock_gate, _ = make_mock_gate(
            [TestGateNodes.L1_HIGH, TestGateNodes.L2_LOW, TestGateNodes.L3_HIGH]
        )
        result = probe_tree(
            SITUATION, constraint_field=constraint, n_branch=3, depth=3, gate_fn=mock_gate
        )
        gates = result["gate_nodes"]
        assert len(gates) == 1
        g = gates[0]
        assert g["layer"] == 2
        assert g["saturation_heuristic"] is True
        assert any("③ 不可延期" in c for c in g["conditions"])

    # --- F5：路徑爆炸截斷 ---
    def test_f5_path_truncation(self):
        from spectrum_os.synth.probe import _enumerate_paths
        root = {"label": "root", "children": [
            {"label": "a", "children": [
                {"label": "a1", "children": []}, {"label": "a2", "children": []},
            ]},
            {"label": "b", "children": [
                {"label": "b1", "children": []}, {"label": "b2", "children": []},
            ]},
        ]}
        paths, truncated = _enumerate_paths(root, max_paths=2)
        assert len(paths) == 2
        assert truncated is True
        paths_all, truncated_all = _enumerate_paths(root)
        assert len(paths_all) == 4
        assert truncated_all is False

    def test_f5_dead_constant_removed(self):
        import spectrum_os.synth.probe as probe
        assert not hasattr(probe, "MAX_BRANCH_JSON_NODES")
        assert hasattr(probe, "MAX_PATHS_PER_TREE")

    def test_f5_probe_tree_records_truncated(self):
        # 正常樹不觸發截斷 → meta truncated=False
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        assert result["meta"]["truncated"] is False
        assert all(t["meta"]["truncated"] is False for t in result["trees"])

    # --- F6：ROUTING_LEAK raise 路徑（probe_tree 內）---
    def test_f6_leak_raise_path(self):
        def leak_gate(prompt, api_key, **kwargs):
            return json.dumps(
                {"layer": 1, "branches": [valid_branch("x")], "note": "TASK ROUTING 洩漏"},
                ensure_ascii=False,
            )
        with pytest.raises(TreeProbeError, match="ROUTING_LEAK_DETECTED"):
            probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=leak_gate)

    def test_f6_schema_mismatch_raise_path(self):
        def bad_gate(prompt, api_key, **kwargs):
            return json.dumps({"layer": 1, "branches": [], "candidates": []})
        with pytest.raises(TreeProbeError, match="SCHEMA_MISMATCH"):
            probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=bad_gate)

    # --- F7：probe_select 副作用順序 + re_calibrate 嚴格布林 ---
    def test_f7_invalid_verdict_no_residue(self):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        with pytest.raises(TreeProbeError, match="verdict"):
            probe_select(result, selected_label="動員令凍結", verdict="maybe")
        assert "collapse" not in result  # 驗證失敗不留殘留
        assert "state_log_entry" not in result

    def test_f7_recalibrate_strict_bool(self):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        probe_select(result, selected_label="動員令凍結", re_calibrate="false")
        assert result["state_log_entry"]["re_calibrate"] is False
        probe_select(result, selected_label="動員令凍結", re_calibrate="true")
        assert result["state_log_entry"]["re_calibrate"] is True

    # --- F8：全域 _log_path 洩漏 ---
    def test_f8_probe_tree_does_not_set_global_log(self, tmp_path):
        log_path = str(tmp_path / "state_log.jsonl")
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(
            SITUATION, n_branch=3, depth=3, gate_fn=mock_gate, state_log_path=log_path
        )
        assert _verify._log_path is None  # 生成不落 log——不設全域
        assert not os.path.exists(log_path)  # 生成不建立檔案
        assert result["meta"]["state_log_path"] == log_path
        # 收束才落盤：無顯式 path 時退 meta 路徑
        probe_select(result, selected_label="動員令凍結")
        assert os.path.exists(log_path)
        with open(log_path, encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        assert any(e.get("selected_branch") == "動員令凍結" for e in lines)

    def test_f8_explicit_path_overrides_meta(self, tmp_path):
        meta_path = str(tmp_path / "meta.jsonl")
        explicit_path = str(tmp_path / "explicit.jsonl")
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(
            SITUATION, n_branch=3, depth=3, gate_fn=mock_gate, state_log_path=meta_path
        )
        probe_select(result, selected_label="動員令凍結", state_log_path=explicit_path)
        assert not os.path.exists(meta_path)
        assert os.path.exists(explicit_path)

    # --- F9：contamination 降權 + rigidity_cross_tree basis ---
    def test_f9_contamination_downweight(self):
        from spectrum_os.synth.probe import _layer_rigidity_components
        def mk(contam=False):
            branches = [
                valid_branch("支一", rigidity_prevalence=0.1),
                valid_branch("支二", rigidity_prevalence=0.1),
                valid_branch("支三", rigidity_prevalence=0.9),
            ]
            if contam:
                branches[2]["contamination"] = True  # 仿 _mechanical_check_layer 標記
            return branches
        comp_expr = _layer_rigidity_components(mk(False))
        comp_cont = _layer_rigidity_components(mk(True))
        assert comp_cont["components"]["llm_mean_rigidity"] < comp_expr["components"]["llm_mean_rigidity"]
        assert comp_cont["rigidity_prevalence"] < comp_expr["rigidity_prevalence"]
        assert comp_expr["components"]["contamination_weight"] == 0.5

    def test_f9_rigidity_cross_tree_basis(self):
        mock_gate, _ = make_mock_gate(
            [LAYER_1, LAYER_2, LAYER_3, LAYER_1, LAYER_2, LAYER_3]
        )
        result = probe_tree(SITUATION, n_branch=3, n_sample=2, depth=3, gate_fn=mock_gate)
        assert len(result["trees"]) == 2
        for e in result["rigidity_map"]:
            assert e["confidence_band"]["basis"] == "rigidity_cross_tree"
