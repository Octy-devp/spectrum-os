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
    MAX_BRANCHES_SAFETY_CAP,
    NECESSITY_HINT_KEY,
    TreeProbeError,
    _build_layer_payload,
    _generate_one_tree_iter,
    _mechanical_check_layer,
    _prevalence_band,
    _rigidity_band,
    _semantic_path_similarity,
    assert_code_compliance,
    assert_tree_gate_contract,
    cluster_archetypes_semantic,
    convergence_view,
    detect_script,
    max_len_for_code,
    max_words_for_code,
    probe_expand_layer,
    probe_select,
    probe_tree,
    probe_tree_manual,
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


def _code_aware_tree(label, **branch_overrides):
    """語碼感知長度測試用：單層單分支樹輸出（可事後加頂層 ``code`` 欄位）。"""
    return {"layer": 1, "date_ref": "1914-07", "branches": [valid_branch(label, **branch_overrides)]}


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
# 1.5  n_branch 提示性語義（2026-08-03：分支數限制兩層解耦）
# ---------------------------------------------------------------------------

class TestNBranchHintSemantics:
    """n_branch 由「硬性上限」改為「提示性 + 機械安全網」後的語義。

    動機（追源）：舊 prompt「≤ N_branch」把機械安全上限誤當 LLM 必須遵守的指令
    ——LLM 恆輸出 5 分支且多數不填 parent → 樹塌成 5 條平行鏈 → 樹寬 1.36 結構常數。
    新語義：prompt 改為「幾根柱就幾條路」（提示性）+「每一條分支必須標明 parent」；
    機械層只在超過 MAX_BRANCHES_SAFETY_CAP（荒謬濾網）時 raise。
    """

    def test_n_branch_hint_range_accepted(self):
        # 提示性範圍 [1,20]——不再是硬性 [3,8]
        for nb in (1, 2, 5, 20):
            mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2])
            result = probe_tree(SITUATION, n_branch=nb, depth=2, gate_fn=mock_gate)
            assert result["meta"]["params"]["n_branch"] == nb

    def test_n_branch_absurd_value_rejected(self):
        # 超出提示性範圍仍拒（0 / 負 / 21+）——範圍本身只是提示參考，仍須是正整數；
        # 校驗在 gate_fn/API 檢查之前就拋錯，不需 mock
        for nb in (0, -1, 21, 100):
            with pytest.raises(ValueError, match="n_branch"):
                probe_tree(SITUATION, n_branch=nb, depth=2)

    def test_branches_exceeding_hint_not_rejected(self):
        # 提示性核心：超過 n_branch 提示值（3→5）不再拒——張力幾何可自然給更多
        l1 = {"layer": 1, "date_ref": "1914-07",
              "branches": [valid_branch(f"路{i}") for i in range(5)]}
        l2 = {"layer": 2, "date_ref": "1914-08",
              "branches": [valid_branch(f"續{i}", parent=f"路{i}") for i in range(5)]}
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        tree = result["trees"][0]
        assert len(tree["layers"][0]["branches"]) == 5
        assert len(tree["layers"][1]["branches"]) == 5
        assert tree["meta"]["n_paths"] == 5

    def test_branches_exceeding_safety_cap_rejected(self):
        # 機械安全網：超過 MAX_BRANCHES_SAFETY_CAP（20）的荒謬輸出才拒
        bad = {"layer": 1, "date_ref": "1914-07",
               "branches": [valid_branch(f"路{i}") for i in range(25)]}
        mock_gate, _ = make_mock_gate([bad])
        with pytest.raises(TreeProbeError, match="max_branches"):
            probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)

    def test_safety_cap_boundary(self):
        # 邊界：恰 20 過、21 拒
        ok = {"layer": 1, "date_ref": "1914-07",
              "branches": [valid_branch(f"路{i}") for i in range(20)]}
        ok2 = {"layer": 2, "date_ref": "1914-08",
               "branches": [valid_branch(f"續{i}", parent=f"路{i}") for i in range(20)]}
        mock_gate, _ = make_mock_gate([ok, ok2])
        result = probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        assert len(result["trees"][0]["layers"][0]["branches"]) == 20

        bad = {"layer": 1, "date_ref": "1914-07",
               "branches": [valid_branch(f"路{i}") for i in range(21)]}
        mock_gate2, _ = make_mock_gate([bad])
        with pytest.raises(TreeProbeError, match="max_branches"):
            probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate2)

    def test_layer_payload_marks_n_branch_as_hint(self):
        # payload 保留 n_branch 數字（提示性參考）+ n_branch_hint 標記
        payload = _build_layer_payload(SITUATION, None, [], layer=1, n_branch=3, w=0.5)
        assert payload["n_branch"] == 3
        assert payload["n_branch_hint"] is True
        # 帶反射對象時仍標記，且 reflection 結構不變
        payload2 = _build_layer_payload(
            SITUATION, None, [{"label": "父"}], layer=2, n_branch=5, w=0.5
        )
        assert payload2["n_branch"] == 5
        assert payload2["n_branch_hint"] is True
        assert payload2["reflection"]["layer"] == 1

    def test_contract_still_enforces_explicit_max_branches(self):
        # 契約函數本身行為不變：呼叫者顯式給 max_branches 時仍拒超過值
        bad = {"layer": 1, "branches": [valid_branch(f"分支{i}") for i in range(4)]}
        with pytest.raises(TreeProbeError, match="max_branches"):
            assert_tree_gate_contract(bad, max_branches=3)
        # 但同一輸出超過「提示值 3」、未超過顯式安全網 20 → 過
        assert_tree_gate_contract(bad, max_branches=20)

    def test_safety_cap_is_constant_decoupled_from_hint(self):
        # 兩層解耦：安全網是固定常數，不隨提示值縮放
        assert MAX_BRANCHES_SAFETY_CAP == 20


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

    def test_situation_echo_recorded_not_rejected_when_labels_given(self):
        # 2026-08-03 起：situation echo 全面撤銷攔截——記錄供人機收束，不 raise
        notes: list[str] = []
        assert_tree_gate_contract(
            {"layer": 1, "branches": [valid_branch("巴爾幹")]},
            ["巴爾幹"], echo_notes=notes,
        )
        assert any("situation echo" in n for n in notes)

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


# ---------------------------------------------------------------------------
# F1-F7 修補（決策樹探針 pilot live 失敗）——2026-08-01
# ---------------------------------------------------------------------------

class TestProbeF1F7Repairs:
    """F1-F7 修補驗收：成本控制 / 月級 sharp / 軸詞彙鏡射 / 樹厚度 / situation echo。

    全部 mock gate / 零真 API。
    """

    # --- F2：月級 date_ref × 日級殘差表 → sharp 掃描命中 ---
    def test_sharp_hit_month_scan_hits_day_keys(self):
        # LLM 輸出月級 date_ref（"1914-08"），殘差表是日級鍵（"1914-08-03"）——
        # _sharp_hit 須掃描該月前綴鍵（F2）。
        residuals = {
            "1914-08-03": {"criteria": ["saturation"]},
            "1914-08-19": {"criteria": ["decoupling"]},
        }
        mock_gate, _ = make_mock_gate(
            [TestGateNodes.L1_HIGH, TestGateNodes.L2_LOW, TestGateNodes.L3_HIGH]
        )
        result = probe_tree(
            SITUATION, constraint_field={"residuals": residuals},
            n_branch=3, depth=3, gate_fn=mock_gate,
        )
        gates = result["gate_nodes"]
        assert len(gates) >= 1
        g = gates[0]
        assert g["layer"] == 2
        assert any("② 殘差 sharp 命中" in c for c in g["conditions"])

    def test_sharp_hit_month_scan_no_key_in_month(self):
        from spectrum_os.synth.probe import _sharp_hit
        table = {"1914-07-22": {"criteria": ["saturation"]}}
        assert _sharp_hit(table, "1914-08") is False   # 該月無鍵 → 不命中
        assert _sharp_hit(table, "1914-07") is True    # 月級 → 掃前綴命中
        assert _sharp_hit(table, "1914-07-22") is True  # 日級精確命中

    # --- F2/F1：月級聚合 saturation map × 日級 date_ref → 回退該月前綴 ---
    def test_saturation_month_fallback_for_day_ref(self):
        l1 = {
            "layer": 1, "date_ref": "1914-07-20",
            "branches": [
                valid_branch("a", rigidity_prevalence=0.9),
                valid_branch("b", rigidity_prevalence=0.85),
            ],
        }
        l2 = {
            "layer": 2, "date_ref": "1914-08-03",
            "branches": [
                valid_branch("x", rigidity_prevalence=0.2, parent="a"),
                valid_branch("y", rigidity_prevalence=0.3, parent="b"),
            ],
        }
        constraint = {
            "residuals": {"1914-08-03": {"criteria": ["saturation"]}},
            # F1：月級聚合 map——日級 date_ref 回退 "1914-08"
            "saturation": {"1914-08": 0.99},
        }
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(
            SITUATION, constraint_field=constraint, n_branch=3, depth=2, gate_fn=mock_gate
        )
        gates = result["gate_nodes"]
        assert len(gates) == 1
        g = gates[0]
        assert g["layer"] == 2
        assert g["saturation_heuristic"] is True
        assert g["saturation"] == 0.99
        assert any("③ 不可延期" in c for c in g["conditions"])

    # --- F3：兩軸名詞字首（契約層降級：fatal → echo_notes 記錄，決策 1/2）---
    def test_axis_echo_prefix_nonfatal_recorded_contract(self):
        for label in ["繼承的動員令", "湧現的合作社", "替代方案"]:
            bad = {"layer": 1, "branches": [valid_branch(label)]}
            notes: list[str] = []
            assert_tree_gate_contract(bad, echo_notes=notes)  # 不再 raise
            assert notes and any("axis echo" in n for n in notes)

    # --- 決策 4：example echo 契約層降為記錄（範例只是形狀）---
    def test_example_echo_nonfatal_recorded_contract(self):
        from spectrum_os.synth.gate_prompts import PROMPT_EXAMPLE_LABELS
        from spectrum_os.synth.probe import AXIS_ECHO_PREFIXES
        # 挑一個不在軸名詞字首集合的範例標籤（避免同時命中 axis echo）
        example_label = next(
            lbl for lbl in PROMPT_EXAMPLE_LABELS
            if not lbl.startswith(AXIS_ECHO_PREFIXES)
        )
        bad = {"layer": 1, "branches": [valid_branch(example_label)]}
        notes: list[str] = []
        assert_tree_gate_contract(bad, echo_notes=notes)  # 不再 raise
        assert notes and any("example echo" in n for n in notes)

    # --- F3：兩軸名詞字首（2026-08-03 全面撤銷攔截：記錄不拒）---
    def test_axis_echo_prefix_nonfatal_pipeline(self):
        l1 = {
            "layer": 1, "date_ref": "1914-07",
            "branches": [
                valid_branch("繼承的動員令"),
                valid_branch("乾淨一"),
                valid_branch("乾淨二"),
            ],
        }
        l2 = {
            "layer": 2, "date_ref": "1914-08",
            "branches": [valid_branch("續一"), valid_branch("續二"), valid_branch("續三")],
        }
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        # axis echo 不再進 rejected——全部放行，記入 echo_notes 供人機收束
        rejected_labels = [r["label"] for r in result["rejected"]]
        assert "繼承的動員令" not in rejected_labels
        echo_text = "\n".join(result.get("echo_notes", []))
        assert "axis echo" in echo_text
        # 分支存活 → 樹正常建立
        assert len(result["trees"]) == 1

    # --- F6：每父平均子數記錄 + 低發散警示判據 ---
    def test_avg_children_per_parent_recorded(self):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        avg = result["trees"][0]["meta"]["avg_children_per_parent"]
        # 純鏈 mock：root 3 子 + 3×L1 各 1 子 + 3×L2 各 1 子 → (3+3+3)/7 ≈ 1.2857
        assert abs(avg - 9 / 7) < 1e-4

    def test_branching_parent_higher_avg_children(self):
        l1 = {
            "layer": 1, "date_ref": "1914-07",
            "branches": [valid_branch("a", rigidity_prevalence=0.8)],
        }
        l2 = {
            "layer": 2, "date_ref": "1914-08",
            "branches": [
                valid_branch("a1", parent="a"),
                valid_branch("a2", parent="a"),
            ],
        }
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        avg = result["trees"][0]["meta"]["avg_children_per_parent"]
        # root 1 子 + a 2 子 → (1 + 2) / 2 = 1.5 > 純鏈 1.2857
        assert abs(avg - 1.5) < 1e-4

    # --- F7：probe_tree 包裝 situation → 處境-echo 2026-08-03 起記錄不拒 ---
    def test_situation_echo_recorded_not_rejected_via_probe_tree(self):
        echo_layer = {
            "layer": 1, "date_ref": "1914-07",
            "branches": [
                valid_branch("巴爾幹"),  # SITUATION.local_texture.region → 處境標籤
                valid_branch("乾淨一"),
                valid_branch("乾淨二"),
            ],
        }
        l2 = {
            "layer": 2, "date_ref": "1914-08",
            "branches": [valid_branch("續一"), valid_branch("續二"), valid_branch("續三")],
        }
        mock_gate, _ = make_mock_gate([echo_layer, l2])
        result = probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        rejected_labels = [r["label"] for r in result["rejected"]]
        assert "巴爾幹" not in rejected_labels
        echo_text = "\n".join(result.get("echo_notes", []))
        assert "situation echo" in echo_text

    def test_extract_situation_labels_accepts_bare_situation(self):
        # F7：_extract_situation_labels 兼容 bare situation dict（防禦性）
        from spectrum_os.synth.alt_gate import _extract_situation_labels
        labels = _extract_situation_labels(SITUATION)
        assert labels is not None
        assert "巴爾幹" in labels  # local_texture 字串值
        assert SITUATION["digest"] in labels  # digest 全文
        # 包裝形狀仍正常
        wrapped = _extract_situation_labels({"situation": SITUATION})
        assert wrapped == labels


# ---------------------------------------------------------------------------
# 8. W1：DCA 基底接回——角色向量 / 文法轉移 / 嵌套遞歸 / 速率矩陣 / 6D 向量
# ---------------------------------------------------------------------------

class TestDcaRoleVector:
    """W1 驗收（PLAN-23 §6.1 角色向量 ⟨thread: role⟩ + §五 文法 + 具體限制）。

    - ``roles`` = CLAD 角色**集合/向量**（可多，叠加）——第一公民（可選但建議）。
    - ``role_instances`` = 嵌套實例（c1→c2→…），深度跟隨樹層（≤ layer）。
    - 文法轉移 = ``validate_alternative``（含結構性零 direction→alternative /
      lag→direction）——契約級 fatal。
    - rate_matrix 零強度邊 = 具體限制的可選輸入——``_mechanical_check_layer``
      非致命拒（rate_zero → rejected 清單）；無 rate_matrix 行為不變（世界無關）。
    - 6d_vector 存在時 rigid 併入三進制方向輔助分量（小權重）；不存在時不變。

    全部 mock gate / 零真 API。
    """

    def _parent(self, roles, label="父", rigidity=0.9):
        return [{"label": label, "roles": roles, "rigidity_prevalence": rigidity}]

    # --- roles 值合法/非法（契約） ---
    def test_roles_valid_passes(self):
        ok = {"layer": 1, "branches": [valid_branch("x", roles=["crisis"])]}
        assert_tree_gate_contract(ok)

    def test_roles_multi_superposition_passes(self):
        # 同一 situation 叠加多角色（thread A crisis、thread B direction）
        ok = {"layer": 1, "branches": [valid_branch("x", roles=["direction", "crisis"])]}
        assert_tree_gate_contract(ok)
        assert ok["branches"][0]["roles"] == ["crisis", "direction"]  # 標準化（ROLES 次序）

    def test_roles_single_string_accepted(self):
        # LLM 慣用輸出單一 str → 包裝為 [str]（內容仍嚴格驗）
        ok = {"layer": 1, "branches": [valid_branch("x", roles="crisis")]}
        assert_tree_gate_contract(ok)
        assert ok["branches"][0]["roles"] == ["crisis"]

    def test_roles_illegal_value_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("x", roles=["not_a_role"])]}
        with pytest.raises(TreeProbeError, match="roles"):
            assert_tree_gate_contract(bad)

    def test_roles_empty_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("x", roles=[])]}
        with pytest.raises(TreeProbeError, match="roles"):
            assert_tree_gate_contract(bad)

    def test_roles_not_collection_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("x", roles=3)]}
        with pytest.raises(TreeProbeError, match="roles"):
            assert_tree_gate_contract(bad)

    # --- role_instances 結構（契約） ---
    def test_role_instances_valid(self):
        ok = {"layer": 1, "branches": [
            valid_branch(
                "x",
                roles=["crisis", "lag"],
                role_instances={"crisis": "c1", "lag": ["l1", "l2"]},
            ),
        ]}
        assert_tree_gate_contract(ok)

    def test_role_instances_illegal_role_key_rejected(self):
        bad = {"layer": 1, "branches": [
            valid_branch("x", role_instances={"crisis": "c1", "foo": "f1"}),
        ]}
        with pytest.raises(TreeProbeError, match="role_instances"):
            assert_tree_gate_contract(bad)

    def test_role_instances_bad_value_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("x", role_instances={"crisis": 3})]}
        with pytest.raises(TreeProbeError, match="role_instances"):
            assert_tree_gate_contract(bad)

    def test_role_instances_not_dict_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("x", role_instances=["c1"])]}
        with pytest.raises(TreeProbeError, match="role_instances"):
            assert_tree_gate_contract(bad)

    def test_role_instances_empty_instance_rejected(self):
        bad = {"layer": 1, "branches": [valid_branch("x", role_instances={"crisis": ""})]}
        with pytest.raises(TreeProbeError, match="role_instances"):
            assert_tree_gate_contract(bad)

    # --- 嵌套遞歸（c1→c2）深度跟隨樹層 ---
    def test_nested_instance_depth_follows_layer(self):
        nested = {"crisis": {"c1": "c2"}}  # 深度 2
        ok = {"layer": 2, "branches": [valid_branch("x", role_instances=nested)]}
        assert_tree_gate_contract(ok)  # 層 2 ≥ 深度 2
        bad = {"layer": 1, "branches": [valid_branch("x", role_instances=nested)]}
        with pytest.raises(TreeProbeError, match="嵌套深度"):
            assert_tree_gate_contract(bad)  # 層 1 < 深度 2 → 拒

    def test_nested_instance_over_max_depth_rejected(self):
        deep = {"crisis": {"c1": {"c2": {"c3": {"c4": {"c5": "c6"}}}}}}  # 深度 6
        bad = {"layer": 4, "branches": [valid_branch("x", role_instances=deep)]}
        with pytest.raises(TreeProbeError, match="嵌套深度"):
            assert_tree_gate_contract(bad, max_depth=4)

    # --- 文法轉移（validate_alternative，父 → 子角色） ---
    def test_grammar_transition_valid_passes(self):
        # crisis → lag / alternative / crisis 均合法
        for cr in ("lag", "alternative", "crisis"):
            out = {"layer": 2, "branches": [valid_branch("子", parent="父", roles=[cr])]}
            assert_tree_gate_contract(out, parent_branches=self._parent(["crisis"]))

    def test_grammar_structural_zero_direction_to_alternative(self):
        # 方向承諾不可撤銷：direction → alternative 結構性零
        # 🔴 2026-08-02：降為非致命（CLAD=閱讀文法非世界序列）——記入 echo_notes
        out = {"layer": 2, "branches": [
            valid_branch("子", parent="父", roles=["alternative"]),
        ]}
        notes: list[str] = []
        assert_tree_gate_contract(
            out, parent_branches=self._parent(["direction"]), echo_notes=notes
        )
        assert any("文法轉移異常" in n for n in notes)

    def test_grammar_structural_zero_lag_to_direction(self):
        # Lag 需經 Alternative 中介：lag → direction 結構性零
        # 🔴 2026-08-02：降為非致命——記入 echo_notes
        out = {"layer": 2, "branches": [
            valid_branch("子", parent="父", roles=["direction"]),
        ]}
        notes: list[str] = []
        assert_tree_gate_contract(
            out, parent_branches=self._parent(["lag"]), echo_notes=notes
        )
        assert any("文法轉移異常" in n for n in notes)

    def test_grammar_multi_role_superposition_existential(self):
        # 父叠加 {direction, crisis}——子 alternative 可經 crisis→alternative（存在性）→ 合法
        out = {"layer": 2, "branches": [
            valid_branch("子", parent="父", roles=["alternative"]),
        ]}
        assert_tree_gate_contract(out, parent_branches=self._parent(["direction", "crisis"]))

    def test_grammar_default_parent_main_line(self):
        # 無 parent 欄位 → 掛主線（_llm_rigidity 最高者）——文法異常記 notes 非 raise
        parents = [
            {"label": "低剛性", "roles": ["crisis"], "_llm_rigidity": 0.2},
            {"label": "主線", "roles": ["direction"], "_llm_rigidity": 0.9},
        ]
        out = {"layer": 2, "branches": [valid_branch("子", roles=["alternative"])]}
        notes: list[str] = []
        assert_tree_gate_contract(out, parent_branches=parents, echo_notes=notes)
        # 主線是 direction → 子 alternative 是結構性零 → 記入 notes（非 raise）
        assert any("文法轉移異常" in n for n in notes)
        parents2 = [
            {"label": "主線", "roles": ["crisis"], "_llm_rigidity": 0.9},
            {"label": "次線", "roles": ["direction"], "_llm_rigidity": 0.2},
        ]
        notes2: list[str] = []
        assert_tree_gate_contract(out, parent_branches=parents2, echo_notes=notes2)
        # 主線是 crisis → 子 alternative 合法 → 無異常
        assert not any("文法轉移異常" in n for n in notes2)

    def test_grammar_no_parent_branches_skips(self):
        # 無 parent_branches（世界無關 fallback）→ 不檢查轉移，仍過
        out = {"layer": 2, "branches": [valid_branch("子", roles=["alternative"])]}
        assert_tree_gate_contract(out)

    def test_grammar_violation_via_probe_tree_nonfatal(self):
        # 🔴 2026-08-02：文法結構性零不再 fatal——經 probe_tree 全管線應**通過**
        #（違反降為記錄，交人機收束），樹仍可展開
        l1 = {"layer": 1, "date_ref": "1914-07", "branches": [
            valid_branch("父", roles=["direction"]),
            valid_branch("支二", roles=["crisis"]),
            valid_branch("支三", roles=["lag"]),
        ]}
        l2 = {"layer": 2, "date_ref": "1914-08", "branches": [
            valid_branch("子", parent="父", roles=["alternative"]),  # direction→alternative 結構性零（記録）
            valid_branch("續二", parent="支二", roles=["lag"]),
            valid_branch("續三", parent="支三", roles=["crisis"]),
        ]}
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        # 樹正常展開（文法異常被記錄而非阻斷）
        assert "trees" in result

    # --- rate_matrix 零強度（具體限制的可選輸入） ---
    # ROLES 次序：(crisis, lag, alternative, direction)
    RATE_ZERO_ALT = [
        [0.5, 0.5, 0.0, 0.0],   # crisis → alternative = 0
        [0.3, 0.3, 0.4, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.4, 0.3, 0.0, 0.3],
    ]

    def test_rate_zero_rejected_when_matrix_present(self):
        l1 = {"layer": 1, "date_ref": "1914-07", "branches": [
            valid_branch("父", roles=["crisis"]),
            valid_branch("支二", roles=["lag"]),
        ]}
        l2 = {"layer": 2, "date_ref": "1914-08", "branches": [
            valid_branch("子", parent="父", roles=["alternative"]),  # crisis→alternative=0 → rate_zero
            valid_branch("續二", parent="父", roles=["lag"]),        # crisis→lag=0.5 → 存活
        ]}
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(
            SITUATION, constraint_field={"rate_matrix": self.RATE_ZERO_ALT},
            n_branch=3, depth=2, gate_fn=mock_gate,
        )
        tree = result["trees"][0]
        # 「子」被非致命拒（進 rejected 清單，不靜默丟棄），「續二」存活
        assert any(r["label"] == "子" and r.get("rate_zero") for r in result["rejected"])
        assert all(b["label"] != "子" for b in tree["layers"][1]["branches"])
        assert any(b["label"] == "續二" for b in tree["layers"][1]["branches"])

    def test_rate_zero_absent_unchanged(self):
        # 無 rate_matrix → 零強度不檢查（世界無關 fallback：行為不變）
        l1 = {"layer": 1, "date_ref": "1914-07", "branches": [
            valid_branch("父", roles=["crisis"]),
            valid_branch("支二", roles=["lag"]),
        ]}
        l2 = {"layer": 2, "date_ref": "1914-08", "branches": [
            valid_branch("子", parent="父", roles=["alternative"]),
            valid_branch("續二", parent="父", roles=["lag"]),
        ]}
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        tree = result["trees"][0]
        assert any(b["label"] == "子" for b in tree["layers"][1]["branches"])
        assert not any(r.get("rate_zero") for r in result["rejected"])

    def test_rate_zero_dict_wrapped_matrix(self):
        # estimate_rate_matrix 輸出形狀 {"matrix": 4x4} 也接受
        l1 = {"layer": 1, "date_ref": "1914-07", "branches": [
            valid_branch("父", roles=["crisis"]),
            valid_branch("支二", roles=["lag"]),
        ]}
        l2 = {"layer": 2, "date_ref": "1914-08", "branches": [
            valid_branch("子", parent="父", roles=["alternative"]),
            valid_branch("續二", parent="父", roles=["lag"]),
        ]}
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(
            SITUATION,
            constraint_field={"rate_matrix": {"matrix": self.RATE_ZERO_ALT}},
            n_branch=3, depth=2, gate_fn=mock_gate,
        )
        assert any(r["label"] == "子" and r.get("rate_zero") for r in result["rejected"])

    # --- 6d_vector：三進制方向輔助分量 ---
    def test_6d_vector_adds_ternary_component(self):
        from spectrum_os.synth.probe import _layer_rigidity_components
        vec = {"d1": 1, "d2": 0, "d3": 0, "d4": 12, "d5": 0.0, "d6": 0.5}
        branches = [
            valid_branch("a", roles=["direction"]),
            valid_branch("b", roles=["direction"]),
            valid_branch("c", roles=["direction"]),
        ]
        comp_with = _layer_rigidity_components(branches, situation_vector=vec)
        comp_without = _layer_rigidity_components(branches)
        # 全 direction → 投影 (1,-1,-1) vs 參考 (1,0,0) → alignment = 1/3（非 None）
        assert comp_with["components"]["ternary_alignment"] == pytest.approx(1 / 3, abs=1e-3)
        assert comp_with["components"]["situation_vector"] == vec
        assert comp_with["components"]["vector_blend_w"] == 0.1
        # 無 6d_vector → 行為不變（無 ternary 分量，blend 與既有完全一致）
        assert comp_without["components"]["ternary_alignment"] is None
        assert comp_without["components"]["vector_blend_w"] is None
        # labels a/b/c 全離散 → base = 0.5*0.6 + 0.5*0.0 = 0.3；with = 0.9*0.3 + 0.1*(1/3)
        assert comp_with["rigidity_prevalence"] == pytest.approx(0.9 * 0.3 + 0.1 / 3, abs=1e-3)
        assert comp_with["rigidity_prevalence"] != comp_without["rigidity_prevalence"]

    def test_6d_vector_through_probe_tree(self):
        sit = dict(SITUATION)
        sit["6d_vector"] = {"d1": 1, "d2": 0, "d3": 0}
        l1 = {"layer": 1, "date_ref": "1914-07", "branches": [
            valid_branch("a", roles=["crisis"]),
            valid_branch("b", roles=["crisis"]),
            valid_branch("c", roles=["crisis"]),
        ]}
        l2 = {"layer": 2, "date_ref": "1914-08", "branches": [
            valid_branch("a1", parent="a", roles=["lag"]),
            valid_branch("b1", parent="b", roles=["alternative"]),
            valid_branch("c1", parent="c", roles=["crisis"]),
        ]}
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(sit, n_branch=3, depth=2, gate_fn=mock_gate)
        comps = result["rigidity_map"][0]["components"]
        assert comps["ternary_alignment"] is not None
        assert comps["situation_vector"] == sit["6d_vector"]
        # roles 隨節點進入樹輸出（第一公民——不是丟失的暫存欄位）
        tree = result["trees"][0]
        assert all(b.get("roles") for le in tree["layers"] for b in le["branches"])

    # --- 向後相容 + prompt ---
    def test_backward_compat_no_roles_still_pass(self):
        # 舊 schema（無 roles）仍過——roles 可選（設計決策，見 assert_tree_gate_contract docstring）
        assert_tree_gate_contract(LAYER_1, max_branches=5, max_depth=3)
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        tree = result["trees"][0]
        assert all("roles" not in b for le in tree["layers"] for b in le["branches"])
        assert all(not b.get("rate_zero") for le in tree["layers"] for b in le["branches"])

    def test_prompt_mentions_roles_and_nesting(self):
        assert "roles" in TREE_GENERATE_SYSTEM_PROMPT
        assert "role_instances" in TREE_GENERATE_SYSTEM_PROMPT
        assert "direction→alternative" in TREE_GENERATE_SYSTEM_PROMPT
        assert "When called with [task:tree_generate]" in GATE_SYSTEM_PROMPT_UNIFIED
        assert "role_instances" in GATE_SYSTEM_PROMPT_UNIFIED
        # 既有子句保留（向後相容）
        for clause in ("【操作】", "【約束】", "【兩軸判準】", "【成熟】", "【鐵律】"):
            assert clause in TREE_GENERATE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# T7（F3）：standing_wave_per_layer 逐層變體——多層/邊界測試
# ---------------------------------------------------------------------------

class TestStandingWavePerLayerEdges:
    def test_multi_layer_grouped_independent(self):
        # 每層獨立分組——不是既有 standing_wave 的跨步 union
        layers = [
            {"layer": 1, "date_ref": "1914-07", "branches": [
                valid_branch("甲"), valid_branch("甲"), valid_branch("乙"),
            ]},
            {"layer": 2, "date_ref": "1914-08", "branches": [
                valid_branch("丙"), valid_branch("丙"), valid_branch("丙"),
            ]},
        ]
        res = standing_wave_per_layer(layers)
        assert len(res["per_layer"]) == 2
        l1, l2 = res["per_layer"]
        assert l1["layer"] == 1 and l2["layer"] == 2
        p1 = {t["label"]: t["prevalence"] for t in l1["nodes"] + l1["antinodes"]}
        assert p1["甲"] == pytest.approx(2 / 3, abs=1e-3)
        assert p1["乙"] == pytest.approx(1 / 3, abs=1e-3)
        # layer 2 全同 → 全 nodes（prevalence 1.0）、無 antinodes、剛性 1.0
        assert [t["label"] for t in l2["nodes"]] == ["丙"]
        assert l2["antinodes"] == []
        assert l2["layer_rigidity"] == pytest.approx(1.0, abs=1e-3)

    def test_empty_layers(self):
        res = standing_wave_per_layer([])
        assert res["per_layer"] == []
        assert res["meta"]["total_layers"] == 0

    def test_single_branch_is_node(self):
        layers = [{"layer": 1, "date_ref": "1914-07", "branches": [valid_branch("唯一")]}]
        res = standing_wave_per_layer(layers)
        per = res["per_layer"][0]
        assert [t["label"] for t in per["nodes"]] == ["唯一"]
        assert per["antinodes"] == []
        assert per["layer_rigidity"] == 1.0

    def test_no_weights_equals_equal_weights(self):
        layers = [{"layer": 1, "branches": [valid_branch("甲"), valid_branch("乙")]}]
        no_w = standing_wave_per_layer(layers)
        eq_w = standing_wave_per_layer(layers, path_weights={1: [1, 1]})
        p_no = {
            t["label"]: t["prevalence"]
            for t in no_w["per_layer"][0]["nodes"] + no_w["per_layer"][0]["antinodes"]
        }
        p_eq = {
            t["label"]: t["prevalence"]
            for t in eq_w["per_layer"][0]["nodes"] + eq_w["per_layer"][0]["antinodes"]
        }
        assert p_no == p_eq

    def test_zero_weight_branch_remasked_not_dropped(self):
        # prevalence==0 的分支不消失——維持叠加交人（remasking，§12.5 步驟 4）
        layers = [
            {"layer": 1, "date_ref": "1914-07", "branches": [
                valid_branch("無路"), valid_branch("有路"),
            ]},
        ]
        res = standing_wave_per_layer(layers, path_weights={1: [0, 1]})
        per = res["per_layer"][0]
        labels = [t["label"] for t in per["nodes"] + per["antinodes"] + per["remasked"]]
        assert "無路" in labels  # 被 remask 而非丟棄


# ---------------------------------------------------------------------------
# T8（F10）：世界原型——語義距離聚類（零 API，樹自身特徵）
# ---------------------------------------------------------------------------

class TestSemanticArchetypes:
    def test_near_duplicate_labels_ranked_above_unrelated(self):
        # 字元 n-gram 捕捉形近標籤——Jaccard 集合層面零重疊、n-gram 層面高相似
        p_near_a = [valid_branch("動員令凍結", roles=["crisis"])]
        p_near_b = [valid_branch("動員令解凍", roles=["crisis"])]
        p_far = [valid_branch("國際調停介入", roles=["alternative"])]
        sim_near = _semantic_path_similarity(p_near_a, p_near_b)
        sim_far = _semantic_path_similarity(p_near_a, p_far)
        assert sim_near > sim_far
        assert sim_near > 0.5  # 形近 pair 相似度顯著

    def test_role_profile_contributes(self):
        # 同標籤、不同角色 → 相似度低於同標籤同角色（DCA 文法親和納入）
        a1 = [valid_branch("開戰", roles=["crisis"])]
        a2 = [valid_branch("開戰", roles=["crisis"])]
        b = [valid_branch("開戰", roles=["alternative"])]
        assert _semantic_path_similarity(a1, a2) > _semantic_path_similarity(a1, b)

    def test_groups_near_duplicate_paths_into_same_cluster(self):
        # 三組形近 pair——語義聚類把它們各自歸組（Jaccard 集合層面無法分辨）
        paths = [
            [valid_branch("動員令凍結", roles=["crisis"])],
            [valid_branch("動員令解凍", roles=["crisis"])],
            [valid_branch("國際調停介入", roles=["alternative"])],
            [valid_branch("國際斡旋介入", roles=["alternative"])],
            [valid_branch("地方調撥自主", roles=["direction"])],
            [valid_branch("地方自主調撥", roles=["direction"])],
        ]
        archetypes, meta = cluster_archetypes_semantic(paths, n_min=2, n_max=3)
        assert meta["k"] == 3
        members = [set(a["member_path_ids"]) for a in archetypes]
        assert {0, 1} in members
        assert {2, 3} in members
        assert {4, 5} in members

    def test_k_selection_within_target_range(self):
        paths = [[valid_branch(f"路徑{i}")] for i in range(6)]
        _archetypes, meta = cluster_archetypes_semantic(paths, n_min=5, n_max=8)
        assert 5 <= meta["k"] <= 8
        assert meta["fell_back"] is False

    def test_insufficient_paths_falls_back_honestly(self):
        # 路徑數 < n_min → 誠實回退，不硬湊 5–8
        paths = [[valid_branch("a")], [valid_branch("b")], [valid_branch("c")]]
        archetypes, meta = cluster_archetypes_semantic(paths, n_min=5, n_max=8)
        assert meta["k"] == 3
        assert meta["fell_back"] is True
        assert len(archetypes) == 3

    def test_empty_paths(self):
        archetypes, meta = cluster_archetypes_semantic([])
        assert archetypes == []
        assert meta["k"] == 0

    def test_deterministic(self):
        paths = [
            [valid_branch("動員令凍結", roles=["crisis"])],
            [valid_branch("動員令解凍", roles=["crisis"])],
            [valid_branch("國際調停介入", roles=["alternative"])],
        ]
        _a1, m1 = cluster_archetypes_semantic(paths, n_min=2, n_max=3)
        _a2, m2 = cluster_archetypes_semantic(paths, n_min=2, n_max=3)
        assert m1["k"] == m2["k"]
        assert [a["member_path_ids"] for a in _a1] == [a["member_path_ids"] for a in _a2]

    def test_probe_tree_wires_semantic_archetypes(self):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        assert "archetype_meta" in result
        assert all(a["method"] == "semantic_hybrid" for a in result["archetypes"])
        assert result["archetype_meta"]["k"] == len(result["archetypes"])
        assert result["archetype_meta"]["n_paths"] == len(result["trees"][0]["paths"])


# ---------------------------------------------------------------------------
# T9（F6 程式碼部分）：收束視圖 UX + state_log 迴路
# ---------------------------------------------------------------------------

class TestConvergenceView:
    def test_view_sections_and_rigidity_distribution(self):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        view = convergence_view(result)
        assert view["sections"] == [
            "rigidity_distribution", "archetype_cards", "gate_node_cards", "selection",
        ]
        assert len(view["rigidity_distribution"]) == 3
        for e in view["rigidity_distribution"]:
            assert "confidence_band" in e
            assert 0.0 <= e["rigidity_prevalence"] <= 1.0
            # 不切 hard/soft、無「必然」標記
            assert "necessity" not in json.dumps(e, ensure_ascii=False)
        assert view["selection"]["selected"] is None

    def test_archetype_cards_include_representative_path(self):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        view = convergence_view(result)
        assert view["archetype_cards"]
        for card in view["archetype_cards"]:
            assert isinstance(card["representative_path"], list)
            assert card["representative_path"]  # 非空——root 至少一節點
            assert card["labels"]

    def test_gate_node_cards_include_branch_labels(self):
        residuals = {"1914-08": {"criteria": ["saturation", "decoupling"]}}
        mock_gate, _ = make_mock_gate(
            [TestGateNodes.L1_HIGH, TestGateNodes.L2_LOW, TestGateNodes.L3_HIGH]
        )
        result = probe_tree(
            SITUATION, constraint_field={"residuals": residuals},
            n_branch=3, depth=3, gate_fn=mock_gate,
        )
        view = convergence_view(result)
        assert view["gate_node_cards"]
        g = view["gate_node_cards"][0]
        assert g["layer"] == 2
        assert "軍部強行開戰" in g["branch_labels"]
        assert "動員叫停" in g["branch_labels"]

    def test_selection_reflects_collapse(self):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        probe_select(
            result, selected_label="動員令凍結", verdict="selected", re_calibrate=True
        )
        view = convergence_view(result)
        assert view["selection"]["selected"] == "動員令凍結"
        assert view["selection"]["layer"] == 1
        assert "地方調撥自主" in view["selection"]["unselected"]
        assert view["selection"]["verdict"] == "selected"
        assert view["selection"]["re_calibrate"] is True

    def test_state_log_query_compat(self, tmp_path, monkeypatch):
        # probe_select 落盤到 data/state_log.jsonl → query_state_log/summarize 可讀
        monkeypatch.chdir(tmp_path)
        _verify.init_log("data/state_log.jsonl")
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        probe_select(
            result, selected_label="動員令凍結", verdict="selected", re_calibrate=True
        )
        entries = _verify.query_state_log()
        assert any(
            e.get("gate_type") == "probe_tree" and e.get("selected_branch") == "動員令凍結"
            for e in entries
        )
        summary = _verify.summarize_state_log()
        assert summary["total_entries"] >= 1
        assert summary["last_entry"]["gate_type"] == "probe_tree"

    def test_verdict_branches_recorded(self):
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        for v in ("selected", "unselected", "human_override"):
            probe_select(result, selected_label="動員令凍結", verdict=v)
            entry = result["state_log_entry"]
            assert entry["verdict"] == v
            assert entry["gate_type"] == "probe_tree"

    def test_append_atomicity(self, tmp_path):
        # 多次收束 → 每次恰好一筆 JSONL 行，檔案全程可解析
        log_path = str(tmp_path / "state_log.jsonl")
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        for _ in range(3):
            probe_select(
                result, selected_label="動員令凍結", verdict="selected",
                state_log_path=log_path,
            )
        with open(log_path, encoding="utf-8") as fh:
            lines = [l for l in fh if l.strip()]
        assert len(lines) == 3
        for line in lines:
            rec = json.loads(line)  # 每行都是完整 JSON
            assert rec["gate_type"] == "probe_tree"

    def test_failed_select_does_not_append(self, tmp_path):
        # F7：verdict 驗證在副作用之前——失敗不留殘留、不 append。
        # 驗證發生在 init_log 之前 → 失敗時檔案根本不建立（零殘留，比空檔更嚴格）。
        log_path = str(tmp_path / "state_log.jsonl")
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        with pytest.raises(TreeProbeError):
            probe_select(
                result, selected_label="動員令凍結", verdict="invalid",
                state_log_path=log_path,
            )
        assert not os.path.exists(log_path)  # 檔案不存在 = 零殘留
        assert _verify._state_log == []  # 記憶體緩衝同樣零殘留


# ---------------------------------------------------------------------------
# T11：remasking / 信賴帶校準（Wald→Wilson + 邊界案例）
# ---------------------------------------------------------------------------

class TestBandCalibration:
    def test_n1_remasks_unknown_full_band(self):
        band = _prevalence_band(1, 1)
        assert band["confidence"] == "UNKNOWN"   # 樣本不足 → remask
        assert band["basis"] == "wilson_path"
        assert band["lower"] == 0.0 and band["upper"] == 1.0  # 全寬——不偽裝信心

    def test_n2_measured_but_wide(self):
        band = _prevalence_band(1, 2)
        assert band["confidence"] == "measured"  # N=2 ≥ MIN_PATHS_FOR_CONFIDENCE
        assert band["upper"] - band["lower"] > 0.5  # 寬帶——不偽裝高信心

    def test_all_zero_remasks_unknown(self):
        band = _prevalence_band(0, 3)
        assert band["confidence"] == "UNKNOWN"   # prevalence==0 → remask（§12.5 步驟 4）

    def test_boundary_one_not_collapsed(self):
        # Wald 在 p=1 時零寬度 [1,1]——Wilson 給誠實寬帶（T11 校準）
        band = _prevalence_band(2, 2)
        assert band["confidence"] == "measured"
        assert band["lower"] < 0.5               # 下界不坍縮到 1.0
        assert band["upper"] == 1.0

    def test_no_paths(self):
        band = _prevalence_band(0, 0)
        assert band["confidence"] == "UNKNOWN"
        assert band["basis"] == "no_paths"
        assert band["lower"] == 0.0 and band["upper"] == 1.0

    def test_cross_tree_var_basis(self):
        band = _prevalence_band(3, 6, cross_tree=[0.5, 0.5, 0.5])
        assert band["basis"] == "cross_tree_var"
        assert band["confidence"] == "measured"

    def test_cross_tree_boundary_uses_wilson_fallback(self):
        # 跨樹全一致（std=0，邊界 1.0）→ 退 Wilson，不零寬度
        band = _prevalence_band(2, 2, cross_tree=[1.0, 1.0])
        assert band["basis"] == "cross_tree_var"
        assert band["lower"] < 0.5

    def test_rigidity_band_n1_unknown_full_band(self):
        band = _rigidity_band([0.5], 1)
        assert band["confidence"] == "UNKNOWN"
        assert band["basis"] == "rigidity_wilson_path"
        assert band["lower"] == 0.0 and band["upper"] == 1.0

    def test_rigidity_band_boundary_not_collapsed(self):
        band = _rigidity_band([1.0], 3)
        assert band["confidence"] == "measured"
        assert band["lower"] < 0.7  # Wilson——非 [1,1]
        assert band["upper"] == 1.0

    def test_rigidity_band_cross_tree(self):
        band = _rigidity_band([0.4, 0.6], 6)
        assert band["basis"] == "rigidity_cross_tree"
        assert band["confidence"] == "measured"

    def test_rigidity_band_empty(self):
        band = _rigidity_band([], 0)
        assert band["confidence"] == "UNKNOWN"
        assert band["basis"] == "no_data"


# ---------------------------------------------------------------------------
# 語碼分層（機械層）：system_message 注入 / 語碼感知長度 / detect_script / 合規審計
# ---------------------------------------------------------------------------

class TestSystemMessageInjection:
    """任務 1：system_message 注入——預設行為不變、注入替代穿透呼叫鏈。"""

    def test_default_system_message_is_tree_prompt(self):
        seen: dict = {}
        _calls = [0]

        def mock_gate(prompt, api_key, **kwargs):
            seen["system_message"] = kwargs.get("system_message")
            _calls[0] += 1
            # 依層號回傳不同 LAYER——層間 echo 攔截（語義中介）拒重複 label
            return json.dumps(LAYER_1 if _calls[0] == 1 else LAYER_2, ensure_ascii=False)

        probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        assert seen["system_message"] == TREE_GENERATE_SYSTEM_PROMPT

    def test_custom_system_message_reaches_gate_fn(self):
        custom = "CUSTOM-語碼分層-system-prompt"
        seen: dict = {}
        _calls = [0]

        def mock_gate(prompt, api_key, **kwargs):
            seen["system_message"] = kwargs.get("system_message")
            _calls[0] += 1
            return json.dumps(LAYER_1 if _calls[0] == 1 else LAYER_2, ensure_ascii=False)

        probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
                   system_message=custom)
        assert seen["system_message"] == custom

    def test_custom_system_message_applies_all_layers(self):
        custom = "CUSTOM-2"
        seen: list = []
        _calls = [0]

        def mock_gate(prompt, api_key, **kwargs):
            seen.append(kwargs.get("system_message"))
            _calls[0] += 1
            return json.dumps(LAYER_1 if _calls[0] == 1 else LAYER_2, ensure_ascii=False)

        probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate,
                   system_message=custom)
        assert seen == [custom, custom]


class TestCodeLengthHelpers:
    """任務 2 輔助：max_len_for_code / max_words_for_code 語碼→上限映射。"""

    def test_max_len_for_code_mapping(self):
        assert max_len_for_code("zh") == 20
        assert max_len_for_code("ja") == 20
        assert max_len_for_code("ko") == 20
        assert max_len_for_code("en") == 60
        assert max_len_for_code("fr") == 60
        assert max_len_for_code("de") == 60
        assert max_len_for_code("sr") == 60
        assert max_len_for_code("ru") == 60
        assert max_len_for_code(None) == 20       # 無語碼 → 舊行為
        assert max_len_for_code("xx") == 20      # 未知語碼 → 舊行為

    def test_max_words_for_code_mapping(self):
        assert max_words_for_code("zh") is None
        assert max_words_for_code("de") == 8
        assert max_words_for_code("ru") == 8
        assert max_words_for_code(None) is None
        assert max_words_for_code("xx") is None

    def test_field_specific_limits(self):
        assert max_len_for_code("de", "grounding") == 240
        assert max_len_for_code("de", "condition") == 120
        assert max_len_for_code(None, "grounding") == 80
        assert max_len_for_code(None, "condition") == 40


class TestCodeAwareLength:
    """任務 2：語碼感知長度契約——德語複合詞（拉丁語碼）通過、CJK 維持 20 字符、無 code 舊行為。"""

    # 52 字符、5 詞——> 20 字符會被誤殺；code="de"（60 字符 / 8 詞）通過
    GERMAN_COMPOUND = "Bewegliche Verteidigung mit getrennten Schwerpunkten"

    def test_german_compound_passes_with_latin_code(self):
        assert_tree_gate_contract(_code_aware_tree(self.GERMAN_COMPOUND), code="de")

    def test_german_compound_rejected_without_code(self):
        # 向後相容：無語碼 → 舊 20 字符上限，52 字符被拒
        with pytest.raises(TreeProbeError, match="label"):
            assert_tree_gate_contract(_code_aware_tree(self.GERMAN_COMPOUND))

    def test_cjk_label_keeps_20_char_limit_with_code(self):
        # CJK 語碼（zh）維持 20 字符上限
        ok = _code_aware_tree("動員令凍結與國際調停介入的複合情境")  # 17 字符
        assert_tree_gate_contract(ok, code="zh")
        long_label = "動員令凍結與國際調停介入的複合情境下的多層次張力"  # 24 字符
        with pytest.raises(TreeProbeError, match="label"):
            assert_tree_gate_contract(_code_aware_tree(long_label), code="zh")

    def test_no_code_preserves_old_behavior(self):
        # 無語碼：17 字符通過、24 字符被拒（舊行為不變）
        assert_tree_gate_contract(_code_aware_tree("動員令凍結與國際調停介入的複合情境"))
        with pytest.raises(TreeProbeError, match="label"):
            assert_tree_gate_contract(
                _code_aware_tree("動員令凍結與國際調停介入的複合情境下的多層次張力")
            )

    def test_word_limit_rejects_overlong_latin_label(self):
        # 拉丁語碼：<60 字符但 >8 詞 → 詞數上限拒絕
        label = "a b c d e f g h i j k l m n"  # 14 詞、27 字符
        with pytest.raises(TreeProbeError, match="詞數"):
            assert_tree_gate_contract(_code_aware_tree(label), code="de")

    def test_code_from_output_top_level_field(self):
        # output 頂層 ``code`` 欄位亦走語碼感知路徑
        out = _code_aware_tree(self.GERMAN_COMPOUND)
        out["code"] = "de"
        assert_tree_gate_contract(out)

    def test_grounding_code_aware(self):
        # grounding：無語碼 80 字符上限；code="de" 240 字符 / 40 詞
        long_grounding = "處境內支撐 " * 30  # 90 字符 > 80、< 240
        with pytest.raises(TreeProbeError, match="grounding"):
            assert_tree_gate_contract(_code_aware_tree("動員令凍結", grounding=long_grounding))
        assert_tree_gate_contract(
            _code_aware_tree("Mobilization freeze", grounding=long_grounding), code="de"
        )

    def test_condition_length_only_when_code_given(self):
        # conditions：無語碼不檢查（舊行為）；code 給定時走語碼感知（40 詞 > 24 → 拒）
        long_cond = " ".join(["condition"] * 40)
        out = _code_aware_tree("動員令凍結", conditions=[long_cond])
        assert_tree_gate_contract(out)  # 無語碼 → 不檢查（向後相容）
        with pytest.raises(TreeProbeError, match="conditions"):
            assert_tree_gate_contract(out, code="de")

    def test_probe_tree_threads_code_end_to_end(self):
        _calls = [0]

        def mock_gate(prompt, api_key, **kwargs):
            _calls[0] += 1
            # 依層號回傳不同 label + 正確 layer——層間 echo 攔截（語義中介）拒重複 label
            if _calls[0] == 1:
                return json.dumps(_code_aware_tree(self.GERMAN_COMPOUND), ensure_ascii=False)
            return json.dumps(
                {"layer": 2, "date_ref": "1914-08",
                 "branches": [valid_branch("Bewegliche Verteidigung")]},
                ensure_ascii=False,
            )

        result = probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate, code="de")
        assert result["meta"]["params"]["code"] == "de"

    def test_probe_tree_without_code_keeps_old_contract(self):
        def mock_gate(prompt, api_key, **kwargs):
            return json.dumps(_code_aware_tree(self.GERMAN_COMPOUND), ensure_ascii=False)

        with pytest.raises(TreeProbeError, match="label"):
            probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)


class TestDetectScript:
    """任務 3a：detect_script——中文/英文/俄文/混合/未知各回傳正確。"""

    def test_chinese_is_cjk(self):
        assert detect_script("動員令凍結") == "cjk"

    def test_english_is_latin(self):
        assert detect_script("Mobilization freeze") == "latin"

    def test_french_accented_is_latin(self):
        assert detect_script("café déjà vu") == "latin"

    def test_german_compound_is_latin(self):
        assert detect_script(TestCodeAwareLength.GERMAN_COMPOUND) == "latin"

    def test_russian_is_cyrillic(self):
        assert detect_script("Мобилизация заморожена") == "cyrillic"

    def test_mixed_scripts(self):
        assert detect_script("動員令 freeze") == "mixed"

    def test_punctuation_only_unknown(self):
        assert detect_script("!!! ???") == "unknown"
        assert detect_script("") == "unknown"
        assert detect_script("   ") == "unknown"

    def test_non_string_unknown(self):
        assert detect_script(None) == "unknown"


class TestCodeCompliance:
    """任務 3b：assert_code_compliance——語碼合規審計輔助（不接入管線）。"""

    def test_german_label_complies_with_de(self):
        assert assert_code_compliance("Bewegliche Verteidigung", ["de"]) is True

    def test_english_label_complies_with_de(self):
        # de 允許 latin 書寫——英語 label 亦是 latin → 合規（書寫系統層級，非語言層級）
        assert assert_code_compliance("Mobilization freeze", ["de"]) is True

    def test_cjk_label_violates_de(self):
        assert assert_code_compliance("動員令凍結", ["de"]) is False

    def test_cyrillic_label_violates_de(self):
        assert assert_code_compliance("Мобилизация", ["de"]) is False

    def test_cyrillic_label_complies_with_sr_and_ru(self):
        assert assert_code_compliance("Мобилизация", ["sr"]) is True
        assert assert_code_compliance("Мобилизация", ["ru"]) is True

    def test_unknown_label_does_not_block(self):
        assert assert_code_compliance("!!!", ["de"]) is True

    def test_unknown_code_does_not_block(self):
        assert assert_code_compliance("動員令凍結", ["xx"]) is True

    def test_mixed_label_partial_compliance(self):
        # mixed：任一成分書寫系統命中允許集合即 True
        assert assert_code_compliance("動員令 freeze", ["de"]) is True
        assert assert_code_compliance("動員令 freeze", ["zh"]) is True


class TestMechanicalLayerCodeCompliance:
    """任務 2：語碼合規接入 _mechanical_check_layer——不合語碼 → rejected（非致命）；mixed 寬鬆（決策 3）。"""

    def _layer(self, labels):
        return {"layer": 1, "branches": [valid_branch(l) for l in labels]}

    def test_cjk_label_rejected_when_code_de(self):
        passed, rejected = _mechanical_check_layer(
            self._layer(["動員令凍結", "Bewegliche Verteidigung"]),
            None, code="de",
        )
        assert [r["label"] for r in rejected] == ["動員令凍結"]
        assert any("書寫系統不合語碼" in " ".join(r["reject_reasons"]) for r in rejected)
        assert [b["label"] for b in passed] == ["Bewegliche Verteidigung"]

    def test_cyrillic_label_rejected_when_code_de(self):
        passed, rejected = _mechanical_check_layer(
            self._layer(["Мобилизация"]), None, code="de",
        )
        assert [r["label"] for r in rejected] == ["Мобилизация"]
        assert len(passed) == 0

    def test_latin_label_passes_when_code_de(self):
        passed, rejected = _mechanical_check_layer(
            self._layer(["Bewegliche Verteidigung", "Mobilization freeze"]),
            None, code="de",
        )
        assert len(rejected) == 0
        assert len(passed) == 2

    def test_sr_latin_cyrillic_dual_script_passes(self):
        # 決策 3：sr 拉丁/西里爾雙書寫——各自合規
        passed, rejected = _mechanical_check_layer(
            self._layer(["Мобилизация заморожена", "Mobilizacija"]),
            None, code="sr",
        )
        assert len(rejected) == 0
        assert len(passed) == 2

    def test_sr_mixed_script_passes(self):
        # 決策 3：mixed 寬鬆——任一成分命中 sr 允許集合（latin/cyrillic）即 True
        passed, rejected = _mechanical_check_layer(
            self._layer(["Мобилизация freeze"]), None, code="sr",
        )
        assert len(rejected) == 0
        assert len(passed) == 1

    def test_no_code_does_not_check(self):
        # 向後相容：code=None → 語碼合規不檢查
        passed, rejected = _mechanical_check_layer(
            self._layer(["動員令凍結"]), None,
        )
        assert len(rejected) == 0
        assert len(passed) == 1

    def test_probe_tree_code_compliance_rejects_cjk_label(self):
        """端到端：code="de" 時 CJK label 分支被機械層非致命拒 → rejected 清單；拉丁分支存活。"""
        l1 = {"layer": 1, "date_ref": "1914-07", "branches": [
            valid_branch("動員令凍結"),  # CJK → 不合 de 語碼 → rejected
            valid_branch("Mobilization freeze"),
            valid_branch("Bewegliche Verteidigung"),
        ]}
        l2 = {"layer": 2, "date_ref": "1914-08", "branches": [
            valid_branch("Continuation one", parent="Mobilization freeze"),
            valid_branch("Continuation two", parent="Bewegliche Verteidigung"),
            valid_branch("Continuation three", parent="Bewegliche Verteidigung"),
        ]}
        mock_gate, _ = make_mock_gate([l1, l2])
        result = probe_tree(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate, code="de")
        rejected_labels = [r["label"] for r in result["rejected"]]
        assert "動員令凍結" in rejected_labels
        assert any(
            "書寫系統不合語碼" in " ".join(r.get("reject_reasons", []))
            for r in result["rejected"]
        )
        # 拉丁分支存活 → 樹仍建立
        assert len(result["trees"]) == 1
        assert all(
            b["label"] != "動員令凍結"
            for b in result["trees"][0]["layers"][0]["branches"]
        )


# ---------------------------------------------------------------------------
# 9. 層間 echo 攔截（T16 語義中介）——判定精化：只拒「label 相同且承義全同」
# ---------------------------------------------------------------------------

class TestLayerEchoSemanticInterception:
    """層間 echo 攔截（T16 語義中介，2026-08-02）判定精化。

    - (a) 真 echo：label 相同 **且** 承義欄位（binding/perspective/grounding/
      conditions）全同/缺失 ＝ 馬可夫原地踏步（轉移矩陣退回恆等）→ 拒。
    - (b) 假 echo：label 相同但承義欄位開出新路 ＝ 合法結構延續 → 放行，
      記入 echo_notes（PLAN-23 §12.4 echo 降權非致命）。
    實證：純 label 精確匹配下 78-81% 層間拒收是假 echo 誤殺（L3 塌單鏈根因）。
    """

    @staticmethod
    def _branch(label, binding=None, perspective=None, conditions=None,
                grounding="處境內支撐"):
        b = {
            "label": label,
            "grounding": grounding,
            "axis_A": "emergent",
            "axis_B": "expression",
            "rigidity_prevalence": 0.6,
            "conditions": conditions or ["邊條件"],
        }
        if binding is not None:
            b["binding"] = binding
        if perspective is not None:
            b["perspective"] = perspective
        return b

    def _parent(self):
        return [self._branch("俄國總動員令", binding="簽署的舊束縛", perspective="總參謀部")]

    def test_true_echo_same_label_and_denotation_recorded_not_rejected(self):
        # (a) label + 承義全同 → 2026-08-03 起不拒（echo 全面撤銷攔截）——記錄供人機收束
        notes: list[str] = []
        passed, rejected = _mechanical_check_layer(
            {"layer": 3, "branches": [
                self._branch("俄國總動員令", binding="簽署的舊束縛", perspective="總參謀部")]},
            None, parent_branches=self._parent(), echo_notes=notes,
        )
        assert len(passed) == 1
        assert len(rejected) == 0
        assert any("候選原地踏步" in n for n in notes)

    def test_true_echo_both_lack_denotation_recorded_not_rejected(self):
        # (a') 兩者皆無 binding/perspective → 2026-08-03 起不拒——記錄供人機收束
        notes: list[str] = []
        passed, rejected = _mechanical_check_layer(
            {"layer": 3, "branches": [self._branch("俄國總動員令")]},
            None, parent_branches=[self._branch("俄國總動員令")], echo_notes=notes,
        )
        assert len(passed) == 1
        assert len(rejected) == 0
        assert any("候選原地踏步" in n for n in notes)

    def test_false_echo_new_binding_released(self):
        # (b) label 同但 binding 全新 → 放行（誤殺修復核心案例）
        passed, rejected = _mechanical_check_layer(
            {"layer": 3, "branches": [
                self._branch("俄國總動員令", binding="以鐵路運力為約束的階段性動員")]},
            None, parent_branches=self._parent(),
        )
        assert len(rejected) == 0
        assert len(passed) == 1

    def test_false_echo_new_perspective_released(self):
        passed, rejected = _mechanical_check_layer(
            {"layer": 3, "branches": [
                self._branch("俄國總動員令", binding="簽署的舊束縛", perspective="倫敦外交部")]},
            None, parent_branches=self._parent(),
        )
        assert len(rejected) == 0
        assert len(passed) == 1

    def test_false_echo_new_conditions_released(self):
        passed, rejected = _mechanical_check_layer(
            {"layer": 3, "branches": [
                self._branch("俄國總動員令", binding="簽署的舊束縛",
                             perspective="總參謀部", conditions=["波蘭走廊駐軍密度"])]},
            None, parent_branches=self._parent(),
        )
        assert len(rejected) == 0
        assert len(passed) == 1

    def test_false_echo_recorded_to_echo_notes(self):
        notes: list[str] = []
        _mechanical_check_layer(
            {"layer": 3, "branches": [self._branch("俄國總動員令", binding="新束縛")]},
            None, parent_branches=self._parent(), echo_notes=notes,
        )
        assert any("層間 label 延續" in n for n in notes)

    def test_echo_notes_none_quiet(self):
        # echo_notes=None → 假 echo 放行但靜默（向後相容）
        passed, rejected = _mechanical_check_layer(
            {"layer": 3, "branches": [self._branch("俄國總動員令", binding="新束縛")]},
            None, parent_branches=self._parent(),
        )
        assert len(rejected) == 0
        assert len(passed) == 1

    def test_new_label_always_passes(self):
        passed, rejected = _mechanical_check_layer(
            {"layer": 3, "branches": [self._branch("德國海軍北海集結", binding="新束縛")]},
            None, parent_branches=self._parent(),
        )
        assert len(rejected) == 0
        assert len(passed) == 1

    def test_probe_tree_l3_no_collapse_with_semantic_continuation(self):
        """端到端：L2 分支在 L3 被「label 延續但承義新」的合法發展 → 全部放行，L3 不塌單鏈。

        舊判據（純 label 精確匹配）下，本測試的 L3 會因 label 重複 L2 而全被拒
        → L3 塌成單鏈（甚至整層 TreeProbeError）。精化後全部放行。
        """
        l1 = {"layer": 1, "date_ref": "1914-07", "branches": [valid_branch("動員令簽署")]}
        l2 = {"layer": 2, "date_ref": "1914-08", "branches": [
            valid_branch("俄國總動員令", binding="簽署的舊束縛", perspective="總參謀部"),
            valid_branch("德國最後通牒", binding="通牒期限", perspective="柏林外交部"),
        ]}
        l3 = {"layer": 3, "date_ref": "1914-09", "branches": [
            valid_branch("俄國總動員令", binding="以鐵路運力為約束的階段性動員",
                         perspective="沙俄總參謀部"),
            valid_branch("德國最後通牒", binding="英國調停介入後的軟化",
                         perspective="倫敦外交部"),
            valid_branch("奧匈對塞宣戰", binding="貝希托爾德主導", perspective="維也納戰爭部"),
        ]}
        mock_gate, _ = make_mock_gate([l1, l2, l3])
        result = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        tree = result["trees"][0]
        l3_branches = [b for le in tree["layers"] if le["layer"] == 3 for b in le["branches"]]
        assert len(l3_branches) == 3  # 全部存活——不塌成單鏈
        # 假 echo（label 延續）被記錄而非丟棄
        assert any("層間 label 延續" in n for n in result["echo_notes"])


# ---------------------------------------------------------------------------
# 13. T18 手動逐層觸發（probe_tree_manual / probe_expand_layer / 生成器）
# ---------------------------------------------------------------------------

class TestProbeTreeManual:
    def test_manual_basic_two_layers(self):
        # 兩層、on_layer 每次選第一條 → 樹能建、層數正確
        mock_gate, calls = make_mock_gate([LAYER_1, LAYER_2])
        seen = []

        def on_layer(layer_entry, ctx):
            seen.append(layer_entry["layer"])
            return layer_entry["branches"][0]

        result = probe_tree_manual(
            SITUATION, n_branch=3, depth=2, gate_fn=mock_gate, on_layer=on_layer
        )
        assert seen == [1, 2]
        assert len(calls) == 2  # 每層 1 call（成本錨定）
        assert result["meta"]["calls"] == 2
        assert result["meta"]["manual"] is True
        assert "stopped_at_layer" not in result["meta"]
        tree = result["trees"][0]
        assert len(tree["layers"]) == 2
        assert all(len(le["branches"]) == 3 for le in tree["layers"])
        # 單父反射下，未選的 L1 分支成為葉 → 路徑 = 3（L2 子）+ 2（L1 葉）= 5
        assert len(tree["paths"]) == 5

    def test_manual_reflect_on_single_parent(self):
        # 人選方向後，下一層 payload 的 reflection 只有 1 個父（打破 1:1 續鏈）
        mock_gate, calls = make_mock_gate([LAYER_1, LAYER_2])

        def on_layer(layer_entry, ctx):
            return layer_entry["branches"][0]  # 「動員令凍結」

        result = probe_tree_manual(
            SITUATION, n_branch=3, depth=2, gate_fn=mock_gate, on_layer=on_layer
        )
        payload2 = json.loads(calls[1].split("\n[task:tree_generate]")[0])
        refs = payload2["reflection"]["passed_branches"]
        assert len(refs) == 1  # 不是全部 passed——只有人選的那 1 條
        assert refs[0]["label"] == "動員令凍結"
        assert payload2["reflection"]["layer"] == 1
        # 一個父展開多子：3 個 L2 分支全掛在「動員令凍結」下 → avg_children > 1
        tree = result["trees"][0]
        l1 = {b["label"]: b for b in tree["layers"][0]["branches"]}
        assert len(l1["動員令凍結"]["children"]) == 3
        assert tree["meta"]["avg_children_per_parent"] == pytest.approx(3.0, abs=1e-3)

    def test_manual_select_different_branch_injected(self):
        # 選不同分支 → 下一層 payload 的反射對象不同（驗證注入機制）
        mock_gate, calls = make_mock_gate([LAYER_1, LAYER_2])

        def on_layer(layer_entry, ctx):
            return layer_entry["branches"][1]  # 「地方調撥自主」

        probe_tree_manual(
            SITUATION, n_branch=3, depth=2, gate_fn=mock_gate, on_layer=on_layer
        )
        payload2 = json.loads(calls[1].split("\n[task:tree_generate]")[0])
        assert [b["label"] for b in payload2["reflection"]["passed_branches"]] == [
            "地方調撥自主"
        ]

    def test_manual_stop_early(self):
        # on_layer 回傳 None → 樹停在該層
        mock_gate, calls = make_mock_gate([LAYER_1, LAYER_2])

        def on_layer(layer_entry, ctx):
            return None

        result = probe_tree_manual(
            SITUATION, n_branch=3, depth=2, gate_fn=mock_gate, on_layer=on_layer
        )
        assert len(calls) == 1  # 只呼叫了第一層
        assert result["meta"]["stopped_at_layer"] == 1
        assert result["meta"]["calls"] == 1
        tree = result["trees"][0]
        assert len(tree["layers"]) == 1
        assert tree["meta"]["calls"] == 1

    def test_manual_invalid_chosen_raises(self):
        # on_layer 回傳不在本層的分支 → TreeProbeError
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2])

        def on_layer(layer_entry, ctx):
            return {"label": "不存在的分支"}

        with pytest.raises(TreeProbeError):
            probe_tree_manual(
                SITUATION, n_branch=3, depth=2, gate_fn=mock_gate, on_layer=on_layer
            )

    def test_manual_auto_mode_equals_probe_tree(self):
        # 向後相容：on_layer=None（自動）與 probe_tree 單樹結果等價
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        auto = probe_tree(SITUATION, n_branch=3, depth=3, gate_fn=mock_gate)
        mock_gate2, _ = make_mock_gate([LAYER_1, LAYER_2, LAYER_3])
        manual = probe_tree_manual(
            SITUATION, n_branch=3, depth=3, gate_fn=mock_gate2
        )
        assert auto["trees"][0] == manual["trees"][0]
        assert auto["prevalence"] == manual["prevalence"]
        assert auto["rigidity_map"] == manual["rigidity_map"]
        assert "manual" not in manual["meta"]

    def test_manual_validation(self):
        # 校驗與 probe_tree 相同
        mock_gate, _ = make_mock_gate([LAYER_1])
        with pytest.raises(ValueError):
            probe_tree_manual(SITUATION, n_branch=0, depth=2, gate_fn=mock_gate)
        with pytest.raises(ValueError):
            probe_tree_manual(SITUATION, n_branch=3, depth=5, gate_fn=mock_gate)
        with pytest.raises(ValueError):
            probe_tree_manual(SITUATION, n_branch=3, depth=2, w=3.0, gate_fn=mock_gate)


class TestGenerateOneTreeIter:
    def test_generator_direct_yield_and_send(self):
        # 直接驅動生成器：yield 層序正確；send([人選]) 注入下一層 reflect_on
        mock_gate, calls = make_mock_gate([LAYER_1, LAYER_2])
        gen = _generate_one_tree_iter(
            SITUATION, None, n_branch=3, depth=2, w=0.5, gate_fn=mock_gate,
            api_key="", model="m", max_tokens=100, temperature=0.6,
            situation_labels=None,
        )
        k1, le1, state1, passed1 = next(gen)
        assert k1 == 1
        assert [b["label"] for b in le1["branches"]] == [
            "動員令凍結", "地方調撥自主", "國際調停介入",
        ]
        assert state1["calls"] == 1
        # 注入 [人選那條] = 「動員令凍結」
        chosen = next(b for b in passed1 if b["label"] == "動員令凍結")
        k2, le2, state2, passed2 = gen.send([chosen])
        assert k2 == 2
        assert state2["calls"] == 2
        assert state2["layers"] == [le1, le2]
        payload2 = json.loads(calls[1].split("\n[task:tree_generate]")[0])
        assert [b["label"] for b in payload2["reflection"]["passed_branches"]] == [
            "動員令凍結"
        ]
        # 耗盡
        with pytest.raises(StopIteration):
            gen.send(None)

    def test_generator_send_none_auto(self):
        # send(None) → 自動續接全部 passed（向後相容模擬）
        mock_gate, calls = make_mock_gate([LAYER_1, LAYER_2])
        gen = _generate_one_tree_iter(
            SITUATION, None, n_branch=3, depth=2, w=0.5, gate_fn=mock_gate,
            api_key="", model="m", max_tokens=100, temperature=0.6,
            situation_labels=None,
        )
        next(gen)
        gen.send(None)  # 自動模式：下一層 reflect_on = passed（3 父）
        payload2 = json.loads(calls[1].split("\n[task:tree_generate]")[0])
        assert len(payload2["reflection"]["passed_branches"]) == 3


class TestProbeExpandLayer:
    def test_expand_layer_stepwise(self):
        # 首次 → layer 1；帶 state + reflect_on=[人選那條] → layer 2；每步恰 1 call
        mock_gate, calls = make_mock_gate([LAYER_1, LAYER_2])
        r1 = probe_expand_layer(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        assert r1["layer_entry"]["layer"] == 1
        assert len(r1["state"]["layers"]) == 1
        assert r1["calls"] == 1
        assert len(calls) == 1
        # 無反射對象 → 第一層 payload 無 reflection
        payload1 = json.loads(calls[0].split("\n[task:tree_generate]")[0])
        assert "reflection" not in payload1

        chosen = r1["layer_entry"]["branches"][0]  # 「動員令凍結」
        r2 = probe_expand_layer(
            SITUATION, state=r1["state"], reflect_on=[chosen], gate_fn=mock_gate
        )
        assert r2["layer_entry"]["layer"] == 2
        assert len(r2["state"]["layers"]) == 2
        assert r2["calls"] == 1
        assert len(calls) == 2
        tree = r2["tree"]
        assert len(tree["layers"]) == 2
        # 下一層 payload 的 reflection 只有 1 個父
        payload2 = json.loads(calls[1].split("\n[task:tree_generate]")[0])
        assert [b["label"] for b in payload2["reflection"]["passed_branches"]] == [
            "動員令凍結"
        ]

    def test_expand_layer_result_consumable_by_probe_select(self):
        # expand 的 result 可被 probe_select 消費（T19 收束接縫）
        mock_gate, _ = make_mock_gate([LAYER_1])
        r1 = probe_expand_layer(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        sel = probe_select(r1["result"], selected_label="動員令凍結", verdict="selected")
        assert sel["collapse"]["selected"] == "動員令凍結"
        assert set(sel["collapse"]["unselected"]) == {"地方調撥自主", "國際調停介入"}

    def test_expand_layer_beyond_depth_raises(self):
        # 已達 max depth 後再展開 → TreeProbeError
        mock_gate, _ = make_mock_gate([LAYER_1, LAYER_2])
        r1 = probe_expand_layer(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        r2 = probe_expand_layer(
            SITUATION, state=r1["state"],
            reflect_on=[r1["layer_entry"]["branches"][0]], gate_fn=mock_gate,
        )
        with pytest.raises(TreeProbeError):
            probe_expand_layer(
                SITUATION, state=r2["state"],
                reflect_on=[r2["layer_entry"]["branches"][0]], gate_fn=mock_gate,
            )

    def test_expand_layer_reflect_on_foreign_branch_raises(self):
        # reflect_on 含非上一層分支 → TreeProbeError（T18 防禦，gate 呼叫前攔截）
        mock_gate, calls = make_mock_gate([LAYER_1])
        r1 = probe_expand_layer(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        foreign = valid_branch("不存在於上一層的分支")
        with pytest.raises(TreeProbeError, match="reflect_on 分支不屬於上一層"):
            probe_expand_layer(
                SITUATION, state=r1["state"],
                reflect_on=[foreign], gate_fn=mock_gate,
            )
        assert len(calls) == 1  # 防禦在第二層 gate 呼叫前攔截，mock 未被消費

    def test_expand_layer_reflect_on_from_prev_layer_ok(self):
        # reflect_on 全屬上一層 → 正常展開（多父可並存）
        mock_gate, calls = make_mock_gate([LAYER_1, LAYER_2])
        r1 = probe_expand_layer(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        parents = r1["layer_entry"]["branches"][:2]  # 上一層兩條 passed
        r2 = probe_expand_layer(
            SITUATION, state=r1["state"], reflect_on=parents, gate_fn=mock_gate,
        )
        assert r2["layer_entry"]["layer"] == 2
        assert len(calls) == 2
        payload2 = json.loads(calls[1].split("\n[task:tree_generate]")[0])
        assert [b["label"] for b in payload2["reflection"]["passed_branches"]] == [
            b["label"] for b in parents
        ]

    def test_expand_layer_first_call_with_reflect_on_raises(self):
        # 首次呼叫（無上一層）帶非空 reflect_on → TreeProbeError（不能有「父」）
        mock_gate, calls = make_mock_gate([])
        with pytest.raises(TreeProbeError, match="reflect_on 分支不屬於上一層"):
            probe_expand_layer(
                SITUATION, n_branch=3, depth=2,
                reflect_on=[valid_branch("任意分支")], gate_fn=mock_gate,
            )
        assert len(calls) == 0  # 防禦攔截，gate 從未被呼叫

    def test_expand_layer_first_call_no_reflect_on_ok(self):
        # 首次呼叫 reflect_on=None → 正常（首層從處境展開）
        mock_gate, calls = make_mock_gate([LAYER_1])
        r1 = probe_expand_layer(SITUATION, n_branch=3, depth=2, gate_fn=mock_gate)
        assert r1["layer_entry"]["layer"] == 1
        assert len(calls) == 1
        payload1 = json.loads(calls[0].split("\n[task:tree_generate]")[0])
        assert "reflection" not in payload1
