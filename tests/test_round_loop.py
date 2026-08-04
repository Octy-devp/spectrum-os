"""Tests for synth/round_loop.py — 回合協定編排器（force-model-social-dynamics.md §2.8）。

一輪 = Mode A（動力學步進直到撕裂）→ STOP → 探針負載 → 人選結算 → 寫回 → 下一輪。

覆蓋：
1. 無撕裂輪——max_steps 內無撕裂 → status=max_steps，軌跡完整。
2. 撕裂即停——mock trigger 在步驟 k 觸發 → 立刻停，status=tear，不跨過撕裂點。
3. 閉合迴圈——tear → settle → 下一輪從新狀態繼續 → 再觸發——至少 3 輪。
4. 人選決定生效——settlement_fn 回傳 mod_alpha/mod_beta 調整 → 下一輪
   effective_rates 反映（乘法合成、呼叫時讀取）。
5. 持久化——store 給定時，round snapshot + field_log 寫出且可 reload。
6. 無 LLM 依賴——import round_loop 不新增任何 LLM/api/prompt 模組。
7. 協定守規——未結算不可跨輪；無待結算不可 settle。
"""

from __future__ import annotations

import importlib
import inspect
import sys

import numpy as np
import pytest

from spectrum_os.contracts import FieldLog
from spectrum_os.kernel.forces import ForceFieldDynamics
from spectrum_os.synth.convergence_store import (
    list_rounds,
    load_field_log,
    load_round_snapshot,
)
from spectrum_os.synth.round_loop import RoundLoop, RoundProtocolError


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


# ---------------------------------------------------------------------------
# 1. 無撕裂輪
# ---------------------------------------------------------------------------

class TestNoTearRound:
    def test_max_steps_round_trajectory_complete(self):
        eng = _engine(dt=0.5)
        loop = RoundLoop(eng, lambda t, ctx: False, lambda payload: {})
        result = loop.run_one_round(max_steps=10)

        assert result["status"] == "max_steps"
        assert result["tear_info"] is None
        assert result["round"] == 1
        assert result["n_steps"] == 10
        assert len(result["trajectory"]) == 10
        # 軌跡完整：10 步從 t=0.5 到 t=5.0
        times = [s["t"] for s in result["trajectory"]]
        assert times == pytest.approx([0.5 * k for k in range(1, 11)])
        # S ∈ [0, 1] 不變式全軌跡成立
        S = np.array([s["S"][0] for s in result["trajectory"]])
        assert np.all((S >= 0.0) & (S <= 1.0))
        # 引擎時間推進到窗末
        assert eng.t == pytest.approx(5.0)
        # 未結算前 rounds_completed 為 0
        assert loop.rounds_completed() == 0

    def test_probe_payload_has_state_and_trajectory(self):
        eng = _engine(dt=0.5)
        loop = RoundLoop(eng, lambda t, ctx: False, lambda payload: {})
        result = loop.run_one_round(max_steps=3)
        probe = result["probe"]
        assert probe["round"] == 1 and probe["status"] == "max_steps"
        assert "state" in probe and "trajectory" in probe and "s_series" in probe
        assert "R" in probe["state"] and "C" in probe["state"] and "S" in probe["state"]
        assert len(probe["s_series"]) == 3


# ---------------------------------------------------------------------------
# 2. 撕裂即停（Mode A 永不跨越撕裂點）
# ---------------------------------------------------------------------------

class TestTearStopsImmediately:
    def test_tear_at_time_k_stops_without_crossing(self):
        """trigger 在 t >= 4.0 觸發 → 軌跡止於 4.0（含），不含任何 t > 4.0。"""
        eng = _engine(dt=0.5)
        loop = RoundLoop(eng, lambda t, ctx: t >= 4.0, lambda payload: {})
        result = loop.run_one_round(max_steps=100)

        assert result["status"] == "tear"
        assert result["tear_info"]["triggered"] is True
        assert result["tear_info"]["at_time"] == pytest.approx(4.0)
        # 步進前檢查：撕裂點在 4.0 → 8 步（0.5..4.0）
        assert result["n_steps"] == 8
        times = [s["t"] for s in result["trajectory"]]
        assert times[-1] == pytest.approx(4.0)
        # 不跨過：無任何一步超過撕裂點
        assert all(t <= 4.0 + 1e-9 for t in times)
        # 引擎時間停在撕裂點，未繼續推進
        assert eng.t == pytest.approx(4.0)

    def test_tear_at_step_k_counting_mock(self):
        """mock trigger 第 k 次檢查才觸發 → 立刻停（第 k 步不跨出）。"""
        eng = _engine(dt=0.5)
        calls = {"n": 0}

        def trigger(t, ctx):
            calls["n"] += 1
            return calls["n"] >= 5  # 第 5 次檢查（t=2.0 前）觸發

        loop = RoundLoop(eng, trigger, lambda payload: {})
        result = loop.run_one_round(max_steps=100)
        assert result["status"] == "tear"
        # 步進前檢查：第 5 次檢查時已步 4 步（t=2.0）
        assert result["n_steps"] == 4
        assert result["trajectory"][-1]["t"] == pytest.approx(2.0)
        assert eng.t == pytest.approx(2.0)

    def test_round_starting_on_tear_yields_empty_trajectory(self):
        """輪起始即撕裂 → 軌跡為空、status=tear（Mode A 一步未走）。"""
        eng = _engine(dt=0.5)
        loop = RoundLoop(eng, lambda t, ctx: True, lambda payload: {})
        result = loop.run_one_round(max_steps=100)
        assert result["status"] == "tear"
        assert result["n_steps"] == 0
        assert result["trajectory"] == []
        assert eng.t == pytest.approx(0.0)
        # 探針仍有撕裂狀態快照（人選仍可結算）
        assert result["probe"]["state"]["t"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. 閉合迴圈（tear → settle → 下一輪 → 再觸發）
# ---------------------------------------------------------------------------

class TestClosedLoop:
    def test_three_rounds_continue_from_new_state(self):
        eng = _engine(dt=0.5)
        tears = [5.0, 9.0, 13.0]
        idx = {"i": 0}

        def trigger(t, ctx):
            if idx["i"] < len(tears) and t >= tears[idx["i"]]:
                idx["i"] += 1
                return True
            return False

        loop = RoundLoop(eng, trigger, lambda payload: {})
        expected_t = 0.0
        for expected_tear in tears:
            result = loop.run_one_round(max_steps=100)
            assert result["status"] == "tear"
            assert result["tear_info"]["at_time"] == pytest.approx(expected_tear)
            # 每一輪從上一輪結算後的新狀態起點繼續（時間單調遞進）
            assert result["start_state"]["t"] == pytest.approx(expected_t)
            loop.settle()
            expected_t = expected_tear

        assert loop.rounds_completed() == 3
        assert [r["round"] for r in loop.history()] == [1, 2, 3]
        assert [r["round"] for r in loop.settlement_history()] == [1, 2, 3]
        assert eng.t == pytest.approx(13.0)
        assert loop.phase() == "ready"

    def test_rounds_with_different_settlement_decisions(self):
        """每輪人選給不同 decision → 各輪 FieldLog 各自記錄、可跨輪累積。"""
        eng = _engine(dt=0.5)
        loop = RoundLoop(eng, lambda t, ctx: t >= 4.0, lambda payload: {})

        loop.run_one_round(max_steps=100)
        loop.settle({"mod_alpha": 1.0, "accepted_path": "path-a",
                     "rejected_paths": ["path-b", "path-c"]})
        loop.run_one_round(max_steps=100)
        loop.settle({"mod_alpha": 1.5, "accepted_path": "path-d"})

        assert loop.rounds_completed() == 2
        fl = loop.field_log
        assert len(fl.entries) == 2
        # FieldLog 契約：add_entry(layer, selected, unselected) → entry = {layer,
        # selected, unselected}——每輪記錄在 selected 鍵下。
        assert fl.entries[0]["selected"]["accepted"] == "path-a"
        assert fl.entries[0]["selected"]["rejected"] == ["path-b", "path-c"]
        assert fl.entries[1]["selected"]["accepted"] == "path-d"
        # 未選（rejected）維持疊加，不刪除；第二輪無 rejected → unselected 為空
        assert fl.entries[1]["unselected"] == []


# ---------------------------------------------------------------------------
# 4. 人選決定生效（decision → 下一輪 effective_rates 反映）
# ---------------------------------------------------------------------------

class TestSettlementDecisionsTakeEffect:
    def test_mod_adjustment_reflected_in_effective_rates(self):
        eng = _engine(alpha=0.012, beta=0.010, dt=0.5)
        loop = RoundLoop(eng, lambda t, ctx: False, lambda payload: {})
        loop.run_one_round(max_steps=2)

        # 基線：mods identity → α_eff=0.012, β_eff=0.010
        a0, b0 = eng.effective_rates(0.0)
        assert a0.tolist() == pytest.approx([0.012])
        assert b0.tolist() == pytest.approx([0.010])

        # 人選回傳 mod_alpha=2.0, mod_beta=0.5
        rec = loop.settle({"mod_alpha": 2.0, "mod_beta": 0.5})
        assert rec["applied"]["mod_alpha"] == [2.0]
        assert rec["applied"]["mod_beta"] == [0.5]
        assert rec["written"] == {}

        # 合成 mod 呼叫時讀取 → settle 立即生效
        a1, b1 = eng.effective_rates(0.0)
        assert a1.tolist() == pytest.approx([0.024])  # 0.012 × 2.0
        assert b1.tolist() == pytest.approx([0.005])  # 0.010 × 0.5

        # 下一輪步進全程反映
        result2 = loop.run_one_round(max_steps=2)
        t2 = result2["trajectory"][-1]["t"]
        a2, b2 = eng.effective_rates(t2)
        assert a2.tolist() == pytest.approx([0.024])
        assert b2.tolist() == pytest.approx([0.005])
        assert loop.round_modulation()[0].tolist() == [2.0]
        assert loop.round_modulation()[1].tolist() == [0.5]

    def test_mod_adjustment_actually_changes_dynamics(self):
        """α_eff 提高（β_eff 降低）→ C 被更強壓制：下一輪的 C 軌跡更低。"""
        eng = _engine(alpha=0.012, beta=0.010, dt=0.5)
        loop = RoundLoop(eng, lambda t, ctx: False, lambda payload: {})
        r1 = loop.run_one_round(max_steps=2)
        loop.settle({"mod_alpha": 2.0, "mod_beta": 0.5})
        r2 = loop.run_one_round(max_steps=2)

        C1 = np.array([s["C"][0] for s in r1["trajectory"]])
        C2 = np.array([s["C"][0] for s in r2["trajectory"]])
        # 兩輪同長（2 步）；α_eff 加倍 + β_eff 減半 → dC = c·C − α_eff·R 更負
        assert C2[-1] < C1[-1]
        assert np.all(C2 < C1)

    def test_settlement_fn_decision_used_when_settle_without_arg(self):
        """settle() 不帶參數 → 內部呼叫 settlement_fn(probe) 取 decision。"""
        eng = _engine(alpha=0.012, dt=0.5)
        seen = {}

        def settlement(payload):
            seen["payload"] = payload
            return {"mod_alpha": 1.5}

        loop = RoundLoop(eng, lambda t, ctx: t >= 3.0, settlement)
        result = loop.run_one_round(max_steps=100)
        assert result["status"] == "tear"
        loop.settle()

        p = seen["payload"]
        assert p["round"] == 1 and p["status"] == "tear"
        assert p["tear_info"]["at_time"] == pytest.approx(3.0)
        assert "state" in p and "trajectory" in p and "s_series" in p
        a, _ = eng.effective_rates(0.0)
        assert a.tolist() == pytest.approx([0.012 * 1.5])

    def test_state_override_applied_to_next_round(self):
        """decision["state"] 覆寫節點狀態 → 下一輪從覆寫後的 R/C 起點。"""
        eng = _engine(dt=0.5)
        loop = RoundLoop(eng, lambda t, ctx: False, lambda payload: {})
        loop.run_one_round(max_steps=2)
        rec = loop.settle({"state": {"R": 100.0, "C": 1.0, "t": 20.0}})
        assert rec["applied"]["state"] == {"R": [100.0], "C": [1.0], "t": 20.0}
        # 下一輪起點 = 覆寫後的狀態
        result = loop.run_one_round(max_steps=2)
        assert result["start_state"]["t"] == pytest.approx(20.0)
        assert result["start_state"]["R"][0] == pytest.approx(100.0)
        assert result["start_state"]["C"][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. 持久化（store 給定 → snapshot + field_log 寫出且可 reload）
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_field_log_and_snapshot_written_and_reloadable(self, tmp_path):
        eng = _engine(dt=0.5)
        store = {
            "field_log_path": tmp_path / "field_log.json",
            "snapshot_dir": tmp_path / "rounds",
        }
        loop = RoundLoop(eng, lambda t, ctx: t >= 4.0,
                         lambda payload: {"accepted_path": "route-a"}, store=store)
        result = loop.run_one_round(max_steps=100)
        assert result["status"] == "tear"
        rec = loop.settle()
        assert set(rec["written"]) == {"field_log", "snapshot"}

        # field log 落盤 + reload
        fl = load_field_log(store["field_log_path"])
        assert isinstance(fl, FieldLog)
        assert len(fl.entries) == 1
        # FieldLog 契約：每輪記錄在 entry["selected"] 鍵下
        assert fl.entries[0]["selected"]["round"] == 1
        assert fl.entries[0]["selected"]["accepted"] == "route-a"

        # round snapshot 落盤 + reload
        snap = load_round_snapshot(1, store["snapshot_dir"])
        assert snap["round"] == 1
        assert snap["field_log_entry"]["round"] == 1
        assert snap["path"] == ["route-a"]
        assert list_rounds(store["snapshot_dir"]) == [1]

        # 再一輪 → 2 筆 entry + 2 份快照
        loop.run_one_round(max_steps=100)
        loop.settle()
        fl2 = load_field_log(store["field_log_path"])
        assert len(fl2.entries) == 2
        assert list_rounds(store["snapshot_dir"]) == [1, 2]
        assert load_round_snapshot(2, store["snapshot_dir"])["round"] == 2

    def test_no_store_runs_pure_in_memory(self, tmp_path):
        eng = _engine(dt=0.5)
        loop = RoundLoop(eng, lambda t, ctx: False, lambda payload: {})
        loop.run_one_round(max_steps=3)
        rec = loop.settle({"accepted_path": "path-a"})
        assert rec["written"] == {}
        assert loop.rounds_completed() == 1
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# 6. 無 LLM 依賴（import round_loop 不觸發任何 LLM/api/prompt 模組）
# ---------------------------------------------------------------------------

LLM_MARKERS = ("api", "llm", "prompt", "openai", "anthropic", "deepseek", "gate")


class TestNoLLMDependency:
    def test_import_adds_no_llm_modules(self):
        import spectrum_os.synth  # noqa: F401 — package __init__ 已載入 LLM-gated 模組

        before = set(sys.modules)
        rl = importlib.import_module("spectrum_os.synth.round_loop")
        new_mods = set(sys.modules) - before
        bad = sorted(m for m in new_mods if any(mk in m.lower() for mk in LLM_MARKERS))
        assert bad == [], f"import round_loop 觸發 LLM 模組：{bad}"

    def test_module_namespace_has_no_llm_names(self):
        rl = importlib.import_module("spectrum_os.synth.round_loop")
        for name in list(vars(rl)):
            if any(mk in name.lower() for mk in LLM_MARKERS):
                raise AssertionError(f"round_loop 命名空間含 LLM 名：{name}")

    def test_source_imports_are_llm_free(self):
        rl = importlib.import_module("spectrum_os.synth.round_loop")
        src = inspect.getsource(rl)
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                low = stripped.lower()
                for mk in LLM_MARKERS:
                    assert mk not in low, f"round_loop 源碼含 LLM import：{line}"


# ---------------------------------------------------------------------------
# 7. 協定守規（§2.8 流程錯誤 → RoundProtocolError）
# ---------------------------------------------------------------------------

class TestProtocolRules:
    def test_cannot_run_next_round_before_settlement(self):
        eng = _engine(dt=0.5)
        loop = RoundLoop(eng, lambda t, ctx: t >= 4.0, lambda payload: {})
        loop.run_one_round(max_steps=100)
        assert loop.phase() == "awaiting_settlement"
        with pytest.raises(RoundProtocolError):
            loop.run_one_round(max_steps=100)  # 未結算不可跨輪
        # 結算後才可繼續
        loop.settle()
        assert loop.phase() == "ready"
        result = loop.run_one_round(max_steps=100)
        assert result["round"] == 2

    def test_cannot_settle_without_pending_round(self):
        eng = _engine(dt=0.5)
        loop = RoundLoop(eng, lambda t, ctx: False, lambda payload: {})
        with pytest.raises(RoundProtocolError):
            loop.settle({})

    def test_settle_without_decision_and_no_callback_raises(self):
        eng = _engine(dt=0.5)
        loop = RoundLoop(eng, lambda t, ctx: False, settlement_fn=None)
        loop.run_one_round(max_steps=2)
        with pytest.raises(RoundProtocolError):
            loop.settle()  # 無 decision 且無 settlement_fn
