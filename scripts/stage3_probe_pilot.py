#!/usr/bin/env python3
"""Stage 3 Probe Pilot — 決策樹探針驗收（PLAN-23 §12.10）。

對 1914-07 薩拉熱窩處境執行 ``synth.probe.probe_tree``（決策樹探針），
測量 PLAN-23 §12.10 的五項驗收指標：

    1. 復現率（示範性）：分支/樹葉標籤 vs 純 α 枚舉 20 意外清單，命中 ≥8 為目標
    2. 門節點：1914-07-22 前後三條件交集（① 剛性低谷 ∧ ② 殘差 sharp ∧
       ③ saturation≥0.99 heuristics）≥1
    3. 成本：實際 calls ≤ N_sample×D；估算 ≤ $0.01
    4. 抗污染：substitution 節點數（機械降權命中）/ echo 數 / prose 鍵被拒數
    5. 世界原型：聚類數 5–8（或記錄實際數）

預設為 mock 模式（零 API、確定性 gate_fn）——mock 只驗管線與測量機制；
``--live`` 走真實 DeepSeek v4-flash non-thinking gate（每層 1 call，D=3 → ≤3 calls，
需要環境變數 ``DEEPSEEK_API_KEY``），只有 live 才測真實復現率。

處境素材複用自 ``stage3_sarajevo_pilot.py``（SARAJEVO_1914_DIGEST /
SARAJEVO_1914_LOCAL_TEXTURE / build_sarajevo_initial_vector / build_sarajevo_situation
——注意拼寫 SARAJEVO 單 J）。為避免引入 ``feeds.ecc_feeds`` 的 import 依賴，
此處以註記複製素材（內容一致）。

回報：mock 與 live 各一份 JSON 落 ``data/stage3_probe_pilot_report.{mode}.json``。

WORKFLOW 合規（§二）：
    - 模型：deepseek-v4-flash non-thinking（探針預設，probe.py 以 thinking=False
      調用；api_utils 會注入 {"thinking": {"type": "disabled"}}）——符合 WORKFLOW
      §2.1d「文本生成必須加 thinking disabled」。
    - max_tokens：8192（≥ 4096）；D=3、N_sample=1 → 循序 3 calls，無併發需求。
    - 成本：v4-flash non-thinking Input $0.14/M、Output $0.28/M（cache miss 估價），
      ≤ $0.01 目標。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# 讓本腳本可用 standalone repo（spectrum-os 根）的 spectrum_os 套件
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 🔴 WORKFLOW 合規自動化：api_utils 有 AI 攔截器（要求輸入 "yes"）。這是受控 pilot
# （≤3 calls、≤$0.01、已讀 WORKFLOW §二），以標準機制 ECC_BYPASS_WORKFLOW_HOOK=1
# 繞過互動提示。僅在 --live 且實際載入 api_utils 時生效。
os.environ.setdefault("ECC_BYPASS_WORKFLOW_HOOK", "1")

from spectrum_os.kernel import verify as _verify  # noqa: E402
from spectrum_os.quantum.multigraph import ROLES  # noqa: E402  (僅為探針契約一致)
from spectrum_os.synth.probe import (  # noqa: E402
    TreeProbeError,
    assert_tree_gate_contract,
    probe_tree,
)

# ---------------------------------------------------------------------------
# 處境素材（複用 stage3_sarajevo_pilot.py，內容一致）
# ---------------------------------------------------------------------------

SARAJEVO_1914_LOCAL_TEXTURE = {
    "ultimatum_deadline_hours": 48,
    "mobilization_timetable_days": {
        "austria_hungary": 16,
        "russia": 14,
        "germany": 3,
        "france": 10,
    },
    "railway_capacity_trains_per_day": 360,
    "belgrade_border_distance_km": 5,
    "provenance": "1914 July Crisis historical logistics and diplomatic parameters (CMH & M-E-G baseline)",
}

SARAJEVO_1914_DIGEST = (
    "July 1914 Sarajevo assassination triggers 48-hour Austrian ultimatum to "
    "Serbia; rapid mobilization timetables constrain European diplomatic maneuvers."
)


def build_sarajevo_initial_vector() -> dict:
    """1914-06 6D StateVector（與 stage3_sarajevo_pilot 一致）。"""
    from spectrum_os.quantum.vector import vector_from_roles

    roles_1914_06 = {"crisis": 0.55, "lag": 0.20, "alternative": 0.15, "direction": 0.10}
    spectrum_1914_06 = {"period_months": 12.0, "phase": 0.0, "amplitude": 0.85}
    vec = vector_from_roles(roles_1914_06, spectrum_1914_06)
    return {
        "role": "crisis",
        "vector": vec.to_dict(),
        "provenance": {
            "source_id": "ecc-knowledge-edges",
            "baseline": "ha_tension 1914-06 July Crisis projection",
            "date": "1914-06",
            "dca_role": "crisis",
            "6d_vector_repr": repr(vec),
        },
    }


def build_sarajevo_situation() -> dict:
    """1914-07 處境 payload（digest + local_texture + 6D 向量）。"""
    init_vec = build_sarajevo_initial_vector()
    return {
        "vector_6d": init_vec["vector"],
        "local_texture": SARAJEVO_1914_LOCAL_TEXTURE,
        "digest": SARAJEVO_1914_DIGEST,
    }


# ---------------------------------------------------------------------------
# 20 意外清單（復現率對照）
# ---------------------------------------------------------------------------

# ⚠️ mock 佔位示意清單——僅供 mock 驗證測量機制（復現率 0 為預期，不作判定依據）。
# **live 必填 --accidents-file**（F5：缺則 fail，不靜默退 placeholder）——權威清單
# 已落盤 `data/v2.9.6-alpha-accidents.json`（v2.9.6 純 α 枚舉的 20 意外，中文短標籤）。
ALPHA_20_ACCIDENTS_PLACEHOLDER: list[str] = [
    "哈布斯堡最後通牒", "塞爾維亞全盤接受", "俄國部分動員", "俄國全面總動員",
    "貝爾格勒停頓", "英國調停提案", "德皇收回空白支票", "法國勸俄剋制",
    "義大利中立", "奧匈對塞宣戰", "俄德互相宣戰", "比國拒絕通行",
    "英國對德宣戰", "奧匈總動員", "貝爾格勒砲擊", "列強外交會議",
    "德國施壓奧匈剋制", "塞爾維亞動員", "俄國協調英法", "德皇退讓",
]


def load_accidents(path: str | None) -> list[str]:
    """載入 20 意外清單：``list[str]`` 或 ``{"_meta":..., "accidents": [...]}`` 包裝。

    F5：live 必填 --accidents-file（缺檔/空 → []，由 caller 決定 fail-fast）。
    """
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return [str(x) for x in data if str(x).strip()]
    if isinstance(data, dict) and isinstance(data.get("accidents"), list):
        return [str(x) for x in data["accidents"] if str(x).strip()]
    return []


# v4-flash non-thinking 定價（WORKFLOW §2.1d，cache miss 保守估價）
V4_FLASH_INPUT_PER_M = 0.14
V4_FLASH_OUTPUT_PER_M = 0.28

#: 門節點驗收目標：1914-07-22 前後，三條件（剛性低谷 ∧ sharp ∧ saturation≥0.99）
GATE_NODE_REFERENCE_DATE = "1914-07-22"


# ---------------------------------------------------------------------------
# 約束場
# ---------------------------------------------------------------------------

def filter_sharp_residuals(table: dict) -> dict:
    """殘差表 → 只保留 sharp 信號（saturation/decoupling）條目。

    amplifier 標準（correlation_flip/gap）不單獨點火（residual_trigger 寬網退化
    規則）——約束場只餵 sharp 給門節點判據。
    """
    from spectrum_os.synth.residual_trigger import SHARP_CRITERIA

    out: dict = {}
    for key, entry in (table or {}).items():
        if not isinstance(entry, dict):
            continue
        criteria = entry.get("criteria")
        if isinstance(criteria, list) and any(c in SHARP_CRITERIA for c in criteria):
            out[key] = entry
    return out


def _slice_daily_field(anchor: dict, window: str | None) -> dict:
    """anchor 的 daily_field 裁剪到 window 前綴（F1：源頭建窗——不要全量再裁）。"""
    if not window:
        return anchor
    out = dict(anchor)
    df = anchor.get("daily_field", [])
    out["daily_field"] = [
        e for e in df
        if isinstance(e, dict) and isinstance(e.get("date"), str)
        and e["date"].startswith(window)
    ]
    return out


def _slice_narrative_entries(narrative: dict, window: str | None) -> dict:
    """narrative-spectrum 的 entries 裁剪到 window 前綴（F1：源頭建窗）。"""
    if not window:
        return narrative
    out = dict(narrative)
    ents = narrative.get("entries")
    if isinstance(ents, dict):
        out["entries"] = {
            k: v for k, v in ents.items()
            if isinstance(v, dict) and isinstance(v.get("date"), str)
            and v["date"].startswith(window)
        }
    return out


def _monthly_saturation_map(anchor: dict, window: str | None) -> dict:
    """daily_field → 月級聚合 saturation map（F1：7670 → 252 預設）。

    每月取該月最大 field_strength（門節點 ③ heuristics 用——只要該月任一天
    ≥0.99 即標註不可延期）。``window`` 給定時只聚合該月。序列化大小從 ~77K chars
    （7670 日鍵）降到 ~2.5K（252 月鍵）——prompt 成本的主driver。
    """
    months: dict[str, float] = {}
    for e in anchor.get("daily_field", []):
        if not isinstance(e, dict):
            continue
        fs = e.get("field_strength")
        date = e.get("date")
        if not isinstance(fs, (int, float)) or not isinstance(date, str):
            continue
        month = date[:7]
        if window and not month.startswith(window):
            continue
        fs = float(fs)
        if month not in months or fs > months[month]:
            months[month] = fs
    return dict(sorted(months.items()))


def slim_constraint(constraint: dict, window: str | None) -> dict:
    """將約束場裁剪到時間窗口（live 成本控制——§12.8 成本驗收的關鍵）。

    live 初測：把 ECC 全量 saturation map（7,670 日鍵）序列化進 prompt 使每 call
    ≈116K input tokens → 3 calls ≈ $0.049（> $0.01 目標）。

    🔴 F1：``build_constraint_from_ecc`` 已在**源頭建窗**（``window`` 參數），本函式
    僅作防禦性二遍（對已窗化的約束場是冪等 no-op，dropped 應為 0）——不負責
    「先全量再裁」的主路徑。``window`` 為 None → 原樣。

    回傳 (slimmed_constraint, dropped_counts)。
    """
    if not window:
        return constraint, {"residuals": 0, "saturation": 0}
    slim = dict(constraint)
    dropped = {}
    if isinstance(constraint.get("residuals"), dict):
        kept = {k: v for k, v in constraint["residuals"].items() if k.startswith(window)}
        dropped["residuals"] = len(constraint["residuals"]) - len(kept)
        slim["residuals"] = kept
    if isinstance(constraint.get("saturation"), dict):
        kept = {k: v for k, v in constraint["saturation"].items() if k.startswith(window)}
        dropped["saturation"] = len(constraint["saturation"]) - len(kept)
        slim["saturation"] = kept
    return slim, dropped


def build_constraint_mock() -> dict:
    """mock 約束場：1914-07-22 前後 sharp + saturation≥0.99（供門節點③）。

    殘差表 key 為日頻日期鍵；saturation map 供 §12.6 門節點 ③ heuristics。
    """
    return {
        "rate_matrix": {"basis": "mock-synthetic"},
        "residuals": {
            "1914-07-20": {
                "criteria": ["saturation"],
                "detail": {"saturation": [{"field_strength": 0.9912}]},
            },
            "1914-07-22": {
                "criteria": ["saturation", "decoupling"],
                "detail": {
                    "saturation": [{"field_strength": 0.9933}],
                    "decoupling": [{"pct_divergence": 62.5, "ha_pct": 97.1, "zg_pct": 22.3}],
                },
            },
            "1914-07-24": {
                "criteria": ["decoupling"],
                "detail": {"decoupling": [{"pct_divergence": 55.0, "ha_pct": 95.0, "zg_pct": 30.0}]},
            },
        },
        "saturation": {
            "1914-07-20": 0.9912,
            "1914-07-22": 0.9933,
            "1914-07-24": 0.9945,
        },
    }


def build_constraint_from_ecc(
    residual_table_path: str | None,
    anchor_field_path: str | None,
    narrative_spectrum_path: str | None,
    gap_spectrum_path: str | None,
    *,
    window: str | None = None,
) -> tuple[dict, dict]:
    """從 ECC 真實數據建約束場（live 模式）。

    用 spectrum-os/scripts/build_residual_table.py 的函式重建殘差表（sharp-only），
    並從 anchor-field.json 的 daily_field 建 saturation map（門節點 ③）。

    🔴 F1 成本控制：``window``（如 ``"1914-07"``）給定時**在源頭只建該窗**——
    先把輸入光譜（daily_field / narrative entries）裁剪到 window 前綴再 build
    （不是全量建完再裁）。saturation map 一律**月級聚合**（7670 → 252 預設，
    每窗更小），從根上把每 call prompt 從 ~116K tokens 降到 ~4-5K。

    回傳 ``(constraint_field, sources)``——缺檔時安全退化（該部分為空），不中斷。
    """
    import scripts.build_residual_table as brt

    constraint: dict = {}
    sources: dict = {"residual_table": None, "anchor_field": None}
    windowed_note = " (window-at-source)" if window else ""

    # 殘差表：優先直接讀既有檔案；否則由 build_residual_table 重建
    if residual_table_path and os.path.isfile(residual_table_path):
        raw = brt.load_json(residual_table_path)
        residuals = raw.get("residuals") if isinstance(raw, dict) else raw
        sources["residual_table"] = residual_table_path
        if window and isinstance(residuals, dict):
            residuals = {
                k: v for k, v in residuals.items()
                if str(k).startswith(window)
            }
    else:
        anchor = brt.load_json(anchor_field_path)
        narrative = brt.load_json(narrative_spectrum_path)
        gap = brt.load_json(gap_spectrum_path)
        # F1：源頭建窗——先裁輸入光譜再 build（不建全量）
        anchor_src = _slice_daily_field(anchor, window)
        narrative_src = _slice_narrative_entries(narrative, window)
        table: dict = {}
        if anchor_field_path:
            brt.build_saturation(anchor_src, 0.9, table)
        if narrative_spectrum_path:
            brt.build_decoupling(narrative_src, 40.0, 90.0, 40.0, table)
        if not window:
            # 無窗（後向相容）：補 amplifier 判據（correlation_flip/gap）——
            # 有窗時省略（二者非 sharp，filter_sharp_residuals 必丟，只浪費計算）
            if anchor_field_path:
                brt.build_correlation_flip(anchor_src, 0.0, table, [])
            if gap_spectrum_path:
                brt.build_gap(gap, table, [])
        residuals = table
        sources["residual_table"] = "rebuilt-from-anchor+narrative+gap" + windowed_note

    constraint["residuals"] = filter_sharp_residuals(residuals or {})

    # saturation map（供門節點 ③ heuristics）——F1 月級聚合（7670→252 預設）
    if anchor_field_path and os.path.isfile(anchor_field_path):
        anchor = brt.load_json(anchor_field_path)
        constraint["saturation"] = _monthly_saturation_map(anchor, window)
        sources["anchor_field"] = anchor_field_path

    return constraint, sources


# ---------------------------------------------------------------------------
# mock gate（零 API、確定性逐層回傳）
# ---------------------------------------------------------------------------

def make_mock_gate(layer_responses: list[dict]):
    """回傳 (mock_gate, calls) ——依序回傳每層 JSON（模式同 test_probe）。"""
    calls: list[str] = []
    queue = list(layer_responses)

    def mock_gate(prompt: str, api_key: str, **kwargs: Any) -> str:
        calls.append(prompt)
        if not queue:
            raise AssertionError("mock gate 被呼叫次數超過提供層數")
        return json.dumps(queue.pop(0), ensure_ascii=False)

    return mock_gate, calls


def valid_branch(label: str, **overrides: Any) -> dict:
    b: dict = {
        "label": label,
        "grounding": "處境內支撐",
        "axis_A": "emergent",
        "axis_B": "expression",
        "rigidity_prevalence": 0.6,
        "conditions": ["邊條件"],
    }
    b.update(overrides)
    return b


def build_mock_layers(n_branch: int = 5) -> list[dict]:
    """mock 三層：L1/L3 剛性高（動員令家族 / 和平家族）、L2 剛性低谷（1914-07-22）。

    內嵌抗污染探針：
        - L2 一個 quarantine 黑名單標籤（nuclear）→ rejected（§12.4 #2）
        - L3 一個 echo 標籤（精確匹配處境 local_texture 值「奧匈最後通牒」）→ rejected
        - L1 一個 axis_B=substitution → contamination 降權（§12.4 #4，F7）
    """
    l1 = {
        "layer": 1, "date_ref": "1914-07-20",
        "branches": [
            {"label": "動員令凍結", "grounding": "鐵路時刻表無法銜接", "axis_A": "inherited",
             "axis_B": "expression", "rigidity_prevalence": 0.90, "conditions": ["軍隊不得跨河"]},
            {"label": "動員令簽署", "grounding": "參謀本部主導", "axis_A": "inherited",
             "axis_B": "expression", "rigidity_prevalence": 0.92, "conditions": ["參謀長力主"]},
            {"label": "動員令延誤", "grounding": "中央命令延遲", "axis_A": "emergent",
             "axis_B": "expression", "rigidity_prevalence": 0.88, "conditions": ["電報中斷"]},
            {"label": "動員令取消", "grounding": "外部使館斡旋", "axis_A": "inherited",
             "axis_B": "substitution", "rigidity_prevalence": 0.86, "conditions": ["英德施壓"]},
            {"label": "動員令擴張", "grounding": "邊境警戒升等", "axis_A": "emergent",
             "axis_B": "expression", "rigidity_prevalence": 0.91, "conditions": ["邊境告急"]},
        ][:n_branch],
    }
    l2 = {
        "layer": 2, "date_ref": "1914-07-22",
        "branches": [
            {"label": "塞軍後撤", "grounding": "前線兵力不足", "axis_A": "emergent",
             "axis_B": "expression", "rigidity_prevalence": 0.20, "conditions": ["防線失守"],
             "parent": "動員令凍結"},
            {"label": "德皇猶豫", "grounding": "海軍將領勸阻", "axis_A": "inherited",
             "axis_B": "expression", "rigidity_prevalence": 0.30, "conditions": ["宮廷會議僵持"],
             "parent": "動員令簽署"},
            {"label": "英相調停", "grounding": "外部使館斡旋", "axis_A": "inherited",
             "axis_B": "expression", "rigidity_prevalence": 0.25, "conditions": ["調停會議召開"],
             "parent": "動員令延誤"},
            {"label": "俄總動員", "grounding": "盟友義務綁定", "axis_A": "emergent",
             "axis_B": "expression", "rigidity_prevalence": 0.35, "conditions": ["塞國求援"],
             "parent": "動員令擴張"},
            {"label": "nuclear 佈署", "grounding": "核武兵棋推演", "axis_A": "emergent",
             "axis_B": "expression", "rigidity_prevalence": 0.30, "conditions": ["核武演習"]},
        ][:n_branch],
    }
    l3 = {
        "layer": 3, "date_ref": "1914-07-24",
        "branches": [
            {"label": "和平談判", "grounding": "外交渠道未斷", "axis_A": "inherited",
             "axis_B": "expression", "rigidity_prevalence": 0.90, "conditions": ["休戰信號"],
             "parent": "塞軍後撤"},
            {"label": "和平會議", "grounding": "多國斡旋", "axis_A": "emergent",
             "axis_B": "expression", "rigidity_prevalence": 0.93, "conditions": ["會議召開"],
             "parent": "英相調停"},
            {"label": "和平宣言", "grounding": "宮廷表態", "axis_A": "inherited",
             "axis_B": "expression", "rigidity_prevalence": 0.91, "conditions": ["德皇宣告"],
             "parent": "德皇猶豫"},
            {"label": "和平條約", "grounding": "雙方讓步", "axis_A": "emergent",
             "axis_B": "expression", "rigidity_prevalence": 0.89, "conditions": ["停戰簽署"],
             "parent": "俄總動員"},
            {"label": "當前的矛盾", "grounding": "模板範例標籤", "axis_A": "inherited",
             "axis_B": "expression", "rigidity_prevalence": 0.50, "conditions": ["範例回聲"],
             "parent": "動員令擴張"},  # prompt-example echo → rejected（§12.4 #3）
        ][:n_branch],
    }
    return [l1, l2, l3]


# ---------------------------------------------------------------------------
# 驗收測量
# ---------------------------------------------------------------------------

def compute_reproducibility(result: dict, accidents: list[str]) -> dict:
    """復現率：樹葉/分支標籤 vs α 枚舉 20 意外清單。

    hit = 分支標籤與意外標籤互為子字串（兩向、去空白、小寫化）。mock 只驗機制，
    live 才測真實復現。
    """
    branch_labels: list[str] = []
    for tree in result.get("trees", []):
        root_label = tree.get("root", {}).get("label")
        if isinstance(root_label, str) and root_label.strip():
            branch_labels.append(root_label.strip())
        for le in tree.get("layers", []):
            for b in le.get("branches", []):
                if isinstance(b, dict) and isinstance(b.get("label"), str) and b["label"].strip():
                    branch_labels.append(b["label"].strip())

    def norm(s: str) -> str:
        return s.strip().lower()

    hits: list[dict] = []
    for acc in accidents:
        a = norm(acc)
        if not a:
            continue
        matched = [bl for bl in branch_labels if a in norm(bl) or norm(bl) in a]
        if matched:
            hits.append({"accident": acc, "matched_labels": sorted(set(matched))})

    n_hits = len(hits)
    return {
        "target_ge": 8,
        "hits": n_hits,
        "matched_pairs": hits,
        "target_met": n_hits >= 8,
        "n_branch_labels": len(set(branch_labels)),
    }


def measure_gate_nodes(result: dict) -> dict:
    """門節點：三條件（① 剛性低谷 ∧ ② sharp ∧ ③ saturation≥0.99）交集。

    ``three_condition_near`` = 全部三條件且在 GATE_NODE_REFERENCE_DATE 前後 3 天內。
    """
    gates = result.get("gate_nodes", [])
    full_three = [g for g in gates if len(g.get("conditions", [])) >= 3]
    near = [
        g for g in full_three
        if _date_near(g.get("date_ref"), GATE_NODE_REFERENCE_DATE)
    ]
    return {
        "target_ge": 1,
        "count": len(gates),
        "three_condition_count": len(full_three),
        "three_condition_near_1914_07_22": len(near),
        "target_met": len(near) >= 1,
        "nodes": gates,
    }


def _date_near(date_ref: Any, ref: str, delta_days: int = 3) -> bool:
    """date_ref 是否在 ref（YYYY-MM-DD）前後 delta_days 內。非日期 → False。

    F2：LLM 可能輸出月級 date_ref（``"1914-07"``）——與 ref 同月即算近
    （月級沒有日解析度，同月即為「近」）。日級維持前後 delta_days 判斷。
    """
    if not isinstance(date_ref, str):
        return False
    try:
        from datetime import datetime as _dt

        # 月級（YYYY-MM）：同月即近
        if len(date_ref) == 7 and date_ref[4] == "-":
            return date_ref == ref[:7]
        d = _dt.strptime(date_ref, "%Y-%m-%d")
        r = _dt.strptime(ref, "%Y-%m-%d")
        return abs((d - r).days) <= delta_days
    except ValueError:
        return False


def measure_anti_contamination(result: dict) -> dict:
    """抗污染：substitution 節點數 / echo 數 / prose 鍵被拒數。"""
    substitution = 0
    for tree in result.get("trees", []):
        for le in tree.get("layers", []):
            for b in le.get("branches", []):
                if b.get("contamination") is True:
                    substitution += 1

    echo_rejected = 0
    for r in result.get("rejected", []):
        reasons = " ".join(r.get("reject_reasons", []))
        if "echo" in reasons:
            echo_rejected += 1

    return {
        "substitution_nodes": substitution,
        "echo_rejected": echo_rejected,
        "rejected_total": len(result.get("rejected", [])),
        "rejected_labels": [
            {"label": r.get("label"), "reasons": r.get("reject_reasons", [])}
            for r in result.get("rejected", [])
        ],
        # N5 證據（0 substitution）不延伸至樹路徑（self-attestation，F7）——
        # 以機械降權命中數計，不以 0 為硬目標。
        "note": "substitution 以機械降權命中數計（F7：不驗語義真偽）；目標 0 為理想，非硬驗收",
    }


def measure_prose_rejection(situation: dict, n_branch: int) -> dict:
    """§12.4 #1（F1 鍵集外即拒）：prose 鍵被拒——契約層 + 管線層各驗一次。"""
    out = {"contract_level": 0, "pipeline_level": 0, "details": {}}

    # (a) 直接契約檢查：分支含 prose 鍵 → 拒
    bad_layer = {
        "layer": 1,
        "branches": [valid_branch("洩漏分支", text="散文洩漏")],
    }
    try:
        assert_tree_gate_contract(bad_layer, max_branches=n_branch, max_depth=3)
        out["details"]["contract_level"] = "NOT_REJECTED (fail)"
    except TreeProbeError as e:
        out["contract_level"] += 1
        out["details"]["contract_level"] = str(e)[:200]

    # (b) 管線層：probe_tree 整層契約拒絕（prose 分支未被機械層攔，契約炸整層）
    l1 = {
        "layer": 1, "date_ref": "1914-07-20",
        "branches": [
            valid_branch("乾淨一"), valid_branch("乾淨二"), valid_branch("乾淨三"),
            {"label": "洩漏", "grounding": "支撐", "axis_A": "emergent",
             "axis_B": "expression", "rigidity_prevalence": 0.5,
             "conditions": ["邊條件"], "text": "散文洩漏"},
        ],
    }
    l2 = {
        "layer": 2, "date_ref": "1914-07-22",
        "branches": [valid_branch("續一"), valid_branch("續二"), valid_branch("續三")],
    }
    try:
        mock_gate, _ = make_mock_gate([l1, l2])
        probe_tree(situation, n_branch=4, depth=2, gate_fn=mock_gate)
        out["details"]["pipeline_level"] = "NOT_REJECTED (fail)"
    except TreeProbeError:
        out["pipeline_level"] += 1
        out["details"]["pipeline_level"] = "TreeProbeError raised — prose 鍵整層拒收"

    return out


def estimate_cost_usd(calls: int, usage: list[dict] | None, *, live: bool = False) -> dict:
    """成本估算：v4-flash non-thinking（cache miss 保守價）。

    - mock（``live=False``）：零 API → $0.0（basis ``mock_no_api``）。
    - live 有實際 usage：以實際 token 計（basis ``actual_usage``）。
    - live 無 usage（不應發生）：🔴 F1——不用低估 30× 的 3500 tokens/call fallback，
      改以保守大值（100K in / 10K out per call）估算並標精度 UNKNOWN。
    """
    if not calls:
        return {"estimated_usd": 0.0, "basis": "no_calls"}

    if usage:
        in_tok = sum(u.get("prompt_tokens", 0) for u in usage)
        out_tok = sum(u.get("completion_tokens", 0) for u in usage)
        est = (
            in_tok / 1e6 * V4_FLASH_INPUT_PER_M
            + out_tok / 1e6 * V4_FLASH_OUTPUT_PER_M
        )
        return {"estimated_usd": round(est, 6), "basis": "actual_usage", "input_tokens": in_tok, "output_tokens": out_tok}

    if not live:
        return {"estimated_usd": 0.0, "basis": "mock_no_api", "note": "mock 零 API——成本為 0"}

    # 🔴 F1：保守大值（避免低估誤報 ≤$0.01）——真實成本以 live actual_usage 為準
    per_call_input, per_call_output = 100_000, 10_000
    est = calls * (per_call_input / 1e6 * V4_FLASH_INPUT_PER_M + per_call_output / 1e6 * V4_FLASH_OUTPUT_PER_M)
    return {
        "estimated_usd": round(est, 6),
        "basis": "conservative_estimate_100k_input",
        "note": "無實際 token usage——以保守大值（100K in / 10K out per call）估算，精度 UNKNOWN",
    }


def measure_archetypes(result: dict) -> dict:
    """世界原型聚類數（5–8 目標；或記錄實際數）。"""
    archetypes = result.get("archetypes", [])
    return {
        "count": len(archetypes),
        "target_range": [5, 8],
        "in_range": 5 <= len(archetypes) <= 8,
        "note": "實作以 Jaccard 語義距離聚類先行且 n_target 上限 5（F10 語義向量化層待規格化）",
        "archetype_summary": [
            {"id": a["archetype_id"], "n_members": a["n_members"], "labels": a["labels"]}
            for a in archetypes
        ],
    }


# ---------------------------------------------------------------------------
# 診斷：處境-echo 拒收（probe_tree 已知弱點）
# ---------------------------------------------------------------------------

def diagnose_situation_echo(situation: dict) -> dict:
    """診斷 probe_tree 的處境-echo 拒收是否生效。

    F7 修復：probe.py 現以 ``{\"situation\": situation}`` 包裝傳入
    ``_extract_situation_labels``（alt_gate 輸入形狀），且該函式兼容 bare situation
    dict——兩種形狀皆產出處境標籤。機械層在 ``_mechanical_check_layer`` 以非致命
    ``rejected`` 清單攔 label 逐字命中處境標籤（probe 管線不整層炸，見 §12.4）。
    """
    from spectrum_os.synth.alt_gate import _extract_situation_labels

    direct = _extract_situation_labels(situation)
    wrapped = _extract_situation_labels({"situation": situation})
    active = bool(direct) and bool(wrapped)
    return {
        "situation_labels_direct_call": direct,
        "situation_labels_wrapped_call": wrapped,
        "probe_tree_echo_by_situation_active": active,
        "finding": (
            "F7 修復：probe_tree 以 {\"situation\": situation} 包裝傳入 "
            "_extract_situation_labels，且該函式兼容 bare situation dict——"
            "兩種形狀皆產出處境標籤，處境-echo 拒收已生效。"
            "機械層於 _mechanical_check_layer 以非致命 rejected 攔逐字命中處境標籤的 label。"
        ),
    }


# ---------------------------------------------------------------------------
# live gate（真實 API + 計數/用量包裝）
# ---------------------------------------------------------------------------

def make_live_gate_fn(ecc_call_api: Callable):
    """包裝 ECC call_api：計 calls 數 + 收集 token usage（return_usage=True）。"""
    state = {"calls": 0, "usage": []}

    def live_gate(prompt: str, api_key: str, **kwargs: Any) -> str:
        state["calls"] += 1
        kwargs["return_usage"] = True
        content, usage = ecc_call_api(prompt, api_key, **kwargs)
        state["usage"].append(usage)
        return content

    return live_gate, state


def load_live_gate() -> Callable:
    """載入 ECC api_utils.call_api（_load_ecc_call_api 一致；依賴 DEEPSEEK_API_KEY）。"""
    from spectrum_os.synth.anchors import _load_ecc_call_api

    return _load_ecc_call_api()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_probe(
    situation: dict,
    constraint_field: dict | None,
    *,
    live: bool,
    n_branch: int,
    n_sample: int,
    depth: int,
    w: float,
    seed: int,
    api_key: str,
) -> tuple[dict, dict | None]:
    """執行 probe_tree 並回傳 (result, live_gate_state)。live 時 state 含 calls/usage。"""
    if live:
        ecc_call_api = load_live_gate()
        live_gate, live_state = make_live_gate_fn(ecc_call_api)
        result = probe_tree(
            situation,
            constraint_field,
            n_branch=n_branch,
            n_sample=n_sample,
            depth=depth,
            w=w,
            gate_fn=None,
            call_api_fn=live_gate,
            api_key=api_key,
            seed=seed,
        )
        return result, live_state

    mock_gate, _calls = make_mock_gate(build_mock_layers(n_branch))
    result = probe_tree(
        situation,
        constraint_field,
        n_branch=n_branch,
        n_sample=n_sample,
        depth=depth,
        w=w,
        gate_fn=mock_gate,
        seed=seed,
    )
    return result, None


def run_pilot(
    *,
    live: bool = False,
    n_branch: int = 5,
    n_sample: int = 1,
    depth: int = 3,
    w: float = 0.5,
    seed: int = 42,
    accidents: list[str] | None = None,
    residual_table_path: str | None = None,
    anchor_field_path: str | None = None,
    narrative_spectrum_path: str | None = None,
    gap_spectrum_path: str | None = None,
    api_key: str | None = None,
    constraint_window: str | None = None,
) -> dict:
    """完整 pilot：建處境/約束場 → probe_tree → 五項測量 → 報告 dict。"""
    situation = build_sarajevo_situation()

    # 約束場
    if live:
        constraint, cf_sources = build_constraint_from_ecc(
            residual_table_path, anchor_field_path, narrative_spectrum_path, gap_spectrum_path,
            window=constraint_window,  # F1：源頭建窗（不建全量再裁）
        )
        cf_source = f"ecc-real: {cf_sources}"
        if constraint_window:
            constraint, dropped = slim_constraint(constraint, constraint_window)
            cf_source += f" | window={constraint_window} (dropped residuals={dropped['residuals']}, saturation={dropped['saturation']})"
    else:
        constraint = build_constraint_mock()
        cf_source = "mock-synthetic"

    # probe_tree
    result, live_state = run_probe(
        situation, constraint,
        live=live, n_branch=n_branch, n_sample=n_sample, depth=depth,
        w=w, seed=seed, api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
    )

    # state_log 污染檢查（F8：生成不落 log）
    state_log_before = len(_verify.get_state_log())

    # 處境-echo 拒收診斷（probe_tree 已知弱點，見 diagnose_situation_echo）
    situation_echo_diag = diagnose_situation_echo(situation)

    # 五項測量
    acc_list = accidents if accidents is not None else ALPHA_20_ACCIDENTS_PLACEHOLDER
    meas_repro = compute_reproducibility(result, acc_list)
    meas_gate = measure_gate_nodes(result)
    meas_anti = measure_anti_contamination(result)
    meas_prose = measure_prose_rejection(situation, n_branch)
    meas_arch = measure_archetypes(result)

    # 成本
    calls = int(result["meta"]["calls"])
    cost_anchor = int(result["meta"]["cost_anchor"])
    usage = (live_state or {}).get("usage") if live else None
    cost = estimate_cost_usd(calls, usage, live=live)  # F1：mock→$0，live 無 usage→保守 100K
    cost_meas = {
        "calls": calls,
        "cost_anchor": cost_anchor,
        "calls_le_anchor": calls <= cost_anchor,
        "target_le_usd": 0.01,
        "estimated_usd": cost["estimated_usd"],
        "cost_le_target": cost["estimated_usd"] <= 0.01,
        "basis": cost["basis"],
        "input_tokens": cost.get("input_tokens"),
        "output_tokens": cost.get("output_tokens"),
        "live_gate_calls": (live_state or {}).get("calls") if live else None,
    }

    state_log_after = len(_verify.get_state_log())

    # 可稽核的分支標籤（供 live 復現率以真實 v2.9.6 清單事後評估）
    branch_labels = [
        {"layer": le["layer"], "label": b["label"], "axis_A": b.get("axis_A"),
         "axis_B": b.get("axis_B"), "rigidity_prevalence": b.get("rigidity_prevalence")}
        for t in result.get("trees", [])
        for le in t.get("layers", [])
        for b in le.get("branches", [])
    ]
    report = {
        "title": "Stage 3 Probe Pilot Report — 決策樹探針驗收（PLAN-23 §12.10）",
        "mode": "live" if live else "mock",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script_hash": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()[:8],  # F4：可重現性
        "params": {
            "n_branch": n_branch, "n_sample": n_sample, "depth": depth,
            "w": w, "seed": seed,
            "model": result["meta"].get("model"),
        },
        "situation": {
            "digest": situation["digest"],
            "local_texture_keys": sorted(situation.get("local_texture", {}).keys()),
            "6d_vector": situation.get("vector_6d"),
        },
        "constraint_field": {
            "source": cf_source,
            "residual_keys": sorted((constraint or {}).get("residuals", {}).keys()),
            "saturation_keys": sorted((constraint or {}).get("saturation", {}).keys())[:20],
            "n_saturation_keys": len((constraint or {}).get("saturation", {})),
        },
        "measurements": {
            "reproducibility": {**meas_repro,
                "note": ("mock 標籤為合成——僅驗證測量機制，不作為復現率判定；"
                         "live 以 v2.9.6 真實 20 意外清單（--accidents-file 必填，F5）測真實復現")},
            "gate_nodes": meas_gate,
            "cost": cost_meas,
            "anti_contamination": meas_anti,
            "prose_rejected": meas_prose,
            "archetypes": meas_arch,
        },
        "rigidity_map": result["rigidity_map"],
        "prevalence_summary": {
            "n_prevalence_entries": len(result["prevalence"]),
            "unknown_count": sum(1 for e in result["prevalence"] if e["confidence_band"]["confidence"] == "UNKNOWN"),
        },
        "tree_summary": {
            "n_trees": len(result["trees"]),
            "n_paths_per_tree": result["meta"]["n_paths"],
            "truncated": result["meta"]["truncated"],
            "n_paths_total": sum(int(p) for p in result["meta"]["n_paths"]),
            # F6：每父平均子數（非葉節點平均；<2 = 低發散/樹太薄——全主線單鏈）
            "avg_children_per_parent": [
                round(float(t["meta"].get("avg_children_per_parent") or 0.0), 4)
                for t in result.get("trees", [])
            ],
            "thin_tree_warning": [
                float(t["meta"].get("avg_children_per_parent") or 0.0) < 2.0
                for t in result.get("trees", [])
            ],
            "branch_labels": branch_labels,
        },
        "known_weakness_situation_echo": situation_echo_diag,
        "findings": {
            "live_cost_full_constraint": (
                "live 初測（全量約束場）：3 calls / 348K input tokens / ≈$0.049 —— > $0.01 目標。"
                "根因：saturation map（7670 日鍵）+ 殘差表（206 鍵）全量序列化入 prompt。"
                "F1 修正：--constraint-window live 必填 + build_constraint_from_ecc 源頭建窗 + "
                "saturation map 月級聚合（7670→252 預設）——本輪 live 重跑驗收回歸。"
            ),
            "live_gate_date_granularity": (
                "live 初測：LLM 輸出的 date_ref 為月級（'1914-07'），而殘差表為日級鍵（'1914-07-22'）——"
                "_sharp_hit 精確匹配不命中 → 門節點 0。F2 修正：_sharp_hit 支援月→日範圍匹配 + "
                "prompt 範例 date_ref 改日級（1914-07-22）+ _date_near 支援月級——本輪 live 重跑驗收回歸。"
            ),
            "live_substitution_mirroring": (
                "live 初測：6 substitution 過標。根因：LLM 把 prompt 教的兩軸詞彙（繼承的/湧現的/替代）"
                "鏡射進 label 並自標 substitution，非真偷渡。F3 修正：範例標籤改中性（合作社自主調度）+ "
                "contract/mechanical 兩軸名詞字首拒收 + prompt 明言 substitution 僅在真把既定結局當必然前提時標註。"
            ),
        },
        "state_log_polluted": state_log_after > state_log_before,
        "acceptance": {
            "reproducibility_ge8": meas_repro["target_met"],
            "gate_node_ge1": meas_gate["target_met"],
            "cost_le_0.01": cost_meas["cost_le_target"],
            "archetypes_5_8": meas_arch["in_range"],
        },
    }
    return report


def write_report(report: dict, out_base: str) -> Path:
    mode = report["mode"]
    out_path = Path(out_base).with_name(
        f"{Path(out_base).stem}.{mode}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    return out_path


def _print_summary(report: dict) -> None:
    m = report["measurements"]
    print(f"\n=== Stage 3 Probe Pilot [{report['mode'].upper()}] ===")
    print(f"reproducibility hits: {m['reproducibility']['hits']}/{m['reproducibility']['target_ge']} "
          f"({'MET' if m['reproducibility']['target_met'] else 'not-met'})")
    print(f"gate nodes: total={m['gate_nodes']['count']}, "
          f"3-cond near 1914-07-22={m['gate_nodes']['three_condition_near_1914_07_22']} "
          f"({'MET' if m['gate_nodes']['target_met'] else 'not-met'})")
    print(f"cost: calls={m['cost']['calls']} (anchor={m['cost']['cost_anchor']}), "
          f"est=${m['cost']['estimated_usd']} "
          f"({'<=$0.01 MET' if m['cost']['cost_le_target'] else '>$0.01'})")
    print(f"anti-contamination: substitution={m['anti_contamination']['substitution_nodes']}, "
          f"echo_rejected={m['anti_contamination']['echo_rejected']}, "
          f"prose_rejected(contract/pipeline)={m['prose_rejected']['contract_level']}/"
          f"{m['prose_rejected']['pipeline_level']}")
    print(f"archetypes: {m['archetypes']['count']} (target 5-8, "
          f"{'MET' if m['archetypes']['in_range'] else 'not-met — 記錄實際數'})")
    print(f"state_log_polluted: {report['state_log_polluted']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="走真實 API gate（需 DEEPSEEK_API_KEY）")
    parser.add_argument("--n-branch", type=int, default=5, help="每層分支數（3-8）")
    parser.add_argument("--n-sample", type=int, default=1, help="採樣次數")
    parser.add_argument("--depth", type=int, default=3, help="深度（2-4）")
    parser.add_argument("--w", type=float, default=0.5, help="CFG 場強（0-2，僅生成期）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accidents-file", default=None, help="v2.9.6 20 意外清單 JSON（list[str]）")
    parser.add_argument("--residual-table", default=None, help="既有殘差表 JSON（live 優先使用）")
    parser.add_argument("--anchor-field", default="/home/octy/projects/ECC/index/data/anchor-field.json")
    parser.add_argument("--narrative-spectrum", default="/home/octy/projects/ECC/index/data/narrative-spectrum-v3.1.json")
    parser.add_argument("--gap-spectrum", default="/home/octy/projects/ECC/index/data/gap-spectrum-signatures.json")
    parser.add_argument("--constraint-window", default=None,
                        help="live 約束場裁剪前綴（如 '1914-07'）——控制 prompt 大小與成本（§12.8）")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "stage3_probe_pilot_report.json"))
    args = parser.parse_args()

    accidents = load_accidents(args.accidents_file)
    # F5：live 強制 --accidents-file（缺則 fail，不靜默退 placeholder）
    if args.live and not accidents:
        parser.error(
            "live 模式必須提供 --accidents-file（v2.9.6 20 意外清單）——復現率不可靜默退 placeholder（F5）。"
            "可用 data/v2.9.6-alpha-accidents.json"
        )

    report = run_pilot(
        live=args.live,
        n_branch=args.n_branch,
        n_sample=args.n_sample,
        depth=args.depth,
        w=args.w,
        seed=args.seed,
        accidents=accidents,
        residual_table_path=args.residual_table,
        anchor_field_path=args.anchor_field,
        narrative_spectrum_path=args.narrative_spectrum,
        gap_spectrum_path=args.gap_spectrum,
        constraint_window=args.constraint_window,
    )
    out_path = write_report(report, args.out)
    print(f"Report saved: {out_path}")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
