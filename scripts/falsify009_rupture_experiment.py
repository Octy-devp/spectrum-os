#!/usr/bin/env python3
"""FALSIFY-009 — rupture 聯合簽名特異性檢定（零 API，純 numpy）。

依據
----
- `ECC/index/data/research-reports/20260728-rupture-theory-evaluation.md` §五
  （實驗程序三項 + 證偽/通過條件）
- `ECC/docs/plans/PLAN-23-spectrum-computer-engine.md` §6.5（rupture_watch 設計：
  路 A 結構 / 路 B 動力 / 路 C 慢化；偽影過濾在機械入口做；單路永不 ASSERTED）

三個任務
--------
1. 偽影占比正式測量：三棵 warfare 樹統計 identical_text / staleness 置零 /
   template_switch，並對比「過濾前（無 §6.5 去重）vs 過濾後（現行 loader）」
   的 decoherence_watch 熵序列與 alerts。
2. Slutsky 零假設檢定（核心，支柱 4 / 路 B 的生死題）：對真實 sector
   （ha_tension、return_mood）與 100 條「隨機衝擊 + 移動平均」合成序列跑
   同一個 rolling-origin 誤差聚集檢測器（wave.extrapolate 滾動回測，
   最大窗口 z-score）。真實 ≤ 合成分佈 95 分位 → 路 B 單獨不成立。
3. 陰性對照：track0 平穩段（r00-r03）不應觸發 rupture 信號
   （外加 1915 平穩段 r00-r04、cycle2 薄序列）。

判決（對齊研究報告 §五與 PLAN-23 §九 FALSIFY-009 行）
----------------------------------------------------
- 過濾後已標記窗口全部消失 → 退回設計階段
- 已知撕裂時刻不觸發而陰性亂觸發 → 理論駁回
- 合成聚集率 ≥ 真實 → 路 B 降為從屬通道
- ≥2 個獨立窗口 ASSERTED 且全部陰性對照不觸發 → 通過

邊界：ECC 數據唯讀；不動 kernel/quantum 現有邏輯（過濾前版本在本腳本內
以公開 API 重構，不改庫）；種子固定可重現；報告落盤 data/（gitignored）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 優先使用本 repo 的 spectrum_os

import yaml  # noqa: E402

from spectrum_os.kernel.wave import extrapolate  # noqa: E402
from spectrum_os.quantum.decoherence import decoherence_watch  # noqa: E402
from spectrum_os.quantum.multigraph import (  # noqa: E402
    RoleAssignment,
    RoleMultigraph,
    _WARFARE_KEY_TO_ROLE,
    field_mass,
    load_warfare_tree,
)

# ── 常數（全部機械規則，無手編調參） ─────────────────────────────
ECC_WARFARE_FIELDS = Path("/home/octy/projects/ECC/index/warfare/data/fields")
SECTORS_JSON = ROOT / "data" / "sectors" / "sectors.json"
FEED_REPORT_JSON = ROOT / "data" / "sectors" / "feed_report.json"
DEFAULT_OUTPUT = ROOT / "data" / "falsify009_report.json"

TREES = ["experiment-track0", "anti-intervention-war-1915", "anti-intervention-war-cycle2"]
FACTIONS = ["rus", "dsr"]

# 已標記 rupture 窗口（研究報告 §五：1915 r05–07；track0 r03 的 12 平行
# alternatives 在 dca-branch-tree.yaml 無對應測量——track0 r03 僅 2 分支，
# 該標記來自 campaign 日誌而非樹數據，見報告 task1.note）
MARKED_WINDOWS = {
    "anti-intervention-war-1915": {"rus": (5, 7), "dsr": (5, 7)},
}

SEED = 20260728
N_SYNTH = 100            # 每個 sector 的合成序列數（長度對齊該 sector）
HORIZON = 3              # rolling-origin 預測跨度
MIN_TRAIN = 36           # 最小訓練長度（對齊 extrapolate 預設 long_window）
CLUSTER_W = 6            # 誤差聚集窗口長度（月/步）
MA_WINDOW_RANGE = (4, 12)  # Slutsky 移動平均窗口抽樣範圍（含端點）
P95_Q = 95.0

# 檢測器 sanity check 的陽性對照參數：末段衝擊方差放大 + 水平位移。
# 斷裂點取 0.85 而非 0.7：斷裂後的 origin 必須是誤差序列中的少數，
# 否則「持續高誤差區」佔多數、被全域均值/標準差稀釋而不再呈現為聚集
# （n=72 實測：break@0.7 → 後段 20/34 origins，分數反而低於零假設 p50）。
POS_CTRL_BREAK = 0.85
POS_CTRL_SIGMA_MULT = 4.0
POS_CTRL_LEVEL_SHIFT = 1.5


# ────────────────────────────────────────────────────────────────
# 任務 1：偽影占比
# ────────────────────────────────────────────────────────────────

def build_unfiltered_graph(data: dict, system_id: str) -> RoleMultigraph:
    """重構 §6.5 過濾前的 loader 行為（在本腳本內，不改庫）。

    不標 identical_text 旗標的對照變體（loader 現行只標旗標、不清零；
    本變體連旗標也不打）。staleness 置零（field_mass）是既有行為，保留；
    template_switch 標記不影響熵序列，略過。
    """
    graph = RoleMultigraph()
    for faction, entries in (data.get("factions") or {}).items():
        branch_counts: dict[int, int] = {}
        for entry in entries:
            rnd = entry.get("round")
            if rnd is None:
                continue
            branch_idx = branch_counts.get(rnd, 0)
            branch_counts[rnd] = branch_idx + 1
            state_id = f"{faction}/r{int(rnd):02d}"
            thread = f"{state_id}/b{branch_idx}"
            graph.state_times[state_id] = float(rnd)
            for key, value in (entry.get("dca") or {}).items():
                role = _WARFARE_KEY_TO_ROLE.get(key)
                if role is None:
                    continue
                mass = field_mass(value)
                graph.add(RoleAssignment(
                    state_id, thread, role,
                    weight=1.0 if mass > 0 else 0.0,
                    mass=mass,
                    stale=(mass == 0),
                    note="",
                    source=system_id,
                ))
    return graph


def _alarm_list(watch) -> list[dict]:
    return [
        {"from_t": a["from_t"], "t": a["t"], "drop": round(a["drop"], 6)}
        for a in watch.values["alarms"]
    ]


def _state_flag_summary(graph: RoleMultigraph, state_id: str) -> dict:
    """單一 state 的 provenance 旗標摘要（機械標記，判讀歸閘層）。"""
    assignments = graph.state_assignments(state_id)
    n_total = len(assignments)
    n_stale = sum(1 for a in assignments if a.stale)
    n_dup = sum(1 for a in assignments if a.note.startswith("identical_text:"))
    n_active = sum(1 for a in assignments if a.weight > 0)
    meta = graph.state_meta.get(state_id, {})
    flagged = bool(meta.get("template_switch")) or n_dup > 0 or n_active == 0 or (
        n_total > 0 and n_stale / n_total >= 0.5
    )
    return {
        "state_id": state_id,
        "n_assignments": n_total,
        "n_active": n_active,
        "n_stale": n_stale,
        "n_identical_text": n_dup,
        "template": meta.get("template"),
        "template_switch": bool(meta.get("template_switch")),
        "flagged": flagged,
    }


def measure_tree_artifacts(tree: str) -> dict:
    """單棵樹的偽影統計 + 過濾前後 decoherence_watch 對比。"""
    path = ECC_WARFARE_FIELDS / tree / "dca-branch-tree.yaml"
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    filtered = load_warfare_tree(str(path))
    unfiltered = build_unfiltered_graph(raw, system_id=tree)

    n_dup = sum(
        1 for aas in filtered.assignments.values() for a in aas
        if a.note.startswith("identical_text:")
    )
    n_stale = sum(1 for aas in filtered.assignments.values() for a in aas if a.stale)
    n_assign = sum(len(aas) for aas in filtered.assignments.values())
    switches = {
        sid: meta["template"]
        for sid, meta in sorted(filtered.state_meta.items())
        if meta.get("template_switch")
    }

    factions: dict[str, dict] = {}
    for fac in FACTIONS:
        watch_pre = decoherence_watch(unfiltered, fac)
        watch_post = decoherence_watch(filtered, fac)
        series_pre = [round(p["entropy"], 6) for p in watch_pre.values["series"]]
        series_post = [round(p["entropy"], 6) for p in watch_post.values["series"]]
        alarms_post = _alarm_list(watch_post)
        # 每個過濾後警報的落點 state provenance 旗標
        for alarm in alarms_post:
            sid = f"{fac}/r{int(alarm['t']):02d}"
            alarm["target_flags"] = _state_flag_summary(filtered, sid)
        factions[fac] = {
            "rounds": [p["t"] for p in watch_post.values["series"]],
            "entropy_unfiltered": series_pre,
            "entropy_filtered": series_post,
            "alarms_unfiltered": _alarm_list(watch_pre),
            "alarms_filtered": alarms_post,
            "verdict_filtered": watch_post.verdict.name,
        }

    return {
        "path": str(path),
        "counts": {
            "assignments_total": n_assign,
            "identical_text": n_dup,
            "stale_zeroed": n_stale,
            "stale_share": round(n_stale / n_assign, 4) if n_assign else 0.0,
            "template_switch_points": switches,
        },
        "factions": factions,
    }


def evaluate_marked_windows(task1: dict) -> dict:
    """已標記窗口在過濾後是否消失——機械警報層與候選層（未旗標）各算一次。

    「候選層」= §6.5 判定的 rupture 可報信號：過濾後警報落點 state 不帶任何
    provenance 旗標（template_switch / identical_text / 半數以上 stale / 全未測）。
    """
    out = {}
    for tree, windows in MARKED_WINDOWS.items():
        per_fac = {}
        for fac, (lo, hi) in windows.items():
            alarms = task1[tree]["factions"][fac]["alarms_filtered"]
            in_window = [a for a in alarms if lo <= a["t"] <= hi]
            unflagged = [a for a in in_window if not a["target_flags"]["flagged"]]
            per_fac[fac] = {
                "window": [lo, hi],
                "mechanical_alarms_in_window": in_window,
                "unflagged_alarms_in_window": unflagged,
                "vanished_mechanically": len(in_window) == 0,
                "vanished_as_candidate": len(unflagged) == 0,
            }
        out[tree] = per_fac
    return out


# ────────────────────────────────────────────────────────────────
# 任務 2：Slutsky 零假設（路 B 特異性）
# ────────────────────────────────────────────────────────────────

def slutsky_series(rng: np.random.Generator, n: int) -> np.ndarray:
    """隨機衝擊 + 移動平均平滑（Yule/Slutsky 零假設程序，numpy-only）。

    窗口 k 每條序列從 MA_WINDOW_RANGE 抽樣，避免把零假設調參到單一 k。
    輸出 z-正規化（與真實序列同處理，比較的是時間結構而非尺度）。
    """
    k = int(rng.integers(MA_WINDOW_RANGE[0], MA_WINDOW_RANGE[1] + 1))
    eps = rng.standard_normal(n + k - 1)
    y = np.convolve(eps, np.ones(k) / k, mode="valid")
    return _znormalize(y)


def positive_control_series(rng: np.random.Generator, n: int) -> np.ndarray:
    """陽性對照：Slutsky 基底 + 末段機制斷裂（方差放大 + 水平位移）。

    用於確認檢測器對真實斷裂有反應——若連陽性對照都不超過零假設 95 分位，
    則「真實 ≤ p95」不能解讀為路 B 無特異性，而是檢測器無鑑別力。
    """
    k = int(rng.integers(MA_WINDOW_RANGE[0], MA_WINDOW_RANGE[1] + 1))
    eps = rng.standard_normal(n + k - 1)
    y = np.convolve(eps, np.ones(k) / k, mode="valid")
    br = int(n * POS_CTRL_BREAK)
    y[br:] = y[br:] * POS_CTRL_SIGMA_MULT + POS_CTRL_LEVEL_SHIFT
    return _znormalize(y)


def _znormalize(y: np.ndarray) -> np.ndarray:
    sd = float(np.std(y))
    if sd < 1e-12:
        return y - float(np.mean(y))
    return (y - float(np.mean(y))) / sd


def rolling_origin_errors(y: np.ndarray, horizon: int = HORIZON,
                          min_train: int = MIN_TRAIN) -> tuple[np.ndarray, np.ndarray]:
    """wave.extrapolate 滾動回測：每個 origin 只用 t 之前的數據預測 horizon 步。

    回傳 (origins, rmse)。點預測不依賴 Monte Carlo 帶，n_simulations 取小值
    僅為走通完整呼叫路徑；seed 固定（MC 殘差重抽不影響點預測）。
    """
    n = len(y)
    origins, errs = [], []
    for t in range(min_train, n - horizon + 1):
        res = extrapolate(y[:t], horizon=horizon, n_simulations=8, seed=SEED)
        fc = np.asarray(res["forecast"], dtype=np.float64)
        if np.any(np.isnan(fc)):
            continue
        errs.append(float(np.sqrt(np.mean((fc - y[t:t + horizon]) ** 2))))
        origins.append(t)
    return np.asarray(origins, dtype=int), np.asarray(errs, dtype=np.float64)


def cluster_score(errors: np.ndarray, origins: np.ndarray,
                  w: int = CLUSTER_W) -> dict:
    """最大窗口 z-score：滑動窗口均值相對全誤差序列的偏離（scan statistic 形式）。

    雙版本並報（判決以 plain 為主、robust 為敏感性檢查）：
    - ``plain``：均值/標準差 z——任務書「最大窗口 z-score」的直譯；
    - ``robust``：中位數/MAD z——抵抗「聚集本身抬高全域 sd 而自我稀釋」
      的掩蔽效應（真實序列若含真聚集，plain 會低估它）。
    兩者皆對序列自身尺度不變（內部 z），真實/合成可比。
    """
    if len(errors) < w:
        return {"plain": 0.0, "robust": 0.0, "argmax_origin": None}
    csum = np.concatenate([[0.0], np.cumsum(errors)])
    window_means = (csum[w:] - csum[:-w]) / w

    mu = float(np.mean(errors))
    sd = float(np.std(errors))
    z_plain = (window_means - mu) / sd if sd >= 1e-12 else np.zeros_like(window_means)

    med = float(np.median(errors))
    mad = float(np.median(np.abs(errors - med))) * 1.4826
    z_rob = (window_means - med) / mad if mad >= 1e-12 else np.zeros_like(window_means)

    i = int(np.argmax(z_plain))
    return {
        "plain": float(z_plain[i]),
        "robust": float(z_rob[int(np.argmax(z_rob))]),
        "argmax_origin": int(origins[i]),
    }


def run_slutsky_sector(sector_id: str, values: list[float], months: list[str] | None,
                       n_synth: int, rng: np.random.Generator) -> dict:
    """單一 sector：真實序列 vs n_synth 條 Slutsky 合成的聚集分數對比。"""
    y_real = _znormalize(np.asarray(values, dtype=np.float64))
    n = len(y_real)

    o_real, e_real = rolling_origin_errors(y_real)
    sc_real = cluster_score(e_real, o_real)
    score_real = sc_real["plain"]
    arg_real = sc_real["argmax_origin"]

    synth_plain, synth_robust = [], []
    for _ in range(n_synth):
        ys = slutsky_series(rng, n)
        o_s, e_s = rolling_origin_errors(ys)
        sc_s = cluster_score(e_s, o_s)
        synth_plain.append(sc_s["plain"])
        synth_robust.append(sc_s["robust"])
    synth = np.asarray(synth_plain)
    synth_rob = np.asarray(synth_robust)

    # 陽性對照（檢測器 sanity check）
    y_pos = positive_control_series(rng, n)
    o_p, e_p = rolling_origin_errors(y_pos)
    sc_pos = cluster_score(e_p, o_p)

    p95 = float(np.percentile(synth, P95_Q))
    p95_rob = float(np.percentile(synth_rob, P95_Q))
    empirical_p = float(np.mean(synth >= score_real))  # 合成聚集率 ≥ 真實 的比例
    specific = score_real > p95

    def _label(origin: int | None) -> str | None:
        if origin is None or months is None or origin >= len(months):
            return None
        return months[origin]

    return {
        "sector": sector_id,
        "n": n,
        "n_origins": len(o_real),
        "horizon": HORIZON,
        "min_train": MIN_TRAIN,
        "cluster_w": CLUSTER_W,
        "score_real": round(score_real, 4),
        "score_real_robust": round(sc_real["robust"], 4),
        "max_window_origin_index": arg_real,
        "max_window_origin_month": _label(arg_real),
        "synth": {
            "n": n_synth,
            "mean": round(float(np.mean(synth)), 4),
            "std": round(float(np.std(synth)), 4),
            "p50": round(float(np.percentile(synth, 50)), 4),
            "p95": round(p95, 4),
            "max": round(float(np.max(synth)), 4),
            "robust_p50": round(float(np.percentile(synth_rob, 50)), 4),
            "robust_p95": round(p95_rob, 4),
        },
        "empirical_p_synth_ge_real": round(empirical_p, 4),
        "real_gt_p95": specific,
        "real_robust_gt_p95": bool(sc_real["robust"] > p95_rob),
        "positive_control": {
            "score_plain": round(sc_pos["plain"], 4),
            "score_robust": round(sc_pos["robust"], 4),
            "max_window_origin_month": _label(sc_pos["argmax_origin"]),
            "plain_gt_p95": bool(sc_pos["plain"] > p95),
            "robust_gt_p95": bool(sc_pos["robust"] > p95_rob),
        },
    }


# ────────────────────────────────────────────────────────────────
# 任務 3：陰性對照
# ────────────────────────────────────────────────────────────────

def negative_controls(task1: dict) -> dict:
    """平穩段不應觸發 rupture 信號（過濾後機械警報計）。

    - track0 r00-r03（用戶指定）：rus/dsr 皆不應有警報落點 t ≤ 3
      （外加全序列 r00-r04 一併報告）。
    - 1915 平穩段 r00-r04：警報落點 t ≤ 4 皆不應存在（標記窗口自 r05 起）。
    - cycle2 僅 2 回合：序列太薄，verdict 應為 CONTESTED 且無警報。
    """
    checks = []

    for fac in FACTIONS:
        alarms = task1["experiment-track0"]["factions"][fac]["alarms_filtered"]
        in_stable = [a for a in alarms if a["t"] <= 3]
        checks.append({
            "control": f"track0 r00-r03 ({fac})",
            "alarms_in_segment": in_stable,
            "pass": len(in_stable) == 0,
            "alarms_full_series": alarms,
        })

    for fac in FACTIONS:
        alarms = task1["anti-intervention-war-1915"]["factions"][fac]["alarms_filtered"]
        in_stable = [a for a in alarms if a["t"] <= 4]
        checks.append({
            "control": f"1915 r00-r04 stable segment ({fac})",
            "alarms_in_segment": in_stable,
            "pass": len(in_stable) == 0,
        })

    for fac in FACTIONS:
        fac_data = task1["anti-intervention-war-cycle2"]["factions"][fac]
        checks.append({
            "control": f"cycle2 thin series ({fac})",
            "verdict": fac_data["verdict_filtered"],
            "alarms": fac_data["alarms_filtered"],
            "pass": len(fac_data["alarms_filtered"]) == 0,
        })

    return {
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks),
        "note": "Slutsky 合成分佈本身即路 B 的統計陰性對照（見 task2）。",
    }


# ────────────────────────────────────────────────────────────────
# 判決組裝
# ────────────────────────────────────────────────────────────────

def assemble_verdicts(task1: dict, marked: dict, task2: dict, task3: dict) -> dict:
    # R1：過濾後已標記窗口全部消失 → 退回設計階段
    mech_vanished = all(
        fac_res["vanished_mechanically"]
        for tree in marked.values() for fac_res in tree.values()
    )
    cand_vanished = all(
        fac_res["vanished_as_candidate"]
        for tree in marked.values() for fac_res in tree.values()
    )
    n_flagged_alarms = sum(
        len(fac_res["mechanical_alarms_in_window"])
        for tree in marked.values() for fac_res in tree.values()
    )
    r1 = {
        "triggered_as_candidates": cand_vanished,
        "triggered_mechanically": mech_vanished,
        "mechanical_alarms_surviving_in_marked_windows": n_flagged_alarms,
        "reading": (
            "候選層（§6.5 語義：旗標窗口降 UNKNOWN、不報）下已標記窗口全部消失"
            if cand_vanished else "存在未旗標的標記窗口警報"
        ) + (
            "；但機械警報仍存留作為 marker（設計本意：機械標記、閘層判讀）"
            if not mech_vanished else ""
        ),
    }

    # R2：已知撕裂時刻（β₁ 1915 開戰＝標記窗口 r05-r07）不觸發 ∧ 陰性亂觸發 → 駁回
    known_triggered = any(
        len(fac_res["mechanical_alarms_in_window"]) > 0
        for tree in marked.values() for fac_res in tree.values()
    )
    negative_fired = not task3["all_pass"]
    r2 = {
        "triggered": (not known_triggered) and negative_fired,
        "known_rupture_triggered": known_triggered,
        "negative_control_fired": negative_fired,
        "note": "兩條件須同時成立才駁回",
    }

    # R3：合成聚集率 ≥ 真實 → 路 B 降為從屬通道（逐 sector 判定；判決以 plain 為主）
    per_sector = {
        sid: {
            "score_real": res["score_real"],
            "score_real_robust": res["score_real_robust"],
            "synth_p95": res["synth"]["p95"],
            "synth_robust_p95": res["synth"]["robust_p95"],
            "empirical_p": res["empirical_p_synth_ge_real"],
            "real_gt_p95": res["real_gt_p95"],
            "real_robust_gt_p95": res["real_robust_gt_p95"],
            "detector_power_plain": res["positive_control"]["plain_gt_p95"],
            "detector_power_robust": res["positive_control"]["robust_gt_p95"],
        }
        for sid, res in task2["sectors"].items()
    }
    all_demote = all(not v["real_gt_p95"] for v in per_sector.values())
    any_specific = any(v["real_gt_p95"] for v in per_sector.values())
    robust_agrees = all(
        (v["real_robust_gt_p95"] == v["real_gt_p95"]) for v in per_sector.values()
    )
    detector_powered = all(
        v["detector_power_plain"] or v["detector_power_robust"]
        for v in per_sector.values()
    )
    r3 = {
        "per_sector": per_sector,
        "path_b_demoted": all_demote,
        "path_b_partial": any_specific and not all_demote,
        "robust_score_agrees": robust_agrees,
        "detector_powered_all_sectors": detector_powered,
        "reading": (
            "全部真實 sector ≤ 合成 p95 → 路 B 單獨不成立，降為從屬通道"
            if all_demote
            else "至少一個 sector > 合成 p95 → 路 B 在該 sector 有特異性"
        ) + ("" if detector_powered else
             "；⚠ 但至少一個 sector 的陽性對照未過 p95——該處陰性結果含檢測器無力成分"),
    }

    # R4：≥2 個獨立窗口 ASSERTED 且全部陰性對照不觸發 → 通過
    unflagged_windows = sum(
        len(fac_res["unflagged_alarms_in_window"])
        for tree in marked.values() for fac_res in tree.values()
    )
    r4 = {
        "triggered": unflagged_windows >= 2 and task3["all_pass"],
        "unflagged_asserted_windows": unflagged_windows,
        "negative_controls_all_pass": task3["all_pass"],
    }

    # 最終判決（主判決 + 子判決；條件間不互斥，按嚴重度陳列）
    if r2["triggered"]:
        final = "駁回"
    elif r1["triggered_as_candidates"]:
        final = "退回設計階段"
        if r3["path_b_demoted"]:
            final += "；路 B 降為從屬通道"
    elif r4["triggered"]:
        final = "通過"
    elif r3["path_b_demoted"]:
        final = "路 B 降為從屬通道"
    else:
        final = "不通過（未命中任何既定條件的完全形——見各項細節）"

    return {
        "R1_return_to_design": r1,
        "R2_theory_rejected": r2,
        "R3_path_b_demotion": r3,
        "R4_pass": r4,
        "final": final,
    }


# ────────────────────────────────────────────────────────────────
# 主程序
# ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="FALSIFY-009 rupture 聯合簽名特異性檢定")
    parser.add_argument("--n-synth", type=int, default=N_SYNTH,
                        help="每 sector 的 Slutsky 合成序列數（預設 100）")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("FALSIFY-009 — rupture 聯合簽名特異性檢定")
    print(f"seed={args.seed}  n_synth={args.n_synth}  horizon={HORIZON}  "
          f"min_train={MIN_TRAIN}  cluster_w={CLUSTER_W}")
    print("=" * 72)

    # ── 任務 1 ──
    print("\n【任務 1】偽影占比正式測量（三棵 warfare 樹）")
    task1: dict[str, dict] = {}
    for tree in TREES:
        res = measure_tree_artifacts(tree)
        task1[tree] = res
        c = res["counts"]
        print(f"\n▸ {tree}")
        print(f"  identical_text={c['identical_text']}  stale 置零={c['stale_zeroed']}"
              f"（{c['stale_share']:.1%} of {c['assignments_total']}）"
              f"  template_switch={list(c['template_switch_points']) or '無'}")
        for fac in FACTIONS:
            f = res["factions"][fac]
            print(f"  {fac}: H 過濾前={f['entropy_unfiltered']}")
            print(f"        H 過濾後={f['entropy_filtered']}")
            print(f"        alerts 前={f['alarms_unfiltered']}")
            print(f"        alerts 後={[(a['from_t'], a['t'], a['drop']) for a in f['alarms_filtered']]}"
                  f"（落點旗標：{[a['target_flags']['flagged'] for a in f['alarms_filtered']]}）")

    marked = evaluate_marked_windows(task1)
    print("\n  已標記窗口（1915 r05-r07）過濾後狀態：")
    for tree, per_fac in marked.items():
        for fac, r in per_fac.items():
            print(f"  {tree}/{fac}: 機械警報={len(r['mechanical_alarms_in_window'])}"
                  f"  未旗標候選={len(r['unflagged_alarms_in_window'])}"
                  f"  候選層消失={r['vanished_as_candidate']}")
    task1_note = (
        "track0 r03「12 平行 alternatives」標記在 dca-branch-tree.yaml 無對應測量"
        "（track0 樹 r03 僅 2 分支、熵與 r01-r04 相同）——該標記來自 campaign 日誌"
        "（mini_rupture_thresholds.yaml:163 的 examples），非樹數據，故機械檢定"
        "只覆蓋 1915 r05-r07。"
    )
    print(f"\n  註：{task1_note}")

    # ── 任務 2 ──
    print("\n【任務 2】Slutsky 零假設檢定（路 B 特異性，核心）")
    sectors = json.loads(SECTORS_JSON.read_text(encoding="utf-8"))
    months_map: dict[str, list[str] | None] = {sid: None for sid in sectors}
    if FEED_REPORT_JSON.exists():
        feed = json.loads(FEED_REPORT_JSON.read_text(encoding="utf-8"))
        for sid, sdata in feed.get("sectors", {}).items():
            months_map[sid] = sdata.get("series", {}).get("months")

    task2_sectors = {}
    for sid in ["ha_tension", "return_mood"]:
        values = sectors[sid]["timeseries"]
        res = run_slutsky_sector(sid, values, months_map.get(sid), args.n_synth, rng)
        task2_sectors[sid] = res
        print(f"\n▸ {sid}（n={res['n']}，origins={res['n_origins']}）")
        print(f"  真實聚集分數 = {res['score_real']}（robust {res['score_real_robust']}）"
              f"（最大窗口始於 index {res['max_window_origin_index']}"
              f" = {res['max_window_origin_month']}）")
        s = res["synth"]
        print(f"  合成（{s['n']} 條）：mean={s['mean']}  p50={s['p50']}  "
              f"p95={s['p95']}（robust p95={s['robust_p95']}）  max={s['max']}")
        print(f"  合成 ≥ 真實 比例（empirical p）= {res['empirical_p_synth_ge_real']}")
        print(f"  判定：真實 {'>' if res['real_gt_p95'] else '≤'} p95 → "
              f"{'有特異性' if res['real_gt_p95'] else '路 B 單獨不成立'}"
              f"（robust 版同判 = {res['real_robust_gt_p95'] == res['real_gt_p95']}）")
        pc = res["positive_control"]
        pc_power = pc["plain_gt_p95"] or pc["robust_gt_p95"]
        print(f"  陽性對照（機制斷裂 sanity check）：plain={pc['score_plain']}"
              f"（{'>' if pc['plain_gt_p95'] else '≤'} p95）"
              f"  robust={pc['score_robust']}（{'>' if pc['robust_gt_p95'] else '≤'} robust p95）"
              f"（{pc['max_window_origin_month']}） → 檢測器{'有' if pc_power else '無'}鑑別力")

    task2 = {"sectors": task2_sectors}

    # ── 任務 3 ──
    print("\n【任務 3】陰性對照")
    task3 = negative_controls(task1)
    for c in task3["checks"]:
        print(f"  {'✓' if c['pass'] else '✗'} {c['control']}"
              + (f"  alarms={c.get('alarms_in_segment', c.get('alarms'))}"
                 if not c["pass"] else ""))
    print(f"  全部陰性對照通過 = {task3['all_pass']}")

    # ── 判決 ──
    verdicts = assemble_verdicts(task1, marked, task2, task3)
    print("\n" + "=" * 72)
    print("【判決】")
    print(f"  R1 退回設計（標記窗口過濾後全消失）：候選層={verdicts['R1_return_to_design']['triggered_as_candidates']}"
          f"  機械層={verdicts['R1_return_to_design']['triggered_mechanically']}")
    print(f"     {verdicts['R1_return_to_design']['reading']}")
    print(f"  R2 理論駁回（已知不觸發∧陰性亂觸發）：{verdicts['R2_theory_rejected']['triggered']}"
          f"（已知觸發={verdicts['R2_theory_rejected']['known_rupture_triggered']}，"
          f"陰性觸發={verdicts['R2_theory_rejected']['negative_control_fired']}）")
    print(f"  R3 路 B 降級（合成聚集率≥真實）：{verdicts['R3_path_b_demotion']['path_b_demoted']}"
          f" — {verdicts['R3_path_b_demotion']['reading']}")
    print(f"  R4 通過（≥2 未旗標窗口 ASSERTED ∧ 陰性全不觸發）：{verdicts['R4_pass']['triggered']}"
          f"（未旗標窗口={verdicts['R4_pass']['unflagged_asserted_windows']}）")
    print(f"\n  ★ 最終判決：{verdicts['final']}")
    print("=" * 72)

    # ── 落盤 ──
    report = {
        "experiment": "FALSIFY-009 — rupture 聯合簽名特異性檢定",
        "generated": datetime.now(timezone.utc).isoformat(),
        "design_refs": [
            "ECC/index/data/research-reports/20260728-rupture-theory-evaluation.md §五",
            "ECC/docs/plans/PLAN-23-spectrum-computer-engine.md §6.5",
        ],
        "seed": args.seed,
        "method": {
            "task1": "現行 loader（identical_text 旗標 + staleness 置零 + template_switch 標記）"
                     " vs 腳本內無旗標對照變體——"
                     "decoherence_watch 熵序列與 alerts 對比；警報落點 state 的 "
                     "provenance 旗標（template_switch / identical_text / stale≥半數 / 全未測）。",
            "task2": f"rolling-origin 回測（wave.extrapolate，horizon={HORIZON}，"
                     f"min_train={MIN_TRAIN}，點預測 RMSE）→ 誤差序列 → 最大窗口 "
                     f"z-score（w={CLUSTER_W}；plain 均值/sd 為主判決，median/MAD robust "
                     "版為敏感性檢查）。零假設：" f"{args.n_synth} 條/sector "
                     f"「N(0,1) 衝擊 + MA(k) 平滑」（k~U{MA_WINDOW_RANGE}，z-正規化，"
                     "長度對齊該 sector）。判定：真實 ≤ 合成 p95 → 路 B 單獨不成立。"
                     f"陽性對照：Slutsky 基底 + 末段（{POS_CTRL_BREAK:.0%} 起）σ×"
                     f"{POS_CTRL_SIGMA_MULT:g} + 水平位移 {POS_CTRL_LEVEL_SHIFT:g}"
                     "——確認檢測器對注入斷裂有鑑別力，否則陰性結果不可解讀。",
            "task3": "track0 r00-r03 平穩段（外加 1915 r00-r04、cycle2 薄序列）"
                     "不應觸發過濾後警報。",
            "caveats": [
                "return_mood 觀測密度僅 0.333（LOCF 高原）——其誤差結構部分來自填充，"
                "非敘事本身；feed 政策本來就把低密度 sector verdict 視為上界樂觀。",
                "ha_tension 在 5.0 飽和（多個月）且 1915-06/1915-11 = 0.0（數據薄）——"
                "研究報告缺口 6。",
                "ha_tension 屬 α₁、return_mood 屬 β₁——兩者各自對自己的零假設分佈"
                "取百分位（類型 III），互不直接比較原始值（比較類型學紅線）。",
            ],
        },
        "task1_artifact_share": {**task1, "marked_windows": marked, "note": task1_note},
        "task2_slutsky": task2,
        "task3_negative_control": task3,
        "verdicts": verdicts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n報告已落盤：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
