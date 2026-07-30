#!/usr/bin/env python3
"""ECC → Spectrum OS feed — PLAN-23 §四 待完成 B 項（餵真實資料）。

從 ECC 真實數據（**唯讀**——本腳本不修改 ECC 任何檔案）聚合四個月頻 sector：

| sector        | 世界線 | 數據源 |
|:--------------|:------:|:-------|
| hk_mood       | β₁     | compiled-scenes.json × narrative-spectrum-v3.5.json（arc=香港成長篇） |
| germany_mood  | β₁     | 同上（arc=德意志篇） |
| return_mood   | β₁     | 同上（arc=重返歐洲篇） |
| ha_tension    | α₁     | historical-atmosphere.json time_series[*].tension_level |

⚠ ha_tension 不是普通 sector——依 PLAN-23 §〇․五 與 narrative-spectrum v3.5
`_meta.weber_variables`（independent = ha），ha(t) 是**重力基準（自變量）**，
是敘事 mood（zg，因變量）的比較座標。註冊為 sector 僅為讓 wave.decompose
可作用於它，不代表它與 β₁ mood sector 同類（比較類型學：原始值不可直接比）。

數值欄位依據（SSOT 追蹤）
-------------------------
- scenes-raw frontmatter 的 mood 是**類別字串**（例：
  `index/scenes-raw/香港英國篇-孤兒游戲記錄/scene-000-01.md:13` → `mood: calm`），
  compiled-scenes.json 的 scene dict 同樣只有 mood/mood_description，**無數值欄**
  → 依任務指示 fallback 到 narrative-spectrum-v3.5.json 的 per-scene `mood_valence`
  （同檔另有 coordinates.zg，但 zg 已是 ECC pipeline 的長波平滑產物——
  餵進 decompose 會二次平滑，故取較原始的 mood_valence）。
- ha 取 `tension_level`：historical-atmosphere.json `_meta.note` 明言
  「tension_level 是主要 atmosphere 信號源。其餘欄位為輔助指標。」

月頻聚合與缺口政策（機械規則，非手編；全部記入 feed_report.json）
----------------------------------------------------------------
1. 只收 date_precision ∈ {day, month} 的場景；year/unknown 精度排除
   （寧缺勿濫——不憑空發明日期，排除數計入報告）。
2. 月值 = 當月場景 mood_valence 的算術平均；無場景的月 = 無觀測。
3. kernel 的 moving_average 以前綴和實作，序列內部 NaN 會向後傳播、摧毀整條
   longwave，故缺口採 **forward-fill（LOCF：情緒持續到新證據出現）**。
   這不是 PLAN-23 §7.1 禁止的線性內插（不產生斜坡偽週期），但會產生高原、
   抬高短 lag 自相關——**低密度 sector 的 decompose verdict 應視為上界樂觀**，
   報告附 density（觀測月/網格月）供判讀；UNKNOWN 是合法結果。
4. 填充連跑 > MAX_FILL_RUN（12 月）= 資訊真空，不切齊、不填充：
   改取「觀測段」（相鄰觀測 gap ≤ MAX_FILL_RUN 的最大連段；並列取觀測最多者，
   再並列取最新者）。香港篇孤立點 1900-11 因此被捨棄（其後 73 月真空），
   保留 1907-01..1908-01 段。被捨棄的段記入報告 dropped_segments。
5. decompose 窗口：n ≥ 48 用 kernel 預設 (36, 6)；短序列用
   (max(3, n//2), max(2, long//3))——機械規則，窗口記入報告。

輸出（皆在 spectrum-os 本地 data 層，不進 ECC SSOT）
----------------------------------------------------
- data/sectors/sectors.json     — osc.sectors.save() 原生序列化（四 sector）
- data/sectors/feed_report.json — 來源/覆蓋率/缺口/decompose 摘要（provenance）

重跑：`python feeds/ecc_feeds.py`（需 numpy；ECC/.venv 可用）。
腳本內建自檢（見 self_checks）；快照級斷言見 tests/test_ecc_feeds.py。
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# --- 保證 import 的是本 repo 的 spectrum_os（ECC/.venv 內裝的是 submodule 副本） ---
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from spectrum_os import osc  # noqa: E402
from spectrum_os.kernel._types import Verdict  # noqa: E402
from spectrum_os.sources import SourceSpec, register_source  # noqa: E402

# --- ECC 數據源（唯讀） -------------------------------------------------------
ECC_DATA = Path("/home/octy/projects/ECC/index/data")
COMPILED_SCENES = ECC_DATA / "compiled-scenes.json"
NARRATIVE_SPECTRUM = ECC_DATA / "narrative-spectrum-v3.5.json"
HISTORICAL_ATMOSPHERE = ECC_DATA / "historical-atmosphere.json"

SECTORS_DIR = REPO_ROOT / "data" / "sectors"

# 篇章 → sector id
MOOD_ARCS = {
    "hk_mood": "香港成長篇",
    "germany_mood": "德意志篇",
    "return_mood": "重返歐洲篇",
}

MAX_FILL_RUN = 12          # 月；超過即視為資訊真空，切段
DEFAULT_WINDOWS = (36, 6)  # kernel 預設 (long, mid)


def get_ecc_source_specs() -> list[SourceSpec]:
    """Return declarative SourceSpec objects for all ECC data feeds (PLAN-23 §6.6)."""
    return [
        SourceSpec(
            driver="file",
            locator=str(COMPILED_SCENES),
            emits="Sector",
            mapping={"values": "values"},
            meta={"arc": MOOD_ARCS["hk_mood"]},
            source_id="hk_mood",
        ),
        SourceSpec(
            driver="file",
            locator=str(COMPILED_SCENES),
            emits="Sector",
            mapping={"values": "values"},
            meta={"arc": MOOD_ARCS["germany_mood"]},
            source_id="germany_mood",
        ),
        SourceSpec(
            driver="file",
            locator=str(COMPILED_SCENES),
            emits="Sector",
            mapping={"values": "values"},
            meta={"arc": MOOD_ARCS["return_mood"]},
            source_id="return_mood",
        ),
        SourceSpec(
            driver="file",
            locator=str(HISTORICAL_ATMOSPHERE),
            emits="Sector",
            mapping={"values": "values"},
            meta={"kind": "historical_tension_gravity_baseline"},
            source_id="ha_tension",
        ),
        SourceSpec(
            driver="file",
            locator=str(Path("/home/octy/projects/ECC/index/warfare/data/fields/experiment-track0/dca-branch-tree.yaml")),
            emits="RoleTrajectory",
            mapping={"points": "points", "thread_id": "thread_id"},
            source_id="ecc-warfare-track0",
        ),
        SourceSpec(
            driver="file",
            locator=str(ECC_DATA / "knowledge-dca-edges.yaml"),
            emits="DCASubstrate",
            mapping={"nodes": "nodes", "edges": "edges"},
            bias_flags=[
                {
                    "kind": "under_measurement",
                    "scope": {"roles": ["lag"]},
                    "note": "lag role is under-measured in knowledge edges",
                },
                {
                    "kind": "orphan_region",
                    "scope": {"nodes": ["210-political-economy-*"]},
                    "note": "210-political-economy-* orphan nodes region",
                },
            ],
            source_id="ecc-knowledge-edges",
        ),
        SourceSpec(
            driver="file",
            locator=str(ECC_DATA / "clad-gene-pool.jsonl"),
            emits="CLADCorpus",
            mapping={"entries": "entries"},
            source_id="ecc-clad-books",
        ),
    ]


def register_ecc_sources() -> list[str]:
    """Register all ECC SourceSpec declarations into spectrum_os.sources registry."""
    ids = []
    for spec in get_ecc_source_specs():
        ids.append(register_source(spec))
    return ids


# Automatically register ECC sources on module import
register_ecc_sources()


# ---------------------------------------------------------------------------
# 月份工具
# ---------------------------------------------------------------------------

def _month_index(month: str) -> int:
    """`YYYY-MM` → 連續月序（0-based）：1900-01 → 1900*12。"""
    y, m = month.split("-")
    return int(y) * 12 + (int(m) - 1)


def _index_month(idx: int) -> str:
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def _parse_month(date_str: str | None) -> str | None:
    """從 `YYYY-MM-DD` / `YYYY-MM-??` 抽出 `YYYY-MM`；`YYYY-??-??` 等返回 None。"""
    if not date_str:
        return None
    m = re.match(r"^(\d{4})-(\d{2})", date_str)
    return m.group(0) if m else None


# ---------------------------------------------------------------------------
# 數據提取
# ---------------------------------------------------------------------------

def load_scene_mood_observations(arc: str) -> dict:
    """從 compiled-scenes × narrative-spectrum 聚合某篇章的逐月 mood_valence 觀測。

    Returns
    -------
    dict with keys:
      ``observed`` — {"YYYY-MM": {"mean": float, "n": int}}
      ``stats``    — scenes_total / scenes_used / excluded / join_missing / unmapped
    """
    compiled = json.load(open(COMPILED_SCENES, encoding="utf-8"))["scenes"]
    entries = json.load(open(NARRATIVE_SPECTRUM, encoding="utf-8"))["entries"]

    raw: dict[str, list[float]] = {}
    excluded: Counter = Counter()
    scenes_total = 0
    join_missing = 0
    unmapped = 0

    for s in compiled:
        if s.get("arc") != arc:
            continue
        scenes_total += 1
        precision = s.get("date_precision")
        if precision not in ("day", "month"):
            excluded[f"precision_{precision}"] += 1
            continue
        month = _parse_month(s.get("date"))
        if month is None:
            excluded["unparseable_date"] += 1
            continue
        entry = entries.get(s.get("file"))
        if entry is None:
            join_missing += 1
            continue
        mv = entry.get("mood_valence")
        if mv is None:
            unmapped += 1  # narrative-spectrum 未映射的 mood 類別
            continue
        raw.setdefault(month, []).append(float(mv))

    observed = {
        month: {"mean": sum(vals) / len(vals), "n": len(vals)}
        for month, vals in sorted(raw.items())
    }
    return {
        "observed": observed,
        "stats": {
            "arc": arc,
            "scenes_total": scenes_total,
            "scenes_used": sum(o["n"] for o in observed.values()),
            "excluded": dict(excluded),
            "join_missing": join_missing,
            "unmapped_mood_valence": unmapped,
        },
    }


def build_monthly_series(observed: dict[str, dict]) -> dict:
    """把逐月觀測（稀疏）轉成連續月網格序列（forward-fill + 觀測段切割）。

    規則見模組 docstring 第 3/4 條。Returns dict with ``months``, ``values``,
    ``stats``（含 density、longest_fill_run、dropped_segments）。
    """
    months_obs = sorted(observed)
    if not months_obs:
        raise ValueError("no observed months")

    # --- 切段：相鄰觀測的填充連跑 > MAX_FILL_RUN → 資訊真空，斷開 ---
    segments: list[list[str]] = [[months_obs[0]]]
    for prev, cur in zip(months_obs, months_obs[1:]):
        fill_run = (_month_index(cur) - _month_index(prev)) - 1
        if fill_run > MAX_FILL_RUN:
            segments.append([cur])
        else:
            segments[-1].append(cur)

    # 取觀測最多的段；並列取最新者
    best = max(segments, key=lambda seg: (len(seg), _month_index(seg[-1])))
    dropped = [
        {
            "start": seg[0],
            "end": seg[-1],
            "observed_months": len(seg),
            "reason": f"separated from kept segment by a fill run > {MAX_FILL_RUN} months",
        }
        for seg in segments
        if seg is not best
    ]

    # --- 建網格 + forward-fill ---
    start_idx, end_idx = _month_index(best[0]), _month_index(best[-1])
    months = [_index_month(i) for i in range(start_idx, end_idx + 1)]
    values: list[float] = []
    last: float | None = None
    longest_fill_run = 0
    run = 0
    filled = 0
    for month in months:
        if month in observed:
            last = observed[month]["mean"]
            run = 0
        else:
            filled += 1
            run += 1
            longest_fill_run = max(longest_fill_run, run)
        values.append(last)

    grid_n = len(months)
    return {
        "months": months,
        "values": values,
        "stats": {
            "grid_start": months[0],
            "grid_end": months[-1],
            "grid_months": grid_n,
            "observed_months": len(best),
            "filled_months": filled,
            "density": len(best) / grid_n,
            "longest_fill_run": longest_fill_run,
            "dropped_segments": dropped,
        },
    }


def load_ha_tension() -> dict:
    """α₁ 歷史張力（重力基準）——historical-atmosphere.json time_series.tension_level。

    252 個月全覆蓋（1900-01..1920-12），無缺口，不涉及任何填充。
    tension_level==0.0 是真值（該月無事件），不是缺失。
    """
    ts = json.load(open(HISTORICAL_ATMOSPHERE, encoding="utf-8"))["time_series"]
    months = [m["date"] for m in ts]
    values = [float(m["tension_level"]) for m in ts]
    return {
        "months": months,
        "values": values,
        "stats": {
            "grid_start": months[0],
            "grid_end": months[-1],
            "grid_months": len(months),
            "observed_months": len(months),
            "filled_months": 0,
            "density": 1.0,
            "longest_fill_run": 0,
            "zero_months": sum(1 for v in values if v == 0.0),
            "dropped_segments": [],
        },
    }


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------

def adaptive_windows(n: int) -> tuple[int, int]:
    """機械窗口規則：n ≥ 48 用 kernel 預設 (36, 6)；短序列按比例縮。"""
    if n >= 48:
        return DEFAULT_WINDOWS
    long_w = max(3, n // 2)
    mid_w = max(2, long_w // 3)
    return long_w, mid_w


def _wave_summary(arr) -> dict:
    vals = [float(v) for v in arr if not math.isnan(v)]
    if not vals:
        return {"valid": 0}
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return {
        "valid": len(vals),
        "mean": round(mean, 4),
        "std": round(math.sqrt(var), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def decompose_sector(sector_id: str) -> dict:
    """對已註冊的 sector 跑 osc.wave.decompose，回傳可 JSON 化的摘要。"""
    sec = osc.sectors.get(sector_id)
    long_w, mid_w = adaptive_windows(len(sec.timeseries))
    res = osc.wave.decompose(sector_id, long_window=long_w, mid_window=mid_w)
    return {
        "long_window": long_w,
        "mid_window": mid_w,
        "verdict": res.verdict.value,
        "dominant_periods": [int(p) for p in res.dominant_periods],
        "valid_range": [int(res.valid_range[0]), int(res.valid_range[1])],
        "confidence_reason": res.confidence_reason,
        "waves": {
            "longwave": _wave_summary(res.values["longwave"]),
            "midwave": _wave_summary(res.values["midwave"]),
            "shortwave": _wave_summary(res.values["shortwave"]),
        },
        "deviation": None,  # 無 targets（β₁ 計畫目標不可機械推導——PLAN-23 §7.3 synth 閘層待做）
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_all() -> dict:
    """聚合四個 sector 的月頻序列（純函數，不註冊、不寫檔）。"""
    out = {}
    for sector_id, arc in MOOD_ARCS.items():
        ext = load_scene_mood_observations(arc)
        series = build_monthly_series(ext["observed"])
        out[sector_id] = {
            "worldline": "β₁",
            "kind": "narrative_mood",
            "source": {
                "structure": str(COMPILED_SCENES),
                "values": str(NARRATIVE_SPECTRUM),
                "field": "mood_valence (per-scene; compiled-scenes.json 僅有類別 mood)",
                "filter": f"arc == {arc!r}, date_precision ∈ {{day, month}}",
            },
            "extraction": ext["stats"],
            "monthly_observations": ext["observed"],
            **series,
        }
    ha = load_ha_tension()
    out["ha_tension"] = {
        "worldline": "α₁",
        "kind": "historical_tension_gravity_baseline",
        "source": {
            "values": str(HISTORICAL_ATMOSPHERE),
            "field": "time_series[*].tension_level（_meta.note: 主要 atmosphere 信號源）",
            "note": "ha(t) 是重力基準/自變量（weber IV），非普通 sector",
        },
        "extraction": None,
        **ha,
    }
    return out


def self_checks(built: dict) -> list[str]:
    """結構性自檢（不依賴具體數字快照——快照斷言在 tests/）。"""
    notes = []
    for sid, b in built.items():
        months, values = b["months"], b["values"]
        assert len(months) == len(values), f"{sid}: months/values 長度不一致"
        assert all(isinstance(v, float) and math.isfinite(v) for v in values), (
            f"{sid}: 序列含 NaN/inf（ffill 後不應存在）"
        )
        idx = [_month_index(m) for m in months]
        assert all(b_ - a_ == 1 for a_, b_ in zip(idx, idx[1:])), f"{sid}: 月網格不連續"
        obs = b.get("monthly_observations")
        if obs:
            # 網格端點必須是觀測月，且觀測值必須精確落在對應網格點上
            assert months[0] in obs and months[-1] in obs, f"{sid}: 網格端點非觀測月"
            for m, o in obs.items():
                if months[0] <= m <= months[-1]:
                    gi = _month_index(m) - _month_index(months[0])
                    assert abs(values[gi] - o["mean"]) < 1e-9, f"{sid}: 觀測月 {m} 未對齊網格"
        st = b["stats"]
        assert st["observed_months"] + st["filled_months"] == st["grid_months"], (
            f"{sid}: observed+filled != grid"
        )
        assert 0 < st["density"] <= 1.0, f"{sid}: density 異常"
        if sid == "ha_tension":
            assert len(values) == 252 and months[0] == "1900-01" and months[-1] == "1920-12", (
                "ha_tension: 應為 1900-01..1920-12 共 252 月"
            )
            assert st["filled_months"] == 0, "ha_tension: 不應有填充月"
        else:
            assert all(-1.0 <= v <= 1.0 for v in values), f"{sid}: mood_valence 超出 [-1,1] 常識界"
        notes.append(f"{sid}: OK (n={len(values)}, density={st['density']:.2f})")
    return notes


def register_sectors(built: dict) -> list[str]:
    """用 osc.sectors.create() 註冊（無 targets——β₁ 計畫目標待 synth 閘層）。"""
    ids = []
    for sid, b in built.items():
        sec = osc.sectors.create(sid, timeseries=list(b["values"]))
        ids.append(sec.id)
    return ids


def main() -> int:
    print("=== ECC → Spectrum OS feed（PLAN-23 §四-B）===")
    print(f"來源（唯讀）：{ECC_DATA}")

    built = build_all()

    print("\n--- 自檢 ---")
    for note in self_checks(built):
        print(" ", note)

    ids = register_sectors(built)

    SECTORS_DIR.mkdir(parents=True, exist_ok=True)
    store_path = SECTORS_DIR / "sectors.json"
    osc.sectors.save(str(store_path))

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "plan": "PLAN-23 §四 待完成 B（餵真實資料）",
        "gap_policy": {
            "rule": "forward-fill（LOCF）；填充連跑 > MAX_FILL_RUN=12 月則切段，取觀測最多段",
            "why_not_linear_interpolation": "PLAN-23 §7.1 禁令——線性內插低密度定性數據會產生偽週期",
            "caveat": "ffill 高原會抬高短 lag 自相關；低密度 sector 的 verdict 為上界樂觀，須併讀 density",
        },
        "sectors": {},
    }

    print("\n--- decompose ---")
    header = f"{'sector':<14} {'n':>4} {'density':>7} {'win(L/M)':>8} {'verdict':<10} {'periods':<12} valid_range"
    print(header)
    print("-" * len(header))
    for sid in ids:
        b = built[sid]
        dec = decompose_sector(sid)
        report["sectors"][sid] = {
            "worldline": b["worldline"],
            "kind": b["kind"],
            "source": b["source"],
            "extraction": b["extraction"],
            "series": {
                "months": b["months"],
                "values": [round(v, 6) for v in b["values"]],
                "stats": b["stats"],
            },
            "monthly_observations": b.get("monthly_observations"),
            "decompose": dec,
        }
        print(
            f"{sid:<14} {len(b['values']):>4} {b['stats']['density']:>7.2f} "
            f"{dec['long_window']}/{dec['mid_window']:<6} {dec['verdict']:<10} "
            f"{str(dec['dominant_periods']):<12} {dec['valid_range']}"
        )

    report_path = SECTORS_DIR / "feed_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n已寫出：{store_path}")
    print(f"已寫出：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
