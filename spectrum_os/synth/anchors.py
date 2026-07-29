"""synth.anchors — LLM gate producing structured anchors (never pointwise values).

PLAN-23 §7.3 + §六․五 (抗污染憲章) implementation:

- 角色限定 (憲章 #1): the gate is a *validator*, not an RP — anchors are
  parameter extraction, not decisions.  Low temperature (0.0 default).
- 知識截斷 (憲章 #2): the sector_spec carries numbers with sources
  (calibration), not narratives.
- LLM 只產錨點，永不逐點寫序列值 — enforced structurally: the contract
  rejects any pointwise series keys, and ``response_format=json_object``
  keeps the output space small.
- confidence 由機械規則回填 (FIX-21 教訓): whatever self-assessment the LLM
  writes into ``confidence`` is discarded and recomputed from whether each
  anchor carries a source citation.
- 機械契約檢查 (FIX-01/06 教訓): :func:`assert_contract` is the single
  producer/consumer contract — ``anchors()`` validates on production and
  ``synth.expand.expand()`` re-validates on consumption.

API access reuses ECC's ``scripts/api_utils.py:call_api`` (prefix caching,
whitelist, anti-leak).  The API key is read from the ``DEEPSEEK_API_KEY``
environment variable by the caller — never stored, never printed.
"""

import importlib.util
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Contract — the mechanical agreement between anchors() and expand()
# ---------------------------------------------------------------------------

#: Top-level keys every anchors dict must carry.
CONTRACT_TOP_KEYS = (
    "turning_points",     # list of {month_index, level, source}
    "magnitudes",         # {baseline, unit, noise_sigma?, ar_rho?}
    "event_shocks",       # list of {month_index, delta, duration_months, source}
    "confidence",         # mechanically backfilled — LLM content discarded
    "calibration_sources" # non-empty list of str
)

#: Keys that would indicate the LLM tried to write pointwise series values.
#: Their presence is a contract violation — the gate must never do that.
_FORBIDDEN_SERIES_KEYS = ("series", "values", "timeseries", "data", "points")

def _resolve_api_utils_path() -> str:
    """Resolve the path to ECC's api_utils.py, preferring env var."""
    env = os.environ.get("ECC_API_UTILS_PATH", "")
    if env and os.path.isfile(env):
        return env
    # Fallback: sibling monorepo (spectrum-os lives next to ECC)
    sibling = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "ECC", "scripts", "api_utils.py"
    )
    sibling = os.path.abspath(sibling)
    if os.path.isfile(sibling):
        return sibling
    raise FileNotFoundError(
        "Cannot locate ECC's api_utils.py. "
        "Set ECC_API_UTILS_PATH to the absolute path of "
        "ECC/scripts/api_utils.py, or install spectrum-os "
        "as a sibling of the ECC monorepo."
    )

_ECC_API_UTILS_PATH = _resolve_api_utils_path()
_DEFAULT_MODEL = "deepseek-v4-flash"


class AnchorContractError(ValueError):
    """Raised when an anchors dict violates the producer/consumer contract."""


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def assert_contract(anchors: dict, n_months: int | None = None) -> None:
    """Mechanical contract check between ``anchors()`` and ``expand()``.

    Raises :class:`AnchorContractError` with a precise message on the first
    violation found.  ``n_months`` (when given) bounds ``month_index``.
    """
    if not isinstance(anchors, dict):
        raise AnchorContractError(
            f"anchors must be a dict, got {type(anchors).__name__}")

    # ---- forbidden pointwise series (gate must never write series values) --
    for key in _FORBIDDEN_SERIES_KEYS:
        if key in anchors:
            raise AnchorContractError(
                f"forbidden key '{key}': the anchors gate must never write "
                f"pointwise series values (PLAN-23 §7.3)")

    # ---- required top-level keys -------------------------------------------
    missing = [k for k in CONTRACT_TOP_KEYS if k not in anchors]
    if missing:
        raise AnchorContractError(f"missing required keys: {missing}")

    # ---- turning_points ------------------------------------------------------
    tps = anchors["turning_points"]
    if not isinstance(tps, list):
        raise AnchorContractError("turning_points must be a list")
    seen_months: set[int] = set()
    for i, tp in enumerate(tps):
        if not isinstance(tp, dict):
            raise AnchorContractError(f"turning_points[{i}] must be a dict")
        m = tp.get("month_index")
        if not isinstance(m, int) or isinstance(m, bool) or m < 0:
            raise AnchorContractError(
                f"turning_points[{i}].month_index must be an int >= 0, got {m!r}")
        if n_months is not None and m >= n_months:
            raise AnchorContractError(
                f"turning_points[{i}].month_index {m} out of range "
                f"[0, {n_months})")
        if m in seen_months:
            raise AnchorContractError(
                f"turning_points[{i}].month_index {m} duplicated")
        seen_months.add(m)
        if not _is_number(tp.get("level")):
            raise AnchorContractError(
                f"turning_points[{i}].level must be numeric, "
                f"got {tp.get('level')!r}")

    # ---- magnitudes ----------------------------------------------------------
    mags = anchors["magnitudes"]
    if not isinstance(mags, dict):
        raise AnchorContractError("magnitudes must be a dict")
    if not _is_number(mags.get("baseline")) or mags["baseline"] <= 0:
        raise AnchorContractError(
            f"magnitudes.baseline must be a positive number, "
            f"got {mags.get('baseline')!r}")
    if not isinstance(mags.get("unit"), str) or not mags["unit"].strip():
        raise AnchorContractError("magnitudes.unit must be a non-empty string")
    if "noise_sigma" in mags and (
            not _is_number(mags["noise_sigma"]) or mags["noise_sigma"] < 0):
        raise AnchorContractError(
            f"magnitudes.noise_sigma must be a number >= 0, "
            f"got {mags['noise_sigma']!r}")
    if "ar_rho" in mags and (
            not _is_number(mags["ar_rho"]) or not (-1.0 < mags["ar_rho"] < 1.0)):
        raise AnchorContractError(
            f"magnitudes.ar_rho must be in (-1, 1), got {mags['ar_rho']!r}")

    # ---- event_shocks ----------------------------------------------------------
    shocks = anchors["event_shocks"]
    if not isinstance(shocks, list):
        raise AnchorContractError("event_shocks must be a list")
    for i, sh in enumerate(shocks):
        if not isinstance(sh, dict):
            raise AnchorContractError(f"event_shocks[{i}] must be a dict")
        m = sh.get("month_index")
        if not isinstance(m, int) or isinstance(m, bool) or m < 0:
            raise AnchorContractError(
                f"event_shocks[{i}].month_index must be an int >= 0, got {m!r}")
        if n_months is not None and m >= n_months:
            raise AnchorContractError(
                f"event_shocks[{i}].month_index {m} out of range "
                f"[0, {n_months})")
        if not _is_number(sh.get("delta")):
            raise AnchorContractError(
                f"event_shocks[{i}].delta must be numeric, "
                f"got {sh.get('delta')!r}")
        dur = sh.get("duration_months")
        if not isinstance(dur, int) or isinstance(dur, bool) or dur < 0:
            raise AnchorContractError(
                f"event_shocks[{i}].duration_months must be an int >= 0 "
                f"(0 = permanent step), got {dur!r}")

    # ---- calibration_sources ---------------------------------------------------
    sources = anchors["calibration_sources"]
    if (not isinstance(sources, list) or not sources
            or not all(isinstance(s, str) and s.strip() for s in sources)):
        raise AnchorContractError(
            "calibration_sources must be a non-empty list of strings — "
            "anchors must cite their calibration sources (憲章 #2)")


# ---------------------------------------------------------------------------
# Mechanical confidence backfill (FIX-21: never trust LLM self-assessment)
# ---------------------------------------------------------------------------

def _mechanical_confidence(parsed: dict) -> dict:
    """Recompute confidence from source-citation coverage.

    Every turning point / shock with a non-empty ``source`` string counts as
    sourced.  The fraction drives a three-way basis label.  Whatever the LLM
    wrote into ``confidence`` is discarded.
    """
    items = list(parsed["turning_points"]) + list(parsed["event_shocks"])
    if not items:
        fraction = 0.0
    else:
        sourced = sum(
            1 for it in items
            if isinstance(it.get("source"), str) and it["source"].strip())
        fraction = sourced / len(items)
    if fraction >= 0.8:
        basis = "sourced"
    elif fraction >= 0.4:
        basis = "mixed"
    else:
        basis = "inferred"
    return {
        "sourced_fraction": round(fraction, 3),
        "basis": basis,
        "note": ("mechanically backfilled from source-citation coverage; "
                 "LLM self-assessment discarded (FIX-21)"),
    }


# ---------------------------------------------------------------------------
# Prompt construction — LLM = 活的數學 (PLAN-23 v2.5 §〇.五 向量契約)
# ---------------------------------------------------------------------------

ANCHORS_SYSTEM_PROMPT_V25 = (
    "你是一個活的數學——在 6D 向量上做形式計算，不做世界大事的詮釋。\n"
    "\n"
    "【輸入】每個狀態以 6D 向量傳入：\n"
    "  [D1 D2 D3, D4 D5 D6]\n"
    "  D1–D3（三進制方向判定）：頻譜方向的投影，取值 {−1, 0, +1}\n"
    "    D1 = 意志方向（+1 向心凝聚 / −1 離心分散 / 0 無方向）\n"
    "    D2 = 阻力（+1 高阻力 / −1 低阻力 / 0 無差別）\n"
    "    D3 = 關係（+1 正向耦合 / −1 負向耦合 / 0 無耦合）\n"
    "  D4–D6（多頻譜內容）：頻譜內生傾向的實數值\n"
    "    D4 = 主頻譜軸頻率（ha/zg/sa/gt 等軸的歸一化頻率，0–1）\n"
    "    D5 = 相位偏移（−π 到 +π，表徵該軸與基準的相位差）\n"
    "    D6 = 振幅（歸一化振幅，0–1，表徵該軸的信號強度）\n"
    "\n"
    "【任務——純形式計算，禁止詮釋】\n"
    "1. turning_points：從 D1–D3 的級聯變化中計算方向判定切換點。\n"
    "   每次 D1/D2/D3 任一分量發生符號翻轉（+1↔−1 或 +1↔0）即為 turning point。\n"
    "   month_index 取翻轉發生的月份；level 取翻轉後的 D4（歸一化頻率）×baseline。\n"
    "2. magnitudes：從 D4–D6 的振幅/頻率分佈中估算基準水平與噪聲參數。\n"
    "   baseline = D6 振幅 × 校準常數；noise_sigma = baseline 的 1-3%；\n"
    "   ar_rho = D5 相位偏移的自相關推估值（0.2–0.5）。\n"
    "3. event_shocks：從 D1–D3 的離散跳變（符號翻轉）中判定階躍量與持續月數。\n"
    "   delta = 翻轉前後 D6 振幅差（有符號）；duration_months 由 D5 相位的恢復半衰期估算。\n"
    "\n"
    "🔴 禁令（違反者輸出為污染——不可接受）：\n"
    "- 禁止生成敘事描述、歷史解讀、世界大事分析\n"
    "- 禁止解釋「為什麼」——只輸出向量計算結果，不解釋因果\n"
    "- 禁止角色扮演、禁止評論、禁止開場白、禁止 markdown 圍欄\n"
    "- 輸入語言紀律：放入向量結構 → 產出向量計算；放入世界散文 → 產出污染詮釋（禁止）\n"
    "\n"
    "【輸出格式】純結構 JSON——英文鍵名，數值數據，零散文。\n"
    "每個 turning_point 與 event_shock 必須附 source（引用校準來源或事件標籤），\n"
    "但 source 是標籤字串，不是敘事——禁止在 source 欄位寫解釋性文字。"
)

# Legacy system message (pre-v2.5) — kept for backward compatibility.
_SYSTEM_MESSAGE_V24 = (
    "你是一個結構化數據驗證器。唯一任務：把輸入的校準數字與事件清單，"
    "轉換為指定 schema 的錨點 JSON。你只做參數提取，不做敘事、不做評論、"
    "不做角色扮演。你只輸出 JSON——直接輸出，無開場白。"
)

_SCHEMA_RULES = """\
【輸出 schema——機械契約，鍵名不可更改】
{
  "turning_points": [
    {"month_index": <整數, 0 起算>, "level": <數字, 該月起的月度產量水平>,
     "source": "<校準來源字串>"}
  ],
  "magnitudes": {
    "baseline": <數字, 第 0 個 turning point 之前的月度基準水平>,
    "unit": "<單位字串>",
    "noise_sigma": <數字, 月度噪聲標準差, 約為 baseline 的 1-3%>,
    "ar_rho": <數字, AR(1) 噪聲自相關係數, 0.2-0.5>
  },
  "event_shocks": [
    {"month_index": <整數>, "delta": <數字, 正負階躍量>,
     "duration_months": <整數, 0 = 永久性階躍, >0 = 暫時性衝擊持續月數>,
     "source": "<事件來源字串>"}
  ],
  "confidence": {},
  "calibration_sources": ["<你實際引用的校準來源>"]
}

【鐵律】
1. turning_points 定義的是分段常數（階梯函數）——month_index 起的水平
   一直保持到下一個 turning point。禁止任何形式的內插假設。
2. 你只產錨點。禁止輸出逐月序列值（禁止 series/values/timeseries/data/
   points 等鍵）。序列展開由下游 numpy 機械完成。
3. 數量級必須錨定給定的校準數字（α₁ 校準源），每個 turning_point 與
   event_shock 都必須附 source 字串說明依據（校準來源或 β₁ 事件名）。
4. month_index 一律在 [0, __N_MONTHS__) 範圍內。
5. confidence 欄位留空物件 {} ——置信度由機械規則回填，不接受自評。
6. 直接輸出 JSON，無開場白、無解釋、無 markdown 圍欄。
"""


def _build_prompt(sector_spec: dict) -> str:
    """Build the user prompt from a sector spec.

    sector_spec keys: ``name``, ``description``, ``period`` ({start, end,
    n_months}), ``calibration`` (list of {metric, value, unit, source}),
    ``beta1_events`` (list of {month_index, event}).
    """
    period = sector_spec["period"]
    lines: list[str] = []
    lines.append(f"【Sector】{sector_spec['name']}")
    lines.append(f"【說明】{sector_spec['description']}")
    lines.append(
        f"【期間】{period['start']} 至 {period['end']}，"
        f"共 {period['n_months']} 個月（month_index 0 = {period['start']}）")
    lines.append("")
    lines.append("【α₁ 校準數字——只給數字，不給史評】")
    for cal in sector_spec["calibration"]:
        lines.append(
            f"- {cal['metric']}: {cal['value']} {cal['unit']}"
            f"（來源: {cal['source']}）")
    lines.append("")
    lines.append("【β₁ 事件清單——世界線內已發生的事】")
    for ev in sector_spec["beta1_events"]:
        lines.append(f"- month_index {ev['month_index']}"
                     f"（{ev.get('month', '?')}）: {ev['event']}")
    lines.append("")
    if sector_spec.get("constraints"):
        lines.append("【物質邊界】")
        for c in sector_spec["constraints"]:
            lines.append(f"- {c}")
        lines.append("")
    lines.append(_SCHEMA_RULES.replace("__N_MONTHS__", str(period["n_months"])))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ECC api_utils loader (key stays in the environment — never in this repo)
# ---------------------------------------------------------------------------

def _load_ecc_call_api() -> Callable:
    """Import ``call_api`` from ECC's scripts/api_utils.py."""
    if not os.path.exists(_ECC_API_UTILS_PATH):
        raise RuntimeError(
            f"ECC api_utils not found at {_ECC_API_UTILS_PATH} — "
            f"the synth gate requires ECC's shared API module")
    spec = importlib.util.spec_from_file_location(
        "ecc_api_utils", _ECC_API_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.call_api


def _parse_json_loose(raw: str) -> dict:
    """Parse LLM output as JSON, tolerating a markdown fence wrapper."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise AnchorContractError(
            f"LLM output is not valid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise AnchorContractError(
            f"LLM output must be a JSON object, got {type(parsed).__name__}")
    return parsed


# ---------------------------------------------------------------------------
# Public gate
# ---------------------------------------------------------------------------

def anchors(sector_spec: dict, *,
            call_api_fn: Callable | None = None,
            api_key: str | None = None,
            model: str = _DEFAULT_MODEL,
            temperature: float = 0.0,
            max_tokens: int = 4096) -> dict:
    """One Flash call → contract-checked structured anchors.

    Parameters
    ----------
    sector_spec
        Dict with ``name``, ``description``, ``period``, ``calibration``,
        ``beta1_events`` (see :func:`_build_prompt`).
    call_api_fn
        Injection point for tests — a callable with the same signature as
        ECC's ``call_api``.  When None, the real ECC api_utils is loaded.
    api_key
        DeepSeek API key.  When None, read from ``DEEPSEEK_API_KEY`` env var.
    model, temperature, max_tokens
        Passed through to the API call.  Temperature defaults to 0.0
        (validator role — anchors must converge, not diverge).

    Returns
    -------
    dict
        Contract-checked anchors with mechanically backfilled ``confidence``
        and a ``generated_by`` provenance block.
    """
    if call_api_fn is None:
        call_api_fn = _load_ecc_call_api()
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError(
            "API key required: pass api_key or set DEEPSEEK_API_KEY")

    prompt = _build_prompt(sector_spec)
    raw = call_api_fn(
        prompt,
        api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system_message=ANCHORS_SYSTEM_PROMPT_V25,
        response_format={"type": "json_object"},
        thinking=False,
    )

    parsed = _parse_json_loose(raw)
    n_months = sector_spec.get("period", {}).get("n_months")
    assert_contract(parsed, n_months=n_months)

    # Mechanical backfill — LLM self-assessment is discarded (FIX-21).
    parsed["confidence"] = _mechanical_confidence(parsed)
    parsed["generated_by"] = {
        "gate": "synth.anchors",
        "model": model,
        "temperature": temperature,
        "called_at": datetime.now(timezone.utc).isoformat(),
    }
    return parsed
