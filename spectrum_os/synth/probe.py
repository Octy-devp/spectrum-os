"""synth.probe — 決策樹探針（約束剛性測量器）。

PLAN-23 §十二（v1.2）實作。決策樹探針 = §11.3 的操作化：

- **機器不輸出「必然」**：不切 hard/soft、不設閾值——只輸出**連續 rigidity_prevalence
  + confidence_band**；「必然/偶然」由人機收束時人詮釋（§12.5 F2 回退）。
- **F1 回退**：不做逐節點文法/速率/週期過濾——只做**整樹剛性統計**
  （分支間離散度）+ 保留 quarantine/回聲/兩軸值/樹結構檢查（§12.4）。
- **無 adversary**：axis_B 是 LLM self-attestation，機械只驗值合法性（§12.7 F7）。
- 每深度層 1 call（D 次）；``[task:tree_generate]`` prompt；N_branch/N_sample/D/w 參數（§12.2）。

本模組**不進 kernel/**——kernel 保持 numpy-only 純潔；機械檢查全用 numpy + 字串。

設計與 §12.2 / 12.3 / 12.5 / 12.6 / 12.7 對齊：

- **Phase A 逐層生成**：第 k 層 call ``gate_fn``（``[task:tree_generate]``）→ 產出本層
  ≤ ``n_branch`` 分支；第 k 層接收處境 + 約束場 + 第 k−1 層已過檢查的分支作為反射對象
  （reflex of reflex = 層間迭代）。輸出：樹狀結構（root + branches + conditions）。
- **Phase B 機械測量/檢查**：§12.4 五項（整樹剛性統計 / quarantine / 回聲 / 兩軸值 / 樹結構）。
  被拒節點 → ``rejected`` 清單（人機收束檢視，不靜默丟棄）。
- **Phase C 連續剛性測量**：樹→路徑枚舉 → 每標籤/每層連續 prevalence（0-1）+ 信賴帶
  （樣本變異）；``prevalence==0`` 或樣本不足 → UNKNOWN（remasking）；不切閾值、不標必然。
- **Phase D 輸出**：約束剛性地圖（逐層 ``{layer, date_ref, rigidity_prevalence[連續],
  confidence_band, tags[]}``）+ 世界原型（低剛性區聚類，Jaccard/共現語義距離聚類，
  🔴 明言非 ``cluster.run``，待語義向量化層 F10）+ 門節點
  （① 剛性低谷 ∧ ② 殘差 sharp 命中 → 標記；③ saturation≥0.99 僅作 heuristics 標註，F8）。
- **Phase E 收束回寫**：``probe_select``——人選定分支 → 樹坍縮；未選分支標 ``unselected``
  （寫回 α₂ 空間，不刪除）；state_log 記錄（仿 ``_write_alt_gate_state_log``，
  ``gate_type: "probe_tree"`` + ``re_calibrate``，§12.7 F6）。

成本：≤ ``n_sample × depth`` calls（Phase A 每層 1 call，§12.8 錨定——不隨節點數漂移，
故**無重試**：單層契約失敗即拋錯）。
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Callable

import numpy as np

from spectrum_os.kernel import verify as _verify
from spectrum_os.quantum.multigraph import ROLES
from spectrum_os.quantum.quarantine import mechanical_filter
from spectrum_os.synth.alt_gate import (
    _extract_situation_labels,
    _FORBIDDEN_PROSE_KEYS,
    _has_shell_refusal_or_placeholder,
)
from spectrum_os.synth.anchors import (
    _FORBIDDEN_SERIES_KEYS,
    _load_ecc_call_api,
    _parse_json_loose,
)
from spectrum_os.synth.gate_prompts import (
    PROMPT_EXAMPLE_LABELS,
    TREE_GENERATE_SYSTEM_PROMPT,
    check_routing_leak_or_schema,
)
from spectrum_os.synth.residual_trigger import SHARP_CRITERIA, load_residual_table

# ---------------------------------------------------------------------------
# 契約常數
# ---------------------------------------------------------------------------

AXIS_A_VALUES: tuple[str, ...] = ("inherited", "emergent")
AXIS_B_VALUES: tuple[str, ...] = ("expression", "substitution")

#: 機器不輸出的鍵——出現即拒（§12.5 F2 回退：必然由人詮釋）。
NECESSITY_HINT_KEY = "necessity_hint"

#: remasking 閾值：路徑數 < 此值視為「樣本不足」→ UNKNOWN（維持叠加交人，§12.5 步驟 4）。
MIN_PATHS_FOR_CONFIDENCE = 2
#: 跨樹樣本數 < 此值時不採樣本變異信賴帶（退 Wald）。
MIN_SAMPLES = 2

MAX_LABEL_LEN = 20
MAX_GROUNDING_LEN = 80
MAX_CONDITION_LEN = 40

#: 分支節點允許鍵（F1：鍵集外即拒——prose 鍵尤其拒收，複用 ``_FORBIDDEN_PROSE_KEYS``）。
#: 核心 8 契約鍵 + F1 回退可選欄位（role/strength/period_months，出現才驗合法性）
#: + 機械層內部標記（``contamination``/``rejected`` 由 ``_mechanical_check_layer``
#: 注入到存活分支，非 LLM 輸出鍵——契約在 Phase B 之後跑，須放行）。
_ALLOWED_BRANCH_KEYS: frozenset[str] = frozenset(
    {
        "label", "grounding", "axis_A", "axis_B", "rigidity_prevalence",
        "confidence_band", "conditions", "parent",
        "role", "strength", "period_months",
        "contamination", "rejected",
    }
)

#: 單樹路徑數上界（路徑爆炸防護，F5）——n_branch≤8、depth≤4 → 最壞 8^4=4096 葉。
#: 超過即截斷並記 meta ``truncated: true``（不無限遞迴）。
MAX_PATHS_PER_TREE = 4096

#: 整樹剛性 blend 權重：rigidity = w*LLM 均值 + (1-w)*機械離散補數（§12.4 #1）。
_RIGIDITY_BLEND_W = 0.5

#: substitution 節點在剛性聚合中的降權權重（§12.4 #4：contamination 降權，不判語義真偽）。
_CONTAMINATION_WEIGHT = 0.5

_Z = 1.96  # 95% 信賴帶


class TreeProbeError(ValueError):
    """Raised when probe_tree output violates contract or mechanical checks."""


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _has_necessity_hint(obj: Any) -> bool:
    """遞迴檢查任何層級是否出現 ``necessity_hint`` 鍵（出現即拒，§12.5）。

    F10 註記：**僅攔欄位、不攔詞**——label 寫「必然開戰」照過（詞層級 quarantine 屬
    可選硬化，v1.2 鐵律下不啟用：機器不切必然，詞義由人機收束時人詮釋）。
    """
    if isinstance(obj, dict):
        if NECESSITY_HINT_KEY in obj:
            return True
        return any(_has_necessity_hint(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_necessity_hint(x) for x in obj)
    return False


def _validate_confidence_band(cb: Any, context: str = "") -> None:
    """驗證 LLM 自報的 confidence_band 合法性（不判語義真偽）。"""
    if not isinstance(cb, dict):
        raise TreeProbeError(f"{context}confidence_band 必須是 dict，got {type(cb).__name__}")
    lower = cb.get("lower")
    upper = cb.get("upper")
    if not _is_number(lower) or not _is_number(upper):
        raise TreeProbeError(f"{context}confidence_band.lower/upper 必須是數值")
    if not (0.0 <= float(lower) <= 1.0 and 0.0 <= float(upper) <= 1.0):
        raise TreeProbeError(f"{context}confidence_band 值必須在 [0, 1]")
    if float(lower) > float(upper):
        raise TreeProbeError(f"{context}confidence_band.lower 不能大於 upper")
    n = cb.get("n")
    if n is not None and (not _is_number(n) or n < 0):
        raise TreeProbeError(f"{context}confidence_band.n 必須是非負數值，got {n!r}")


# ---------------------------------------------------------------------------
# 機械契約：assert_tree_gate_contract
# ---------------------------------------------------------------------------

def assert_tree_gate_contract(
    output_data: dict,
    situation_labels: list[str] | None = None,
    *,
    max_branches: int | None = None,
    max_depth: int | None = None,
) -> None:
    """機械契約檢查（仿 ``assert_alt_gate_contract``，PLAN-23 §12.3）。

    檢查：樹結構（層 ≤ max_branches、深度 ≤ max_depth、無循環、條件非空）、
    兩軸值合法性、回聲/空殼/placeholder 拒收、**無 necessity_hint**（出現即拒）、
    **分支鍵集外即拒**（F1：只允許 ``_ALLOWED_BRANCH_KEYS``，prose 鍵
    narrative/description/explanation/text/summary/prose 尤其拒收，複用
    ``_FORBIDDEN_PROSE_KEYS``）。

    ⚠️ 與 alt_gate 的差異：echo/quarantine 在此處為**結構性**契約檢查
    （給 ``situation_labels`` 時回聲即拒）；``probe_tree`` 內部改以逐節點
    ``rejected`` 清單**非致命**處理（一層一個壞分支不該浪費整層 call，
    見 ``_mechanical_check_layer``）——所以 probe_tree 呼叫本契約時不傳
    ``situation_labels``，由 Phase B 逐節點處理。
    """
    if not isinstance(output_data, dict):
        raise TreeProbeError(f"output must be dict, got {type(output_data).__name__}")

    for key in _FORBIDDEN_SERIES_KEYS:
        if key in output_data:
            raise TreeProbeError(
                f"forbidden key '{key}': probe gate must never output pointwise series values"
            )

    if _has_necessity_hint(output_data):
        raise TreeProbeError(
            f"necessity_hint detected: 機器不輸出「必然」——'{NECESSITY_HINT_KEY}' 出現即拒（§12.5 F2 回退）"
        )

    if _has_shell_refusal_or_placeholder(output_data):
        raise TreeProbeError("output contains forbidden placeholder or shell refusal string")

    required_keys = ("layer", "branches")
    missing = [k for k in required_keys if k not in output_data]
    if missing:
        raise TreeProbeError(f"missing required keys: {missing}")

    layer = output_data["layer"]
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 1:
        raise TreeProbeError(f"layer must be int >= 1, got {layer!r}")
    if max_depth is not None and layer > max_depth:
        raise TreeProbeError(f"layer {layer} exceeds max_depth {max_depth}")

    date_ref = output_data.get("date_ref")
    if date_ref is not None and not isinstance(date_ref, str):
        raise TreeProbeError(f"date_ref must be str, got {type(date_ref).__name__}")

    branches = output_data["branches"]
    if not isinstance(branches, list):
        raise TreeProbeError(f"branches must be a list, got {type(branches).__name__}")
    if max_branches is not None and len(branches) > max_branches:
        raise TreeProbeError(
            f"branches count {len(branches)} exceeds max_branches {max_branches}"
        )
    if not branches:
        raise TreeProbeError("branches 不能為空——每層至少 1 分支")

    situation_set: set[str] | None = None
    if situation_labels:
        situation_set = {
            s.strip() for s in situation_labels if isinstance(s, str) and s.strip()
        }

    seen_labels: set[str] = set()
    for i, b in enumerate(branches):
        if not isinstance(b, dict):
            raise TreeProbeError(f"branch[{i}] must be dict, got {type(b).__name__}")

        # F1：鍵集外即拒——prose 鍵明確拒收（複用 alt_gate 的 _FORBIDDEN_PROSE_KEYS），
        # 其餘未知鍵也拒（分支只允許 _ALLOWED_BRANCH_KEYS）。
        for pkey in _FORBIDDEN_PROSE_KEYS:
            if pkey in b:
                raise TreeProbeError(
                    f"branch[{i}] 含禁制 prose 鍵 '{pkey}'（鍵集外即拒，F1）"
                )
        extra = set(b.keys()) - _ALLOWED_BRANCH_KEYS
        if extra:
            raise TreeProbeError(
                f"branch[{i}] 含未允許鍵 {sorted(extra)}——鍵集外即拒"
                f"（允許 {sorted(_ALLOWED_BRANCH_KEYS)}，prose 鍵尤其拒收）"
            )

        req = ("label", "grounding", "axis_A", "axis_B", "rigidity_prevalence", "conditions")
        missing_b = [k for k in req if k not in b]
        if missing_b:
            raise TreeProbeError(f"branch[{i}] missing required keys: {missing_b}")

        label = b["label"]
        if not isinstance(label, str) or not label.strip():
            raise TreeProbeError(f"branch[{i}].label 必須是非空 str，got {label!r}")
        label = label.strip()
        if len(label) > MAX_LABEL_LEN:
            raise TreeProbeError(f"branch[{i}].label 超過長度上限 (<= {MAX_LABEL_LEN}): {label!r}")
        if label in PROMPT_EXAMPLE_LABELS:
            raise TreeProbeError(
                f"example echo: branch[{i}].label 重複 prompt 範例標籤 '{label}'"
            )
        if situation_set is not None and label in situation_set:
            raise TreeProbeError(
                f"situation echo: branch[{i}].label '{label}' 重複處境標籤"
            )
        if label in seen_labels:
            raise TreeProbeError(
                f"duplicate label within layer: branch[{i}].label '{label}' 重複（破壞 parent 匹配/路徑語義）"
            )
        seen_labels.add(label)
        clean, matches = mechanical_filter(label)
        if not clean:
            raise TreeProbeError(f"branch[{i}].label 未過 mechanical filter: {matches}")

        grounding = b["grounding"]
        if not isinstance(grounding, str) or not grounding.strip():
            raise TreeProbeError(f"branch[{i}].grounding 必須是非空 str")
        if len(grounding) > MAX_GROUNDING_LEN:
            raise TreeProbeError(f"branch[{i}].grounding 超過長度上限 (<= {MAX_GROUNDING_LEN})")
        clean, matches = mechanical_filter(grounding)
        if not clean:
            raise TreeProbeError(f"branch[{i}].grounding 未過 mechanical filter: {matches}")

        axis_a = b["axis_A"]
        if axis_a not in AXIS_A_VALUES:
            raise TreeProbeError(f"branch[{i}].axis_A 必須是 {AXIS_A_VALUES}，got {axis_a!r}")
        axis_b = b["axis_B"]
        if axis_b not in AXIS_B_VALUES:
            raise TreeProbeError(f"branch[{i}].axis_B 必須是 {AXIS_B_VALUES}，got {axis_b!r}")

        rp = b["rigidity_prevalence"]
        if not _is_number(rp) or not (0.0 <= float(rp) <= 1.0):
            raise TreeProbeError(
                f"branch[{i}].rigidity_prevalence 必須是 [0,1] 連續值，got {rp!r}"
            )

        if "confidence_band" in b:
            _validate_confidence_band(b["confidence_band"], f"branch[{i}].")

        conditions = b["conditions"]
        if not isinstance(conditions, list) or not conditions:
            raise TreeProbeError(f"branch[{i}].conditions 必須是非空 list（邊條件非空）")
        for j, cond in enumerate(conditions):
            if not isinstance(cond, str) or not cond.strip():
                raise TreeProbeError(f"branch[{i}].conditions[{j}] 必須是非空 str")
            clean, matches = mechanical_filter(cond)
            if not clean:
                raise TreeProbeError(
                    f"branch[{i}].conditions[{j}] 未過 mechanical filter: {matches}"
                )

        # 可選父欄位：指向上一層分支 label
        if "parent" in b and not isinstance(b["parent"], str):
            raise TreeProbeError(
                f"branch[{i}].parent 必須是 str，got {type(b['parent']).__name__}"
            )

        # F1 回退：role/strength/period_months 非強制欄位——出現才驗合法性
        if "role" in b and b["role"] not in ROLES:
            raise TreeProbeError(f"branch[{i}].role 非法: {b['role']!r}（允許 {ROLES}）")
        if "strength" in b and (
            not _is_number(b["strength"]) or not (0.0 <= float(b["strength"]) <= 1.0)
        ):
            raise TreeProbeError(f"branch[{i}].strength 必須是 [0,1]，got {b['strength']!r}")
        if "period_months" in b and (
            not _is_number(b["period_months"]) or float(b["period_months"]) < 0
        ):
            raise TreeProbeError(
                f"branch[{i}].period_months 必須是非負數值，got {b['period_months']!r}"
            )


# ---------------------------------------------------------------------------
# 機械測量：整樹剛性統計（§12.4 #1，F1 回退——非逐節點拒絕）
# ---------------------------------------------------------------------------

def _char_ngrams(s: str) -> set[str]:
    """字元 unigram + bigram 集合（處理單字標籤時比純 bigram 更穩健）。"""
    s = s.strip()
    if not s:
        return set()
    ngrams: set[str] = set()
    for i in range(len(s)):
        ngrams.add(s[i : i + 1])
    for i in range(len(s) - 1):
        ngrams.add(s[i : i + 2])
    return ngrams


def _jaccard_sim(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _label_dispersion(labels: list[str]) -> float:
    """分支標籤間離散度（0-1）：0 = 同質（剛性高），1 = 異質（選擇多）。

    以字元 n-gram Jaccard 距離的成對均值計——純字串機械測量，零 LLM。
    """
    if not labels:
        return 0.0
    bigrams = [_char_ngrams(l) for l in labels]
    if len(bigrams) < 2:
        return 0.0
    dists: list[float] = []
    for a, b in combinations(bigrams, 2):
        union = a | b
        if not union:
            dists.append(0.0)
        else:
            dists.append(1.0 - len(a & b) / len(union))
    return float(np.mean(dists))


def _role_dispersion(branches: list[dict]) -> float | None:
    """分支間 role 離散度（歸一化熵，0-1）；無 role 欄位時回 None（F1 可選）。"""
    roles = [b.get("role") for b in branches if b.get("role")]
    if len(roles) < 2:
        return None
    counts = [roles.count(r) for r in ROLES]
    total = sum(counts)
    if total == 0:
        return None
    probs = np.array([c / total for c in counts if c > 0], dtype=np.float64)
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_ent = math.log2(float(len([c for c in counts if c > 0]))) if any(counts) else 1.0
    return entropy / max_ent if max_ent > 0 else 0.0


def _period_dispersion(branches: list[dict]) -> float | None:
    """分支間 period_months 離散度（歸一化變異係數，0-1）；無欄位時回 None（F1 可選）。"""
    periods = [
        float(b["period_months"])
        for b in branches
        if b.get("period_months") is not None
    ]
    if len(periods) < 2:
        return None
    arr = np.asarray(periods, dtype=np.float64)
    mean = float(arr.mean())
    if mean == 0.0:
        return 0.0
    cv = float(arr.std(ddof=1)) / mean
    return float(np.clip(cv, 0.0, 1.0))


def _layer_rigidity_components(branches: list[dict]) -> dict:
    """整樹剛性統計（§12.4 #1）：分支間離散度 → 納入 rigidity 分佈。

    rigidity = blend_w * LLM 均值 + (1 - blend_w) * 機械離散補數。
    substitution 節點以 ``_CONTAMINATION_WEIGHT`` 降權（§12.4 #4，F7——不判語義真偽）。

    F2 判定：**label 語義離散為主代理、role/period 為輔助顯示**——role_dispersion /
    period_dispersion（F1 可選欄位）只進 components，刻意不混入 blend（防過度工程、
    保持機械確定性）。速率離散度不在分支節點（rate_matrix 存於 constraint_field，
    非節點欄位）故不納入本函數測量。
    """
    labels = [b["label"].strip() for b in branches]
    dispersion = _label_dispersion(labels)

    rigidities = [float(b.get("rigidity_prevalence", 0.5)) for b in branches]
    weights = [
        _CONTAMINATION_WEIGHT if b.get("contamination") else 1.0 for b in branches
    ]
    if weights:
        llm_mean = float(np.average(rigidities, weights=weights))
    else:
        llm_mean = 0.5

    mechanical_rigidity = 1.0 - dispersion
    rigidity = float(
        np.clip(
            _RIGIDITY_BLEND_W * llm_mean + (1.0 - _RIGIDITY_BLEND_W) * mechanical_rigidity,
            0.0,
            1.0,
        )
    )

    return {
        "rigidity_prevalence": round(rigidity, 4),
        "components": {
            "llm_mean_rigidity": round(llm_mean, 4),
            "label_dispersion": round(dispersion, 4),
            "mechanical_rigidity": round(mechanical_rigidity, 4),
            "role_dispersion": _role_dispersion(branches),
            "period_dispersion": _period_dispersion(branches),
            "blend_w": _RIGIDITY_BLEND_W,
            "contamination_weight": _CONTAMINATION_WEIGHT,
        },
    }


# ---------------------------------------------------------------------------
# Phase B：逐節點機械檢查（quarantine / 回聲 / 兩軸降權）
# ---------------------------------------------------------------------------

def _mechanical_check_layer(
    layer_entry: dict,
    situation_labels: list[str] | None,
) -> tuple[list[dict], list[dict]]:
    """對一層分支做逐節點機械檢查：quarantine L1 / 回聲 / 空殼 / 兩軸 substitution 降權。

    回傳 ``(passed, rejected)``——rejected 節點進 rejected 清單供人機收束檢視，
    不靜默丟棄（§12.4）。被拒分支不作為下一層的反射對象。
    """
    passed: list[dict] = []
    rejected: list[dict] = []
    situation_set: set[str] = set()
    if situation_labels:
        situation_set = {
            s.strip() for s in situation_labels if isinstance(s, str) and s.strip()
        }

    for b in layer_entry.get("branches", []):
        reasons: list[str] = []
        label = str(b.get("label", "")).strip()

        # 空殼 / placeholder（None / ? / <...> 等）→ rejected（非致命，供人檢視）
        if _has_shell_refusal_or_placeholder(b):
            reasons.append("shell/placeholder: 分支含空殼或佔位字串")

        clean, matches = mechanical_filter(label)
        if not clean:
            reasons.append(f"quarantine: label 含時空黑名單詞 {matches}")
        g = str(b.get("grounding", "")).strip()
        gclean, gmatches = mechanical_filter(g)
        if not gclean:
            reasons.append(f"quarantine: grounding 含時空黑名單詞 {gmatches}")
        for cond in b.get("conditions", []):
            cclean, cmatches = mechanical_filter(str(cond))
            if not cclean:
                reasons.append(f"quarantine: condition 含時空黑名單詞 {cmatches}")
                break

        if label in PROMPT_EXAMPLE_LABELS:
            reasons.append("echo: label 重複 prompt 範例標籤")
        if label in situation_set:
            reasons.append("echo: label 重複處境標籤")

        # 兩軸 substitution → contamination 降權（F7：不驗語義真偽，機械只標記）
        if b.get("axis_B") == "substitution":
            b["contamination"] = True
        else:
            b["contamination"] = False

        if reasons:
            b["rejected"] = True
            b["reject_reasons"] = reasons
            rejected.append(b)
        else:
            b["rejected"] = False
            passed.append(b)
    return passed, rejected


# ---------------------------------------------------------------------------
# Phase C：連續剛性測量（樹→路徑枚舉 → prevalence + 信賴帶 + remasking）
# ---------------------------------------------------------------------------

def _prevalence_band(
    count: int,
    n: int,
    *,
    cross_tree: list[float] | None = None,
    z: float = _Z,
) -> dict:
    """信賴帶（95%）。

    - ``cross_tree``（n_sample>1）：跨樹樣本變異 → 樣本均數 ± z·SE。
    - 單樹/匯總：以路徑數 n 的 Wald 區間。
    - remasking：``n < MIN_PATHS_FOR_CONFIDENCE`` 或 ``prevalence==0`` → ``UNKNOWN``
      （維持叠加交人，§12.5 步驟 4——只在門上坍縮，不在機械層提前閉合）。
    """
    if n <= 0:
        return {
            "lower": 0.0,
            "upper": 1.0,
            "n": 0,
            "confidence": "UNKNOWN",
            "basis": "no_paths",
        }
    p = count / n
    if cross_tree is not None and len(cross_tree) > 1:
        arr = np.asarray(cross_tree, dtype=np.float64)
        mean_p = float(arr.mean())
        std_p = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        se = std_p / math.sqrt(float(len(arr)))
        lower = max(0.0, mean_p - z * se)
        upper = min(1.0, mean_p + z * se)
        confidence = (
            "UNKNOWN" if (mean_p == 0.0 or len(arr) < MIN_SAMPLES) else "measured"
        )
        return {
            "lower": round(lower, 4),
            "upper": round(upper, 4),
            "n": n,
            "confidence": confidence,
            "basis": "cross_tree_var",
        }
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    lower = max(0.0, p - z * se)
    upper = min(1.0, p + z * se)
    confidence = "UNKNOWN" if (p == 0.0 or n < MIN_PATHS_FOR_CONFIDENCE) else "measured"
    return {
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "n": n,
        "confidence": confidence,
        "basis": "wald_path",
    }


def _enumerate_paths(
    root: dict, *, max_paths: int | None = None
) -> tuple[list[list[dict]], bool]:
    """樹→根到葉路徑枚舉（§12.5 步驟 1）。每條路徑 = 一條完整世界線。

    ``max_paths``：路徑爆炸防護（F5）——超過即截斷遞迴，回傳 ``(paths, truncated)``；
    ``truncated=True`` 時 caller 記入 meta（不無限遞迴）。None = 不限制。
    """
    paths: list[list[dict]] = []
    truncated = False

    def walk(node: dict, path: list[dict]) -> None:
        nonlocal truncated
        if truncated:
            return
        path = path + [node]
        children = node.get("children")
        if not children:
            paths.append(path)
            if max_paths is not None and len(paths) >= max_paths:
                truncated = True
            return
        for child in children:
            walk(child, path)
            if truncated:
                return

    walk(root, [])
    return paths, truncated


def _path_weights_from_paths(layers: list[dict], paths: list[list[dict]]) -> dict:
    """逐層逐分支路徑權重（通過該分支的根到葉路徑數）。

    路徑為可變長度——未抵達某層的路徑不計入該層（與 Phase C 條件 prevalence 一致）。
    """
    weights: dict[int, list[int]] = {}
    for layer_entry in layers:
        layer = layer_entry["layer"]
        labels = [b["label"] for b in layer_entry["branches"]]
        counts = [0] * len(labels)
        for path in paths:
            if len(path) <= layer:
                continue  # 未抵達此層
            node = path[layer]  # root 在 index 0，layer k 在 index k
            for i, lb in enumerate(labels):
                if lb == node["label"]:
                    counts[i] += 1
                    break
        weights[layer] = counts
    return weights


def standing_wave_per_layer(
    layers: list[dict],
    *,
    path_weights: dict[int, list[int]] | None = None,
) -> dict:
    """逐層 standing wave（F3 變體，PLAN-23 §12.9）：按 layer 分組的 prevalence。

    🔴 不是既有 ``standing_wave()`` 的跨步 union / divergence_curve（那些依賴
    role 序列，不可直接吃樹）——本變體以「層」為單位：每層內各標籤的 prevalence
    （可選 path_weights 使權重 = 通過該分支的路徑數），並輸出該層離散度與機械剛性。

    Args:
        layers: ``[{"layer": int, "date_ref": str|None, "branches": [{label,...}, ...]}, ...]``。
        path_weights: 可選，``{layer: [每分支權重]}``；None 時每分支等權。
            （由 ``_path_weights_from_paths`` 產生，與 Phase C prevalence 一致。）

    Returns:
        ``{"per_layer": [{"layer", "date_ref", "nodes"[], "antinodes"[], "dispersion",
        "layer_rigidity"}], "meta": {...}}``——``nodes`` = prevalence==1.0（剛性最高），
        ``antinodes`` = 0<prevalence<1 升冪；不切 hard/soft、不標必然。
    """
    if not layers:
        return {
            "per_layer": [],
            "meta": {"total_layers": 0, "total_unique_tags": 0, "basis": "layer_grouped"},
        }

    out: list[dict] = []
    all_tags: set[str] = set()

    for li, layer_entry in enumerate(layers):
        branches = layer_entry.get("branches", [])
        layer = layer_entry.get("layer", li + 1)
        labels = [
            b["label"].strip()
            for b in branches
            if isinstance(b, dict) and isinstance(b.get("label"), str) and b["label"].strip()
        ]

        if path_weights is not None:
            w = path_weights.get(layer)
            weights = [
                int(round(x)) if isinstance(x, (int, float)) else 1
                for x in (w or [])
            ]
            weights = (weights + [1] * len(labels))[: len(labels)]
        else:
            weights = [1] * len(labels)

        total = float(sum(weights))
        tag_stats: list[dict] = []
        seen: set[str] = set()
        for label, wgt in zip(labels, weights):
            if label in seen:
                continue
            seen.add(label)
            count = sum(ww for (ll, ww) in zip(labels, weights) if ll == label)
            all_tags.add(label)
            tag_stats.append(
                {
                    "label": label,
                    "prevalence": round(count / total, 4) if total else 0.0,
                    "confidence_band": _prevalence_band(int(count), int(total)),
                }
            )

        nodes = sorted(
            [t for t in tag_stats if t["prevalence"] == 1.0], key=lambda t: t["label"]
        )
        contingent = [t for t in tag_stats if 0.0 < t["prevalence"] < 1.0]
        contingent.sort(key=lambda t: (t["prevalence"], t["label"]))
        # remasking：prevalence==0 或樣本不足 → UNKNOWN，維持叠加交人（§12.5 步驟 4）
        remasked = [
            t
            for t in tag_stats
            if t["confidence_band"]["confidence"] == "UNKNOWN"
        ]

        dispersion = _label_dispersion(labels)
        out.append(
            {
                "layer": layer,
                "date_ref": layer_entry.get("date_ref"),
                "nodes": nodes,
                "antinodes": contingent,
                "remasked": remasked,
                "dispersion": round(dispersion, 4),
                "layer_rigidity": round(1.0 - dispersion, 4),
            }
        )

    return {
        "per_layer": out,
        "meta": {
            "total_layers": len(out),
            "total_unique_tags": len(all_tags),
            "basis": "layer_grouped",
        },
    }


# ---------------------------------------------------------------------------
# Phase A：逐層生成（每層 1 call）
# ---------------------------------------------------------------------------

def _branch_ref(b: dict) -> dict:
    """上一層分支的反射摘要（給下一層當反射對象）。"""
    return {
        "label": b.get("label"),
        "axis_A": b.get("axis_A"),
        "axis_B": b.get("axis_B"),
        "rigidity_prevalence": b.get("rigidity_prevalence"),
        "conditions": b.get("conditions", []),
    }


def _build_layer_payload(
    situation: Any,
    constraint_field: Any,
    reflect_on: list[dict],
    *,
    layer: int,
    n_branch: int,
    w: float,
) -> dict:
    """構造第 k 層 user payload。

    🔴 邊界必須在 payload 裡（S4 教訓）：處境與約束場全放 user payload，
    LLM 不得自行「想起」約束。``w`` 只在此生成期 prompt 內作用（約束強度），
    不作機械閾值（§12.5 刪 ε(w)）。
    """
    payload: dict[str, Any] = {
        "task": "tree_generate",
        "layer": layer,
        "n_branch": n_branch,
        "w": w,  # CFG 場強（生成期 only）
        "situation": situation,
    }
    if constraint_field is not None:
        payload["constraint_field"] = constraint_field
    if reflect_on:
        payload["reflection"] = {
            "layer": layer - 1,
            "passed_branches": [_branch_ref(b) for b in reflect_on],
        }
    return payload


def _parse_tree_json(raw: str) -> dict:
    try:
        return _parse_json_loose(raw)
    except Exception as e:  # AnchorContractError → TreeProbeError
        raise TreeProbeError(f"tree_generate 輸出無法解析: {e}") from e


def _resolve_parent(branch: dict, prev_passed: list[dict]) -> dict | None:
    """解析分支的父節點：優先 ``parent`` 欄位比對；缺省 → 主線（LLM 剛性最高者）。

    ``parent`` 指向已拒節點（不在 prev_passed）時退主線。回 None 表示父 = root（第 1 層）。
    """
    parent_label = branch.get("parent")
    if parent_label is not None:
        for pb in prev_passed:
            if pb["label"] == parent_label:
                return pb
    if prev_passed:
        return max(
            prev_passed,
            key=lambda pb: (pb.get("_llm_rigidity", 0.0), -prev_passed.index(pb)),
        )
    return None


def _root_label(situation: Any) -> str:
    """根節點標籤：處境 digest 首行（≤20 字），否則「處境」。"""
    if isinstance(situation, dict):
        digest = situation.get("digest")
        if isinstance(digest, str) and digest.strip():
            first = digest.strip().splitlines()[0].strip()
            return first[:MAX_LABEL_LEN]
    return "處境"


def _generate_one_tree(
    situation: Any,
    constraint_field: Any,
    *,
    n_branch: int,
    depth: int,
    w: float,
    gate_fn: Callable,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    situation_labels: list[str] | None,
) -> dict:
    """生成單棵決策樹（Phase A + B + C）。每層 1 call，共 depth 次。"""
    root = {
        "label": _root_label(situation),
        "grounding": "處境（root）",
        "layer": 0,
        "parent_label": None,
        "axis_A": "inherited",
        "axis_B": "expression",
        "children": [],
    }
    layers: list[dict] = []
    rejected: list[dict] = []
    calls = 0
    reflect_on: list[dict] = []

    for k in range(1, depth + 1):
        payload = _build_layer_payload(
            situation, constraint_field, reflect_on,
            layer=k, n_branch=n_branch, w=w,
        )
        user_prompt = f"{json.dumps(payload, ensure_ascii=False)}\n[task:tree_generate]"
        raw = gate_fn(
            user_prompt,
            api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_message=TREE_GENERATE_SYSTEM_PROMPT,
            response_format={"type": "json_object"},
            thinking=False,
        )
        calls += 1

        parsed = _parse_tree_json(raw)
        # F6：路由洩漏/結構失配只在 check_routing_leak_or_schema 一處處理（內部已含
        # ROUTING_LEAK_KEYWORDS 掃描 + 任務 schema 檢查）——不重複手動迴圈。
        # 注意：洩漏且無法解析的回應會在解析階段先以「無法解析」拒絕（仍是 TreeProbeError）。
        leak_err = check_routing_leak_or_schema(raw, parsed, "tree_generate")
        if leak_err:
            raise TreeProbeError(leak_err)

        # Phase B：先逐節點機械檢查（quarantine/回聲/空殼/兩軸降權）→ passed / rejected
        passed, layer_rejected = _mechanical_check_layer(parsed, situation_labels)
        for b in layer_rejected:
            b["layer"] = k
        rejected.extend(layer_rejected)

        # 結構性契約：只對「存活分支」驗結構（文法/兩軸/placeholder/necessity/長度）。
        # quarantine/echo/空殼 已在上一步逐節點非致命處理——不該在契約層炸掉整層。
        if not passed:
            raise TreeProbeError(
                f"layer {k} 全部分支被機械拒——reflex 無法承載下一層（§12.12 #1：不承載即拒）"
            )
        surviving = dict(parsed)
        surviving["branches"] = list(passed)
        assert_tree_gate_contract(
            surviving, None, max_branches=n_branch, max_depth=depth
        )

        layer_entry = {
            "layer": k,
            "date_ref": parsed.get("date_ref"),
            "branches": [],
        }

        # 建樹：解析父節點連結
        for b in passed:
            b["layer"] = k
            b["_llm_rigidity"] = float(b["rigidity_prevalence"])
            b["_llm_band"] = b.get("confidence_band")
            b["children"] = []
            parent = _resolve_parent(b, reflect_on)
            if parent is None:
                b["parent_label"] = root["label"]
                root["children"].append(b)
            else:
                b["parent_label"] = parent["label"]
                parent["children"].append(b)
            layer_entry["branches"].append(b)

        layers.append(layer_entry)
        reflect_on = passed

    # Phase C：路徑枚舉 → prevalence + 信賴帶
    # 路徑為可變長度（根→葉；無子節點的分支即葉）。每層的分母 = **抵達該層的
    # 路徑數**（條件 prevalence）——與 `standing_wave_per_layer` 的 path 加權一致；
    # 提前結束的世界線不稀釋後續層的選擇分佈。
    paths, paths_truncated = _enumerate_paths(root, max_paths=MAX_PATHS_PER_TREE)
    n_paths = len(paths)
    prevalence: list[dict] = []
    for layer_entry in layers:
        layer = layer_entry["layer"]
        labels = [b["label"] for b in layer_entry["branches"]]
        counts = [0] * len(labels)
        reached = 0
        for path in paths:
            if len(path) <= layer:
                continue  # 路徑未抵達此層（提前為葉）
            reached += 1
            node = path[layer]
            for i, lb in enumerate(labels):
                if lb == node["label"]:
                    counts[i] += 1
                    break
        for label, cnt in zip(labels, counts):
            prevalence.append(
                {
                    "layer": layer,
                    "label": label,
                    "prevalence": round(cnt / reached, 4) if reached else 0.0,
                    "confidence_band": _prevalence_band(cnt, reached),
                    "count": cnt,
                    "n": reached,
                }
            )

    # 每層整樹剛性統計 + 機械信賴帶
    rigidity_map: list[dict] = []
    for layer_entry in layers:
        branches = layer_entry["branches"]
        comps = _layer_rigidity_components(branches)
        # 該層抵達路徑數（reflex 承載的樣本數）
        reached_k = sum(
            1 for p in paths if len(p) > layer_entry["layer"]
        )
        band = _rigidity_band([comps["rigidity_prevalence"]], reached_k)
        rigidity_map.append(
            {
                "layer": layer_entry["layer"],
                "date_ref": layer_entry.get("date_ref"),
                "rigidity_prevalence": comps["rigidity_prevalence"],
                "confidence_band": band,
                "tags": _top_labels(prevalence, layer_entry["layer"], top_k=5),
                "components": comps["components"],
                "n_paths": reached_k,
            }
        )

    # 逐層 standing wave（path 加權）
    standing_wave = standing_wave_per_layer(
        layers, path_weights=_path_weights_from_paths(layers, paths)
    )

    return {
        "root": root,
        "layers": layers,
        "paths": paths,
        "prevalence": prevalence,
        "rigidity_map": rigidity_map,
        "standing_wave": standing_wave,
        "rejected": rejected,
        "meta": {
            "calls": calls,
            "n_paths": n_paths,
            "truncated": paths_truncated,  # F5：路徑爆炸截斷旗標
            "params": {},
        },
    }


def _rigidity_band(rigidities: list[float], n_paths: int) -> dict:
    """整樹剛性信賴帶。

    - 跨樹（>1 棵）：跨樹樣本變異 → 樣本均數 ± z·SE，basis ``rigidity_cross_tree``。
    - 單樹：以該層抵達路徑數 n 的 Wald 區間，basis ``rigidity_wald_path``
      （估的是**路徑枚舉**的抽樣誤差，不是 LLM 生成變異——後者無跨樹樣本時誠實標 UNKNOWN）。
    - remasking：``mean==0`` 或路徑數不足 → ``UNKNOWN``（維持叠加交人，§12.5 步驟 4）。
    """
    arr = np.asarray(rigidities, dtype=np.float64)
    if arr.size == 0:
        return {"lower": 0.0, "upper": 1.0, "n": 0, "confidence": "UNKNOWN", "basis": "no_data"}
    mean = float(arr.mean())
    if arr.size > 1:
        std = float(arr.std(ddof=1))
        se = std / math.sqrt(float(arr.size))
        basis = "rigidity_cross_tree"
    else:
        std = 0.0
        se = math.sqrt(max(mean * (1.0 - mean), 0.0) / max(n_paths, 1))
        basis = "rigidity_wald_path"
    lower = max(0.0, mean - _Z * se)
    upper = min(1.0, mean + _Z * se)
    confidence = (
        "UNKNOWN"
        if (mean == 0.0 or n_paths < MIN_PATHS_FOR_CONFIDENCE)
        else "measured"
    )
    return {
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "n": n_paths,
        "confidence": confidence,
        "basis": basis,
    }


def _top_labels(prevalence: list[dict], layer: int, top_k: int = 5) -> list[str]:
    """該層代表標籤（prevalence 最高前 top_k）。"""
    entries = [e for e in prevalence if e["layer"] == layer]
    entries.sort(key=lambda e: e["prevalence"], reverse=True)
    return [e["label"] for e in entries[:top_k]]


# ---------------------------------------------------------------------------
# 跨樣本聚合（n_sample > 1）
# ---------------------------------------------------------------------------

def _aggregate_prevalence(trees: list[dict]) -> list[dict]:
    """跨樣本聚合每 (layer, label) prevalence + 信賴帶（樣本變異）。"""
    by_key: dict[tuple[int, str], list[dict]] = {}
    for t in trees:
        for entry in t.get("prevalence", []):
            key = (entry["layer"], entry["label"])
            by_key.setdefault(key, []).append(entry)

    out: list[dict] = []
    for (layer, label), entries in by_key.items():
        total_count = sum(int(e["count"]) for e in entries)
        total_n = sum(int(e["n"]) for e in entries)
        prev = total_count / total_n if total_n else 0.0
        cross = [float(e["prevalence"]) for e in entries]
        band = _prevalence_band(
            total_count, total_n, cross_tree=cross if len(cross) > 1 else None
        )
        out.append(
            {
                "layer": layer,
                "label": label,
                "prevalence": round(prev, 4),
                "confidence_band": band,
                "count": total_count,
                "n": total_n,
            }
        )
    return out


def _aggregate_rigidity_map(trees: list[dict], prevalence: list[dict]) -> list[dict]:
    """跨樣本聚合逐層剛性地圖。"""
    by_layer: dict[int, list[dict]] = {}
    for t in trees:
        for entry in t.get("rigidity_map", []):
            by_layer.setdefault(entry["layer"], []).append(entry)

    out: list[dict] = []
    for layer in sorted(by_layer):
        entries = by_layer[layer]
        rigidities = [float(e["rigidity_prevalence"]) for e in entries]
        n_paths_total = sum(int(e["n_paths"]) for e in entries)
        mean_r = float(np.mean(rigidities))
        band = _rigidity_band(
            rigidities if len(rigidities) > 1 else [mean_r], n_paths_total
        )
        components = entries[0].get("components", {})
        out.append(
            {
                "layer": layer,
                "date_ref": entries[0].get("date_ref"),
                "rigidity_prevalence": round(mean_r, 4),
                "confidence_band": band,
                "tags": _top_labels(prevalence, layer, top_k=5),
                "components": components,
                "n_paths": n_paths_total,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Phase D：世界原型 + 門節點
# ---------------------------------------------------------------------------

def _cluster_archetypes(
    paths: list[list[dict]], *, n_target: int = 5, sim_threshold: float = 0.3
) -> list[dict]:
    """世界原型（§12.6 / §12.10 #5）。

    🔴 明確**不是** ``kernel/cluster.run``（其吃 11 維頻譜簽名，不可直接吃樹——F10）。
    此處為語義距離聚類**先行**：以每條路徑的標籤集合 Jaccard 相似度的貪婪凝聚，
    待語義向量化層規格化後替換。低剛性區（選擇空間大的路徑）自然成為不同原型。
    """
    if not paths:
        return []
    label_sets = [
        set(
            n["label"]
            for n in path
            if isinstance(n, dict) and isinstance(n.get("label"), str) and n["label"]
        )
        for path in paths
    ]
    clusters: list[tuple[set[str], list[int]]] = []
    for i, ls in enumerate(label_sets):
        if not ls:
            clusters.append((set(), [i]))
            continue
        best: int | None = None
        best_sim = sim_threshold
        for ci, (cset, _cids) in enumerate(clusters):
            if not cset:
                continue
            sim = _jaccard_sim(ls, cset)
            if sim >= best_sim:
                best_sim = sim
                best = ci
        if best is None:
            clusters.append((set(ls), [i]))
        else:
            clusters[best][0].update(ls)
            clusters[best][1].append(i)

    clusters.sort(key=lambda c: len(c[1]), reverse=True)
    archetypes: list[dict] = []
    for idx, (cset, cids) in enumerate(clusters[:n_target]):
        archetypes.append(
            {
                "archetype_id": idx,
                "labels": sorted(cset),
                "member_path_ids": cids,
                "n_members": len(cids),
                "representative_path_id": cids[0],
                "method": "jaccard_semantic_distance",
                "note": "語義距離聚類先行；待語義向量化層（F10）規格化後替換 kernel/cluster.run",
            }
        )
    return archetypes


def _local_minima(values: list[float]) -> list[int]:
    """剛性低谷（局部極小）索引（§12.6 ①）。端點以單側鄰居判。"""
    idx: list[int] = []
    for i in range(len(values)):
        left = values[i - 1] if i > 0 else None
        right = values[i + 1] if i < len(values) - 1 else None
        if (left is None or values[i] < left) and (right is None or values[i] < right):
            idx.append(i)
    return idx


def _sharp_hit(table: dict, date_ref: Any) -> bool:
    """殘差 sharp 命中（§12.6 ②）：該 date_ref 條目命中 saturation/decoupling。"""
    if not date_ref:
        return False
    entry = table.get(str(date_ref))
    if not isinstance(entry, dict):
        return False
    criteria = entry.get("criteria")
    if not isinstance(criteria, list):
        return False
    return any(c in SHARP_CRITERIA for c in criteria)


def _saturation_value(
    saturation_map: Any,
    table: dict,
    date_ref: Any,
    layer: Any,
) -> float | None:
    """取該層 saturation（③ heuristics 標註用），兼容兩種數據形狀（F4）。

    數據形狀契約：
    - (a) 頂層 map：``constraint_field["saturation"] = {date_ref|layer: 0.99}`` 或純量；
    - (b) 殘差條目內：``constraint_field["residuals"] = {date_ref: {"criteria": [...],
      "saturation": 0.99}}``——本函數回退查 ``table[str(date_ref)]["saturation"]``。

    回 None = 該層無 saturation 資訊（③ 不標註）。
    """
    if saturation_map is None:
        saturation_map = {}
    if isinstance(saturation_map, dict):
        val = saturation_map.get(str(date_ref)) if date_ref is not None else None
        if val is None:
            val = saturation_map.get(str(layer))
        if val is None:
            val = saturation_map.get(int(layer) if isinstance(layer, (int, float)) else layer)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    elif isinstance(saturation_map, (int, float)) and not isinstance(saturation_map, bool):
        return float(saturation_map)
    # 形狀 (b)：殘差條目內 saturation（F4——真實約束場可能把 saturation 放這裡）。
    if date_ref is not None:
        entry = table.get(str(date_ref))
        if isinstance(entry, dict):
            inner = entry.get("saturation")
            if isinstance(inner, (int, float)) and not isinstance(inner, bool):
                return float(inner)
    return None


def _find_gate_nodes(
    rigidity_map: list[dict],
    constraint_field: Any,
) -> list[dict]:
    """門節點標記（§12.6）：① 剛性低谷 ∧ ② 殘差 sharp 命中 → 門。

    ③ 不可延期：``saturation ≥ 0.99`` 僅作 heuristics 標註（非機械驗收條件，F8）——
    門節點卡上標註，人保留最終覆核。門節點 = 內核第二域的指針——頻譜能測到剛性低谷
    （選擇空間），但跳躍本身不屬於測量。
    """
    if not rigidity_map:
        return []

    constraint = constraint_field if isinstance(constraint_field, dict) else {}
    table = load_residual_table(constraint.get("residuals"))
    saturation_map = constraint.get("saturation")

    rigidities = [float(e["rigidity_prevalence"]) for e in rigidity_map]
    troughs = set(_local_minima(rigidities))

    gate_nodes: list[dict] = []
    for i, entry in enumerate(rigidity_map):
        is_trough = i in troughs
        date_ref = entry.get("date_ref")
        sharp = _sharp_hit(table, date_ref)
        saturation_val = _saturation_value(saturation_map, table, date_ref, entry["layer"])
        heuristic = saturation_val is not None and saturation_val >= 0.99

        if is_trough and sharp:
            conditions = ["① 剛性低谷（選擇空間最大）", "② 殘差 sharp 命中"]
            if heuristic:
                conditions.append("③ 不可延期（saturation≥0.99 heuristics）")
            gate_nodes.append(
                {
                    "layer": entry["layer"],
                    "date_ref": date_ref,
                    "rigidity_prevalence": entry["rigidity_prevalence"],
                    "conditions": conditions,
                    "saturation_heuristic": heuristic,
                    "saturation": saturation_val,
                    "note": "③ 不可延期：saturation≥0.99 僅作 heuristics 標註，人保留最終覆核（F8）",
                }
            )
    return gate_nodes


# ---------------------------------------------------------------------------
# 主入口：probe_tree
# ---------------------------------------------------------------------------

def probe_tree(
    situation: Any,
    constraint_field: Any = None,
    *,
    n_branch: int = 5,
    n_sample: int = 1,
    depth: int = 3,
    w: float = 0.5,
    gate_fn: Callable | None = None,
    call_api_fn: Callable | None = None,
    state_log_path: str | None = None,
    seed: int = 42,
    api_key: str | None = None,
    model: str = "deepseek-v4-flash",
    max_tokens: int = 8192,
    temperature: float = 0.6,
) -> dict:
    """決策樹探針——約束剛性測量器（PLAN-23 §十二 v1.2）。

    Args:
        situation: 社會物質結構 digest + local_texture + 6D 向量（§12.2 處境）。
        constraint_field: 約束場（承載量）——速率矩陣 + 頻譜數據 + 殘差表 sharp 信號
            （``{"rate_matrix":..., "residuals":..., "saturation":...}``，§12.2）。
        n_branch: 每層分支數（3–8，pilot 5）。
        n_sample: 採樣次數（≥1，pilot 1）——成本 = n_sample × depth calls（§12.8）。
        depth: 深度（2–4，pilot 3）——每層 1 call。
        w: CFG 場強（0–2，pilot 0.5–1.0）——**只在生成期 prompt 內作用**（約束強度），
            不作機械閾值（§12.5 刪 ε(w)）。
        gate_fn: LLM 呼叫函式（注入點，測試用 mock）——簽名同 ``call_api``：
            ``(prompt, api_key, *, model, max_tokens, temperature, system_message,
            response_format, thinking) -> str``。None 時退 ``call_api_fn``。
        call_api_fn: 真實 API 呼叫（None 時載入 ECC 的 api_utils.call_api）。
        state_log_path: 可選——僅存入 meta 供 Phase E ``probe_select`` 收束落盤。
            🔴 F8：生成本身**不落 log**（不設 verify 全域 log）——只有收束（probe_select）
            落盤，避免全域 ``_log_path`` 洩漏到後續無 path 的操作。
        seed: 保留——目前機械層全確定性，未來隨機子採樣用。
        api_key: DeepSeek API key。僅在走真實 API（gate_fn=None）時需要。
        model / max_tokens / temperature: 傳給 gate_fn。

    Returns:
        dict：
        - ``trees``: 每樣本一棵樹（root / layers / paths / prevalence / rigidity_map /
          standing_wave / rejected / meta）。
        - ``prevalence``: 跨樣本每 (layer, label) 連續 prevalence + 信賴帶（remasking → UNKNOWN）。
        - ``rigidity_map``: 約束剛性地圖（逐層 ``{layer, date_ref, rigidity_prevalence,
          confidence_band, tags[], components}``）。
        - ``archetypes``: 世界原型（Jaccard 語義距離聚類，非 cluster.run）。
        - ``gate_nodes``: 門節點卡片（① 剛性低谷 ∧ ② sharp 命中；③ saturation heuristics）。
        - ``rejected``: 被拒節點清單（人機收束檢視，不靜默丟棄）。
        - ``meta``: params / calls / cost_anchor（= n_sample × depth）/ n_paths /
          truncated（路徑爆炸截斷旗標，F5）/ state_log_path（僅供收束落盤，F8）。
    """
    if n_branch < 3 or n_branch > 8:
        raise ValueError(f"n_branch must be in [3, 8], got {n_branch}")
    if depth < 2 or depth > 4:
        raise ValueError(f"depth must be in [2, 4], got {depth}")
    if n_sample < 1:
        raise ValueError(f"n_sample must be >= 1, got {n_sample}")
    if not (0.0 <= w <= 2.0):
        raise ValueError(f"w must be in [0, 2], got {w}")

    if gate_fn is None:
        if call_api_fn is None:
            call_api_fn = _load_ecc_call_api()
        gate_fn = call_api_fn
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("API key required: pass api_key or set DEEPSEEK_API_KEY")

    situation_labels = _extract_situation_labels(situation)

    trees: list[dict] = []
    for _ in range(n_sample):
        tree = _generate_one_tree(
            situation,
            constraint_field,
            n_branch=n_branch,
            depth=depth,
            w=w,
            gate_fn=gate_fn,
            api_key=api_key or "",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            situation_labels=situation_labels,
        )
        trees.append(tree)

    total_calls = sum(int(t["meta"]["calls"]) for t in trees)
    prevalence = _aggregate_prevalence(trees)
    rigidity_map = _aggregate_rigidity_map(trees, prevalence)
    all_paths = [p for t in trees for p in t["paths"]]
    archetypes = _cluster_archetypes(all_paths)
    gate_nodes = _find_gate_nodes(rigidity_map, constraint_field)
    rejected = [r for t in trees for r in t["rejected"]]

    result = {
        "trees": trees,
        "prevalence": prevalence,
        "rigidity_map": rigidity_map,
        "archetypes": archetypes,
        "gate_nodes": gate_nodes,
        "rejected": rejected,
        "meta": {
            "params": {
                "n_branch": n_branch,
                "n_sample": n_sample,
                "depth": depth,
                "w": w,
                "seed": seed,
            },
            "calls": total_calls,
            "cost_anchor": n_sample * depth,
            "n_paths": [int(t["meta"]["n_paths"]) for t in trees],
            "truncated": any(bool(t["meta"].get("truncated")) for t in trees),
            "model": model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    # F8：生成不落 log——不設 verify 全域 log（避免後續無 path 的 probe_select / 其他
    # gate 洩漏寫入同一檔案）。state_log_path 僅存入 meta，供收束時 ``probe_select``
    # 獨佔觸發落盤（§12.7 F6：生成不落 log、只有收束落 log）。
    result["meta"]["state_log_path"] = state_log_path

    return result


# ---------------------------------------------------------------------------
# Phase E：人機交錯收束——probe_select
# ---------------------------------------------------------------------------

def _find_node(tree: dict, label: str, layer: int | None = None) -> dict | None:
    """在樹中尋找 label（可限 layer）的節點。"""
    if tree.get("root", {}).get("label") == label and (layer is None or layer == 0):
        return tree["root"]
    for layer_entry in tree.get("layers", []):
        if layer is not None and layer_entry["layer"] != layer:
            continue
        for b in layer_entry.get("branches", []):
            if b["label"] == label:
                return b
    return None


def _sibling_labels(tree: dict, node: dict) -> list[str]:
    """同層其餘分支 label（未選分支）。"""
    layer = node["layer"]
    return [
        b["label"]
        for layer_entry in tree.get("layers", [])
        if layer_entry["layer"] == layer
        for b in layer_entry.get("branches", [])
        if b["label"] != node["label"]
    ]


def _coerce_bool(x: Any) -> bool:
    """嚴格布林解析（F7）：``"false"/"0"/False/None`` → False，其餘 → True。"""
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1"}


def probe_select(
    result: dict,
    *,
    selected_label: str,
    layer: int | None = None,
    verdict: str = "selected",
    re_calibrate: bool = False,
    state_log_path: str | None = None,
    tree_index: int = 0,
    **extra: Any,
) -> dict:
    """人機收束：人選定分支 → 樹坍縮；未選分支標 ``unselected``（寫回 α₂，不刪除）。

    ``tree_index``（F3）：n_sample>1 多樹聚合時，坍縮以 ``trees[tree_index]`` 為準
    （預設 0 = 第一棵）；``unselected``/state_log 只反映該樹。越界即拒。

    F7：verdict 合法性在副作用（collapse 寫入）**之前**驗證——驗證失敗不留殘留；
    ``re_calibrate`` 用嚴格布林解析（``"false"`` → False，不當 Truthy）。

    F8：落盤路徑 = 顯式 ``state_log_path``，否則退 ``result["meta"]["state_log_path"]``
    （probe_tree 存入；生成不落 log、只有收束落 log）。

    記錄 state_log（仿 ``_write_alt_gate_state_log``，§12.7 F6）：
    ``{"ts", "prediction_id": "probe_tree-<ts>", "gate_type": "probe_tree",
    "verdict": "selected|unselected|human_override", "re_calibrate": <bool>,
    "n_branch", "layer", "tree_index", "selected_branch", "unselected_branches"}``。

    ``re_calibrate``：與機械判定相左時由人置 true，供 ``kernel/verify`` 校準
    （``query_state_log`` / ``summarize_state_log``）。
    """
    trees = result.get("trees") or []
    if not trees:
        raise TreeProbeError("result 無 trees——無法選擇分支")
    if not (0 <= tree_index < len(trees)):
        raise TreeProbeError(
            f"tree_index {tree_index} 越界（trees 共 {len(trees)} 棵）——"
            f"多樹聚合必須指定收束對象樹"
        )

    # F7：verdict 合法性先驗證，再做任何副作用（collapse 寫入）——驗證失敗不留殘留。
    if verdict not in ("selected", "unselected", "human_override"):
        raise TreeProbeError(
            f"verdict 必須是 selected|unselected|human_override，got {verdict!r}"
        )

    tree = trees[tree_index]
    node = _find_node(tree, selected_label, layer)
    if node is None:
        raise TreeProbeError(
            f"selected_label '{selected_label}' 不在樹中"
            f"（trees[{tree_index}]，layer={layer}）"
        )
    layer_actual = node["layer"]
    siblings = _sibling_labels(tree, node)

    collapse = {
        "selected": selected_label,
        "layer": layer_actual,
        "tree_index": tree_index,
        "unselected": siblings,
        "note": "未選分支標 unselected 寫回 α₂ 空間（不刪除）；多樹下以 trees[tree_index] 為準",
    }
    result["collapse"] = collapse

    re_calibrate_bool = _coerce_bool(re_calibrate)  # F7：嚴格布林（"false" → False）
    ts = datetime.now(timezone.utc).isoformat()
    entry = {
        "ts": ts,
        "prediction_id": f"probe_tree-{ts}",
        "gate_type": "probe_tree",
        "verdict": verdict,
        "re_calibrate": re_calibrate_bool,
        "n_branch": result.get("meta", {}).get("params", {}).get("n_branch"),
        "layer": layer_actual,
        "tree_index": tree_index,
        "selected_branch": selected_label,
        "unselected_branches": siblings,
    }
    for k, v in extra.items():
        entry[k] = v

    # F8：收束才落盤——路徑優先顯式參數，否則退 meta（probe_tree 存入）。
    path = state_log_path or result.get("meta", {}).get("state_log_path")
    _verify._state_log.append(entry)
    if path is not None:
        _verify.init_log(path)
        _verify._append_jsonl(_verify._log_path, entry)
    elif _verify._log_path is not None:
        _verify._append_jsonl(_verify._log_path, entry)

    result["state_log_entry"] = entry
    return result


__all__ = [
    "AXIS_A_VALUES",
    "AXIS_B_VALUES",
    "NECESSITY_HINT_KEY",
    "TreeProbeError",
    "assert_tree_gate_contract",
    "standing_wave_per_layer",
    "probe_tree",
    "probe_select",
    "_prevalence_band",
    "_label_dispersion",
    "_cluster_archetypes",
    "_find_gate_nodes",
]
