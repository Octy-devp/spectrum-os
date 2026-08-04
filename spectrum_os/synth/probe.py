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
  分支（數目提示性——由張力幾何決定；機械層僅在超過 ``MAX_BRANCHES_SAFETY_CAP`` 時拒收）；
  第 k 層接收處境 + 約束場 + 第 k−1 層已過檢查的分支作為反射對象
  （reflex of reflex = 層間迭代）。輸出：樹狀結構（root + branches + conditions）。
- **Phase B 機械測量/檢查**：§12.4 五項（整樹剛性統計 / quarantine / 回聲 / 兩軸值 / 樹結構）。
  被拒節點 → ``rejected`` 清單（人機收束檢視，不靜默丟棄）。
- **W1（DCA 基底接回）**：分支節點 ``roles``（CLAD 角色向量，可多叠加）+ ``role_instances``
  （嵌套實例，深度跟隨樹層）；文法轉移契約（``validate_alternative``，含結構性零）；速率矩陣
  零強度邊機械拒（具體限制的可選輸入）；6D 向量三進制方向輔助分量（小權重）。
- **Phase C 連續剛性測量**：樹→路徑枚舉 → 每標籤/每層連續 prevalence（0-1）+ 信賴帶
  （樣本變異）；``prevalence==0`` 或樣本不足 → UNKNOWN（remasking）；不切閾值、不標必然。
- **Phase D 輸出**：約束剛性地圖（逐層 ``{layer, date_ref, rigidity_prevalence[連續],
  confidence_band, tags[]}``）+ 世界原型（低剛性區聚類，Jaccard/共現語義距離聚類，
  🔴 明言非 ``cluster.run``，待語義向量化層 F10）+ 門節點
  （① 剛性低谷 ∧ ② 殘差 sharp 命中 → 標記；③ saturation≥0.99 僅作 heuristics 標註，F8）。
- **Phase E 收束回寫**：``probe_select``——人選定分支 → 樹坍縮；未選分支標 ``unselected``
  （寫回 α₂ 空間，不刪除）；state_log 記錄（仿 ``_write_alt_gate_state_log``，
  ``gate_type: "probe_tree"`` + ``re_calibrate``，§12.7 F6）。
- **T18 手動逐層**：``probe_tree_manual``（``on_layer`` 回呼 = 人機收束接縫）+ ``probe_expand_layer``
  （單層原語）+ ``_generate_one_tree_iter``（暫停的生成器）——``reflect_on`` 由「全部 passed」
  改為「[人選的那條]」，打破 1:1 續鏈（樹寬 1.3636 根因）。自動路徑（``probe_tree``）與
  手動路徑共用 ``_generate_layer_once`` / ``_finalize_tree`` / ``_assemble_probe_result``。

成本：≤ ``n_sample × depth`` calls（Phase A 每層 1 call，§12.8 錨定——不隨節點數漂移，
故**無重試**：單層契約失敗即拋錯）。
"""

from __future__ import annotations

import json
import math
import os
import unicodedata
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Callable, Sequence

import numpy as np

from spectrum_os.kernel import verify as _verify
from spectrum_os.quantum.dca_grammar import validate_alternative
from spectrum_os.quantum.multigraph import ROLES
from spectrum_os.quantum.quarantine import mechanical_filter
from spectrum_os.quantum.vector import project_ternary
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

# ---------------------------------------------------------------------------
# 語碼感知長度契約（混合語言語碼分層，機械層）
# ---------------------------------------------------------------------------

#: 語碼 → 欄位 → (字符上限, 詞數上限)。
#: - CJK 語碼（zh/ja/ko）：維持現字符上限（label 20 / grounding 80 / condition 40），
#:   不設詞數上限（CJK 以字符為天然單位，無空格斷詞）。
#: - 拉丁語碼（en/fr/de/sr）：放寬字符上限（×3）+ 詞數上限——德語複合詞如
#:   "Bewegliche Verteidigung mit getrennten Schwerpunkten"（52 字符、5 詞）不誤殺。
#: - 西里爾語碼（ru；sr 亦可用西里爾書寫）：同拉丁上限。
#: 向後相容：無語碼/未知語碼 → 舊行為（僅字符上限，無詞數上限）。
_LANGUAGE_CODE_LIMITS: dict[str, dict[str, tuple[int, int | None]]] = {
    "zh": {"label": (20, None), "grounding": (80, None), "condition": (40, None)},
    "ja": {"label": (20, None), "grounding": (80, None), "condition": (40, None)},
    "ko": {"label": (20, None), "grounding": (80, None), "condition": (40, None)},
    "en": {"label": (60, 8), "grounding": (240, 40), "condition": (120, 24)},
    "fr": {"label": (60, 8), "grounding": (240, 40), "condition": (120, 24)},
    "de": {"label": (60, 8), "grounding": (240, 40), "condition": (120, 24)},
    "sr": {"label": (60, 8), "grounding": (240, 40), "condition": (120, 24)},
    "ru": {"label": (60, 8), "grounding": (240, 40), "condition": (120, 24)},
}

#: 欄位 → 舊字符上限（無語碼/未知語碼回退，向後相容）。
_DEFAULT_FIELD_LIMITS: dict[str, int] = {
    "label": MAX_LABEL_LEN,
    "grounding": MAX_GROUNDING_LEN,
    "condition": MAX_CONDITION_LEN,
}

#: 語碼 → 允許書寫系統（語碼合規審計用）。塞爾維亞語為拉丁/西里爾雙書寫。
_CODE_ALLOWED_SCRIPTS: dict[str, frozenset[str]] = {
    "zh": frozenset({"cjk"}),
    "ja": frozenset({"cjk"}),
    "ko": frozenset({"cjk"}),
    "en": frozenset({"latin"}),
    "fr": frozenset({"latin"}),
    "de": frozenset({"latin"}),
    "sr": frozenset({"latin", "cyrillic"}),
    "ru": frozenset({"cyrillic"}),
}


def max_len_for_code(code: str | None, field: str = "label") -> int:
    """語碼 → 指定欄位字符長度上限（語碼感知長度契約）。

    向後相容：``code`` 為 None / 未知語碼 → 舊字符上限（label 20 / grounding 80 /
    condition 40）。CJK 語碼（zh/ja/ko）維持現上限；拉丁/西里爾語碼（en/fr/de/sr/ru）
    放寬至 ×3（label 60 / grounding 240 / condition 120）。
    """
    if not code:
        return _DEFAULT_FIELD_LIMITS.get(field, MAX_LABEL_LEN)
    limits = _LANGUAGE_CODE_LIMITS.get(code)
    if limits is None:
        return _DEFAULT_FIELD_LIMITS.get(field, MAX_LABEL_LEN)
    return limits[field][0]


def max_words_for_code(code: str | None, field: str = "label") -> int | None:
    """語碼 → 指定欄位詞數上限（拉丁/西里爾語碼用；None/CJK → None = 不設詞數上限）。"""
    if not code:
        return None
    limits = _LANGUAGE_CODE_LIMITS.get(code)
    if limits is None:
        return None
    return limits[field][1]


def _count_words(text: str) -> int:
    """以空白分隔計詞數（拉丁/西里爾語碼的詞數上限用）。"""
    return sum(1 for tok in text.split() if tok.strip())


def _classify_char(ch: str) -> str | None:
    """單字元 → 書寫系統類別（cjk/latin/cyrillic）；空白/標點/數字/無法歸類 → None。

    優先字符範圍（明確且快，Cyrillic U+0400–U+04FF、CJK U+4E00–U+9FFF 等），
    ``unicodedata.name()`` 兜底（CJK 擴展 B+、兼容表意文字、諺文等）。
    """
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:  # CJK 統一表意 / 擴展 A
        return "cjk"
    if 0xAC00 <= o <= 0xD7AF or 0x3040 <= o <= 0x30FF:  # 諺文 / 假名
        return "cjk"
    if 0x0400 <= o <= 0x04FF or 0x0500 <= o <= 0x052F:  # 西里爾 / 西里爾補充
        return "cyrillic"
    if 0x0041 <= o <= 0x005A or 0x0061 <= o <= 0x007A:  # 基本拉丁
        return "latin"
    if 0x00C0 <= o <= 0x024F or 0x1E00 <= o <= 0x1EFF:  # 拉丁擴充（é/ü/š…）
        return "latin"
    try:
        name = unicodedata.name(ch, "")
    except ValueError:
        name = ""
    if name.startswith("CJK") or name.startswith("HANGUL") or name.startswith("HIRAGANA") \
            or name.startswith("KATAKANA"):
        return "cjk"
    if name.startswith("CYRILLIC"):
        return "cyrillic"
    if "LATIN" in name:
        return "latin"
    return None


def detect_script(text: str) -> str:
    """偵測文本主要書寫系統（機械層，供語碼合規審計）。

    回傳 ``"cjk"`` / ``"latin"`` / ``"cyrillic"`` / ``"mixed"`` / ``"unknown"``。
    空白/標點/數字不計入判定；無可歸類字符 → ``"unknown"``；多種書寫系統共存 →
    ``"mixed"``。純機械判別——不判語義、不做翻譯判斷。
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    counts: dict[str, int] = {}
    for ch in text:
        cls = _classify_char(ch)
        if cls is not None:
            counts[cls] = counts.get(cls, 0) + 1
    if not counts:
        return "unknown"
    if len(counts) == 1:
        return next(iter(counts))
    return "mixed"


def assert_code_compliance(label: str, allowed_codes: Sequence[str]) -> bool:
    """語碼合規概念（審計輔助，**不接入管線**）：label 的偵測書寫系統是否屬於
    ``allowed_codes`` 任一語碼的允許集合（供未來審計使用）。

    - ``"unknown"``（純標點/空白）→ True（無從判定，不阻斷）。
    - ``"mixed"`` → 任一成分書寫系統命中允許集合即 True。
    - 未知語碼（不在 ``_CODE_ALLOWED_SCRIPTS``）→ 視為允許一切（不阻斷）。
    """
    detected = detect_script(label)
    if detected == "unknown":
        return True
    allowed: set[str] = set()
    for code in allowed_codes:
        allowed |= set(_CODE_ALLOWED_SCRIPTS.get(code, ()))
    if not allowed:
        return True
    if detected == "mixed":
        scripts: set[str] = set()
        for ch in label:
            cls = _classify_char(ch)
            if cls is not None:
                scripts.add(cls)
        return bool(scripts & allowed)
    return detected in allowed

#: 兩軸詞彙鏡射拒收（F3）：「繼承的/湧現的/替代」是 prompt 教的兩軸名詞——
#: LLM 把軸名詞鏡射進 label 並自標 substitution，非真偷渡。字首即拒
#: （echo 類，與 PROMPT_EXAMPLE_LABELS 拒收同構——機械只攔字首，不判語義真偽）。
AXIS_ECHO_PREFIXES: tuple[str, ...] = ("繼承的", "湧現的", "替代")

#: 分支節點允許鍵（F1：鍵集外即拒——prose 鍵尤其拒收，複用 ``_FORBIDDEN_PROSE_KEYS``）。
#: 核心 8 契約鍵 + F1 回退可選欄位（role/strength/period_months，出現才驗合法性）
#: + W1 DCA 角色向量（roles 可多叠加 / role_instances 嵌套）
#: + 機械層內部標記（``contamination``/``rejected``/``rate_zero`` 由
#: ``_mechanical_check_layer`` 注入到分支，非 LLM 輸出鍵——契約在 Phase B 之後跑，須放行）。
_ALLOWED_BRANCH_KEYS: frozenset[str] = frozenset(
    {
        "label", "grounding", "axis_A", "axis_B", "rigidity_prevalence",
        "confidence_band", "conditions", "parent",
        "role", "strength", "period_months",
        "roles", "role_instances",
        "perspective", "binding",  # T16：鏡角（這條路從誰的眼睛看）＋對抗的束縛
        "contamination", "rejected", "rate_zero",
    }
)

#: 單樹路徑數上界（路徑爆炸防護，F5）——depth≤4、每層最多安全網 20 分支時理論葉數可達
#: 20^4=160000，但真正的爆炸防護是下方截斷（超過即截斷並記 meta ``truncated: true``，
#: 不無限遞迴）。
MAX_PATHS_PER_TREE = 4096

#: 單層分支數機械安全網（2026-08-03 提示性解耦）：n_branch 現為**提示性參考**
#: （prompt 已改為「幾根柱就幾條路」，不再要求「≤ N_branch」）——本常數是**獨立**的
#: 荒謬輸出濾網：一層超過 20 分支幾乎必然是壞輸出（JSON 黏連/重複/跑飛），對其 raise；
#: 真正路徑爆炸防護是 MAX_PATHS_PER_TREE 截斷 + max_tokens 封頂。固定值與提示**解耦**
#: （不隨 n_branch 縮放）——「超過提示值」不再拒，只有「超過安全網」才拒。
MAX_BRANCHES_SAFETY_CAP = 20

#: 整樹剛性 blend 權重：rigidity = w*LLM 均值 + (1-w)*機械離散補數（§12.4 #1）。
_RIGIDITY_BLEND_W = 0.5

#: substitution 節點在剛性聚合中的降權權重（§12.4 #4：contamination 降權，不判語義真偽）。
_CONTAMINATION_WEIGHT = 0.5

#: W1：DCA 角色向量——roles 建議為強制（DCA 是通用文法），但為向後相容設為**可選**：
#: 舊 schema（無 roles）仍過；一旦某分支帶 roles，文法轉移 / 6D / 速率輔助即自動啟動。
ROLES_REQUIRED = False

#: W1：role_instances 嵌套深度上限——遞歸在具體限制下：嵌套深度跟隨樹層
#: （層 k 最多 k 層實例 c1→c2→…），全局再以本常數封頂（與 probe 深度上界一致）。
MAX_ROLE_INSTANCE_DEPTH = 4

#: W1：6D 向量在剛性 blend 中的輔助權重（小權重——主代理仍是 label 語義離散，§12.4 #1）。
_VECTOR_BLEND_W = 0.1

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
# W1：DCA 角色向量 / 文法轉移 / 6D 輔助 helpers
# ---------------------------------------------------------------------------

def _as_role_list(roles: Any) -> list[str]:
    """規範角色為 list[str]（容忍單一 str）；非集合回 []。

    不做內容驗證（內容驗證在契約層 ``_normalize_roles``）——此處只防
    機械層對原始 LLM 輸出（可能 str）逐字元迭代的錯誤。
    """
    if isinstance(roles, str):
        return [roles]
    if isinstance(roles, (list, tuple, set, frozenset)):
        return [r for r in roles if isinstance(r, str)]
    return []


def _normalize_roles(roles: Any, context: str = "") -> list[str]:
    """標準化 + 驗證 ``roles`` 欄位（CLAD 角色集合/向量，可多叠加）。

    接受單一 str（LLM 慣用輸出）或 str 集合；內容必須 ⊆ ``ROLES`` 且非空；
    回傳去重、按 ROLES 次序排序的列表。非法即 TreeProbeError。
    """
    if isinstance(roles, str):
        roles = [roles]
    if not isinstance(roles, (list, tuple, set, frozenset)):
        raise TreeProbeError(
            f"{context}roles 必須是 CLAD 角色集合/列表（可多，叠加），"
            f"got {type(roles).__name__}"
        )
    out: list[str] = []
    for r in roles:
        if not isinstance(r, str) or r not in ROLES:
            raise TreeProbeError(f"{context}roles 含非法角色 {r!r}（允許 {ROLES}）")
        if r not in out:
            out.append(r)
    if not out:
        raise TreeProbeError(f"{context}roles 不能為空——空集合無意義（缺省請省略欄位）")
    out.sort(key=lambda r: ROLES.index(r))
    return out


def _role_instance_depth(value: Any) -> int:
    """單一實例值的嵌套深度：str/list = 1（如 \"c1\" 或 [\"l1\", \"l2\"]）；

    dict（實例 id → 子實例）= 1 + 子值最大深度（如 {\"c1\": \"c2\"} = 2）。
    """
    if isinstance(value, dict):
        if not value:
            return 1
        return 1 + max(_role_instance_depth(v) for v in value.values())
    return 1


def _validate_instance_value(value: Any, *, context: str = "") -> None:
    """驗證單一角色實例值：str | list[str] | dict(實例 id → 子實例)。"""
    if isinstance(value, str):
        if not value.strip():
            raise TreeProbeError(f"{context}實例 id 不能為空字串")
        return
    if isinstance(value, list):
        if not value:
            raise TreeProbeError(f"{context}實例列表不能為空")
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise TreeProbeError(
                    f"{context}實例列表元素必須是非空 str，got {item!r}"
                )
        return
    if isinstance(value, dict):
        if not value:
            raise TreeProbeError(f"{context}嵌套實例 dict 不能為空")
        for k, v in value.items():
            if not isinstance(k, str) or not k.strip():
                raise TreeProbeError(f"{context}嵌套實例 id 必須是非空 str，got {k!r}")
            _validate_instance_value(v, context=f"{context}{k}.")
        return
    raise TreeProbeError(
        f"{context}實例值必須是 str | list[str] | dict（嵌套），got {type(value).__name__}"
    )


def _validate_role_instances(
    role_instances: Any, *, max_depth: int, context: str = ""
) -> int:
    """驗證 ``role_instances`` 結構（角色 → 實例 id / 嵌套實例）+ 嵌套深度。

    嵌套深度跟隨樹層：層 k 的角色實例可含子實例 c2/c3……，但總深度 ≤ 樹層
    （遞歸在具體限制下，非無限）。回傳嵌套深度。
    """
    if not isinstance(role_instances, dict):
        raise TreeProbeError(
            f"{context}role_instances 必須是 dict（角色 → 實例 id/嵌套），"
            f"got {type(role_instances).__name__}"
        )
    if not role_instances:
        raise TreeProbeError(f"{context}role_instances 不能為空 dict")
    for role, value in role_instances.items():
        if not isinstance(role, str) or role not in ROLES:
            raise TreeProbeError(
                f"{context}role_instances 鍵 {role!r} 非法（允許 {ROLES}）"
            )
        _validate_instance_value(value, context=f"{context}role_instances[{role}].")
    # 嵌套深度：頂層 dict 是容器本身（不計層）——深度 = 各角色實例值深度的最大值。
    depth = max(_role_instance_depth(v) for v in role_instances.values())
    if depth > max_depth:
        raise TreeProbeError(
            f"{context}role_instances 嵌套深度 {depth} 超過樹層上限 {max_depth}"
            f"（遞歸在具體限制下，深度跟隨樹層）"
        )
    return depth


def _extract_situation_vector(situation: Any) -> dict | None:
    """從 situation 提取 6D 向量 dict（StateVector dict，須含 d1/d2/d3）；無則 None。

    W1：世界無關 fallback——situation 不帶 ``6d_vector`` 時回 None，機械層行為不變。
    """
    if not isinstance(situation, dict):
        return None
    vec = situation.get("6d_vector")
    if not isinstance(vec, dict):
        return None
    if not all(k in vec for k in ("d1", "d2", "d3")):
        return None
    return vec


def _role_distribution_from_branches(branches: list[dict]) -> dict[str, float]:
    """層內分支 ``roles`` 的歸一化分佈（出現次數）；無任何 roles 時回空 dict。"""
    counts: dict[str, float] = {r: 0.0 for r in ROLES}
    for b in branches:
        for r in _as_role_list(b.get("roles")):
            if r in ROLES:
                counts[r] += 1.0
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {r: c / total for r, c in counts.items()}


def _ternary_alignment(
    situation_vector: dict | None, role_dist: dict[str, float]
) -> float | None:
    """6D 輔助分量：situation 6D 方向 × 層內角色三進制投影的一致性（0-1）。

    用 ``project_ternary``（quantum/vector.py）把角色分佈投影到 D1-D3，與 situation
    ``6d_vector`` 的 (d1,d2,d3) 逐位比對——一致位元比例 = alignment。
    無 6d_vector 或無角色分佈 → None（世界無關 fallback：rigid 行為不變）。
    """
    if situation_vector is None:
        return None
    if not role_dist:
        return None
    d1, d2, d3 = project_ternary(role_dist)
    ref = (
        int(situation_vector.get("d1", 0)),
        int(situation_vector.get("d2", 0)),
        int(situation_vector.get("d3", 0)),
    )
    matches = sum(1 for a, b in zip((d1, d2, d3), ref) if a == b)
    return matches / 3.0


def _as_rate_matrix(rate_matrix: Any):
    """把 rate_matrix 規範為 4x4 numpy 陣列；形狀非法回 None（世界無關 fallback）。

    兼容兩種形狀：``{"matrix": [[...]]}``（estimate_rate_matrix 輸出）或裸 4x4。
    """
    if isinstance(rate_matrix, dict):
        rate_matrix = rate_matrix.get("matrix", rate_matrix)
    if rate_matrix is None:
        return None
    try:
        m = np.asarray(rate_matrix, dtype=np.float64)
    except Exception:
        return None
    if m.shape != (len(ROLES), len(ROLES)):
        return None
    return m


def _check_rate_zero(
    branch: dict, parent_branches: list[dict], rate_matrix: Any
) -> bool:
    """速率矩陣零強度檢查：父角色 → 子角色轉移在 rate_matrix 下全為零 → True。

    具體限制（可選輸入）：rate_matrix 存在才生效。叠加多角色下，子分支任一角色
    存在任一正強度父→子轉移即視為「活著」（False）。父 = root 或角色缺失時
    恆 False（世界無關 fallback——不誤傷）。
    """
    child_roles = _as_role_list(branch.get("roles"))
    if not child_roles:
        return False
    parent = _resolve_parent(branch, parent_branches)
    if parent is None:
        return False
    parent_roles = _as_role_list(parent.get("roles"))
    if not parent_roles:
        return False
    m = _as_rate_matrix(rate_matrix)
    if m is None:
        return False
    for cr in child_roles:
        if cr not in ROLES:
            continue
        strengths = [
            float(m[ROLES.index(pr)][ROLES.index(cr)])
            for pr in parent_roles
            if pr in ROLES
        ]
        if any(s > 0.0 for s in strengths):
            return False
    return True


def _check_role_transitions(
    output_data: dict, parent_branches: list[dict], notes: list[str]
) -> None:
    """文法轉移檢查（父 → 子角色，含結構性零）——**非致命**（2026-08-02 修正）。

    🔴 認識論修正：CLAD 是**閱讀文法**，不是**世界序列**——同一狀態可同時是
    crisis/lag/alternative/direction（多線程叠加，multigraph）；
    ``validate_alternative`` 的單向遞歸（crisis→lag→alternative→direction）
    是**觀察已發生事件**的工具，不是**生成限制**。強制父→子沿 CLAD 單向轉移
    = 把叠加態強制坍縮成單線序列（破壞量子容器）。
    因此：違反**不再 raise**（fatal），改為記入 ``notes``（非致命，人機收束
    ``convergence_view`` 檢視）——可能是真偷渡（structure violation），
    也可能是新結構（革命性翻轉：alternative→lag 受阻、crisis→direction 直接決裂）。
    """
    layer = output_data["layer"]
    for i, b in enumerate(output_data.get("branches", [])):
        child_roles = _as_role_list(b.get("roles"))
        if not child_roles:
            continue
        parent = _resolve_parent(b, parent_branches)
        if parent is None:
            continue
        parent_roles = _as_role_list(parent.get("roles"))
        if not parent_roles:
            continue
        for cr in child_roles:
            if not any(
                validate_alternative(pr, cr, depth=layer) for pr in parent_roles
            ):
                notes.append(
                    f"grammar: branch[{i}] 文法轉移異常 父角色 {parent_roles} → "
                    f"子角色 {cr!r}（validate_alternative 拒——含結構性零 "
                    f"direction→alternative / lag→direction；降為記錄，語義由人機收束判斷）"
                )


def assert_tree_gate_contract(
    output_data: dict,
    situation_labels: list[str] | None = None,
    *,
    max_branches: int | None = None,
    max_depth: int | None = None,
    parent_branches: list[dict] | None = None,
    code: str | None = None,
    echo_notes: list[str] | None = None,
) -> None:
    """機械契約檢查（仿 ``assert_alt_gate_contract``，PLAN-23 §12.3）。

    檢查：樹結構（層 ≤ max_branches、深度 ≤ max_depth、無循環、條件非空）、
    兩軸值合法性、回聲/空殼/placeholder 拒收、**無 necessity_hint**（出現即拒）、
    **分支鍵集外即拒**（F1：只允許 ``_ALLOWED_BRANCH_KEYS``，prose 鍵
    narrative/description/explanation/text/summary/prose 尤其拒收，複用
    ``_FORBIDDEN_PROSE_KEYS``）。

    **W1（DCA 基底）**：
    - ``roles``（可選但建議）：CLAD 角色**集合/向量**（可多，叠加）——值必須 ⊆ ROLES。
    - ``role_instances``（可選）：角色 → 實例 id / 嵌套實例；嵌套深度跟隨樹層（≤ layer）。
    - ``parent_branches``（可選）：上一層存活分支——給定時做**文法轉移**契約：
      子分支每個角色須是父分支某角色的合法轉移（``validate_alternative``，含結構性零
      direction→alternative / lag→direction）。
    **roles 為可選欄位（``ROLES_REQUIRED`` = False）是設計決策**：DCA 是通用文法，
    但為向後相容（舊 schema 無 roles 仍過），不設硬性強制；一旦某分支帶 roles，
    文法 / 6D / 速率輔助即自動啟動。速率矩陣零強度邊**不在本契約**處理——那是
    「具體限制」的可選輸入，由 ``_mechanical_check_layer`` 非致命拒（人機收束檢視，
    不靜默丟棄）。

    ⚠️ 與 alt_gate 的差異：echo/quarantine 在此處為**結構性**契約檢查
    （給 ``situation_labels`` 時回聲即拒）；``probe_tree`` 內部改以逐節點
    ``rejected`` 清單**非致命**處理（一層一個壞分支不該浪費整層 call，
    見 ``_mechanical_check_layer``）——所以 probe_tree 呼叫本契約時不傳
    ``situation_labels``，由 Phase B 逐節點處理。

    **語碼感知長度契約（機械層，混合語言語碼分層）**：``code`` 給定時（keyword
    參數或 output 頂層 ``code`` 欄位），label/grounding/condition 長度檢查走語碼
    感知路徑（``max_len_for_code`` / ``max_words_for_code``）——CJK 維持字符上限、
    拉丁/西里爾放寬字符 + 詞數上限。無語碼 → 舊行為（label ≤ 20 字符、grounding
    ≤ 80 字符；conditions 舊行為無長度檢查，維持不啟用）。

    **語義中介取代黑名單（決策 1/2/4，2026-08-03 全面撤銷 echo 攔截）**：
    ``PROMPT_EXAMPLE_LABELS``、``AXIS_ECHO_PREFIXES``、situation echo、層間 echo
    （含承義欄位全同的「候選原地踏步」）**一律不再 raise / reject**——改為 append
    到 ``echo_notes``（可選參數，None 時靜默略過，向後相容）。A/B 實測證明
    同相深化（同 action 同 binding 的進一步發展）會被誤判為原地踏步而整層全拒。
    語義判定完全交給人機收束（``convergence_view``）。
    """
    if not isinstance(output_data, dict):
        raise TreeProbeError(f"output must be dict, got {type(output_data).__name__}")

    # 語碼感知（機械層）：支援 keyword 參數或 output 頂層 ``code`` 欄位——
    # 皆無 → None（舊行為：字符上限 20/80/40、無詞數上限）。
    if code is None and isinstance(output_data.get("code"), str):
        code = output_data["code"]

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
        # 語碼感知長度（機械層）：無語碼 → 舊行為（20 字符）；拉丁/西里爾語碼 →
        # 放寬字符上限 + 詞數上限（德語複合詞不誤殺）。
        label_char_limit = max_len_for_code(code, "label")
        label_word_limit = max_words_for_code(code, "label")
        if len(label) > label_char_limit:
            raise TreeProbeError(
                f"branch[{i}].label 超過長度上限 (<= {label_char_limit} 字符): {label!r}"
            )
        if label_word_limit is not None and _count_words(label) > label_word_limit:
            raise TreeProbeError(
                f"branch[{i}].label 超過詞數上限 (<= {label_word_limit} 詞): {label!r}"
            )
        if label in PROMPT_EXAMPLE_LABELS:
            if echo_notes is not None:
                echo_notes.append(
                    f"example echo: branch[{i}].label 重複 prompt 範例標籤 '{label}'"
                    f"（非致命——範例只是形狀，人機收束檢視）"
                )
        if label.startswith(AXIS_ECHO_PREFIXES):
            if echo_notes is not None:
                echo_notes.append(
                    f"axis echo: branch[{i}].label '{label}' 以兩軸名詞為字首"
                    f"（F3：鏡射 prompt 教的軸詞彙——繼承的/湧現的/替代；非致命，人機收束檢視）"
                )
        if situation_set is not None and label in situation_set:
            if echo_notes is not None:
                echo_notes.append(
                    f"situation echo: branch[{i}].label '{label}' 重複處境標籤"
                    f"（非致命——以處境為參照的合法延續可能是同相干涉，人機收束檢視）"
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
        # 語碼感知長度（grounding）：無語碼 → 舊行為（80 字符）。
        g_char_limit = max_len_for_code(code, "grounding")
        g_word_limit = max_words_for_code(code, "grounding")
        if len(grounding) > g_char_limit:
            raise TreeProbeError(
                f"branch[{i}].grounding 超過長度上限 (<= {g_char_limit} 字符)"
            )
        if g_word_limit is not None and _count_words(grounding) > g_word_limit:
            raise TreeProbeError(
                f"branch[{i}].grounding 超過詞數上限 (<= {g_word_limit} 詞)"
            )
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
        cond_char_limit = max_len_for_code(code, "condition")
        cond_word_limit = max_words_for_code(code, "condition")
        for j, cond in enumerate(conditions):
            if not isinstance(cond, str) or not cond.strip():
                raise TreeProbeError(f"branch[{i}].conditions[{j}] 必須是非空 str")
            clean, matches = mechanical_filter(cond)
            if not clean:
                raise TreeProbeError(
                    f"branch[{i}].conditions[{j}] 未過 mechanical filter: {matches}"
                )
            # 語碼感知長度：conditions 舊行為無長度檢查——只有語碼給定時才啟用
            # （維持向後相容）。
            if code:
                if len(cond) > cond_char_limit:
                    raise TreeProbeError(
                        f"branch[{i}].conditions[{j}] 超過長度上限 (<= {cond_char_limit} 字符)"
                    )
                if cond_word_limit is not None and _count_words(cond) > cond_word_limit:
                    raise TreeProbeError(
                        f"branch[{i}].conditions[{j}] 超過詞數上限 (<= {cond_word_limit} 詞)"
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

        # W1：DCA 角色向量（第一公民）——可選，但一旦出現即驗結構 + 標準化。
        if "roles" in b:
            b["roles"] = _normalize_roles(b["roles"], f"branch[{i}].")
        if "role_instances" in b:
            _validate_role_instances(
                b["role_instances"],
                max_depth=min(
                    layer if isinstance(layer, int) else MAX_ROLE_INSTANCE_DEPTH,
                    max_depth or MAX_ROLE_INSTANCE_DEPTH,
                ),
                context=f"branch[{i}].",
            )

    # W1：文法轉移檢查（父 → 子角色，含結構性零）——parent_branches 給定時才檢查。
    # 🔴 2026-08-02：降為非致命（CLAD=閱讀文法非世界序列）——違反記入 echo_notes，
    # 語義由人機收束判斷（可能是真偷渡，也可能是新結構：alternative→lag 受阻、
    # crisis→direction 直接決裂）。
    if parent_branches:
        # 注意：echo_notes 為空 list 時 `or []` 會丟掉原 list（空 list 是 falsy）——
        # 用 None 判斷。
        _check_role_transitions(
            output_data, parent_branches, echo_notes if echo_notes is not None else []
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


def _layer_rigidity_components(
    branches: list[dict], *, situation_vector: dict | None = None
) -> dict:
    """整樹剛性統計（§12.4 #1）：分支間離散度 → 納入 rigidity 分佈。

    rigidity = blend_w * LLM 均值 + (1 - blend_w) * 機械離散補數。
    substitution 節點以 ``_CONTAMINATION_WEIGHT`` 降權（§12.4 #4，F7——不判語義真偽）。

    **W1（6D 輔助分量）**：situation 帶 ``6d_vector`` 且層內分支帶 ``roles`` 時，
    以 ``project_ternary``（quantum/vector.py）的角色方向投影 × situation 方向的
    一致性（``ternary_alignment``，0-1）以小權重 ``_VECTOR_BLEND_W`` 併入 blend：
    ``rigidity = (1-w_vec)*base + w_vec*alignment``。無任一輸入 → 行為不變
    （世界無關 fallback）。``rate_zero`` 分支由 ``_mechanical_check_layer`` 非致命拒，
    不進本函數（被拒節點不成反射對象）。

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
    base = float(
        np.clip(
            _RIGIDITY_BLEND_W * llm_mean
            + (1.0 - _RIGIDITY_BLEND_W) * mechanical_rigidity,
            0.0,
            1.0,
        )
    )

    # W1：6D 三進制方向輔助分量（小權重）——主代理仍是 label 語義離散。
    ternary = _ternary_alignment(
        situation_vector, _role_distribution_from_branches(branches)
    )
    if ternary is not None:
        rigidity = float(
            np.clip(
                (1.0 - _VECTOR_BLEND_W) * base + _VECTOR_BLEND_W * ternary,
                0.0,
                1.0,
            )
        )
    else:
        rigidity = base

    return {
        "rigidity_prevalence": round(rigidity, 4),
        "components": {
            "llm_mean_rigidity": round(llm_mean, 4),
            "label_dispersion": round(dispersion, 4),
            "mechanical_rigidity": round(mechanical_rigidity, 4),
            "role_dispersion": _role_dispersion(branches),
            "period_dispersion": _period_dispersion(branches),
            "ternary_alignment": round(ternary, 4) if ternary is not None else None,
            "situation_vector": situation_vector,
            "vector_blend_w": _VECTOR_BLEND_W if ternary is not None else None,
            "blend_w": _RIGIDITY_BLEND_W,
            "contamination_weight": _CONTAMINATION_WEIGHT,
        },
    }


# ---------------------------------------------------------------------------
# Phase B：逐節點機械檢查（quarantine / 回聲 / 兩軸降權）
# ---------------------------------------------------------------------------

def _same_denotation(child: dict, parent: dict) -> bool:
    """層間 echo 記錄用的承義比較：label 相同的候選與父分支，承義欄位是否全同。

    2026-08-03 起**不再用於攔截**（echo 全面撤銷攔截，見 ``_mechanical_check_layer``）——
    只用來在 echo_notes 中標記「候選原地踏步」（承義欄位全同）供人機收束檢視。

    承義欄位（binding/perspective/grounding）＋邊條件（conditions）是「義」，
    label 只是「形」。全同＝可能是同相深化（合法干涉）或真複製（原地踏步）——
    機械層不判，交人機收束。
    """
    for field in ("binding", "perspective", "grounding"):
        child_val = (child.get(field) or "").strip()
        parent_val = (parent.get(field) or "").strip()
        if child_val != parent_val:
            return False
    child_conds = [str(c).strip() for c in child.get("conditions", [])]
    parent_conds = [str(c).strip() for c in parent.get("conditions", [])]
    return child_conds == parent_conds


def _mechanical_check_layer(
    layer_entry: dict,
    situation_labels: list[str] | None,
    *,
    parent_branches: list[dict] | None = None,
    rate_matrix: Any = None,
    code: str | None = None,
    echo_notes: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """對一層分支做逐節點機械檢查：quarantine L1 / 回聲 / 空殼 / 兩軸 substitution 降權。

    **echo 全面撤銷攔截（2026-08-03）**：example echo / axis echo / situation echo /
    層間 echo（含承義欄位全同的「候選原地踏步」）一律不再 rejected / raise——全部
    降為 ``echo_notes`` 記錄，語義判定完全交給人機收束（convergence_view）。
    A/B 實測（``scripts/ab_prompt_test.py``）證明層間 echo 攔截誤殺「同相深化」
    （同 action 同 binding 的進一步發展）→ 整層全拒 → 樹崩潰。同相的繼續壓是
    合法干涉，機械層不假設自己能判「原地踏步」。

    **語碼合規（決策 3）**：``code`` 給定時，label 書寫系統不合語碼
    （``assert_code_compliance`` 為 False）→ rejected 清單（非致命）。mixed 寬鬆——
    ``assert_code_compliance`` 已對 mixed 回 True（sozio-Gestalten 天然混合）。
    ``code`` 為 None → 不檢查（向後相容）。

    **W1**：``rate_matrix``（具體限制的可選輸入）存在時，父→子角色轉移強度為零的
    邊機械**拒**（``rate_zero`` → rejected 清單，人機收束檢視——測量不靜默丟棄）。
    無 rate_matrix 或無父/子角色時不動（世界無關 fallback）。

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
            if echo_notes is not None:
                echo_notes.append(
                    f"example echo: label 重複 prompt 範例標籤 '{label}'"
                    f"（非致命——範例只是形狀，人機收束檢視）"
                )
        if label.startswith(AXIS_ECHO_PREFIXES):
            if echo_notes is not None:
                echo_notes.append(
                    f"axis echo: label '{label}' 以兩軸名詞為字首"
                    f"（F3：鏡射 prompt 教的軸詞彙——繼承的/湧現的/替代；非致命，人機收束檢視）"
                )
        if label in situation_set:
            if echo_notes is not None:
                echo_notes.append(
                    f"situation echo: label '{label}' 重複處境標籤"
                    f"（非致命——以處境為參照的合法延續可能是同相干涉，人機收束檢視）"
                )
        # 層間 echo：2026-08-03 徹底撤掉攔截——label 重複上一層（含承義欄位全同）
        # 不再拒收。A/B 實測（ab_prompt_test）證明：同相深化（同 action 同 binding 的
        # 進一步發展）會被誤判為「真原地踏步」而全拒（B 版 2/4 崩潰於 layer 2 全拒）。
        # 同相的繼續壓是合法干涉，不是自我複製。一律降為 echo_notes 記錄，語義判定
        # 完全交給人機收束（convergence_view）——機械層不再假設自己能判「原地踏步」。
        if parent_branches:
            parent_by_label = {
                str(p.get("label", "")).strip(): p
                for p in parent_branches if isinstance(p, dict)
            }
            parent = parent_by_label.get(label)
            if parent is not None and echo_notes is not None:
                if _same_denotation(b, parent):
                    echo_notes.append(
                        f"層間 label 延續（候選原地踏步，放行）：'{label}' 承義欄位與"
                        f"上一層全同——可能是同相深化或真複製，由人機收束判定"
                    )
                else:
                    echo_notes.append(
                        f"層間 label 延續（非 echo，放行）：'{label}' 承義欄位開出新路"
                        f"（binding={b.get('binding')!r} perspective={b.get('perspective')!r}）"
                    )

        # 語碼合規（決策 3：mixed 寬鬆——assert_code_compliance 已對 mixed 寬鬆）：
        # 書寫系統不合語碼 → rejected（非致命，人機收束檢視）。code=None → 不檢查。
        if code is not None and not assert_code_compliance(label, [code]):
            reasons.append("code: label 書寫系統不合語碼")

        # 兩軸 substitution → contamination 降權（F7：不驗語義真偽，機械只標記）
        if b.get("axis_B") == "substitution":
            b["contamination"] = True
        else:
            b["contamination"] = False

        # W1：rate_matrix 零強度邊機械拒（具體限制的可選輸入——有 rate_matrix 才生效）。
        # 世界無關 fallback：無 rate_matrix 或無父/子角色時不動（行為不變）。
        b["rate_zero"] = False
        if rate_matrix is not None and parent_branches is not None:
            if _check_rate_zero(b, parent_branches, rate_matrix):
                b["rate_zero"] = True
                reasons.append(
                    "rate_zero: 父→子角色轉移在速率矩陣下強度為零（具體限制，機械拒）"
                )

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

def _wilson_interval(p: float, n: int, z: float = _Z) -> tuple[float, float]:
    """Wilson score interval（95%）——p 在邊界 0/1 時**不坍縮為零寬度**。

    Wald 在 p∈{0,1} 時 SE=0 → 帶寬為 0 → 偽裝高信心（樣本少時尤其嚴重：
    N=2 全命中 → [1.0, 1.0]）。Wilson 以正態近似倒推，邊界處給出誠實的寬帶
    （§12.10 #1 ②：信賴帶誠實——樣本不足 → 寬帶/UNKNOWN，不偽裝高信心）。
    內部 p 值 Wilson ≈ Wald，行為連續。
    """
    if n <= 0:
        return 0.0, 1.0
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p), 0.0) / n + z2 / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _prevalence_band(
    count: int,
    n: int,
    *,
    cross_tree: list[float] | None = None,
    z: float = _Z,
) -> dict:
    """信賴帶（95%）。

    - ``cross_tree``（n_sample>1）：跨樹樣本變異 → 樣本均數 ± z·SE；若跨樹
      std=0（全樹一致，含邊界 0/1）→ 退 Wilson（以總樣本數 n 估，避免零寬度偽信心）。
    - 單樹/匯總：以路徑數 n 的 **Wilson** 區間（非 Wald——Wald 在邊界坍縮，T11 校準）。
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
        if std_p > 0.0:
            se = std_p / math.sqrt(float(len(arr)))
            lower = max(0.0, mean_p - z * se)
            upper = min(1.0, mean_p + z * se)
        else:
            # 全樹一致（std=0）——含邊界 0/1：退 Wilson 以總樣本數估，
            # 避免「全樹一致」被誤當高信心（樣本少時邊界零寬度是偽裝）。
            lower, upper = _wilson_interval(mean_p, n, z=z)
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
    lower, upper = _wilson_interval(p, n, z=z)
    confidence = "UNKNOWN" if (p == 0.0 or n < MIN_PATHS_FOR_CONFIDENCE) else "measured"
    if n < MIN_PATHS_FOR_CONFIDENCE:
        # 樣本不足（T11）→ 全寬 [0,1]——維持叠加交人，不偽裝任何信心
        lower, upper = 0.0, 1.0
    return {
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "n": n,
        "confidence": confidence,
        "basis": "wilson_path",
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
    """上一層分支的反射摘要（給下一層當反射對象）。

    T16 語義中介（2026-08-02）：補傳**承義欄位**（perspective/binding/grounding）——
    下一層反射者若只有 label（形）而無鏡角/束縛/可地面化支撐（義），
    就只能複製 label，無法「反射出義被推至極限後裂開的新方向」。
    轉移矩陣要從恆等（自我複製）變成真轉移，反射對象必須含義。
    """
    return {
        "label": b.get("label"),
        "axis_A": b.get("axis_A"),
        "axis_B": b.get("axis_B"),
        "rigidity_prevalence": b.get("rigidity_prevalence"),
        "conditions": b.get("conditions", []),
        # W1：DCA 角色向量/嵌套實例——供下一層反射時做文法延續（父→子合法轉移）。
        "roles": b.get("roles"),
        "role_instances": b.get("role_instances"),
        # T16 語義中介：承義欄位——鏡角（從誰的眼睛看）＋對抗的束縛＋可地面化支撐。
        "perspective": b.get("perspective"),
        "binding": b.get("binding"),
        "grounding": b.get("grounding"),
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

    2026-08-03：``n_branch`` 為**提示性參考**（非硬上限）——prompt 已改為
    「幾根柱就幾條路」，實際分支數由張力幾何決定；機械層僅在超過
    ``MAX_BRANCHES_SAFETY_CAP`` 時拒收。payload 以 ``n_branch_hint: true``
    標記此語義（供下游/人類辨識）。
    """
    payload: dict[str, Any] = {
        "task": "tree_generate",
        "layer": layer,
        # 2026-08-03：n_branch 只是**提示性參考**（prompt 已不再要求「≤ N_branch」，
        # 改為「幾根柱就幾條路」），不是硬上限——機械安全網是 MAX_BRANCHES_SAFETY_CAP。
        # 標記供下游/人類辨識這是提示而非約束。
        "n_branch": n_branch,
        "n_branch_hint": True,
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

    2026-08-04：**樹寬 1.3636 根因修正後的 docstring**——1.3636 唯一對應「完美 1:1
    平行鏈」（11 個父各恰 1 子），這需要 LLM 恆給互不重複的合法 parent label；
    主線 fallback 只會產生星形 avg=5.0，**不是** 1.3636 的來源（舊註解「大量省略
    parent→掛主線→1.36」為誤診）。prompt 已撤回「必須標明 parent」強制句，改為
    「parent 可選、新路可為場的獨立坍縮」——缺省掛主線保留為兜底安全網。
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


def _generate_layer_once(
    situation: Any,
    constraint_field: Any,
    reflect_on: list[dict],
    *,
    k: int,
    depth: int,
    n_branch: int,
    w: float,
    gate_fn: Callable,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    code: str | None,
    system_message: str | None,
    situation_labels: list[str] | None,
    rate_matrix: Any,
    situation_vector: dict | None,
    root: dict,
    echo_notes: list[str],
    timeout: int,
) -> tuple[list[dict], dict, list[dict], int]:
    """生成單層（Phase A/B 單層原語，T18 拆解）——payload 組裝 → gate 呼叫 → 解析 →
    機械檢查 → 結構契約 → 建樹。第 k 層。

    ``reflect_on`` 由外部注入：自動模式 = 上一層 passed（全量回饋）；手動模式 =
    **[人選的那條]**（T18——打破 1:1 續鏈：一個父可展開 3-4 個子）。

    回傳 ``(passed, layer_entry, layer_rejected, calls)``——``layer_entry["branches"]``
    即 ``passed``（同一批 dict 物件）；建樹副作用寫入 ``root`` / ``echo_notes``
    （可變容器，呼叫者持有）。
    """
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
        system_message=(
            TREE_GENERATE_SYSTEM_PROMPT if system_message is None else system_message
        ),
        response_format={"type": "json_object"},
        thinking=False,
        timeout=timeout,
    )

    parsed = _parse_tree_json(raw)
    # F6：路由洩漏/結構失配只在 check_routing_leak_or_schema 一處處理（內部已含
    # ROUTING_LEAK_KEYWORDS 掃描 + 任務 schema 檢查）——不重複手動迴圈。
    # 注意：洩漏且無法解析的回應會在解析階段先以「無法解析」拒絕（仍是 TreeProbeError）。
    leak_err = check_routing_leak_or_schema(raw, parsed, "tree_generate")
    if leak_err:
        raise TreeProbeError(leak_err)

    # Phase B：先逐節點機械檢查（quarantine/回聲/空殼/兩軸降權 + W1 rate_zero）→
    # passed / rejected。rate_matrix 為「具體限制」可選輸入——缺席時零強度不檢查。
    passed, layer_rejected = _mechanical_check_layer(
        parsed, situation_labels,
        parent_branches=reflect_on, rate_matrix=rate_matrix,
        code=code, echo_notes=echo_notes,
    )
    for b in layer_rejected:
        b["layer"] = k

    # 結構性契約：只對「存活分支」驗結構（文法/兩軸/placeholder/necessity/長度）。
    # quarantine/echo/空殼 已在上一步逐節點非致命處理——不該在契約層炸掉整層。
    if not passed:
        raise TreeProbeError(
            f"layer {k} 全部分支被機械拒——reflex 無法承載下一層（§12.12 #1：不承載即拒）"
        )
    surviving = dict(parsed)
    surviving["branches"] = list(passed)
    # W1：parent_branches 給定 → 契約做 DCA 文法轉移檢查（含結構性零，fatal）。
    # 2026-08-03：max_branches 用**獨立安全網**（MAX_BRANCHES_SAFETY_CAP），不再
    # 用提示值 n_branch——「超過提示值」不拒（張力幾何可能自然給更多），
    # 只有「超過荒謬上限」才 raise。真正路徑爆炸防護是 MAX_PATHS_PER_TREE。
    assert_tree_gate_contract(
        surviving, None, max_branches=MAX_BRANCHES_SAFETY_CAP, max_depth=depth,
        parent_branches=reflect_on, code=code,
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

    return passed, layer_entry, layer_rejected, 1


def _new_probe_state(
    situation: Any,
    constraint_field: Any,
    *,
    n_branch: int,
    w: float,
    gate_fn: Callable,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
    code: str | None,
    system_message: str | None,
    situation_labels: list[str] | None,
    timeout: int,
    depth: int,
) -> dict:
    """手動逐層會話的累積狀態（T18）——config + 累積輸出，供 probe_expand_layer /
    _generate_one_tree_iter 跨次呼叫攜帶。SSOT 仍是 YAML/JSON 源——此處只是單次
    手動會話的工作記憶（世界無關）。
    """
    constraint = constraint_field if isinstance(constraint_field, dict) else {}
    root = {
        "label": _root_label(situation),
        "grounding": "處境（root）",
        "layer": 0,
        "parent_label": None,
        "axis_A": "inherited",
        "axis_B": "expression",
        "children": [],
    }
    return {
        "situation": situation,
        "constraint_field": constraint_field,
        "n_branch": n_branch,
        "w": w,
        "gate_fn": gate_fn,
        "api_key": api_key,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "code": code,
        "system_message": system_message,
        "situation_labels": situation_labels,
        "timeout": timeout,
        "depth": depth,
        "rate_matrix": constraint.get("rate_matrix"),
        "situation_vector": _extract_situation_vector(situation),
        "root": root,
        "layers": [],
        "rejected": [],
        "echo_notes": [],
        "calls": 0,
    }


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
    code: str | None = None,
    system_message: str | None = None,
    situation_labels: list[str] | None,
    timeout: int = 180,
) -> dict:
    """生成單棵決策樹（Phase A + B + C）。每層 1 call，共 depth 次。

    W1：從 constraint_field 提取 ``rate_matrix``（具體限制的可選輸入）與 situation
    的 ``6d_vector``（6D 輔助分量）——皆為可選，缺席時行為不變（世界無關）。

    語碼分層（機械層）：
    - ``code``：語碼（zh/ja/ko/en/fr/de/sr/ru…）——None（預設）→ 長度契約舊行為。
    - ``system_message``：替代 system prompt——None（預設）→
      ``TREE_GENERATE_SYSTEM_PROMPT``（prompt 凍結中：本參數只允許呼叫者注入替代，
      不修改任何 prompt 文本）。
    """
    constraint = constraint_field if isinstance(constraint_field, dict) else {}
    rate_matrix = constraint.get("rate_matrix")
    situation_vector = _extract_situation_vector(situation)
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
    echo_notes: list[str] = []
    calls = 0
    reflect_on: list[dict] = []

    for k in range(1, depth + 1):
        # T18：單層邏輯已抽為 _generate_layer_once（自動與手動共用）——reflect_on
        # 由外部注入（自動 = 上一層 passed；手動 = [人選的那條]）。
        passed, layer_entry, layer_rejected, dcalls = _generate_layer_once(
            situation, constraint_field, reflect_on,
            k=k, depth=depth, n_branch=n_branch, w=w,
            gate_fn=gate_fn, api_key=api_key, model=model,
            max_tokens=max_tokens, temperature=temperature,
            code=code, system_message=system_message,
            situation_labels=situation_labels, rate_matrix=rate_matrix,
            situation_vector=situation_vector, root=root,
            echo_notes=echo_notes, timeout=timeout,
        )
        calls += dcalls
        rejected.extend(layer_rejected)
        layers.append(layer_entry)
        reflect_on = passed

    return _finalize_tree(
        root, layers, rejected, echo_notes, calls, situation_vector
    )


def _finalize_tree(
    root: dict,
    layers: list[dict],
    rejected: list[dict],
    echo_notes: list[str],
    calls: int,
    situation_vector: dict | None,
) -> dict:
    """Phase C/D 收尾（T18 抽離）——路徑枚舉 → prevalence + 信賴帶 → 剛性圖 →
    standing wave → 低發散警示 → 組裝樹 dict。自動（_generate_one_tree）與手動
    （probe_tree_manual / probe_expand_layer）共用同一收尾——自動/手動零分歧。
    """
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
        comps = _layer_rigidity_components(branches, situation_vector=situation_vector)
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

    # F6：低發散警示——每父平均子數（非葉節點）。1.0 = 純單鏈主線（無真分岔，
    # 每父恰好 1 子）；>1 = 真分岔。供 pilot 對 n_paths 過薄（=n_branch 級）時警示。
    all_nodes = [root] + [b for le in layers for b in le["branches"]]
    non_leaf = [n for n in all_nodes if n["children"]]
    avg_children_per_parent = round(
        float(np.mean([len(n["children"]) for n in non_leaf])) if non_leaf else 0.0,
        4,
    )

    return {
        "root": root,
        "layers": layers,
        "paths": paths,
        "prevalence": prevalence,
        "rigidity_map": rigidity_map,
        "standing_wave": standing_wave,
        "rejected": rejected,
        "echo_notes": echo_notes,
        "meta": {
            "calls": calls,
            "n_paths": n_paths,
            "truncated": paths_truncated,  # F5：路徑爆炸截斷旗標
            "avg_children_per_parent": avg_children_per_parent,  # F6：低發散警示
            "params": {},
        },
    }


def _generate_one_tree_iter(
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
    code: str | None = None,
    system_message: str | None = None,
    situation_labels: list[str] | None,
    timeout: int = 180,
):
    """逐層生成器（T18「暫停的生成器」）——每生成一層 yield 一次，外部控制流
    （人機收束）可暫停。

    每次 yield ``(k, layer_entry, state, passed)``：
    - ``k``：層號；``layer_entry``：本層分支（人從 ``layer_entry["branches"]`` 挑選）；
    - ``state``：累積狀態快照（root/layers/rejected/echo_notes/calls/situation_vector）——
      外部 break / close 後以最後的快照 ``_finalize_tree``；
    - ``passed``：本層存活分支（= 自動模式的下一層反射對象）。

    外部以 ``gen.send(next_reflect_on)`` 注入下一層 reflect_on：
    - ``[branch]``（人選的那條）→ 下一層 reflect_on = [branch]（打破 1:1 續鏈）；
    - ``None`` → 下一層 reflect_on = passed（自動續接全部，向後相容）。

    生成器本身不做 Phase C 收尾——收尾職責在外部控制流（``_finalize_tree``）。
    """
    state = _new_probe_state(
        situation, constraint_field,
        n_branch=n_branch, w=w, gate_fn=gate_fn, api_key=api_key,
        model=model, max_tokens=max_tokens, temperature=temperature,
        code=code, system_message=system_message,
        situation_labels=situation_labels, timeout=timeout, depth=depth,
    )
    reflect_on: list[dict] = []

    for k in range(1, depth + 1):
        passed, layer_entry, layer_rejected, dcalls = _generate_layer_once(
            state["situation"], state["constraint_field"], reflect_on,
            k=k, depth=state["depth"], n_branch=state["n_branch"], w=state["w"],
            gate_fn=state["gate_fn"], api_key=state["api_key"], model=state["model"],
            max_tokens=state["max_tokens"], temperature=state["temperature"],
            code=state["code"], system_message=state["system_message"],
            situation_labels=state["situation_labels"], rate_matrix=state["rate_matrix"],
            situation_vector=state["situation_vector"], root=state["root"],
            echo_notes=state["echo_notes"], timeout=state["timeout"],
        )
        state["calls"] += dcalls
        state["rejected"].extend(layer_rejected)
        state["layers"].append(layer_entry)
        snapshot = {
            "root": state["root"],
            "layers": state["layers"],
            "rejected": state["rejected"],
            "echo_notes": state["echo_notes"],
            "calls": state["calls"],
            "situation_vector": state["situation_vector"],
        }
        next_reflect_on = yield k, layer_entry, snapshot, passed
        if next_reflect_on is not None:
            reflect_on = list(next_reflect_on)
        else:
            reflect_on = passed


def _rigidity_band(rigidities: list[float], n_paths: int) -> dict:
    """整樹剛性信賴帶。

    - 跨樹（>1 棵）：跨樹樣本變異 → 樣本均數 ± z·SE，basis ``rigidity_cross_tree``；
      std=0（全樹一致，含邊界 0/1）→ 退 Wilson（避免零寬度偽信心，T11 校準）。
    - 單樹：以該層抵達路徑數 n 的 **Wilson** 區間，basis ``rigidity_wilson_path``
      （估的是**路徑枚舉**的抽樣誤差，不是 LLM 生成變異——後者無跨樹樣本時誠實標 UNKNOWN）。
    - remasking：``mean==0`` 或路徑數不足 → ``UNKNOWN``（維持叠加交人，§12.5 步驟 4）。
    """
    arr = np.asarray(rigidities, dtype=np.float64)
    if arr.size == 0:
        return {"lower": 0.0, "upper": 1.0, "n": 0, "confidence": "UNKNOWN", "basis": "no_data"}
    mean = float(arr.mean())
    if arr.size > 1:
        std = float(arr.std(ddof=1))
        if std > 0.0:
            se = std / math.sqrt(float(arr.size))
            lower = max(0.0, mean - _Z * se)
            upper = min(1.0, mean + _Z * se)
        else:
            lower, upper = _wilson_interval(mean, max(n_paths, 1))
        basis = "rigidity_cross_tree"
    else:
        lower, upper = _wilson_interval(mean, max(n_paths, 1))
        basis = "rigidity_wilson_path"
    confidence = (
        "UNKNOWN"
        if (mean == 0.0 or n_paths < MIN_PATHS_FOR_CONFIDENCE)
        else "measured"
    )
    if n_paths < MIN_PATHS_FOR_CONFIDENCE:
        # 樣本不足（T11）→ 全寬 [0,1]——維持叠加交人，不偽裝任何信心
        lower, upper = 0.0, 1.0
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


# ---------------------------------------------------------------------------
# T8（F10）：世界原型——語義距離聚類層（零 API，全樹自身特徵）
# ---------------------------------------------------------------------------

#: 語義距離相似度權重——label 字元 n-gram 為主、role/兩軸為輔（皆從樹導出，世界無關）。
_LABEL_SIM_W = 0.6
_ROLE_SIM_W = 0.25
_AXIS_SIM_W = 0.15

#: 路徑數超過此值時語義聚類（O(n²) 凝聚 + silhouette）成本過高 → 退 Jaccard 基底
#: （仍機械、零 API；pilot 路徑數通常 < 100，此為防護上限）。
MAX_PATHS_FOR_SEMANTIC_CLUSTER = 128


def _path_label_ngrams(path: list[dict]) -> set[str]:
    """路徑所有標籤的字元 n-gram 聯集（T8：軟 Jaccard 用，捕捉形近標籤）。"""
    out: set[str] = set()
    for n in path:
        if (
            isinstance(n, dict)
            and isinstance(n.get("label"), str)
            and n["label"].strip()
        ):
            out |= _char_ngrams(n["label"])
    return out


def _path_role_profile(path: list[dict]) -> dict[str, float]:
    """路徑內所有節點 ``roles``/``role`` 的 ROLES 計數（T8：DCA 文法親和）。"""
    counts: dict[str, float] = {r: 0.0 for r in ROLES}
    for n in path:
        if not isinstance(n, dict):
            continue
        roles = _as_role_list(n.get("roles"))
        single = n.get("role")
        if single in ROLES:
            roles = roles + [single]
        for r in roles:
            if r in ROLES:
                counts[r] += 1.0
    return counts


def _path_axis_profile(path: list[dict]) -> dict[str, float]:
    """路徑內軸 A/B 計數輪廓（T8：inherited/emergent × expression/substitution）。"""
    prof: dict[str, float] = {
        "inherited": 0.0, "emergent": 0.0,
        "expression": 0.0, "substitution": 0.0,
    }
    for n in path:
        if not isinstance(n, dict):
            continue
        a = n.get("axis_A")
        if a in ("inherited", "emergent"):
            prof[a] += 1.0
        b = n.get("axis_B")
        if b in ("expression", "substitution"):
            prof[b] += 1.0
    return prof


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """兩個計數輪廓的餘弦相似度（0-1）；皆零向量 → 1.0（無資訊不懲罰）。"""
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _semantic_path_similarity(path_a: list[dict], path_b: list[dict]) -> float:
    """兩路徑的語義距離相似度（0-1，T8/F10）——混合三特徵：

    ``w_label·softJaccard(字元 n-gram) + w_role·cos(角色輪廓) + w_axis·cos(兩軸輪廓)``。

    🔴 設計取捨：**不依賴 CLADCorpus / clad-gene-pool.jsonl**——那些語料在 ECC 側
    （feeds/sources_cache），spectrum-os 的零-API 機械層必須獨立可測；字元 n-gram +
    角色向量 + 兩軸輪廓全從樹自身導出（世界無關）。且比純 Jaccard（集合重疊）更細：
    形近標籤（「動員令凍結」vs「動員令解凍」）在集合層面零重疊，在 n-gram 層面高相似。
    """
    label_sim = _jaccard_sim(
        _path_label_ngrams(path_a), _path_label_ngrams(path_b)
    )
    role_sim = _cosine_sim(_path_role_profile(path_a), _path_role_profile(path_b))
    axis_sim = _cosine_sim(_path_axis_profile(path_a), _path_axis_profile(path_b))
    return float(
        np.clip(
            _LABEL_SIM_W * label_sim + _ROLE_SIM_W * role_sim + _AXIS_SIM_W * axis_sim,
            0.0,
            1.0,
        )
    )


def _average_linkage_sim(cluster_a: list[int], cluster_b: list[int], sim: np.ndarray) -> float:
    """凝聚層次聚類的平均連結（平均成對相似度）。"""
    pairs = [(i, j) for i in cluster_a for j in cluster_b]
    return float(np.mean([sim[i, j] for i, j in pairs]))


def _agglomerative_at_k(sim: np.ndarray, k: int) -> np.ndarray:
    """凝聚層次聚類（平均連結），在 k 簇時切——回傳每樣本的簇標籤。

    迭代合併平均相似度最高的兩簇，直到簇數 == k；平手時取最小索引對
    （確定性）。k >= n 時回傳全單例標籤。
    """
    n = sim.shape[0]
    if k >= n:
        return np.arange(n, dtype=np.intp)
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > k:
        best_pair: tuple[int, int] | None = None
        best_sim = -1.0
        for ai in range(len(clusters)):
            for bi in range(ai + 1, len(clusters)):
                s = _average_linkage_sim(clusters[ai], clusters[bi], sim)
                if s > best_sim:
                    best_sim = s
                    best_pair = (ai, bi)
        assert best_pair is not None
        ai, bi = best_pair
        merged = clusters[ai] + clusters[bi]
        clusters = [c for ci, c in enumerate(clusters) if ci not in (ai, bi)]
        clusters.append(merged)
    labels = np.zeros(n, dtype=np.intp)
    for ci, members in enumerate(clusters):
        for m in members:
            labels[m] = ci
    return labels


def _silhouette_similarity(labels: np.ndarray, sim: np.ndarray) -> float:
    """相似度矩陣上的平均 silhouette（-1..1，越高越好）。

    對每個樣本：a = 同簇平均相似度、b = 最大異簇平均相似度；
    s = (a-b)/max(a,b)；單例簇樣本不計（a 無定義）。
    """
    n = sim.shape[0]
    scores: list[float] = []
    for i in range(n):
        ci = labels[i]
        same = np.where(labels == ci)[0]
        same = same[same != i]
        if same.size == 0:
            continue
        a = float(np.mean(sim[i, same]))
        others = np.where(labels != ci)[0]
        if others.size == 0:
            continue
        b_vals: list[float] = []
        for cj in np.unique(labels[others]):
            members = others[labels[others] == cj]
            b_vals.append(float(np.mean(sim[i, members])))
        b = max(b_vals)
        denom = max(a, b)
        scores.append((a - b) / denom if denom > 0 else 0.0)
    if not scores:
        return 0.0
    return float(np.mean(scores))


def cluster_archetypes_semantic(
    paths: list[list[dict]],
    *,
    n_min: int = 5,
    n_max: int = 8,
    similarity: Callable | None = None,
) -> tuple[list[dict], dict]:
    """世界原型——語義距離聚類（T8/F10，§12.6 / §12.10 #5）。

    取代純 Jaccard 貪婪（``_cluster_archetypes``，pilot 僅 4 個原型）：
    - **全相似度矩陣 + 平均連結凝聚**（非依序貪婪——後者依賴輸入順序、易收斂到
      少數大簇）。
    - **k 自動選取**：在 [n_min, n_max]（預設 5–8，§12.10 #5 目標）內取平均
      silhouette 最高的 k；數據不足（路徑數 < n_min、或無法形成非單例簇）時
      **誠實回退**——不以假分割硬湊 5–8。
    - 特徵全從樹自身導出（字元 n-gram / 角色輪廓 / 兩軸輪廓，零 API、世界無關）。

    Returns:
        ``(archetypes, meta)``——archetypes 與 ``_cluster_archetypes`` 同構
        （外加 ``mean_sim``，代表路徑 = 簇內平均相似度最高者）；meta 記錄
        k / silhouette / 權重 / 路徑數 / 是否回退。
    """
    if not paths:
        return (
            [],
            {"k": 0, "n_paths": 0, "silhouette": None, "method": "semantic_hybrid"},
        )
    n = len(paths)
    if n > MAX_PATHS_FOR_SEMANTIC_CLUSTER:
        # O(n²) 凝聚 + silhouette 成本過高 → 退 Jaccard 基底（仍機械、零 API）。
        base = _cluster_archetypes(paths)
        return base, {
            "k": len(base),
            "n_paths": n,
            "silhouette": None,
            "method": "jaccard_fallback",
            "note": f"n_paths={n} 超過上限 {MAX_PATHS_FOR_SEMANTIC_CLUSTER}，退 Jaccard 基底",
        }

    sim_fn = similarity if similarity is not None else _semantic_path_similarity
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sim_fn(paths[i], paths[j]))
            sim[i, j] = sim[j, i] = float(np.clip(s, 0.0, 1.0))
        sim[i, i] = 1.0

    lo = min(n_min, n)
    hi = min(n_max, n)
    best_k: int | None = None
    best_sil = -2.0
    for k in range(lo, hi + 1):
        if k == n:
            continue  # 全單例——silhouette 無定義
        labels = _agglomerative_at_k(sim, k)
        if np.max(np.bincount(labels, minlength=n)) < 2:
            continue  # 無非單例簇——silhouette 無意義
        sil = _silhouette_similarity(labels, sim)
        if sil > best_sil:
            best_sil = sil
            best_k = k
    if best_k is None:
        # 誠實回退：k = min(n_max, n)，不硬湊 5–8。
        best_k = min(n_max, n)
        labels = _agglomerative_at_k(sim, best_k)
        best_sil = None
        fell_back = True
    else:
        labels = _agglomerative_at_k(sim, best_k)
        fell_back = False

    archetypes: list[dict] = []
    for ci in range(best_k):
        members = np.where(labels == ci)[0]
        cids = [int(m) for m in members]
        if len(cids) > 1:
            rep = min(
                cids,
                key=lambda i: -float(np.mean([sim[i, j] for j in cids if j != i])),
            )
            mean_sim = float(
                np.mean([sim[i, j] for i in cids for j in cids if j > i])
            )
        else:
            rep = cids[0]
            mean_sim = 1.0
        label_set = sorted(
            {
                p["label"].strip()
                for m in cids
                for p in paths[m]
                if isinstance(p, dict)
                and isinstance(p.get("label"), str)
                and p["label"].strip()
            }
        )
        archetypes.append(
            {
                "archetype_id": ci,
                "labels": label_set,
                "member_path_ids": cids,
                "n_members": len(cids),
                "representative_path_id": rep,
                "mean_sim": round(mean_sim, 4),
                "method": "semantic_hybrid",
                "note": (
                    "語義距離聚類（字元 n-gram + 角色輪廓 + 兩軸輪廓；"
                    "平均連結 + silhouette k 選取）；非 kernel/cluster.run（F10）"
                ),
            }
        )

    meta = {
        "k": best_k,
        "n_paths": n,
        "silhouette": round(best_sil, 4) if best_sil is not None else None,
        "method": "semantic_hybrid",
        "weights": {
            "label_ngram": _LABEL_SIM_W,
            "role": _ROLE_SIM_W,
            "axis": _AXIS_SIM_W,
        },
        "k_range": [lo, hi],
        "fell_back": fell_back,
    }
    return archetypes, meta


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
    """殘差 sharp 命中（§12.6 ②）：該 date_ref 條目命中 saturation/decoupling。

    F2：LLM 可能輸出月級 date_ref（``"1914-07"``）而殘差表是日級鍵
    （``"1914-07-22"``）——精確匹配不命中。月級 date_ref 掃描該月前綴鍵任一
    sharp 即命中（仿 residual_trigger 的 day→month fallback，但此處是
    month→day 掃描，方向相反）。日級 date_ref 維持精確匹配。
    """
    if not date_ref:
        return False
    key = str(date_ref)
    parts = key.split("-")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        # 月級（YYYY-MM）：掃描該月前綴鍵，任一 sharp 判據即命中
        for k, entry in table.items():
            if not isinstance(entry, dict):
                continue
            if not str(k).startswith(key):
                continue
            criteria = entry.get("criteria")
            if isinstance(criteria, list) and any(c in SHARP_CRITERIA for c in criteria):
                return True
        return False
    entry = table.get(key)
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
        # F1：月級聚合 map——日級 date_ref（"1914-07-22"）回退該月前綴（"1914-07"）
        if val is None and isinstance(date_ref, str) and len(date_ref.split("-")) == 3:
            val = saturation_map.get(date_ref[:7])
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
    code: str | None = None,
    system_message: str | None = None,
    timeout: int = 180,
) -> dict:
    """決策樹探針——約束剛性測量器（PLAN-23 §十二 v1.2）。

    Args:
        situation: 社會物質結構 digest + local_texture + 6D 向量（§12.2 處境）。
        constraint_field: 約束場（承載量）——速率矩陣 + 頻譜數據 + 殘差表 sharp 信號
            （``{"rate_matrix":..., "residuals":..., "saturation":...}``，§12.2）。
        n_branch: 每層分支數**提示**（1–20，pilot 5）——提示性參考：實際分支數由
            張力幾何決定（prompt「幾根柱就幾條路」），機械層只在超過
            ``MAX_BRANCHES_SAFETY_CAP``（20）時拒收。
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
        code: 語碼（zh/ja/ko/en/fr/de/sr/ru…）——None（預設）→ 長度契約舊行為
            （label ≤ 20 字符等）；拉丁/西里爾語碼 → 放寬字符 + 詞數上限。
        system_message: 替代 system prompt——None（預設）→ ``TREE_GENERATE_SYSTEM_PROMPT``。
            🔴 prompt 凍結中：本參數只允許呼叫者注入替代，不修改任何 prompt 文本。

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
        - ``echo_notes``: 層間 label 延續的記錄（假 echo——label 同但承義欄位開出
          新路，放行並記錄，供人機收束檢視；PLAN-23 §12.4 echo 降權非致命）。
        - ``meta``: params / calls / cost_anchor（= n_sample × depth）/ n_paths /
          truncated（路徑爆炸截斷旗標，F5）/ state_log_path（僅供收束落盤，F8）。
    """
    if n_branch < 1 or n_branch > 20:
        raise ValueError(f"n_branch must be in [1, 20] (hint only, not a cap), got {n_branch}")
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

    # F7：_extract_situation_labels 期待 {"situation": ...} 包裝（alt_gate 輸入形狀）——
    # 直接傳 bare situation dict 恆回 None → 處境-echo 拒收靜默失效。此處包裝，
    # 讓 situation_labels 真正帶處境標籤（防禦性：_extract_situation_labels 本身
    # 也兼容 bare situation 形狀）。
    situation_labels = _extract_situation_labels({"situation": situation})

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
            code=code,
            system_message=system_message,
            situation_labels=situation_labels,
            timeout=timeout,
        )
        trees.append(tree)

    return _assemble_probe_result(
        trees, constraint_field,
        n_branch=n_branch, n_sample=n_sample, depth=depth, w=w,
        seed=seed, code=code, model=model, state_log_path=state_log_path,
    )


# ---------------------------------------------------------------------------
# Phase E：人機交錯收束——收束視圖 + probe_select
# ---------------------------------------------------------------------------

def _assemble_probe_result(
    trees: list[dict],
    constraint_field: Any,
    *,
    n_branch: int,
    n_sample: int,
    depth: int,
    w: float,
    seed: int,
    code: str | None,
    model: str,
    state_log_path: str | None,
) -> dict:
    """把一或多棵樹組裝為 probe 結果 dict（prevalence / rigidity_map / archetypes /
    gate_nodes / meta）。自動（probe_tree）與手動（probe_tree_manual）共用。
    """
    total_calls = sum(int(t["meta"]["calls"]) for t in trees)
    prevalence = _aggregate_prevalence(trees)
    rigidity_map = _aggregate_rigidity_map(trees, prevalence)
    all_paths = [p for t in trees for p in t["paths"]]
    # T8（F10）：世界原型——語義距離聚類（取代純 Jaccard 貪婪 `_cluster_archetypes`）。
    archetypes, archetype_meta = cluster_archetypes_semantic(all_paths)
    gate_nodes = _find_gate_nodes(rigidity_map, constraint_field)
    rejected = [r for t in trees for r in t["rejected"]]
    echo_notes = [n for t in trees for n in t.get("echo_notes", [])]

    result = {
        "trees": trees,
        "prevalence": prevalence,
        "rigidity_map": rigidity_map,
        "archetypes": archetypes,
        "archetype_meta": archetype_meta,
        "gate_nodes": gate_nodes,
        "rejected": rejected,
        "echo_notes": echo_notes,
        "meta": {
            "params": {
                "n_branch": n_branch,
                "n_sample": n_sample,
                "depth": depth,
                "w": w,
                "seed": seed,
                "code": code,
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


def probe_tree_manual(
    situation: Any,
    constraint_field: Any = None,
    *,
    n_branch: int = 5,
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
    code: str | None = None,
    system_message: str | None = None,
    timeout: int = 180,
    on_layer: Callable | None = None,
) -> dict:
    """手動逐層觸發（T18）——生成器逐層暫停，``on_layer`` 回呼是人機收束的接縫。

    ``on_layer(layer_entry, ctx)`` → 回傳 **[選定的 branch dict]**（從
    ``layer_entry["branches"]`` 挑一個）作為下一層 reflect_on（打破 1:1 續鏈：
    一個父可展開多子）；回傳 **None** → 中止（樹停在該層）。
    ``ctx = {"k", "passed", "rejected", "echo_notes"}``。
    ``on_layer=None``（預設）→ 自動續接全部 passed（模擬 probe_tree 單樹行為，
    向後相容）。

    🔴 世界無關：不引入任何世界特定詞；與 probe_tree 共用 _generate_layer_once /
    _finalize_tree / _assemble_probe_result——自動與手動路徑的單層邏輯零分歧。

    場狀態壓縮注入（PLAN §12.14「reflect_on = [人選那條] + 場狀態」）留待後續——
    本實作先做最小可行：只注入 [人選那條]（經 _build_layer_payload 的
    reflection.passed_branches）。
    """
    if n_branch < 1 or n_branch > 20:
        raise ValueError(f"n_branch must be in [1, 20] (hint only, not a cap), got {n_branch}")
    if depth < 2 or depth > 4:
        raise ValueError(f"depth must be in [2, 4], got {depth}")
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

    # F7：同 probe_tree——包裝 situation 讓 situation_labels 真正帶處境標籤。
    situation_labels = _extract_situation_labels({"situation": situation})

    gen = _generate_one_tree_iter(
        situation, constraint_field,
        n_branch=n_branch, depth=depth, w=w,
        gate_fn=gate_fn, api_key=api_key or "", model=model,
        max_tokens=max_tokens, temperature=temperature,
        code=code, system_message=system_message,
        situation_labels=situation_labels, timeout=timeout,
    )
    state: dict | None = None
    stopped_at: int | None = None
    # 🔴 手動驅動生成器（不能用 `for ... in gen`）：``gen.send()`` 本身會消耗下一次
    # yield——與 for 迴圈組合會雙重推進、跳過一層（T18 實測）。
    try:
        k, layer_entry, gen_state, passed = next(gen)
    except StopIteration:
        raise TreeProbeError("生成器未產出任何層——depth 必須 ≥ 2（已校驗）") from None
    while True:
        state = gen_state
        if on_layer is None:
            # 自動模式：續接全部 passed（向後相容——模擬 probe_tree 單樹行為）。
            try:
                k, layer_entry, gen_state, passed = gen.send(None)
            except StopIteration:
                break
            continue
        chosen = on_layer(layer_entry, {
            "k": k,
            "passed": passed,
            "rejected": gen_state["rejected"],
            "echo_notes": gen_state["echo_notes"],
        })
        if chosen is None:
            # 中止：樹停在該層。
            stopped_at = k
            gen.close()
            break
        # 防禦：人選的必須是本層分支之一（T18——reflect_on 只能是「人選的那條」）。
        if not isinstance(chosen, dict) or not any(
            chosen is b or chosen.get("label") == b.get("label") for b in passed
        ):
            raise TreeProbeError(
                f"on_layer 回傳的分支不在 layer {k} 的 branches 中——"
                f"必須從 layer_entry['branches'] 挑一個（回傳 None = 中止）"
            )
        try:
            k, layer_entry, gen_state, passed = gen.send([chosen])
        except StopIteration:
            break

    assert state is not None, "生成器未產出任何層——depth 必須 ≥ 2（已校驗）"
    tree = _finalize_tree(
        state["root"], state["layers"], state["rejected"],
        state["echo_notes"], state["calls"], state["situation_vector"],
    )
    result = _assemble_probe_result(
        [tree], constraint_field,
        n_branch=n_branch, n_sample=1, depth=depth, w=w,
        seed=seed, code=code, model=model, state_log_path=state_log_path,
    )
    if on_layer is not None:
        result["meta"]["manual"] = True
    if stopped_at is not None:
        result["meta"]["stopped_at_layer"] = stopped_at
    return result


def probe_expand_layer(
    situation: Any,
    constraint_field: Any = None,
    *,
    state: dict | None = None,
    reflect_on: list[dict] | None = None,
    n_branch: int = 5,
    depth: int = 3,
    w: float = 0.5,
    gate_fn: Callable | None = None,
    call_api_fn: Callable | None = None,
    seed: int = 42,
    api_key: str | None = None,
    model: str = "deepseek-v4-flash",
    max_tokens: int = 8192,
    temperature: float = 0.6,
    code: str | None = None,
    system_message: str | None = None,
    timeout: int = 180,
) -> dict:
    """生成單層（T18 手動逐層觸發原語）——``reflect_on`` = [人選的那條]，回傳該層結果。

    ``state``：手動會話累積狀態（首次呼叫 None → 自動初始化；後續傳回上回的 state，
    state 一旦建立即為權威——後續的 n_branch/w/gate_fn/model 等參數被忽略）。
    ``reflect_on``：下一層反射對象——[人選的那條 branch]；首次（None）→ []
    （從處境展開，同現行第一層）。``depth``：契約 max_depth（須 ≥ 已生成層數 + 1）。

    回傳 ``{"layer_entry", "passed", "state", "tree", "result", "calls"}``——
    ``tree`` 為截至該層的已 finalize 工作樹（供 convergence_view / probe_select
    人看）；``result`` 為 probe_select 可消費的最小 result 形狀（trees=[tree]）。

    人機流程（T19 收束 UX）：看 tree → probe_select(result, selected_label=...) →
    取 collapse.selected 對應的 branch → 下一層 reflect_on=[該 branch]。
    """
    if n_branch < 1 or n_branch > 20:
        raise ValueError(f"n_branch must be in [1, 20] (hint only, not a cap), got {n_branch}")
    if not (0.0 <= w <= 2.0):
        raise ValueError(f"w must be in [0, 2], got {w}")

    if state is None:
        if gate_fn is None:
            if call_api_fn is None:
                call_api_fn = _load_ecc_call_api()
            gate_fn = call_api_fn
            if api_key is None:
                api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not api_key:
                raise ValueError("API key required: pass api_key or set DEEPSEEK_API_KEY")
        situation_labels = _extract_situation_labels({"situation": situation})
        state = _new_probe_state(
            situation, constraint_field,
            n_branch=n_branch, w=w, gate_fn=gate_fn, api_key=api_key or "",
            model=model, max_tokens=max_tokens, temperature=temperature,
            code=code, system_message=system_message,
            situation_labels=situation_labels, timeout=timeout, depth=depth,
        )

    k = len(state["layers"]) + 1
    if k > state["depth"]:
        raise TreeProbeError(
            f"已達 max depth（{state['depth']}）——無法再展開第 {k} 層"
        )
    # 🔴 防禦（T18）：reflect_on（若有）必須全部屬於上一層的 passed（label 集合，
    # 與 _resolve_parent 比對方式一致）。違反 → payload 對 LLM 是語義謊言
    # （reflection.layer=k-1 但分支來自別層），且 _resolve_parent fallback 會把
    # 本層子掛到上一層父 → depth-jump。
    if reflect_on:
        prev_labels = (
            {b["label"] for b in state["layers"][-1]["branches"]}
            if state["layers"] else set()
        )
        if not prev_labels:
            raise TreeProbeError(
                "reflect_on 分支不屬於上一層——首次呼叫（k=1）沒有上一層，"
                "reflect_on 必須為 None/空（首層從處境展開，不能有「父」）"
            )
        foreign = [
            (b.get("label") if isinstance(b, dict) else repr(b))
            for b in reflect_on
            if not (isinstance(b, dict) and b.get("label") in prev_labels)
        ]
        if foreign:
            raise TreeProbeError(
                f"reflect_on 分支不屬於上一層——{foreign} 不在 layer {k - 1} 的 "
                f"branches（上一層 passed）中；reflect_on 必須從上一層 branches 挑選"
            )
    passed, layer_entry, layer_rejected, dcalls = _generate_layer_once(
        state["situation"], state["constraint_field"], reflect_on or [],
        k=k, depth=state["depth"], n_branch=state["n_branch"], w=state["w"],
        gate_fn=state["gate_fn"], api_key=state["api_key"], model=state["model"],
        max_tokens=state["max_tokens"], temperature=state["temperature"],
        code=state["code"], system_message=state["system_message"],
        situation_labels=state["situation_labels"], rate_matrix=state["rate_matrix"],
        situation_vector=state["situation_vector"], root=state["root"],
        echo_notes=state["echo_notes"], timeout=state["timeout"],
    )
    state["calls"] += dcalls
    state["rejected"].extend(layer_rejected)
    state["layers"].append(layer_entry)
    tree = _finalize_tree(
        state["root"], state["layers"], state["rejected"],
        state["echo_notes"], state["calls"], state["situation_vector"],
    )
    return {
        "layer_entry": layer_entry,
        "passed": passed,
        "state": state,
        "tree": tree,
        "result": {
            "trees": [tree],
            "meta": {
                "state_log_path": None,
                "params": {"n_branch": state["n_branch"]},
            },
        },
        "calls": dcalls,
    }


def convergence_view(result: dict) -> dict:
    """收束視圖（§12.7）——**可程式化輸出的資料結構**，供人/介面消費。

    三區段（§12.7 (a)(b)(c)）+ 選擇狀態：
    - ``rigidity_distribution``：約束剛性分佈（逐層 rigidity + 信賴帶，不切 hard/soft）。
    - ``archetype_cards``：世界原型卡片（標籤集 + 代表路徑標籤序列）。
    - ``gate_node_cards``：門節點卡片（三條件命中 + 該層分叉清單）。
    - ``selection``：``probe_select`` 後的坍縮狀態（未選分支標 unselected，不刪除）。

    全部從 ``probe_tree`` 產物機械重組——零 API、零 LLM。**人只在此視圖上決定**
    「這算必然還是有選擇」（§12.1 / §12.7：必然性由人詮釋，不由機器標）。
    """
    trees = result.get("trees") or []
    rigidity = [
        {
            "layer": e["layer"],
            "date_ref": e.get("date_ref"),
            "rigidity_prevalence": e["rigidity_prevalence"],
            "confidence_band": e["confidence_band"],
            "tags": e.get("tags", []),
        }
        for e in result.get("rigidity_map", [])
    ]

    # 代表路徑：archetype 的 member_path_ids 指向 all_paths（跨樹平鋪索引）。
    all_paths = [p for t in trees for p in (t.get("paths") or [])]
    archetype_cards: list[dict] = []
    for a in result.get("archetypes", []):
        rep_id = a.get("representative_path_id")
        rep_path: list[str] = []
        if isinstance(rep_id, int) and 0 <= rep_id < len(all_paths):
            rep_path = [
                n.get("label")
                for n in all_paths[rep_id]
                if isinstance(n, dict) and isinstance(n.get("label"), str)
            ]
        archetype_cards.append(
            {
                "archetype_id": a.get("archetype_id"),
                "labels": a.get("labels", []),
                "n_members": a.get("n_members"),
                "representative_path": rep_path,
                "method": a.get("method"),
            }
        )

    # 門節點卡片 + 該層分叉清單（跨樹去重）。
    def _layer_branch_labels(layer: int) -> list[str]:
        seen: list[str] = []
        for t in trees:
            for le in t.get("layers", []):
                if le.get("layer") != layer:
                    continue
                for b in le.get("branches", []):
                    lb = b.get("label")
                    if lb and lb not in seen:
                        seen.append(lb)
        return seen

    gate_cards: list[dict] = []
    for g in result.get("gate_nodes", []):
        gate_cards.append(
            {
                "layer": g["layer"],
                "date_ref": g.get("date_ref"),
                "rigidity_prevalence": g["rigidity_prevalence"],
                "conditions": g.get("conditions", []),
                "saturation_heuristic": g.get("saturation_heuristic"),
                "branch_labels": _layer_branch_labels(g["layer"]),
            }
        )

    if "collapse" not in result:
        selection: dict = {
            "selected": None,
            "layer": None,
            "unselected": [],
            "verdict": None,
            "re_calibrate": None,
        }
    else:
        entry = result.get("state_log_entry") or {}
        selection = {
            "selected": result["collapse"].get("selected"),
            "layer": result["collapse"].get("layer"),
            "unselected": result["collapse"].get("unselected", []),
            "verdict": entry.get("verdict"),
            "re_calibrate": entry.get("re_calibrate"),
        }

    return {
        "sections": [
            "rigidity_distribution",
            "archetype_cards",
            "gate_node_cards",
            "selection",
        ],
        "rigidity_distribution": rigidity,
        "archetype_cards": archetype_cards,
        "gate_node_cards": gate_cards,
        "selection": selection,
    }


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
    # 原子性：先 init_log（啟動檔案模式）再把 entry 入記憶體緩衝——否則 init_log 的
    # memory→file 過渡 flush 會把「本筆」重複寫入（每次收束恰一筆 JSONL 行）。
    path = state_log_path or result.get("meta", {}).get("state_log_path")
    if path is not None:
        _verify.init_log(path)
    _verify._state_log.append(entry)
    if _verify._log_path is not None:
        _verify._append_jsonl(_verify._log_path, entry)

    result["state_log_entry"] = entry
    return result


__all__ = [
    "AXIS_A_VALUES",
    "AXIS_B_VALUES",
    "AXIS_ECHO_PREFIXES",
    "NECESSITY_HINT_KEY",
    "TreeProbeError",
    "assert_tree_gate_contract",
    "standing_wave_per_layer",
    "cluster_archetypes_semantic",
    "convergence_view",
    "probe_tree",
    "probe_tree_manual",
    "probe_expand_layer",
    "probe_select",
    "detect_script",
    "assert_code_compliance",
    "max_len_for_code",
    "max_words_for_code",
    "_prevalence_band",
    "_label_dispersion",
    "_cluster_archetypes",
    "_find_gate_nodes",
    "_sharp_hit",
    "_saturation_value",
    "_generate_one_tree",
    "_generate_one_tree_iter",
]
