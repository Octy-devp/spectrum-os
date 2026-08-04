"""synth.round_loop — 回合協定編排器（force-model-social-dynamics.md §2.8, v1.5）。

一輪的定義（§2.8）：

    Mode A（動力學步進，直到撕裂）→ STOP → 探針負載 → 人選結算 → 寫回 → 下一輪

五步：
1. **Mode A 步進**——以當下狀態（R/C/P_R）+ **當下可觀測**的外部場，``dynamics.step``
   推進一步。外部場只餵當下可觀測值，不預先餵未來序列（§2.8 不變式）。
2. **撕裂偵測**——每步在步進**之前**查 ``residual_trigger``（saturation/decoupling）。
   **撕裂即停**：Mode A 永不跨越撕裂點——trigger 在時刻 t 回報撕裂時，軌跡止於 t
   （含 t，但不含任何 t' > t）；若輪起始即撕裂，軌跡為空、status="tear"。
3. **探針負載**——STOP 後建 probe payload（撕裂狀態、殘差表、S(t) 軌跡、R/C/P_R
   快照）——交給人選（``settlement_fn``）。
4. **結算**——人選回傳下一輪的決策（``decision`` dict：外部場調製 mod_alpha/mod_beta、
   節點狀態覆寫 state、或接受/拒絶某路徑 accepted_path/rejected_paths）。
   **介面是 callback，不是 LLM**——``settlement_fn(probe_payload) -> decision``。
5. **寫回**——``convergence_store`` 存 round snapshot + FieldLog 追加；下一輪從
   新狀態起點。

不變式（§2.8）：
- 撕裂是輪的邊界——Mode A 永不跨越撕裂點；外部場只餵當下可觀測值。
- 結算是人機接口——每輪的收束點是「人選」，不是模型自動決定。
- 1920 的社會內容只有走完每一輪才知道——不是從 1900 算出。

🔴 零 LLM：本模組只 import numpy + contracts + residual_trigger + convergence_store
（全部純 numpy / stdlib），不觸發任何 api/LLM/prompt 模組。Mode B 委派（§2.8
「enumerate / probe expand」）以**注入 callable**（``probe_converge_fn`` 與
``mode_b`` 設定內的委派函式）或**首次使用時惰性載入**（``importlib.import_module``）
接上——模組層 import 面維持零 LLM。

🆕 Mode B 委派（§2.8 對接——可選路徑，不破壞機械 settlement_fn）：

    run_one_round（Mode A 直到撕裂）→ settle() → 處境組裝器（probe →
    SituationSpec 形狀：digest / local_texture / 6d_vector / actors）
    → 委派 probe_converge_round（expand → 人選 → 坍縮 → FieldLog/snapshot
    統一寫回）→ branch → decision（mod_alpha/mod_beta/state）→ 套用。

兩種形態（``mode_b["style"]``，撕裂語義/設定驅動）：
- ``"probe"``（預設，人選方向）：``probe_converge_round`` 內部路徑——展開一層
  決策樹，人選一條坍縮；未選標 unselected（不刪除）。
- ``"enumerate"``（意外枚舉）：§12.13「Mode B 不空手起跳」——先
  ``record_mode_a_history``（把當輪 ODE 軌跡寫入）→ ``mode_a_posterior_init``
  → ``alt_gate_enumerate``。
給定時，寫回只走 delegate 一條路（統一寫回）；未給定時，維持機械 settlement_fn
（向後相容）。``settle(decision=...)`` 顯式給決策 = 人類覆寫 → 走機械路徑。

外部場調製的注入點：``forces.py`` 提供公開 ``set_modulation``/``current_modulation``
——編排器經公開接口注入合成 mod（乘法合成：``base(t) × round_mod[i]``），不訪問
私有屬性。合成在呼叫時才讀取 ``round_mod``，因此 ``settle(decision)`` 一落地，
``effective_rates`` 立即反映——不需重建引擎。
⚠️ 同一個 dynamics 若被多個 RoundLoop 依序使用會多層合成；建議每 loop 用獨立
dynamics，或傳 ``base_mod_alpha``/``base_mod_beta`` 明確指定基線。

樣式：numpy-only、純 Python 編排、零 LLM、零 prompt。
"""

from __future__ import annotations

import importlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from spectrum_os.contracts import ActorCard, FieldLog
from spectrum_os.synth.convergence_store import (
    save_dashboard,
    save_field_log,
    save_round_snapshot,
)
from spectrum_os.synth.residual_trigger import make_residual_trigger


class RoundProtocolError(Exception):
    """回合協定違規（§2.8 流程錯誤）。

    例如：上一輪未結算就再 run_one_round、或沒有待結算的輪就 settle()。
    撕裂是輪的邊界——未結算前不可跨入下一輪。
    """


def _broadcast_force(value: Any, n: int, name: str) -> np.ndarray:
    """把 scalar / length-n array-like 廣播為 length-n float64 陣列（不可負）。"""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=np.float64)
    if arr.size != n:
        raise ValueError(f"{name} 長度 {arr.size} != n_nodes {n}")
    return arr.reshape(n)


def _resolve_base_mods(
    provided: Any, dynamics: Any, attr: str, n: int
) -> list[Callable[[float], float]]:
    """解析基線外部場 mod：優先使用者提供；否則讀引擎現行 mod；再否則 identity。

    ``provided`` 可以是：None、callable（套所有節點）、scalar 常數、或
    length-n 的 callable 清單。回傳 length-n 的 ``t -> float`` callable 清單。
    """
    if provided is not None:
        if callable(provided):
            return [provided] * n
        if isinstance(provided, (int, float, np.number)):
            return [lambda t, v=float(provided): v] * n
        if isinstance(provided, (list, tuple)):
            if len(provided) != n:
                raise ValueError(f"base mod {attr} 長度 {len(provided)} != n_nodes {n}")
            if not all(callable(m) for m in provided):
                raise ValueError(f"base mod {attr} 每項必須是 callable")
            return list(provided)
        raise ValueError(f"base mod {attr} 必須是 callable / scalar / callable 清單")
    mods = None
    getter = getattr(dynamics, "current_modulation", None)
    if callable(getter):
        try:
            mod_a, mod_b = getter()
            mods = mod_a if attr == "_mod_alpha" else mod_b
        except Exception:
            mods = None
    if not (isinstance(mods, (list, tuple)) and len(mods) == n and all(callable(m) for m in mods)):
        mods = getattr(dynamics, attr, None)
    if isinstance(mods, (list, tuple)) and len(mods) == n and all(callable(m) for m in mods):
        return list(mods)
    return [lambda t: 1.0] * n


# ---------------------------------------------------------------------------
# 處境組裝器（Mode B 委派——probe → SituationSpec 形狀）
# ---------------------------------------------------------------------------


def _json_scalar(value: Any) -> Any:
    """numpy 陣列/純量 → JSON 安全型別（陣列→list，純量→item）。"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_float_list(value: Any) -> list[float] | None:
    """長度-n 陣列/純量 → float list；None/空/不可轉換 → None。"""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.size == 0:
        return None
    return [float(v) for v in arr.reshape(-1)]


def _fmt_arr(value: Any) -> str:
    """陣列/純量 → 短文字（digest 用）。"""
    vals = _as_float_list(value)
    if vals is None:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.4g}"
    return "[" + ", ".join(f"{v:.4g}" for v in vals) + "]"


def _fmt_criteria(crit: Any) -> str:
    """撕裂準則（字串/字串清單/其他）→ 短文字。"""
    if crit is None:
        return "-"
    if isinstance(crit, (list, tuple)):
        return ", ".join(str(c) for c in crit)
    return str(crit)


def _tear_node(state: Mapping, node_ids: Sequence[str], tension: Any) -> dict | None:
    """撕裂節點：張力最高（或 |S−0.5| 最大）的節點；無法判定回 None。"""
    t = _as_float_list(tension)
    if t is not None and len(t) > 0:
        idx = int(np.argmax(t))
        return {
            "node_id": node_ids[idx] if idx < len(node_ids) else str(idx),
            "index": idx,
            "tension": t[idx],
        }
    S = _as_float_list(state.get("S"))
    if S is not None and len(S) > 0:
        idx = int(np.argmax(np.abs(np.asarray(S) - 0.5)))
        return {
            "node_id": node_ids[idx] if idx < len(node_ids) else str(idx),
            "index": idx,
            "s": S[idx],
        }
    return None


def _derive_vector_6d(
    S: Any, tension: Any, status: str | None, tear_info: Mapping
) -> dict:
    """從 ODE 撕裂觀測機械推導 6D 向量（d1/d2/d3 三進制方向 + 主週期/振幅/張力）。

    語義（與 alpha 場 situation yaml 約定一致）：
    - d1：危機方向——撕裂張力高（>0.5）→ 1（顯現），否則 −1。
    - d2：制度穩定方向——S > 0.5（R 主導）→ 1，S < 0.5（C 主導=穩定）→ −1。
    - d3：意外方向——撕裂且帶準則 → 1（意外顯現），否則 −1。
    - d4/d5/d6：主週期（未知 0）/振幅（未知 0）/頻譜張力（= 平均張力）。
    """
    s_mean = float(np.mean(S)) if S is not None and len(S) else 0.5
    tens = float(np.mean(tension)) if tension is not None and len(tension) else 0.0
    d1 = 1 if tens > 0.5 else -1
    d2 = 1 if s_mean > 0.5 else -1
    d3 = 1 if status == "tear" and tear_info.get("criteria") else -1
    return {"d1": d1, "d2": d2, "d3": d3, "d4": 0, "d5": 0.0, "d6": round(tens, 4)}


def assemble_situation(
    probe: Mapping,
    *,
    actors: Mapping[str, Any] | None = None,
    world_digest: str | None = None,
    local_texture_extra: Mapping | None = None,
    vector_6d: Mapping | None = None,
    node_ids: Sequence[str] | None = None,
) -> dict:
    """處境組裝器——把一輪探針輸出物化為 SituationSpec 形狀的 dict。

    回傳鍵（``SituationSpec.to_dict()`` 契約，probe.py 可消費）：
    - ``digest``：撕裂時刻的世界狀態敘述（撕裂狀態 + 殘差準則 + 當輪 S/R/C/P_R；
      有 ``world_digest`` 時置首行）。
    - ``local_texture``：結構化參數——撕裂節點（``tear_node``）、張力
      （``tension``）、P_R 釋放狀態（``p_r_release``）、終態 S/R/C、撕裂準則
      （``criteria``）；``local_texture_extra`` 合併覆寫。
    - ``6d_vector``：含 ``d1``/``d2``/``d3`` 的 dict——``vector_6d`` 給定則透傳，
      否則從 ODE 觀測機械推導（``_derive_vector_6d``）。
    - ``actors``：``{aid: ActorCard.to_dict()}``（含 revolutionary_force /
      conservative_force / latent_force）——``actors`` 值可為 ``ActorCard`` 或
      已序列化 dict（直接透傳）。

    素材來源：alpha 場的 situation yaml + actors/*.yaml（經 ``load_field_actors``
    載入後餵入）+ 當輪 ODE 軌跡（probe）。本函式零 YAML、零 LLM——世界無關，
    素材由呼叫方組裝餵入。
    """
    ti = probe.get("tear_info") or {}
    state = probe.get("state") or {}
    status = probe.get("status")
    n = probe.get("round")
    at = ti.get("at_time")

    R = _as_float_list(state.get("R"))
    C = _as_float_list(state.get("C"))
    S = _as_float_list(state.get("S"))
    P = _as_float_list(state.get("P"))
    tension = _as_float_list(state.get("tension"))

    if node_ids is not None:
        ids = [str(x) for x in node_ids]
    else:
        width = max(len(S) if S is not None else 0, 1)
        ids = [f"node{i}" for i in range(width)]

    local_texture: dict[str, Any] = {
        "round": n,
        "status": status,
        "at_time": at,
        "tear_node": _tear_node(state, ids, tension),
        "tension": tension,
        "p_r_release": P,
        "s_final": S,
        "r_final": R,
        "c_final": C,
        "criteria": _json_scalar(ti.get("criteria")),
    }
    if local_texture_extra:
        local_texture.update(dict(local_texture_extra))

    if vector_6d is not None:
        vec: dict = dict(vector_6d)
    else:
        vec = _derive_vector_6d(S, tension, status, ti)

    parts: list[str] = []
    if world_digest and str(world_digest).strip():
        parts.append(str(world_digest).strip())
    at_s = f"{float(at):g}" if at is not None else "-"
    parts.append(f"第 {n} 輪（round_loop）撕裂 @t={at_s}，status={status}。")
    parts.append(
        f"社會內容合矢量 S={_fmt_arr(S)}（轉型力 R={_fmt_arr(R)}，慣性力 "
        f"C={_fmt_arr(C)}，潛在轉型力 P_R={_fmt_arr(P)}）。"
    )
    parts.append(f"撕裂準則：{_fmt_criteria(ti.get('criteria'))}。")

    actor_dicts: dict[str, Any] = {}
    if actors:
        for aid, card in actors.items():
            if isinstance(card, ActorCard):
                actor_dicts[str(aid)] = card.to_dict()
            elif isinstance(card, dict):
                actor_dicts[str(aid)] = dict(card)

    return {
        "digest": "\n".join(parts),
        "local_texture": local_texture,
        "6d_vector": vec,
        "actors": actor_dicts,
    }


def _actor_card_from_dict(data: Mapping) -> ActorCard:
    """把 alpha 場 actor yaml（dict）→ ActorCard（world-agnostic 契約）。"""
    fb = data.get("force_balance_1900") or data.get("force_balance") or {}
    tensions = data.get("internal_tensions")
    if not isinstance(tensions, list) or not tensions:
        raise ValueError(
            f"actor {data.get('actor_id')!r} internal_tensions 不能為空"
            "（ActorCard 契約——每個 actor 必須有內部張力）"
        )

    def _opt_float(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return ActorCard(
        actor_id=str(data["actor_id"]),
        name=str(data.get("name") or data["actor_id"]),
        inherited_conditions=dict(data.get("inherited_conditions") or {}),
        field_coordinates=dict(data.get("field_coordinates") or {}),
        internal_tensions=list(tensions),
        temporal_states=dict(data.get("temporal_states") or {}),
        revolutionary_force=_opt_float(fb.get("revolutionary_force")),
        conservative_force=_opt_float(fb.get("conservative_force")),
        latent_force=_opt_float(fb.get("latent_force")),
        meta=dict(data.get("meta") or {}),
    )


def load_field_actors(
    field_dir: str | os.PathLike,
    situation_path: str | os.PathLike | None = None,
) -> tuple[str | None, dict, dict | None, dict[str, ActorCard]]:
    """從「場目錄」載入處境素材（situation yaml + ``actors/*.yaml``）。

    🔴 世界無關：路徑由呼叫方給（ECC alpha 場 = continuum/alpha/field 目錄）。
    ``yaml`` 在函式內 lazy import——round_loop 模組層維持零依賴。

    - ``situation_path`` 給定 → 讀該檔；否則掃描 ``field_dir`` 下的
      ``situation*.yaml``（唯一才用；多個 → 要求顯式）。
    - ``actors/`` 目錄下每個 ``*.yaml`` → ``ActorCard``（含 ``force_balance_1900``
      的 revolutionary/conservative/latent force）。

    回傳 ``(digest, local_texture, vector_6d, actors)``——可直接餵給
    ``assemble_situation`` 的 ``world_digest``/``local_texture_extra``/
    ``vector_6d``/``actors``。
    """
    import yaml  # lazy——不進模組層 import 面

    d = Path(field_dir)
    sit_path = situation_path
    if sit_path is None:
        cands = sorted(d.glob("situation*.yaml"))
        if len(cands) == 1:
            sit_path = cands[0]
        elif len(cands) > 1:
            raise ValueError(
                f"field_dir 有多個 situation*.yaml（{cands}）——必須顯式給 situation_path"
            )

    digest: str | None = None
    local_texture: dict = {}
    vector_6d: dict | None = None
    if sit_path is not None and Path(sit_path).exists():
        with open(sit_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if isinstance(data, dict):
            if isinstance(data.get("digest"), str):
                digest = data["digest"]
            if isinstance(data.get("local_texture"), dict):
                local_texture = dict(data["local_texture"])
            if isinstance(data.get("vector_6d"), dict):
                vector_6d = dict(data["vector_6d"])

    actors_dir = d / "actors"
    actors: dict[str, ActorCard] = {}
    if actors_dir.is_dir():
        for f in sorted(actors_dir.glob("*.yaml")):
            with open(f, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict) or not data.get("actor_id"):
                continue
            aid = str(data["actor_id"])
            actors[aid] = _actor_card_from_dict(data)
    return digest, local_texture, vector_6d, actors


def _default_branch_to_decision(branch: Any, result: Mapping) -> dict:
    """預設 branch → decision 對映。

    branch 攜帶 ``mod_alpha``/``mod_beta``/``state`` 鍵 → 直接對映（probe 分支
    可選擇性攜帶；測試/mock 用）。``accepted_path`` = branch label/path；
    ``rejected_paths`` = collapse.unselected（維持疊加）。branch 非 dict → 空 decision。
    """
    decision: dict = {}
    if not isinstance(branch, dict):
        return decision
    for key in ("mod_alpha", "mod_beta", "state"):
        if key in branch:
            decision[key] = branch[key]
    label = branch.get("label") or branch.get("path")
    if label:
        decision["accepted_path"] = str(label)
    collapse = result.get("collapse") or {}
    unselected = collapse.get("unselected") or []
    if unselected:
        decision["rejected_paths"] = [str(u) for u in unselected]
    return decision


class RoundLoop:
    """回合協定編排器——把 Mode A 動力學、撕裂偵測、人選結算、寫回串成閉合迴圈。

    純 Python 編排，零 LLM、零 prompt：``dynamics.step`` 是唯一動力學接口，
    ``residual_trigger`` 是 ``(t, context) -> bool`` 插拔探針，``settlement_fn``
    是 ``(probe_payload) -> decision`` 純函數（人選 callback，測試用 mock）。

    Parameters
    ----------
    dynamics
        ``ForceFieldDynamics`` 或任何有 ``step(dt) -> snapshot dict`` 接口的引擎。
        快照鍵至少含 ``t``/``R``/``C``/``S``（``kernel.forces`` 的 step 快照鍵：
        ``t``/``R``/``C``/``P``/``S``/``T``/``tension``）。
    residual_trigger
        撕裂探針：callable ``(t, context) -> bool``（saturation/decoupling——
        見 ``residual_trigger.SHARP_CRITERIA``），或一張殘差表 Mapping（自動以
        ``make_residual_trigger`` 包裝，``key_fn`` 可用於 date-keyed 表）。
        **撕裂即停**：任一歩進前回報 True → 立刻停，不跨過撕裂點。探針可把
        ``context["residual"]`` 寫入附加的撕裂細節（會被收進 ``tear_info["criteria"]``）。
    settlement_fn
        人選 callback：``(probe_payload) -> decision dict``。``decision`` 鍵：
        - ``mod_alpha`` / ``mod_beta``：下一輪的外部場調製乘數（scalar 或 per-node），
          乘法合成於基線 mod 之上；非負。
        - ``state``：節點狀態覆寫（``{"R"/"C"/"P": scalar|list, "t": float}``）。
        - ``accepted_path`` / ``rejected_paths``：接受/拒絶路徑（記錄於 FieldLog
          與 snapshot；路徑語義屬 Mode B，本編排器只記錄不判定）。
        None（不給）→ ``settle(decision)`` 必須顯式傳 decision。
    store
        持久化（可選）：None = 純記憶體（測試用）；dict
        ``{"field_log_path": Path, "snapshot_dir": Path, "dashboard_path": Path}``
        （路徑給定才寫）；或一個帶 ``field_log_path``/``snapshot_dir`` 屬性 +
        ``save_field_log``/``save_round_snapshot`` 方法的 duck-typed 物件。
    field_log
        可選 ``FieldLog`` 實例；None → 自動建立（field_id 預設為 node_ids 或
        ``"round-loop"``）。
    field_id
        自動建立的 FieldLog 的 field_id（field_log 給定時忽略）。
    max_steps
        每輪 Mode A 步進上限（``run_one_round(max_steps=...)`` 可覆寫）。
    dt
        步進 dt；None → 沿用 ``dynamics._dt``（無則 1.0）。
    key_fn
        當 ``residual_trigger`` 是殘差表時，``(t, context) -> lookup key``
        （date-keyed 表需要，見 ``make_residual_trigger``）。
    context
        每步傳給 trigger 的基底 context（與 round/t 合併；fresh dict per step）。
    base_mod_alpha, base_mod_beta
        基線外部場 mod（callable / scalar / callable 清單）；None → 讀引擎現行 mod。
    meta
        provenance dict，帶入每輪 result 的 ``meta``。
    probe_converge_fn
        🆕 Mode B 人選方向委派（可選）：callable ``(situation, constraint_field=None,
        **kw) -> converge_result``。給定（或 ``mode_b`` 給定）時，撕裂/輪界後
        ``settle()`` 走 Mode B 委派路徑——處境組裝器把探針輸出物化為 SituationSpec
        形狀 → 委派 → branch → decision（mod_alpha/mod_beta/state）→ 套用。
        未給定 → 維持機械 ``settlement_fn``（向後相容）。None 且 ``mode_b`` 給定 →
        首次使用時惰性載入 ``probe_converge_round``（convergence.py）。
    mode_b
        🆕 Mode B 委派設定 dict（可選）：
        - ``style``：``"probe"``（預設，人選方向）| ``"enumerate"``（意外枚舉）
          | callable ``(tear_info, probe) -> str``（撕裂語義驅動）。
        - ``expand_kwargs``：透傳 probe_converge_round 的 **expand_kwargs
          （n_branch/depth/w/gate_fn/...）。
        - ``selected_label_fn``：``(converge_result, ctx) -> label | None`` 人選點
          （None = 只展開不坍縮）。
        - ``branch_to_decision``：``(branch, converge_result, ctx) -> decision``
          （預設 ``_default_branch_to_decision``）。
        - ``actors`` / ``field_dir`` / ``world_digest`` / ``local_texture_extra`` /
          ``vector_6d``：處境組裝器素材（alpha 場 yaml 經 ``load_field_actors``）。
        - ``constraint_field``：委派約束場（None → 從探針組裝殘差表+軌跡）。
        - enumerate 形態：``mode_a_log_path`` / ``record_mode_a_history_fn`` /
          ``mode_a_posterior_init_fn`` / ``enumerate_fn`` / ``select_accident_fn`` /
          ``enumerate_kwargs``（§12.13 順序：record → posterior → enumerate）。
    """

    def __init__(
        self,
        dynamics: Any,
        residual_trigger: Callable[[Any, Any], bool] | Mapping,
        settlement_fn: Callable[[dict], dict] | None = None,
        *,
        store: Any = None,
        field_log: FieldLog | None = None,
        field_id: str | None = None,
        max_steps: int = 1000,
        dt: float | None = None,
        key_fn: Callable[[Any, Any], Any] | None = None,
        context: dict | None = None,
        base_mod_alpha: Any = None,
        base_mod_beta: Any = None,
        meta: dict | None = None,
        # 🆕 Mode B 委派（§2.8 對接）
        probe_converge_fn: Callable | None = None,
        mode_b: Mapping | None = None,
    ) -> None:
        if not callable(getattr(dynamics, "step", None)):
            raise TypeError(
                "dynamics 必須有 step(dt) -> snapshot dict 接口"
                f"（got {type(dynamics).__name__}）"
            )
        self._dynamics = dynamics

        if isinstance(residual_trigger, Mapping):
            self._trigger: Callable[[Any, Any], bool] = make_residual_trigger(
                residual_trigger, key_fn=key_fn
            )
        elif callable(residual_trigger):
            self._trigger = residual_trigger
        else:
            raise TypeError(
                "residual_trigger 必須是 (t, context) -> bool callable 或殘差表 Mapping，"
                f"got {type(residual_trigger).__name__}"
            )

        if settlement_fn is not None and not callable(settlement_fn):
            raise TypeError(
                f"settlement_fn 必須是 (probe_payload) -> decision callable，"
                f"got {type(settlement_fn).__name__}"
            )
        self._settlement_fn = settlement_fn

        self._n = max(int(getattr(dynamics, "n_nodes", 1)), 1)

        if dt is None:
            dt = float(getattr(dynamics, "_dt", 1.0))
        if dt <= 0:
            raise ValueError(f"dt 必須為正，got {dt}")
        self._dt = float(dt)

        max_steps = int(max_steps)
        if max_steps < 1:
            raise ValueError(f"max_steps 必須 ≥ 1，got {max_steps}")
        self._max_steps = max_steps

        # 基線外部場 + 每輪調製層（乘法合成，呼叫時讀取 → settle 立即生效）。
        self._base_mod_alpha = _resolve_base_mods(base_mod_alpha, dynamics, "_mod_alpha", self._n)
        self._base_mod_beta = _resolve_base_mods(base_mod_beta, dynamics, "_mod_beta", self._n)
        self._round_mod_alpha = np.ones(self._n, dtype=np.float64)
        self._round_mod_beta = np.ones(self._n, dtype=np.float64)
        self._install_mod_layer()

        if field_log is not None and not isinstance(field_log, FieldLog):
            raise TypeError(
                f"field_log 必須是 FieldLog 實例（contracts.py），"
                f"got {type(field_log).__name__}"
            )
        if field_log is None:
            node_ids = getattr(dynamics, "node_ids", None)
            fid = field_id or ("-".join(node_ids) if node_ids else "round-loop")
            field_log = FieldLog(field_id=fid)
        self._field_log = field_log

        if store is not None and not isinstance(store, dict) and not (
            hasattr(store, "save_round_snapshot") or hasattr(store, "save_field_log")
        ):
            raise TypeError(
                "store 必須是 None / 路徑 dict / 帶 save_* 方法的物件，"
                f"got {type(store).__name__}"
            )
        self._store = store

        self._context = dict(context or {})
        self._meta = dict(meta or {})

        # 🆕 Mode B 委派（§2.8 對接 probe_converge_round / alt_gate_enumerate）。
        # 零 LLM import：委派函式由呼叫方注入（probe_converge_fn），或首次使用時
        # 以 importlib 惰性載入——模組層維持不觸發任何 api/LLM/prompt 模組。
        if probe_converge_fn is not None and not callable(probe_converge_fn):
            raise TypeError(
                f"probe_converge_fn 必須是 callable（(situation, **kw) -> dict），"
                f"got {type(probe_converge_fn).__name__}"
            )
        if mode_b is not None and not isinstance(mode_b, Mapping):
            raise TypeError(
                f"mode_b 必須是 dict（Mode B 委派設定），got {type(mode_b).__name__}"
            )
        self._probe_converge_fn = probe_converge_fn
        self._mode_b_config: dict = dict(mode_b or {})
        self._node_ids = [str(x) for x in (getattr(dynamics, "node_ids", None) or [])]
        self._field_actors_cache: tuple | None = None
        self._mode_a_history_buffer: list[dict] = []
        self._mode_a_tmp_path: str | None = None
        self._residual_table: Mapping | None = (
            residual_trigger if isinstance(residual_trigger, Mapping) else None
        )

        # 回合狀態機：round 數 = 已結算輪數；phase = ready | awaiting_settlement。
        self._round = 0
        self._phase = "ready"
        self._history: list[dict] = []
        self._settlement_history: list[dict] = []
        self._last_result: dict | None = None
        self._last_payload: dict | None = None
        self._pending_state: dict = {}

    # ------------------------------------------------------------------
    # 外部場調製注入
    # ------------------------------------------------------------------

    def _install_mod_layer(self) -> None:
        """把引擎執行期 mod callable 換成「基線 × 每輪調製」合成（乘法、呼叫時讀取）。

        forces.py 無公開 mod setter——編排器作為引擎的合法驅動方，組合其執行期
        ``_mod_alpha``/``_mod_beta``。合成在呼叫時才讀 ``_round_mod_*``，因此
        ``settle()`` 一落地，``effective_rates`` 立即反映，無需重建引擎。
        generic 引擎（無 ``_mod_alpha``/``_mod_beta``）→ 跳過，mod 層留在
        RoundLoop 內（``round_modulation()`` 可取）。
        """
        dyn = self._dynamics
        setter = getattr(dyn, "set_modulation", None)
        if not callable(setter):
            return
        try:
            setter(
                mod_alpha=[
                    self._composed_mod(b, i, "alpha")
                    for i, b in enumerate(self._base_mod_alpha)
                ],
                mod_beta=[
                    self._composed_mod(b, i, "beta")
                    for i, b in enumerate(self._base_mod_beta)
                ],
            )
        except Exception:
            pass

    def _composed_mod(self, base: Callable[[float], float], i: int, which: str) -> Callable[[float], float]:
        """合成 callable：``base(t) × round_mod[i]``（round_mod 呼叫時讀取）。"""

        def mod(t: float) -> float:
            arr = self._round_mod_alpha if which == "alpha" else self._round_mod_beta
            return float(base(t)) * float(arr[i])

        return mod

    # ------------------------------------------------------------------
    # 快照 / 序列化小工具
    # ------------------------------------------------------------------

    def _snapshot(self) -> dict:
        """目前引擎狀態的完整快照（ForceFieldDynamics 友好；generic 引擎容錯降級）。"""
        dyn = self._dynamics
        out: dict = {"t": getattr(dyn, "t", None)}
        try:
            R, C = dyn.state
            out["R"], out["C"] = R.copy(), C.copy()
        except Exception:
            out["R"] = out["C"] = None
        try:
            out["P"] = dyn.latent_state.copy()
        except Exception:
            out["P"] = None
        for key, meth in (("S", "s"), ("T", "total_capacity"), ("tension", "tension")):
            try:
                out[key] = np.asarray(getattr(dyn, meth)(), dtype=np.float64)
            except Exception:
                out[key] = None
        return out

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """遞迴把 numpy 值轉成 JSON 安全型別（FieldLog / snapshot 落盤用）。"""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: RoundLoop._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [RoundLoop._json_safe(v) for v in value]
        return value

    # ------------------------------------------------------------------
    # 回合驅動：run_one_round → settle（閉合迴圈）
    # ------------------------------------------------------------------

    def run_one_round(self, max_steps: int | None = None) -> dict:
        """Mode A 步進直到撕裂或 max_steps（§2.8 步驟 1–2 + 探針負載）。

        每步先查撕裂（步進前）——trigger 回報撕裂即停，軌跡不含任何超過撕裂點
        的狀態。回傳：

        - ``round``：本輪編號（已結算輪數 + 1）。
        - ``status``：``"tear"`` | ``"max_steps"``。
        - ``n_steps``：本輪實際步數。
        - ``tear_info``：``{"triggered": True, "at_time", "round",
          "criteria"}``（status=max_steps 時為 None）。
        - ``start_state`` / ``final_state``：本輪起點 / 停點快照。
        - ``trajectory``：本輪每步的快照串列（含撕裂時刻本身，但不含其後）。
        - ``s_series``：S(t) 序列（每步的 ``S`` 陣列）。
        - ``probe``：探針負載（原樣交給 ``settlement_fn`` / ``settle()``）。
        - ``meta``：dt / max_steps / node_ids / trigger_errors 等。

        🔴 撕裂是輪的邊界：上一輪未結算（phase=awaiting_settlement）時呼叫 →
        ``RoundProtocolError``。必須先 ``settle()`` 才能進入下一輪。
        """
        if self._phase != "ready":
            raise RoundProtocolError(
                f"第 {self._round} 輪未結算——必須先 settle() 才能 run_one_round()"
                "（§2.8：撕裂→結算→寫回→下一輪）"
            )
        if max_steps is None:
            max_steps = self._max_steps
        max_steps = int(max_steps)
        if max_steps < 1:
            raise ValueError(f"max_steps 必須 ≥ 1，got {max_steps}")

        n = self._round + 1
        dyn = self._dynamics

        start_state = self._snapshot()
        trajectory: list[dict] = []
        status = "max_steps"
        tear_info: dict | None = None
        tear_ctx: dict | None = None
        trigger_errors = 0

        for _ in range(max_steps):
            t = getattr(dyn, "t", None)
            ctx = dict(self._context)
            ctx["round"] = n
            ctx["t"] = t
            try:
                fired = bool(self._trigger(t, ctx))
            except Exception:
                # trigger 是插拔探針——失敗不應使整輪崩潰，視為未觸發（計數）。
                fired = False
                trigger_errors += 1
            if fired:
                status = "tear"
                tear_info = {
                    "triggered": True,
                    "at_time": t,
                    "round": n,
                    "criteria": ctx.get("residual"),
                }
                tear_ctx = ctx
                break
            trajectory.append(dyn.step(self._dt))

        final_state = trajectory[-1] if trajectory else self._snapshot()

        probe = {
            "round": n,
            "status": status,
            "tear_info": tear_info,
            "state": final_state,
            "trajectory": trajectory,
            "s_series": [s["S"] for s in trajectory],
            "context": (
                tear_ctx
                if tear_ctx is not None
                else dict(self._context, round=n)
            ),
        }
        result = {
            "round": n,
            "status": status,
            "n_steps": len(trajectory),
            "tear_info": tear_info,
            "start_state": start_state,
            "final_state": final_state,
            "trajectory": trajectory,
            "s_series": probe["s_series"],
            "probe": probe,
            "meta": {
                "dt": self._dt,
                "max_steps": max_steps,
                "node_ids": getattr(dyn, "node_ids", None),
                "trigger_errors": trigger_errors,
                "round_loop": "spectrum_os.synth.round_loop",
                **self._meta,
            },
        }
        self._phase = "awaiting_settlement"
        self._last_result = result
        self._last_payload = probe
        self._history.append(result)
        return result

    def settle(self, decision: dict | None = None) -> dict:
        """人選結算 → 寫回 → 準備下一輪（§2.8 步驟 4–5）。

        ``decision``：人選回傳的決策 dict。None → 內部呼叫 ``settlement_fn``
        （``settlement_fn(probe_payload)``）；兩者皆無 → ``RoundProtocolError``。

        決策鍵（都選填）：
        - ``mod_alpha`` / ``mod_beta``：下一輪外部場調製乘數（乘法合成，立即生效）。
        - ``state``：節點狀態覆寫（``R``/``C``/``P``/``t``）。
        - ``accepted_path`` / ``rejected_paths``：接受/拒絶路徑（記錄於 FieldLog）。

        寫回：FieldLog 追加本輪 entry（round / decision / mod_alpha / mod_beta /
        accepted / rejected）；store 給定時另存 field log + round snapshot
        （＋可選 dashboard）。

        回傳 ``{"round", "applied", "written", "decision"}``——``applied`` 是實際
        生效的調整，``written`` 是本輪寫出的檔案路徑 dict。
        """
        if self._phase != "awaiting_settlement":
            raise RoundProtocolError(
                "沒有待結算的輪——先 run_one_round() 再 settle()"
            )
        n = self._round + 1
        # 🆕 Mode B 委派路徑：未給顯式 decision 且啟用委派 → 交 probe_converge_fn
        # （expand → 人選 → 坍縮 → FieldLog/snapshot 統一寫回）。顯式 decision =
        # 人類覆寫 → 走機械路徑（向後相容）。
        if decision is None and self.mode_b_active:
            return self._settle_via_mode_b(n)
        if decision is None:
            if self._settlement_fn is None:
                raise RoundProtocolError(
                    "settle() 未給 decision 且無 settlement_fn——兩者至少其一"
                )
            decision = self._settlement_fn(self._last_payload)
        if decision is None:
            decision = {}
        if not isinstance(decision, dict):
            raise TypeError(
                f"decision 必須是 dict，got {type(decision).__name__}"
            )

        applied = self._apply_decision(decision)

        accepted = decision.get("accepted_path") or decision.get("path")
        rejected = decision.get("rejected_paths") or decision.get("rejected_path") or []
        if not isinstance(rejected, (list, tuple)):
            rejected = [rejected]

        # FieldLog 契約：add_entry(layer, selected, unselected)——selected 是結構化
        # 結算條目（不可把 decision 整包塞進 selected），unselected 維持疊加。
        entry = {
            "round": n,
            "accepted": accepted,
            "rejected": [str(p) for p in rejected],
            "mod_alpha": self._round_mod_alpha.tolist(),
            "mod_beta": self._round_mod_beta.tolist(),
            "state": applied.get("state"),
        }
        self._field_log.add_entry(n, entry, [{"label": str(p)} for p in rejected])

        written = self._persist_round(n, entry, accepted)

        self._round += 1
        self._phase = "ready"
        record = {
            "round": n,
            "applied": applied,
            "written": written,
            "decision": self._json_safe(decision),
        }
        self._settlement_history.append(record)
        return record

    # ------------------------------------------------------------------
    # 結算的落地細節
    # ------------------------------------------------------------------

    def _set_round_mod(self, which: str, value: Any) -> np.ndarray:
        """設定下一輪外部場調製乘數（非負；合成 callable 呼叫時讀取 → 立即生效）。"""
        arr = _broadcast_force(value, self._n, f"decision.mod_{which}")
        if np.any(arr < 0):
            raise ValueError(
                f"mod_{which} 必須非負（外部場乘數不可為負），got {arr.tolist()}"
            )
        if which == "alpha":
            self._round_mod_alpha = arr
        else:
            self._round_mod_beta = arr
        return arr

    def _apply_state(self, state: Any) -> dict:
        """套用節點狀態覆寫（R/C/P/t）到引擎執行期狀態。

        generic 引擎（無 ``_R``/``_C`` 等）→ 只記錄於 ``pending_state``，不崩潰。
        """
        if not isinstance(state, dict):
            raise TypeError(
                f"decision['state'] 必須是 dict，got {type(state).__name__}"
            )
        dyn = self._dynamics
        applied: dict = {}
        for key in ("R", "C", "P"):
            if key in state:
                arr = _broadcast_force(state[key], self._n, f"state.{key}")
                arr = np.maximum(arr, 0.0)
                if hasattr(dyn, f"_{key}"):
                    setattr(dyn, f"_{key}", arr)
                applied[key] = arr.tolist()
        if "t" in state:
            t = float(state["t"])
            if hasattr(dyn, "_t"):
                dyn._t = t
            applied["t"] = t
        self._pending_state = dict(state)
        return applied

    def _persist_round(self, n: int, field_log_entry: dict, accepted: Any) -> dict:
        """store 給定時，本輪寫回 field log + round snapshot（＋dashboard）。"""
        written: dict = {}
        if self._store is None:
            return written
        if isinstance(self._store, dict):
            fl_path = self._store.get("field_log_path")
            snap_dir = self._store.get("snapshot_dir")
            dash_path = self._store.get("dashboard_path")
        else:
            fl_path = getattr(self._store, "field_log_path", None)
            snap_dir = getattr(self._store, "snapshot_dir", None)
            dash_path = getattr(self._store, "dashboard_path", None)

        if fl_path is not None:
            written["field_log"] = str(save_field_log(self._field_log, fl_path))
        if snap_dir is not None:
            last = self._last_result or {}
            round_result = {
                "round": n,
                "view": self._render_view(last),
                "collapse": None,
                "path": [str(accepted)] if accepted else [],
                "field_log_entry": field_log_entry,
                "state_log_entry": {
                    "round": n,
                    "status": last.get("status"),
                    "n_steps": last.get("n_steps"),
                    "tear_at": (last.get("tear_info") or {}).get("at_time"),
                },
            }
            written["snapshot"] = str(save_round_snapshot(round_result, n, snap_dir))
        if dash_path is not None:
            written["dashboard"] = str(save_dashboard(self._render_view(last), dash_path))
        return written

    # ------------------------------------------------------------------
    # 🆕 Mode B 委派（§2.8 對接——probe_converge_round / alt_gate_enumerate）
    # ------------------------------------------------------------------

    def assemble_situation(self, probe: Mapping | None = None) -> dict:
        """處境組裝器（公開）——把最近一輪（或給定）探針輸出物化為 SituationSpec 形狀。

        委派給模組級 ``assemble_situation``，並把 mode_b 設定裡的 ``field_dir`` /
        ``actors`` / ``world_digest`` / ``local_texture_extra`` / ``vector_6d``
        一起餵入。回傳 ``{"digest", "local_texture", "6d_vector", "actors"}``
        （probe.py 可消費）。
        """
        if probe is None:
            if self._last_payload is None:
                raise RoundProtocolError(
                    "尚無探針輸出——先 run_one_round() 再 assemble_situation()"
                )
            probe = self._last_payload
        return self._assemble_situation_for(probe)

    def _assemble_situation_for(self, probe: Mapping) -> dict:
        cfg = self._mode_b_config
        actors = cfg.get("actors")
        world_digest = cfg.get("world_digest")
        lt_extra = cfg.get("local_texture_extra")
        vec = cfg.get("vector_6d")
        field_dir = cfg.get("field_dir")
        if actors is None and field_dir is not None:
            if self._field_actors_cache is None:
                self._field_actors_cache = load_field_actors(field_dir)
            fdigest, flt, fvec, fcards = self._field_actors_cache
            actors = fcards
            if world_digest is None:
                world_digest = fdigest
            if lt_extra is None:
                lt_extra = flt
            if vec is None:
                vec = fvec
        return assemble_situation(
            probe,
            actors=actors,
            world_digest=world_digest,
            local_texture_extra=lt_extra,
            vector_6d=vec,
            node_ids=self._node_ids,
        )

    def _apply_decision(self, decision: Mapping) -> dict:
        """套用決策的外部場調製 / 狀態覆寫（機械 + Mode B 共用）。"""
        applied: dict = {}
        if "mod_alpha" in decision:
            arr = self._set_round_mod("alpha", decision["mod_alpha"])
            applied["mod_alpha"] = arr.tolist()
        if "mod_beta" in decision:
            arr = self._set_round_mod("beta", decision["mod_beta"])
            applied["mod_beta"] = arr.tolist()
        if "state" in decision:
            applied["state"] = self._apply_state(decision["state"])
        return applied

    def _resolve_style(self, probe: Mapping) -> str:
        """Mode B 形態：``"probe"``（人選方向）| ``"enumerate"``（意外枚舉）。

        ``mode_b["style"]`` 可為字串或 ``(tear_info, probe) -> str`` callable
        （撕裂語義驅動）。預設 "probe"。
        """
        style = self._mode_b_config.get("style", "probe")
        if callable(style):
            return str(style(probe.get("tear_info"), probe))
        return "enumerate" if style == "enumerate" else "probe"

    def _resolve_probe_converge(self) -> Callable:
        """人選方向委派函式：給定直接用；否則惰性載入 probe_converge_round。

        惰性（importlib.import_module）——round_loop 模組層維持零 LLM import；
        只有實際走 Mode B 委派時才載入 convergence（內部含 LLM-gated probe）。
        """
        if self._probe_converge_fn is not None:
            return self._probe_converge_fn
        mod = importlib.import_module("spectrum_os.synth.convergence")
        self._probe_converge_fn = mod.probe_converge_round
        return self._probe_converge_fn

    def _resolve_delegate(self, key: str, module: str, attr: str) -> Callable:
        """Mode B 設定內可注入的委派函式；None → 惰性載入 ``module.attr``。"""
        fn = self._mode_b_config.get(key)
        if fn is not None:
            return fn
        mod = importlib.import_module(f"spectrum_os.synth.{module}")
        return getattr(mod, attr)

    def _mb_constraint_field(self, probe: Mapping) -> Any:
        """Mode B 委派的約束場：設定給定 → 用設定；否則從探針組裝（殘差表 + 軌跡）。"""
        cf = self._mode_b_config.get("constraint_field")
        if cf is not None:
            return cf
        return {
            "round": probe.get("round"),
            "status": probe.get("status"),
            "tear_info": RoundLoop._json_safe(probe.get("tear_info")),
            "residual_table": (
                dict(self._residual_table) if self._residual_table is not None else None
            ),
            "s_series": RoundLoop._json_safe(probe.get("s_series")),
        }

    def _delegate_write_kwargs(self) -> dict:
        """把 store 的落盤路徑轉為 delegate（probe_converge_round）的寫回參數。"""
        if self._store is None:
            return {}
        if isinstance(self._store, dict):
            fl = self._store.get("field_log_path")
            sd = self._store.get("snapshot_dir")
            db = self._store.get("dashboard_path")
        else:
            fl = getattr(self._store, "field_log_path", None)
            sd = getattr(self._store, "snapshot_dir", None)
            db = getattr(self._store, "dashboard_path", None)
        out: dict = {}
        if fl is not None:
            out["field_log_path"] = str(fl)
        if sd is not None:
            out["snapshot_dir"] = str(sd)
        if db is not None:
            out["dashboard_path"] = str(db)
        return out

    def _select_label(self, result: Mapping, ctx: dict) -> str | None:
        """人選點：``selected_label_fn(converge_result, ctx) -> label | None``。"""
        fn = self._mode_b_config.get("selected_label_fn")
        if fn is None:
            return None
        return fn(result, ctx)

    def _branch_to_decision(self, branch: Any, result: Mapping, ctx: dict) -> dict:
        """branch → decision 對映：設定給定 → 用設定；否則預設對映。"""
        fn = self._mode_b_config.get("branch_to_decision")
        if fn is not None:
            return fn(branch, result, ctx)
        return _default_branch_to_decision(branch, result)

    def _mode_b_probe(self, probe: Mapping) -> tuple[dict, dict]:
        """人選方向（Mode B 形態一）：委派 probe_converge_round——expand → 人選 → 坍縮
        → FieldLog/snapshot 統一寫回。回傳 ``(decision, converge_result)``。

        兩段式（§2.8 人機接口）：第一呼只展開給人看（selected_label=None，不寫）；
        ``selected_label_fn`` 決定選哪條（None = 只展開不坍縮）；第二呼帶
        selected_label + field_log + 落盤路徑 → delegate 完成坍縮與統一寫回。
        """
        n = probe.get("round")
        situation = self._assemble_situation_for(probe)
        constraint_field = self._mb_constraint_field(probe)
        ctx = {
            "round": n,
            "probe": probe,
            "situation": situation,
            "constraint_field": constraint_field,
        }
        converge = self._resolve_probe_converge()
        kw = dict(self._mode_b_config.get("expand_kwargs") or {})

        expand_only = converge(
            situation=situation,
            constraint_field=constraint_field,
            selected_label=None,
            field_log=None,
            **kw,
        )
        label = self._select_label(expand_only, ctx)
        if label is None:
            # 只展開不坍縮——本輪無人選（無決策、不寫回）。
            return {}, expand_only

        converge_result = converge(
            situation=situation,
            constraint_field=constraint_field,
            selected_label=label,
            field_log=self._field_log,  # 統一寫回：delegate 寫入同一 FieldLog
            **self._delegate_write_kwargs(),
            **kw,
        )
        branch = (converge_result.get("field_log_entry") or {}).get("selected")
        decision = self._branch_to_decision(branch, converge_result, ctx)
        return decision, converge_result

    def _mode_a_log_path(self) -> str:
        """§12.13 Mode A 歷史 JSONL 路徑：設定給定 → 用之；否則每 loop 一個 temp。"""
        path = self._mode_b_config.get("mode_a_log_path")
        if path is not None:
            return str(path)
        if self._mode_a_tmp_path is None:
            fd, name = tempfile.mkstemp(prefix="round_loop_mode_a_", suffix=".jsonl")
            os.close(fd)
            self._mode_a_tmp_path = name
        return self._mode_a_tmp_path

    def _mode_a_history_entry(self, probe: Mapping) -> dict:
        """把當輪 ODE 軌跡寫成 Mode A 歷史條目（§12.13 ①，純機械）。"""
        ti = probe.get("tear_info") or {}
        state = probe.get("state") or {}
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "mode": "ode",
            "round": probe.get("round"),
            "role_sequence": [],
            "labels": list(self._node_ids),
            "rate_matrix": None,
            "spectrum": {
                "s_series": RoundLoop._json_safe(probe.get("s_series")),
                "final_S": RoundLoop._json_safe(state.get("S")),
            },
            "tear_info": RoundLoop._json_safe(ti),
            "source": "spectrum_os.synth.round_loop:mode_b",
        }

    def _enumerate_input(self, probe: Mapping, posterior: Any) -> dict:
        """意外枚舉的 input_data：SituationSpec 形狀 + §12.13 後驗（資料層注入）。"""
        data = dict(self._assemble_situation_for(probe))
        data["mode_a_history"] = posterior
        data["round"] = probe.get("round")
        return data

    def _select_accident(self, accidents: list, ctx: dict) -> dict | None:
        """意外人選：``select_accident_fn(accidents, ctx)``；None → 第一個。"""
        fn = self._mode_b_config.get("select_accident_fn")
        if fn is not None:
            return fn(accidents, ctx)
        return accidents[0] if accidents else None

    def _accident_to_decision(self, accident: Any, result: Mapping, ctx: dict) -> dict:
        """意外 → decision 對映：accepted = pattern（或首實例），rejected = 其餘。"""
        fn = self._mode_b_config.get("branch_to_decision")
        if fn is not None:
            return fn(accident, result, ctx)
        decision: dict = {}
        if not isinstance(accident, dict):
            return decision
        pattern = accident.get("pattern")
        instances = accident.get("instances") or []
        label = pattern or (instances[0] if instances else None)
        if label:
            decision["accepted_path"] = str(label)
        others: list[str] = []
        for a in result.get("accidents") or []:
            if a is accident:
                continue
            if not isinstance(a, dict):
                continue
            p = a.get("pattern") or ((a.get("instances") or [None])[0])
            if p:
                others.append(str(p))
        if others:
            decision["rejected_paths"] = others
        return decision

    def _mode_b_enumerate(self, probe: Mapping) -> tuple[dict, dict]:
        """意外枚舉（Mode B 形態二）：§12.13——先 record_mode_a_history（ODE 軌跡
        寫入）→ mode_a_posterior_init → alt_gate_enumerate。回傳 ``(decision, result)``。
        """
        ctx = {"round": probe.get("round"), "probe": probe}
        path = self._mode_a_log_path()
        entry = self._mode_a_history_entry(probe)
        record_fn = self._resolve_delegate(
            "record_mode_a_history_fn", "alt_gate", "record_mode_a_history"
        )
        record_fn(path, entry)
        self._mode_a_history_buffer.append(entry)

        posterior_fn = self._resolve_delegate(
            "mode_a_posterior_init_fn", "alt_gate", "mode_a_posterior_init"
        )
        posterior = posterior_fn(list(self._mode_a_history_buffer))

        input_data = self._enumerate_input(probe, posterior)
        ekw = dict(self._mode_b_config.get("enumerate_kwargs") or {})
        enum_fn = self._resolve_delegate("enumerate_fn", "alt_gate", "alt_gate_enumerate")
        result = enum_fn(
            input_data, mode_a_history=list(self._mode_a_history_buffer), **ekw
        )

        accidents = result.get("accidents") or []
        accident = self._select_accident(accidents, ctx)
        decision = self._accident_to_decision(accident, result, ctx)
        return decision, result

    def _settle_via_mode_b(self, n: int) -> dict:
        """Mode B 結算：委派 → 對映 decision → 套用 → 統一寫回（不重複寫）。"""
        probe = self._last_payload
        style = self._resolve_style(probe)
        if style == "enumerate":
            decision, delegated = self._mode_b_enumerate(probe)
        else:
            decision, delegated = self._mode_b_probe(probe)
        if decision is None:
            decision = {}
        if not isinstance(decision, dict):
            raise TypeError(
                f"Mode B decision 必須是 dict，got {type(decision).__name__}"
            )
        applied = self._apply_decision(decision)
        # 統一寫回：FieldLog / snapshot 已由 delegate 落盤——round_loop 不重複寫。
        written = (delegated or {}).get("written", {})
        if not isinstance(written, dict):
            written = {}
        self._round += 1
        self._phase = "ready"
        record = {
            "round": n,
            "applied": applied,
            "written": written,
            "decision": self._json_safe(decision),
            "mode_b": {
                "style": style,
                "delegated": True,
                "delegate_wrote": bool(written),
            },
        }
        self._settlement_history.append(record)
        return record

    @staticmethod
    def _render_view(result: dict) -> str:
        """把一輪結果渲染成短文字（snapshot 的 view 鍵）。"""
        ti = result.get("tear_info") or {}
        at = ti.get("at_time")
        at_s = f"@{float(at):g}" if at is not None else "-"
        status = result.get("status", "?")
        if status == "tear":
            tail = f"tear {at_s}"
        else:
            tail = "max_steps"
        return (
            f"[round {result.get('round')}] status={status} · "
            f"steps={result.get('n_steps')} · {tail}"
        )

    # ------------------------------------------------------------------
    # 查詢
    # ------------------------------------------------------------------

    def rounds_completed(self) -> int:
        """已結算（settle 完成）的輪數。"""
        return self._round

    def phase(self) -> str:
        """回合狀態機相位：``"ready"``（可 run）| ``"awaiting_settlement"``。"""
        return self._phase

    @property
    def mode_b_active(self) -> bool:
        """是否啟用 Mode B 委派（``probe_converge_fn`` 或 ``mode_b`` 設定給定）。"""
        return self._probe_converge_fn is not None or bool(self._mode_b_config)

    def history(self) -> list[dict]:
        """所有已跑完（含待結算）的輪結果（含 ``trajectory`` 等 numpy 內容）。"""
        return list(self._history)

    def settlement_history(self) -> list[dict]:
        """所有已結算的記錄（round / applied / written / decision）。"""
        return list(self._settlement_history)

    def probe(self) -> dict | None:
        """最近一輪的探針負載（None = 尚未跑任何輪）。"""
        return self._last_payload

    def pending_state(self) -> dict:
        """最近一次 decision 的 state 覆寫（generic 引擎可在這裡取）。"""
        return dict(self._pending_state)

    @property
    def field_log(self) -> FieldLog:
        """本 loop 的 FieldLog（每輪結算追加一筆）。"""
        return self._field_log

    def current_state(self) -> dict:
        """目前引擎狀態快照（t/R/C/P/S/T/tension）。"""
        return self._snapshot()

    def round_modulation(self) -> tuple[np.ndarray, np.ndarray]:
        """目前每輪調製乘數 ``(mod_alpha, mod_beta)``（length-n，read-only copies）。"""
        return self._round_mod_alpha.copy(), self._round_mod_beta.copy()

    def effective_rates(self, t: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """合成後的抑制效率 ``(α_eff, β_eff)`` 在時刻 t（含每輪調製）。

        引擎支援 ``effective_rates``（ForceFieldDynamics）→ 直接委派（已反映
        合成 mod）。generic 引擎 → 回傳每輪調製乘數（不含 α⁰/β⁰ 基線——那是
        引擎的領域）。
        """
        dyn = self._dynamics
        if hasattr(dyn, "effective_rates"):
            return dyn.effective_rates(t)
        return self._round_mod_alpha.copy(), self._round_mod_beta.copy()


__all__ = [
    "RoundLoop",
    "RoundProtocolError",
    "assemble_situation",
    "load_field_actors",
]
