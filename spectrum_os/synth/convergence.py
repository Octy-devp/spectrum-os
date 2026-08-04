"""Human–machine convergence UX — PLAN-23 §11.7 / §12.7 (T19 N1).

把「人看樹 → 人選 → 寫回場 log」的收束流程（奇門的「門」——『沒有民主就沒有用』）
做成可用介面。**世界無關**：零 ECC 路徑 / 零世界特定詞。

關係（下游消費 T18/T17a 既有產物，不碰其既有函數）：
- ``probe_expand_layer``（probe.py，T18）展開一層，回傳 ``tree``（截至該層的工作樹，
  人看的對象）與 ``result``（``probe_select`` 可消費的最小形狀）。
- ``probe_select``（probe.py）人選一條 → 坍縮；未選分支標 ``unselected`` 寫回
  （**不刪除**，維持疊加）。
- ``contracts.FieldLog``（contracts.py，T17a）場的歷史 log——``add_entry(layer,
  selected, unselected)`` 把 selected + unselected 一起寫入，維持疊加。

命名說明：probe.py 已有 ``convergence_view(result)``——probe_tree 產物的**可程式化
資料視圖**（rigidity / archetype / gate 卡片，供程式消費）。本模組的
``convergence_view`` 是**樹的渲染函數**（format=text/markdown/json，人選方向前的
『看』）。兩者同名不同職——本模組**刻意不** re-export 進 ``synth/__init__.py``
（避免遮蔽 probe 版本）；使用方直接 ``from spectrum_os.synth.convergence import ...``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from spectrum_os.contracts import FieldLog  # noqa: F401  (type hint 用)
from spectrum_os.synth.convergence_store import (
    save_dashboard as _store_save_dashboard,
)
from spectrum_os.synth.convergence_store import (
    save_field_log as _store_save_field_log,
)
from spectrum_os.synth.convergence_store import (
    save_round_snapshot as _store_save_round_snapshot,
)
from spectrum_os.synth.probe import TreeProbeError, probe_expand_layer, probe_select

#: include 允許的欄位 → 人類可讀標籤（頻譜電腦自身詞彙，非任何世界的詞）。
#: label 是節點標題（不帶前綴渲染）；rigidity 對映 ``rigidity_prevalence``。
_INCLUDE_LABELS: dict[str, str] = {
    "label": "label",
    "perspective": "鏡角",
    "binding": "束縛",
    "grounding": "依憑",
    "rigidity": "剛性",
    "conditions": "條件",
    "roles": "roles",
}

_VALID_FORMATS: tuple[str, ...] = ("text", "markdown", "json")

_DEFAULT_INCLUDE: tuple[str, ...] = (
    "label", "perspective", "binding", "grounding", "rigidity",
)


def _resolve_tree(tree_or_result: dict) -> tuple[dict, dict | None]:
    """從 tree 或 probe result 抽出 ``(tree, collapse)``。

    - probe result（``{"trees": [...]}``）→ 取第一棵樹 + ``result["collapse"]``。
    - 裸 tree（``{"root": ..., "layers": [...]}``）→ 原樣 + 內嵌 ``collapse``。
    ``collapse`` 由 ``probe_select`` 寫入（``{"selected", "layer", "unselected"}``）。
    """
    if not isinstance(tree_or_result, dict):
        raise TypeError(
            "tree_or_result 必須是 dict（tree 或 probe result），"
            f"got {type(tree_or_result).__name__}"
        )
    if "trees" in tree_or_result:
        trees = tree_or_result["trees"]
        if not isinstance(trees, list) or not trees or not isinstance(trees[0], dict):
            raise TypeError("tree_or_result['trees'] 必須是非空 list[dict]")
        return trees[0], tree_or_result.get("collapse")
    if "root" in tree_or_result and "layers" in tree_or_result:
        return tree_or_result, tree_or_result.get("collapse")
    raise TypeError(
        "tree_or_result 既不是 tree（缺 root/layers）也不是 probe result（缺 trees）"
    )


def _selection_of(
    node: dict,
    collapse: dict | None,
    walked: set[str] | None = None,
) -> str | None:
    """分支的收束角色：None = 可選；'selected' / 'unselected'。

    - ``collapse`` 只標記其記錄的那一層（``collapse["layer"]``）——該層的
      selected → （已選）、unselected → （未選）；其餘層不受 collapse 影響。
    - ``walked``（歷輪已選 label 集合，N2 缺口 2）標記「人已經走過的路」——只要
      label 在集合內（不限層）即（已選）；collapse 層內的標記優先於 walked。
    """
    if collapse:
        layer = collapse.get("layer")
        if layer is not None and int(node.get("layer", -1)) == int(layer):
            label = node.get("label")
            if label == collapse.get("selected"):
                return "selected"
            if label in (collapse.get("unselected") or []):
                return "unselected"
    if walked and node.get("label") in walked:
        return "selected"
    return None


def _selection_marker(
    node: dict, collapse: dict | None, walked: set[str] | None = None
) -> str:
    """分支行的收束角色標記（人看的『這條在收束中的角色』）。"""
    role = _selection_of(node, collapse, walked)
    if role == "selected":
        return "（已選）"
    if role == "unselected":
        return "（未選）"
    if int(node.get("layer", 0)) > 0:
        return "（可選）"
    return ""  # root（layer 0）不是分支，不可選


def _field_parts(branch: dict, include: tuple[str, ...]) -> list[str]:
    """把 include 的欄位渲染為「標籤=值」片段（label 是標題，不在此列）。"""
    parts: list[str] = []
    for key in include:
        if key not in _INCLUDE_LABELS or key == "label":
            continue
        if key == "rigidity":
            value = branch.get("rigidity_prevalence", branch.get("rigidity"))
        else:
            value = branch.get(key)
        if value is None or value == "" or value == []:
            rendered = "—"
        elif isinstance(value, (list, tuple)):
            rendered = ", ".join(str(v) for v in value)
        else:
            rendered = str(value)
        parts.append(f"{_INCLUDE_LABELS[key]}={rendered}")
    return parts


def _node_summary(
    node: dict,
    include: tuple[str, ...],
    collapse: dict | None,
    walked: set[str] | None = None,
) -> str:
    """單節點一行：label + 欄位片段 + 收束角色標記。"""
    label = str(node.get("label", "?"))
    parts = _field_parts(node, include)
    body = label if not parts else f"{label}  " + "  ".join(parts)
    return f"{body}{_selection_marker(node, collapse, walked)}"


def _render_text(
    node: dict,
    depth: int,
    prefix: str,
    is_last: bool,
    out: list[str],
    include: tuple[str, ...],
    collapse: dict | None,
    max_layer: int | None,
    walked: set[str] | None = None,
) -> None:
    """遞迴渲染 text 樹（box-drawing）。``depth`` 追蹤樹深；層號用節點 layer。

    root（depth 0）渲染為標頭列（無分支接頭）；其餘節點用 ├─ / └─ / │ 接頭。
    """
    layer = int(node.get("layer", depth))
    if depth == 0:
        # root 是處境標頭（非分支）：只渲染 label，不帶欄位註記/可選標記。
        line = f"[root] {node.get('label', '?')}"
    else:
        connector = "└─ " if is_last else "├─ "
        line = prefix + connector
        if layer > 0:
            line += f"L{layer} "
        line += _node_summary(node, include, collapse, walked)
    out.append(line)
    children = node.get("children") or []
    if max_layer is not None and layer + 1 > max_layer:
        children = []
    child_prefix = prefix + ("" if depth == 0 else ("   " if is_last else "│  "))
    for i, child in enumerate(children):
        _render_text(
            child, depth + 1, child_prefix, i == len(children) - 1,
            out, include, collapse, max_layer, walked,
        )


def _render_markdown(
    node: dict,
    depth: int,
    include: tuple[str, ...],
    collapse: dict | None,
    max_layer: int | None,
    out: list[str],
    walked: set[str] | None = None,
) -> None:
    """遞迴渲染 markdown 樹（巢狀 bullet）。"""
    layer = int(node.get("layer", depth))
    indent = "  " * depth
    if layer == 0:
        out.append(f"- **{node.get('label', '?')}**  `[root]`")
    else:
        body = _node_summary(node, include, collapse, walked)
        out.append(f"{indent}- `L{layer}` {body}")
    children = node.get("children") or []
    if max_layer is not None and layer + 1 > max_layer:
        children = []
    for child in children:
        _render_markdown(child, depth + 1, include, collapse, max_layer, out, walked)


def _render_json(tree: dict, collapse: dict | None, walked: set[str] | None = None) -> dict:
    """json 格式：原樣結構化輸出（供程式消費）+ 收束選擇元資料。"""
    return {
        "format": "json",
        "tree": tree,
        "collapse": collapse,
        "selection": (
            {
                "selected": collapse.get("selected"),
                "layer": collapse.get("layer"),
                "unselected": collapse.get("unselected", []),
            }
            if collapse
            else None
        ),
        "walked": sorted(walked) if walked else [],
        "selectable": [
            {"layer": int(le.get("layer", 0)), "label": b.get("label")}
            for le in tree.get("layers", [])
            for b in le.get("branches", [])
            if _selection_of(b, collapse, walked) is None
        ],
    }


def convergence_view(
    tree_or_result: dict,
    *,
    layer: int | None = None,
    format: str = "text",  # "text" | "markdown" | "json"
    include: tuple[str, ...] = _DEFAULT_INCLUDE,
    path: list[str] | None = None,
    field_log: FieldLog | None = None,
) -> str | dict:
    """人機收束視覺化——把樹展開結果渲染給人看（人選方向前的『看』）。

    ``tree_or_result``：``probe_expand_layer`` 回傳的 ``tree``，或 probe result
    （``{"trees": [...]}``，可含 ``probe_select`` 寫入的 ``collapse``）。

    - ``format="text"``：人類可讀的文字樹（層縮排、每分支 label + 視角 + 束縛 +
      依憑 + 剛性，root 在頂）。
    - ``format="markdown"``：Markdown 樹（供 README / 筆記）。
    - ``format="json"``：原樣結構化輸出（供程式消費）。

    ``layer``：只渲染到第 N 層（含）；None = 全部。``include``：每分支要顯示的
    欄位（label/perspective/binding/grounding/rigidity/conditions/roles）。

    每個分支都標記它在收束中的角色——預設「（可選）」（可被 ``probe_select`` 選中）；
    若樹帶 ``collapse``（已被選過）→ 該層標「（已選）」/「（未選）」。

    ``path``（N2 缺口 2）：歷輪已選的 label 清單——這些「人已經走過的路」標
    「（已選）」（不限層）；``field_log`` 亦可直接傳入，從場 log 的 selected entry
    讀已選路徑（兩者併入同一集合；collapse 層內的標記優先）。

    純函數、零 UI 依賴、零 API。世界無關——不引入任何世界特定詞。
    """
    if format not in _VALID_FORMATS:
        raise ValueError(
            f"format 必須是 {'|'.join(_VALID_FORMATS)}，got {format!r}"
        )
    bad = [k for k in include if k not in _INCLUDE_LABELS]
    if bad:
        raise ValueError(
            f"include 含不支援欄位 {bad}——允許 {sorted(_INCLUDE_LABELS)}"
        )
    tree, collapse = _resolve_tree(tree_or_result)
    root = tree.get("root") or {}
    if not root:
        raise ValueError("tree 無 root——無法渲染（空樹）")
    max_layer = layer if layer is not None else max(
        (int(le.get("layer", 0)) for le in tree.get("layers", [])), default=0
    )

    # 🆕 累積路徑（N2 缺口 2）：``path``（歷輪已選 label）與 ``field_log``（場 log
    # 內的 selected entry）併入 walked 集合——渲染時把「人已經走過哪條」標（已選）。
    walked: set[str] = set(str(x) for x in (path or []))
    if field_log is not None:
        for e in field_log.entries:
            sel = e.get("selected")
            if isinstance(sel, dict) and sel.get("label"):
                walked.add(str(sel["label"]))

    if format == "json":
        return _render_json(tree, collapse, walked)

    out: list[str] = []
    if format == "markdown":
        out.append("# 收束視圖")
        _render_markdown(root, 0, include, collapse, max_layer, out, walked)
    else:
        _render_text(root, 0, "", True, out, include, collapse, max_layer, walked)
    return "\n".join(out)


def probe_converge_round(
    *,
    situation: Any,
    constraint_field: Any = None,
    state: dict | None = None,
    reflect_on: list[dict] | None = None,
    selected_label: str | None = None,  # None → 只展開給人看，不選
    field_log: FieldLog | None = None,  # 若有 → 收束後寫回場 log
    state_log_path: str | None = None,
    path: list[str] | None = None,  # 歷輪已選 label（累積路徑標記）
    view_format: str = "text",
    view_layer: int | None = None,
    view_include: tuple[str, ...] | None = None,
    dashboard_path: str | Path | None = None,  # 若有 → 視圖寫成 dashboard 檔
    field_log_path: str | Path | None = None,  # 若有 → FieldLog 寫成 JSON 檔
    snapshot_dir: str | Path | None = None,    # 若有 → 每輪寫輕量快照
    **expand_kwargs: Any,
) -> dict:
    """一輪收束：展開一層 → （可選）人選 → （可選）寫回場 log → （可選）落盤。

    流程：
    1. ``probe_expand_layer(state=state, reflect_on=reflect_on, ...)`` → 展開一層
       （``expand_kwargs`` 透傳：n_branch/depth/w/gate_fn/seed/api_key/model/...）。
    2. 若 ``selected_label`` 給定 → ``probe_select(expanded["result"],
       selected_label=selected_label, state_log_path=state_log_path)`` → 坍縮，
       取 collapse.selected 對應的分支；未選分支標 ``unselected``（不刪除）。
    3. 若 ``field_log`` 給定且有選定 → ``field_log.add_entry(layer,
       selected_branch, unselected_branches)``（selected + unselected 一起寫入，
       維持疊加）。
    4. 🆕 選後渲染（N2 缺口 1）：``convergence_view`` 在收束後才算——本輪有選 →
       用含 collapse 的 ``expanded["result"]`` 渲染（selected 標（已選）、同層未選
       標（未選））；沒選 → 全部（可選）。``view_format``/``view_layer``/
       ``view_include`` 控制渲染；``path``/``field_log`` 透傳為累積路徑標記。
    5. 🆕 落盤（T19 companion——「指向哪就寫到哪」，仿 warfare dashboard）：
       - ``dashboard_path`` 給定 → 把 ``view``（文字樹/markdown）寫成 dashboard 檔。
       - ``field_log_path`` 給定且 ``field_log`` 給定 → FieldLog 序列化為 JSON 寫檔
         （selected + unselected 疊加落盤，跨 session 續接）。
       - ``snapshot_dir`` 給定 → 每輪寫 ``converge-round-{N:02d}.json`` 輕量快照
         （round = field_log 筆數或 path 長度 + 1）。
       每次寫入前自動 .bak 備份（仿 ``save_oob_to_profiles``）。
    6. 回傳 ``next_state = expanded["state"]``、``next_reflect_on =
       [selected_branch]``（若有選）、``path``（累積路徑 = 歷輪 + 本輪）——
       供下一輪 ``probe_converge_round`` 續接。

    回傳 ``{"view", "expanded", "collapse", "next_state", "next_reflect_on",
    "field_log_entry", "state_log_entry", "path", "written"}``。

    ``written``：本輪實際寫出的檔案路徑 dict——``{"dashboard"|"field_log"|
    "snapshot": Path}``（只含真的寫了鍵）。未指定對應路徑參數 → 該鍵缺席。

    ``selected_label=None``：只展開給人不選（不坍縮、不寫場 log、不落盤）——
    人看了以後可以決定是否收束。若 ``selected_label`` 給定但不在本層 branches →
    ``TreeProbeError``（在 ``probe_select`` 副作用之前攔截）。

    ``state_log_path=None``（F8 fallback）：``probe_select`` 只寫 in-memory
    state log（``kernel.verify``，cap 10000，``query_state_log`` 可取）——回傳
    ``state_log_entry`` 讓呼叫方在無落盤時也能看到本輪收束記錄。
    """
    expanded = probe_expand_layer(
        situation, constraint_field, state=state, reflect_on=reflect_on,
        **expand_kwargs,
    )

    collapse: dict | None = None
    selected_branch: dict | None = None
    unselected_branches: list[dict] = []
    state_log_entry: dict | None = None
    if selected_label is not None:
        branches = expanded["layer_entry"]["branches"]
        by_label = {b["label"]: b for b in branches}
        if selected_label not in by_label:
            raise TreeProbeError(
                f"selected_label '{selected_label}' 不在本層 branches（layer "
                f"{expanded['layer_entry']['layer']}）——必須從 convergence_view "
                f"的（可選）/ selectable 分支中挑選"
            )
        collapsed = probe_select(
            expanded["result"],
            selected_label=selected_label,
            state_log_path=state_log_path,
        )
        collapse = collapsed.get("collapse")
        state_log_entry = collapsed.get("state_log_entry")
        selected_branch = by_label[selected_label]
        unselected_branches = [b for b in branches if b["label"] != selected_label]

    field_log_entry: dict | None = None
    if field_log is not None and selected_branch is not None:
        layer = int(selected_branch.get("layer", expanded["layer_entry"]["layer"]))
        field_log.add_entry(layer, selected_branch, unselected_branches)
        field_log_entry = field_log.entries[-1]

    # 🆕 選後渲染（N2 缺口 1）：view 反映收束後狀態——本輪有選 → 用含 collapse 的
    # result 重渲染（selected 標（已選）、同層未選標（未選））；沒選 → 全部（可選）。
    # ``path``/``field_log`` 透傳：歷輪已選標（已選）（N2 缺口 2）。
    view = convergence_view(
        expanded["result"] if selected_branch is not None else expanded["tree"],
        layer=view_layer,
        format=view_format,
        include=view_include if view_include is not None else _DEFAULT_INCLUDE,
        path=path,
        field_log=field_log,
    )

    # 🆕 落盤（T19 companion）：只展開沒選 → 不落盤（無可寫的收束狀態）。
    written: dict[str, Path] = {}
    if selected_branch is not None:
        if dashboard_path is not None:
            if not isinstance(view, str):
                raise TypeError(
                    "dashboard_path 需要 text/markdown 視圖（str）——"
                    f"view_format 不能是 'json'（got {view_format!r}）"
                )
            written["dashboard"] = _store_save_dashboard(view, dashboard_path)
        if field_log_path is not None and field_log is not None:
            written["field_log"] = _store_save_field_log(field_log, field_log_path)
        if snapshot_dir is not None:
            round_num = (len(field_log.entries) if field_log is not None else 0) or (
                len(path or []) + 1
            )
            written["snapshot"] = _store_save_round_snapshot(
                {
                    "view": view,
                    "collapse": collapse,
                    "path": (list(path or []) + [selected_label]),
                    "field_log_entry": field_log_entry,
                    "state_log_entry": state_log_entry,
                },
                round_num,
                snapshot_dir,
            )

    return {
        "view": view,
        "expanded": expanded,
        "collapse": collapse,
        "next_state": expanded["state"],
        "next_reflect_on": [selected_branch] if selected_branch is not None else None,
        "field_log_entry": field_log_entry,
        "state_log_entry": state_log_entry,
        "path": (list(path or []) + [selected_label])
        if selected_label is not None
        else list(path or []),
        "written": written,
    }


__all__ = [
    "_DEFAULT_INCLUDE",
    "convergence_view",
    "probe_converge_round",
]
