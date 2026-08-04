# T19 — 收束 UX（Convergence UX）

> **工單**：T19（spectrum-os 待辦表）
> **日期**：2026-08-04
> **定位**：人機收束「奇門的門」的操作化——『沒有民主就沒有用』。把「人看樹 → 人選 → 寫回場 log」的收束流程做成可用介面。
> **產物**：`spectrum_os/synth/convergence.py` 新增 `convergence_view`（text/markdown/json 渲染）+ `probe_converge_round`（一輪收束）。
> **衡量**：E2E 三輪 mock gate 流程全通；`FieldLog` 維持疊加（selected + unselected 一起寫入）；世界無關檢定通過。

---

## 一、目標

下游消費 T18/T17a 既有產物（不碰其既有函數）：

- `probe_expand_layer`（probe.py，T18）展開一層，回傳 `tree`（截至該層的工作樹，人看的對象）與 `result`（`probe_select` 可消費的最小形狀）。
- `probe_select`（probe.py）人選一條 → 坍縮；未選分支標 `unselected` 寫回（**不刪除**，維持疊加）。
- `contracts.FieldLog`（contracts.py，T17a）場的歷史 log——`add_entry(layer, selected, unselected)` 把 selected + unselected 一起寫入。

## 二、API

### `convergence_view(tree_or_result, *, layer=None, format="text", include=("label","perspective","binding","grounding","rigidity"), path=None, field_log=None)`

人選方向前的『看』——純函數、零 UI 依賴、零 API：

- `format="text"`：box-drawing 文字樹（root 標頭在頂，`L1`/`L2` 層號縮排）。
- `format="markdown"`：巢狀 bullet 樹（供 README / 筆記）。
- `format="json"`：原樣結構化輸出（供程式消費，含 `selectable` 清單）。
- 每個分支標記收束角色：預設「（可選）」；樹帶 `collapse` → 該層「（已選）」/「（未選）」；他層不受影響。
- `path` / `field_log`（N2 缺口 2）：累積路徑——歷輪已選的 label 標「（已選）」（不限層），collapse 層內標記優先。

### `probe_converge_round(*, situation, constraint_field=None, state=None, reflect_on=None, selected_label=None, field_log=None, state_log_path=None, path=None, view_format="text", view_layer=None, view_include=None, **expand_kwargs)`

一輪收束：展開一層 → （可選）人選 → （可選）寫回場 log → 選後渲染：

1. `probe_expand_layer(...)` → 展開一層（`expand_kwargs` 透傳：n_branch/depth/w/gate_fn/seed/api_key/model/...）。
2. `selected_label` 給定 → `probe_select(...)` → 坍縮；未選標 `unselected`（不刪除）。
3. `field_log` 給定且有選 → `add_entry(layer, selected_branch, unselected_branches)`（維持疊加）。
4. **選後渲染**（N2 缺口 1）：本輪有選 → 用含 collapse 的 result 渲染（selected「（已選）」、同層未選「（未選）」）；沒選 → 全部「（可選）」。
5. 回傳 `{"view", "expanded", "collapse", "next_state", "next_reflect_on", "field_log_entry", "state_log_entry", "path"}`——供下一輪續接。

`selected_label=None`：只展開給人不選（不坍縮、不寫場 log）。`selected_label` 不在本層 branches → `TreeProbeError`（在 `probe_select` 副作用之前攔截）。

## 三、命名（防遮蔽）

- `spectrum_os.synth.convergence_view`（`synth/__init__.py` re-export）= probe 產物的**可程式化資料視圖**（rigidity / archetype / gate 卡片）。
- `spectrum_os.synth.convergence.convergence_view` = **樹的渲染函數**（text/markdown/json，人選方向前的『看』）。
- 本模組**刻意不** re-export 進 `synth/__init__.py`（避免遮蔽 probe 版本）；使用方直接 `from spectrum_os.synth.convergence import ...`。

## 四、驗收

1. **E2E 三輪 mock gate 流程全通**：expand→view→select→next_reflect_on=[1 父]→FieldLog 3 筆。
2. **打破 1:1 實證**：手動路徑 `avg_children_per_parent`=2.67（route-a→3 子, route-a2→2 子）；自動 `probe_tree`=1.29（每父 1 子，1.3636 血統）——手動路徑為人機收束提供結構性出口。
3. **世界無關**：convergence.py 對禁詞模式零命中（唯一「ECC」在 docstring 否定句「零 ECC 路徑」）；渲染輸出零世界特定詞。
4. **防禦**：`selected_label` 攔截在 `probe_select` 副作用前（test 驗證 `_verify._state_log == []` 無殘留）；`state_log_path=None` 時 in-memory fallback 仍可取 `state_log_entry`（F8）。
5. **全套測試全綠**：621 → 631（test_convergence.py 21 → 31）。

## 五、施工步驟

```
N1: convergence_view 三格式渲染（text/markdown/json）+ 收束角色標記
N2: probe_converge_round 一輪收束（expand → select → FieldLog 寫回 → 選後渲染）
N3: N2 缺口修復：選後渲染（缺口 1）+ 累積路徑標記（缺口 2）+ state_log_entry fallback（缺口 3）
N4: 新增測試 + 全套跑綠
→ 人機收束：與 T18 on_layer 回呼對齊；UX 交人類 pilot
```

## 六、明確不做（後話）

- ❌ 多人投票機制 / 完整 UI（見 TASKLOG 已知弱點 #6）——本實作是可程式化介面，非 GUI。
- ❌ 場狀態壓縮注入（`reflect_on` = [人選那條] + 場狀態）——T18 後話。
- ❌ 自動路徑的 1:1 解除——自動 `probe_tree` 保持向後相容。
