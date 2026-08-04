# T18 — 手動逐層觸發（Manual Layer Trigger）

> **工單**：T18（spectrum-os 待辦表）
> **日期**：2026-08-04
> **定位**：打破 `probe_tree` 自動路徑的 **1:1 續鏈鎖死**（反射設計把上一層全部通過分支全量回饋為反射對象 → LLM 慣性落入「一一對應」，樹寬指標鎖死在 1.3636）。
> **產物**：`spectrum_os/synth/probe.py` 新增 `probe_tree_manual`（on_layer 回呼）+ `probe_expand_layer`（公開單層原語）。
> **衡量**：手動路徑 `reflect_on` 長度 = 1（非 5）；一個父可展開多子；`avg_children_per_parent` 脫離 1.3636。

---

## 一、目標

把「人選方向」插入探針生成：生成器逐層暫停，`on_layer` 回呼是人機收束的接縫——人從 `layer_entry["branches"]` 挑一個，只有那條成為下一層的 `reflect_on`（打破 1:1 引誘：一個父可展開多子）。世界無關：不引入任何世界特定詞。

## 二、API

### `probe_tree_manual(situation, constraint_field=None, *, n_branch=5, depth=3, w=0.5, gate_fn=None, call_api_fn=None, state_log_path=None, seed=42, api_key=None, model="deepseek-v4-flash", max_tokens=8192, temperature=0.6, code=None, system_message=None, timeout=180, on_layer=None)`

- `on_layer(layer_entry, ctx)` → 回傳 **[選定的 branch dict]** 作為下一層 reflect_on；回傳 **None** → 中止（樹停在該層，`meta.stopped_at_layer`）。
- `ctx = {"k", "passed", "rejected", "echo_notes"}`。
- `on_layer=None`（預設）→ 自動續接全部 passed（模擬 `probe_tree` 單樹行為，向後相容）。
- 與 `probe_tree` 共用 `_generate_layer_once` / `_finalize_tree` / `_assemble_probe_result`——自動與手動路徑的單層邏輯零分歧。

### `probe_expand_layer(situation, constraint_field=None, *, state=None, reflect_on=None, n_branch=5, depth=3, w=0.5, ...)`

- 公開單層原語：每次呼叫展開一層；`state` 跨次攜帶（首呼 None → 自動初始化；state 一旦建立即為**權威**——後續 n_branch/w/gate_fn/model 被忽略）。
- `reflect_on` = [人選的那條]；首次（None）→ []（從處境展開）。
- 回傳 `{"layer_entry", "passed", "state", "tree", "result", "calls"}`——`tree` 為截至該層的工作樹（供 `convergence_view` / `probe_select` 人看）；`result` 為 `probe_select` 可消費的最小形狀。

## 三、設計原則

1. **reuse via extraction，非平行複製**——手動/自動共用單層邏輯，不維護兩套。
2. **防禦先於 gate**——`reflect_on` 必須屬於上一層 passed（label 集合比對）：無上一層（k=1）或任一 label 非屬上一層 → `TreeProbeError`，在 gate 呼叫前攔截（零額外 API）。原風險：payload 寫 `reflection.layer=k-1` 但分支來自別層 = 對 LLM 的語義謊言；`_resolve_parent` fallback → depth-jump。
3. **生成器手動驅動**——`probe_tree_manual` 不能用 `for ... in gen`（`gen.send()` 本身消耗下一次 yield，與 for 迴圈組合會雙重推進、跳過一層）：`next(gen)` 起步 + `while True` + `k, ..., = gen.send(...)` 接收回傳值。

## 四、驗收

1. **打破 1:1 實證**（N2 獨立驗證 30/30 通過）：`probe_tree_manual` + on_layer 人選第一條（n_branch=5）→ 下一層 payload `reflection.passed_branches` 長度 = 1（非 5）；一個父展開 3 子；`avg_children_per_parent` > 1（含 root 統計：root(5 子)+人選父(3 子) → 4.0；n_branch=3 時才是 3.0）。
2. **`probe_expand_layer` 連續 3 次安全**：state 累積、每步 1 call、超 depth 才 raise。
3. **reflect_on 防禦測試**：`test_expand_layer_reflect_on_foreign_branch_raises` / `_from_prev_layer_ok` / `_first_call_with_reflect_on_raises` / `_first_call_no_reflect_on_ok`（+4）。
4. **世界無關**：probe.py 邏輯碼零世界特定詞命中。
5. **全套測試全綠**：584 → 596（+12：manual 7 / generator 2 / expand 3）。

## 五、施工步驟

```
N1: _generate_one_tree_iter 改為暫停的生成器（每層 yield，gen.send([人選那條]) 注入下一層 reflect_on）
N2: probe_tree_manual（on_layer 回呼）
N3: probe_expand_layer（公開單層原語 + state 累積）
N4: reflect_on 防禦（屬上一層 passed 檢驗）
N5: 新增測試 + 全套跑綠
→ 人機收束：on_layer 回呼與人類 UX 對齊（下游 T19 消費）
```

## 六、明確不做（後話）

- ❌ 場狀態壓縮注入（`reflect_on` = [人選那條] + 場狀態，PLAN §12.14）——本實作先做最小可行：只注入 [人選那條]。
- ❌ 收束 UX / 視覺化——T19。
- ❌ 自動路徑的 1:1 解除——自動 `probe_tree` 保持向後相容，1:1 血統仍在（見 TASKLOG「樹寬指標失效根因」）。
