# 工單：S4 重飛性能優化——序列化瓶頸移除（不觸動設計機制）

> **日期**：2026-07-31
> **目標**：把 Stage 3 薩拉熱窩 pilot 的 6–12 分鐘壓到 ~1–2 分鐘，**且不改變任何 PLAN-23 設計意圖**
> **立場**：reporter 模式（observe→compress）與 n_samples=3 是刻意品質機制，**禁止**以性能名義移除

---

## 零、設計紅線（不可逾越）

| 機制 | 設計意圖 | 禁止的操作 |
|:---|:---|:---|
| reporter 兩階段 (observe→compress) | 生成熱、檢驗冷（Inventory §四.3） | ❌ 禁止停用 |
| n_samples=3 | 機械置信度回填的樣本離散度（FIX-21） | ❌ 禁止降到 2 |
| 同 branch 內 gate points 順序 | `existing_labels` 依賴鏈（累積標籤） | ❌ 禁止 branch 內並行 |
| adversary 依賴 stage1 輸出 | 對抗審查兩段調用 | ❌ 禁止合併 |
| KV-cache 共享前綴 | 97.31% 理論共享率 | ⚠️ 並行不得破壞 system prompt 共享 |

## 一、唯一修改點：跨 branch + samples 並行

**現狀**（`ensemble.py` Phase 2/3）：`for b_idx in gate_points` 嚴格順序，全管線零並發。

**修改**：Phase 2 的生成調用改為 `ThreadPoolExecutor` 並行：

```
安全的並行維度:
  ✅ 不同 branch 之間（gate_input 完全獨立）
  ✅ 同 gate point 內 3 個 samples（同 prompt，獨立調用）
  
不可並行的維度:
  ⛔ 同 branch 內多個 gate points（existing_labels 依賴鏈）
  ⛔ adversary 需 stage1 完成後
```

### 實作方案

```
1. Phase 2 重構:
   for b_idx 並行 (max_workers=N):
     該 branch 內 gate points 仍順序（保依賴鏈）
     每個 gate point 內 3 samples 可再並行（可選，先單層）

2. Phase 3 重構:
   for gp 並行（adversary 只需 stage1_res，跨 branch 獨立）

3. N 值: WORKFLOW 並發 100-200 → max_workers 建議 8-16（保守，
   因 API rate limit 未知；先 8，觀察失敗率）
```

### 需要處理的風險

| 風險 | 處理 |
|:---|:---|
| API rate limit | max_workers=8 起步；失敗率>5% 則降至 4 |
| KV-cache 共享 | system prompt 不變（`GATE_SYSTEM_PROMPT_UNIFIED` 保持共享），並行不影響前綴 |
| 執行緒安全 | `call_api` 用獨立 urllib.request 呼叫，無共享狀態；`verify` 的 state_log 需確認 append 是否 thread-safe（JSONL append 原子性） |
| 確定性 | seed 用於 Markov 採樣（Phase 1，同步進行），API 調用本質非確定性，並行不改語義 |
| 重試 | 每 call 內建 MAX_RETRIES=2，並行下重試仍各自獨立 |

## 二、驗收

1. **功能等價**：並行前後同一 seed 的 `standing_wave.nodes` / `anchor_branch_ratio` 差異僅來自 API 生成的非確定性（不應有系統性差異）
2. **294/294 tests 全綠**
3. **計時**：`time` 前後對照，牆鐘 6-12min → 預期 1-2min
4. **reporter 模式保留**：`--reporter` 路徑仍產出 observe→compress 兩階段調用
5. **n_samples=3 保留**：`confidence` 仍基於 3 樣本離散度
6. **cache 命中率不下降**：console log `[API Usage]` Hit 率 ≥ 85%

## 三、明確不做的事

- ❌ 不停用 reporter（S4 刻意設計）
- ❌ 不降 n_samples
- ❌ 不縮 thread_history 視窗（未驗證語義影響）
- ❌ 不減少 adversary（對抗審查是抗污染憲章 §六.五-5）
- ❌ 不優化 prompt（非本次範圍）

## 四、預期成本

- 修改：`ensemble.py`（~40 行）、測試 `tests/test_ensemble.py` 增並行回歸
- 估時：0.5-1 天
- 加速：6-12×（牆鐘）
- 成本：不變（調用數不變）

## 五、執行禁令

- 禁止 `git commit`
- 禁止改 `gate_prompts.py`、`alt_gate.py` 的生成邏輯
- 產物只落 `data/`
