# Spectrum OS 開發日誌（`docs/TASKLOG.md`）

> **職能**：OS 本體（spectrum-os repo）的開發狀態追蹤——每次 commit、測試數、驗收、已知 bug、待辦。
> **規格**：OS 本體規格在 `docs/PLAN.md`；ECC 應用層在 ECC `docs/plans/PLAN-23-spectrum-computer-engine.md` + ECC `index/data/SPECTRUM-TASKLOG.md`。
> **定位**：本文件記錄「OS 怎麼長大」，不記錄「OS 被用來做什麼」（那在應用層）。

## 📌 現況快照（2026-08-07）

| 項 | 值 |
|:---|:---|
| HEAD | `59af3d3`（[Forces] v1.6 βC→P_R 壓縮通道） |
| 測試 | **764 passed / 10.57s**（30 test files，2026-08-07 實跑） |
| 模組規模 | 36+ 模組（含 forces.py 815 行 + round_loop.py 1358 行） |
| 規格-代碼差距 | `docs/v25-spec-code-gap.md`：5 項全數 conscious divergence（已收斂） |
| push 狀態 | `59af3d3` 已 push 至 origin/master |

### ✅ 已 commit（4ead001，2026-08-04 12:14）+ 後續 forces/RoundLoop 三 commit

> 上表工作樹已於 `4ead001`（T17a/T18/T19 + 收束持久層，657 tests）commit。
> **08-04 後另起三條新線（forces/RoundLoop）**——本文件當時未追蹤，2026-08-07 補記：

| commit | 內容 |
|:---|:---|
| `59af3d3`（08-07） | [Forces] v1.6 βC→P_R 壓縮通道（壓制轉移非毀滅、growth_r_fn/growth_c_fn/release_fn 基板回饋）——764 tests |
| `808672f`（08-04 21:56） | [RoundLoop] 回合協定編排器 + Mode B 委派 + **TREE 干涉網段** + forces 公開 setter——753 tests |
| `ea369e7`（08-04 20:28） | [Forces] 社會動力學第一定理引擎（R/C/P_R 對抗力場 + 四象限耦合 + from_actor_cards）——716 tests |
| `4ead001`（08-04 12:14） | T17a 插件契約 + T18 手動逐層 + T19 收束 UX/持久層——657 tests |

## 📊 最近 commit 歷史（Stage 3 收束段）

| commit | 內容 |
|:---|:---|
| `59af3d3` | [Forces] v1.6 βC→P_R 壓縮通道（壓制轉移非毀滅、基板回饋）——764 tests |
| `808672f` | [RoundLoop] 回合協定編排器 + Mode B 委派 + TREE 干涉網段 + forces setter——753 tests |
| `ea369e7` | [Forces] 社會動力學第一定理引擎（R/C/P_R + 四象限 + from_actor_cards）——716 tests |
| `4ead001` | T17a 插件契約 + T18 手動逐層 + T19 收束 UX/持久層——657 tests |
| `1abf6d4` | T16 語義中介取代黑名單 + T6 定稿 prompt + 層間 echo 語義化（L3 單鏈修復）——548 tests |
| `12ddb68` | P2 補齊：standing_wave_per_layer + 語義聚類 + 收束視圖 + Mode A→B 接線 + remasking 校準 |
| `8257744` | 探針修補 F1-F10 + W1 DCA 基底接回 + probe pilot 腳本 |
| `d39c3c1` | 決策樹探針實作（synth/probe.py：五階段 A-E、v1.2 無必然硬輸出）——421 tests |
| `afaf1f2` | N6：correlation_flip 引擎層降級為 amplifier |
| `9da4466` | N5：pilot 二相態接線（residual_table + key_fn + sharp-only 過濾） |
| `c378563` / `b1ca766` | N4：residual_trigger 接線（二相態切換） |
| `6fba133` | N1-N3：DIGEST 社會物質結構升級 + adversary 處境感知 + 以處境為參照的回聲拒絕 |
| `b49a1b3` | S7 版本 B：歷史的意外（殘差觸發模式）alt_gate_enumerate 實作 |
| `1931872` | 閘層 prompt 重寫：繼承的條件（兩軸框架）+ 中性範例標籤 |
| `86b3bde` | S4 驗收修補三項（verify.init_log flush、pilot 報告 branch_tags、observe 錯字）294/294 |
| `cd9b71c` | 薩拉熱窩 live 首飛接線（unified=True + 時代視野改相對截止——通用化） |
| `293d1cd` | prompt 技藝：統一閘 prompt 全肯定句化 |
| `a140a2b` | KV-cache 最大化（GATE_SYSTEM_PROMPT_UNIFIED + ensemble Batch-by-Phase）291/291 |
| `14b794d` | OS-2 Stage 3 本體 + OS-3 pilot（alt_gate + ensemble + standing_wave + 薩拉熱窩）285/285 |
| `aa3ceee` | OS-1 通用化地基（sources registry + 通用貨幣三件套）265/265 |
| `8206810` | H schema + G2 速率估計閘 + HMM 疊加重建橋（markov v1.1）241/241 |
| `41a7dd6` | §4.6 Markov 脊椎 v1（轉移計數+Dirichlet 後驗+地平線四量+譜隙）224/224 |
| `49c9d49` | v2.5 量子層實裝（6D StateVector + entangle + DCA grammar + quarantine + V25 prompt）210/210 |

## 🐛 樹寬指標失效根因（2026-08-03 追源，2026-08-04 二修）

> **現象**：A/B/C/D prompt 對照測試（`scripts/ab_prompt_test.py`，同一處境同一 seed）中樹寬指標全面失效——所有版本、所有 seed 的 `avg_children_per_parent` 恆為 **1.3636**、`n_paths` 恆為 **5**。指標失去版本區分度，成為結構常數。

**真正的根因（2026-08-04 二修確認）**：不是機械碼、不是 `n_branch` 主因——是**反射設計的 1:1 續鏈**，讓 LLM 慣性落入「一一對應」：

1. `reflect_on = passed`（probe.py L1632，HEAD 既有）把上一層全部通過分支**全量回饋**為反射對象（5 個父）。
2. 分支數恆 5（payload `n_branch=5`＋prompt「幾根柱就幾條路」＋處境約 5 柱）——LLM 每層恆產 5 條。
3. 既有句「本層每一條新路必須與上一層及同層的路**相位不同**」（HEAD 既有）→ LLM 的最簡解法 = 5 條新路各續 1 條舊路（一一對應）。
4. 結果：完美 1:1 → 5 條平行鏈。**1.3636 = 15/11 唯一對應此結構**（11 個父各恰 1 子）；主線 fallback 只會產生星形 avg=5.0——故「parent 多數缺省→掛主線」為**誤診**（LLM 恆給互不重複的合法 parent 才構成 1:1）。
5. 2026-08-03 加的「必須標明 parent」句是**加劇**（隱式 1:1 → 顯式，並引入 NoneType 失敗），**不是元兇**（fix 前 A/B/C 已全鎖 1.3636）——已於 2026-08-04 **撤回**，改為「parent 可選、新路可為場的獨立坍縮」。
6. `n_branch` 為次要因素（D 版同帶「≤N_branch」仍偶爾突圍）。唯一突圍（D seed=7 → 2.1429/9）為**溫度 lucky draw**；候選機制 = D/C2 的**內化框架**（突圍 3/5 次 vs A/B/C 的 1 次，n 小未定）。

**追源鏈（程式碼先於規格）**：
- `d39c3c1`（2026-08-01，spectrum-os）探針初始 commit 帶入反射/建樹機制。
- `b94d1d4b`（2026-08-02，ECC）規格 §十二 才回寫——程式碼先於規格。
- **真正安全網已存在且獨立**：`MAX_PATHS_PER_TREE=4096`（F5 路徑爆炸防護）+ `MAX_BRANCHES_SAFETY_CAP=20`（2026-08-03 加，與 LLM 指令解耦）。

**修正方向（2026-08-04 已落地部分）**：
- ✅ **撤回「必須標明 parent」強制句**（4 處）→ 改為「樹的形狀由場決定：同一父可掛多子；新路可為場的獨立坍縮（不續舊路——省略 parent）；不要為了續接而續接」——打破 1:1 引誘。
- ✅ `_resolve_parent` docstring 修正（刪誤診）。
- ✅ **T18/T19 結構性解鎖**（2026-08-04）：手動逐層路徑（`probe_tree_manual`/`probe_expand_layer`）的 `reflect_on` = [人選那條]——實證 `avg_children_per_parent` 脫離 1.3636（手動 2.67 vs 自動 1.29，見 `docs/tickets/T18`/`T19`）。1:1 血統在自動路徑仍在（自動 probe_tree 每父 1 子）；手動路徑為人機收束提供結構性出口。
- ⏳ **待驗證**：自動路徑重跑 A/B/C/D 看樹寬是否脫離 1.3636；若仍鎖，試「反射壓縮」（`reflect_on` 從 5 父清單改為場的整體描述）或驗證 D/C2 內化框架。

**實作狀態**：✅ 2026-08-04 撤回完成（**559 tests 全綠**，詩性內容完整保留）。T6 樹寬驗證**待重跑**（見待辦）。

## �📝 待辦（OS 本體層）

| # | 項目 | 狀態 | 依賴 |
|:---:|:---|:---:|:---|
| T17a | **插件契約 schema（OS 層·世界無關）**：定義場 folder + 行動者卡的 dataclass 契約（`contracts.py`）——field.yaml + actor card（承繼條件/場域座標/內在張力/時空演化） | ✅ 已完成（2026-08-04） | 產物：`contracts.py` 新增 FieldSpec/ActorCard/SituationSpec/FieldLog + `to_dict()`（vector_6d→6d_vector 對映）+ 29 測試（含管線整合）；**584 tests 全綠**；OS-1 世界無關檢定通過；工單 `docs/tickets/T17a-plugin-contract-schema.md` |
| T18 | **手動逐層觸發**：`probe_tree` → `probe_expand_layer`（reflect_on=[人選那條]+場狀態）→ 停 → `probe_select` → 人觸發下一層；打破 1:1 續鏈 | ✅ 已完成（2026-08-04） | 產物：`probe_tree_manual`（on_layer 回呼，None=中止）+ `probe_expand_layer`（公開單層原語，state 跨次攜帶、reflect_on 屬上一層防禦）；與 probe_tree 共用單層邏輯零分歧；工單 `docs/tickets/T18-manual-layer-trigger.md` |
| T19 | **收束 UX**（convergence_view 視覺化）：人看樹 → 選 → 坍縮寫回場 folder（unselected 保留） | ✅ 已完成（2026-08-04） | 產物：`synth/convergence.py`（`convergence_view` 渲染 text/markdown/json + `probe_converge_round` 一輪收束，選後渲染 + 累積路徑 + FieldLog 寫回）；31 測試；工單 `docs/tickets/T19-convergence-ux.md` |
| T6 | **樹寬驗證**：撤回「必須標明 parent」強制句後重跑 A/B/C/D——驗證 `avg_children_per_parent` 脫離 1.3636（2026-08-03 fix 已實跑未解鎖；2026-08-04 改為場的獨立坍縮精神後待重跑，見「🐛 樹寬指標失效根因」） | ⏳ 未跑 | 撤回 ✅ 已落地（2026-08-04，559 tests）；待重跑 A/B/C/D |
| T5 | **prompt 設計——剛性 blend**：記者/編輯分工——LLM 帶定性素材（開/閉），機械測量剛性數字；現 blend 被 LLM 自報主導（mechanical≈0.004） | 🔵 設計中 | 需人類簽署：兩軸分開、structural_rigidity 新增 |
| T16 | **prompt 設計——語言本土化/世界線無關**：labels 用處境語言（β₁ 德/俄/英/粵、γ 英文）；語義中介取代黑名單已落地；剩餘：`lang_field` 進 situation（步 0）＋`PROMPT_EXAMPLE_LABELS` 處境語言化 | 🟡 部分落地 | 需人類追認（T16/T16c 實作已落地但提案 DRAFT） |
| T4 | **v1.3 五項剛性地圖品質驗收 pilot**（①剛性低谷↔歷史時刻 ②信賴帶誠實/remasking ③原型處境內生 ④人機收束 re_calibrate 率 ⑤成本/抗污染維持） | ⏳ 未跑 | T5/T16 定稿後；mock 門節點 1/1 MET 已預驗 |
| T16b | **第二世界線端到端註冊**（γ：世行 40 年 GDP 或天氣）——證明機器世界無關（α/β/γ/δ…平行，各有階段，只在校準剎那共享 timing）。**核心待驗證能力** | ⏳ 未做 | substrate 準備 + prompt 語言機制；worldbank-gdp 已註冊（雙源 dry-run 已跑） |
| S6 | 時間游標（舊定位暫緩）：分支用自己的日曆（僅分支標籤+物理曆法），**禁止** α₁ 事件日曆 | 暫緩 | 本體論落實 |
| G3 | verify 校準曲線擴展至速率預測（FALSIFY-010 對照執行） | 待辦 | 已實現轉移累積 |
| Mode A→B | Mode A 歷史日誌 + Mode B 後驗初始化接線（PLAN §11.12） | ⏳ 待實作 | — |

## 🧭 認識論層待辦（人機收束）

- ✅ **人機收束（奇門的「門」）已操作化（T19，2026-08-04）**：`convergence_view`（text/markdown/json 渲染）+ `probe_converge_round`（一輪收束）落地——「沒有民主就沒有用」從口號變成可用介面（可程式化，非 GUI）。
- ⏳ 剩餘：多人投票機制 / 完整 UX pilot（見已知弱點 #6）；T18 後話「場狀態壓縮注入」。
- 三段航路：天書（機器）→ 奇門（人機交互）→ 意識宇宙（機器成為意識的生境）——目前在天書→奇門交界，奇門的「門」已可開。

## 🐛 已知弱點（下輪 debug 審查對象，PLAN §11.11）

1. reflex 層間迭代不保證收斂（D>4 風險高）
2. 樹→路徑組合爆炸（$\prod N_{branch}$ 截斷需 pilot 校準）
3. α₂ 語義對齊層未規格化
4. 門節點③代理閾值（0.99 vs 0.9）需 pilot 對照
5. w 的絕對標度缺驗收錨點
6. 收束介面 UX/多人投票機制待 pilot
7. CFG 公式落地需新增條件 prevalence 欄位
8. 世界原型聚類需語義向量化層

## ⚠️ 需人類追認事項

- **T16/T16c 提案**：實作已落地 commit（`1abf6d4`）但提案文件（ECC `docs/plans/T16-hybrid-language-proposal.md`、`T16c-quantum-semantic-prompt.md`）仍為 DRAFT
- **T5 blend 重構**：需人類簽署兩軸分開方案（mechanical/structural 分離）
- **FALSIFY-009 含義**：數據已落盤（`data/falsify009_report.json`），含義留待人類判斷

## 📁 實驗/驗證腳本

| 腳本 | 內容 | 報告 |
|:---|:---|:---|
| `scripts/stage3_probe_pilot.py` | 決策樹探針 mock 實證（零 API） | `data/stage3_probe_pilot_report.live.json` |
| `scripts/stage3_sarajevo_pilot.py` | 薩拉熱窩 live pilot（錨點對照落 state_log） | `data/stage3_sarajevo_live_report*.json` |
| `scripts/stage3_universality_check.py` | 雙源通用性（ECC sarajevo + worldbank-gdp） | `data/stage3_universality_report.json` |
| `scripts/fetch_economic_data_demo.py` | 世行 40 年 GDP → DCA role → Markov | `data/world_economic_spectrum_report.json` |
| `scripts/markov_warfare_demo.py` | Markov 戰爭推演 demo | `data/markov_demo_report.json` |
| `scripts/falsify009_rupture_experiment.py` | rupture 聯合簽名特異性檢定 | `data/falsify009_report.json` |
| `scripts/gate_cache_benchmark.py` | KV-cache 共享率 benchmark | `data/gate_cache_benchmark.json` |
| `scripts/rate_gate_liveness.py` | G2 速率估計閘 liveness | — |
| `scripts/pilot_danube_steel.py` | danube_steel synthetic pilot | `data/danube_steel_nspv.json` |

## 文件職能變更紀錄

- **2026-08-03**：本文件（TASKLOG.md）+ `docs/PLAN.md` 建立——OS 本體從 ECC PLAN-23 獨立（三分：規格→PLAN.md、日誌→TASKLOG.md、應用→ECC PLAN-23）。
