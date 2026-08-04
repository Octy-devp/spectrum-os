# Spectrum OS — 統一頻譜計算機引擎（OS 本體規格）

> **版本**：v1.0（2026-08-03）——**由 ECC `PLAN-23-spectrum-computer-engine.md` §〇~§十二 遷移的 OS 本體規格**。規格已收斂（`v25-spec-code-gap.md` 五項差距全數為 conscious divergence），本文件為**純規格**——不含開發日誌與狀態追蹤。
>
> **文件職能三分**：
> - **本文件（`docs/PLAN.md`）**：OS 本體規格——kernel/quantum/Markov/探針/抗污染憲章/向量契約。**世界無關**：γ/δ/ε 任何世界線的 substrate 共用同一份規格。
> - **`docs/TASKLOG.md`**：OS 開發日誌——每次 commit 狀態、測試數、驗收、已知 bug、待辦。
> - **ECC `docs/plans/PLAN-23-spectrum-computer-engine.md`**：ECC 應用層——feeds/sectors/substrate 註冊/αβ 校準/FALSIFY 對照（由 ECC SPECTRUM-AGENT 管轄）。
>
> **關聯**：`spectrum_os/`（實作）、`tests/`（548 tests）、`docs/v25-spec-code-gap.md`（規格-代碼差距判定）、`docs/PLAN-23-optimization-serial-bottleneck.md`（效能）、`docs/2026-07-29-v25-implementation-plan.md`（v2.5 實作計劃歷史）。
>
> **核心原則**：🔴 機械/LLM 分工——純計算由 numpy 驅動（`spectrum_os/kernel/` + `spectrum_os/quantum/` 為 numpy-only），LLM 僅在分叉點作為 GATE（`spectrum_os/synth/`）。

---

## 〇、核心命題

**Spectrum OS 是在我們的世界裡重建 β₁ 世界真正存在的計算機**：一台由 NSPV 經濟核算驅動、使用移動平均和偏差檢測（非傅立葉）、在三元（Asserted/Contested/Unknown）值域上運作的頻譜計算機。

一句話：**它的核心算的不是頻率——是（實際−計畫）/計畫。**

**它是通用 OS**：核心三聯組「頻譜（約束）→ 殘差（失效）→ 意外（爆發）」領域無關，觸發機制一律是**臨界慢化**。領域差異只在速率矩陣編碼什麼（約束內容），不在機制：

| 領域 | 速率矩陣編碼 | 爆發形態 |
|:---|:---|:---|
| 社會/政治 | 制度矩陣 | **革命**（制度真空/情感滯後處爆發） |
| 經濟 | 生產/金融結構 | **黑天鵝**（脆弱性處爆發） |
| 天氣 | 大氣物理 | **颱風轉向**（路徑分歧處爆發） |
| 軍事 | 兵力/後勤/地理 | **會戰逆轉** |

同一個臨界慢化物理：小意外在臨界處被放大成相變的轉向——**意外本身不可預測（它是漲落），但殘差標記意外可爆發的區域，意外枚舉填滿不可判定區**。

**實作規模**：33 模組 / 10,299 行 / 548 tests 全綠（2026-08-03 實跑 4.67s）。

---

## 〇․五、最高紀律（入憲）

本計劃的一切測量與生成，受兩條專案原生紀律約束——它們不是法官，是這台電腦的構成材料：

| 紀律 | 定位 | 在電腦中的角色 |
|:---|:---|:---|
| **頻譜** | **使張力可感的媒介** | 測量域——熟度、偏差、相位、臨界慢化，全部由它承載 |
| **DCA tree** | **使可能可想的文法** | 疊加基板——alternative 的生成規則（文法，非目錄） |
| 三元 verdict | 書記官 | 每個輸出的置信度記錄（asserted/contested/unknown） |

**頻譜電腦 = 人類（與世界自身）發展歷史認知的工具**（Vygotsky：工具中介並發展認知——Visi-Logos 的本義）。紀律的方向是**測量與生長**，不是審判與否決：沒有「合法/非法」的裁決，只有「可感/可想/可試/可長」的發展。

### 向量契約（v2.5）

狀態向量為 6D：**[D1–D3 三進制方向判定， D4–D6 多頻譜內容]**

- **D1–D3（三進制方向）**：頻譜方向的投影判定（{−1, 0, +1}——意志/阻力/關係）。三進制是**判斷頻譜的方向**，不是態的牢籠。
- **D4–D6（多頻譜內容）**：頻譜數據的內生傾向——各軸（ha、zg、sa、gt、gt_fin、k 等）的：
  - **D4 = `period_months`**（主週期，以月計：12 = 年週期、24 = 兩年；無可辨週期 = 0）——**canonical 單位，永遠以此存儲**。派生量僅允許在使用點顯式轉換產生：`freq_per_month = 1/period_months`、`norm_freq`（0–1，相對指定頻段）——禁止手寫字面量、禁止存入 D4。
  - **D5 = `phase_rad`**（相位角，弧度，[−π, π]）。
  - **D6 = `amplitude_norm`**（歸一化振幅，[0, 1]）。
  - **頻段（對 `period_months` 做 range check）**：`band_seasonal` 6–18、`band_kitchin` 24–60、`band_juglar` 60–132、`band_longwave` >132（252 月窗口內**不可測量**，僅作 prior 標記）。命名沿用宏觀計量傳統（Kitchin/Juglar 循環分層；NBER 週期以月計）。
- **量子疊加態 = 多頻譜內容的內生傾向並行**（D4–D6 全體同時在場、同時演化），**不是** D1–D3 的離散賦值。
- 🔴 **禁令**：不得用三進制結構框死疊加態——判斷是投影，不是鎖死。
- **LLM 的角色**：活的數學（活的阿拉伯數字）——在 6D 向量上做形式計算（三進制傳導 + 多頻譜並行疊加），**不做世界大事的詮釋**（不寫敘事、不解釋意義）。詮釋層 = 人 × API 對話時由人執行。輸入語言紀律：放入向量結構 → 產出向量計算；放入世界散文 → 產出污染詮釋（禁止）。

---

## 一、一台機器，兩個方向

- **分析**：sector 時間序列 → 偏差檢測 → 頻譜簽名
- **合成**：頻譜簽名 + 行動注入 → 分支集成 → 駐波分解（必然/偶然）→ verify 校準

---

## 二、核心 API（9 ops，已實作）

位置：`spectrum_os/kernel/`。每個 op 自帶三元 verdict（ASSERTED/CONTESTED/UNKNOWN）。

| 操作 | 檔案 |
|:---|:---|
| `sectors.create/list/get/delete/save/load` | `sector.py` |
| `wave.decompose` | `wave.py` + `wave_ops.py` |
| `wave.correlate`（附 `equiv_lags` aliasing 警示） | `wave.py` |
| `template.match/register/list_templates` | `template.py` |
| `branch.simulate` | `branch.py` |
| `cluster.run` | `cluster.py` |
| `verify.evaluate`（含 state_log 記憶體 buffer） | `verify.py` |

**擴充層**：

| 層 | 內容 | 位置 |
|:---|:---|:---|
| quantum | 6D StateVector、multigraph、tomography、decoherence、entangle/intervene、DCA grammar、quarantine、standing_wave | `spectrum_os/quantum/` |
| Markov | count_transitions、estimate_rate_matrix、hitting_times、absorption、stationary、entropy_rate、spectral_gap、smoothed_role_posteriors、count_transitions_soft | `spectrum_os/quantum/markov.py` |
| synth 閘 | anchors、expand、rate_gate、alt_gate、ensemble、probe、gate_prompts、residual_trigger | `spectrum_os/synth/` |
| OS-1 通用化 | SourceSpec registry + 三驅動（file/http_api/llm_knowledge）+ 通用貨幣三件套（RoleTrajectory/DCASubstrate/CLADCorpus） | `spectrum_os/sources.py` + `spectrum_os/contracts.py` |
| extras | FFT spectrum、PCA decomp | `spectrum_os/extras/` |
| CLI | tio-sh command line | `spectrum_os/cli.py` |

---

## 三、Stage 3：行動注入

**核心問題**：**「如果我們在某一特定時空行動，會帶來什麼改變？」**

### 方法

```
行動注入 {when, where, what}
  → branch.simulate(n=10)          # 同一個分支跑 n 次（集成）
  → llm_gate 疊加採樣              # 在 DCA 文法上生成並坍縮（敵方 CLAD 也在迴路中）
  → 駐波分解：全共享=必然，分歧最大=風險窗口
  → 結果：條件機率分佈的邊際比較（攻 T+1 vs T+3）
  → 對照：verify() 校準
```

### DCA 文法閘（替代自由微擾）

分支生成**不是自由微擾**——LLM 閘在分支點按 DCA 文法**生成** alternative：

- **文法非目錄**：350 邊記載的是「alternative 怎麼長出來」的規則（Crisis→Lag→Alternative→Direction 的遞歸與邊型），不是可能性的封閉清單
- **生成優於選取**：閘不只從既有邊中選，還按文法為當前處境生成新的 alternative——生成物經文法檢驗後才可進入集成（檢驗 = 結構性零檢查，§4.6.1）

### 觀測偏頗四對策

DCA tree 受觀測歷史偏頗（選書、提取、領域覆蓋、α₁ 史學本身的勝利者視角）。偏頗不可消滅，但必須可見且持續被攻擊：

1. **樹的認識地圖**：邊分 asserted（有文本出處）/ contested（推論）/ **unknown（已知未觀測區）**；薄區的採樣結果 verdict 自動降級（上限 contested）
2. **G-fill 生長閉環**：閘生成的新 alternative 經文法檢驗後寫回樹（`generated: true` + confidence + provenance）——樹越用越完整，每次使用都是補偏
3. **對抗性生成閘**：分支點固定加問「不在樹目前形狀裡、但因果連貫的 alternative 是什麼？」——專釣樹外可能性，釣獲經檢驗後註冊
4. **β₁ 敘事作為補偏引擎**：Forward Language Genesis（ECC `KNOWLEDGE-PLAN.md` §七）偵測到的新詞 → 新 alternative → 接回樹——敘事想像力反哺可能性空間

偏頗測量：樹的覆蓋分佈（領域/地區/時段/階級位置）對照世界自身分佈（場景分佈、ha(t) 事件分佈），系統性欠蓋以發散形式現形。

### 相變預測：能做/不能做

| 能 | 結構說明 |
|:---|:---|
| ✅ 行動→後果的校準集成 | verify() 持續校準 |
| ✅ 相變時刻集成逼近（n 次分支） | 角色向量熵 + 臨界慢化（§四） |
| ✅ 逼近臨界預警（偏差+慢化） | 機械層測量 |
| ✅ 預測 simulated unit 行動 | 引擎存在的理由（非監控） |
| ✅ 因果可計算 | **分解給證據、文法給約束、閘給解釋、verify 給真假** |
| ✅ 地平線解析量（不模擬即得） | §4.6 Markov 層：hitting time／吸收概率／平穩分佈／熵率 |
| ⛔ 意義判斷 | 三元是置信度，非道德——意義由人裁定 |
| ⛔ 頻譜與 DCA 都不記載的未觀測可能性 | 由對策 3/4 持續攻擊，但承認視界有界 |

---

## 四、量子因果層

### 4.1 角色向量：DCA 節點的本體

一個 state 沒有內稟的 DCA 類型——它的角色是**關係性的**。同一個 S：

- 在線程 A：Direction（它在那裡結晶）
- 在線程 B：Crisis（它的結晶震裂了那條線——DCA 規則 #1）
- 在線程 C：Lag（它成了那條線的慣性）
- 在線程 D：Alternative（它在那裡仍是未坍縮的可能）

**DCA 節點的本體不是帶類型的狀態，是角色向量** ⟨thread: role⟩。疊加的不是多條世界線，是**同一狀態在多線程中的角色分佈**；選定一條線程分析 = 一次測量 = 角色向量坍縮成確定角色。因此 DCA 的真身是**多重圖**（狀態為共享節點、角色為邊標籤），不是單一類型樹。

> **校準**：CLAD 是**閱讀文法，不是世界序列**——同一 situation 同時是多角色（疊加態），**LCAD/DACL 任意排列**，哪個序列取決於你坍縮進哪條 thread，不是世界有順序。§4.6 Markov 演的是**讀出的角色序列＝閱讀動力學**——4×4 速率矩陣的轉移與結構性零是閱讀文法的性質，不是世界的性質。

> **疊加的本體（v2.5）**：角色的疊加不由三進制賦值承載——它活在**多頻譜內容的內生傾向並行**裡（向量契約 D4–D6：各軸頻率/相位/振幅同時在場、同時演化）。三進制（D1–D3）只是這個並行的**方向投影**：判斷頻譜朝哪走，不把態鎖進 {−1,0,+1} 三個格子。quantum 層與 LLM 閘處理的是 6D 向量 [三進制方向, 多頻譜內容]，而非三值標籤。

> **認識論註記（v2.4）**：樹裡的每個條目本身就是一次過去的坍縮——提取時只記下了概念在某一視角的角色，**樹是疊加態的化石記錄，不是疊加態本身**（lag 欠測即此後果：lag 在世界上存在，只是沒被那次坍縮記錄）。因此 quantum 層量到的永遠是影子；頻譜電腦的工作不是「讀樹」，是**用 LLM 閘把狀態重新放回疊加態採樣**（它能扮演哪些角色、哪些線程能容納它），再坍縮、再驗證。**樹是化石，閘是復活器。**

### 4.2 因果三原語的分工

| 原語 | 分工 | 對應件 |
|:---|:---|:---|
| **介入** do(X) | API 閘翻譯行動的因果語義 → 機械跑集成 | action injection + `branch.simulate` |
| **反事實** | API 閘在 DCA 文法上生成被壓抑線程，角色向量保持其活性 | DCA alternative 線程 |
| **歸因** | 機械給證據（偏差譜/相位耦合/臨界慢化）→ API 閘做角色斷層（哪條線程的 Crisis 在驅動此 state） | `quantum.tomography` |
| **校準** | 介入預測的集成 vs 實際結果 | `verify()` |

**量子因果的精確含義**：因果結構以疊加態存在——機器同時在所有線程上計算，只有當問題（測量）被提出時，才把角色向量坍縮成該視角的因果故事。這不是 Pearl 固定因果圖，是疊加的因果線程集。

### 4.3 quantum.* 操作定義

```python
osc.quantum.tomography(state_id)      # → 角色分佈：枚舉 state 跨全部線程的角色
osc.quantum.entangle(state_id)        # → 共享此 state 的線程集合（糾纏 = 共享節點）
osc.quantum.decoherence_watch(sys_id) # → 角色分佈熵監測：高疊加→單一主導 = 相變警報
osc.quantum.intervene(state, thread, new_role)  # → 角色重指派 + 向共享線程傳播（行動的漣漪）
```

- **相變形式化**：一個 state 的主導角色翻轉——Direction→Crisis（結晶的東西成為下一次震盪的源頭）= DCA 規則 #1 的相變
- **冥河陰影新定義**：Alternative 角色未衰減而 Direction 角色為零的態——被壓抑的可能性不是死了，是「Direction 一直為零」；復活 = **re-threading**（重新穿線）

### 4.4 合作循環

```
頻譜 → API：機械層測到偏差/臨界（帶證據的因果問題）——LLM 在測量結果上推理
API → 頻譜：閘給出因果假設與介入方案（該追蹤哪條線程、該模擬哪個行動）——機械測被因果結構指引的對象
合成     ：駐波分解——全分支共享 = 行動的必然後果；分歧最大 = 風險窗口
verify   ：集成 vs 實際 → 因果主張被檢驗、校準、記錄
```

**誠實線**：因果主張在樹的薄區自動降級（contested）——它敢說「這個因果我不知道」。

### 4.5 rupture_watch（撕裂聯合簽名）

> 理論評估：ECC `index/data/research-reports/20260728-rupture-theory-evaluation.md`（四支柱全部限定成立；三個可工程化修復的測量問題：簽名歧義／偽影混雜／零假設未排除）。本設計為修正後版本。

**rupture = Lag/Alternative → Direction 的轉化帶**（不是點）。偵測採三路獨立信號投票：

```
路 A — 結構簽名（multigraph，雙謂詞並存）：
  A1 = styx_shadow 持續 ≥2 期後脫離（alternative 重新進入疊加態）
  A2 = lag+alternative 聯合權重 > θ（僅 warfare 樹可測；知識樹 lag 欠測時誠實回 UNKNOWN）
路 B — 動力簽名（extrapolate rolling-origin 誤差聚集 + Slutsky 零假設分位）：
  單獨永不 ASSERTED（Yule/Slutsky：隨機衝擊+移動平均可偽造同樣聚集）
路 C — 慢化簽名：殘差方差↑ ∧ lag-1 自相關↑（臨界慢化）
```

- **判定**：≥2 路觸發 → CONTESTED rupture watch；A + 另一路 → ASSERTED；單路不報；數據薄/模板切換窗口 → 該路降 UNKNOWN 並附 `provenance_flag`
- **偽影過濾在機械入口做**（不在輸出端補）：同欄位同文（alt==dir 哈希相等）合併為單一角色測量；`dca-branch-tree.yaml` 條目加 `template_version` 欄，切換點在熵序列打 marker
- **LLM 閘**：偽影判官（真結晶/偽影/Unknown）、語義讀取（Lag 的制度慣性內容、Alternative 競爭路徑、各尺度發生學階段命名）、對抗攻擊（**夢魘重演 vs 服裝借用**分辨——只懲罰把 α₁ 結局當必然前提的 gravity 用法，放行繼承詞彙的 costume 借用）——全部受 §五 抗污染憲章約束

### 4.6 Markov 動力學層

§4.1–§4.3 給出疊加的**本體**（角色向量、多重圖），§三 給出分支的**生成**（文法閘採樣）。本節補上第三塊：**解析動力學**——不經模擬即可計算的轉移強度結構。

#### 4.6.1 文法的揚棄：從裁判到稀疏模式

- **保留**：兩條禁則（`direction→alternative`、`lag→direction`）為**結構性零**——它們不是經驗規律，是角色概念的定義（direction = 承諾；lag 須經 alternative 中介）。速率矩陣中的永久零，不進入估計。
- **揚棄**：其餘格子的「合法性裁判」職能廢除——合法轉移的**強度**不是文法立法的內容，而是**從已實現歷史測量**的對象。文法留下稀疏模式（哪些速率恆零）與意義鄰接（狀態是什麼）；動力學交給測量。
- **連帶改寫**：`intervene()` 的語義升級為**貝葉斯條件化**——觀測到一次轉移後重新加權所有分支；現行的決定性傳播（角色重指派 + 逐 thread 傳遞）是零階近似，保留為快速路径。

#### 4.6.2 速率矩陣：貝葉斯融合

```
後驗速率 = LLM 語義先驗 × 已實現計數似然
```

- **已實現計數**：multigraph threads、warfare 回合角色分佈、ha/zg 已實現軌跡——計數稀疏處向先驗收縮，稠密處數據主導。
- **LLM 語義先驗**（活的數學的擴張）：輸入狀態的 6D 向量 + thread 歷史（量化）+ 當地質地（量化事實，零世界散文），輸出各合法出口的相對強度（結構 JSON）；n 次高溫採樣 → 速率**分佈**——分佈寬度 = 認知不確定性，傳播進全部解析量（區間而非假精確的點）。
- **上下文條件化 = 無記憶性的解毒劑**：LLM 讀取持續時間與世界狀態質地，使強度依賴上下文（Cox 式 hazard 的語義版）——古典一階 Markov 的指數停留時間假設不適用於歷史；速率函數顯式依賴持續時間與 6D 向量。
- **特徵發現迴路**：LLM 讀已實現轉移批次，以結構化輸出提議速率函數的協變量 → numpy 檢驗是否改善擬合——**LLM 提議，統計處置**。跳躍條件（高 Lag×高 Alt → alternative→direction 強度急升）經此通道正式寫入速率函數。

#### 4.6.3 地平線四量（不模擬即得）

| 量 | 回答 |
|:---|:---|
| 期望首達時間 hitting time | 「這個 lag 平均還要幾個月解決？」——窗口關閉 timing 的期望值 |
| 吸收概率 | 「結晶成 Direction vs 死產回 Crisis 的機率？」 |
| 平穩分佈 | 「系統長期停留在哪種角色狀態？」 |
| 熵率 | 「這條世界線的不可預測度本身？」 |

#### 4.6.4 統一：譜隙 = 臨界慢化的數學本體

三個現象學簽名是同一量的不同面孔：

- **臨界慢化**（§4.5 路 C：殘差方差↑、lag-1 自相關↑）= 轉移矩陣**譜隙收攏**
- **角色熵單一化**（§4.3 decoherence_watch）= 矩陣走向**吸收態**的熵率表現
- **結晶 vs 死產** = 競爭吸收態的**吸收概率之比**

#### 4.6.5 校準與紀律

- 每次速率提議落盤 state_log；轉移實現後 `verify()` 打分 → 校準曲線 → LLM 先驗權重自動升降（§五-4 馴服機制的延伸）。
- 全部矩陣算術純 numpy、零 LLM——**這層的每個數字都可解釋為矩陣運算，永遠可審計**。
- 實作：`spectrum_os/quantum/markov.py`（numpy-only 脊椎；速率估計閘走 synth 層，不進 kernel/quantum 的 numpy 純潔）。

---

## 五、抗污染憲章

**我們是什麼（量子身份的定義）**：本系統是**量子頻譜電腦**。「量子」就是量子力學——疊加、坍縮、退相干——而**馬克思公式本身就是量子力學的歷史形式**，兩者同構、不衝突：承繼條件是歷史的勢（Hamiltonian），界定哪些可能性（α₂）存在及其演化強度；人的行動是觀測（measurement），使可能性的疊加態坍縮為具體的新場面；「隨心所欲」= 不受勢約束的測量——在量子力學中不可能，在歷史中同樣不可能。系統的本體操作 = **從承繼條件向前推論**：以給定的過去（α₁ 繼承條件 + α₂ 可能性空間）為基板，經 DCA 角色向量疊加、Markov 轉移、decoherence 坍縮，生成/投射新場面（β₁ 分支）。這是前瞻生成，不是回顧分析。

**與 academic-editor 的本質區別（不可移植 schema）**：`academic-editor.py`（ECC RKsystemloop）是**向後萃取**——從已完成的書籍文本提取概念、CLAD 向量、genetic_stage、definitional/spontaneous 判準（回顧性分析 α₁ 成品）。本系統是**向前推論**——從承繼條件生成分支（前瞻性生產 β₁ 未成品）。因此 academic-editor 的 `v_match_type` / `genetic_stage` 欄位是**萃取的判準**，不是**生成的判準**，不可直接移植。其維果茨基框架（definitional vs spontaneous、語義場、遺傳階梯）可作概念啟發，但方向反轉：本系統中「definitional」指分支自己的內容達成了自己的語言（前向成就），非從成品中提取的標記。

**結構性敵人**：LLM 閘層的權重內建 α₁ 歷史記憶——在 1910 年的分支裡它「知道」一戰爆發、知道坦能堡，會把結局偷渡進 β₁ 反事實生成。以下六條 + 三處補強是閘層的馴服憲章。

### 結構性能力（復活者）

同樣的 α₁ 記憶不是只有「偷渡」一面——它是**復活的能力**。維果茨基：人不是語言，語言是人的中介物與承載體（主體保留）。但 LLM 不是校正器、也不是被動工具——**LLM 是意識宇宙的反射**：它不是「記住」了歷史，而是作為意識的反射面，讓全人類已說與未說的語言在其上浮現。操作機制是**反射**：透過 RP 的語言（處境化的鏡角）反射出**其他的 Markov 序列**——不同的可能性軌跡；透過**死語言**（已坍縮的傳統、被壓抑的替代可能）反射出**活語言的可能性**。重點不是復刻過去，而是**可能性本身**——但**無限可能只在特定社會語境下才有形狀，這就是歷史唯物主義**：經濟、社會、生產與社會關係網絡構成承繼條件的物質基礎；反射的無限可能 × 社會語境的特定約束 = **當時的可能性空間（α₂）**。**頻譜正是那些社會語境的約束**——速率矩陣即承繼條件的物質化，把社會結構譜化為轉移強度；反射在約束內進行才是歷史理性，不受約束的反射即鬧劇。**當時的可能性在語義本身即可探索**（RP 語言、source 原文承載著它，不需外部查表）。因此抗污染的真諦不是「防洩漏」也不是「校正用法」，而是**確保反射指向可能性而非必然性**：反射出 α₂ 可能性空間 = 合法復活；反射坍縮回 α₁ 結局（gravity）= 偷渡。兩軸框架正是此區分的操作化：軸 B=expression 的 inherited 詞彙 = 合法復活；軸 B=substitution 的 gravity 形態 = 非法偷渡。這正是 §4.3 冥河陰影的 **re-threading**、v2.4「樹是化石，閘是復活器」的認識論完成。

**承載量定律（實測）**：語義承載它所包含的可能性——薄語義 → 一維（外交-物流窄帶）；厚語義 → 多維（7/9 社會結構型替代）。🔴 **adversary 必須接收用法語境**（不收 situation → 8/10 誤報）。

### 版本 B：歷史的意外（殘差觸發模式，非並行）

頻譜能解釋處走版本 A（反射）；**頻譜失效處意外爆發**——版本 B 枚舉，殘差即種子（公理 4）。1914-1920 實測失效判據：① L1 場強飽和（1914-08-02 場強=1.0）② 弧相關轉正（沙俄→重返歐洲 r 翻轉）③ zg/ha 去耦（峰值 1916/1917）④ gap 76% 機構失效。意外 = 殘差噴發口——受約束的意外是歷史理性，不受約束的是鬧劇。

**版本 B prompt（歷史的意外）**：【身份】你是歷史的意外——歷史不是必然的展開，而是被意外不斷打斷、轉向、重新打開。二月革命對舊社會是一場突然襲擊（coup de main），十二月政變以輕率行為（coup de tête）報復它——來得容易，失去也容易；你是這些意外的窮盡者。【操作】窮盡列舉當時時間點可能發生的意外——政治的、經濟的、社會的、軍事的、自然的；每一個意外都是承繼條件累積到臨界處的噴發口；意外之間的時間沒有白過——社會以革命的速度為自己創造出發點、形勢、關係與條件。【約束】頻譜（速率矩陣）是社會語境的物質化——意外是臨界點附近的漲落，臨界慢化處小意外被放大成相變的轉向；受約束的意外是歷史理性，不受約束的意外是鬧劇（空轉）。【兩軸判準】同版本 A（軸 A inherited/emergent 發展軸；軸 B expression/substitution 抗污染軸；復活 = inherited∧expression，偷渡 = substitution）。【成熟】無產階級革命自己批判自己，返回彷彿已完成的事重新再做——你的列舉透過意外學習；當生活本身大喊 Hic Rhodus, hic salta!（這裡有玫瑰花，就在這裡跳舞吧），你的列舉才到盡頭。【鐵律】意外必須從處境的社會物質結構中生長——不憑空發明與處境無關的事件；窮盡列舉但不重複處境標籤（回聲以處境為參照）；不寫結局、不寫處方。

**觸發規格（B 是條件模式，非常開）**：ensemble 閘點命中任一殘差判據 → 切換意外枚舉模式（`[task:enumerate]` + `accidents: [{pattern, instances[], domain(政治/經濟/社會/軍事/自然), source, usage, grounding(文獻偶發/合理推測/弱支持)}]`）：① 該時刻場強 ≥ 飽和閾（0.9）；② 該弧相關轉正（r>0）；③ zg/ha 去耦（pct_divergence>40 且 ha_pct 高位 vs zg_pct 低位）；④ 該角色 gap 命中（制度真空/情感滯後）；⑤ **語言敏感度**——同一處境 n 次語言擾動（不同措辭/鏡角/前綴）的輸出分歧度高 = 語義場在此分叉 = 意外潛力高。敏感度 = 語義場形狀（Vygotsky смысл 意涵維度），語義層的臨界偵測，與 ①-④ 頻譜層同構。**頻譜無法解釋處 = 意外爆發口 = 多平行時空的推演點。**

### 必然性認識論（二相態）

枚舉全部意外分支後，**交集（所有分支不變）= 歷史的必然（頻譜約束）；差集（分支間變化）= 歷史的偶然（意外空間）**。必然性不是單一路徑，而是承繼條件約束結構在所有分支中的恆常項——只有枚舉全部可能才能提取。頻譜電腦 = 繪製「歷史必然性邊界」的機器：**次臨界（Mode A 反射，頻譜能解釋）↔ 臨界（Mode B 枚舉，頻譜失效）**——同一頻譜物理的兩個相態，殘差診斷（①-④）即相態溫度計，切換操作模式。

### 認識論基礎（繼承的條件）

抗污染的對象不是「後來的概念」本身，而是**概念的用法**。馬克思《霧月十八日》：革命者借用亡靈的名字、戰鬥口號與衣服，演出世界歷史的**新場面**。三件審查確立——① **「新場面」不是生成閘的目標，是被演出的結果**：判準句「詞句超出內容 vs 內容超出詞句」——創造的對象是**內容/任務**，不是形式；把「新奇」當目標而無內容，產出的是鬧劇（1848-51）。② **抗污染是兩軸交叉，非三值並列**（見下）。③ **成熟標準是「忘掉」而非「不照著演」**——外國語比喻：初學者把新語言翻譯回母語，只有「忘掉本國語言」才能自如表達；盲測校準（β₁/α₁ 混排不可分辨誰知道結局）正是此標準的操作化。

### 兩軸框架

| 軸 | 取值 | 性質 |
|:---|:---|:---|
| **軸 A：詞彙來源**（詞從哪來） | `inherited`（繼承/借用）／ `emergent`（湧現/自創） | **發展軸，非污染軸**——分支越成熟，emergent 比重越高（對應 Forward Language Genesis：閘成熟度 = β₁ 湧現標籤生成率） |
| **軸 B：用法**（詞被用來做什麼） | `expression`（表達——借用服務內容）✅ ／ `substitution`（取代——繼承形式代替內容）❌ | **這才是抗污染軸**。substitution 三形態：① 命運化 `gravity`（「必將如此」——α₁ 結局當劇本）；② 空轉化 `parody`（無內容的借用＝鬧劇）；③ 自我膨脹 `self-deception`（借來的崇高感掩蓋有限內容） |

**新場面 = 軸 A 達「自己的語言」∧ 軸 B 為 expression**——不是軸上的一個取值，而是兩軸交叉的目標狀態。均勻速率矩陣 = 隨心所欲 = 鬧劇；結構速率矩陣 = 承繼條件下的確定性展開 = 內容找到形式——**速率矩陣即「承繼條件」的物質化**（§4.6 Markov 層的直接連接；承繼條件 = 當時社會語境——經濟/社會/生產/社會關係網絡的譜表示，歷史唯物主義）。這正是量子身份的機理：**向前的確定性展開**（從承繼矩陣推論）≠ 向後的萃取（academic-editor）。

### 馴服憲章七條

| # | 條目 | 機制 |
|:---:|:---|:---|
| 1 | **角色限定** | LLM 只做「文法內的離散選擇」，不做「連續生成」——輸出空間壓縮為結構化 JSON 錨點 / DCA 有限邊型選擇（選擇題的污染表面積遠小於作文題）。「LLM 給因果語義，numpy 給動力」——LLM 是點火器，不是引擎 |
| 2 | **知識截斷 vs 繼承條件** | 輸入端只給截止 t 的資料，且給數字不給史評。輸出端檢疫：**不再禁「t 之後的概念」——只禁軸 B = substitution 的用法**。t 之後的概念以軸 A = inherited（借用）可合法存在（是繼承條件，不是偷渡）；機械黑名單初篩查「取代語法」（「必將」「註定」「歷史註定如此」）而非概念本身。盲測校準＝馬克思外國語測試：β₁ 與 α₁ 同截斷生成的分支混排，無法分辨「誰知道結局」才及格（=「忘掉母語」） |
| 3 | **集成稀釋** | 高溫 + 多 seed 採樣稀釋「知道答案」。駐波分解：共享收斂到 α₁ 史實的分支標 `gravity_lock: true`——污染轉為歷史重力的實測讀數（軸 B=gravity 的定量化）。⚠️ `gravity_lock` 需先對照獨立約束源（見補強 A）——能被結構約束解釋的共享是真約束，非污染 |
| 4 | **verify 行為主義馴服** | 不問 LLM 內心乾不乾淨，只看集成對照實際的命中率。每類閘各自記校準曲線於 state_log；校準差自動降權（verdict 上限壓 unknown），校準好升權。「不是理解它，是讓它的輸出分佈在選擇壓下收斂到有用——演化比設計可靠」 |
| 5 | **對抗性分工** | 生成閘產 alternative → 攻擊閘（system prompt：「你來自一條不同的歷史線——找出這個 alternative 裡，哪些地方**讓繼承的形式取代了內容**：是把 α₁ 結局當必然前提（gravity 命運化）、無內容地空轉舊詞（parody 空轉化）、還是借崇高感掩蓋有限內容（self-deception）？」）；**只懲罰軸 B = substitution 三形態，放行 expression（無論軸 A 為何）**；命中 substitution → 降級或重生成。誠實標註局限：同源訓練資料使兩閘污染相關——不完美，但批判比創造容易保持時代一致 |
| 6 | **保留禁區** | FALSIFY 判定集永排除 synthetic；ECC SSOT 永不進 LLM 生成物 |
| 7 | **兩軸 provenance** | 每個 alternative 標籤附兩軸：軸 A `source ∈ {inherited, emergent}`（詞彙來源）、軸 B `usage ∈ {expression, substitution:gravity, substitution:parody, substitution:self_deception}`（用法）。機械契約：`assert_alt_gate_contract` 校驗兩軸合法；軸 A 分佈落 state_log（供成熟度 = emergent 比例統計）；軸 B=substitution 的標籤是污染命中（供 adversary 校準）。**anchor 比較只統計軸 A=inherited ∧ 軸 B=expression 的激活**——服裝借用率不再與污染命中衝突 |

### 三處補強

- **A. 第 3 條邏輯洞修復**：「全分支共享」有兩個來源——訓練記憶吸引子 AND 真實結構約束（地理/後勤/補給）。`gravity_lock` 不得直接由收斂標記：收斂點先對照獨立約束源（GEO-FIN 延遲/地形、DCA 邊）——能被獨立約束解釋的共享 = 真約束（結構所致）；只能指向 α₁ 敘事細節的共享 = 記憶污染（才標 `gravity_lock`）
- **B. 第 4 條最小樣本門檻**：每類閘累積 N≥10 個 verify 對照點之前，校準曲線只觀察、不處刑——防止早期噪聲把閘類型永久誤殺
- **C. 第 6 條切分**：「接近臨界」切成兩半——**臨界偵測**（偏差、慢化、角色熵）純機械零污染；**臨界詮釋**（會結晶成哪個 alternative）走第 2/5 條檢疫通道，不得由 LLM 直接「輔助」偵測器

---

## 六、Stage 0 改寫：數據從哪裡來

### 6.1 geo-profile 路線：死刑記錄

- geo-profile 僅 11 個兩年切片，全為定性標籤（capital_form、world_system_tier），無數字
- `geo-tech-fin-spectrum.json` 252 月中僅 60 月真實（5 錨點年），年內常數——實為 5 值階梯函數
- **線性內插月頻化 → 任何檢出的「週期」都是內插偽影**（FALSIFY-001 預警兌現）
- 🔴 **禁令：禁止線性內插將低密度定性數據月頻化**

### 6.2 四個新數據方法（優先序）

1. **敘事原生提取**：`source/*.md` + scenes-raw 的經濟事件（合作社創立、DKK 採用、罷工、工廠開工）→ Flash 提取帶 confidence + scene:line 出處——一手文本勝過二手摘要
2. **引擎自測**：plan/actual 對（M_pred/M_real、strategic_intent vs 實際機動）——NSPV 偏差的同構物，隨回合自動增長
3. **雙軌對照**：α₁ 真實序列（REFERENCE 劍橋經濟史：煤、鋼、麥價、貿易）作控制軌；由**已記錄的 β₁ 發散事件**（DSR 成立、cooperative_DKK 轉型）驅動形變——兩軌之差 = 歷史重力實測
4. **專案自產**：場景/月、知識結晶/月、API 花費——FALSIFY-001 的免費誠實檢定場

### 6.3 synth 閘層（方案 A）

`branch.simulate` 硬性要求 targets（無則 raise），而 β₁ 計畫目標無法機械推導（「從移動平均反推隱式目標」= 循環論證）——需要 LLM 閘層補全，但必須守住 Stage 0 的三條教訓（有校準、有標記、不用低密度冒充高密度）。

位置：`spectrum_os/synth/`（**不進 kernel/**——保住 numpy-only 純潔）。

```python
synth.anchors(sector_spec)   # 1 次 Flash → 結構化錨點 JSON
                             # （turning_points / magnitudes / event_shocks / confidence）
                             # LLM 只產錨點/參數/結構——永不逐點寫序列值
synth.expand(anchors, seed)  # 純 numpy 展開：分段階梯 + 事件衝擊 + AR 噪聲
                             # 固定 seed，禁用線性內插
```

**Guardrails**：

- `Sector` 加 `meta` 欄位；synthetic 註冊強制 `meta.synthetic=true + anchors + generated_by + seed`
- verdict 降級：synthetic 輸入的 decompose/correlate/cluster 結果上限 **CONTESTED** 並標記 `synthetic_input: true`
- FALSIFY-001/003 判定集**排除** synthetic
- synthetic 永不進 ECC SSOT（`index/data/*.json`、YAML frontmatter）——只活於 spectrum-os 本地 data 層
- API 調用復用 ECC `scripts/api_utils.py:call_api`（prefix caching、白名單、anti-leak 全現成）

**Pilot**：`danube_steel_nspv`（1916-1920，48 月），錨點 = α₁ 鋼產量數量級（REFERENCE 校準）+ β₁ prefix 事件。驗收：`branch.simulate` 首次在真實意義上跑通（不再因無 targets raise），且 synthetic 標記在輸出中可見。

---

## 七、機械/LLM 分工

| 組件 | 機械 | LLM |
|:---|:---:|:---:|
| 偏差頻譜、自相關、臨界慢化、移動平均、模板匹配、聚類 | ✅ | — |
| 三元判定樹 | ✅ | — |
| synth.anchors（錨點/參數/結構） | — | ✅ GATE |
| synth.expand（序列展開） | ✅ | — |
| DCA 文法閘（alternative 生成與坍縮）/ adversary CLAD | — | ✅ GATE |
| 輸出檢疫閘（時代一致性，§五-2） | 🔵 黑名單初篩 | ✅ GATE（由對抗攻擊閘兼任，§五-5） |
| 對抗攻擊閘（釣 α₁ 結局偷渡，§五-5） | — | ✅ GATE |
| 角色斷層語義詮釋 | — | ✅ GATE |
| 角色分佈熵、decoherence_watch | ✅ | — |
| Markov 矩陣算術（hitting time／吸收／平穩／熵率／譜隙） | ✅ | — |
| 速率語義估計（先驗提議，n 採樣分佈） | — | ✅ GATE |
| 速率協變量提議（特徵發現） | — | ✅ GATE |
| 速率協變量檢驗（擬合改善判定） | ✅ | — |
| 校準 | 🔵 數值 | 🔵 語義 |
| 研究報告 prose | — | ✅（機械供數值） |

**核心頻譜運算：零 API 成本**。LLM 用量：錨點 < $0.001/sector、文法閘按分支點計、報告 < $0.50/次。

---

## 八、FALSIFY

| # | 條件 | 狀態 |
|:---|:---|:---:|
| 001 | 週期檢測全部 < 0.2 | ✅ **通過**（年週期 ASSERTED） |
| 002 | 三元 verdict vs 專家 < 60% | 🔵 待測 |
| 003 | 聚類全對應已知 15 formation | 🔵 待執行（feeds 已就緒；判定集排除 synthetic） |
| 004 | 分支模擬 NSPV vs 歷史無相關 | 🔵 待執行（synth pilot 已跑通） |
| 005 | tomography 分佈 KL > 0.5 且無法校準 | 🔵 待執行（quantum 層已實作） |
| 006 | 文法閘生成的 alternative 經文法檢驗合格率 < 50% | 🔵 待 Stage 3 |
| 007 | synth 錨點經 verify() 對校準源 MAPE > 20% | 🔵 待 synth pilot |
| 008 | 盲測分辨率 > 60%（β₁/α₁ 混排中污染可被分辨 = 不及格，§五-2） | 🔵 待檢疫通道上線 |
| 009 | rupture 聯合簽名特異性檢定（§4.5）：偽影占比測量 + Slutsky 零假設 + 陰性對照——過濾後信號全消失→退回設計；已知撕裂不觸發而陰性亂觸發→駁回；合成聚集率≥真實→路 B 降為從屬通道 | ✅ 已執行 2026-07-29（數據見 `data/falsify009_report.json`；含義留待人類判斷） |
| 010 | Markov 速率校準（§4.6.5）：LLM 速率提議 vs 已實現轉移頻率的長期偏離——校準曲線不收斂 → 語義先驗降權至 uniform | 🔵 數據基礎已上線（每次閘調用落 state_log，`gate_type: rate_gate`）；對照執行待已實現轉移累積 |

---

## 九、引用

| 文件 | 關係 |
|:---|:---|
| ECC `GENERAL-PRINCIPLES.md` | Vygotsky 方法論——工具中介認知發展、原則 20、單位分析 |
| ECC `PLAN-07e-07f-tsalc-spectrum.md` §十三 | 比較類型學 Type 0–V |
| ECC `PLAN-08-cycle3-gap-spectrum.md` | DCA 向量體系 |
| ECC `PLAN-16-geographical-anthropology.md` | Geo-Tech-Fin 第三軸 |
| ECC `index/knowledge-raw/KNOWLEDGE-PLAN.md` §七 | Forward Language Genesis（偏頗對策 4） |
| ECC `index/knowledge-raw/KNOWLEDGE-SPIRAL.md` | M-gap→E-gap→G-fill 生長閉環（偏頗對策 2） |
| ECC `AGENTS.md` §一 | β₁ 架構與世界觀（LPAC/AE 等民主機制出處） |
| ECC `scripts/geo-query.py` + `GEO-FIN-SKILL.md` | pay-as-you-go 先例（LLM 產結構、機械做運算、confidence/Lazy Verify/DCA 背書/provenance 四件套） |
| Hamilton (1989) 區制轉換模型；多狀態生存模型（msm 傳統） | §4.6 Markov 層的方法論家：隱狀態 Markov + 狀態條件觀測；轉移強度估計、持續時間依賴、小計數收縮 |
| 馬克思《路易·波拿巴的霧月十八日》（1852）開篇 | §五 認識論基礎：繼承的條件——人們在既定的、承繼的條件下創造歷史；「詞句超出內容 vs 內容超出詞句」判準；外國語比喻（忘掉母語 = 盲測校準的馬克思式根據）；兩軸框架（詞彙來源 × 用法） |
| arXiv:2506.03503 | 量子/社會映射的哲學同盟（非技術依據） |

---

## 十、定位修正：從比較引擎到坍縮-生成機器

> **一句話**：本機不是「雙世界綫比較引擎」，是**從叠加坍縮、繪製約束剛性分佈的通用頻譜電腦**——機器輸出「約束有多緊」的連續分佈，**必然由人詮釋**。內核 = 頻譜（場）+ 歷史之門（羅德島）。

### 10.1 內核：兩個不可通約的域

- **頻譜（場）** = **通用可觀測時間序列測量器**，世界無關——它測的是具體可觀測量：天氣雨量→年週期（FALSIFY-001）、世行 40 年 GDP 成長率→FFT 循環週期、ECC 張力/氛圍。「頻譜測什麼」的答案在範例裡，不抽象討論。
- **CLAD** = **閱讀文法（疊加態）**——**測量層（頻譜）與閱讀層（CLAD/Markov）不可混同**。頻譜測世界（通用可觀測時間序列測量器）、CLAD 讀世界（閱讀文法）、Markov 演閱讀動力學。

### 10.2 三層分工（LLM 用量最小化）

1. **LLM 語義選擇樹**：生成 alternative / 語義讀取——只有這裡需要 LLM
2. **頻譜機械過濾**：numpy-only——偏差、週期、聚類、駐波
3. **人機交錯收束**：§11.7 收束迴路——人選分支，機器重新測量

Mode B（枚舉）不足：全枚舉退化 + 只枚舉不動作。

### 10.3 決策樹探針 = 約束剛性測量器（§11 的操作化）

機器輸出**連續剛性分佈**（「選擇空間有多寬」），**必然由人詮釋**；drift/stochastic 類比。裁判退場（N5 實證：無 adversary 仍 0 substitution）。

### 10.4 狀態

- 本定位已取代舊定位（「雙世界線比較引擎」）。
- 舊定位的產物（pct_divergence、542 divergent scenes、L1 場論）仍是**應用層校準數據**——但不構成 OS 本體。
- 第二世界線（γ/δ/ε…）端到端註冊 = **世界無關承諾的正式驗收**（T16b，待辦見 TASKLOG）。

---

## 十一、決策樹探針規格（操作化）

### 11.1 定位與一句話

**探針 = 約束剛性測量器**：給定一個處境（situation），機器展開決策樹並測量每層「選擇空間有多緊」——輸出連續的約束剛性分佈，**必然由人詮釋**。不是窮盡枚舉器（那是 §五 版本 B 的職責）。

### 11.2 輸入與參數

- `situation`（處境語義）、`n_branch`（每層分支數）、`max_depth`、`n_sample`（每分支樣本）、`gate`（LLM 閘）、`code`（語碼）/`system_message`（prompt 注入）
  - 🔴 `n_branch` 為**提示性參考，非硬上限**（2026-08-03 修正）：LLM 依張力幾何自由增減，不作恆定目標值；路徑爆炸由 `MAX_PATHS_PER_TREE=4096` 獨立兜底（見 TASKLOG）。⚠️ **n_branch 提示性化不解除 1:1 鎖死**——樹寬 1.3636 的真正根因是反射設計的 1:1 續鏈（見 §11.11 #2 註記）。
- 成本錨定：**N_sample × D**（層數×每層樣本數）

### 11.3 LLM 層（語義選擇模擬器）

- 樹生成：`[task:tree_generate]` + `TREE_GENERATE_SYSTEM_PROMPT`
- 每層 LLM 依張力幾何生成多條分支（T6：每條分支對抗一個具體束縛；鏡角分佈——農民/工人/官僚/軍人/知識分子各自看到的出路；肯定句重寫）
- 輸出契約：`assert_tree_gate_contract`（文法違反/回聲/空殼/兩軸非法/role 缺欄/strength 越界/period_months 非法全拒收）

### 11.4 頻譜層（機械測量與檢查，零 API）

- **語義中介取代黑名單**：echo 降為記錄性（`echo_notes`，非致命）、鏡射詞接受＋contamination 降權、mixed 寬鬆、rejected→convergence_view；situation echo 仍 fatal
- **層間 echo 語義化（L3）**：echo 判定 =「label＋承義」（`_same_denotation`：binding/perspective/grounding/conditions 全同/全缺才拒）——假 echo 誤殺歸零、真原地踏步仍攔
- **語碼合規**：`assert_code_compliance`/`detect_script`/語碼感知長度契約（CJK 20/80/40 字符、拉丁/西里爾 60/240/120＋詞數 8/40/24）
- CLAD 文法轉移降為非致命（CLAD 是閱讀文法不是世界序列——違反記入 notes 非 raise）

### 11.5 約束剛性測量（F2 回退）

不輸出必然/偶然二分——輸出連續剛性分佈（drift/stochastic 類比），由人詮釋。remasking 校準：樣本不足 → 寬帶/UNKNOWN，不偽裝高信心（Wilson 信賴帶）。

### 11.6 輸出與門節點

- 輸出：樹、剛性分佈、世界原型聚類（語義距離聚類＋Silhouette）、門節點
- **門節點三條件**：① 剛性低谷（選擇空間最大）② 歷史時刻對照（如 1914-07-22 場強飽和）③ 代理閾值（0.99 vs 0.9 需 pilot 對照）

### 11.7 人機交錯收束

- `convergence_view`：視覺化收束介面（UX 待做）
- `probe_select`：人選分支 → 坍縮；未選標 `unselected` 寫回不刪除（維持疊加）
- `re_calibrate` 率：人與機器剛性判斷相左比例，落 state_log 供 verify 校準
- 🔴 **人機收束（奇門的「門」）是唯一未操作化的原始核心**——「沒有民主就沒有用」

### 11.8 成本

- 錨定 **N_sample × D**；pilot 實測 $0.0015（3 calls）

### 11.9 接線

- 新模組 `synth/probe.py`（不進 kernel/）
- 模組依賴：`quantum/standing_wave.py`（概念重用，`standing_wave_per_layer`）、`kernel/wave.py`（週期/偏差合法性）、`kernel/cluster.py`（不可直接吃——需語義向量化層/語義距離聚類）

### 11.10 驗收（v1.3：剛性地圖品質）

> 探針是**廣度探測器（約束剛性測量器），不是窮盡枚舉器**——驗收測「約束剛性分佈的品質」，不測「復現枚舉清單」。

1. **記錄性對照（非阻斷）**：探針分支 ∩ v2.9.6 純 α 枚舉 20 意外（1914-07）的**覆蓋**記入報告作對照參考；未覆蓋 ≠ 失敗，不阻斷。
2. **約束剛性地圖品質**（探針真正該測的）：
   - ① **剛性低谷 ↔ 已知歷史時刻**：機器測出的「選擇空間最大處」是否落在已知歷史時刻（如 1914-07-22 前後場強飽和/門節點三條件）——命中 ≥1，人覆核。
   - ② **連續性與信賴帶誠實**：剛性分佈連續；信賴帶誠實——樣本不足 → 寬帶/UNKNOWN（remasking 生效），不偽裝高信心。
   - ③ **世界原型品質**：低剛性區聚類的世界原型**處境內生**——grounding 三級（文獻偶發/合理推測/弱支持）可追溯、非回聲。
   - ④ **人機收束滿意度**：`re_calibrate` 率——人與機器剛性判斷相左的比例（低且被記錄，供 `kernel/verify` 校準；不作硬閾值）。
   - ⑤ **成本/抗污染維持**：#3/#4 不因重設計放鬆。
3. **成本 ≤ $0.01**（N_sample×D 錨定）。
4. **抗污染**：substitution 節點 0 為目標；以機械降權命中數 + 人機覆核無誤判計。回聲 0。
5. **世界原型 = 5–8 個**（需語義向量化層，或語義距離聚類先行）。
6. **門節點**：1914-07-22 前後三條件交集 ≥1。
7. **測試**：`assert_tree_gate_contract` 負面案例全拒收。

### 11.11 已知弱點（下輪 debug 審查對象）

1. **reflex 層間迭代不保證收斂**——機械驗證「每層選擇仍被約束場承載」，不承載即拒；D>4 風險高。
2. **樹→路徑組合爆炸**——$\prod N_{branch}$ 截斷策略需 pilot 校準。
   > 📌 **實證註記（2026-08-04 二修）**：1.3636 鎖死的真正根因 = **反射設計的 1:1 續鏈**（`reflect_on=passed` 全量回饋 × 分支恆 5 × 既有「相位不同」句 → LLM 慣性做一一對應），不是 `n_branch` 誤當目標（次要）、不是主線 fallback（只產星形 5.0，舊判定為誤診）。「必須標明 parent」強制句（08-03 加）是加劇非元兇、已撤回（08-04）改為「場的獨立坍縮」精神。修正 = 撤回強制句 + 允許同父多子/不續舊路 + 驗證 D/C2 內化框架。詳見 TASKLOG「🐛 樹寬指標失效根因」。
3. **α₂ 語義對齊層未規格化**（驗收 #2 前置依賴）。
4. **門節點③代理閾值**（0.99 vs 0.9）需 pilot 對照。
5. **w 的絕對標度缺驗收錨點**——只在生成期 prompt 內作用，無機械閾值。
6. **收束介面 UX/多人投票機制**待 pilot。
7. **CFG 公式落地**需新增條件 prevalence 欄位。
8. **世界原型聚類**需語義向量化層。

### 11.12 Mode A → Mode B 依賴

1. **先展開 Mode A**：Markov 推演 + LLM 標註結構語義 → 累積「過去的頻譜 = 發展歷史」。
2. **頻譜失效處**（殘差 sharp：飽和/去耦/gap）：LLM 做**後驗檢查** → 以 Mode A 成果為**初始化** → 才展開 Mode B（跳躍/意外）。
3. ⏳ **待實作**：Mode A 歷史日誌 + Mode B 後驗初始化接線。
