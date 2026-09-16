# Spectrum OS — 修改律（MODIFICATION LAW）

> **版本**：v1.0（2026-09-15）
> **職能聲明**：**本檔是 spectrum-os 所有寫入行為的成文紀律。正本在此（spectrum-os repo）。** ECC 側散落的相關紀律（`AGENTS.md` §2.2、`.agents/skills/ecc-guardrails/SKILL.md`、`scripts/check-spectrum-os-sync.sh` 檔頭、`docs/…/SESSION-HANDOFF-*.md`）在收編後**只作指針**——發現衝突時，以本檔為準；本檔未寫的，ECC 側有權保留，但不得與本檔牴觸。
>
> **與既有文檔的分工**：
>
> | 文件 | 管什麼 | 不管什麼 |
> |:---|:---|:---|
> | `docs/PLAN.md` | OS 本體**規格**——kernel/quantum/Markov/探針/抗污染憲章/向量契約 | 不寫流程、不寫事故史 |
> | `docs/TASKLOG.md` | OS 開發**日誌**——每次 commit、測試數、驗收、已知 bug | 不寫規範條文 |
> | **本檔（`docs/MODIFICATION-LAW.md`）** | **寫入行為的成文紀律**——寫在哪、怎麼提交、如何驗證、何時凍結 | 不寫規格（去 PLAN.md）、不寫進度（去 TASKLOG.md） |
>
> **性質**：本檔不是設計文件的附錄，是**墓碑的集合**。每一條紀律底下都埋著一次真實事故——日期、commit、症狀、根因。條文之所以是條文，是因為有人已經付過代價。閱讀者請把每條當成「上次這樣做壞了什麼」，而不是「上級要求什麼」。
>
> **本檔的哲學前提（見 §3.2「移動的裂縫」的自我指涉）**：本檔寫下那一刻也開始死亡。它記錄的是**當前的**斷裂點與已知防洪堤；新的裂縫會在本檔之外出現。當你發現本檔某條已與現實不符，那不是本檔的權威失效，是本檔正在變舊——**應提出修訂，而不是靜默繞過**。

---

## 第一條 單一寫入點

**真本體＝`/home/octy/projects/spectrum-os`。這是唯一可寫入 spectrum-os 代碼的 checkout。**

| 位置 | 性質 | 可寫？ |
|:---|:---|:---:|
| `/home/octy/projects/spectrum-os` | 真本體（standalone repo，`master` 有 branch 支撐） | ✅ 唯一寫入點 |
| `ECC/spectrum-os/` | git submodule，**唯讀 checkout**，通常處於 **detached HEAD** | 🚫 禁止開發 |
| 任何 `git worktree` 臨時檢出 | 驗證用（見第四條） | 🚫 驗證用，不提交 |

**禁止在 submodule 直接開發，即使「只是改一行」。** 在 submodule 產生的 commit 會落在 detached HEAD 上，兩邊 repo 互不相識——這正是 09-05 事故的形狀。

**違反偵測**：ECC 側 `scripts/check-spectrum-os-sync.sh` 五項檢查中的第 ② 項（submodule 是否有真本體沒有的 commit）與第 ⑤ 項（submodule 工作樹是否乾淨）即為此條的機械強制。此檢查器在 ECC 側，因為**消費者才持有 submodule 佈局**——生產者（本 repo）看不到這種漂移。

### 🪦 事故墓碑：2026-09-05 雙 checkout 分叉（`f8f524a` 之後）

- **症狀**：發現兩個 spectrum-os checkout（真本體 + ECC submodule）各自從 `f8f524a` 分叉，**互不認識對方的 commit**，且**兩邊的獨有 commit 都還沒推 origin**。更危險的是 submodule 的 `65b6871` 只靠 detached HEAD 支撐，沒有任何 branch 指向它——任何人一句 `git checkout master` 就會讓它變成孤兒。
- **根因**：**不是工具問題，是寫入點不唯一**。有人在 submodule 直接開發。
- **催生**：`ECC/scripts/check-spectrum-os-sync.sh`（五項檢查）＋「同步三規則」（見第二條）＋真本體 `archive/submodule-5d-20260912` 分支（`65b6871` 的歷史保全——內容已被 `master` 取代，**不併入** master）。
- **教訓（原話）**：「本檢查讓『多寫入點』與『stale pointer』變成會響的訊號，而不是靜默漂移。」

---

## 第二條 提交紀律（順序不可逆）

**任何 spectrum-os 變更的正規流轉，必須依序完成四步，不得跳步、不得換序：**

```
真本體 commit  →  push origin  →  ECC 側 git submodule update --remote  →  ECC 側 commit pointer
   (1)              (2)                        (3)                               (4)
```

| 步 | 動作 | 為什麼必須在這一格 |
|:---:|:---|:---|
| 1 | 在 `/home/octy/projects/spectrum-os` 提交（見第四條驗證後） | 唯一的寫入點 |
| 2 | `git push origin master` | 讓 commit 進入共享歷史——**這是「脫離孤兒」的唯一動作** |
| 3 | 在 ECC：`git submodule update --remote spectrum-os` | 讓唯讀 checkout 追上真本體 |
| 4 | 在 ECC 提交 pointer bump（訊息格式見第三條） | 消費者錨定版本 |

### 🔴 未 push 的本地 commit ＝ 孤兒風險

**本地 commit 不是保存，只是「還沒死透」。** 只要它還沒 push：

- 它只存在於這台機器的工作目錄；
- 它可能只靠 **detached HEAD** 支撐（submodule 尤其如此）——沒有 branch 指向它，`git checkout`、`git gc`、`git submodule update` 都可能讓它消失；
- 另一方（另一 checkout／另一 agent／人類）**完全看不見**它，因此會基於「不存在的歷史」做出決定；
- 交叉比對時，兩邊會各自聲稱自己的版本是「最新」。

**規則**：做完第 1 步後，**立即做第 2 步**，不要「攢一批再推」。需要多 commit 時，逐個 push 的成本遠低於事後考古。

### 🪦 事故墓碑：09-05 分叉的「兩邊獨有 commit 都未推」

見第一條。**雙分叉之所以能發生，前提正是「兩邊都沒 push」**——若有任一邊先 push，另一邊立刻會看到分歧。未 push 是分叉的**培養基**。

---

## 第三條 指針格式

ECC 側的 submodule pointer bump **必須是獨立、可辨識、名實相符的 commit**。

### 3.1 commit 訊息格式

```
[SPECTRUM-OS] pointer: <old7>→<new7>
```

- `old7`／`new7`：pointer bump 前後的 submodule commit hash **前七碼**。
- 訊息中**不得**夾帶其他改動的描述。pointer commit 只做 pointer。
- 若同一次確實還改了其他東西，**分成兩個 commit**——指針與內容不可混裝。

### 3.2 `.gitmodules` 設定

```ini
[submodule "spectrum-os"]
    ...
    ignore = untracked
```

原因：唯讀 checkout 產生的 `__pycache__` 等**未追蹤**檔案，會讓 ECC 出現假 `M spectrum-os` 訊號。`ignore = untracked` 消除假訊號，**真髒污（已追蹤檔案被改）仍會回報**——這正是我們要的：雜訊消失，警報保留。

### 3.3 為何要求「名實相符」

commit 訊息是**唯一的非同步通道**。人類與 agent 靠它判斷「這次動了什麼範圍」，然後決定要不要跟進、要不要重跑下游。訊息若低估範圍，**讀者按訊息規劃、實際 diff 卻大得多**——下一個問題就會在錯誤的前提上被回答。

### 🪦 事故墓碑：`2bdac138` 名不符實 pointer

- **症狀**：pointer commit 的**標題寫「pointer 收斂」**，實際 diff **包含 777 個檔案**。
- **傷害**：任何以訊息為準的審閱、二分定位、影響評估，全部失效。標題說「只動了指標」，實際是巨型改動被藏在最小標題下。
- **催生**：本條指針格式紀律。
- **教訓**：**標題就是承諾**。標題的範圍必須等於 diff 的範圍；做不到，就拆 commit。

---

## 第四條 提交完整性驗證

**規則：宣稱某個 commit 可用之前，必須在乾淨 worktree 上驗證它。**

```bash
git -C /home/octy/projects/spectrum-os worktree add /tmp/fresh <commit>
cd /tmp/fresh && pytest
git -C /home/octy/projects/spectrum-os worktree remove /tmp/fresh
```

### 🔴 「本機測試通過」≠「提交可通過」

工作樹同時含新舊檔案時，測試只證明**工作樹自洽**，不證明**提交自洽**。這是 `python -m pytest` 的結構性盲區——它在「有未提交改動的工作樹」上跑，看不到提交層的不一致。**只有乾淨 worktree 能重現「別人 clone 之後會看到什麼」。**

**觸發時機**：任何 commit 之後、宣告完成之前、push 之前。若第四條與第五條同時適用（改動觸及 `kernel/forces.py`），兩者都要跑。

### 🪦 事故墓碑：`7abf6c7` 提交不完整（探針缺口）

- **症狀**：`git worktree add /tmp/fresh 7abf6c7` 後跑測試 → `1 failed, 54 passed`，`TypeError: FastChannelConfig got an unexpected keyword argument 'co_share'`。
- **根因**：`7abf6c7` 提交了新的 `forces.py` 與三個測試檔，**但沒提交同步更新後的探針腳本**——工作樹上它們已改（可用），提交裡仍是舊版（用 `co_share`）：

  | 檔案 | 提交版 | 工作樹版 |
  |:---|:---:|:---:|
  | `scripts/fast_channel_dnspv_harness.py` | ❌ 舊（`co_share` ×6） | ✅ 新 |
  | `scripts/fast_channel_alpha_rc_probe.py` | ❌ 舊 | ✅ 新 |
  | `scripts/two_mode_digestion_probe.py` | ❌ 舊 | ✅ 新 |

- **為何本機看不到**：`test_forces_fast_channel.py::TestHarnessSmoke` 會 import harness。**本機工作樹已改，所以綠燈**；乾淨 checkout 是紅燈。
- **修法**：`19f9e92`（[Forces] 探針腳本同步 substrate 化）。
- **教訓（原話）**：「『本機測試通過』不等於『提交可通過』。」

---

## 第五條 `forces.py` 煙霧義務

**任何觸及 `kernel/forces.py` 的變更，必跑以下全套：**

1. **`pytest tests/test_forces_fast_channel.py::TestHarnessSmoke`**——最少門檻，驗證 harness 與 kernel 的介面契約仍對得上。
2. **同步探針腳本**——`scripts/fast_channel_dnspv_harness.py`、`scripts/fast_channel_alpha_rc_probe.py`、`scripts/two_mode_digestion_probe.py`。改了 kernel 的 substrate 槽（如 `co_share`/`debt_stress` → `phi_drive`/`phi_suppress`），**探針必須同一個 commit 改完**（見第四條墓碑）。
3. **ECC 側受影響產物生成器重跑並比對關鍵量**——`forces.py` 是 ECC substrate（`ECC/scripts/ecc_substrate.py`）的注入目標；kernel 介面一動，ECC 側產物（fast5 輸出、overlay、project-almanac-dynamics）可能靜默變值。**不比對就不知道**。

**為什麼 `forces.py` 需要專條**：它是**世界與引擎的接觸面**。其餘 kernel 模組的變更，作用域限於引擎內部；`forces.py` 一改，世界側（substrate 注入、探針、ECC 生成器）全部在射程內——這是唯一一條「改一處、四方受影響」的路徑。

> **交叉引用**：第六條（純度）管「什麼內容不可以進 `forces.py`」；第五條管「改動 `forces.py` 之後必須做什麼」。

---

## 第六條 純度守則（kernel 零世界詞）

**`kernel/` 與 `quantum/` 中不得出現世界特定的字串**。世界概念住在 substrate 與 extensions，**不住 kernel**。

### 6.1 機械強制

`tests/test_sources.py::TestCoreGeneralityMechanicalCheck::test_no_ecc_paths_in_core_modules`——掃描核心模組，禁止字串 `"/home/octy/projects/ECC"` 與 `"index/"` 出現。

守備範圍（`tests/test_sources.py`）：

```
spectrum_os/contracts.py, spectrum_os/sources.py,
spectrum_os/kernel/**, spectrum_os/quantum/**, spectrum_os/synth/**,
scripts/stage3_sarajevo_pilot.py, scripts/stage3_universality_check.py,
tests/test_alt_gate.py, tests/test_ensemble.py,
tests/test_standing_wave.py, tests/test_stage3_pilot.py
```

**注意**：世界的**名字**（如 `sigma_soviet`、`co_share`、`debt_stress`）不一定長得像路徑——見 6.3 墓碑。機械掃描擋得住字串，擋不住概念；概念靠 6.2。

### 6.2 「engine never fits parameters」（`kernel/forces.py:64`，條文引用）

> *Parameters (α, β, a, c, κ, μ, λ, ρ) are **SEMANTIC ANCHORS**, not physical constants: sign and relative magnitude carry the judgement; absolute values are neither available nor needed. **The engine never fits parameters.***

**推論（動態 lock 否決事件的裁定）**：既然參數是語義錨而非物理常數，那麼**用數據去反推／鎖定參數值，是把世界塞回引擎**。當有人提議「加一條動態 lock 把參數釘到實測值」時，該提議違反本條——**引擎不得 fit 參數**，參數的判準是符號與相對量級，不是擬合誤差。

### 🪦 事故墓碑 A：`ea369e7` 世界詞內嵌 kernel

- **症狀**：kernel 內直接出現 `co_share`／`debt_stress`／`sigma_soviet` 三個世界概念——違反 substrate 契約（世界只提供**形式**，引擎提供**結構**）。
- **催生**：kernel 零世界詞掃描＋substrate 化（`co_share` → `phi_drive`、`debt_stress` → `phi_suppress`，經 substrate callable 注入）。
- **教訓**：世界詞進 kernel 是**層級錯位**（見第七條），不是命名品味問題。

### 🪦 事故墓碑 B：ECC 絕對路徑內嵌 kernel

- **症狀**：兩支 kernel 模組帶 ECC 絕對路徑 → `test_sources` **持續失飛**。
- **催生**：`test_no_ecc_paths_in_core_modules`。
- **教訓**：核心模組不知道自己在哪台機器上——**一旦知道，它就不再是 OS**。

---

## 第七條 substrate 四層聲明制（本檔新立）

**每一項修改都必須聲明它屬於哪一層。四層：**

| 層 | 位置 | 內容 | 世界相關？ |
|:---|:---|:---|:---:|
| **kernel** | `spectrum_os/kernel/`、`spectrum_os/quantum/` | 世界無關的機制：頻譜→殘差→意外、ODE、Markov、量子層 | 🚫 **世界無關**（見 `PLAN.md:6`） |
| **substrate** | 世界側，如 ECC `scripts/ecc_substrate.py` | 世界的**形式**，以 callable 注入：`c_source_fn`／`k_inflow_fn`／`phi_drive_fn` 等 substrate 槽 | ✅ 世界特定 |
| **extensions** | `spectrum_os/extensions/` | 外掛動力學：`gnp_dynamics.py`、`ar_frigorifico_dynamics.py`、`market_suite_dynamics.py` | ✅ 世界特定（**可選、可插拔**） |
| **data** | 卷宗／觀測／校準輸入 | 具體數值、時間序列、快照 | ✅ 世界特定 |

### 7.1 聲明義務

**每次修改必須在 commit 訊息或 TASKLOG 條目中明確聲明層級。** 範例：

```
[Forces] kernel 層：FastChannelConfig 新增 phi_drive 槽（世界無關，substrate 注入）
[Forces] substrate 層：ecc_substrate.py 以 phi_drive_fn 取代 co_share 常數
```

**未聲明層級 = 未完成修改。**

### 7.2 🔴 錯層即事故

**同一件事放錯層，就是事故**——不是風格問題，是**架構破壞**：

| 錯層 | 後果 |
|:---|:---|
| 世界概念寫進 kernel | kernel 不再世界無關 → OS 退化為「ECC 的內部工具」→ 第六條紅燈 |
| 引擎機制寫進 substrate | 世界側重複實作引擎邏輯 → 每個世界各自實作一遍 → 無法比較 |
| 世界概念寫進 extensions 又反向依賴 kernel | 外掛不再是外掛 → 可插拔性消失 |

### 7.3 📌 註記：本條同時是 substrate 契約規格化的起點

**現行 substrate 契約僅一句聲明**（`docs/PLAN.md:6`）：

> 「**世界無關**：γ/δ/ε 任何世界線的 substrate 共用同一份規格。」

**這句話確立了原則，但沒有定義介面。** 目前 substrate 契約的**實際**內容散落在：`kernel/forces.py` 的 substrate 槽 docstring（`growth_c_fn`／`release_fn`／`phi_drive` 等）、`ECC/scripts/ecc_substrate.py` 的實作、以及 `tests/test_forces_fast_channel.py` 的介面測試。

**本條既是紀律（四層聲明＋錯層即事故），也是契約規格化的起點**——把「哪些槽是 substrate 的法定介面」「每個槽的語義與守恆要求」「新增槽的程序」從散落狀態收攏為明文。**此規格化尚未完成**；完成前，substrate 契約的事實來源是 `forces.py` 的槽定義與 ECC 側 `ecc_substrate.py` 的實作。

### 🪦 事故墓碑：`ea369e7`（錯層的第一次代價）

見第六條墓碑 A。**`ea369e7` 的教訓不只是「kernel 要乾淨」，而是「層級必須先聲明」**——因為當時沒有層級聲明制，世界概念是**無意識地**長進 kernel 的。事後清理（`36df362` 的 substrate 化）比事前聲明貴得多。

---

## 第八條 凍結與解凍程序（本檔首立）

### 8.1 凍結令：必須明列範圍

**凍結令不得寫成「引擎凍結」四個字。** 必須回答：**不改什麼、不改到什麼程度、為何凍結、解凍前置是什麼。**

**照 09-11 體例**（`SESSION-START-20260911.md:231`）：

> **引擎凍結範圍**：除已完成的 P2 池地板外，**不改 PARAMS、不加器械、不接時滯**——等數據，不磨刀。

拆解此體例的四個要件：

| 要件 | 該例的內容 |
|:---|:---|
| ① **例外白名單** | 「除已完成的 P2 池地板外」——凍結令不回溯已授權項 |
| ② **禁止清單（具體動詞）** | 「不改 PARAMS、不加器械、不接時滯」——**不是「不優化」，是可枚舉的動作** |
| ③ **凍結理由** | 「等數據」——凍結是**等待約束解除**，不是「不要動」 |
| ④ **隱含解凍前置** | 數據到位（P3 的 1920 實測 observation 卷） |

**違反偵測**：若凍結令無法回答「這算不算違反？」——例如有人提議「只是重構一下 `forces.py`，不算改參數吧」，而凍結令沒說——則該凍結令**規格不足**，應補列。

### 8.2 解凍程序（無先例，本次首立）

**09-11 凍結令下達至今，尚無解凍先例。以下為首立程序：**

1. **須用戶明示裁定**——解凍**不可由 agent 自行判斷**「數據好像夠了」。凍結的解除是**裁定**，不是推論。
2. **裁定必須限定範圍**：**定點解凍**（指定具體條目，如「解凍時滯耦合」）或**全面解凍**（明文寫「全面」）。**未限定的解凍令視為規格不足**，應回問——因為「解凍」的預設語義歧義：是指凍結令全部項目解除，還是僅指某項解除？
3. **解凍後，先跑兩項驗證**（理由見下）：
   - **第五條煙霧**：`pytest tests/test_forces_fast_channel.py::TestHarnessSmoke` ＋同步探針 ＋ ECC 側生成器比對；
   - **第四條 worktree 驗證**：`git worktree add /tmp/fresh <commit> && pytest`。

**為何解凍後必跑這兩項**：凍結期間引擎**未被觸碰**，但也**未被驗證**。基礎設施（探針、ECC 側生成器、Python 版本、依賴）在凍結期間持續變動——凍結只凍結了引擎，沒凍結環境。**解凍後的第一次改動，同時是環境漂移的第一次曝光**。這兩項驗證是曝光的手段。

### 8.3 凍結與解凍的關係

```
凍結令（範圍明列） ──→ 凍結期（引擎不動，環境在動） ──→ 用戶裁定解凍（定點/全面）
                                                              ↓
                                                第五條煙霧 + 第四條 worktree
```

**凍結不是「暫停」，是「把引擎釘在一個已驗證的座標上」**——因此解凍時必須重新證明那個座標還在。

---

## 附錄 A — 事故年表

| 日期 | commit | 事故 | 症狀 | 催生的條文 |
|:---|:---|:---|:---|:---|
| 2026-08-04 | `ea369e7` | **世界詞內嵌 kernel** | `co_share`／`debt_stress`／`sigma_soviet` 直接寫在 kernel，違反 substrate 契約 | 第六條（純度）· 第七條（四層聲明） |
| 2026-09-05 | 分叉起於 `f8f524a` 之後 | **雙 checkout 分叉** | 真本體與 submodule 互不相識、兩邊獨有 commit 未推、`65b6871` 僅靠 detached HEAD 支撐（孤兒） | 第一條（單一寫入點）· 第二條（提交紀律） |
| 2026-09-09 | — | **kernel 帶 ECC 絕對路徑** | 兩支 kernel 模組含 ECC 絕對路徑 → `test_sources` 持續失飛 | 第六條 6.1（`test_no_ecc_paths_in_core_modules`） |
| 2026-09-11 | — | **凍結令下達** | `SESSION-START-20260911.md:231`——不改 PARAMS、不加器械、不接時滯 | 第八條（凍結規格＋解凍程序首立） |
| 2026-09-12 | `7abf6c7` | **提交不完整** | 含新 `forces.py` 但缺同步探針腳本 → 乾淨 clone `TypeError`，**本機看不見** | 第四條（worktree 驗證）· 第五條（煙霧義務） |
| 2026-09-13 | `2bdac138` | **名不符實 pointer** | 標題「pointer 收斂」，實含 **777 檔** | 第三條（指針格式） |
| — | — | **動態 lock 否決** | 提議為參數加動態 lock（以數據鎖定參數值），經裁定否決 | 第六條 6.2（engine never fits parameters，`forces.py:64`） |

**未掛 commit 的兩條**（`2026-09-09` 路徑污染、動態 lock 否決）為事件性裁定；`09-11` 凍結令為程序性裁定。

---

## 附錄 B — 散落紀律的源頭清單與收編指向

本檔收編前，修改紀律散落在以下位置。**收編後，各源頭只作指針，正本為本檔。**

| # | 源頭（位置） | 原本承載的紀律 | 收編至 | 收編後定位 |
|:---:|:---|:---|:---|:---|
| 1 | `ECC/scripts/check-spectrum-os-sync.sh`（檔頭 :1–19） | 「寫入點不唯一」根因、孤兒風險、五項檢查 | 第一條 | **機械強制器**——仍是違反偵測的實作，紀律本體在本檔 |
| 2 | `ECC/index/locations/data/almanac/_sessions/SESSION-HANDOFF-20260912-kernel-fork.md` §五 | 同步三規則、單一寫入點流程圖 | 第二條 | **事故現場記錄**——歷史證據，不再作規範來源 |
| 3 | 同上 §八 | 提交不完整根因、乾淨 worktree 驗證 | 第四條 · 第五條 | 同上 |
| 4 | `ECC/AGENTS.md:141–144` | 「本體不在 ECC monorepo 內」「開發與 commit 請到真本體」 | 第一條 | **ECC 側摘要指針**——詳細紀律見本檔 |
| 5 | `ECC/.agents/skills/ecc-guardrails/SKILL.md` | ECC 側寫入前查證程序（SSOT 先讀後寫、dry-run） | 不直接收編 | **ECC 側技能，管 ECC 數據寫入**——與本檔並行，兩者不衝突（本檔管 spectrum-os 寫入） |
| 6 | `docs/PLAN.md:6` | 「世界無關」聲明 | 第七條 · 附錄 A | **substrate 契約的一句話起點**——第七條 §7.3 註記其規格化尚未完成 |
| 7 | `README.md:5` | "The ECC project is its **first substrate**" | 第七條 | **定位聲明**——世界＝substrate 的公開表述 |
| 8 | `docs/TASKLOG.md` | 每次 commit／測試數／push 狀態 | 不直接收編 | **日誌**——記錄「發生過什麼」，本檔記錄「應該怎麼做」 |
| 9 | `spectrum_os/kernel/forces.py:64` | "The engine never fits parameters" | 第六條 6.2 | **條文原文引用**——保留在代碼 docstring 中，本檔賦予其紀律效力 |
| 10 | `tests/test_sources.py:203–235` | `test_no_ecc_paths_in_core_modules` 守備範圍 | 第六條 6.1 | **機械強制器**——範圍變更時須同步更新本檔 |
| 11 | `SESSION-START-20260911.md:231`（ECC `almanac/_sessions/`） | 凍結令體例 | 第八條 8.1 | **體例範本**——凍結令撰寫照此四要件 |

### 未收編項（明示保留）

- **ECC 側應用層紀律**：`ecc-guardrails`（SSOT 先讀後寫、ground-truth 注入、dry-run）管 ECC 的**數據**寫入，與本檔管 spectrum-os 的**代碼**寫入是**不同客體**。兩者並行，無隸屬關係。
- **`docs/TASKLOG.md` 的現況快照**：本檔不重複記錄測試數、HEAD hash、push 狀態——那是日誌的職能，且會迅速過期。

---

**本檔結束。修訂請提請用戶裁定；發現條文與現實不符時，記錄裂縫，不要靜默繞過。**
