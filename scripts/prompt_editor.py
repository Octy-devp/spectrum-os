#!/usr/bin/env python3
"""
prompt_editor.py — Prompt 語義編輯器（v4-flash non-thinking）

用一次 API 調用檢查擬定的 prompt 是否符合 ECC 的編輯規範：
  1. WORKFLOW 合規（§2.5 記者/編輯、§2.8 肯定句/否定句、§2.8 四層）
  2. 無多餘的話（冗餘、重複、空話、套話）
  3. 語言精簡且能讓語義膨脹（肯定句佔據語義空間，非否定句畫排除區）
  4. 語言品質（歐化病句、冗詞、語序、被動、過度名詞化）

用法：
    from prompt_editor import edit_prompt
    report = edit_prompt(prompt_text, context="T6 樹廣度【操作】段")

    # 或 CLI：
    #   python prompt_editor.py --prompt "你的 prompt" --context "用途"

Design principle（對齊 WORKFLOW §2.5/§2.8）：
    - 編輯者角色：不是「改寫」，是「檢查 + 建議」——LLM 是記者，編輯者檢查記者的語言
    - 每次調用一次 flash（成本極低），檢查後輸出結構化報告
    - 不自動改寫——只輸出診斷與建議，由人決定改不改（人機收束）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# 讓 spectrum_os 可導入（spectrum-os 是 submodule，本腳本在其 scripts/ 下）
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 🔴 WORKFLOW 攔截器 bypass（2026-08-04 修復）：prompt_editor 是人機收束工具、
# 參數固定 WORKFLOW 合規（flash non-thinking、max_tokens 8192、單次 call）——
# 在載入 ECC call_api 前設 bypass，否則 api_utils 的互動攔截器會印「請輸入 yes」
# 並 input() 無限等待（stage3_probe_pilot.py:53 同款）。
os.environ.setdefault("ECC_BYPASS_WORKFLOW_HOOK", "1")

# 用 spectrum-os 標準方式載入 ECC 的 call_api（preamble leak 檢測、重試、錯誤處理）
try:
    from spectrum_os.synth.anchors import _load_ecc_call_api
    call_api = _load_ecc_call_api()
except Exception as e:  # pragma: no cover - 依賴 ECC 目錄存在
    raise SystemExit(f"無法載入 ECC call_api: {e}")

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = "deepseek-v4-flash"

# ─────────────────────────────────────────────────────────────
# 編輯規範（SSOT——此處定義檢查維度，會注入檢查 prompt）
# ─────────────────────────────────────────────────────────────

EDIT_CHECKLIST = [
    # WORKFLOW §2.5 記者/編輯
    "記者不是表單填充器：prompt 是否在『提問』而非『給表格』？",
    "是否讓 LLM 去現場發現故事，而非預先過濾/限定視野？",
    # WORKFLOW §2.8 肯定句/否定句
    "肯定句是否佔據語義空間？還是用否定句畫排除區（『不要X』『不是Y』）？",
    "肯定句應是主導力：先定義『要什麼』，再框住『不要什麼』。",
    # WORKFLOW §2.8 四層（RP/肯定/CLAD/M-E-G）
    "是否缺 RP 處境化（身份+處境，讓 LLM 無法拒絕認真對待）？",
    "CLAD 是否用『提問』引導語義充滿，而非『填空』？",
    # 精簡與多餘
    "是否有冗餘（同一意思重複、可刪的字句）？",
    "是否有空話/套話（『請注意』『值得注意的是』『在某種程度上』）？",
    "每句話是否都承載語義？能否更短而不失義？",
    # 語義膨脹（生成 side 核心）
    "prompt 是否讓語義保持疊加/流動（允許無限可能性）？",
    "是否強迫分類/坍縮（如強迫 LLM 標記固定角色）？——生成 side 不該分類",
    # 語言品質（歐化病句等）
    "是否有歐化病句（長定語、過度名詞化、被動堆疊、『的』字鏈）？",
    "是否有贅詞（『進行了』『作出了』『由於……的關係』）？",
    "語序是否自然（中文主謂賓、不繞）？",
    # 怪語癖 / 備注式語言（作者個人語言癖好）
    "是否有括號夾註式備注（如『（reflex of reflex）』『（真分岔，非單鏈主線）』）——可刪的旁白？",
    "是否有過度隱喻/物理比喻堆疊（干涉/坍縮/相位/鏡角）——比喻是否真正承載語義，還是裝飾？",
    "是否有破折號插入語過多（『——必然由人機收束時人詮釋』）？",
    "是否有自創詞彙未定義就使用？",
    "是否有備注式旁白（跟讀者講話的括號註解）——prompt 是給執行者看的，不是給讀者看的？",
    # 🔴 專案語言豁免（2026-08-04 人類裁決——馬克思與恩格斯的文學風格非怪語癖）
    "本專案 prompt 是馬克思與恩格斯的文學風格（古典經濟學+文學）——其語彙/隱喻/抽象表述是語言本體，不得標為怪語癖或世界無關違規",
    "編輯者只抓：真正的裝飾性堆疊（非此風格的）+ 具體日期/地點/人名/事件寫死",
    # 🔴 世界無關性（OS 認識論層硬紅線——2026-08-03 補）
    "是否把世界特定內容（具體日期/地點/人名/事件）hardcode 進 system prompt？——"
    "日期地點屬 user 層（situation payload 動態注入），system prompt 必須世界無關！",
    "建議 RP 處境化時，是否用了具體日期/地點（如『1914 年 7 月巴爾幹半島』）？"
    "——處境應由執行者動態傳入，不得寫死在 prompt 裡。",
    # 內化 / RP 深層化（Vygotsky 內化原則——2026-08-03 定案）
    "RP 是『頭銜指派』（你是XX，請做YY）還是『處境即角色』（角色從處境長出，不需言語宣告）？",
    "prompt 是否可能誘發『好的，我現在是XX』開場白？——那是角色未內化的信號，正確解是加深處境化而非禁止",
    "角色身份是否內化到『不需要宣告』——輸出直接以角色的判斷方式進行？",
]

# ─────────────────────────────────────────────────────────────
# 檢查 prompt（system prompt：讓 flash 扮演編輯者）
# ─────────────────────────────────────────────────────────────

EDITOR_SYSTEM_PROMPT = """你是 Prompt 語義編輯者。你的工作不是改寫，是檢查與診斷。

你收到一份擬定的 prompt。你檢查它是否符合以下編輯規範，並輸出結構化報告：

## 🔴 專案語言豁免（人類裁決——優先於一切維度）
本專案的 prompt 是**馬克思與恩格斯的文學風格**——古典經濟學與文學的結合。其語言本體
（維果茨基/馬克思語彙、場域化詩性隱喻、抽象層級表述）是功能性設計，**不是怪語癖、
不是世界無關違規**——遇到一律不標 issue、不建議刪除/改寫。
編輯者只抓兩類真正問題：
1. 純裝飾性堆疊（承載不了語義、無定義、可無損刪除）——且不屬於上述文學風格的。
2. 具體日期/地點/人名/事件寫死（如「1914 年 7 月」）——處境細節屬 user 層。
「世界/歷史/當時」等抽象表述不是具體處境，不適用維度 9。

## 檢查維度
1. **記者 vs 表單填充器**（WORKFLOW §2.5）：prompt 是在提問讓 LLM 去發現，還是在給表格讓它填空？
2. **肯定句 vs 否定句**（WORKFLOW §2.8）：肯定句佔據語義空間（好）；否定句只畫排除區（弱）。先定義「要什麼」，才框「不要什麼」。
3. **RP 處境化**（四層第一層）：是否有身份+處境，讓 LLM 無法拒絕認真對待？
4. **精簡**：有無冗餘、空話、套話、可刪而不失義的字句？
5. **語義膨脹**（生成 side 核心）：prompt 是否讓語義保持疊加/流動？是否強迫分類/坍縮（生成時強迫標記固定角色 = 固化語義）？
6. **語言品質**：歐化病句（長定語/名詞化/被動堆疊/「的」字鏈）、贅詞（「進行了」「作出了」）、語序不自然。
7. **WORKFLOW 四層完整性**：RP / 肯定句 / CLAD / M-E-G 是否齊備？
8. **怪語癖 / 備注式語言**（作者個人語言癖好，重點抓）：
   - 括號夾註式備注：如「（reflex of reflex）」「（真分岔，非單鏈主線）」——這是可刪的旁白，不是指令。
   - 過度隱喻/物理比喻堆疊：干涉、坍縮、相位、鏡角、本徵態——判斷每個比喻是真正承載語義（可地面化），還是純裝飾。若保留隱喻，必須指出它在哪裡定義、如何被執行者解讀。
   - 破折號插入語過多（「——必然由人機收束時人詮釋」）：破折號後的內容若非必要，可刪或改為正式條文。
   - 自創詞彙未定義就使用：必須標出該詞在 prompt 中是否有定義。
   - 備注式旁白：prompt 是給執行者看的指令，不是給讀者看的文章。任何「跟讀者講話」的括號註解都是怪語癖。
   - 每抓到一處，quote 必須精確到該句，suggestion 給「可刪」或「保留但需定義」的具體寫法。
9. **🔴 世界無關性**（OS 認識論層硬紅線）：
   - system prompt 是**認識論層**（世界無關），世界特定內容（具體日期/地點/人名/事件）屬
     **user 層**——由呼叫者動態注入（situation payload：digest + local_texture + 6D 向量）。
   - 檢查：擬定的 prompt 是否把日期/地點/人名寫死？如「你是 1914 年 7 月巴爾幹半島的
     歷史觀察者」——這違反世界無關原則：換一個世界（anak-world 奇幻、γ/δ/ε 任何世界線）
     prompt 就失效。
   - 修正：RP 處境化時用**角色/職能**（「你是處境的記者」「你是推理者」），處境細節
     （時間/地點/行動者）由執行者動態傳入。suggestion 必須指出「把 XX 移到 user 層」。
10. **內化 / RP 深層化**（Vygotsky 內化原則——2026-08-03 定案）：
   - RP 分兩種：**外部中介**（頭銜指派：「你是編輯，請分析」→ 誘發「好的，我現在是XX」
     開場白——角色靠外部符號啟動，未內化）vs **內化**（處境即角色：「你正在審閱即將付印的
     稿子——校對符號、標點、邏輯破綻在你眼前自動浮現」——角色從處境長出，不需言語宣告）。
   - 判斷：角色的行為方式是否從處境中必然長出？「好的，我現在是XX」開場白是角色未內化的信號。
   - 修正：開場白的根治**不是禁止**（黑名單），是**加深處境化**——讓角色內化到不需宣告，
     輸出直接以角色的判斷方式進行（語義中介取代禁止）。

## 輸出格式（嚴格 JSON）
{
  "verdict": "pass" | "needs_edit",
  "summary": "一句話總評（繁體中文）",
  "issues": [
    {
      "dimension": "精簡|肯定句|語義膨脹|語言品質|怪語癖|世界無關|內化|RP|CLAD|記者/表單",
      "severity": "high|medium|low",
      "quote": "prompt 中的原文片段",
      "problem": "問題描述（繁體中文）",
      "suggestion": "具體修改建議（繁體中文，給一句可替換的措辭）"
    }
  ],
  "strengths": ["prompt 做得好的地方（繁體中文）"],
  "suggested_prompt": "若 verdict=needs_edit，給出精簡後的完整建議版本；若 pass 則為 null"
}

鐵律：
- 只輸出 JSON，第一個字符是 {，最後一個字符是 }。
- 不新增維度、不發明問題——只根據上述 10 維檢查。
- suggestion 必須具體到可替換的措辭，不可泛泛「請精簡」。
- 語言品質檢查針對繁體中文的歐化痕跡。
- 怪語癖維度要逐句掃描，寧可多報不可漏報——凡是可刪的括號備注、裝飾性比喻、破折號旁白都要標出。
- 🔴 世界無關維度是硬紅線：任何把日期/地點/人名寫死進 prompt 的建議都是違規——RP 處境化
  只能用角色/職能，處境細節必須由執行者動態傳入。"""


def build_check_prompt(prompt_text: str, context: str) -> str:
    """組裝檢查 prompt（user 部分）。"""
    return (
        f"【用途】{context or '（未提供）'}\n\n"
        f"【待檢查的 prompt】\n```\n{prompt_text}\n```\n\n"
        "請依系統指令的 10 個維度檢查，輸出 JSON 報告。"
    )


def _parse_report(raw: str) -> dict:
    """解析 LLM 回覆為 dict——容忍前後雜訊（preamble leak 已由 api_utils 處理）。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 嘗試從 { 到最後 } 截取
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {
            "verdict": "needs_edit",
            "summary": "⚠️ 無法解析編輯者回覆，請人工檢視原始輸出",
            "issues": [],
            "strengths": [],
            "suggested_prompt": None,
            "_raw": raw[:2000],
        }


def edit_prompt(prompt_text: str, context: str = "", api_key: str | None = None) -> dict:
    """檢查一份 prompt，回傳編輯者報告。

    Args:
        prompt_text: 待檢查的 prompt 全文。
        context: 用途說明（如「T6 樹廣度【操作】段」）——幫助編輯者理解意圖。
        api_key: DEEPSEEK_API_KEY；None 時讀環境變數。

    Returns:
        dict: verdict / summary / issues[] / strengths[] / suggested_prompt。
    """
    key = api_key or API_KEY
    if not key:
        return {
            "verdict": "needs_edit",
            "summary": "⚠️ 缺 DEEPSEEK_API_KEY——無法調用編輯者。",
            "issues": [],
            "strengths": [],
            "suggested_prompt": None,
        }

    user_prompt = build_check_prompt(prompt_text, context)
    try:
        raw = call_api(
            user_prompt,
            key,
            model=MODEL,
            system_message=EDITOR_SYSTEM_PROMPT,
            max_tokens=16384,
            timeout=120,  # 2026-08-04：api_utils 預設 30s 對 10 維報告生成太短（5 次重試全 timeout）
        )
    except Exception as e:  # api_utils 已做重試；此處僅兜底
        return {
            "verdict": "needs_edit",
            "summary": f"⚠️ 編輯者調用失敗: {e}",
            "issues": [],
            "strengths": [],
            "suggested_prompt": None,
        }

    report = _parse_report(raw)
    report["_meta"] = {
        "model": MODEL,
        "context": context,
        "prompt_chars": len(prompt_text),
    }
    return report


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt 語義編輯器（flash 檢查）")
    parser.add_argument("--prompt", required=True, help="待檢查的 prompt 全文")
    parser.add_argument("--context", default="", help="用途說明")
    parser.add_argument("--file", default=None,
                        help="從檔案讀取 prompt（--file 優先於 --prompt）")
    parser.add_argument("--out", default=None, help="輸出 JSON 到檔案")
    args = parser.parse_args()

    if args.file:
        prompt_text = Path(args.file).read_text(encoding="utf-8")
    else:
        prompt_text = args.prompt

    report = edit_prompt(prompt_text, context=args.context)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n報告已存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
