#!/usr/bin/env python3
"""Stage 3 首跑：薩拉熱窩分支注入實驗（PLAN-23 §五）。

分支點 1914-06-28：A=刺殺成功（α₁ 側，可校準）、B=刺殺被阻止（β₁ 側）、
C=阻止但 24 月內大戰仍爆發（中間型）。每分支一次 v4-flash 呼叫（生成+收束
同時），駐波分解（共享=必然/分歧=偶然），A 分支對真實歷史錨點校準。

機械/LLM 分工：生成與收束由 LLM（GATE），駐波比對與錨點校準由機械。
理解權歸人：本腳本只產出數據與標記，不做判讀。
"""
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ECC = Path("/home/octy/projects/ECC")

import importlib.util
spec = importlib.util.spec_from_file_location("api_utils", ECC / "scripts" / "api_utils.py")
api_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api_utils)
call_api = api_utils.call_api

# WORKFLOW 自動化 hook（campaign-runner 先例）
Path("/tmp/.ecc_workflow_hook_timestamp").write_text(str(time.time()))

SYSTEM = """You are a counterfactual history analysis engine for the year 1914.
You receive a branch point and generate a plausible worldline trajectory from it.
Output STRICT YAML only — no prose outside the YAML block.
Mark every claim in trajectory/events as [fact-before-1914-06] or [inference].
Use English for all structural fields."""

CONTEXT = """Facts as of 1914-06-28 (no opinions):
- Triple Entente: France, Russia, Britain. Central Powers: Germany, Austria-Hungary (Ottoman leaning).
- Schlieffen plan finalized 1913: France first via Belgium, then Russia.
- Russian general mobilization plan known to general staffs.
- Balkan Wars 1912-13 just ended; Serbia doubled territory; Austria-Hungary wants it contained.
- Anglo-German naval race ongoing since 1906. French revanchism over Alsace-Lorraine since 1871.
- Franz Ferdinand, heir to Austria-Hungary, is in Sarajevo today for military maneuvers.
"""

BRANCHES = [
    ("A", "Gavrilo Princip shoots and kills Franz Ferdinand and his wife on 1914-06-28 in Sarajevo."),
    ("B", "The assassination attempt on Franz Ferdinand fails on 1914-06-28; Princip is arrested before firing. No shots hit the couple."),
    ("C", "The assassination attempt on Franz Ferdinand fails, but a general European great-power war begins within 24 months through another trigger."),
]

SCHEMA = """Output this exact YAML schema (no other text):
branch_id: <A|B|C>
premise: <one sentence>
trajectory:
  - phase: <name>
    window: <YYYY-MM to YYYY-MM>
    events: [<key events, each marked [fact-before-1914-06] or [inference]>]
crystallization_direction: <one sentence: what this worldline converges toward by 1920>
windows:
  - possibility: <a possibility>
    state: <opened|closed|contested>
    timing: <when it opens/closes>
necessity_candidates: [<things that would happen in ANY worldline from this branch point>]
contingency: [<things unique to this worldline>]
provenance_notes: <one sentence on your confidence basis>"""

ANCHORS_1914_1918 = {
    "austrian ultimatum to serbia (1914-07)": ["ultimatum"],
    "war declarations (1914-08)": ["declaration of war", "declares war", "declared war", "war on"],
    "german invasion of belgium (1914-08)": ["belgium", "liège", "liege"],
    "tannenberg (1914-08)": ["tannenberg"],
    "first marne (1914-09)": ["marne"],
    "trench stalemate (1915)": ["trench", "stalemate"],
    "verdun (1916)": ["verdun"],
    "somme (1916)": ["somme"],
    "russian revolution (1917)": ["revolution", "february", "october", "tsar"],
    "us entry (1917)": ["united states", "america", "u.s."],
    "armistice (1918-11)": ["armistice"],
}


def generate_branch(branch_id: str, premise: str, api_key: str) -> dict:
    user = f"{CONTEXT}\nBranch premise: {premise}\n\n{SCHEMA}"
    text = call_api(user, api_key, model="deepseek-v4-flash", max_tokens=4096,
                    temperature=0.6, timeout=180, system_message=SYSTEM,
                    thinking=False)
    import yaml
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("yaml"):
            cleaned = cleaned[4:]
    try:
        return yaml.safe_load(cleaned)
    except Exception as e:
        return {"branch_id": branch_id, "parse_error": str(e), "raw": text[:2000]}


def flatten_text(obj) -> str:
    if isinstance(obj, dict):
        return " ".join(flatten_text(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(flatten_text(v) for v in obj)
    return str(obj)


def standing_wave(branches: dict) -> dict:
    """Mechanical necessity/contingency split across branches."""
    nec = {bid: set(map(str.lower, b.get("necessity_candidates", []) or []))
           for bid, b in branches.items() if isinstance(b, dict)}
    if len(nec) < 2:
        return {"shared": [], "note": "insufficient branches"}
    shared_concepts = {}
    for bid, items in nec.items():
        for item in items:
            words = set(item.split()) - {"the", "a", "an", "of", "to", "in", "on", "and", "or", "by", "for", "with", "from", "as", "at", "be", "is", "are"}
            for w in words:
                if len(w) > 3:
                    shared_concepts.setdefault(w, set()).add(bid)
    shared = sorted([w for w, bids in shared_concepts.items()
                     if len(bids) == len(nec)], key=lambda w: -len(shared_concepts[w]))
    return {"shared_concepts_all_branches": shared,
            "per_branch_items": {k: sorted(v) for k, v in nec.items()}}


def verify_anchor(branch: dict) -> dict:
    text = flatten_text(branch).lower()
    hits = {name: any(kw in text for kw in kws) for name, kws in ANCHORS_1914_1918.items()}
    return {"hits": hits, "hit_rate": round(sum(hits.values()) / len(hits), 3)}


def main():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    branches = {}
    for bid, premise in BRANCHES:
        print(f"生成分支 {bid}…", flush=True)
        branches[bid] = generate_branch(bid, premise, api_key)

    wave = standing_wave(branches)
    verify_a = verify_anchor(branches.get("A", {}))

    report = {
        "experiment": "Stage 3 首跑：薩拉熱窩分支注入（1914-06-28）",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "branches": branches,
        "standing_wave": wave,
        "verify_branch_A_vs_real_history": verify_a,
    }
    out = REPO / "data" / "sarajevo_branch_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 駐波（全分支共享概念）:", wave["shared_concepts_all_branches"][:15])
    print("=== A 分支錨點校準:", f"{verify_a['hit_rate']:.0%}",
          [k for k, v in verify_a["hits"].items() if not v])
    print(f"報告落盤: {out}")


if __name__ == "__main__":
    main()
