#!/usr/bin/env python3
"""
ab_prompt_test.py — Prompt A/B 對照測試

對兩版 system prompt（A=原始 / B=編輯者建議版）用同一處境、同一約束場、
同一 seed 跑 live probe_tree，比較生成品質指標。

指標：
  - avg_children_per_parent（樹寬——T6 核心：>1 = 真分岔，1.0 = 單鏈主線）
  - n_paths（路徑數）
  - echo_notes（層間 label 延續數——假 echo 記錄）
  - rejected（被拒節點數）
  - gate_nodes（門節點命中）
  - prose_rejection（散文式拒絕率）
  - anti_contamination（抗污染）
  - archetypes（世界原型數）
  - 成本（calls / USD）

用法：
    export $(grep -E '^DEEPSEEK_API_KEY=' /home/octy/projects/ECC/.env | xargs)
    /home/octy/projects/ECC/.venv/bin/python ab_prompt_test.py \
        --a /tmp/tree_original.txt --b /tmp/tree_suggested3.txt \
        --seeds 42 7 2026 --depth 3 --out /tmp/ab_report.json

設計：
  - 同一 (seed, depth, n_branch) 下 A/B 各跑一次 live，一次只變 system_message。
  - 多 seed 取平均，避免單次 LLM 隨機性主導結論。
  - prompt 凍結中：本腳本只注入替代 system_message，不改任何 prompt 文本。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage3_probe_pilot import (  # noqa: E402
    build_sarajevo_situation,
    build_constraint_from_ecc,
    make_live_gate_fn,
    load_live_gate,
    compute_reproducibility,
    measure_gate_nodes,
    measure_anti_contamination,
    measure_prose_rejection,
    measure_archetypes,
    estimate_cost_usd,
    ALPHA_20_ACCIDENTS_PLACEHOLDER,
)
from spectrum_os.synth.probe import probe_tree  # noqa: E402


def run_one(
    situation: dict,
    constraint: dict,
    system_message: str,
    *,
    n_branch: int,
    depth: int,
    w: float,
    seed: int,
    api_key: str,
    code: str | None,
    timeout: int = 45,
) -> tuple[dict, dict]:
    """跑一棵 live 樹 + 全部測量，回傳 (measurements, result)。"""
    ecc_call_api = load_live_gate()
    live_gate, live_state = make_live_gate_fn(ecc_call_api)
    result = probe_tree(
        situation,
        constraint,
        n_branch=n_branch,
        n_sample=1,
        depth=depth,
        w=w,
        gate_fn=None,
        call_api_fn=live_gate,
        api_key=api_key,
        seed=seed,
        code=code,
        system_message=system_message,
        timeout=timeout,
    )

    tree = result["trees"][0]
    m = {
        "avg_children_per_parent": tree["meta"]["avg_children_per_parent"],
        "n_paths": tree["meta"]["n_paths"],
        "calls": tree["meta"]["calls"],
        "truncated": tree["meta"]["truncated"],
        "echo_notes": len(result.get("echo_notes", [])),
        "rejected": len(result.get("rejected", [])),
        "gate_nodes": len(result.get("gate_nodes", [])),
        "prose_rejection": measure_prose_rejection(situation, n_branch),
        "anti_contamination": measure_anti_contamination(result),
        "archetypes": measure_archetypes(result),
        "reproducibility": compute_reproducibility(
            result, ALPHA_20_ACCIDENTS_PLACEHOLDER
        ),
        "rigidity_prevalence": [
            r["rigidity_prevalence"] for r in result.get("rigidity_map", [])
        ],
        "layer_branch_counts": [len(l["branches"]) for l in tree["layers"]],
    }
    usage = (live_state or {}).get("usage")
    cost = estimate_cost_usd(int(tree["meta"]["calls"]), usage, live=True)
    m["cost_usd"] = cost.get("estimated_usd")
    m["usage_in_tokens"] = cost.get("input_tokens")
    m["usage_out_tokens"] = cost.get("output_tokens")
    return m, result


def main() -> int:
    ap = argparse.ArgumentParser(description="Prompt A/B/C/D 對照測試")
    ap.add_argument("--a", required=True, help="A 版 prompt 檔案")
    ap.add_argument("--b", required=True, help="B 版 prompt 檔案")
    ap.add_argument("--c", default=None, help="C 版 prompt 檔案（可選）")
    ap.add_argument("--d", default=None, help="D 版 prompt 檔案（可選）")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42], help="seed 清單")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--n-branch", type=int, default=5)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--code", default=None, help="語碼（T16）")
    ap.add_argument("--timeout", type=int, default=45,
                    help="單次 API 呼叫 timeout 秒（預設 45——偶發連線掛死時快速失敗重試）")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "ab_prompt_report.json"))
    args = ap.parse_args()

    versions: list[tuple[str, str]] = [("A", args.a), ("B", args.b)]
    if args.c:
        versions.append(("C", args.c))
    if args.d:
        versions.append(("D", args.d))
    prompts = {tag: Path(p).read_text(encoding="utf-8") for tag, p in versions}
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("❌ 缺 DEEPSEEK_API_KEY")
        return 1

    situation = build_sarajevo_situation()
    constraint, cf_sources = build_constraint_from_ecc(
        None,
        "/home/octy/projects/ECC/index/data/anchor-field.json",
        "/home/octy/projects/ECC/index/data/narrative-spectrum-v3.1.json",
        "/home/octy/projects/ECC/index/data/gap-spectrum-signatures.json",
    )

    rows: list[dict] = []
    for seed in args.seeds:
        for tag, prompt in prompts.items():
            print(f"▶ [{tag}] seed={seed} depth={args.depth} n_branch={args.n_branch} …")
            try:
                m, _ = run_one(
                    situation, constraint, prompt,
                    n_branch=args.n_branch, depth=args.depth, w=args.w,
                    seed=seed, api_key=api_key, code=args.code, timeout=args.timeout,
                )
                rows.append({"tag": tag, "seed": seed, **m})
                print(f"  tree_width={m['avg_children_per_parent']} "
                      f"n_paths={m['n_paths']} echo={m['echo_notes']} "
                      f"rejected={m['rejected']} gate={m['gate_nodes']} "
                      f"$={m['cost_usd']:.4f}")
            except Exception as e:
                rows.append({
                    "tag": tag, "seed": seed,
                    "error": f"{type(e).__name__}: {e}",
                })
                print(f"  ❌ {type(e).__name__}: {str(e)[:120]}")

    # 彙總
    def avg(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    def errs(tag: str) -> list[str]:
        return [r.get("error", "") for r in rows if r["tag"] == tag and r.get("error")]

    agg: dict[str, dict] = {}
    for tag in prompts:
        rs = [r for r in rows if r["tag"] == tag and not r.get("error")]
        agg[tag] = {
            "runs": len(rs),
            "errors": errs(tag),
            "avg_tree_width": avg([r["avg_children_per_parent"] for r in rs]),
            "avg_n_paths": avg([r["n_paths"] for r in rs]),
            "avg_echo": avg([r["echo_notes"] for r in rs]),
            "avg_rejected": avg([r["rejected"] for r in rs]),
            "avg_gate_nodes": avg([r["gate_nodes"] for r in rs]),
            "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in rs), 4),
        }

    best_tag = max(agg, key=lambda t: agg[t]["avg_tree_width"])
    report = {
        "meta": {
            "files": {t: p for t, p in versions},
            "seeds": args.seeds, "depth": args.depth,
            "n_branch": args.n_branch, "w": args.w,
            "prompt_lens": {t: len(p) for t, p in prompts.items()},
            "constraint_source": cf_sources,
        },
        "rows": rows,
        "agg": agg,
        "verdict_hint": f"{best_tag} 勝（樹寬最高）",
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 彙總 ===")
    for tag in prompts:
        a = agg[tag]
        print(f"[{tag}] 成功={a['runs']}/{len(args.seeds)} 樹寬={a['avg_tree_width']} "
              f"n_paths={a['avg_n_paths']} echo={a['avg_echo']} "
              f"rejected={a['avg_rejected']} gate={a['avg_gate_nodes']} "
              f"$={a['total_cost_usd']}")
        if a["errors"]:
            print(f"   ❌ 失敗: {a['errors']}")
    print(f"verdict_hint: {report['verdict_hint']}")
    print(f"報告: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
