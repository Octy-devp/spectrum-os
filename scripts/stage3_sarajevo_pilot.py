#!/usr/bin/env python3
"""Stage 3 Pilot: Sarajevo 1914 Counterfactual Branch Experiment (PLAN-23 §5).

Executes full Stage 3 pipeline on Sarajevo 1914 July Crisis counterfactual state:
1. Data source access via sources.py registry (NO direct file open of ECC files).
2. Initial state vector: 1914-06 6D StateVector with explicit provenance.
3. Action injection: {"t": 1, "reweight": {"role": "alternative", "factor": 2.0}}.
4. Pay-as-you-go ensemble simulation (n=10 branches, horizon=24 steps).
5. Concept-level standing wave decomposition (nodes, antinodes, divergence_curve).
6. Mechanical anchor comparison against alpha_1 historical anchor tags with verify.evaluate.
7. Output JSON report saved to data/stage3_sarajevo_report.json.

Default mode is dry-run. Run with --live to invoke actual alt_gate LLM gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

# Ensure local spectrum_os is imported
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectrum_os import sources, osc
from spectrum_os.kernel import verify
from spectrum_os.quantum.standing_wave import standing_wave
from spectrum_os.quantum.vector import StateVector, vector_from_roles
from spectrum_os.synth.alt_gate import alt_gate
from spectrum_os.synth.ensemble import run_ensemble
from feeds.ecc_feeds import register_ecc_sources


ANCHORS_1914_1918 = {
    "austrian ultimatum to serbia (1914-07)": ["ultimatum", "通牒"],
    "war declarations (1914-08)": ["declaration of war", "declares war", "declared war", "war on", "宣戰"],
    "german invasion of belgium (1914-08)": ["belgium", "liège", "liege", "比利時"],
    "tannenberg (1914-08)": ["tannenberg", "坦能堡"],
    "first marne (1914-09)": ["marne", "馬恩河"],
    "trench stalemate (1915)": ["trench", "stalemate", "壕溝", "對峙"],
    "verdun (1916)": ["verdun", "凡爾登"],
    "somme (1916)": ["somme", "索姆河"],
    "russian revolution (1917)": ["revolution", "february", "october", "tsar", "革命", "沙皇"],
    "us entry (1917)": ["united states", "america", "u.s.", "美國", "美軍"],
    "armistice (1918-11)": ["armistice", "停戰"],
}

SARAJEVO_1914_LOCAL_TEXTURE = {
    "ultimatum_deadline_hours": 48,
    "mobilization_timetable_days": {
        "austria_hungary": 16,
        "russia": 14,
        "germany": 3,
        "france": 10,
    },
    "railway_capacity_trains_per_day": 360,
    "belgrade_border_distance_km": 5,
    "provenance": "1914 July Crisis historical logistics and diplomatic parameters (CMH & M-E-G baseline)",
}

SARAJEVO_1914_DIGEST = """【1914年7月 波士尼亞與赫塞哥維納——社會物質結構】
一、經濟：農業危機——1908 併吞後連續歉收，穀價下跌；農民對高利貸者與稅吏的負債累積，1913-14 冬季部分縣份欠稅查封。土地關係：半封建 kmet 佃農制，多數農地由佃農耕作，實物地租上繳地主（beg/aga 多為穆斯林）或帝國特許經營者；1909 土地改革草案在帝國議會停滯。奧斯曼債務與關稅：1878 占領後攤付奧斯曼公共債務，地方稅收抽成上繳維也納財政部。鐵路與勞工：1906 通車窄軌 Bosnabahn 連結礦山、農產品集散地與邊境；建設季雇季節勞工，1914 春失業率上升；1907 成立的鐵路工會要求最低工資與八小時制，罷工醞釀中。
二、生產：菸草專賣——帝國 Tabakregie 壟斷收購定價，菸農多為穆斯林小農，1913 收購價下調引發零星拒售。軍需工業：波士尼亞無大型兵工廠，軍需訂單集中維也納、斯泰爾、布爾諾。運輸網絡：鐵路單軌窄軌、日貨運量有限；薩拉熱窩距貝爾格勒邊境 5 公里，兩國鐵路未直連，貨物須邊境駁運；季節性封路（融雪、洪水）常見。
三、社會關係網絡：官僚 vs 南斯拉夫農民/中產——哈布斯堡民政官僚多為德裔與捷克裔把持縣級行政，南斯拉夫裔中產（商人、教師、文書）被排除於高級職位，日常摩擦表現為稅務、徵兵、學校語言爭執。軍隊 vs 文官——駐波 XV 軍團與民政當局在動員權、鐵路優先權、邊境警戒上爭執；軍事法庭與民事法庭管轄權重疊，戒嚴令隨時可局部生效。警察與線民：帝國警察與憲兵維持線民名冊，監視民族主義社團、工會與報紙；1908 後秘密報告系統擴大，密探滲入學生與合作社。民族主義組織的社會基礎：青年波士尼亞（Mlada Bosna）由學生與半知識分子組成，多出身農家靠獎學金就學；與貝爾格勒塞爾維亞學生團體有書信往來；克羅埃西亞裔與塞爾維亞裔農民合作社（zadruga）為宣傳滲透點。邊境社會：塞爾維亞—奧匈邊境走私、越境勞工、親屬網絡使情報與人員雙向流動。
四、約束：奧匈動員 16 日、俄國 14 日、德國 3 日、法國 10 日；48 小時通牒時限；鐵路日容量 360 列；貝爾格勒距邊境 5 公里。"""


def dry_run_gate(gate_input: dict, **_kwargs: Any) -> dict:
    """Mock gate function for dry-run ensemble simulation."""
    t = gate_input.get("state_vector", {}).get("t", 0)
    tags = ["july crisis", "crisis management"]

    if gate_input.get("situation") is not None:
        tags.append("situation_received")
        if "local_texture" in gate_input["situation"]:
            tags.append("texture_attached")

    if t <= 4:
        tags.extend(["ultimatum", "austrian ultimatum"])
    elif t <= 8:
        tags.extend(["declaration of war", "belgium"])
    elif t <= 12:
        tags.extend(["tannenberg", "marne"])
    elif t <= 16:
        tags.extend(["trench stalemate", "verdun"])
    else:
        tags.extend(["russian revolution", "armistice"])

    return {
        "walk": {
            "crisis": "july crisis escalation",
            "lag": "mobilization delay",
            "alive_space": "diplomatic channel",
        },
        "candidates": [
            {
                "label": f"sarajevo_alt_step_{t}",
                "concept_tags": tags,
                "provenance_hint": "novel",
                "novel": True,
                "adversary_flag": False,
                "weight": 1.0,
            }
        ],
        "evidence": ["july crisis tension baseline"],
        "confidence": {"score": 1.0, "basis": "dry_run_mock"},
    }


def build_sarajevo_initial_vector() -> dict:
    """Construct 1914-06 6D StateVector with explicit provenance."""
    roles_1914_06 = {"crisis": 0.55, "lag": 0.20, "alternative": 0.15, "direction": 0.10}
    spectrum_1914_06 = {"period_months": 12.0, "phase": 0.0, "amplitude": 0.85}
    vec = vector_from_roles(roles_1914_06, spectrum_1914_06)
    
    return {
        "role": "crisis",
        "vector": vec.to_dict(),
        "provenance": {
            "source_id": "ecc-knowledge-edges",
            "baseline": "ha_tension 1914-06 July Crisis projection",
            "date": "1914-06",
            "dca_role": "crisis",
            "6d_vector_repr": repr(vec),
        },
    }


def build_sarajevo_situation() -> dict:
    """Construct 1914-07 situation payload for Sarajevo pilot."""
    init_vec = build_sarajevo_initial_vector()
    return {
        "vector_6d": init_vec["vector"],
        "local_texture": SARAJEVO_1914_LOCAL_TEXTURE,
        "digest": SARAJEVO_1914_DIGEST,
    }


def evaluate_anchor_comparison(branches: list[dict]) -> dict:
    """Compare branch trajectories against alpha_1 historical anchors."""
    n_branches = len(branches)
    if n_branches == 0:
        return {"anchor_branch_ratio": {}, "closest_branch_id": 0, "verify_result": {}}

    branch_hits: dict[int, dict[str, bool]] = {}
    branch_scores: dict[int, int] = {}

    for b in branches:
        bid = b.get("branch_id", 0)
        # Collect all tags across steps
        all_tags = []
        for step_tags in b.get("tags_per_step", {}).values():
            all_tags.extend(step_tags)
        text = " ".join(all_tags).lower()

        hits = {}
        for anchor_name, keywords in ANCHORS_1914_1918.items():
            hits[anchor_name] = any(kw.lower() in text for kw in keywords)

        branch_hits[bid] = hits
        branch_scores[bid] = sum(1 for v in hits.values() if v)

    # Compute per-anchor branch ratio
    anchor_branch_ratio: dict[str, float] = {}
    for anchor_name in ANCHORS_1914_1918:
        count = sum(1 for bid in branch_hits if branch_hits[bid][anchor_name])
        anchor_branch_ratio[anchor_name] = round(count / n_branches, 4)

    closest_branch_id = max(branch_scores, key=lambda k: branch_scores[k]) if branch_scores else 0

    # Execute verify.evaluate against historical baseline (all 1.0)
    verify.init_log("data/state_log.jsonl")
    realized_baseline = [1.0] * len(ANCHORS_1914_1918)
    predicted_ratios = [anchor_branch_ratio[a] for a in ANCHORS_1914_1918]

    eval_result = verify.evaluate("3-sarajevo-vs-alpha1", realized_baseline, predicted_ratios)

    return {
        "anchors": ANCHORS_1914_1918,
        "anchor_branch_ratio": anchor_branch_ratio,
        "closest_branch_id": closest_branch_id,
        "closest_branch_hits": branch_scores.get(closest_branch_id, 0),
        "verify_result": {
            "prediction_id": eval_result["prediction_id"],
            "mae": round(eval_result["mae"], 4),
            "mape": round(eval_result["mape"], 4),
            "verdict": eval_result["verdict"].value,
            "re_calibrate": eval_result["re_calibrate"],
        },
    }


def run_sarajevo_pilot(
    live: bool = False,
    reporter: bool = False,
    n_branches: int = 10,
    horizon: int = 24,
    seed: int = 42,
    out_path: str = "data/stage3_sarajevo_report.json",
    n_samples: int = 3,
) -> dict:
    """Run full Sarajevo 1914 pilot pipeline."""
    # 1. Register sources & load via registry
    register_ecc_sources()
    substrate = sources.load_source("ecc-knowledge-edges")
    corpus = sources.load_source("ecc-clad-books")

    # 2. Initial state vector & situation payload
    initial_vec_data = build_sarajevo_initial_vector()
    situation = build_sarajevo_situation()

    # 3. Action injection
    action = {"t": 1, "reweight": {"role": "alternative", "factor": 2.0}}

    # 4. Gate selection
    if live:
        print("[LIVE MODE] Invoking alt_gate LLM gate...")
        estimated_calls = min(12, n_branches * (horizon // 4))
        print(f"Estimated max gate calls: {estimated_calls}")
        gate_fn = alt_gate
    else:
        gate_fn = dry_run_gate

    # 5. Run ensemble
    ensemble_res = run_ensemble(
        initial_vector=initial_vec_data,
        substrate=substrate,
        action=action,
        n_branches=n_branches,
        horizon=horizon,
        gate_fn=gate_fn,
        gate_every=4,
        max_gates=12,
        seed=seed,
        situation=situation,
        unified=True,
        reporter=reporter,
        n_samples=n_samples,
    )

    # 6. Standing wave decomposition
    sw = standing_wave(ensemble_res["branches"])

    # 7. Anchor comparison & verify.evaluate
    anchor_comp = evaluate_anchor_comparison(ensemble_res["branches"])

    # 8. Extract quarantine and adversary hit stats
    quarantine_hits = 0
    adversary_hits = 0
    for b in ensemble_res["branches"]:
        for step_tags in b.get("tags_per_step", {}).values():
            if any("quarantine" in str(t).lower() for t in step_tags):
                quarantine_hits += 1
            if any("adversary" in str(t).lower() for t in step_tags):
                adversary_hits += 1

    # 9. Build final report
    report = {
        "title": "Stage 3 Sarajevo 1914 Counterfactual Branch Pilot Report",
        "mode": "live" if live else "dry-run",
        "reporter": reporter,
        "initial_vector": initial_vec_data,
        "situation_payload": situation,
        "action_injection": action,
        "substrate_info": {
            "source_id": substrate.source_id,
            "nodes_count": len(substrate.nodes),
            "edges_count": len(substrate.edges),
        },
        "corpus_info": {
            "source_id": corpus.source_id,
            "entries_count": len(corpus.entries),
        },
        "branches_summary": {
            "n_branches": ensemble_res["meta"]["n_branches"],
            "horizon": ensemble_res["meta"]["horizon"],
            "total_gate_calls": ensemble_res["meta"]["total_gate_calls"],
            "seed": ensemble_res["meta"]["seed"],
        },
        # Per-branch tag inventory — makes anchor hits auditable from the artifact
        "branch_tags": {
            str(b.get("branch_id", i)): sorted(
                {t for tags in b.get("tags_per_step", {}).values() for t in tags}
            )
            for i, b in enumerate(ensemble_res["branches"])
        },
        "standing_wave": sw,
        "anchor_comparison": anchor_comp,
        "gate_calls_and_cost": {
            "total_gate_calls": ensemble_res["meta"]["total_gate_calls"],
            "mode": "live" if live else "dry-run",
            "estimated_cost_usd": (
                round(ensemble_res["meta"]["total_gate_calls"] * 0.001, 4) if live else 0.0
            ),
        },
        "quarantine_and_adversary_hits": {
            "quarantine_hits": quarantine_hits,
            "adversary_hits": adversary_hits,
        },
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report successfully saved to {out_file}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 Sarajevo 1914 Pilot")
    parser.add_argument("--live", action="store_true", help="Run with live LLM gate")
    parser.add_argument("--reporter", action="store_true", help="Run with reporter mode")
    parser.add_argument("--branches", type=int, default=10, help="Number of branches")
    parser.add_argument("--horizon", type=int, default=24, help="Horizon steps")
    parser.add_argument("--n-samples", type=int, default=3, help="Number of candidate samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="data/stage3_sarajevo_report.json", help="Output path")

    args = parser.parse_args()
    run_sarajevo_pilot(
        live=args.live,
        reporter=args.reporter,
        n_branches=args.branches,
        horizon=args.horizon,
        seed=args.seed,
        out_path=args.out,
        n_samples=args.n_samples,
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
