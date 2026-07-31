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
from datetime import date, timedelta
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
from spectrum_os.synth.anchors import _load_ecc_call_api
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


def dry_run_enumerate_gate(gate_input: dict, **_kwargs: Any) -> dict:
    """Mock Mode-B enumerate gate for dry-run ensemble simulation (zero LLM)."""
    t = gate_input.get("state_vector", {}).get("t", 0)
    return {
        "accidents": [
            {
                "pattern": f"railway_labor_action_t{t}",
                "instances": [f"iron_workers_walkout_t{t}", f"coal_depot_shutdown_t{t}"],
                "domain": "社會",
                "source": "emergent",
                "usage": "expression",
                "grounding": "合理推測",
                "world_development": f"1914-{t}: 鐵路勞工集體行動癱瘓動員節點",
            }
        ],
        "generated_by": {"gate": "dry_run_enumerate", "n_accidents": 1},
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
    """Construct 1914-07 situation payload for Sarajevo pilot.

    ``date`` is the situation's physical-calendar base (1914-07-01). N5's
    ``residual_key_fn`` advances it by ``RESIDUAL_STEP_DAYS`` per Markov step
    — a month-key approximation of the pilot's own clock (S6 time-cursor spec
    is out of scope; no alpha_1 event calendar is injected).
    """
    init_vec = build_sarajevo_initial_vector()
    return {
        "vector_6d": init_vec["vector"],
        "local_texture": SARAJEVO_1914_LOCAL_TEXTURE,
        "digest": SARAJEVO_1914_DIGEST,
        "date": "1914-07-01",
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


# --- N5: residual-trigger wiring (Mode B two-phase, version B) ---

RESIDUAL_STEP_DAYS = 7  # physical-calendar step size (month-key approximation)


def _step_to_date(base_date: str, t: int, step_days: int = RESIDUAL_STEP_DAYS) -> str:
    """Map Markov step ``t`` to a physical-calendar date.

    Each step advances the situation's base date by ``step_days`` calendar
    days. This is the pilot's own clock (S6 time cursor is out of scope) —
    deliberately no alpha_1 event calendar.
    """
    return (date.fromisoformat(base_date) + timedelta(days=t * step_days)).isoformat()


def _filter_sharp_residual_table(residual_table: Any) -> dict:
    """Run-specific sharpness policy for the Sarajevo pilot.

    Only the date-localized sharp criteria (saturation / decoupling) fire
    Mode B. ``correlation_flip`` is arc-granularity — 91 continuous months
    (1913-06..1920-12) covering the ENTIRE pilot window — the same broad-net
    degeneracy that demoted ``gap`` to amplifier-only in Round-3.5: firing on
    it would switch every gate point to Mode B and leave the reflect path
    (standing wave / anchors) structurally empty. ``gap`` is dropped here too
    (it never fires alone in module logic; the enumerate prompt reads the
    situation, not the table).
    """
    from spectrum_os.synth.residual_trigger import load_residual_table

    table = load_residual_table(residual_table)
    firing_criteria = {"saturation", "decoupling"}
    out: dict = {}
    for key, entry in table.items():
        if not isinstance(entry, dict):
            continue
        firing = [c for c in entry.get("criteria", []) if c in firing_criteria]
        if firing:
            out[key] = {"criteria": firing, "detail": entry.get("detail", {})}
    return out


def _build_mode_b_summary(
    sharp_table: dict, situation_date: str, gate_ts: tuple = (3, 7, 11)
) -> list[dict]:
    """Per-gate-point Mode A/B decision (deterministic, zero LLM cost)."""
    from spectrum_os.synth.residual_trigger import should_enumerate

    summary = []
    for t in gate_ts:
        date_key = _step_to_date(situation_date, t)
        entry = sharp_table.get(date_key, {})
        criteria = entry.get("criteria", []) if isinstance(entry, dict) else []
        fired = should_enumerate(sharp_table, date_key)
        summary.append(
            {
                "t": t,
                "date": date_key,
                "mode": "enumerate" if fired else "reflect",
                "criteria": criteria,
            }
        )
    return summary


def run_sarajevo_pilot(
    live: bool = False,
    reporter: bool = False,
    n_branches: int = 10,
    horizon: int = 24,
    seed: int = 42,
    out_path: str = "data/stage3_sarajevo_report.json",
    n_samples: int = 3,
    residual_table: Any = None,
) -> dict:
    """Run full Sarajevo 1914 pilot pipeline.

    N5: when ``residual_table`` (dict or JSON path) is provided, the Mode-B
    residual trigger is wired into ``run_ensemble`` — gate points whose
    physical-calendar date (see ``_step_to_date``) hits a sharp residual
    criterion (saturation / decoupling — see ``_filter_sharp_residual_table``)
    switch to ``alt_gate_enumerate`` (Mode B) instead of the reflect path.
    Live cost is tracked through a counting ``call_api_fn`` wrapper.
    """
    api_call_count = {"n": 0}

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
    gate_kwargs: dict[str, Any] = {}
    enumerate_gate_fn = None  # live: default alt_gate_enumerate
    if live:
        print("[LIVE MODE] Invoking alt_gate LLM gate...")
        estimated_calls = min(12, n_branches * (horizon // 4))
        print(f"Estimated max gate calls: {estimated_calls}")
        gate_fn = alt_gate
        # N5: count actual API calls through a wrapper (cost tracking).
        real_call_api = _load_ecc_call_api()

        def counting_call_api(*args: Any, **kwargs: Any) -> str:
            api_call_count["n"] += 1
            return real_call_api(*args, **kwargs)

        gate_kwargs["call_api_fn"] = counting_call_api
    else:
        gate_fn = dry_run_gate
        # N5: Mode B must NOT hit the real API in dry-run — mock enumerate gate.
        enumerate_gate_fn = dry_run_enumerate_gate

    # N5: residual-trigger wiring (Mode B two-phase)
    residual_trigger = None
    residual_key_fn = None
    mode_b_summary: list[dict] = []
    if residual_table is not None:
        sharp_table = _filter_sharp_residual_table(residual_table)
        situation_date = situation.get("date", "1914-07-01")
        gate_ts = tuple(t for t in range(horizon - 1) if (t + 1) % 4 == 0)
        residual_trigger = sharp_table  # Mapping -> auto-wrapped by run_ensemble
        residual_key_fn = lambda t, ctx: _step_to_date(  # noqa: E731
            ctx["situation"].get("date", situation_date), t
        )
        mode_b_summary = _build_mode_b_summary(sharp_table, situation_date, gate_ts)
        print(
            f"[N5] sharp residual table: {len(sharp_table)} keys; gate-point modes: "
            + ", ".join(
                f"t={s['t']} {s['date']} {s['mode']}" for s in mode_b_summary
            )
        )

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
        residual_trigger=residual_trigger,
        residual_key_fn=residual_key_fn,
        enumerate_gate_fn=enumerate_gate_fn,
        **gate_kwargs,
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
            "total_enumerate_calls": ensemble_res["meta"].get("total_enumerate_calls", 0),
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
            "total_enumerate_calls": ensemble_res["meta"].get("total_enumerate_calls", 0),
            "n_reflect_points": ensemble_res["meta"]["total_gate_calls"]
            - ensemble_res["meta"].get("total_enumerate_calls", 0),
            "actual_api_calls": api_call_count["n"],
            "mode": "live" if live else "dry-run",
            "estimated_cost_usd": (
                round(api_call_count["n"] * 0.001, 4) if live else 0.0
            ),
        },
        "mode_b_switching": {
            "policy": (
                "sharp-only (saturation/decoupling); correlation_flip demoted to "
                "amplifier for this pilot (arc-granularity 91-month net covers the "
                "entire window — broad-net degeneracy, Round-3.5 precedent)"
            ),
            "step_days": RESIDUAL_STEP_DAYS,
            "situation_date": situation.get("date"),
            "gate_points": mode_b_summary,
        },
        # Mode-B accident inventory (never flows into tags — F10)
        "mode_b_accidents": {
            str(b.get("branch_id", i)): {
                str(t): accs for t, accs in b.get("accidents_by_step", {}).items()
            }
            for i, b in enumerate(ensemble_res["branches"])
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
    parser.add_argument(
        "--residual-table",
        type=str,
        default=None,
        help="Residual table JSON path (N5 Mode-B two-phase trigger)",
    )

    args = parser.parse_args()
    run_sarajevo_pilot(
        live=args.live,
        reporter=args.reporter,
        n_branches=args.branches,
        horizon=args.horizon,
        seed=args.seed,
        out_path=args.out,
        n_samples=args.n_samples,
        residual_table=args.residual_table,
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
