"""synth.rate_gate — LLM Rate Matrix Estimation Gate (H schema + G2).

PLAN-23 §6.6.2 + WORKFLOW §2.8 implementation:
- 身份錨: "你是這個歷史處境的活數學——你分配的每個強度決定哪個可能性繼續活著。"
- CLAD 處境化問句: 活矛盾 (C) -> 束縛 (L) -> 活出口 (A) -> 強度分配 (D).
- 肯定句佔據語義空間: "你只能在 legal_exits 內分配強度——這就是你的空間。"
- β₁ 錨定塊: Cutoff 1920-12, 數字與邊界事實, 零歷史史評.
- 機械否決層:
  - Placeholder & empty shell rejection ("None", "?", "按戰略意圖行動", placeholders).
  - Forbidden exit zeroing & legal_exits scope enforcement.
  - Mechanical confidence backfill from n-sample dispersion (FIX-21: LLM self-assessment discarded).
  - Mechanical filter (Quarantine Layer 1) on text fields.
- 貝葉斯融合 (§6.6.2): fused_posterior_counts = realized_counts + w_prior * llm_mean_rates.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from spectrum_os.quantum.dca_grammar import DCA_GRAMMAR
from spectrum_os.quantum.multigraph import ROLES
from spectrum_os.quantum.quarantine import mechanical_filter
from spectrum_os.kernel import verify as _verify
from spectrum_os.synth.anchors import (
    _FORBIDDEN_SERIES_KEYS,
    _load_ecc_call_api,
    _parse_json_loose,
)

RATE_GATE_SYSTEM_PROMPT_V1 = (
    "你是這個歷史處境的活數學——你分配的每個強度決定哪個可能性繼續活著。\n"
    "\n"
    "【空間約束】\n"
    "你只能在 legal_exits 內分配強度——這就是你的空間。\n"
    "每個強度都必須來自這個狀態獨有的處境——能套用到任何其他狀態的數字就是死數字，死數字會被機械層銷毀。\n"
    "\n"
    "【認知路徑——CLAD 問句序列】\n"
    "你必須沿著以下四步進行推理與強度分配：\n"
    "1. 這個狀態此刻被什麼束縛著？（C 活矛盾 / L 束縛）\n"
    "2. 哪幾條出口是活的？（A 活出口，必須是 legal_exits 的子集）\n"
    "3. 各出口的相對強度憑什麼這樣分？（D 強度分配，rates 數值和為 1.0）\n"
    "\n"
    "【β₁ 世界線時代約束（截止時間 t = 1920-12）】\n"
    "- 物質邊界：鋼鐵、鐵路、軍隊動員與後勤限制為硬約束。\n"
    "- 給數字不給史評：只分析客觀處境與限制，禁止給予後世道德評價或史學結論。\n"
    "\n"
    "【輸出格式】純結構 JSON——無開場白、無 markdown 圍欄、無解釋段落。\n"
    "Format:\n"
    "{\n"
    '  "walk": {\n'
    '    "crisis": "補給線斷裂",\n'
    '    "lag": "動員法令未完成",\n'
    '    "alive_exits": ["lag", "alternative"]\n'
    "  },\n"
    '  "rates": {\n'
    '    "lag": 0.60,\n'
    '    "alternative": 0.40\n'
    "  },\n"
    '  "evidence": ["鐵路延遲3日", "角色已持續14月"]\n'
    "}\n"
    "\n"
    "🔴 鐵律：\n"
    "- walk 的每個標籤必須 ≤ 20 字，且只能包含當前處境短語，禁止段落散文。\n"
    "- rates 只包含 legal_exits 內的鍵，數值 >= 0 且歸一化。\n"
    "- 絕對禁止在輸出中包含 confidence 欄位（置信度由機械層回填）。\n"
    "- 絕對禁止 placeholder（如 <...>, None, ?, 按戰略意圖行動）。\n"
)


class RateGateContractError(ValueError):
    """Raised when rate_gate output violates contract or mechanical checks."""


def _has_shell_refusal_or_placeholder(obj: Any) -> bool:
    """Recursively check for empty shell refusals or placeholder markers."""
    if obj is None:
        return True
    if isinstance(obj, str):
        s = obj.strip()
        lower = s.lower()
        if (
            "none" in lower
            or "?" in s
            or "按戰略意圖行動" in s
            or re.search(r"<[^>]+>", s)
        ):
            return True
    elif isinstance(obj, dict):
        return any(_has_shell_refusal_or_placeholder(v) for v in obj.values())
    elif isinstance(obj, list):
        return any(_has_shell_refusal_or_placeholder(x) for x in obj)
    return False


def assert_rate_gate_contract(output_data: dict, legal_exits: list[str]) -> None:
    """Mechanical contract assertion for rate_gate output.

    PLAN-23 §6.6.2 + FIX-01/06/14/15 implementation.
    """
    if not isinstance(output_data, dict):
        raise RateGateContractError(f"output must be dict, got {type(output_data).__name__}")

    for key in _FORBIDDEN_SERIES_KEYS:
        if key in output_data:
            raise RateGateContractError(
                f"forbidden key '{key}': rate gate must never output pointwise series values"
            )

    required_keys = ("walk", "rates", "evidence")
    missing = [k for k in required_keys if k not in output_data]
    if missing:
        raise RateGateContractError(f"missing required keys: {missing}")

    if _has_shell_refusal_or_placeholder(output_data):
        raise RateGateContractError("output contains forbidden placeholder or shell refusal string")

    # Check walk
    walk = output_data["walk"]
    if not isinstance(walk, dict):
        raise RateGateContractError(f"walk must be a dict, got {type(walk).__name__}")
    for k, v in walk.items():
        if k == "alive_exits":
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise RateGateContractError(f"walk key 'alive_exits' must be list of str, got {v!r}")
        else:
            if not isinstance(v, str):
                raise RateGateContractError(f"walk tag '{k}' must be str, got {v!r}")
            if len(v) > 50:
                raise RateGateContractError(f"walk tag '{k}' exceeds length limit: {v!r}")
            clean, matches = mechanical_filter(v)
            if not clean:
                raise RateGateContractError(f"walk tag '{k}' failed mechanical filter: {matches}")

    # Check evidence
    evidence = output_data["evidence"]
    if not isinstance(evidence, list):
        raise RateGateContractError(f"evidence must be a list, got {type(evidence).__name__}")
    for item in evidence:
        if not isinstance(item, str):
            raise RateGateContractError(f"evidence item must be str, got {type(item).__name__}")
        clean, matches = mechanical_filter(item)
        if not clean:
            raise RateGateContractError(f"evidence item failed mechanical filter: {matches}")

    # Check rates
    rates = output_data["rates"]
    if not isinstance(rates, dict):
        raise RateGateContractError(f"rates must be a dict, got {type(rates).__name__}")

    legal_set = set(legal_exits)
    for exit_name, val in rates.items():
        if exit_name not in legal_set and val > 0.0:
            raise RateGateContractError(
                f"rates key '{exit_name}' is not in legal_exits {legal_exits}"
            )
        if (
            not isinstance(val, (int, float))
            or isinstance(val, bool)
            or not np.isfinite(val)
            or val < 0
        ):
            raise RateGateContractError(
                f"rates value for '{exit_name}' must be non-negative finite float, got {val!r}"
            )


def mechanical_confidence_rate_gate(
    rate_samples: list[dict[str, float]],
    legal_exits: list[str],
) -> dict:
    """Compute mechanical confidence from n-sample dispersion (FIX-21).

    LLM self-assessment is discarded.
    """
    if not rate_samples or not legal_exits:
        return {
            "mean_std": 0.0,
            "score": 0.0,
            "basis": "n_sample_dispersion",
            "note": "no samples or legal exits provided",
        }

    stds = []
    for exit_name in legal_exits:
        vals = [float(sample.get(exit_name, 0.0)) for sample in rate_samples]
        stds.append(float(np.std(vals)))

    mean_std = float(np.mean(stds))
    confidence_score = float(max(0.0, min(1.0, 1.0 - 2.0 * mean_std)))

    return {
        "mean_std": round(mean_std, 4),
        "score": round(confidence_score, 4),
        "basis": "n_sample_dispersion",
        "note": "mechanically computed from n-sample dispersion; LLM self-assessment discarded (FIX-21)",
    }


def _build_rate_gate_prompt(input_data: dict) -> str:
    """Construct user prompt JSON string from input_data."""
    return json.dumps(input_data, indent=2, ensure_ascii=False)


def _write_rate_gate_state_log(result: dict) -> None:
    """Append a gate-call record to the verify state_log (FALSIFY-010 data foundation).

    Follows ``kernel.verify``'s own write pattern: always append to the
    in-memory buffer; persist to JSONL only when ``verify.init_log()`` has
    enabled file logging.  ``mae``/``mape`` keys are intentionally omitted so
    aggregate readers skip these entries (verdict comparison happens later,
    when transitions realize — that comparison is FALSIFY-010).
    """
    called_at = result["generated_by"]["called_at"]
    entry = {
        "ts": called_at,
        "prediction_id": f"rate_gate-{called_at}",
        "gate_type": "rate_gate",
        "verdict": "",
        "re_calibrate": False,
        "n_samples": result["generated_by"]["n_samples"],
        "w_prior": result["w_prior"],
        "llm_mean_rates": result["llm_mean_rates"],
        "fused_rates": result["fused_rates"],
        "confidence_score": result["confidence"]["score"],
    }
    _verify._state_log.append(entry)
    if _verify._log_path is not None:
        _verify._append_jsonl(_verify._log_path, entry)


def rate_gate(
    input_data: dict,
    *,
    call_api_fn: Callable | None = None,
    api_key: str | None = None,
    model: str = "deepseek-v4-flash",
    temperature: float = 0.6,
    n_samples: int = 3,
    w_prior: float = 5.0,
    max_tokens: int = 2048,
) -> dict:
    """LLM Rate Estimation Gate with H schema + G2 Bayesian fusion.

    PLAN-23 §6.6.2 + WORKFLOW §2.8 implementation.

    Args:
        input_data: Input dict containing 'state_vector', 'thread_history',
            'local_texture', 'legal_exits', 'realized_counts'.
        call_api_fn: Callable for API invocation (injected for testing/mocking).
        api_key: DeepSeek API key. Read from DEEPSEEK_API_KEY env var if None.
        model: Model name. Default 'deepseek-v4-flash'.
        temperature: Sampling temperature (default 0.6).
        n_samples: Number of sampling calls (default 3).
        w_prior: Weight of LLM prior in Bayesian fusion (default 5.0).
        max_tokens: Token limit for completion.

    Returns:
        Dict containing llm_samples, llm_mean_rates, confidence,
        realized_counts, w_prior, fused_posterior_counts, fused_rates,
        generated_by provenance.
    """
    if call_api_fn is None:
        call_api_fn = _load_ecc_call_api()
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("API key required: pass api_key or set DEEPSEEK_API_KEY")

    legal_exits = input_data.get("legal_exits", [])
    if not legal_exits or not isinstance(legal_exits, list):
        raise ValueError("input_data must specify a non-empty list of legal_exits")

    user_prompt = _build_rate_gate_prompt(input_data)

    rate_samples: list[dict[str, float]] = []
    walks: list[dict] = []
    evidences: list[list[str]] = []

    for _ in range(n_samples):
        raw = call_api_fn(
            user_prompt,
            api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_message=RATE_GATE_SYSTEM_PROMPT_V1,
            response_format={"type": "json_object"},
            thinking=False,
        )

        parsed = _parse_json_loose(raw)
        assert_rate_gate_contract(parsed, legal_exits=legal_exits)

        # Enforce rates zeroing for non-legal exits & normalize over legal_exits
        raw_rates = parsed["rates"]
        clean_rates = {e: float(raw_rates.get(e, 0.0)) for e in legal_exits}
        total_r = sum(clean_rates.values())
        if total_r > 0:
            norm_rates = {e: clean_rates[e] / total_r for e in legal_exits}
        else:
            norm_rates = {e: 1.0 / len(legal_exits) for e in legal_exits}

        rate_samples.append(norm_rates)
        walks.append(parsed["walk"])
        evidences.append(parsed["evidence"])

    # Compute mean LLM rates
    llm_mean_rates = {}
    for exit_name in legal_exits:
        llm_mean_rates[exit_name] = float(
            np.mean([s[exit_name] for s in rate_samples])
        )
    tot = sum(llm_mean_rates.values())
    if tot > 0:
        llm_mean_rates = {k: v / tot for k, v in llm_mean_rates.items()}

    # Mechanical confidence backfill (FIX-21: discard LLM confidence if present)
    confidence = mechanical_confidence_rate_gate(rate_samples, legal_exits)

    # Bayesian fusion (§6.6.2): posterior_counts = realized_counts + w_prior * llm_mean_rates
    realized_counts = input_data.get("realized_counts", {})
    fused_posterior_counts = {}
    for r in ROLES:
        if r in legal_exits:
            rc = float(realized_counts.get(r, 0.0))
            pr = float(llm_mean_rates.get(r, 0.0))
            fused_posterior_counts[r] = rc + w_prior * pr
        else:
            fused_posterior_counts[r] = 0.0

    denom = sum(fused_posterior_counts.values())
    fused_rates = {}
    for r in ROLES:
        fused_rates[r] = fused_posterior_counts[r] / denom if denom > 0 else 0.0

    # Write workflow timestamp hook
    try:
        with open("/tmp/.ecc_workflow_hook_timestamp", "w", encoding="utf-8") as fh:
            fh.write(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass

    result = {
        "llm_samples": rate_samples,
        "llm_walks": walks,
        "llm_evidences": evidences,
        "llm_mean_rates": llm_mean_rates,
        "confidence": confidence,
        "realized_counts": realized_counts,
        "w_prior": float(w_prior),
        "fused_posterior_counts": fused_posterior_counts,
        "fused_rates": fused_rates,
        "generated_by": {
            "gate": "synth.rate_gate",
            "model": model,
            "n_samples": n_samples,
            "temperature": temperature,
            "called_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    # FALSIFY-010 data foundation: every gate call is recorded (§6.6.5)
    _write_rate_gate_state_log(result)

    return result
