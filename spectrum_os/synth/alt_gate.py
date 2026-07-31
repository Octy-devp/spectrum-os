"""synth.alt_gate — LLM Alternative Generation Gate (H schema family second instance).

PLAN-23 §5 + §6.5 (抗污染憲章) + WORKFLOW §2.8 implementation:
- M-E-G:
  - M = Current state (6D vector / thread history / texture) + substrate existing labels.
  - E = CLAD question focusing on A (alive possibilities: divergence > convergence).
  - G = Structured alternative candidates.
- Output contract assertion `assert_alt_gate_contract`.
- Mechanical veto layer:
  - Grammar check: transition validation via `validate_alternative`.
  - Mechanical deduplication against existing labels (`novel: false`).
  - Placeholder & empty shell rejection ("None", "?", "按戰略意圖行動", placeholders).
  - Prose keys refusal (narrative / description / explanation forbidden in candidate dicts).
  - Mechanical confidence backfill from n-sample label dispersion (LLM self-assessment discarded).
  - Mechanical filter (Quarantine Layer 1) on text fields.
- 2-Stage Adversarial Call (§六․五-5):
  - Stage 1: Candidate generation.
  - Stage 2: Adversary attack prompt ("你來自一條不同的歷史線——找出這些候選偷渡了哪個既定結局").
  - Flagged candidates get `adversary_flag: true` and weight halved (`weight: 0.5`).
- State log recording: `gate_type: "alt_gate"`.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from spectrum_os.kernel import verify as _verify
from spectrum_os.quantum.dca_grammar import validate_alternative
from spectrum_os.quantum.quarantine import mechanical_filter
from spectrum_os.synth.anchors import (
    _FORBIDDEN_SERIES_KEYS,
    _load_ecc_call_api,
    _parse_json_loose,
)

ALT_GATE_SYSTEM_PROMPT_V1 = (
    "你是這個歷史處境的活數學——你負責生成可能存在的活出口概念（A 活出口／發散優於收束）。\n"
    "\n"
    "【空間與任務】\n"
    "你必須給出當前處境下可能活著的出口與概念標籤。發散優於收束。\n"
    "每個標籤都必須來自這個狀態獨有的處境，禁止段落散文與史評詮釋。\n"
    "\n"
    "【輸出格式】純結構 JSON——無開場白、無 markdown 圍欄、無解釋段落。\n"
    "Format:\n"
    "{\n"
    '  "walk": {\n'
    '    "crisis": "補給線斷裂",\n'
    '    "lag": "動員法令未完成",\n'
    '    "alive_space": "邊界物資替換機制"\n'
    "  },\n"
    '  "candidates": [\n'
    "    {\n"
    '      "label": "物資替換點",\n'
    '      "concept_tags": ["物資網絡", "替代通道"],\n'
    '      "provenance_hint": "novel",\n'
    '      "novel": true\n'
    "    }\n"
    "  ],\n"
    '  "evidence": ["物資流動滯後"]\n'
    "}\n"
    "\n"
    "🔴 鐵律：\n"
    "- walk 的每個標籤必須 ≤ 20 字，且只能包含當前處境短語，禁止段落散文。\n"
    "- candidates 中的 label 必須 ≤ 12 字，concept_tags 為短標籤陣列。\n"
    "- 絕對禁止在 candidates 中包含 prose 鍵（如 narrative, description, explanation）。\n"
    "- 絕對禁止在輸出中包含 confidence 欄位（置信度由機械層回填）。\n"
    "- 絕對禁止 placeholder（如 <...>, None, ?, 按戰略意圖行動）。\n"
)

ADVERSARY_SYSTEM_PROMPT_V1 = (
    "你來自一條不同的歷史線——找出這些候選偷渡了哪個既定結局。\n"
    "你必須審查傳入的候選標籤，指出哪些偷渡了既定歷史結局。\n"
    "輸出格式：純 JSON。\n"
    "{\n"
    '  "flagged_labels": ["偷渡結局標籤"],\n'
    '  "reasons": {"偷渡結局標籤": "偷渡既定結局"}\n'
    "}\n"
)

_FORBIDDEN_PROSE_KEYS = ("narrative", "description", "explanation", "text", "summary", "prose")


class AltGateContractError(ValueError):
    """Raised when alt_gate output violates contract or mechanical checks."""


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


def assert_alt_gate_contract(output_data: dict) -> None:
    """Mechanical contract assertion for alt_gate output.

    PLAN-23 §5 + §6.5 implementation.
    """
    if not isinstance(output_data, dict):
        raise AltGateContractError(f"output must be dict, got {type(output_data).__name__}")

    for key in _FORBIDDEN_SERIES_KEYS:
        if key in output_data:
            raise AltGateContractError(
                f"forbidden key '{key}': alt gate must never output pointwise series values"
            )

    required_keys = ("walk", "candidates", "evidence")
    missing = [k for k in required_keys if k not in output_data]
    if missing:
        raise AltGateContractError(f"missing required keys: {missing}")

    if _has_shell_refusal_or_placeholder(output_data):
        raise AltGateContractError("output contains forbidden placeholder or shell refusal string")

    # Check walk
    walk = output_data["walk"]
    if not isinstance(walk, dict):
        raise AltGateContractError(f"walk must be a dict, got {type(walk).__name__}")
    for k, v in walk.items():
        if not isinstance(v, str):
            raise AltGateContractError(f"walk tag '{k}' must be str, got {v!r}")
        if len(v) > 50:
            raise AltGateContractError(f"walk tag '{k}' exceeds length limit: {v!r}")
        clean, matches = mechanical_filter(v)
        if not clean:
            raise AltGateContractError(f"walk tag '{k}' failed mechanical filter: {matches}")

    # Check evidence
    evidence = output_data["evidence"]
    if not isinstance(evidence, list):
        raise AltGateContractError(f"evidence must be a list, got {type(evidence).__name__}")
    for item in evidence:
        if not isinstance(item, str):
            raise AltGateContractError(f"evidence item must be str, got {type(item).__name__}")
        clean, matches = mechanical_filter(item)
        if not clean:
            raise AltGateContractError(f"evidence item failed mechanical filter: {matches}")

    # Check candidates
    candidates = output_data["candidates"]
    if not isinstance(candidates, list):
        raise AltGateContractError(f"candidates must be a list, got {type(candidates).__name__}")

    for i, cand in enumerate(candidates):
        if not isinstance(cand, dict):
            raise AltGateContractError(f"candidate[{i}] must be a dict, got {type(cand).__name__}")

        # Check prose keys refusal
        for pkey in _FORBIDDEN_PROSE_KEYS:
            if pkey in cand:
                raise AltGateContractError(
                    f"candidate[{i}] contains forbidden prose key '{pkey}'"
                )

        req_cand_keys = ("label", "concept_tags", "provenance_hint", "novel")
        missing_cand = [k for k in req_cand_keys if k not in cand]
        if missing_cand:
            raise AltGateContractError(f"candidate[{i}] missing required keys: {missing_cand}")

        label = cand["label"]
        if not isinstance(label, str):
            raise AltGateContractError(f"candidate[{i}].label must be str, got {label!r}")
        if len(label) > 20:
            raise AltGateContractError(f"candidate[{i}].label exceeds length limit (<=20): {label!r}")
        clean, matches = mechanical_filter(label)
        if not clean:
            raise AltGateContractError(f"candidate[{i}].label failed mechanical filter: {matches}")

        tags = cand["concept_tags"]
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise AltGateContractError(f"candidate[{i}].concept_tags must be list of str, got {tags!r}")
        for t in tags:
            clean, matches = mechanical_filter(t)
            if not clean:
                raise AltGateContractError(f"candidate[{i}].concept_tag '{t}' failed mechanical filter: {matches}")

        prov = cand["provenance_hint"]
        if not isinstance(prov, str):
            raise AltGateContractError(f"candidate[{i}].provenance_hint must be str, got {prov!r}")
        clean, matches = mechanical_filter(prov)
        if not clean:
            raise AltGateContractError(f"candidate[{i}].provenance_hint failed mechanical filter: {matches}")

        if not isinstance(cand["novel"], bool):
            raise AltGateContractError(f"candidate[{i}].novel must be bool, got {cand['novel']!r}")

        # If candidate specifies role transitions, check with validate_alternative
        if "from_role" in cand and "to_role" in cand:
            if not validate_alternative(cand["from_role"], cand["to_role"]):
                raise AltGateContractError(
                    f"candidate[{i}] role transition '{cand['from_role']}' -> '{cand['to_role']}' is invalid per DCA grammar"
                )


def mechanical_confidence_alt_gate(candidate_samples: list[list[dict]]) -> dict:
    """Compute mechanical confidence from n-sample candidate label dispersion.

    LLM self-assessment is discarded.
    """
    if not candidate_samples:
        return {
            "mean_jaccard": 0.0,
            "score": 0.0,
            "basis": "n_sample_label_dispersion",
            "note": "no candidate samples provided",
        }

    label_sets = [
        set(cand["label"].strip() for cand in sample if isinstance(cand, dict) and "label" in cand)
        for sample in candidate_samples
    ]

    n = len(label_sets)
    if n <= 1:
        return {
            "mean_jaccard": 1.0,
            "score": 1.0,
            "basis": "n_sample_label_dispersion",
            "note": "single sample provided",
        }

    jaccards = []
    for i in range(n):
        for j in range(i + 1, n):
            s1, s2 = label_sets[i], label_sets[j]
            union = s1 | s2
            if not union:
                jaccards.append(1.0)
            else:
                jaccards.append(len(s1 & s2) / len(union))

    mean_jaccard = float(np.mean(jaccards))

    return {
        "mean_jaccard": round(mean_jaccard, 4),
        "score": round(mean_jaccard, 4),
        "basis": "n_sample_label_dispersion",
        "note": "mechanically computed from n-sample label dispersion; LLM self-assessment discarded",
    }


def _build_alt_gate_prompt(input_data: dict) -> str:
    """Construct user prompt JSON string from input_data."""
    return json.dumps(input_data, indent=2, ensure_ascii=False)


def _write_alt_gate_state_log(result: dict) -> None:
    """Append a gate-call record to the verify state_log."""
    called_at = result["generated_by"]["called_at"]
    entry = {
        "ts": called_at,
        "prediction_id": f"alt_gate-{called_at}",
        "gate_type": "alt_gate",
        "verdict": "",
        "re_calibrate": False,
        "n_samples": result["generated_by"]["n_samples"],
        "n_candidates": len(result.get("candidates", [])),
        "confidence_score": result["confidence"]["score"],
    }
    _verify._state_log.append(entry)
    if _verify._log_path is not None:
        _verify._append_jsonl(_verify._log_path, entry)


def alt_gate(
    input_data: dict,
    *,
    call_api_fn: Callable | None = None,
    api_key: str | None = None,
    model: str = "deepseek-v4-flash",
    temperature: float = 0.6,
    n_samples: int = 3,
    max_tokens: int = 2048,
) -> dict:
    """LLM Alternative Generation Gate with H schema + 2-Stage Adversarial Call.

    Args:
        input_data: Input dict containing 'state_vector', 'thread_history',
            'local_texture', and optional 'existing_labels'.
        call_api_fn: Callable for API invocation (injected for testing/mocking).
        api_key: DeepSeek API key. Read from DEEPSEEK_API_KEY env var if None.
        model: Model name. Default 'deepseek-v4-flash'.
        temperature: Sampling temperature (default 0.6).
        n_samples: Number of sampling calls for generation (default 3).
        max_tokens: Token limit for completion.

    Returns:
        Dict containing walk, candidates, evidence, confidence, generated_by provenance.
    """
    if call_api_fn is None:
        call_api_fn = _load_ecc_call_api()
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("API key required: pass api_key or set DEEPSEEK_API_KEY")

    user_prompt = _build_alt_gate_prompt(input_data)
    existing_labels_set = set(input_data.get("existing_labels", []))

    # Stage 1: Candidate Generation Call
    candidate_samples: list[list[dict]] = []
    walks: list[dict] = []
    evidences: list[list[str]] = []

    for _ in range(n_samples):
        raw = call_api_fn(
            user_prompt,
            api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_message=ALT_GATE_SYSTEM_PROMPT_V1,
            response_format={"type": "json_object"},
            thinking=False,
        )

        parsed = _parse_json_loose(raw)
        assert_alt_gate_contract(parsed)

        candidate_samples.append(parsed["candidates"])
        walks.append(parsed["walk"])
        evidences.append(parsed["evidence"])

    # Aggregate & mechanically deduplicate candidates
    seen_labels: set[str] = set()
    aggregated_candidates: list[dict] = []

    for sample in candidate_samples:
        for cand in sample:
            label = cand["label"].strip()
            # Copy candidate dict to avoid mutating original
            c_dict = dict(cand)
            c_dict["label"] = label
            # Mechanical deduplication check
            if label in existing_labels_set or label in seen_labels:
                c_dict["novel"] = False
            else:
                c_dict["novel"] = True
                seen_labels.add(label)

            # Prevent duplicate label entries in final candidate list
            if not any(existing["label"] == label for existing in aggregated_candidates):
                aggregated_candidates.append(c_dict)

    # Mechanical confidence backfill
    confidence = mechanical_confidence_alt_gate(candidate_samples)

    # Stage 2: Independent Adversarial Call (§六․五-5)
    adv_input = {
        "candidate_labels": [c["label"] for c in aggregated_candidates],
        "candidate_tags": [c["concept_tags"] for c in aggregated_candidates],
    }
    adv_prompt = json.dumps(adv_input, ensure_ascii=False)

    adv_raw = call_api_fn(
        adv_prompt,
        api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        system_message=ADVERSARY_SYSTEM_PROMPT_V1,
        response_format={"type": "json_object"},
        thinking=False,
    )

    adv_parsed = _parse_json_loose(adv_raw)
    flagged_set = set(adv_parsed.get("flagged_labels", []))

    # Apply adversarial flags and weight halving
    final_candidates: list[dict] = []
    for cand in aggregated_candidates:
        c_dict = dict(cand)
        if c_dict["label"] in flagged_set:
            c_dict["adversary_flag"] = True
            c_dict["weight"] = 0.5
        else:
            c_dict["adversary_flag"] = False
            c_dict["weight"] = 1.0
        c_dict["generated"] = True
        final_candidates.append(c_dict)

    # Aggregate walk and evidence from first sample
    final_walk = walks[0] if walks else {}
    final_evidence = list(dict.fromkeys(sum(evidences, [])))

    result = {
        "walk": final_walk,
        "candidates": final_candidates,
        "evidence": final_evidence,
        "confidence": confidence,
        "generated_by": {
            "gate": "synth.alt_gate",
            "model": model,
            "n_samples": n_samples,
            "temperature": temperature,
            "called_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    _write_alt_gate_state_log(result)

    return result
