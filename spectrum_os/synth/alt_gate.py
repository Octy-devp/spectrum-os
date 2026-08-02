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
  - Stage 2: Adversary usage-judge prompt (判別 expression vs substitution，以處境為據).
  - Flagged candidates get `adversary_flag: true` and weight halved (`weight: 0.5`).
- State log recording: `gate_type: "alt_gate"`.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from spectrum_os.kernel import verify as _verify
from spectrum_os.quantum.dca_grammar import validate_alternative
from spectrum_os.quantum.markov import count_transitions
from spectrum_os.quantum.multigraph import ROLES
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
    "此刻你是用法裁判——你判定每個候選把承繼的詞彙用成了表達（expression）還是取代（substitution）。\n"
    "你接收：處境（社會物質結構與鏡角）+ 候選（labels、concept_tags）。\n"
    "【兩軸判準】\n"
    "- expression：借用的詞服務於分支自己的內容——復活，合法。\n"
    "- substitution·gravity：把既定結局當必然前提塞入——偷渡，非法。\n"
    "- substitution·parody：無內容的借用——鬧劇，非法。\n"
    "- substitution·self-deception：借來的崇高感掩蓋有限內容——非法。\n"
    "【鐵律】判定必須以處境為據——同一個詞，在這種社會語境下可能是復活，換一種語境就是偷渡。\n"
    "不可無處境空判；也不可因詞彙來自過去就當偷渡（繼承是合法的）。\n"
    "輸出格式：純 JSON。\n"
    "{\n"
    '  "flagged_labels": ["偷渡結局標籤"],\n'
    '  "reasons": {"偷渡結局標籤": "substitution（gravity）——把既定結局當必然前提"}\n'
    "}\n"
)

_FORBIDDEN_PROSE_KEYS = ("narrative", "description", "explanation", "text", "summary", "prose")


class AltGateContractError(ValueError):
    """Raised when alt_gate output violates contract or mechanical checks."""


def _has_shell_refusal_or_placeholder(obj: Any) -> bool:
    """Recursively check for empty shell refusals or placeholder markers.

    🔴 2026-08-02 修復（T6 live pilot 發現）：標量 ``None`` **不再**判為空殼——
    第 1 層分支的 ``parent: null``（無父，語義合法）曾被誤判 → 整層被拒（flaky）。
    空殼判定只針對**字串內容**（"none"/"?"/"<...>"/拒絕語）的遞迴掃描。
    ``None`` 值本身是「缺席」（合法）；空 list/dict 由各語義檢查
    （conditions 非空 / roles 非空 / candidates 檢查）各司其職，不在此判。
    """
    if obj is None:
        return False  # 標量 None = 缺席（合法），非空殼拒絕
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


def assert_alt_gate_contract(
    output_data: dict,
    situation_labels: list[str] | None = None,
    *,
    echo_notes: list[str] | None = None,
) -> None:
    """Mechanical contract assertion for alt_gate output.

    PLAN-23 §5 + §6.5 implementation.

    N3 (Round 2): when ``situation_labels`` is provided, candidate LABELS that
    verbatim-repeat a situation-provided label are rejected (echo rejection is
    referenced by situation, not by a fixed vocabulary). Only LABEL is checked —
    concept_tags may reference situation material (revival / expression is legal).

    **語義中介取代黑名單（決策 1/4）**：``PROMPT_EXAMPLE_LABELS`` 命中**不再 raise**
    ——改為 append 到 ``echo_notes``（可選，None 時靜默略過，向後相容）。範例只是形狀
    （決策 4）；**situation echo 仍 fatal**——那是語義判準（以處境為參照，不是黑名單）。
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

    situation_set: set[str] | None = None
    if situation_labels:
        situation_set = {
            s.strip() for s in situation_labels if isinstance(s, str) and s.strip()
        }

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
        if label.strip() in PROMPT_EXAMPLE_LABELS:
            if echo_notes is not None:
                echo_notes.append(
                    f"example echo: candidate[{i}].label matches prompt example label "
                    f"'{label.strip()}'（降權，非致命——範例只是形狀）"
                )
        if situation_set is not None and label.strip() in situation_set:
            raise AltGateContractError(
                f"situation echo: candidate[{i}].label '{label.strip()}' repeats a label already provided by the situation"
            )
        if len(label) > 20:
            raise AltGateContractError(f"candidate[{i}].label exceeds length limit (<=20): {label!r}")
        clean, matches = mechanical_filter(label)
        if not clean:
            raise AltGateContractError(f"candidate[{i}].label failed mechanical filter: {matches}")

        tags = cand["concept_tags"]
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise AltGateContractError(f"candidate[{i}].concept_tags must be list of str, got {tags!r}")
        for t in tags:
            if t.strip() in PROMPT_EXAMPLE_LABELS:
                if echo_notes is not None:
                    echo_notes.append(
                        f"example echo: candidate[{i}].concept_tag matches prompt example label "
                        f"'{t.strip()}'（降權，非致命——範例只是形狀）"
                    )
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


def _extract_situation_labels(input_data: dict) -> list[str] | None:
    """Extract situation-referenced labels from ``input_data["situation"]``.

    N3 (Round 2): echo rejection is referenced by situation — the situation's
    digest full text plus local_texture string values become the reference
    labels. Returns None when no situation payload is provided (behavior
    unchanged).

    F7: probe.py 曾直接傳 bare situation dict（非包裝），使 ``input_data["situation"]``
    恆 None → 處境-echo 拒收靜默失效。此處防禦性兼容兩種形狀：包裝
    ``{"situation": {...}}``（alt_gate 輸入）與 bare situation dict
    （自身含 ``digest`` / ``local_texture`` 時視為處境本身）。
    """
    situation = input_data.get("situation")
    if not isinstance(situation, dict):
        # F7：bare situation dict 形狀（probe 直接傳處境）——含處境特徵鍵即當處境。
        if isinstance(input_data.get("digest"), str) or isinstance(
            input_data.get("local_texture"), dict
        ):
            situation = input_data
        else:
            return None
    labels: list[str] = []
    digest = situation.get("digest")
    if isinstance(digest, str) and digest.strip():
        labels.append(digest.strip())
    local_texture = situation.get("local_texture")
    if isinstance(local_texture, dict):
        for v in local_texture.values():
            if isinstance(v, str) and v.strip():
                labels.append(v.strip())
    return labels or None


def _build_alt_gate_prompt(input_data: dict) -> str:
    """Construct user prompt JSON string from input_data."""
    return json.dumps(input_data, indent=2, ensure_ascii=False)


def _write_alt_gate_state_log(result: dict, observations: list[str] | None = None) -> None:
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
    if observations:
        entry["observation"] = observations if len(observations) > 1 else observations[0]
    _verify._state_log.append(entry)
    if _verify._log_path is not None:
        _verify._append_jsonl(_verify._log_path, entry)


from spectrum_os.synth.gate_prompts import (
    GATE_SYSTEM_PROMPT_UNIFIED,
    PROMPT_EXAMPLE_LABELS,
    ROUTING_LEAK_KEYWORDS,
    VERSION_B_SYSTEM_PROMPT,
    check_routing_leak_or_schema,
)


def alt_gate_generate(
    input_data: dict,
    *,
    call_api_fn: Callable | None = None,
    api_key: str | None = None,
    model: str = "deepseek-v4-flash",
    temperature: float = 0.6,
    n_samples: int = 3,
    max_tokens: int = 2048,
    unified: bool = False,
    reporter: bool = False,
    **_kwargs: Any,
) -> dict:
    """Stage 1: Candidate Generation for LLM Alternative Gate."""
    if call_api_fn is None:
        call_api_fn = _load_ecc_call_api()
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("API key required: pass api_key or set DEEPSEEK_API_KEY")

    sys_prompt = GATE_SYSTEM_PROMPT_UNIFIED if (unified or reporter) else ALT_GATE_SYSTEM_PROMPT_V1
    existing_labels_set = set(input_data.get("existing_labels", []))
    situation_labels = _extract_situation_labels(input_data)
    # 語義中介取代黑名單（決策 4）：example echo 不再 fatal——記錄到 echo_notes（人機收束檢視）。
    echo_notes: list[str] = []

    candidate_samples: list[list[dict]] = []
    walks: list[dict] = []
    evidences: list[list[str]] = []
    observations: list[str] = []

    for _ in range(n_samples):
        if reporter:
            # --- Two-Stage Reporter Generation Path: observe -> compress ---
            # 1. Observe call (hot temp 0.6–0.7, prose output <= 150 words, NO format examples)
            obs_user_prompt = f"{_build_alt_gate_prompt(input_data)}\n[task:observe]"
            obs_raw = call_api_fn(
                obs_user_prompt,
                api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system_message=sys_prompt,
                thinking=False,
            )
            leak_err = check_routing_leak_or_schema(obs_raw, None, "observe")
            if leak_err:
                raise AltGateContractError(leak_err)
            observation_text = obs_raw.strip()
            observations.append(observation_text)

            # 2. Compress call (cool temp 0.0–0.25, distills observation prose into contract structure)
            compress_input = {
                "observation": observation_text,
                "state_vector": input_data.get("state_vector"),
                "situation": input_data.get("situation"),
            }
            comp_user_prompt = f"{_build_alt_gate_prompt(compress_input)}\n[task:compress]"

            parsed = None
            for attempt in range(2):
                curr_temp = min(0.25, 0.1 + attempt * 0.1)
                raw = call_api_fn(
                    comp_user_prompt,
                    api_key,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=curr_temp,
                    system_message=sys_prompt,
                    response_format={"type": "json_object"},
                    thinking=False,
                )
                try:
                    for lk in ROUTING_LEAK_KEYWORDS:
                        if lk in raw:
                            raise AltGateContractError(f"ROUTING_LEAK_DETECTED: response contains prompt leak keyword '{lk}'")
                    candidate_parsed = _parse_json_loose(raw)
                    leak_err = check_routing_leak_or_schema(raw, candidate_parsed, "compress")
                    if leak_err:
                        raise AltGateContractError(leak_err)
                    assert_alt_gate_contract(candidate_parsed, situation_labels, echo_notes=echo_notes)
                    parsed = candidate_parsed
                    break
                except Exception as e:
                    if attempt == 0:
                        continue
                    if isinstance(e, AltGateContractError):
                        raise e
                    raise AltGateContractError(str(e)) from e

            candidate_samples.append(parsed["candidates"])
            walks.append(parsed["walk"])
            evidences.append(parsed["evidence"])

        else:
            # --- Legacy Direct Path ---
            base_prompt = _build_alt_gate_prompt(input_data)
            user_prompt = f"{base_prompt}\n[task:generate]" if unified else base_prompt
            parsed = None
            for attempt in range(2):
                curr_temp = temperature + (0.1 if attempt == 1 else 0.0)
                raw = call_api_fn(
                    user_prompt,
                    api_key,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=curr_temp,
                    system_message=sys_prompt,
                    response_format={"type": "json_object"},
                    thinking=False,
                )
                try:
                    if unified:
                        for lk in ROUTING_LEAK_KEYWORDS:
                            if lk in raw:
                                raise AltGateContractError(f"ROUTING_LEAK_DETECTED: response contains prompt leak keyword '{lk}'")
                    candidate_parsed = _parse_json_loose(raw)
                    if unified:
                        leak_err = check_routing_leak_or_schema(raw, candidate_parsed, "generate")
                        if leak_err:
                            raise AltGateContractError(leak_err)
                    assert_alt_gate_contract(candidate_parsed, situation_labels, echo_notes=echo_notes)
                    parsed = candidate_parsed
                    break
                except Exception as e:
                    if attempt == 0:
                        continue
                    if isinstance(e, AltGateContractError):
                        raise e
                    raise AltGateContractError(str(e)) from e

            candidate_samples.append(parsed["candidates"])
            walks.append(parsed["walk"])
            evidences.append(parsed["evidence"])

    # Aggregate & mechanically deduplicate candidates
    seen_labels: set[str] = set()
    aggregated_candidates: list[dict] = []

    for sample in candidate_samples:
        for cand in sample:
            label = cand["label"].strip()
            c_dict = dict(cand)
            c_dict["label"] = label
            if label in existing_labels_set or label in seen_labels:
                c_dict["novel"] = False
            else:
                c_dict["novel"] = True
                seen_labels.add(label)

            if not any(existing["label"] == label for existing in aggregated_candidates):
                aggregated_candidates.append(c_dict)

    confidence = mechanical_confidence_alt_gate(candidate_samples)

    res = {
        "walks": walks,
        "evidences": evidences,
        "aggregated_candidates": aggregated_candidates,
        "confidence": confidence,
        "model": model,
        "n_samples": n_samples,
        "temperature": temperature,
        "situation": input_data.get("situation"),
    }
    if echo_notes:
        res["echo_notes"] = echo_notes
    if reporter and observations:
        res["observations"] = observations
    return res


def alt_gate_adversary(
    stage1_res: dict,
    *,
    call_api_fn: Callable | None = None,
    api_key: str | None = None,
    model: str = "deepseek-v4-flash",
    max_tokens: int = 2048,
    unified: bool = False,
    **_kwargs: Any,
) -> dict:
    """Stage 2: Adversarial Usage-Judge Review for LLM Alternative Gate.

    The adversary judges each candidate as expression (revival, legal) vs
    substitution (smuggling, illegal), grounded in the situation payload
    carried in ``stage1_res["situation"]`` (social-material structure + mirror
    angle when present). Flagged candidates get ``adversary_flag: true`` and
    halved weight.
    """
    if call_api_fn is None:
        call_api_fn = _load_ecc_call_api()
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("API key required: pass api_key or set DEEPSEEK_API_KEY")

    aggregated_candidates = stage1_res["aggregated_candidates"]
    walks = stage1_res["walks"]
    evidences = stage1_res["evidences"]
    confidence = stage1_res["confidence"]
    situation = stage1_res.get("situation")

    adv_input = {
        "candidate_labels": [c["label"] for c in aggregated_candidates],
        "candidate_tags": [c["concept_tags"] for c in aggregated_candidates],
    }
    if isinstance(situation, dict):
        adv_input["situation"] = situation
    base_adv_prompt = json.dumps(adv_input, ensure_ascii=False)
    adv_prompt = f"{base_adv_prompt}\n[task:adversary]" if (unified or "observations" in stage1_res) else base_adv_prompt
    sys_prompt = GATE_SYSTEM_PROMPT_UNIFIED if (unified or "observations" in stage1_res) else ADVERSARY_SYSTEM_PROMPT_V1

    adv_parsed = None
    for attempt in range(2):
        adv_raw = call_api_fn(
            adv_prompt,
            api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            system_message=sys_prompt,
            response_format={"type": "json_object"},
            thinking=False,
        )
        try:
            if unified or "observations" in stage1_res:
                for lk in ROUTING_LEAK_KEYWORDS:
                    if lk in adv_raw:
                        raise AltGateContractError(f"ROUTING_LEAK_DETECTED: response contains prompt leak keyword '{lk}'")
            candidate_adv = _parse_json_loose(adv_raw)
            if unified or "observations" in stage1_res:
                leak_err = check_routing_leak_or_schema(adv_raw, candidate_adv, "adversary")
                if leak_err:
                    raise AltGateContractError(leak_err)
            adv_parsed = candidate_adv
            break
        except Exception as e:
            if attempt == 0:
                continue
            if isinstance(e, AltGateContractError):
                raise e
            raise AltGateContractError(str(e)) from e

    flagged_set = set(adv_parsed.get("flagged_labels", []))

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

    final_walk = walks[0] if walks else {}
    final_evidence = list(dict.fromkeys(sum(evidences, [])))

    result = {
        "walk": final_walk,
        "candidates": final_candidates,
        "evidence": final_evidence,
        "confidence": confidence,
        "generated_by": {
            "gate": "synth.alt_gate",
            "model": stage1_res.get("model", model),
            "n_samples": stage1_res.get("n_samples", 3),
            "temperature": stage1_res.get("temperature", 0.6),
            "called_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    _write_alt_gate_state_log(result, observations=stage1_res.get("observations"))
    return result


def alt_gate(
    input_data: dict,
    *,
    call_api_fn: Callable | None = None,
    api_key: str | None = None,
    model: str = "deepseek-v4-flash",
    temperature: float = 0.6,
    n_samples: int = 3,
    max_tokens: int = 2048,
    unified: bool = False,
    reporter: bool = False,
    **kwargs: Any,
) -> dict:
    """LLM Alternative Generation Gate with H schema + 2-Stage Adversarial Call.

    Args:
        input_data: Input dict containing 'state_vector', 'thread_history',
            'local_texture', and optional 'existing_labels' / 'situation'
            (the situation payload grounds the adversary's usage judgment).
        call_api_fn: Callable for API invocation (injected for testing/mocking).
        api_key: DeepSeek API key. Read from DEEPSEEK_API_KEY env var if None.
        model: Model name. Default 'deepseek-v4-flash'.
        temperature: Sampling temperature (default 0.6).
        n_samples: Number of sampling calls for generation (default 3).
        max_tokens: Token limit for completion.
        unified: If True, use GATE_SYSTEM_PROMPT_UNIFIED with [task:X] tags.
        reporter: If True, use two-stage reporter generation path (observe -> compress).

    Returns:
        Dict containing walk, candidates, evidence, confidence, generated_by provenance.
    """
    stage1_res = alt_gate_generate(
        input_data,
        call_api_fn=call_api_fn,
        api_key=api_key,
        model=model,
        temperature=temperature,
        n_samples=n_samples,
        max_tokens=max_tokens,
        unified=unified,
        reporter=reporter,
        **kwargs,
    )
    return alt_gate_adversary(
        stage1_res,
        call_api_fn=call_api_fn,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        unified=unified,
        **kwargs,
    )


alt_gate.stage1 = alt_gate_generate
alt_gate.stage2 = alt_gate_adversary


# --- Version B: Historical Accident Enumeration ([task:enumerate]) ---

ACCIDENT_DOMAINS = {"政治", "經濟", "社會", "軍事", "自然"}
ACCIDENT_SOURCES = {"inherited", "emergent"}
ACCIDENT_USAGES = {"expression", "substitution"}
ACCIDENT_GROUNDINGS = {"文獻偶發", "合理推測", "弱支持"}


def validate_accident(acc: dict, index: int) -> None:
    """Validate one accident entry from [task:enumerate] output."""
    if not isinstance(acc, dict):
        raise AltGateContractError(f"accident[{index}] must be dict, got {type(acc).__name__}")
    if not isinstance(acc.get("pattern"), str) or not acc["pattern"].strip():
        raise AltGateContractError(f"accident[{index}].pattern must be non-empty str")
    if len(acc["pattern"]) > 40:
        raise AltGateContractError(f"accident[{index}].pattern exceeds length limit (<=40)")
    # F4: quarantine — pattern must not be an empty shell refusal / placeholder
    if _has_shell_refusal_or_placeholder(acc["pattern"]):
        raise AltGateContractError(
            f"accident[{index}].pattern contains placeholder or shell refusal"
        )
    if not isinstance(acc.get("instances"), list):
        raise AltGateContractError(f"accident[{index}].instances must be list")
    # F3: burst mode requires >= 1 concrete instance
    if not acc["instances"]:
        raise AltGateContractError(
            f"accident[{index}].instances must not be empty (burst mode requires >= 1 concrete instance)"
        )
    for j, inst in enumerate(acc["instances"]):
        if not isinstance(inst, str):
            raise AltGateContractError(f"accident[{index}].instances[{j}] must be str")
        # F4: quarantine each instance string
        if _has_shell_refusal_or_placeholder(inst):
            raise AltGateContractError(
                f"accident[{index}].instances[{j}] contains placeholder or shell refusal"
            )
    dom = acc.get("domain", "")
    if dom not in ACCIDENT_DOMAINS:
        raise AltGateContractError(
            f"accident[{index}].domain must be one of {sorted(ACCIDENT_DOMAINS)}, got {dom!r}"
        )
    src = acc.get("source", "")
    if src not in ACCIDENT_SOURCES:
        raise AltGateContractError(
            f"accident[{index}].source must be one of {sorted(ACCIDENT_SOURCES)}, got {src!r}"
        )
    usage = acc.get("usage", "")
    if usage not in ACCIDENT_USAGES:
        raise AltGateContractError(
            f"accident[{index}].usage must be one of {sorted(ACCIDENT_USAGES)}, got {usage!r}"
        )
    grd = acc.get("grounding", "")
    if grd not in ACCIDENT_GROUNDINGS:
        raise AltGateContractError(
            f"accident[{index}].grounding must be one of {sorted(ACCIDENT_GROUNDINGS)}, got {grd!r}"
        )
    if not isinstance(acc.get("world_development"), str) or not acc["world_development"].strip():
        raise AltGateContractError(f"accident[{index}].world_development must be non-empty str")
    # F4: quarantine world_development
    if _has_shell_refusal_or_placeholder(acc["world_development"]):
        raise AltGateContractError(
            f"accident[{index}].world_development contains placeholder or shell refusal"
        )


# ---------------------------------------------------------------------------
# Mode A → Mode B 依賴（PLAN-23 §12.13）
# ---------------------------------------------------------------------------

#: 主導轉移判定的最低速率（> 此值才進 dominant_transitions 摘要）。
MODE_A_DOMINANT_TRANSITION_MIN = 0.05


def record_mode_a_history(path: str | os.PathLike, entry: dict) -> None:
    """記錄一筆 Mode A 歷史（§12.13 ①，純機械部分）。

    Mode A = Markov 推演 + 頻譜累積——「過去的頻譜 = 發展歷史」。本函式 append
    一筆 entry 到 JSONL（open/append/close 單行，仿 ``_verify._append_jsonl``）——
    純資料累積、零 LLM。建議欄位：
    ``{"ts", "mode": "markov"|"spectrum"|"llm_annotation", "role_sequence": [...],
    "rate_matrix": {...}, "labels": [...], "spectrum": {...}, "source": str}``。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_mode_a_history(path: str | os.PathLike | None) -> list[dict]:
    """讀取 Mode A 歷史（§12.13 ①）——JSONL，跳過損壞行；檔案不存在回 []。"""
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _as_4x4(rate_matrix: Any):
    """規範 rate_matrix 為 4x4 numpy；形狀非法回 None（世界無關 fallback）。"""
    if isinstance(rate_matrix, dict):
        rate_matrix = rate_matrix.get("matrix", rate_matrix)
    try:
        m = np.asarray(rate_matrix, dtype=np.float64)
    except Exception:
        return None
    if m.shape != (len(ROLES), len(ROLES)):
        return None
    return m


def _dominant_transitions(
    rate_matrix: Any, *, min_rate: float = MODE_A_DOMINANT_TRANSITION_MIN
) -> list[dict]:
    """從速率矩陣抽主導轉移（> min_rate 的非零格），依強度降冪。

    純機械（§12.13 ② 後驗初始化的一部分）——「過去頻譜代表的發展歷史」的
    Markov 側摘要。無 rate_matrix → []。
    """
    m = _as_4x4(rate_matrix)
    if m is None:
        return []
    out: list[dict] = []
    for i in range(len(ROLES)):
        for j in range(len(ROLES)):
            v = float(m[i, j])
            if v > min_rate:
                out.append({"from": ROLES[i], "to": ROLES[j], "rate": round(v, 4)})
    out.sort(key=lambda d: d["rate"], reverse=True)
    return out


def _accumulated_labels(history: list[dict]) -> list[str]:
    """歷史中累積的標籤（去重、保序）——spectrum/ensemble 側摘要。"""
    seen: list[str] = []
    for entry in history:
        for lab in entry.get("labels", []) or []:
            if isinstance(lab, str) and lab.strip() and lab not in seen:
                seen.append(lab.strip())
    return seen


def mode_a_posterior_init(mode_a_history: list[dict]) -> dict:
    """Mode B 後驗初始化（§12.13 ②，純機械部分）。

    從 Mode A 歷史萃取「過去的頻譜 = 發展歷史」的機械摘要，作為枚舉前初始化：
    - ``n_entries``：歷史筆數。
    - ``rate_matrix`` / ``dominant_transitions``：Markov 側（最後一筆有效速率矩陣
      + 主導轉移）。
    - ``role_statistics``：歷史角色序列的轉移計數（``count_transitions``）。
    - ``accumulated_labels``：累積標籤（spectrum/ensemble 側）。
    - ``spectrum_snapshot``：最後一筆 spectrum 摘要（若有）。
    - ``annotation``：機械註記——LLM 結構語義標註待 prompt（PROMPT-DEPENDENT），
      見 ``_annotate_structural_semantics``。

    零 LLM、零 API。Mode B 枚舉前以本摘要為初始化，不再空手起跳（§12.13）。
    """
    if not mode_a_history:
        return {"n_entries": 0, "source": "mode_a_history", "annotation": "no_history"}

    last_rates: Any = None
    last_spectrum: Any = None
    role_sequences: list[list[str]] = []
    for entry in mode_a_history:
        if "rate_matrix" in entry:
            last_rates = entry["rate_matrix"]
        if "spectrum" in entry:
            last_spectrum = entry["spectrum"]
        seq = entry.get("role_sequence")
        if isinstance(seq, list) and seq and all(isinstance(r, str) for r in seq):
            role_sequences.append(seq)

    counts, anomalies = (
        count_transitions(role_sequences) if role_sequences else (None, [])
    )
    m = _as_4x4(last_rates)
    rate_summary: dict | None = None
    if m is not None:
        rate_summary = {
            "matrix": [[float(v) for v in row] for row in m.tolist()],
            "dominant_transitions": _dominant_transitions(m),
        }

    return {
        "n_entries": len(mode_a_history),
        "source": "mode_a_history",
        "rate_matrix": rate_summary,
        "role_statistics": {
            "n_sequences": len(role_sequences),
            "transition_counts": counts.tolist() if counts is not None else None,
            "anomalies": anomalies,
        },
        "accumulated_labels": _accumulated_labels(mode_a_history),
        "spectrum_snapshot": last_spectrum,
        "annotation": (
            "mechanical posterior init（§12.13 ② 純機械）；"
            "LLM 結構語義標註待 prompt 設計（PROMPT-DEPENDENT，"
            "見 _annotate_structural_semantics）"
        ),
    }


def _annotate_structural_semantics(posterior: dict) -> dict:
    """LLM 標註結構語義（§12.13 ①）——🔴 **PROMPT-DEPENDENT**。

    需要人類/主 agent 設計「結構語義標註」prompt（把 Mode A 的頻譜摘要轉成
    結構語義註記）才能實作。目前回傳機械 posterior 不變——Mode B 仍可機械
    初始化；語義標註待 prompt 定稿後於此接上。**本輪禁止自行改 prompt。**
    """
    return posterior


def alt_gate_enumerate(
    input_data: dict,
    *,
    call_api_fn: Callable | None = None,
    api_key: str | None = None,
    model: str = "deepseek-v4-flash",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    mode_a_history: list[dict] | None = None,
    mode_a_log_path: str | None = None,
    **_kwargs: Any,
) -> dict:
    """Version B: Historical Accident Enumeration Gate ([task:enumerate]).

    Exhaustively enumerates the accidents possible at the given moment from the
    situation's social-material structure. Triggered where the spectrum fails
    (criticality / residual). Output schema:
        {"accidents": [{"pattern", "instances", "domain", "source", "usage",
                        "grounding", "world_development"}]}

    §12.13（Mode A → Mode B 依賴）：Mode B **不再空手起跳**——先展開 Mode A
    （Markov 推演 + 頻譜累積 → 發展歷史），枚舉前以 Mode A 成果為**後驗初始化**：
    - ``mode_a_history``：Mode A 歷史條目清單（直接傳入）。
    - ``mode_a_log_path``：或指定 JSONL 路徑（``record_mode_a_history`` 寫入）。
    初始化摘要以資料欄位 ``input_data["mode_a_history"]`` 注入 payload（**資料層**，
    不改任何 prompt 措辭）；LLM 結構語義標註見 ``_annotate_structural_semantics``
    （PROMPT-DEPENDENT，本輪凍結）。回傳含 ``mode_a_posterior`` 供追溯。

    Note:
        Not suitable as a ``gate_fn`` for ``run_ensemble``: this gate returns
        ``{"accidents": [...]}`` instead of ``{"candidates": [...]}``, so wiring
        it in as an ensemble gate would silently yield no candidates.
    """
    if call_api_fn is None:
        call_api_fn = _load_ecc_call_api()
    if api_key is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("API key required: pass api_key or set DEEPSEEK_API_KEY")

    # §12.13：Mode B 後驗初始化（純機械）——注入 payload 資料層，不動 prompt 字串。
    payload = dict(input_data)
    posterior: dict | None = None
    history = mode_a_history
    if history is None and mode_a_log_path is not None:
        history = load_mode_a_history(mode_a_log_path)
    if history:
        posterior = _annotate_structural_semantics(mode_a_posterior_init(history))
        payload["mode_a_history"] = posterior

    sys_prompt = VERSION_B_SYSTEM_PROMPT
    user_prompt = (
        f"{_build_alt_gate_prompt(payload)}\n"
        "[task:enumerate] 窮盡列舉此刻可能發生的意外。輸出 JSON："
        '{"accidents": [{"pattern": "爆發模式", "instances": ["特定實例1", "特定實例2"], '
        '"domain": "政治|經濟|社會|軍事|自然", "source": "inherited|emergent", '
        '"usage": "expression|substitution", '
        '"grounding": "文獻偶發|合理推測|弱支持", "world_development": "這個意外導向的世界發展"}]}'
    )
    parsed = None
    accidents: list = []
    for attempt in range(2):
        curr_temp = temperature + (0.15 if attempt == 1 else 0.0)
        raw = call_api_fn(
            user_prompt,
            api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=curr_temp,
            system_message=sys_prompt,
            response_format={"type": "json_object"},
            thinking=False,
        )
        try:
            # F6: the routing-leak scan deliberately runs before parsing — leak
            # detection keeps precedence over parse errors (a leaked prompt is
            # the harder failure, so it surfaces first).
            for lk in ROUTING_LEAK_KEYWORDS:
                if lk in raw:
                    raise AltGateContractError(
                        f"ROUTING_LEAK_DETECTED: response contains prompt leak keyword '{lk}'"
                    )
            # F1: normalize AnchorContractError (or any other Exception) raised
            # by _parse_json_loose into AltGateContractError, mirroring the
            # except pattern of alt_gate_generate.
            try:
                parsed = _parse_json_loose(raw)
            except Exception as e:
                raise AltGateContractError(str(e)) from e
            err = check_routing_leak_or_schema(raw, parsed, "enumerate")
            if err:
                raise AltGateContractError(err)
            accidents = parsed.get("accidents", [])
            if not isinstance(accidents, list) or not accidents:
                raise AltGateContractError("task [task:enumerate] returned empty accidents list")
            for i, acc in enumerate(accidents):
                validate_accident(acc, i)
            break
        except AltGateContractError as e:
            # F2: model-output failures (parse / schema / contract / validation)
            # are retryable — retry once with a bumped temperature, then surface.
            if attempt == 0:
                continue
            raise e
        except Exception as e:
            # F1: non-contract exceptions are not retryable — normalize and
            # propagate immediately.
            raise AltGateContractError(str(e)) from e
    return {
        "accidents": accidents,
        "mode_a_posterior": posterior,  # §12.13：Mode B 後驗初始化（無 Mode A 時 None）
        "generated_by": {
            "gate": "alt_gate_enumerate",
            "model": model,
            "temperature": temperature,
            "n_accidents": len(accidents),
            "called_at": datetime.now(timezone.utc).isoformat(),
        },
    }


