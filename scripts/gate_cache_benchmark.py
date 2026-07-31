#!/usr/bin/env python3
"""synth.gate_cache_benchmark — KV-cache Hit Rate & Sharing Benchmark.

Computes theoretical prefix sharing ratio in mock mode, and executes small-scale
live API benchmark (--live) measuring actual DeepSeek prompt_cache_hit_tokens vs miss_tokens.
Outputs data/gate_cache_benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectrum_os.synth.gate_prompts import GATE_SYSTEM_PROMPT_UNIFIED
from spectrum_os.synth.rate_gate import RATE_GATE_SYSTEM_PROMPT_V1
from spectrum_os.synth.alt_gate import (
    ALT_GATE_SYSTEM_PROMPT_V1,
    ADVERSARY_SYSTEM_PROMPT_V1,
    alt_gate,
)
from spectrum_os.synth.ensemble import run_ensemble


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (using ~3 bytes per token for CJK/English mix)."""
    return max(1, len(text.encode("utf-8")) // 3)


def run_dry_run_benchmark() -> dict[str, Any]:
    """Compute theoretical prefix sharing ratio in mock/dry-run mode."""
    print("=== KV-Cache Prefix Sharing Ratio (Dry-Run Benchmark) ===")

    # System prompt token estimates
    rate_sys_tokens = estimate_tokens(RATE_GATE_SYSTEM_PROMPT_V1)
    alt_sys_tokens = estimate_tokens(ALT_GATE_SYSTEM_PROMPT_V1)
    adv_sys_tokens = estimate_tokens(ADVERSARY_SYSTEM_PROMPT_V1)
    unified_sys_tokens = estimate_tokens(GATE_SYSTEM_PROMPT_UNIFIED)

    sample_user_msg = '{"state_vector": {"role": "crisis"}, "thread_history": ["crisis"], "existing_labels": []}'
    user_msg_tokens = estimate_tokens(sample_user_msg)

    # Legacy mode: 3 distinct system prompts -> 0% system prompt cross-gate sharing
    legacy_total_prompt_tokens = (
        (rate_sys_tokens + user_msg_tokens)
        + (alt_sys_tokens + user_msg_tokens)
        + (adv_sys_tokens + user_msg_tokens)
    )

    # Unified mode: Single shared system prompt across all 3 gates
    unified_prefix_sharing_ratio = round(unified_sys_tokens / (unified_sys_tokens + user_msg_tokens), 4)

    print(f"Legacy System Prompt Tokens: rate={rate_sys_tokens}, alt={alt_sys_tokens}, adv={adv_sys_tokens}")
    print(f"Unified Shared System Prompt Tokens: {unified_sys_tokens}")
    print(f"User Message Tokens (approx): {user_msg_tokens}")
    print(f"Unified Mode System Prefix Sharing Ratio: {unified_prefix_sharing_ratio * 100:.2f}%\n")

    return {
        "mode": "dry-run",
        "theoretical_sharing": {
            "legacy_sys_tokens": {
                "rate": rate_sys_tokens,
                "alt": alt_sys_tokens,
                "adv": adv_sys_tokens,
            },
            "unified_sys_tokens": unified_sys_tokens,
            "user_msg_tokens": user_msg_tokens,
            "unified_prefix_sharing_ratio": unified_prefix_sharing_ratio,
        },
    }


def run_live_benchmark() -> dict[str, Any]:
    """Execute live API benchmark measuring prompt_cache_hit_tokens."""
    print("=== KV-Cache Live Hit Rate Benchmark (--live) ===")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY environment variable is required for live benchmark.")
        sys.exit(1)

    # Import call_api from ECC api_utils
    from spectrum_os.synth.anchors import _load_ecc_call_api
    ecc_call_api = _load_ecc_call_api()

    # Scale: 2 branches x 2 gate points (horizon 12, gate_every 4) x n_samples=2
    # Live estimation equation:
    # 2 branches * 2 gate points * 2 samples (generate) + 4 points (adversary) = 12 API calls per mode
    # Total live API calls = 24 API calls (12 legacy + 12 unified)
    n_branches = 2
    horizon = 12
    gate_every = 4
    n_samples = 2

    estimated_calls_per_mode = n_branches * (horizon // gate_every) * n_samples + n_branches * (horizon // gate_every)
    total_estimated_calls = estimated_calls_per_mode * 2

    print("--- Estimated API Calls & Cost Equation ---")
    print(f"Equation per mode: {n_branches} branches * 2 points * {n_samples} samples (generate) + 4 points (adversary) = {estimated_calls_per_mode} calls")
    print(f"Total Live API Calls: {estimated_calls_per_mode} (legacy) + {estimated_calls_per_mode} (unified) = {total_estimated_calls} calls")
    print(f"Estimated Cost: ~{total_estimated_calls} calls * $0.0002 = ${total_estimated_calls * 0.0002:.4f} USD\n")

    modes_stats = {}

    for unified_flag, mode_name in [(False, "legacy"), (True, "unified")]:
        print(f"Running [{mode_name.upper()} MODE] (unified={unified_flag})...")
        usage_records = []

        def tracking_call_api_fn(prompt, key, model="deepseek-v4-flash", max_tokens=2048, temperature=0.6, system_message="", response_format=None, thinking=False):
            try:
                res, usage = ecc_call_api(
                    prompt, key, model=model, max_tokens=max_tokens, temperature=temperature,
                    system_message=system_message, response_format=response_format, thinking=thinking,
                    return_usage=True
                )
                usage_records.append(usage)
                hit = usage.get("prompt_cache_hit_tokens", 0)
                miss = usage.get("prompt_cache_miss_tokens", 0)
                print(f"  Call #{len(usage_records)}: prompt_tokens={usage.get('prompt_tokens', 0)} (Hit: {hit}, Miss: {miss})")
                return res
            except TypeError:
                res = ecc_call_api(
                    prompt, key, model=model, max_tokens=max_tokens, temperature=temperature,
                    system_message=system_message, response_format=response_format, thinking=thinking
                )
                return res

        run_ensemble(
            "crisis",
            n_branches=n_branches,
            horizon=horizon,
            gate_fn=alt_gate,
            gate_every=gate_every,
            max_gates=5,
            seed=42,
            call_api_fn=tracking_call_api_fn,
            api_key=api_key,
            n_samples=n_samples,
            unified=unified_flag,
        )

        total_hits = sum(u.get("prompt_cache_hit_tokens", 0) for u in usage_records)
        total_misses = sum(u.get("prompt_cache_miss_tokens", 0) for u in usage_records)
        total_prompt = sum(u.get("prompt_tokens", 0) for u in usage_records)
        hit_rate = round(total_hits / total_prompt, 4) if total_prompt > 0 else 0.0

        print(f"[{mode_name.upper()} RESULT] Calls: {len(usage_records)}, Total Prompt Tokens: {total_prompt}, Cache Hits: {total_hits}, Misses: {total_misses}, Hit Rate: {hit_rate * 100:.2f}%\n")

        modes_stats[mode_name] = {
            "total_calls": len(usage_records),
            "total_prompt_tokens": total_prompt,
            "cache_hit_tokens": total_hits,
            "cache_miss_tokens": total_misses,
            "cache_hit_rate": hit_rate,
            "usage_records": usage_records,
        }

    dry_run_data = run_dry_run_benchmark()
    return {
        "mode": "live",
        "theoretical_sharing": dry_run_data["theoretical_sharing"],
        "live_benchmark": modes_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="synth gate KV-cache benchmark")
    parser.add_argument("--live", action="store_true", help="Run live API cache benchmark")
    args = parser.parse_args()

    if args.live:
        benchmark_results = run_live_benchmark()
    else:
        benchmark_results = run_dry_run_benchmark()

    out_file = REPO_ROOT / "data" / "gate_cache_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

    print(f"Benchmark report saved to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
