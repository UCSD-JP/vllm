#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline replay for Elastic KV JSONL trace.

Reads a trace file (produced by VLLM_ELASTIC_KV_TRACE=/path/to/trace.jsonl)
and re-evaluates decisions under different config parameter grids.

Usage:
    python scripts/replay_elastic_kv.py trace.jsonl [--sweep]

With --sweep, replays each record under a grid of c_reload_ms and h_cap
values to show how decisions would change.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm.v1.core.prefix_protect import PrefixProtectionConfig, ProtectionDecision


def replay_record(record: dict, cfg: PrefixProtectionConfig) -> str:
    """Re-evaluate a single trace record with the given config."""
    h_eff = record.get('h_eff', 1)
    groups_est = record.get('groups_est', 1)
    touched = record.get('touched_cached_blocks', 0)
    preempt_tokens = record.get('preempt_tokens', 0)
    n_decode = record.get('n_decode', 0)
    n_prefill = record.get('n_prefill', 0)
    can_fully_protect = record.get('can_fully_protect', False)

    decision = cfg.decide(
        h_eff=h_eff,
        groups_to_evict=groups_est,
        touched_cached_blocks=touched,
        preempt_computed_tokens=preempt_tokens,
        n_decode=n_decode,
        n_prefill=n_prefill,
        can_fully_protect=can_fully_protect,
    )
    return decision.value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace_file', help='Path to JSONL trace file')
    parser.add_argument('--sweep', action='store_true',
                        help='Sweep over parameter grid')
    args = parser.parse_args()

    records = []
    with open(args.trace_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found in trace file.")
        return

    print(f"Loaded {len(records)} records from {args.trace_file}")

    # Default config: use runtime config from first trace record if available
    first_cfg = records[0].get('cfg', {})
    if first_cfg:
        cfg = PrefixProtectionConfig(
            local_num_experts=first_cfg.get('E', 512),
            group_size=first_cfg.get('G', 2),
            top_k=first_cfg.get('top_k', 10),
            c_reload_ms=first_cfg.get('c_reload_ms', 0.63),
            block_size=first_cfg.get('block_size', 16),
            h_cap=first_cfg.get('h_cap', 64),
            num_layers=first_cfg.get('num_layers', 1),
            l_sync_prefill=first_cfg.get('l_sync_prefill', 2.0),
            t_recompute_ms_per_token=first_cfg.get('t_recompute', 0.260),
            t_prefill_tok_us=first_cfg.get('t_prefill_tok_us', 15.0),
            p_reuse=first_cfg.get('p_reuse', 1.0),
        )
        print(f"Using runtime config from trace: {first_cfg}")
    else:
        cfg = PrefixProtectionConfig()
        print("No runtime config in trace, using defaults")
    print("\n--- Default config replay ---")
    decision_counts: dict[str, int] = {}
    for rec in records:
        d = replay_record(rec, cfg)
        decision_counts[d] = decision_counts.get(d, 0) + 1
        original = rec.get('decision', '')
        if d != original:
            print(f"  ts={rec.get('ts', 0):.3f}: {original} → {d}")

    print(f"\nDecision distribution: {decision_counts}")

    if args.sweep:
        print("\n--- Parameter sweep ---")
        c_reload_values = [0.1, 0.3, 0.63, 1.0, 2.0]
        h_cap_values = [16, 32, 64, 128]

        for c_reload in c_reload_values:
            for h_cap in h_cap_values:
                # Inherit runtime config from trace, override sweep params
                base = dict(
                    local_num_experts=first_cfg.get('E', 512),
                    group_size=first_cfg.get('G', 2),
                    top_k=first_cfg.get('top_k', 10),
                    num_layers=first_cfg.get('num_layers', 1),
                    l_sync_prefill=first_cfg.get('l_sync_prefill', 2.0),
                    t_recompute_ms_per_token=first_cfg.get(
                        't_recompute', 0.260),
                    t_prefill_tok_us=first_cfg.get('t_prefill_tok_us', 15.0),
                    p_reuse=first_cfg.get('p_reuse', 1.0),
                    block_size=first_cfg.get('block_size', 16),
                ) if first_cfg else {}
                cfg = PrefixProtectionConfig(
                    c_reload_ms=c_reload, h_cap=h_cap, **base)
                counts: dict[str, int] = {}
                for rec in records:
                    d = replay_record(rec, cfg)
                    counts[d] = counts.get(d, 0) + 1
                protect_pct = (
                    counts.get('protect_and_expand', 0) / len(records) * 100)
                reclaim_pct = (
                    counts.get('reclaim_cached', 0) / len(records) * 100)
                print(f"  c_reload={c_reload:.2f} h_cap={h_cap:3d}: "
                      f"protect={protect_pct:5.1f}% reclaim={reclaim_pct:5.1f}% "
                      f"counts={counts}")


if __name__ == '__main__':
    main()
