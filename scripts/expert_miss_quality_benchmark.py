#!/usr/bin/env python3
"""Expert Cache Miss Quality Benchmark — KL/Flip/Overlap Measurement

Measures the quality impact of expert cache misses by comparing logprobs
from normal vs miss-injected inference runs.

Usage:
    # On Paladin (with model weights available):
    export HF_HOME=/mnt/raid0_ssd/huggingface
    export TMPDIR=/mnt/raid0_ssd/jinpyo/tmp

    # Run all conditions:
    python expert_miss_quality_benchmark.py --output_dir /tmp/quality_eval

    # Run a single condition:
    python expert_miss_quality_benchmark.py --miss_rate 0.02 --miss_layers all

    # Use custom prompts:
    python expert_miss_quality_benchmark.py --prompts_file prompts.json

Requirements:
    - vLLM with expert offload v2 + miss injection support
    - enforce_eager=True (avoid CUDA graph complexity)
    - TP=2 for Qwen3-Next-80B-A3B-Instruct
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── Default prompts (coding-style, similar to SWE-bench) ──────────────
DEFAULT_PROMPTS = [
    "Write a Python function that implements binary search on a sorted list. The function should return the index of the target element, or -1 if not found.",
    "Explain how to implement a LRU cache in Python using OrderedDict. Provide a complete working example with get and put methods.",
    "Write a Python class that implements a min-heap data structure with insert, extract_min, and peek operations.",
    "Implement a function to find the longest common subsequence of two strings using dynamic programming.",
    "Write a Python decorator that retries a function up to 3 times with exponential backoff on exceptions.",
    "Implement a thread-safe producer-consumer queue in Python using threading primitives.",
    "Write a Python function that serializes a binary tree to a string and deserializes it back.",
    "Implement Dijkstra's shortest path algorithm for a weighted directed graph in Python.",
    "Write a Python function to detect cycles in a directed graph using DFS with coloring.",
    "Implement a rate limiter class using the token bucket algorithm in Python.",
    "Write a Python function that merges k sorted lists into a single sorted list efficiently.",
    "Implement a trie data structure in Python with insert, search, and startsWith methods.",
    "Write a Python function to find all permutations of a string without using itertools.",
    "Implement an LFU (Least Frequently Used) cache with O(1) get and put operations.",
    "Write a Python function that evaluates a mathematical expression string with +, -, *, / and parentheses.",
    "Implement a concurrent web scraper in Python using asyncio and aiohttp.",
    "Write a Python class implementing a skip list with insert, search, and delete operations.",
    "Implement a function to find the median of two sorted arrays in O(log(min(m,n))) time.",
    "Write a Python function that compresses a string using run-length encoding.",
    "Implement a basic regex engine in Python that supports '.', '*', and literal characters.",
    "Write a function to convert a nested dictionary to a flat dictionary with dot-separated keys.",
    "Implement a Python function for topological sorting of a DAG using Kahn's algorithm.",
    "Write a concurrent file downloader in Python that downloads multiple URLs in parallel.",
    "Implement a Python function to find the k-th largest element in an unsorted array without sorting.",
    "Write a Python class that implements a balanced BST (AVL tree) with rotations.",
    "Implement a function to solve the N-Queens problem and return all valid board configurations.",
    "Write a Python function that finds all bridges in an undirected graph.",
    "Implement a simple key-value store in Python with TTL (time-to-live) expiration support.",
    "Write a Python function that implements the KMP string matching algorithm.",
    "Implement a function to find the strongly connected components of a directed graph using Tarjan's algorithm.",
    "Write a Python class implementing a segment tree with range sum queries and point updates.",
    "Implement a Python function to solve a Sudoku puzzle using backtracking.",
    "Write a function that performs matrix multiplication without using numpy.",
    "Implement a consistent hashing ring in Python for distributed cache routing.",
    "Write a Python function to find the minimum window substring containing all characters of a pattern.",
    "Implement a Python class for a disjoint set (union-find) with path compression and union by rank.",
    "Write a function to generate all valid combinations of n pairs of parentheses.",
    "Implement a Python class for a bloom filter with configurable false positive rate.",
    "Write a Python function that implements the A* pathfinding algorithm on a 2D grid.",
    "Implement a function to find the longest palindromic substring using Manacher's algorithm.",
    "Write a Python class implementing a priority queue with decrease-key operation.",
    "Implement a function to count the number of inversions in an array using merge sort.",
    "Write a Python function that implements depth-limited search with iterative deepening.",
    "Implement a lock-free stack in Python using compare-and-swap semantics.",
    "Write a function to find the maximum flow in a network using the Ford-Fulkerson algorithm.",
    "Implement a Python function for reservoir sampling to select k items from a stream of unknown length.",
    "Write a Python class implementing a red-black tree with insert and delete operations.",
    "Implement a function to find the shortest path in a maze using BFS with wall-breaking ability.",
    "Write a Python function that converts an infix expression to postfix notation using the shunting-yard algorithm.",
    "Implement a Python class for a persistent data structure (immutable list with structural sharing).",
]


@dataclass
class QualityResult:
    """Results from comparing normal vs miss-injected logprobs."""
    miss_rate: float
    miss_layers: str
    num_prompts: int
    num_tokens_compared: int
    top1_flip_rate: float
    top5_overlap_mean: float
    approx_kl_mean: float
    approx_kl_std: float
    approx_kl_max: float
    top1_flip_by_position: list = field(default_factory=list)
    wall_time_s: float = 0.0


def extract_logprobs_from_output(output):
    """Extract per-token logprobs from a vLLM RequestOutput."""
    tokens = []
    for completion in output.outputs:
        if completion.logprobs is None:
            continue
        for step_logprobs in completion.logprobs:
            # step_logprobs is a dict: {token_id: Logprob}
            tokens.append(step_logprobs)
    return tokens


def compute_quality_metrics(baseline_outputs, injected_outputs,
                            num_logprobs: int = 50) -> QualityResult:
    """Compare logprobs between baseline and miss-injected runs."""
    top1_flips = 0
    top5_overlaps = []
    kl_divs = []
    total_tokens = 0
    flip_by_position = {}

    for b_out, i_out in zip(baseline_outputs, injected_outputs):
        b_tokens = extract_logprobs_from_output(b_out)
        i_tokens = extract_logprobs_from_output(i_out)

        n_compare = min(len(b_tokens), len(i_tokens))
        for pos in range(n_compare):
            b_step = b_tokens[pos]
            i_step = i_tokens[pos]

            if not b_step or not i_step:
                continue

            total_tokens += 1

            # Top-1 flip: does the most likely token change?
            b_top1 = max(b_step, key=lambda tid: b_step[tid].logprob)
            i_top1 = max(i_step, key=lambda tid: i_step[tid].logprob)
            if b_top1 != i_top1:
                top1_flips += 1
                flip_by_position[pos] = flip_by_position.get(pos, 0) + 1

            # Top-5 overlap: how much do the top-5 sets overlap?
            b_top5 = set(sorted(b_step, key=lambda tid: b_step[tid].logprob,
                                reverse=True)[:5])
            i_top5 = set(sorted(i_step, key=lambda tid: i_step[tid].logprob,
                                reverse=True)[:5])
            overlap = len(b_top5 & i_top5) / 5.0
            top5_overlaps.append(overlap)

            # Approximate KL divergence over shared tokens
            # KL(P_baseline || P_injected) = sum(p_b * log(p_b / p_i))
            shared_tokens = set(b_step.keys()) & set(i_step.keys())
            if len(shared_tokens) > 1:
                kl = 0.0
                for tid in shared_tokens:
                    p_b = math.exp(b_step[tid].logprob)
                    p_i = math.exp(i_step[tid].logprob)
                    if p_b > 1e-10 and p_i > 1e-10:
                        kl += p_b * math.log(p_b / p_i)
                kl_divs.append(max(kl, 0.0))  # clip negative numerical noise

    if total_tokens == 0:
        return QualityResult(
            miss_rate=0, miss_layers="", num_prompts=0,
            num_tokens_compared=0, top1_flip_rate=0,
            top5_overlap_mean=1.0, approx_kl_mean=0,
            approx_kl_std=0, approx_kl_max=0)

    kl_arr = np.array(kl_divs) if kl_divs else np.array([0.0])

    # Flip rate by position (binned every 10 tokens)
    max_pos = max(flip_by_position.keys()) if flip_by_position else 0
    n_prompts = len(baseline_outputs)
    flip_by_pos_binned = []
    for bin_start in range(0, max_pos + 10, 10):
        bin_flips = sum(flip_by_position.get(p, 0)
                        for p in range(bin_start, bin_start + 10))
        flip_by_pos_binned.append(bin_flips / max(n_prompts, 1))

    return QualityResult(
        miss_rate=0,  # filled by caller
        miss_layers="",  # filled by caller
        num_prompts=len(baseline_outputs),
        num_tokens_compared=total_tokens,
        top1_flip_rate=top1_flips / total_tokens,
        top5_overlap_mean=float(np.mean(top5_overlaps)),
        approx_kl_mean=float(kl_arr.mean()),
        approx_kl_std=float(kl_arr.std()),
        approx_kl_max=float(kl_arr.max()),
        top1_flip_by_position=flip_by_pos_binned,
    )


def run_inference(model_name: str, prompts: list, tp_size: int,
                  max_tokens: int, num_logprobs: int,
                  miss_rate: float, miss_layers: Optional[str],
                  max_model_len: int = 4096):
    """Run inference with given miss injection config. Returns outputs."""
    # Set env vars BEFORE importing vllm
    os.environ["VLLM_EXPERT_OFFLOAD_ENABLE"] = "1"
    os.environ["VLLM_EXPERT_INJECT_MISS_RATE"] = str(miss_rate)
    if miss_layers and miss_layers != "all":
        os.environ["VLLM_EXPERT_INJECT_MISS_LAYERS"] = miss_layers
    elif "VLLM_EXPERT_INJECT_MISS_LAYERS" in os.environ:
        del os.environ["VLLM_EXPERT_INJECT_MISS_LAYERS"]

    # Import vllm here to pick up env vars
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        enforce_eager=True,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
    )

    params = SamplingParams(
        max_tokens=max_tokens,
        logprobs=num_logprobs,
        temperature=0.0,  # greedy for reproducibility
    )

    outputs = llm.generate(prompts, params)
    return outputs


def run_single_condition(model_name: str, prompts: list, tp_size: int,
                         max_tokens: int, num_logprobs: int,
                         miss_rate: float, miss_layers: Optional[str],
                         baseline_outputs, max_model_len: int = 4096,
                         ) -> QualityResult:
    """Run one miss injection condition and compare with baseline."""
    print(f"\n{'='*60}")
    print(f"Condition: miss_rate={miss_rate}, layers={miss_layers or 'all'}")
    print(f"{'='*60}")

    t0 = time.time()
    injected_outputs = run_inference(
        model_name, prompts, tp_size, max_tokens, num_logprobs,
        miss_rate, miss_layers, max_model_len)
    wall_time = time.time() - t0

    result = compute_quality_metrics(baseline_outputs, injected_outputs,
                                     num_logprobs)
    result.miss_rate = miss_rate
    result.miss_layers = miss_layers or "all"
    result.wall_time_s = wall_time

    print(f"  Top-1 flip rate:  {result.top1_flip_rate*100:.2f}%")
    print(f"  Top-5 overlap:    {result.top5_overlap_mean*100:.1f}%")
    print(f"  Approx KL mean:   {result.approx_kl_mean:.6f}")
    print(f"  Approx KL max:    {result.approx_kl_max:.6f}")
    print(f"  Tokens compared:  {result.num_tokens_compared}")
    print(f"  Wall time:        {wall_time:.1f}s")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Expert cache miss quality benchmark")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Tokens to generate per prompt")
    parser.add_argument("--num_logprobs", type=int, default=50,
                        help="Top-k logprobs to request")
    parser.add_argument("--num_prompts", type=int, default=50,
                        help="Number of prompts to use")
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="JSON file with list of prompt strings")
    parser.add_argument("--output_dir", type=str, default=None)

    # Single condition mode
    parser.add_argument("--miss_rate", type=float, default=None,
                        help="Run single condition with this miss rate")
    parser.add_argument("--miss_layers", type=str, default=None,
                        help="Comma-separated layer indices, or 'all'")

    args = parser.parse_args()

    # Load prompts
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
    else:
        prompts = DEFAULT_PROMPTS

    prompts = prompts[:args.num_prompts]
    print(f"Using {len(prompts)} prompts, {args.max_tokens} tokens each")
    print(f"Model: {args.model}, TP={args.tp_size}")

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(f"/tmp/quality_eval_{int(time.time())}")
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # ── Pass 1: Baseline (no injection) ────────────────────────
    print(f"\n{'='*60}")
    print("BASELINE: miss_rate=0 (no injection)")
    print(f"{'='*60}")
    t0 = time.time()
    baseline_outputs = run_inference(
        args.model, prompts, args.tp_size, args.max_tokens,
        args.num_logprobs, miss_rate=0.0, miss_layers=None,
        max_model_len=args.max_model_len)
    baseline_time = time.time() - t0
    print(f"Baseline complete in {baseline_time:.1f}s")

    # Save baseline generations for reference
    baseline_gens = []
    for out in baseline_outputs:
        baseline_gens.append({
            "prompt": out.prompt[:100] + "..." if len(out.prompt) > 100 else out.prompt,
            "generated": out.outputs[0].text if out.outputs else "",
        })
    with open(out_dir / "baseline_generations.json", "w") as f:
        json.dump(baseline_gens, f, indent=2)

    # ── Pass 2+: Miss-injected conditions ──────────────────────
    if args.miss_rate is not None:
        # Single condition mode
        conditions = [(args.miss_rate, args.miss_layers)]
    else:
        # Full experiment matrix
        conditions = [
            (0.02, None),       # Realistic: 2% all layers
            (0.04, "0"),        # Layer 0 only at 4%
            (0.02, "47"),       # Layer 47 only at 2%
            (0.02, "23"),       # Layer 23 only (control)
            (0.05, None),       # High: 5% all layers
            (0.10, None),       # Extreme: 10% all layers
        ]

    results = []
    for miss_rate, miss_layers in conditions:
        result = run_single_condition(
            args.model, prompts, args.tp_size, args.max_tokens,
            args.num_logprobs, miss_rate, miss_layers,
            baseline_outputs, args.max_model_len)
        results.append(result)

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<25} {'Flip%':>7} {'Top5':>7} {'KL_mean':>10} {'KL_max':>10}")
    print("-" * 65)
    for r in results:
        label = f"r={r.miss_rate:.0%} L={r.miss_layers}"
        print(f"{label:<25} {r.top1_flip_rate*100:>6.2f}% "
              f"{r.top5_overlap_mean*100:>6.1f}% "
              f"{r.approx_kl_mean:>10.6f} {r.approx_kl_max:>10.6f}")

    # Decision gate
    print(f"\n{'='*60}")
    print("DECISION GATE")
    print(f"{'='*60}")
    realistic = next((r for r in results if r.miss_rate == 0.02
                       and r.miss_layers == "all"), None)
    if realistic:
        kl_ok = realistic.approx_kl_mean < 0.01
        flip_ok = realistic.top1_flip_rate < 0.01
        print(f"  Realistic (2% all): KL={realistic.approx_kl_mean:.6f} "
              f"({'OK' if kl_ok else 'WARN'}), "
              f"Flip={realistic.top1_flip_rate*100:.2f}% "
              f"({'OK' if flip_ok else 'WARN'})")
        if kl_ok and flip_ok:
            print("  → PASS: Misses have negligible quality impact")
            print("  → W2/W3 are nice-to-have optimizations")
        else:
            print("  → FAIL: Misses affect quality significantly")
            print("  → W2 (3-C) and/or W3 (3-D) recommended")

    # Layer comparison
    l0 = next((r for r in results if r.miss_layers == "0"), None)
    l23 = next((r for r in results if r.miss_layers == "23"), None)
    l47 = next((r for r in results if r.miss_layers == "47"), None)
    if l0 and l23:
        print(f"\n  Layer comparison:")
        print(f"    L0  (4%): flip={l0.top1_flip_rate*100:.2f}%, "
              f"KL={l0.approx_kl_mean:.6f}")
        print(f"    L23 (2%): flip={l23.top1_flip_rate*100:.2f}%, "
              f"KL={l23.approx_kl_mean:.6f}")
        if l47:
            print(f"    L47 (2%): flip={l47.top1_flip_rate*100:.2f}%, "
                  f"KL={l47.approx_kl_mean:.6f}")
        if l0.top1_flip_rate > 2 * l23.top1_flip_rate:
            print("    → L0 miss is significantly more harmful than mid-layer")
            print("    → W2 (3-C Layer 0 Eager) strongly recommended")

    # Save results
    results_data = []
    for r in results:
        results_data.append({
            "miss_rate": r.miss_rate,
            "miss_layers": r.miss_layers,
            "num_prompts": r.num_prompts,
            "num_tokens_compared": r.num_tokens_compared,
            "top1_flip_rate": r.top1_flip_rate,
            "top5_overlap_mean": r.top5_overlap_mean,
            "approx_kl_mean": r.approx_kl_mean,
            "approx_kl_std": r.approx_kl_std,
            "approx_kl_max": r.approx_kl_max,
            "wall_time_s": r.wall_time_s,
        })
    with open(out_dir / "quality_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {out_dir / 'quality_results.json'}")


if __name__ == "__main__":
    main()
