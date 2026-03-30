# SPDX-License-Identifier: Apache-2.0
"""Prefix Protection V4 cost model — work-conserving decision (Ce/Cc/Cp/Cb).

Ce = Cost_expert_evict  — per-token DMA reload stall from evicted experts
Cc = Cost_cached_reclaim — re-prefilling destroyed prefix cache blocks
Cp = Cost_req_preempt   — recomputing preempted request tokens
Cb = Cost_partial_expand — MPC first-action: floor expand + reclaim remainder

Decision:
  USE_UNCACHED                — uncached free blocks suffice, no cost
  PROTECT_AND_EXPAND          — Ce is cheapest: expand KV via expert eviction
  RECLAIM_CACHED              — Cc is cheapest: reclaim cached blocks
  PREEMPT                     — Cp is cheapest: preempt a running request
  PARTIAL_EXPAND_AND_RECLAIM  — Cb is cheapest: partial expand + reclaim rest

4-way argmin(Ce, Cc, Cp, Cb). Cb degenerates to inf when floor=0.
Fallback is RECLAIM_CACHED.

V4 Ce (traffic-bound, per-token amortized):
  rho  = (groups_to_evict × G) / E        — miss rate (uniform routing)
  Ce   = top_k × rho × c_reload × h_eff × 1000   (ms → µs)

  Physical basis: expert reload is DMA I/O → stall ∝ miss count.
  Per-step stall = batch × per-token stall; per-request share = stall / batch
  → batch cancels out (amortized).

Cc = p_reuse × (touched_blocks × block_size × t_prefill_tok + overhead)
Cp = t_sched + t_queue + partial_tail_tokens × t_prefill_tok
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class CallerKind(Enum):
    RUNNING = "running"
    WAITING = "waiting"


class ProtectionDecision(Enum):
    USE_UNCACHED = "use_uncached"
    PROTECT_AND_EXPAND = "protect_and_expand"
    RECLAIM_CACHED = "reclaim_cached"
    PREEMPT = "preempt"         # preempt another victim
    PARTIAL_EXPAND_AND_RECLAIM = "partial_expand_and_reclaim"


@dataclass
class PrefixProtectionConfig:
    """V4 4-way prefix protection cost model configuration (Ce/Cc/Cp/Cb)."""

    # --- Ce parameters (V4 traffic-bound) ---
    local_num_experts: int = 512   # E: number of local experts
    group_size: int = 2            # G: experts per group
    top_k: int = 10                # top_k per token
    c_reload_ms: float = 0.63     # C_reload per expert (ms, topology dep.)
    h_cap: int = 64                # H_cap (remaining decode steps clamp)

    # --- Cp parameters ---
    t_recompute_ms_per_token: float = 0.260  # ms/tok (V3 t_recompute)

    # --- Cc parameters ---
    block_size: int = 16           # tokens per block
    t_prefill_tok_us: float = 15.0 # per-token prefill cost for cache rebuild
    p_reuse: float = 1.0           # probability of prefix reuse (static fallback)
    p_reuse_alpha: float = 0.5     # dynamic p_reuse = alpha * hit_rate
    cc_age_scale: float = 2000.0   # recency age decay half-scale for Cc
    cc_gamma: float = 1.0          # gamma exponent for low-hit Cc suppression
    cc_hit_threshold: float = 0.0  # hit_rate threshold below which gamma applies
    t_sched_us: float = 500.0      # scheduler overhead (µs)
    t_queue_us: float = 1000.0     # queue overhead (µs)

    # --- Decode-freeze (step-boundary mode) ---
    decode_freeze_expand: bool = True  # VLLM_PP_DECODE_FREEZE (default ON)

    enable: bool = False

    @classmethod
    def from_env(cls) -> "PrefixProtectionConfig":
        """VLLM_PREFIX_PROTECTION_ENABLE=1 to activate."""
        return cls(
            enable=os.environ.get(
                "VLLM_PREFIX_PROTECTION_ENABLE", "0") == "1",
            top_k=int(os.environ.get("VLLM_PP_TOP_K", "10")),
            group_size=int(os.environ.get("VLLM_PP_GROUP_SIZE", "2")),
            local_num_experts=int(
                os.environ.get("VLLM_PP_NUM_EXPERTS", "512")),
            c_reload_ms=float(
                os.environ.get("VLLM_PP_C_RELOAD_MS", "0.63")),
            h_cap=int(os.environ.get("VLLM_PP_H_CAP", "64")),
            t_recompute_ms_per_token=float(
                os.environ.get("VLLM_PP_T_RECOMPUTE", "0.260")),
            block_size=int(os.environ.get("VLLM_PP_BLOCK_SIZE", "16")),
            t_prefill_tok_us=float(
                os.environ.get("VLLM_PP_T_PREFILL_TOK", "15.0")),
            t_sched_us=float(os.environ.get("VLLM_PP_T_SCHED", "500.0")),
            t_queue_us=float(os.environ.get("VLLM_PP_T_QUEUE", "1000.0")),
            p_reuse=float(os.environ.get("VLLM_PP_P_REUSE", "1.0")),
            p_reuse_alpha=float(
                os.environ.get("VLLM_PP_P_REUSE_ALPHA", "0.5")),
            cc_age_scale=float(
                os.environ.get("VLLM_PP_CC_AGE_SCALE", "2000.0")),
            cc_gamma=float(
                os.environ.get("VLLM_PP_CC_GAMMA", "1.0")),
            cc_hit_threshold=float(
                os.environ.get("VLLM_PP_CC_HIT_THRESHOLD", "0.0")),
            decode_freeze_expand=(
                os.environ.get("VLLM_PP_DECODE_FREEZE", "1") == "1"),
        )

    # ------------------------------------------------------------------ #
    # V4 cost functions
    # ------------------------------------------------------------------ #

    def compute_ce(self, h_eff: int, groups_to_evict: int,
                   n_decode: int = 0, n_prefill: int = 0) -> float:
        """V4 Cost_expert_evict — traffic-bound DMA model (microseconds).

        rho  = (groups_to_evict × G) / E   — fraction of experts evicted
        Ce   = top_k × rho × c_reload × h_eff × 1000

        Per-token: top_k experts routed, each has rho probability of
        being evicted (uniform routing).  Each miss = c_reload ms DMA.
        Batch cancels via amortization (see module docstring).
        """
        E = self.local_num_experts
        G = self.group_size
        if E <= 0 or groups_to_evict <= 0:
            return 0.0

        rho = min((groups_to_evict * G) / E, 1.0)  # clamp: can't evict > 100%
        stall_per_token_ms = self.top_k * rho * self.c_reload_ms
        return stall_per_token_ms * h_eff * 1000  # ms → µs

    def compute_cc(self, touched_cached_blocks: int,
                   hit_rate: float | None = None,
                   recency_age: float = 0.0) -> float:
        """Cc: cached free block reclaim cost (microseconds).

        Prefix re-prefill cost. Linear in touched_cached_blocks.
        When hit_rate is provided and p_reuse_alpha > 0, uses dynamic
        p_reuse = alpha * hit_rate instead of static p_reuse.

        Recency age decay: reclaim candidates (front of LRU cached queue)
        have high age and low reuse probability. Cost is discounted:
            p_effective = p_base / (1 + recency_age / cc_age_scale)

        With cc_age_scale=2000 and front_step_age as input:
            age=0     → p_eff = p (fresh, full cost)
            age=2000  → p_eff = p × 0.50 (half-life)
            age=14000 → p_eff = p × 0.125 (Ce-competitive)
        """
        if touched_cached_blocks <= 0:
            return 0.0
        if hit_rate is not None and self.p_reuse_alpha > 0:
            p = self.p_reuse_alpha * hit_rate
            # Threshold-gated suppression: low-hit region gets extra decay
            if (self.cc_hit_threshold > 0
                    and self.cc_gamma != 1.0
                    and hit_rate < self.cc_hit_threshold):
                p *= (hit_rate / self.cc_hit_threshold) ** (
                    self.cc_gamma - 1.0)
        else:
            p = self.p_reuse  # static fallback
        # Recency age decay: cold blocks are cheaper to reclaim
        if recency_age > 0 and self.cc_age_scale > 0:
            p = p / (1.0 + recency_age / self.cc_age_scale)
        return p * (
            touched_cached_blocks * self.block_size * self.t_prefill_tok_us
            + self.t_sched_us + self.t_queue_us)

    def compute_cp_running(self, victim_computed_tokens: int,
                           t_queue_override: float | None = None) -> float:
        """Running victim preemption cost (microseconds).

        Model: preempting frees full blocks (-> cached queue, handled by Cc)
        + at most 1 partial tail block (-> uncached queue).
        Cost = scheduler overhead + queue delay + partial tail replay.
        If no partial tail (computed % block_size == 0), uncached_yield=0
        -> Cp=inf (useless preemption for uncached deficit).

        Args:
            victim_computed_tokens: Computed tokens of preempt victim.
            t_queue_override: Measured queue delay EMA (µs). When provided,
                replaces static t_queue_us for runtime-calibrated Cp.
        """
        if victim_computed_tokens <= 0:
            return 0.0
        partial_tail_tokens = victim_computed_tokens % self.block_size
        if partial_tail_tokens == 0:
            return float('inf')  # no uncached yield -> useless preemption
        t_queue = (t_queue_override
                   if t_queue_override is not None else self.t_queue_us)
        return (self.t_sched_us + t_queue
                + partial_tail_tokens * self.t_prefill_tok_us)

    # ------------------------------------------------------------------ #
    # V4 3-way decision
    # ------------------------------------------------------------------ #

    def decide(self, h_eff: int, groups_to_evict: int,
               touched_cached_blocks: int,
               preempt_computed_tokens: int,
               n_decode: int, n_prefill: int,
               can_fully_protect: bool,
               caller: CallerKind = CallerKind.RUNNING,
               hit_rate: float | None = None,
               recency_age: float = 0.0,
               protection_gap: int = 0,
               floor_expand_groups: int = 0,
               floor_expand_blocks: int = 0,
               t_queue_override: float | None = None,
               has_decode: bool = False,
               ) -> ProtectionDecision:
        """Work-conserving 4-way cost selection (Ce/Cc/Cp/Cb).

        4-way argmin over corner solutions + Plan B (partial expand):
          Ce — full expand cost
          Cc — full cached reclaim cost
          Cp — gap-normalized preempt cost
          Cb — MPC first-action: Ce_floor + Cc_remainder (inf when floor=0)
        Fallback is RECLAIM_CACHED (progress guaranteed, no DEFER).

        Args:
            h_eff: Effective remaining decode steps (V3 H_eff).
            groups_to_evict: Expert groups to evict for protection.
            touched_cached_blocks: Cached blocks that would be reclaimed.
            preempt_computed_tokens: Computed tokens of best preempt
                victim (yield-aware, partial-tail model).  0 when
                running is empty.
            n_decode: Number of decode requests in batch.
            n_prefill: Number of prefill requests in batch.
            can_fully_protect: True if remaining expand capacity >=
                protection_gap.  When False, PROTECT_AND_EXPAND is
                structurally blocked (partial expand prevention).
            caller: RUNNING or WAITING (logged for diagnostics).
            hit_rate: Sliding-window prefix cache hit rate for dynamic Cc.
            recency_age: Step-based recency age of LRU front cached block.
                Higher age → lower p_reuse → cheaper Cc (cold blocks).
            protection_gap: Uncached block deficit (= required - noncached_free).
                Used to normalize Cp: each preempt yields ~1 block,
                so Cp_total = Cp_single × protection_gap.
            floor_expand_groups: Quantum-floored group count for Plan B.
                0 → Cb=inf → degenerates to 3-way.
            floor_expand_blocks: KV blocks from floor_expand_groups.
                0 → Cb=inf.

        Returns:
            ProtectionDecision enum.
        """
        if protection_gap <= 0 and touched_cached_blocks == 0:
            return ProtectionDecision.USE_UNCACHED

        # Decode-freeze: block expert eviction when decode tokens present
        _can_expand = (can_fully_protect
                       and not (self.decode_freeze_expand and has_decode))
        Ce = (self.compute_ce(h_eff, groups_to_evict, n_decode, n_prefill)
              if _can_expand else float('inf'))
        # Cc is inf when no cached blocks exist to reclaim
        Cc = (self.compute_cc(touched_cached_blocks, hit_rate=hit_rate,
                              recency_age=recency_age)
              if touched_cached_blocks > 0 else float('inf'))
        Cp_single = self.compute_cp_running(
            preempt_computed_tokens, t_queue_override=t_queue_override)
        if Cp_single <= 0:
            Cp_single = float('inf')
        # Each preempt yields at most 1 uncached block (partial tail).
        # Ce/Cc price the full gap; Cp must be normalized to match.
        gap = max(1, protection_gap if protection_gap > 0
                  else touched_cached_blocks)
        Cp = Cp_single * gap

        # Plan B: partial expand (quantum floor) + reclaim remainder
        # (MPC first-action cost estimate)
        # Also blocked by decode-freeze (partial expand still evicts experts)
        Cb = float('inf')
        if (floor_expand_groups > 0 and floor_expand_blocks > 0
                and _can_expand):
            remainder = max(0, gap - floor_expand_blocks)
            if remainder > 0 and touched_cached_blocks > 0:
                Ce_floor = self.compute_ce(
                    h_eff, floor_expand_groups, n_decode, n_prefill)
                Cc_rem = self.compute_cc(
                    min(remainder, touched_cached_blocks),
                    hit_rate=hit_rate, recency_age=recency_age)
                Cb = Ce_floor + Cc_rem

        costs = [
            (Ce, ProtectionDecision.PROTECT_AND_EXPAND),
            (Cc, ProtectionDecision.RECLAIM_CACHED),
            (Cp, ProtectionDecision.PREEMPT),
            (Cb, ProtectionDecision.PARTIAL_EXPAND_AND_RECLAIM),
        ]
        _, best = min(costs, key=lambda x: x[0])
        return best

