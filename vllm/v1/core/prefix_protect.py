# SPDX-License-Identifier: Apache-2.0
"""Prefix Protection V3 cost model — 3-way decision (Ce / Cc / Cp).

Ce = Cost_expert_evict  — reloading evicted experts during decode
Cc = Cost_cached_reclaim — re-prefilling destroyed prefix cache blocks
Cp = Cost_req_preempt   — recomputing preempted request tokens

Decision:
  USE_UNCACHED         — uncached free blocks suffice, no cost
  PROTECT_AND_EXPAND   — Ce is cheapest: expand KV via expert eviction
  RECLAIM_CACHED       — Cc is cheapest: reclaim cached blocks
  PREEMPT              — Cp is cheapest: preempt a running request

V3 formulas (VAMP_V3_TWO_LAYER_DESIGN.md):
  Ce = stall_step × H_eff
    stall_step = k_per_layer × P_hit_one × C_reload
    P_hit_one  = 1 − (1 − G/E)^m_eff
    m_eff      = top_k × (N_decode + L_sync × N_prefill)
  Cp = N_computed × t_recompute_ms_per_token × 1000  (→ µs)
  Cc = p_reuse × (touched_blocks × block_size × t_prefill_tok_us + overhead)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class ProtectionDecision(Enum):
    USE_UNCACHED = "use_uncached"
    PROTECT_AND_EXPAND = "protect_and_expand"
    RECLAIM_CACHED = "reclaim_cached"
    PREEMPT = "preempt"


@dataclass
class PrefixProtectionConfig:
    """V3 3-way prefix protection cost model configuration."""

    # --- Ce parameters (V3 Expected Hit Groups) ---
    local_num_experts: int = 512   # E: number of local experts
    group_size: int = 2            # G: experts per group
    top_k: int = 10                # top_k_eff
    c_reload_ms: float = 0.63     # C_reload per expert (ms, topology dep.)
    num_layers: int = 1            # L_local
    h_cap: int = 64                # H_cap
    l_sync_prefill: float = 2.0   # L_sync_prefill weight

    # --- Cp parameters ---
    t_recompute_ms_per_token: float = 0.260  # ms/tok (V3 t_recompute)

    # --- Cc parameters ---
    block_size: int = 16           # tokens per block
    t_prefill_tok_us: float = 15.0 # per-token prefill cost for cache rebuild
    p_reuse: float = 1.0           # probability of prefix reuse
    t_sched_us: float = 500.0      # scheduler overhead (µs)
    t_queue_us: float = 1000.0     # queue overhead (µs)

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
            num_layers=int(os.environ.get("VLLM_PP_NUM_LAYERS", "1")),
            l_sync_prefill=float(
                os.environ.get("VLLM_PP_L_SYNC_PREFILL", "2.0")),
            t_recompute_ms_per_token=float(
                os.environ.get("VLLM_PP_T_RECOMPUTE", "0.260")),
            block_size=int(os.environ.get("VLLM_PP_BLOCK_SIZE", "16")),
            t_prefill_tok_us=float(
                os.environ.get("VLLM_PP_T_PREFILL_TOK", "15.0")),
            t_sched_us=float(os.environ.get("VLLM_PP_T_SCHED", "500.0")),
            t_queue_us=float(os.environ.get("VLLM_PP_T_QUEUE", "1000.0")),
            p_reuse=float(os.environ.get("VLLM_PP_P_REUSE", "1.0")),
        )

    # ------------------------------------------------------------------ #
    # V3 cost functions
    # ------------------------------------------------------------------ #

    def compute_ce(self, h_eff: int, groups_to_evict: int,
                   n_decode: int, n_prefill: int) -> float:
        """V3 Cost_expert_evict (microseconds).

        stall_step = k_per_layer × P_hit_one × C_reload
        P_hit_one  = 1 − (1 − G/E)^m_eff
        m_eff      = top_k × (N_decode + L_sync × N_prefill)
        Ce         = stall_step × H_eff × 1000  (ms → µs)
        """
        E = self.local_num_experts
        G = self.group_size
        if E <= 0 or groups_to_evict <= 0:
            return 0.0

        k_per_layer = groups_to_evict / max(1, self.num_layers)
        m_eff = self.top_k * (n_decode + self.l_sync_prefill * n_prefill)
        p_hit_one = 1.0 - (1.0 - G / E) ** m_eff if m_eff > 0 else 0.0
        stall_step = k_per_layer * p_hit_one * self.c_reload_ms
        return stall_step * h_eff * 1000  # → µs

    def compute_cc(self, touched_cached_blocks: int) -> float:
        """Cc: cached free block reclaim cost (microseconds).

        Prefix re-prefill cost. Linear in touched_cached_blocks.
        """
        if touched_cached_blocks <= 0:
            return 0.0
        return self.p_reuse * (
            touched_cached_blocks * self.block_size * self.t_prefill_tok_us
            + self.t_sched_us + self.t_queue_us)

    def compute_cp(self, preempt_computed_tokens: int) -> float:
        """V3 Cost_req_preempt (microseconds).

        NOTE: single-candidate heuristic.
        The scheduler may preempt multiple requests, so this value is
        a lower bound on actual total preempt cost.
        """
        if preempt_computed_tokens <= 0:
            return 0.0
        return preempt_computed_tokens * self.t_recompute_ms_per_token * 1000

    # ------------------------------------------------------------------ #
    # V3 3-way decision
    # ------------------------------------------------------------------ #

    def decide(self, h_eff: int, groups_to_evict: int,
               touched_cached_blocks: int,
               preempt_computed_tokens: int,
               n_decode: int, n_prefill: int,
               can_fully_protect: bool) -> ProtectionDecision:
        """3-way minimum cost selection.

        Args:
            h_eff: Effective remaining decode steps (V3 H_eff).
            groups_to_evict: Expert groups to evict for protection.
            touched_cached_blocks: Cached blocks that would be reclaimed.
            preempt_computed_tokens: Computed tokens of worst preempt
                candidate (single-candidate heuristic, lower bound).
            n_decode: Number of decode requests in batch.
            n_prefill: Number of prefill requests in batch.
            can_fully_protect: True if remaining expand capacity >=
                protection_gap.  When False, PROTECT_AND_EXPAND is
                structurally blocked (partial expand prevention).

        Returns:
            ProtectionDecision enum.
        """
        if touched_cached_blocks == 0:
            return ProtectionDecision.USE_UNCACHED

        Ce = (self.compute_ce(h_eff, groups_to_evict, n_decode, n_prefill)
              if can_fully_protect else float('inf'))
        Cc = self.compute_cc(touched_cached_blocks)
        Cp = self.compute_cp(preempt_computed_tokens)

        # When Cp==0 (no preempt candidates), PREEMPT is not an option.
        if Cp <= 0:
            Cp = float('inf')

        if Ce <= Cc and Ce <= Cp:
            return ProtectionDecision.PROTECT_AND_EXPAND
        if Cc <= Ce and Cc <= Cp:
            return ProtectionDecision.RECLAIM_CACHED
        return ProtectionDecision.PREEMPT

    # ------------------------------------------------------------------ #
    # Legacy compatibility: should_protect() for existing callers
    # ------------------------------------------------------------------ #

    def should_protect(self, b_eff: int, h_eff: int,
                       groups_to_evict: int,
                       protection_gap: int = 1) -> bool:
        """Legacy 2-way: Ce < Cp means protect."""
        if groups_to_evict <= 0:
            return True
        ce = self.compute_ce(h_eff, groups_to_evict, b_eff, 0)
        cc = self.compute_cc(protection_gap)
        return ce < cc
