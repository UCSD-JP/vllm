# SPDX-License-Identifier: Apache-2.0
"""Prefix Protection cost model — Ce vs Cp comparison.

Ce = cost of expert eviction (reloading evicted experts during decode)
Cp = cost of prefix reclaim (re-prefilling after cached blocks destroyed)

If Ce < Cp → protect cached prefix blocks (expand KV via expert eviction).
If Ce >= Cp → allow cached reclaim (normal allocation).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class PrefixProtectionConfig:
    """Configuration for prefix protection cost model.

    Controls whether the scheduler should protect cached prefix blocks
    from eviction by expanding KV cache via expert eviction instead.
    """

    # Ce parameters
    top_k: int = 10
    group_size: int = 2
    num_experts: int = 512
    c_reload_eff_ms: float = 0.252
    h_floor: int = 1
    h_cap: int = 64

    # Cp parameters
    block_size: int = 544
    t_prefill_tok_us: float = 15.0
    t_sched_us: float = 500.0
    t_queue_us: float = 1000.0
    p_reuse: float = 1.0

    enable: bool = False

    @classmethod
    def from_env(cls) -> "PrefixProtectionConfig":
        """VLLM_PREFIX_PROTECTION_ENABLE=1 to activate."""
        return cls(
            enable=os.environ.get(
                "VLLM_PREFIX_PROTECTION_ENABLE", "0") == "1",
            top_k=int(os.environ.get("VLLM_PP_TOP_K", "10")),
            group_size=int(os.environ.get("VLLM_PP_GROUP_SIZE", "2")),
            num_experts=int(os.environ.get("VLLM_PP_NUM_EXPERTS", "512")),
            c_reload_eff_ms=float(
                os.environ.get("VLLM_PP_C_RELOAD_MS", "0.252")),
            h_cap=int(os.environ.get("VLLM_PP_H_CAP", "64")),
            h_floor=int(os.environ.get("VLLM_PP_H_FLOOR", "1")),
            block_size=int(os.environ.get("VLLM_PP_BLOCK_SIZE", "544")),
            t_prefill_tok_us=float(
                os.environ.get("VLLM_PP_T_PREFILL_TOK", "15.0")),
            t_sched_us=float(os.environ.get("VLLM_PP_T_SCHED", "500.0")),
            t_queue_us=float(os.environ.get("VLLM_PP_T_QUEUE", "1000.0")),
            p_reuse=float(os.environ.get("VLLM_PP_P_REUSE", "1.0")),
        )

    def compute_ce(self, b_eff: int, h_eff: int,
                   groups_to_evict: int) -> float:
        """Compute expert eviction cost in microseconds.

        k_evicted = groups_to_evict * group_size (dynamic).
        """
        k_evicted = groups_to_evict * self.group_size
        return (b_eff * self.top_k * (k_evicted / self.num_experts)
                * self.c_reload_eff_ms * h_eff * 1000)

    def compute_cp(self, protection_gap: int = 1) -> float:
        """Compute prefix reclaim cost in microseconds.

        Scales with protection_gap: losing N cached blocks means
        re-prefilling N * block_size tokens plus per-request overhead.
        Without scaling, Cp is constant and Ce always dominates for
        large protection gaps, making protection structurally impossible.
        """
        return self.p_reuse * (
            protection_gap * self.block_size * self.t_prefill_tok_us
            + self.t_sched_us + self.t_queue_us)

    def should_protect(self, b_eff: int, h_eff: int,
                       groups_to_evict: int,
                       protection_gap: int = 1) -> bool:
        """Ce < Cp means expert eviction is cheaper → protect prefix."""
        if groups_to_evict <= 0:
            return True  # no expand needed = always protect
        return (self.compute_ce(b_eff, h_eff, groups_to_evict)
                < self.compute_cp(protection_gap))
