# SPDX-License-Identifier: Apache-2.0
"""Elastic KV configuration — expert → KV one-way expansion MVP."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ElasticKVConfig:
    enable: bool = False
    min_resident_ratio: float = 0.5
    expand_group_quantum: int = 4
    expert_eviction_policy: str = "lru"

    # Computed at runtime (not from env)
    max_expand_blocks: int = 0
    pages_per_block: int = 0
    group_pages: int = 0
    per_tensor_block_bytes: dict[int, int] | None = None
    page_size: int = 0

    # Runtime Ce params (set by worker, not from env)
    local_num_experts: int = 0
    expert_group_size: int = 0
    expert_top_k: int = 0
    num_layers: int = 0
    c_reload_ms: float = 0.63  # default PCIe

    @classmethod
    def from_env(cls) -> "ElasticKVConfig":
        return cls(
            enable=os.environ.get("VLLM_ELASTIC_KV_ENABLE", "0") == "1",
            min_resident_ratio=float(
                os.environ.get("VLLM_ELASTIC_KV_MIN_RESIDENT", "0.5")),
            expand_group_quantum=int(
                os.environ.get("VLLM_ELASTIC_KV_GROUPS_PER_EXPAND", "4")),
            expert_eviction_policy=os.environ.get(
                "VLLM_ELASTIC_KV_EVICT_POLICY", "lru"),
            max_expand_blocks=int(
                os.environ.get("VLLM_ELASTIC_KV_MAX_EXPAND_BLOCKS", "0")),
        )

    def compute_derived(
        self,
        pool,  # VMMPagePool
        per_tensor_block_bytes: dict[int, int],
    ) -> None:
        """Compute max_expand_blocks from VMM geometry."""
        from vllm.vmm_pool import max_blocks_for_pages

        total_expert_pages = pool.num_expert_pages
        evictable_pages = int(
            total_expert_pages * (1.0 - self.min_resident_ratio))
        self.max_expand_blocks = max_blocks_for_pages(
            evictable_pages, per_tensor_block_bytes, pool.page_size)

        # Store runtime geometry for scheduler prefix protection cost model
        from vllm.vmm_pool import pages_for_blocks
        self.pages_per_block = pages_for_blocks(
            1, per_tensor_block_bytes, pool.page_size)
        self.group_pages = pool.group_pages
        self.per_tensor_block_bytes = dict(per_tensor_block_bytes)
        self.page_size = pool.page_size

        natural_pages = self.expand_group_quantum * pool.group_pages
        natural_blocks = max_blocks_for_pages(
            natural_pages, per_tensor_block_bytes, pool.page_size)

        logger.info(
            "ElasticKV: expand_group_quantum=%d → %d pages → %d blocks/call, "
            "max_expand=%d blocks total "
            "(%d evictable / %d total expert pages, "
            "min_resident=%.0f%%)",
            self.expand_group_quantum, natural_pages, natural_blocks,
            self.max_expand_blocks,
            evictable_pages, total_expert_pages,
            self.min_resident_ratio * 100,
        )
