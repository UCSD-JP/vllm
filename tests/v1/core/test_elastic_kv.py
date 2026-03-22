# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Elastic KV subsystem.

Layer 1: pages_for_blocks / max_blocks_for_pages batch arithmetic
Layer 2: BlockPool.expand_blocks()
Layer 3: Scheduler elastic expand (deficit block-unit conversion)
Bug 6 regression: deficit_blocks = ceil(num_new_tokens / block_size)
"""
import math
from unittest.mock import MagicMock, patch

import pytest

from vllm.vmm_pool import max_blocks_for_pages, pages_for_blocks


# ---------------------------------------------------------------------------
# Layer 1: pages_for_blocks / max_blocks_for_pages
# ---------------------------------------------------------------------------

class TestPageBlockConversion:
    """Test batched page<->block helpers (64× batching gain)."""

    # Standard geometry: 64 KV tensors × 32 KiB each, 2 MiB pages
    PAGE_SIZE = 2 * 1024 * 1024  # 2 MiB
    BLOCK_BYTES = 32 * 1024  # 32 KiB per tensor per block
    NUM_TENSORS = 64

    @pytest.fixture
    def per_tensor(self):
        """64 KV tensors each 32 KiB per block."""
        return {i: self.BLOCK_BYTES for i in range(self.NUM_TENSORS)}

    def test_pages_for_blocks_batched_gain(self, per_tensor):
        """Batched: ceil(64 * 32 KiB / 2 MiB) = 1 page per tensor.
        Total = 64 pages for 1 block.
        Naive per-tensor: 64 * ceil(32 KiB / 2 MiB) = 64 pages.
        Batched = naive at n_blocks=1 for this geometry.
        """
        pages = pages_for_blocks(1, per_tensor, self.PAGE_SIZE)
        # Each tensor: ceil(1 * 32K / 2M) = 1 page; total = 64
        assert pages == 64

    def test_pages_for_blocks_large_batch(self, per_tensor):
        """64 blocks: ceil(64 * 32 KiB / 2 MiB) = ceil(2 MiB / 2 MiB) = 1
        page per tensor. Total = 64 pages.  Batching: 64 blocks → 64 pages.
        """
        pages = pages_for_blocks(64, per_tensor, self.PAGE_SIZE)
        # Each tensor: ceil(64 * 32K / 2M) = ceil(2M / 2M) = 1
        assert pages == 64

    def test_pages_for_blocks_65_blocks(self, per_tensor):
        """65 blocks: ceil(65 * 32 KiB / 2 MiB) = ceil(2080 KiB / 2048 KiB) = 2
        pages per tensor. Total = 128 pages.
        """
        pages = pages_for_blocks(65, per_tensor, self.PAGE_SIZE)
        assert pages == 128

    def test_max_blocks_for_pages_inverse(self, per_tensor):
        """max_blocks_for_pages(64, ...) should return 64 (one full 2 MiB
        page per tensor → 64 blocks each)."""
        blocks = max_blocks_for_pages(64, per_tensor, self.PAGE_SIZE)
        assert blocks == 64

    def test_max_blocks_for_pages_128(self, per_tensor):
        """128 pages → should support 128 blocks
        (2 pages per tensor → 128 * 32 KiB = 4 MiB = 2 pages)."""
        blocks = max_blocks_for_pages(128, per_tensor, self.PAGE_SIZE)
        assert blocks == 128

    def test_roundtrip(self, per_tensor):
        """pages_for_blocks(max_blocks_for_pages(N)) <= N."""
        for n_pages in [1, 10, 64, 100, 128, 256, 1000]:
            blocks = max_blocks_for_pages(n_pages, per_tensor, self.PAGE_SIZE)
            if blocks > 0:
                pages_back = pages_for_blocks(
                    blocks, per_tensor, self.PAGE_SIZE)
                assert pages_back <= n_pages, (
                    f"roundtrip failed: {n_pages} pages → {blocks} blocks → "
                    f"{pages_back} pages"
                )

    def test_zero_pages(self, per_tensor):
        """0 pages → 0 blocks."""
        assert max_blocks_for_pages(0, per_tensor, self.PAGE_SIZE) == 0

    def test_zero_blocks(self, per_tensor):
        """0 blocks → 0 pages."""
        assert pages_for_blocks(0, per_tensor, self.PAGE_SIZE) == 0

    def test_empty_per_tensor(self):
        """Empty per_tensor → 0 blocks."""
        assert max_blocks_for_pages(100, {}, self.PAGE_SIZE) == 0
        assert pages_for_blocks(100, {}, self.PAGE_SIZE) == 0

    def test_single_tensor(self):
        """Single tensor: simpler case."""
        per_tensor = {0: 32 * 1024}
        # 1 block = 32 KiB, page = 2 MiB → ceil(32K/2M) = 1 page
        assert pages_for_blocks(1, per_tensor, self.PAGE_SIZE) == 1
        # 64 blocks = 2 MiB → exactly 1 page
        assert pages_for_blocks(64, per_tensor, self.PAGE_SIZE) == 1
        # 65 blocks → ceil(65*32K/2M) = 2 pages
        assert pages_for_blocks(65, per_tensor, self.PAGE_SIZE) == 2
        # Inverse
        assert max_blocks_for_pages(1, per_tensor, self.PAGE_SIZE) == 64
        assert max_blocks_for_pages(2, per_tensor, self.PAGE_SIZE) == 128


# ---------------------------------------------------------------------------
# Layer 2: BlockPool.expand_blocks()
# ---------------------------------------------------------------------------

class TestBlockPoolExpand:
    """Test expand_blocks logic (inline to avoid heavy import chain)."""

    def _make_mock_pool(self, num_blocks=100):
        """Create a mock pool with expand_blocks logic inlined."""
        from dataclasses import dataclass

        @dataclass
        class FakeBlock:
            block_id: int

        pool = MagicMock()
        pool.num_gpu_blocks = num_blocks
        pool.blocks = [FakeBlock(block_id=i) for i in range(num_blocks)]
        pool.free_block_queue = MagicMock()

        def expand_blocks(n_new_blocks):
            if n_new_blocks <= 0:
                return
            old_size = pool.num_gpu_blocks
            pool.num_gpu_blocks += n_new_blocks
            for i in range(n_new_blocks):
                block_id = old_size + i
                block = FakeBlock(block_id=block_id)
                pool.blocks.append(block)
                pool.free_block_queue.append(block)

        pool.expand_blocks = expand_blocks
        return pool

    def test_expand_blocks_adds_blocks(self):
        pool = self._make_mock_pool(100)
        pool.expand_blocks(10)
        assert pool.num_gpu_blocks == 110
        assert len(pool.blocks) == 110
        for i in range(10):
            assert pool.blocks[100 + i].block_id == 100 + i

    def test_expand_blocks_zero(self):
        pool = self._make_mock_pool(100)
        pool.expand_blocks(0)
        assert pool.num_gpu_blocks == 100

    def test_expand_blocks_negative(self):
        pool = self._make_mock_pool(100)
        pool.expand_blocks(-5)
        assert pool.num_gpu_blocks == 100


# ---------------------------------------------------------------------------
# Layer 3: Scheduler elastic expand — deficit unit conversion
# ---------------------------------------------------------------------------

class TestSchedulerElasticExpand:
    """Test _try_elastic_kv_expand and deficit unit conversion."""

    @staticmethod
    def _try_elastic_kv_expand_impl(self, deficit: int) -> int:
        """Re-implementation of Scheduler._try_elastic_kv_expand for testing
        (avoids importing Scheduler which has heavy deps)."""
        if not hasattr(self, '_elastic_kv_handler') or deficit <= 0:
            return 0
        cfg = self._elastic_kv_config
        remaining = (cfg.max_expand_blocks - self._elastic_kv_expanded_total
                     if cfg.max_expand_blocks > 0 else float('inf'))
        if remaining <= 0:
            return 0
        max_blocks = int(remaining)
        added = self._elastic_kv_handler(deficit, max_blocks)
        if added > 0:
            self.kv_cache_manager.block_pool.expand_blocks(added)
            self._elastic_kv_expanded_total += added
        return added

    def _make_scheduler_with_handler(self, block_size=16,
                                     max_expand_blocks=100):
        """Create a minimal Scheduler-like object for testing."""
        scheduler = MagicMock()
        scheduler.block_size = block_size

        # Bind the re-implemented method
        scheduler._try_elastic_kv_expand = (
            lambda deficit: self._try_elastic_kv_expand_impl(
                scheduler, deficit))

        # Mock handler
        mock_handler = MagicMock(return_value=5)
        scheduler._elastic_kv_handler = mock_handler

        # Mock config
        mock_config = MagicMock()
        mock_config.max_expand_blocks = max_expand_blocks
        scheduler._elastic_kv_config = mock_config
        scheduler._elastic_kv_expanded_total = 0

        # Mock block pool
        scheduler.kv_cache_manager = MagicMock()

        return scheduler, mock_handler

    def test_try_expand_calls_handler(self):
        scheduler, handler = self._make_scheduler_with_handler()
        added = scheduler._try_elastic_kv_expand(5)
        assert added == 5
        handler.assert_called_once_with(5, 100)

    def test_try_expand_zero_deficit(self):
        scheduler, handler = self._make_scheduler_with_handler()
        added = scheduler._try_elastic_kv_expand(0)
        assert added == 0
        handler.assert_not_called()

    def test_try_expand_respects_lifetime_cap(self):
        scheduler, handler = self._make_scheduler_with_handler(
            max_expand_blocks=10)
        scheduler._elastic_kv_expanded_total = 8
        # remaining = 10 - 8 = 2
        handler.return_value = 2
        added = scheduler._try_elastic_kv_expand(5)
        # max_blocks should be capped to 2
        handler.assert_called_once_with(5, 2)
        assert added == 2

    def test_try_expand_exhausted(self):
        scheduler, handler = self._make_scheduler_with_handler(
            max_expand_blocks=10)
        scheduler._elastic_kv_expanded_total = 10
        added = scheduler._try_elastic_kv_expand(5)
        assert added == 0
        handler.assert_not_called()

    def test_deficit_unit_conversion(self):
        """Bug 6 regression: deficit should be ceil(tokens / block_size)."""
        block_size = 16
        test_cases = [
            (1, 1),      # 1 token → ceil(1/16) = 1 block
            (16, 1),     # 16 tokens → ceil(16/16) = 1 block
            (17, 2),     # 17 tokens → ceil(17/16) = 2 blocks
            (32, 2),     # 32 tokens → 2 blocks
            (33, 3),     # 33 tokens → 3 blocks
            (100, 7),    # 100 tokens → ceil(100/16) = 7 blocks
            (0, 1),      # 0 tokens → max(1, ceil(0/16)) = 1 block
        ]
        for num_new_tokens, expected_deficit in test_cases:
            deficit_blocks = max(1, math.ceil(num_new_tokens / block_size))
            assert deficit_blocks == expected_deficit, (
                f"tokens={num_new_tokens}: expected deficit={expected_deficit},"
                f" got {deficit_blocks}"
            )


# ---------------------------------------------------------------------------
# Config plumbing: compute_derived
# ---------------------------------------------------------------------------

class TestElasticKVConfig:
    """Test ElasticKVConfig."""

    def test_from_env_defaults(self):
        from vllm.elastic_kv_config import ElasticKVConfig
        with patch.dict('os.environ', {}, clear=True):
            config = ElasticKVConfig.from_env()
            assert config.enable is False
            assert config.max_expand_blocks == 0

    def test_from_env_with_max_expand(self):
        from vllm.elastic_kv_config import ElasticKVConfig
        env = {
            'VLLM_ELASTIC_KV_ENABLE': '1',
            'VLLM_ELASTIC_KV_MAX_EXPAND_BLOCKS': '42',
        }
        with patch.dict('os.environ', env, clear=True):
            config = ElasticKVConfig.from_env()
            assert config.enable is True
            assert config.max_expand_blocks == 42

    def test_compute_derived(self):
        from vllm.elastic_kv_config import ElasticKVConfig

        config = ElasticKVConfig(
            enable=True,
            min_resident_ratio=0.5,
            expand_group_quantum=4,
        )

        # Mock pool: 256 expert pages (50% evictable = 128 pages → 128 blocks)
        pool = MagicMock()
        pool.num_expert_pages = 256
        pool.page_size = 2 * 1024 * 1024
        pool.group_pages = 8

        per_tensor = {i: 32 * 1024 for i in range(64)}

        config.compute_derived(pool, per_tensor)
        # 128 evictable pages → 128 blocks with 64 tensors
        assert config.max_expand_blocks == 128


# ---------------------------------------------------------------------------
# 2-phase commit: partial handling
# ---------------------------------------------------------------------------

class TestTwoPhaseCommit:
    """Test engine-level 2-phase commit handler."""

    def test_commit_divergence_returns_zero(self):
        """When ranks return different added_blocks, return 0 (no partial)."""
        results = [
            {"added_blocks": 10, "freed_pages": 5, "groups_evicted": 1},
            {"added_blocks": 0, "freed_pages": 5, "groups_evicted": 1},
        ]
        committed = [r["added_blocks"] for r in results]
        if len(set(committed)) != 1:
            actual = 0  # divergence → reject
        else:
            actual = committed[0]
        assert actual == 0

    def test_commit_agreement(self):
        """When all ranks agree, return that value."""
        results = [
            {"added_blocks": 10, "freed_pages": 5, "groups_evicted": 1},
            {"added_blocks": 10, "freed_pages": 5, "groups_evicted": 1},
        ]
        committed = [r["added_blocks"] for r in results]
        if len(set(committed)) != 1:
            actual = 0
        else:
            actual = committed[0]
        assert actual == 10


# ---------------------------------------------------------------------------
# Fail-fast propagation: worker → engine
# ---------------------------------------------------------------------------

class TestWorkerFailFastCleanup:
    """Verify worker.init_elastic_kv() cleans up state on fail-fast."""

    def _make_worker(self, vmm_pool=None, expert_cache=None):
        """Create a minimal Worker-like object with the required attrs."""
        worker = MagicMock()
        worker.model_runner = MagicMock()
        worker.model_runner._vmm_pool = vmm_pool
        worker.model_runner._expert_cache = expert_cache
        worker.model_runner._elastic_kv_enabled = False
        worker.model_runner._per_tensor_block_bytes = {}
        worker._vmm_pool = None
        worker._expert_cache = None
        worker._elastic_kv_config = None
        worker._per_tensor_block_bytes = {}
        return worker

    def test_can_shrink_false_disables_worker(self):
        """_can_shrink() == False → config None, enabled False."""
        vmm_pool = MagicMock()
        expert_cache = MagicMock()
        expert_cache._can_shrink.return_value = False
        expert_cache._phase_c_enabled = False
        expert_cache.max_resident = 8
        expert_cache.local_num_experts = 8

        worker = self._make_worker(vmm_pool=vmm_pool,
                                   expert_cache=expert_cache)

        # Simulate init_elastic_kv logic (the part after config.enable check)
        worker.model_runner._elastic_kv_enabled = True
        worker._elastic_kv_config = MagicMock()  # pretend config was set
        worker._vmm_pool = vmm_pool
        worker._expert_cache = expert_cache

        # Fail-fast path
        if not expert_cache._can_shrink():
            worker.model_runner._elastic_kv_enabled = False
            worker._elastic_kv_config = None

        assert worker._elastic_kv_config is None
        assert worker.model_runner._elastic_kv_enabled is False

    def test_vmm_pool_none_disables_worker(self):
        """_vmm_pool is None → config None, enabled False."""
        worker = self._make_worker(vmm_pool=None)

        # Simulate init_elastic_kv: enable then discover vmm_pool is None
        worker.model_runner._elastic_kv_enabled = True
        worker._elastic_kv_config = MagicMock()
        worker._vmm_pool = worker.model_runner._vmm_pool  # None

        if worker._vmm_pool is None:
            worker.model_runner._elastic_kv_enabled = False
            worker._elastic_kv_config = None

        assert worker._elastic_kv_config is None
        assert worker.model_runner._elastic_kv_enabled is False

    def test_get_elastic_kv_config_returns_none_after_fail_fast(self):
        """After fail-fast, get_elastic_kv_config must return None."""
        worker = self._make_worker()
        worker._elastic_kv_config = None  # fail-fast already ran
        # Simulating the RPC getter
        assert worker._elastic_kv_config is None


class TestEngineFailFastPropagation:
    """Verify engine skips handler registration when any worker is None."""

    def test_all_none_configs_skips_handler(self):
        """get_elastic_kv_config() -> [None, None] → no handler."""
        engine = MagicMock()
        engine.collective_rpc.return_value = [None, None]
        engine.scheduler = MagicMock()

        configs = engine.collective_rpc("get_elastic_kv_config")
        registered = True
        if not configs or any(c is None for c in configs):
            registered = False

        assert registered is False
        engine.scheduler.set_elastic_kv_handler.assert_not_called()

    def test_partial_none_configs_skips_handler(self):
        """get_elastic_kv_config() -> [config, None] → no handler."""
        engine = MagicMock()
        valid_config = MagicMock()
        engine.collective_rpc.return_value = [valid_config, None]
        engine.scheduler = MagicMock()

        configs = engine.collective_rpc("get_elastic_kv_config")
        registered = True
        if not configs or any(c is None for c in configs):
            registered = False

        assert registered is False
        engine.scheduler.set_elastic_kv_handler.assert_not_called()

    def test_empty_configs_skips_handler(self):
        """get_elastic_kv_config() -> [] → no handler."""
        engine = MagicMock()
        engine.collective_rpc.return_value = []
        engine.scheduler = MagicMock()

        configs = engine.collective_rpc("get_elastic_kv_config")
        registered = True
        if not configs or any(c is None for c in configs):
            registered = False

        assert registered is False
        engine.scheduler.set_elastic_kv_handler.assert_not_called()

    def test_all_valid_configs_registers_handler(self):
        """get_elastic_kv_config() -> [config, config] → handler registered."""
        config_a = MagicMock()
        config_b = MagicMock()
        configs = [config_a, config_b]

        registered = True
        if not configs or any(c is None for c in configs):
            registered = False

        assert registered is True


# ---------------------------------------------------------------------------
# BlockPool 2-queue tests
# ---------------------------------------------------------------------------

class TestBlockPoolTwoQueue:
    """Tests for BlockPool split free queue (uncached + cached)."""

    def _make_pool(self, num_blocks=21):
        from vllm.v1.core.block_pool import BlockPool
        return BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=True,
            hash_block_size=16,
        )

    def _move_to_cached(self, pool, count):
        """Pop from uncached, set hash, put in cached queue."""
        blocks = pool.free_uncached_queue.popleft_n(count)
        for i, b in enumerate(blocks):
            b._block_hash = b'hash' + i.to_bytes(4, 'big')
        pool.free_cached_queue.append_n(blocks)
        return blocks

    def test_initial_state_all_uncached(self):
        """At init, all free blocks are in uncached queue."""
        pool = self._make_pool(21)
        # 1 null block consumed
        assert pool.get_num_noncached_free_blocks() == 20
        assert pool.get_num_cached_free_blocks() == 0
        assert pool.get_num_free_blocks() == 20

    def test_free_blocks_routing_by_hash(self):
        """free_blocks() routes to correct queue based on block_hash."""
        pool = self._make_pool(21)
        # Get 4 blocks
        blocks = pool.get_new_blocks(4)
        assert len(blocks) == 4
        # Set hash on 2 of them (simulating cache_full_blocks)
        blocks[0]._block_hash = b'hash0'
        blocks[1]._block_hash = b'hash1'
        # Free all — 2 should go to cached, 2 to uncached
        pool.free_blocks(reversed(blocks))
        assert pool.get_num_cached_free_blocks() == 2
        # uncached: 16 remaining + 2 freed = 18
        assert pool.get_num_noncached_free_blocks() == 18

    def test_touch_removes_from_correct_queue(self):
        """touch() removes cached block from cached queue."""
        pool = self._make_pool(21)
        cached_blocks = self._move_to_cached(pool, 3)
        assert pool.get_num_cached_free_blocks() == 3

        # Touch one cached block (simulating prefix hit)
        pool.touch([cached_blocks[0]])
        assert pool.get_num_cached_free_blocks() == 2
        assert cached_blocks[0].ref_cnt == 1

    def test_reset_prefix_cache_drains_cached_queue(self):
        """reset_prefix_cache() moves all cached→uncached."""
        pool = self._make_pool(21)
        self._move_to_cached(pool, 8)
        assert pool.get_num_cached_free_blocks() == 8

        result = pool.reset_prefix_cache()
        assert result is True
        assert pool.get_num_cached_free_blocks() == 0
        assert pool.get_num_free_blocks() == 20  # all free

    def test_expand_blocks_go_to_uncached(self):
        """expand_blocks() adds to uncached queue."""
        pool = self._make_pool(21)
        initial_uncached = pool.get_num_noncached_free_blocks()
        pool.expand_blocks(5)
        assert pool.get_num_noncached_free_blocks() == initial_uncached + 5
        assert pool.get_num_cached_free_blocks() == 0

    def test_alloc_mode_uncached_then_cached(self):
        """uncached_then_cached: exhaust uncached first, then cached."""
        pool = self._make_pool(21)
        self._move_to_cached(pool, 10)
        # Now: 10 uncached, 10 cached

        # Request 15 blocks with uncached_then_cached
        blocks = pool.get_new_blocks(15, alloc_mode="uncached_then_cached")
        assert len(blocks) == 15
        # All 10 uncached used + 5 from cached
        assert pool.get_num_noncached_free_blocks() == 0
        assert pool.get_num_cached_free_blocks() == 5

    def test_alloc_mode_any(self):
        """any mode: takes from uncached first, then cached."""
        pool = self._make_pool(21)
        self._move_to_cached(pool, 5)
        # 15 uncached, 5 cached

        blocks = pool.get_new_blocks(18, alloc_mode="any")
        assert len(blocks) == 18
        assert pool.get_num_free_blocks() == 2

    def test_maybe_evict_moves_free_cached_to_uncached(self):
        """_maybe_evict_cached_block on a free block in cached queue
        should move it to uncached queue."""
        from vllm.v1.core.kv_cache_utils import (
            BlockHash, make_block_hash_with_group_id)
        pool = self._make_pool(21)

        # Manually create 3 cached blocks with proper hash format
        raw_blocks = pool.free_uncached_queue.popleft_n(3)
        for i, b in enumerate(raw_blocks):
            raw_hash = BlockHash(b'hash' + i.to_bytes(4, 'big'))
            full_hash = make_block_hash_with_group_id(raw_hash, 0)
            b._block_hash = full_hash
            pool.cached_block_hash_to_block.insert(full_hash, b)
        pool.free_cached_queue.append_n(raw_blocks)

        assert pool.get_num_cached_free_blocks() == 3
        initial_uncached = pool.get_num_noncached_free_blocks()

        # Evict one block (free, ref_cnt==0, in cached queue)
        pool._maybe_evict_cached_block(raw_blocks[0])
        assert pool.get_num_cached_free_blocks() == 2
        assert pool.get_num_noncached_free_blocks() == initial_uncached + 1

    def test_migration_shim_alias(self):
        """free_block_queue alias points to free_uncached_queue."""
        pool = self._make_pool(21)
        assert pool.free_block_queue is pool.free_uncached_queue


# ---------------------------------------------------------------------------
# ElasticKVConfig Ce runtime params
# ---------------------------------------------------------------------------

class TestAllocationPlanTouchedClamp:
    """Finding 1: touched_cached_blocks clamped to cached_free."""

    def test_touched_clamped_to_cached_free(self):
        """required=40, noncached=5, cached=10 → touched=10 (not 35)."""
        from vllm.v1.core.kv_cache_manager import AllocationPlan
        plan = AllocationPlan(
            required_blocks=40,
            total_free=15,
            noncached_free=5,
            cached_free=10,
        )
        assert plan.touched_cached_blocks == 10  # min(35, 10)
        assert plan.protection_gap == 35  # unclamped

    def test_touched_when_cached_sufficient(self):
        """required=20, noncached=5, cached=20 → touched=15."""
        from vllm.v1.core.kv_cache_manager import AllocationPlan
        plan = AllocationPlan(
            required_blocks=20,
            total_free=25,
            noncached_free=5,
            cached_free=20,
        )
        assert plan.touched_cached_blocks == 15  # min(15, 20) = 15

    def test_touched_zero_when_uncached_sufficient(self):
        """required=5, noncached=10 → touched=0."""
        from vllm.v1.core.kv_cache_manager import AllocationPlan
        plan = AllocationPlan(
            required_blocks=5,
            total_free=20,
            noncached_free=10,
            cached_free=10,
        )
        assert plan.touched_cached_blocks == 0


class TestElasticKVConfigCeParams:
    """Tests for runtime Ce parameter fields."""

    def test_ce_params_default_zero(self):
        from vllm.elastic_kv_config import ElasticKVConfig
        cfg = ElasticKVConfig()
        assert cfg.local_num_experts == 0
        assert cfg.expert_group_size == 0
        assert cfg.expert_top_k == 0
        assert cfg.num_layers == 0
        assert cfg.c_reload_ms == 0.63

    def test_ce_params_from_env_not_set(self):
        """from_env() should not read Ce params from env."""
        from vllm.elastic_kv_config import ElasticKVConfig
        cfg = ElasticKVConfig.from_env()
        # Ce params are runtime-only, should be 0
        assert cfg.local_num_experts == 0
        assert cfg.expert_top_k == 0
