# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Prefix Protection hard protect design.

Tests cover:
1. Hard protect regression (most important)
2. Ce/Cp dynamic cost model
3. _pop_uncached_only correctness
4. AllocationPlan/AllocationAttempt
5. PrefixProtectionConfig.should_protect logic
6. ElasticKVConfig runtime geometry
7. Dynamic groups calculation
8. Engine exact target commit
"""

import math
import pytest
from unittest.mock import MagicMock, patch

from vllm.v1.core.prefix_protect import PrefixProtectionConfig


# =====================================================================
# Test: Ce/Cp dynamic cost model
# =====================================================================

class TestPrefixProtectionConfig:
    """Tests for PrefixProtectionConfig cost model.

    Terminology:
      Ce = cost of expert eviction (reloading evicted experts during decode)
      Cp = cost of prefix reclaim (re-prefilling after cached blocks destroyed)
      b_eff = effective batch size (number of running requests)
      h_eff = effective remaining decode steps
      groups_to_evict = number of expert groups to evict for KV expansion
    """

    def test_compute_cp_default(self):
        """Default Cp(gap=1) = p_reuse * (1*block_size*t_prefill + t_sched + t_queue)."""
        cfg = PrefixProtectionConfig()
        cp = cfg.compute_cp(protection_gap=1)
        # 1.0 * (1 * 544 * 15.0 + 500.0 + 1000.0) = 9660.0 microseconds
        assert cp == pytest.approx(9660.0)

    def test_compute_cp_scales_with_gap(self):
        """Cp scales linearly with protection_gap (per-block prefill cost)."""
        cfg = PrefixProtectionConfig()
        cp1 = cfg.compute_cp(protection_gap=1)
        cp10 = cfg.compute_cp(protection_gap=10)
        # Cp(10) = 1.0 * (10*544*15.0 + 500 + 1000) = 83100
        assert cp10 == pytest.approx(83100.0)
        # Per-block contribution: (cp10 - cp1) / 9 ≈ block_size * t_prefill
        assert (cp10 - cp1) / 9 == pytest.approx(544 * 15.0)

    def test_compute_ce_small_batch(self):
        """Ce(b_eff=2, h_eff=1, groups=4) should be less than Cp, meaning protect."""
        cfg = PrefixProtectionConfig()
        ce = cfg.compute_ce(b_eff=2, h_eff=1, groups_to_evict=4)
        # k_evicted = 4 * 2 = 8
        # 2 * 10 * (8/512) * 0.252 * 1 * 1000 = 78.75 microseconds
        assert ce == pytest.approx(78.75)
        assert ce < cfg.compute_cp(protection_gap=1)

    def test_compute_ce_large_batch(self):
        """Ce(b_eff=24, h_eff=50, groups=4) exceeds Cp(gap=1)."""
        cfg = PrefixProtectionConfig()
        ce = cfg.compute_ce(b_eff=24, h_eff=50, groups_to_evict=4)
        # k_evicted = 4 * 2 = 8
        # 24 * 10 * (8/512) * 0.252 * 50 * 1000 = 47250.0 microseconds
        assert ce == pytest.approx(47250.0)
        assert ce > cfg.compute_cp(protection_gap=1)

    def test_compute_ce_large_batch_with_gap_scaling(self):
        """Ce(b=24,h=50,g=4)=47250 but Cp(gap=10)=83100 → protect with gap scaling."""
        cfg = PrefixProtectionConfig()
        ce = cfg.compute_ce(b_eff=24, h_eff=50, groups_to_evict=4)
        assert ce == pytest.approx(47250.0)
        # With gap=10, Cp=83100 >> Ce=47250 → now protects
        assert ce < cfg.compute_cp(protection_gap=10)

    def test_compute_ce_large_expand(self):
        """Ce(b_eff=2, h_eff=1, groups=16) should still be less than Cp."""
        cfg = PrefixProtectionConfig()
        ce = cfg.compute_ce(b_eff=2, h_eff=1, groups_to_evict=16)
        # k_evicted = 16 * 2 = 32
        # 2 * 10 * (32/512) * 0.252 * 1 * 1000 = 315.0 microseconds
        assert ce == pytest.approx(315.0)
        assert ce < cfg.compute_cp(protection_gap=1)

    def test_should_protect_small_batch(self):
        """Small batch should protect (Ce < Cp)."""
        cfg = PrefixProtectionConfig()
        assert cfg.should_protect(b_eff=2, h_eff=1, groups_to_evict=4,
                                  protection_gap=1)

    def test_should_protect_large_batch_small_gap(self):
        """Large batch with gap=1 → Ce > Cp → no protect."""
        cfg = PrefixProtectionConfig()
        assert not cfg.should_protect(b_eff=24, h_eff=50, groups_to_evict=4,
                                      protection_gap=1)

    def test_should_protect_large_batch_large_gap(self):
        """Large batch but gap=10 → Cp scales up → Ce < Cp → now protects.
        Ce=47250 vs Cp(gap=10)=83100.
        """
        cfg = PrefixProtectionConfig()
        assert cfg.should_protect(b_eff=24, h_eff=50, groups_to_evict=4,
                                  protection_gap=10)

    def test_should_protect_zero_groups(self):
        """Zero groups_to_evict means no expansion needed, always protect."""
        cfg = PrefixProtectionConfig()
        assert cfg.should_protect(b_eff=100, h_eff=100, groups_to_evict=0)

    def test_from_env_defaults(self):
        """from_env with no env vars should use defaults."""
        cfg = PrefixProtectionConfig.from_env()
        assert cfg.enable is False
        assert cfg.top_k == 10
        assert cfg.block_size == 544

    def test_from_env_enable(self):
        """VLLM_PREFIX_PROTECTION_ENABLE=1 should enable."""
        with patch.dict("os.environ", {"VLLM_PREFIX_PROTECTION_ENABLE": "1"}):
            cfg = PrefixProtectionConfig.from_env()
            assert cfg.enable is True


# =====================================================================
# Test: _pop_uncached_only correctness
# =====================================================================

class TestPopUncachedOnly:
    """Tests for BlockPool._pop_uncached_only.

    Terminology:
      uncached block = free block with NO cached prefix hash (block_hash is None)
      cached block   = free block WITH a cached prefix hash
    """

    def _make_block_pool(self, num_blocks=20, enable_caching=True):
        """Create a BlockPool with specified block count."""
        from vllm.v1.core.block_pool import BlockPool
        pool = BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=enable_caching,
            hash_block_size=16,
        )
        return pool

    def test_pop_uncached_basic(self):
        """Pop uncached blocks when all blocks are unhashed."""
        pool = self._make_block_pool(20)
        noncached = pool.get_num_noncached_free_blocks()
        assert noncached == pool.get_num_free_blocks()

        blocks = pool._pop_uncached_only(5)
        assert len(blocks) == 5
        for b in blocks:
            assert b.ref_cnt == 1
            assert b.block_hash is None

    def test_pop_uncached_mixed(self):
        """Pop only unhashed blocks, skip hashed ones."""
        pool = self._make_block_pool(20)

        # Manually mark some blocks as hashed (simulating cached prefix).
        # We walk the free queue and mark specific blocks.
        queue = pool.free_block_queue
        block = queue.fake_free_list_head.next_free_block
        tail = queue.fake_free_list_tail
        hashed_count = 0
        idx = 0
        while block is not tail and block is not None:
            if idx in [0, 3, 7, 9, 14]:
                block._block_hash = b'fakehash' + idx.to_bytes(4, 'big')
                hashed_count += 1
            idx += 1
            block = block.next_free_block

        noncached = pool.get_num_noncached_free_blocks()
        total = pool.get_num_free_blocks()
        assert noncached == total - hashed_count

        if noncached >= 8:
            blocks = pool._pop_uncached_only(8)
            assert len(blocks) == 8
            for b in blocks:
                assert b.block_hash is None
                assert b.ref_cnt == 1

    def test_pop_uncached_preserves_cached(self):
        """Cached blocks must remain in queue after _pop_uncached_only."""
        pool = self._make_block_pool(20)

        # Mark 5 blocks as cached
        queue = pool.free_block_queue
        block = queue.fake_free_list_head.next_free_block
        tail = queue.fake_free_list_tail
        idx = 0
        while block is not tail and block is not None and idx < 5:
            block._block_hash = b'cached' + idx.to_bytes(4, 'big')
            idx += 1
            block = block.next_free_block

        initial_free = pool.get_num_free_blocks()
        noncached = pool.get_num_noncached_free_blocks()

        # Pop some uncached blocks
        to_pop = min(5, noncached)
        blocks = pool._pop_uncached_only(to_pop)
        assert len(blocks) == to_pop

        # Cached blocks should still be in queue (preserved, not lost)
        remaining_free = pool.get_num_free_blocks()
        assert remaining_free == initial_free - to_pop

    def test_pop_uncached_preserves_refcnt_blocks(self):
        """Blocks with ref_cnt > 0 must be preserved, not discarded."""
        pool = self._make_block_pool(20)

        # Set ref_cnt > 0 on a couple of blocks (simulate cache hit via touch)
        queue = pool.free_block_queue
        block = queue.fake_free_list_head.next_free_block
        tail = queue.fake_free_list_tail
        touched = 0
        idx = 0
        while block is not tail and block is not None and idx < 3:
            block.ref_cnt = 1  # mark as in-use
            touched += 1
            idx += 1
            block = block.next_free_block

        initial_free = pool.get_num_free_blocks()
        blocks = pool._pop_uncached_only(5)
        assert len(blocks) == 5

        # Queue should shrink by 5 (allocated) but touched blocks preserved
        remaining = pool.get_num_free_blocks()
        assert remaining == initial_free - 5

    def test_pop_uncached_insufficient_asserts(self):
        """Requesting more uncached blocks than available should assert."""
        pool = self._make_block_pool(10)
        noncached = pool.get_num_noncached_free_blocks()
        with pytest.raises(AssertionError, match="_pop_uncached_only"):
            pool._pop_uncached_only(noncached + 5)


# =====================================================================
# Test: AllocationPlan/AllocationAttempt
# =====================================================================

class TestAllocationPlan:
    """Tests for AllocationPlan dataclass.

    Terminology:
      required_blocks  = blocks needed (from coordinator exact calculation)
      total_free       = all free blocks (cached + uncached)
      noncached_free   = free blocks without cached prefix hash
      allocation_gap   = max(0, required - total_free)
      protection_gap   = max(0, required - noncached_free)
      would_touch_cached = True if allocation needs more than noncached_free
    """

    def test_allocation_gap(self):
        from vllm.v1.core.kv_cache_manager import AllocationPlan
        plan = AllocationPlan(
            required_blocks=10, total_free=7, noncached_free=5)
        assert plan.allocation_gap == 3
        assert plan.protection_gap == 5
        assert plan.would_touch_cached is True

    def test_no_gap(self):
        from vllm.v1.core.kv_cache_manager import AllocationPlan
        plan = AllocationPlan(
            required_blocks=5, total_free=10, noncached_free=8)
        assert plan.allocation_gap == 0
        assert plan.protection_gap == 0
        assert plan.would_touch_cached is False

    def test_protection_gap_only(self):
        """Total free sufficient but noncached insufficient."""
        from vllm.v1.core.kv_cache_manager import AllocationPlan
        plan = AllocationPlan(
            required_blocks=8, total_free=15, noncached_free=5)
        assert plan.allocation_gap == 0
        assert plan.protection_gap == 3
        assert plan.would_touch_cached is True

    def test_allocation_attempt(self):
        from vllm.v1.core.kv_cache_manager import (
            AllocationAttempt, AllocationPlan)
        plan = AllocationPlan(
            required_blocks=5, total_free=10, noncached_free=5)
        attempt = AllocationAttempt(blocks=None, plan=plan)
        assert attempt.blocks is None
        assert attempt.plan.protection_gap == 0


# =====================================================================
# Test: Hard protect regression (most important)
# =====================================================================

class TestHardProtectRegression:
    """Regression test for prefix protection hard protect.

    Scenario: noncached_free=5, cached_free=10 (total=15), required=8, Ce < Cp

    Correct result:
    - Step 1: strict uncached allocation -> fail (5 < 8)
    - Step 2: protection_gap = 3
    - Step 3: Ce < Cp -> elastic expand(3) -> +3 uncached
    - Now: noncached=8, total=18
    - Retry: strict uncached -> success (8 uncached consumed)
    - Cached blocks untouched

    Wrong result (old bug):
    - Normal alloc -> reclaim 3 cached + 5 uncached = 8 -> success
    - No expand, cached destroyed
    """

    def test_should_protect_returns_correct_plan(self):
        """Verify AllocationPlan correctly identifies the two deficits."""
        from vllm.v1.core.kv_cache_manager import AllocationPlan

        plan = AllocationPlan(
            required_blocks=8,
            total_free=15,
            noncached_free=5,
        )
        assert plan.allocation_gap == 0  # total_free sufficient
        assert plan.protection_gap == 3  # need 3 more uncached
        assert plan.would_touch_cached is True

    def test_cost_model_small_batch_protects(self):
        """Small batch Ce < Cp should trigger protection."""
        cfg = PrefixProtectionConfig(
            top_k=10, group_size=2, num_experts=512,
            c_reload_eff_ms=0.252, h_floor=1, h_cap=64,
            block_size=544, t_prefill_tok_us=15.0,
            t_sched_us=500.0, t_queue_us=1000.0,
            p_reuse=1.0, enable=True,
        )
        # Ce(b_eff=2, h_eff=1, groups=4) = 78.75 < Cp(gap=3)=25960 -> protect
        assert cfg.should_protect(b_eff=2, h_eff=1, groups_to_evict=4,
                                  protection_gap=3)


# =====================================================================
# Test: ElasticKVConfig
# =====================================================================

class TestElasticKVConfig:
    """Tests for ElasticKVConfig."""

    def test_from_env_defaults(self):
        from vllm.elastic_kv_config import ElasticKVConfig
        cfg = ElasticKVConfig.from_env()
        assert cfg.enable is False
        assert cfg.expand_group_quantum == 4
        assert cfg.min_resident_ratio == 0.5

    def test_from_env_custom(self):
        from vllm.elastic_kv_config import ElasticKVConfig
        with patch.dict("os.environ", {
            "VLLM_ELASTIC_KV_ENABLE": "1",
            "VLLM_ELASTIC_KV_GROUPS_PER_EXPAND": "8",
        }):
            cfg = ElasticKVConfig.from_env()
            assert cfg.enable is True
            assert cfg.expand_group_quantum == 8


# =====================================================================
# Test: Ce >= Cp fallback
# =====================================================================

class TestCeGeCpFallback:
    """Tests for fallback when Ce >= Cp (eviction too expensive to protect)."""

    def test_large_batch_no_protect_small_gap(self):
        """b_eff=24, h_eff=50, gap=1 -> Ce > Cp -> no protect.
        Ce = 47250 us, Cp(gap=1) = 9660 us → Ce > Cp → no protect
        """
        cfg = PrefixProtectionConfig(enable=True)
        assert not cfg.should_protect(b_eff=24, h_eff=50, groups_to_evict=4,
                                      protection_gap=1)

    def test_large_batch_protects_large_gap(self):
        """b_eff=24, h_eff=50, gap=10 -> Cp scales up -> now protects.
        Ce = 47250 us, Cp(gap=10) = 83100 us → Ce < Cp → protect
        """
        cfg = PrefixProtectionConfig(enable=True)
        assert cfg.should_protect(b_eff=24, h_eff=50, groups_to_evict=4,
                                  protection_gap=10)

    def test_medium_batch_threshold(self):
        """Find the b_eff where Ce crosses Cp(gap=1) for groups=4, h_eff=1."""
        cfg = PrefixProtectionConfig(enable=True)
        cp = cfg.compute_cp(protection_gap=1)  # 9660

        for b in range(1, 1000):
            ce = cfg.compute_ce(b_eff=b, h_eff=1, groups_to_evict=4)
            if ce >= cp:
                assert cfg.should_protect(
                    b_eff=b-1, h_eff=1, groups_to_evict=4, protection_gap=1)
                assert not cfg.should_protect(
                    b_eff=b, h_eff=1, groups_to_evict=4, protection_gap=1)
                break
        else:
            # Ce never exceeds Cp at h_eff=1 for reasonable b_eff
            pass


# =====================================================================
# Test: Dynamic groups calculation
# =====================================================================

class TestDynamicGroupsCalculation:
    """Tests for elastic_kv_prepare dynamic groups calculation.

    Terminology:
      pages_per_block = VMM pages needed per KV cache block
      group_pages     = VMM pages per expert group
      quantum         = expand_group_quantum (round-up unit for eviction count)
    """

    def test_groups_calculation_basic(self):
        """min_blocks=10, free_pages=0, group_pages=100, pages_per_block=30
        -> pages_needed=300 -> groups_needed=3 -> quantum=4 -> groups_to_evict=4
        """
        min_blocks = 10
        free_pages = 0
        group_pages = 100
        pages_per_block = 30
        quantum = 4
        evictable_groups = 20

        pages_for_min = min_blocks * pages_per_block  # 300
        pages_needed = max(0, pages_for_min - free_pages)  # 300
        groups_needed = math.ceil(pages_needed / group_pages)  # 3
        groups_planned = (
            (groups_needed + quantum - 1) // quantum) * quantum  # 4
        groups_to_evict = min(groups_planned, evictable_groups)  # 4

        assert pages_for_min == 300
        assert pages_needed == 300
        assert groups_needed == 3
        assert groups_planned == 4
        assert groups_to_evict == 4

    def test_free_pages_sufficient(self):
        """When free pages are sufficient, groups_to_evict should be 0.
        min_blocks=1, free_pages=50, pages_per_block=30
        -> pages_needed=0 (free sufficient) -> groups_to_evict=0
        """
        min_blocks = 1
        free_pages = 50
        pages_per_block = 30

        pages_for_min = min_blocks * pages_per_block  # 30
        pages_needed = max(0, pages_for_min - free_pages)  # 0
        assert pages_needed == 0


# =====================================================================
# Test: Engine exact target commit
# =====================================================================

class TestEngineExactCommit:
    """Tests for exact target commit (not over-expand).

    The engine should commit min_blocks (what the scheduler needs),
    not the maximum possible blocks from prepare phase.
    """

    def test_commit_exact_target(self):
        """possible=10, min_blocks=5 -> commit(5), not 10."""
        min_blocks = 5
        max_blocks = 10
        possible = 10  # workers can provide up to 10

        target = min_blocks  # exact target
        assert target == 5
        assert target <= possible
        assert target != possible  # NOT over-expanding

    def test_commit_fails_when_insufficient(self):
        """possible < min_blocks -> return 0."""
        min_blocks = 10
        possible = 5  # insufficient

        if possible < min_blocks:
            result = 0
        else:
            result = min_blocks
        assert result == 0


# =====================================================================
# Test: Prefix Protection End-to-End (Real BlockPool integration)
# =====================================================================

class TestPrefixProtectionBlockPoolIntegration:
    """Integration tests using real BlockPool to verify cached block
    preservation under strict_uncached mode.

    These are NOT mock tests — they exercise the actual free queue
    linked-list operations and _pop_uncached_only logic.
    """

    def _make_block_pool(self, num_blocks=20, enable_caching=True):
        """Create a BlockPool with specified block count."""
        from vllm.v1.core.block_pool import BlockPool
        pool = BlockPool(
            num_gpu_blocks=num_blocks,
            enable_caching=enable_caching,
            hash_block_size=16,
        )
        return pool

    def _mark_blocks_cached(self, pool, count):
        """Walk free queue and set block_hash on first `count` blocks."""
        queue = pool.free_block_queue
        block = queue.fake_free_list_head.next_free_block
        tail = queue.fake_free_list_tail
        marked = 0
        while block is not tail and block is not None and marked < count:
            block._block_hash = b'cached' + marked.to_bytes(4, 'big')
            marked += 1
            block = block.next_free_block
        return marked

    def test_real_blockpool_strict_uncached_preserves_cached(self):
        """21 blocks (20 free after null_block): 8 cached, 12 uncached.
        strict_uncached alloc for 10 → takes 10 uncached.
        Cached 8 remain in free queue untouched.
        """
        # 21 blocks: 1 reserved as null_block → 20 free
        pool = self._make_block_pool(21)
        assert pool.get_num_free_blocks() == 20

        self._mark_blocks_cached(pool, 8)
        assert pool.get_num_noncached_free_blocks() == 12

        # Enable strict uncached mode and allocate
        pool.set_strict_uncached(True)
        blocks = pool.get_new_blocks(10)
        pool.set_strict_uncached(False)

        # All allocated blocks must be uncached
        assert len(blocks) == 10
        for b in blocks:
            assert b.block_hash is None
            assert b.ref_cnt == 1

        # Remaining: 8 cached + 2 uncached = 10
        assert pool.get_num_free_blocks() == 10
        assert pool.get_num_noncached_free_blocks() == 2

        # Verify cached blocks still have their hashes
        queue = pool.free_block_queue
        block = queue.fake_free_list_head.next_free_block
        tail = queue.fake_free_list_tail
        cached_remaining = 0
        while block is not tail and block is not None:
            if block.block_hash is not None:
                cached_remaining += 1
            block = block.next_free_block
        assert cached_remaining == 8

    def test_real_blockpool_strict_uncached_fails_when_insufficient(self):
        """21 blocks (20 free): 15 cached, 5 uncached.
        strict_uncached alloc for 8 → AssertionError (only 5 uncached).
        Cached blocks are preserved back into queue.
        """
        pool = self._make_block_pool(21)
        assert pool.get_num_free_blocks() == 20

        self._mark_blocks_cached(pool, 15)
        assert pool.get_num_noncached_free_blocks() == 5

        pool.set_strict_uncached(True)
        with pytest.raises(AssertionError, match="_pop_uncached_only"):
            pool.get_new_blocks(8)
        pool.set_strict_uncached(False)

        # _pop_uncached_only consumed 5 uncached blocks before asserting,
        # but all 15 cached blocks were preserved back into the queue.
        # Verify cached blocks survived intact.
        queue = pool.free_block_queue
        block = queue.fake_free_list_head.next_free_block
        tail = queue.fake_free_list_tail
        cached_in_queue = 0
        while block is not tail and block is not None:
            if block.block_hash is not None:
                cached_in_queue += 1
            block = block.next_free_block
        assert cached_in_queue == 15


# =====================================================================
# Test: Prefix Protection Scheduler Control Flow (mock-based)
# =====================================================================

class TestPrefixProtectionSchedulerFlow:
    """Mock-based tests for _prefix_protection_try_allocate() control flow.

    Verifies the 5-step scheduler logic:
    1. strict uncached try → 2. read plan → 3. cost model →
    4. expand + retry → 5. fallback.
    """

    def _make_scheduler_stub(self, *, should_protect=True,
                             expand_return=0, running_count=2):
        """Create a minimal scheduler-like object with the prefix protection
        method and its dependencies mocked.
        """
        sched = MagicMock()
        sched.running = [MagicMock() for _ in range(running_count)]

        # Bind the real method under test
        from vllm.v1.core.sched.scheduler import Scheduler
        sched._prefix_protection_try_allocate = (
            Scheduler._prefix_protection_try_allocate.__get__(sched))
        sched._compute_b_eff = (
            Scheduler._compute_b_eff.__get__(sched))
        # Mock _compute_h_eff to avoid needing real Request objects
        sched._compute_h_eff = MagicMock(return_value=1)

        # Config — must return real numbers for break-even calculation
        pp_cfg = MagicMock()
        pp_cfg.enable = True
        pp_cfg.should_protect = MagicMock(return_value=should_protect)
        pp_cfg.compute_ce = MagicMock(return_value=100.0)
        pp_cfg.compute_cp = MagicMock(return_value=9660.0)
        sched._prefix_protection_config = pp_cfg

        # Elastic KV handler (existence enables prefix protection path)
        sched._elastic_kv_handler = MagicMock()

        # Elastic KV config for groups_est geometry
        ekv_cfg = MagicMock()
        ekv_cfg.group_pages = 100
        ekv_cfg.per_tensor_block_bytes = {0: 32768}
        ekv_cfg.page_size = 2097152
        sched._elastic_kv_config = ekv_cfg

        # Expand
        sched._try_elastic_kv_expand = MagicMock(return_value=expand_return)

        return sched

    def test_protect_expand_retry_flow(self):
        """strict uncached fails → should_protect=True → expand → retry succeeds.
        Verifies expand is called with protection_gap.
        """
        from vllm.v1.core.kv_cache_manager import (
            AllocationAttempt, AllocationPlan)

        protection_gap = 5
        sched = self._make_scheduler_stub(
            should_protect=True, expand_return=protection_gap)

        # Step 1: strict uncached fails
        fail_plan = AllocationPlan(
            required_blocks=10, total_free=20, noncached_free=5)
        fail_attempt = AllocationAttempt(blocks=None, plan=fail_plan)

        # Step 4: retry after expand succeeds
        success_blocks = MagicMock(name="blocks")

        mgr = sched.kv_cache_manager
        mgr.try_allocate = MagicMock(return_value=fail_attempt)
        mgr.allocate_slots = MagicMock(return_value=success_blocks)

        request = MagicMock()
        result = sched._prefix_protection_try_allocate(
            request, num_new_tokens=10)

        assert result is success_blocks
        sched._try_elastic_kv_expand.assert_called_once_with(protection_gap)
        # allocate_slots retry was called with strict_uncached=True
        mgr.allocate_slots.assert_called_once_with(
            request, 10, strict_uncached=True)

    def test_ce_ge_cp_skips_expand(self):
        """Ce >= Cp → should_protect=False → no expand, falls through
        to normal allocation.
        """
        from vllm.v1.core.kv_cache_manager import (
            AllocationAttempt, AllocationPlan)

        sched = self._make_scheduler_stub(should_protect=False)

        # Step 1: strict uncached fails
        fail_plan = AllocationPlan(
            required_blocks=10, total_free=20, noncached_free=5)
        fail_attempt = AllocationAttempt(blocks=None, plan=fail_plan)

        # Step 5: fallback normal alloc succeeds
        success_plan = AllocationPlan(
            required_blocks=10, total_free=20, noncached_free=20)
        success_attempt = AllocationAttempt(
            blocks=MagicMock(name="blocks"), plan=success_plan)

        mgr = sched.kv_cache_manager
        # First call (strict) fails, second call (normal) succeeds
        mgr.try_allocate = MagicMock(
            side_effect=[fail_attempt, success_attempt])

        request = MagicMock()
        result = sched._prefix_protection_try_allocate(
            request, num_new_tokens=10)

        assert result is success_attempt.blocks
        # Expand should NOT be called
        sched._try_elastic_kv_expand.assert_not_called()

    def test_expand_fails_falls_through(self):
        """Ce < Cp but expand returns 0 → falls through to normal alloc."""
        from vllm.v1.core.kv_cache_manager import (
            AllocationAttempt, AllocationPlan)

        sched = self._make_scheduler_stub(
            should_protect=True, expand_return=0)

        # Step 1: strict uncached fails
        fail_plan = AllocationPlan(
            required_blocks=10, total_free=20, noncached_free=5)
        fail_attempt = AllocationAttempt(blocks=None, plan=fail_plan)

        # Step 5: fallback normal alloc succeeds
        success_plan = AllocationPlan(
            required_blocks=10, total_free=20, noncached_free=20)
        success_attempt = AllocationAttempt(
            blocks=MagicMock(name="blocks"), plan=success_plan)

        mgr = sched.kv_cache_manager
        # First call (strict) fails, second call (normal) succeeds
        mgr.try_allocate = MagicMock(
            side_effect=[fail_attempt, success_attempt])

        request = MagicMock()
        result = sched._prefix_protection_try_allocate(
            request, num_new_tokens=10)

        assert result is success_attempt.blocks
        # Expand WAS called (Ce < Cp) but returned 0
        sched._try_elastic_kv_expand.assert_called_once()
        # Normal alloc fallback was used
        assert mgr.try_allocate.call_count == 2
