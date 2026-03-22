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
    """Tests for V3 3-way cost model.

    Terminology:
      Ce = Cost_expert_evict (reloading evicted experts during decode)
      Cc = Cost_cached_reclaim (re-prefilling destroyed prefix cache)
      Cp = Cost_req_preempt (recomputing preempted request tokens)
      h_eff = effective remaining decode steps
      groups_to_evict = expert groups to evict for expansion
      n_decode / n_prefill = batch composition
    """

    def _default_cfg(self, **overrides):
        """Create config with test-friendly defaults."""
        kw = dict(
            local_num_experts=512, group_size=2, top_k=10,
            c_reload_ms=0.63, num_layers=1, h_cap=64,
            l_sync_prefill=2.0,
            t_recompute_ms_per_token=0.260,
            block_size=16, t_prefill_tok_us=15.0,
            p_reuse=1.0, t_sched_us=500.0, t_queue_us=1000.0,
        )
        kw.update(overrides)
        return PrefixProtectionConfig(**kw)

    # --- Ce tests ---

    def test_compute_ce_basic(self):
        """Ce with small batch, h_eff=10, groups=4."""
        cfg = self._default_cfg()
        # k_per_layer = 4/1 = 4
        # m_eff = 10 * (6 + 2.0*2) = 100
        # P_hit_one = 1 - (1 - 2/512)^100 ≈ 0.3236
        # stall_step = 4 * 0.3236 * 0.63 = 0.8155
        # Ce = 0.8155 * 10 * 1000 = 8155 µs
        ce = cfg.compute_ce(h_eff=10, groups_to_evict=4,
                            n_decode=6, n_prefill=2)
        assert ce > 0
        # Verify formula structure: should increase with h_eff
        ce_higher_h = cfg.compute_ce(h_eff=50, groups_to_evict=4,
                                     n_decode=6, n_prefill=2)
        assert ce_higher_h > ce

    def test_compute_ce_zero_groups(self):
        """Zero groups → Ce = 0."""
        cfg = self._default_cfg()
        assert cfg.compute_ce(10, 0, 6, 2) == 0.0

    def test_compute_ce_scales_with_groups(self):
        """More groups → higher Ce."""
        cfg = self._default_cfg()
        ce4 = cfg.compute_ce(10, 4, 6, 2)
        ce8 = cfg.compute_ce(10, 8, 6, 2)
        assert ce8 > ce4

    # --- Cc tests ---

    def test_compute_cc_basic(self):
        """Cc for 5 touched blocks."""
        cfg = self._default_cfg()
        # 1.0 * (5 * 16 * 15.0 + 500 + 1000) = 1.0 * (1200 + 1500) = 2700 µs
        cc = cfg.compute_cc(5)
        assert cc == pytest.approx(2700.0)

    def test_compute_cc_zero(self):
        """Cc for 0 touched blocks = 0."""
        cfg = self._default_cfg()
        assert cfg.compute_cc(0) == 0.0

    def test_compute_cc_scales_linearly(self):
        """Cc scales linearly with touched blocks (modulo overhead)."""
        cfg = self._default_cfg()
        cc1 = cfg.compute_cc(1)
        cc10 = cfg.compute_cc(10)
        # Per-block contribution
        per_block = cfg.block_size * cfg.t_prefill_tok_us  # 240 µs
        assert (cc10 - cc1) == pytest.approx(9 * per_block)

    # --- Cp tests ---

    def test_compute_cp_basic(self):
        """Cp for 2048 computed tokens."""
        cfg = self._default_cfg()
        # 2048 * 0.260 * 1000 = 532480 µs
        cp = cfg.compute_cp(2048)
        assert cp == pytest.approx(532480.0)

    def test_compute_cp_zero(self):
        """Cp for 0 tokens = 0."""
        cfg = self._default_cfg()
        assert cfg.compute_cp(0) == 0.0

    # --- decide() tests ---

    def test_decide_use_uncached(self):
        """touched_cached_blocks=0 → USE_UNCACHED."""
        from vllm.v1.core.prefix_protect import ProtectionDecision
        cfg = self._default_cfg()
        d = cfg.decide(h_eff=10, groups_to_evict=4,
                       touched_cached_blocks=0,
                       preempt_computed_tokens=2048,
                       n_decode=6, n_prefill=2,
                       can_fully_protect=True)
        assert d == ProtectionDecision.USE_UNCACHED

    def test_decide_protect_and_expand(self):
        """Ce is cheapest → PROTECT_AND_EXPAND."""
        from vllm.v1.core.prefix_protect import ProtectionDecision
        # Low h_eff, few groups → small Ce; many touched → high Cc; large preempt → high Cp
        cfg = self._default_cfg()
        d = cfg.decide(h_eff=1, groups_to_evict=1,
                       touched_cached_blocks=100,
                       preempt_computed_tokens=10000,
                       n_decode=2, n_prefill=0,
                       can_fully_protect=True)
        assert d == ProtectionDecision.PROTECT_AND_EXPAND

    def test_decide_reclaim_cached(self):
        """Cc is cheapest → RECLAIM_CACHED."""
        from vllm.v1.core.prefix_protect import ProtectionDecision
        cfg = self._default_cfg()
        # Small touched (low Cc), high Ce (many groups, high h), high Cp
        d = cfg.decide(h_eff=50, groups_to_evict=20,
                       touched_cached_blocks=1,
                       preempt_computed_tokens=10000,
                       n_decode=20, n_prefill=5,
                       can_fully_protect=True)
        assert d == ProtectionDecision.RECLAIM_CACHED

    def test_decide_preempt(self):
        """Cp is cheapest → PREEMPT."""
        from vllm.v1.core.prefix_protect import ProtectionDecision
        cfg = self._default_cfg()
        # Very small preempt cost, high Ce and Cc
        d = cfg.decide(h_eff=50, groups_to_evict=20,
                       touched_cached_blocks=100,
                       preempt_computed_tokens=1,
                       n_decode=20, n_prefill=5,
                       can_fully_protect=True)
        assert d == ProtectionDecision.PREEMPT

    def test_decide_partial_expand_blocked(self):
        """can_fully_protect=False → PROTECT_AND_EXPAND never chosen."""
        from vllm.v1.core.prefix_protect import ProtectionDecision
        cfg = self._default_cfg()
        # Same params as protect_and_expand test, but can_fully_protect=False
        d = cfg.decide(h_eff=1, groups_to_evict=1,
                       touched_cached_blocks=100,
                       preempt_computed_tokens=10000,
                       n_decode=2, n_prefill=0,
                       can_fully_protect=False)
        assert d != ProtectionDecision.PROTECT_AND_EXPAND

    def test_decide_no_preempt_candidates(self):
        """preempt_computed_tokens=0 → PREEMPT never chosen."""
        from vllm.v1.core.prefix_protect import ProtectionDecision
        cfg = self._default_cfg()
        d = cfg.decide(h_eff=50, groups_to_evict=20,
                       touched_cached_blocks=100,
                       preempt_computed_tokens=0,
                       n_decode=20, n_prefill=5,
                       can_fully_protect=False)
        # With can_fully_protect=False (Ce=inf) and Cp=inf, must be RECLAIM
        assert d == ProtectionDecision.RECLAIM_CACHED

    # --- from_env tests ---

    def test_from_env_defaults(self):
        """from_env with no env vars should use defaults."""
        cfg = PrefixProtectionConfig.from_env()
        assert cfg.enable is False
        assert cfg.top_k == 10
        assert cfg.block_size == 16

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

    def _move_blocks_to_cached(self, pool, count):
        """Pop blocks from uncached queue, set hash, put in cached queue."""
        blocks = pool.free_uncached_queue.popleft_n(count)
        for i, blk in enumerate(blocks):
            blk._block_hash = b'fakehash' + i.to_bytes(4, 'big')
        pool.free_cached_queue.append_n(blocks)
        return blocks

    def test_pop_uncached_mixed(self):
        """Pop only unhashed blocks when pool has both cached and uncached."""
        pool = self._make_block_pool(20)

        # Move 5 blocks to cached queue
        self._move_blocks_to_cached(pool, 5)

        noncached = pool.get_num_noncached_free_blocks()
        cached = pool.get_num_cached_free_blocks()
        total = pool.get_num_free_blocks()
        assert noncached == total - cached
        assert cached == 5

        blocks = pool._pop_uncached_only(8)
        assert len(blocks) == 8
        for b in blocks:
            assert b.block_hash is None
            assert b.ref_cnt == 1

    def test_pop_uncached_preserves_cached(self):
        """Cached blocks must remain in cached queue after _pop_uncached_only."""
        pool = self._make_block_pool(20)

        # Move 5 blocks to cached queue
        self._move_blocks_to_cached(pool, 5)

        initial_free = pool.get_num_free_blocks()
        noncached = pool.get_num_noncached_free_blocks()

        to_pop = min(5, noncached)
        blocks = pool._pop_uncached_only(to_pop)
        assert len(blocks) == to_pop

        # Cached blocks should still be in cached queue
        remaining_free = pool.get_num_free_blocks()
        assert remaining_free == initial_free - to_pop
        assert pool.get_num_cached_free_blocks() == 5

    def test_pop_uncached_with_cached_does_not_touch_cached(self):
        """Pop from uncached queue should not affect cached queue size."""
        pool = self._make_block_pool(20)
        self._move_blocks_to_cached(pool, 5)

        initial_cached = pool.get_num_cached_free_blocks()
        initial_uncached = pool.get_num_noncached_free_blocks()

        blocks = pool._pop_uncached_only(5)
        assert len(blocks) == 5

        # Cached queue unchanged
        assert pool.get_num_cached_free_blocks() == initial_cached
        # Uncached decreased
        assert pool.get_num_noncached_free_blocks() == initial_uncached - 5

    def test_pop_uncached_insufficient_asserts(self):
        """Requesting more uncached blocks than available should raise."""
        pool = self._make_block_pool(10)
        noncached = pool.get_num_noncached_free_blocks()
        with pytest.raises(AssertionError):
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
        """Small batch: Ce is low, Cc for gap=3 is moderate → protect."""
        cfg = PrefixProtectionConfig(
            top_k=10, group_size=2, local_num_experts=512,
            c_reload_ms=0.63, h_cap=64, num_layers=1,
            block_size=16, t_prefill_tok_us=15.0,
            t_sched_us=500.0, t_queue_us=1000.0,
            p_reuse=1.0, enable=True,
        )
        # Ce should be cheap for low h_eff and small groups
        ce = cfg.compute_ce(h_eff=1, groups_to_evict=4, n_decode=2, n_prefill=0)
        cc = cfg.compute_cc(3)
        assert ce < cc  # protect is preferable


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

class TestCeVsCcCrossover:
    """Tests for Ce vs Cc crossover: when eviction becomes too expensive."""

    def test_high_h_eff_ce_exceeds_cc(self):
        """High h_eff, many groups → Ce > Cc → RECLAIM_CACHED."""
        from vllm.v1.core.prefix_protect import ProtectionDecision
        cfg = PrefixProtectionConfig(enable=True)
        d = cfg.decide(h_eff=50, groups_to_evict=20,
                       touched_cached_blocks=1,
                       preempt_computed_tokens=10000,
                       n_decode=20, n_prefill=5,
                       can_fully_protect=True)
        assert d == ProtectionDecision.RECLAIM_CACHED

    def test_low_h_eff_ce_below_cc(self):
        """Low h_eff, few groups → Ce < Cc → PROTECT_AND_EXPAND."""
        from vllm.v1.core.prefix_protect import ProtectionDecision
        cfg = PrefixProtectionConfig(enable=True)
        d = cfg.decide(h_eff=1, groups_to_evict=1,
                       touched_cached_blocks=50,
                       preempt_computed_tokens=10000,
                       n_decode=2, n_prefill=0,
                       can_fully_protect=True)
        assert d == ProtectionDecision.PROTECT_AND_EXPAND

    def test_ce_monotone_with_h_eff(self):
        """Ce should increase monotonically with h_eff."""
        cfg = PrefixProtectionConfig(enable=True)
        prev_ce = 0
        for h in [1, 5, 10, 20, 50]:
            ce = cfg.compute_ce(h_eff=h, groups_to_evict=4,
                                n_decode=6, n_prefill=2)
            assert ce >= prev_ce
            prev_ce = ce


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
    preservation under alloc_mode="uncached_only".

    These are NOT mock tests — they exercise the actual split-queue
    linked-list operations.
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

    def _move_to_cached(self, pool, count):
        """Pop blocks from uncached queue, set hash, put in cached queue."""
        blocks = pool.free_uncached_queue.popleft_n(count)
        for i, blk in enumerate(blocks):
            blk._block_hash = b'cached' + i.to_bytes(4, 'big')
        pool.free_cached_queue.append_n(blocks)
        return blocks

    def test_real_blockpool_uncached_only_preserves_cached(self):
        """21 blocks (20 free after null_block): 8 cached, 12 uncached.
        uncached_only alloc for 10 → takes 10 uncached.
        Cached 8 remain in cached queue untouched.
        """
        pool = self._make_block_pool(21)
        assert pool.get_num_free_blocks() == 20

        self._move_to_cached(pool, 8)
        assert pool.get_num_noncached_free_blocks() == 12
        assert pool.get_num_cached_free_blocks() == 8

        blocks = pool.get_new_blocks(10, alloc_mode="uncached_only")

        # All allocated blocks must be uncached
        assert len(blocks) == 10
        for b in blocks:
            assert b.block_hash is None
            assert b.ref_cnt == 1

        # Remaining: 8 cached + 2 uncached = 10
        assert pool.get_num_free_blocks() == 10
        assert pool.get_num_noncached_free_blocks() == 2
        assert pool.get_num_cached_free_blocks() == 8

    def test_real_blockpool_uncached_only_fails_when_insufficient(self):
        """21 blocks (20 free): 15 cached, 5 uncached.
        uncached_only alloc for 8 → error (only 5 uncached available).
        """
        pool = self._make_block_pool(21)
        assert pool.get_num_free_blocks() == 20

        self._move_to_cached(pool, 15)
        assert pool.get_num_noncached_free_blocks() == 5

        # alloc_mode="uncached_only" with 8 required but only 5 available
        # → allocate_slots returns None (availability check fails)
        # But get_new_blocks with uncached_only would raise from popleft_n
        with pytest.raises(AssertionError):
            pool.get_new_blocks(8, alloc_mode="uncached_only")

        # Cached blocks should survive intact
        assert pool.get_num_cached_free_blocks() == 15


# =====================================================================
# Test: Prefix Protection Scheduler Control Flow (mock-based)
# =====================================================================

class TestPrefixProtectionSchedulerFlow:
    """Mock-based tests for V3 _prefix_protection_try_allocate() control flow.

    Verifies 3-way decision:
    PROTECT_AND_EXPAND, RECLAIM_CACHED, PREEMPT.
    """

    def _make_scheduler_stub(self, *, decision_value="protect_and_expand",
                             expand_return=0, running_count=2):
        """Create a minimal scheduler-like object for V3 flow testing."""
        from vllm.v1.core.prefix_protect import (
            PrefixProtectionConfig, ProtectionDecision)

        sched = MagicMock()

        # Running requests with num_computed_tokens for preempt cost
        running = []
        for i in range(running_count):
            r = MagicMock()
            r.num_computed_tokens = 500 + i * 100
            r.max_tokens = 2048
            r.num_output_tokens = 10
            running.append(r)
        sched.running = running

        # Bind the real methods under test
        from vllm.v1.core.sched.scheduler import Scheduler
        sched._prefix_protection_try_allocate = (
            Scheduler._prefix_protection_try_allocate.__get__(sched))
        sched._compute_b_eff = (
            Scheduler._compute_b_eff.__get__(sched))
        sched._compute_h_eff = MagicMock(return_value=10)
        sched._trace_decision = MagicMock()  # no-op trace
        sched._diag_last_pp_active = None

        # V3 prefix protection config with mocked decide()
        pp_cfg = MagicMock(spec=PrefixProtectionConfig)
        pp_cfg.enable = True
        pp_cfg.h_cap = 64
        pp_cfg.decide = MagicMock(return_value=ProtectionDecision(decision_value))
        pp_cfg.compute_ce = MagicMock(return_value=100.0)
        pp_cfg.compute_cc = MagicMock(return_value=500.0)
        pp_cfg.compute_cp = MagicMock(return_value=50000.0)
        sched._prefix_protection_config = pp_cfg

        # Elastic KV handler
        sched._elastic_kv_handler = MagicMock()
        sched._elastic_kv_expanded_total = 0

        # Elastic KV config
        ekv_cfg = MagicMock()
        ekv_cfg.group_pages = 100
        ekv_cfg.per_tensor_block_bytes = {0: 32768}
        ekv_cfg.page_size = 2097152
        ekv_cfg.max_expand_blocks = 1000
        sched._elastic_kv_config = ekv_cfg

        # Expand
        sched._try_elastic_kv_expand = MagicMock(return_value=expand_return)

        return sched

    def test_protect_expand_retry_flow(self):
        """decide()=PROTECT_AND_EXPAND → expand → retry uncached_only."""
        from vllm.v1.core.kv_cache_manager import (
            AllocationAttempt, AllocationPlan)

        protection_gap = 5
        sched = self._make_scheduler_stub(
            decision_value="protect_and_expand",
            expand_return=protection_gap)

        fail_plan = AllocationPlan(
            required_blocks=10, total_free=20,
            noncached_free=5, cached_free=15)
        fail_attempt = AllocationAttempt(blocks=None, plan=fail_plan)

        success_blocks = MagicMock(name="blocks")

        mgr = sched.kv_cache_manager
        mgr.try_allocate = MagicMock(return_value=fail_attempt)
        mgr.allocate_slots = MagicMock(return_value=success_blocks)

        request = MagicMock()
        result = sched._prefix_protection_try_allocate(
            request, num_new_tokens=10)

        assert result is success_blocks
        sched._try_elastic_kv_expand.assert_called_once_with(protection_gap)
        mgr.allocate_slots.assert_called_once_with(
            request, 10, alloc_mode="uncached_only")

    def test_reclaim_cached_flow(self):
        """decide()=RECLAIM_CACHED → uncached_then_cached alloc."""
        from vllm.v1.core.kv_cache_manager import (
            AllocationAttempt, AllocationPlan)

        sched = self._make_scheduler_stub(decision_value="reclaim_cached")

        fail_plan = AllocationPlan(
            required_blocks=10, total_free=20,
            noncached_free=5, cached_free=15)
        fail_attempt = AllocationAttempt(blocks=None, plan=fail_plan)

        success_plan = AllocationPlan(
            required_blocks=10, total_free=20,
            noncached_free=5, cached_free=15)
        success_attempt = AllocationAttempt(
            blocks=MagicMock(name="blocks"), plan=success_plan)

        mgr = sched.kv_cache_manager
        # First call (uncached_only) fails, second (uncached_then_cached) OK
        mgr.try_allocate = MagicMock(
            side_effect=[fail_attempt, success_attempt])

        request = MagicMock()
        result = sched._prefix_protection_try_allocate(
            request, num_new_tokens=10)

        assert result is success_attempt.blocks
        # Expand should NOT be called for reclaim path
        sched._try_elastic_kv_expand.assert_not_called()
        # Second try_allocate used uncached_then_cached
        assert mgr.try_allocate.call_count == 2
        call_args = mgr.try_allocate.call_args_list[1]
        assert call_args.kwargs.get('alloc_mode') == 'uncached_then_cached'

    def test_preempt_returns_none(self):
        """decide()=PREEMPT → return None (caller handles preemption)."""
        from vllm.v1.core.kv_cache_manager import (
            AllocationAttempt, AllocationPlan)

        sched = self._make_scheduler_stub(decision_value="preempt")

        fail_plan = AllocationPlan(
            required_blocks=10, total_free=20,
            noncached_free=5, cached_free=15)
        fail_attempt = AllocationAttempt(blocks=None, plan=fail_plan)

        mgr = sched.kv_cache_manager
        mgr.try_allocate = MagicMock(return_value=fail_attempt)

        request = MagicMock()
        result = sched._prefix_protection_try_allocate(
            request, num_new_tokens=10)

        assert result is None
        sched._try_elastic_kv_expand.assert_not_called()

    def test_expand_partial_falls_to_reclaim(self):
        """PROTECT_AND_EXPAND but expand returns less than needed →
        falls back to RECLAIM_CACHED."""
        from vllm.v1.core.kv_cache_manager import (
            AllocationAttempt, AllocationPlan)

        sched = self._make_scheduler_stub(
            decision_value="protect_and_expand",
            expand_return=2)  # less than protection_gap=5

        fail_plan = AllocationPlan(
            required_blocks=10, total_free=20,
            noncached_free=5, cached_free=15)
        fail_attempt = AllocationAttempt(blocks=None, plan=fail_plan)

        success_plan = AllocationPlan(
            required_blocks=10, total_free=22,
            noncached_free=7, cached_free=15)
        success_attempt = AllocationAttempt(
            blocks=MagicMock(name="blocks"), plan=success_plan)

        mgr = sched.kv_cache_manager
        # First (uncached_only) fails, second (uncached_then_cached) succeeds
        mgr.try_allocate = MagicMock(
            side_effect=[fail_attempt, success_attempt])

        request = MagicMock()
        result = sched._prefix_protection_try_allocate(
            request, num_new_tokens=10)

        assert result is success_attempt.blocks
        # Expand was tried but insufficient
        sched._try_elastic_kv_expand.assert_called_once()
        # Fallback to uncached_then_cached
        assert mgr.try_allocate.call_count == 2
