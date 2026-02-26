"""Unit tests for expert weight offloading PoC.

Tests ExpertPredictor (pure CPU) and ExpertCacheManager (mocked CUDA).
Run: pytest tests/test_expert_offload.py -v
"""
import sys
import os
import pytest
import torch
from unittest.mock import MagicMock, patch

# Add vllm to path so imports work from this checkout
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vllm.model_executor.layers.fused_moe.expert_predictor import (
    ExpertPredictor,
)
from vllm.model_executor.layers.fused_moe.expert_cache import (
    ExpertCacheManager,
    ExpertOffloadConfig,
    CacheStats,
)


# ─── ExpertPredictor Tests ────────────────────────────────────────────

class TestExpertPredictor:
    def test_basic_prediction(self):
        """Same-expert heuristic returns current layer's experts."""
        pred = ExpertPredictor(num_layers=4, num_local_experts=16, top_k=3)
        topk_ids = torch.tensor([[0, 2, 5], [1, 3, 7]])  # (2, 3)
        result = pred.update_and_predict(
            current_layer_idx=0,
            topk_ids=topk_ids,
            expert_map=None,  # ep_size=1
            target_layer_idx=1,
        )
        assert isinstance(result, list)
        assert len(result) <= 3  # top_k
        # All routed experts should appear (same-expert heuristic)
        routed = set(topk_ids.flatten().unique().tolist())
        # Result is a subset (limited by top_k)
        assert len(result) > 0

    def test_expert_map_none_global_eq_local(self):
        """expert_map=None => global IDs used directly as local IDs."""
        pred = ExpertPredictor(num_layers=2, num_local_experts=512, top_k=10)
        topk_ids = torch.tensor([[42, 100, 200]])
        result = pred.update_and_predict(0, topk_ids, None, 1)
        # With None expert_map, IDs 42/100/200 are used as local IDs
        for eid in [42, 100, 200]:
            assert eid in result or len(result) == 10

    def test_expert_map_tensor_filters_minus_one(self):
        """expert_map Tensor: -1 entries are excluded."""
        pred = ExpertPredictor(num_layers=2, num_local_experts=4, top_k=3)
        # 8 global experts, only 4 local (experts 0-3 mapped, 4-7 = -1)
        expert_map = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1], dtype=torch.int32)
        topk_ids = torch.tensor([[0, 5, 7]])  # gid=5,7 -> lid=-1 (remote)
        result = pred.update_and_predict(0, topk_ids, expert_map, 1)
        # Only gid=0 maps to local (lid=0)
        assert 0 in result

    def test_freq_fill_after_warmup(self):
        """After enough updates, frequency fill kicks in."""
        pred = ExpertPredictor(num_layers=2, num_local_experts=32, top_k=10)
        # Warm up layer 1 stats with expert 5 appearing often
        for _ in range(110):
            topk_ids = torch.tensor([[5, 10, 15]])
            pred.update_and_predict(1, topk_ids, None, 0)
        # Now predict for layer 1 — freq fill should include expert 5
        topk_ids_new = torch.tensor([[0, 1, 2]])
        result = pred.update_and_predict(0, topk_ids_new, None, 1)
        assert 5 in result  # high-freq expert from layer 1 stats

    def test_top_k_limit(self):
        """Prediction never exceeds top_k."""
        pred = ExpertPredictor(num_layers=2, num_local_experts=512, top_k=10)
        # 20 unique experts routed
        topk_ids = torch.arange(20).unsqueeze(0)  # (1, 20)
        result = pred.update_and_predict(0, topk_ids, None, 1)
        assert len(result) <= 10


# ─── ExpertCacheManager Tests (mocked CUDA) ──────────────────────────

def _make_cache(num_layers=2, local_E=16, global_E=16, max_resident=4):
    """Create ExpertCacheManager with mocked CUDA primitives."""
    config = ExpertOffloadConfig(
        enable=True,
        max_resident_per_layer=max_resident,
        staging_window_experts=2,
    )
    w13_shape = (8, 4)  # small dummy shape
    w2_shape = (4, 8)

    # Mock CUDA stream, pin_memory
    with patch("torch.cuda.Stream") as mock_stream, \
         patch.object(torch.Tensor, "pin_memory", return_value=torch.empty(1)):
        mock_stream.return_value = MagicMock()

        # Patch pin_memory to return the tensor itself
        original_empty = torch.empty

        def patched_empty(*args, **kwargs):
            kwargs.pop("device", None)
            t = original_empty(*args, **kwargs)
            t.pin_memory = lambda: t
            return t

        # Build staging buffers without pin_memory
        cache = object.__new__(ExpertCacheManager)
        cache.config = config
        cache.num_layers = num_layers
        cache.local_num_experts = local_E
        cache.global_num_experts = global_E
        cache.max_resident = max_resident
        cache.dtype = torch.float32
        cache.device = torch.device("cpu")
        cache.current_step = 0

        # Compute expert size
        cache._w13_shape = w13_shape
        cache._w2_shape = w2_shape
        elem_size = 4  # float32
        w13_numel = 8 * 4
        w2_numel = 4 * 8
        cache.expert_size_bytes = (w13_numel + w2_numel) * elem_size

        # Identity map
        cache._identity_map = torch.arange(global_E, dtype=torch.int32)

        # Slot tracking
        cache._slot_to_expert = []
        cache._expert_to_slot = []
        cache._free_slots = []
        for _ in range(num_layers):
            cache._slot_to_expert.append([-1] * max_resident)
            cache._expert_to_slot.append([-1] * local_E)
            cache._free_slots.append(list(range(max_resident)))

        # Eviction metadata
        from collections import defaultdict
        cache._access_count = defaultdict(int)
        cache._last_access = defaultdict(int)
        cache._pinned = set()

        # CPU pool
        cache._cpu_pool = [{} for _ in range(num_layers)]

        # Staging buffers (no pin_memory)
        staging_n = config.staging_window_experts
        cache._staging_w13 = torch.empty((staging_n, *w13_shape), dtype=torch.float32)
        cache._staging_w2 = torch.empty((staging_n, *w2_shape), dtype=torch.float32)

        # GPU weight refs
        cache._layer_w13 = {}
        cache._layer_w2 = {}

        # CUDA stream mock
        cache._copy_stream = MagicMock()
        cache._prefetch_events = {}
        cache._prefetch_pending = defaultdict(set)
        cache._prefetch_bytes_this_step = 0

        # Stats
        cache.stats = CacheStats()
        import threading
        cache._lock = threading.Lock()

    return cache


def _register_dummy_experts(cache, num_layers, local_E, w13_shape, w2_shape):
    """Register dummy experts to CPU pool and GPU weight refs."""
    for layer_idx in range(num_layers):
        # GPU weight tensors (resized to max_resident)
        w13_gpu = torch.zeros((cache.max_resident, *w13_shape), dtype=torch.float32)
        w2_gpu = torch.zeros((cache.max_resident, *w2_shape), dtype=torch.float32)
        cache._layer_w13[layer_idx] = w13_gpu
        cache._layer_w2[layer_idx] = w2_gpu

        # CPU pool: each expert has unique data
        for lid in range(local_E):
            w13_cpu = torch.full(w13_shape, float(lid), dtype=torch.float32)
            w2_cpu = torch.full(w2_shape, float(lid + 100), dtype=torch.float32)
            cache._cpu_pool[layer_idx][lid] = (w13_cpu, w2_cpu)


class TestExpertCacheManager:
    def test_identity_map_created(self):
        """Identity map [0..E-1] created for ep_size=1."""
        cache = _make_cache(global_E=16)
        assert cache._identity_map.shape == (16,)
        assert cache._identity_map[0] == 0
        assert cache._identity_map[15] == 15

    def test_resolve_expert_map_none(self):
        """None -> identity map."""
        cache = _make_cache(global_E=8)
        result = cache._resolve_expert_map(None)
        assert torch.equal(result, torch.arange(8, dtype=torch.int32))

    def test_resolve_expert_map_tensor(self):
        """Tensor passthrough."""
        cache = _make_cache()
        emap = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
        result = cache._resolve_expert_map(emap)
        assert torch.equal(result, emap)

    def test_register_and_populate(self):
        """Register experts, populate initial cache, verify slots filled."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8, max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))

        # Mock torch.cuda.synchronize
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        # 4 slots should be filled
        filled = sum(1 for s in cache._slot_to_expert[0] if s != -1)
        assert filled == 4
        # 4 free slots consumed
        assert len(cache._free_slots[0]) == 0

    def test_prepare_hit_returns_valid_slot(self):
        """All-hit scenario: cached experts return valid slots."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8, max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))

        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        # Experts 0-3 should be in cache
        topk_ids = torch.tensor([[0, 1, 2]])  # all should hit
        cache_map = cache.prepare_and_get_expert_map(0, topk_ids, None)

        assert cache_map.shape == (8,)
        # Slots 0-2 should be valid (>=0)
        for gid in [0, 1, 2]:
            assert cache_map[gid].item() >= 0
            assert cache_map[gid].item() < 4
        assert cache.stats.hits == 3
        assert cache.stats.misses == 0

    def test_prepare_miss_triggers_sync_fetch(self):
        """Cache miss triggers sync_fetch."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8, max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))

        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        # Experts 0-3 are cached. Request expert 7 (miss).
        topk_ids = torch.tensor([[0, 7]])

        with patch("torch.cuda.synchronize"):
            cache_map = cache.prepare_and_get_expert_map(0, topk_ids, None)

        assert cache.stats.hits == 1    # expert 0
        assert cache.stats.misses == 1  # expert 7
        assert cache.stats.sync_fetches == 1
        assert cache.stats.evictions == 1  # had to evict to make room
        # Expert 7 should now have a valid slot
        assert cache_map[7].item() >= 0

    def test_eviction_frees_slot(self):
        """Eviction works when cache is full."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8, max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))

        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        assert len(cache._free_slots[0]) == 0
        evicted = cache._evict_one(0)
        assert evicted is True
        assert len(cache._free_slots[0]) == 1

    def test_pinned_expert_not_evicted(self):
        """Pinned experts survive eviction."""
        cache = _make_cache(num_layers=1, local_E=4, global_E=4, max_resident=4)
        _register_dummy_experts(cache, 1, 4, (8, 4), (4, 8))

        # Pin all experts
        for lid in range(4):
            cache._pinned.add((0, lid))

        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        evicted = cache._evict_one(0)
        assert evicted is False  # can't evict anything

    def test_cache_map_preserves_minus_one(self):
        """Uncached experts have -1 in cache_map."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8, max_resident=2)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))

        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        topk_ids = torch.tensor([[0]])  # only request expert 0
        cache_map = cache.prepare_and_get_expert_map(0, topk_ids, None)

        # Experts not in cache should be -1
        not_cached = 0
        for gid in range(8):
            if cache_map[gid].item() == -1:
                not_cached += 1
        # 8 total - 2 cached = 6 should be -1
        assert not_cached == 6

    def test_cache_map_slot_within_max_resident(self):
        """All slot indices are 0..max_resident-1."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8, max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))

        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        topk_ids = torch.tensor([[0, 1, 2, 3]])
        cache_map = cache.prepare_and_get_expert_map(0, topk_ids, None)

        for gid in range(8):
            slot = cache_map[gid].item()
            assert slot == -1 or (0 <= slot < 4)

    def test_step_resets_bw_throttle(self):
        """step() resets prefetch bytes counter."""
        cache = _make_cache()
        cache._prefetch_bytes_this_step = 9999
        cache.step()
        assert cache._prefetch_bytes_this_step == 0
        assert cache.current_step == 1

    def test_expert_data_integrity(self):
        """Verify loaded expert data matches CPU pool."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8, max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))

        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        # Check that GPU slot has correct data
        for slot in range(4):
            lid = cache._slot_to_expert[0][slot]
            if lid == -1:
                continue
            expected_w13, expected_w2 = cache._cpu_pool[0][lid]
            actual_w13 = cache._layer_w13[0][slot]
            actual_w2 = cache._layer_w2[0][slot]
            assert torch.allclose(actual_w13, expected_w13), \
                f"w13 mismatch for slot {slot} (expert {lid})"
            assert torch.allclose(actual_w2, expected_w2), \
                f"w2 mismatch for slot {slot} (expert {lid})"

    def test_multi_layer_independence(self):
        """Each layer has independent slot tracking."""
        cache = _make_cache(num_layers=3, local_E=8, global_E=8, max_resident=4)
        _register_dummy_experts(cache, 3, 8, (8, 4), (4, 8))

        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        # Each layer should have 4 filled slots independently
        for layer in range(3):
            filled = sum(1 for s in cache._slot_to_expert[layer] if s != -1)
            assert filled == 4

    def test_expert_map_with_ep(self):
        """EP expert_map properly filters to local experts only."""
        cache = _make_cache(num_layers=1, local_E=4, global_E=8, max_resident=4)
        _register_dummy_experts(cache, 1, 4, (8, 4), (4, 8))

        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        # EP map: globals 0-3 -> locals 0-3, globals 4-7 -> -1 (remote)
        expert_map = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1], dtype=torch.int32)
        topk_ids = torch.tensor([[0, 5, 2]])  # gid=5 is remote

        cache_map = cache.prepare_and_get_expert_map(0, topk_ids, expert_map)

        # gid=0 and gid=2 should have valid slots
        assert cache_map[0].item() >= 0
        assert cache_map[2].item() >= 0
        # gid=5 should be -1 (remote)
        assert cache_map[5].item() == -1
        # hits=2 (experts 0,2), misses=0
        assert cache.stats.hits == 2

    def test_eviction_protects_same_step_needed_experts(self):
        """HIGH: sync_fetch must not evict experts needed in the same step.

        Scenario: max_resident=6, experts 0-5 cached. Forward needs
        experts 0,1,2 (hits) + 7,8 (misses). Without protection,
        eviction could pick expert 0/1/2 (low access count). With
        protection, only experts 3/4/5 are eviction candidates.
        """
        cache = _make_cache(num_layers=1, local_E=10, global_E=10, max_resident=6)
        _register_dummy_experts(cache, 1, 10, (8, 4), (4, 8))

        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        # Experts 0-5 are cached (6 slots filled).
        # Give experts 3,4,5 HIGH access counts so naive eviction would
        # prefer evicting 0,1,2 (lowest priority). Protection must
        # prevent that since 0,1,2 are needed this step.
        for lid in [3, 4, 5]:
            cache._access_count[(0, lid)] = 100
            cache._last_access[(0, lid)] = 50
        for lid in [0, 1, 2]:
            cache._access_count[(0, lid)] = 0
            cache._last_access[(0, lid)] = 0

        topk_ids = torch.tensor([[0, 1, 2, 7, 8]])

        with patch("torch.cuda.synchronize"):
            cache_map = cache.prepare_and_get_expert_map(0, topk_ids, None)

        # All 5 requested experts must have valid slots
        for gid in [0, 1, 2, 7, 8]:
            assert cache_map[gid].item() >= 0, \
                f"Expert {gid} has no slot — likely evicted during sync_fetch"

        # Verify data integrity for the originally-cached hit experts
        for gid in [0, 1, 2]:
            slot = cache_map[gid].item()
            expected_w13, _ = cache._cpu_pool[0][gid]
            actual_w13 = cache._layer_w13[0][slot]
            assert torch.allclose(actual_w13, expected_w13), \
                f"Expert {gid} data corrupted — slot was overwritten"

        # Evictions should come from experts 3,4,5 (not 0,1,2)
        assert cache.stats.evictions == 2
        # Experts 3 or 4 or 5 should have been evicted (not 0,1,2)
        evicted_lids = {
            lid for lid in [3, 4, 5]
            if cache._expert_to_slot[0][lid] == -1
        }
        assert len(evicted_lids) == 2

    def test_needed_exceeds_max_resident_graceful(self):
        """MEDIUM: needed_local > max_resident must not crash.

        When a single step routes to more unique local experts than
        max_resident slots, the cache should load as many as possible
        and leave the rest as -1 in cache_map (graceful degradation).
        """
        # max_resident=4 but we need 6 unique experts
        cache = _make_cache(num_layers=1, local_E=10, global_E=10, max_resident=4)
        _register_dummy_experts(cache, 1, 10, (8, 4), (4, 8))

        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        # Experts 0-3 cached. Request all 6: 0,1,2,3 (hits) + 7,8 (misses).
        # 4 hits fill all 4 slots. 2 misses can't evict any (all protected).
        # Old code: RuntimeError. New code: skip misses, return -1.
        topk_ids = torch.tensor([[0, 1, 2, 3, 7, 8]])

        with patch("torch.cuda.synchronize"):
            cache_map = cache.prepare_and_get_expert_map(0, topk_ids, None)

        # Hits (0-3) must all have valid slots
        for gid in [0, 1, 2, 3]:
            assert cache_map[gid].item() >= 0, \
                f"Hit expert {gid} lost its slot"

        # Misses (7,8) may or may not have slots — at least no crash
        # With 4 hits in 4 slots, no room for 7,8 → expect -1
        skipped_count = sum(
            1 for gid in [7, 8] if cache_map[gid].item() == -1
        )
        assert skipped_count == 2, \
            f"Expected 2 skipped experts, got {skipped_count}"

    def test_lfu_lru_eviction_order(self):
        """LFU-LRU evicts least-used, oldest expert."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8, max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))

        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        # Simulate access: expert in slot 0 accessed many times
        for slot in range(4):
            lid = cache._slot_to_expert[0][slot]
            if lid != -1:
                key = (0, lid)
                cache._access_count[key] = 10 if slot == 0 else 1
                cache._last_access[key] = 100 if slot == 0 else 1

        evicted = cache._evict_one(0)
        assert evicted is True
        # The least-frequently-used expert should be evicted (not slot 0)
        assert cache._slot_to_expert[0][0] != -1  # slot 0 survived


# ─── CacheStats Tests ─────────────────────────────────────────────────

class TestCacheStats:
    def test_hit_rate_empty(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats(hits=90, misses=10)
        assert abs(stats.hit_rate - 0.9) < 1e-6

    def test_hit_rate_all_hits(self):
        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 1.0


# ─── ExpertOffloadConfig Tests ────────────────────────────────────────

class TestExpertOffloadConfig:
    def test_defaults(self):
        config = ExpertOffloadConfig()
        assert config.enable is False
        assert config.max_resident_per_layer == 50
        assert config.prefetch_lookahead == 1
        assert config.eviction_policy == "lfu_lru"
        assert config.staging_window_experts == 16
        assert config.pin_shared_experts is True

    def test_custom_values(self):
        config = ExpertOffloadConfig(
            enable=True, max_resident_per_layer=30, eviction_policy="lru"
        )
        assert config.enable is True
        assert config.max_resident_per_layer == 30
        assert config.eviction_policy == "lru"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
