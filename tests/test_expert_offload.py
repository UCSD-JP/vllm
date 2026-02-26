"""Unit tests for expert weight offloading PoC.

Tests ExpertPredictor (pure CPU) and ExpertCacheManager (mocked CUDA).
Run: pytest tests/test_expert_offload.py -v --noconftest
"""
import sys
import os
import importlib.util
import pytest
import torch
from unittest.mock import MagicMock, patch

# Direct module import to avoid fused_moe/__init__.py cascade (which pulls in
# layer.py and its heavy dependencies). We only need expert_cache and predictor.
_vllm_root = os.path.join(os.path.dirname(__file__), "..", "vllm",
                          "model_executor", "layers", "fused_moe")

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_pred_mod = _load_module(
    "expert_predictor",
    os.path.join(_vllm_root, "expert_predictor.py"))
_cache_mod = _load_module(
    "expert_cache",
    os.path.join(_vllm_root, "expert_cache.py"))

ExpertPredictor = _pred_mod.ExpertPredictor
ExpertCacheManager = _cache_mod.ExpertCacheManager
ExpertOffloadConfig = _cache_mod.ExpertOffloadConfig
CacheStats = _cache_mod.CacheStats


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


# ─── v2: CUDA-graph-compatible pre_step() Tests ─────────────────────

def _make_mock_layer(global_E=16, local_E=16, max_num_tokens=64, top_k=10):
    """Create a mock FusedMoE-like layer with v2 persistent buffers."""
    layer = MagicMock()
    layer.global_num_experts = global_E
    layer.top_k = top_k

    # v2 persistent buffers (simulated register_buffer on CPU)
    layer._cache_map = torch.full((global_E,), -1, dtype=torch.int32)
    layer._routing_snapshot = torch.zeros(
        max_num_tokens * top_k, dtype=torch.int32)
    layer._routing_len = torch.zeros(1, dtype=torch.int32)

    # Expert map (ep_size=1: None means identity)
    layer._expert_map = None

    return layer


class TestPreStep:
    """Tests for v2 pre_step() predict-and-preload interface."""

    def test_pre_step_first_call_uses_initial_cache(self):
        """First call (routing_len=0) uses initial cache, no misses."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer = _make_mock_layer(global_E=8, local_E=8)
        result = cache.pre_step([layer])

        assert result['total_misses'] == 0
        assert cache.current_step == 1
        assert 'miss_ratio' in result
        assert 't_pre_step_us' in result

    def test_pre_step_updates_cache_map(self):
        """pre_step writes correct slot mapping to _cache_map."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer = _make_mock_layer(global_E=8, local_E=8)
        cache.pre_step([layer])

        for gid in range(4):
            slot = layer._cache_map[gid].item()
            assert 0 <= slot < 4, f"Expert {gid} should have valid slot"
        for gid in range(4, 8):
            assert layer._cache_map[gid].item() == -1

    def test_routing_snapshot_roundtrip(self):
        """Write routing_snapshot → pre_step reads it → correct needed."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer = _make_mock_layer(global_E=8, local_E=8)
        routing = torch.tensor([0, 1, 2], dtype=torch.int32)
        layer._routing_snapshot[:3].copy_(routing)
        layer._routing_len.fill_(3)

        result = cache.pre_step([layer])

        assert result['total_misses'] == 0
        assert cache.stats.hits == 3

    def test_pre_step_detects_misses(self):
        """Routing needs expert NOT in cache → miss counted."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer = _make_mock_layer(global_E=8, local_E=8)
        routing = torch.tensor([0, 1, 7], dtype=torch.int32)
        layer._routing_snapshot[:3].copy_(routing)
        layer._routing_len.fill_(3)

        result = cache.pre_step([layer])

        assert result['total_misses'] == 1
        assert cache.stats.misses == 1
        assert cache.stats.hits == 2
        assert layer._cache_map[7].item() >= 0

    def test_pre_step_multi_layer(self):
        """pre_step handles multiple layers independently."""
        cache = _make_cache(num_layers=2, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 2, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layers = [
            _make_mock_layer(global_E=8, local_E=8),
            _make_mock_layer(global_E=8, local_E=8),
        ]
        layers[0]._routing_snapshot[:1].copy_(
            torch.tensor([0], dtype=torch.int32))
        layers[0]._routing_len.fill_(1)
        layers[1]._routing_snapshot[:1].copy_(
            torch.tensor([6], dtype=torch.int32))
        layers[1]._routing_len.fill_(1)

        result = cache.pre_step(layers)
        assert result['total_misses'] == 1

    def test_pre_step_with_expert_map(self):
        """pre_step correctly handles EP expert_map (global→local)."""
        cache = _make_cache(num_layers=1, local_E=4, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 4, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer = _make_mock_layer(global_E=8, local_E=4)
        layer._expert_map = torch.tensor(
            [0, 1, 2, 3, -1, -1, -1, -1], dtype=torch.int32)
        routing = torch.tensor([0, 5], dtype=torch.int32)
        layer._routing_snapshot[:2].copy_(routing)
        layer._routing_len.fill_(2)

        result = cache.pre_step([layer])

        assert result['total_misses'] == 0
        assert cache.stats.hits == 1
        assert layer._cache_map[0].item() >= 0
        assert layer._cache_map[5].item() == -1

    def test_eager_fallback_on_high_miss_ratio(self):
        """pre_step returns miss_ratio for proportional fallback."""
        cache = _make_cache(num_layers=1, local_E=16, global_E=16,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 16, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer = _make_mock_layer(global_E=16, local_E=16)
        routing = torch.tensor(list(range(4, 16)), dtype=torch.int32)
        layer._routing_snapshot[:12].copy_(routing)
        layer._routing_len.fill_(12)

        result = cache.pre_step([layer])

        assert result['total_misses'] > 0
        assert result['miss_ratio'] > 0.5  # most experts missed

    def test_cache_map_inplace_preserves_identity(self):
        """_cache_map.data_ptr() is preserved across pre_step calls."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer = _make_mock_layer(global_E=8, local_E=8)
        ptr_before = layer._cache_map.data_ptr()
        cache.pre_step([layer])
        ptr_after = layer._cache_map.data_ptr()
        assert ptr_before == ptr_after, \
            "data_ptr must be preserved for CUDA graph compatibility"

    def test_pre_step_skips_layers_without_buffers(self):
        """Layers without _cache_map are skipped gracefully."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer_no_buffer = MagicMock()
        layer_no_buffer._cache_map = None
        result = cache.pre_step([layer_no_buffer])
        assert result['total_misses'] == 0

    def test_padding_does_not_affect_needed_local(self):
        """HIGH-1: num_tokens trims padding from routing snapshot."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer = _make_mock_layer(global_E=8, local_E=8, top_k=2)
        # 2 real tokens × top_k=2 = 4 valid routing entries
        # Then 2 padding tokens × top_k=2 = 4 padding entries with expert 7
        real_routing = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        pad_routing = torch.tensor([7, 7, 7, 7], dtype=torch.int32)
        layer._routing_snapshot[:4].copy_(real_routing)
        layer._routing_snapshot[4:8].copy_(pad_routing)
        layer._routing_len.fill_(8)  # graph records padded length

        # Without num_tokens: expert 7 would be needed (from padding)
        result_no_trim = cache.pre_step([layer], num_tokens=0)
        cache.current_step -= 1  # reset for next call

        # Reset stats
        cache.stats.hits = 0
        cache.stats.misses = 0

        # With num_tokens=2: only first 4 entries (2 tokens × top_k=2)
        result_trimmed = cache.pre_step([layer], num_tokens=2)

        # Trimmed version should NOT need expert 7 (padding)
        assert result_trimmed['total_misses'] == 0  # experts 0-3 all cached

    def test_miss_ratio_invariant_to_padding(self):
        """HIGH-1: miss_ratio with padding trim equals ratio without."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer = _make_mock_layer(global_E=8, local_E=8, top_k=2)
        # 1 real token needing experts [0, 5]
        # padding tokens all route to expert 0
        layer._routing_snapshot[:2].copy_(
            torch.tensor([0, 5], dtype=torch.int32))
        layer._routing_snapshot[2:6].copy_(
            torch.tensor([0, 0, 0, 0], dtype=torch.int32))
        layer._routing_len.fill_(6)  # padded

        result = cache.pre_step([layer], num_tokens=1)
        # With trim: only [0, 5] needed. Expert 5 is miss.
        assert result['total_routed'] == 2
        assert result['miss_ratio'] == 0.5  # 1 miss / 2 routed


# ─── Batched D2H + Vectorized Cache Map Tests ────────────────────────

class TestBatchedD2H:
    """Tests for batched D2H optimization (Steps 2-4)."""

    def test_set_expert_slot_syncs_tensor(self):
        """_set_expert_slot updates both list and tensor mirror."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        # Init tensor mirror manually
        cache._expert_to_slot_t = torch.full(
            (1, 8), -1, dtype=torch.int32)
        for eid in range(8):
            cache._expert_to_slot_t[0, eid] = \
                cache._expert_to_slot[0][eid]

        # Now call wrapper and verify both are updated
        cache._set_expert_slot(0, 5, 2)
        assert cache._expert_to_slot[0][5] == 2
        assert cache._expert_to_slot_t[0, 5].item() == 2

        # Clear slot
        cache._set_expert_slot(0, 5, -1)
        assert cache._expert_to_slot[0][5] == -1
        assert cache._expert_to_slot_t[0, 5].item() == -1

    def test_vectorized_cache_map_matches_loop(self):
        """Vectorized _update_cache_map produces same result as loop."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer = _make_mock_layer(global_E=8, local_E=8)

        # Get loop result (before tensor mirror exists)
        cache._update_cache_map(0, layer)
        loop_result = layer._cache_map.clone()

        # Now init tensor mirror + emap cache
        cache._expert_to_slot_t = torch.full(
            (1, 8), -1, dtype=torch.int32)
        for eid in range(8):
            cache._expert_to_slot_t[0, eid] = \
                cache._expert_to_slot[0][eid]
        cache._emap_cpu_cache = [None]  # ep_size=1

        # Get vectorized result
        layer._cache_map.fill_(-1)
        cache._update_cache_map(0, layer)
        vec_result = layer._cache_map.clone()

        assert torch.equal(loop_result, vec_result), \
            f"Mismatch: loop={loop_result} vs vec={vec_result}"

    def test_vectorized_cache_map_with_ep(self):
        """Vectorized path with expert_map (EP)."""
        cache = _make_cache(num_layers=1, local_E=4, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 4, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer = _make_mock_layer(global_E=8, local_E=4)
        emap = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1],
                            dtype=torch.int32)
        layer._expert_map = emap

        # Loop result
        cache._update_cache_map(0, layer)
        loop_result = layer._cache_map.clone()

        # Init vectorized path
        cache._expert_to_slot_t = torch.full(
            (1, 4), -1, dtype=torch.int32)
        for eid in range(4):
            cache._expert_to_slot_t[0, eid] = \
                cache._expert_to_slot[0][eid]
        cache._emap_cpu_cache = [emap.clone()]

        layer._cache_map.fill_(-1)
        cache._update_cache_map(0, layer)
        vec_result = layer._cache_map.clone()

        assert torch.equal(loop_result, vec_result), \
            f"Mismatch: loop={loop_result} vs vec={vec_result}"

    def test_batched_d2h_init_creates_buffers(self):
        """_init_batched_d2h creates all required buffers."""
        cache = _make_cache(num_layers=2, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 2, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layers = [
            _make_mock_layer(global_E=8, local_E=8),
            _make_mock_layer(global_E=8, local_E=8),
        ]

        with patch("torch.cuda.Stream") as mock_stream, \
             patch("torch.cuda.Event") as mock_event:
            mock_stream.return_value = MagicMock()
            mock_event.return_value = MagicMock()
            cache._init_batched_d2h(layers)

        assert hasattr(cache, '_batched_d2h_ready')
        assert hasattr(cache, '_routing_len_gpu')
        assert hasattr(cache, '_routing_snap_gpu')
        assert hasattr(cache, '_routing_len_cpu')
        assert hasattr(cache, '_routing_snap_cpu')
        assert hasattr(cache, '_emap_cpu_cache')
        assert hasattr(cache, '_expert_to_slot_t')
        assert hasattr(cache, '_active_layer_indices')

        assert cache._routing_len_gpu.shape == (2,)
        assert cache._expert_to_slot_t.shape == (2, 8)
        assert len(cache._emap_cpu_cache) == 2
        assert len(cache._active_layer_indices) == 2

    def test_batched_pre_step_same_result(self):
        """Batched pre_step produces same cache_map as unbatched."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 1, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        layer = _make_mock_layer(global_E=8, local_E=8)
        routing = torch.tensor([0, 1, 2], dtype=torch.int32)
        layer._routing_snapshot[:3].copy_(routing)
        layer._routing_len.fill_(3)

        # Patch CUDA operations for CPU-only test
        with patch("torch.cuda.Stream") as mock_stream, \
             patch("torch.cuda.Event") as mock_event:
            mock_stream_inst = MagicMock()
            mock_stream.return_value = mock_stream_inst
            mock_event_inst = MagicMock()
            mock_event.return_value = mock_event_inst

            # Need to mock pin_memory for CPU tensors
            orig_pin = torch.Tensor.pin_memory
            torch.Tensor.pin_memory = lambda self: self
            try:
                result = cache.pre_step([layer])
            finally:
                torch.Tensor.pin_memory = orig_pin

        assert result['total_misses'] == 0
        assert cache.stats.hits == 3

        # Verify cache_map is correct
        for gid in range(4):
            slot = layer._cache_map[gid].item()
            assert 0 <= slot < 4

    def test_get_timing_summary(self):
        """get_timing_summary returns formatted string."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        cache.__init_v2_scratch = lambda: None
        cache._timing = {
            't_pre_step_total_us': 1000.0,
            't_gpu_gather_us': 100.0,
            't_d2h_us': 200.0,
            't_classify_us': 300.0,
            't_cache_map_us': 150.0,
            't_fetch_us': 50.0,
            't_sync_us': 200.0,
            'pre_step_calls': 10,
        }
        summary = cache.get_timing_summary()
        assert "avg over 10 calls" in summary
        assert "t_pre_step_total_us: 100.0 us/call" in summary
        assert "t_d2h_us: 20.0 us/call" in summary

    def test_invalidate_emap_cache_none(self):
        """invalidate_emap_cache(None) sets cache entry to None."""
        cache = _make_cache(num_layers=2, local_E=8, global_E=8,
                            max_resident=4)
        cache._emap_cpu_cache = [
            torch.tensor([0, 1, -1, -1], dtype=torch.int32),
            None
        ]
        cache.invalidate_emap_cache(0)
        assert cache._emap_cpu_cache[0] is None
        # Out of range is safe
        cache.invalidate_emap_cache(99)

    def test_invalidate_emap_cache_recache(self):
        """invalidate_emap_cache with new_expert_map re-caches immediately."""
        cache = _make_cache(num_layers=1, local_E=4, global_E=8,
                            max_resident=4)
        old_emap = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1],
                                dtype=torch.int32)
        cache._emap_cpu_cache = [old_emap.clone()]

        # EPLB changes mapping: experts 4-7 now local, 0-3 remote
        new_emap = torch.tensor([-1, -1, -1, -1, 0, 1, 2, 3],
                                dtype=torch.int32)
        cache.invalidate_emap_cache(0, new_expert_map=new_emap)

        # Cache should have new mapping, not None
        assert cache._emap_cpu_cache[0] is not None
        assert torch.equal(cache._emap_cpu_cache[0], new_emap)

    def test_get_timing_summary_before_init(self):
        """get_timing_summary returns safe message before any pre_step."""
        cache = _make_cache(num_layers=1, local_E=8, global_E=8,
                            max_resident=4)
        # _timing not yet created (no pre_step called)
        assert not hasattr(cache, '_timing')
        summary = cache.get_timing_summary()
        assert "not yet initialized" in summary

    def test_stacked_scratch_contiguous(self):
        """scratch_maps are views into contiguous _stacked_scratch_cpu."""
        cache = _make_cache(num_layers=2, local_E=8, global_E=8,
                            max_resident=4)
        _register_dummy_experts(cache, 2, 8, (8, 4), (4, 8))
        with patch("torch.cuda.synchronize"):
            cache.populate_initial_cache()

        # Trigger __init_v2_scratch
        cache._ExpertCacheManager__init_v2_scratch()

        # Verify stacked_scratch_cpu exists and scratch_maps are views
        assert hasattr(cache, '_stacked_scratch_cpu')
        assert cache._stacked_scratch_cpu.shape == (2, 8)
        for i in range(2):
            # Same storage pointer
            assert (cache._scratch_maps[i].data_ptr()
                    == cache._stacked_scratch_cpu[i].data_ptr())

        # Write to scratch_maps[0] and verify it's visible in stacked
        cache._scratch_maps[0].fill_(42)
        assert cache._stacked_scratch_cpu[0, 0].item() == 42
        # Scratch maps[1] unchanged
        assert cache._stacked_scratch_cpu[1, 0].item() == -1

    def test_eplb_invalidation_hook(self):
        """update_expert_map triggers re-cache with new expert_map."""
        cache = _make_cache(num_layers=1, local_E=4, global_E=8,
                            max_resident=4)
        old_emap = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1],
                                dtype=torch.int32)
        cache._emap_cpu_cache = [old_emap.clone()]

        # Simulate what layer.update_expert_map() does:
        # new_expert_map after EPLB rebalancing
        new_emap = torch.tensor([0, -1, 2, -1, 1, -1, 3, -1],
                                dtype=torch.int32)
        cache.invalidate_emap_cache(0, new_expert_map=new_emap)

        # Cache should have new mapping (not None, not old)
        assert cache._emap_cpu_cache[0] is not None
        assert torch.equal(cache._emap_cpu_cache[0], new_emap)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
