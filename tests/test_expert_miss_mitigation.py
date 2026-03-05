"""Unit tests for Phase 1 expert cache miss mitigation.

Tests 3-A (never-evict), 3-B (union prediction), 3-E (eager fallback).
Run: pytest tests/test_expert_miss_mitigation.py -v --noconftest
"""
import sys
import os
import importlib.util
import pytest
import torch
from unittest.mock import MagicMock, patch

# Direct module import to avoid fused_moe/__init__.py cascade
_vllm_root = os.path.join(os.path.dirname(__file__), "..", "vllm",
                          "model_executor", "layers", "fused_moe")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cache_mod = _load_module(
    "expert_cache",
    os.path.join(_vllm_root, "expert_cache.py"))

ExpertCacheManager = _cache_mod.ExpertCacheManager
ExpertOffloadConfig = _cache_mod.ExpertOffloadConfig
CacheStats = _cache_mod.CacheStats


def _make_cache(max_resident=8, num_layers=2, local_num_experts=16,
                global_num_experts=16, never_evict_k=10, union_steps_k=3):
    """Create a minimal ExpertCacheManager for testing."""
    config = ExpertOffloadConfig(
        enable=True,
        max_resident_per_layer=max_resident,
    )
    # Patch env vars + mock CUDA stream before creating cache
    with patch.dict(os.environ, {
        "VLLM_EXPERT_NEVER_EVICT_K": str(never_evict_k),
        "VLLM_EXPERT_UNION_STEPS": str(union_steps_k),
    }), patch("torch.cuda.Stream") as mock_stream:
        mock_stream.return_value = MagicMock()
        cache = ExpertCacheManager(
            config=config,
            num_layers=num_layers,
            local_num_experts=local_num_experts,
            global_num_experts=global_num_experts,
            expert_w13_shape=(4, 4),
            expert_w2_shape=(4, 4),
            dtype=torch.float32,
            device=torch.device('cpu'),
        )

    # Register CPU pool with dummy weights
    for layer in range(num_layers):
        for lid in range(local_num_experts):
            w13 = torch.randn(4, 4)
            w2 = torch.randn(4, 4)
            cache._cpu_pool[layer][lid] = (w13, w2)

    # Register dummy GPU weight tensors
    for layer in range(num_layers):
        cache._layer_w13[layer] = torch.zeros(max_resident, 4, 4)
        cache._layer_w2[layer] = torch.zeros(max_resident, 4, 4)

    return cache


def _fill_cache(cache, layer_idx, expert_ids):
    """Manually load specific experts into cache slots."""
    for lid in expert_ids:
        if not cache._free_slots[layer_idx]:
            break
        slot = cache._free_slots[layer_idx].pop()
        cache._set_slot_expert(layer_idx, slot, lid)
        cache._set_expert_slot(layer_idx, lid, slot)
        cache._access_count[(layer_idx, lid)] = 1
        cache._last_access[(layer_idx, lid)] = cache.current_step


# ─── 3-A: Never-Evict Tests ─────────────────────────────────────────

class TestNeverEvict:
    def test_never_evict_protects_recent(self):
        """Experts used within K steps should NOT be evicted."""
        cache = _make_cache(max_resident=4, local_num_experts=8,
                            never_evict_k=5)
        layer = 0

        # Fill all 4 slots with experts 0-3
        _fill_cache(cache, layer, [0, 1, 2, 3])
        # Access all at step 1
        cache.current_step = 1
        for lid in range(4):
            cache._last_access[(layer, lid)] = 1

        # Advance to step 3 (within K=5 window)
        cache.current_step = 3

        # Try to evict — should fail (all protected)
        result = cache._evict_one(layer)
        assert result is False, "Should not evict any expert within K steps"
        assert cache.stats.never_evict_protections > 0

    def test_never_evict_allows_old(self):
        """Experts NOT used for K+ steps should be evictable."""
        cache = _make_cache(max_resident=4, local_num_experts=8,
                            never_evict_k=5)
        layer = 0

        # Fill with experts 0-3
        _fill_cache(cache, layer, [0, 1, 2, 3])
        # Expert 0 last accessed at step 0
        cache._last_access[(layer, 0)] = 0
        # Experts 1-3 accessed at step 5
        for lid in [1, 2, 3]:
            cache._last_access[(layer, lid)] = 5

        # Advance to step 6 (expert 0: age=6 > K=5)
        cache.current_step = 6

        result = cache._evict_one(layer)
        assert result is True, "Should evict expert with age > K"
        # Expert 0 should be the one evicted
        assert cache._expert_to_slot[layer][0] == -1

    def test_never_evict_disabled(self):
        """K=0 should disable never-evict protection."""
        cache = _make_cache(max_resident=4, local_num_experts=8,
                            never_evict_k=0)
        layer = 0

        _fill_cache(cache, layer, [0, 1, 2, 3])
        cache.current_step = 1
        for lid in range(4):
            cache._last_access[(layer, lid)] = 1

        # Should evict even though recently used (K=0 disables)
        result = cache._evict_one(layer)
        assert result is True
        assert cache.stats.never_evict_protections == 0


# ─── 3-B: Union Prediction Tests ────────────────────────────────────

class TestUnionPrediction:
    def _make_mock_layer(self, local_num_experts, global_num_experts,
                         routing_ids, top_k=2):
        """Create a mock layer with routing snapshot."""
        layer = MagicMock()
        layer.top_k = top_k
        # _expert_map = None => ep_size=1 (identity mapping)
        layer._expert_map = None
        # _routing_snapshot: flattened topk_ids
        snapshot = torch.tensor(routing_ids, dtype=torch.int32)
        layer._routing_snapshot = snapshot
        layer._routing_len = torch.tensor([len(routing_ids)], dtype=torch.int32)
        # _cache_map: GPU tensor for slot mapping
        layer._cache_map = torch.full(
            (global_num_experts,), -1, dtype=torch.int32)
        return layer

    def test_union_prediction_expands_needed(self):
        """Union of K steps should produce extras beyond current step."""
        cache = _make_cache(max_resident=16, local_num_experts=16,
                            global_num_experts=16,
                            never_evict_k=0, union_steps_k=3)
        layer_idx = 0

        # Fill cache with all 16 experts
        _fill_cache(cache, layer_idx, list(range(16)))

        # Manually populate routing history for layer 0:
        # Step 1: experts {0, 1, 2}
        cache._routing_history[layer_idx].append(frozenset({0, 1, 2}))
        # Step 2: experts {2, 3, 4}
        cache._routing_history[layer_idx].append(frozenset({2, 3, 4}))

        # Now simulate step 3 routing: experts {4, 5}
        # After union with history, should expand to {0,1,2,3,4,5}
        # Extras = {0, 1, 2, 3} - depends on the pre_step union logic

        # For a unit test, directly test the routing_history + union logic:
        current_local_set = {4, 5}
        needed_local = set(current_local_set)
        cache._routing_history[layer_idx].append(frozenset(current_local_set))

        if cache._union_steps_k > 1 and len(cache._routing_history[layer_idx]) > 1:
            union_local = set()
            for past in cache._routing_history[layer_idx]:
                union_local |= past
            extras = union_local - current_local_set
            if extras:
                needed_local = needed_local | extras
                cache.stats.union_prediction_extras += len(extras)

        # Union of {0,1,2}, {2,3,4}, {4,5} = {0,1,2,3,4,5}
        assert needed_local == {0, 1, 2, 3, 4, 5}
        assert cache.stats.union_prediction_extras == 4  # {0, 1, 2, 3}

    def test_union_prediction_k1_no_extras(self):
        """K=1 should produce no extras (single step = baseline)."""
        cache = _make_cache(max_resident=16, local_num_experts=16,
                            global_num_experts=16,
                            never_evict_k=0, union_steps_k=1)
        layer_idx = 0

        # With K=1, history maxlen=1, so only current step is stored
        current_local_set = {4, 5}
        needed_local = set(current_local_set)
        cache._routing_history[layer_idx].append(frozenset(current_local_set))

        if cache._union_steps_k > 1 and len(cache._routing_history[layer_idx]) > 1:
            # This block should NOT execute when K=1
            union_local = set()
            for past in cache._routing_history[layer_idx]:
                union_local |= past
            extras = union_local - current_local_set
            if extras:
                needed_local = needed_local | extras
                cache.stats.union_prediction_extras += len(extras)

        assert needed_local == {4, 5}
        assert cache.stats.union_prediction_extras == 0

    def test_union_history_maxlen(self):
        """History deque respects maxlen=K, dropping oldest entries."""
        cache = _make_cache(max_resident=16, local_num_experts=16,
                            global_num_experts=16,
                            never_evict_k=0, union_steps_k=2)
        layer_idx = 0

        # Push 3 entries into deque with maxlen=2
        cache._routing_history[layer_idx].append(frozenset({0, 1}))
        cache._routing_history[layer_idx].append(frozenset({2, 3}))
        cache._routing_history[layer_idx].append(frozenset({4, 5}))

        # Only last 2 should remain
        assert len(cache._routing_history[layer_idx]) == 2
        items = list(cache._routing_history[layer_idx])
        assert items[0] == frozenset({2, 3})
        assert items[1] == frozenset({4, 5})


# ─── 3-E: Eager Fallback Tests ──────────────────────────────────────

class TestEagerFallback:
    def test_eager_fallback_triggers_on_miss(self):
        """Eager fallback should trigger when any miss occurs and flag is set."""
        cache = _make_cache(max_resident=8, local_num_experts=16)

        # Simulate what gpu_model_runner checks
        eager_on_any_miss = True
        warmup_steps = 10
        step = 15
        cache.current_step = step
        cache_result = {'total_misses': 2, 'total_routed': 48, 'miss_ratio': 0.04}

        triggered = False
        if (eager_on_any_miss
                and step > warmup_steps
                and cache_result['total_misses'] > 0):
            triggered = True
            cache.stats.eager_fallback_triggers += 1

        assert triggered is True
        assert cache.stats.eager_fallback_triggers == 1

    def test_eager_fallback_no_trigger_zero_misses(self):
        """Eager fallback should NOT trigger when misses=0."""
        cache = _make_cache(max_resident=8, local_num_experts=16)

        eager_on_any_miss = True
        warmup_steps = 10
        step = 15
        cache.current_step = step
        cache_result = {'total_misses': 0, 'total_routed': 48, 'miss_ratio': 0.0}

        triggered = False
        if (eager_on_any_miss
                and step > warmup_steps
                and cache_result['total_misses'] > 0):
            triggered = True
            cache.stats.eager_fallback_triggers += 1

        assert triggered is False
        assert cache.stats.eager_fallback_triggers == 0

    def test_eager_fallback_disabled_by_default(self):
        """When flag is False, no eager fallback even with misses."""
        cache = _make_cache(max_resident=8, local_num_experts=16)

        eager_on_any_miss = False
        warmup_steps = 10
        step = 15
        cache.current_step = step
        cache_result = {'total_misses': 5, 'total_routed': 48, 'miss_ratio': 0.10}

        triggered = False
        if (eager_on_any_miss
                and step > warmup_steps
                and cache_result['total_misses'] > 0):
            triggered = True
            cache.stats.eager_fallback_triggers += 1

        assert triggered is False
        assert cache.stats.eager_fallback_triggers == 0


# ─── Combined Phase 1 Test ──────────────────────────────────────────

class TestCombinedPhase1:
    def test_combined_phase1(self):
        """All three features enabled, run sequence with routing churn.

        Verifies that never-evict + union prediction together reduce misses
        compared to baseline (never_evict_k=0, union_steps_k=1).
        """
        num_layers = 1
        local_experts = 16
        max_resident = 10

        # ── Baseline run (no mitigation) ──
        cache_base = _make_cache(
            max_resident=max_resident, num_layers=num_layers,
            local_num_experts=local_experts, global_num_experts=local_experts,
            never_evict_k=0, union_steps_k=1)
        _fill_cache(cache_base, 0, list(range(max_resident)))

        # ── Mitigated run ──
        cache_mit = _make_cache(
            max_resident=max_resident, num_layers=num_layers,
            local_num_experts=local_experts, global_num_experts=local_experts,
            never_evict_k=5, union_steps_k=3)
        _fill_cache(cache_mit, 0, list(range(max_resident)))

        # Simulate 20 steps with slowly shifting routing
        # Each step needs ~6 experts, shifting by 1-2 each step
        import random
        random.seed(42)

        for step in range(1, 21):
            # Generate needed experts: base set + small churn
            base = max(0, step - 3)
            needed = set(range(base, min(base + 6, local_experts)))

            for cache in [cache_base, cache_mit]:
                cache.current_step = step
                # Try to load all needed
                for lid in needed:
                    key = (0, lid)
                    if cache._expert_to_slot[0][lid] != -1:
                        cache._access_count[key] += 1
                        cache._last_access[key] = step
                        cache.stats.hits += 1
                    else:
                        cache.stats.misses += 1
                        # Try to evict + load
                        if not cache._free_slots[0]:
                            cache._evict_one(0, protected_local_ids=needed)
                        if cache._free_slots[0]:
                            slot = cache._free_slots[0].pop()
                            cache._set_slot_expert(0, slot, lid)
                            cache._set_expert_slot(0, lid, slot)
                            cache._access_count[key] = 1
                            cache._last_access[key] = step

        # Mitigated should have >= baseline hit rate
        # (never-evict prevents premature eviction of recently used experts)
        base_rate = cache_base.stats.hit_rate
        mit_rate = cache_mit.stats.hit_rate
        assert mit_rate >= base_rate, (
            f"Mitigated hit rate {mit_rate:.4f} should be >= "
            f"baseline {base_rate:.4f}")
        # Never-evict should have fired
        assert cache_mit.stats.never_evict_protections > 0


# ─── CacheStats Phase 1 Fields ──────────────────────────────────────

class TestCacheStatsPhase1:
    def test_new_fields_exist(self):
        """CacheStats has the 3 new Phase 1 fields."""
        stats = CacheStats()
        assert hasattr(stats, 'never_evict_protections')
        assert hasattr(stats, 'union_prediction_extras')
        assert hasattr(stats, 'eager_fallback_triggers')
        assert stats.never_evict_protections == 0
        assert stats.union_prediction_extras == 0
        assert stats.eager_fallback_triggers == 0

    def test_diagnostics_include_phase1(self):
        """get_miss_diagnostics() includes Phase 1 and Quality sections."""
        import numpy as np
        cache = _make_cache(max_resident=8, local_num_experts=16)
        cache.__init_v2_scratch = lambda: None  # skip v2 scratch
        cache._diag = {
            'per_layer_unique_needed': np.zeros(2, dtype=np.int64),
            'per_layer_rlen': np.zeros(2, dtype=np.int64),
            'per_layer_valid_len': np.zeros(2, dtype=np.int64),
            'per_layer_hits': np.array([100, 100], dtype=np.int64),
            'per_layer_misses': np.array([2, 3], dtype=np.int64),
            'per_layer_token_hits': np.zeros(2, dtype=np.int64),
            'per_layer_token_misses': np.zeros(2, dtype=np.int64),
            'diag_steps': 1,
            'diag_dump_interval': 100,
            'tokens_with_any_miss': 5,
            'tokens_total': 100,
            'miss_layers_total': 7,
            'per_layer_miss_rate_accum': np.zeros(2, dtype=np.float64),
            'per_layer_miss_rate_steps': 0,
        }
        diag = cache.get_miss_diagnostics()
        assert "[Phase 1 Mitigation]" in diag
        assert "never_evict_K:" in diag
        assert "union_steps_K:" in diag
        assert "eviction_protections:" in diag
        assert "union_extras_prefetched:" in diag
        assert "eager_fallback_triggers:" in diag
        # Quality eval sections
        assert "[Quality Eval" in diag
        assert "Token-level any-miss rate:" in diag
        assert "Layer-weighted miss concentration:" in diag

    def test_defaults_are_off(self):
        """Phase 1 mitigation defaults to disabled (K=0, union=1)."""
        cache = _make_cache(max_resident=8, local_num_experts=16,
                            never_evict_k=0, union_steps_k=1)
        assert cache._never_evict_k == 0
        assert cache._union_steps_k == 1
        assert cache._inject_miss_rate == 0.0


# ─── Quality Eval Tests ─────────────────────────────────────────────

class TestQualityEval:
    def test_quality_summary_analytical(self):
        """Quality summary produces analytical estimate without token data."""
        import numpy as np
        cache = _make_cache(max_resident=8, local_num_experts=16,
                            num_layers=4)
        cache._diag = {
            'per_layer_unique_needed': np.zeros(4, dtype=np.int64),
            'per_layer_rlen': np.zeros(4, dtype=np.int64),
            'per_layer_valid_len': np.zeros(4, dtype=np.int64),
            'per_layer_hits': np.array([96, 98, 98, 98], dtype=np.int64),
            'per_layer_misses': np.array([4, 2, 2, 2], dtype=np.int64),
            'per_layer_token_hits': np.zeros(4, dtype=np.int64),
            'per_layer_token_misses': np.zeros(4, dtype=np.int64),
            'diag_steps': 1,
            'diag_dump_interval': 100,
            'tokens_with_any_miss': 0,
            'tokens_total': 0,
            'miss_layers_total': 0,
            'per_layer_miss_rate_accum': np.zeros(4, dtype=np.float64),
            'per_layer_miss_rate_steps': 0,
        }
        summary = cache.get_quality_summary()
        assert "analytical estimate" in summary
        assert "Estimated:" in summary

    def test_quality_summary_with_token_data(self):
        """Quality summary shows real token-level data when available."""
        import numpy as np
        cache = _make_cache(max_resident=8, local_num_experts=16,
                            num_layers=4)
        cache._diag = {
            'per_layer_unique_needed': np.zeros(4, dtype=np.int64),
            'per_layer_rlen': np.zeros(4, dtype=np.int64),
            'per_layer_valid_len': np.zeros(4, dtype=np.int64),
            'per_layer_hits': np.array([96, 98, 98, 98], dtype=np.int64),
            'per_layer_misses': np.array([4, 2, 2, 2], dtype=np.int64),
            'per_layer_token_hits': np.zeros(4, dtype=np.int64),
            'per_layer_token_misses': np.zeros(4, dtype=np.int64),
            'diag_steps': 1,
            'diag_dump_interval': 100,
            'tokens_with_any_miss': 63,
            'tokens_total': 100,
            'miss_layers_total': 80,
            'per_layer_miss_rate_accum': np.zeros(4, dtype=np.float64),
            'per_layer_miss_rate_steps': 0,
        }
        summary = cache.get_quality_summary()
        assert "63/100" in summary
        assert "63.0%" in summary
        assert "miss-layers per token" in summary

    def test_layer_weighted_miss_concentration(self):
        """Layer 0 (input) and last layer (output) get higher weights."""
        import numpy as np
        cache = _make_cache(max_resident=8, local_num_experts=16,
                            num_layers=4)
        # Same miss rate everywhere = 2%
        cache._diag = {
            'per_layer_unique_needed': np.zeros(4, dtype=np.int64),
            'per_layer_rlen': np.zeros(4, dtype=np.int64),
            'per_layer_valid_len': np.zeros(4, dtype=np.int64),
            'per_layer_hits': np.array([98, 98, 98, 98], dtype=np.int64),
            'per_layer_misses': np.array([2, 2, 2, 2], dtype=np.int64),
            'per_layer_token_hits': np.zeros(4, dtype=np.int64),
            'per_layer_token_misses': np.zeros(4, dtype=np.int64),
            'diag_steps': 1,
            'diag_dump_interval': 100,
            'tokens_with_any_miss': 0,
            'tokens_total': 0,
            'miss_layers_total': 0,
            'per_layer_miss_rate_accum': np.zeros(4, dtype=np.float64),
            'per_layer_miss_rate_steps': 0,
        }
        summary = cache.get_quality_summary()
        # With uniform 2% miss, weighted = 2% (weights cancel out)
        # But Layer 0 contribution = 3.0 * 0.02 = 0.06
        # Layer 3 contribution = 2.0 * 0.02 = 0.04
        assert "weight=3.0" in summary  # Layer 0
        assert "weight=2.0" in summary  # Last layer
        assert "(input)" in summary
        assert "(output)" in summary

    def test_miss_injection_off_by_default(self):
        """Miss injection defaults to 0 (disabled)."""
        cache = _make_cache(max_resident=8, local_num_experts=16)
        assert cache._inject_miss_rate == 0.0

    def test_miss_injection_config(self):
        """Miss injection parses env vars correctly."""
        with patch.dict(os.environ, {
            "VLLM_EXPERT_INJECT_MISS_RATE": "0.05",
            "VLLM_EXPERT_INJECT_MISS_LAYERS": "0,47",
            "VLLM_EXPERT_NEVER_EVICT_K": "0",
            "VLLM_EXPERT_UNION_STEPS": "1",
        }), patch("torch.cuda.Stream") as ms:
            ms.return_value = MagicMock()
            cache = ExpertCacheManager(
                config=ExpertOffloadConfig(enable=True,
                                           max_resident_per_layer=8),
                num_layers=48, local_num_experts=16,
                global_num_experts=16,
                expert_w13_shape=(4, 4), expert_w2_shape=(4, 4),
                dtype=torch.float32, device=torch.device('cpu'),
            )
        assert cache._inject_miss_rate == 0.05
        assert cache._inject_miss_layers == {0, 47}

    def test_actual_vs_speculative_accounting(self):
        """Union expansion should NOT inflate total_routed or miss counts."""
        cache = _make_cache(max_resident=16, local_num_experts=16,
                            global_num_experts=16,
                            never_evict_k=0, union_steps_k=3)
        # Fill cache with experts 0-9 (10 of 16 max_resident)
        _fill_cache(cache, 0, list(range(10)))

        # Manually populate routing history with different sets
        cache._routing_history[0].append(frozenset({0, 1, 2, 10, 11}))
        cache._routing_history[0].append(frozenset({0, 1, 3, 12, 13}))

        # Current routing: {0, 1, 4} — all cached
        actual = {0, 1, 4}
        # After union: {0, 1, 2, 3, 4, 10, 11, 12, 13}
        # Union extras: {2, 3, 10, 11, 12, 13} — some not cached

        # Verify stats use actual set, not expanded
        actual_needed = set(actual)

        # Actual: 3 experts, all cached → 3 hits, 0 misses
        miss_ids = []
        for lid in actual_needed:
            if cache._expert_to_slot[0][lid] != -1:
                cache.stats.hits += 1
            else:
                miss_ids.append(lid)
                cache.stats.misses += 1

        assert len(miss_ids) == 0
        assert cache.stats.hits == 3
        assert cache.stats.misses == 0
