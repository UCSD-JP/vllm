# SPDX-License-Identifier: Apache-2.0
"""Expert Cache Manager for MoE weight offloading (MVP elastic KV).

Trimmed from the research fork:
- VA (value-aware) scoring → removed (LRU only)
- grow_from_kv / grow_from_free_pages → NotImplementedError stubs
- pressure epoch / decay scoring → removed
- miss injection / prefill sync optimization → removed

Retained:
- Core slot management (_slot_to_expert, _expert_to_slot)
- CPU pinned backing store + async DMA
- LRU-based eviction
- shrink_for_pages (renamed from shrink_for_kv, simplified)
- predict-and-preload (pre_step / pre_step_single_layer)
- cache_map persistent buffer (CUDA graph compatible)
"""

from __future__ import annotations

import enum
import logging
import math
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_EXPERT_DEBUG = os.environ.get("VLLM_EXPERT_DEBUG", "0") == "1"
_EXPERT_TRACE = os.environ.get("VLLM_EXPERT_TRACE", "0") == "1"
_EAGER_LRU_SKIP = os.environ.get("VLLM_EAGER_LRU_SKIP", "0") == "1"
_EAGER_CAP_LOG = os.environ.get("VLLM_EAGER_CAP_LOG", "0") == "1"
_KERNEL_PROFILE = os.environ.get("VLLM_KERNEL_PROFILE", "0") == "1"
_SYNC_FETCH_DEVICE_SYNC = (
    os.environ.get("VLLM_SYNC_FETCH_DEVICE_SYNC", "1") == "1")
_EAGER_BREAKDOWN = os.environ.get("VLLM_EAGER_BREAKDOWN", "0") == "1"
_FIXED_TAIL = os.environ.get("VLLM_FIXED_TAIL", "0") == "1"
_STEP_BOUNDARY = os.environ.get("VLLM_STEP_BOUNDARY", "0") == "1"
_CUTOFF_BOUNDARY = os.environ.get("VLLM_CUTOFF_BOUNDARY", "0") == "1"

# ── Eager breakdown accumulators (VLLM_EAGER_BREAKDOWN=1) ──
if _EAGER_BREAKDOWN:
    import time as _bk_time
    _bk_interval = 500  # log every N calls
    _bk_calls = 0
    _bk_hits = 0
    _bk_misses = 0
    # Stage timings (cumulative μs)
    _bk_s1a_gather_us = 0.0    # cache_map[valid_ids] + .any() (GPU enqueue)
    _bk_s1b_sync_us = 0.0      # .item() scalar sync (GPU→CPU drain)
    _bk_s2_hit_us = 0.0        # hit path: unique+cpu+numpy+LRU
    _bk_s3_classify_us = 0.0   # miss path: unique+cpu+numpy+sets
    _bk_s4_fetch_us = 0.0      # sync_fetch + scratch
    _bk_s5_post_us = 0.0       # _update_cache_map + scratch + prewarm

# ── Copy profiling (VLLM_KERNEL_PROFILE=1) ──
if _KERNEL_PROFILE:
    import time as _time
    _cp_interval = 200
    _cp_calls = 0
    _cp_total_us = 0.0
    _cp_count = 0
    _cp_experts_total = 0
    # Expert delta: per-layer routing churn, separated by phase
    # key = (phase, layer_idx) where phase = 'D' (decode) or 'P' (prefill)
    # Phase determined by ExpertCache._prefillish_step (set by runner)
    _ed_prev_gids: "Dict[tuple, set]" = {}  # type: ignore
    _ed_delta_total_d = 0  # decode
    _ed_delta_count_d = 0
    _ed_routed_total_d = 0
    _ed_delta_total_p = 0  # prefill
    _ed_delta_count_p = 0
    _ed_routed_total_p = 0
    # Layer cycle (intra-step): Li → Li+1 within same step/phase
    _lc_last_t: float = 0.0
    _lc_last_layer: int = -1
    _lc_last_phase: str = ''
    _lc_total_us_p = 0.0  # prefill only (decode has no adjacent layers)
    _lc_count_p = 0
    # Step cycle (cross-step): same layer across steps (L0→L0 next step)
    # key = (phase, layer_idx) → last timestamp
    _sc_prev_t: "Dict[tuple, float]" = {}  # type: ignore
    _sc_total_us_d = 0.0
    _sc_count_d = 0
    _sc_total_us_p = 0.0
    _sc_count_p = 0
    # TailDelta: same-layer cross-step non-resident set churn (prefill only)
    # "If bank kept last step's non-resident experts, how many need fresh copy?"
    _td_prev_miss: "Dict[int, set]" = {}  # layer_idx → set of miss gids
    _td_delta_total = 0  # new experts not in prev bank
    _td_reuse_total = 0  # experts already in prev bank
    _td_miss_total = 0   # total non-resident per call
    _td_count = 0
_trace_fd = None
if _EXPERT_TRACE:
    import json as _json
    _trace_base = os.environ.get(
        "VLLM_EXPERT_TRACE_PATH", "/tmp/expert_trace")
    _trace_path = f"{_trace_base}_pid{os.getpid()}.jsonl"
    _trace_fd = open(_trace_path, "a", buffering=1)  # line-buffered append
    logger.info("Expert trace enabled: %s", _trace_path)


@dataclass
class ExpertOffloadConfig:
    """Expert offloading configuration."""
    enable: bool = False
    max_resident_per_layer: int = 50
    prefetch_lookahead: int = 1
    eviction_policy: str = "lru"
    staging_window_experts: int = 16
    prefetch_max_bytes_per_step: int = 64 * (1 << 20)  # 64MB
    pin_shared_experts: bool = True


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefetch_hits: int = 0
    sync_fetches: int = 0
    deduped_prefetches: int = 0
    token_hits: int = 0
    token_misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def token_hit_rate(self) -> float:
        total = self.token_hits + self.token_misses
        return self.token_hits / total if total > 0 else 0.0


class _BankState(enum.IntEnum):
    IDLE = 0
    FILLING = 1   # H2D in progress on copy_stream
    READY = 2     # ready_event recorded, awaiting consume
    IN_USE = 3    # compute stream consuming


@dataclass
class _ScratchBank:
    w13: torch.Tensor
    w2: torch.Tensor
    ready_event: torch.cuda.Event
    done_event: torch.cuda.Event
    gid_buffer: torch.Tensor            # int64, pre-alloc for index_copy_
    state: _BankState = _BankState.IDLE
    owner_layer_idx: int = -1
    owner_layer: object = None
    current_gids: Optional[torch.Tensor] = None
    n_active: int = 0
    # Phase 2: prewarm metadata
    prewarm_target_layer: int = -1
    prewarmed_gids: Optional[List[int]] = None
    prewarmed_lids: Optional[List[int]] = None
    gid_to_slot: Optional[Dict[int, int]] = None
    consumed_gids: Optional[List[int]] = None
    # Split prefetch: w13 done before w2
    w13_ready_event: Optional[torch.cuda.Event] = None
    n_prewarmed: int = 0


class ExpertCacheManager:
    """Manages expert weight offloading with LRU eviction.

    MVP elastic KV: supports one-way expert → KV page conversion.
    """

    def __init__(
        self,
        config: ExpertOffloadConfig,
        num_layers: int,
        local_num_experts: int,
        global_num_experts: int,
        expert_w13_shape: Tuple[int, ...],
        expert_w2_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.config = config
        self.num_layers = num_layers
        self.local_num_experts = local_num_experts
        self.global_num_experts = global_num_experts
        self.max_resident = config.max_resident_per_layer
        self.dtype = dtype
        self.device = device
        self.current_step = 0

        # Per-expert size
        self._w13_shape = expert_w13_shape
        self._w2_shape = expert_w2_shape
        w13_numel = 1
        for d in expert_w13_shape:
            w13_numel *= d
        w2_numel = 1
        for d in expert_w2_shape:
            w2_numel *= d
        elem_size = torch.tensor([], dtype=dtype).element_size()
        self.expert_size_bytes = (w13_numel + w2_numel) * elem_size

        # Identity map for ep_size=1
        self._identity_map = torch.arange(
            global_num_experts, dtype=torch.int32, device=device)

        # Per-layer slot tracking
        self._slot_to_expert: List[List[int]] = []
        self._expert_to_slot: List[List[int]] = []
        self._free_slots: List[List[int]] = []
        for _ in range(num_layers):
            self._slot_to_expert.append([-1] * self.max_resident)
            self._expert_to_slot.append([-1] * local_num_experts)
            self._free_slots.append(
                list(range(self.max_resident - 1, -1, -1)))

        # numpy mirror for vectorized slot lookups in eager boundary
        self._expert_to_slot_np = np.full(
            (num_layers, local_num_experts), -1, dtype=np.int32)

        # Eviction metadata
        self._access_count = np.zeros(
            (num_layers, local_num_experts), dtype=np.int32)
        self._last_access = np.zeros(
            (num_layers, local_num_experts), dtype=np.int32)
        self._pinned: Set[Tuple[int, int]] = set()

        # CPU pinned backing store
        self._cpu_pool: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = [
            {} for _ in range(num_layers)]

        # GPU weight tensor references
        self._layer_w13: Dict[int, torch.Tensor] = {}
        self._layer_w2: Dict[int, torch.Tensor] = {}

        # CUDA stream + prefetch
        self._copy_stream = torch.cuda.Stream(device=device)
        self._prefetch_events: Dict[int, torch.cuda.Event] = {}
        self._prefetch_pending: Dict[int, Set[int]] = defaultdict(set)
        self._prefetch_bytes_this_step: int = 0
        self._collect_routing_freq = False  # set by gpu_model_runner

        # Phase flags: set by runner before each forward pass
        self._prefillish_step: bool = False
        self._has_decode_tokens: bool = False

        # Eager boundary instrumentation (opt-in via _collect_routing_freq)
        self._eager_calls = 0
        self._eager_hits = 0
        self._eager_misses = 0
        self._eager_miss_experts = 0
        self._eager_fetch_failures = 0
        self._eager_log_interval = 500

        # Stats
        self.stats = CacheStats()
        self._lock = threading.Lock()

        # VMM dynamic pool (set via set_vmm_pool())
        self._vmm_pool = None
        self._vmm_enabled: bool = False
        self._dynamic_max_resident: List[int] = [
            self.max_resident for _ in range(num_layers)]
        self._unmapped_slots: List[Set[int]] = [
            set() for _ in range(num_layers)]

        # Cache-map dirty flag
        self._cache_map_needs_rebuild = False
        self._dormant: bool = True  # runtime: set by runner each step; True=safe initial

        # Phase C stats
        self._phase_c_enabled: bool = False
        self._phase_c_stats: Dict[str, int] = {
            'shrink_calls': 0,
            'shrink_pages': 0,
        }

        # 3-D: Group boundary state (set by gpu_worker)
        self._group_ranges: List[List[int]] = []
        # 3-D instrumentation
        self._group_boundary_calls: int = 0
        self._group_prefetch_experts: int = 0
        self._group_prefetch_loaded: int = 0
        self._group_prefetch_skipped: int = 0
        self._group_log_interval: int = 200

        # ── Scratch bank state (set by gpu_worker via set_scratch()) ──
        self._scratch_banks: List[_ScratchBank] = []
        self._scratch_threshold: int = 0  # = max_resident (base capacity)
        self._scratch_capacity: int = 0   # num scratch slots per bank
        self._scratch_slot_values: Optional[torch.Tensor] = None  # shared
        self._scratch_current_bank: int = 0
        # MoE layer refs for lookahead prefetch (set by gpu_worker)
        self._moe_layers: List = []
        # Prewarm: known non-resident H2D + partial consume (default OFF)
        self._prewarm_enabled: bool = (
            os.environ.get("VLLM_EXPERT_PREWARM", "0") == "1")
        self._prewarm_cap: int = int(
            os.environ.get("VLLM_PREWARM_CAP", "64"))
        # Scratch instrumentation
        self._scratch_reserve_calls: int = 0
        self._scratch_experts_loaded: int = 0
        self._scratch_log_interval: int = 200
        # ── Layer-concentrated shrink policy ──
        # VLLM_SHRINK_TAIL_N: only the last N layers are shrink-eligible.
        # L0 is always protected regardless of this setting.
        # Default: num_layers (all layers eligible except L0 — preserves
        # prior behavior except L0 protection). Set smaller to concentrate.
        _tail_n = int(os.environ.get("VLLM_SHRINK_TAIL_N",
                                     str(num_layers)))
        _tail_n = max(0, min(_tail_n, num_layers))
        self._shrink_tail_n: int = _tail_n  # raw config for logging
        self._shrink_eligible: Set[int] = set(
            range(num_layers - _tail_n, num_layers))
        self._shrink_eligible.discard(0)  # L0 always protected

        # Prewarm instrumentation
        self._pw_consumes: int = 0
        self._pw_H_total: int = 0
        self._pw_R_total: int = 0
        self._pw_Rfill_total: int = 0
        self._pw_fallback: int = 0

        # ── Fixed-tail double-buffer (VLLM_FIXED_TAIL=1) ──
        self._fixed_tail_active: bool = False
        self._ft_tail_lids: List[List[int]] = []   # per-layer CPU lid lists
        self._ft_current_bank: int = 0

        # ── Step-boundary static topology (VLLM_STEP_BOUNDARY=1) ──
        self._static_topo_active: bool = False
        self._topology_dirty: bool = False
        self._sb_tail_lids: List[List[int]] = []   # per-layer non-resident lids
        self._sb_tail_gids: List[List[int]] = []   # per-layer non-resident gids
        self._sb_current_bank: int = 0

        # ── TP-size (set by gpu_worker, default 1) ──
        self._tp_size: int = 1

        # ── Cutoff-boundary (VLLM_CUTOFF_BOUNDARY=1) ──
        self._cutoff_verified: bool = False   # True after first verification (pass or fail)
        self._cutoff_supported: bool = _CUTOFF_BOUNDARY  # env-based; cleared on verify fail
        self._cutoff_active: bool = False
        self._resident_cutoff: int = local_num_experts  # init: all resident
        self._cutoff_tail_lids: List[int] = []  # shared across L1..L47
        self._cutoff_current_bank: int = 0

    # ================================================================
    # Backward-compatible scratch properties
    # ================================================================

    @property
    def _scratch_w13(self) -> Optional[torch.Tensor]:
        return self._scratch_banks[0].w13 if self._scratch_banks else None

    @property
    def _scratch_w2(self) -> Optional[torch.Tensor]:
        return self._scratch_banks[0].w2 if self._scratch_banks else None

    @property
    def _scratch_in_use(self) -> bool:
        return any(b.state == _BankState.IN_USE for b in self._scratch_banks)

    @property
    def _scratch_owner_layer_idx(self) -> int:
        for b in self._scratch_banks:
            if b.state == _BankState.IN_USE:
                return b.owner_layer_idx
        return -1

    def set_moe_layers(self, layers: List) -> None:
        """Store MoE layer refs for lookahead prefetch."""
        self._moe_layers = layers
        # O2: cache per-layer expert_map as CPU numpy for vectorized lookups.
        # _expert_map is a static registered buffer (layer.py:473), immutable
        # unless EP remap occurs. Avoids emap.cpu() D2H (~5μs) per layer per step.
        self._emap_cpu_np: List[Optional[np.ndarray]] = [None] * self.num_layers
        for i, lyr in enumerate(layers):
            emap = getattr(lyr, '_expert_map', None)
            if emap is not None:
                self._emap_cpu_np[i] = emap.cpu().numpy()

    # ================================================================
    # Slot helpers
    # ================================================================

    def _set_expert_slot(self, layer_idx: int, lid: int, slot: int):
        self._expert_to_slot[layer_idx][lid] = slot
        if hasattr(self, '_expert_to_slot_np'):
            self._expert_to_slot_np[layer_idx, lid] = slot
        if hasattr(self, '_expert_to_slot_t'):
            self._expert_to_slot_t[layer_idx, lid] = slot

    def _set_slot_expert(self, layer_idx: int, slot: int, lid: int):
        self._slot_to_expert[layer_idx][slot] = lid

    def _resolve_expert_map(
        self, original_expert_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if original_expert_map is not None:
            return original_expert_map
        return self._identity_map

    # ================================================================
    # Registration
    # ================================================================

    def register_layer(
        self, layer_idx: int,
        w13_data: torch.Tensor, w2_data: torch.Tensor,
    ):
        self._layer_w13[layer_idx] = w13_data
        self._layer_w2[layer_idx] = w2_data

    def register_expert_cpu(
        self, layer_idx: int, local_expert_id: int,
        w13_weight: torch.Tensor, w2_weight: torch.Tensor,
        is_shared: bool = False,
    ):
        if w13_weight.device.type == 'cpu':
            w13_cpu = w13_weight.detach().clone().pin_memory()
            w2_cpu = w2_weight.detach().clone().pin_memory()
        else:
            w13_cpu = w13_weight.detach().cpu().pin_memory()
            w2_cpu = w2_weight.detach().cpu().pin_memory()
        self._cpu_pool[layer_idx][local_expert_id] = (w13_cpu, w2_cpu)
        if is_shared and self.config.pin_shared_experts:
            self._pinned.add((layer_idx, local_expert_id))

    # ================================================================
    # Initial population
    # ================================================================

    def populate_initial_cache(self):
        for layer_idx in range(self.num_layers):
            for lid in range(self.local_num_experts):
                if (layer_idx, lid) in self._pinned:
                    self._load_to_slot(layer_idx, lid)
            for lid in range(self.local_num_experts):
                if not self._free_slots[layer_idx]:
                    break
                if self._expert_to_slot[layer_idx][lid] != -1:
                    continue
                self._load_to_slot(layer_idx, lid)
        torch.cuda.synchronize(self.device)

    def _load_to_slot(self, layer_idx: int, local_id: int) -> bool:
        if not self._free_slots[layer_idx]:
            return False
        slot = self._free_slots[layer_idx].pop()
        self._set_slot_expert(layer_idx, slot, local_id)
        self._set_expert_slot(layer_idx, local_id, slot)
        w13_cpu, w2_cpu = self._cpu_pool[layer_idx][local_id]
        self._layer_w13[layer_idx][slot].copy_(w13_cpu)
        self._layer_w2[layer_idx][slot].copy_(w2_cpu)
        return True

    # ================================================================
    # Core: prepare_and_get_expert_map (from forward_cuda)
    # ================================================================

    def prepare_and_get_expert_map(
        self, layer_idx: int, topk_ids: torch.Tensor,
        original_expert_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        emap = self._resolve_expert_map(original_expert_map)
        if layer_idx in self._prefetch_events:
            event = self._prefetch_events.pop(layer_idx)
            torch.cuda.current_stream(self.device).wait_event(event)
            self._prefetch_pending.pop(layer_idx, None)

        needed_local: Set[int] = set()
        emap_cpu = emap.cpu()
        for gid in topk_ids.flatten().unique().tolist():
            if 0 <= gid < emap_cpu.shape[0]:
                lid = emap_cpu[gid].item()
                if lid != -1:
                    needed_local.add(lid)

        miss_ids: List[int] = []
        with self._lock:
            for lid in needed_local:
                if self._expert_to_slot[layer_idx][lid] != -1:
                    self._access_count[layer_idx, lid] += 1
                    self._last_access[layer_idx, lid] = self.current_step
                    self.stats.hits += 1
                else:
                    miss_ids.append(lid)
                    self.stats.misses += 1

        if miss_ids:
            cached_needed = needed_local - set(miss_ids)
            self._sync_fetch(layer_idx, miss_ids,
                             protected_local_ids=cached_needed)

        cache_map = torch.full(
            (self.global_num_experts,), -1,
            dtype=torch.int32, device=self.device)
        for gid in range(emap_cpu.shape[0]):
            lid = emap_cpu[gid].item()
            if lid == -1:
                continue
            slot = self._expert_to_slot[layer_idx][lid]
            if slot != -1:
                cache_map[gid] = slot
        return cache_map

    # ================================================================
    # Prefetch
    # ================================================================

    def enqueue_prefetch(
        self, target_layer_idx: int, predicted_local_ids: List[int],
    ):
        if target_layer_idx >= self.num_layers:
            return
        if target_layer_idx in self._prefetch_events:
            return

        to_fetch: List[int] = []
        for lid in predicted_local_ids:
            if self._expert_to_slot[target_layer_idx][lid] != -1:
                continue
            if lid in self._prefetch_pending.get(target_layer_idx, set()):
                self.stats.deduped_prefetches += 1
                continue
            to_fetch.append(lid)

        if not to_fetch:
            return

        max_n = self.config.prefetch_max_bytes_per_step // max(
            self.expert_size_bytes, 1)
        remaining = max_n - (
            self._prefetch_bytes_this_step // max(self.expert_size_bytes, 1))
        if remaining <= 0:
            return
        to_fetch = to_fetch[:min(len(to_fetch), remaining)]

        for _ in range(len(to_fetch)):
            if self._free_slots[target_layer_idx]:
                break
            self._evict_one(target_layer_idx)

        event = torch.cuda.Event()
        loaded: Set[int] = set()
        with torch.cuda.stream(self._copy_stream):
            for lid in to_fetch:
                if not self._free_slots[target_layer_idx]:
                    break
                slot = self._free_slots[target_layer_idx].pop()
                self._set_slot_expert(target_layer_idx, slot, lid)
                self._set_expert_slot(target_layer_idx, lid, slot)
                self._access_count[target_layer_idx, lid] = 0
                self._last_access[target_layer_idx, lid] = self.current_step
                w13_cpu, w2_cpu = self._cpu_pool[target_layer_idx][lid]
                self._layer_w13[target_layer_idx][slot].copy_(
                    w13_cpu, non_blocking=True)
                self._layer_w2[target_layer_idx][slot].copy_(
                    w2_cpu, non_blocking=True)
                loaded.add(lid)
            event.record(self._copy_stream)

        self._prefetch_events[target_layer_idx] = event
        self._prefetch_pending[target_layer_idx] = loaded
        self._prefetch_bytes_this_step += len(loaded) * self.expert_size_bytes

    # ================================================================
    # Sync fetch
    # ================================================================

    def _sync_fetch(
        self, layer_idx: int, local_ids: List[int],
        protected_local_ids: Optional[Set[int]] = None,
        skip_device_sync: bool = False,
    ) -> Tuple[int, int, Optional[Dict[str, int]]]:
        """Returns (fetched, skipped, evict_fail_reason or None).

        skip_device_sync: If True, skip the final torch.cuda.synchronize().
            Use when caller manages stream ordering (e.g. prewarm path).
            The .copy_() calls are already ordered on the current stream.
            Global device sync can also be disabled process-wide via
            VLLM_SYNC_FETCH_DEVICE_SYNC=0 for eager offload experiments.
        """
        protected = set(protected_local_ids) if protected_local_ids else set()
        skipped = 0
        fetched = 0
        for i, lid in enumerate(local_ids):
            if not self._free_slots[layer_idx]:
                if not self._evict_one(layer_idx, protected):
                    skipped += len(local_ids) - i
                    break
            slot = self._free_slots[layer_idx].pop()
            self._set_slot_expert(layer_idx, slot, lid)
            self._set_expert_slot(layer_idx, lid, slot)
            self._access_count[layer_idx, lid] = 1
            self._last_access[layer_idx, lid] = self.current_step
            protected.add(lid)
            w13_cpu, w2_cpu = self._cpu_pool[layer_idx][lid]
            self._layer_w13[layer_idx][slot].copy_(w13_cpu)
            self._layer_w2[layer_idx][slot].copy_(w2_cpu)
            self.stats.sync_fetches += 1
            fetched += 1
        evict_reason = None
        if skipped > 0:
            # Eviction failure breakdown
            n_empty = 0
            n_unmapped = 0
            n_pinned = 0
            n_protected = 0
            n_pending = 0
            n_evictable = 0
            pending_set = self._prefetch_pending.get(layer_idx, set())
            for s in range(len(self._slot_to_expert[layer_idx])):
                lid = self._slot_to_expert[layer_idx][s]
                if lid == -1:
                    n_empty += 1
                elif (self._vmm_enabled
                      and s in self._unmapped_slots[layer_idx]):
                    n_unmapped += 1
                elif (layer_idx, lid) in self._pinned:
                    n_pinned += 1
                elif protected and lid in protected:
                    n_protected += 1
                elif lid in pending_set:
                    n_pending += 1
                else:
                    n_evictable += 1
            evict_reason = dict(empty=n_empty, unmapped=n_unmapped,
                                pinned=n_pinned, protected=n_protected,
                                pending=n_pending)
            # Always log fetch_fail with full diagnostics
            n_free = len(self._free_slots[layer_idx])
            n_total = len(self._slot_to_expert[layer_idx])
            _dmr = getattr(self, '_dynamic_max_resident', None)
            dyn_max = (_dmr[layer_idx] if _dmr is not None
                       and layer_idx < len(_dmr)
                       else self.max_resident)
            logger.warning(
                "[Fetch-Fail] L%d step=%d skipped=%d fetched=%d "
                "requested=%d | slots: total=%d free=%d empty=%d "
                "unmapped=%d pinned=%d protected=%d pending=%d "
                "evictable=%d | dyn_max=%d floor=%d scratch=%d",
                layer_idx, self.current_step, skipped, fetched,
                len(local_ids), n_total, n_free, n_empty,
                n_unmapped, n_pinned, n_protected, n_pending,
                n_evictable, dyn_max,
                self._resident_floor(layer_idx)
                if hasattr(self, '_resident_floor') else -1,
                self._scratch_capacity)
        if not skip_device_sync and _SYNC_FETCH_DEVICE_SYNC:
            torch.cuda.synchronize(self.device)
        return fetched, skipped, evict_reason

    def _async_fetch(
        self, layer_idx: int, local_ids: List[int],
        protected: Optional[Set[int]] = None,
    ):
        protected = set(protected) if protected else set()
        with torch.cuda.stream(self._copy_stream):
            for lid in local_ids:
                if not self._free_slots[layer_idx]:
                    if not self._evict_one(layer_idx, protected):
                        continue
                slot = self._free_slots[layer_idx].pop()
                self._set_slot_expert(layer_idx, slot, lid)
                self._set_expert_slot(layer_idx, lid, slot)
                self._access_count[layer_idx, lid] = 1
                self._last_access[layer_idx, lid] = self.current_step
                protected.add(lid)
                w13_cpu, w2_cpu = self._cpu_pool[layer_idx][lid]
                self._layer_w13[layer_idx][slot].copy_(
                    w13_cpu, non_blocking=True)
                self._layer_w2[layer_idx][slot].copy_(
                    w2_cpu, non_blocking=True)

    # ================================================================
    # Eviction (LRU only)
    # ================================================================

    def _evict_one(
        self, layer_idx: int,
        protected_local_ids: Optional[Set[int]] = None,
    ) -> bool:
        best_slot = -1
        best_priority = float('inf')
        for slot in range(len(self._slot_to_expert[layer_idx])):
            lid = self._slot_to_expert[layer_idx][slot]
            if lid == -1:
                continue
            if (self._vmm_enabled
                    and slot in self._unmapped_slots[layer_idx]):
                continue
            if (layer_idx, lid) in self._pinned:
                continue
            if protected_local_ids and lid in protected_local_ids:
                continue
            if lid in self._prefetch_pending.get(layer_idx, set()):
                continue
            # LRU: lowest last_access = best victim
            priority = int(self._last_access[layer_idx, lid])
            if priority < best_priority:
                best_priority = priority
                best_slot = slot

        if best_slot == -1:
            return False
        lid = self._slot_to_expert[layer_idx][best_slot]
        self._set_slot_expert(layer_idx, best_slot, -1)
        self._set_expert_slot(layer_idx, lid, -1)
        self._free_slots[layer_idx].append(best_slot)
        self.stats.evictions += 1
        return True

    # ================================================================
    # VMM dynamic pool interface
    # ================================================================

    def set_vmm_pool(self, pool) -> None:
        self._vmm_pool = pool
        self._vmm_enabled = True
        logger.info("ExpertCacheManager: VMM pool attached")

    def has_evicted_experts(self) -> bool:
        """Return True if any expert slots are currently evicted (unmapped).

        Two eviction sources:
        1. max_resident < local_num_experts → some experts never fit in GPU
        2. VMM shrink → _unmapped_slots non-empty
        """
        if self.max_resident < self.local_num_experts:
            return True
        for layer in range(self.num_layers):
            if self._unmapped_slots[layer]:
                return True
        return False

    def _can_shrink(self) -> bool:
        """Shared eligibility check for expert eviction.

        Used by both count_evictable_groups (prepare) and
        shrink_for_pages (commit) to ensure identical guard logic.
        Without this, prepare can propose groups that commit refuses
        to evict, causing assert failures.
        """
        if not self._vmm_enabled or self._vmm_pool is None:
            return False
        if not self._phase_c_enabled:
            if self.max_resident >= self.local_num_experts:
                return False
        return True

    def _resident_floor(self, layer_idx: int) -> int:
        """Minimum resident experts to maintain scratch coverage invariant.

        resident_floor = localE - scratch_capacity
        When scratch not active, returns 0 (no floor).
        """
        if self._scratch_capacity > 0:
            return max(0, self.local_num_experts - self._scratch_capacity)
        return 0

    def _evictable_groups(self) -> Dict:
        """Build group-level evictability map (shared by prepare and commit).

        A group is evictable iff ALL its mapped slots are unpinned.
        This matches _select_lru_victims semantics: one pinned member
        poisons the whole group.

        Returns:
            Dict[(layer, group_idx)] → list of (layer, slot) members.
            Only groups where all members are unpinned are included.
        """
        pool = self._vmm_pool
        if pool is None:
            return {}
        group_size = pool.group_size
        # Track: group_key → (has_pinned, members)
        groups: Dict[Tuple[int, int], Tuple[bool, list]] = {}
        for layer in range(self.num_layers):
            if layer not in self._shrink_eligible:
                continue  # layer-concentrated shrink: skip protected layers
            for slot in range(len(self._slot_to_expert[layer])):
                lid = self._slot_to_expert[layer][slot]
                if lid == -1:
                    continue
                if slot in self._unmapped_slots[layer]:
                    continue
                group_idx = slot // group_size
                gk = (layer, group_idx)
                is_pinned = (layer, lid) in self._pinned
                if gk not in groups:
                    groups[gk] = (is_pinned, [(layer, slot)])
                else:
                    prev_pinned, members = groups[gk]
                    members.append((layer, slot))
                    groups[gk] = (prev_pinned or is_pinned, members)
        # Filter: unpinned AND floor-safe (scratch coverage invariant)
        pool = self._vmm_pool
        group_size = pool.group_size if pool else 1
        result = {}
        for gk, (pinned, members) in groups.items():
            if pinned or not members:
                continue
            layer = gk[0]
            floor = self._resident_floor(layer)
            if self._dynamic_max_resident[layer] - group_size < floor:
                continue  # floor guard: would break coverage invariant
            result[gk] = members
        return result

    def _scored_evictable_groups(self) -> List[Tuple[int, Tuple, list]]:
        """Score evictable groups by LRU coldest-first.

        Shared by count_evictable_groups (prepare) and _select_lru_victims
        (commit) so both see identical ordering and candidates.

        Returns:
            List of (max_access, group_key, members) sorted coldest-first.
        """
        evictable_map = self._evictable_groups()
        if not evictable_map:
            return []
        scored = []
        for gk, members in evictable_map.items():
            layer = gk[0]
            max_access = 0
            for _, slot in members:
                lid = self._slot_to_expert[layer][slot]
                if lid >= 0:
                    max_access = max(
                        max_access, int(self._last_access[layer, lid]))
            scored.append((max_access, gk, members))
        scored.sort()
        return scored

    def _greedy_select(
        self,
        scored: List[Tuple[int, Tuple, list]],
        max_groups: int,
    ) -> List[list]:
        """Greedy group selection with cumulative per-layer floor guard.

        Shared by count_evictable_groups (prepare) and _select_lru_victims
        (commit) to guarantee prepare never over-promises.

        Args:
            scored: output of _scored_evictable_groups()
            max_groups: maximum number of groups to select

        Returns:
            List of member lists for selected groups.
        """
        remaining = list(self._dynamic_max_resident)
        pool = self._vmm_pool
        group_size = pool.group_size if pool else 1

        selected = []
        for _, gk, members in scored:
            if len(selected) >= max_groups:
                break
            layer = gk[0]
            floor = self._resident_floor(layer)
            if remaining[layer] - group_size < floor:
                continue  # cumulative floor guard
            remaining[layer] -= group_size
            selected.append(members)
        return selected

    def _cutoff_verify_once(self) -> bool:
        """Lazy one-time verification of cutoff preconditions.

        Called from count_evictable_groups / shrink_for_pages / pre_step
        before dispatching to cutoff path. Returns current _cutoff_supported.
        """
        if self._cutoff_verified:
            return self._cutoff_supported
        self._cutoff_verified = True
        fail_reason = None

        # 1) Identity: lid == slot for all layers
        for li in range(self.num_layers):
            for lid in range(self.local_num_experts):
                if self._expert_to_slot[li][lid] != lid:
                    fail_reason = (
                        f"lid!=slot at layer={li} lid={lid} "
                        f"slot={self._expert_to_slot[li][lid]}")
                    break
            if fail_reason:
                break

        # 2) Pinned experts must be below shrink floor
        if not fail_reason and self._pinned:
            floor = max(
                0, self.local_num_experts - self._scratch_capacity)
            for layer_idx, lid in self._pinned:
                if lid >= floor:
                    fail_reason = (
                        f"pinned expert ({layer_idx},{lid}) "
                        f">= floor={floor}")
                    break

        # 3) cpu_pool completeness for all experts in L1..L47
        if not fail_reason:
            for li in range(1, self.num_layers):
                for lid in range(self.local_num_experts):
                    if lid not in self._cpu_pool[li]:
                        fail_reason = (
                            f"cpu_pool[{li}][{lid}] missing")
                        break
                if fail_reason:
                    break

        if fail_reason:
            self._cutoff_supported = False
            logger.warning(
                "[CutoffBoundary] precondition failed: %s — "
                "falling back to default path (permanent)",
                fail_reason)
        else:
            logger.info(
                "[CutoffBoundary] preconditions verified, "
                "cutoff path enabled")
        return self._cutoff_supported

    def count_evictable_groups(self) -> int:
        """Count groups that can actually be evicted (prepare-safe).

        Uses the same greedy cumulative floor guard as _select_lru_victims
        to guarantee prepare never over-promises vs commit.
        """
        if (_CUTOFF_BOUNDARY
                and self._cutoff_verify_once()):
            return self._cutoff_count_evictable_groups()
        if not self._can_shrink():
            return 0
        scored = self._scored_evictable_groups()
        # No limit — count all that pass cumulative guard
        selected = self._greedy_select(scored, len(scored))
        return len(selected)

    def shrink_for_pages(self, min_pages: int) -> Tuple[int, int]:
        """Evict expert groups to free at least min_pages physical pages.

        Simplified from shrink_for_kv: no VA scoring, no pressure epoch.

        Returns:
            (freed_pages, groups_evicted)
        """
        if (_CUTOFF_BOUNDARY
                and self._cutoff_verify_once()):
            return self._cutoff_shrink(min_pages)
        if self._fixed_tail_active:
            logger.debug(
                "[FixedTail] shrink_for_pages called but topology frozen. "
                "Returning 0 (KV pressure handled via preemption).")
            return 0, 0
        if not self._can_shrink():
            return 0, 0

        pool = self._vmm_pool

        groups_needed = math.ceil(min_pages / pool.group_pages)
        victims = self._select_lru_victims(
            groups_needed * pool.group_size)
        if not victims:
            gs = pool.group_size
            n_protected = self.num_layers - len(self._shrink_eligible)
            floor = self._resident_floor(0)
            at_floor = [(l, self._dynamic_max_resident[l])
                        for l in self._shrink_eligible
                        if self._dynamic_max_resident[l] - gs < floor]
            logger.info(
                "shrink_for_pages: no victims (needed=%d pages, "
                "protected_layers=%d, at_floor=%d/%d, floor=%d)",
                min_pages, n_protected,
                len(at_floor), len(self._shrink_eligible), floor)
            return 0, 0

        freed_pages = pool.unmap_expert_slots(victims)
        groups_evicted = self._update_tracking_after_evict(victims)
        self._cache_map_needs_rebuild = True
        if _STEP_BOUNDARY:
            self._topology_dirty = True

        self._phase_c_stats['shrink_calls'] += 1
        self._phase_c_stats['shrink_pages'] += freed_pages
        logger.info(
            "shrink_for_pages: freed %d groups → %d pages "
            "(floor=%d dyn_res_min=%d scratch=%d)",
            groups_evicted, freed_pages,
            self._resident_floor(0),
            min(self._dynamic_max_resident),
            self._scratch_capacity)
        return freed_pages, groups_evicted

    def _select_lru_victims(self, count: int) -> List[Tuple[int, int]]:
        """Select coldest expert slots by LRU (group-aware).

        Uses _scored_evictable_groups() + _greedy_select() — identical
        logic as count_evictable_groups() to prevent prepare/commit mismatch.
        """
        scored = self._scored_evictable_groups()
        if not scored:
            return []
        pool = self._vmm_pool
        groups_needed = math.ceil(count / (pool.group_size if pool else 1))
        selected = self._greedy_select(scored, groups_needed)
        victims = []
        for members in selected:
            victims.extend(members)
        return victims[:count]

    def _update_tracking_after_evict(
        self, victims: List[Tuple[int, int]],
    ) -> int:
        """Update cache tracking after pool.unmap_expert_slots."""
        if not self._vmm_pool:
            return 0
        pool = self._vmm_pool
        seen_groups: Set[Tuple[int, int]] = set()
        for layer, slot in victims:
            group_idx = slot // pool.group_size
            gk = (layer, group_idx)
            if gk in seen_groups:
                continue
            seen_groups.add(gk)
            for gs_offset in range(pool.group_size):
                group_slot = group_idx * pool.group_size + gs_offset
                if group_slot >= len(self._slot_to_expert[layer]):
                    continue
                lid = self._slot_to_expert[layer][group_slot]
                if lid != -1:
                    self._set_expert_slot(layer, lid, -1)
                    self._set_slot_expert(layer, group_slot, -1)
                    self.stats.evictions += 1
                self._unmapped_slots[layer].add(group_slot)
                if group_slot in self._free_slots[layer]:
                    self._free_slots[layer].remove(group_slot)
            self._dynamic_max_resident[layer] = max(
                0, self._dynamic_max_resident[layer] - pool.group_size)
        return len(seen_groups)

    # ================================================================
    # Reverse path stubs (MVP: disabled)
    # ================================================================

    def grow_from_kv(self, *args, **kwargs):
        raise NotImplementedError("MVP: expert recovery disabled.")

    def grow_from_free_pages(self, *args, **kwargs):
        raise NotImplementedError("MVP: expert recovery disabled.")

    # ================================================================
    # Step
    # ================================================================

    def step(self):
        self.current_step += 1
        self._prefetch_bytes_this_step = 0

    # ================================================================
    # pre_step (simplified v2)
    # ================================================================

    def pre_step(self, layers, num_tokens: int = 0,
                 has_prefill: bool = False) -> dict:
        """Called BEFORE CUDA graph replay. Predict-and-preload.

        Simplified: no batched D2H, no pressure epoch, no miss injection.
        """
        # Mark ready after first call (CUDA graph capture is complete).
        # This gates eager_pre_step_single_layer() during graph warmup.
        if not getattr(self, '_batched_d2h_ready', False):
            self._batched_d2h_ready = True

        # ── Fixed-tail lazy activation ──
        if (not self._fixed_tail_active
                and _FIXED_TAIL
                and self._scratch_banks
                and len(self._scratch_banks) >= 2
                and any(self._expert_to_slot_np[li].min() == -1
                        for li in range(self.num_layers))):
            self._activate_fixed_tail(layers)

        # Fixed-tail: skip all dynamic logic (LRU, prefetch, rebuild)
        if self._fixed_tail_active:
            self.current_step += 1
            return {'total_misses': 0, 'total_routed': 0,
                    'miss_ratio': 0.0, 'layer_stats': []}

        # ── Cutoff-boundary mode ──
        if (_CUTOFF_BOUNDARY
                and self._scratch_banks
                and len(self._scratch_banks) >= 2
                and self._cutoff_verify_once()):
            if not self._cutoff_active:
                if self._resident_cutoff < self.local_num_experts:
                    self._cutoff_active = True
                    self._topology_dirty = True
                    logger.info("[CutoffBoundary] ACTIVATING: cutoff=%d",
                                self._resident_cutoff)

            if self._cutoff_active:
                if self._topology_dirty:
                    self._cutoff_recompute_topology()
                    self._cutoff_rebuild_cache_maps()
                    self._topology_dirty = False

                # No boot prefetch: L0 is resident-only.
                # First bank fill happens via cutoff_prefetch_next(0)
                # during L0's forward pass.

                self.current_step += 1
                return {'total_misses': 0, 'total_routed': 0,
                        'miss_ratio': 0.0, 'layer_stats': []}

        # ── Step-boundary static topology ──
        if (_STEP_BOUNDARY
                and self._scratch_banks
                and len(self._scratch_banks) >= 2):
            # Activation: first time non-resident experts detected
            if not self._static_topo_active:
                has_evicted = any(
                    self._expert_to_slot_np[li].min() == -1
                    for li in range(self.num_layers))
                if has_evicted:
                    self._topology_dirty = True
                    self._static_topo_active = True
                    logger.info("[StepBoundary] ACTIVATING: "
                                "non-resident experts detected")

            if self._static_topo_active:
                # LRU batch update from previous step's routing snapshots
                self._update_lru_from_snapshots(layers)

                # Topology recomputation (only when dirty)
                if self._topology_dirty:
                    self._recompute_step_topology(layers)
                    self._patch_step_cache_maps(layers)
                    self._start_initial_prefetch()
                    self._topology_dirty = False

                self.current_step += 1
                return {'total_misses': 0, 'total_routed': 0,
                        'miss_ratio': 0.0, 'layer_stats': []}

        # Fast path: all experts resident, no shrink happened
        if self.max_resident >= self.local_num_experts:
            if not self._phase_c_enabled:
                self.current_step += 1
                return {'total_misses': 0, 'total_routed': 0,
                        'miss_ratio': 0.0, 'layer_stats': []}
            elif (not self._cache_map_needs_rebuild
                  and not any(self._unmapped_slots)):
                self.current_step += 1
                return {'total_misses': 0, 'total_routed': 0,
                        'miss_ratio': 0.0, 'layer_stats': []}

        total_misses = 0
        total_routed = 0
        layer_stats = []

        for i, layer in enumerate(layers):
            if i >= self.num_layers:
                break
            if not hasattr(layer, '_routing_snapshot'):
                layer_stats.append({'misses': 0, 'routed': 0})
                continue
            if not hasattr(layer, '_cache_map'):
                layer_stats.append({'misses': 0, 'routed': 0})
                continue

            result = self.pre_step_single_layer(
                i, layer, num_tokens)
            layer_stats.append(result)
            total_misses += result.get('misses', 0)
            total_routed += result.get('routed', 0)

        if self._cache_map_needs_rebuild:
            # Rebuild cache maps for all layers
            for i, layer in enumerate(layers):
                if i >= self.num_layers:
                    break
                if hasattr(layer, '_cache_map') and layer._cache_map is not None:
                    self._update_cache_map(i, layer)
            self._cache_map_needs_rebuild = False

        self.current_step += 1
        self._prefetch_bytes_this_step = 0

        miss_ratio = (total_misses / total_routed
                      if total_routed > 0 else 0.0)
        return {
            'total_misses': total_misses,
            'total_routed': total_routed,
            'miss_ratio': miss_ratio,
            'layer_stats': layer_stats,
        }

    def pre_step_single_layer(
        self, layer_idx: int, layer, num_tokens: int = 0,
    ) -> dict:
        """Per-layer predict-and-preload."""
        if not hasattr(layer, '_routing_snapshot') \
                or layer._routing_snapshot is None:
            return {'misses': 0, 'routed': 0}
        if not hasattr(layer, '_routing_len') \
                or layer._routing_len is None:
            return {'misses': 0, 'routed': 0}

        # Read routing snapshot from previous step
        rlen = int(layer._routing_len.item())
        if rlen <= 0:
            return {'misses': 0, 'routed': 0}

        snap = layer._routing_snapshot[:rlen]

        # Opt-in routing frequency histogram
        expert_freq = None
        if self._collect_routing_freq:
            expert_freq = torch.bincount(
                snap.clamp(min=0), minlength=self.global_num_experts)

        unique_gids = snap.unique().tolist()

        # Convert global→local
        emap = (layer._expert_map if hasattr(layer, '_expert_map')
                else None)
        needed_local: Set[int] = set()
        if emap is not None:
            emap_cpu = emap.cpu()
            for gid in unique_gids:
                if 0 <= gid < emap_cpu.shape[0]:
                    lid = emap_cpu[gid].item()
                    if lid != -1:
                        needed_local.add(lid)
        else:
            needed_local = {gid for gid in unique_gids
                           if 0 <= gid < self.local_num_experts}

        # Hit/miss classification
        miss_ids = []
        hit_ids = set()
        for lid in needed_local:
            if self._expert_to_slot[layer_idx][lid] != -1:
                self._last_access[layer_idx, lid] = self.current_step
                hit_ids.add(lid)
            else:
                miss_ids.append(lid)

        # Fetch misses
        if miss_ids:
            self._async_fetch(layer_idx, miss_ids, protected=hit_ids)
            self._copy_stream.synchronize()
            self._cache_map_needs_rebuild = True

        return {
            'misses': len(miss_ids),
            'routed': len(needed_local),
            'expert_freq': expert_freq,
        }

    def eager_pre_step_single_layer(self, layer_idx: int,
                                    topk_ids: torch.Tensor, layer):
        """Eager routing: fresh routing → sync fetch → cache_map update.

        Called from eager_routing_boundary custom op INSIDE forward pass
        (between graph pieces). Uses current-step routing (not N-1 stale).

        IMPORTANT: The expert union is computed from topk_ids.flatten()
        which is the WHOLE call/chunk's tokens. If the full prefill batch
        is passed, need can approach localE (e.g. 446/512), overwhelming
        the resident budget. Use prefill chunking to reduce per-call union.
        See EAGER_ROUTING_SPEC.md section 2.

        GPU fast path: checks cache_map on GPU first (~10us).
        Only falls back to expensive CPU path when there's a miss.
        """
        # Guard: skip during CUDA graph capture/warmup.
        # _batched_d2h_ready is False until first pre_step() completes.
        if not getattr(self, "_batched_d2h_ready", False):
            return
        # Fixed-tail: thin prefetch only, NO .item(), NO .cpu(), NO cache_map
        if self._fixed_tail_active:
            self._fixed_tail_prefetch_next(layer_idx)
            return

        # Cycle timing (two axes):
        # 1. LayerCycle: Li→Li+1 intra-step (prefill overlap window)
        # 2. StepCycle: same-layer cross-step (decode full-step budget)
        if _KERNEL_PROFILE:
            global _lc_last_t, _lc_last_layer, _lc_last_phase
            global _lc_total_us_p, _lc_count_p
            global _sc_total_us_d, _sc_count_d, _sc_total_us_p, _sc_count_p
            _t_now = _time.perf_counter()
            _cur_phase = 'P' if self._prefillish_step else 'D'
            # LayerCycle: adjacent layers, same phase (prefill overlap window)
            if (_lc_last_layer >= 0
                    and layer_idx == _lc_last_layer + 1
                    and _cur_phase == _lc_last_phase):
                _lc_us = (_t_now - _lc_last_t) * 1e6
                if _cur_phase == 'P':
                    _lc_total_us_p += _lc_us
                    _lc_count_p += 1
            _lc_last_t = _t_now
            _lc_last_layer = layer_idx
            _lc_last_phase = _cur_phase
            # StepCycle: same (phase, layer) across steps
            _sc_key = (_cur_phase, layer_idx)
            _sc_prev = _sc_prev_t.get(_sc_key)
            if _sc_prev is not None:
                _sc_us = (_t_now - _sc_prev) * 1e6
                if _cur_phase == 'D':
                    _sc_total_us_d += _sc_us
                    _sc_count_d += 1
                else:
                    _sc_total_us_p += _sc_us
                    _sc_count_p += 1
            _sc_prev_t[_sc_key] = _t_now

        # Use current-step topk_ids directly (NOT _routing_snapshot).
        # _routing_snapshot is written inside moe_forward() after
        # forward_impl() and used by inter_group_step for next-group
        # prefetch prediction.
        valid_ids = topk_ids.flatten()
        if valid_ids.numel() <= 0:
            return

        # GPU fast path: check cache_map for misses without CPU roundtrip
        if _EAGER_BREAKDOWN:
            _bk_t0 = _bk_time.perf_counter()
        cache_hits = layer._cache_map[valid_ids]       # GPU gather
        _any_neg = (cache_hits < 0).any()               # GPU bool reduce
        if _EAGER_BREAKDOWN:
            global _bk_calls, _bk_hits, _bk_misses
            global _bk_s1a_gather_us, _bk_s1b_sync_us
            global _bk_s2_hit_us, _bk_s3_classify_us
            global _bk_s4_fetch_us, _bk_s5_post_us
            _bk_s1a_gather_us += (_bk_time.perf_counter() - _bk_t0) * 1e6
            _bk_t0b = _bk_time.perf_counter()
        has_miss = _any_neg.item()                      # GPU→CPU scalar sync
        if _EAGER_BREAKDOWN:
            _bk_s1b_sync_us += (_bk_time.perf_counter() - _bk_t0b) * 1e6
            _bk_calls += 1

        self._eager_calls += 1

        if not has_miss:
            self._eager_hits += 1
            if _EAGER_BREAKDOWN:
                _bk_t1 = _bk_time.perf_counter()
            if not _EAGER_LRU_SKIP:
                # O1+O2: vectorized LRU update (replaces Python for-loop)
                unique_arr = valid_ids.unique().cpu().numpy()
                emap_np = (self._emap_cpu_np[layer_idx]
                           if hasattr(self, '_emap_cpu_np') else None)
                if emap_np is not None:
                    valid_mask = (unique_arr >= 0) & (
                        unique_arr < len(emap_np))
                    lids = emap_np[unique_arr[valid_mask]]
                    lid_valid = lids[lids != -1]
                    slots = self._expert_to_slot_np[layer_idx, lid_valid]
                    resident = lid_valid[slots != -1]
                    if len(resident) > 0:
                        self._last_access[layer_idx, resident] = (
                            self.current_step)
                # Expert delta: routing churn vs previous step (phase-separated)
                if _KERNEL_PROFILE:
                    global _ed_delta_total_d, _ed_delta_count_d, _ed_routed_total_d
                    global _ed_delta_total_p, _ed_delta_count_p, _ed_routed_total_p
                    _phase = 'P' if self._prefillish_step else 'D'
                    _ed_key = (_phase, layer_idx)
                    cur_set = set(unique_arr.tolist())
                    prev = _ed_prev_gids.get(_ed_key)
                    if prev is not None:
                        _d = len(cur_set - prev)
                        if _phase == 'D':
                            _ed_delta_total_d += _d
                            _ed_routed_total_d += len(cur_set)
                            _ed_delta_count_d += 1
                        else:
                            _ed_delta_total_p += _d
                            _ed_routed_total_p += len(cur_set)
                            _ed_delta_count_p += 1
                    _ed_prev_gids[_ed_key] = cur_set
                # Trace: zero-miss row for complete N→N+1 coverage
                if _EXPERT_TRACE and _trace_fd is not None:
                    _trace_fd.write(_json.dumps({
                        "step": self.current_step,
                        "layer": layer_idx,
                        "routed_gids": sorted(unique_arr.tolist()),
                        "miss_gids": [],
                        "n_routed": len(unique_arr),
                        "n_miss": 0,
                        "n_cached": len(unique_arr),
                        "top_freq": [],
                    }) + "\n")
            elif _EXPERT_TRACE and _trace_fd is not None:
                # LRU skip mode: still need trace if enabled
                unique_arr = valid_ids.unique().cpu().numpy()
                _trace_fd.write(_json.dumps({
                    "step": self.current_step,
                    "layer": layer_idx,
                    "routed_gids": sorted(unique_arr.tolist()),
                    "miss_gids": [],
                    "n_routed": len(unique_arr),
                    "n_miss": 0,
                    "n_cached": len(unique_arr),
                    "top_freq": [],
                }) + "\n")
            # ── Prewarm trigger on hit path (next layer may have misses) ──
            if (self._prewarm_enabled and self._scratch_banks
                    and self._prefillish_step):
                self._prewarm_next_layer(layer_idx)
            if _EAGER_BREAKDOWN:
                _bk_s2_hit_us += (_bk_time.perf_counter() - _bk_t1) * 1e6
                _bk_hits += 1
                self._bk_maybe_log()
            self._eager_periodic_log(layer_idx)
            return

        # Miss detected → CPU fallback path (O1+O2: vectorized)
        if _EAGER_BREAKDOWN:
            _bk_t3 = _bk_time.perf_counter()
        self._eager_misses += 1
        unique_arr = valid_ids.unique().cpu().numpy()
        emap_np = (self._emap_cpu_np[layer_idx]
                   if hasattr(self, '_emap_cpu_np') else None)
        if emap_np is not None:
            valid_mask = (unique_arr >= 0) & (unique_arr < len(emap_np))
            valid_gids = unique_arr[valid_mask]
            lids = emap_np[valid_gids]
            mapped = lids != -1
            mapped_gids = valid_gids[mapped]
            mapped_lids = lids[mapped]

            # Slot check via numpy mirror
            slots = self._expert_to_slot_np[layer_idx, mapped_lids]
            is_cached = slots != -1

            # Batch LRU for cached
            cached_lids = mapped_lids[is_cached]
            if len(cached_lids) > 0:
                self._last_access[layer_idx, cached_lids] = self.current_step
            cached_needed = set(cached_lids.tolist())

            # Miss list
            miss_lids_arr = mapped_lids[~is_cached]
            miss_gids_arr = mapped_gids[~is_cached]
            miss_ids = miss_lids_arr.tolist()
            # lid_to_gid dict (needed for scratch overflow gid lookup + trace)
            lid_to_gid: Dict[int, int] = dict(
                zip(miss_lids_arr.tolist(), miss_gids_arr.tolist()))
            lid_to_gid.update(dict(
                zip(cached_lids.tolist(), mapped_gids[is_cached].tolist())))
            needed_local = set(mapped_lids.tolist())
        else:
            # Fallback: identity map (no expert_map), original Python path
            unique_gids = unique_arr.tolist()
            needed_local = {gid for gid in unique_gids
                           if 0 <= gid < self.local_num_experts}
            lid_to_gid = {gid: gid for gid in needed_local}
            miss_ids = []
            cached_needed: Set[int] = set()
            for lid in needed_local:
                if self._expert_to_slot[layer_idx][lid] != -1:
                    self._last_access[layer_idx, lid] = self.current_step
                    cached_needed.add(lid)
                else:
                    miss_ids.append(lid)

        self._eager_miss_experts += len(miss_ids)

        # Expert delta: routing churn vs previous step (miss path, phase-separated)
        if _KERNEL_PROFILE:
            _phase = 'P' if self._prefillish_step else 'D'
            _ed_key = (_phase, layer_idx)
            cur_set = set(unique_arr.tolist())
            prev = _ed_prev_gids.get(_ed_key)
            if prev is not None:
                _d = len(cur_set - prev)
                if _phase == 'D':
                    _ed_delta_total_d += _d
                    _ed_routed_total_d += len(cur_set)
                    _ed_delta_count_d += 1
                else:
                    _ed_delta_total_p += _d
                    _ed_routed_total_p += len(cur_set)
                    _ed_delta_count_p += 1
            _ed_prev_gids[_ed_key] = cur_set

            # TailDelta: prefill only — bank reuse across steps for same layer
            # "If bank kept prev step's miss set, how many need fresh H2D?"
            if _phase == 'P' and miss_ids:
                global _td_delta_total, _td_reuse_total
                global _td_miss_total, _td_count
                miss_gid_set = set(
                    miss_gids_arr.tolist() if emap_np is not None
                    else [lid_to_gid.get(lid, lid) for lid in miss_ids])
                prev_miss = _td_prev_miss.get(layer_idx)
                if prev_miss is not None:
                    reuse = len(miss_gid_set & prev_miss)
                    delta = len(miss_gid_set) - reuse
                    _td_reuse_total += reuse
                    _td_delta_total += delta
                    _td_miss_total += len(miss_gid_set)
                    _td_count += 1
                _td_prev_miss[layer_idx] = miss_gid_set

        # ── Trace: per-layer routing + miss gids for correlation analysis ──
        # Records ALL layers (including zero-miss) for complete N→N+1 analysis
        unique_gids_list = unique_arr.tolist()
        if _EXPERT_TRACE and _trace_fd is not None:
            miss_gids = sorted(lid_to_gid.get(lid, lid)
                               for lid in miss_ids) if miss_ids else []
            # Top-32 routing frequency for entropy/distribution analysis
            gid_counts = {}
            for g in valid_ids.tolist():
                gid_counts[g] = gid_counts.get(g, 0) + 1
            top_freq = sorted(gid_counts.items(),
                              key=lambda kv: kv[1], reverse=True)[:32]
            _trace_fd.write(_json.dumps({
                "step": self.current_step,
                "layer": layer_idx,
                "routed_gids": sorted(unique_gids_list),
                "miss_gids": miss_gids,
                "n_routed": len(unique_gids_list),
                "n_miss": len(miss_ids),
                "n_cached": len(cached_needed),
                "top_freq": top_freq,
            }) + "\n")

        # ── Stage 3→4 boundary ──
        if _EAGER_BREAKDOWN:
            _bk_s3_classify_us += (_bk_time.perf_counter() - _bk_t3) * 1e6
            _bk_t4 = _bk_time.perf_counter()

        # ── Prewarm V2: Early H/R split before sync_fetch ──
        _pw_bank = None
        _pw_bank_idx = -1
        H_gids: List[int] = []
        H_lids: List[int] = []
        R_gids: List[int] = []
        R_lids: List[int] = []
        if miss_ids and self._prewarm_enabled and self._scratch_banks:
            # P1: find prewarm bank targeting this layer
            for _pbi, _pb in enumerate(self._scratch_banks):
                if (_pb.prewarm_target_layer == layer_idx
                        and _pb.state in (_BankState.READY,
                                          _BankState.FILLING)
                        and _pb.gid_to_slot is not None):
                    _pw_bank = _pb
                    _pw_bank_idx = _pbi
                    break
            # P2: H/R classification (before sync_fetch!)
            if _pw_bank is not None:
                prewarmed_set = set(_pw_bank.prewarmed_gids)
                for lid in miss_ids:
                    gid = lid_to_gid.get(lid, lid)
                    if gid in prewarmed_set:
                        H_gids.append(gid)
                        H_lids.append(lid)
                    else:
                        R_gids.append(gid)
                        R_lids.append(lid)

        # ── Sync fetch: R-only when prewarm active, else full miss_ids ──
        fetched, skipped, evict_reason = 0, 0, None
        if _pw_bank is not None:
            if R_lids:
                fetched, skipped, evict_reason = self._sync_fetch(
                    layer_idx, R_lids,
                    protected_local_ids=cached_needed,
                    skip_device_sync=True)
            # R_overflow = R tail that couldn't be fetched
            R_overflow_gids = R_gids[fetched:]
            R_overflow_lids = R_lids[fetched:]
        elif miss_ids:
            fetched, skipped, evict_reason = self._sync_fetch(
                layer_idx, miss_ids,
                protected_local_ids=cached_needed)

        self._eager_fetch_failures += skipped

        # Per-miss-call capacity diagnostics
        if miss_ids and _EAGER_CAP_LOG:
            self._log_eager_capacity(
                layer_idx, needed_local, cached_needed, miss_ids,
                fetched, skipped, evict_reason)

        # ── Stage 4→5 boundary ──
        if _EAGER_BREAKDOWN:
            _bk_s4_fetch_us += (_bk_time.perf_counter() - _bk_t4) * 1e6
            _bk_t5 = _bk_time.perf_counter()

        # Update cache_map for this layer (base slots: fetched R now resident)
        self._update_cache_map(layer_idx, layer)

        # ── Scratch: prewarm consume or hard fallback ──
        if _pw_bank is not None:
            bank_avail = self._scratch_capacity - _pw_bank.n_prewarmed
            if len(R_overflow_gids) <= bank_avail:
                # NORMAL: all H + all R_overflow fit in bank
                if H_gids or R_overflow_gids:
                    self._consume_prewarm(
                        _pw_bank, _pw_bank_idx, layer_idx,
                        H_gids, R_overflow_gids, R_overflow_lids, layer)
                else:
                    self._discard_prewarm(_pw_bank)
            else:
                # HARD FALLBACK: R_overflow > residual. Discard prewarm
                # and re-serve via reserve_scratch (uses full capacity,
                # reclaims wasted prewarm slots for better coverage).
                all_overflow_gids = H_gids + R_overflow_gids
                all_overflow_lids = H_lids + R_overflow_lids
                self._discard_prewarm(_pw_bank)
                if all_overflow_gids:
                    self.reserve_scratch(
                        layer_idx, all_overflow_gids,
                        all_overflow_lids, layer)
                self._pw_fallback += 1
        elif skipped > 0 and self._scratch_banks:
            # No prewarm → legacy reserve_scratch fallback
            overflow_lids = miss_ids[fetched:]
            overflow_gids = [lid_to_gid[lid] for lid in overflow_lids
                            if lid in lid_to_gid]
            if overflow_gids:
                self.reserve_scratch(
                    layer_idx, overflow_gids, overflow_lids, layer)

        # ── Prewarm trigger for next layer ──
        # Known non-resident based → no miss_ids gate needed
        if (self._prewarm_enabled and self._scratch_banks
                and self._prefillish_step):
            self._prewarm_next_layer(layer_idx)

        if _EAGER_BREAKDOWN:
            _bk_s5_post_us += (_bk_time.perf_counter() - _bk_t5) * 1e6
            _bk_misses += 1
            self._bk_maybe_log()
        self._eager_periodic_log(layer_idx)

    def _bk_maybe_log(self):
        """Periodic breakdown log (VLLM_EAGER_BREAKDOWN=1)."""
        if not _EAGER_BREAKDOWN:
            return
        global _bk_calls, _bk_hits, _bk_misses
        global _bk_s1a_gather_us, _bk_s1b_sync_us
        global _bk_s2_hit_us, _bk_s3_classify_us
        global _bk_s4_fetch_us, _bk_s5_post_us
        if _bk_calls > 0 and _bk_calls % _bk_interval == 0:
            n = _bk_calls
            s1_total = _bk_s1a_gather_us + _bk_s1b_sync_us
            logger.info(
                "[EagerBreakdown] n=%d hit=%d miss=%d | "
                "S1a_gather=%.0fμs S1b_sync=%.0fμs (S1=%.0fμs) "
                "S2_hit=%.0fμs S3_classify=%.0fμs "
                "S4_fetch=%.0fμs S5_post=%.0fμs | "
                "avg: S1a=%.1f S1b=%.1f S2=%.1f S3=%.1f S4=%.1f S5=%.1f",
                n, _bk_hits, _bk_misses,
                _bk_s1a_gather_us, _bk_s1b_sync_us, s1_total,
                _bk_s2_hit_us, _bk_s3_classify_us,
                _bk_s4_fetch_us, _bk_s5_post_us,
                _bk_s1a_gather_us / max(n, 1),
                _bk_s1b_sync_us / max(n, 1),
                _bk_s2_hit_us / max(_bk_hits, 1),
                _bk_s3_classify_us / max(_bk_misses, 1),
                _bk_s4_fetch_us / max(_bk_misses, 1),
                _bk_s5_post_us / max(_bk_misses, 1))
            # Reset for next window
            _bk_calls = 0
            _bk_hits = 0
            _bk_misses = 0
            _bk_s1a_gather_us = 0.0
            _bk_s1b_sync_us = 0.0
            _bk_s2_hit_us = 0.0
            _bk_s3_classify_us = 0.0
            _bk_s4_fetch_us = 0.0
            _bk_s5_post_us = 0.0

    def _eager_periodic_log(self, layer_idx: int):
        """Log eager boundary stats every N calls."""
        if (self._eager_calls > 0
                and self._eager_calls % self._eager_log_interval == 0):
            logger.info(
                "[Eager] L%d calls=%d hit=%d miss=%d "
                "miss_experts=%d fetch_fail=%d",
                layer_idx, self._eager_calls,
                self._eager_hits, self._eager_misses,
                self._eager_miss_experts, self._eager_fetch_failures)
            # Expert delta log (decode / prefill separated)
            if _KERNEL_PROFILE:
                if _ed_delta_count_d > 0:
                    _avg_d = _ed_delta_total_d / _ed_delta_count_d
                    _avg_r_d = _ed_routed_total_d / _ed_delta_count_d
                    logger.info(
                        "[ExpertDelta:decode] n=%d "
                        "avg_new=%.1f avg_routed=%.1f churn=%.1f%%",
                        _ed_delta_count_d, _avg_d, _avg_r_d,
                        _avg_d / max(_avg_r_d, 1) * 100)
                if _ed_delta_count_p > 0:
                    _avg_p = _ed_delta_total_p / _ed_delta_count_p
                    _avg_r_p = _ed_routed_total_p / _ed_delta_count_p
                    logger.info(
                        "[ExpertDelta:prefill] n=%d "
                        "avg_new=%.1f avg_routed=%.1f churn=%.1f%%",
                        _ed_delta_count_p, _avg_p, _avg_r_p,
                        _avg_p / max(_avg_r_p, 1) * 100)
                # LayerCycle: Li→Li+1 intra-step (prefill overlap window)
                if _lc_count_p > 0:
                    _avg_lp = _lc_total_us_p / _lc_count_p
                    logger.info(
                        "[LayerCycle:prefill] n=%d "
                        "avg=%.0fμs (%.2fms)/MoE-layer "
                        "(= overlap window for cross-layer copy)",
                        _lc_count_p, _avg_lp, _avg_lp / 1000)
                # StepCycle: same-layer cross-step (full step budget)
                if _sc_count_d > 0:
                    _avg_sd = _sc_total_us_d / _sc_count_d
                    logger.info(
                        "[StepCycle:decode] n=%d "
                        "avg=%.0fμs (%.2fms)/step "
                        "(= full decode step wall-clock)",
                        _sc_count_d, _avg_sd, _avg_sd / 1000)
                if _sc_count_p > 0:
                    _avg_sp = _sc_total_us_p / _sc_count_p
                    logger.info(
                        "[StepCycle:prefill] n=%d "
                        "avg=%.0fμs (%.2fms)/step",
                        _sc_count_p, _avg_sp, _avg_sp / 1000)
                # TailDelta: bank reuse viability
                if _td_count > 0:
                    _avg_miss = _td_miss_total / _td_count
                    _avg_delta = _td_delta_total / _td_count
                    _avg_reuse = _td_reuse_total / _td_count
                    _pct = _avg_reuse / max(_avg_miss, 1) * 100
                    logger.info(
                        "[TailDelta:prefill] n=%d "
                        "avg_miss=%.1f avg_delta=%.1f avg_reuse=%.1f "
                        "(%.0f%% bank reusable across steps)",
                        _td_count, _avg_miss, _avg_delta,
                        _avg_reuse, _pct)
            # Prewarm stats
            if self._pw_consumes > 0:
                _pw_h = self._pw_H_total / self._pw_consumes
                _pw_r = self._pw_R_total / self._pw_consumes
                _pw_rf = self._pw_Rfill_total / self._pw_consumes
                logger.info(
                    "[Prewarm] consumes=%d fallback=%d "
                    "H_avg=%.1f R_avg=%.1f Rfill_avg=%.1f",
                    self._pw_consumes, self._pw_fallback,
                    _pw_h, _pw_r, _pw_rf)

    def _log_eager_capacity(self, layer_idx: int,
                              needed_local: Set[int],
                              cached_needed: Set[int],
                              miss_ids: List[int],
                              fetched: int, skipped: int,
                              evict_reason: Optional[Dict[str, int]]):
        """Log per-miss-call capacity breakdown (post-fetch)."""
        total_slots = len(self._slot_to_expert[layer_idx])
        free_count = len(self._free_slots[layer_idx])
        unmapped = len(self._unmapped_slots[layer_idx])
        reason_str = ""
        if evict_reason:
            reason_str = (
                f" evict_fail=[empty={evict_reason['empty']} "
                f"unmap={evict_reason['unmapped']} "
                f"pin={evict_reason['pinned']} "
                f"prot={evict_reason['protected']} "
                f"pend={evict_reason['pending']}]")
        logger.info(
            "[Eager-Cap] L%d step=%d localE=%d need=%d cached=%d "
            "miss=%d fetched=%d skipped=%d | "
            "free=%d unmapped=%d slots=%d%s",
            layer_idx, self.current_step, self.local_num_experts,
            len(needed_local), len(cached_needed), len(miss_ids),
            fetched, skipped,
            free_count, unmapped, total_slots, reason_str)

    # ================================================================
    # 3-D: Inter-group step (group boundary handler)
    # ================================================================

    def inter_group_step(self, group_idx: int, layers):
        """3-D Group boundary: previous group's fresh routing → async prefetch next.

        Called from expert_group_boundary custom op between graph pieces.
        Uses current-step routing from previous group layers to predict
        and async-prefetch experts for the next group.

        Args:
            group_idx: 1-based index of the NEXT group (after boundary).
            layers: List of all FusedMoE layer modules (ordered by layer idx).
        """
        if not self._group_ranges or group_idx < 1:
            return
        if group_idx >= len(self._group_ranges):
            return

        # Skip when no eviction this step (runtime).
        # _dormant is set by runner each step via has_evicted_experts().
        # When dormant, routing snapshots are stale (_snapshot_active=False).
        # NOTE: removed static `max_resident >= local_num_experts` guard.
        # With VMM Phase C, max_resident = max_slots (e.g. 512 == local_E)
        # even when actual resident count < local_E after shrink.
        # _dormant alone correctly gates on runtime eviction state.
        if self._dormant:
            return

        self._group_boundary_calls += 1

        prev_range = self._group_ranges[group_idx - 1]
        next_range = self._group_ranges[group_idx]

        # Phase A: Read fresh routing from previous group layers (GPU→CPU)
        # Only read layers with active snapshot capture — stale snapshots
        # (from previous steps when _snapshot_active was true) are skipped.
        prev_routing_sets: Dict[int, Set[int]] = {}
        for i in prev_range:
            if i >= len(layers) or layers[i] is None:
                continue
            layer = layers[i]
            # Guard: skip layers without active routing capture
            if not getattr(layer, '_snapshot_active', False):
                continue
            if not hasattr(layer, '_routing_snapshot'):
                continue
            rlen_t = getattr(layer, '_routing_len', None)
            rlen = int(rlen_t[0].item()) if rlen_t is not None else 0
            if rlen <= 0:
                continue
            snap = layer._routing_snapshot[:rlen].cpu()
            unique_gids = {int(g) for g in snap.unique().tolist() if g >= 0}
            # Map global → local expert IDs
            emap = (layer._expert_map if hasattr(layer, '_expert_map')
                    else None)
            if emap is not None:
                emap_cpu = emap.cpu() if emap.device.type != 'cpu' else emap
                needed_local: Set[int] = set()
                for gid in unique_gids:
                    if 0 <= gid < emap_cpu.shape[0]:
                        lid = emap_cpu[gid].item()
                        if lid != -1:
                            needed_local.add(lid)
            else:
                needed_local = {gid for gid in unique_gids
                                if 0 <= gid < self.local_num_experts}
            prev_routing_sets[i] = needed_local

        if not prev_routing_sets:
            return

        # Phase B: Predict next group routing using previous group's tail
        # Heuristic: adjacent layers have correlated routing.
        # Use union of last 4 layers in previous group.
        tail_layers = sorted(prev_routing_sets.keys())[-4:]
        prediction_union: Set[int] = set()
        for li in tail_layers:
            prediction_union |= prev_routing_sets[li]

        # Phase C: Fetch predicted experts for each next-group layer
        # We launch DMA on copy_stream, then synchronize before updating
        # cache_map to ensure weights are fully copied before compute
        # can observe the new slot mappings.
        layers_with_fetches: List[int] = []
        total_miss = 0
        total_loaded = 0
        total_skipped = 0
        for i in next_range:
            if i >= len(layers) or layers[i] is None:
                continue
            miss_ids = []
            cached_needed: Set[int] = set()
            for lid in prediction_union:
                if lid < self.local_num_experts:
                    if self._expert_to_slot[i][lid] != -1:
                        cached_needed.add(lid)
                    else:
                        miss_ids.append(lid)
            total_miss += len(miss_ids)
            if miss_ids:
                self._async_fetch(i, miss_ids, protected=cached_needed)
                layers_with_fetches.append(i)
                for lid in miss_ids:
                    if self._expert_to_slot[i][lid] != -1:
                        total_loaded += 1
                    else:
                        total_skipped += 1

        # Synchronize copy_stream BEFORE updating cache_map.
        # This ensures all H2D weight copies are complete before
        # the next graph piece can observe the new slot mappings.
        if layers_with_fetches:
            self._copy_stream.synchronize()
            for i in layers_with_fetches:
                self._update_cache_map(i, layers[i])

        self._group_prefetch_experts += len(prediction_union)
        self._group_prefetch_loaded += total_loaded
        self._group_prefetch_skipped += total_skipped

        # Periodic log
        if (self._group_boundary_calls > 0
                and self._group_boundary_calls % self._group_log_interval == 0):
            logger.info(
                "[Group3D] calls=%d predict_union=%d "
                "loaded=%d skipped=%d (cumulative)",
                self._group_boundary_calls,
                self._group_prefetch_experts,
                self._group_prefetch_loaded,
                self._group_prefetch_skipped)

    def _update_cache_map(self, layer_idx: int, layer):
        """Rebuild cache_map for a layer. In-place copy for CUDA graph compat."""
        cache_map = torch.full(
            (self.global_num_experts,), -1, dtype=torch.int32)

        emap = (layer._expert_map if hasattr(layer, '_expert_map')
                else None)
        if emap is not None:
            emap_cpu = emap.cpu() if emap.device.type != 'cpu' else emap
            for gid in range(emap_cpu.shape[0]):
                lid = emap_cpu[gid].item()
                if lid == -1:
                    continue
                slot = self._expert_to_slot[layer_idx][lid]
                if slot != -1:
                    cache_map[gid] = slot
        else:
            for lid in range(self.local_num_experts):
                slot = self._expert_to_slot[layer_idx][lid]
                if slot != -1:
                    cache_map[lid] = slot

        # In-place copy preserves data_ptr (CUDA graph safe)
        layer._cache_map.copy_(cache_map.to(layer._cache_map.device))

    # ================================================================
    # Scratch bank: shared overflow buffer for hard guarantee
    # ================================================================

    def set_scratch(
        self,
        scratch_w13: torch.Tensor,
        scratch_w2: torch.Tensor,
        threshold: int,
        capacity: int,
        num_banks: int = 2,
    ) -> None:
        """Configure scratch bank(s). Called once by gpu_worker during init.

        Args:
            scratch_w13: shared scratch tensor [num_banks*capacity, N13, K]
            scratch_w2: shared scratch tensor [num_banks*capacity, N2, K2]
            threshold: base capacity (= max_resident). Slots >= threshold
                       are scratch slots in the kernel.
            capacity: number of scratch slots per bank.
            num_banks: 1 = single-bank miss-driven, 2 = double-buffer.
        """
        assert capacity > 0, "scratch capacity must be > 0"
        if threshold + capacity > self.global_num_experts:
            logger.info(
                "Scratch virtual slots [%d, %d) exceed global_num_experts %d "
                "(expected for virtual slot namespace)",
                threshold, threshold + capacity, self.global_num_experts,
            )

        self._scratch_threshold = threshold
        self._scratch_capacity = capacity

        # Shared slot_values: [threshold, threshold+1, ..., threshold+capacity-1]
        # Both banks share the same slot namespace in _cache_map
        self._scratch_slot_values = torch.arange(
            threshold, threshold + capacity,
            dtype=torch.int32, device=self.device)

        # Create bank(s) by slicing the num_banks*capacity tensors
        self._scratch_banks = []
        for bi in range(num_banks):
            off = bi * capacity
            ready_ev = torch.cuda.Event()
            done_ev = torch.cuda.Event()
            # Record initial done event so first reserve doesn't block
            done_ev.record()
            gid_buf = torch.empty(
                capacity, dtype=torch.int64, device=self.device)
            w13_ready_ev = torch.cuda.Event()
            bank = _ScratchBank(
                w13=scratch_w13[off:off + capacity],
                w2=scratch_w2[off:off + capacity],
                ready_event=ready_ev,
                done_event=done_ev,
                gid_buffer=gid_buf,
                w13_ready_event=w13_ready_ev,
            )
            self._scratch_banks.append(bank)

        self._scratch_current_bank = 0

        # 1-bank mode: prewarm needs 2 banks, force off
        if num_banks < 2 and self._prewarm_enabled:
            self._prewarm_enabled = False
            logger.info("Scratch 1-bank mode: prewarm forced OFF")

        logger.info(
            "Scratch configured: threshold=%d capacity=%d "
            "banks=%d w13=%s w2=%s device=%s prewarm=%s cap=%d",
            threshold, capacity, len(self._scratch_banks),
            list(scratch_w13.shape), list(scratch_w2.shape), self.device,
            self._prewarm_enabled, self._prewarm_cap)

        # Scratch-aware resident floor validation
        floor = self._resident_floor(0)
        if self.max_resident < floor:
            logger.warning(
                "Scratch-aware floor UNSATISFIABLE: max_resident=%d < floor=%d "
                "(localE=%d scratch=%d). Skip/quality loss expected.",
                self.max_resident, floor,
                self.local_num_experts, self._scratch_capacity)
        else:
            group_sz = self._vmm_pool.group_size if self._vmm_pool else 1
            headroom = self.max_resident - floor
            logger.info(
                "Scratch-aware floor: floor=%d headroom=%d groups "
                "(max_res=%d localE=%d scratch=%d)",
                floor, headroom // group_sz,
                self.max_resident, self.local_num_experts,
                self._scratch_capacity)

        # Layer-concentrated shrink policy log
        n_eligible = len(self._shrink_eligible)
        if n_eligible < self.num_layers:
            eligible_range = (min(self._shrink_eligible),
                              max(self._shrink_eligible)) if n_eligible else (-1, -1)
            logger.info(
                "Shrink policy: %d/%d layers eligible (L%d-L%d), "
                "L0 protected, tail_n=%d (env VLLM_SHRINK_TAIL_N=%s)",
                n_eligible, self.num_layers,
                eligible_range[0], eligible_range[1],
                self._shrink_tail_n,
                os.environ.get("VLLM_SHRINK_TAIL_N", "<default>"))
        else:
            logger.info(
                "Shrink policy: all %d layers eligible (L0 protected), "
                "tail_n=%d (set VLLM_SHRINK_TAIL_N to concentrate offload)",
                self.num_layers, self._shrink_tail_n)

    def _discard_prewarm(self, bank: _ScratchBank) -> None:
        """Discard a prewarmed bank (READY or FILLING → IDLE)."""
        if _EXPERT_DEBUG:
            logger.info(
                "[Scratch] discard_prewarm: target_layer=%d prewarmed_n=%d",
                bank.prewarm_target_layer,
                len(bank.prewarmed_gids) if bank.prewarmed_gids else 0)
        bank.state = _BankState.IDLE
        bank.prewarm_target_layer = -1
        bank.prewarmed_gids = None
        bank.prewarmed_lids = None
        bank.gid_to_slot = None
        bank.consumed_gids = None
        bank.n_prewarmed = 0
        bank.owner_layer_idx = -1
        bank.owner_layer = None
        bank.current_gids = None
        bank.n_active = 0

    # ================================================================
    # Fixed-tail double-buffer (VLLM_FIXED_TAIL=1)
    # ================================================================

    def _activate_fixed_tail(self, layers) -> None:
        """One-time activation: compute per-layer tail, patch cache_map."""
        slots_np = self._expert_to_slot_np
        local_E = self.local_num_experts

        self._ft_tail_lids = []
        for li in range(self.num_layers):
            tail = [lid for lid in range(local_E)
                    if slots_np[li][lid] == -1
                    and lid in self._cpu_pool[li]]
            self._ft_tail_lids.append(tail)

        max_tail = max(len(t) for t in self._ft_tail_lids)
        layers_with_tail = sum(1 for t in self._ft_tail_lids if t)

        # Invariant: tail must fit in scratch
        assert max_tail <= self._scratch_capacity, (
            f"[FixedTail] max_tail={max_tail} > scratch_capacity="
            f"{self._scratch_capacity}. Reduce shrink or increase scratch.")
        # L0 should have no tail (always fully resident)
        assert len(self._ft_tail_lids[0]) == 0, (
            f"[FixedTail] L0 has {len(self._ft_tail_lids[0])} non-resident "
            f"experts. L0 must be fully resident.")

        # Static cache_map patch (one-time)
        emap_np_list = getattr(self, '_emap_cpu_np', None)
        slot_values = self._scratch_slot_values
        for li in range(self.num_layers):
            tail = self._ft_tail_lids[li]
            if not tail:
                continue
            layer = self._moe_layers[li]
            # Build gid list for this layer's tail experts
            gid_list = []
            emap_np = emap_np_list[li] if emap_np_list else None
            if emap_np is not None:
                lid_to_gid = {}
                for gid in range(len(emap_np)):
                    lid = int(emap_np[gid])
                    if lid in tail and lid not in lid_to_gid:
                        lid_to_gid[lid] = gid
                for lid in tail:
                    if lid in lid_to_gid:
                        gid_list.append(lid_to_gid[lid])
            else:
                gid_list = list(tail)

            n = len(gid_list)
            assert n == len(tail), (
                f"[FixedTail] L{li}: gid_list={n} != tail={len(tail)}. "
                f"Some lid→gid mapping failed. "
                f"Missing lids: {set(tail) - set(lid_to_gid.keys()) if emap_np is not None else set()}")
            gid_t = torch.tensor(gid_list, dtype=torch.int64,
                                 device=self.device)
            layer._cache_map.index_copy_(0, gid_t, slot_values[:n])

        self._fixed_tail_active = True
        self._cache_map_needs_rebuild = False
        logger.info(
            "[FixedTail] ACTIVE: max_tail=%d layers_with_tail=%d/%d "
            "scratch_capacity=%d",
            max_tail, layers_with_tail, self.num_layers,
            self._scratch_capacity)

    def _fixed_tail_prefetch_next(self, layer_idx: int) -> None:
        """Prefetch next layer's tail into alternate bank (no .item())."""
        next_idx = layer_idx + 1
        if next_idx >= self.num_layers:
            return
        tail_lids = self._ft_tail_lids[next_idx]
        if not tail_lids:
            return  # next layer fully resident, no prefetch

        alt = 1 - self._ft_current_bank
        bank = self._scratch_banks[alt]

        # Sequential invariant: bank MUST be IDLE
        assert bank.state == _BankState.IDLE, (
            f"[FixedTail] bank {alt} not IDLE: {bank.state.name} "
            f"owner={bank.owner_layer_idx}. Layer cycle too short?")

        n = len(tail_lids)
        bank.state = _BankState.FILLING
        self._copy_stream.wait_event(bank.done_event)
        with torch.cuda.stream(self._copy_stream):
            for i, lid in enumerate(tail_lids):
                w13_cpu, w2_cpu = self._cpu_pool[next_idx][lid]
                bank.w13[i].copy_(w13_cpu, non_blocking=True)
                bank.w2[i].copy_(w2_cpu, non_blocking=True)
            bank.ready_event.record(self._copy_stream)

        bank.state = _BankState.READY
        bank.owner_layer_idx = next_idx
        bank.owner_layer = self._moe_layers[next_idx]
        bank.n_active = n
        self._ft_current_bank = alt

    # ================================================================
    # Step-boundary static topology (VLLM_STEP_BOUNDARY=1)
    # ================================================================

    def _update_lru_from_snapshots(self, layers) -> None:
        """Batch LRU update from previous step's routing snapshots.

        Called at step boundary only — .item() is acceptable here
        (not on forward hot path).
        """
        for li, layer in enumerate(layers):
            if li >= self.num_layers:
                break
            if (not hasattr(layer, '_routing_len')
                    or layer._routing_len is None):
                continue
            rlen = int(layer._routing_len.item())  # step boundary OK
            if rlen <= 0:
                continue
            snapshot = layer._routing_snapshot[:rlen]
            unique_gids = snapshot.unique().cpu().numpy()
            emap_np = self._emap_cpu_np[li]
            if emap_np is None:
                continue
            valid = unique_gids[
                (unique_gids >= 0) & (unique_gids < len(emap_np))]
            lids = emap_np[valid]
            resident = lids[
                (lids >= 0) & (self._expert_to_slot_np[li][lids] >= 0)]
            if len(resident) > 0:
                self._last_access[li, resident] = self.current_step

    def _recompute_step_topology(self, layers) -> None:
        """Scan all layers, compute per-layer non-resident (tail) lists.

        Updates _sb_tail_lids and _sb_tail_gids.
        """
        slots_np = self._expert_to_slot_np
        local_E = self.local_num_experts
        emap_np_list = getattr(self, '_emap_cpu_np', None)

        self._sb_tail_lids = []
        self._sb_tail_gids = []
        for li in range(self.num_layers):
            tail_lids = [lid for lid in range(local_E)
                         if slots_np[li][lid] == -1
                         and lid in self._cpu_pool[li]]
            # Build gid list (lid → gid mapping via emap)
            gid_list = []
            emap_np = emap_np_list[li] if emap_np_list else None
            if emap_np is not None:
                lid_to_gid = {}
                for gid in range(len(emap_np)):
                    lid = int(emap_np[gid])
                    if lid in tail_lids and lid not in lid_to_gid:
                        lid_to_gid[lid] = gid
                gid_list = [lid_to_gid[lid] for lid in tail_lids
                            if lid in lid_to_gid]
            else:
                gid_list = list(tail_lids)

            assert len(gid_list) == len(tail_lids), (
                f"[StepBoundary] L{li}: gid_list={len(gid_list)} != "
                f"tail_lids={len(tail_lids)}. Some lid→gid mapping failed. "
                f"Missing lids: "
                f"{set(tail_lids) - set(lid_to_gid.keys()) if emap_np is not None else set()}")
            self._sb_tail_lids.append(tail_lids)
            self._sb_tail_gids.append(gid_list)

        max_tail = max(len(t) for t in self._sb_tail_lids) if self._sb_tail_lids else 0
        layers_with_tail = sum(1 for t in self._sb_tail_lids if t)

        assert max_tail <= self._scratch_capacity, (
            f"[StepBoundary] max_tail={max_tail} > scratch_capacity="
            f"{self._scratch_capacity}. Reduce shrink or increase scratch.")

        logger.info(
            "[StepBoundary] topology recomputed: max_tail=%d "
            "layers_with_tail=%d/%d scratch=%d",
            max_tail, layers_with_tail, self.num_layers,
            self._scratch_capacity)

    def _patch_step_cache_maps(self, layers) -> None:
        """Patch _cache_map for all layers with scratch slot mappings.

        One-time per topology change. Maps tail gids → scratch slot indices.
        """
        slot_values = self._scratch_slot_values
        for li in range(self.num_layers):
            gid_list = self._sb_tail_gids[li]
            if not gid_list:
                continue
            layer = self._moe_layers[li]
            if not hasattr(layer, '_cache_map') or layer._cache_map is None:
                continue
            n = len(gid_list)
            gid_t = torch.tensor(gid_list, dtype=torch.int64,
                                 device=self.device)
            layer._cache_map.index_copy_(0, gid_t, slot_values[:n])

        self._cache_map_needs_rebuild = False

    def _start_initial_prefetch(self) -> None:
        """Prefetch first layer that has a tail into bank 0.

        Called once after topology recompute. If L0 has no tail,
        find the first layer with tail and prefetch it.
        """
        if not self._sb_tail_lids:
            return

        # Find first layer with non-empty tail
        first_li = -1
        for li in range(self.num_layers):
            if self._sb_tail_lids[li]:
                first_li = li
                break
        if first_li < 0:
            return  # all layers fully resident

        tail_lids = self._sb_tail_lids[first_li]
        n = len(tail_lids)

        # Use bank 0 for initial prefetch
        bank = self._scratch_banks[0]
        assert bank.state == _BankState.IDLE, (
            f"[StepBoundary] bank 0 not IDLE for initial prefetch: "
            f"{bank.state.name}")

        bank.state = _BankState.FILLING
        self._copy_stream.wait_event(bank.done_event)
        with torch.cuda.stream(self._copy_stream):
            for i, lid in enumerate(tail_lids):
                w13_cpu, _ = self._cpu_pool[first_li][lid]
                bank.w13[i].copy_(w13_cpu, non_blocking=True)
            if bank.w13_ready_event is not None:
                bank.w13_ready_event.record(self._copy_stream)
            for i, lid in enumerate(tail_lids):
                _, w2_cpu = self._cpu_pool[first_li][lid]
                bank.w2[i].copy_(w2_cpu, non_blocking=True)
            bank.ready_event.record(self._copy_stream)

        bank.state = _BankState.READY
        bank.owner_layer_idx = first_li
        bank.owner_layer = self._moe_layers[first_li]
        bank.n_active = n
        self._sb_current_bank = 0

        logger.info(
            "[StepBoundary] initial prefetch: L%d tail=%d bank=0",
            first_li, n)

    def step_prefetch_next(self, layer_idx: int) -> None:
        """Double-buffer prefetch: load next layer's tail into alt bank.

        Called from forward_cuda after bank.ready_event wait, before kernel.
        Overlaps H2D (copy_stream) with compute (current stream).
        """
        next_idx = layer_idx + 1
        if next_idx >= self.num_layers:
            return
        if not self._sb_tail_lids:
            return
        tail_lids = self._sb_tail_lids[next_idx]
        if not tail_lids:
            return  # next layer fully resident, no prefetch needed

        # If next layer is already armed on any bank, skip duplicate prefetch
        for b in self._scratch_banks:
            if (b.owner_layer_idx == next_idx
                    and b.state in (_BankState.READY, _BankState.FILLING)):
                return

        alt = 1 - self._sb_current_bank
        bank = self._scratch_banks[alt]

        # Alt bank must be IDLE (released by previous layer)
        assert bank.state == _BankState.IDLE, (
            f"[StepBoundary] bank {alt} not IDLE for prefetch L{next_idx}: "
            f"{bank.state.name} owner={bank.owner_layer_idx}")

        n = len(tail_lids)
        bank.state = _BankState.FILLING
        self._copy_stream.wait_event(bank.done_event)
        with torch.cuda.stream(self._copy_stream):
            for i, lid in enumerate(tail_lids):
                w13_cpu, _ = self._cpu_pool[next_idx][lid]
                bank.w13[i].copy_(w13_cpu, non_blocking=True)
            if bank.w13_ready_event is not None:
                bank.w13_ready_event.record(self._copy_stream)
            for i, lid in enumerate(tail_lids):
                _, w2_cpu = self._cpu_pool[next_idx][lid]
                bank.w2[i].copy_(w2_cpu, non_blocking=True)
            bank.ready_event.record(self._copy_stream)

        bank.state = _BankState.READY
        bank.owner_layer_idx = next_idx
        bank.owner_layer = self._moe_layers[next_idx]
        bank.n_active = n
        self._sb_current_bank = alt

    # ================================================================
    # Cutoff-boundary (VLLM_CUTOFF_BOUNDARY=1)
    # ================================================================

    def _cutoff_shrink(self, min_pages: int) -> Tuple[int, int]:
        """Suffix-cut shrink: evict highest-lid experts, L1..L47 uniform.

        L0 excluded (always resident-only). L1..L47 evict same lids.
        _shrink_eligible ignored — cutoff mode uses its own policy.
        """
        pool = self._vmm_pool
        if not pool:
            return 0, 0
        gs = pool.group_size
        gp = pool.group_pages
        n_layers = self.num_layers - 1  # L0 excluded
        if n_layers <= 0:
            return 0, 0

        pages_per_step = n_layers * gp
        steps_needed = math.ceil(min_pages / pages_per_step)
        new_cutoff = self._resident_cutoff - steps_needed * gs

        # Floor guard: scratch must fit tail
        floor = max(0, self.local_num_experts - self._scratch_capacity)
        new_cutoff = max(new_cutoff, floor)
        if new_cutoff >= self._resident_cutoff:
            return 0, 0

        # Collect victims: L1..L47, lids [new_cutoff, old_cutoff)
        victims = []
        for layer in range(1, self.num_layers):
            for lid in range(new_cutoff, self._resident_cutoff):
                slot = self._expert_to_slot[layer][lid]
                if slot >= 0:
                    victims.append((layer, slot))

        if not victims:
            return 0, 0

        freed_pages = pool.unmap_expert_slots(victims)
        groups_evicted = self._update_tracking_after_evict(victims)

        old_cutoff = self._resident_cutoff
        self._resident_cutoff = new_cutoff
        self._topology_dirty = True
        self._cache_map_needs_rebuild = True

        logger.info(
            "[CutoffBoundary] shrink: cutoff %d→%d, freed %d pages, "
            "%d groups evicted (%d layers, floor=%d scratch=%d)",
            old_cutoff, new_cutoff, freed_pages, groups_evicted,
            n_layers, floor, self._scratch_capacity)
        return freed_pages, groups_evicted

    def _cutoff_count_evictable_groups(self) -> int:
        """Count evictable groups under cutoff policy (prepare-safe).

        Must match _cutoff_shrink logic so prepare never over-promises.
        Returns total per-layer group evictions (not cutoff steps), so
        worker's `groups * group_pages` formula gives correct page count.

        One cutoff step = (num_layers-1) per-layer group evictions.
        """
        pool = self._vmm_pool
        if not pool:
            return 0
        gs = pool.group_size
        n_layers = self.num_layers - 1  # L0 excluded
        if n_layers <= 0:
            return 0
        cutoff = self._resident_cutoff
        floor = max(0, self.local_num_experts - self._scratch_capacity)
        evictable_lids = max(0, cutoff - floor)
        cutoff_steps = evictable_lids // gs
        # Each cutoff step evicts from (num_layers-1) layers
        return cutoff_steps * n_layers

    def _cutoff_recompute_topology(self):
        """Deterministic tail from cutoff. Same for L1..L47."""
        cutoff = self._resident_cutoff
        localE = self.local_num_experts

        self._cutoff_tail_lids = list(range(cutoff, localE))
        n_tail = len(self._cutoff_tail_lids)

        assert n_tail <= self._scratch_capacity, (
            f"[CutoffBoundary] tail={n_tail} > scratch="
            f"{self._scratch_capacity}")
        # Pinned/cpu_pool already verified at latch time (pre_step)

        logger.info("[CutoffBoundary] topology: cutoff=%d tail=%d scratch=%d",
                    cutoff, n_tail, self._scratch_capacity)

    def _cutoff_rebuild_cache_maps(self):
        """One-time full cache_map rebuild from cutoff.

        L0: always resident-only (all experts → slot = lid).
        L1..L47: lids [0, cutoff) → resident slot,
                 lids [cutoff, localE) → scratch slot.
        """
        cutoff = self._resident_cutoff
        threshold = self._scratch_threshold
        emap_np_list = getattr(self, '_emap_cpu_np', None)

        for li in range(self.num_layers):
            cache_map = torch.full(
                (self.global_num_experts,), -1, dtype=torch.int32)
            emap_np = emap_np_list[li] if emap_np_list else None
            is_l0 = (li == 0)

            if emap_np is not None:
                for gid in range(len(emap_np)):
                    lid = int(emap_np[gid])
                    if lid < 0:
                        continue
                    if is_l0 or lid < cutoff:
                        slot = self._expert_to_slot[li][lid]
                        if slot >= 0:
                            cache_map[gid] = slot
                    else:
                        cache_map[gid] = threshold + (lid - cutoff)
            else:
                if is_l0:
                    for lid in range(self.local_num_experts):
                        slot = self._expert_to_slot[li][lid]
                        if slot >= 0:
                            cache_map[lid] = slot
                else:
                    for lid in range(cutoff):
                        slot = self._expert_to_slot[li][lid]
                        if slot >= 0:
                            cache_map[lid] = slot
                    for i, lid in enumerate(
                            range(cutoff, self.local_num_experts)):
                        cache_map[lid] = threshold + i

            self._moe_layers[li]._cache_map.copy_(
                cache_map.to(self._moe_layers[li]._cache_map.device))

        self._cache_map_needs_rebuild = False

    def cutoff_prefetch_next(self, layer_idx: int):
        """Double-buffer: load next layer's tail into alt bank.

        Called from every layer's forward (including L0 which has no scratch).
        L0→L1: first bank fill (bank[0]).
        L1→L2, ...: alternating banks.
        """
        next_idx = layer_idx + 1
        if next_idx >= self.num_layers:
            return
        tail = self._cutoff_tail_lids
        if not tail:
            return

        # First call (from L0): use bank 0. Otherwise alternate.
        if layer_idx == 0:
            target_bank = 0
        else:
            target_bank = 1 - self._cutoff_current_bank

        bank = self._scratch_banks[target_bank]
        assert bank.state == _BankState.IDLE, (
            f"[CutoffBoundary] bank {target_bank} not IDLE: "
            f"{bank.state.name}")
        bank.state = _BankState.FILLING
        self._copy_stream.wait_event(bank.done_event)
        with torch.cuda.stream(self._copy_stream):
            # w13 first → record w13_ready_event → w2 → record ready_event
            # Allows w1 kernel to start while w2 is still copying.
            for i, lid in enumerate(tail):
                w13_cpu, _ = self._cpu_pool[next_idx][lid]
                bank.w13[i].copy_(w13_cpu, non_blocking=True)
            if bank.w13_ready_event is not None:
                bank.w13_ready_event.record(self._copy_stream)
            for i, lid in enumerate(tail):
                _, w2_cpu = self._cpu_pool[next_idx][lid]
                bank.w2[i].copy_(w2_cpu, non_blocking=True)
            bank.ready_event.record(self._copy_stream)

        bank.state = _BankState.READY
        bank.owner_layer_idx = next_idx
        bank.owner_layer = self._moe_layers[next_idx]
        bank.n_active = len(tail)
        self._cutoff_current_bank = target_bank

    def reserve_scratch(
        self,
        layer_idx: int,
        miss_gids: List[int],
        miss_lids: List[int],
        layer,
    ) -> Tuple[bool, int]:
        """Load overflow experts into scratch and in-place patch _cache_map.

        Returns:
            (success, bank_idx). If success=True, forward_cuda must
            wait_event(bank.ready_event) then release_scratch(layer, bank_idx).
        """
        if not miss_gids or not self._scratch_banks:
            return (False, -1)

        n = min(len(miss_gids), self._scratch_capacity)
        if n == 0:
            return (False, -1)

        # ── Bank selection ──
        bidx = self._scratch_current_bank
        bank = self._scratch_banks[bidx]
        if bank.state != _BankState.IDLE:
            if len(self._scratch_banks) == 1:
                # 1-bank: non-IDLE = sequential invariant violation
                raise AssertionError(
                    f"Single scratch bank not IDLE: "
                    f"state={bank.state.name} owner={bank.owner_layer_idx}. "
                    "Sequential forward invariant violated.")
            # Try alternate bank (2-bank mode)
            bidx = 1 - bidx
            bank = self._scratch_banks[bidx]
            if bank.state != _BankState.IDLE:
                # Both busy → sacrifice prefetch bank (READY/FILLING)
                victim = None
                for vi, vb in enumerate(self._scratch_banks):
                    if vb.state in (_BankState.READY, _BankState.FILLING):
                        victim = (vi, vb)
                        break
                if victim is not None:
                    self._discard_prewarm(victim[1])
                    bidx, bank = victim
                else:
                    _states = [(i, b.state.name, b.owner_layer_idx)
                               for i, b in enumerate(self._scratch_banks)]
                    raise AssertionError(
                        f"All scratch banks busy: {_states}. "
                        "Sequential forward invariant violated.")

        # ── Wait for previous compute on this bank ──
        bank.state = _BankState.FILLING
        self._copy_stream.wait_event(bank.done_event)

        # ── H2D: load overflow experts into scratch on copy_stream ──
        if _KERNEL_PROFILE:
            _ev_cp_s = torch.cuda.Event(enable_timing=True)
            _ev_cp_e = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self._copy_stream):
            if _KERNEL_PROFILE:
                _ev_cp_s.record(self._copy_stream)
            for i in range(n):
                lid = miss_lids[i]
                w13_cpu, w2_cpu = self._cpu_pool[layer_idx][lid]
                bank.w13[i].copy_(w13_cpu, non_blocking=True)
                bank.w2[i].copy_(w2_cpu, non_blocking=True)
            if _KERNEL_PROFILE:
                _ev_cp_e.record(self._copy_stream)
            bank.ready_event.record(self._copy_stream)

        bank.state = _BankState.READY

        if _KERNEL_PROFILE:
            global _cp_calls, _cp_total_us, _cp_count, _cp_experts_total
            _cp_calls += 1
            _cp_experts_total += n
            if _cp_calls % _cp_interval == 0:
                self._copy_stream.synchronize()
                _c_us = _ev_cp_s.elapsed_time(_ev_cp_e) * 1000
                _cp_total_us += _c_us
                _cp_count += 1
                if _cp_calls % (_cp_interval * 10) == 0:
                    _avg_c = _cp_total_us / _cp_count if _cp_count else 0
                    _avg_e = _cp_experts_total / _cp_calls if _cp_calls else 0
                    logger.info(
                        "[CopyProfile] calls=%d "
                        "copy_avg=%.0fμs (n=%d) "
                        "avg_experts=%.1f/call",
                        _cp_calls, _avg_c, _cp_count, _avg_e)
                    _cp_total_us = 0.0
                    _cp_count = 0

        # ── In-place patch _cache_map ──
        gid_cpu = torch.tensor(miss_gids[:n], dtype=torch.int64)
        bank.gid_buffer[:n].copy_(gid_cpu)
        gid_view = bank.gid_buffer[:n]
        slot_view = self._scratch_slot_values[:n]
        layer._cache_map.index_copy_(0, gid_view, slot_view)

        # ── Save owner metadata ──
        bank.owner_layer_idx = layer_idx
        bank.owner_layer = layer
        bank.current_gids = gid_view
        bank.consumed_gids = list(miss_gids[:n])
        bank.n_active = n

        # ── Bank alternation (Phase 2A, 2-bank only) ──
        if len(self._scratch_banks) > 1:
            self._scratch_current_bank = 1 - bidx

        # Instrumentation
        self._scratch_reserve_calls += 1
        self._scratch_experts_loaded += n
        if _EXPERT_DEBUG:
            if (self._scratch_reserve_calls <= 1
                    or self._scratch_reserve_calls
                    % self._scratch_log_interval == 0):
                logger.info(
                    "[Scratch] reserve[%d] bank=%d: layer=%d, gids=%s, "
                    "slot_range=[%d,%d)",
                    self._scratch_reserve_calls, bidx, layer_idx,
                    list(miss_gids[:min(n, 8)]),
                    self._scratch_threshold, self._scratch_threshold + n)

        return (True, bidx)

    def release_scratch(self, layer, bank_idx: int = -1) -> None:
        """Restore _cache_map and record done event on current compute stream.

        Must be called immediately after kernel returns in forward_cuda,
        while still on the compute stream.
        """
        # Find the bank
        if bank_idx >= 0:
            bank = self._scratch_banks[bank_idx]
        else:
            # Legacy fallback: find by owner match
            bank = None
            for bi, b in enumerate(self._scratch_banks):
                if b.state == _BankState.IN_USE:
                    bank = b
                    bank_idx = bi
                    break
            if bank is None:
                return

        assert bank.state == _BankState.IN_USE, (
            f"release_scratch: bank {bank_idx} state={bank.state}, "
            f"expected IN_USE")

        # Safety: verify caller's layer matches
        assert layer is bank.owner_layer, (
            f"release_scratch called with wrong layer. "
            f"bank {bank_idx} owner_layer_idx={bank.owner_layer_idx}")

        # ── Fixed-tail / Step-boundary / Cutoff: static map → no restore ──
        if (self._fixed_tail_active or self._static_topo_active
                or self._cutoff_active):
            bank.done_event.record()
            bank.owner_layer_idx = -1
            bank.owner_layer = None
            bank.current_gids = None
            bank.consumed_gids = None
            bank.n_active = 0
            bank.state = _BankState.IDLE
            return

        # ── Restore _cache_map ──
        restore_gids = bank.current_gids
        if bank.consumed_gids is not None:
            if isinstance(bank.consumed_gids, list):
                gid_cpu = torch.tensor(
                    bank.consumed_gids, dtype=torch.int64)
                bank.gid_buffer[:len(bank.consumed_gids)].copy_(gid_cpu)
                restore_gids = bank.gid_buffer[:len(bank.consumed_gids)]
        if restore_gids is not None:
            bank.owner_layer._cache_map.index_fill_(
                0, restore_gids, -1)

        # ── Record done event on current compute stream ──
        bank.done_event.record()

        if _EXPERT_DEBUG:
            logger.info(
                "[Scratch] release: bank=%d layer_idx=%d n_active=%d",
                bank_idx, bank.owner_layer_idx, bank.n_active)

        # ── Clear bank state ──
        bank.owner_layer_idx = -1
        bank.owner_layer = None
        bank.current_gids = None
        bank.consumed_gids = None
        bank.n_active = 0
        bank.prewarmed_gids = None
        bank.prewarmed_lids = None
        bank.gid_to_slot = None
        bank.n_prewarmed = 0
        bank.prewarm_target_layer = -1
        bank.state = _BankState.IDLE

    def _prewarm_next_layer(self, current_layer_idx: int) -> None:
        """Prewarm known non-resident experts for next layer into scratch.

        Unlike lookahead prefetch (prediction-based), prewarm uses KNOWN
        non-resident state from emap + slot tracking. No prediction needed.

        H2D only — does NOT patch _cache_map. Patch deferred to consume.
        Prefill-only: caller must gate on _prefillish_step.
        """
        if not self._scratch_banks or len(self._scratch_banks) < 2:
            return
        next_idx = current_layer_idx + 1
        if next_idx >= self.num_layers:
            return

        # ── Non-resident collection (emap full scan, O(global_E) ≈ 256) ──
        emap_np = (self._emap_cpu_np[next_idx]
                   if hasattr(self, '_emap_cpu_np') else None)
        slots_np = self._expert_to_slot_np[next_idx]
        nr_gids: List[int] = []
        nr_lids: List[int] = []
        if emap_np is not None:
            for gid in range(len(emap_np)):
                lid = int(emap_np[gid])
                if (lid >= 0
                        and slots_np[lid] == -1
                        and lid in self._cpu_pool[next_idx]):
                    nr_gids.append(gid)
                    nr_lids.append(lid)
        else:
            miss_mask = np.where(slots_np == -1)[0]
            for lid in miss_mask:
                lid = int(lid)
                if lid in self._cpu_pool[next_idx]:
                    nr_gids.append(lid)
                    nr_lids.append(lid)

        if not nr_gids:
            return

        # ── Hottest-first truncation (last_access descending) ──
        cap = self._prewarm_cap
        if len(nr_gids) > cap:
            la = self._last_access[next_idx]
            paired = sorted(
                zip(nr_gids, nr_lids),
                key=lambda gl: la[gl[1]],
                reverse=True)
            nr_gids = [g for g, _ in paired[:cap]]
            nr_lids = [l for _, l in paired[:cap]]

        # ── Bank selection (try current, then alternate) ──
        bidx = self._scratch_current_bank
        bank = self._scratch_banks[bidx]
        if bank.state != _BankState.IDLE:
            bidx = 1 - bidx
            bank = self._scratch_banks[bidx]
            if bank.state != _BankState.IDLE:
                return  # both busy, no-op

        n = min(len(nr_gids), self._scratch_capacity)
        if n == 0:
            return

        # ── Build gid→slot mapping ──
        gid_to_slot: Dict[int, int] = {}
        for i in range(n):
            gid_to_slot[nr_gids[i]] = self._scratch_threshold + i

        # ── H2D on copy_stream ──
        bank.state = _BankState.FILLING
        self._copy_stream.wait_event(bank.done_event)
        with torch.cuda.stream(self._copy_stream):
            for i in range(n):
                lid = nr_lids[i]
                w13_cpu, w2_cpu = self._cpu_pool[next_idx][lid]
                bank.w13[i].copy_(w13_cpu, non_blocking=True)
                bank.w2[i].copy_(w2_cpu, non_blocking=True)
            bank.ready_event.record(self._copy_stream)

        bank.state = _BankState.READY

        # ── Store prewarm metadata (NO _cache_map patch) ──
        bank.prewarm_target_layer = next_idx
        bank.prewarmed_gids = list(nr_gids[:n])
        bank.prewarmed_lids = list(nr_lids[:n])
        bank.gid_to_slot = gid_to_slot
        bank.n_prewarmed = n
        bank.consumed_gids = None
        bank.owner_layer_idx = -1
        bank.owner_layer = None

        if _EXPERT_DEBUG:
            logger.info(
                "[Prewarm] bank=%d next_layer=%d n_prewarmed=%d",
                bidx, next_idx, n)

    def _consume_prewarm(
        self,
        bank: _ScratchBank,
        bank_idx: int,
        layer_idx: int,
        H_gids: List[int],
        R_overflow_gids: List[int],
        R_overflow_lids: List[int],
        layer,
    ) -> None:
        """Consume prewarmed bank: H from prewarm slots, R_overflow async-copied.

        1. Enqueue R_overflow copies on copy_stream (after prewarm H2D)
        2. Re-record ready_event (now covers prewarm + residual)
        3. Compute stream wait_event → patch cache_map
        4. Transfer bank ownership

        No device-wide sync: compute stream only waits on bank.ready_event.
        forward_cuda's existing wait_event(ready_event) covers everything.
        """
        n_rfill = len(R_overflow_gids)

        # ── R_overflow → bank residual slots on copy_stream ──
        # Enqueued AFTER prewarm H2D (same stream → serialized)
        if n_rfill > 0:
            with torch.cuda.stream(self._copy_stream):
                for i in range(n_rfill):
                    off = bank.n_prewarmed + i
                    lid = R_overflow_lids[i]
                    w13_cpu, w2_cpu = self._cpu_pool[layer_idx][lid]
                    bank.w13[off].copy_(w13_cpu, non_blocking=True)
                    bank.w2[off].copy_(w2_cpu, non_blocking=True)
                # Re-record: now covers prewarm + residual fill
                bank.ready_event.record(self._copy_stream)

        # ── Compute stream waits for all bank writes ──
        torch.cuda.current_stream().wait_event(bank.ready_event)

        # ── cache_map patch via index_copy_ ──
        consumed_gids = H_gids + R_overflow_gids
        n_consumed = len(consumed_gids)
        # H slots from gid_to_slot, R_overflow from threshold + n_prewarmed + i
        slot_list = [bank.gid_to_slot[g] for g in H_gids]
        slot_list += [self._scratch_threshold + bank.n_prewarmed + i
                      for i in range(n_rfill)]
        gid_t = torch.tensor(consumed_gids, dtype=torch.int64,
                             device=self.device)
        slot_t = torch.tensor(slot_list, dtype=torch.int32,
                              device=self.device)
        bank.gid_buffer[:n_consumed].copy_(gid_t)
        gid_view = bank.gid_buffer[:n_consumed]
        layer._cache_map.index_copy_(0, gid_view, slot_t)

        # ── Bank ownership (stay READY — forward_cuda transitions to IN_USE) ──
        bank.owner_layer_idx = layer_idx
        bank.owner_layer = layer
        bank.current_gids = gid_view
        bank.consumed_gids = consumed_gids
        bank.n_active = n_consumed
        bank.prewarm_target_layer = -1
        self._scratch_current_bank = 1 - bank_idx

        # ── Instrumentation ──
        self._pw_consumes += 1
        self._pw_H_total += len(H_gids)
        self._pw_R_total += len(R_overflow_gids)
        self._pw_Rfill_total += n_rfill

        # First consume: always log (confirms prewarm is working)
        if self._pw_consumes == 1:
            logger.info(
                "[Prewarm] ACTIVE: first consume bank=%d layer=%d "
                "H=%d R_fill=%d cap=%d",
                bank_idx, layer_idx, len(H_gids), n_rfill,
                self._prewarm_cap)
        elif _EXPERT_DEBUG:
            logger.info(
                "[Prewarm] consume: bank=%d layer=%d H=%d R_fill=%d total=%d",
                bank_idx, layer_idx, len(H_gids), n_rfill, n_consumed)

