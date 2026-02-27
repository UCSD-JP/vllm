# SPDX-License-Identifier: Apache-2.0
"""Expert Cache Manager for MoE weight offloading.

Single implementation path:
- w13_weight resized to (max_resident, ...)
- cache_map replaces expert_map temporarily in forward_cuda()
- CPU pinned backing store (direct async DMA, no staging)
"""

import torch
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExpertOffloadConfig:
    """Expert offloading configuration."""
    enable: bool = False
    max_resident_per_layer: int = 50
    prefetch_lookahead: int = 1
    eviction_policy: str = "lfu_lru"
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
    # Token-weighted counters (each routing entry, not unique IDs)
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


class ExpertCacheManager:
    """Manages expert weight data in resized w13/w2 tensors.

    After initialization:
    - layer.w13_weight has shape (max_resident, 2*N, K)
    - layer.w2_weight has shape (max_resident, K, N)
    - Only max_resident experts fit on GPU at a time
    - The rest live in CPU pageable memory

    Per forward call:
    1. prepare_and_get_expert_map() ensures routed experts are in GPU slots
    2. Returns cache_map: expert_map remapped to slot indices
    3. Kernel uses cache_map instead of original expert_map

    expert_map=None handling:
    - TP4(ep_size=1) sets _expert_map=None
    - Cache creates identity map [0,1,...,E-1] as replacement
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

        # Per-expert size calculation
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

        # Identity map for ep_size=1 (expert_map=None replacement)
        self._identity_map = torch.arange(
            global_num_experts, dtype=torch.int32, device=device
        )

        # Per-layer slot tracking
        self._slot_to_expert: List[List[int]] = []
        self._expert_to_slot: List[List[int]] = []
        self._free_slots: List[List[int]] = []
        for _ in range(num_layers):
            self._slot_to_expert.append([-1] * self.max_resident)
            self._expert_to_slot.append([-1] * local_num_experts)
            self._free_slots.append(list(range(self.max_resident)))

        # Eviction metadata
        self._access_count: Dict[Tuple[int, int], int] = defaultdict(int)
        self._last_access: Dict[Tuple[int, int], int] = defaultdict(int)
        self._pinned: Set[Tuple[int, int]] = set()

        # CPU pinned backing store — enables direct async DMA to GPU
        # without intermediate staging copies (pageable→pinned eliminated)
        self._cpu_pool: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = [
            {} for _ in range(num_layers)
        ]

        # GPU weight tensor references (set by register_layer)
        self._layer_w13: Dict[int, torch.Tensor] = {}
        self._layer_w2: Dict[int, torch.Tensor] = {}

        # CUDA stream + prefetch events
        self._copy_stream = torch.cuda.Stream(device=device)
        self._prefetch_events: Dict[int, torch.cuda.Event] = {}
        self._prefetch_pending: Dict[int, Set[int]] = defaultdict(set)
        self._prefetch_bytes_this_step: int = 0

        # Stats
        self.stats = CacheStats()
        self._lock = threading.Lock()

    def _set_expert_slot(self, layer_idx: int, lid: int, slot: int):
        """Write to _expert_to_slot + tensor mirror."""
        self._expert_to_slot[layer_idx][lid] = slot
        if hasattr(self, '_expert_to_slot_t'):
            self._expert_to_slot_t[layer_idx, lid] = slot

    def _set_slot_expert(self, layer_idx: int, slot: int, lid: int):
        """Write to _slot_to_expert."""
        self._slot_to_expert[layer_idx][slot] = lid

    def _resolve_expert_map(
        self, original_expert_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return identity map if expert_map is None (ep_size=1)."""
        if original_expert_map is not None:
            return original_expert_map
        return self._identity_map

    # --- Registration (Worker init) ---

    def register_layer(
        self,
        layer_idx: int,
        w13_data: torch.Tensor,
        w2_data: torch.Tensor,
    ):
        """Store reference to resized weight tensor data."""
        self._layer_w13[layer_idx] = w13_data
        self._layer_w2[layer_idx] = w2_data

    def register_expert_cpu(
        self,
        layer_idx: int,
        local_expert_id: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        is_shared: bool = False,
    ):
        """Copy expert weight to CPU pinned pool for direct async DMA.

        Pinned memory enables non_blocking GPU copies without staging.
        """
        if w13_weight.device.type == 'cpu':
            w13_cpu = w13_weight.detach().clone().pin_memory()
            w2_cpu = w2_weight.detach().clone().pin_memory()
        else:
            w13_cpu = w13_weight.detach().cpu().pin_memory()
            w2_cpu = w2_weight.detach().cpu().pin_memory()
        self._cpu_pool[layer_idx][local_expert_id] = (w13_cpu, w2_cpu)
        if is_shared and self.config.pin_shared_experts:
            self._pinned.add((layer_idx, local_expert_id))

    # --- Initial Population ---

    def populate_initial_cache(self):
        """Cold start: fill max_resident slots per layer."""
        for layer_idx in range(self.num_layers):
            # Pinned (shared) experts first
            for lid in range(self.local_num_experts):
                if (layer_idx, lid) in self._pinned:
                    self._load_to_slot(layer_idx, lid)
            # Fill remaining
            for lid in range(self.local_num_experts):
                if not self._free_slots[layer_idx]:
                    break
                if self._expert_to_slot[layer_idx][lid] != -1:
                    continue
                self._load_to_slot(layer_idx, lid)
        torch.cuda.synchronize(self.device)

    def _load_to_slot(self, layer_idx: int, local_id: int) -> bool:
        """CPU pinned -> GPU slot (synchronous, no staging)."""
        if not self._free_slots[layer_idx]:
            return False
        slot = self._free_slots[layer_idx].pop()
        self._set_slot_expert(layer_idx, slot, local_id)
        self._set_expert_slot(layer_idx, local_id, slot)

        w13_cpu, w2_cpu = self._cpu_pool[layer_idx][local_id]
        self._layer_w13[layer_idx][slot].copy_(w13_cpu)
        self._layer_w2[layer_idx][slot].copy_(w2_cpu)
        return True

    # --- Core: Called from forward_cuda() ---

    def prepare_and_get_expert_map(
        self,
        layer_idx: int,
        topk_ids: torch.Tensor,
        original_expert_map: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Ensure routed experts are cached and return cache_map.

        Called from forward_cuda() after select_experts(), before kernel().

        Returns:
            cache_map: Tensor[global_num_experts] int32
            cache_map[gid] = slot_idx or -1
        """
        # Step 0: Resolve expert_map (None -> identity)
        emap = self._resolve_expert_map(original_expert_map)

        # Step 1: Wait prefetch from previous layer
        if layer_idx in self._prefetch_events:
            event = self._prefetch_events.pop(layer_idx)
            torch.cuda.current_stream(self.device).wait_event(event)
            self._prefetch_pending.pop(layer_idx, None)

        # Step 2: Identify needed local experts
        needed_local: Set[int] = set()
        emap_cpu = emap.cpu()
        for gid in topk_ids.flatten().unique().tolist():
            if 0 <= gid < emap_cpu.shape[0]:
                lid = emap_cpu[gid].item()
                if lid != -1:
                    needed_local.add(lid)

        # Step 3: Hit/miss classification
        miss_ids: List[int] = []
        with self._lock:
            for lid in needed_local:
                key = (layer_idx, lid)
                if self._expert_to_slot[layer_idx][lid] != -1:
                    self._access_count[key] += 1
                    self._last_access[key] = self.current_step
                    self.stats.hits += 1
                else:
                    miss_ids.append(lid)
                    self.stats.misses += 1

        # Step 4: Sync fetch misses (protect already-cached needed experts)
        if miss_ids:
            cached_needed = needed_local - set(miss_ids)
            self._sync_fetch(layer_idx, miss_ids, protected_local_ids=cached_needed)

        # Step 5: Build cache_map
        cache_map = torch.full(
            (self.global_num_experts,), -1, dtype=torch.int32, device=self.device
        )
        for gid in range(emap_cpu.shape[0]):
            lid = emap_cpu[gid].item()
            if lid == -1:
                continue
            slot = self._expert_to_slot[layer_idx][lid]
            if slot != -1:
                cache_map[gid] = slot
        return cache_map

    # --- Prefetch ---

    def enqueue_prefetch(
        self,
        target_layer_idx: int,
        predicted_local_ids: List[int],
    ):
        """Async prefetch for next layer. Overlaps with current kernel."""
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

        # BW throttle
        max_n = self.config.prefetch_max_bytes_per_step // max(
            self.expert_size_bytes, 1
        )
        remaining = max_n - (
            self._prefetch_bytes_this_step // max(self.expert_size_bytes, 1)
        )
        if remaining <= 0:
            return
        to_fetch = to_fetch[:min(len(to_fetch), remaining)]

        # Make room
        for _ in range(len(to_fetch)):
            if self._free_slots[target_layer_idx]:
                break
            self._evict_one(target_layer_idx)

        # Async copy
        event = torch.cuda.Event()
        loaded: Set[int] = set()

        with torch.cuda.stream(self._copy_stream):
            for i, lid in enumerate(to_fetch):
                if not self._free_slots[target_layer_idx]:
                    break

                slot = self._free_slots[target_layer_idx].pop()
                self._set_slot_expert(target_layer_idx, slot, lid)
                self._set_expert_slot(target_layer_idx, lid, slot)
                self._access_count[(target_layer_idx, lid)] = 0
                self._last_access[(target_layer_idx, lid)] = self.current_step

                w13_cpu, w2_cpu = self._cpu_pool[target_layer_idx][lid]
                self._layer_w13[target_layer_idx][slot].copy_(
                    w13_cpu, non_blocking=True
                )
                self._layer_w2[target_layer_idx][slot].copy_(
                    w2_cpu, non_blocking=True
                )
                loaded.add(lid)

            event.record(self._copy_stream)

        self._prefetch_events[target_layer_idx] = event
        self._prefetch_pending[target_layer_idx] = loaded
        self._prefetch_bytes_this_step += len(loaded) * self.expert_size_bytes

    # --- Sync Fetch ---

    def _sync_fetch(
        self,
        layer_idx: int,
        local_ids: List[int],
        protected_local_ids: Optional[Set[int]] = None,
    ):
        """Blocking CPU->GPU fetch. Last resort for cache misses.

        If needed_local > max_resident, some experts cannot be loaded.
        Instead of crashing, we skip them — they stay as -1 in cache_map
        and the kernel treats them like remote (EP) experts.
        """
        # Protect both already-cached needed experts AND experts being
        # loaded in this batch (they occupy slots as we iterate).
        protected = set(protected_local_ids) if protected_local_ids else set()
        skipped = 0
        for i, lid in enumerate(local_ids):
            if not self._free_slots[layer_idx]:
                if not self._evict_one(layer_idx, protected):
                    # Capacity exceeded: can't fit all needed experts.
                    # Skip remaining — they'll be -1 in cache_map.
                    skipped += len(local_ids) - i
                    break

            slot = self._free_slots[layer_idx].pop()
            self._set_slot_expert(layer_idx, slot, lid)
            self._set_expert_slot(layer_idx, lid, slot)
            self._access_count[(layer_idx, lid)] = 1
            self._last_access[(layer_idx, lid)] = self.current_step
            protected.add(lid)  # protect newly loaded expert from eviction

            w13_cpu, w2_cpu = self._cpu_pool[layer_idx][lid]
            self._layer_w13[layer_idx][slot].copy_(w13_cpu)
            self._layer_w2[layer_idx][slot].copy_(w2_cpu)

            self.stats.sync_fetches += 1

        if skipped > 0:
            logger.warning(
                "Layer %d: needed_local (%d) > max_resident (%d), "
                "skipped %d experts (will be -1 in cache_map)",
                layer_idx,
                len(protected),  # total protected at this point
                self.max_resident,
                skipped,
            )
        torch.cuda.synchronize(self.device)

    # --- Eviction ---

    def _evict_one(
        self,
        layer_idx: int,
        protected_local_ids: Optional[Set[int]] = None,
    ) -> bool:
        best_slot = -1
        best_priority = float('inf')

        for slot in range(self.max_resident):
            lid = self._slot_to_expert[layer_idx][slot]
            if lid == -1:
                continue
            key = (layer_idx, lid)
            if key in self._pinned:
                continue
            if protected_local_ids and lid in protected_local_ids:
                continue
            if lid in self._prefetch_pending.get(layer_idx, set()):
                continue

            if self.config.eviction_policy == "lru":
                priority = self._last_access.get(key, 0)
            elif self.config.eviction_policy == "lfu":
                priority = self._access_count.get(key, 0)
            else:  # lfu_lru
                age = max(self.current_step - self._last_access.get(key, 0) + 1, 1)
                priority = self._access_count.get(key, 0) / age

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

    # --- Step (v1, kept for backward compat) ---

    def step(self):
        """Called once per forward step. Resets BW throttle."""
        self.current_step += 1
        self._prefetch_bytes_this_step = 0

    # ================================================================
    # v2: CUDA-graph-compatible predict-and-preload interface
    # ================================================================

    def __init_v2_scratch(self):
        """Lazy-init per-layer scratch maps to avoid repeated allocation."""
        if hasattr(self, '_scratch_maps'):
            return
        # Stacked CPU scratch tensor — views become per-layer scratch maps.
        # Contiguous layout enables single bulk CPU→GPU copy.
        # Pin memory for faster CPU→GPU DMA (avoids driver staging).
        _scratch = torch.full(
            (self.num_layers, self.global_num_experts), -1,
            dtype=torch.int32)
        self._stacked_scratch_cpu = (
            _scratch.pin_memory() if torch.cuda.is_available() else _scratch
        )
        self._scratch_maps: List[torch.Tensor] = [
            self._stacked_scratch_cpu[i]
            for i in range(self.num_layers)
        ]
        # Timing instrumentation
        self._timing = {
            't_pre_step_total_us': 0.0,
            't_gpu_gather_us': 0.0,
            't_d2h_us': 0.0,
            't_classify_us': 0.0,
            't_cache_map_us': 0.0,
            't_cache_map_build_us': 0.0,
            't_cache_map_upload_us': 0.0,
            't_fetch_us': 0.0,
            't_sync_us': 0.0,
            't_deferred_sync_us': 0.0,
            'pre_step_calls': 0,
            'bytes_fetched_total': 0,
            'experts_fetched_total': 0,
        }
        # Miss-rate diagnostic counters (per-layer detail, periodic dump)
        self._diag = {
            'per_layer_unique_needed': np.zeros(self.num_layers, dtype=np.int64),
            'per_layer_rlen': np.zeros(self.num_layers, dtype=np.int64),
            'per_layer_valid_len': np.zeros(self.num_layers, dtype=np.int64),
            'per_layer_hits': np.zeros(self.num_layers, dtype=np.int64),
            'per_layer_misses': np.zeros(self.num_layers, dtype=np.int64),
            'per_layer_token_hits': np.zeros(self.num_layers, dtype=np.int64),
            'per_layer_token_misses': np.zeros(self.num_layers, dtype=np.int64),
            'diag_steps': 0,
            'diag_dump_interval': 100,  # log every N steps
        }
        # Expert gating distribution: per-expert access counts
        self._gating_histogram = np.zeros(
            self.global_num_experts, dtype=np.int64)
        self._gating_steps = 0
        # Async DMA hiding: deferred sync state
        self._prev_needs_sync = False
        # Per-layer set of slot indices with in-flight DMA.
        # These slots are allocated but weight data not yet written.
        # Must be excluded from cache_map until sync completes.
        self._pending_dma_slots: Dict[int, Set[int]] = {}

    def _init_batched_d2h(self, layers):
        """Lazy-init batched D2H infrastructure on first pre_step call.

        Creates:
        - Stacked GPU buffers for GPU→GPU gather
        - Pinned CPU landing pads for bulk D2H
        - Expert map CPU cache (avoid repeated .cpu())
        - _expert_to_slot tensor mirror
        - Dedicated D2H stream + event
        """
        if hasattr(self, '_batched_d2h_ready'):
            return
        self._batched_d2h_ready = True

        num_layers = len(layers)
        # Determine max routing snapshot length across layers
        max_snap_len = 0
        for layer in layers:
            if hasattr(layer, '_routing_snapshot') \
                    and layer._routing_snapshot is not None:
                try:
                    slen = int(layer._routing_snapshot.shape[0])
                    max_snap_len = max(max_snap_len, slen)
                except (TypeError, AttributeError):
                    pass
        if max_snap_len == 0:
            max_snap_len = 4096  # fallback

        # Detect device from first layer's buffer
        dev = self.device
        for layer in layers:
            if hasattr(layer, '_routing_len') \
                    and layer._routing_len is not None:
                try:
                    d = layer._routing_len.device
                    if isinstance(d, torch.device):
                        dev = d
                        break
                except (AttributeError, TypeError):
                    pass
        # Normalize: ensure dev is a torch.device
        if not isinstance(dev, torch.device):
            dev = torch.device('cpu')
        self._is_cuda = dev.type == 'cuda'

        # 2a. Stacked GPU buffers (for GPU→GPU gather)
        self._routing_len_gpu = torch.zeros(
            num_layers, dtype=torch.int32, device=dev)
        self._routing_snap_gpu = torch.zeros(
            num_layers, max_snap_len, dtype=torch.int32, device=dev)
        self._max_snap_len = max_snap_len

        # 2b. Pinned CPU landing pads (for bulk D2H)
        self._routing_len_cpu = torch.zeros(
            num_layers, dtype=torch.int32)
        self._routing_snap_cpu = torch.zeros(
            num_layers, max_snap_len, dtype=torch.int32)
        if self._is_cuda:
            try:
                self._routing_len_cpu = self._routing_len_cpu.pin_memory()
                self._routing_snap_cpu = \
                    self._routing_snap_cpu.pin_memory()
            except RuntimeError:
                pass

        # 2c. Expert map CPU cache
        self._emap_cpu_cache: List[Optional[torch.Tensor]] = []
        for layer in layers:
            if hasattr(layer, '_expert_map') \
                    and layer._expert_map is not None:
                emap = layer._expert_map
                self._emap_cpu_cache.append(
                    emap.cpu() if emap.device.type != 'cpu' else emap.clone())
            else:
                self._emap_cpu_cache.append(None)

        # 2d. _expert_to_slot tensor mirror (CPU, int32)
        self._expert_to_slot_t = torch.full(
            (num_layers, self.local_num_experts), -1, dtype=torch.int32)
        # Copy from Python lists
        for li in range(min(num_layers, self.num_layers)):
            for eid in range(self.local_num_experts):
                self._expert_to_slot_t[li, eid] = \
                    self._expert_to_slot[li][eid]

        # 2e. Dedicated D2H stream + event
        if self._is_cuda:
            try:
                self._d2h_stream = torch.cuda.Stream(device=dev)
                self._d2h_event = torch.cuda.Event()
            except RuntimeError:
                self._d2h_stream = None
                self._d2h_event = None
        else:
            self._d2h_stream = None
            self._d2h_event = None

        # Track layer indices for active layers (have _cache_map)
        self._active_layer_indices: List[int] = []
        for i, layer in enumerate(layers):
            if hasattr(layer, '_cache_map') and layer._cache_map is not None:
                self._active_layer_indices.append(i)

        # 2f. Stacked GPU buffer for batched cache_map upload
        # Instead of 48× CPU→GPU per layer, do one bulk copy then
        # 48× fast GPU→GPU scatter.
        self.__init_v2_scratch()  # ensure _stacked_scratch_cpu exists
        if self._is_cuda:
            self._stacked_scratch_gpu = torch.full(
                (num_layers, self.global_num_experts), -1,
                dtype=torch.int32, device=dev)
        else:
            self._stacked_scratch_gpu = None

    def invalidate_emap_cache(self, layer_idx: int,
                              new_expert_map: Optional[torch.Tensor] = None):
        """Re-cache expert_map for a layer after EPLB update.

        Args:
            layer_idx: Layer index to invalidate.
            new_expert_map: Updated expert_map tensor. If provided, immediately
                re-cached (avoids identity fallback which is wrong for EP).
                If None, the next pre_step will use identity fallback
                (only safe for ep_size=1).
        """
        if not hasattr(self, '_emap_cpu_cache') \
                or layer_idx >= len(self._emap_cpu_cache):
            return
        if new_expert_map is not None:
            self._emap_cpu_cache[layer_idx] = (
                new_expert_map.cpu()
                if new_expert_map.device.type != 'cpu'
                else new_expert_map.clone())
        else:
            self._emap_cpu_cache[layer_idx] = None

    def _globals_to_locals(
        self,
        layer_idx: int,
        global_ids: Set[int],
        layer=None,
    ) -> Set[int]:
        """Convert global expert IDs to local expert IDs.

        For ep_size=1 (expert_map=None): global_id == local_id.
        For EP: uses layer._expert_map to translate.
        """
        if layer is not None and hasattr(layer, '_expert_map') \
                and layer._expert_map is not None:
            emap = layer._expert_map
            emap_cpu = emap.cpu() if emap.device.type != 'cpu' else emap
            local_ids = set()
            for gid in global_ids:
                if 0 <= gid < emap_cpu.shape[0]:
                    lid = emap_cpu[gid].item()
                    if lid != -1:
                        local_ids.add(lid)
            return local_ids
        else:
            # ep_size=1: identity mapping
            return {gid for gid in global_ids
                    if 0 <= gid < self.local_num_experts}

    def _async_fetch(
        self,
        layer_idx: int,
        local_ids: List[int],
        protected: Optional[Set[int]] = None,
    ):
        """Async CPU→GPU fetch on copy_stream.

        pre_step() calls copy_stream.synchronize() once after all layers.
        """
        protected = set(protected) if protected else set()
        with torch.cuda.stream(self._copy_stream):
            for i, lid in enumerate(local_ids):
                if not self._free_slots[layer_idx]:
                    if not self._evict_one(layer_idx, protected):
                        self.stats.sync_fetches += 1  # track skips
                        continue
                slot = self._free_slots[layer_idx].pop()
                self._set_slot_expert(layer_idx, slot, lid)
                self._set_expert_slot(layer_idx, lid, slot)
                self._access_count[(layer_idx, lid)] = 1
                self._last_access[(layer_idx, lid)] = self.current_step
                protected.add(lid)

                w13_cpu, w2_cpu = self._cpu_pool[layer_idx][lid]
                self._layer_w13[layer_idx][slot].copy_(
                    w13_cpu, non_blocking=True)
                self._layer_w2[layer_idx][slot].copy_(
                    w2_cpu, non_blocking=True)

                # Track DMA bytes for instrumentation
                if hasattr(self, '_timing'):
                    nbytes = (w13_cpu.nbytes + w2_cpu.nbytes)
                    self._timing['bytes_fetched_total'] += nbytes
                    self._timing['experts_fetched_total'] += 1

    def _update_cache_map(self, layer_idx: int, layer,
                          skip_gpu_upload: bool = False):
        """In-place update layer._cache_map with current slot mapping.

        Vectorized: uses _expert_to_slot_t tensor mirror and _emap_cpu_cache
        to avoid per-element .item() calls. Falls back to loop if batched
        D2H not initialized yet.

        Uses .copy_() to preserve data_ptr() for CUDA graph compatibility.

        Args:
            skip_gpu_upload: If True, only writes CPU scratch map.
                GPU upload is deferred to batched bulk copy in pre_step().
        """
        self.__init_v2_scratch()
        scratch = self._scratch_maps[layer_idx]

        if hasattr(self, '_expert_to_slot_t'):
            # Vectorized path (zero .item() calls)
            e2s = self._expert_to_slot_t[layer_idx]  # [local_E] int32

            emap_cpu = (self._emap_cpu_cache[layer_idx]
                        if hasattr(self, '_emap_cpu_cache')
                        and layer_idx < len(self._emap_cpu_cache)
                        else None)

            if emap_cpu is not None:
                # EP path: emap[gid] -> lid, e2s[lid] -> slot
                valid = emap_cpu >= 0               # [global_E] bool
                lids = emap_cpu.clamp(min=0).long()  # safe index
                slots = e2s[lids]                    # vectorized gather
                mask = valid & (slots >= 0)
                scratch.fill_(-1)
                scratch[mask] = slots[mask]
            else:
                # ep_size=1: global_id == local_id
                scratch.fill_(-1)
                local_e = min(self.local_num_experts, scratch.shape[0])
                valid = e2s[:local_e] >= 0
                scratch[:local_e] = torch.where(
                    valid, e2s[:local_e],
                    torch.tensor(-1, dtype=torch.int32))
        else:
            # Fallback: scalar loop (before _init_batched_d2h)
            scratch.fill_(-1)
            if hasattr(layer, '_expert_map') \
                    and layer._expert_map is not None:
                emap = layer._expert_map
                emap_cpu = emap.cpu() \
                    if emap.device.type != 'cpu' else emap
                for gid in range(emap_cpu.shape[0]):
                    lid = emap_cpu[gid].item()
                    if lid != -1:
                        slot = self._expert_to_slot[layer_idx][lid]
                        if slot != -1:
                            scratch[gid] = slot
            else:
                for lid in range(self.local_num_experts):
                    slot = self._expert_to_slot[layer_idx][lid]
                    if slot != -1:
                        scratch[lid] = slot

        if not skip_gpu_upload:
            # In-place copy preserves data_ptr
            layer._cache_map.copy_(scratch.to(layer._cache_map.device))

    def _globals_to_locals_cached(
        self,
        layer_idx: int,
        global_ids: Set[int],
    ) -> Set[int]:
        """CPU-only global→local conversion using cached expert_map."""
        emap_cpu = self._emap_cpu_cache[layer_idx]
        if emap_cpu is not None:
            local_ids = set()
            for gid in global_ids:
                if 0 <= gid < emap_cpu.shape[0]:
                    lid = emap_cpu[gid].item()
                    if lid != -1:
                        local_ids.add(lid)
            return local_ids
        else:
            # ep_size=1: identity mapping
            return {gid for gid in global_ids
                    if 0 <= gid < self.local_num_experts}

    def pre_step(self, layers, num_tokens: int = 0) -> dict:
        """v2 entry point: called BEFORE CUDA graph replay.

        Batched D2H optimization: gathers all routing data from all layers
        into stacked GPU buffers, does a single bulk D2H transfer, then
        processes everything on CPU without further GPU sync.

        Args:
            layers: List of FusedMoE layer modules.
            num_tokens: Actual unpadded token count this step (HIGH-1 fix).
                Used to compute valid_routing_len = num_tokens * top_k.
                If 0, falls back to graph-recorded _routing_len.

        Returns:
            dict with 'total_misses', 'total_routed', 'miss_ratio',
            and timing fields for instrumentation.
        """
        t0 = time.monotonic()
        _nvtx = torch.cuda.nvtx if hasattr(torch.cuda, 'nvtx') else None
        if _nvtx:
            _nvtx.range_push("expert_cache::pre_step")

        self.__init_v2_scratch()
        self._init_batched_d2h(layers)

        # ── Phase 0: Deferred sync from previous step ─────────────
        if _nvtx:
            _nvtx.range_push("phase0_deferred_sync")
        t_dsync_start = time.monotonic()
        if self._prev_needs_sync:
            self._copy_stream.synchronize()
            self._prev_needs_sync = False
            # Previous pending slots now have valid data — clear them.
            # Next _update_cache_map will naturally include them.
            self._pending_dma_slots.clear()
        t_dsync = time.monotonic() - t_dsync_start
        self._timing['t_deferred_sync_us'] += t_dsync * 1e6
        if _nvtx:
            _nvtx.range_pop()  # phase0

        self.current_step += 1
        self._prefetch_bytes_this_step = 0
        total_misses = 0
        total_routed = 0
        needs_sync = False

        # ── Phase A: GPU→GPU gather ──────────────────────────────
        if _nvtx:
            _nvtx.range_push("phaseA_gpu_gather")
        t_gather_start = time.monotonic()
        active_indices = self._active_layer_indices
        for i in active_indices:
            layer = layers[i]
            self._routing_len_gpu[i].copy_(layer._routing_len[0])
            snap_len = min(int(layer._routing_snapshot.shape[0]),
                           self._max_snap_len)
            self._routing_snap_gpu[i, :snap_len].copy_(
                layer._routing_snapshot[:snap_len])
        t_gather = time.monotonic() - t_gather_start
        if _nvtx:
            _nvtx.range_pop()  # phaseA

        # ── Phase B: Bulk D2H (single stream, 1 sync) ───────────
        if _nvtx:
            _nvtx.range_push("phaseB_d2h")
        t_d2h_start = time.monotonic()
        if self._d2h_stream is not None:
            with torch.cuda.stream(self._d2h_stream):
                self._routing_len_cpu.copy_(
                    self._routing_len_gpu, non_blocking=True)
                self._routing_snap_cpu.copy_(
                    self._routing_snap_gpu, non_blocking=True)
                self._d2h_event.record(self._d2h_stream)
            self._d2h_event.synchronize()  # single sync point
        else:
            # CPU-only fallback (testing)
            self._routing_len_cpu.copy_(self._routing_len_gpu)
            self._routing_snap_cpu.copy_(self._routing_snap_gpu)
        t_d2h = time.monotonic() - t_d2h_start
        if _nvtx:
            _nvtx.range_pop()  # phaseB

        # ── Phase C: CPU-only processing (no GPU sync) ───────────
        if _nvtx:
            _nvtx.range_push("phaseC_classify_fetch")
        t_classify_start = time.monotonic()
        for i in active_indices:
            layer = layers[i]

            # 1. Read routing from CPU landing pad
            raw_rlen = int(self._routing_len_cpu[i].item())
            rlen = raw_rlen
            valid_len = raw_rlen  # track for diagnostics
            if num_tokens > 0:
                top_k = getattr(layer, 'top_k', 10)
                valid_len = num_tokens * top_k
                rlen = min(rlen, valid_len)

            if rlen > 0:
                topk_cpu = self._routing_snap_cpu[i, :rlen]
                needed_global = set(topk_cpu.unique().tolist())
                needed_local = self._globals_to_locals_cached(
                    i, needed_global)
                # Gating distribution: accumulate per-expert counts
                # Sample first active layer only (O(n) numpy bincount)
                if i == active_indices[0]:
                    ids = topk_cpu.numpy().astype(np.int64)
                    ids = ids[(ids >= 0) & (ids < self.global_num_experts)]
                    if len(ids) > 0:
                        self._gating_histogram += np.bincount(
                            ids, minlength=self.global_num_experts)
                    self._gating_steps += 1
            else:
                topk_cpu = None
                # First step (warmup): use initial cache contents
                needed_local = set()
                for lid in range(self.local_num_experts):
                    if self._expert_to_slot[i][lid] != -1:
                        needed_local.add(lid)

            total_routed += len(needed_local)

            # 2. Hit/miss classification (unique-ID based)
            miss_ids = []
            for lid in needed_local:
                key = (i, lid)
                if self._expert_to_slot[i][lid] != -1:
                    self._access_count[key] += 1
                    self._last_access[key] = self.current_step
                    self.stats.hits += 1
                else:
                    miss_ids.append(lid)
                    self.stats.misses += 1

            total_misses += len(miss_ids)

            # 2b. Token-weighted hit/miss (each routing entry counted)
            if topk_cpu is not None and rlen > 0:
                miss_set = set(miss_ids)
                # Count per-token routing entries that hit vs miss
                # Convert global→local for each routing entry
                emap_cpu = (self._emap_cpu_cache[i]
                            if hasattr(self, '_emap_cpu_cache')
                            and i < len(self._emap_cpu_cache)
                            else None)
                tk_hit = 0
                tk_miss = 0
                for gid_t in topk_cpu.tolist():
                    gid = int(gid_t)
                    if emap_cpu is not None:
                        if 0 <= gid < emap_cpu.shape[0]:
                            lid = int(emap_cpu[gid].item())
                        else:
                            lid = -1
                    else:
                        lid = gid if 0 <= gid < self.local_num_experts else -1
                    if lid == -1:
                        continue
                    if lid in miss_set:
                        tk_miss += 1
                    else:
                        tk_hit += 1
                self.stats.token_hits += tk_hit
                self.stats.token_misses += tk_miss

            # 2c. Per-layer diagnostics accumulation
            if hasattr(self, '_diag'):
                li = min(i, self.num_layers - 1)
                self._diag['per_layer_rlen'][li] += raw_rlen
                self._diag['per_layer_valid_len'][li] += valid_len
                self._diag['per_layer_unique_needed'][li] += len(needed_local)
                self._diag['per_layer_hits'][li] += (
                    len(needed_local) - len(miss_ids))
                self._diag['per_layer_misses'][li] += len(miss_ids)
                if topk_cpu is not None and rlen > 0:
                    self._diag['per_layer_token_hits'][li] += tk_hit
                    self._diag['per_layer_token_misses'][li] += tk_miss

            # 3. Async-fetch misses (DMA issued, not synced)
            t_fetch_start = time.monotonic()
            if miss_ids:
                cached_needed = needed_local - set(miss_ids)
                # Record slots BEFORE fetch so we know which are pending
                pre_fetch_slots = set()
                for lid in miss_ids:
                    slot = self._expert_to_slot[i][lid]
                    if slot != -1:
                        pre_fetch_slots.add(slot)
                self._async_fetch(i, miss_ids, protected=cached_needed)
                needs_sync = True
                # Track NEW slots allocated by _async_fetch (not in pre_fetch)
                pending = set()
                for lid in miss_ids:
                    slot = self._expert_to_slot[i][lid]
                    if slot != -1 and slot not in pre_fetch_slots:
                        pending.add(slot)
                if pending:
                    self._pending_dma_slots[i] = pending
            t_fetch_delta = time.monotonic() - t_fetch_start
            self._timing['t_fetch_us'] += t_fetch_delta * 1e6

            # 4. Update _cache_map CPU scratch (vectorized, no GPU upload)
            t_cmap_start = time.monotonic()
            self._update_cache_map(i, layer, skip_gpu_upload=True)
            t_cmap_delta = time.monotonic() - t_cmap_start
            self._timing['t_cache_map_us'] += t_cmap_delta * 1e6
            self._timing['t_cache_map_build_us'] += t_cmap_delta * 1e6

        t_classify = time.monotonic() - t_classify_start
        if _nvtx:
            _nvtx.range_pop()  # phaseC

        # ── Phase C1.5: Mask pending DMA slots from cache_map ──────
        # Slots with in-flight DMA have allocated slots but no valid
        # weight data yet.  Set cache_map entries pointing to those
        # slots to -1 so the kernel skips them this step.  Data will
        # be available after deferred sync at the start of next step.
        if self._pending_dma_slots:
            for li, pending_slots in self._pending_dma_slots.items():
                scratch = self._scratch_maps[li]
                for slot in pending_slots:
                    # scratch[gid]==slot means gid maps to this pending slot
                    scratch[scratch == slot] = -1

        # ── Phase C2: Batched cache_map GPU upload ─────────────────
        if _nvtx:
            _nvtx.range_push("phaseC2_cache_map_upload")
        t_cmap_upload_start = time.monotonic()
        if (hasattr(self, '_stacked_scratch_gpu')
                and self._stacked_scratch_gpu is not None):
            # Bulk CPU→GPU (pinned, non_blocking ~10us for 96KB)
            self._stacked_scratch_gpu.copy_(
                self._stacked_scratch_cpu, non_blocking=True)
            # GPU→GPU scatter to each layer's _cache_map
            # All on default stream → ordered before graph replay
            for i in active_indices:
                layers[i]._cache_map.copy_(
                    self._stacked_scratch_gpu[i], non_blocking=True)
        else:
            # CPU-only or non-CUDA: direct copy
            for i in active_indices:
                layers[i]._cache_map.copy_(
                    self._scratch_maps[i].to(layers[i]._cache_map.device))
        t_cmap_upload = time.monotonic() - t_cmap_upload_start
        self._timing['t_cache_map_us'] += t_cmap_upload * 1e6
        self._timing['t_cache_map_upload_us'] += t_cmap_upload * 1e6
        if _nvtx:
            _nvtx.range_pop()  # phaseC2

        # ── Phase D: Deferred — DMA overlaps with graph replay ────
        # Instead of blocking here, we defer the sync to the start
        # of next pre_step().  The copy_stream DMA (~6.5ms) runs
        # concurrently with graph replay (~7.7ms) on default stream.
        # Since cache_map excludes pending slots, no data race.
        t_sync_start = time.monotonic()
        if needs_sync:
            self._prev_needs_sync = True
            # No synchronize() here — that's the whole point!
        t_sync = time.monotonic() - t_sync_start

        t_total = time.monotonic() - t0
        if _nvtx:
            _nvtx.range_pop()  # expert_cache::pre_step

        # miss_ratio for proportional fallback
        miss_ratio = (total_misses / total_routed
                      if total_routed > 0 else 0.0)

        # Timing instrumentation
        self._timing['t_pre_step_total_us'] += t_total * 1e6
        self._timing['t_gpu_gather_us'] += t_gather * 1e6
        self._timing['t_d2h_us'] += t_d2h * 1e6
        self._timing['t_classify_us'] += t_classify * 1e6
        self._timing['t_sync_us'] += t_sync * 1e6
        self._timing['pre_step_calls'] += 1

        # Periodic miss-rate diagnostic dump
        if hasattr(self, '_diag'):
            self._diag['diag_steps'] += 1
            interval = self._diag['diag_dump_interval']
            if self._diag['diag_steps'] % interval == 0:
                logger.info(self.get_miss_diagnostics())

        return {
            'total_misses': total_misses,
            'total_routed': total_routed,
            'miss_ratio': miss_ratio,
            't_pre_step_us': t_total * 1e6,
            't_gpu_gather_us': t_gather * 1e6,
            't_d2h_us': t_d2h * 1e6,
            't_classify_us': t_classify * 1e6,
            't_sync_us': t_sync * 1e6,
        }

    def get_timing_summary(self) -> str:
        """Return human-readable timing summary (avg per pre_step call)."""
        if not hasattr(self, '_timing'):
            return "ExpertCache timing: not yet initialized (no pre_step calls)"
        n = max(self._timing.get('pre_step_calls', 0), 1)
        lines = [f"ExpertCache timing (avg over {n} calls):"]
        for key in ['t_pre_step_total_us', 't_gpu_gather_us',
                     't_deferred_sync_us', 't_d2h_us',
                     't_classify_us', 't_cache_map_us',
                     't_cache_map_build_us', 't_cache_map_upload_us',
                     't_fetch_us', 't_sync_us']:
            val = self._timing.get(key, 0.0)
            lines.append(f"  {key}: {val/n:.1f} us/call")
        # DMA bytes instrumentation
        total_bytes = self._timing.get('bytes_fetched_total', 0)
        total_experts = self._timing.get('experts_fetched_total', 0)
        bytes_per_step = total_bytes / n
        experts_per_step = total_experts / n
        t_fetch_total_s = self._timing.get('t_fetch_us', 0.0) / 1e6
        eff_gbps = (total_bytes / t_fetch_total_s / 1e9
                    if t_fetch_total_s > 0 else 0.0)
        lines.append(
            f"  dma_bytes/step: {bytes_per_step/1e6:.2f} MB "
            f"({experts_per_step:.1f} experts/step, "
            f"eff_bw: {eff_gbps:.1f} GB/s)")
        lines.append(
            f"  hit_rate(unique-ID): {self.stats.hit_rate:.4f} "
            f"(hits={self.stats.hits}, misses={self.stats.misses})")
        lines.append(
            f"  hit_rate(token-wtd): {self.stats.token_hit_rate:.4f} "
            f"(tok_hits={self.stats.token_hits}, "
            f"tok_misses={self.stats.token_misses})")
        return "\n".join(lines)

    def get_gating_summary(self) -> str:
        """Return expert gating distribution summary.

        Shows top-20 most routed experts, entropy, and coverage stats.
        """
        total = self._gating_histogram.sum()
        if total == 0:
            return "Gating: no routing data collected yet"
        # Normalize to probability
        probs = self._gating_histogram / total
        nonzero = probs[probs > 0]
        entropy = -np.sum(nonzero * np.log2(nonzero))
        max_entropy = np.log2(self.global_num_experts)
        # Coverage: how many experts received > 0.1% of routing
        active_01 = int(np.sum(probs > 0.001))
        active_1 = int(np.sum(probs > 0.01))
        # Top-20
        top_idx = np.argsort(self._gating_histogram)[::-1][:20]
        lines = [
            f"Expert gating distribution (layer 0, {self._gating_steps} steps, "
            f"{int(total)} total routings):",
            f"  entropy: {entropy:.2f} / {max_entropy:.2f} bits "
            f"({entropy/max_entropy*100:.1f}% of uniform)",
            f"  active experts: {active_01} (>0.1%), {active_1} (>1%)",
            f"  top-20:"
        ]
        for rank, idx in enumerate(top_idx):
            count = self._gating_histogram[idx]
            pct = count / total * 100
            lines.append(f"    #{rank+1} expert {idx}: "
                         f"{int(count)} ({pct:.2f}%)")
        return "\n".join(lines)

    def get_miss_diagnostics(self) -> str:
        """Return detailed miss-rate diagnostics for verification.

        Reports per-layer and aggregate stats to answer:
        1. Is valid_len truncating rlen? (measurement artifact)
        2. How many unique experts does each layer need? (locality check)
        3. Unique-ID hit rate vs token-weighted hit rate (definition check)
        4. Per-layer variance (are some layers worse than others?)
        """
        if not hasattr(self, '_diag'):
            return "Miss diagnostics: not initialized"
        d = self._diag
        steps = max(d['diag_steps'], 1)
        lines = [f"=== Miss-Rate Diagnostics (step {self.current_step}, "
                 f"last {steps} steps) ==="]

        # ── Aggregate stats ──
        lines.append(f"\n[Aggregate]")
        lines.append(
            f"  unique-ID hit_rate: {self.stats.hit_rate:.4f} "
            f"(hits={self.stats.hits}, misses={self.stats.misses})")
        lines.append(
            f"  token-wtd hit_rate: {self.stats.token_hit_rate:.4f} "
            f"(tok_hits={self.stats.token_hits}, "
            f"tok_misses={self.stats.token_misses})")
        lines.append(
            f"  max_resident: {self.max_resident}, "
            f"local_experts: {self.local_num_experts}, "
            f"global_experts: {self.global_num_experts}")

        # ── valid_len vs rlen check ──
        lines.append(f"\n[valid_len vs rlen — artifact check]")
        total_rlen = d['per_layer_rlen'].sum()
        total_vlen = d['per_layer_valid_len'].sum()
        if total_rlen > 0:
            truncation_pct = max(0, (total_rlen - total_vlen)) / total_rlen * 100
            lines.append(
                f"  avg rlen/layer/step: {total_rlen / steps / self.num_layers:.1f}")
            lines.append(
                f"  avg valid_len/layer/step: {total_vlen / steps / self.num_layers:.1f}")
            lines.append(
                f"  truncation: {truncation_pct:.1f}% of routing entries clipped")
        else:
            lines.append("  no routing data yet")

        # ── Per-layer detail (sample 5 layers: 0, 11, 23, 35, 47) ──
        lines.append(f"\n[Per-Layer Detail (avg/step over {steps} steps)]")
        lines.append(
            f"  {'layer':>5} {'rlen':>7} {'vlen':>7} "
            f"{'uniq_need':>9} {'uid_hit':>7} {'uid_miss':>8} "
            f"{'uid_rate':>8} {'tok_hit':>8} {'tok_miss':>8} "
            f"{'tok_rate':>8}")
        sample_layers = [0, 11, 23, 35, 47]
        for li in sample_layers:
            if li >= self.num_layers:
                continue
            rlen_avg = d['per_layer_rlen'][li] / steps
            vlen_avg = d['per_layer_valid_len'][li] / steps
            uniq_avg = d['per_layer_unique_needed'][li] / steps
            h = d['per_layer_hits'][li]
            m = d['per_layer_misses'][li]
            uid_rate = h / (h + m) if (h + m) > 0 else 0.0
            th = d['per_layer_token_hits'][li]
            tm = d['per_layer_token_misses'][li]
            tok_rate = th / (th + tm) if (th + tm) > 0 else 0.0
            lines.append(
                f"  {li:>5} {rlen_avg:>7.0f} {vlen_avg:>7.0f} "
                f"{uniq_avg:>9.1f} {h/steps:>7.1f} {m/steps:>8.2f} "
                f"{uid_rate:>8.4f} {th/steps:>8.1f} {tm/steps:>8.2f} "
                f"{tok_rate:>8.4f}")

        # ── Unique-needed distribution across all layers ──
        lines.append(f"\n[Unique-Needed Stats]")
        uniq_all = d['per_layer_unique_needed'] / steps
        lines.append(
            f"  min={uniq_all.min():.1f}, max={uniq_all.max():.1f}, "
            f"mean={uniq_all.mean():.1f}, std={uniq_all.std():.1f}")
        over_max = int(np.sum(uniq_all > self.max_resident))
        lines.append(
            f"  layers needing > max_resident({self.max_resident}): "
            f"{over_max}/{self.num_layers}")

        # ── Cache occupancy snapshot ──
        lines.append(f"\n[Cache Occupancy Snapshot]")
        for li in sample_layers:
            if li >= self.num_layers:
                continue
            occupied = sum(
                1 for s in range(self.max_resident)
                if self._slot_to_expert[li][s] != -1)
            lines.append(
                f"  layer {li}: {occupied}/{self.max_resident} slots used")

        return "\n".join(lines)
