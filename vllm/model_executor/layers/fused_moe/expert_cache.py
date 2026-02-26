# SPDX-License-Identifier: Apache-2.0
"""Expert Cache Manager for MoE weight offloading.

Single implementation path:
- w13_weight resized to (max_resident, ...)
- cache_map replaces expert_map temporarily in forward_cuda()
- CPU pageable backing store + pinned staging window
"""

import torch
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging

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

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


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

        # CPU pageable backing store
        self._cpu_pool: List[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = [
            {} for _ in range(num_layers)
        ]

        # Pinned staging window
        staging_n = config.staging_window_experts
        self._staging_w13 = torch.empty(
            (staging_n, *expert_w13_shape), dtype=dtype, device='cpu'
        ).pin_memory()
        self._staging_w2 = torch.empty(
            (staging_n, *expert_w2_shape), dtype=dtype, device='cpu'
        ).pin_memory()

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
        """Copy expert weight to CPU pageable pool."""
        w13_cpu = w13_weight.detach().cpu()
        w2_cpu = w2_weight.detach().cpu()
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
        """CPU -> staging -> GPU slot (synchronous)."""
        if not self._free_slots[layer_idx]:
            return False
        slot = self._free_slots[layer_idx].pop()
        self._slot_to_expert[layer_idx][slot] = local_id
        self._expert_to_slot[layer_idx][local_id] = slot

        w13_cpu, w2_cpu = self._cpu_pool[layer_idx][local_id]
        self._staging_w13[0].copy_(w13_cpu)
        self._staging_w2[0].copy_(w2_cpu)
        self._layer_w13[layer_idx][slot].copy_(self._staging_w13[0])
        self._layer_w2[layer_idx][slot].copy_(self._staging_w2[0])
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
        to_fetch = to_fetch[:min(
            len(to_fetch), remaining, self.config.staging_window_experts
        )]

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
                if i >= self.config.staging_window_experts:
                    break

                slot = self._free_slots[target_layer_idx].pop()
                self._slot_to_expert[target_layer_idx][slot] = lid
                self._expert_to_slot[target_layer_idx][lid] = slot
                self._access_count[(target_layer_idx, lid)] = 0
                self._last_access[(target_layer_idx, lid)] = self.current_step

                w13_cpu, w2_cpu = self._cpu_pool[target_layer_idx][lid]
                self._staging_w13[i].copy_(w13_cpu)
                self._staging_w2[i].copy_(w2_cpu)
                self._layer_w13[target_layer_idx][slot].copy_(
                    self._staging_w13[i], non_blocking=True
                )
                self._layer_w2[target_layer_idx][slot].copy_(
                    self._staging_w2[i], non_blocking=True
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
            self._slot_to_expert[layer_idx][slot] = lid
            self._expert_to_slot[layer_idx][lid] = slot
            self._access_count[(layer_idx, lid)] = 1
            self._last_access[(layer_idx, lid)] = self.current_step
            protected.add(lid)  # protect newly loaded expert from eviction

            si = i % self.config.staging_window_experts
            w13_cpu, w2_cpu = self._cpu_pool[layer_idx][lid]
            self._staging_w13[si].copy_(w13_cpu)
            self._staging_w2[si].copy_(w2_cpu)
            self._layer_w13[layer_idx][slot].copy_(self._staging_w13[si])
            self._layer_w2[layer_idx][slot].copy_(self._staging_w2[si])

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
        self._slot_to_expert[layer_idx][best_slot] = -1
        self._expert_to_slot[layer_idx][lid] = -1
        self._free_slots[layer_idx].append(best_slot)
        self.stats.evictions += 1
        return True

    # --- Step ---

    def step(self):
        """Called once per forward step. Resets BW throttle."""
        self.current_step += 1
        self._prefetch_bytes_this_step = 0
