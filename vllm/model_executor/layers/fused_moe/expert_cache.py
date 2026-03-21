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

# Module-level ref for group boundary custom op
_global_expert_cache_ref: Dict = {}


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
        self._dormant: bool = False

        # Phase C stats
        self._phase_c_enabled: bool = False
        self._phase_c_stats: Dict[str, int] = {
            'shrink_calls': 0,
            'shrink_pages': 0,
        }

    # ================================================================
    # Slot helpers
    # ================================================================

    def _set_expert_slot(self, layer_idx: int, lid: int, slot: int):
        self._expert_to_slot[layer_idx][lid] = slot
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
    ):
        protected = set(protected_local_ids) if protected_local_ids else set()
        skipped = 0
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
        if skipped > 0:
            logger.warning(
                "Layer %d: needed > max_resident (%d), skipped %d experts",
                layer_idx, self.max_resident, skipped)
        torch.cuda.synchronize(self.device)

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
        """Return True if any expert slots are currently evicted (unmapped)."""
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
        # Filter: only fully unpinned groups
        return {gk: members
                for gk, (pinned, members) in groups.items()
                if not pinned and members}

    def count_evictable_groups(self) -> int:
        """Count groups that can be evicted (not pinned, have residents).

        Uses _can_shrink() and _evictable_groups() to match the exact
        same eligibility as shrink_for_pages / _select_lru_victims.
        """
        if not self._can_shrink():
            return 0
        return len(self._evictable_groups())

    def shrink_for_pages(self, min_pages: int) -> Tuple[int, int]:
        """Evict expert groups to free at least min_pages physical pages.

        Simplified from shrink_for_kv: no VA scoring, no pressure epoch.

        Returns:
            (freed_pages, groups_evicted)
        """
        if not self._can_shrink():
            return 0, 0

        pool = self._vmm_pool

        if self._dormant:
            self._dormant = False

        groups_needed = math.ceil(min_pages / pool.group_pages)
        victims = self._select_lru_victims(
            groups_needed * pool.group_size)
        if not victims:
            return 0, 0

        freed_pages = pool.unmap_expert_slots(victims)
        groups_evicted = self._update_tracking_after_evict(victims)
        self._cache_map_needs_rebuild = True

        self._phase_c_stats['shrink_calls'] += 1
        self._phase_c_stats['shrink_pages'] += freed_pages
        logger.info(
            "shrink_for_pages: freed %d groups → %d pages",
            groups_evicted, freed_pages)
        return freed_pages, groups_evicted

    def _select_lru_victims(self, count: int) -> List[Tuple[int, int]]:
        """Select coldest expert slots by LRU (group-aware).

        Uses _evictable_groups() for consistent pinned-group semantics.
        """
        evictable_map = self._evictable_groups()
        if not evictable_map:
            return []

        # Score each group by max(last_access) across members
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

        # Sort coldest-first
        scored.sort()

        victims = []
        for _, _, members in scored:
            if len(victims) >= count:
                break
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
        # Fast path: all experts resident, no shrink happened
        if self.max_resident >= self.local_num_experts:
            if not self._phase_c_enabled:
                self.current_step += 1
                return {'total_misses': 0, 'total_routed': 0,
                        'miss_ratio': 0.0}
            elif (not self._cache_map_needs_rebuild
                  and not any(self._unmapped_slots)):
                self.current_step += 1
                return {'total_misses': 0, 'total_routed': 0,
                        'miss_ratio': 0.0}

        total_misses = 0
        total_routed = 0

        for i, layer in enumerate(layers):
            if i >= self.num_layers:
                break
            if not hasattr(layer, '_routing_snapshot'):
                continue
            if not hasattr(layer, '_cache_map'):
                continue

            result = self.pre_step_single_layer(
                i, layer, num_tokens)
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
        }

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
