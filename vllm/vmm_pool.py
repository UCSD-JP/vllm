"""CUDA VMM-backed dynamic Expert↔KV page pool.

Manages a shared pool of 2MB physical pages that can be dynamically
remapped between Expert weight storage and KV cache regions using
CUDA Virtual Memory Management (cuMemMap / cuMemUnmap).

Usage:
    pool = VMMPagePool(device_id=0, expert_total_bytes=..., kv_max_bytes=...,
                       expert_slot_bytes=..., num_layers=48, max_slots_per_layer=256)
    # Expert tensors backed by VMM VA
    w13, w2 = pool.get_expert_tensor(layer=0, slot=0, w13_shape, w2_shape, dtype)
    # Dynamic transfer: expert pages → KV
    new_kv = pool.transfer_expert_to_kv([(layer, slot), ...])
    # Reverse: KV pages → expert
    pool.transfer_kv_to_expert(kv_offsets, [(layer, slot), ...])

Requires: pip install cuda-python (or nvidia-cuda-python)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# cuda-python import with version fallback
# ---------------------------------------------------------------------------
try:
    from cuda.bindings.driver import (
        CUmemAccess_flags,
        CUmemAccessDesc,
        CUmemAllocationGranularity_flags,
        CUmemAllocationProp,
        CUmemAllocationType,
        CUmemLocationType,
        CUresult,
        cuInit,
        cuMemAddressFree,
        cuMemAddressReserve,
        cuMemCreate,
        cuMemGetAllocationGranularity,
        cuMemMap,
        cuMemRelease,
        cuMemSetAccess,
        cuMemUnmap,
    )
    _CUDA_BINDINGS = True
except ImportError:
    try:
        from cuda.cuda import (
            CUmemAccess_flags,
            CUmemAccessDesc,
            CUmemAllocationGranularity_flags,
            CUmemAllocationProp,
            CUmemAllocationType,
            CUmemLocationType,
            CUresult,
            cuInit,
            cuMemAddressFree,
            cuMemAddressReserve,
            cuMemCreate,
            cuMemGetAllocationGranularity,
            cuMemMap,
            cuMemRelease,
            cuMemSetAccess,
            cuMemUnmap,
        )
        _CUDA_BINDINGS = True
    except ImportError:
        _CUDA_BINDINGS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_cuda(err, msg: str = ""):
    """Raise RuntimeError on CUDA driver error."""
    if isinstance(err, tuple):
        err = err[0]
    if isinstance(err, int):
        if err != 0:
            raise RuntimeError(f"CUDA VMM error code {err}: {msg}")
    elif err != CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA VMM error {err.name} ({int(err)}): {msg}")


def _round_up(x: int, granularity: int) -> int:
    return math.ceil(x / granularity) * granularity


# ---------------------------------------------------------------------------
# Batched page ↔ block helpers (MVP elastic KV)
# ---------------------------------------------------------------------------

def pages_for_blocks(
    n_blocks: int,
    per_tensor_block_bytes: Dict[int, int],
    page_size: int,
) -> int:
    """Return total physical pages needed for *n_blocks* KV blocks (batched).

    Batching gain: ``ceil(64 * 32 KiB / 2 MiB) = 1`` vs
    ``64 × ceil(32 KiB / 2 MiB) = 64``.
    """
    return sum(
        math.ceil(n_blocks * pb / page_size)
        for pb in per_tensor_block_bytes.values()
        if pb > 0
    )


def max_blocks_for_pages(
    n_pages: int,
    per_tensor_block_bytes: Dict[int, int],
    page_size: int,
) -> int:
    """Return max KV blocks achievable from *n_pages* free pages (binary search)."""
    pbs = [pb for pb in per_tensor_block_bytes.values() if pb > 0]
    if not pbs:
        return 0
    lo, hi = 0, n_pages * page_size // max(sum(pbs), 1)
    hi = max(hi, 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        total = sum(math.ceil(mid * pb / page_size) for pb in pbs)
        if total <= n_pages:
            lo = mid
        else:
            hi = mid - 1
    return lo


class _CUDAPointerWrapper:
    """Wraps a raw CUdeviceptr so ``torch.as_tensor`` can consume it via
    the ``__cuda_array_interface__`` protocol (zero-copy)."""

    def __init__(self, ptr: int, size_bytes: int):
        self._ptr = ptr
        self._size = size_bytes

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self._size,),
            "typestr": "|u1",
            "data": (self._ptr, False),
            "version": 3,
        }


def vmm_ptr_to_tensor(
    ptr: int,
    shape: tuple,
    dtype: torch.dtype,
    device_idx: int = 0,
) -> torch.Tensor:
    """Create a PyTorch tensor backed by a VMM virtual address (zero-copy).

    The caller MUST ensure the underlying VMM pages remain mapped for the
    lifetime of the returned tensor.  Accessing an unmapped VA causes a
    GPU fault.
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    numel = 1
    for s in shape:
        numel *= s
    size_bytes = numel * elem_size
    wrapper = _CUDAPointerWrapper(ptr, size_bytes)
    raw = torch.as_tensor(wrapper, device=f"cuda:{device_idx}")
    return raw.view(dtype).reshape(shape)


def is_vmm_available() -> bool:
    """Return True if CUDA VMM bindings are importable and a GPU is present."""
    if not _CUDA_BINDINGS:
        return False
    try:
        # CUDA driver must be initialized before any cuMem* calls
        cuInit(0)
        prop = CUmemAllocationProp()
        prop.type = CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = 0
        err, _ = cuMemGetAllocationGranularity(
            prop,
            CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
        return err == CUresult.CUDA_SUCCESS
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PageState(IntEnum):
    FREE = 0       # physical page allocated, not mapped to any VA
    EXPERT = 1     # mapped into Expert VA region
    KV = 2         # mapped into KV VA region


@dataclass
class PhysPage:
    """Tracks one physical 2 MB page."""
    page_id: int
    handle: int  # CUmemGenericAllocationHandle
    state: PageState = PageState.FREE
    mapped_va: int = 0
    # Expert bookkeeping (valid when state == EXPERT)
    expert_layer: int = -1
    expert_slot: int = -1
    expert_page_offset: int = -1
    # KV bookkeeping (valid when state == KV)
    kv_va_offset: int = -1


# ---------------------------------------------------------------------------
# VMMPagePool
# ---------------------------------------------------------------------------

class VMMPagePool:
    """Manages a shared physical page pool between Expert and KV VA regions.

    Physical pages (2 MB each on current GPUs) can be dynamically mapped /
    unmapped between the two regions without copying data.
    """

    def __init__(
        self,
        device_id: int,
        expert_total_bytes: int,
        kv_max_bytes: int,
        expert_slot_bytes: int,
        num_layers: int,
        max_slots_per_layer: int,
        dtype: torch.dtype = torch.float16,
        total_phys_override: Optional[int] = None,
    ):
        if not _CUDA_BINDINGS:
            raise RuntimeError(
                "cuda-python is required for VMMPagePool. "
                "Install with: pip install cuda-python"
            )

        # Ensure CUDA driver is initialized (idempotent)
        cuInit(0)

        self.device_id = device_id
        self.num_layers = num_layers
        self.max_slots_per_layer = max_slots_per_layer
        self.dtype = dtype

        # -- 1. Build allocation properties ----------------------------------
        prop = CUmemAllocationProp()
        prop.type = CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device_id
        self._prop = prop

        # -- 2. Query page granularity (typically 2 MB) ----------------------
        err, granularity = cuMemGetAllocationGranularity(
            prop,
            CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
        )
        _check_cuda(err, "cuMemGetAllocationGranularity")
        self.page_size: int = granularity

        # -- 3. Compute per-slot and per-group page geometry ------------------
        self.expert_slot_bytes = expert_slot_bytes
        self.slot_pages: int = math.ceil(expert_slot_bytes / self.page_size)
        self.slot_aligned_bytes: int = self.slot_pages * self.page_size

        # Grouped remap: find smallest group_size where
        # group_size * expert_slot_bytes is page-aligned (zero padding).
        # TP2 (3MiB): group_size=2 → 6MiB = 3 pages
        # TP1 (6MiB): group_size=1 → 6MiB = 3 pages
        # TP4 (1.5MiB): group_size=4 → 6MiB = 3 pages
        self.group_size: int = 1
        for gs in range(1, 9):
            if (gs * expert_slot_bytes) % self.page_size == 0:
                self.group_size = gs
                break
        self.group_pages: int = (
            self.group_size * expert_slot_bytes) // self.page_size
        self.group_bytes: int = self.group_pages * self.page_size

        # -- 4. Compute VA region sizes --------------------------------------
        # Use group-aligned size for VA reservation (zero padding)
        num_groups = math.ceil(max_slots_per_layer / self.group_size)
        self._expert_va_size = _round_up(
            num_layers * num_groups * self.group_bytes,
            self.page_size,
        )
        self._kv_va_size = _round_up(kv_max_bytes, self.page_size)

        # -- 5. Reserve VA ranges --------------------------------------------
        err, expert_va_raw = cuMemAddressReserve(
            self._expert_va_size, self.page_size, 0, 0,
        )
        _check_cuda(err, "cuMemAddressReserve(expert)")
        self._expert_va_base: int = int(expert_va_raw)

        err, kv_va_raw = cuMemAddressReserve(
            self._kv_va_size, self.page_size, 0, 0,
        )
        _check_cuda(err, "cuMemAddressReserve(kv)")
        self._kv_va_base: int = int(kv_va_raw)

        # -- 6. Build access descriptor (reused for every cuMemSetAccess) ----
        access_desc = CUmemAccessDesc()
        access_desc.location.type = CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access_desc.location.id = device_id
        access_desc.flags = (
            CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        )
        self._access_desc = access_desc

        # -- 7. Lazy physical allocation (no cuMemCreate here) ----------------
        # Physical pages are created on-demand via allocate_and_map_slot().
        # This avoids double-residency OOM when replacing PyTorch tensors
        # with VMM-backed ones layer-by-layer.
        self._pages: List[PhysPage] = []
        self._free_pages: List[int] = []
        self._next_page_id: int = 0

        # -- 8. Tracking structures ------------------------------------------
        # (layer, slot, page_offset) → page_id
        self._expert_pages: Dict[Tuple[int, int, int], int] = {}
        # kv_va_offset → page_id
        self._kv_pages: Dict[int, int] = {}
        self._kv_next_va_offset: int = 0
        self._kv_mapped_count: int = 0
        # Fix #1: Recycle freed KV VA offsets instead of append-only
        self._kv_free_offsets: List[int] = []  # reusable KV VA offsets

        self._initial_expert_slots = 0

        # -- Phase C: dynamic expert↔KV exchange --------------------------
        self._phase_c_mode: bool = False   # enabled by VLLM_VMM_PHASE_C=1
        # KV tensor tracking (VMM-backed KV cache)
        self._kv_tensors: Dict[int, Dict] = {}       # tensor_idx → info
        self._kv_va_allocated: int = 0                # bytes allocated in KV VA

        # Split tensor layout [all_w13][all_w2] tracking.
        # Set via set_tensor_layout() before any layer allocation.
        self._w13_bytes: int = 0            # per-expert w13 size in bytes
        self._w2_bytes: int = 0             # per-expert w2 size in bytes
        self._w13_pages_per_group: int = 0  # w13 pages per expert group
        self._w2_pages_per_group: int = 0   # w2 pages per expert group
        self._layout_configured: bool = False
        self._layer_num_slots: int = 0      # actual slots per layer

        # Phase C: per-tensor expansion tracking (set by expand_kv_physical_pages)

        # Sentinel page: a single physical page mapped to ALL unmapped expert
        # VA regions.  This keeps the full stacked weight tensor accessible
        # (Triton validates pointer accessibility at kernel launch in eager
        # mode), while cache_map = -1 prevents routing to sentinel-backed
        # experts.  Lazily created on first shrink.
        self._sentinel_handle: Optional[int] = None
        self._sentinel_vas: Set[int] = set()  # VAs currently sentinel-mapped

        # ensure_safe_remap guard
        self._sync_done_this_step: bool = False
        self._in_forward: bool = False
        self._total_sync_us: float = 0.0

        logger.info(
            "VMMPagePool: device=%d, page_size=%dB (%.1f MiB), "
            "lazy_alloc=True, (%d layers x %d max_slots), "
            "group_size=%d, group_pages=%d, slot_bytes=%d, "
            "expert_va=0x%x (%d MiB), kv_va=0x%x (%d MiB)",
            device_id, self.page_size, self.page_size / (1 << 20),
            num_layers, max_slots_per_layer,
            self.group_size, self.group_pages, expert_slot_bytes,
            self._expert_va_base, self._expert_va_size // (1 << 20),
            self._kv_va_base, self._kv_va_size // (1 << 20),
        )

    # =====================================================================
    # Split tensor layout configuration
    # =====================================================================

    def set_tensor_layout(self, w13_bytes: int, w2_bytes: int) -> None:
        """Configure the [all_w13][all_w2] split layout for expert-aware grouping.

        Must be called BEFORE any layer allocation.  The split layout means
        each expert's data is split across two non-contiguous VA regions:
        - w13 block: slots 0..N-1, each w13_bytes
        - w2 block: slots 0..N-1, each w2_bytes (starts after all w13)

        Expert group G's pages span both regions (non-contiguous in VA):
        - w13 pages at VA offsets [G*w13ppg*ps .. (G+1)*w13ppg*ps)
        - w2 pages at VA offsets [w13_total + G*w2ppg*ps .. ...)
        """
        assert w13_bytes + w2_bytes == self.expert_slot_bytes, (
            f"w13({w13_bytes}) + w2({w2_bytes}) != slot({self.expert_slot_bytes})"
        )
        self._w13_bytes = w13_bytes
        self._w2_bytes = w2_bytes

        # Verify group_size * w13/w2 are each page-aligned
        w13_group_bytes = self.group_size * w13_bytes
        w2_group_bytes = self.group_size * w2_bytes
        assert w13_group_bytes % self.page_size == 0, (
            f"group_size({self.group_size}) * w13({w13_bytes}) = "
            f"{w13_group_bytes} not page-aligned (ps={self.page_size})"
        )
        assert w2_group_bytes % self.page_size == 0, (
            f"group_size({self.group_size}) * w2({w2_bytes}) = "
            f"{w2_group_bytes} not page-aligned (ps={self.page_size})"
        )

        self._w13_pages_per_group = w13_group_bytes // self.page_size
        self._w2_pages_per_group = w2_group_bytes // self.page_size
        assert (self._w13_pages_per_group + self._w2_pages_per_group
                == self.group_pages), (
            f"w13ppg({self._w13_pages_per_group}) + "
            f"w2ppg({self._w2_pages_per_group}) != "
            f"group_pages({self.group_pages})"
        )
        self._layout_configured = True

        logger.info(
            "VMM tensor layout: w13=%d w2=%d bytes/expert, "
            "w13ppg=%d w2ppg=%d pages/group",
            w13_bytes, w2_bytes,
            self._w13_pages_per_group, self._w2_pages_per_group,
        )

    def _get_group_va_offsets(self, group_idx: int) -> List[int]:
        """Return VA byte offsets (from layer base) for each page in group G.

        For the split [all_w13][all_w2] layout, returns non-contiguous
        offsets that span both the w13 and w2 VA regions.

        Example (TP2, w13=2MiB, w2=1MiB, ps=2MiB, group_size=2):
            Group G → [2G*ps, (2G+1)*ps, (510+G)*ps]
            i.e., 2 w13 pages + 1 w2 page (non-contiguous)
        """
        ps = self.page_size
        ns = self._layer_num_slots
        offsets: List[int] = []

        # w13 pages: contiguous within the w13 block
        w13_pg_base = group_idx * self._w13_pages_per_group
        for i in range(self._w13_pages_per_group):
            offsets.append((w13_pg_base + i) * ps)

        # w2 pages: in the w2 block (starts after all w13 data)
        w2_block_start_pages = (ns * self._w13_bytes) // ps
        w2_pg_base = w2_block_start_pages + group_idx * self._w2_pages_per_group
        for i in range(self._w2_pages_per_group):
            offsets.append((w2_pg_base + i) * ps)

        return offsets

    # =====================================================================
    # Stream synchronization (Fix #2: safe remap guarantee)
    # =====================================================================

    def ensure_safe_remap(self) -> None:
        """Synchronize all CUDA streams before any map/unmap operation.

        cuMemMap/Unmap are CPU-side calls that take effect immediately.
        Any GPU kernel or DMA still accessing the affected VA will cause
        a fault.  This method MUST be called before transfer operations
        when GPU work may be in-flight.

        Phase C enhancement: guards against remap during forward pass,
        tracks sync time, and ensures at most one sync per step.
        """
        if self._sync_done_this_step:
            return
        if self._in_forward:
            raise RuntimeError(
                "ensure_safe_remap called during forward pass — "
                "remap is only safe at step boundaries"
            )
        t0 = time.monotonic()
        torch.cuda.synchronize(self.device_id)
        elapsed_us = (time.monotonic() - t0) * 1e6
        self._sync_done_this_step = True
        self._total_sync_us += elapsed_us
        if elapsed_us > 1000:
            logger.warning("ensure_safe_remap stall: %.0fus", elapsed_us)

    def mark_step_start(self) -> None:
        """Reset per-step sync flag. Called at step boundary."""
        self._sync_done_this_step = False

    # =====================================================================
    # Lazy physical page allocation
    # =====================================================================

    def _create_phys_pages(self, n: int) -> List[int]:
        """Allocate ``n`` physical pages via cuMemCreate. Returns page_ids."""
        new_ids: List[int] = []
        for _ in range(n):
            err, handle_raw = cuMemCreate(
                self.page_size, self._prop, 0)
            _check_cuda(
                err,
                f"cuMemCreate(page {self._next_page_id})",
            )
            page = PhysPage(
                page_id=self._next_page_id,
                handle=int(handle_raw),
            )
            self._pages.append(page)
            new_ids.append(self._next_page_id)
            self._next_page_id += 1
        return new_ids

    def allocate_and_map_slot(self, layer: int, slot: int) -> None:
        """Allocate physical pages for one expert slot and map into VA.

        Creates ``slot_pages`` new physical pages (cuMemCreate) and maps
        them into the expert VA region.  This is the primary entry point
        for per-slot VMM initialization.
        """
        # Reuse free pages first, allocate new ones for any deficit
        needed = self.slot_pages
        reuse = min(needed, len(self._free_pages))
        allocate = needed - reuse

        if allocate > 0:
            new_ids = self._create_phys_pages(allocate)
            self._free_pages.extend(new_ids)

        self._map_expert_slot_internal(layer, slot)
        self._initial_expert_slots += 1

    def allocate_and_map_layer_slot_aligned(
        self, layer: int, num_slots: int,
    ) -> int:
        """Allocate group-aligned pages for a layer (Phase C).

        Experts are grouped (group_size experts per group) so that each
        group uses exactly group_pages pages with zero padding.
        E.g., TP2 (3MiB/expert): group_size=2, group_pages=3, 0% waste.

        IMPORTANT: Uses split-layout-aware page tracking.  The tensor
        layout is [all_w13][all_w2], so each expert group's pages are
        non-contiguous in VA space.  Pages are tracked by expert group
        (not by sequential page index).

        Remap unit = group (not single expert).

        Returns:
            Number of physical pages allocated.
        """
        if not self._phase_c_mode:
            raise RuntimeError(
                "allocate_and_map_layer_slot_aligned called but "
                "_phase_c_mode=False. Use tight_pack for Phase A.")
        if not self._layout_configured:
            raise RuntimeError(
                "set_tensor_layout() must be called before "
                "allocate_and_map_layer_slot_aligned()")
        assert num_slots % self.group_size == 0, (
            f"num_slots({num_slots}) must be multiple of "
            f"group_size({self.group_size})")

        self._layer_num_slots = num_slots
        num_groups = num_slots // self.group_size
        num_pages = num_groups * self.group_pages

        reuse = min(num_pages, len(self._free_pages))
        allocate = num_pages - reuse
        if allocate > 0:
            new_ids = self._create_phys_pages(allocate)
            self._free_pages.extend(new_ids)

        # Map ALL pages sequentially (fills entire layer VA)
        va_base = self.get_expert_va(layer, 0)
        page_at_pg_idx: Dict[int, int] = {}
        for pg_idx in range(num_pages):
            if not self._free_pages:
                raise RuntimeError(
                    f"No free pages for layer {layer} "
                    f"page {pg_idx}/{num_pages}")
            page_id = self._free_pages.pop()
            va = va_base + pg_idx * self.page_size
            self._map_page_to_va(page_id, va)
            page = self._pages[page_id]
            page.state = PageState.EXPERT
            page.expert_layer = layer
            page_at_pg_idx[pg_idx] = page_id

        # Build expert-group tracking using split-layout-aware mapping.
        # Each group's pages are non-contiguous in VA space.
        for group_idx in range(num_groups):
            va_offsets = self._get_group_va_offsets(group_idx)
            for offset, va_off in enumerate(va_offsets):
                pg_idx = va_off // self.page_size
                page_id = page_at_pg_idx[pg_idx]
                page = self._pages[page_id]
                page.expert_slot = group_idx
                page.expert_page_offset = offset
                self._expert_pages[(layer, group_idx, offset)] = page_id

        self._initial_expert_slots += num_slots
        return num_pages

    def allocate_and_map_layer_tight_pack(
        self, layer: int, num_slots: int,
    ) -> int:
        """Allocate tight-packed pages for a layer (Phase A).

        Uses ``ceil(num_slots * expert_slot_bytes / page_size)`` pages —
        no per-slot padding. Per-expert remap is NOT possible.

        Returns:
            Number of physical pages allocated.
        """
        if self._phase_c_mode:
            raise RuntimeError(
                "allocate_and_map_layer_tight_pack called but "
                "_phase_c_mode=True. Use slot_aligned for Phase C.")
        total_bytes = num_slots * self.expert_slot_bytes
        num_pages = math.ceil(total_bytes / self.page_size)

        reuse = min(num_pages, len(self._free_pages))
        allocate = num_pages - reuse
        if allocate > 0:
            new_ids = self._create_phys_pages(allocate)
            self._free_pages.extend(new_ids)

        va_base = self.get_expert_va(layer, 0)
        for pg_idx in range(num_pages):
            if not self._free_pages:
                raise RuntimeError(
                    f"No free pages for layer {layer} "
                    f"page {pg_idx}/{num_pages}")
            page_id = self._free_pages.pop()
            va = va_base + pg_idx * self.page_size
            self._map_page_to_va(page_id, va)
            page = self._pages[page_id]
            page.state = PageState.EXPERT
            page.expert_layer = layer
            page.expert_slot = -1
            page.expert_page_offset = pg_idx
            self._expert_pages[(layer, pg_idx, 0)] = page_id

        self._initial_expert_slots += num_slots
        return num_pages

    # Backward compat alias
    allocate_and_map_layer_contiguous = allocate_and_map_layer_slot_aligned

    # =====================================================================
    # Low-level page operations
    # =====================================================================

    def _map_page_to_va(self, page_id: int, va: int) -> None:
        """Map a FREE physical page to a virtual address and set access."""
        page = self._pages[page_id]
        assert page.state == PageState.FREE, (
            f"Page {page_id} is {page.state.name}, expected FREE"
        )
        err, = cuMemMap(va, self.page_size, 0, page.handle, 0)
        _check_cuda(err, f"cuMemMap(page={page_id}, va=0x{va:x})")
        err, = cuMemSetAccess(va, self.page_size, [self._access_desc], 1)
        _check_cuda(err, f"cuMemSetAccess(page={page_id}, va=0x{va:x})")
        page.mapped_va = va

    def _unmap_page(self, page_id: int) -> None:
        """Unmap a page from its current VA. Page becomes FREE."""
        page = self._pages[page_id]
        if page.mapped_va == 0:
            return
        err, = cuMemUnmap(page.mapped_va, self.page_size)
        _check_cuda(err, f"cuMemUnmap(page={page_id}, va=0x{page.mapped_va:x})")
        page.mapped_va = 0
        page.state = PageState.FREE
        page.expert_layer = -1
        page.expert_slot = -1
        page.expert_page_offset = -1
        page.kv_va_offset = -1

    # =====================================================================
    # Sentinel page management (Phase C)
    # =====================================================================

    def _ensure_sentinel(self) -> None:
        """Lazily create the sentinel physical page (one per pool)."""
        if self._sentinel_handle is not None:
            return
        err, handle_raw = cuMemCreate(self.page_size, self._prop, 0)
        _check_cuda(err, "cuMemCreate(sentinel)")
        self._sentinel_handle = int(handle_raw)
        logger.info("VMMPagePool: sentinel page created (handle=0x%x)",
                     self._sentinel_handle)

    def _map_sentinel_to_va(self, va: int) -> None:
        """Map the sentinel page to a VA (must be currently unmapped)."""
        self._ensure_sentinel()
        err, = cuMemMap(va, self.page_size, 0, self._sentinel_handle, 0)
        _check_cuda(err, f"cuMemMap(sentinel, va=0x{va:x})")
        err, = cuMemSetAccess(va, self.page_size, [self._access_desc], 1)
        _check_cuda(err, f"cuMemSetAccess(sentinel, va=0x{va:x})")
        self._sentinel_vas.add(va)

    def _unmap_sentinel_from_va(self, va: int) -> None:
        """Unmap sentinel from a VA (before mapping a real page there)."""
        if va not in self._sentinel_vas:
            return
        err, = cuMemUnmap(va, self.page_size)
        _check_cuda(err, f"cuMemUnmap(sentinel, va=0x{va:x})")
        self._sentinel_vas.discard(va)

    # =====================================================================
    # Expert region
    # =====================================================================

    def get_expert_va(self, layer: int, slot: int) -> int:
        """Return the VA start address for an expert slot.

        Uses group-aligned VA layout: each layer gets
        ceil(max_slots/group_size) * group_bytes of VA.
        Within a layer, expert data is contiguous at expert_slot_bytes
        stride (no padding gaps within groups).
        """
        num_groups_per_layer = math.ceil(
            self.max_slots_per_layer / self.group_size)
        per_layer_va = num_groups_per_layer * self.group_bytes
        return (
            self._expert_va_base
            + layer * per_layer_va
            + slot * self.expert_slot_bytes
        )

    def _map_expert_slot_internal(self, layer: int, slot: int) -> None:
        """Map free pages for the group containing ``slot``.

        Group-aware with split-layout VA: maps ``group_pages`` pages to
        the correct non-contiguous VA offsets in the [w13][w2] layout.
        """
        group_idx = slot // self.group_size
        layer_va = self.get_expert_va(layer, 0)
        va_offsets = self._get_group_va_offsets(group_idx)
        for offset, va_off in enumerate(va_offsets):
            if not self._free_pages:
                raise RuntimeError(
                    f"No free pages to map expert group ({layer}, {group_idx})"
                )
            page_id = self._free_pages.pop()
            va = layer_va + va_off
            # Remove sentinel mapping if present (Phase C grow-back)
            self._unmap_sentinel_from_va(va)
            self._map_page_to_va(page_id, va)
            page = self._pages[page_id]
            page.state = PageState.EXPERT
            page.expert_layer = layer
            page.expert_slot = group_idx
            page.expert_page_offset = offset
            self._expert_pages[(layer, group_idx, offset)] = page_id

    def map_expert_slot(self, layer: int, slot: int) -> None:
        """Public API: map an expert slot from free pages."""
        self._map_expert_slot_internal(layer, slot)

    def unmap_expert_slot(self, layer: int, slot: int) -> List[int]:
        """Unmap all pages of an expert slot → FREE. Returns freed page_ids.

        Group-aware: converts expert slot → group_idx, unmaps all
        group_pages for that group.
        """
        freed: List[int] = []
        group_idx = slot // self.group_size
        for offset in range(self.group_pages):
            key = (layer, group_idx, offset)
            page_id = self._expert_pages.pop(key, None)
            if page_id is None:
                continue
            va = self._pages[page_id].mapped_va
            self._unmap_page(page_id)
            self._free_pages.append(page_id)
            freed.append(page_id)
            # Map sentinel so stacked weight tensor stays accessible
            if self._phase_c_mode and va != 0:
                self._map_sentinel_to_va(va)
        return freed

    def is_expert_slot_mapped(self, layer: int, slot: int) -> bool:
        """Check if the group containing this expert slot is mapped."""
        group_idx = slot // self.group_size
        return (layer, group_idx, 0) in self._expert_pages

    def get_expert_tensor(
        self,
        layer: int,
        slot: int,
        w13_shape: tuple,
        w2_shape: tuple,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create PyTorch tensor views into a mapped expert slot's VA.

        Layout within each slot: [w13_data | w2_data | padding]
        The padding (if any) fills the slot up to ``slot_aligned_bytes``.
        """
        va = self.get_expert_va(layer, slot)
        elem_size = torch.tensor([], dtype=dtype).element_size()

        w13_numel = 1
        for d in w13_shape:
            w13_numel *= d
        w13_bytes = w13_numel * elem_size

        w13 = vmm_ptr_to_tensor(va, w13_shape, dtype, self.device_id)
        w2 = vmm_ptr_to_tensor(va + w13_bytes, w2_shape, dtype, self.device_id)
        return w13, w2

    def get_expert_layer_tensors(
        self,
        layer: int,
        num_slots: int,
        w13_per_expert: tuple,
        w2_per_expert: tuple,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create contiguous [num_slots, *shape] tensors for an entire layer.

        Returns standard contiguous tensors backed by VMM VA — fully
        compatible with torch.compile / inductor / CUDA graphs.

        Layout in VMM VA:
            [w13 block: num_slots × w13_per_expert] [w2 block: num_slots × w2_per_expert]
        Both blocks are contiguous (no padding between slots).

        Returns: (w13_stacked, w2_stacked) both [num_slots, *per_expert_shape]
        """
        layer_va = self.get_expert_va(layer, 0)
        elem_size = torch.tensor([], dtype=dtype).element_size()

        w13_numel_per = 1
        for d in w13_per_expert:
            w13_numel_per *= d
        w2_numel_per = 1
        for d in w2_per_expert:
            w2_numel_per *= d

        w13_total_bytes = num_slots * w13_numel_per * elem_size
        w2_total_bytes = num_slots * w2_numel_per * elem_size

        # w13: contiguous block at layer_va
        w13 = vmm_ptr_to_tensor(
            layer_va,
            (num_slots, *w13_per_expert),
            dtype, self.device_id,
        )
        # w2: contiguous block immediately after w13
        w2 = vmm_ptr_to_tensor(
            layer_va + w13_total_bytes,
            (num_slots, *w2_per_expert),
            dtype, self.device_id,
        )
        return w13, w2

    # =====================================================================
    # KV region
    # =====================================================================

    def map_kv_pages(self, n_pages: int) -> List[Tuple[int, int]]:
        """Map ``n_pages`` free pages into KV VA.

        Reuses previously freed KV VA offsets before appending new ones,
        preventing VA space exhaustion during long-running sessions.

        Returns: [(kv_va, page_id), ...]
        """
        result: List[Tuple[int, int]] = []
        for _ in range(n_pages):
            if not self._free_pages:
                break
            # Prefer reusing freed KV VA offsets
            if self._kv_free_offsets:
                va_offset = self._kv_free_offsets.pop()
            elif self._kv_next_va_offset + self.page_size <= self._kv_va_size:
                va_offset = self._kv_next_va_offset
                self._kv_next_va_offset += self.page_size
            else:
                logger.warning("KV VA exhausted, cannot map more KV pages")
                break
            page_id = self._free_pages.pop()
            va = self._kv_va_base + va_offset
            self._map_page_to_va(page_id, va)
            page = self._pages[page_id]
            page.state = PageState.KV
            page.kv_va_offset = va_offset
            self._kv_pages[va_offset] = page_id
            result.append((va, page_id))
            self._kv_mapped_count += 1
        return result

    def unmap_kv_pages(self, kv_va_offsets: List[int]) -> List[int]:
        """Unmap specific KV pages → FREE. Recycles VA offsets for reuse."""
        freed: List[int] = []
        for offset in kv_va_offsets:
            page_id = self._kv_pages.pop(offset, None)
            if page_id is None:
                continue
            self._unmap_page(page_id)
            self._free_pages.append(page_id)
            self._kv_free_offsets.append(offset)  # recycle VA offset
            self._kv_mapped_count -= 1
            freed.append(page_id)
        return freed

    # =====================================================================
    # Phase C: VMM-backed KV cache tensors
    # =====================================================================

    def allocate_kv_tensor(
        self, tensor_idx: int, initial_size: int, max_size: int,
    ) -> torch.Tensor:
        """Create a VMM-backed KV cache tensor.

        VA is reserved for max_size but only initial_size bytes get physical
        pages.  block_pool.num_gpu_blocks limits access range, so unmapped
        VA beyond initial pages is never touched.

        Args:
            tensor_idx: index of the KVCacheTensor in the config
            initial_size: bytes to back with physical pages now
            max_size: total VA reservation (for future expert→KV expansion)

        Returns:
            torch.Tensor backed by VMM VA (dtype=int8, shape=(max_size,))
        """
        # VA bound check: ensure we don't exceed reserved KV VA region
        aligned_max_check = _round_up(max_size, self.page_size)
        if self._kv_va_allocated + aligned_max_check > self._kv_va_size:
            raise RuntimeError(
                f"KV VA exhausted: need {self._kv_va_allocated + max_size} "
                f"bytes but only {self._kv_va_size} reserved. "
                f"Increase kv_max_bytes at pool creation."
            )

        # Round up VA allocation to page boundary so the next tensor's
        # VA base is always page-aligned (cuMemMap requirement).
        aligned_max = _round_up(max_size, self.page_size)
        va_base = self._kv_va_base + self._kv_va_allocated
        self._kv_va_allocated += aligned_max

        n_initial_pages = math.ceil(initial_size / self.page_size)
        new_ids = self._create_phys_pages(n_initial_pages)
        for i, page_id in enumerate(new_ids):
            va = va_base + i * self.page_size
            self._map_page_to_va(page_id, va)
            page = self._pages[page_id]
            page.state = PageState.KV
            page.kv_va_offset = (va_base - self._kv_va_base) + i * self.page_size
            self._kv_pages[page.kv_va_offset] = page_id
            self._kv_mapped_count += 1

        self._kv_tensors[tensor_idx] = {
            'va_base': va_base,
            'initial_size': initial_size,
            'max_size': max_size,
            'mapped_bytes': n_initial_pages * self.page_size,
        }

        # HIGH-2 fix: advance _kv_next_va_offset past allocated region so
        # transfer_expert_to_kv() (Phase A path) never collides with
        # KV tensor pages.
        self._kv_next_va_offset = max(
            self._kv_next_va_offset, self._kv_va_allocated)

        from vllm.vmm_pool import vmm_ptr_to_tensor
        return vmm_ptr_to_tensor(va_base, (max_size,), torch.int8, self.device_id)

    def total_expert_mapped_bytes(self) -> int:
        """Total bytes currently mapped in the expert VA region."""
        return len(self._expert_pages) * self.page_size

    # =====================================================================
    # Phase C: expert unmap + KV physical expansion
    # =====================================================================

    def unmap_expert_slots(
        self, victims: List[Tuple[int, int]],
    ) -> int:
        """Unmap expert slots and return pages to free pool.

        Group-aware: converts expert slot → group_idx, unmaps all
        group_pages for that group. Deduplicates groups so each group
        is unmapped at most once (e.g., if both experts 2 and 3 are
        victims, group 1 is unmapped once, not twice).

        Args:
            victims: [(layer, slot), ...] expert slots to sacrifice.

        Returns:
            Number of pages freed.
        """
        self.ensure_safe_remap()
        freed = 0
        # Deduplicate by group: (layer, group_idx) → first seen
        seen_groups: Set[Tuple[int, int]] = set()
        for layer, slot in victims:
            group_idx = slot // self.group_size
            group_key = (layer, group_idx)
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            for offset in range(self.group_pages):
                key = (layer, group_idx, offset)
                page_id = self._expert_pages.pop(key, None)
                if page_id is None:
                    continue
                va = self._pages[page_id].mapped_va
                self._unmap_page(page_id)
                self._free_pages.append(page_id)
                freed += 1
                # Map sentinel so the stacked weight tensor VA stays
                # accessible (Triton validates pointers in eager mode).
                if self._phase_c_mode and va != 0:
                    self._map_sentinel_to_va(va)
        return freed

    def expand_kv_physical_pages(
        self,
        n_new_blocks: int,
        per_tensor_block_bytes: Dict[int, int],
    ) -> int:
        """Map free pages at correct sequential KV tensor VA offsets.

        Transactional: pre-computes total pages needed across all tensors,
        clamps ``n_new_blocks`` to what free pages and VA space can support,
        then maps.  This guarantees every returned block is fully backed
        in ALL tensors (no partial mapping / leaked state).

        Args:
            n_new_blocks: number of new KV blocks requested.
            per_tensor_block_bytes: {tensor_idx: bytes_per_block} for each
                KV tensor.

        Returns:
            Actual number of blocks fully backed across ALL tensors.
        """
        self.ensure_safe_remap()

        if n_new_blocks <= 0 or not self._kv_tensors:
            return 0

        # ── Phase 1: compute per-tensor pages needed and clamp ──────────
        # For each tensor, pages_for_n_blocks = ceil(n * block_bytes / page_size).
        # Also clamp by remaining VA space (max_size - mapped_bytes).
        per_tensor_pages: Dict[int, int] = {}   # tensor_idx → pages needed
        max_blocks_by_va = n_new_blocks  # min across tensors by VA limit

        for tensor_idx in sorted(self._kv_tensors.keys()):
            info = self._kv_tensors[tensor_idx]
            block_bytes = per_tensor_block_bytes.get(tensor_idx, 0)
            if block_bytes <= 0:
                continue

            va_remaining = info['max_size'] - info['mapped_bytes']
            blocks_by_va = va_remaining // block_bytes
            if blocks_by_va < max_blocks_by_va:
                max_blocks_by_va = blocks_by_va

        n_new_blocks = min(n_new_blocks, max_blocks_by_va)
        if n_new_blocks <= 0:
            return 0

        # Compute total pages needed across all tensors
        total_pages_needed = 0
        for tensor_idx in sorted(self._kv_tensors.keys()):
            block_bytes = per_tensor_block_bytes.get(tensor_idx, 0)
            if block_bytes <= 0:
                continue
            pages = math.ceil(n_new_blocks * block_bytes / self.page_size)
            per_tensor_pages[tensor_idx] = pages
            total_pages_needed += pages

        # Clamp by available free pages — use batched binary search
        available = len(self._free_pages)
        if total_pages_needed > available:
            n_new_blocks = max_blocks_for_pages(
                available, per_tensor_block_bytes, self.page_size)
            if n_new_blocks <= 0:
                return 0
            # Recompute per-tensor pages with clamped block count
            total_pages_needed = 0
            for tidx in per_tensor_pages:
                block_bytes = per_tensor_block_bytes[tidx]
                pages = math.ceil(n_new_blocks * block_bytes / self.page_size)
                per_tensor_pages[tidx] = pages
                total_pages_needed += pages

        # ── Phase 2: map pages (guaranteed to succeed) ──────────────────
        for tensor_idx in sorted(per_tensor_pages.keys()):
            info = self._kv_tensors[tensor_idx]
            n_pages = per_tensor_pages[tensor_idx]

            for _ in range(n_pages):
                page_id = self._free_pages.pop()
                va = info['va_base'] + info['mapped_bytes']
                self._map_page_to_va(page_id, va)
                page = self._pages[page_id]
                page.state = PageState.KV
                kv_offset = (
                    info['va_base'] - self._kv_va_base + info['mapped_bytes']
                )
                page.kv_va_offset = kv_offset
                self._kv_pages[kv_offset] = page_id
                self._kv_mapped_count += 1
                info['mapped_bytes'] += self.page_size

        if n_new_blocks > 0:
            logger.info(
                "expand_kv_physical_pages: %d blocks fully backed "
                "(%d pages mapped)", n_new_blocks, total_pages_needed,
            )
        return n_new_blocks

    def contract_kv_physical_pages(
        self,
        n_blocks: int,
        per_tensor_block_bytes: Dict[int, int],
    ) -> int:
        """Undo expand_kv_physical_pages: unmap the last n_blocks of KV pages.

        Used for rollback when multi-rank commit diverges. Pages go back
        to the free list (NOT to expert — one-way MVP still holds).

        Returns actual number of blocks rolled back.
        """
        if n_blocks <= 0 or not self._kv_tensors:
            return 0

        # For each tensor, unmap the last N pages (most recently mapped)
        for tensor_idx in sorted(self._kv_tensors.keys(), reverse=True):
            info = self._kv_tensors[tensor_idx]
            block_bytes = per_tensor_block_bytes.get(tensor_idx, 0)
            if block_bytes <= 0:
                continue

            pages_to_unmap = math.ceil(n_blocks * block_bytes / self.page_size)

            for _ in range(pages_to_unmap):
                if info['mapped_bytes'] <= 0:
                    break
                # Walk back: unmap the highest-offset KV page for this tensor
                info['mapped_bytes'] -= self.page_size
                va = info['va_base'] + info['mapped_bytes']
                kv_offset = info['va_base'] - self._kv_va_base + info['mapped_bytes']
                page_id = self._kv_pages.pop(kv_offset, None)
                if page_id is not None:
                    self._unmap_page(page_id)
                    self._free_pages.append(page_id)
                    self._kv_mapped_count -= 1

        logger.info(
            "contract_kv_physical_pages: rolled back %d blocks", n_blocks)
        return n_blocks

    # =====================================================================
    # Transfer operations (Expert ↔ KV)
    # =====================================================================

    def transfer_expert_to_kv(
        self,
        victims: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Unmap expert slots and remap their pages as KV pages.

        Calls ``ensure_safe_remap()`` to synchronize all GPU work before
        touching VA mappings.

        Args:
            victims: [(layer, slot), ...] expert slots to sacrifice

        Returns: [(kv_va, page_id), ...] newly mapped KV pages
        """
        self.ensure_safe_remap()
        new_kv: List[Tuple[int, int]] = []
        seen_groups: Set[Tuple[int, int]] = set()
        for layer, slot in victims:
            group_idx = slot // self.group_size
            group_key = (layer, group_idx)
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            for offset in range(self.group_pages):
                key = (layer, group_idx, offset)
                page_id = self._expert_pages.pop(key, None)
                if page_id is None:
                    continue
                # Unmap from expert VA
                self._unmap_page(page_id)
                # Remap into KV VA (reuse freed offsets first)
                if self._kv_free_offsets:
                    kv_offset = self._kv_free_offsets.pop()
                elif self._kv_next_va_offset + self.page_size <= self._kv_va_size:
                    kv_offset = self._kv_next_va_offset
                    self._kv_next_va_offset += self.page_size
                else:
                    # KV VA exhausted — return page to free pool
                    self._free_pages.append(page_id)
                    logger.warning("KV VA exhausted during expert→KV transfer")
                    continue
                kv_va = self._kv_va_base + kv_offset
                self._map_page_to_va(page_id, kv_va)
                page = self._pages[page_id]
                page.state = PageState.KV
                page.kv_va_offset = kv_offset
                self._kv_pages[kv_offset] = page_id
                new_kv.append((kv_va, page_id))
                self._kv_mapped_count += 1
        return new_kv

    def transfer_kv_to_expert(self, *args, **kwargs):
        """MVP: reverse path disabled. KV → expert transfer not supported."""
        raise NotImplementedError(
            "MVP: reverse path disabled. KV → expert transfer not supported."
        )

    def map_free_pages_to_experts(self, *args, **kwargs):
        """MVP: reverse path disabled. Expert recovery not supported."""
        raise NotImplementedError(
            "MVP: reverse path disabled. Expert recovery not supported."
        )

    # =====================================================================
    # Queries & stats
    # =====================================================================

    @property
    def num_free_pages(self) -> int:
        return len(self._free_pages)

    @property
    def num_expert_pages(self) -> int:
        return len(self._expert_pages)

    @property
    def num_kv_pages(self) -> int:
        return self._kv_mapped_count

    @property
    def total_pages(self) -> int:
        return len(self._pages)

    def get_stats(self) -> Dict:
        stats = {
            "total_pages": self.total_pages,
            "free_pages": self.num_free_pages,
            "expert_pages": self.num_expert_pages,
            "kv_pages": self.num_kv_pages,
            "page_size_mb": self.page_size / (1 << 20),
            "expert_mb": self.num_expert_pages * self.page_size / (1 << 20),
            "kv_mb": self.num_kv_pages * self.page_size / (1 << 20),
            "free_mb": self.num_free_pages * self.page_size / (1 << 20),
            "slot_pages": self.slot_pages,
            "slot_aligned_bytes": self.slot_aligned_bytes,
        }
        if self._phase_c_mode:
            stats["phase_c_mode"] = True
            stats["kv_tensors_mapped_mb"] = sum(
                info['mapped_bytes'] / (1 << 20)
                for info in self._kv_tensors.values()
            )
            stats["total_sync_us"] = self._total_sync_us
            stats["sentinel_vas"] = len(self._sentinel_vas)
        return stats

    # =====================================================================
    # Cleanup
    # =====================================================================

    def destroy(self) -> None:
        """Release all VMM resources (unmap, release handles, free VA)."""
        # 1. Unmap all pages that are still mapped
        for page in self._pages:
            if page.mapped_va != 0:
                try:
                    cuMemUnmap(page.mapped_va, self.page_size)
                except Exception:
                    pass
                page.mapped_va = 0

        # 2. Release physical handles
        for page in self._pages:
            try:
                cuMemRelease(page.handle)
            except Exception:
                pass

        # 3. Free VA reservations
        if self._expert_va_base:
            try:
                cuMemAddressFree(self._expert_va_base, self._expert_va_size)
            except Exception:
                pass
            self._expert_va_base = 0

        if self._kv_va_base:
            try:
                cuMemAddressFree(self._kv_va_base, self._kv_va_size)
            except Exception:
                pass
            self._kv_va_base = 0

        self._pages.clear()
        self._free_pages.clear()
        self._expert_pages.clear()
        self._kv_pages.clear()

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass
