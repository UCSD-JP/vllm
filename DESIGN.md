# Elastic KV: Dynamic Expert ↔ KV Cache Reallocation

> vLLM 0.15.1 fork · Branch: `elastic-kv-mvp`

## Overview

Elastic KV enables **dynamic reallocation** between MoE expert weight memory and KV cache memory at runtime. When KV cache pressure rises (long-context / high-concurrency), expert weight slots are evicted to GPU virtual memory pages that are then mapped as KV cache blocks. When KV pressure drops, the system can expand expert capacity back.

**Key insight**: MoE models (e.g., Qwen3-Next-80B-A3B with 512 experts/GPU) keep all experts resident on GPU but only route to ~8 per token. The unused expert memory can be dynamically repurposed for KV cache.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Scheduler                       │
│  prefix_protect.py: Ce/Cc/Cp/Cb cost model       │
│  scheduler.py: can_fully_protect → expand/reclaim │
├────────────────────┬────────────────────────────┤
│   GPU Worker       │     ExpertCacheManager      │
│  gpu_worker.py     │     expert_cache.py          │
│  - init offload    │     - slot management        │
│  - set_moe_layers  │     - shrink_for_pages       │
│  - topology wire   │     - count_evictable_groups  │
│                    │     - double-buffer prefetch   │
├────────────────────┴────────────────────────────┤
│  Forward Hot Path (fused_moe / layer / modular)  │
│  - cache_map lookup (no .item() sync)            │
│  - scratch bank consume + prefetch_next          │
│  - routing snapshot capture                      │
└─────────────────────────────────────────────────┘
```

### File Map (18 modified files)

| Layer | File | Role |
|-------|------|------|
| Config | `vllm/config/compilation.py` | Compilation config for custom ops |
| Scheduler | `vllm/v1/core/sched/scheduler.py` | Elastic KV prepare/commit, PP integration |
| Scheduler | `vllm/v1/core/prefix_protect.py` | V4.2 cost model (Ce/Cc/Cp/Cb) |
| Worker | `vllm/v1/worker/gpu_worker.py` | Expert offload init, topology wiring |
| Worker | `vllm/v1/worker/gpu_model_runner.py` | Step profiler, prefillish flag |
| Cache | `vllm/model_executor/layers/fused_moe/expert_cache.py` | Core: slot mgmt, LRU, shrink/expand, CB/SB |
| Forward | `vllm/model_executor/layers/fused_moe/fused_moe.py` | Routing snapshot, scratch kernel dispatch |
| Forward | `vllm/model_executor/layers/fused_moe/layer.py` | FusedMoE layer with cache_map wiring |
| Forward | `vllm/model_executor/layers/fused_moe/modular_kernel.py` | Modular kernel scratch support |
| Forward | `vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py` | forward_cuda: bank wait, prefetch_next dispatch |
| Model | `vllm/model_executor/models/qwen3_next.py` | Qwen3-Next offload-aware weight init |
| Test | `tests/v1/core/test_prefix_protection.py` | PP cost model unit tests |

## Core Data Structures

### ExpertCacheManager (`expert_cache.py`)

Central manager for expert weight offloading. One instance per GPU worker.

```python
# Per-layer slot tracking
_expert_to_slot[layer][lid] → slot_idx  # -1 = evicted
_slot_to_expert[layer][slot] → lid      # -1 = empty
_last_access[layer, lid] → step         # LRU timestamp

# CPU backing store (pinned memory for async DMA)
_cpu_pool[layer][lid] → (w13_cpu, w2_cpu)

# VMM pool integration
_vmm_pool: VMMExpertPool  # virtual memory manager for page-level alloc

# Double-buffer scratch banks (for non-resident expert serving)
_scratch_banks: List[_ScratchBank]  # 2 banks, alternating fill/consume
_scratch_capacity: int              # max experts per bank
_scratch_threshold: int             # slot index where scratch starts
```

### _ScratchBank

Double-buffer state machine for overlapping H2D copy with compute:

```
IDLE → FILLING → READY → IN_USE → IDLE
         ↑                          │
         └──────────────────────────┘

IDLE:     Bank available for new prefetch
FILLING:  H2D copy in progress on copy_stream
READY:    ready_event recorded, awaiting consume by compute stream
IN_USE:   Compute stream consuming weights (done_event pending)
```

### Prefix Protection Cost Model (`prefix_protect.py`)

4-way argmin decision for block allocation:

| Cost | Formula | When cheapest |
|------|---------|---------------|
| **Ce** (expert evict) | `top_k × ρ × c_reload × h_eff × 1000` | Low KV pressure, many evictable experts |
| **Cc** (cache reclaim) | `p_reuse × (blocks × block_size × t_prefill + overhead)` | Cold cached blocks available |
| **Cp** (preempt) | `t_sched + t_queue + tail_tokens × t_prefill` | Long-running request can be preempted |
| **Cb** (partial expand) | Floor expand + reclaim remainder | Partial expand cheaper than full |

## Offload Modes

### 1. Default: LRU Demand-Driven

- **Init**: All experts resident (`max_resident = local_E`)
- **Shrink trigger**: KV cache pressure → scheduler calls `shrink_for_pages()`
- **Eviction**: LRU scoring with greedy per-layer floor guard
- **Forward**: Sync fetch on cache miss (`.item()` required — breaks CUDA graph)

### 2. Fixed Tail (`VLLM_FIXED_TAIL=1`)

- Static non-resident set determined at init
- Double-buffer prefetch (no `.item()` on hot path)
- CUDA graph compatible
- No topology changes after activation

### 3. Step-Boundary (`VLLM_STEP_BOUNDARY=1`) — **Plan B**

Per-step topology recompute at step boundaries (between scheduler steps), not during forward.

**Flow per step:**
```
pre_step()
  ├── _update_lru_from_snapshots()    # batch LRU from routing snapshots
  ├── _recompute_step_topology()      # scan non-resident → _sb_tail_lids
  ├── _patch_step_cache_maps()        # update cache_map with scratch slots
  └── _start_initial_prefetch()       # prefetch first tail layer → bank 0

forward_cuda(layer_idx)
  ├── wait_event(active_bank.ready_event)
  ├── step_prefetch_next(layer_idx)   # arm next layer's tail → alt bank
  └── kernel(w13, w2, cache_map)
```

**Duplicate arm guard** (bugfix): `step_prefetch_next()` checks all banks before arming to prevent `_start_initial_prefetch()` + first `step_prefetch_next(0)` from double-arming L1:

```python
# If next layer is already armed on any bank, skip duplicate prefetch
for b in self._scratch_banks:
    if (b.owner_layer_idx == next_idx
            and b.state in (_BankState.READY, _BankState.FILLING)):
        return
```

### 4. Cutoff-Boundary (`VLLM_CUTOFF_BOUNDARY=1`) — **Plan A**

Uniform suffix-cut eviction. Single integer `_resident_cutoff` replaces per-layer LRU scoring.

**Invariant**: `lids [0, cutoff) = resident`, `lids [cutoff, local_E) = scratch (non-resident)`.

L0 is always fully resident (no eviction). L1..L47 share the same cutoff.

**Shrink**: Move cutoff left by `group_size` steps → evict highest-lid experts uniformly across L1..L47.

**Expand**: Move cutoff right → reclaim experts from CPU pool.

**Lazy verification** (`_cutoff_verify_once()`): Three preconditions checked once on first dispatch:
1. Identity: `lid == slot` for all layers (no slot shuffling)
2. Pinned experts below shrink floor
3. `cpu_pool` completeness for L1..L47

Called from `count_evictable_groups()`, `shrink_for_pages()`, and `pre_step()`. If any check fails, permanently falls back to LRU path.

**Cache map rebuild**: Full deterministic rebuild from cutoff value — no incremental patching needed.

## Environment Variables

### Feature Flags
| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_EXPERT_OFFLOAD_ENABLE` | `0` | Enable expert offloading |
| `VLLM_VMM_EXPERT_POOL` | `0` | Enable VMM-based expert pool |
| `VLLM_FIXED_TAIL` | `0` | Fixed tail mode |
| `VLLM_STEP_BOUNDARY` | `0` | Step-boundary mode (Plan B) |
| `VLLM_CUTOFF_BOUNDARY` | `0` | Cutoff-boundary mode (Plan A) |
| `VLLM_EXPERT_PREWARM` | `0` | Prewarm prefetch |

### Tuning
| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_EXPERT_MAX_RESIDENT` | `512` | Max resident experts per layer |
| `VLLM_SCRATCH_CAPACITY` | `0` | Scratch bank capacity (experts) |
| `VLLM_DECODE_FREEZE` | `0` | Freeze topology during decode |

### Diagnostics
| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_EXPERT_DEBUG` | `0` | Debug logging |
| `VLLM_EXPERT_TRACE` | `0` | JSONL trace output |
| `VLLM_KERNEL_PROFILE` | `0` | Kernel timing profiler |
| `VLLM_EAGER_BREAKDOWN` | `0` | Per-stage timing breakdown |
| `VLLM_STEP_PROFILE` | `0` | Step boundary profiler |

## Deployment

**Critical**: Never use `PYTHONPATH` to dev repo — causes +60% TPOT regression from import resolution overhead.

Always deploy by copying patched files to site-packages:
```bash
SITE=$(python -c "import vllm; print(vllm.__path__[0])")
# Copy all 12 modified vllm/ files:
cp vllm/config/compilation.py                              $SITE/config/
cp vllm/model_executor/layers/fused_moe/expert_cache.py    $SITE/model_executor/layers/fused_moe/
cp vllm/model_executor/layers/fused_moe/fused_moe.py       $SITE/model_executor/layers/fused_moe/
cp vllm/model_executor/layers/fused_moe/layer.py            $SITE/model_executor/layers/fused_moe/
cp vllm/model_executor/layers/fused_moe/modular_kernel.py   $SITE/model_executor/layers/fused_moe/
cp vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py $SITE/model_executor/layers/fused_moe/
cp vllm/model_executor/models/qwen3_next.py                $SITE/model_executor/models/
cp vllm/v1/core/prefix_protect.py                          $SITE/v1/core/
cp vllm/v1/core/sched/scheduler.py                         $SITE/v1/core/sched/
cp vllm/v1/worker/gpu_model_runner.py                      $SITE/v1/worker/
cp vllm/v1/worker/gpu_worker.py                            $SITE/v1/worker/
```

## Known Issues

### `can_fully_protect=False` → expand blocked (pre-existing)

`scheduler.py:546`: `can_fully_protect = remaining_cap >= protection_gap`.
Long-context requests produce `protection_gap > remaining_cap` → `Ce=inf` → expand permanently blocked. This is a pre-existing PP design issue, not caused by CB/SB changes. Under investigation.

### `_phase_c_enabled` never set to True

LRU fallback's `_can_shrink()` returns False when `max_resident >= local_num_experts` and `_phase_c_enabled=False`. This variable is initialized to False and never updated, blocking the LRU shrink path when all experts start resident (demand-driven mode).
