# Expert Offload v2 — Benchmark Results

## Setup

- **Hardware**: Paladin 2×H100 SXM (93GB each), PCIe topology (NOT NVLink)
- **Model**: Qwen3-Next-80B-A3B-Instruct, TP=2, BF16
- **Config**: `VLLM_EXPERT_OFFLOAD_ENABLE=1`, `VLLM_EXPERT_MAX_RESIDENT=400`
- **MoE**: 512 global experts, 48 layers, top-10 routing
- **Per-expert size**: w13=2.097MB + w2=1.049MB = 3.146MB (BF16, TP2-sharded)
- **vLLM**: v0.15.1, CUDA graph mode (enforce_eager=false)

## Evolution: v5 → v6 → v7

### Architecture Changes

| Component | v5 (staging) | v6 (pinned pool) | v7 (pinned scratch) |
|-----------|:---:|:---:|:---:|
| CPU expert pool | Pageable | **Pinned** | Pinned |
| DMA path | pageable → staging → GPU | pinned → GPU (direct) | pinned → GPU (direct) |
| Copies per expert | 4 | **2** | 2 |
| cache_map scratch | N/A | Pageable CPU | **Pinned CPU** |
| cache_map upload | Per-layer sync | Bulk sync | **Bulk non_blocking** |

### Performance Comparison

| Metric | v5 (staging) | v6 (pinned) | v7 (pinned scratch) | v5→v7 |
|--------|:---:|:---:|:---:|:---:|
| t_pre_step_total | 12,640 us | 6,399 us | **3,528 us** | **-72%** |
| t_cache_map_upload | N/A | ~3,100 us | **369 us** | — |
| t_fetch | ~6,500 us | 153 us | **72 us** | **-99%** |
| t_classify | ~4,000 us | 2,001 us | **1,822 us** | **-54%** |
| eff_bw | 2.1-2.6 GB/s | 10.2 GB/s | **10.2 GB/s** | **4×** |
| t_deferred_sync | 4.2 us | 0.7 us | **0.3 us** | ~0 |
| **TPOT b=1 out=128** | **10.2ms** | **7.3ms** | **7.0ms** | **-31%** |
| hit_rate | 99.1% | 99.9% | **100.0%** | +0.9pp |

### v6→v7 Root Cause Analysis

**Problem**: v6 `_stacked_scratch_cpu` was a pageable (non-pinned) tensor.
`copy_()` from pageable CPU → GPU forces the CUDA driver to:
1. Allocate a temporary pinned staging buffer
2. `memcpy` from pageable to staging (CPU)
3. DMA from staging to GPU
4. Free the staging buffer

For a 96KB cache_map tensor, this driver overhead dominated: **3,100us for 96KB = 0.03 GB/s**.

**Fix**: `.pin_memory()` on `_stacked_scratch_cpu` + `non_blocking=True` on all copies.
Result: GPU copies are enqueued instantly (~370us Python loop time), execute on default
stream, and complete before CUDA graph replay (same-stream ordering guarantee).

## Comprehensive Benchmark (v6, pinned pool)

### TPOT by Concurrency and Arrival Pattern

| Input | Output | Conc | Arrival | TPOT p50 | TPOT p95 | TTFT p50 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 128 | 32 | 1 | burst | 7.3ms | 8.5ms | 75ms |
| 128 | 32 | 8 | burst | 8.5ms | 9.3ms | 185ms |
| 128 | 32 | 8 | stagger | 7.3ms | 7.4ms | 48ms |
| 128 | 32 | 32 | burst | 10.8ms | 13.3ms | 383ms |
| 128 | 32 | 32 | stagger | 7.3ms | 7.4ms | 47ms |
| 128 | 128 | 1 | burst | 7.3ms | 7.3ms | 47ms |
| 128 | 128 | 8 | burst | 8.5ms | 8.7ms | 171ms |
| 128 | 128 | 8 | stagger | 7.9ms | 8.4ms | 58ms |
| 128 | 128 | 32 | burst | 10.7ms | 11.7ms | 238ms |
| 128 | 128 | 32 | stagger | 7.9ms | 8.7ms | 58ms |
| 1024 | 32 | 1 | burst | 7.3ms | 7.3ms | 114ms |
| 1024 | 32 | 8 | burst | 8.5ms | 9.0ms | 260ms |
| 1024 | 32 | 8 | stagger | 7.3ms | 7.3ms | 106ms |
| 1024 | 32 | 32 | burst | 10.8ms | 14.9ms | 567ms |
| 1024 | 32 | 32 | stagger | 7.3ms | 7.4ms | 106ms |
| 1024 | 128 | 1 | burst | 7.3ms | 7.3ms | 106ms |
| 1024 | 128 | 8 | burst | 8.5ms | 8.7ms | 255ms |
| 1024 | 128 | 8 | stagger | 7.9ms | 8.5ms | 119ms |
| 1024 | 128 | 32 | burst | 10.8ms | 11.8ms | 521ms |
| 1024 | 128 | 32 | stagger | 8.1ms | 8.7ms | 117ms |

### Key Observations

1. **TPOT insensitive to input_len**: in=128 vs in=1024 → identical 7.3ms at c=1.
   Expert routing patterns are similar regardless of prompt length.

2. **Concurrency is the dominant factor**: c=1 → c=32 burst increases TPOT by +48% (7.3→10.8ms).
   This is due to batched MoE routing hitting more diverse experts per step.

3. **Staggered >> burst arrival**: At c=32, burst TPOT=10.8ms vs staggered=7.9ms (-27%).
   Burst causes all prefills to overlap, leading to high TTFT (567ms vs 106ms)
   and temporarily higher cache miss rates.

4. **p95 close to p50**: Tail latency is well-controlled (typically <15% above p50),
   indicating the expert cache provides consistent performance.

## Expert Cache Detailed Timing

### v7 Breakdown (TP0, steady state, avg over 5200+ calls)

| Phase | Time (us) | % of total | Description |
|-------|:---------:|:----------:|-------------|
| **t_pre_step_total** | **3,528** | **100%** | Total pre_step wall-clock |
| t_gpu_gather (A) | 942 | 26.7% | GPU→GPU routing snapshot gather |
| t_d2h (B) | 366 | 10.4% | Bulk D2H (single stream sync) |
| t_classify (C) | 1,822 | 51.6% | CPU classification loop (48 layers) |
| — cache_map_build | 736 | 20.9% | Per-layer cache_map scratch update |
| — t_fetch | 72 | 2.0% | Async DMA queueing (misses only) |
| — pure classify | ~1,014 | 28.7% | Hit/miss detection, eviction |
| t_cache_map_upload (C2) | 369 | 10.5% | Pinned CPU → GPU bulk + scatter |
| t_deferred_sync | 0.3 | 0.01% | Previous step DMA completion |

### v6 Breakdown (for comparison, avg over 20500 calls)

| Phase | v6 time (us) | v7 time (us) | Change |
|-------|:---:|:---:|:---:|
| t_pre_step_total | 6,399 | **3,528** | **-45%** |
| t_gpu_gather (A) | 881 | 942 | +7% |
| t_d2h (B) | 416 | 366 | -12% |
| t_classify (C) | 2,001 | 1,822 | -9% |
| t_cache_map_upload (C2) | **~3,100** | **369** | **-88%** |
| t_fetch | 153 | 72 | -53% |
| t_deferred_sync | 0.7 | 0.3 | -57% |

**Key insight**: v6's "unaccounted 40%" was actually `t_cache_map_upload` — the pageable
scratch → GPU copy was taking 3.1ms for 96KB due to CUDA driver staging overhead.
After pinning, this dropped to 369us (Python loop overhead only).

### pre_step vs CUDA Graph Overlap

```
v7 timeline (b=1 steady state):
         pre_step (3.5ms)    idle (3.5ms)
├────────────────────────────┼────────────────────────────┤
│  A  │ B │     C      │ C2 │                            │
├─────┴───┴────────────┴────┴────────────────────────────┤
│          CUDA graph replay (~7.0ms)                     │
├─────────────────────────────────────────────────────────┤
                          TPOT ≈ 7.0ms

v6 timeline (b=1 steady state):
         pre_step (6.4ms)
├─────────────────────────────────────────────────────┤
│  A  │ B │   C   │         C2 (3.1ms!)              │
├─────┴───┴───────┴──────────────────────────────────┤
│          CUDA graph replay (~7.3ms)                 │
├─────────────────────────────────────────────────────┤
                          TPOT ≈ 7.3ms
```

**Result**: pre_step (3.5ms) now finishes well before graph replay (7.0ms),
making pre_step fully hidden. Further pre_step optimization won't improve TPOT
at b=1 — the bottleneck is now purely CUDA graph execution.

## Expert Gating Distribution

### Short-term (500 steps, cold start)

| Metric | Value |
|--------|-------|
| Entropy | 8.02 / 9.00 bits (89.2% of uniform) |
| Active experts (>0.1%) | 269 / 512 |
| Active experts (>1%) | 5 / 512 |

### Long-term (20,500 steps, steady state)

| Metric | Value |
|--------|-------|
| Entropy | **4.31 / 9.00 bits (47.9% of uniform)** |
| Active experts (>0.1%) | **21 / 512** |
| Active experts (>1%) | **11 / 512** |
| Top expert (#326) | **9.14%** |
| Top-8 experts combined | **~72.7%** |

**Key finding**: Entropy drops dramatically from 89.2% (cold) to 47.9% (warm).
At steady state, 11 experts handle >1% each, and the top 8 each receive ~9%.
This explains the 100% hit rate with max_res=400 — only ~21 experts are actively
needed, far below the 400 cached slots.

**Implication for contiguous DMA**: Since misses average only 0.2 experts/step
in steady state, contiguous CPU pool layout would primarily help cold start
(~500 steps) and burst arrival scenarios. Steady-state benefit is negligible.

## GPU Memory Profile

| Stage | GPU Usage |
|-------|-----------|
| After model load | 58.53 GiB (compact [400,...] tensors) |
| During offloading (48 layers) | 58.6 GiB (stable, no spike) |
| After CUDA graph capture | ~89 GiB |
| KV cache available | 24.02 GiB (524,416 tokens) |

**Comparison**: Without offload-aware loading, create_weights() would allocate
full [512,...] tensors → ~91 GiB GPU peak → OOM on 93 GiB H100.

## Remaining Optimization Opportunities

### 1. t_classify Vectorization (1.8ms → target <0.5ms)
Classification iterates 48 layers in Python. Each layer does:
`topk_cpu.unique()` → set operations → hit/miss detection → cache_map build.
Potential: batch all 48 layers' topk_ids into a single numpy array,
vectorize the entire classify phase.

### 2. GPU Gather Optimization (0.9ms)
48 GPU→GPU copies of routing snapshots. Could use a custom CUDA kernel
to gather all snapshots in a single kernel launch.

### 3. Contiguous CPU Pool for Cold Start DMA
Current: per-expert individual pinned tensors (3.15MB each, scattered).
Proposed: `[local_E, ...]` contiguous pinned buffer per layer.
Impact: only during cold start / cache churn. Negligible at steady state.
Priority: LOW (99.9%+ hit rate makes this almost irrelevant).
