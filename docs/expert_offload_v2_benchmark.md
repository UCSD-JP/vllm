# Expert Offload v2 — Benchmark Results

## Setup

- **Hardware**: Paladin 2×H100 SXM (93GB each), PCIe topology (NOT NVLink)
- **Model**: Qwen3-Next-80B-A3B-Instruct, TP=2, BF16
- **Config**: `VLLM_EXPERT_OFFLOAD_ENABLE=1`, `VLLM_EXPERT_MAX_RESIDENT=400`
- **MoE**: 512 global experts, 48 layers, top-10 routing
- **Per-expert size**: w13=2.097MB + w2=1.049MB = 3.146MB (BF16, TP2-sharded)
- **vLLM**: v0.15.1, CUDA graph mode (enforce_eager=false)

## Evolution: v5 (staging) → v6 (pinned pool)

### Architecture Change

| Component | v5 | v6 |
|-----------|----|----|
| CPU pool | Pageable memory | **Pinned memory** |
| DMA path | CPU pageable → pinned staging → GPU | CPU pinned → GPU **(direct)** |
| Copies per expert | 4 (2× staging + 2× DMA) | **2** (direct DMA only) |

### Performance Comparison

| Metric | v5 (staging) | v6 (pinned) | Improvement |
|--------|:---:|:---:|:---:|
| t_pre_step_total | 12,640 us | **6,399 us** | **-49%** |
| t_fetch | ~6,500 us | **153 us** | **-98%** |
| t_classify | ~4,000 us | 2,001 us | -50% |
| eff_bw | 2.1-2.6 GB/s | **10.2 GB/s** | **4.3×** |
| t_deferred_sync | 4.2 us | 3.5 us | ~same |
| TPOT b=1 out=128 | 10.2ms | **7.3ms** | **-28%** |
| hit_rate | 99.1% | 99.9% | +0.8pp |

**Note**: eff_bw 10.2 GB/s is still 32% of theoretical PCIe Gen4 (32 GB/s).
Remaining gap is due to small per-expert copies (3.15MB × ~5 experts = 10 individual DMA calls).

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

## Expert Cache Detailed Timing (v6, steady state)

| Phase | Time (us/call) | % of total | Description |
|-------|:-:|:-:|---|
| t_pre_step_total | 6,399 | 100% | Total pre_step wall-clock |
| t_gpu_gather (A) | 969 | 15.1% | GPU→GPU routing gather |
| t_d2h (B) | 364 | 5.7% | Bulk D2H (single sync) |
| t_classify (C) | 2,001 | 31.3% | CPU hit/miss classification |
| t_cache_map (C2) | ~300 | 4.7% | Batched cache_map upload |
| t_fetch | 153 | 2.4% | Async DMA queueing |
| t_deferred_sync | 3.5 | 0.05% | Previous DMA completion |
| **unaccounted** | ~2,609 | 40.8% | Python overhead, GIL, etc. |

**Bottleneck**: t_classify (2ms) + unaccounted overhead (2.6ms) = 72% of pre_step.
DMA itself is negligible (153us + 3.5us sync).

## Expert Gating Distribution (Layer 0, 500 steps)

| Metric | Value |
|--------|-------|
| Entropy | 8.02 / 9.00 bits (89.2% of uniform) |
| Active experts (>0.1% routing share) | 269 / 512 |
| Active experts (>1% routing share) | 5 / 512 |
| Top expert (#474) | 1.56% |
| Top-5 range | 1.16% – 1.56% |

**Interpretation**: Near-uniform routing across 269 active experts means the cache
needs ~270 slots to avoid misses in steady state. With max_res=400, this provides
comfortable headroom → 99.9% hit rate.

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

### 1. Contiguous CPU Pool for Bulk DMA
Current: per-expert individual pinned tensors (3.15MB each, scattered allocations).
Proposed: single contiguous pinned buffer per layer `[local_E, ...]`.
Expected: fewer DMA calls, better PCIe utilization (target: 20+ GB/s).

### 2. pre_step Python Overhead (2.6ms unaccounted)
The ~40% unaccounted time suggests Python/GIL overhead in the hot loop.
Options: C++ extension for classify phase, or torch.compile on classification.

### 3. t_classify Optimization (2.0ms)
Classification iterates over 48 layers × ~10 needed experts.
Vectorized numpy/torch operations could reduce this.
