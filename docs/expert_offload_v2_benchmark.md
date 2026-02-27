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
| t_pre_step_total | 12,640 us | 6,399 us | **3,701 us** | **-71%** |
| t_cache_map_upload | N/A | ~3,100 us | **371 us** | — |
| t_fetch | ~6,500 us | 153 us | **78 us** | **-99%** |
| t_classify | ~4,000 us | 2,001 us | **1,957 us** | **-51%** |
| eff_bw | 2.1-2.6 GB/s | 10.2 GB/s | **10.1 GB/s** | **4×** |
| t_deferred_sync | 4.2 us | 0.7 us | **0.4 us** | ~0 |
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

## Comprehensive Benchmark (v7, pinned scratch)

### TPOT by Concurrency and Arrival Pattern

| Input | Output | Conc | Arrival | TPOT p50 | TPOT p95 | TTFT p50 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 128 | 32 | 1 | burst | 7.0ms | 9.8ms | 1441ms |
| 128 | 32 | 8 | burst | 8.2ms | 8.7ms | 196ms |
| 128 | 32 | 8 | stagger | 7.0ms | 7.1ms | 49ms |
| 128 | 32 | 32 | burst | 10.6ms | 12.9ms | 346ms |
| 128 | 32 | 32 | stagger | 7.0ms | 7.5ms | 61ms |
| 128 | 128 | 1 | burst | 7.0ms | 7.2ms | 48ms |
| 128 | 128 | 8 | burst | 8.2ms | 8.5ms | 177ms |
| 128 | 128 | 8 | stagger | 7.6ms | 8.1ms | 59ms |
| 128 | 128 | 32 | burst | 10.5ms | 11.5ms | 291ms |
| 128 | 128 | 32 | stagger | 7.6ms | 8.6ms | 70ms |
| 1024 | 32 | 1 | burst | 7.0ms | 7.1ms | 121ms |
| 1024 | 32 | 8 | burst | 8.2ms | 9.2ms | 265ms |
| 1024 | 32 | 8 | stagger | 7.0ms | 7.2ms | 108ms |
| 1024 | 32 | 32 | burst | 10.6ms | 14.8ms | 566ms |
| 1024 | 32 | 32 | stagger | 7.0ms | 7.4ms | 113ms |
| 1024 | 128 | 1 | burst | 7.0ms | 7.4ms | 147ms |
| 1024 | 128 | 8 | burst | 8.3ms | 8.8ms | 294ms |
| 1024 | 128 | 8 | stagger | 7.7ms | 8.5ms | 136ms |
| 1024 | 128 | 32 | burst | 10.6ms | 11.7ms | 568ms |
| 1024 | 128 | 32 | stagger | 7.8ms | 8.6ms | 122ms |

### v6 → v7 TPOT Comparison

| Config | v6 | v7 | Change |
|--------|:---:|:---:|:---:|
| c=1, b=1 | 7.3ms | **7.0ms** | -4% |
| c=8 burst | 8.5ms | **8.2ms** | -4% |
| c=8 stagger | 7.3-7.9ms | **7.0-7.6ms** | -4% |
| c=32 burst | 10.8ms | **10.6ms** | -2% |
| c=32 stagger | 7.3-8.1ms | **7.0-7.8ms** | -4% |

Consistent 0.3ms improvement across all configs. Smaller improvement at c=32 burst
because batched MoE routing increases CUDA graph execution time.

### Key Observations

1. **TPOT insensitive to input_len**: in=128 vs in=1024 → identical 7.0ms at c=1.
   Expert routing patterns are similar regardless of prompt length.

2. **Concurrency is the dominant factor**: c=1 → c=32 burst increases TPOT by +51% (7.0→10.6ms).
   This is due to batched MoE routing hitting more diverse experts per step.

3. **Staggered >> burst arrival**: At c=32, burst TPOT=10.6ms vs staggered=7.0ms (-34%).
   Burst causes all prefills to overlap, leading to high TTFT (566ms vs 113ms)
   and temporarily higher cache miss rates.

4. **p95 close to p50**: Tail latency is well-controlled (typically <15% above p50),
   indicating the expert cache provides consistent performance.

5. **First request TTFT anomaly**: c=1 in=128 out=32 shows TTFT=1441ms. This is a
   cold start artifact (first request triggers CUDA graph capture for new batch size).
   Subsequent requests settle to 48-147ms.

## Expert Cache Detailed Timing

### v7 Breakdown (TP0, steady state, avg over 5200+ calls)

| Phase | Time (us) | % of total | Description |
|-------|:---------:|:----------:|-------------|
| **t_pre_step_total** | **3,701** | **100%** | Total pre_step wall-clock |
| t_gpu_gather (A) | 942 | 25.5% | GPU→GPU routing snapshot gather |
| t_d2h (B) | 410 | 11.1% | Bulk D2H (single stream sync) |
| t_classify (C) | 1,957 | 52.9% | CPU classification loop (48 layers) |
| — cache_map_build | 764 | 20.6% | Per-layer cache_map scratch update |
| — t_fetch | 78 | 2.1% | Async DMA queueing (misses only) |
| — pure classify | ~1,115 | 30.1% | Hit/miss detection, eviction |
| t_cache_map_upload (C2) | 371 | 10.0% | Pinned CPU → GPU bulk + scatter |
| t_deferred_sync | 0.4 | 0.01% | Previous step DMA completion |

### v5 → v6 → v7 Phase Comparison

| Phase | v5 | v6 | v7 | v5→v7 |
|-------|:---:|:---:|:---:|:---:|
| t_pre_step_total | 12,640 | 6,399 | **3,701** | **-71%** |
| t_gpu_gather (A) | — | 881 | 942 | — |
| t_d2h (B) | — | 416 | 410 | — |
| t_classify (C) | ~4,000 | 2,001 | 1,957 | -51% |
| t_cache_map_upload (C2) | — | **~3,100** | **371** | — |
| t_fetch | ~6,500 | 153 | 78 | **-99%** |
| t_deferred_sync | 4.2 | 0.7 | 0.4 | -90% |

### pre_step vs CUDA Graph Overlap

```
v7 timeline (b=1 steady state):
         pre_step (3.7ms)    idle (3.3ms)
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

**Result**: pre_step (3.7ms) now finishes well before graph replay (7.0ms),
making pre_step fully hidden. Further pre_step optimization won't improve TPOT
at b=1 — the bottleneck is now purely CUDA graph execution.

## CUDA Graph Replay Time Analysis (~7ms)

TPOT의 병목이 pre_step에서 CUDA graph replay로 이동했으므로,
7ms의 구성을 분석하여 NVLink 등으로 줄일 수 있는지 검토한다.

### TPOT 구성 — 합산이 아닌 병렬(overlap)

```
Step N의 TPOT:

CPU thread:  [sched] [pre_step 3.7ms] [post]       → CPU_total ≈ 4ms
                     ↓ graph launch
GPU default: ────────[graph replay: compute+comm ~5ms]──→
GPU copy_st: ────────────[DMA 0.3ms]──→ (miss expert fetch, if any)
                                              ↓ sync
                                     ←─ TPOT ≈ 7ms ─→

TPOT = max(CPU_total, GPU_graph) + sync_overhead
     = max(4ms, 5ms) + ~2ms overhead
     ≈ 7ms
```

CPU와 GPU는 **병렬 실행**. TPOT는 합산이 아니라 둘 중 긴 쪽이 결정.
Expert DMA (copy_stream)는 둘 다와 병렬이므로 TPOT에 영향 0.

### GPU Graph 내부 구성

Nsight 프로파일링 데이터 (Paladin TP2-FP16, agentic) 기반:

| Component | c=1 비중 | c=8 비중 | 설명 |
|-----------|:---:|:---:|------|
| NCCL AllReduce | ~25% | ~52% | TP AllReduce (48 layers × 2 calls) |
| GEMM (ATTN+FFN+OUT) | ~28% | ~10% | 행렬 곱셈 커널 |
| MoE routing+expert | ~16% | ~23% | Expert selection + gated FFN |
| Attention | ~12% | ~5% | Flash attention 커널 |
| H2D copy / host prep | ~8% | ~5% | 스케줄러→GPU 데이터 전달 |
| GPU launch gaps | ~5% | ~2% | 커널 간 idle (CUDA graph에서 최소화) |

**핵심**: c=1에서 comm 비중 ~25%, c≥8에서 **52%로 급증**. 고 concurrency에서 comm이 지배.

### Miss Expert Copy 시간

| 항목 | 시간 | 실행 위치 | TPOT 영향 |
|------|:---:|:---:|:---:|
| t_fetch (DMA 큐잉) | 78 us/step avg | CPU (pre_step 내) | **0** — pre_step이 graph에 숨겨짐 |
| 실제 DMA 전송 | ~312 us/expert | GPU copy_stream | **0** — graph replay와 병렬 |
| t_deferred_sync | 0.4 us/step | CPU (다음 step 시작) | **0** — DMA가 graph 중 이미 완료 |

Miss expert copy는 **3중 overlap** 구조:
1. DMA 큐잉(78us)은 pre_step 내에서 발생 → pre_step이 graph에 숨겨짐
2. 실제 DMA(312us/expert)는 copy_stream → default stream의 graph(7ms)와 병렬
3. 다음 step에서 확인 시(0.4us) 이미 완료

TPOT에 영향을 주려면 한 step에서 **~20+ experts miss** 필요 (312us × 20 = 6.2ms > graph 5ms).
실측 miss = 0.2 experts/step이므로 해당 없음.

### AllReduce 데이터량 (TP2, per decode token)

| Batch | ATTN AR | FFN AR | OUT AR | Total | PCIe 시간 | NVLink 시간 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 4KB | 4KB | 296KB | 0.3MB | 0.04ms | 0.002ms |
| 8 | 32KB | 32KB | 2.4MB | 2.5MB | 0.31ms | 0.019ms |
| 32 | 128KB | 128KB | 9.5MB | 10MB | 1.25ms | 0.075ms |
| 64 | 256KB | 256KB | 19MB | 20MB | 2.50ms | 0.15ms |

계산 기준: PCIe effective BW = 8 GB/s (`tp_comm_efficiency_scale=0.125`),
NVLink effective BW = 133 GB/s (H200 NV18 full mesh, `scale=3.59`).

### NVLink 적용 시 TPOT 예측

| Component | H100 PCIe (Paladin) | H200 NVLink | 변화 |
|-----------|:---:|:---:|:---:|
| Compute (GEMM+ATTN+MoE) | ~3.5ms | ~2.5ms | -30% (HBM 4800 vs 3350 GB/s) |
| AllReduce (b=1) | ~1.3ms | ~0.05ms | **-96%** |
| GPU graph total | ~5ms | ~3ms | -40% |
| CPU iteration (parallel) | ~4ms | ~4ms | 0% (Python, 불변) |
| **TPOT = max(CPU, GPU) + sync** | **~7ms** | **~5-6ms** | **-15~29%** |

**실측 비교**: Cloud H200 4×NVLink TP2에서 Qwen3-Next agentic c=1 TPOT = **7-8ms**.
예측(5-6ms)보다 높은 이유:
1. GPU graph가 3ms로 줄어도 **CPU 4ms가 critical path**가 됨
2. H200 cloud 인스턴스의 NUMA/scheduler 차이
3. Expert offload 미적용 (cloud 벤치는 non-offload TP2, 다른 코드 경로)

### NVLink으로 줄일 수 있는 부분 vs 없는 부분

**줄일 수 있는 것 (NVLink 효과)**:
- TP AllReduce 시간: PCIe 1.3ms → NVLink 0.05ms (b=1)
- GPU graph: ~5ms → ~3ms
- 특히 c≥8에서 효과 큼 (comm 비중 52% → ~5%)

**줄일 수 없는 것**:
- CPU iteration overhead (~4ms) — Python/PyTorch scheduling, GIL. 하드웨어 무관
- NVLink으로 GPU graph를 CPU보다 빠르게 만들면, **CPU가 새로운 병목**이 됨
- Expert offload pre_step — 이미 graph 안에 숨겨져 있으므로 TPOT 무관

**결론**: NVLink은 GPU graph를 5ms→3ms로 줄이지만, CPU가 ~4ms이므로
**TPOT는 max(4ms, 3ms) + sync ≈ 5-6ms**. CPU 4ms가 새로운 hard floor.
이를 넘으려면 C++ custom scheduler나 torch.compile 기반 graph-only execution 필요.

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

## Contiguous DMA Layout — 결론

### 리뷰 포인트와 분석

| # | 리뷰 포인트 | 판정 | 근거 |
|---|------------|------|------|
| 1 | Fancy indexing ≠ single large DMA | **동의** | `gpu[slots] = cpu[ids]`는 PyTorch 내부에서 gather/scatter + 임시 텐서 생성. 개별 `copy_()`가 오히려 깨끗함 |
| 2 | 현재 병목은 fetch가 아님 | **완전 동의** | v7: t_fetch=78us (2%), miss=0.2/step. DMA BW를 3× 올려도 TPOT 변화 0 |
| 3 | 연속 DMA는 src+dst 둘 다 연속 필요 | **동의** | CPU contiguous해도 GPU slot이 LRU-random이면 합침 불가. 슬롯 정책까지 변경 필요 |
| 4 | TPOT vs pre_step 측정 불일치 | **부분 동의** | v7에서 pre_step(3.7ms) << graph(7.0ms)이므로 불일치 완화. NVTX 구현 완료, nsys 재시작시 검증 가능 |

### DMA 방향 확인

Expert weight copy 방향은 **항상 CPU → GPU** (단방향):
- `_load_to_slot()`: CPU pinned → GPU slot (sync, initial populate)
- `_async_fetch()`: CPU pinned → GPU slot (async, non_blocking, copy_stream)
- `_sync_fetch()`: CPU pinned → GPU slot (sync, last resort)
- `_prefetch_next_layer()`: CPU pinned → GPU slot (async, non_blocking)

GPU → CPU expert weight copy는 **존재하지 않음**.
GPU→CPU는 routing snapshot D2H만 (topk_ids, int32, ~수 KB).

### 최종 결론

Contiguous DMA Layout 우선순위: **LOW → SKIP**.

이유:
1. Steady state miss = 0.2 experts/step → DMA 자체가 pre_step의 2%
2. pre_step 전체가 CUDA graph 안에 숨겨져 있어 TPOT에 직접 영향 없음
3. 구현 시 fancy indexing 함정 (추가 오버헤드), 슬롯 정책 변경 필요
4. Cold start (~500 steps) 개선만 가능, steady state 이점 없음

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

### pre_step 내부 (TPOT 영향 없음, headroom 확보용)

| 항목 | 현재 | 목표 | 방법 |
|------|:---:|:---:|------|
| t_classify | 1,957 us | <500 us | 48 layers numpy batch vectorization |
| t_gpu_gather | 942 us | <200 us | Custom CUDA kernel (single launch) |
| t_cache_map_build | 764 us | <200 us | Vectorized e2s→cache_map 변환 |

이 최적화들은 TPOT을 직접 줄이지 않지만 (이미 graph 안에 숨겨짐),
headroom을 3.7ms → ~1ms로 줄여서 CUDA graph이 더 길어져도 (c=32 등) 여유 확보.

### TPOT를 줄이려면

| 방법 | 예상 효과 | 비고 |
|------|:---:|------|
| NVLink (H200) | -15~25% at c=1 | Comm 1.7ms → 0.1ms, but CPU 3.1ms 불변 |
| TP4 (4 GPU) | -30~40% | Compute 2× 분산, comm 증가 |
| C++ scheduler | -20~30% | CPU iteration 3.1ms → ~1ms, Python 제거 |
| FP8 quantization | -20~30% | GEMM throughput 2×, memory BW 절감 |
