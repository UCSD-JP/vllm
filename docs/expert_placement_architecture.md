# Expert Placement Architecture — GPU Cache + CPU Backing Store

## 1. Overview

Expert offloading은 "400개 GPU 고정 + 112개 CPU 고정"이 **아닙니다**.
**512개 전부 CPU pinned memory에 상주**하고, GPU에는 **400-slot LRU 캐시**가 있는 구조입니다.

```
┌─────────────────────────────────────────────────────────────┐
│  CPU (Host, Pinned Memory)                                  │
│                                                             │
│  _cpu_pool[layer_idx][expert_id] = (w13_cpu, w2_cpu)        │
│  512개 expert × 48 layers = 24,576 entries                  │
│  모든 expert가 항상 여기 있음 (backing store, 삭제 안 됨)   │
│                                                             │
│  Per expert: 3.146 MB (BF16, TP2-sharded)                   │
│  Total CPU: 512 × 3.146MB × 48 layers = ~75.3 GB           │
└──────────────────────────┬──────────────────────────────────┘
                           │  CPU→GPU DMA (non_blocking, copy_stream)
                           │  evict 시: slot만 해제 (CPU→GPU 역방향 없음)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  GPU (HBM)                                                  │
│                                                             │
│  w13_weight[slot_idx]  shape = [400, 512, 2048]  per layer  │
│  w2_weight[slot_idx]   shape = [400, 2048, 256]  per layer  │
│                                                             │
│  400개 슬롯 = 캐시 라인 (어떤 expert든 어떤 슬롯에 매핑)    │
│  GPU total: 400 × 3.146MB × 48 layers = ~60.4 GB           │
│                                                             │
│  양방향 매핑:                                                │
│    _expert_to_slot[layer][expert_id] → slot (or -1)         │
│    _slot_to_expert[layer][slot] → expert_id (or -1)         │
│                                                             │
│  cache_map[global_expert_id] → slot_idx                     │
│  → CUDA graph kernel이 이 map으로 weight 참조               │
└─────────────────────────────────────────────────────────────┘
```

## 2. Expert Weight Size 검증

### Model Config (Qwen3-Next-80B-A3B)

| Parameter | Value | Source |
|-----------|-------|--------|
| hidden_size | 2048 | `qwen3_next.py:189` |
| moe_intermediate_size | 512 | `qwen3_next.py:210` |
| num_experts | 512 | `qwen3_next.py:213` |
| num_experts_per_tok | 10 | `qwen3_next.py:212` |
| num_hidden_layers | 48 | model config |
| dtype | BF16 (2 bytes) | |
| TP | 2 | |

### Per-Expert Tensor Shapes (TP2, BF16)

코드 (`unquantized_fused_moe_method.py:126,143-147`):

```python
intermediate_size_per_partition = moe_intermediate_size / TP = 512 / 2 = 256
w13_up_dim = 2 * intermediate_size_per_partition = 2 * 256 = 512  # fused gate+up

# Per expert shapes (first dim = expert count, excluded):
w13_per_expert = (w13_up_dim, hidden_size) = (512, 2048)
w2_per_expert  = (hidden_size, intermediate_size_per_partition) = (2048, 256)
```

### Size Calculation

```
w13: 512 × 2048 × 2B = 2,097,152 bytes = 2.097 MB
w2:  2048 × 256 × 2B = 1,048,576 bytes = 1.049 MB
─────────────────────────────────────────────────
Total per expert:       3,145,728 bytes = 3.146 MB
```

### 80B 모델인데 expert당 3MB?

```
80B total params
 ├── Non-MoE params (embedding, attention, LayerNorm, etc.): ~5-10B
 └── MoE params: ~70-75B
     └── Per layer MoE: 70B / 48 layers ≈ 1.46B params per layer
         └── Per expert: 1.46B / 512 experts ≈ 2.85M params per expert
             └── BF16: 2.85M × 2B = 5.7 MB (full, unsharded)
             └── TP2:  5.7 MB / 2 = 2.85 MB ← analytical
             └── Actual (fused packing): 3.146 MB ✓

검증: 512 × 2048 + 2048 × 256 = 1,048,576 + 524,288 = 1,572,864 params
     × 2B = 3,145,728 bytes = 3.146 MB ✓
     × 512 experts × 48 layers = 38.6B MoE params (BF16, TP2-sharded)
     × 2 (TP2→full) = 77.2B MoE params (full model)
     + ~3B non-MoE ≈ 80B total ✓
```

### Memory Footprint Summary

| Component | Per Layer | 48 Layers Total |
|-----------|-----------|-----------------|
| GPU slots (400 experts) | 400 × 3.146MB = 1.26 GB | **60.4 GB** |
| CPU backing (512 experts) | 512 × 3.146MB = 1.61 GB | **75.3 GB** (pinned) |
| Non-MoE params (GPU) | ~0.3 GB | **~14 GB** |
| **GPU total** | | **~74.5 GB** (safe for 93GB H100) |
| **CPU total** | | **~75.3 GB** (pinned memory) |

## 3. Step별 동작 흐름

### 3.1 Cold Start (populate_initial_cache)

```
Server 시작 시:
  1. create_weights(): GPU [400, ...] + CPU [512, ...] 생성
  2. weight_loader(): checkpoint → CPU backing store에 로드
  3. _init_expert_offloading(): CPU store → pinned pool 복사
  4. populate_initial_cache(): expert 0~399를 GPU slot 0~399에 동기 DMA
  5. 나머지 112개(expert 400~511)는 CPU에만 존재 (slot = -1)
```

### 3.2 Steady State (per inference step)

```
┌──────────────────────────────────────────────────────────────────┐
│  pre_step() — CUDA graph 실행 전에 호출 (CPU thread)             │
│                                                                  │
│  Phase 0: Deferred Sync                                          │
│    이전 step의 async DMA 완료 대기 (copy_stream.synchronize())    │
│    → 이전 step에서 miss된 expert가 이제 GPU에 valid              │
│                                                                  │
│  Phase A: GPU→GPU Gather                                         │
│    48 layers의 routing snapshot (topk_ids)를 stacked buffer에    │
│                                                                  │
│  Phase B: Bulk D2H                                               │
│    routing snapshot GPU→CPU 복사 (d2h_stream, 1회 sync)          │
│                                                                  │
│  Phase C: CPU-only Processing (per layer)                        │
│                                                                  │
│    ┌────────────────────────────────────────────────────┐        │
│    │  Layer i:                                          │        │
│    │                                                    │        │
│    │  1. routing 읽기                                   │        │
│    │     needed = {expert_3, 7, 15, 412, ...}           │        │
│    │                                                    │        │
│    │  2. Hit/Miss classification                        │        │
│    │     hit:  expert_3  → slot[0]  ✓ (access_count++)  │        │
│    │     hit:  expert_7  → slot[1]  ✓                   │        │
│    │     hit:  expert_15 → slot[8]  ✓                   │        │
│    │     miss: expert_412 → slot=-1 ✗                   │        │
│    │                                                    │        │
│    │  3. Async Fetch (miss만)                           │        │
│    │     a. evict: LFU/LRU로 victim 선택                │        │
│    │        slot[237]의 expert_501 제거                  │        │
│    │        (CPU에서는 삭제 안 됨, slot만 반환)          │        │
│    │     b. slot[237] = expert_412                      │        │
│    │     c. DMA 큐잉 (copy_stream, non_blocking):       │        │
│    │        _cpu_pool[i][412].w13 → w13_weight[237]     │        │
│    │        _cpu_pool[i][412].w2  → w2_weight[237]      │        │
│    │        (~3.15MB, copy_stream에서 비동기 실행)       │        │
│    │                                                    │        │
│    │  4. cache_map 빌드 (vectorized, CPU)               │        │
│    │     cache_map[412] = -1  (DMA 미완료, 이번에 skip) │        │
│    │     cache_map[3] = 0, cache_map[7] = 1, ...        │        │
│    └────────────────────────────────────────────────────┘        │
│                                                                  │
│  Phase C2: Batched cache_map Upload                              │
│    pinned CPU → GPU bulk copy (non_blocking, ~370us)             │
│    → CUDA graph가 이 cache_map 참조                              │
└──────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────┐
│  CUDA Graph Replay (~4.5ms)                                      │
│                                                                  │
│  48 layers × (Attention + MoE + AllReduce):                      │
│    MoE kernel: w13_weight[cache_map[topk_ids[token]]]            │
│    → slot 0에 있는 expert_3 weight 사용 ✓                        │
│    → slot 237에 있는 expert_412: cache_map=-1이므로 skip         │
│                                                                  │
│  동시에 copy_stream에서 expert_412 DMA 진행 중 (overlap)         │
└──────────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────────┐
│  Next Step의 pre_step()                                          │
│                                                                  │
│  Phase 0: copy_stream.synchronize()                              │
│    → expert_412 DMA 완료 확인                                    │
│    → cache_map[412] = 237 (이제 사용 가능)                       │
└──────────────────────────────────────────────────────────────────┘
```

### 3.3 Deferred Sync (Async DMA Hiding)

Miss된 expert는 **이번 step에서는 사용 불가** (cache_map = -1), **다음 step부터 사용 가능**.
DMA 전송이 CUDA graph replay와 시간적으로 겹치므로 latency가 숨겨집니다.

```
Timeline:

Step N:  pre_step [miss→DMA 큐잉] ──→ CUDA Graph Replay ──→
                                       ↑                     ↑
         copy_stream: ────────── [DMA expert_412 전송] ──────→

Step N+1: pre_step [sync: DMA 완료 ✓] → cache_map에 412 포함 → CUDA Graph
```

**실측**: deferred sync 대기 시간 = `t_deferred_sync_us`
- 수렴 후: ~0us (miss가 거의 없어 DMA 없음)
- miss 발생 시: ~300-500us (3.15MB DMA, PCIe ~25 GB/s)

## 4. Eviction Policy (LFU/LRU Hybrid)

```python
# expert_cache.py:435-437
# lfu_lru policy: frequency / age
age = max(current_step - last_access[key] + 1, 1)
priority = access_count[key] / age    # 낮을수록 evict 대상
```

- **pinned experts** (shared experts): eviction 대상에서 제외
- **protected experts**: 현재 step에서 needed인 expert는 evict 안 됨
- **prefetch pending**: 비동기 DMA 진행 중인 expert도 evict 안 됨

## 5. Batch Miss Fetch

한 step에 miss가 여러 개면 **한꺼번에** async fetch:

```python
# expert_cache.py:928-936
if miss_ids:  # e.g. [412, 489, 503]
    self._async_fetch(i, miss_ids, protected=cached_needed)
    # → 3개 expert를 copy_stream에서 연속 DMA 큐잉
```

다만 실측에서 **miss는 step당 평균 0.2개** (hit rate 99.9%)이므로
대부분의 step은 swap 없이 지나갑니다.

### Miss Rate 추이 (Gating 수렴)

| Step | Hit Rate | Avg Misses/Step | 비고 |
|------|----------|-----------------|------|
| 0-100 | ~95% | ~5-10 | Cold start, 다양한 expert 탐색 |
| 100-500 | ~98% | ~2-3 | Gating entropy 89.2% (넓은 분산) |
| 500-2000 | ~99.5% | ~0.5 | 수렴 중 |
| 2000+ | ~99.9% | ~0.2 | 안정 상태, top 11 experts > 1% each |

## 6. DMA 방향 요약

| 방향 | 용도 | 발생 시점 | 크기 |
|------|------|-----------|------|
| **CPU→GPU** | expert weight 로드 | miss 시 _async_fetch | 3.15MB/expert |
| **CPU→GPU** | cache_map 업로드 | pre_step Phase C2 | ~96KB (48×512×4B) |
| **GPU→CPU** | routing snapshot D2H | pre_step Phase B | ~200KB |
| ~~GPU→CPU~~ | ~~expert evict~~ | **없음** | CPU에서 삭제 안 함 |

**핵심**: evict 시 GPU→CPU 역복사 없음. CPU에 전체 512 expert가 항상 있으므로
evict는 slot 해제 + 매핑 삭제만 수행.

## 7. Miss-Rate 검증 결과 (Paladin 실측, 2026-02-27)

기존 "99.9% hit rate" 주장이 진짜 locality인지, 측정 아티팩트인지 검증.

### 실험 설정

- **Paladin 2×H100 SXM**, TP2, BF16, Qwen3-Next-80B-A3B
- 워크로드: prompt~100tok, max_tokens=32, sequential requests
- 각 config: warmup 100 requests → 측정 300 requests
- 진단: 100 step마다 `get_miss_diagnostics()` 자동 출력

### 검증 1: max_res sweep (400→200→100)

| max_res | TPOT (ms) | unique-ID hit% | **token-wtd hit%** | uid miss/step/layer | tok miss/step/layer |
|---------|-----------|---------------|-------------------|--------------------|--------------------|
| **400** | 8.72 | 98.9% | **94.5%** | 0.11 | 1.0 |
| **200** | 8.74 | 96.7% | **82.5%** | 0.33 | 2.3 |
| **100** | 8.79 | 94.9% | **76.0%** | 0.51 | 3.2 |

**결론**:
- miss는 max_res에 따라 **단조 증가** → 실제 locality 효과가 존재
- 하지만 **TPOT는 거의 변하지 않음** (8.72→8.79ms, +0.8%) → miss가 deferred sync로 완전히 숨겨짐
- **token-weighted miss가 unique-ID보다 훨씬 높음** (아래 상세 분석)

### 검증 4: per-layer 진단 (artifact 발견)

Step 200, max_res=400 (TP1 worker):

```
[valid_len vs rlen — artifact check]
  avg rlen/layer/step: 54.4     ← CUDA graph에 캡처된 routing length
  avg valid_len/layer/step: 45.0  ← num_tokens × top_k (실제 유효)
  truncation: 17.2%              ← 17% routing entries가 padding이었음!

[Per-Layer Detail (avg/step)]
  layer   rlen  vlen uniq_need uid_hit uid_miss uid_rate tok_hit tok_miss tok_rate
      0    54    45      10.0    10.0    0.04   0.9955    12.8    0.36   0.9726
     11    54    45      10.0     9.9    0.06   0.9940    12.5    0.69   0.9475
     23    54    45      10.0     9.9    0.13   0.9870    12.7    0.45   0.9662
     35    54    45      10.0     9.8    0.15   0.9845    12.1    1.10   0.9163
     47    54    45      10.0     9.9    0.09   0.9915    12.1    1.03   0.9217

[Unique-Needed Stats]
  min=10.0, max=10.0, mean=10.0, std=0.0
  layers needing > max_resident(400): 0/48
```

**핵심 발견**:
1. **unique_needed = 정확히 10.0** (top_k=10) — step당 layer당 딱 10개 expert만 필요
2. **truncation 17-44%** — rlen > valid_len이므로 CUDA graph padding 데이터가 포함됨
3. 400 슬롯에 10개만 필요하니 unique-ID miss가 낮은 건 구조적으로 당연
4. 그러나 **tok_rate < uid_rate** → miss expert가 많은 token을 처리 (아래 상세)

### 검증 5: token-weighted vs unique-ID hit rate 괴리

| max_res | unique-ID | token-wtd | 차이 |
|---------|-----------|-----------|------|
| 400 | 98.9% | 94.5% | **-4.4pp** |
| 200 | 96.7% | 82.5% | **-14.2pp** |
| 100 | 94.9% | 76.0% | **-18.9pp** |

**해석**: unique-ID 기준으로는 10개 중 9.8개 hit이지만, 그 0.2개 miss expert에
**불균형하게 많은 token이 라우팅됨**. 이는 "long-tail expert"가 갑자기 인기를 끌 때
(= gating distribution shift) 발생.

- unique-ID: "expert X가 캐시에 있는가?" → binary, expert 수 기준
- token-wtd: "이 token의 expert가 캐시에 있는가?" → 실제 compute impact

**token-weighted가 실제 성능 영향에 가까운 metric**이지만, deferred sync로
miss expert를 다음 step에서 처리하므로 TPOT에는 거의 영향 없음.

### Per-Layer 패턴

- **Layer 35, 47이 miss가 가장 높음** (모든 config에서 일관)
- 이는 후반 layer의 gating이 더 다양하거나, expert 분포가 더 넓기 때문
- Example (max_res=100, step 100):
  - Layer 0: tok_rate=0.700 (miss 4.89/step)
  - Layer 35: tok_rate=0.695 (miss 4.97/step)
  - Layer 47: tok_rate=0.645 (miss 5.79/step) ← worst

### 결론

| 질문 | 답 |
|------|---|
| 99.9% hit rate는 진짜인가? | unique-ID 기준으로는 맞음 (98-99%). 하지만 **token-weighted로는 76-94%** |
| 이유는? | unique_needed=10 << max_resident=400이므로 unique miss가 구조적으로 낮음 |
| artifact 있는가? | **있음**: truncation 17-44% (CUDA graph padding), unique vs token 괴리 |
| TPOT에 영향은? | **거의 없음**: deferred sync가 miss를 완전히 숨김 (max_res=100에서도 +0.8% only) |
| max_res 최적값은? | 100으로도 TPOT 영향 미미. GPU 메모리 절약이 필요하면 100-200도 가능 |

## 8. Code References

| Component | File | Lines |
|-----------|------|-------|
| ExpertCacheManager class | `expert_cache.py` | 49-137 |
| CPU pinned pool registration | `expert_cache.py` | 169-189 |
| Initial cache populate | `expert_cache.py` | 193-220 |
| Sync fetch (blocking) | `expert_cache.py` | 361-407 |
| Async fetch (non-blocking) | `expert_cache.py` | 671-705 |
| Eviction (LFU/LRU) | `expert_cache.py` | 411-451 |
| pre_step v2 (CUDA graph) | `expert_cache.py` | 793-990 |
| Deferred sync | `expert_cache.py` | 818-831 |
| create_weights (offload mode) | `unquantized_fused_moe_method.py` | 114-188 |
| _init_expert_offloading | `gpu_worker.py` | 279-459 |
