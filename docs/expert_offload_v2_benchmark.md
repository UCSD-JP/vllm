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

### Eager Mode vs CUDA Graph — 왜 Graph가 필수인가

Eager mode에서는 매 forward pass마다 Python이 커널을 하나씩 발사한다:

```
Python:  [select_experts] → launch kernel → [routing] → launch kernel → ...
GPU:     .........[kernel].........[kernel].........[kernel]...
              ↑ idle gap      ↑ idle gap      ↑ idle gap
```

Qwen3-Next-80B TP2 기준 **한 step에 ~4,858개 커널**:
- 48 MoE layers × (ATTN + FFN/MoE + OUT_PROJ) = ~144 compute groups
- 각 group 내부: select_experts, GEMM, scatter, activation 등 다수 커널
- 96 AllReduce calls (48 layers × 2)
- 각 커널 launch마다 CPU→GPU 커맨드 전송 ~11us

결과: CPU launch overhead만 `11us × 4,858 = 53ms`, GPU idle gap `2us × 4,858 = 10ms`.
이것이 eager mode TPOT 90ms의 정체이며, expert offload v1이 CUDA graph 불가로 56.5ms인 이유.

### CUDA Graph Replay가 실제로 하는 것

**녹화 (서버 시작 시 1회)**:
```python
with torch.cuda.graph(cudagraph):
    output = model.forward(dummy_input)  # 4,858개 커널 시퀀스 녹화
```

GPU가 모든 커널의 순서, 파라미터, 메모리 주소를 기록. **텐서의 data_ptr() (메모리 주소)가 고정**됨.

**재생 (매 step)**:
```python
cudagraph.replay()  # C++ 레벨에서 단일 호출. Python 코드 실행 없음.
```

이 한 줄이 GPU에서 실행하는 전체 시퀀스:

```
GPU (단일 replay 안에서 순차 실행):
├─ Layer 0:
│   ├─ select_experts kernel    (topk routing, GPU-only 연산)
│   ├─ _routing_snapshot.copy_() (다음 step용 라우팅 캡처, GPU→GPU)
│   ├─ expert_map = _cache_map 읽기 (pre_step이 미리 써놓은 값)
│   ├─ MoE GEMM kernel (w13[slot] × hidden_state)
│   ├─ activation kernel (SiLU)
│   ├─ MoE GEMM kernel (w2[slot] × intermediate)
│   ├─ AllReduce (TP2: GPU0 ↔ GPU1 결과 합산)
│   ├─ ATTN kernel (Q·K^T, softmax, ×V)
│   ├─ ATTN output GEMM
│   └─ AllReduce
├─ Layer 1: (동일 구조)
├─ ...
├─ Layer 47: (동일 구조)
└─ LM head: hidden → logits

총 시간: ~5ms GPU compute + ~2ms sync overhead = ~7ms
```

핵심: 이 전체 과정에서 **Python 코드는 한 줄도 실행되지 않음**. CPU는 `replay()` 호출 후 대기.
4,858개 커널이 GPU 내부에서 연쇄 실행되며, 변경되는 것은 **고정 메모리 주소의 데이터**뿐.

### pre_step이 실제로 하는 것

512개 expert 중 400개만 GPU에 상주하므로, **다음 step에서 커널이 읽을 expert 매핑 테이블**을
graph replay 전에 미리 갱신해야 한다. 이것이 pre_step의 역할.

```
pre_step(3.7ms) 내부:

Phase A — GPU→GPU gather (942us, 25.5%)
  48 layers의 _routing_snapshot (이전 step에서 GPU가 기록한 topk_ids)을
  stacked GPU buffer로 모음. GPU 내부 D2D copy 48회.

Phase B — Bulk D2H (410us, 11.1%)
  모아진 routing 데이터를 GPU→CPU로 한 번에 전송.
  pinned CPU memory에 착지. 단일 stream, 단일 sync.

Phase C — CPU classify (1,957us, 52.9%)
  CPU에서 순수 Python/tensor 연산:
  ├─ 각 layer별 "이전 step에서 어떤 expert가 라우팅됐나?" 파싱
  ├─ hit/miss 분류: GPU slot에 있으면 hit, 없으면 miss
  ├─ miss인 expert → async DMA 큐잉 (CPU pinned → GPU slot, copy_stream)
  │   (78us로 큐잉만 하고 완료를 기다리지 않음)
  └─ cache_map scratch 업데이트 (expert_id → slot_id 매핑 테이블)

Phase C2 — Batched GPU upload (371us, 10.0%)
  48개 layer의 cache_map을 pinned CPU → GPU로 한 번에 전송.
  layer._cache_map.copy_()로 고정 메모리 주소의 값만 교체.
  → 이후 graph replay 시 커널이 이 주소를 읽으면 새 값이 보임.
```

pre_step의 본질: **"이전 step의 라우팅 결과를 보고, 다음 step에서 커널이 읽을
expert→slot 매핑 테이블을 미리 채워놓기"**. 1-step-behind prediction.

### Overlap의 실체 — "Fully Hidden"이 의미하는 것

pre_step은 CPU에서, graph replay는 GPU에서 실행. 서로 다른 하드웨어이므로 **병렬 실행** 가능.

```
시간축 →

CPU thread:  [sched][───── pre_step 3.7ms ─────][graph launch][idle 3.3ms]
                    │                            │
                    │ (cache_map 값 갱신)          │ (replay() 호출, non-blocking)
                    │                            ↓
GPU default: [이전 step 마무리]──────────────────[graph replay 7.0ms──────────]
                                                 │
GPU copy_st: ───────[DMA 0.3ms]──→               │ (miss expert 비동기 전송)
                                                 │
                    ←──────────── TPOT ≈ 7.0ms ──────────────→
```

TPOT 결정 공식:

```
TPOT = max(CPU_total, GPU_graph) + sync_overhead

CPU_total = sched + pre_step + post ≈ 4ms
GPU_graph ≈ 7ms (5ms compute + 2ms overhead)

TPOT = max(4ms, 7ms) + α ≈ 7ms
```

**"Fully hidden"의 의미**: pre_step이 3.7ms이든 1ms이든, `CPU_total < GPU_graph`인 한
TPOT은 변하지 않음. GPU가 더 오래 걸리므로 CPU 쪽 시간이 완전히 "숨겨짐".
pre_step을 0ms로 만들어도 TPOT은 여전히 ~7ms.

반대로 **v5에서는 pre_step = 12.6ms** → `CPU_total ≈ 13ms > GPU_graph 7ms` → CPU가 병목.
그래서 TPOT = 10.2ms. v5→v7로 pre_step을 12.6ms→3.7ms로 줄여서 CPU가 GPU보다
빨라지는 전환점을 넘긴 것이 핵심 성과.

### 1-Step-Behind Prediction — 왜 필요하고, 왜 동작하는가

#### 문제: graph 안에서는 라우팅 결과를 볼 수 없다

CUDA graph replay 중에는 Python이 실행되지 않는다. 즉 **step N에서 어떤 expert가
라우팅됐는지를 step N 안에서 CPU가 알 수 없다**.

```
Step N의 graph replay 안:
  select_experts(hidden_state) → topk_ids = [expert 3, 7, 42, ...]
  ↑ 이 결과는 GPU 메모리에만 존재
  ↑ CPU는 모름 (graph 중 Python 실행 안 됨)
  ↑ 따라서 "expert 42가 GPU에 없으니 지금 로드하자"는 불가능
```

#### 해결: 이전 step의 라우팅으로 다음 step을 예측

```
Step N-1 (graph replay 중):
  ├─ select_experts() → topk_ids = [3, 7, 42, 156, ...]
  ├─ _routing_snapshot.copy_(topk_ids)  ← GPU 메모리에 기록만 해둠
  └─ kernel(expert_map=_cache_map)      ← 이미 준비된 매핑으로 실행

Step N (graph replay 전):
  ├─ pre_step():
  │   ├─ _routing_snapshot 읽기 (GPU→CPU)  ← Step N-1의 라우팅 결과
  │   ├─ "step N-1에서 expert 3,7,42,156을 썼으니
  │   │    step N에서도 비슷할 것이다" ← 예측
  │   ├─ miss인 expert → DMA로 GPU에 로드
  │   └─ _cache_map 갱신 (expert→slot 매핑)
  └─ graph replay:
      └─ kernel이 갱신된 _cache_map으로 실행
```

핵심: step N의 pre_step은 **step N-1의 라우팅**을 보고 준비. 항상 1 step 뒤.

#### 예측이 틀리면 어떻게 되나

Step N-1에서 expert [3, 7, 42]를 썼는데, step N에서 expert [3, 7, **99**]가 라우팅된 경우:

```
pre_step: step N-1 기반으로 expert 3, 7, 42를 GPU에 준비
graph replay: select_experts() → expert 3, 7, 99 필요

expert 3:  _cache_map[3] = slot 5  → 정상 (hit)
expert 7:  _cache_map[7] = slot 12 → 정상 (hit)
expert 99: _cache_map[99] = -1     → slot 없음 (miss)
           → MoE kernel이 expert 99의 기여를 0으로 처리
           → top-10 중 1개 expert 누락 = softmax weight의 ~10% 손실
```

Miss의 영향:
- 해당 token의 해당 layer에서 MoE output이 ~10% 약해짐 (1/top_k)
- **1 step만** — 다음 step의 pre_step에서 expert 99를 로드
- Decode는 autoregressive → 다음 토큰에서 self-correct 가능
- 48 layers 전부에서 동시에 같은 expert miss 확률은 극히 낮음

#### 왜 예측이 거의 맞나 — Hit rate 98.8~100%

Decode 특성상 **연속된 step의 라우팅이 매우 안정적**:

```
Step N-1: input = "The capital of France is"
          hidden_state → routing → expert [3, 7, 42, 156, ...]

Step N:   input = " Par"  (다음 토큰 1개 추가)
          hidden_state → routing → expert [3, 7, 42, 156, ...]
          ↑ 거의 동일 (같은 문맥, 비슷한 hidden state)
```

안정적인 이유:
1. **문맥 연속성**: 토큰 1개 추가로 hidden state가 급변하지 않음
2. **Expert popularity bias**: 512개 중 상위 ~21개가 대부분 차지 (entropy 47.9%)
3. **max_resident=400**: 512개 중 400개 상주 → 어떤 expert든 이미 있을 확률 78%.
   인기 편중 포함 시 실질 miss rate ≈ 0

실측 결과:

| 조건 | Hit rate | Miss/step | 설명 |
|------|:---:|:---:|------|
| res=300 | 98.8% | 5.7 | 48 layers × 1.2% miss |
| res=400, cold start | 99.1% | ~2 | 초기 워밍업 중 |
| **res=400, steady state** | **100.0%** | **0.2** | 거의 miss 없음 |

#### Eager (v1) vs Predict-and-Preload (v2) Trade-off

| | v1 (eager, 실시간) | v2 (predict, 1-step-behind) |
|---|---|---|
| 정보 시점 | **현재** step 라우팅 | **이전** step 라우팅 |
| 정확도 | 100% | 98.8~100% |
| CUDA graph | **불가** | **호환** |
| TPOT | 56.5ms | **7.0ms** |
| Miss 시 | 없음 | 해당 expert 기여 0 (1 step) |

100% 정확도를 위해 56.5ms를 쓰는 것보다, 98.8% 정확도로 7.0ms를 달성하는 것이 8× 빠름.

#### Miss Coverage 방안 — 진짜 100%를 원한다면

**방안 A: Piecewise CUDA Graph (설계서 §2.1 Codex 제안)**

select_experts()만 eager로 분리하여 실시간 라우팅을 확보:

```
A단계 (eager): 48 layers의 select_experts() 실행 → 정확한 expert 목록 확보
               miss expert DMA fetch (동기)
B단계 (graph): MoE kernel + ATTN + AllReduce만 replay
```

- 장점: **100% 정확도 + CUDA graph 속도**
- 단점: full-model graph를 layer 단위로 분할 필요, 96개 추가 커널 launch (~1ms),
  piecewise CUDA graph 인프라 구현 복잡도 높음

**방안 B: Eager Fallback Threshold 강화**

현재 구현에 이미 존재하는 메커니즘:

```python
# gpu_model_runner.py
if miss_count > threshold:
    cudagraph_mode = CUDAGraphMode.NONE  # 이번 step만 eager
```

- 현재 threshold: miss_ratio > 0.95 (거의 안 발동)
- **miss_count > 0** 으로 변경 시: miss 발생 step만 eager로 실행 (~90ms)
- Steady state에서 100% hit → 사실상 항상 graph mode
- Transition 시에만 가끔 eager → 평균 TPOT 영향 미미

**방안 C: Speculative Pre-fetch 확장**

pre_step에서 이전 routing 외에 popularity/frequency 기반으로 추가 expert를 미리 로드.
DMA 트래픽 증가 대신 miss rate 추가 감소. 현재 이미 100%이므로 실익 없음.

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

## Miss Mitigation — 100% Hit Rate 달성 방안

**실측 hit rate (c=16, SWE-bench, res=400)**: 99.38% (100%가 아님). 상세 분석 및 설계:

- 실측 데이터: [`expert_cache_hit_analysis.md`](expert_cache_hit_analysis.md)
- **Miss mitigation 설계서**: [`expert_cache_miss_mitigation.md`](expert_cache_miss_mitigation.md)

권장 조합 요약:

| Phase | 방안 | Hit Rate | TPOT Overhead |
|:---:|------|:---:|:---:|
| 1 | Never-evict + Multi-step union + Eager fallback(miss>0) | **100%** | ~+0.2ms |
| 2 | + Layer 0 eager routing | **~99.9%+** (fallback 극소화) | +0.1ms |
| 3 | Per-layer piecewise graph (필요 시) | **100% (구조적)** | +1.6ms |

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
