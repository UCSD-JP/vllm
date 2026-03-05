# Expert Cache Miss Mitigation — 100% Hit Rate 달성 설계

Related docs:
- Hit rate 실측 데이터: [`expert_cache_hit_analysis.md`](expert_cache_hit_analysis.md)
- v2 벤치마크/타이밍: [`expert_offload_v2_benchmark.md`](expert_offload_v2_benchmark.md)
- v2 설계/코드: [`expert_offload_v2_design.md`](../docs/expert_offload_v2_design.md)
- **품질 영향 측정 prompt**: [`expert_miss_quality_eval_prompt.md`](expert_miss_quality_eval_prompt.md)

---

## 1. 문제 정의

### 1.1 설계 문서 주장 vs 실측

| 항목 | 설계 문서 주장 | 실측 (c=16, SWE-bench, res=400) |
|------|:-----------:|:-----------------------------:|
| Steady state hit rate | 100% | **99.38%** (0.6% miss 잔존) |
| 초기 hit rate | — | **97.6%** (2.4% miss) |
| Miss/step | ~0 | **~0.6%** (최종), **~2%** (초기) |

**결론**: "100% steady state"는 c=1 단일 요청 벤치마크에서만 성립.
Concurrent agentic workload (c=16)에서는 routing diversity 증가로 miss가 유의미하게 발생.

### 1.2 Miss의 원인: 1-Step-Behind Prediction

현재 v7의 expert cache는 **predict-and-preload** 방식:
- pre_step에서 step N-1의 routing snapshot을 읽고 cache를 준비
- graph replay에서 step N의 실제 routing으로 실행
- step N이 step N-1과 다른 expert를 라우팅하면 miss

```
Step N-1 routing: [expert 3, 7, 42, 99]  ← pre_step이 아는 정보
Step N   routing: [expert 3, 7, 42, 201] ← 실제 필요

expert 201이 400개 상주 슬롯에 없으면 → cache_map[201] = -1 → 기여 0
```

**pre_step이 발견한 miss는 이미 DMA copy로 해결됨** (t_fetch=78us).
문제는 pre_step이 **발견할 수 없는** miss — step N 고유의 새로운 routing.

### 1.3 Per-Layer 분석 (실측)

| Layer | UID Hit Rate | Unique Needed | 특성 |
|------:|:-----------:|:------------:|:----:|
| 0 | **95.85%** | 103.3 | 가장 volatile (입력 embedding 직결) |
| 11 | 97.91% | 56.3 | 안정 |
| 23 | 97.98% | 57.6 | 안정 |
| 35 | 97.62% | 59.5 | 안정 |
| 47 | 98.19% | 83.6 | 출력 layer, 두번째로 많은 unique expert |

Layer 0이 bottleneck: 입력 token embedding에 직접 의존하므로 routing이 가장 volatile.
새 request 합류 시 routing 급변이 Layer 0에서 가장 심함.

### 1.4 왜 "늦게 copy"할 수 없나

```
pre_step (graph 전):
  miss 발견 → DMA copy 가능 ✓  (이미 하고 있음)
  BUT: step N의 routing은 아직 모름 → 뭘 copy할지 모름 ✗

graph replay (실행 중):
  step N의 routing 확정 → 뭘 copy할지 알지만
  GPU만 실행 중, CPU 개입 불가 → copy 불가 ✗

graph 완료 (실행 후):
  output 이미 나옴 → copy해봤자 이 step에는 이미 늦음 ✗
  중간 layer hidden state 덮어써짐 → 부분 재실행도 불가 ✗
```

근본 제약: **step N의 routing을 알려면 step N의 graph를 실행해야 하고,
실행하면 이미 miss가 발생한 후.** 이 순환을 깨는 것이 핵심.

---

## 2. Eager Mode vs CUDA Graph — 왜 Graph가 필수인가

### 2.1 Eager Mode의 비용

Eager mode에서는 매 forward pass마다 Python이 커널을 하나씩 발사:

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
Eager mode TPOT ≈ 90ms. CUDA graph TPOT ≈ 7ms. **12× 차이.**

### 2.2 Eager가 유리한 경우

| 케이스 | 이유 |
|--------|------|
| **Forward 내부 동적 분기** | Mixture of Depths, Early exit — layer별 조건부 실행 |
| **Variable shape (prefill)** | 요청마다 prompt 길이 다름 → graph 캡처 불가 |
| **Expert offload v1** | graph 내부 GPU→CPU sync → `cudaErrorStreamCaptureInvalidated` |
| **Speculative decoding verify** | Accept/reject 분기가 data-dependent |
| **메모리 동적 관리** | KV cache resize, GPU↔CPU swap (HMA 비호환) |
| **디버깅/프로파일링** | nsys에서 per-kernel 가시성 필요 시 `enforce_eager=True` |

---

## 3. Miss Mitigation 방안

### 3-A: Never-Evict Recent Policy

**개요**: 최근 K step에서 사용된 expert를 eviction 후보에서 제외.

**원리**:
```python
def _evict_one(self, layer_idx, protected):
    recent = self._recently_used[layer_idx]  # {lid: last_step}
    protected = protected | {
        lid for lid, step in recent.items()
        if self.current_step - step < K  # K=10 정도
    }
    # 나머지: 기존 LRU eviction
```

**분석**:
- 400 slots, ~21 active experts → 379 여유 slots
- 최근 10 step의 union ≈ 30-50 unique experts → 충분히 들어감
- Transition 시 이전 routing의 expert가 보존됨 → miss 감소

**비용**: TPOT overhead 0, 구현 ~20줄.

### 3-B: Multi-Step Union Prediction

**개요**: pre_step에서 N-1 routing 대신 최근 K step의 union을 사용.

**원리**:
```python
def pre_step(self, layers):
    # Phase B: D2H 후
    self._routing_history.append(current_routing)  # deque(maxlen=K)
    needed = set()
    for past_routing in self._routing_history:
        needed |= past_routing
    # needed 기반으로 miss 분류 → 더 넓은 범위 pre-fetch
```

**분석**:
- K=3: 최근 3 step의 routing union → prediction 범위 ~3× 확대
- 새로 등장한 expert가 2-3 step 전에 이미 등장했을 확률 높음
- 400 slots에 union이 충분히 들어감 (max unique ~133)

**비용**: TPOT overhead 0, 구현 ~30줄.

### 3-C: Layer 0 Eager Routing

**개요**: 가장 volatile한 Layer 0의 select_experts()만 graph 전에 eager 실행.

**원리**:
```
현재:
  pre_step (N-1 prediction for L0-47) → graph [L0 ... L47]

제안:
  pre_step (N-1 prediction for L1-47)
  → embedding(eager, ~5us)
  → Layer 0 select_experts(eager, ~20us)
  → Layer 0 miss fetch (DMA)
  → cache_map[Layer 0] 갱신 (정확한 routing 기반)
  → graph [L0 MoE+ATTN ... L47]
```

**분석**:
- Layer 0: 95.85% → **100%** (실시간 routing이므로 miss 불가)
- Layer 0이 전체 miss의 ~25% 담당 → 전체 miss rate ~25% 감소
- 추가로: Layer 0의 실제 routing을 L1-47 prediction에 합류 가능
  (Layer 0에서 새로 등장한 expert를 다른 layer에도 pre-load)

**비용**: ~100us (embedding + router + classify), **TPOT +1.4%**. 구현 중간 (graph entry point 분리 필요).

### 3-D: Per-Layer Piecewise CUDA Graph

**개요**: 48 layers 각각의 select_experts()를 eager로 분리, MoE+ATTN+AR만 graph.

**원리**:
```
[L0 routing(eager)] → [L0 MoE+ATTN+AR(graph)] → [L1 routing(eager)] → ...
     ↑ 정확한                                          ↑ 정확한
   expert 목록                                       expert 목록
```

**분석**:
- **100% 정확도** (구조적으로 miss 불가)
- per-layer CPU work (~77us) < per-layer GPU time (~146us) → 파이프라이닝 가능

```
GPU:  [L0 graph 146us][L1 graph 146us][L2 graph 146us]...
CPU:  [            ][L0 routing 77us][L1 routing 77us]...
                     ↑ GPU L0 실행 중 CPU가 L0 결과 읽고 L1 준비
```

**비용 breakdown**:

| 항목 | 비용 |
|------|:---:|
| 48 graph launch (vs 1) | ~480us |
| 48 eager select_experts | ~960us |
| sync/transition | ~200us |
| **Total extra** | **~1,640us** |
| **예상 TPOT** | **~8.6ms (+23%)** |

구현 복잡도 높음: vLLM `CUDAGraphWrapper` 분할, per-layer graph capture 인프라 필요.

### 3-E: Eager Fallback on Any Miss

**개요**: 기존 구현의 threshold를 `miss_count > 0`으로 강화.

**원리**:
```python
# gpu_model_runner.py (이미 존재하는 코드)
if miss_count > 0:  # 현재: miss_ratio > 0.95
    cudagraph_mode = CUDAGraphMode.NONE  # 이번 step만 eager
```

**분석**:
- Miss 발생 step만 eager (~90ms) → 해당 step은 100% 정확
- 3-A/3-B와 조합: miss rate가 0.1% 이하로 떨어지면 eager 발동 극소
- 단독 사용 시 문제: c=16에서 초기 2% miss → 대부분 step이 eager → TPOT 급증

**비용**: 구현 trivial (threshold 값 변경 1줄). TPOT는 miss rate에 비례.

---

## 4. 방안 비교

| 방안 | 예상 Hit Rate | TPOT Overhead | 구현 난이도 | 구현량 |
|------|:---:|:---:|:---:|:---:|
| 현재 v7 (baseline) | 99.38% (최종) | 0 | 완료 | — |
| **3-A** Never-evict recent | ~99.5-99.7% | **0** | **낮음** | ~20줄 |
| **3-B** Multi-step union | ~99.5-99.7% | **0** | **낮음** | ~30줄 |
| **3-C** Layer 0 eager | Layer 0: 100% | **+1.4%** | 중간 | ~80줄 |
| **3-D** Per-layer piecewise | **100%** | **+23%** | 높음 | ~300줄 |
| **3-E** Eager fallback (miss>0) | 100% per step | miss율 비례 | **trivial** | 1줄 |

---

## 5. 권장 조합

### Phase 1: 소프트웨어 최적화 (graph 변경 없음)

```
3-A (Never-evict) + 3-B (Multi-step union) + 3-E (Eager fallback miss>0)
```

**기대 효과**:
- 3-A + 3-B: miss rate 99.38% → ~99.8%+ (eviction 안정화 + prediction 확장)
- 3-E: 남은 ~0.2% miss step만 eager → 평균 TPOT 영향 미미
- **결과: 100% accuracy, TPOT ~7.0ms + (0.2% × 83ms) ≈ 7.2ms**
- 구현: ~50줄, graph 구조 변경 없음

### Phase 2: Layer 0 분리 (선택적)

```
Phase 1 + 3-C (Layer 0 eager routing)
```

**추가 효과**:
- Layer 0의 95.85% → 100% (가장 volatile layer 완전 해결)
- Layer 0 실제 routing을 L1-47 prediction에 활용 → 전체 miss rate 추가 감소
- Eager fallback 발동 빈도 더욱 감소
- **결과: ~99.9%+, TPOT ~7.1ms**

### Phase 3: 구조적 100% (필요 시)

```
3-D (Per-layer piecewise CUDA graph)
```

**효과**:
- 구조적으로 miss 불가 (실시간 routing → 즉시 fetch → 실행)
- Prediction, eviction policy 최적화 불필요
- **결과: 100%, TPOT ~8.6ms (+23%)**
- Phase 1-2의 결과가 충분하면 불필요

---

## 6. 미결 검증 항목

Phase 1-2 구현 전에 실측이 필요한 데이터:

| 항목 | 방법 | 목적 |
|------|------|------|
| Per-step routing Jaccard similarity | routing snapshot 연속 비교 | N-1 prediction 정확도 직접 측정 |
| 3-A 적용 후 hit rate | K=5,10,20 sweep | never-evict의 실제 효과 |
| 3-B 적용 후 hit rate | history=2,3,5 sweep | multi-step union의 실제 효과 |
| Layer 0 routing vs L1-47 상관관계 | cross-layer Jaccard | Layer 0 eager의 다른 layer 개선 효과 |
| Eager fallback 발동 빈도 (3-A+3-B 후) | miss_count>0 카운트 | 평균 TPOT 영향 추정 |

---

## 7. Appendix: CUDA Graph Replay 내부 동작

### 7.1 Graph가 실행하는 것 (per layer)

```
GPU (단일 replay 안에서 순차 실행, Python 없음):
├─ Layer L:
│   ├─ select_experts kernel    (topk routing, GPU-only 연산)
│   ├─ _routing_snapshot.copy_() (다음 step용 라우팅 캡처, GPU→GPU)
│   ├─ expert_map = _cache_map 읽기 (pre_step이 미리 써놓은 값)
│   ├─ MoE GEMM kernel (w13[slot] × hidden_state)
│   ├─ activation kernel (SiLU)
│   ├─ MoE GEMM kernel (w2[slot] × intermediate)
│   ├─ AllReduce (TP: GPU간 결과 합산)
│   ├─ ATTN kernel (Q·K^T, softmax, ×V)
│   ├─ ATTN output GEMM
│   └─ AllReduce
```

### 7.2 pre_step이 실행하는 것

```
pre_step(3.7ms) — graph replay 전, CPU에서 실행:

Phase A — GPU→GPU gather (942us, 25.5%)
  48 layers의 _routing_snapshot을 stacked GPU buffer로 수집

Phase B — Bulk D2H (410us, 11.1%)
  GPU→CPU 한 번에 전송, pinned CPU memory 착지

Phase C — CPU classify (1,957us, 52.9%)
  ├─ 각 layer별 hit/miss 분류
  ├─ miss expert → async DMA 큐잉 (CPU pinned → GPU slot)
  └─ cache_map scratch 업데이트

Phase C2 — Batched GPU upload (371us, 10.0%)
  pinned CPU → GPU bulk copy, layer._cache_map.copy_()
```

### 7.3 Overlap 구조

```
CPU:  [sched][───── pre_step 3.7ms ─────][graph launch][idle 3.3ms]
GPU:  [이전 step 마무리]────────────────[graph replay 7.0ms──────────]
GPU copy_st: ───────[DMA 0.3ms]──→                (miss expert 비동기 전송)
                    ←──────────── TPOT ≈ 7.0ms ──────────────→

TPOT = max(CPU_total, GPU_graph) + sync
     = max(4ms, 7ms) + α ≈ 7ms

pre_step(3.7ms) < graph(7ms) → pre_step은 완전히 숨겨짐 ("fully hidden")
```
