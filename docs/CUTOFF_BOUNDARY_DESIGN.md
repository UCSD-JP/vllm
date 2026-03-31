# Cutoff Boundary: Deterministic Expert Offloading for MoE Inference

## 1. Problem Statement

MoE 모델(Mixtral, DeepSeek-V3 등)에서 expert weight는 GPU 메모리의 대부분을 차지한다.
KV cache 확장이 필요할 때 expert weight를 CPU로 offload하고, 해당 expert가 routing될 때
다시 GPU로 load하는 **dynamic offloading**이 기본 접근이다.

기존 LRU 기반 dynamic path의 문제:

| 단계 | 연산 | Overhead |
|------|------|----------|
| Routing snapshot D2H | `topk_ids.unique().cpu()` | GPU→CPU sync (~5μs) |
| Hit/miss 분류 | Python set 연산 per-layer | ~10-50μs/layer |
| Miss fetch | `copy_stream.synchronize()` | **Blocking** H2D (수 ms) |
| Cache map rebuild | `_update_cache_map()` per-layer | CPU tensor 생성 + H2D copy |
| LRU bookkeeping | `_last_access` 갱신 | numpy scatter per-layer |
| **총 per-step cost** | 48 layers × 위 전부 | **O(layers × experts)** |

핵심 병목은 **reactive fetch**: miss가 발생해야 H2D를 시작하므로,
kernel이 H2D 완료를 기다려야 한다 (overlap 불가능).

## 2. Key Insight: Monotonic Shrink + Uniform Topology

Elastic KV의 shrink는 **단조 감소(monotonic decrease)**: 한 번 줄어든 resident 수는
다시 늘어나지 않는다 (grow 연산이 없는 one-way 전환).

이 특성을 활용하면:

1. Expert를 `[0, cutoff)` (resident) / `[cutoff, local_E)` (tail, CPU) 로 **정적 분할**
2. Tail은 **모든 layer에서 동일** (L1-L47, L0는 항상 전체 resident)
3. Cache map은 cutoff 변경 시에만 rebuild (매 step이 아니라 매 shrink)
4. 다음 layer의 tail을 **현재 layer kernel과 동시에 prefetch** (double-buffer)

이것이 Cutoff Boundary(CB)의 핵심이다.

## 3. Architecture

```
            ┌─────────────────────────────────────────────┐
            │              Expert Weight Space              │
            │                                               │
            │  ┌───────────────┬──────────────────────┐    │
            │  │  Resident      │      Tail (CPU)       │    │
            │  │  [0, cutoff)   │  [cutoff, local_E)    │    │
            │  │  GPU weight    │  Scratch bank via     │    │
            │  │  tensor에 존재  │  H2D prefetch         │    │
            │  └───────────────┴──────────────────────┘    │
            │        ↑ shrink: cutoff 감소                   │
            │        │ (monotonic, 복구 없음)                │
            └─────────────────────────────────────────────┘

  Forward 흐름 (L_i → L_{i+1}):

    L_i kernel (compute stream)     L_{i+1} H2D (copy stream)
    ┌──────────────────────┐       ┌─────────────────────┐
    │  wait(bank_A.ready)  │       │                     │
    │  MoE kernel          │       │  bank_B ← CPU tail  │
    │  release(bank_A)     │       │  record(ready_event) │
    └──────────────────────┘       └─────────────────────┘
              ↓                              ↓
    L_{i+1} kernel                  L_{i+2} H2D
    ┌──────────────────────┐       ┌─────────────────────┐
    │  wait(bank_B.ready)  │       │                     │
    │  MoE kernel          │       │  bank_A ← CPU tail  │
    │  release(bank_B)     │       │  record(ready_event) │
    └──────────────────────┘       └─────────────────────┘
```

## 4. CB vs LRU: Per-Step Overhead 비교

### 4.1 LRU Dynamic Path (기존)

매 step마다 **모든 layer**에 대해:

```
pre_step():
  for each layer (48회):
    routing_snapshot.unique().cpu()     ← GPU→CPU sync, ~5μs
    hit/miss 분류 (Python set ops)     ← ~10-50μs
    if miss:
      _async_fetch(miss_lids)          ← H2D enqueue
      copy_stream.synchronize()        ← BLOCKING wait (수 ms)
      _cache_map_needs_rebuild = True
  if rebuild:
    for each layer: _update_cache_map() ← CPU tensor alloc + GPU copy

forward_cuda() per-layer:
  (no prefetch, reactive only)
  kernel runs with whatever is in GPU
```

**Cost**: `O(L × E)` Python ops + blocking H2D sync + cache map rebuild
**Critical path**: miss → sync → kernel (직렬)

### 4.2 Cutoff Boundary (본 설계)

pre_step(): **topology dirty일 때만** (shrink 발생 시)

```
pre_step():
  if _topology_dirty:                  ← shrink가 cutoff를 낮춘 경우만
    _cutoff_recompute_topology()       ← list(range(cutoff, E)), O(1)
    _cutoff_rebuild_cache_maps()       ← 1회 full rebuild
    _topology_dirty = False
  return immediately                   ← 매 step O(1)

forward_cuda() per-layer:
  scan banks → READY + owner match    ← O(2) bank scan
  READY → IN_USE                      ← state flip
  wait(bank.ready_event)              ← non-blocking if H2D finished
  cutoff_prefetch_next(layer_idx)     ← OVERLAP: L_{i+1} H2D on copy_stream
  kernel(scratch_w13, scratch_w2)     ← kernel ∥ next H2D
  release_scratch(bank)               ← record done_event, state → IDLE
```

**Cost**: `O(1)` per step (no routing decode, no LRU, no miss classify)
**Critical path**: prefetch(L_{i+1}) ∥ kernel(L_i) (병렬)

### 4.3 왜 CB가 빠른가

| 요소 | LRU | CB | 근거 |
|------|-----|-----|------|
| Routing D2H | 매 step × 48 layers | 없음 | 정적 topology → routing 무관 |
| `.item()` / `.cpu()` sync | 매 miss 시 | 없음 (forward에서 0회) | cache map이 정적 |
| Hit/miss 분류 | Python set ops ×48 | 없음 | tail은 미리 결정됨 |
| H2D timing | miss 감지 **후** reactive | 이전 layer kernel **중** proactive | double-buffer overlap |
| Cache map rebuild | miss마다 per-layer | shrink 시 1회 | cutoff 변경 빈도 ≪ step 빈도 |
| `copy_stream.synchronize()` | 매 miss | 없음 | `wait_event`로 대체 (non-blocking) |
| LRU bookkeeping | `_last_access` 매 hit | 없음 | CB는 LRU 사용 안 함 |
| Bank state machine | 1 bank, miss-driven | 2 bank, prefetch-driven | double-buffer rotation |

**정량적 차이 (실측, DeepSeek-V3 TP2 H200)**:

```
LRU dynamic (eager):  TPOT p50 ~168ms, hit path ~120μs/layer
CB (cutoff=384):      TPOT p50 ~128ms, pre_step ~0μs (O(1) return)
```

핵심: CB의 **per-step CPU overhead는 거의 0**. LRU는 miss가 없어도 routing decode + LRU update가 발생.

## 5. Implementation Details

### 5.1 Precondition Verification (`_cutoff_verify_once`)

CB 활성화 전 1회 검증 (이후 결과 캐시):

1. **Identity mapping**: `lid == slot` for all (layer, expert)
   - 이유: CB는 suffix-cut으로 evict하므로, slot 할당이 identity여야 cutoff 기준 분할이 유효
2. **Pinned guard**: pinned expert가 resident 범위 안에 있는지
3. **CPU pool completeness**: L1-L47 모든 expert의 CPU backing 존재

실패 시 → LRU fallback (permanent, 재시도 없음).

### 5.2 Bank State Machine

```
IDLE ──(prefetch)──→ FILLING ──(ready_event)──→ READY
                                                  │
                                          (forward_cuda scan)
                                                  │
                                                  ↓
                                               IN_USE ──(release)──→ IDLE
```

- **IDLE**: 빈 bank, prefetch 가능
- **FILLING**: copy_stream에서 H2D 진행 중
- **READY**: H2D 완료, ready_event 기록됨, forward 대기
- **IN_USE**: kernel이 이 bank의 weight를 사용 중

Invariant: forward는 READY→IN_USE만 수행, prefetch는 IDLE→FILLING만 수행.

### 5.3 Double-Buffer Rotation

```python
# L0 (resident-only) → prefetch L1 into bank[0]
cutoff_prefetch_next(0)  →  bank[0].state = READY, owner = L1

# L1 forward: consume bank[0], prefetch L2 into bank[1]
forward_cuda(L1):
  bank[0]: READY → IN_USE
  cutoff_prefetch_next(1) → bank[1]: IDLE → READY, owner = L2
  kernel(scratch=bank[0])
  release(bank[0]) → IDLE

# L2 forward: consume bank[1], prefetch L3 into bank[0]
forward_cuda(L2):
  bank[1]: READY → IN_USE
  cutoff_prefetch_next(2) → bank[0]: IDLE → READY, owner = L3
  kernel(scratch=bank[1])
  release(bank[1]) → IDLE

# ... 48 layers: 0↔1 alternation
```

### 5.4 Uniform Tail (L1-L47 동일)

CB의 핵심 단순화: **모든 non-L0 layer가 동일한 tail을 공유**.

```python
_cutoff_tail_lids = list(range(cutoff, local_num_experts))
```

이유:
- Shrink가 uniform: `_cutoff_shrink()`는 L1-L47에서 동일한 `[new_cutoff, old_cutoff)` evict
- Identity mapping 전제: lid X가 evict되면 모든 layer에서 동시에 evict
- Prefetch 단순화: 매 layer마다 동일한 `tail_lids`를 bank에 load

### 5.5 Shrink 정책: Suffix-Cut

```python
# cutoff 30 → 28: lids [28, 29]를 L1-L47에서 evict
new_cutoff = _resident_cutoff - steps_needed * group_size
floor = max(0, local_num_experts - scratch_capacity)
new_cutoff = max(new_cutoff, floor)  # tail이 scratch에 들어가야 함
```

- **Steps**: `ceil(min_pages / (n_layers × group_pages))`
- **Floor guard**: tail size ≤ scratch_capacity
- **단조**: cutoff는 감소만 함 (grow 없음)

### 5.6 Cache Map 구조

```
cache_map[gid]:
  L0:  항상 resident slot (gid → expert_to_slot[0][lid])
  L1-L47:
    lid < cutoff  → expert_to_slot[li][lid]   (resident)
    lid ≥ cutoff  → threshold + (lid - cutoff) (scratch slot)
    lid not local → -1
```

`threshold = max_resident` = resident slot 수. Scratch slot은 threshold부터 시작.
Kernel은 `expert_map[gid] >= threshold`이면 scratch tensor에서 읽음.

### 5.7 Static Cache Map → No Restore

LRU path에서는 `release_scratch()`가 cache_map을 원복해야 함
(miss 시 임시로 scratch slot을 넣었으므로).

CB에서는 cache_map이 **정적**: shrink 후 rebuild된 map이 다음 shrink까지 유효.
따라서 `release_scratch()`는:

```python
if self._cutoff_active:
    bank.done_event.record()
    bank.state = _BankState.IDLE
    return  # no cache_map restoration
```

이것만으로도 release path에서 `index_fill_` (GPU scatter) 1회를 절약.

### 5.8 TP Gate: TP≤2 전용

CB는 **kernel execution 중 H2D overlap**에 의존한다.
TP4에서는 shard가 작아 kernel 시간이 짧고, overlap window가 부족하다:

```
TP2: kernel ~8.8ms/layer → H2D 5.6ms 숨기기 가능
TP4: kernel ~4.4ms/layer → H2D 5.6ms > kernel → wait 발생
```

따라서 TP4+에서는 CB를 비활성화하고 dynamic fallback 사용:

```python
if _CUTOFF_BOUNDARY and self._tp_size <= 2:
    # CB path
else:
    # LRU dynamic path
```

Gate 위치 (4곳 일관 적용):
- `count_evictable_groups()`
- `shrink_for_pages()`
- `pre_step()` cutoff activation
- `pre_step()` step-boundary activation

## 6. Correctness Invariants

1. **cutoff는 단조 감소**: `new_cutoff < old_cutoff` always
2. **L0 불변**: L0는 항상 전체 resident (cutoff 무관)
3. **Floor guard**: `tail_size = local_E - cutoff ≤ scratch_capacity`
4. **Bank IDLE invariant**: prefetch는 IDLE bank에만 write, forward는 READY bank만 consume
5. **Identity mapping**: CB 활성화 전 `lid == slot` 검증 (1회)
6. **CPU pool**: L1-L47 모든 expert에 CPU backing 존재 (1회 검증)
7. **Mutual exclusion**: CB, step-boundary, fixed-tail은 상호 배타적

## 7. Configuration

| Env Variable | Default | Description |
|---|---|---|
| `VLLM_CUTOFF_BOUNDARY` | `0` | CB 활성화 |
| `VLLM_EXPERT_SCRATCH_CAPACITY` | `0` | Scratch bank 크기 (expert 수) |
| `VLLM_SCRATCH_BANKS` | `2` | Bank 수 (CB는 2 강제) |

## 8. Experimental Results (DeepSeek-V3, TP2, H200)

```
Vanilla (no offload):     TPS=86.8,  TPOT=109.2ms, TTFT=2850ms
CB (cutoff=384, c=32):    TPS=99.8,  TPOT=135.1ms, TTFT=1839ms
  - TPS:     +15.0%
  - TTFT:    -35.5%
  - TTFT p99: -64.4%
  - Preempt: 20 → 1
  - Prefix:  8.5% → 76.3%
```

TPOT 증가는 expert offload의 고유 비용 (H2D latency가 kernel에 완전히 숨겨지지 않음).
그러나 TTFT와 preemption 감소로 **사용자 체감 latency는 개선**.
