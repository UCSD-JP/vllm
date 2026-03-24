# Elastic KV + Prefix Protection — V4 Design Document

**Base**: vLLM 0.15.1 (pinned)
**Branch**: `elastic-kv-mvp`
**Date**: 2026-03-24

---

## 1. Overview

Elastic KV는 MoE(Mixture of Experts) 모델에서 expert weight 메모리와 KV cache 메모리를
동적으로 재분배하는 시스템이다. CUDA VMM(Virtual Memory Management)을 활용하여 expert
페이지를 해제하고 KV cache 블록으로 전환한다.

Prefix Protection은 KV cache 확장 시 prefix cache(재사용 가능한 캐시 블록)를 보호하여
불필요한 cache miss를 방지하는 비용 기반 의사결정 모델이다.

### Version History

| Version | 핵심 변경 |
|---------|----------|
| MVP | Expert→KV 단방향 확장, VMM 기반 |
| V3 | 3-way cost model (Ce/Cc/Cp), split queues, JSONL trace |
| V3.1 | CallerKind 분리, dynamic p_reuse, recency age tracking |
| V3.2 | Work-conserving (DEFER 제거), PREEMPT waiting 허용 |
| V3.2c | Routing snapshot dormant fix, step-based recency age |
| V4 | Traffic-bound Ce model (rho × top_k × c_reload), num_layers/l_sync 제거 |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Scheduler                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  _prefix_protection_try_allocate()                   │    │
│  │    ├─ CallerKind.RUNNING (while-loop, preempt OK)   │    │
│  │    └─ CallerKind.WAITING (single-shot, skip on fail)│    │
│  │                                                      │    │
│  │  decide() → USE_UNCACHED / PROTECT / RECLAIM / PREEMPT │ │
│  │    Ce = expert eviction cost                         │    │
│  │    Cc = cached block reclaim cost (dynamic p_reuse)  │    │
│  │    Cp = request preemption cost (partial-tail model)  │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  KVCacheManager                                      │    │
│  │    try_allocate(alloc_mode) → AllocationAttempt       │    │
│  │    allocate_slots(alloc_mode) → KVCacheBlocks         │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  BlockPool                                           │    │
│  │    free_uncached_queue ◄── 해시 없음 (재사용 불가)    │    │
│  │    free_cached_queue   ◄── 해시 있음 (prefix 재사용)  │    │
│  │    alloc_mode: uncached_only / uncached_then_cached / any│ │
│  │    advance_step() → step-based recency age            │    │
│  │    front_step_age → LRU front block 나이              │    │
│  └──────────────────────┬──────────────────────────────┘    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  _try_elastic_kv_expand(deficit)                     │    │
│  │    handler(min_blocks, max_blocks) → added_blocks     │    │
│  │    Expert pages → KV blocks (VMM cuMemMap)            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     GPU Worker                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  GPUModelRunner.execute_model()                      │    │
│  │    expert_cache.pre_step()  → expert eviction/reload  │    │
│  │    vmm_pool.mark_step_start()                         │    │
│  │    _snapshot_active = has_evicted_experts()            │    │
│  │    model.forward() → FusedMoE.forward_cuda()          │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  FusedMoE Layer                                       │    │
│  │    _snapshot_active: bool  (per-step, model_runner set)│   │
│  │    if _snapshot_active:                                │    │
│  │      torch.topk → _routing_snapshot (expert prediction)│   │
│  │    else: skip (dormant, no overhead)                   │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Dual CUDA Graph Dispatch                             │    │
│  │    offload_active=True  → graph with snapshot ops     │    │
│  │    offload_active=False → clean graph (dormant)       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Cost Model (Prefix Protection V4)

### 3.1 Ce — Expert Eviction Cost (V4 Traffic-Bound DMA Model)

KV 확장을 위해 expert를 evict하면, decode 중 해당 expert가 routing될 때 DMA reload stall 발생.

**V4 모델**: Expert reload는 DMA I/O → stall ∝ miss count (linear). Per-token amortized.

```
rho               = (groups_to_evict × G) / E   — evicted fraction
stall_per_token   = top_k × rho × C_reload      — expected DMA per token (ms)
Ce                = stall_per_token × H_eff × 1000   (ms → µs)
```

**Batch cancellation**: Per-step stall = batch × per-token stall.
Per-request share = stall / batch. Batch cancels out → per-token model is already amortized.

- `E`: local expert 수 (e.g. 512)
- `G`: group size (e.g. 2)
- `top_k`: experts routed per token (e.g. 10)
- `H_eff`: min remaining decode steps, clamped to [1, H_cap]
- `can_fully_protect=False` → Ce = ∞ (partial expand 방지)

예시: `groups=4, G=2, E=512, top_k=10, c_reload=0.63ms, h_eff=10`
→ rho = 8/512 = 0.015625, stall = 10 × 0.015625 × 0.63 = 0.098ms/tok
→ Ce = 0.098 × 10 × 1000 = 984 µs

### 3.2 Cc — Cached Block Reclaim Cost

Prefix cache 블록을 회수하면 향후 동일 prefix 요청 시 re-prefill 필요.

```
Cc = p_eff × (touched_blocks × block_size × t_prefill_tok_us + t_sched + t_queue)
```

**Dynamic p_reuse**: `p_eff = p_reuse_alpha × hit_rate` (sliding window 500 requests)

**Recency age decay**: LRU front 블록의 step age가 높으면 재사용 확률이 낮음.
```
p_eff = p_base / (1 + recency_age / cc_age_scale)
```

| recency_age | p_eff / p_base | 의미 |
|-------------|----------------|------|
| 0 | 1.00 | 방금 freed (hot) |
| 2,000 | 0.50 | half-life |
| 14,000 | 0.125 | Ce-competitive |

### 3.3 Cp — Request Preemption Cost (V3.1 Partial-Tail Model)

Preemption은 full blocks → cached queue (Cc가 처리), partial tail → uncached queue.
Uncached deficit 해소에는 partial tail만 기여.

```
partial_tail = victim_computed_tokens % block_size
if partial_tail == 0: Cp = ∞  (useless preemption)
else: Cp = t_sched + t_queue + partial_tail × t_prefill_tok_us
```

**Victim selection**: Scheduler policy와 동일한 victim 사용 (FIFO → `running[-1]`, Priority → max).

### 3.4 Decision Logic

```python
def decide(h_eff, groups_to_evict, touched_cached_blocks,
           preempt_computed_tokens, ..., caller):
    if touched_cached_blocks == 0:
        return USE_UNCACHED

    Ce = compute_ce(...)       # ∞ if can't fully protect
    Cc = compute_cc(...)       # dynamic p_reuse + age decay
    Cp = compute_cp_running(...)  # partial-tail model

    min_cost = min(Ce, Cc, Cp)
    if min_cost == Ce: return PROTECT_AND_EXPAND
    if min_cost == Cc: return RECLAIM_CACHED
    # Cp is cheapest:
    if caller == RUNNING:    return PREEMPT
    if caller == WAITING:
        if running > 0:      return PREEMPT    # work-conserving
        else:                 return RECLAIM_CACHED  # no victim → reclaim
```

**Work-conserving**: WAITING path는 항상 progress (DEFER 없음).

---

## 4. Split Block Queues

Vanilla vLLM은 단일 `free_block_queue`를 사용. Elastic KV는 두 개의 queue로 분리:

| Queue | 내용 | Allocation 우선순위 |
|-------|------|-------------------|
| `free_uncached_queue` | 해시 없음, 재사용 불가 | 항상 먼저 소비 |
| `free_cached_queue` | 해시 있음, prefix 재사용 가능 | 보호 대상, 필요 시만 회수 |

### Allocation Modes

| Mode | 동작 | 사용처 |
|------|------|--------|
| `uncached_only` | uncached queue에서만 할당 | PP try (step 1) |
| `uncached_then_cached` | uncached 우선, 부족 시 cached | RECLAIM_CACHED decision |
| `any` | uncached 우선, 부족 시 cached | PP OFF fallback |

### Step-Based Recency Age

블록이 cached queue에 append될 때 `_cached_free_step = current_step` 기록.
`front_step_age = current_step - front_block._cached_free_step`.

이점:
- Block churn 없이도 age 증가 (livelock 탈출)
- `advance_step()`: scheduler round마다 호출

Hit/Evict age 별도 추적:
- `touch()` → `_record_recency_age(age, kind='hit')`
- `_maybe_evict_cached_block()` → `_record_recency_age(age, kind='evict')`

---

## 5. Routing Snapshot (Dormant Fix)

Expert cache가 dormant 상태 (모든 expert가 resident)일 때, `forward_cuda()`의
`torch.topk` routing snapshot이 불필요.

**Fix**: `_snapshot_active: bool` 플래그를 model_runner가 매 step 설정:
- `execute_model()`: `snap = expert_cache.has_evicted_experts()`
- `_dummy_run()`: `snap = forced_batch_desc.offload_active` (CUDA graph capture)

→ Dormant 시 snapshot 연산 스킵.

---

## 6. File Inventory

### Modified Files (11)

| File | 변경 내용 |
|------|----------|
| `vllm/forward_context.py` | `BatchDescriptor.offload_active` field |
| `vllm/model_executor/layers/fused_moe/layer.py` | `_snapshot_active`, dormant skip in `forward_cuda()` |
| `vllm/v1/core/block_pool.py` | Split queues, alloc_mode, step-based recency age, `advance_step()`, `front_step_age` |
| `vllm/v1/core/kv_cache_coordinator.py` | AllocationPlan: `protection_gap`, `touched_cached_blocks` |
| `vllm/v1/core/kv_cache_manager.py` | `try_allocate()` → AllocationAttempt wrapper, `alloc_mode` 전달 |
| `vllm/v1/core/sched/scheduler.py` | Prefix protection decision, `_try_elastic_kv_expand()`, JSONL trace, CachingMetrics, CallerKind |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `alloc_mode` 전달 |
| `vllm/v1/cudagraph_dispatcher.py` | Dual CUDA graph key (`offload_active` axis) |
| `vllm/v1/engine/core.py` | `ElasticKVConfig` init + handler registration |
| `vllm/v1/worker/gpu_model_runner.py` | Expert cache pre_step, `_snapshot_active` setting, VMM pool |
| `vllm/v1/worker/gpu_worker.py` | `ElasticKVConfig` compute_derived, Ce params propagation |

### New Files (5)

| File | 내용 |
|------|------|
| `vllm/elastic_kv_config.py` | ElasticKVConfig dataclass, env parsing, VMM geometry |
| `vllm/model_executor/layers/fused_moe/expert_cache.py` | ExpertCacheManager: eviction/reload, routing prediction |
| `vllm/model_executor/layers/fused_moe/expert_predictor.py` | Expert routing prediction (offline/online) |
| `vllm/v1/core/prefix_protect.py` | PrefixProtectionConfig, CallerKind, ProtectionDecision, cost model |
| `vllm/vmm_pool.py` | VMMPagePool: CUDA VMM page management, 2-phase commit |

### Test Files (2)

| File | Tests |
|------|-------|
| `tests/v1/core/test_prefix_protection.py` | 1073 lines, Ce/Cc/Cp model, CallerKind, decisions |
| `tests/v1/core/test_elastic_kv.py` | 772 lines, page math, expand, recency age |

---

## 7. Environment Variables

### Elastic KV (core)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ELASTIC_KV_ENABLE` | `0` | Enable expert→KV expansion |
| `VLLM_ELASTIC_KV_MIN_RESIDENT` | `0.5` | Min fraction of expert pages to keep |
| `VLLM_ELASTIC_KV_GROUPS_PER_EXPAND` | `4` | Expert groups per expand call |
| `VLLM_ELASTIC_KV_EVICT_POLICY` | `lru` | Expert eviction policy |
| `VLLM_ELASTIC_KV_MAX_EXPAND_BLOCKS` | `0` | Hard cap (0=auto from min_resident) |
| `VLLM_ELASTIC_KV_TRACE` | `` | JSONL trace file path |

### Prefix Protection (cost model)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_PREFIX_PROTECTION_ENABLE` | `0` | Enable prefix protection |
| `VLLM_PP_H_CAP` | `64` | H_eff cap (max remaining decode steps) |
| `VLLM_PP_C_RELOAD_MS` | `0.63` | Expert reload latency (ms, topology dep.) |
| `VLLM_PP_T_PREFILL_TOK` | `15.0` | Per-token prefill cost (µs) |
| `VLLM_PP_P_REUSE` | `1.0` | Static p_reuse (fallback when alpha=0) |
| `VLLM_PP_P_REUSE_ALPHA` | `0.5` | Dynamic scaling: `p = alpha × hit_rate` |
| `VLLM_PP_CC_AGE_SCALE` | `2000.0` | Recency age half-life for Cc decay |
| `VLLM_PP_T_SCHED` | `500.0` | Scheduler overhead (µs) |
| `VLLM_PP_T_QUEUE` | `1000.0` | Queue overhead (µs) |
| `VLLM_PP_TOP_K` | `10` | Top-k for Ce calculation |
| `VLLM_PP_GROUP_SIZE` | `2` | Expert group size |
| `VLLM_PP_NUM_EXPERTS` | `512` | Local expert count |
| `VLLM_PP_T_RECOMPUTE` | `0.260` | Per-token recompute cost (ms) |
| `VLLM_PP_BLOCK_SIZE` | `16` | Tokens per block |
| `VLLM_PP_DIAG_INTERVAL` | `100` | Diagnostic log interval (steps) |

---

## 8. Deployment

### Site-Packages Method (Required)

PYTHONPATH 방식은 +60% TPOT regression을 유발한다 (import resolution overhead).
반드시 site-packages에 직접 복사해야 한다.

```bash
# 타겟 서버에서
SITE=$(python -c "import vllm; print(vllm.__path__[0])")

# 수정 파일 복사 (dev 머신에서 scp)
scp vllm/forward_context.py                            target:$SITE/forward_context.py
scp vllm/elastic_kv_config.py                          target:$SITE/elastic_kv_config.py
scp vllm/vmm_pool.py                                   target:$SITE/vmm_pool.py
scp vllm/model_executor/layers/fused_moe/layer.py      target:$SITE/model_executor/layers/fused_moe/layer.py
scp vllm/model_executor/layers/fused_moe/expert_cache.py    target:$SITE/model_executor/layers/fused_moe/expert_cache.py
scp vllm/model_executor/layers/fused_moe/expert_predictor.py target:$SITE/model_executor/layers/fused_moe/expert_predictor.py
scp vllm/v1/core/block_pool.py                         target:$SITE/v1/core/block_pool.py
scp vllm/v1/core/kv_cache_coordinator.py               target:$SITE/v1/core/kv_cache_coordinator.py
scp vllm/v1/core/kv_cache_manager.py                   target:$SITE/v1/core/kv_cache_manager.py
scp vllm/v1/core/prefix_protect.py                     target:$SITE/v1/core/prefix_protect.py
scp vllm/v1/core/single_type_kv_cache_manager.py       target:$SITE/v1/core/single_type_kv_cache_manager.py
scp vllm/v1/core/sched/scheduler.py                    target:$SITE/v1/core/sched/scheduler.py
scp vllm/v1/cudagraph_dispatcher.py                    target:$SITE/v1/cudagraph_dispatcher.py
scp vllm/v1/engine/core.py                             target:$SITE/v1/engine/core.py
scp vllm/v1/worker/gpu_model_runner.py                 target:$SITE/v1/worker/gpu_model_runner.py
scp vllm/v1/worker/gpu_worker.py                       target:$SITE/v1/worker/gpu_worker.py
```

### Rollback

```bash
pip install vllm==0.15.1 --force-reinstall --no-deps
```

### Server Start

```bash
python -u -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 49152 --max-num-seqs 128 \
    --enable-prefix-caching \
    --trust-remote-code
```

Features ON:
```bash
export VLLM_ELASTIC_KV_ENABLE=1
export VLLM_PREFIX_PROTECTION_ENABLE=1
export VLLM_PP_H_CAP=64
export VLLM_PP_P_REUSE_ALPHA=0.5
export VLLM_PP_CC_AGE_SCALE=2000
```

---

## 9. Benchmark Results (V3.2c, c=32 t=12)

Qwen3-Next-80B-A3B, 2×H100 PCIe, site-packages deployment.

| Metric | Vanilla | Features ON | Delta |
|--------|---------|-------------|-------|
| TPOT | 80.4 ms | 75.7 ms | -6% |
| TPS | 260.1 | 342.3 | +32% |
| Prefix Cache Hit | 47.8% | 78.0% | +30.2pp |
| T11 TTFT | 65.8 s | 8.0 s | 8× faster |

---

## 10. JSONL Trace Format

`VLLM_ELASTIC_KV_TRACE=/path/to/trace.jsonl` 설정 시 매 decision마다 기록.

```json
{
  "ts": 1711234567.89,
  "required_blocks": 3,
  "uncached_free": 1200,
  "cached_free": 800,
  "touched_cached_blocks": 2,
  "protection_gap": 2,
  "allocation_gap": 0,
  "expanded_total": 64,
  "h_eff": 32,
  "groups_est": 1,
  "n_decode": 28,
  "n_prefill": 4,
  "Ce": 8155.0,
  "Cc": 1260.0,
  "Cp": 1515.0,
  "can_fully_protect": true,
  "remaining_cap": 192,
  "decision": "reclaim_cached",
  "preempt_tokens": 49,
  "caller": "running",
  "hit_rate": 0.72,
  "p_reuse_eff": 0.36,
  "recency_age": {"hit": {"mean": 450, "max": 3200, "count": 120}, "evict": {"mean": 8500, "max": 15000, "count": 45}},
  "cfg": {"E": 512, "G": 2, "top_k": 10, "c_reload_ms": 0.63, "block_size": 16, ...}
}
```

---

## 11. Known Limitations (POC)

1. **Victim mismatch**: Cp estimation은 scheduler policy victim 기준이지만,
   multi-round preemption에서 victim이 변경될 수 있음
2. **One-way expansion**: Expert→KV만 지원. KV→Expert 역확장 미구현
3. **α auto-calibration**: `p_reuse_alpha`, `cc_age_scale` 수동 설정.
   Recency age 통계로 자동 보정 가능하나 미구현
4. **PYTHONPATH deployment 불가**: Import resolution overhead로 인한 TPOT regression.
   반드시 site-packages 직접 복사 필요

---

## 12. Test Verification

```bash
python -m pytest -q tests/v1/core/test_prefix_protection.py tests/v1/core/test_elastic_kv.py
```

Expected: 100+ passed (Ce/Cc/Cp model, CallerKind, decisions, recency age, page math, expand).
