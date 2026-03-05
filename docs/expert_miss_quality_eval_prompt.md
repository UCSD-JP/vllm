# Expert Cache Miss — Quality Impact 측정 스크립트 작성 Prompt

## 배경

Expert offload v7에서 1-step-behind prediction으로 인한 cache miss가 존재한다.
실측 (c=16, SWE-bench, res=400): steady state hit rate = 99.38%, 초기 97.6%.

**목표**: miss가 모델 출력 품질에 미치는 영향을 정량화하여,
"이 정도 miss는 허용 가능" vs "반드시 100% 필요" 판단 근거를 확보.

## 관련 문서 (반드시 읽어야 할 것)

| 문서 | 경로 | 내용 |
|------|------|------|
| Hit rate 실측 | `docs/expert_cache_hit_analysis.md` | c=16 SWE-bench per-layer hit rate |
| Miss mitigation 설계 | `vllm/docs/expert_cache_miss_mitigation.md` | 5가지 방안, 비교표, 권장 조합 |
| v2 벤치마크 | `vllm/docs/expert_offload_v2_benchmark.md` | v5→v7 진화, timing, CUDA graph 설명 |
| v2 코드 리뷰 | `docs/expert_offload_v2_code_review.md` | expert_cache.py 코드 구조 |

## 핵심 코드 위치

| 파일 | 위치 | 내용 |
|------|------|------|
| `vllm/model_executor/layers/fused_moe/expert_cache.py` | L981-1035 | hit/miss 분류 + per-layer 카운터 |
| 같은 파일 | L1226-1339 | `get_miss_diagnostics()` — 기존 진단 출력 |
| 같은 파일 | L522-548 | timing + diag dict 초기화 |
| 같은 파일 | L759-823 | `_update_cache_map()` — cache_map 갱신 (miss injection point) |
| `vllm/v1/worker/gpu_model_runner.py` | pre_step 호출부 | miss_count → eager fallback 분기 |

## 기존 인프라 (활용 가능)

- `_diag['per_layer_hits/misses']`: per-layer hit/miss 카운터 (이미 누적 중)
- `_diag['per_layer_token_hits/misses']`: token-weighted per-layer 카운터
- `_diag['per_layer_unique_needed']`: layer별 unique expert 수
- `_gating_histogram`: expert별 routing frequency
- `VLLM_EXPERT_DIAG_DUMP=100`: 100 step마다 자동 출력
- `_routing_history`: multi-step union용 deque (3-B 구현 시 추가됨)

---

## 측정 #1: Token-Layer "Any Miss" 비율

### 목적

현재 metric은 "expert 단위 hit rate" (98.4%).
하지만 품질에 직접 영향을 주는 것은 **"토큰 하나가 어떤 layer에서든 한번이라도 miss를 겪었는가"**.

### 정의

```
per_token_any_miss_rate = (any miss가 있는 토큰 수) / (전체 토큰 수)

1개 토큰이 48 layers를 통과하면서 top-10 routing을 하면:
- 각 layer에서 miss 확률 = 1 - layer_hit_rate
- 48 layers 전부 hit할 확률 ≈ ∏(layer_hit_rate_i)
- any miss 확률 = 1 - ∏(layer_hit_rate_i)

예시 (실측 기반):
  Layer 0: 95.85%, Layer 11-35: ~98%, Layer 47: 98.19%
  48-layer 전부 hit 확률 ≈ 0.9585 × 0.98^46 × 0.9819 ≈ 0.37
  → any miss 확률 ≈ 63%  ← expert 단위 98%와 체감이 매우 다름
```

### 구현 방법 (expert_cache.py에 추가)

```python
# pre_step Phase C에서 (L981 부근), per-step token-layer miss 추적:

# 새 카운터 (_diag에 추가):
#   'tokens_with_any_miss': 0   (1개 layer라도 miss인 토큰 수)
#   'tokens_total': 0           (전체 토큰 수)

# 각 step에서:
# token_miss_flags = np.zeros(num_tokens, dtype=bool)  # per-token
# for layer_idx in range(48):
#     for each token's top-k routing:
#         if any expert in top-k has cache_map = -1:
#             token_miss_flags[token_idx] = True
# tokens_with_any_miss += token_miss_flags.sum()
# tokens_total += num_tokens
```

**이미 있는 것**: `topk_cpu` (per-layer routing), `miss_set` (per-layer miss experts).
**추가 필요**: per-token miss flag 집계 (topk_cpu와 miss_set의 intersection per token).

### 출력 형식

```
[Expert Cache Quality] Step 2000:
  Expert-level hit rate:     98.41%
  Token-level any-miss rate: 63.2%   ← 이것이 실제 품질 영향 proxy
  Avg miss layers per token: 2.1 / 48
```

---

## 측정 #2: Layer별 Miss 집중도 (가중 분석)

### 목적

Layer 0 miss 2%와 Layer 35 miss 2%는 품질 영향이 다르다:
- **Layer 0**: 입력 embedding 직후 → miss가 전체 forward propagation에 파급
- **Layer 47**: 출력 직전 → logit에 직접 영향
- **중간 layers**: residual connection이 miss 영향을 희석

### 구현 방법 (기존 데이터 활용)

```python
# 이미 있는 데이터: _diag['per_layer_misses'], _diag['per_layer_hits']
# 추가 계산만 하면 됨:

# 가중치 정의 (heuristic, 추후 #3의 KL 결과로 교정)
LAYER_WEIGHT = {
    0: 3.0,      # 입력 layer — 파급 효과 최대
    47: 2.0,     # 출력 layer — logit 직결
    # 나머지: 1.0
}

weighted_miss_rate = sum(
    LAYER_WEIGHT.get(i, 1.0) * per_layer_miss_rate[i]
    for i in range(48)
) / sum(LAYER_WEIGHT.get(i, 1.0) for i in range(48))
```

### 출력 형식

```
[Expert Cache Quality] Layer Miss Distribution:
  Layer 0:  miss=4.15%, weight=3.0 → contribution=12.45%
  Layer 1-46: miss=~2.0%, weight=1.0 → contribution=~2.0% each
  Layer 47: miss=1.81%, weight=2.0 → contribution=3.62%
  Weighted miss rate: 2.8% (vs unweighted 2.0%)
  Miss concentration: Layer 0 accounts for 25% of total weighted miss
```

---

## 측정 #3: 출력 Logit KL Divergence / Top-1 Flip Rate

### 목적

Miss가 최종 출력에 얼마나 영향을 주는지 **직접** 측정.
동일 입력에서 (A) 정상 실행 vs (B) miss 주입 실행을 비교.

### 실험 설계

```
Step 1: Normal run (offload ON, 모든 expert 정상)
  → logits_normal[vocab_size]

Step 2: Miss-injected run (동일 입력, 특정 layer에 miss 주입)
  → logits_miss[vocab_size]

비교:
  KL(normal || miss) = sum(p_normal * log(p_normal / p_miss))
  top1_flip = (argmax(logits_normal) != argmax(logits_miss))
```

### Miss 주입 방법 (expert_cache.py `_update_cache_map` 활용)

```python
# 환경변수로 miss injection 제어:
# VLLM_EXPERT_INJECT_MISS_RATE=0.02    (2% random miss)
# VLLM_EXPERT_INJECT_MISS_LAYERS=0,47  (특정 layer만)
# VLLM_EXPERT_INJECT_MISS_LOG=1        (logit 비교 모드)

# _update_cache_map() 끝에서 (L823 부근):
if self._inject_miss_rate > 0:
    mask = torch.rand(scratch.shape) < self._inject_miss_rate
    scratch[mask] = -1  # 인위적 miss 주입
```

### Logit 비교 구현 (gpu_model_runner.py에 추가)

```python
# execute_model() 끝에서:
if VLLM_EXPERT_INJECT_MISS_LOG:
    # 매 N step마다 2-pass:
    # Pass 1: normal cache_map → logits_A
    # Pass 2: injected cache_map → logits_B (같은 input_ids 재사용)

    probs_A = F.softmax(logits_A, dim=-1)
    probs_B = F.softmax(logits_B, dim=-1)

    kl_div = F.kl_div(probs_B.log(), probs_A, reduction='batchmean')
    top1_A = logits_A.argmax(dim=-1)
    top1_B = logits_B.argmax(dim=-1)
    top1_flip_rate = (top1_A != top1_B).float().mean()

    logger.info(f"[Miss Quality] KL={kl_div:.6f}, top1_flip={top1_flip_rate:.4f}")
```

### 실험 matrix

| 조건 | Miss Rate | Miss Layers | 목적 |
|------|:---------:|:-----------:|------|
| Baseline | 0% | — | 기준선 |
| 실측 수준 | 2% | 전체 | 현재 v7의 실제 영향 |
| Layer 0 only | 4% | Layer 0 | Layer 0 miss의 단독 영향 |
| Layer 47 only | 2% | Layer 47 | 출력 layer miss의 단독 영향 |
| 중간 layer only | 2% | Layer 23 | 중간 layer miss (대조군) |
| High miss | 5% | 전체 | 안전 마진 확인 |
| Extreme | 10% | 전체 | 파손 임계점 탐색 |

### 출력 형식

```
[Miss Quality Impact] Injection: rate=2%, layers=all
  KL divergence:    0.0023 (baseline=0, 높을수록 품질 저하)
  Top-1 flip rate:  0.3%   (다음 토큰 예측이 바뀌는 비율)
  Top-5 overlap:    99.8%  (상위 5개 토큰 집합 일치율)
  Mean logit delta:  0.015  (logit 절대값 평균 변화)
```

---

## 측정 #4: SWE-bench Score 트래킹

### 목적

#1-#3은 proxy metric. 최종적으로 **실제 task 성공률**에 영향이 있는지 직접 확인.

### 실험 설계

| Config | Description | 목적 |
|--------|-------------|------|
| A: Offload OFF | 전체 expert GPU 상주 (baseline) | 100% 정확도 기준선 |
| B: Offload ON (v7) | res=400, 1-step-behind | 현재 구현의 영향 |
| C: Offload ON + Phase 1 | res=400 + 3-A + 3-B + 3-E | mitigation 효과 |
| D: Offload ON + Phase 2 | + Layer 0 eager | 추가 개선 효과 |

### SWE-bench 실행

```bash
# MAB repo에서 실행 (multi-agent-bench)
# 각 config별 동일 task set으로 3회 반복

# Config A: Offload OFF
export VLLM_EXPERT_OFFLOAD_ENABLE=0
python run_swe_bench.py --tasks subset_50 --repeat 3

# Config B: Offload ON (v7 current)
export VLLM_EXPERT_OFFLOAD_ENABLE=1
export VLLM_EXPERT_MAX_RESIDENT=400
python run_swe_bench.py --tasks subset_50 --repeat 3

# Config C: + Phase 1
export VLLM_EXPERT_NEVER_EVICT_K=10
export VLLM_EXPERT_UNION_STEPS=3
export VLLM_EXPERT_EAGER_ON_ANY_MISS=1
python run_swe_bench.py --tasks subset_50 --repeat 3
```

### 트래킹 metric

```
| Config | Solve Rate | Avg Tokens | Avg TPOT | Hit Rate | Any-Miss% |
|--------|:----------:|:----------:|:--------:|:--------:|:---------:|
| A (OFF)     | X%  | Y    | Z ms  | 100%   | 0%   |
| B (v7)      | X%  | Y    | Z ms  | 99.4%  | ~63% |
| C (Phase 1) | X%  | Y    | Z ms  | 99.8%+ | ~?%  |
| D (Phase 2) | X%  | Y    | Z ms  | 99.9%+ | ~?%  |
```

**핵심 비교**: A vs B의 solve rate 차이가 통계적으로 유의한지.
50 tasks × 3 repeats = 150 data points per config.

---

## 구현 우선순위

```
P0 (즉시, 코드 변경 최소):
  #1 Token-layer any-miss 비율 — pre_step에 ~20줄 추가
  #2 Layer별 miss 집중도 — 기존 데이터로 계산만 추가

P1 (이번 주):
  #3 KL divergence / top-1 flip — miss injection + 2-pass 비교

P2 (결과 보고 후):
  #4 SWE-bench score — A/B/C/D config 비교 실험
```

## 산출물

1. `scripts/analyze_expert_miss_quality.py` — #1, #2 오프라인 분석 (기존 로그 파싱)
2. `expert_cache.py` 수정 — #1 실시간 any-miss 카운터 추가
3. `scripts/expert_miss_injection_benchmark.py` — #3 KL/flip 측정 스크립트
4. SWE-bench 결과 CSV — #4 config별 solve rate

## 참고: 기존 환경변수

| Variable | Default | 용도 |
|----------|---------|------|
| `VLLM_EXPERT_DIAG_DUMP` | 100 | 진단 출력 간격 (steps) |
| `VLLM_EXPERT_NEVER_EVICT_K` | 10 | 3-A never-evict window |
| `VLLM_EXPERT_UNION_STEPS` | 3 | 3-B union history |
| `VLLM_EXPERT_EAGER_ON_ANY_MISS` | 0 | 3-E eager fallback |
| `VLLM_EXPERT_MAX_RESIDENT` | 50 | GPU 상주 expert 수 |

## 실행 환경

- **Paladin**: `jinpyo@paladin.ucsd.edu`, 2×H100 SXM, TP2
- **Server**: `bash /home/jinpyo/llm_serving/run_server.sh tp2-fp16`
- **HF_HOME**: `/mnt/raid0_ssd/huggingface`
- **TMPDIR**: `/mnt/raid0_ssd/jinpyo/tmp`
- **로그**: `/mnt/raid0_ssd/jinpyo/kv_pressure_results/logs/`
