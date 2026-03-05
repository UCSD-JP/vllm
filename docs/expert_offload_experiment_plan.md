# Expert Offload 실험 계획서

Date: 2026-02-28 (실험 결과 업데이트: 2026-03-01)
관련 문서:
- [Miss Mitigation 설계](expert_cache_miss_mitigation.md)
- [품질 영향 측정 Prompt](expert_miss_quality_eval_prompt.md)
- [Hit Rate 실측 데이터](../docs/expert_cache_hit_analysis.md)
- [v2 벤치마크](expert_offload_v2_benchmark.md)

---

## 실험 목표

1. Expert cache miss가 **모델 출력 품질**에 미치는 영향을 정량화
2. Mitigation 방안 (Phase 1/2)의 효과를 검증
3. "이 정도 miss는 허용 가능" vs "반드시 100% 필요" 판단 근거 확보
4. 서로 다른 `max_resident` 값에서의 miss rate 변화 확인

---

## 실험 Config 정의

### Config A: Offload OFF (Baseline)

```
전체 512개 expert가 GPU에 상주.
100% 정확도 기준선 — miss 불가능.
```

| 환경변수 | 값 |
|----------|------|
| `VLLM_EXPERT_OFFLOAD_ENABLE` | `0` |

**용도**: 모든 비교의 기준점.

### Config B: Offload ON — 현재 v7 ✅ 실측 완료

```
res=400, 1-step-behind prediction.
현재 구현 그대로.
```

| 환경변수 | 값 |
|----------|------|
| `VLLM_EXPERT_OFFLOAD_ENABLE` | `1` |
| `VLLM_EXPERT_MAX_RESIDENT` | `400` |

**용도**: v7 기본 구현의 품질 영향 측정.

**실측 결과** (SWE-agent c=16, SWE-bench Lite 16 tasks):
| Metric | 값 |
|--------|:--:|
| Expert UID hit rate | 99.3% |
| Token-wtd hit rate | 99.38% |
| Token any-miss rate | **67.1%** (335,340/499,598) |
| Avg miss-layers/token | 3.75 / 48 |
| Weighted miss rate | 0.79% |
| Layer 0 miss | 4.11% |
| Layer 47 miss | 0.39% |

로그: `server_toolcall_maxres400_mml49152_20260228_101618.log`

### Config C: Offload ON + Phase 1 Full (3-A + 3-B + 3-E) ✅ 실측 완료 — ❌ 실패

```
res=400 + 3-A (Never-evict K=10) + 3-B (Union steps=3) + 3-E (Eager fallback on miss>0).
Graph 구조 변경 없이 소프트웨어 최적화만 적용.
```

| 환경변수 | 값 | 설명 |
|----------|------|------|
| `VLLM_EXPERT_OFFLOAD_ENABLE` | `1` | |
| `VLLM_EXPERT_MAX_RESIDENT` | `400` | |
| `VLLM_EXPERT_NEVER_EVICT_K` | `10` | 3-A: 최근 10 step 사용 expert eviction 금지 |
| `VLLM_EXPERT_UNION_STEPS` | `3` | 3-B: 최근 3 step routing union으로 prediction 확장 |
| `VLLM_EXPERT_EAGER_ON_ANY_MISS` | `1` | 3-E: miss > 0이면 해당 step만 eager fallback |

**기대 효과** (사전):
- Hit rate: 99.38% → ~99.8%+

**실측 결과** — **Baseline 대비 전 지표 악화**:
| Metric | 값 | vs Baseline |
|--------|:--:|:-----------:|
| Expert UID hit rate | 99.20% | ❌ -0.1%p |
| Token-wtd hit rate | 99.20% | ❌ -0.18%p |
| Token any-miss rate | **81.0%** (286,189/353,113) | ❌ +13.9%p |
| Avg miss-layers/token | 5.35 / 48 | ❌ +1.60 |
| Weighted miss rate | 0.94% | ❌ +0.15%p |
| Layer 0 miss | 5.20% | ❌ +1.09%p |
| Layer 47 miss | 0.40% | ≈ same |
| eviction_protections | 66M | 🔴 root cause |
| union_extras | 44.7M | |
| eager_triggers | 15,011 | |

**실패 원인 분석**: 3-A (Never-evict K=10)이 66M회 eviction을 차단하여 cache가 경직됨.
res=400 >> unique_needed_max=128이므로 LRU만으로 충분히 좋은 eviction 정책.
과도한 보호가 새 expert를 위한 slot 확보를 방해 → miss 증가.

로그: `server_toolcall_maxres400_mml49152_20260228_104634.log`

### Config D: 3-B Union Only (3-A OFF, 3-E OFF) ✅ 실측 완료 — ❌ 효과 없음

```
3-A 실패 분석 후, 3-B union prediction만 단독 테스트.
res=400 + 3-B (Union steps=3) only.
```

| 환경변수 | 값 | 설명 |
|----------|------|------|
| `VLLM_EXPERT_OFFLOAD_ENABLE` | `1` | |
| `VLLM_EXPERT_MAX_RESIDENT` | `400` | |
| `VLLM_EXPERT_NEVER_EVICT_K` | `0` | 3-A OFF |
| `VLLM_EXPERT_UNION_STEPS` | `3` | 3-B: 최근 3 step routing union |
| `VLLM_EXPERT_EAGER_ON_ANY_MISS` | `0` | 3-E OFF |

**실측 결과** — Baseline과 유사, 약간 악화:
| Metric | 값 | vs Baseline |
|--------|:--:|:-----------:|
| Expert UID hit rate | 99.29% | ≈ same |
| Token-wtd hit rate | 99.14% | ❌ -0.24%p |
| Token any-miss rate | **70.4%** (279,272/396,832) | ❌ +3.3%p |
| Avg miss-layers/token | 3.62 / 48 | ✅ -0.13 |
| Weighted miss rate | 0.83% | ❌ +0.04%p |
| Layer 0 miss | 4.64% | ❌ +0.53%p |
| Layer 47 miss | 0.38% | ≈ same |
| eviction_protections | 0 | ✅ (3-A OFF) |
| union_extras | 61.3M | |
| eager_triggers | 0 | (3-E OFF) |

**분석**: 61.3M extra experts를 prefetch했으나, 과거 routing이 미래 routing을 예측하지 못함.
오히려 불필요한 prefetch가 cache pressure를 높여 유용한 expert가 evict되는 역효과.
**1-step-behind prediction miss는 capacity miss가 아닌 prediction miss** → history-based 접근으로는 해결 불가.

로그: `server_toolcall_maxres400_mml49152_20260228_140425.log`

### Config E: Phase 2 — Layer 0 Eager Routing (미실행)

```
3-C (Layer 0 eager routing).
가장 volatile한 Layer 0만 graph 전에 eager로 routing + miss fetch 수행.
Phase 1이 실패했으므로 Phase 2 단독 효과 측정이 필요.
```

| 환경변수 | 값 | 설명 |
|----------|------|------|
| `VLLM_EXPERT_OFFLOAD_ENABLE` | `1` | |
| `VLLM_EXPERT_MAX_RESIDENT` | `400` | |
| `VLLM_EXPERT_LAYER0_EAGER` | `1` | 3-C: Layer 0 select_experts를 eager 실행 |

**기대 효과**:
- Layer 0: 95.89% → **100%** (실시간 routing, 구조적 miss 불가)
- Layer 0이 전체 miss의 ~25% → 전체 miss rate ~25% 추가 감소

---

## 실험 #1: Token-Layer Any-Miss 비율 (P0)

### 목적

Expert 단위 hit rate (98.4%)와 **토큰 관점 품질 proxy** 사이의 괴리 정량화.

### 정의

```
per_token_any_miss_rate = (48개 layer 중 1개라도 miss 있는 토큰 수) / (전체 토큰 수)

예시 (실측 기반, step 100):
  48-layer 전부 hit 확률 ≈ 0.9585 × 0.98^46 × 0.9819 ≈ 0.37
  → any miss 확률 ≈ 63%  ← expert 단위 98%와 체감이 매우 다름
```

### 실행

- **Workload**: `benchmark_serving.py` (synthetic agentic, c=16)
- **Config**: B, C, D
- **코드 수정**: `expert_cache.py` pre_step에 ~20줄 추가
- **데이터 수집**: 매 100 step 자동 출력 (`VLLM_EXPERT_DIAG_DUMP=100`)
- **로그 위치**: `/mnt/raid0_ssd/jinpyo/kv_pressure_results/logs/`

### 출력 형식

```
[Expert Cache Quality] Step 2000:
  Expert-level hit rate:     98.41%
  Token-level any-miss rate: 63.2%   ← 실제 품질 영향 proxy
  Avg miss layers per token: 2.1 / 48
```

### 실측 결과 (2026-03-01)

| Config | Expert Hit Rate | Any-Miss Rate | Avg Miss-Layers | 해석 |
|--------|:--------------:|:-------------:|:---------------:|------|
| B (Baseline) | 99.38% | **67.1%** | 3.75/48 | 2/3 토큰이 1개+ layer에서 miss |
| C (Phase 1 Full) | 99.20% | **81.0%** | 5.35/48 | ❌ 3-A 과보호로 악화 |
| D (3-B Only) | 99.14% | **70.4%** | 3.62/48 | ❌ union prediction 효과 없음 |

**사전 기대 vs 실측**:

| Config | 기대 Any-Miss | 실측 Any-Miss | 판정 |
|--------|:------------:|:------------:|:----:|
| B | ~25% (최종) | 67.1% | 기대보다 2.7× 높음 |
| C | ~9% | 81.0% | ❌ 오히려 악화 |
| D | (신규) | 70.4% | ❌ baseline과 유사 |

**핵심 발견**: 사전 예측이 크게 빗나간 이유는 miss가 capacity miss가 아닌 **prediction miss** (1-step-behind)이기 때문.
res=400 >> unique_needed_max=128 → headroom 3.2×로 LRU가 이미 최적에 가까움.
History-based mitigation (3-A, 3-B)은 prediction miss에 무효.

---

## 실험 #2: Layer별 Miss 집중도 (P0)

### 목적

Layer 0 miss 2%와 Layer 35 miss 2%는 품질 영향이 다름:
- **Layer 0**: 입력 embedding 직후 → miss가 전체 forward propagation에 파급
- **Layer 47**: 출력 직전 → logit에 직접 영향
- **중간 layers**: residual connection이 miss 영향을 희석

### 구현

기존 `_diag['per_layer_misses']`, `_diag['per_layer_hits']` 데이터 활용.
추가 계산만 수행:

```python
LAYER_WEIGHT = {0: 3.0, 47: 2.0}  # 나머지: 1.0
# heuristic — #3의 KL 결과로 교정 예정

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

## 실험 #3: KL Divergence / Top-1 Flip Rate (P1)

### 목적

Miss가 **최종 출력 logit에 얼마나 영향을 주는지** 직접 측정.

### 방법

동일 입력에서 2-pass 비교:
1. **Pass 1**: Normal run (offload ON, 모든 expert 정상) → `logits_normal`
2. **Pass 2**: Miss-injected run (특정 layer에 miss 주입) → `logits_miss`

```python
KL(normal || miss) = sum(p_normal * log(p_normal / p_miss))
top1_flip = (argmax(logits_normal) != argmax(logits_miss))
```

### Miss 주입 방법

```python
# 환경변수:
# VLLM_EXPERT_INJECT_MISS_RATE=0.02    (2% random miss)
# VLLM_EXPERT_INJECT_MISS_LAYERS=0,47  (특정 layer만)
# VLLM_EXPERT_INJECT_MISS_LOG=1        (logit 비교 모드)

# _update_cache_map() 끝에서:
if self._inject_miss_rate > 0:
    mask = torch.rand(scratch.shape) < self._inject_miss_rate
    scratch[mask] = -1  # 인위적 miss 주입 → 해당 expert 기여 = 0
```

### Workload

- **50개 고정 prompt** (SWE-bench subset에서 추출)
  - 동일 입력 보장 (2-pass 비교에 필수)
  - 각 prompt에서 100 토큰 생성 → 50×100 = 5,000 비교 데이터 포인트

### 실험 Matrix

| 조건 | Miss Rate | Miss Layers | 목적 |
|------|:---------:|:-----------:|------|
| Baseline | 0% | — | 기준선 (noise floor 확인) |
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

### 기대 결과 & 판단 기준

| 지표 | "허용 가능" | "주의 필요" | "불가" |
|------|:----------:|:----------:|:------:|
| KL divergence | < 0.01 | 0.01-0.1 | > 0.1 |
| Top-1 flip rate | < 1% | 1-5% | > 5% |
| Top-5 overlap | > 99% | 95-99% | < 95% |

---

## 실험 #4: SWE-bench Score 트래킹 (P2)

### 목적

#1-#3은 proxy metric. **실제 task 성공률에 미치는 영향** 직접 확인.

### Config별 실행

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

# Config D: + Phase 2
export VLLM_EXPERT_LAYER0_EAGER=1
python run_swe_bench.py --tasks subset_50 --repeat 3
```

### 트래킹 Metrics

| Config | Solve Rate | Avg Tokens | Avg TPOT | Hit Rate | Any-Miss% |
|--------|:----------:|:----------:|:--------:|:--------:|:---------:|
| A (OFF)     | X%  | Y    | Z ms  | 100%   | 0%   |
| B (v7)      | X%  | Y    | Z ms  | 99.4%  | ~63% |
| C (Phase 1) | X%  | Y    | Z ms  | 99.8%+ | ~?%  |
| D (Phase 2) | X%  | Y    | Z ms  | 99.9%+ | ~?%  |

### 통계적 유의성

- 50 tasks × 3 repeats = **150 data points** per config
- McNemar's test 또는 paired t-test로 A vs B 유의성 검정
- 핵심: **A vs B의 solve rate 차이가 1% 이내이면 miss는 실질적으로 무해**

---

## 실험 #5: max_resident Sweep (보조)

### 목적

res=400이 "최적"인지, 더 줄일 수 있는지 확인.

### Prediction (분석적 — 로그 기반)

`scripts/predict_miss_rate_by_resident.py --from-stats` 결과:

| res | Capacity Miss | Prediction Miss | Combined | Any-Miss% | Headroom |
|:---:|:------------:|:---------------:|:--------:|:---------:|:--------:|
| 100 | 5.50% | 1.51% | 7.02% | 96.5% | 0.8x |
| 150 | 0.00% | 1.60% | 1.60% | 53.9% | 1.2x |
| 200 | 0.00% | 1.60% | 1.60% | 53.9% | 1.6x |
| 300 | 0.00% | 1.60% | 1.60% | 53.9% | 2.4x |
| 400 | 0.00% | 1.60% | 1.60% | 53.9% | 3.2x |
| 500 | 0.00% | 1.60% | 1.60% | 53.9% | 4.0x |

**핵심 발견**:
- **res=150 이상이면 capacity miss 0%** (unique_needed max=128 < 150)
- 150과 400의 miss rate 차이 없음 — **모든 miss가 N-1 prediction 오차에서 발생**
- res를 300으로 줄여도 miss rate 동일 (단, GPU 메모리 100개 expert 분량 절약)
- 그러나 res < 150이면 capacity miss가 급증 → 불가

### 실측 검증 (선택적)

```bash
# res=300으로 실행하여 prediction 검증
export VLLM_EXPERT_MAX_RESIDENT=300
# 동일 SWE-bench workload로 hit rate 비교
```

---

## 구현 우선순위

```
P0 (즉시, 코드 변경 최소):
  실험 #1 — pre_step에 ~20줄 추가 (token-layer any-miss 카운터)
  실험 #2 — 기존 데이터로 계산만 추가 (~10줄)
  실험 #5 — 이미 완료 (predict_miss_rate_by_resident.py)

P1 (이번 주):
  실험 #3 — miss injection + 2-pass logit 비교 (~80줄)

P2 (결과 보고 후):
  실험 #4 — SWE-bench A/B/C/D 실행 (MAB repo)
```

---

## 산출물

| 산출물 | 파일 | 상태 |
|--------|------|:----:|
| res별 miss rate 예측 | `scripts/predict_miss_rate_by_resident.py` | ✅ 완료 |
| #1 실시간 any-miss 카운터 | `expert_cache.py` `get_quality_summary()` | ✅ 완료 |
| #2 layer-weighted miss 집중도 | `expert_cache.py` `get_quality_summary()` | ✅ 완료 |
| Phase 1 mitigation 구현 | `expert_cache.py` 3-A/3-B + `gpu_model_runner.py` 3-E | ✅ 완료 |
| Phase 1 unit tests (19개) | `tests/test_expert_miss_mitigation.py` | ✅ 완료 |
| 실측 B (Baseline) | Paladin, c=16 SWE-bench | ✅ 완료 |
| 실측 C (Phase 1 Full) | Paladin, c=16 SWE-bench | ✅ 완료 — 실패 |
| 실측 D (3-B Union Only) | Paladin, c=16 SWE-bench | ✅ 완료 — 효과 없음 |
| #3 KL/flip 측정 스크립트 | miss injection 인프라 구현됨 | 실측 미실행 |
| SWE-bench score A/B 비교 | MAB repo | TODO |
| 종합 결과 정리 | `docs/expert_miss_quality_results.md` | TODO |

---

## 실행 환경

| 항목 | 값 |
|------|------|
| **서버** | Paladin: `jinpyo@paladin.ucsd.edu`, 2×H100 SXM |
| **vLLM 서버** | `bash /home/jinpyo/llm_serving/run_server.sh tp2-fp16` |
| **HF_HOME** | `/mnt/raid0_ssd/huggingface` |
| **TMPDIR** | `/mnt/raid0_ssd/jinpyo/tmp` |
| **로그** | `/mnt/raid0_ssd/jinpyo/kv_pressure_results/logs/` |
| **진단 간격** | `VLLM_EXPERT_DIAG_DUMP=100` (매 100 step) |
| **MAB (SWE-bench)** | `/home/jp/paper_resource/multi-agent-bench` |

---

## 실험 결과 종합 분석 (2026-03-01)

### 3-Config 비교표

| Metric | B (Baseline) | C (Phase1 Full) | D (3-B Only) |
|--------|:-----------:|:---------------:|:------------:|
| Expert UID hit rate | 99.3% | 99.20% | 99.29% |
| Token-wtd hit rate | 99.38% | 99.20% | 99.14% |
| **Token any-miss rate** | **67.1%** | **81.0%** | **70.4%** |
| Avg miss-layers/token | 3.75/48 | 5.35/48 | 3.62/48 |
| Weighted miss rate | 0.79% | 0.94% | 0.83% |
| Layer 0 miss | 4.11% | 5.20% | 4.64% |
| Layer 47 miss | 0.39% | 0.40% | 0.38% |
| eviction_protections | 0 | 66M | 0 |
| union_extras | 0 | 44.7M | 61.3M |
| eager_triggers | 0 | 15,011 | 0 |

### 핵심 발견

1. **Expert-level hit rate와 token-level any-miss rate의 괴리가 매우 큼**
   - Expert hit rate 99.3%인데 token any-miss rate 67.1% → 2/3 토큰이 영향받음
   - 48 layer × 8 experts/layer → 384 expert slot/token, 1개라도 miss 확률 높음

2. **모든 miss가 prediction miss (N-1 routing 오차)**
   - res=400 >> unique_needed_max=128 → capacity miss = 0%
   - history-based 접근 (3-A never-evict, 3-B union)은 근본적으로 무효
   - 3-A: 과보호로 오히려 악화 (66M eviction 차단 → cache 경직)
   - 3-B: 과거 routing ≠ 미래 routing, 불필요한 prefetch가 cache pressure 유발

3. **Phase 1 mitigation은 전략적으로 실패**
   - 사전 기대: any-miss 67% → ~9% (Phase 1), ~5% (Phase 2)
   - 실측: 67% → 81% (악화) / 70% (효과 없음)
   - **원인**: miss의 본질을 잘못 진단 (capacity miss로 가정했으나 실제로는 prediction miss)

### 남은 전략 옵션

| 옵션 | 접근 | 효과 예상 | 복잡도 |
|------|------|----------|:------:|
| **3-C Layer 0 Eager** | Layer 0만 CUDA graph 밖에서 실행 | Layer 0 miss 4.1% → 0% | 중 |
| **3-D Piecewise Graph** | 전 layer를 per-layer graph로 분리 | 이론상 0% miss | 높음 |
| **현상 유지** | 67% any-miss 허용 | #3 KL로 실제 품질 영향 확인 | 없음 |
| **res 축소** | res=300 (메모리 절약) | miss rate 동일 (prediction-limited) | 없음 |

### 다음 단계 권장

- **#3 KL divergence 실험**을 우선 진행하여 67% any-miss가 실제로 품질에 영향을 주는지 확인
  - KL < 0.01이면 현상 유지로 충분 → 논문에서 "negligible quality impact" 주장 가능
  - KL > 0.01이면 3-C (Layer 0 Eager) 또는 3-D (Piecewise Graph) 필요

---

## 판단 프레임워크

실험 완료 후 최종 판단:

```
IF (#3 KL < 0.01 AND top1_flip < 1% AND #4 A_vs_B solve_rate_diff < 1%):
    → "v7의 0.6% miss는 실질적으로 무해. Phase 1 mitigation만으로 충분."
    → 논문: "Expert offload achieves 99.4% hit rate with negligible quality impact"

ELIF (#3 KL < 0.1 AND top1_flip < 5%):
    → "미미한 영향 있음. Phase 1 + Phase 2 적용 권장."
    → 논문: "With mitigation, expert offload reaches 99.9%+ hit rate"

ELSE:
    → "유의미한 품질 저하. Per-layer piecewise graph (3-D) 필수."
    → 논문: "100% accuracy requires per-layer graph at +23% TPOT cost"
```
