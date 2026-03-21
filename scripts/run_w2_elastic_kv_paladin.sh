#!/bin/bash
# =============================================================================
# W2 Elastic KV MVP — Paladin (2×H100, TP=2)
# =============================================================================
# PYTHONPATH override로 기존 site-packages 대신 elastic-kv-mvp 코드 사용.
# 기존 gpusim은 건드리지 않음.
#
# Configs:
#   B0: Baseline (no offload, no elastic KV)
#   B1: Expert offload only (no elastic KV) — 기존 gpusim 동작
#   E1: Elastic KV conservative (groups=2, min_resident=0.7)
#   E2: Elastic KV balanced    (groups=4, min_resident=0.5)
#   E3: Elastic KV aggressive  (groups=8, min_resident=0.3)
#
# Usage:
#   bash run_w2_elastic_kv_paladin.sh           # 전체
#   bash run_w2_elastic_kv_paladin.sh B0 E2     # 선택
# =============================================================================
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

# ── Paladin 필수 환경변수 (disk full 방지) ──
export HF_HOME=/mnt/raid0_ssd/huggingface
export TMPDIR=/mnt/raid0_ssd/jinpyo/tmp
mkdir -p "$TMPDIR"

# ── GLIBCXX 호환성: flashinfer JIT가 시스템 GCC로 컴파일 → 시스템 libstdc++ 필요 ──
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# ── Elastic KV 코드 경로 (PYTHONPATH로 site-packages 앞에 삽입) ──
ELASTIC_KV_ROOT="/mnt/raid0_ssd/jinpyo/vllm_elastic_kv"
export PYTHONPATH="$ELASTIC_KV_ROOT:${PYTHONPATH:-}"

MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
PORT=8000
GPU_MEM_UTIL=0.90
TP=2
MML=49152
MAX_SEQS=128
START_TIME=$(date +%s)

RESULTS=/mnt/raid0_ssd/jinpyo/results/w2_elastic_kv_v3
LOGDIR=/mnt/raid0_ssd/jinpyo/results/w2_elastic_kv_v3/logs
mkdir -p "$RESULTS" "$LOGDIR"

# ── Config matrix ──────────────────────────────────────────────────
# Format: "LABEL|OFFLOAD|VMM|PHASE_C|ELASTIC_KV|GROUPS_PER_EXPAND|MIN_RESIDENT|MAX_EXPAND_BLOCKS"
ALL_CONFIGS=(
    "B0|0|0|0|0|0|0|0"
    "B1|1|1|1|0|0|0|0"
    "E1|1|1|1|1|2|0.7|0"
    "E2|1|1|1|1|4|0.5|0"
    "E3|1|1|1|1|8|0.3|0"
)

# Parse CLI args
SELECTED=()
for arg in "$@"; do
    SELECTED+=("$arg")
done
if [ ${#SELECTED[@]} -eq 0 ]; then
    SELECTED=("B0" "B1" "E1" "E2" "E3")
fi

log() { echo "[$(date '+%H:%M:%S')] [+$(($(date +%s)-START_TIME))s] $*"; }

kill_server() {
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 5
}

start_server() {
    local label=$1 offload=$2 vmm=$3 phase_c=$4 elastic=$5
    local groups_per_expand=$6 min_resident=$7 max_expand=$8

    kill_server

    # Clear all env vars
    unset VLLM_EXPERT_OFFLOAD_ENABLE VLLM_EXPERT_MAX_RESIDENT
    unset VLLM_VMM_EXPERT_POOL VLLM_VMM_PHASE_C VLLM_EXPERT_DIAG_DUMP
    unset VLLM_ELASTIC_KV_ENABLE VLLM_ELASTIC_KV_GROUPS_PER_EXPAND
    unset VLLM_ELASTIC_KV_MIN_RESIDENT VLLM_ELASTIC_KV_MAX_EXPAND_BLOCKS
    unset VLLM_PREFIX_PROTECTION_ENABLE
    unset VLLM_PHASE_C_ORCHESTRATOR VLLM_ORCH_VERSION
    unset VLLM_V3_ALPHA VLLM_V3_S_MAX VLLM_V3_TPOT_SLO_MS
    unset VLLM_V3_H_CAP VLLM_V3_H_FLOOR VLLM_V3_K_CAP_OVERRIDE

    # Expert offload
    export VLLM_EXPERT_OFFLOAD_ENABLE=$offload
    [ "$vmm" = "1" ] && export VLLM_VMM_EXPERT_POOL=1
    [ "$phase_c" = "1" ] && export VLLM_VMM_PHASE_C=1
    export VLLM_EXPERT_DIAG_DUMP=100

    # Elastic KV
    if [ "$elastic" = "1" ]; then
        export VLLM_ELASTIC_KV_ENABLE=1
        export VLLM_ELASTIC_KV_GROUPS_PER_EXPAND=$groups_per_expand
        export VLLM_ELASTIC_KV_MIN_RESIDENT=$min_resident
        [ "$max_expand" -gt 0 ] && export VLLM_ELASTIC_KV_MAX_EXPAND_BLOCKS=$max_expand
        # Prefix protection: protect cached prefix blocks via expert eviction
        export VLLM_PREFIX_PROTECTION_ENABLE=1
    fi

    SERVER_LOG="$LOGDIR/server_${label}_$(date +%Y%m%d_%H%M%S).log"

    log "Starting $label: offload=$offload vmm=$vmm phaseC=$phase_c elastic=$elastic"
    [ "$elastic" = "1" ] && log "  elastic: groups=$groups_per_expand min_resident=$min_resident max_expand=$max_expand"
    log "  PYTHONPATH=$ELASTIC_KV_ROOT (elastic-kv-mvp override)"

    PYTHONPATH="$ELASTIC_KV_ROOT:${PYTHONPATH:-}" \
    python -u -m vllm.entrypoints.openai.api_server \
        --model $MODEL --host 0.0.0.0 --port $PORT \
        --tensor-parallel-size $TP \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MML --max-num-seqs $MAX_SEQS \
        --enable-prefix-caching \
        --enable-prompt-tokens-details \
        --enable-log-requests \
        --trust-remote-code \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    log "Waiting for server (PID=$SERVER_PID)..."
    for i in $(seq 1 360); do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            log "ERROR: Server died during startup"
            tail -30 "$SERVER_LOG"
            return 1
        fi
        if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
            log "Server ready ($((i*5))s)"
            grep -i "offload\|phase.c\|elastic.kv\|VMM\|expert.*freed\|geometry\|Elastic KV" \
                "$SERVER_LOG" 2>/dev/null | head -10 || true
            return 0
        fi
        sleep 5
    done
    log "ERROR: Server timeout"
    return 1
}

run_benchmark() {
    local label="$1" sessions="$2" concurrency="$3" turns="${4:-8}"
    local outfile="$RESULTS/${label}_c${sessions}_t${turns}.json"

    log "  benchmark: $label c=$sessions t=$turns"

    # benchmark_kv_heavy.py 위치 (기존 gpusim 스크립트 또는 로컬)
    local BENCH_SCRIPT=""
    for p in \
        ~/scripts/benchmark_kv_heavy.py \
        "$ELASTIC_KV_ROOT/scripts/benchmark_kv_heavy.py" \
        ~/benchmark_kv_heavy.py; do
        if [ -f "$p" ]; then
            BENCH_SCRIPT="$p"
            break
        fi
    done

    if [ -z "$BENCH_SCRIPT" ]; then
        log "  WARNING: benchmark_kv_heavy.py not found, skipping"
        return
    fi

    python3 "$BENCH_SCRIPT" \
        --concurrent_sessions "$sessions" --num_turns "$turns" --concurrency "$concurrency" \
        --model "$MODEL" \
        --base_url "http://localhost:$PORT/v1" \
        --output_dir "$RESULTS" \
        2>&1 || log "  WARNING: $label benchmark failed"

    if [ -f "$outfile" ]; then
        python3 -c "
import json
with open('$outfile') as f:
    d = json.load(f)
s = d.get('summary', {})
kv = s.get('kv', {})
print('  >> OK=%d/%d  TPOT=%.1fms  KV_peak=%.1f%%  Preempt=%d' % (
    s.get('successful', 0), s.get('total_requests', 0),
    s.get('tpot_ms', {}).get('mean', 0),
    kv.get('gpu_kv_peak_pct', 0),
    kv.get('preemptions', 0)))
" 2>&1 || true
    fi
    sleep 5
}

check_health() {
    curl -sf http://localhost:$PORT/health > /dev/null 2>&1
}

# =============================================================================
# MAIN LOOP
# =============================================================================
log "============================================================"
log "Elastic KV MVP Experiment — Paladin TP=$TP"
log "PYTHONPATH override: $ELASTIC_KV_ROOT"
log "Configs: ${SELECTED[*]}"
log "============================================================"

for cfg_str in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r label offload vmm phase_c elastic groups_per_expand min_resident max_expand <<< "$cfg_str"

    found=0
    for s in "${SELECTED[@]}"; do
        [ "$s" = "$label" ] && found=1 && break
    done
    [ "$found" = "0" ] && continue

    log ""
    log "============================================================"
    log "$label: offload=$offload elastic=$elastic"
    log "============================================================"

    if ! start_server "$label" "$offload" "$vmm" "$phase_c" "$elastic" \
                      "$groups_per_expand" "$min_resident" "$max_expand"; then
        log "SKIP $label — server failed to start"
        continue
    fi

    # W2 workloads: c=32 only (서버 재시작 간 비교용)
    run_benchmark "$label" 32 32 8

    kill_server
    sleep 10
done

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "============================================================"
log "ELASTIC KV MVP — RESULTS SUMMARY"
log "============================================================"

python3 << 'PYEOF'
import json, glob, os

results_dir = "/mnt/raid0_ssd/jinpyo/results/w2_elastic_kv_v3"

print(f"\n{'Config':<40} {'OK':>9} {'TPOT':>8} {'KV_peak':>9} {'Preempt':>8}")
print("-" * 80)

for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
    if "server_" in f or "/logs/" in f:
        continue
    try:
        with open(f) as fh:
            d = json.load(fh)
        s = d.get("summary", {})
        kv = s.get("kv", s.get("kv_metrics", {}))
        name = os.path.basename(f).replace(".json", "")
        ok = s.get("successful", 0)
        total = s.get("total_requests", 0)
        tpot = s.get("tpot_ms", {}).get("mean", 0)
        kv_peak = kv.get("gpu_kv_peak_pct", 0)
        preempt = kv.get("preemptions", 0)
        print(f"{name:<40} {ok:>4}/{total:<4} {tpot:>7.1f}ms {kv_peak:>8.1f}% {preempt:>8}")
    except Exception as e:
        print(f"  SKIP {os.path.basename(f)}: {e}")

PYEOF

log ""
log "Results: $RESULTS/"
log "Logs:    $LOGDIR/"
log "Total elapsed: $(($(date +%s)-START_TIME))s"
