#!/bin/bash
# =============================================================================
# Step-Boundary PoC — TP4 Experiment (solab 4×H200 NVL)
# =============================================================================
#
# Configs:
#   V0: Vanilla baseline (no offload, no elastic KV)
#   V42: V4.2 baseline (elastic KV + PP, eager boundary — current best)
#   SB1: Step-Boundary + decode-freeze ON  (PoC baseline)
#   SB0: Step-Boundary + decode-freeze OFF (ablation)
#
# 성공 기준:
#   1. Quality = vanilla (GSM8K/IFEval 동일)
#   2. Forward .item() = 0  (SB1/SB0)
#   3. TPOT T10 overhead < 20ms (현재 V4.2 = +127ms)
#   4. TPS ≥ 342 (V4.2 baseline)
#
# Usage:
#   bash scripts/run_step_boundary_tp4.sh              # 전체
#   bash scripts/run_step_boundary_tp4.sh V0 SB1       # 선택
#
# =============================================================================
set -euo pipefail

source /home/ucsd/miniconda3/etc/profile.d/conda.sh
conda activate vllm
export HF_HOME=/home/ucsd/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/home/ucsd/tmp
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/stubs:${LIBRARY_PATH:-}
mkdir -p "$TMPDIR"

MODEL="/home/ucsd/huggingface/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct/snapshots/9c7f2fbe84465e40164a94cc16cd30b6999b0cc7"
PORT=8000
GPU_MEM_UTIL=0.90
TP=4
MML=49152
MAX_SEQS=192
START_TIME=$(date +%s)

RESULTS=~/results/step_boundary_tp4
mkdir -p "$RESULTS"

# ── Config matrix ──
# LABEL|OFFLOAD|ELASTIC|STEP_BOUNDARY|DECODE_FREEZE|SCRATCH_CAP|SCRATCH_BANKS
ALL_CONFIGS=(
    "V0|0|0|0|1|0|1"
    "V42|1|1|0|1|128|2"
    "SB1|1|1|1|1|128|2"
    "SB0|1|1|1|0|128|2"
)

# Parse CLI args
SELECTED=()
for arg in "$@"; do SELECTED+=("$arg"); done
if [ ${#SELECTED[@]} -eq 0 ]; then
    SELECTED=("V0" "V42" "SB1" "SB0")
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
    local label=$1 offload=$2 elastic=$3 step_boundary=$4
    local decode_freeze=$5 scratch_cap=$6 scratch_banks=$7
    kill_server

    # Clear all env vars
    unset VLLM_EXPERT_OFFLOAD_ENABLE VLLM_EXPERT_MAX_RESIDENT
    unset VLLM_VMM_EXPERT_POOL VLLM_VMM_PHASE_C VLLM_EXPERT_DIAG_DUMP
    unset VLLM_ELASTIC_KV_ENABLE VLLM_ELASTIC_KV_GROUPS_PER_EXPAND
    unset VLLM_ELASTIC_KV_MIN_RESIDENT VLLM_ELASTIC_KV_MAX_EXPAND_BLOCKS
    unset VLLM_PREFIX_PROTECTION_ENABLE VLLM_ELASTIC_KV_TRACE
    unset VLLM_STEP_BOUNDARY VLLM_FIXED_TAIL
    unset VLLM_PP_DECODE_FREEZE VLLM_PP_C_RELOAD_MS VLLM_PP_H_CAP
    unset VLLM_PP_P_REUSE_ALPHA VLLM_PP_CC_AGE_SCALE
    unset VLLM_EXPERT_SCRATCH_CAPACITY VLLM_SCRATCH_BANKS
    unset VLLM_EXPERT_EAGER_ROUTING_PREFILL VLLM_EXPERT_EAGER_ROUTING_DECODE
    unset VLLM_STEP_PROFILE VLLM_STATIC_PROBE
    unset VLLM_EXPERT_TRACE VLLM_EXPERT_DEBUG

    # ALWAYS: no trace, no debug (I/O is biggest overhead)
    export VLLM_EXPERT_TRACE=0
    export VLLM_EXPERT_DEBUG=0

    if [ "$offload" = "1" ]; then
        export VLLM_EXPERT_OFFLOAD_ENABLE=1
        export VLLM_VMM_EXPERT_POOL=1
        export VLLM_VMM_PHASE_C=1
        export VLLM_EXPERT_DIAG_DUMP=100
    fi

    if [ "$elastic" = "1" ]; then
        export VLLM_ELASTIC_KV_ENABLE=1
        export VLLM_ELASTIC_KV_GROUPS_PER_EXPAND=4
        export VLLM_ELASTIC_KV_MIN_RESIDENT=0.5
        export VLLM_PREFIX_PROTECTION_ENABLE=1
        # V4.2 best params
        export VLLM_PP_C_RELOAD_MS=0.63
        export VLLM_PP_H_CAP=64
        export VLLM_PP_P_REUSE_ALPHA=0.5
        export VLLM_PP_CC_AGE_SCALE=2000
    fi

    if [ "$scratch_cap" -gt 0 ]; then
        export VLLM_EXPERT_SCRATCH_CAPACITY=$scratch_cap
        export VLLM_SCRATCH_BANKS=$scratch_banks
    fi

    if [ "$step_boundary" = "1" ]; then
        export VLLM_STEP_BOUNDARY=1
        # Step-boundary needs eager routing for bank detection
        export VLLM_EXPERT_EAGER_ROUTING_PREFILL=all
        export VLLM_EXPERT_EAGER_ROUTING_DECODE=all
    elif [ "$offload" = "1" ] && [ "$scratch_cap" -gt 0 ]; then
        # V4.2: existing eager routing config
        export VLLM_EXPERT_EAGER_ROUTING_PREFILL=all
        export VLLM_EXPERT_EAGER_ROUTING_DECODE=all
    fi

    export VLLM_PP_DECODE_FREEZE=$decode_freeze

    # Step profiler for T10 analysis
    export VLLM_STEP_PROFILE=1

    SERVER_LOG="$RESULTS/server_${label}_$(date +%Y%m%d_%H%M%S).log"

    log "Starting $label: offload=$offload elastic=$elastic step_boundary=$step_boundary decode_freeze=$decode_freeze scratch=$scratch_cap"

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -u -m vllm.entrypoints.openai.api_server \
        --model $MODEL --host 0.0.0.0 --port $PORT \
        --tensor-parallel-size $TP \
        --gpu-memory-utilization $GPU_MEM_UTIL \
        --max-model-len $MML --max-num-seqs $MAX_SEQS \
        --enable-prefix-caching --trust-remote-code \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    log "Waiting for server (PID=$SERVER_PID)..."
    for i in $(seq 1 360); do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            log "ERROR: Server died during startup"
            tail -40 "$SERVER_LOG"
            return 1
        fi
        if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
            log "Server ready ($((i*5))s)"
            grep -iE "StepBoundary|ACTIVATING|FixedTail|ElasticKV:|offload|phase.c|shrink|scratch" \
                "$SERVER_LOG" 2>/dev/null | head -15 || true
            return 0
        fi
        sleep 5
    done
    log "ERROR: Server timeout"
    return 1
}

run_bench() {
    local label="$1" sessions="$2" concurrency="$3" turns="${4:-12}"
    local outfile="$RESULTS/${label}_c${sessions}_t${turns}.json"

    log "  bench: $label c=$sessions t=$turns"

    local BENCH_SCRIPT=""
    for p in ~/scripts/benchmark_kv_heavy.py ~/benchmark_kv_heavy.py; do
        [ -f "$p" ] && BENCH_SCRIPT="$p" && break
    done
    if [ -z "$BENCH_SCRIPT" ]; then
        log "  WARNING: benchmark_kv_heavy.py not found"
        return
    fi

    python3 "$BENCH_SCRIPT" \
        --sessions "$sessions" --turns "$turns" --concurrency "$concurrency" \
        --model "$MODEL" \
        --server "http://localhost:$PORT/v1" \
        --output "$outfile" \
        2>&1 || log "  WARNING: $label bench failed"

    if [ -f "$outfile" ]; then
        python3 -c "
import json, sys
with open('$outfile') as f:
    d = json.load(f)
s = d.get('summary', {})
kv = s.get('kv', {})
tpot = s.get('tpot_ms', {})
ttft = s.get('ttft_ms', {})
tps = s.get('throughput_tps', 0)
print('  >> OK=%d/%d  TPOT_p50=%.1f TPOT_p99=%.1f  TTFT_p99=%.0f  TPS=%.1f' % (
    s.get('successful', 0), s.get('total_requests', 0),
    tpot.get('p50', 0), tpot.get('p99', 0),
    ttft.get('p99', 0), tps))
print('     KV_peak=%.1f%%  Prefix\$=%.1f%%  Preempt=%d' % (
    kv.get('gpu_kv_peak_pct', 0),
    (kv.get('prefix_cache_hit_rate', 0) or 0) * 100,
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
for cfg_str in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r label offload elastic step_boundary decode_freeze scratch_cap scratch_banks <<< "$cfg_str"

    found=0
    for s in "${SELECTED[@]}"; do [ "$s" = "$label" ] && found=1 && break; done
    [ "$found" = "0" ] && continue

    log "============================================================"
    log "$label: offload=$offload elastic=$elastic SB=$step_boundary DF=$decode_freeze"
    log "============================================================"

    if ! start_server "$label" "$offload" "$elastic" "$step_boundary" \
                      "$decode_freeze" "$scratch_cap" "$scratch_banks"; then
        log "SKIP $label — server failed to start"
        continue
    fi

    # Primary benchmark: c=32 t=12 (V4.2 comparable)
    run_bench "$label" 32 32 12

    # Stress test (optional, if server survives)
    if check_health; then
        run_bench "$label" 64 64 12
    fi

    kill_server
    sleep 10
done

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "============================================================"
log "STEP-BOUNDARY TP4 EXPERIMENT SUMMARY"
log "============================================================"

python3 << 'PYEOF'
import json, glob, os

results_dir = os.path.expanduser("~/results/step_boundary_tp4")

print(f"\n{'Config':<30} {'OK':>9} {'TPOT_p50':>9} {'TPOT_p99':>9} "
      f"{'TTFT_p99':>9} {'TPS':>7} {'KV%':>6} {'Pfx$':>6} {'Pmpt':>5}")
print("-" * 100)

for f in sorted(glob.glob(os.path.join(results_dir, "*_c*_t*.json"))):
    try:
        with open(f) as fh:
            d = json.load(fh)
        s = d.get("summary", {})
        kv = s.get("kv", {})
        tpot = s.get("tpot_ms", {})
        ttft = s.get("ttft_ms", {})
        name = os.path.basename(f).replace(".json", "")
        ok = s.get("successful", 0)
        total = s.get("total_requests", 0)
        tps = s.get("throughput_tps", 0)
        prefix = kv.get("prefix_cache_hit_rate")
        prefix_s = "%.0f%%" % (prefix*100) if prefix is not None else "N/A"
        print(f"{name:<30} {ok:>4}/{total:<4} {tpot.get('p50', 0):>8.1f}ms "
              f"{tpot.get('p99', 0):>8.1f}ms {ttft.get('p99', 0):>8.0f}ms "
              f"{tps:>6.0f} {kv.get('gpu_kv_peak_pct', 0):>5.0f}% "
              f"{prefix_s:>5} {kv.get('preemptions', 0):>5}")
    except Exception as e:
        print(f"  SKIP {os.path.basename(f)}: {e}")

PYEOF

log ""
log "Results: $RESULTS/"
log "Total elapsed: $(($(date +%s)-START_TIME))s"
