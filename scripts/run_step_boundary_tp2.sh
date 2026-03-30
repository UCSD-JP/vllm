#!/bin/bash
# =============================================================================
# Step-Boundary PoC — TP2 Experiment (Paladin 2×H100)
# =============================================================================
#
# TP2 = scratch_capacity < max possible tail → overflow 가능.
# STEP_BOUNDARY는 max_tail > scratch에서 assert fail.
# 따라서 TP2에서는 STEP_BOUNDARY 비활성화, V4.2 대비 비교만 수행.
#
# Configs:
#   V0:  Vanilla baseline (no offload)
#   V42: V4.2 (elastic KV + PP, eager boundary)
#   SB1: Step-Boundary + decode-freeze ON  (TP2 feasibility test)
#   SB0: Step-Boundary + decode-freeze OFF (ablation)
#
# NOTE: SB1/SB0는 scratch=128로 시도. max_tail > 128이면 assert fail → skip.
#       성공하면 TP2에서도 step-boundary 유효.
#
# Usage:
#   bash scripts/run_step_boundary_tp2.sh              # 전체
#   bash scripts/run_step_boundary_tp2.sh V0 V42       # V4.2 비교만
#
# =============================================================================
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export HF_HOME=/mnt/raid0_ssd/huggingface
export TMPDIR=/mnt/raid0_ssd/jinpyo/tmp
mkdir -p "$TMPDIR"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
PORT=8000
GPU_MEM_UTIL=0.90
TP=2
MML=49152
MAX_SEQS=128
START_TIME=$(date +%s)

RESULTS=/mnt/raid0_ssd/jinpyo/results/step_boundary_tp2
mkdir -p "$RESULTS"

# LABEL|OFFLOAD|ELASTIC|STEP_BOUNDARY|DECODE_FREEZE|SCRATCH_CAP|SCRATCH_BANKS
ALL_CONFIGS=(
    "V0|0|0|0|1|0|1"
    "V42|1|1|0|1|128|2"
    "SB1|1|1|1|1|128|2"
    "SB0|1|1|1|0|128|2"
)

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
        export VLLM_EXPERT_EAGER_ROUTING_PREFILL=all
        export VLLM_EXPERT_EAGER_ROUTING_DECODE=all
    elif [ "$offload" = "1" ] && [ "$scratch_cap" -gt 0 ]; then
        export VLLM_EXPERT_EAGER_ROUTING_PREFILL=all
        export VLLM_EXPERT_EAGER_ROUTING_DECODE=all
    fi

    export VLLM_PP_DECODE_FREEZE=$decode_freeze
    export VLLM_STEP_PROFILE=1

    SERVER_LOG="$RESULTS/server_${label}_$(date +%Y%m%d_%H%M%S).log"

    log "Starting $label: offload=$offload elastic=$elastic SB=$step_boundary DF=$decode_freeze scratch=$scratch_cap"

    CUDA_VISIBLE_DEVICES=0,1 \
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
            # Check for StepBoundary assert (expected for TP2 if tail > scratch)
            if grep -q "StepBoundary.*max_tail.*scratch_capacity" "$SERVER_LOG" 2>/dev/null; then
                log "EXPECTED: TP2 tail overflow — step-boundary not viable at scratch=$scratch_cap"
                grep "StepBoundary" "$SERVER_LOG" | head -3
            else
                tail -40 "$SERVER_LOG"
            fi
            return 1
        fi
        if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
            log "Server ready ($((i*5))s)"
            grep -iE "StepBoundary|ACTIVATING|ElasticKV:|offload|scratch" \
                "$SERVER_LOG" 2>/dev/null | head -10 || true
            return 0
        fi
        sleep 5
    done
    log "ERROR: Server timeout"
    return 1
}

run_bench() {
    local label="$1" sessions="$2" concurrency="$3" turns="${4:-8}"
    local outfile="$RESULTS/${label}_c${sessions}_t${turns}.json"

    log "  bench: $label c=$sessions t=$turns"

    local BENCH_SCRIPT=""
    for p in \
        ~/scripts/benchmark_kv_heavy.py \
        /mnt/raid0_ssd/jinpyo/vllm_elastic_kv/scripts/benchmark_kv_heavy.py \
        ~/benchmark_kv_heavy.py; do
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
import json
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

    run_bench "$label" 32 32 8

    if check_health; then
        run_bench "$label" 64 64 8
    fi

    kill_server
    sleep 10
done

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "============================================================"
log "STEP-BOUNDARY TP2 EXPERIMENT SUMMARY"
log "============================================================"

python3 << 'PYEOF'
import json, glob, os

results_dir = "/mnt/raid0_ssd/jinpyo/results/step_boundary_tp2"

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
