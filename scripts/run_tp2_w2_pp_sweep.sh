#!/bin/bash
# =============================================================================
# TP2 W2 Prefix Protection Sweep — Paladin (2×H100, TP=2)
# =============================================================================
# 4 arms: pp-off, pp-0252, pp-012, pp-003
# Each arm: c=32 t=8
# Common elastic KV: groups=4, min_resident=0.5
# =============================================================================
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export HF_HOME=/mnt/raid0_ssd/huggingface
export TMPDIR=/mnt/raid0_ssd/jinpyo/tmp
mkdir -p "$TMPDIR"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

ELASTIC_KV_ROOT="/mnt/raid0_ssd/jinpyo/vllm_elastic_kv"

MODEL="Qwen/Qwen3-Next-80B-A3B-Instruct"
PORT=8000
GPU_MEM_UTIL=0.90
TP=2
MML=49152
MAX_SEQS=128
START_TIME=$(date +%s)

RESULTS=/mnt/raid0_ssd/jinpyo/results/tp2_w2_pp_sweep
LOGDIR=$RESULTS/logs
mkdir -p "$RESULTS" "$LOGDIR"

# ── Arm matrix ────────────────────────────────────────────────────
# Format: "LABEL|PP_ENABLE|C_RELOAD_MS"
ALL_ARMS=(
    "pp-off|0|0.252"
    "pp-0252|1|0.252"
    "pp-012|1|0.12"
    "pp-003|1|0.03"
)

# Parse CLI args (select specific arms)
SELECTED=()
for arg in "$@"; do
    SELECTED+=("$arg")
done
if [ ${#SELECTED[@]} -eq 0 ]; then
    SELECTED=("pp-off" "pp-0252" "pp-012" "pp-003")
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
    local label=$1 pp_enable=$2 c_reload_ms=$3

    kill_server

    # Clear all env vars
    unset VLLM_EXPERT_OFFLOAD_ENABLE VLLM_EXPERT_MAX_RESIDENT
    unset VLLM_VMM_EXPERT_POOL VLLM_VMM_PHASE_C VLLM_EXPERT_DIAG_DUMP
    unset VLLM_ELASTIC_KV_ENABLE VLLM_ELASTIC_KV_GROUPS_PER_EXPAND
    unset VLLM_ELASTIC_KV_MIN_RESIDENT VLLM_ELASTIC_KV_MAX_EXPAND_BLOCKS
    unset VLLM_PREFIX_PROTECTION_ENABLE
    unset VLLM_PP_C_RELOAD_MS VLLM_PP_P_REUSE VLLM_PP_H_FLOOR VLLM_PP_H_CAP
    unset VLLM_PHASE_C_ORCHESTRATOR VLLM_ORCH_VERSION
    unset VLLM_V3_ALPHA VLLM_V3_S_MAX VLLM_V3_TPOT_SLO_MS
    unset VLLM_V3_H_CAP VLLM_V3_H_FLOOR VLLM_V3_K_CAP_OVERRIDE

    # Expert offload + elastic KV (always on for all arms)
    export VLLM_EXPERT_OFFLOAD_ENABLE=1
    export VLLM_VMM_EXPERT_POOL=1
    export VLLM_VMM_PHASE_C=1
    export VLLM_EXPERT_DIAG_DUMP=100
    export VLLM_ELASTIC_KV_ENABLE=1
    export VLLM_ELASTIC_KV_GROUPS_PER_EXPAND=4
    export VLLM_ELASTIC_KV_MIN_RESIDENT=0.5

    # Prefix protection
    export VLLM_PREFIX_PROTECTION_ENABLE=$pp_enable
    if [ "$pp_enable" = "1" ]; then
        export VLLM_PP_C_RELOAD_MS=$c_reload_ms
        export VLLM_PP_P_REUSE=1.0
        export VLLM_PP_H_FLOOR=1
        export VLLM_PP_H_CAP=64
    fi

    SERVER_LOG="$LOGDIR/server_${label}_$(date +%Y%m%d_%H%M%S).log"

    log "Starting $label: pp_enable=$pp_enable c_reload_ms=$c_reload_ms"
    log "  PYTHONPATH=$ELASTIC_KV_ROOT"

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
            grep -i "ElasticKV:\|prefix.protect\|DIAG-1\|blocks/call" \
                "$SERVER_LOG" 2>/dev/null | head -5 || true
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

    log "  benchmark: $label c=$sessions t=$turns → $outfile"

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

    # Rename output to include arm label
    local default_out="$RESULTS/w2_swe_unknown_s${sessions}_t${turns}.json"
    if [ -f "$default_out" ] && [ "$default_out" != "$outfile" ]; then
        mv "$default_out" "$outfile"
        log "  renamed → $outfile"
    fi

    if [ -f "$outfile" ]; then
        python3 -c "
import json
with open('$outfile') as f:
    d = json.load(f)
s = d.get('summary', {})
kv = s.get('kv', {})
print('  >> OK=%d/%d  TPOT=%.1fms  KV_peak=%.1f%%  Prefix\$=%.1f%%  Preempt=%d' % (
    s.get('successful', 0), s.get('total_requests', 0),
    s.get('tpot_ms', {}).get('mean', 0),
    kv.get('gpu_kv_peak_pct', 0),
    (kv.get('prefix_cache_hit_rate', 0) or 0) * 100,
    kv.get('preemptions', 0)))
" 2>&1 || true
    fi

    # Extract expand/protect stats from server log
    local latest_log=$(ls -t "$LOGDIR"/server_${label}_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        local n_expand=$(grep -c "expand_kv_physical_pages" "$latest_log" 2>/dev/null || echo 0)
        local n_protect=$(grep -c "should_protect=True" "$latest_log" 2>/dev/null || echo 0)
        local n_unprotect=$(grep -c "should_protect=False" "$latest_log" 2>/dev/null || echo 0)
        log "  expand=$n_expand  protect=$n_protect  unprotect=$n_unprotect"
    fi

    sleep 5
}

# =============================================================================
# MAIN LOOP
# =============================================================================
log "============================================================"
log "TP2 W2 Prefix Protection Sweep — Paladin TP=$TP"
log "Arms: ${SELECTED[*]}"
log "============================================================"

for arm_str in "${ALL_ARMS[@]}"; do
    IFS='|' read -r label pp_enable c_reload_ms <<< "$arm_str"

    found=0
    for s in "${SELECTED[@]}"; do
        [ "$s" = "$label" ] && found=1 && break
    done
    [ "$found" = "0" ] && continue

    log ""
    log "============================================================"
    log "$label: pp=$pp_enable c_reload=$c_reload_ms"
    log "============================================================"

    if ! start_server "$label" "$pp_enable" "$c_reload_ms"; then
        log "SKIP $label — server failed to start"
        continue
    fi

    run_benchmark "$label" 32 32 8

    kill_server
    sleep 10
done

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "============================================================"
log "TP2 PP SWEEP — RESULTS SUMMARY"
log "============================================================"

python3 << 'PYEOF'
import json, glob, os

results_dir = os.path.expanduser("/mnt/raid0_ssd/jinpyo/results/tp2_w2_pp_sweep")

print(f"\n{'Arm':<20} {'OK':>9} {'TPOT':>8} {'KV_peak':>9} {'Prefix$':>9} {'Preempt':>8}")
print("-" * 70)

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
        prefix = kv.get("prefix_cache_hit_rate")
        preempt = kv.get("preemptions", 0)
        prefix_s = "%.1f%%" % (prefix*100) if prefix is not None else "N/A"
        print(f"{name:<20} {ok:>4}/{total:<4} {tpot:>7.1f}ms {kv_peak:>8.1f}% {prefix_s:>9} {preempt:>8}")
    except Exception as e:
        print(f"  SKIP {os.path.basename(f)}: {e}")

# Expand stats from logs
print(f"\n{'Arm':<20} {'Expand':>8} {'Protect':>10} {'Unprotect':>11}")
print("-" * 55)

for logf in sorted(glob.glob(os.path.join(results_dir, "logs", "server_*.log"))):
    name = os.path.basename(logf).replace(".log", "")
    # extract arm name
    parts = name.split("_")
    arm = "_".join(parts[1:-2]) if len(parts) > 3 else name
    with open(logf) as fh:
        content = fh.read()
    n_expand = content.count("expand_kv_physical_pages")
    n_protect = content.count("should_protect=True")
    n_unprotect = content.count("should_protect=False")
    print(f"{arm:<20} {n_expand:>8} {n_protect:>10} {n_unprotect:>11}")

PYEOF

log ""
log "Results: $RESULTS/"
log "Logs:    $LOGDIR/"
log "Total elapsed: $(($(date +%s)-START_TIME))s"
