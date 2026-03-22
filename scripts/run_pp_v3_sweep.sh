#!/bin/bash
# =============================================================================
# PP V3 Tuning Sweep — Paladin (2×H100, TP=2)
# =============================================================================
# Sweeps P_REUSE × L_SYNC_PREFILL (phase 1), then H_CAP (phase 2).
# Fixed: c=24, t=8
# JSONL trace enabled for every arm.
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
CONCURRENCY=24
TURNS=8
START_TIME=$(date +%s)

RESULTS=/mnt/raid0_ssd/jinpyo/results/pp_v3_sweep
LOGDIR=$RESULTS/logs
TRACEDIR=$RESULTS/traces
mkdir -p "$RESULTS" "$LOGDIR" "$TRACEDIR"

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
    local arm_name="$1"
    kill_server

    # Clear all relevant env vars
    unset VLLM_EXPERT_OFFLOAD_ENABLE VLLM_EXPERT_MAX_RESIDENT
    unset VLLM_VMM_EXPERT_POOL VLLM_VMM_PHASE_C VLLM_EXPERT_DIAG_DUMP
    unset VLLM_ELASTIC_KV_ENABLE VLLM_ELASTIC_KV_GROUPS_PER_EXPAND
    unset VLLM_ELASTIC_KV_MIN_RESIDENT VLLM_ELASTIC_KV_MAX_EXPAND_BLOCKS
    unset VLLM_PREFIX_PROTECTION_ENABLE
    unset VLLM_PP_C_RELOAD_MS VLLM_PP_P_REUSE VLLM_PP_H_CAP
    unset VLLM_PP_L_SYNC_PREFILL VLLM_PP_T_RECOMPUTE
    unset VLLM_PP_T_PREFILL_TOK VLLM_PP_T_SCHED VLLM_PP_T_QUEUE
    unset VLLM_PP_NUM_EXPERTS VLLM_PP_GROUP_SIZE VLLM_PP_TOP_K
    unset VLLM_PP_NUM_LAYERS VLLM_PP_BLOCK_SIZE
    unset VLLM_ELASTIC_KV_TRACE
    unset VLLM_PHASE_C_ORCHESTRATOR VLLM_ORCH_VERSION
    unset VLLM_V3_ALPHA VLLM_V3_S_MAX VLLM_V3_TPOT_SLO_MS
    unset VLLM_V3_H_CAP VLLM_V3_H_FLOOR VLLM_V3_K_CAP_OVERRIDE

    # Expert offload + elastic KV (base)
    export VLLM_EXPERT_OFFLOAD_ENABLE=1
    export VLLM_VMM_EXPERT_POOL=1
    export VLLM_VMM_PHASE_C=1
    export VLLM_EXPERT_DIAG_DUMP=100
    export VLLM_ELASTIC_KV_ENABLE=1
    export VLLM_ELASTIC_KV_GROUPS_PER_EXPAND=4
    export VLLM_ELASTIC_KV_MIN_RESIDENT=0.5
    export VLLM_PREFIX_PROTECTION_ENABLE=1

    # PP V3 tuning params (set by caller)
    export VLLM_PP_C_RELOAD_MS="${PP_C_RELOAD_MS}"
    export VLLM_PP_P_REUSE="${PP_P_REUSE}"
    export VLLM_PP_H_CAP="${PP_H_CAP}"
    export VLLM_PP_L_SYNC_PREFILL="${PP_L_SYNC_PREFILL}"
    export VLLM_PP_T_RECOMPUTE="${PP_T_RECOMPUTE}"

    # JSONL trace
    export VLLM_ELASTIC_KV_TRACE="$TRACEDIR/${arm_name}.jsonl"

    SERVER_LOG="$LOGDIR/server_${arm_name}.log"

    log "Starting server: $arm_name"
    log "  P_REUSE=$PP_P_REUSE L_SYNC=$PP_L_SYNC_PREFILL H_CAP=$PP_H_CAP T_RECOMPUTE=$PP_T_RECOMPUTE C_RELOAD=$PP_C_RELOAD_MS"

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
    local arm_name="$1"
    local outfile="$RESULTS/${arm_name}.json"

    log "  benchmark: c=$CONCURRENCY t=$TURNS -> $outfile"

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
        --concurrent_sessions "$CONCURRENCY" --num_turns "$TURNS" --concurrency "$CONCURRENCY" \
        --model "$MODEL" \
        --base_url "http://localhost:$PORT/v1" \
        --output_dir "$RESULTS" \
        2>&1 || log "  WARNING: $arm_name benchmark failed"

    # Rename output to include label
    local default_out="$RESULTS/w2_swe_unknown_s${CONCURRENCY}_t${TURNS}.json"
    if [ -f "$default_out" ] && [ "$default_out" != "$outfile" ]; then
        mv "$default_out" "$outfile"
        log "  renamed -> $outfile"
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

    sleep 5
}

run_arm() {
    local arm_name="$1"
    if ! start_server "$arm_name"; then
        log "SKIP $arm_name — server failed"
        return 1
    fi
    run_benchmark "$arm_name"
    kill_server
    sleep 10
}

# =============================================================================
# ARMS — Phase 1: P_REUSE × L_SYNC_PREFILL
# =============================================================================
log "============================================================"
log "PP V3 Tuning Sweep — Paladin TP=$TP c=$CONCURRENCY t=$TURNS"
log "============================================================"

# Common defaults for all arms
PP_C_RELOAD_MS=0.63     # PCIe default (worker doesn't set, PP env controls)
PP_T_RECOMPUTE=0.260    # ms/tok

# --- Phase 1: P_REUSE × L_SYNC_PREFILL (H_CAP=64 fixed) ---
log ""
log "=== Phase 1: P_REUSE × L_SYNC_PREFILL (H_CAP=64) ==="

PP_H_CAP=64

# Arm 1: Reclaim-biased (과확장 최대 억제)
PP_P_REUSE=0.25; PP_L_SYNC_PREFILL=4.0
run_arm "ph1_reuse025_lsync40"

# Arm 2: Balanced (설계 중간값)
PP_P_REUSE=0.50; PP_L_SYNC_PREFILL=2.0
run_arm "ph1_reuse050_lsync20"

# Arm 3: Protect-biased (기존과 유사, expand 선호)
PP_P_REUSE=1.00; PP_L_SYNC_PREFILL=1.0
run_arm "ph1_reuse100_lsync10"

# --- Phase 2: H_CAP (best P_REUSE/L_SYNC from phase 1 고정) ---
# 여기서는 Phase 1 결과를 보고 최적 조합을 선택해야 하지만,
# 일단 모든 조합을 돌립니다. P_REUSE=0.50, L_SYNC=2.0 기준.
log ""
log "=== Phase 2: H_CAP sweep (P_REUSE=0.50, L_SYNC=2.0) ==="

PP_P_REUSE=0.50; PP_L_SYNC_PREFILL=2.0

PP_H_CAP=32
run_arm "ph2_hcap032"

PP_H_CAP=128
run_arm "ph2_hcap128"
# H_CAP=64 already covered by ph1_reuse050_lsync20

# --- Phase 3: T_RECOMPUTE (preempt sensitivity) ---
log ""
log "=== Phase 3: T_RECOMPUTE sweep (P_REUSE=0.50, L_SYNC=2.0, H_CAP=64) ==="

PP_H_CAP=64; PP_P_REUSE=0.50; PP_L_SYNC_PREFILL=2.0

PP_T_RECOMPUTE=0.180
run_arm "ph3_trecomp180"

PP_T_RECOMPUTE=0.350
run_arm "ph3_trecomp350"
# T_RECOMPUTE=0.260 already covered by ph1_reuse050_lsync20

kill_server

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "============================================================"
log "PP V3 SWEEP — RESULTS SUMMARY"
log "============================================================"

python3 << 'PYEOF'
import json, glob, os

results_dir = "/mnt/raid0_ssd/jinpyo/results/pp_v3_sweep"
traces_dir = os.path.join(results_dir, "traces")

print(f"\n{'Arm':<30} {'OK':>9} {'TPOT':>8} {'KV_peak':>9} {'Prefix$':>9} {'Preempt':>8}")
print("-" * 80)

for f in sorted(glob.glob(os.path.join(results_dir, "ph*.json"))):
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
        print(f"{name:<30} {ok:>4}/{total:<4} {tpot:>7.1f}ms {kv_peak:>8.1f}% {prefix_s:>9} {preempt:>8}")
    except Exception as e:
        print(f"  SKIP {os.path.basename(f)}: {e}")

# Decision distribution from traces
print(f"\n{'Arm':<30} {'use_unc':>10} {'protect':>10} {'reclaim':>10} {'preempt':>10} {'total':>8}")
print("-" * 80)

for tf in sorted(glob.glob(os.path.join(traces_dir, "ph*.jsonl"))):
    name = os.path.basename(tf).replace(".jsonl", "")
    counts = {}
    total = 0
    try:
        with open(tf) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    d = rec.get("decision", "?")
                    counts[d] = counts.get(d, 0) + 1
                    total += 1
    except Exception:
        continue
    print(f"{name:<30} {counts.get('use_uncached', 0):>10} "
          f"{counts.get('protect_and_expand', 0):>10} "
          f"{counts.get('reclaim_cached', 0):>10} "
          f"{counts.get('preempt', 0):>10} "
          f"{total:>8}")

# Expand stats from server logs
print(f"\n{'Arm':<30} {'expand_calls':>14} {'pp_partial':>14}")
print("-" * 60)

for logf in sorted(glob.glob(os.path.join(results_dir, "logs", "server_ph*.log"))):
    name = os.path.basename(logf).replace("server_", "").replace(".log", "")
    try:
        with open(logf) as fh:
            content = fh.read()
        n_expand = content.count("expand_kv_physical_pages")
        n_partial = content.count("expand partial")
        print(f"{name:<30} {n_expand:>14} {n_partial:>14}")
    except Exception:
        continue

PYEOF

log ""
log "Results: $RESULTS/"
log "Traces:  $TRACEDIR/"
log "Logs:    $LOGDIR/"
log "Total elapsed: $(($(date +%s)-START_TIME))s"
log ""
log "Replay any arm:  python scripts/replay_elastic_kv.py $TRACEDIR/<arm>.jsonl --sweep"
