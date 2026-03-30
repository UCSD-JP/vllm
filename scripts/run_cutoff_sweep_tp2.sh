#!/bin/bash
# =============================================================================
# Cutoff vs Step-Boundary Sweep — TP2 (Paladin 2×H100)
# =============================================================================
#
# Demand-driven offload: init=all resident, KV pressure → shrink.
# Scratch = LOCAL_E * pct / 100, rounded down to GROUP_SIZE alignment.
#
# NOTE: TP2에서 step-boundary는 max_tail > scratch 시 assert fail 가능.
#       서버 시작 실패하면 자동 SKIP.
#
# Usage:
#   bash scripts/run_cutoff_sweep_tp2.sh              # 전체
#   bash scripts/run_cutoff_sweep_tp2.sh CB_20         # 선택
#
# =============================================================================
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vllm

export HF_HOME=/mnt/raid0_ssd/huggingface
export TMPDIR=/mnt/raid0_ssd/jinpyo/tmp
mkdir -p "$TMPDIR"

KV_DIR=/mnt/raid0_ssd/jinpyo/kv_pressure_results
PORT=8001            # paladin-start-server.sh 고정
CONCURRENCY=20
START_TIME=$(date +%s)

LOCAL_E=512          # experts per GPU (Qwen3-Next-80B-A3B: NOT sharded by TP)
GROUP_SIZE=4         # group alignment

RESULTS=/mnt/raid0_ssd/jinpyo/results/cutoff_sweep_tp2
mkdir -p "$RESULTS"

# ── pct → scratch (group-aligned) ──
pct_to_scratch() {
    local pct=$1
    local raw=$(( LOCAL_E * pct / 100 ))
    echo $(( (raw / GROUP_SIZE) * GROUP_SIZE ))
}

# ── Config matrix ──
# LABEL|OFFLOAD|ELASTIC|CUTOFF|STEP_BOUNDARY|DECODE_FREEZE|PCT|SCRATCH_BANKS
OFFLOAD_PCTS=(20 25 30)

ALL_CONFIGS=()
for pct in "${OFFLOAD_PCTS[@]}"; do
    ALL_CONFIGS+=("CB_${pct}|1|1|1|0|0|${pct}|2")
    ALL_CONFIGS+=("SB_${pct}|1|1|0|1|0|${pct}|2")
done

SELECTED=()
for arg in "$@"; do SELECTED+=("$arg"); done
if [ ${#SELECTED[@]} -eq 0 ]; then
    for pct in "${OFFLOAD_PCTS[@]}"; do
        SELECTED+=("CB_${pct}" "SB_${pct}")
    done
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
    local label=$1 offload=$2 elastic=$3 cutoff=$4 step_boundary=$5
    local decode_freeze=$6 pct=$7 scratch_banks=$8
    local scratch_cap
    scratch_cap=$(pct_to_scratch "$pct")
    kill_server

    # Generate env file
    local ENV_FILE="$RESULTS/env_${label}.sh"
    {
        echo "# Auto-generated: $label (pct=${pct}%, scratch=${scratch_cap}, localE=${LOCAL_E})"
        echo "export VLLM_EXPERT_TRACE=0"
        echo "export VLLM_EXPERT_DEBUG=0"

        if [ "$offload" = "1" ]; then
            echo "export VLLM_EXPERT_OFFLOAD_ENABLE=1"
            echo "export VLLM_VMM_EXPERT_POOL=1"
            echo "export VLLM_VMM_PHASE_C=1"
            echo "export VLLM_EXPERT_DIAG_DUMP=100"
        fi

        if [ "$elastic" = "1" ]; then
            echo "export VLLM_ELASTIC_KV_ENABLE=1"
            echo "export VLLM_ELASTIC_KV_GROUPS_PER_EXPAND=4"
            echo "export VLLM_ELASTIC_KV_MIN_RESIDENT=0.5"
            echo "export VLLM_PREFIX_PROTECTION_ENABLE=1"
            echo "export VLLM_PP_C_RELOAD_MS=0.63"
            echo "export VLLM_PP_H_CAP=64"
            echo "export VLLM_PP_P_REUSE_ALPHA=0.5"
            echo "export VLLM_PP_CC_AGE_SCALE=2000"
        fi

        if [ "$scratch_cap" -gt 0 ]; then
            echo "export VLLM_EXPERT_SCRATCH_CAPACITY=$scratch_cap"
            echo "export VLLM_SCRATCH_BANKS=$scratch_banks"
        fi

        if [ "$cutoff" = "1" ]; then
            echo "export VLLM_CUTOFF_BOUNDARY=1"
            echo "export VLLM_EXPERT_EAGER_ROUTING_PREFILL=all"
            echo "export VLLM_EXPERT_EAGER_ROUTING_DECODE=all"
        elif [ "$step_boundary" = "1" ]; then
            echo "export VLLM_STEP_BOUNDARY=1"
            echo "export VLLM_EXPERT_EAGER_ROUTING_PREFILL=all"
            echo "export VLLM_EXPERT_EAGER_ROUTING_DECODE=all"
        elif [ "$offload" = "1" ] && [ "$scratch_cap" -gt 0 ]; then
            echo "export VLLM_EXPERT_EAGER_ROUTING_PREFILL=all"
            echo "export VLLM_EXPERT_EAGER_ROUTING_DECODE=all"
        fi

        echo "export VLLM_PP_DECODE_FREEZE=$decode_freeze"
        echo "export VLLM_STEP_PROFILE=1"
    } > "$ENV_FILE"

    SERVER_LOG="$KV_DIR/logs/server_${label}_$(date +%Y%m%d_%H%M%S).log"
    log "Starting $label: pct=${pct}% scratch=$scratch_cap → $SERVER_LOG"

    bash "$KV_DIR/paladin-start-server.sh" "$SERVER_LOG" "$ENV_FILE"
}

run_w2adv() {
    local label="$1"
    local W2OUT="$RESULTS/${label}_w2adv.json"

    log "  w2-adv c=$CONCURRENCY → $W2OUT"

    cd /mnt/raid0_ssd/jinpyo
    bash scripts/paladin-w2-adv.sh "$W2OUT" "$CONCURRENCY"

    if [ -f "$W2OUT" ]; then
        python3 -c "
import json
with open('$W2OUT') as f:
    d = json.load(f)
s = d['summary']
tpot = s.get('tpot_ms', {})
ttft = s.get('ttft_ms', {})
tps = s.get('tokens_per_second', 0)
print('  >> TPS=%.1f TPOT_p50=%.1f TPOT_p99=%.1f TTFT_p99=%.0f OK=%s/%s' % (
    tps, tpot.get('p50',0), tpot.get('p99',0),
    ttft.get('p99',0),
    s.get('successful',0), s.get('total_requests',0)))
" 2>&1 || true
    fi
}

run_quality() {
    local label="$1"

    if ! curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
        log "  WARNING: Server died after w2-adv, skipping quality"
        return 1
    fi

    log "  quality: $label (gsm8k + ifeval, c=8)"
    python3 ~/scripts/stage1_quality_benchmark.py collect \
        --label "$label" \
        --concurrency 8 \
        --benchmarks gsm8k ifeval \
        --output-dir "$RESULTS" \
        2>&1 | tee "$RESULTS/${label}_quality.log"

    local QFILE="$RESULTS/${label}.json"
    python3 -c "
import json
try:
    with open('$QFILE') as f:
        d = json.load(f)
    scores = d.get('bench_scores', {})
    for name, val in scores.items():
        if isinstance(val, dict):
            for k, v in val.items():
                print(f'  {name}/{k}: {v}')
        else:
            print(f'  {name}: {val}')
except Exception as e:
    print(f'  quality parse error: {e}')
" 2>&1 || true
}

# =============================================================================
# MAIN LOOP
# =============================================================================
for cfg_str in "${ALL_CONFIGS[@]}"; do
    IFS='|' read -r label offload elastic cutoff step_boundary decode_freeze pct scratch_banks <<< "$cfg_str"
    scratch_cap=$(pct_to_scratch "$pct")

    found=0
    for s in "${SELECTED[@]}"; do [ "$s" = "$label" ] && found=1 && break; done
    [ "$found" = "0" ] && continue

    log "============================================================"
    log "$label: pct=${pct}% scratch=$scratch_cap (localE=$LOCAL_E)"
    log "============================================================"

    if ! start_server "$label" "$offload" "$elastic" "$cutoff" "$step_boundary" \
                      "$decode_freeze" "$pct" "$scratch_banks"; then
        log "SKIP $label — server failed to start"
        continue
    fi

    # 1. W2-ADV benchmark
    run_w2adv "$label"

    # 2. Quality benchmark (gsm8k + ifeval)
    run_quality "$label" || true

    kill_server
    sleep 10
    log "$label complete"
    log ""
done

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "============================================================"
log "CUTOFF SWEEP TP2 SUMMARY"
log "============================================================"

python3 << 'PYEOF'
import json, os

results_dir = "/mnt/raid0_ssd/jinpyo/results/cutoff_sweep_tp2"
labels = ["CB_20", "SB_20", "CB_25", "SB_25", "CB_30", "SB_30"]

print()
print(f"{'Config':<12} {'TPS':>7} {'TPOT_p50':>10} {'TPOT_p99':>10} "
      f"{'TTFT_p50':>10} {'TTFT_p99':>12} {'GSM8K':>7} {'IFEval':>7}")
print("-" * 85)

for label in labels:
    w2 = os.path.join(results_dir, f"{label}_w2adv.json")
    q = os.path.join(results_dir, f"{label}.json")

    tps = tp50 = tp99 = fp50 = fp99 = 0
    gsm = ife = "N/A"

    if os.path.exists(w2):
        try:
            with open(w2) as f:
                d = json.load(f)
            s = d["summary"]
            tps = s.get("tokens_per_second", 0)
            tp50 = s.get("tpot_ms", {}).get("p50", 0)
            tp99 = s.get("tpot_ms", {}).get("p99", 0)
            fp50 = s.get("ttft_ms", {}).get("p50", 0)
            fp99 = s.get("ttft_ms", {}).get("p99", 0)
        except Exception:
            pass

    if os.path.exists(q):
        try:
            with open(q) as f:
                d = json.load(f)
            scores = d.get("bench_scores", {})
            g = scores.get("gsm8k", {})
            if isinstance(g, dict):
                gsm = "%.1f%%" % (g.get("accuracy", 0) * 100)
            i = scores.get("ifeval", {})
            if isinstance(i, dict):
                ife = "%.1f%%" % (i.get("prompt_strict_accuracy", 0) * 100)
        except Exception:
            pass

    print(f"{label:<12} {tps:>7.1f} {tp50:>9.1f}ms {tp99:>9.1f}ms "
          f"{fp50:>9.0f}ms {fp99:>10.0f}ms {gsm:>7} {ife:>7}")

PYEOF

log ""
log "Results: $RESULTS/"
log "Total elapsed: $(($(date +%s)-START_TIME))s"
