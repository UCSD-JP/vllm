#!/bin/bash
# =============================================================================
# Cutoff vs Step-Boundary Sweep — TP4 (solab 4×H200 NVL)
# =============================================================================
#
# Uses solab-start-server.sh (고정 스크립트) + solab-w2-adv.sh (고정 벤치마크)
# Server: max_model_len=262144, max_num_seqs=64, TP=4, gpu_mem=0.90
# W2-ADV: c=32, t=12, fixed-output-tokens=256
#
# Demand-driven offload: init=all resident, KV pressure → shrink.
# Scratch = LOCAL_E * pct / 100, rounded down to GROUP_SIZE alignment.
#
# Usage:
#   bash scripts/run_cutoff_sweep_tp4.sh              # 전체
#   bash scripts/run_cutoff_sweep_tp4.sh CB_20 SB_20  # 선택
#
# =============================================================================
set -euo pipefail

source /home/ucsd/miniconda3/etc/profile.d/conda.sh
conda activate vllm
export HF_HOME=/home/ucsd/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/home/ucsd/tmp
mkdir -p "$TMPDIR"

PORT=8000
START_TIME=$(date +%s)

LOCAL_E=512          # experts per GPU (Qwen3-Next-80B-A3B: 512 per GPU on TP4)
GROUP_SIZE=4         # group alignment

RESULTS=~/results/cutoff_sweep_tp4
mkdir -p "$RESULTS"

# ── pct → scratch (group-aligned) ──
pct_to_scratch() {
    local pct=$1
    local raw=$(( LOCAL_E * pct / 100 ))
    echo $(( (raw / GROUP_SIZE) * GROUP_SIZE ))
}

# ── Config matrix ──
# LABEL|OFFLOAD|ELASTIC|CUTOFF|STEP_BOUNDARY|DECODE_FREEZE|PCT|SCRATCH_BANKS
OFFLOAD_PCTS=(20 30 40)

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

# ── Generate env file for a config ──
make_env() {
    local label=$1 offload=$2 elastic=$3 cutoff=$4 step_boundary=$5
    local decode_freeze=$6 pct=$7 scratch_banks=$8
    local scratch_cap
    scratch_cap=$(pct_to_scratch "$pct")
    local envfile="$RESULTS/env_${label}.sh"

    cat > "$envfile" << ENVEOF
# Auto-generated env for $label (pct=${pct}%, scratch=$scratch_cap)
export VLLM_EXPERT_TRACE=0
export VLLM_EXPERT_DEBUG=0
ENVEOF

    if [ "$offload" = "1" ]; then
        cat >> "$envfile" << 'ENVEOF'
export VLLM_EXPERT_OFFLOAD_ENABLE=1
export VLLM_VMM_EXPERT_POOL=1
export VLLM_VMM_PHASE_C=1
export VLLM_EXPERT_DIAG_DUMP=100
ENVEOF
    fi

    if [ "$elastic" = "1" ]; then
        cat >> "$envfile" << 'ENVEOF'
export VLLM_ELASTIC_KV_ENABLE=1
export VLLM_ELASTIC_KV_GROUPS_PER_EXPAND=4
export VLLM_ELASTIC_KV_MIN_RESIDENT=0.5
export VLLM_PREFIX_PROTECTION_ENABLE=1
export VLLM_PP_C_RELOAD_MS=0.63
export VLLM_PP_H_CAP=64
export VLLM_PP_P_REUSE_ALPHA=0.5
export VLLM_PP_CC_AGE_SCALE=2000
ENVEOF
    fi

    if [ "$scratch_cap" -gt 0 ]; then
        echo "export VLLM_EXPERT_SCRATCH_CAPACITY=$scratch_cap" >> "$envfile"
        echo "export VLLM_SCRATCH_BANKS=$scratch_banks" >> "$envfile"
    fi

    if [ "$cutoff" = "1" ]; then
        cat >> "$envfile" << 'ENVEOF'
export VLLM_CUTOFF_BOUNDARY=1
export VLLM_EXPERT_EAGER_ROUTING_PREFILL=all
export VLLM_EXPERT_EAGER_ROUTING_DECODE=all
ENVEOF
    elif [ "$step_boundary" = "1" ]; then
        cat >> "$envfile" << 'ENVEOF'
export VLLM_STEP_BOUNDARY=1
export VLLM_EXPERT_EAGER_ROUTING_PREFILL=all
export VLLM_EXPERT_EAGER_ROUTING_DECODE=all
ENVEOF
    elif [ "$offload" = "1" ] && [ "$scratch_cap" -gt 0 ]; then
        cat >> "$envfile" << 'ENVEOF'
export VLLM_EXPERT_EAGER_ROUTING_PREFILL=all
export VLLM_EXPERT_EAGER_ROUTING_DECODE=all
ENVEOF
    fi

    echo "export VLLM_PP_DECODE_FREEZE=$decode_freeze" >> "$envfile"
    echo "export VLLM_STEP_PROFILE=1" >> "$envfile"


    echo "$envfile"
}

start_server() {
    local label=$1 offload=$2 elastic=$3 cutoff=$4 step_boundary=$5
    local decode_freeze=$6 pct=$7 scratch_banks=$8
    local scratch_cap
    scratch_cap=$(pct_to_scratch "$pct")

    kill_server

    # Generate env file
    local envfile
    envfile=$(make_env "$label" "$offload" "$elastic" "$cutoff" "$step_boundary" \
                       "$decode_freeze" "$pct" "$scratch_banks")

    SERVER_LOG="$RESULTS/server_${label}_$(date +%Y%m%d_%H%M%S).log"

    log "Starting $label: cutoff=$cutoff SB=$step_boundary DF=$decode_freeze pct=${pct}% scratch=$scratch_cap (localE=$LOCAL_E)"
    log "  env: $envfile"

    # Use fixed solab-start-server.sh (max_model_len=262144, max_num_seqs=64)
    if ! bash ~/scripts/solab-start-server.sh "$SERVER_LOG" "$envfile"; then
        log "ERROR: Server failed to start"
        tail -40 "$SERVER_LOG" 2>/dev/null || true
        return 1
    fi

    log "Server ready"
    grep -iE "CutoffBoundary|StepBoundary|ACTIVATING|ElasticKV:|offload|scratch|min_resident" \
        "$SERVER_LOG" 2>/dev/null | head -15 || true
    return 0
}

run_w2adv() {
    local label="$1"
    local W2OUT="$RESULTS/${label}_w2adv.json"

    log "  w2-adv: $label"
    bash ~/scripts/solab-w2-adv.sh "$W2OUT" 2>&1 | tee "$RESULTS/${label}_w2adv.log"

    python3 -c "
import json
with open('$W2OUT') as f:
    d = json.load(f)
s = d['summary']
tpot = s.get('tpot_ms', {})
ttft = s.get('ttft_ms', {})
tps = s.get('tokens_per_second', 0)
print('  W2-ADV: TPS=%.1f TPOT_p50=%.1f TPOT_p99=%.1f TTFT_p50=%.0f TTFT_p99=%.0f' % (
    tps, tpot.get('p50',0), tpot.get('p99',0), ttft.get('p50',0), ttft.get('p99',0)))
" 2>&1 || true
}

run_quality() {
    local label="$1"
    local run_id="${2:-1}"

    if ! curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
        log "  WARNING: Server died, skipping quality"
        return 1
    fi

    local qlabel="${label}_r${run_id}"
    log "  quality: $qlabel (gsm8k + ifeval + mmlu-pro, c=8)"
    python3 ~/scripts/stage1_quality_benchmark.py collect \
        --label "$qlabel" \
        --concurrency 8 \
        --benchmarks gsm8k ifeval mmlu-pro \
        --output-dir "$RESULTS" \
        2>&1 | tee "$RESULTS/${qlabel}_quality.log"

    local QFILE="$RESULTS/${qlabel}.json"
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

    # 1. W2-ADV benchmark (solab-w2-adv.sh: c=32, t=12)
    run_w2adv "$label"

    # 2. Quality benchmark ×3 (gsm8k + ifeval + mmlu-pro, c=8)
    for run in 1 2 3; do
        run_quality "$label" "$run" || true
    done

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
log "CUTOFF SWEEP TP4 SUMMARY"
log "============================================================"

python3 << 'PYEOF'
import json, os

results_dir = os.path.expanduser("~/results/cutoff_sweep_tp4")
labels = ["CB_20", "SB_20", "CB_30", "SB_30", "CB_40", "SB_40"]

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
