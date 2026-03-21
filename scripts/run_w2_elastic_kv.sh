#!/bin/bash
# =============================================================================
# W2 Elastic KV MVP — B0 + E1-E4 Matrix (solab 4×H200 NVL)
# =============================================================================
# B0: Baseline (no offload, no elastic KV)
# B1: Expert offload only (no elastic KV)
# E1: Elastic KV — conservative (groups_per_expand=2, min_resident=0.7)
# E2: Elastic KV — balanced    (groups_per_expand=4, min_resident=0.5)
# E3: Elastic KV — aggressive  (groups_per_expand=8, min_resident=0.3)
# E4: Elastic KV — max blocks capped (groups_per_expand=4, max_expand=64)
#
# 코드 위치: /home/jp/vllm_elastic_kv/
# 기존 gpusim 대비 차이: elastic KV가 ON이면 KV cache 부족 시
#   expert 메모리를 evict → KV block으로 자동 변환
#
# 실행:
#   ssh solab 'screen -dmS elastickv bash -c "bash ~/run_w2_elastic_kv.sh > ~/elastickv.log 2>&1"'
#
# 특정 config만 실행:
#   bash run_w2_elastic_kv.sh B0 E2
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

# ── vLLM 코드 경로: elastic-kv-mvp 브랜치 사용 ──
# solab에 배포 후 아래 경로를 실제 경로로 변경
VLLM_CODE_PATH="${VLLM_CODE_PATH:-/home/ucsd/vllm_elastic_kv}"

MODEL="/home/ucsd/huggingface/hub/models--Qwen--Qwen3-Next-80B-A3B-Instruct/snapshots/9c7f2fbe84465e40164a94cc16cd30b6999b0cc7"
PORT=8000
GPU_MEM_UTIL=0.90
TP=4
MML=49152
MAX_SEQS=192
START_TIME=$(date +%s)

RESULTS=~/results/w2_elastic_kv
mkdir -p "$RESULTS"

# ── Config matrix ──────────────────────────────────────────────────
# Format: "LABEL|OFFLOAD|VMM|PHASE_C|ELASTIC_KV|GROUPS_PER_EXPAND|MIN_RESIDENT|MAX_EXPAND_BLOCKS"
ALL_CONFIGS=(
    "B0|0|0|0|0|0|0|0"
    "B1|1|1|1|0|0|0|0"
    "E1|1|1|1|1|2|0.7|0"
    "E2|1|1|1|1|4|0.5|0"
    "E3|1|1|1|1|8|0.3|0"
    "E4|1|1|1|1|4|0.5|64"
)

# Parse CLI args
SELECTED=()
for arg in "$@"; do
    SELECTED+=("$arg")
done
if [ ${#SELECTED[@]} -eq 0 ]; then
    SELECTED=("B0" "B1" "E1" "E2" "E3" "E4")
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

    SERVER_LOG="$RESULTS/server_${label}_$(date +%Y%m%d_%H%M%S).log"

    log "Starting $label: offload=$offload vmm=$vmm phaseC=$phase_c elastic=$elastic"
    [ "$elastic" = "1" ] && log "  elastic: groups=$groups_per_expand min_resident=$min_resident max_expand=$max_expand"

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONPATH="$VLLM_CODE_PATH:${PYTHONPATH:-}" \
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
            tail -30 "$SERVER_LOG"
            return 1
        fi
        if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
            log "Server ready ($((i*5))s)"
            grep -i "offload\|phase.c\|elastic.kv\|VMM\|expert.*freed\|geometry" "$SERVER_LOG" 2>/dev/null | head -10 || true
            return 0
        fi
        sleep 5
    done
    log "ERROR: Server timeout"
    return 1
}

run_kv_heavy() {
    local label="$1" sessions="$2" concurrency="$3" turns="${4:-8}"
    local outfile="$RESULTS/${label}_kv_c${sessions}_t${turns}.json"

    log "  kv_heavy: $label c=$sessions t=$turns"
    python3 ~/scripts/benchmark_kv_heavy.py \
        --sessions "$sessions" --turns "$turns" --concurrency "$concurrency" \
        --model "$MODEL" \
        --server "http://localhost:$PORT/v1" \
        --output "$outfile" \
        2>&1 || log "  WARNING: $label kv_heavy failed"

    if [ -f "$outfile" ]; then
        python3 -c "
import json
with open('$outfile') as f:
    d = json.load(f)
s = d.get('summary', {})
kv = s.get('kv', {})
print('  >> OK=%d/%d  TPOT=%.1fms  KV_peak=%.1f%%  Prefix\$=%s  Preempt=%d' % (
    s.get('successful', 0), s.get('total_requests', 0),
    s.get('tpot_ms', {}).get('mean', 0),
    kv.get('gpu_kv_peak_pct', 0),
    '%.1f%%' % (kv['prefix_cache_hit_rate']*100) if kv.get('prefix_cache_hit_rate') is not None else 'N/A',
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
    IFS='|' read -r label offload vmm phase_c elastic groups_per_expand min_resident max_expand <<< "$cfg_str"

    # Skip if not selected
    found=0
    for s in "${SELECTED[@]}"; do
        [ "$s" = "$label" ] && found=1 && break
    done
    [ "$found" = "0" ] && continue

    log "============================================================"
    log "$label: offload=$offload elastic=$elastic"
    [ "$elastic" = "1" ] && log "  groups=$groups_per_expand min_resident=$min_resident max_expand=$max_expand"
    log "============================================================"

    if ! start_server "$label" "$offload" "$vmm" "$phase_c" "$elastic" \
                      "$groups_per_expand" "$min_resident" "$max_expand"; then
        log "SKIP $label — server failed to start"
        continue
    fi

    # W2 workloads: c=32 (기본), c=64 (중간), c=128 (stress)
    run_kv_heavy "$label" 32 32 8
    if check_health; then
        run_kv_heavy "$label" 64 64 8
    fi
    if check_health; then
        run_kv_heavy "$label" 128 128 8
    fi

    kill_server
    sleep 10
done

# =============================================================================
# SUMMARY
# =============================================================================
log ""
log "============================================================"
log "ELASTIC KV MVP EXPERIMENT SUMMARY"
log "============================================================"

python3 << 'PYEOF'
import json, glob, os

print(f"\n{'Config':<45} {'OK':>9} {'TPOT':>8} {'KV_peak':>9} {'Prefix$':>9} {'Preempt':>8}")
print("-" * 95)

for f in sorted(glob.glob(os.path.expanduser("~/results/w2_elastic_kv/*.json"))):
    if "server_" in f:
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
        print(f"{name:<45} {ok:>4}/{total:<4} {tpot:>7.1f}ms {kv_peak:>8.1f}% {prefix_s:>9} {preempt:>8}")
    except Exception as e:
        print(f"  SKIP {os.path.basename(f)}: {e}")

PYEOF

log ""
log "Results in: $RESULTS/"
log "Total elapsed: $(($(date +%s)-START_TIME))s"
