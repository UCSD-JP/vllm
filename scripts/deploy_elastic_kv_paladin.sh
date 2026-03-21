#!/bin/bash
# =============================================================================
# Deploy Elastic KV MVP to Paladin (PYTHONPATH override — 기존 gpusim 유지)
# =============================================================================
# 기존 site-packages의 gpusim 코드를 건드리지 않고,
# /mnt/raid0_ssd/jinpyo/vllm_elastic_kv/ 에 소스를 배포하여
# PYTHONPATH로 우선 로딩합니다.
#
# Usage: bash scripts/deploy_elastic_kv_paladin.sh
# =============================================================================
set -euo pipefail

PALADIN="jinpyo@paladin.ucsd.edu"
LOCAL_REPO="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/mnt/raid0_ssd/jinpyo/vllm_elastic_kv"

echo "=== Elastic KV MVP Deployment to Paladin ==="
echo "Local:  $LOCAL_REPO"
echo "Remote: $PALADIN:$REMOTE_DIR"
echo ""

# 1. rsync vllm/ 디렉토리 (소스 코드만, .git 제외)
echo "[1/3] Syncing vllm source..."
rsync -avz --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.egg-info' \
    --exclude='build/' \
    --exclude='dist/' \
    --exclude='.pytest_cache' \
    "$LOCAL_REPO/vllm/" \
    "$PALADIN:$REMOTE_DIR/vllm/"

# 2. 실험 스크립트 복사
echo ""
echo "[2/3] Copying experiment scripts..."
rsync -avz "$LOCAL_REPO/scripts/" "$PALADIN:$REMOTE_DIR/scripts/"

# 3. 검증
echo ""
echo "[3/3] Verifying deployment..."
ssh "$PALADIN" 'bash -s' << VERIFY_SCRIPT
SITE="$REMOTE_DIR/vllm"
FAIL=0

echo "File sizes on Paladin:"
ls -la \$SITE/elastic_kv_config.py | awk '{print "  elastic_kv_config.py:", \$5, "bytes"}'
ls -la \$SITE/vmm_pool.py | awk '{print "  vmm_pool.py:", \$5, "bytes"}'
ls -la \$SITE/v1/engine/core.py | awk '{print "  core.py:", \$5, "bytes"}'
ls -la \$SITE/v1/worker/gpu_worker.py | awk '{print "  gpu_worker.py:", \$5, "bytes"}'
ls -la \$SITE/v1/worker/gpu_model_runner.py | awk '{print "  gpu_model_runner.py:", \$5, "bytes"}'
ls -la \$SITE/v1/core/sched/scheduler.py | awk '{print "  scheduler.py:", \$5, "bytes"}'
ls -la \$SITE/model_executor/layers/fused_moe/expert_predictor.py | awk '{print "  expert_predictor.py:", \$5, "bytes"}'

echo ""
echo "Critical marker checks:"

check_marker() {
    local file="\$1" pattern="\$2" label="\$3"
    if grep -q "\$pattern" "\$file" 2>/dev/null; then
        echo "  OK  \$label"
    else
        echo "  FAIL \$label"
        FAIL=1
    fi
}

check_marker "\$SITE/elastic_kv_config.py" "VLLM_ELASTIC_KV_ENABLE" "elastic_kv_config: env var"
check_marker "\$SITE/v1/engine/core.py" "_init_elastic_kv_workers" "core.py: phase 1 init"
check_marker "\$SITE/v1/engine/core.py" "_register_elastic_kv_handler" "core.py: phase 2 handler"
check_marker "\$SITE/v1/engine/core.py" "_update_elastic_kv_geometry" "core.py: geometry update"
check_marker "\$SITE/v1/engine/core.py" "elastic_kv_rollback" "core.py: rollback"
check_marker "\$SITE/v1/worker/gpu_worker.py" "init_elastic_kv" "gpu_worker: init_elastic_kv"
check_marker "\$SITE/v1/worker/gpu_worker.py" "update_elastic_kv_geometry" "gpu_worker: geometry update"
check_marker "\$SITE/v1/worker/gpu_worker.py" "elastic_kv_prepare" "gpu_worker: prepare"
check_marker "\$SITE/v1/worker/gpu_worker.py" "elastic_kv_commit" "gpu_worker: commit"
check_marker "\$SITE/v1/worker/gpu_worker.py" "elastic_kv_rollback" "gpu_worker: rollback"
check_marker "\$SITE/v1/worker/gpu_worker.py" "_init_expert_offloading" "gpu_worker: expert offloading"
check_marker "\$SITE/v1/worker/gpu_model_runner.py" "_get_moe_layers" "model_runner: _get_moe_layers"
check_marker "\$SITE/v1/core/sched/scheduler.py" "deficit_blocks" "scheduler: deficit blocks"
check_marker "\$SITE/vmm_pool.py" "contract_kv_physical_pages" "vmm_pool: contract"

if [ "\$FAIL" -ne 0 ]; then
    echo ""
    echo "ERROR: Verification failed!"
    exit 1
fi
echo ""
echo "All checks passed."
VERIFY_SCRIPT

echo ""
echo "=== Deployment complete ==="
echo ""
echo "실험 실행:"
echo "  ssh $PALADIN"
echo "  bash $REMOTE_DIR/scripts/run_w2_elastic_kv_paladin.sh B0 E2"
echo ""
echo "또는 screen으로:"
echo "  ssh $PALADIN 'screen -dmS elastickv bash -c \"bash $REMOTE_DIR/scripts/run_w2_elastic_kv_paladin.sh > ~/elastickv.log 2>&1\"'"
