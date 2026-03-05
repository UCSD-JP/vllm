#!/bin/bash
# Phase 1 Expert Cache Miss Mitigation — Integration Test
#
# Usage: bash scripts/test_phase1_mitigation.sh
#
# Prerequisites:
# - vLLM server running with expert offload enabled
# - SWE-bench benchmark client available
#
# This script sets Phase 1 env vars, starts the server, and runs
# c=16 SWE-bench to compare hit rates vs baseline.

set -euo pipefail

echo "=== Phase 1 Expert Cache Miss Mitigation — Integration Test ==="
echo ""

# Phase 1 env vars
export VLLM_EXPERT_NEVER_EVICT_K=10
export VLLM_EXPERT_UNION_STEPS=3
export VLLM_EXPERT_EAGER_ON_ANY_MISS=0
export VLLM_EXPERT_DIAG_DUMP=100

echo "Configuration:"
echo "  VLLM_EXPERT_NEVER_EVICT_K=${VLLM_EXPERT_NEVER_EVICT_K}"
echo "  VLLM_EXPERT_UNION_STEPS=${VLLM_EXPERT_UNION_STEPS}"
echo "  VLLM_EXPERT_EAGER_ON_ANY_MISS=${VLLM_EXPERT_EAGER_ON_ANY_MISS}"
echo "  VLLM_EXPERT_DIAG_DUMP=${VLLM_EXPERT_DIAG_DUMP}"
echo ""

# Expected results:
# Baseline hit rate: 99.38% (final), 97.6% (step 100)
# Target:           >99.7% (final), >98.5% (step 100)

echo "Expected improvement:"
echo "  Baseline: 99.38% final, 97.6% at step 100"
echo "  Target:   >99.7% final, >98.5% at step 100"
echo ""

# Start server with offload (modify path as needed)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "${SCRIPT_DIR}/start_server_toolcall.sh" ]; then
    echo "Starting vLLM server with expert offload..."
    bash "${SCRIPT_DIR}/start_server_toolcall.sh" &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"

    # Wait for server readiness
    echo "Waiting for server to be ready..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            break
        fi
        sleep 1
    done

    # Run benchmark
    echo ""
    echo "Running SWE-bench c=16..."
    if [ -f "${SCRIPT_DIR}/run_kv_pressure_experiment.py" ]; then
        python "${SCRIPT_DIR}/run_kv_pressure_experiment.py" \
            --concurrency 16 \
            --scenario swebench 2>&1 | tee /tmp/phase1_mitigation_results.log
    else
        echo "WARNING: run_kv_pressure_experiment.py not found"
        echo "Run your benchmark manually with the env vars above"
    fi

    # Collect diagnostic logs
    echo ""
    echo "=== Collecting diagnostic logs ==="
    echo "Check server logs for lines containing:"
    echo "  - 'phase1_mitigation:' (timing summary)"
    echo "  - '[Phase 1 Mitigation]' (miss diagnostics)"
    echo "  - 'Expert eager-on-any-miss:' (3-E triggers)"

    # Cleanup
    echo ""
    echo "Stopping server..."
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
else
    echo "No start_server_toolcall.sh found."
    echo "Start your vLLM server manually with the env vars above,"
    echo "then run your benchmark."
fi

echo ""
echo "=== Done ==="
