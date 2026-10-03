#!/usr/bin/env bash
# ==============================================================================
# collect_environment.sh - Hardware & OS Environment Profiler
# ==============================================================================
# Gathers system specifications and memory diagnostics for VMware / Ubuntu runs.
# ==============================================================================

set -e

OUTPUT_DIR="results/resource_profiles"
mkdir -p "$OUTPUT_DIR"
ENV_LOG="$OUTPUT_DIR/system_environment.log"

echo "=========================================================="
echo " Gathering System Environment Information"
echo "=========================================================="

{
    echo "=== TIMESTAMP ==="
    date -u +"%Y-%m-%dT%H:%M:%SZ"
    echo ""

    echo "=== OPERATING SYSTEM & KERNEL ==="
    uname -a || true
    echo ""

    echo "=== CPU TOPOLOGY ==="
    lscpu || cat /proc/cpuinfo || true
    echo ""

    echo "=== SYSTEM MEMORY (RAM & SWAP) ==="
    free -h || true
    echo ""

    echo "=== PYTHON VERSION & PACKAGES ==="
    python3 --version || python --version || true
    pip list 2>/dev/null || true
    echo ""
} > "$ENV_LOG"

echo "System environment details recorded in $ENV_LOG"
