#!/usr/bin/env bash
# ==============================================================================
# run_resource_profile.sh - VMware / Ubuntu Resource Profiling Runner
# ==============================================================================
# Usage:
#   bash scripts/run_resource_profile.sh <profile_name>
# Examples:
#   bash scripts/run_resource_profile.sh low
#   bash scripts/run_resource_profile.sh medium
#   bash scripts/run_resource_profile.sh high
# ==============================================================================

set -e

PROFILE=${1:-"default"}
OUTPUT_DIR="results/resource_profiles"
mkdir -p "$OUTPUT_DIR"

PROFILE_LOG="$OUTPUT_DIR/profile_${PROFILE}.log"
TIME_OUTPUT="$OUTPUT_DIR/time_${PROFILE}.txt"

echo "=========================================================="
echo " Running Resource Profile: ${PROFILE}"
echo "=========================================================="
echo "Output Log: $PROFILE_LOG"
echo "Time Stats: $TIME_OUTPUT"

# Record snapshot of current memory state
echo "=== System Memory Before Run ===" > "$PROFILE_LOG"
free -h >> "$PROFILE_LOG" 2>&1 || true
echo "" >> "$PROFILE_LOG"

# Execute standard experiment benchmark under GNU time
if command -v /usr/bin/time &> /dev/null; then
    /usr/bin/time -v -o "$TIME_OUTPUT" python3 run_experiment.py --seed 42 --frames 5 --trace-length 2000 >> "$PROFILE_LOG" 2>&1
elif command -v time &> /dev/null; then
    (time python3 run_experiment.py --seed 42 --frames 5 --trace-length 2000) >> "$PROFILE_LOG" 2>&1
else
    python3 run_experiment.py --seed 42 --frames 5 --trace-length 2000 >> "$PROFILE_LOG" 2>&1
fi

echo "=== System Memory After Run ===" >> "$PROFILE_LOG"
free -h >> "$PROFILE_LOG" 2>&1 || true

echo "=========================================================="
echo " Profile [${PROFILE}] execution complete."
echo " Check $TIME_OUTPUT for Max RSS and CPU execution times."
echo "=========================================================="
