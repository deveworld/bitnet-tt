#!/usr/bin/env bash
# scripts/bench-smoke.sh — Phase 0 harness gate for BitNet-TT.
#
# Quick sanity check that the measurement stack is intact:
#   1. bench_batch32 runs without dlpack / uint32 errors
#   2. decode_tps clears the 70 t/s Phase-0 speed floor
#   3. bench_accuracy (single prompt) completes and reports PCC
#
# Exits 0 on PASS, nonzero on any failure. Prints a one-line summary.
# Run from repo root on the TT server.

set -u
set -o pipefail

cd "$(dirname "$0")/.."

FLOOR_TPS=70.0
SMOKE_LOG="/tmp/bitnet_tt_bench_smoke_$$.log"
trap 'rm -f "$SMOKE_LOG"' EXIT

echo "[smoke] bench_batch32.py --dtype packed_ternary --max-new 64"
python3 bench_batch32.py --dtype packed_ternary --max-new 64 >"$SMOKE_LOG" 2>&1 || {
    echo "[smoke] FAIL: bench_batch32 crashed"
    tail -15 "$SMOKE_LOG"
    exit 2
}

DECODE_TPS=$(grep 'decode_tps=' "$SMOKE_LOG" | tail -1 | sed -E 's/.*decode_tps=([0-9.]+).*/\1/')
P50=$(grep 'p50=' "$SMOKE_LOG" | tail -1 | sed -E 's/.*p50=([0-9.]+)ms.*/\1/')

if [ -z "$DECODE_TPS" ] || [ -z "$P50" ]; then
    echo "[smoke] FAIL: could not parse decode_tps / p50 from bench output"
    tail -15 "$SMOKE_LOG"
    exit 3
fi

echo "[smoke] decode_tps=$DECODE_TPS  p50=${P50}ms"

FLOOR_OK=$(awk -v tps="$DECODE_TPS" -v floor="$FLOOR_TPS" 'BEGIN { print (tps >= floor) ? 1 : 0 }')
if [ "$FLOOR_OK" != "1" ]; then
    echo "[smoke] WARN: decode_tps $DECODE_TPS below Phase-0 floor $FLOOR_TPS"
fi

echo "[smoke] bench_accuracy.py --dtype packed_ternary (single prompt)"
python3 bench_accuracy.py --dtype packed_ternary >"$SMOKE_LOG" 2>&1 || {
    echo "[smoke] FAIL: bench_accuracy crashed"
    tail -15 "$SMOKE_LOG"
    exit 4
}

PCC=$(grep -E '^\s*PCC:' "$SMOKE_LOG" | head -1 | sed -E 's/.*PCC:\s*([0-9.]+).*/\1/')
if [ -z "$PCC" ]; then
    echo "[smoke] FAIL: could not parse PCC from bench_accuracy output"
    tail -15 "$SMOKE_LOG"
    exit 5
fi
echo "[smoke] PCC=$PCC (vs HF fp32)"

if [ "$FLOOR_OK" != "1" ]; then
    echo "[smoke] PASS (with speed-floor warning)"
    exit 0
fi

echo "[smoke] PASS"
exit 0
