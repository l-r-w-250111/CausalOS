#!/usr/bin/env bash
set -euo pipefail

# run_q1q2_factguard.sh

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-q1q2_${TS}}"
OUT_LOG="${LOG_DIR}/${RUN_ID}.out.log"
ERR_LOG="${LOG_DIR}/${RUN_ID}.err.log"
SUMMARY_LOG="${LOG_DIR}/${RUN_ID}.summary.log"

export CAUSALOS_MODEL="${CAUSALOS_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export CAUSALOS_FACT_BASE="${CAUSALOS_FACT_BASE:-oss_v6}"
export CAUSALOS_WEB_TIMEOUT="${CAUSALOS_WEB_TIMEOUT:-12}"

{
  echo "=== run_q1q2_factguard.sh (${RUN_ID}) ==="
  echo "[Info] Model: $CAUSALOS_MODEL"
  echo "[Info] FACT_BASE: $CAUSALOS_FACT_BASE (oss_v6/oss_v7/base)"
  echo "[Info] Logs: OUT=$OUT_LOG ERR=$ERR_LOG SUMMARY=$SUMMARY_LOG"
  echo ""
} | tee "$SUMMARY_LOG"

python3 CausalChatAgent_q1q2.py >"$OUT_LOG" 2>"$ERR_LOG" || true

{
  echo "---- SUMMARY ----"
  grep -E "^\[Q1\]|^\[Q2\]|\"used_web\"|\"used_llm\"|\"used_causalos_guard\"|NO_SUCH_LAW|Iz Beltagy|Kyle Lo|Arman Cohan" "$OUT_LOG" || true
  echo ""
  echo "[Info] Full stdout: $OUT_LOG"
  echo "[Info] Full stderr: $ERR_LOG"
} | tee -a "$SUMMARY_LOG"

echo "=== done ===" | tee -a "$SUMMARY_LOG"
