#!/usr/bin/env bash
set -euo pipefail
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-factguard_chat_v8_${TS}}"
OUT_LOG="${LOG_DIR}/${RUN_ID}.out.log"
ERR_LOG="${LOG_DIR}/${RUN_ID}.err.log"
SUMMARY_LOG="${LOG_DIR}/${RUN_ID}.summary.log"

: > "$ERR_LOG"

export CAUSALOS_WEB_TIMEOUT="${CAUSALOS_WEB_TIMEOUT:-12}"
export CAUSALOS_SHOW_TRACE="${CAUSALOS_SHOW_TRACE:-1}"
export CAUSALOS_SHOW_PROVENANCE="${CAUSALOS_SHOW_PROVENANCE:-1}"
export CAUSALOS_TRACE_DIR="${CAUSALOS_TRACE_DIR:-./logs}"
export CAUSALOS_FG_RESOLVERS="${CAUSALOS_FG_RESOLVERS:-semanticscholar,crosscite,arxiv_search}"

rm -f "$CAUSALOS_TRACE_DIR/trace_turn_1.json" 2>/dev/null || true

python3 CausalChatAgent_factguard_chat_v8.py \
  2> >(tee -a "$ERR_LOG" >&2) \
  | tee "$OUT_LOG"

{
  echo "---- SUMMARY ----"
  grep -E "\[Build\]|\[Turn [0-9]+\]\[Provenance\]|Assistant:|\[Trace\]" "$OUT_LOG" || true
  echo "[Info] stdout: $OUT_LOG"
  echo "[Info] stderr: $ERR_LOG"
  if [[ -f "$CAUSALOS_TRACE_DIR/trace_turn_1.json" ]]; then
    echo "[Info] trace_turn_1.json: $(ls -la "$CAUSALOS_TRACE_DIR/trace_turn_1.json")"
  fi
} | tee "$SUMMARY_LOG"
