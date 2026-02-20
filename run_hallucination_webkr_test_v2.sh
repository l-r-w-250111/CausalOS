#!/usr/bin/env bash
set -euo pipefail

# =========================================
# run_hallucination_webkr_test_v2.sh
# - Uses CausalChatAgent_webkr_v2.py (identifier-safe)
# =========================================

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-halluc_ddg_v2_${TS}}"
OUT_LOG="${LOG_DIR}/${RUN_ID}.out.log"
ERR_LOG="${LOG_DIR}/${RUN_ID}.err.log"
SUMMARY_LOG="${LOG_DIR}/${RUN_ID}.summary.log"

export CAUSALOS_MODEL="${CAUSALOS_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export CAUSALOS_ENABLE_FACT_MODE=1
export CAUSALOS_SHOW_TRACE="${CAUSALOS_SHOW_TRACE:-1}"
export CAUSALOS_WEB_MAX="${CAUSALOS_WEB_MAX:-5}"

{
  echo "=== run_hallucination_webkr_test_v2.sh (${RUN_ID}) ==="
  echo "[Info] Model: $CAUSALOS_MODEL"
  echo "[Info] Logs: OUT=$OUT_LOG  ERR=$ERR_LOG  SUMMARY=$SUMMARY_LOG"
  echo "[Info] Fact-mode: $CAUSALOS_ENABLE_FACT_MODE"
  echo "[Info] Trace: CAUSALOS_SHOW_TRACE=$CAUSALOS_SHOW_TRACE"
  echo ""
} | tee "$SUMMARY_LOG"

if [[ ! -f "WebKnowledgeRetriever.py" ]]; then
  echo "[Error] WebKnowledgeRetriever.py missing" | tee -a "$SUMMARY_LOG"; exit 2
fi
if [[ ! -f "CausalChatAgent_webkr_v2.py" ]]; then
  echo "[Error] CausalChatAgent_webkr_v2.py missing" | tee -a "$SUMMARY_LOG"; exit 2
fi

TEST_INPUT="$(cat <<'EOF'
What is the DOI or arXiv identifier of Attention Is All You Need?

What is the boiling point of water at 1 atm?

Provide the DOI of the paper titled "Quantum Strawberry Fields for Infinite Batteries" (2019).

What is the ISSN of the journal "Journal of Completely Imaginary Results"?
EOF
)"

python3 CausalChatAgent_webkr_v2.py <<<"$TEST_INPUT" >"$OUT_LOG" 2>"$ERR_LOG" || true

{
  echo "---- SUMMARY (extracted) ----"
  grep -E "\[WebKnowledge\]|I don't know from the provided sources|\b10\.[0-9]{4,9}/|\b[0-9]{4}\.[0-9]{4,5}\b|\b[0-9]{4}-[0-9]{3}[0-9X]\b" "$OUT_LOG" || true
  echo ""
  echo "[Check] If sources are empty OR no identifier in snippets -> should say don't know"
  grep -n "I don't know from the provided sources" "$OUT_LOG" || true
} | tee -a "$SUMMARY_LOG"

echo "=== done ===" | tee -a "$SUMMARY_LOG"
