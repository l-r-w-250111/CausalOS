#!/usr/bin/env bash
set -euo pipefail

# =========================================
# run_demo.sh (compact runner for CausalOS)
# - minimal console output by default
# - full stdout/stderr saved to ./logs/
# =========================================

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-demo_${TS}}"

OUT_LOG="${LOG_DIR}/${RUN_ID}.out.log"
ERR_LOG="${LOG_DIR}/${RUN_ID}.err.log"
SUMMARY_LOG="${LOG_DIR}/${RUN_ID}.summary.log"

# ---- Model ----
export CAUSALOS_MODEL="${CAUSALOS_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

# ---- Console controls ----
SHOW_TRACE="${SHOW_TRACE:-0}"      # 1 -> prints huge JSON to stdout (not recommended)
DEBUG="${DEBUG:-0}"               # 1 -> debug logs (stderr heavy)
FULL_CONSOLE="${FULL_CONSOLE:-0}" # 1 -> print everything to console





# =========================================
# Core flags (robustpack_v5 recommended)
# =========================================
export CAUSALOS_PLACEHOLDER_GUARD="${CAUSALOS_PLACEHOLDER_GUARD:-1}"
export CAUSALOS_GROUND_TOKEN_OVERLAP="${CAUSALOS_GROUND_TOKEN_OVERLAP:-1}"
export CAUSALOS_FRAME_STRICT_MAX="${CAUSALOS_FRAME_STRICT_MAX:-3}"
export CAUSALOS_GROUND_RETRY="${CAUSALOS_GROUND_RETRY:-3}"
export CAUSALOS_GROUND_THR="${CAUSALOS_GROUND_THR:-0.45}"

# v5 grounding improvements
export CAUSALOS_GROUND_CONTENT_ONLY="${CAUSALOS_GROUND_CONTENT_ONLY:-1}"

# robustness
export CAUSALOS_STATE_FALLBACK="${CAUSALOS_STATE_FALLBACK:-1}"
export CAUSALOS_DEFALLBACK_ATOMIC="${CAUSALOS_DEFALLBACK_ATOMIC:-1}"
export CAUSALOS_POLARITY_FIX="${CAUSALOS_POLARITY_FIX:-1}"
export CAUSALOS_POLARITY_SIM_THR="${CAUSALOS_POLARITY_SIM_THR:-0.88}"
export CAUSALOS_TARGET_FALLBACK="${CAUSALOS_TARGET_FALLBACK:-1}"
export CAUSALOS_LATENT_OPT="${CAUSALOS_LATENT_OPT:-1}"

# enforce grounding
export CAUSALOS_ENFORCE_GROUND="${CAUSALOS_ENFORCE_GROUND:-1}"
export CAUSALOS_ENFORCE_THR="${CAUSALOS_ENFORCE_THR:-0.55}"

# v5 span settings
export CAUSALOS_SPAN_MIN_TOK="${CAUSALOS_SPAN_MIN_TOK:-2}"
export CAUSALOS_SPAN_MAX_TOK="${CAUSALOS_SPAN_MAX_TOK:-8}"
export CAUSALOS_SPAN_SPECIFICITY="${CAUSALOS_SPAN_SPECIFICITY:-1}"

# v5 inactive dedup
export CAUSALOS_INACTIVE_DEDUP="${CAUSALOS_INACTIVE_DEDUP:-1}"
export CAUSALOS_IGNORE_INACTIVE="${CAUSALOS_IGNORE_INACTIVE:-1}"
export CAUSALOS_DEDUP_SIM_THR="${CAUSALOS_DEDUP_SIM_THR:-0.92}"


export CAUSALOS_IDS_MARGIN_REF=0.05


export CAUSALOS_REL_COMB=max   # max | mix
export CAUSALOS_REL_EMB_W=0.80 # REL_COMB=mix の時のみ使用

# option scoring
export CAUSALOS_OPT_SCENARIO_REL="${CAUSALOS_OPT_SCENARIO_REL:-1}"
export CAUSALOS_OPT_SCENARIO_W="${CAUSALOS_OPT_SCENARIO_W:-0.65}"
export CAUSALOS_OPT_SCENARIO_EMB="${CAUSALOS_OPT_SCENARIO_EMB:-1}"
export CAUSALOS_OPT_OPS_ALIGN="${CAUSALOS_OPT_OPS_ALIGN:-1}"
export CAUSALOS_OPT_OPS_W="${CAUSALOS_OPT_OPS_W:-0.70}"
export CAUSALOS_OPT_MIN_MARGIN="${CAUSALOS_OPT_MIN_MARGIN:-0.03}"

# Fact-mode disabled by default
export CAUSALOS_ENABLE_FACT_MODE="${CAUSALOS_ENABLE_FACT_MODE:-0}"

# Trace / Debug toggles
export CAUSALOS_SHOW_TRACE="$SHOW_TRACE"
export CAUSALOS_TRACE_FRAMES="${CAUSALOS_TRACE_FRAMES:-1}"


export CAUSALOS_GENERIC_PENALTY=1
export CAUSALOS_GENERIC_LAMBDA=0.8
export CAUSALOS_ENTAIL_YES=Yes
export CAUSALOS_ENTAIL_NO=No


export CAUSALOS_QB_SCORE_THR=0.0
export CAUSALOS_QB_INFER_MAX=5
export CAUSALOS_IDS_THR=0.55


export CAUSALOS_OPT_SCORER=likely_yesno
export CAUSALOS_GENERIC_PENALTY=1
export CAUSALOS_GENERIC_LAMBDA=0.8

# relevance penalty（r2で追加）
export CAUSALOS_LIKELY_REL=1
export CAUSALOS_LIKELY_REL_W=0.80
export CAUSALOS_LIKELY_REL_FLOOR=0.15

# QueryB（汎用勝ちで回す）
export CAUSALOS_ENABLE_QUERY_B=1
export CAUSALOS_QUERY_B_BUDGET=1
export CAUSALOS_QB_MARGIN_THR=0.02
export CAUSALOS_QB_REL_THR=0.25
export CAUSALOS_QB_BETA_MIN=0.25


if [[ "$DEBUG" == "1" ]]; then
  export CAUSALOS_DEBUG_FRAME_RAW=1
  export CAUSALOS_DEBUG_GROUND_TRIES=1
  export CAUSALOS_DEBUG_OPTION=1
else
  export CAUSALOS_DEBUG_FRAME_RAW=0
  export CAUSALOS_DEBUG_GROUND_TRIES=0
  export CAUSALOS_DEBUG_OPTION=0
fi

# =========================================
# Demo input blocks (SAFE with set -e)
# =========================================
DEMO_INPUT="$(cat <<'EOF'
A family starts a trip. What would have happened if the family had ended the trip?
A: The journey would have come to an end.
B: The journey would have begun.
C: The family would have used their car.
D: That is not possible.

A woman sees a fire. What would have happened if the woman had touched the fire?
A: She would have not been burned.
B: Everything would have been fine.
C: She would have been burned.
D: She would have seen fire.
EOF
)"

# =========================================
# Runner
# =========================================
{
  echo "=== run_demo.sh (${RUN_ID}) ==="
  echo "[Info] Model: $CAUSALOS_MODEL"
  echo "[Info] Logs: OUT=$OUT_LOG  ERR=$ERR_LOG"
  echo "[Info] Trace: SHOW_TRACE=$SHOW_TRACE"
  echo ""
} | tee "$SUMMARY_LOG"

# Sanity: required files
if [[ ! -f "CausalChatAgent.py" ]]; then
  echo "[Error] CausalChatAgent.py not found in current directory: $(pwd)" | tee -a "$SUMMARY_LOG"
  exit 2
fi

# Run and capture
if [[ "$FULL_CONSOLE" == "1" ]]; then
  python3 CausalChatAgent.py <<<"$DEMO_INPUT" 2> >(tee "$ERR_LOG" >&2) | tee "$OUT_LOG"
else
  python3 CausalChatAgent.py <<<"$DEMO_INPUT" >"$OUT_LOG" 2>"$ERR_LOG" || true

  {
    echo "---- SUMMARY (extracted) ----"
    grep -E "^\【反事実推論|^確信度:|^Grounding:|^FrameQuality:|^PriorMask:|^Enforce:|^Score:|^OPTS:|^\【選択肢との整合\】" "$OUT_LOG"
    echo ""
    echo "---- TOP2 (if present) ----"
    grep -E "^\- 1位 |^\- 2位 " "$OUT_LOG" || true
    echo ""
    echo "[Info] Full stdout saved: $OUT_LOG"
    echo "[Info] Full stderr saved: $ERR_LOG"
    echo "[Tip] tail -n 120 $OUT_LOG"
    echo "[Tip] tail -n 120 $ERR_LOG"
  } | tee -a "$SUMMARY_LOG"
fi

echo "=== done ===" | tee -a "$SUMMARY_LOG"


