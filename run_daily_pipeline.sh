#!/usr/bin/env bash
# =============================================================================
# run_daily_pipeline.sh — NSE Volume Breakout: daily data pipeline
# =============================================================================
# Chain: fetch_market_cap.py → data_loading.py → prepare_data.py
# Triggered by launchd at BOTH 04:31 ET and 05:31 ET:
#
#   04:31 ET = 09:31 UTC = 15:01 IST  during EST (Nov  1 – Mar  8)
#   05:31 ET = 09:31 UTC = 15:01 IST  during EDT (Mar  8 – Nov  1)
#
# The IST time-window check (Step 4 below) silently skips whichever firing
# lands outside 14:55–15:35 IST, so only one run per day actually executes.
# This handles US Daylight Saving Time automatically — no plist edits needed.
#
# Logs: logs/pipeline_log_YYYY-MM-DD.log  (one file per IST trading day)
# =============================================================================

set -uo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/Users/rashilshah/Desktop/Volume "
PYTHON="/usr/local/bin/python3"
LOG_DIR="${PROJECT_DIR}/logs"

# ── Sanity-check Python ───────────────────────────────────────────────────────
if [[ ! -x "$PYTHON" ]]; then
    echo "FATAL: Python not found at $PYTHON — pipeline aborted." >&2
    exit 1
fi

# ── Logging helpers ───────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

_ist() {
    # Returns the current datetime in IST formatted as $1 (strftime string).
    "$PYTHON" -c "
from datetime import datetime, timezone, timedelta
ist = timezone(timedelta(hours=5, minutes=30))
print(datetime.now(ist).strftime('$1'))
"
}

TODAY_IST=$(_ist '%Y-%m-%d')
LOG_FILE="${LOG_DIR}/pipeline_log_${TODAY_IST}.log"

log() {
    # Writes timestamped message to the dated log AND stdout (captured by launchd).
    local ts
    ts=$(_ist '%Y-%m-%d %H:%M:%S IST')
    printf '[%s] %s\n' "$ts" "$*" | tee -a "$LOG_FILE"
}

log "============================================================"
log "Pipeline triggered.  Local clock: $(date '+%Y-%m-%d %H:%M:%S %Z')"

# ── Step 1: Weekday check ─────────────────────────────────────────────────────
DOW=$(_ist '%A')   # Monday … Sunday in IST
if [[ "${NSE_PIPELINE_TEST:-}" != "1" ]]; then
    if [[ "$DOW" == "Saturday" || "$DOW" == "Sunday" ]]; then
        log "SKIP: Weekend ($DOW in IST). NSE closed."
        exit 0
    fi
fi

# ── Step 2: NSE trading-holiday check ────────────────────────────────────────
#
# Source: Official NSE holiday circular published each January.
# URL: https://www.nseindia.com/resources/exchange-communication-holidays
#
# HOW TO UPDATE EACH YEAR:
#   1. Download the new NSE holiday circular in January.
#   2. Add a new NSE_HOLIDAYS_YYYY array below (weekday holidays only —
#      weekends are already handled by Step 1 above).
#   3. Append the new array to the NSE_HOLIDAYS=( … ) line at the bottom.
#   4. Re-load the launchd job:
#        launchctl unload  ~/Library/LaunchAgents/com.rashilshah.nse-pipeline.plist
#        launchctl load    ~/Library/LaunchAgents/com.rashilshah.nse-pipeline.plist
#
# Weekends (Sat/Sun) are intentionally OMITTED from these lists because
# Step 1 already skips them before we get here.

NSE_HOLIDAYS_2026=(
    "2026-01-26"   # Republic Day                   (Mon)
    "2026-03-03"   # Holi                           (Tue)
    "2026-03-26"   # Ram Navami                     (Thu)
    "2026-03-31"   # Mahavir Jayanti                (Tue)
    "2026-04-03"   # Good Friday                    (Fri)
    "2026-05-01"   # Maharashtra Day / Labour Day   (Fri)
    "2026-06-26"   # Eid-ul-Adha (Id-Ul-Zuha)       (Fri)

    "2026-09-14"   # Ganesh Chaturthi               (Mon)
    "2026-10-02"   # Gandhi Jayanti                 (Fri)
    "2026-10-20"   # Dussehra                       (Tue)
    "2026-11-24"   # Guru Nanak Jayanti             (Tue)
    "2026-12-25"   # Christmas                      (Fri)
)

# Add future years below, then include them in NSE_HOLIDAYS:
# NSE_HOLIDAYS_2027=(
#     "2027-01-26"   # Republic Day
#     ...
# )

NSE_HOLIDAYS=( "${NSE_HOLIDAYS_2026[@]}" )
# When 2027 is ready: NSE_HOLIDAYS=( "${NSE_HOLIDAYS_2026[@]}" "${NSE_HOLIDAYS_2027[@]}" )

if [[ "${NSE_PIPELINE_TEST:-}" != "1" ]]; then
    for holiday in "${NSE_HOLIDAYS[@]}"; do
        if [[ "$holiday" == "$TODAY_IST" ]]; then
            log "SKIP: NSE holiday today (${TODAY_IST}). Market closed."
            exit 0
        fi
    done
fi

# ── Step 3: IST time-window guard ─────────────────────────────────────────────
# launchd fires at 04:31 ET AND 05:31 ET every weekday.
# Only one of them lands in the 14:55–15:35 IST window.
# The other fires roughly 60 min earlier (≈14:01 IST) and is ignored here.
#
# TEST MODE: set NSE_PIPELINE_TEST=1 to bypass all guards and run immediately.
# Usage:  NSE_PIPELINE_TEST=1 bash run_daily_pipeline.sh
# DO NOT set this in production — it removes all safety checks.

IST_HHMM=$(_ist '%H%M')
IST_INT=$(( 10#$IST_HHMM ))   # strip leading zeros for integer arithmetic

if [[ "${NSE_PIPELINE_TEST:-}" == "1" ]]; then
    log "TEST MODE ACTIVE: bypassing IST time-window and day-of-week checks."
    log "IST time is ${IST_HHMM} — running pipeline unconditionally."
elif (( IST_INT < 1455 || IST_INT > 1535 )); then
    log "SKIP: IST time ${IST_HHMM} is outside the 14:55–15:35 run window."
    log "      (This is the off-DST launchd trigger firing — safe to ignore.)"
    exit 0
else
    log "IST time: ${IST_HHMM} — within window. Pipeline starting."
fi
log "============================================================"

# Track pipeline outcome so notify.py always runs, even on failure.
PIPELINE_STATUS="failed"
FAILED_STEP=""
MCAP_STATUS="fresh"
PIPELINE_START_TS=$(date +%s)

# ── Step 4: fetch_market_cap.py ───────────────────────────────────────────────
# Exit 0 = fresh data; 2 = stale fallback (proceed with warning); 1 = no file at all.
# Pipeline always continues — mcap failure never blocks data_loading.py.
log "--- STEP 1/3: fetch_market_cap.py ---"
cd "$PROJECT_DIR"

"$PYTHON" fetch_market_cap.py >> "$LOG_FILE" 2>&1
MCAP_EXIT=$?

if [[ $MCAP_EXIT -eq 2 ]]; then
    MCAP_STATUS="stale"
    log "STEP 1/3: Market cap fetch FAILED — using stale fallback. Continuing."
elif [[ $MCAP_EXIT -eq 1 ]]; then
    MCAP_STATUS="failed"
    log "STEP 1/3: Market cap fetch FAILED — no fallback. prepare_data.py will use snapshots."
else
    log "--- STEP 1/3 COMPLETE: fetch_market_cap.py succeeded ---"
fi

# ── Step 5: data_loading.py ───────────────────────────────────────────────────
log "--- STEP 2/3: data_loading.py ---"

"$PYTHON" data_loading.py >> "$LOG_FILE" 2>&1
DL_EXIT=$?

if [[ $DL_EXIT -ne 0 ]]; then
    log "STEP 2/3 FAILED  (exit code: ${DL_EXIT})"
    log "  prepare_data.py will NOT run — data fetch was incomplete."
    log "  To retry manually:"
    log "    cd '${PROJECT_DIR}'"
    log "    python3 fetch_market_cap.py && python3 data_loading.py && python3 prepare_data.py"
    FAILED_STEP="data_loading.py"
else
    log "--- STEP 2/3 COMPLETE: data_loading.py succeeded ---"

    # ── Step 6: prepare_data.py ───────────────────────────────────────────────
    log "--- STEP 3/3: prepare_data.py ---"

    "$PYTHON" prepare_data.py >> "$LOG_FILE" 2>&1
    PD_EXIT=$?

    if [[ $PD_EXIT -ne 0 ]]; then
        log "STEP 3/3 FAILED  (exit code: ${PD_EXIT})"
        log "  To retry manually (data fetch was OK, only analysis failed):"
        log "    cd '${PROJECT_DIR}'"
        log "    python3 prepare_data.py"
        FAILED_STEP="prepare_data.py"
    else
        log "--- STEP 3/3 COMPLETE: prepare_data.py succeeded ---"
        PIPELINE_STATUS="success"
        log "============================================================"
        log "PIPELINE COMPLETE."
        log "  Trade list: ${PROJECT_DIR}/results/trade_list_${TODAY_IST}.csv"
        log "  Full log  : ${LOG_FILE}"
        log "============================================================"
    fi
fi

# ── Step 7: Email notification (always runs, even after failure) ──────────────
log "--- Sending notification email ---"
NOTIFY_EXIT=0
"$PYTHON" "${PROJECT_DIR}/notify.py" \
    --log          "$LOG_FILE"           \
    --date         "$TODAY_IST"          \
    --status       "$PIPELINE_STATUS"    \
    --failed-step  "$FAILED_STEP"        \
    --mcap-status  "$MCAP_STATUS"        \
    --start-ts     "$PIPELINE_START_TS"  \
    >> "$LOG_FILE" 2>&1 || NOTIFY_EXIT=$?

if [[ $NOTIFY_EXIT -ne 0 ]]; then
    log "WARNING: Notification email failed to send (notify.py exit ${NOTIFY_EXIT})."
    log "  Check log above for SMTP error details."
    log "  Make sure SENDER_EMAIL / SENDER_APP_PASSWORD are filled in notify.py."
else
    log "Notification email sent."
fi

# Exit with pipeline status so launchd records success vs failure correctly.
if [[ "$PIPELINE_STATUS" == "failed" ]]; then
    exit 1
fi
