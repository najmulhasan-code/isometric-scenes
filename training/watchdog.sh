#!/usr/bin/env bash
# Hard spending limit. launch.sh starts it first, as soon as the pod accepts SSH,
# with setsid nohup outside tmux, so killing tmux does not stop it.
#
#   every minute    checks that run_all.sh is alive (its pid in $RUN_ALL_PID_FILE,
#                   with run_all.sh in /proc/PID/cmdline). If it is gone before
#                   finishing (no KEEP_POD, UPLOAD_FAILED, TERMINATE_FAILED or
#                   FINISHED file), or never started within 30 min, terminates
#                   the pod after a 10-min grace. TERMINATE_FAILED (finish ran,
#                   terminate failed) means terminate now, retrying.
#   deadline - 3 h  writes $WATCHDOG_STOP_FILE and sends SIGTERM to run_all.sh,
#                   which skips its remaining work, uploads what exists and
#                   terminates the pod (run_all.sh, on_signal);
#   deadline        terminates the pod, retrying until the call succeeds.
#
# The deadline is MAX_POD_HOURS (default 16) after the watchdog first started.
# It ignores KEEP_POD at the deadline; to lift the limit, kill this process
# (pkill -f watchdog.sh).

set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WIND_DOWN_HOURS=3
DEAD_GRACE=600          # seconds between finding run_all.sh gone and terminating
START_GRACE=1800        # seconds run_all.sh may take to start

mkdir -p "$LOGS"
exec >> "$LOGS/watchdog.log" 2>&1
trap '' HUP PIPE

load_job_env
MAX_POD_HOURS="${MAX_POD_HOURS:-16}"
load_secrets || log "secrets could not be loaded; will retry before terminating"

# The start time survives a restart of this script.
START_FILE="$JOB_DIR/watchdog_start"
[[ -s "$START_FILE" ]] || date +%s > "$START_FILE"
started="$(cat "$START_FILE")"
deadline=$(( started + ${MAX_POD_HOURS%.*} * 3600 ))
wind_down=$(( deadline - WIND_DOWN_HOURS * 3600 ))
log "watchdog: pod ${POD_NAME:-unknown}: wind-down at $(date -u -d "@$wind_down" +%Y-%m-%dT%H:%M:%SZ)," \
    "terminate at $(date -u -d "@$deadline" +%Y-%m-%dT%H:%M:%SZ)"

# run_all.sh is alive: its recorded pid exists and runs run_all.sh.
run_all_alive() {
    local pid
    [[ -s "$RUN_ALL_PID_FILE" ]] || return 1
    pid="$(cat "$RUN_ALL_PID_FILE")"
    [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "run_all.sh"
}

# Terminate, retrying until the call succeeds; never returns.
terminate_now() {
    log "watchdog: terminating the pod ($1)"
    while true; do
        [[ -n "${PRIME_API_KEY:-}" ]] || load_secrets
        ensure_prime_cli
        terminate_self && exit 0
        sleep 300
    done
}

# One check of run_all.sh, every minute until the deadline.
dead_since=""
check_run_all() {
    if [[ -e "$JOB_DIR/TERMINATE_FAILED" ]]; then
        terminate_now "run_all.sh finished but its terminate call failed"
    fi
    if run_all_alive || [[ -e "$JOB_DIR/KEEP_POD" || -e "$JOB_DIR/UPLOAD_FAILED" || -e "$JOB_DIR/FINISHED" ]]; then
        dead_since=""
        return
    fi
    if [[ ! -s "$RUN_ALL_PID_FILE" ]] && (( $(date +%s) < started + START_GRACE )); then
        return
    fi
    if [[ -z "$dead_since" ]]; then
        dead_since="$(date +%s)"
        log "watchdog: run_all.sh is not running and did not finish; terminating in $DEAD_GRACE s unless it returns"
    elif (( $(date +%s) - dead_since >= DEAD_GRACE )); then
        terminate_now "run_all.sh stopped without finishing"
    fi
}

until (( $(date +%s) >= wind_down )); do
    check_run_all
    sleep 60
done
if run_all_alive; then
    log "watchdog: $WIND_DOWN_HOURS h to the deadline; asking run_all.sh to upload and terminate"
    printf 'wind-down %d h before the %s h pod limit\n%d\n' "$WIND_DOWN_HOURS" "$MAX_POD_HOURS" "$deadline" \
        > "$WATCHDOG_STOP_FILE"
    kill -TERM "$(cat "$RUN_ALL_PID_FILE")"
fi

until (( $(date +%s) >= deadline )); do
    check_run_all
    sleep 60
done
terminate_now "deadline reached"
