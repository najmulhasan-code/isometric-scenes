#!/usr/bin/env bash
# The whole training job, unattended, in tmux on the pod (launch.sh starts it,
# next to watchdog.sh):
#
#   0. pod setup (pod_setup.sh)
#   a. smoke test of every stage: a 2-step run, guard checks, export, a short
#      final eval on both GPUs, rescoring with the evaluation judge
#   b. full run, watched by guard.py
#   c. best checkpoint on `val` (select_checkpoint.py), its LoRA adapter copied for upload
#   d. final evaluation of the base and trained models, in parallel on the two GPUs
#   e. upload to the private Hugging Face repo in $HF_REPO
#   f. the pod terminates itself
#
# Training ends at max_steps or at the wall-clock cap; both continue to c. A stop
# for errors or a stall (guard rules error_rate, stalled) also continues to c if
# a validated checkpoint exists. Any other failure or stop skips to e. So does
# SIGTERM from watchdog.sh, 3 h before its deadline. e and f run exactly once on
# every way out (finish, from the end of main, the signal handler or the EXIT
# trap). If the upload fails after every retry the pod stays up, until
# watchdog.sh terminates it at its deadline, and $JOB_DIR/UPLOAD_FAILED says
# why. $JOB_DIR/KEEP_POD skips step f.
#
# Everything is logged to $JOB_DIR/outputs/logs/run_all.log.

set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUNS="$OUT/runs"
EVAL_DIR="$OUT/final_eval"
EXPORT_DIR="$OUT/model"
GROUPS_FILE="$OUT/process_groups"   # every process group this script starts

BASE_MODEL="Qwen/Qwen3.5-4B"
SEEDS=(0 1 2 3)
MAX_MODEL_LEN=10240                  # as in rl.toml
EVAL_TIMEOUT="${EVAL_TIMEOUT:-4h}"   # per eval or rescoring run: one model, one seed
SERVER_START_TIMEOUT=1800            # seconds for vLLM to load and answer
GUARD_GRACE=90                       # seconds the guard may take to finish after the run
GUARD_RESTARTS=3                     # restarts of a guard that died during the full run
UPLOAD_ATTEMPTS=5
EVAL_JUDGE="google/gemini-3.1-pro-preview"
RESULTS_JUDGE_MAX_TOKENS=8192        # reasoning tokens count here; the training judge keeps 256
LANE_CONCURRENCY=128                 # rescoring per model; the two lanes keep 256 sandboxes in flight
LORA_RANK=32                         # as [trainer.model.lora] in rl.toml
GPU_MEMORY_UTILIZATION=0.92          # as [inference.vllm] in rl.toml
MAX_NUM_SEQS=256
FULL_STEPS=150                       # as max_steps in rl.toml
# Smoke eval: 10 prompts per split, so 20 rows per lane for the evaluation judge. With the
# 5% tolerance, 1 failure in 20 passes and 2 fail.
SMOKE_EVAL_PROMPTS=10
SMOKE_MAX_ERROR_RATE=0.05

mkdir -p "$LOGS" "$RUNS" "$EVAL_DIR"
# The logger ignores HUP, INT and TERM, so it outlives a tmux kill and still
# records finish. This script ignores HUP and PIPE for the same reason.
exec > >(trap '' HUP INT TERM; exec tee -a "$LOGS/run_all.log") 2>&1
trap '' PIPE HUP

PY="$PRIME_RL_DIR/.venv/bin/python"
HF_PY="$TOOLS_DIR/bin/python"
STATUS="RUNNING"
REASON=""
TRAINING_END=""
LANE_PIDS=""
FINISH_STARTED=0
IN_SIGNAL=0
GUARD_PID=""
WIND_DOWN_EXPORT_SECONDS=7200        # export on wind-down only with at least 2 h left
UPLOAD_RESERVE_SECONDS=2700          # the export must leave 45 min before the deadline for the upload
WARNINGS_FILE="$OUT/WARNINGS"        # problems that do not stop the job, shown in STATUS

set_status() {
    STATUS="$1"
    REASON="$2"
    printf '%s\n%s\n' "$STATUS" "$REASON" > "$OUT/STATUS"
    log "status: $STATUS${REASON:+ - $REASON}"
}

# Process groups -------------------------------------------------------------

# Run a command from the prime-rl checkout in a new process group, in the
# background. The group id (= its pid) is recorded in $GROUPS_FILE and left in
# STARTED_PGID. Called directly, never in $(...), so the caller can wait for it.
STARTED_PGID=""
start_group() {
    (cd "$PRIME_RL_DIR" && exec setsid "$@") &
    STARTED_PGID=$!
    echo "$STARTED_PGID" >> "$GROUPS_FILE"
}

# Run a command in its own process group and wait for it; returns its status.
run_group() {
    start_group "$@"
    wait "$STARTED_PGID"
}

stop_group() {
    local pgid="$1"
    kill -TERM -- "-$pgid" 2>/dev/null || return 0
    for _ in $(seq 1 30); do
        kill -0 -- "-$pgid" 2>/dev/null || return 0
        sleep 2
    done
    kill -KILL -- "-$pgid" 2>/dev/null
    return 0
}

# Stop every process group this script has started.
stop_all_groups() {
    local pgid
    [[ -f "$GROUPS_FILE" ]] || return 0
    while read -r pgid; do
        [[ -n "$pgid" ]] && stop_group "$pgid"
    done < "$GROUPS_FILE"
    : > "$GROUPS_FILE"
}

# Wait until no process holds a GPU: no compute process listed, and under 1 GiB
# used on every GPU (inside a container's PID namespace the process list can be
# empty while memory is still held).
GPU_IDLE_MIB=1024
gpus_busy() {
    [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] && return 0
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | awk -v limit="$GPU_IDLE_MIB" '$1 + 0 >= limit { busy = 1 } END { exit busy ? 0 : 1 }'
}

gpus_idle() {
    local waited=0
    while gpus_busy; do
        if (( waited >= 120 )); then
            log "a GPU is still in use:"
            nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
            nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
}

# Stages shared by the smoke test and the real run ----------------------------

# Export step directory $1 to bf16 safetensors in $2, on GPU 0.
# The trained model is the base model plus a LoRA adapter in PEFT format
# (adapter_model.safetensors, adapter_config.json), copied by guard.py from the
# trainer's weight broadcast of that step. convert_dcp_to_bf16.py does not export
# LoRA checkpoints (prime-rl docs/training.md:409 at commit 35fcb52) and prime-rl has no merge tool, so the
# adapter itself is the exported model: copy adapter dir $1 to $2.
export_adapter() {
    local adapter="$1" out_dir="$2"
    rm -rf "$out_dir" && mkdir -p "$out_dir" \
        && cp "$adapter/adapter_model.safetensors" "$adapter/adapter_config.json" "$out_dir/"
}

# Both lanes serve the base model's own tokenizer and chat template; the adapter
# must belong to that base model and carry no tokenizer of its own, so the two
# lanes render every prompt identically.
adapter_matches_base() {
    "$PY" - "$BASE_MODEL" "$1" <<'EOF'
import json
import sys
from pathlib import Path

base, adapter = sys.argv[1], Path(sys.argv[2])
config = json.loads((adapter / "adapter_config.json").read_text())
# The trainer records the model's _name_or_path (trainer/lora.py:211-215): the hub
# id, or a local snapshot path of it in the Hugging Face cache.
name = str(config.get("base_model_name_or_path"))
if name != base and "models--" + base.replace("/", "--") not in name:
    sys.exit(f"the adapter was trained on {name!r}, not {base!r}")
extra = sorted(p.name for p in adapter.iterdir() if p.name.startswith(("tokenizer", "chat_template", "special_tokens")))
if extra:
    sys.exit(f"the adapter carries tokenizer files: {extra}")
print(f"adapter for {base}, rank {config.get('r')}, modules {config.get('target_modules')}")
EOF
}

# Serve the base model on `gpu` at `port`, with LoRA enabled, and load `adapter`
# if given (empty: none). Both lanes start identically, so they differ only in
# the adapter. With LoRA enabled the base model is served as "<model>-base" and
# an adapter is loaded under the model name, as in training (configs/
# inference.py:698-699; inference/vllm/server.py:76-99, the /load_lora_adapter
# route, reached on the engine's port as the orchestrator does, rl.py:776-783).
# The server's process-group id is left in SERVER_PGID.
SERVER_PGID=""
start_server() {
    local adapter="$1" gpu="$2" port="$3" dir="$4" log_file="$5" pgid waited=0
    # Distinct engine and RPC ports, so two servers can share the pod
    # (configs/inference.py:437 backend_port 8100, :77 data_parallel_rpc_port 13345).
    # No reasoning parser, as on the training run's server (rl.toml).
    start_group env CUDA_VISIBLE_DEVICES="$gpu" uv run --no-sync inference \
        --vllm.model "$BASE_MODEL" --vllm.max-model-len "$MAX_MODEL_LEN" --vllm.reasoning-parser None \
        --vllm.enable-lora --vllm.max-lora-rank "$LORA_RANK" \
        --vllm.gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --vllm.max-num-seqs "$MAX_NUM_SEQS" \
        --server.port "$port" --backend-port "$((port + 100))" \
        --vllm.data-parallel-rpc-port "$((13345 + gpu))" --output-dir "$dir/inference" >>"$log_file" 2>&1
    pgid="$STARTED_PGID"
    until curl -sf --max-time 5 "http://localhost:$port/v1/models" >/dev/null; do
        if ! kill -0 -- "-$pgid" 2>/dev/null; then
            log "the inference server on GPU $gpu exited; see $log_file" >&2
            return 1
        fi
        if (( waited >= SERVER_START_TIMEOUT )); then
            log "the inference server on GPU $gpu did not answer within $SERVER_START_TIMEOUT s" >&2
            stop_group "$pgid"
            return 1
        fi
        sleep 10
        waited=$((waited + 10))
    done
    if [[ -n "$adapter" ]] && ! curl -sf --max-time 300 -X POST -H 'Content-Type: application/json' \
            -d "{\"lora_name\": \"$BASE_MODEL\", \"lora_path\": \"$adapter\"}" \
            "http://localhost:$((port + 100))/load_lora_adapter?weight_version=0" >>"$log_file"; then
        log "could not load the adapter $adapter on GPU $gpu; see $log_file" >&2
        stop_group "$pgid"
        return 1
    fi
    SERVER_PGID="$pgid"
}

# One model's evaluation: serve, answer every test and test_unseen prompt once per
# seed (scored online by the training judge), stop the server, check that thinking
# was off, then score the same stored answers with the evaluation judge.
# Usage: eval_lane TAG MODEL GPU PORT DIR LIMIT SEED...   (LIMIT 0 = all prompts)
# Writes the number of failed parts to DIR/FAILED.
eval_lane() {
    local tag="$1" adapter="$2" gpu="$3" port="$4" dir="$5" limit="$6" model
    shift 6
    # The name each lane's requests use: the plain base model, or the adapter.
    if [[ -n "$adapter" ]]; then model="$BASE_MODEL"; else model="$BASE_MODEL-base"; fi
    local seeds=("$@") failed=0 seed judge server rc thinking_args
    local limit_args=()
    (( limit > 0 )) && limit_args=(--select.limit "$limit")
    mkdir -p "$dir"
    if ! start_server "$adapter" "$gpu" "$port" "$dir" "$LOGS/inference-$(basename "$(dirname "$dir")")-$tag.log"; then
        echo $(( ${#seeds[@]} * 2 )) > "$dir/FAILED"   # per seed: the eval and one rescoring
        return 1
    fi
    server="$SERVER_PGID"
    log "$tag: serving $model${adapter:+ (adapter $adapter)} on GPU $gpu, port $port"
    for seed in "${seeds[@]}"; do
        log "$tag: eval, seed $seed"
        if ! run_group timeout "$EVAL_TIMEOUT" uv run --no-sync eval \
                @ "$TRAINING_DIR/final_eval.toml" @ "$EVAL_DIR/seed_$seed.toml" --model "$model" \
                --client.base-url "http://localhost:$port/v1" "${limit_args[@]}" \
                --run.name "seed$seed" --output-dir "$dir" --no-monitors.prime; then
            log "$tag: eval, seed $seed failed"
            failed=$((failed + 1))
        fi
    done
    stop_group "$server"
    log "$tag: server stopped"
    for seed in "${seeds[@]}"; do
        if [[ ! -d "$dir/seed$seed" ]]; then
            log "$tag: seed $seed has no answers; not rescored"
            failed=$((failed + 1))
            continue
        fi
        thinking_args=()
        if ! "$PY" "$TRAINING_DIR/guard.py" thinking --run-dir "$dir/seed$seed"; then
            # Scored anyway, so the results exist; flagged in the summaries and in STATUS.
            log "WARNING: $tag, seed $seed: too many completions have reasoning or a think tag (thinking on?); rescoring anyway and flagging it"
            touch "$dir/seed$seed/THINKING_ON"
            thinking_args=(--thinking-on)
        fi
        for judge in "$EVAL_JUDGE"; do
            log "$tag: rescoring seed $seed with $judge"
            run_group timeout "$EVAL_TIMEOUT" "$PY" "$TRAINING_DIR/rescore.py" \
                --run-dir "$dir/seed$seed" --judge "$judge" --max-tokens "$RESULTS_JUDGE_MAX_TOKENS" \
                --model-tag "$tag" --seed "$seed" --kind eval --concurrency "$LANE_CONCURRENCY" \
                --max-error-rate "$SMOKE_MAX_ERROR_RATE" "${thinking_args[@]}" \
                --out "$dir/seed$seed/rescore/${judge//\//_}.jsonl"
            rc=$?
            if (( rc != 0 )); then
                log "$tag: rescoring seed $seed with $judge exited with $rc"
                failed=$((failed + 1))
            fi
        done
    done
    echo "$failed" > "$dir/FAILED"
}

# Run the base and trained lanes at once, on GPU 0 and GPU 1. Usage:
#   eval_pair OUT_DIR ADAPTER_DIR LIMIT SEED...
eval_pair() {
    local out_dir="$1" trained="$2" limit="$3"
    shift 3
    local seed base_pid trained_pid failed=0 tag flag
    for seed in "$@"; do
        printf '# Written by run_all.sh: the request seed of one final-eval run.\n[sampling]\nseed = %d\n' \
            "$seed" > "$EVAL_DIR/seed_$seed.toml"
    done
    stop_all_groups
    gpus_idle || return 1
    eval_lane base "" 0 8000 "$out_dir/base" "$limit" "$@" \
        > "$LOGS/eval-$(basename "$out_dir")-base.log" 2>&1 &
    base_pid=$!
    eval_lane trained "$trained" 1 8001 "$out_dir/trained" "$limit" "$@" \
        > "$LOGS/eval-$(basename "$out_dir")-trained.log" 2>&1 &
    trained_pid=$!
    LANE_PIDS="$base_pid $trained_pid"
    wait "$base_pid"
    wait "$trained_pid"
    LANE_PIDS=""
    stop_all_groups
    for tag in base trained; do
        failed=$((failed + $(cat "$out_dir/$tag/FAILED" 2>/dev/null || echo 1)))
    done
    for flag in "$out_dir"/*/seed*/THINKING_ON; do
        [[ -e "$flag" ]] && echo "thinking on in ${flag#"$OUT"/}" >> "$WARNINGS_FILE"
    done
    log "eval in $out_dir: $failed failed parts"
    (( failed == 0 ))
}

# 0. Setup -------------------------------------------------------------------

setup() {
    log "0. pod setup"
    if ! bash "$TRAINING_DIR/pod_setup.sh" > "$LOGS/pod_setup.log" 2>&1; then
        tail -n 20 "$LOGS/pod_setup.log"
        set_status SETUP_FAILED "pod_setup.sh failed (see logs/pod_setup.log)"
        return 1
    fi
}

# a. Smoke test --------------------------------------------------------------

smoke() {
    log "a. smoke test"
    local smoke_dir="$RUNS/smoke" judge
    if ! run_group uv run --no-sync rl @ "$TRAINING_DIR/rl.toml" @ "$TRAINING_DIR/smoke.toml" \
            --output-dir "$RUNS" --run.name smoke; then
        set_status SMOKE_FAILED "the smoke run exited with an error"
        return 1
    fi
    stop_all_groups
    "$PY" "$TRAINING_DIR/guard.py" adapters --run-dir "$smoke_dir" --interval 1
    if ! "$PY" "$TRAINING_DIR/guard.py" smoke --run-dir "$smoke_dir" --steps 2 \
            --max-error-rate "$SMOKE_MAX_ERROR_RATE"; then
        set_status SMOKE_FAILED "smoke checks failed (see run_all.log)"
        return 1
    fi
    if ! export_adapter "$smoke_dir/adapters/step_2" "$smoke_dir/export"; then
        set_status SMOKE_FAILED "copying the smoke adapter failed"
        return 1
    fi
    if ! adapter_matches_base "$smoke_dir/export"; then
        set_status SMOKE_FAILED "the smoke adapter does not match the base model"
        return 1
    fi
    if ! eval_pair "$OUT/smoke_eval" "$smoke_dir/export" "$SMOKE_EVAL_PROMPTS" 0; then
        set_status SMOKE_FAILED "the smoke final eval or its rescoring failed (see logs/eval-smoke_eval-*.log)"
        return 1
    fi
    # The smoke checkpoints and export only tested the stages; free the disk.
    rm -rf "$smoke_dir/checkpoints" "$smoke_dir/broadcasts" "$smoke_dir/export"
}

# b. Full run under the guard ------------------------------------------------

# A process that exists and is not a zombie (an exited child not yet waited for).
process_running() {
    [[ -r "/proc/$1/stat" ]] && [[ "$(awk '{ print $3 }' "/proc/$1/stat" 2>/dev/null)" != Z ]]
}

# The guard of the full run, in its own process group. Every 10 steps it logs the
# projected hours left and writes them to outputs/projected_hours; a slow run
# still ends at MAX_TRAIN_HOURS, as normal. --started keeps the wall-clock cap of
# a restarted guard.
start_guard() {
    start_group "$PY" "$TRAINING_DIR/guard.py" watch --run-dir "$RUNS/full" --pgid "$1" --adapter-interval 15 \
        --total-steps "$FULL_STEPS" --projection-file "$OUT/projected_hours" --started "$2" \
        --max-hours "$MAX_TRAIN_HOURS" --stop-file "$OUT/GUARD_STOP"
    GUARD_PID="$STARTED_PGID"
}

train() {
    log "b. full run (wall-clock cap ${MAX_TRAIN_HOURS} h)"
    local rl_pgid rl_status waited=0 rule reason started copier
    rm -f "$OUT/GUARD_STOP"
    started="$(date +%s)"
    start_group uv run --no-sync rl @ "$TRAINING_DIR/rl.toml" --output-dir "$RUNS" --run.name full
    rl_pgid="$STARTED_PGID"
    start_guard "$rl_pgid" "$started"
    # A second, independent copier of the validation-step adapters, every 30 s.
    start_group "$PY" "$TRAINING_DIR/guard.py" adapters --run-dir "$RUNS/full" --interval 15 \
        --every 30 --pgid "$rl_pgid"
    copier="$STARTED_PGID"

    # Every minute: restart the guard if it died while the run goes on, at most
    # GUARD_RESTARTS times. After that training goes on without it; the adapter
    # copier, the wall-clock watchdog and the error checks after training remain.
    local restarts=0
    while process_running "$rl_pgid"; do
        if [[ -n "$GUARD_PID" ]] && ! process_running "$GUARD_PID" && [[ ! -s "$OUT/GUARD_STOP" ]]; then
            if (( restarts < GUARD_RESTARTS )); then
                restarts=$((restarts + 1))
                log "WARNING: the guard is not running; restart $restarts of $GUARD_RESTARTS"
                start_guard "$rl_pgid" "$started"
            else
                log "WARNING: the guard failed $((restarts + 1)) times; training goes on without it"
                echo "the guard failed $((restarts + 1)) times during the full run" > "$OUT/GUARD_FAILED"
                echo "the guard failed during the full run (no stop rules after that)" >> "$WARNINGS_FILE"
                GUARD_PID=""
            fi
        fi
        sleep 60 &
        wait $!
    done
    wait "$rl_pgid"
    rl_status=$?
    # Nothing of the run may outlive it; the guard gets a last look, then stops.
    stop_group "$rl_pgid"
    while [[ -n "$GUARD_PID" ]] && process_running "$GUARD_PID" && (( waited < GUARD_GRACE )); do
        sleep 5
        waited=$((waited + 5))
    done
    [[ -n "$GUARD_PID" ]] && stop_group "$GUARD_PID"
    stop_group "$copier"
    GUARD_PID=""
    stop_all_groups
    "$PY" "$TRAINING_DIR/guard.py" adapters --run-dir "$RUNS/full" --interval 15

    if [[ -s "$OUT/GUARD_STOP" ]]; then
        rule="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["rule"])' "$OUT/GUARD_STOP")"
        reason="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["reason"])' "$OUT/GUARD_STOP")"
        case "$rule" in
            time_cap)
                TRAINING_END="time cap: $reason"
                log "training ended at the wall-clock cap; continuing" ;;
            error_rate|stalled)
                TRAINING_END="stopped ($rule): $reason"
                log "training stopped ($rule); continuing with the validated checkpoints" ;;
            *)
                set_status GUARD_STOPPED "$rule: $reason"
                return 1 ;;
        esac
        echo "$TRAINING_END" > "$OUT/training_end"
        return 0
    fi
    if (( rl_status != 0 )); then
        set_status TRAIN_FAILED "the full run exited with status $rl_status"
        return 1
    fi
    TRAINING_END="max_steps"
    echo "$TRAINING_END" > "$OUT/training_end"
}

# c. Choose and export the checkpoint ----------------------------------------

select_and_export() {
    log "c. checkpoint selection on val"
    local adapter
    "$PY" "$TRAINING_DIR/guard.py" adapters --run-dir "$RUNS/full" --interval 15
    if ! adapter="$("$PY" "$TRAINING_DIR/select_checkpoint.py" --run-dir "$RUNS/full" --env val \
            --out "$OUT/selection.json")"; then
        set_status SELECT_FAILED "no validated checkpoint with an adapter to choose"
        return 1
    fi
    log "chose $adapter"
    if ! export_adapter "$adapter" "$EXPORT_DIR"; then
        set_status EXPORT_FAILED "copying the adapter $adapter failed"
        return 1
    fi
    if ! adapter_matches_base "$EXPORT_DIR"; then
        set_status EXPORT_FAILED "the chosen adapter does not match the base model"
        return 1
    fi
}

# d. Final evaluation --------------------------------------------------------

final_eval() {
    log "d. final evaluation"
    if ! eval_pair "$EVAL_DIR" "$EXPORT_DIR" 0 "${SEEDS[@]}"; then
        set_status EVAL_INCOMPLETE "some eval or rescoring parts failed (see logs/eval-final_eval-*.log)"
        return 1
    fi
}

# e. Upload ------------------------------------------------------------------

record_versions() {
    local revision_file="$HOME/.cache/huggingface/hub/models--${BASE_MODEL/\//--}/refs/main"
    mkdir -p "$OUT/configs"
    cp "$TRAINING_DIR"/*.toml "$TRAINING_DIR"/*.py "$TRAINING_DIR"/*.sh "$OUT/configs/" 2>/dev/null
    {
        echo "prime-rl $(git -C "$PRIME_RL_DIR" rev-parse HEAD 2>/dev/null)"
        git -C "$PRIME_RL_DIR" submodule status 2>/dev/null
        echo "$BASE_MODEL revision $(cat "$revision_file" 2>/dev/null || echo unknown)"
        if [[ "${RENDER_IMAGE:-}" == *@sha256:* ]]; then
            echo "render image digest ${RENDER_IMAGE##*@}"
        else
            echo "render image digest unknown (the reference carries no digest)"
        fi
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null
        echo
        uv pip list --python "$PY" 2>/dev/null
    } > "$OUT/versions.txt"
}

upload() {
    log "e. upload"
    local attempt
    [[ -x "$HF_PY" ]] || { log "no huggingface_hub environment; nothing can be uploaded"; return 1; }
    for attempt in $(seq 1 "$UPLOAD_ATTEMPTS"); do
        if "$HF_PY" - "$OUT" <<'EOF'
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

repo, folder = os.environ["HF_REPO"], Path(sys.argv[1])
api = HfApi()  # reads HF_TOKEN from the environment
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
if not api.repo_info(repo, repo_type="model").private:
    sys.exit("the Hugging Face repo exists and is public; refusing to upload")
# DCP checkpoints and weight broadcasts stay on the pod; the exported model is in model/.
# This job's own log is still being written, so it goes up as a copy afterwards.
api.upload_large_folder(
    repo_id=repo, folder_path=folder, repo_type="model", private=True,
    ignore_patterns=["**/checkpoints/**", "**/broadcasts/**", "**/batches/**", "**/wandb/**", "logs/run_all.log"],
    print_report=False,
)
with tempfile.TemporaryDirectory() as tmp:
    snapshot = Path(tmp) / "run_all.log"
    shutil.copyfile(folder / "logs" / "run_all.log", snapshot)
    api.upload_file(path_or_fileobj=str(snapshot), path_in_repo="logs/run_all.log",
                    repo_id=repo, repo_type="model")
EOF
        then
            log "upload done"
            rm -f "$JOB_DIR/UPLOAD_FAILED"
            return 0
        fi
        log "upload attempt $attempt of $UPLOAD_ATTEMPTS failed"
        (( attempt < UPLOAD_ATTEMPTS )) && sleep $((attempt * 120))
    done
    printf 'FAILED: upload to Hugging Face failed %d times (%s). The pod was not terminated;\nwatchdog.sh terminates it at its deadline. Outputs are in %s.\n' \
        "$UPLOAD_ATTEMPTS" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OUT" > "$JOB_DIR/UPLOAD_FAILED"
    log "FAILED: upload failed after $UPLOAD_ATTEMPTS attempts; the pod stays up until the watchdog deadline"
    return 1
}

# f. Terminate ---------------------------------------------------------------

finish() {
    (( FINISH_STARTED )) && return 0
    FINISH_STARTED=1
    stop_all_groups
    [[ "$STATUS" == RUNNING ]] && set_status DONE "${TRAINING_END:+training end: $TRAINING_END}"
    if [[ -s "$WARNINGS_FILE" ]]; then
        set_status "$STATUS" "${REASON:+$REASON; }WARNING: $(sort -u "$WARNINGS_FILE" | paste -sd ';' -)"
    fi
    # The tools may be missing if setup failed early; upload and terminate need them.
    ensure_tools || log "the upload and terminate tools could not be installed"
    record_versions
    # Keep the pod for the outputs only if there are outputs: a failed setup has none.
    if ! upload && [[ "$STATUS" != SETUP_FAILED ]]; then
        return 0
    fi
    if [[ -e "$JOB_DIR/KEEP_POD" ]]; then
        log "$JOB_DIR/KEEP_POD exists; not terminating"
        return 0
    fi
    log "f. terminate"
    if terminate_self; then
        touch "$JOB_DIR/FINISHED"
    else
        echo "FAILED: terminate failed; watchdog.sh retries it" > "$JOB_DIR/TERMINATE_FAILED"
    fi
}

# SIGTERM from watchdog.sh (wind-down before its deadline), or INT/TERM: skip the
# remaining work, upload, terminate. On a wind-down with at least 2 h left and no
# exported model yet, the best validated checkpoint is exported first, so the
# model is not lost. A second signal, or one during finish, is ignored.
on_signal() {
    local deadline status reason
    (( FINISH_STARTED || IN_SIGNAL )) && { log "signal while stopping; continuing"; return 0; }
    IN_SIGNAL=1
    if [[ -s "$WATCHDOG_STOP_FILE" ]]; then
        status=STOPPED_BY_WATCHDOG
        reason="$(head -n1 "$WATCHDOG_STOP_FILE"); remaining work skipped"
    else
        status=INTERRUPTED
        reason="run_all.sh received a signal; remaining work skipped"
    fi
    set_status "$status" "$reason"
    # shellcheck disable=SC2086
    [[ -n "$LANE_PIDS" ]] && kill $LANE_PIDS 2>/dev/null
    [[ -n "$GUARD_PID" ]] && kill "$GUARD_PID" 2>/dev/null
    stop_all_groups
    deadline="$(sed -n 2p "$WATCHDOG_STOP_FILE" 2>/dev/null)"
    if [[ "$status" == STOPPED_BY_WATCHDOG && "$deadline" =~ ^[0-9]+$ && ! -f "$EXPORT_DIR/adapter_config.json" \
          && -d "$RUNS/full/checkpoints" ]] && (( deadline - $(date +%s) >= WIND_DOWN_EXPORT_SECONDS )); then
        log "wind-down: exporting the best validated checkpoint before the upload"
        # No policy-span scan (it reads the whole trace stream).
        export SELECT_SKIP_SPANS=1
        if select_and_export; then
            reason="$reason; best validated checkpoint exported"
        else
            reason="$reason; export failed (${REASON})"
        fi
        set_status "$status" "$reason"
        if (( deadline - $(date +%s) < UPLOAD_RESERVE_SECONDS )); then
            log "wind-down: less than 45 min to the deadline; uploading now"
        fi
    fi
    finish
    exit 1
}

# Any other way out (an unexpected error, a stray exit): finish once.
on_exit() {
    (( FINISH_STARTED )) && return 0
    [[ "$STATUS" == RUNNING ]] && set_status EXITED_UNEXPECTEDLY "run_all.sh exited before finishing"
    finish
}

main() {
    log "job start"
    # First, so the watchdog's dead-run check covers every exit from here on.
    echo "$$" > "$RUN_ALL_PID_FILE"
    load_job_env
    MAX_TRAIN_HOURS="${MAX_TRAIN_HOURS:-10}"
    if ! load_secrets; then
        # Nothing can be uploaded. SECRETS_FAILED is not a keep-the-pod marker: the
        # watchdog finds run_all.sh gone and terminates the pod after its grace.
        echo "FAILED: secrets could not be loaded" > "$JOB_DIR/SECRETS_FAILED"
        log "FAILED: secrets could not be loaded; nothing was run"
        exit 1
    fi
    trap on_signal INT TERM
    trap on_exit EXIT
    : > "$GROUPS_FILE"
    rm -f "$WARNINGS_FILE"

    setup && smoke && train && select_and_export && final_eval
    finish
    log "job end: $STATUS${REASON:+ - $REASON}"
}

main "$@"
