#!/usr/bin/env bash
# Start the training job on a new Prime pod. Run from the repository root on a local
# machine:
#   bash training/launch.sh
#
# 1. Picks the cheapest on-demand 2-GPU offer of any type in GPU_TYPES, prints it and asks before
#    creating anything.
# 2. Creates the pod without prompts and waits until it accepts SSH. From here
#    on there is no prompt; on any exit before the job starts, the pod is
#    terminated (with retries) and the terminate command is printed.
# 3. Writes the run settings (with the pod id), copies environments/isometric_scenes,
#    training/ and the secrets file, starts watchdog.sh (setsid nohup, outside
#    tmux, so a tmux kill cannot stop it), and checks that the pod can find
#    itself with the Prime key from the secrets file.
# 4. Starts run_all.sh (which begins with pod_setup.sh) in tmux and prints how
#    to watch it. This machine can then disconnect.
#
# Needs: the Prime CLI, logged in, with an SSH key set (`prime config
# set-ssh-key-path`) whose public half is added on the Prime dashboard;
# ssh, scp and tar; Python 3 for reading JSON.
#
# Settings (environment variables): SECRETS_FILE, GPU_TYPES, DISK_GB, POD_IMAGE,
# POD_NAME, MAX_TRAIN_HOURS, MAX_POD_HOURS, VCPUS, MEMORY_GB.
#
# Image: POD_IMAGES are tried in order until the provider accepts one; providers
# support different images, and the availability list does not say which. The
# image's own CUDA and PyTorch are not used: uv installs prime-rl's pinned CUDA 13
# wheels (which also cover the Blackwell RTX PRO 6000) into their own environment,
# and pod_setup.sh checks that the host driver (>= 580) can run them. The
# prime_rl image is not used because it ships its own prime-rl, which need not
# be commit 35fcb52.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECRETS_FILE="${SECRETS_FILE:-$ROOT/training/secrets.env}"
# Any of these fits LoRA on the 4B model (80 GB or more per GPU); the cheapest
# 2-GPU offer among them is used.
GPU_TYPES="${GPU_TYPES:-H100_80GB H200_141GB RTX_PRO_6000B_96GB}"
GPU_COUNT=2
# The least disk to ask for: LoRA checkpoints, the base model, adapters and traces
# (pod_setup.sh requires 300 GB free). An offer with a fixed disk, or a larger
# default, gets its own size. Disk above the offer's default may cost extra.
DISK_GB="${DISK_GB:-400}"
POD_IMAGES="${POD_IMAGES:-ubuntu_22_cuda_12 cuda_12_6_pytorch_2_7}"
POD_IMAGE="${POD_IMAGES%% *}"   # shown in the offer summary; the one used is printed at create
POD_NAME="${POD_NAME:-isometric-rl-$(date -u +%Y%m%d-%H%M)}"
MAX_TRAIN_HOURS="${MAX_TRAIN_HOURS:-10}"
MAX_POD_HOURS="${MAX_POD_HOURS:-16}"   # hard limit: watchdog.sh terminates the pod then
READY_TIMEOUT=3600   # seconds to wait for the pod (setup time is not billed)
export PRIME_DISABLE_VERSION_CHECK=1

fail() { echo "launch: $*" >&2; exit 1; }

for tool in prime ssh scp tar; do
    command -v "$tool" >/dev/null || fail "$tool not found"
done
PYTHON=""
for candidate in python python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c "import json" 2>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
[[ -n "$PYTHON" ]] || fail "no working Python 3 found"
[[ -f "$SECRETS_FILE" ]] || fail "secrets file $SECRETS_FILE not found"
[[ "$POD_NAME" =~ ^[A-Za-z0-9-]*[A-Za-z][A-Za-z0-9-]*$ ]] || fail "POD_NAME may hold letters, digits and dashes only"

# One key of the Prime CLI config (~/.prime/config.json), without printing the rest.
prime_config() {
    "$PYTHON" -c '
import json, pathlib, sys
path = pathlib.Path.home() / ".prime" / "config.json"
config = json.loads(path.read_text()) if path.exists() else {}
print(config.get(sys.argv[1]) or "")
' "$1"
}

# --- 1. The offer -----------------------------------------------------------

echo "Looking for on-demand ${GPU_COUNT}-GPU offers of: ${GPU_TYPES}..."
# One query per GPU type, without --gpu-count (that filter has returned nothing
# while matching offers were listed); the count is checked here instead. Each
# matching offer becomes one tab-separated line, led by its price.
# The list sometimes comes back empty for a moment, so it is read up to 3 times.
offer=""
for listing in 1 2 3; do
    candidates=""
    for gpu in $GPU_TYPES; do
        listed="$(prime availability list --gpu-type "$gpu" -o json)" || continue
        candidates+="$(printf '%s' "$listed" | "$PYTHON" -c '
import json, math, sys
gpu, count = sys.argv[1], int(sys.argv[2])
for o in json.loads(sys.stdin.read().lstrip("\ufeff")).get("gpu_resources", []):
    if o.get("is_spot") or o.get("gpu_count") != count:
        continue
    if not (isinstance(o.get("price_value"), (int, float)) and math.isfinite(o["price_value"])):
        continue
    fields = [o["price_value"], o.get("id"), gpu] + [o.get(f) for f in
              ["provider", "location", "security", "socket", "vcpus", "memory_gb", "disk_gb", "stock_status"]]
    print("\t".join(str(f) for f in fields))
' "$gpu" "$GPU_COUNT")"$'\n'
    done
    offer="$(printf '%s' "$candidates" | { grep -v '^$' || true; } | sort -t$'\t' -k1,1g | head -n 1)"
    [[ -n "$offer" ]] && break
    sleep 10
done
[[ -n "$offer" ]] || fail "no on-demand ${GPU_COUNT}-GPU offer of ${GPU_TYPES} is available"
IFS=$'\t' read -r PRICE OFFER_ID GPU_TYPE PROVIDER LOCATION SECURITY SOCKET OFFER_VCPUS OFFER_MEMORY OFFER_DISK STOCK <<< "$offer"

VCPUS="${VCPUS:-$OFFER_VCPUS}"
MEMORY_GB="${MEMORY_GB:-$OFFER_MEMORY}"
[[ "$VCPUS" =~ ^[0-9]+$ ]] || fail "the offer gives no vCPU count ($OFFER_VCPUS); set VCPUS"
[[ "$MEMORY_GB" =~ ^[0-9]+$ ]] || fail "the offer gives no memory size ($OFFER_MEMORY); set MEMORY_GB"
# disk_gb is the offer's default size, with "+" when it can grow. A fixed disk
# is asked for at its own size; a growable one at no less than its default.
if [[ "$OFFER_DISK" =~ ^([0-9]+)(\+?)$ ]]; then
    offer_default="${BASH_REMATCH[1]}"
    if [[ -z "${BASH_REMATCH[2]}" ]]; then
        (( offer_default >= DISK_GB )) || fail "the offer has a fixed ${offer_default} GB disk, less than the ${DISK_GB} GB needed"
        DISK_GB="$offer_default"
    elif (( offer_default > DISK_GB )); then
        DISK_GB="$offer_default"
    fi
fi

cat <<EOF

Cheapest on-demand offer:
  offer id     $OFFER_ID
  GPUs         ${GPU_COUNT} x ${GPU_TYPE} ($SOCKET)
  provider     $PROVIDER, $LOCATION, $SECURITY, stock $STOCK
  price        \$$PRICE per hour for the pod (disk above the default size may cost extra)
  pod          name $POD_NAME, $VCPUS vCPUs, $MEMORY_GB GB RAM, $DISK_GB GB disk, image $POD_IMAGE
  training cap $MAX_TRAIN_HOURS h, then evaluation; the pod terminates itself at the end
  hard limit   upload and stop at $((MAX_POD_HOURS - 3)) h, pod terminated at $MAX_POD_HOURS h, whatever the job is doing

EOF
read -r -p "Create this pod and start the job? [y/N] " answer
[[ "$answer" =~ ^[Yy]$ ]] || { echo "Nothing created."; exit 0; }

# --- 2. Create and wait -------------------------------------------------------

# Prime CLI calls that only read state are retried, so a transient API error does
# not end the script (and set -e) while a pod is running.
CLI_ATTEMPTS=5

pod_id_by_name() {
    local attempt out
    for attempt in $(seq 1 "$CLI_ATTEMPTS"); do
        if out="$(prime pods list -o json)"; then
            printf '%s' "$out" | "$PYTHON" -c '
import json, sys
ids = [p["id"] for p in json.load(sys.stdin).get("pods", []) if p.get("name") == sys.argv[1]]
print(ids[0] if len(ids) == 1 else "")
' "$POD_NAME" && return 0
        fi
        sleep 10
    done
    return 1
}

pod_state() {
    local attempt out
    for attempt in $(seq 1 "$CLI_ATTEMPTS"); do
        if out="$(prime pods status "$1" -o json)"; then
            # "ssh" is "user@host -p port" once the pod is up; status is ACTIVE once installed.
            printf '%s' "$out" | "$PYTHON" -c '
import json, sys
s = json.load(sys.stdin)
print(s.get("status", ""), s.get("ssh", "N/A"), sep="\t")
' && return 0
        fi
        sleep 10
    done
    return 1
}

existing="$(pod_id_by_name)" || fail "could not list pods"
[[ -z "$existing" ]] || fail "a pod named $POD_NAME already exists"

POD_ID=""
JOB_STARTED=0
# From here on a pod may exist. If the job was not started, terminate the pod
# (with retries) and say so; print the command in case that fails too.
on_exit() {
    local attempt
    (( JOB_STARTED )) && return 0
    trap - EXIT
    echo
    echo "launch did not finish; terminating pod $POD_NAME so it does not keep costing money."
    # A pod just created may take a moment to be listed: look for about 2 min.
    for attempt in $(seq 1 8); do
        [[ -n "$POD_ID" ]] && break
        POD_ID="$(pod_id_by_name)" || POD_ID=""
        [[ -n "$POD_ID" ]] || sleep 15
    done
    if [[ -z "$POD_ID" ]]; then
        echo "No pod named $POD_NAME was found. Check with: prime pods list"
        return 0
    fi
    for attempt in $(seq 1 "$CLI_ATTEMPTS"); do
        if prime pods terminate "$POD_ID" -y; then
            echo "Terminate requested for pod $POD_ID."
            return 0
        fi
        sleep 15
    done
    echo "Terminating failed. Terminate it by hand: prime pods terminate $POD_ID -y"
}
trap on_exit EXIT

# Each image in POD_IMAGES is tried until the provider accepts one ("Provider X
# is not supported for image Y"). The availability list the CLI reads during
# create also sometimes comes back without an offer that is listed a moment
# before and after ("No valid GPU configuration found"), so other failures are
# retried. Nothing is retried once a pod with this name exists.
created=0
create_log="$(mktemp)"
for POD_IMAGE in $POD_IMAGES; do
    for attempt in $(seq 1 "$CLI_ATTEMPTS"); do
        echo "Creating the pod with image $POD_IMAGE (attempt $attempt of $CLI_ATTEMPTS)..."
        if prime pods create --id "$OFFER_ID" --name "$POD_NAME" --disk-size "$DISK_GB" \
            --vcpus "$VCPUS" --memory "$MEMORY_GB" --image "$POD_IMAGE" -y 2>&1 | tee "$create_log"; then
            created=1
            break 2
        fi
        sleep 15
        existing="$(pod_id_by_name)" || fail "could not list pods after a failed create"
        [[ -z "$existing" ]] || { POD_ID="$existing"; created=1; break 2; }
        if grep -qi "not supported for image" "$create_log"; then
            echo "The provider does not support image $POD_IMAGE; trying the next one."
            break
        fi
    done
done
rm -f "$create_log"
(( created )) || fail "the pod could not be created with any of: $POD_IMAGES"

echo "Waiting for the pod..."
STATUS=""
SSH_TARGET=""
waited=0
while (( waited < READY_TIMEOUT )); do
    [[ -n "$POD_ID" ]] || POD_ID="$(pod_id_by_name)" || POD_ID=""
    if [[ -n "$POD_ID" ]] && state="$(pod_state "$POD_ID")"; then
        IFS=$'\t' read -r STATUS SSH_TARGET <<< "$state"
        echo "  status $STATUS"
        [[ "$STATUS" == "ACTIVE" && -n "$SSH_TARGET" && "$SSH_TARGET" != "N/A" ]] && break
    fi
    sleep 20
    waited=$((waited + 20))
done
[[ "$STATUS" == "ACTIVE" && -n "$SSH_TARGET" && "$SSH_TARGET" != "N/A" ]] \
    || fail "the pod had no SSH access after $READY_TIMEOUT s"
echo "pod id $POD_ID"

SSH_TARGET="${SSH_TARGET%%,*}"            # first node if several
SSH_HOST="${SSH_TARGET%% -p *}"
SSH_PORT="22"
[[ "$SSH_TARGET" == *" -p "* ]] && SSH_PORT="${SSH_TARGET##* -p }"
KEY="$(prime_config ssh_key_path)"
[[ -n "$KEY" ]] || fail "no ssh_key_path in the Prime CLI config"
command -v cygpath >/dev/null && KEY="$(cygpath -u "$KEY")"
[[ -f "$KEY" ]] || fail "SSH key $KEY not found"
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
          -o ServerAliveInterval=30)
remote() { ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "$SSH_HOST" "$@"; }

waited=0
until remote true 2>/dev/null; do
    (( waited < 600 )) || fail "SSH to the pod does not answer"
    sleep 15
    waited=$((waited + 15))
done

# --- 3. Watchdog first, then copy ---------------------------------------------

echo "Starting the watchdog on the pod..."
remote "mkdir -p ~/job/src ~/job/outputs/logs"
# Run settings for the pod (not secrets, but account-specific, so not in the repo).
TEAM_ID="$(prime_config team_id)"
{
    echo "POD_NAME=$POD_NAME"
    echo "POD_ID=$POD_ID"
    echo "MAX_TRAIN_HOURS=$MAX_TRAIN_HOURS"
    echo "MAX_POD_HOURS=$MAX_POD_HOURS"
    [[ -n "$TEAM_ID" ]] && echo "PRIME_TEAM_ID=$TEAM_ID"
    true
} | remote "umask 077 && cat > ~/job/job.env"
tar -C "$ROOT" -czf - \
    --exclude='__pycache__' --exclude='.pytest_cache' \
    --exclude='environments/isometric_scenes/runs' --exclude='environments/isometric_scenes/dist' \
    --exclude='training/secrets.env' \
    environments/isometric_scenes training \
    | remote "tar -xzf - -C ~/job/src"
# Piped, and created under umask 077, so the file is never readable by others.
remote "umask 077 && cat > ~/job/secrets.env && chmod 600 ~/job/secrets.env" < "$SECRETS_FILE"
remote 'command -v tmux >/dev/null || { s=""; [ "$(id -u)" = 0 ] || s=sudo;
    $s apt-get update -y >/dev/null && DEBIAN_FRONTEND=noninteractive $s apt-get install -y tmux >/dev/null; }'
remote "setsid nohup bash ~/job/src/training/watchdog.sh > /dev/null 2>&1 < /dev/null &"
sleep 5
remote "pgrep -f 'training/watchdog.sh' > /dev/null" || fail "the watchdog is not running on the pod"

# The pod must be able to find itself with the Prime key in secrets.env, or it
# could not terminate itself; checked here, while someone is at the terminal.
echo "Checking that the pod can find itself through the Prime CLI..."
remote "bash -c '. ~/job/src/training/common.sh && load_secrets && load_job_env &&
    ensure_prime_cli >/dev/null 2>&1 && [ \"\$(pod_id_by_name)\" = \"\$POD_ID\" ]'" \
    || fail "the pod could not find itself with the Prime key in the secrets file"

# --- 4. Start -----------------------------------------------------------------

# The job begins with pod_setup.sh. It runs on the pod; this machine can disconnect.
remote "tmux new-session -d -s job -n run 'bash ~/job/src/training/run_all.sh'"
JOB_STARTED=1

cat <<EOF

The job is running in tmux on pod $POD_NAME (id $POD_ID).
  attach:       ssh -i "$KEY" -p $SSH_PORT $SSH_HOST -t tmux attach -t job   (detach: Ctrl-b d)
  follow log:   ssh -i "$KEY" -p $SSH_PORT $SSH_HOST tail -f job/outputs/logs/run_all.log
  watchdog log: ssh -i "$KEY" -p $SSH_PORT $SSH_HOST tail -f job/outputs/logs/watchdog.log
  status:       ssh -i "$KEY" -p $SSH_PORT $SSH_HOST cat job/outputs/STATUS
  keep the pod: ssh -i "$KEY" -p $SSH_PORT $SSH_HOST touch job/KEEP_POD
  terminate:    prime pods terminate $POD_ID -y
The pod terminates itself after the upload. The watchdog asks the job to upload
and stop $((MAX_POD_HOURS - 3)) h after the pod started, and terminates it at $MAX_POD_HOURS h.
If the upload fails, job/UPLOAD_FAILED says why.
EOF
