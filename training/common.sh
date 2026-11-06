# Shared by pod_setup.sh, run_all.sh and watchdog.sh. Source it; do not run it.

JOB_DIR="${JOB_DIR:-$HOME/job}"
SRC_DIR="$JOB_DIR/src"
TRAINING_DIR="$SRC_DIR/training"
PRIME_RL_DIR="$JOB_DIR/prime-rl"
TOOLS_DIR="$JOB_DIR/tools"
SECRETS_FILE="${SECRETS_FILE:-$JOB_DIR/secrets.env}"
JOB_ENV="$JOB_DIR/job.env"   # POD_NAME, POD_ID, MAX_TRAIN_HOURS, MAX_POD_HOURS, PRIME_TEAM_ID; written by launch.sh
OUT="$JOB_DIR/outputs"
LOGS="$OUT/logs"
RUN_ALL_PID_FILE="$JOB_DIR/run_all.pid"
WATCHDOG_STOP_FILE="$JOB_DIR/WATCHDOG_STOP"   # written by watchdog.sh before it signals run_all.sh

PRIME_CLI_VERSION="0.18.0"   # the Prime CLI version these scripts were written against
HF_HUB_VERSION="1.33.0"
TERMINATE_ATTEMPTS=5
TERMINATE_DELAY=60

export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PRIME_DISABLE_VERSION_CHECK=1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# Export KEY=value lines of a file readable only by its owner. The file is parsed,
# not sourced: blank lines and # comments are skipped, a trailing carriage return
# and one pair of surrounding quotes are removed. Values are never printed.
load_kv_file() {
    local file="$1" line key value
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        [[ -z "${line//[[:space:]]/}" || "$line" =~ ^[[:space:]]*# ]] && continue
        line="${line#export }"
        if [[ ! "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            echo "skipping a line of $file that is not KEY=value" >&2
            continue
        fi
        key="${line%%=*}"
        value="${line#*=}"
        if [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]]; then
            value="${BASH_REMATCH[1]}"
        fi
        export "$key=$value"
    done < "$file"
}

load_secrets() {
    local file="$SECRETS_FILE" mode name
    [[ -f "$file" ]] || { echo "secrets file $file not found" >&2; return 1; }
    mode="$(stat -c %a "$file")"
    [[ "$mode" == "600" ]] || { echo "secrets file $file has mode $mode, not 600" >&2; return 1; }
    load_kv_file "$file"
    for name in HF_TOKEN HF_REPO PRIME_API_KEY RENDER_IMAGE WANDB_API_KEY; do
        [[ -n "${!name:-}" ]] || { echo "$name is not set in $file" >&2; return 1; }
    done
}

load_job_env() {
    [[ -f "$JOB_ENV" ]] && load_kv_file "$JOB_ENV"
    return 0
}

# Any Python 3 that can parse JSON: the tools environment if it exists, else the system one.
json_python() {
    if [[ -x "$TOOLS_DIR/bin/python" ]]; then echo "$TOOLS_DIR/bin/python"; else echo python3; fi
}

# The id of the one pod named $POD_NAME (`prime pods list -o json`: .pods[] = {id, name, ...}).
pod_id_by_name() {
    prime pods list -o json | "$(json_python)" -c '
import json, sys
ids = [p["id"] for p in json.load(sys.stdin).get("pods", []) if p.get("name") == sys.argv[1]]
if len(ids) != 1:
    sys.exit(f"found {len(ids)} pods named {sys.argv[1]}")
print(ids[0])
' "$POD_NAME"
}

# Terminate this pod, retrying. Uses POD_ID from job.env (written by launch.sh),
# and falls back to a lookup by POD_NAME.
terminate_self() {
    local attempt pod_id
    command -v prime >/dev/null || { log "the Prime CLI is not installed"; return 1; }
    for attempt in $(seq 1 "$TERMINATE_ATTEMPTS"); do
        pod_id="${POD_ID:-}"
        [[ -n "$pod_id" || -z "${POD_NAME:-}" ]] || pod_id="$(pod_id_by_name)" || pod_id=""
        if [[ -n "$pod_id" ]] && prime pods terminate "$pod_id" -y; then
            log "terminate requested for pod ${POD_NAME:-$pod_id}"
            return 0
        fi
        log "terminate attempt $attempt of $TERMINATE_ATTEMPTS failed"
        sleep "$TERMINATE_DELAY"
    done
    return 1
}

# uv and the Prime CLI, installed if missing (and curl first, which a plain image may lack).
ensure_prime_cli() {
    local sudo_cmd=""
    if ! command -v curl >/dev/null; then
        [[ "$(id -u)" == 0 ]] || sudo_cmd="sudo"
        $sudo_cmd apt-get update -y >/dev/null && DEBIAN_FRONTEND=noninteractive $sudo_cmd apt-get install -y curl >/dev/null \
            || return 1
    fi
    command -v uv >/dev/null || curl -LsSf --max-time 300 https://astral.sh/uv/install.sh | sh
    command -v prime >/dev/null || uv tool install "prime==$PRIME_CLI_VERSION"
}

# Everything needed to upload logs and terminate the pod, before anything else:
# git and curl, uv, the Prime CLI, and a small huggingface_hub environment that
# never depends on prime-rl's.
ensure_tools() {
    local missing=() tool sudo_cmd=""
    for tool in git curl; do command -v "$tool" >/dev/null || missing+=("$tool"); done
    if (( ${#missing[@]} )); then
        [[ "$(id -u)" == 0 ]] || sudo_cmd="sudo"
        $sudo_cmd apt-get update -y && DEBIAN_FRONTEND=noninteractive $sudo_cmd apt-get install -y "${missing[@]}" \
            || return 1
    fi
    ensure_prime_cli || return 1
    [[ -x "$TOOLS_DIR/bin/python" ]] || uv venv --quiet --python 3.12 "$TOOLS_DIR" || return 1
    "$TOOLS_DIR/bin/python" -c "import huggingface_hub" 2>/dev/null \
        || uv pip install --quiet --python "$TOOLS_DIR/bin/python" "huggingface_hub==$HF_HUB_VERSION"
}

# The keys work for what the job needs at its end. Prints which check failed,
# never a value. Needs load_secrets, load_job_env and ensure_tools first.
check_credentials() {
    local found
    if ! "$TOOLS_DIR/bin/python" - <<'EOF'
import os
import sys

from huggingface_hub import HfApi

api = HfApi()  # reads HF_TOKEN from the environment
try:
    api.whoami()
except Exception as error:
    sys.exit(f"credential check failed: the Hugging Face token was refused ({type(error).__name__})")
try:
    api.create_repo(os.environ["HF_REPO"], repo_type="model", private=True, exist_ok=True)
    private = api.repo_info(os.environ["HF_REPO"], repo_type="model").private
except Exception as error:
    sys.exit(f"credential check failed: the Hugging Face repo could not be created or read ({type(error).__name__})")
if not private:
    sys.exit("credential check failed: the Hugging Face repo exists and is public")
print("Hugging Face token and private repo: ok")

# W&B: ask its API who owns the key (basic auth, user "api"), without printing it.
import base64
import json
import urllib.request

token = base64.b64encode(f"api:{os.environ['WANDB_API_KEY']}".encode()).decode()
request = urllib.request.Request(
    "https://api.wandb.ai/graphql",
    data=json.dumps({"query": "{ viewer { username } }"}).encode(),
    headers={"Authorization": f"Basic {token}", "Content-Type": "application/json", "User-Agent": "isometric-rl-check"},
)
try:
    with urllib.request.urlopen(request, timeout=30) as response:
        viewer = json.load(response).get("data", {}).get("viewer")
except Exception as error:
    sys.exit(f"credential check failed: the W&B key could not be checked ({type(error).__name__})")
if not viewer:
    sys.exit("credential check failed: the W&B key was refused")
print("W&B key: ok")
EOF
    then
        return 1
    fi
    if ! found="$(pod_id_by_name)"; then
        echo "credential check failed: the Prime CLI could not find this pod by name" >&2
        return 1
    fi
    if [[ -n "${POD_ID:-}" && "$found" != "$POD_ID" ]]; then
        echo "credential check failed: the pod found by name is not this pod" >&2
        return 1
    fi
    echo "Prime pod lookup: ok"
}
