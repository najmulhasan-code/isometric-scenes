#!/usr/bin/env bash
# Prepare a fresh GPU pod for the training job. run_all.sh runs it as its first
# stage; it can also be run alone: bash pod_setup.sh
#
# Expects this layout under JOB_DIR (launch.sh copies it there):
#   $JOB_DIR/src/training/                  this folder
#   $JOB_DIR/src/environments/isometric_scenes/
#   $JOB_DIR/secrets.env                    HF_TOKEN, HF_REPO, PRIME_API_KEY, RENDER_IMAGE, WANDB_API_KEY
#
# Order: the small tools needed to upload logs and terminate the pod (git, curl,
# uv, Prime CLI, huggingface_hub); the secrets and a check that the keys work
# (Hugging Face token and private repo, Prime pod lookup); GPUs, driver and
# disk; then prime-rl at the pinned commit, the isometric-scenes package and dry
# runs of every config. Any failure stops the script with a nonzero exit.

set -euo pipefail
# shellcheck source=common.sh
. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PRIME_RL_REPO="https://github.com/PrimeIntellect-ai/prime-rl.git"
PRIME_RL_COMMIT="35fcb52c9496b68903c14136b65fb7fa1255b658"
VERIFIERS_COMMIT="540f40faab8c376b85c6afa2f872683b0ed1d4d1"   # deps/verifiers at that commit
# prime-rl installs CUDA 13 wheels, which need driver >= 580 on Linux; that also
# covers Blackwell GPUs (RTX PRO 6000), which need CUDA 12.8 or later.
MIN_DRIVER_MAJOR=580
# LoRA: 11 DCP checkpoints (the model state plus the adapter's optimizer state,
# trainer/ckpt.py AppState), the base model, adapters and traces.
MIN_FREE_GB=300
BASE_MODEL="Qwen/Qwen3.5-4B"

fail() { echo "pod_setup: $*" >&2; exit 1; }
step() { echo; echo "== $*"; }

step "Tools: git, curl, uv, Prime CLI, huggingface_hub"
# First, so a failure of any later check can still upload the logs and terminate the pod.
ensure_tools || fail "the upload and terminate tools could not be installed"
uv --version
prime --version
"$TOOLS_DIR/bin/python" -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"

step "Secrets and credentials"
load_secrets || fail "secrets could not be loaded"
load_job_env
echo "loaded HF_TOKEN, HF_REPO, PRIME_API_KEY, RENDER_IMAGE"
check_credentials || fail "a credential check failed"

step "GPUs and driver"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found: no NVIDIA driver"
gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
(( gpu_count >= 2 )) || fail "found $gpu_count GPU(s); the job needs 2"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
driver_major="${driver%%.*}"
[[ "$driver_major" =~ ^[0-9]+$ ]] || fail "could not read the driver version ($driver)"
(( driver_major >= MIN_DRIVER_MAJOR )) \
    || fail "driver $driver is too old for CUDA 13 wheels; need >= $MIN_DRIVER_MAJOR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "driver $driver: ok"

step "Disk"
mkdir -p "$JOB_DIR"
free_gb="$(df -BG --output=avail "$JOB_DIR" | tail -n1 | tr -dc '0-9')"
(( free_gb >= MIN_FREE_GB )) || fail "${free_gb} GB free at $JOB_DIR; need at least $MIN_FREE_GB GB"
echo "${free_gb} GB free: ok"

step "prime-rl at $PRIME_RL_COMMIT"
# The submodules are listed with SSH URLs; a fresh pod has no GitHub key.
git config --global url."https://github.com/".insteadOf "git@github.com:"
if [[ ! -d "$PRIME_RL_DIR/.git" ]]; then
    git clone "$PRIME_RL_REPO" "$PRIME_RL_DIR"
fi
git -C "$PRIME_RL_DIR" fetch --quiet origin
git -C "$PRIME_RL_DIR" checkout --quiet "$PRIME_RL_COMMIT"
git -C "$PRIME_RL_DIR" submodule update --init --recursive
[[ "$(git -C "$PRIME_RL_DIR" rev-parse HEAD)" == "$PRIME_RL_COMMIT" ]] || fail "prime-rl is not at $PRIME_RL_COMMIT"
[[ "$(git -C "$PRIME_RL_DIR/deps/verifiers" rev-parse HEAD)" == "$VERIFIERS_COMMIT" ]] \
    || fail "deps/verifiers is not at $VERIFIERS_COMMIT"

step "uv sync --all-extras"
(cd "$PRIME_RL_DIR" && uv sync --all-extras)
PY="$PRIME_RL_DIR/.venv/bin/python"

# Version and location of the installed verifiers, to compare before and after.
verifiers_state() {
    "$PY" - <<'EOF'
import importlib.metadata as m, pathlib, verifiers
print(m.version("verifiers"), pathlib.Path(verifiers.__file__).resolve().parent)
EOF
}

step "isometric-scenes into the prime-rl environment"
before="$(verifiers_state)"
# --no-deps keeps the pinned verifiers; the other dependency, prime-sandboxes
# (>=0.5.0,<0.6), is checked below rather than installed.
uv pip install --python "$PY" --no-deps -e "$SRC_DIR/environments/isometric_scenes"
after="$(verifiers_state)"
[[ "$before" == "$after" ]] || fail "installing isometric-scenes changed verifiers: '$before' -> '$after'"
echo "verifiers unchanged: $after"
"$PY" - <<'EOF'
import importlib.metadata as m
version = tuple(int(p) for p in m.version("prime-sandboxes").split(".")[:2])
assert (0, 5) <= version < (0, 6), f"prime-sandboxes {m.version('prime-sandboxes')} is outside >=0.5.0,<0.6"
from verifiers.v1.utils.loaders import taskset_class
print("taskset:", taskset_class("isometric-scenes").__name__, "| prime-sandboxes", m.version("prime-sandboxes"))
EOF

step "Config dry runs"
DRY="$JOB_DIR/dry-run"
mkdir -p "$DRY"
printf '[sampling]\nseed = 0\n' > "$DRY/seed_0.toml"
cd "$PRIME_RL_DIR"
uv run --no-sync rl @ "$TRAINING_DIR/rl.toml" --output-dir "$DRY" --run.name rl --clean --dry-run
uv run --no-sync rl @ "$TRAINING_DIR/rl.toml" @ "$TRAINING_DIR/smoke.toml" \
    --output-dir "$DRY" --run.name rl-smoke --clean --dry-run
uv run --no-sync eval @ "$TRAINING_DIR/final_eval.toml" @ "$DRY/seed_0.toml" --model "$BASE_MODEL" \
    --output-dir "$DRY" --run.name eval --clean --no-monitors.prime --dry-run
uv run --no-sync inference --vllm.model "$BASE_MODEL" --vllm.reasoning-parser None \
    --vllm.enable-lora --vllm.max-lora-rank 32 \
    --output-dir "$DRY/inference" --dry-run

echo
echo "pod_setup: done"
