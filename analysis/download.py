"""Download the run's results from the model repository on Hugging Face into DATA_DIR.

The repository defaults to the released adapter repository, which also holds the
final-evaluation outputs; RESULTS_REPO overrides it. The revision is pinned to the one
the reported results come from; RESULTS_REVISION overrides it. Downloads only what the
analysis needs: the evaluation judge's scores, the final-evaluation answers, the
checkpoint selection and the version record.

  python download.py
"""

import os
import sys

from huggingface_hub import snapshot_download

from common import DATA_DIR

DEFAULT_REPO = "najmulhasan/qwen3.5-4b-isometric-scenes-lora"
# Set to the public repository commit at release.
REVISION = "701a3db54ab64d88b0bb789710e4a4b27ec9bcba"
PATTERNS = [
    "final_eval/*/seed*/rescore/*",
    "final_eval/*/seed*/monitors/file/traces/stream/*",
    "selection.json",
    "versions.txt",
]


def main() -> int:
    repo = os.environ.get("RESULTS_REPO", DEFAULT_REPO)
    revision = os.environ.get("RESULTS_REVISION", REVISION)
    snapshot_download(repo, repo_type="model", revision=revision, token=os.environ.get("HF_TOKEN"),
                      allow_patterns=PATTERNS, local_dir=DATA_DIR)
    print(f"downloaded revision {revision[:12]} to {DATA_DIR.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
