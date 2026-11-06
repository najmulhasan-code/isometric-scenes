"""Run manifest: what a run was built from, checked again when it resumes.

The manifest records the generator model, the pilot round, the SHA-256 of
the config, templates, example lists and term lists, the git commit of the
build code, whether the working tree was clean, and the Python version. A run only
resumes when the model, pilot round, setting overrides, code commit and
file hashes are unchanged.
"""

import hashlib
import json
import platform
import subprocess
from pathlib import Path

HASHED_DIRS = ("templates", "lists", "terms")
IDENTITY_KEYS = ("model", "pilot_round", "overrides", "code_commit", "files")


def file_hashes(root: Path) -> dict[str, str]:
    paths = [root / "config.toml"]
    for name in HASHED_DIRS:
        paths.extend(sorted(path for path in (root / name).iterdir() if path.is_file()))
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build_manifest(
    root: Path, model: str, pilot_round: int | None, overrides: dict | None = None
) -> dict:
    repo = root.parent
    status = _git(repo, "status", "--porcelain")
    return {
        "model": model,
        "pilot_round": pilot_round,
        "overrides": overrides or {},
        "files": file_hashes(root),
        "code_commit": _git(repo, "rev-parse", "HEAD"),
        "working_tree_clean": None if status is None else status == "",
        "python": platform.python_version(),
    }


def check_or_write(path: Path, manifest: dict) -> list[str]:
    """Write the manifest for a new run, or list how an existing run differs."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        return []
    existing = json.loads(path.read_text(encoding="utf-8"))
    differences = []
    for key in IDENTITY_KEYS:
        if existing.get(key) == manifest[key]:
            continue
        if key == "files":
            old, new = existing.get(key, {}), manifest[key]
            changed = sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))
            differences.append(f"changed files: {', '.join(changed)}")
        else:
            differences.append(f"{key}: {existing.get(key)!r} became {manifest[key]!r}")
    return differences
