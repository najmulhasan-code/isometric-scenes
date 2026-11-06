"""The public raw data: the inputs needed to rebuild the released splits offline.

The export holds:

- runs/<run>/requests.jsonl and manifest.json for every generator run that
  feeds the pool or the late reserves, and for the threshold-check build;
- review/prescreen_round<N>.jsonl, the pre-screen model's replies for each
  review round;
- review/decisions.json, the decisions a person made: the pairs judged the
  same scene in the threshold check, the spot-check removals of each round,
  and which refill run supplied late reserves for which stratum, in order;
- expected/SHA256SUMS and expected/release.json, the published checksums and
  release record that a rebuild is compared against;
- MANIFEST.json, the size and SHA-256 of every file above.

In every API log the provider's request identifiers and other fields that
can identify an account are removed (PRIVATE_KEYS), both from response
bodies and from error bodies kept as text. No step of the rebuild reads
them. Pilot and diagnostic runs, review reading lists, embedding caches and
the build outputs are left out: the rebuild does not read them, or computes
them again.
"""

import getpass
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from .generate import RunLog

PRIVATE_KEYS = frozenset({
    "id",
    "system_fingerprint",
    "request_id",
    "inference_id",
    "signature",
    "user",
    "user_id",
    "team_id",
    "organization",
    "organization_id",
    "account_id",
})
POOL_RUNS = ("main", "refill-1", "refill-2", "refill-3")
THRESHOLD_RUNS = ("main",)
_ERROR_BODY = re.compile(r"\A(HTTP \d+: )(.*)\Z", re.S)
_PRIVATE_PAIR = re.compile(
    r'"(?:' + "|".join(sorted(PRIVATE_KEYS)) + r')"\s*:\s*(?:"(?:[^"\\]|\\.)*"|[^,}\]]*)\s*,?\s*'
)


def scrub(value, removed: Counter):
    """`value` without any PRIVATE_KEYS, at any depth. Counts what was removed."""
    if isinstance(value, dict):
        kept = {}
        for key, item in value.items():
            if key in PRIVATE_KEYS:
                removed[key] += 1
            else:
                kept[key] = scrub(item, removed)
        return kept
    if isinstance(value, list):
        return [scrub(item, removed) for item in value]
    return value


def scrub_error(text: str, removed: Counter) -> str:
    """An error message with PRIVATE_KEYS removed from the error body it quotes."""
    match = _ERROR_BODY.match(text)
    if not match:
        return text
    head, body = match.groups()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        def drop(found: re.Match) -> str:
            removed[found.group(0).split('"')[1]] += 1
            return ""
        return head + _PRIVATE_PAIR.sub(drop, body)
    return head + json.dumps(scrub(data, removed), ensure_ascii=False, separators=(",", ":"))


def scrub_record(record: dict, removed: Counter) -> dict:
    """A log record with the private fields of its response and error removed."""
    out = {}
    for key, value in record.items():
        if key == "response" and value is not None:
            value = scrub(value, removed)
        elif key == "error" and isinstance(value, str):
            value = scrub_error(value, removed)
        out[key] = value
    return out


def write_log(records: list[dict], path: Path) -> None:
    """Write records in the run log format: one JSON object per line, UTF-8, LF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def late_reserve_events(build_dir: Path) -> list[dict]:
    """The refill runs screened for late reserves, in the order they were added."""
    path = build_dir / "review" / "late_candidates.jsonl"
    events = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            event = {"run": row["run"], "stratum": row["stratum"]}
            if not events or events[-1] != event:
                events.append(event)
    return events


def decisions(root: Path, build: str, threshold_build: str, pool_runs, threshold_runs) -> dict:
    """The person's decisions, taken from the review state and the threshold-check result."""
    build_dir = root / "builds" / build
    check_dir = root / "builds" / threshold_build
    state = _json(build_dir / "review" / "state.json")
    result = _json(check_dir / "threshold_check_result.json")
    key = (check_dir / "threshold_check_key.json").read_bytes()
    return {
        "pool_runs": list(pool_runs),
        "threshold_check": {
            "runs": list(threshold_runs),
            "cosine_threshold": _json(check_dir / "summary.json")["cosine_threshold"],
            "judged_same": result["judged_same"],
            "key_sha256": hashlib.sha256(key).hexdigest(),
        },
        "spot_check_removals": [round_["author"]["removed"] for round_ in state["rounds"]],
        "late_reserves": late_reserve_events(build_dir),
    }


def export_public(
    root: Path,
    out: Path,
    build: str = "main",
    threshold_build: str = "threshold-check",
    pool_runs=POOL_RUNS,
    threshold_runs=THRESHOLD_RUNS,
) -> dict:
    """Copy the minimal rebuild inputs from `root` to `out`, with private fields removed.

    `out` must not exist or be empty. Returns the manifest written to
    out/MANIFEST.json.
    """
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    build_dir = root / "builds" / build
    chosen = decisions(root, build, threshold_build, pool_runs, threshold_runs)
    runs = list(dict.fromkeys([*pool_runs, *threshold_runs, *(e["run"] for e in chosen["late_reserves"])]))
    removed: Counter = Counter()

    for name in runs:
        source = root / "runs" / name
        write_log([scrub_record(r, removed) for r in RunLog(source / "requests.jsonl").read()],
                  out / "runs" / name / "requests.jsonl")
        shutil.copyfile(source / "manifest.json", out / "runs" / name / "manifest.json")

    for number in range(1, len(chosen["spot_check_removals"]) + 1):
        name = f"prescreen_round{number}.jsonl"
        records = RunLog(build_dir / "review" / name).read()
        write_log([scrub_record(r, removed) for r in records], out / "review" / name)
    _dump(chosen, out / "review" / "decisions.json")

    (out / "expected").mkdir(parents=True, exist_ok=True)
    for name in ("SHA256SUMS", "release.json"):
        shutil.copyfile(build_dir / "release" / name, out / "expected" / name)

    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {
        "removed_fields": [{"field": k, "count": n} for k, n in sorted(removed.items())],
        "files": {
            p.relative_to(out).as_posix(): {"bytes": p.stat().st_size,
                                            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in files
        },
    }
    _dump(manifest, out / "MANIFEST.json")
    return manifest


def scan_patterns() -> dict[str, re.Pattern]:
    """Patterns for text that should not be published, including this machine's user and home folder."""
    patterns = {
        "windows path": re.compile(r"\b[A-Za-z]:(?:\\\\|\\|/)(?:Users|home|Documents)", re.I),
        "home path": re.compile(r"(?<![\w.])/(?:Users|home|root)/[\w.-]+"),
        "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
        "key": re.compile(r"\b(?:sk-[\w-]{16,}|(?:sk|pk|pit|hf|ghp|gho)_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})"),
        "bearer token": re.compile(r"Bearer\s+[\w.-]{8,}"),
        "private field": re.compile(r'"(?:' + "|".join(sorted(PRIVATE_KEYS)) + r')"\s*:'),
        "team or user id": re.compile(r"\b(?:team|user|account|org)[_-]?id\b\W{0,3}[\w-]{6,}", re.I),
    }
    user = getpass.getuser()
    if len(user) >= 3:
        patterns["user name"] = re.compile(r"\b" + re.escape(user) + r"\b", re.I)
    home = str(Path.home())
    patterns["home folder"] = re.compile(
        "|".join(re.escape(v) for v in {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}), re.I
    )
    return patterns


def scan(out: Path, limit: int = 5) -> list[str]:
    """Lines of the export that match a scan pattern, at most `limit` per file and pattern."""
    hits = []
    patterns = scan_patterns()
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        found = Counter()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for name, pattern in patterns.items():
                match = pattern.search(line)
                if match and found[name] < limit:
                    found[name] += 1
                    start = max(0, match.start() - 30)
                    hits.append(f"{path.relative_to(out).as_posix()}:{number}: {name}: "
                                f"...{line[start:match.end() + 30]}...")
    return hits
