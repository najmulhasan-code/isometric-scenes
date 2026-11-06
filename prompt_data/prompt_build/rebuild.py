"""Rebuild the released splits from the public raw data, offline.

Every step after generation is run again from the raw data:

1. the items of every run log are parsed again from the raw replies and
   compared with the items the log holds;
2. the threshold-check build is made again, and the decision rule is applied
   to the recorded judgments, which must give the configured threshold;
3. the main build is made again from the pool runs;
4. the review is replayed: each round is opened by the code, the pre-screen
   verdicts are read from the recorded replies (a verdict counts only for the
   same candidate and the same request), the spot-check sample is drawn, the
   recorded spot-check removals are applied, and the round is closed. When a
   stratum runs out of reserves, the recorded late-reserve refills are
   screened again, in their recorded order;
5. the release checks run and the split files are written.

Embeddings are computed again with the pinned encoder. The candidates,
review state and late reserves are compared with the hashes in the release
record, and the split files with the published SHA256SUMS. Nothing is sent
over the network; the encoder must already be in the Hugging Face cache.
"""

import dataclasses
import hashlib
import json
from pathlib import Path
from string import Template

from .build import build_settings, run_build, write_outputs
from .config import BuildConfig
from .generate import RunLog, parse_response
from .late_reserves import add_late_reserves
from .prescreen import recorded_round
from .release import release_build
from .review import SPOT_CHECK_SIZE, NoReserves, Review
from .threshold_check import decide


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sums(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        sums[name] = digest
    return sums


def parsed_item_problems(raw: Path, runs: list[str]) -> list[str]:
    """Logged requests whose items differ from the items parsed again from the raw reply."""
    problems = []
    for name in runs:
        for record in RunLog(raw / "runs" / name / "requests.jsonl").read():
            if record.get("response") is None:
                continue
            parsed = parse_response(record["response"], record["stratum"].startswith("distractor"), record["size"])
            if (parsed.items or [], parsed.status) != (record["items"], record["status"]):
                problems.append(f"{name} request {record['stratum']} {record['request_index']}: "
                                "items differ from the reply")
    return problems


def with_threshold(config: BuildConfig, threshold: float) -> BuildConfig:
    return dataclasses.replace(config, build=dataclasses.replace(config.build, cosine_threshold=threshold))


def replay_review(config: BuildConfig, root: Path, raw: Path, build_dir: Path, decisions: dict) -> None:
    """Run the review of a fresh build from the recorded verdicts and decisions."""
    config = build_settings(config, build_dir)
    template = Template((root / "templates" / "prescreen.txt").read_text(encoding="utf-8"))
    removals = decisions["spot_check_removals"]
    pending = list(decisions["late_reserves"])
    review = Review(build_dir, config)
    review.start()
    while True:
        number = len(review.state["rounds"])
        if number > len(removals):
            raise ValueError(f"round {number} has no recorded spot-check result")
        records = RunLog(raw / "review" / f"prescreen_round{number}.jsonl").read()
        found = recorded_round(review, config.prescreen, template, config.categories, records)
        if found is None:
            raise ValueError(f"round {number}: the log has no valid verdict for some item")
        flags = {n: v for n, v in found.items() if v.flag}
        review.flag({n: v.flag for n, v in flags.items()}, {n: v.reason for n, v in flags.items()})
        review.spot_check(SPOT_CHECK_SIZE if number == 1 else 0)
        review.decide({int(n): reason for n, reason in removals[number - 1].items()})
        while True:
            try:
                size = review.close()
                break
            except NoReserves as error:
                # Add the recorded refills in order, up to the first one for this stratum.
                strata = [event["stratum"] for event in pending]
                if error.stratum not in strata:
                    raise
                for event in pending[: strata.index(error.stratum) + 1]:
                    add_late_reserves(config, root, raw, build_dir, event["stratum"], event["run"])
                del pending[: strata.index(error.stratum) + 1]
                review = Review(build_dir, config)
        if size == 0:
            break
    for event in pending:
        add_late_reserves(config, root, raw, build_dir, event["stratum"], event["run"])
    if len(review.state["rounds"]) != len(removals):
        raise ValueError(f"the replay closed after {len(review.state['rounds'])} rounds, "
                         f"the record has {len(removals)}")


def rebuild(config: BuildConfig, root: Path, raw: Path, out: Path, say=print) -> list[str]:
    """Rebuild the splits from `raw` into `out` and list every difference from the published files."""
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    decisions = json.loads((raw / "review" / "decisions.json").read_text(encoding="utf-8"))
    record = json.loads((raw / "expected" / "release.json").read_text(encoding="utf-8"))
    expected = read_sums(raw / "expected" / "SHA256SUMS")
    check = decisions["threshold_check"]
    runs = list(dict.fromkeys([*decisions["pool_runs"], *check["runs"],
                               *(e["run"] for e in decisions["late_reserves"])]))
    problems = parsed_item_problems(raw, runs)
    say(f"run logs: {len(runs)} runs parsed again from the raw replies")

    check_dir = out / "threshold-check"
    check_config = with_threshold(config, check["cosine_threshold"])
    result = run_build(check_config, root, check["runs"], check_dir, runs_root=raw)
    write_outputs(result, check_config, check_dir)
    if sha256(check_dir / "threshold_check_key.json") != check["key_sha256"]:
        problems.append("threshold-check key differs from the one that was read")
    key = json.loads((check_dir / "threshold_check_key.json").read_text(encoding="utf-8"))
    threshold, _ = decide(key, set(check["judged_same"]), check["cosine_threshold"])
    if threshold != config.build.cosine_threshold:
        problems.append(f"threshold check gives {threshold}, config has {config.build.cosine_threshold}")
    say(f"threshold check: {check['cosine_threshold']} -> {threshold}")

    build_dir = out / "main"
    main_config = with_threshold(config, threshold)
    result = run_build(main_config, root, decisions["pool_runs"], build_dir, runs_root=raw)
    write_outputs(result, main_config, build_dir)
    if not result.assignment.complete or result.problems:
        problems.append("the main build is incomplete or fails its checks")
    say(f"build: {sum(c.kept for c in result.candidates)} candidates kept")

    replay_review(main_config, root, raw, build_dir, decisions)
    say(f"review: {len(Review(build_dir, main_config).state['rounds'])} rounds replayed")
    for name, digest in record["inputs"].items():
        path = build_dir / name
        if not path.exists() or sha256(path) != digest:
            problems.append(f"{name} differs from the release record")

    release_problems, digests = release_build(main_config, root, build_dir)
    problems += release_problems
    for name, digest in sorted(expected.items()):
        same = digests.get(name) == digest
        say(f"{name:20s} {'identical' if same else 'DIFFERENT'}")
        if not same:
            problems.append(f"{name} differs from the published file")
    return problems
