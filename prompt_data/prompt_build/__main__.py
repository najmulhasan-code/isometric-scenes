"""Command-line entry point: python -m prompt_build <command>.

Commands:
  check     validate the config, word lists and templates
  plan      show the strata and the number of requests, without sending any
  pilot     send the pilot requests for one round
  diagnose  send six fixed requests with a chosen model and settings, to test
            the request setup; diagnostic outputs never enter the pool
  report    score a run against the pilot pass criteria and write the
            hand-check sample
  generate  send the main-run requests, resuming an earlier run if one exists
  build     clean the pool and assign splits from one or more runs (offline)
  threshold-check
            apply the threshold-check decision rule to the judged pairs
  refill    send further requests for the strata a build reported as short, or
            for one stratum that ran out of reserves during review
  review    run the review of a build, one step at a time
  reserves  screen a refill run's candidates and add them as reserves during review
  release   check the reviewed splits and write the dataset files
  export-public
            copy the inputs needed to rebuild the release into a folder, with
            provider request IDs and account fields removed from the API logs
  rebuild   rebuild the released splits from that folder, offline, and compare
            them with the published checksums

The API key is read from PRIME_API_KEY, which may be set in a .env file at
the repository root. That file is ignored by git.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from string import Template

from .build import build_settings, run_build, write_outputs
from .client import ChatClient
from .config import BuildConfig, load_config
from .generate import Generation, RunLog, pilot_strata, run_main, run_pilot
from .lexicon import ELEMENT_KINDS, Lexicon, load_lexicon, read_word_list
from .manifest import build_manifest, check_or_write
from .plan import ExampleLists, ExampleSchedule, Templates, exposure, seed_block
from .prescreen import TEMPLATE_FIELDS as PRESCREEN_FIELDS
from .prescreen import log_path, prescreen_round
from .public import POOL_RUNS, THRESHOLD_RUNS, export_public, scan
from .rebuild import rebuild
from .late_reserves import add_late_reserves
from .refill import refill_stratum
from .release import release_build
from .review import SPOT_CHECK_SIZE, Review
from .report import build_report, format_report, hand_check_sample
from .strata import Stratum, build_strata
from .threshold_check import decide

DATA_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = DATA_ROOT.parent / ".env"

# The fixed strata of a diagnostic run, taken from the pilot strata.
DIAGNOSTIC_KEYS = (
    "in_scope/single_objects/motion",
    "in_scope/home_interiors/still",
    "in_scope/transport/motion",
    "in_scope/nature_islands/still",
    "distractor/transport/animal/motion",
    "distractor/home_interiors/face_figure/still",
)

TEMPLATE_FIELDS = {
    "in_scope.txt": {"n", "category_name", "category_definition", "focus", "motion_rule"},
    "prescreen.txt": PRESCREEN_FIELDS,
    "distractor.txt": {
        "n",
        "category_name",
        "category_definition",
        "focus",
        "motion_rule",
        "element_rule",
        "insertion_example",
    },
}


def strip_article(entry: str) -> str:
    first, _, rest = entry.partition(" ")
    return rest if first in ("a", "an") and rest else entry


def check_example_lists(lists_dir: Path, lexicon: Lexicon) -> list[str]:
    """Each element example must give exactly one match of its list's kind."""
    problems = []
    for kind in ELEMENT_KINDS:
        path = lists_dir / f"{kind}_examples.txt"
        for entry in read_word_list(path):
            matches = lexicon.matcher.find(strip_article(entry))
            if len(matches) != 1 or matches[0].term.kind != kind:
                found = ", ".join(f"{m.term.lemma} ({m.term.kind})" for m in matches) or "nothing"
                problems.append(f"{path.name}: {entry!r} matches {found}")

    path = lists_dir / "motion_examples.txt"
    for entry in read_word_list(path):
        if not lexicon.mentions_motion(entry):
            problems.append(f"{path.name}: {entry!r} contains no word from motion_terms.txt")
        if lexicon.matcher.find(entry):
            problems.append(f"{path.name}: {entry!r} contains an out-of-scope term")

    path = lists_dir / "insertion_frames.txt"
    for entry in read_word_list(path):
        if entry.count("{element}") != 1 or entry.split()[0] not in ("with", "and", "where"):
            problems.append(f"{path.name}: {entry!r} needs one {{element}} and a leading with, and or where")
        if lexicon.matcher.find(entry.replace("{element}", "")) or lexicon.mentions_motion(entry):
            problems.append(f"{path.name}: {entry!r} contains an out-of-scope or motion term")
    return problems


def check_templates(templates_dir: Path) -> list[str]:
    problems = []
    for name, expected in TEMPLATE_FIELDS.items():
        template = Template((templates_dir / name).read_text(encoding="utf-8"))
        if not template.is_valid():
            problems.append(f"{name}: contains an invalid placeholder")
            continue
        found = set(template.get_identifiers())
        if found != expected:
            problems.append(
                f"{name}: placeholders {sorted(found)} differ from expected {sorted(expected)}"
            )
    return problems


def run_check(root: Path) -> list[str]:
    load_config(root / "config.toml")
    lexicon = load_lexicon(root / "terms")
    return check_example_lists(root / "lists", lexicon) + check_templates(root / "templates")


def parse_env_value(raw: str) -> str:
    """Value of a KEY=VALUE line.

    A quoted value runs to its closing quote, and anything after it is
    ignored. An unquoted value ends at " #". An opening quote without a
    closing one is kept as part of the value.
    """
    value = raw.strip()
    if value[:1] in ("'", '"'):
        end = value.find(value[0], 1)
        if end > 0:
            return value[1:end]
    return value.split(" #", 1)[0].rstrip()


def load_env_file(path: Path) -> None:
    """Read KEY=VALUE lines into the environment, without overriding set variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), parse_env_value(value))


def make_schedule(config: BuildConfig, root: Path, strata: list[Stratum]) -> ExampleSchedule:
    return ExampleSchedule.build(
        strata,
        ExampleLists.load(root / "lists"),
        config.base_seed + config.seed_offsets["example_order"],
        config.generator.items_per_request,
    )


def make_generation(
    config: BuildConfig,
    root: Path,
    run_name: str,
    pilot_round: int | None,
    strata: list[Stratum],
    diagnostic: bool = False,
    overrides: dict | None = None,
) -> Generation:
    load_env_file(ENV_FILE)
    settings = config.generator
    if settings is None:
        raise ValueError("config.toml has no [generator] section")
    if overrides:
        settings = dataclasses.replace(settings, **overrides)
    client = ChatClient(
        base_url=settings.base_url,
        api_key=os.environ.get("PRIME_API_KEY", ""),
        timeout=settings.timeout_seconds,
        team_id=os.environ.get("PRIME_TEAM_ID") or None,
    )
    return Generation(
        run_name=run_name,
        pilot_round=pilot_round,
        seed_block=seed_block(pilot_round, diagnostic),
        base_seed=config.base_seed,
        settings=settings,
        templates=Templates.load(root / "templates"),
        examples=ExampleLists.load(root / "lists"),
        schedule=make_schedule(config, root, strata),
        log=RunLog(root / "runs" / run_name / "requests.jsonl"),
        complete=client.complete,
    )


def show_plan(config: BuildConfig) -> None:
    strata = build_strata(config)
    per_request = config.generator.items_per_request
    requests = sum(math.ceil(s.target / per_request) for s in strata)
    in_scope = [s for s in strata if not s.distractor]
    distractors = [s for s in strata if s.distractor]
    print(f"in-scope strata:   {len(in_scope)}, quota {sum(s.quota for s in in_scope)}")
    print(f"distractor strata: {len(distractors)}, quota {sum(s.quota for s in distractors)}")
    print(f"candidates to collect: {sum(s.target for s in strata)}")
    print(f"requests, at least: {requests} ({per_request} items each)")
    pilot = pilot_strata(strata, config.split_categories("train"))
    print(f"pilot requests per round: {len(pilot)}")
    shown = exposure(make_schedule(config, DATA_ROOT, strata), strata, per_request)
    for name, counts in shown.items():
        total = sum(counts.values())
        print(f"examples shown, {name}: {len(counts)} distinct, largest share {max(counts.values()) / total:.1%}")


def run_build_command(config: BuildConfig, root: Path, runs: list[str], name: str) -> int:
    out_dir = root / "builds" / name
    print(f"cosine threshold: {config.build.cosine_threshold}")
    result = run_build(config, root, runs, out_dir)
    write_outputs(result, config, out_dir)
    for stage, count in result.stages:
        print(f"{stage:26s} {count:5d} kept")
    for split, items in sorted(result.assignment.splits.items()):
        print(f"{split:26s} {len(items):5d} assigned")
    if result.assignment.shortfalls:
        print("short strata (refill needed):")
        for key, missing in sorted(result.assignment.shortfalls.items()):
            print(f"  {key}: {missing} missing")
    print(f"words over the cap: {', '.join(result.words['capped']) or 'none'}")
    if result.problems:
        print(f"final checks: {len(result.problems)} problem(s)")
        for problem in result.problems[:20]:
            print(f"  {problem}")
    else:
        print("final checks: all passed")
    print(f"Outputs: {out_dir}")
    return 0 if result.assignment.complete and not result.problems else 1


def run_threshold_check(config: BuildConfig, root: Path, args: argparse.Namespace) -> int:
    build_dir = root / "builds" / args.build
    threshold = build_settings(config, build_dir).build.cosine_threshold
    key = json.loads((build_dir / "threshold_check_key.json").read_text(encoding="utf-8"))
    numbers = {k["number"] for k in key}
    judged_same = set(args.same)
    if not judged_same <= numbers:
        print(f"pairs {sorted(judged_same - numbers)} are not in the list")
        return 1
    decided, counts = decide(key, judged_same, threshold)
    result = {"judged_same": sorted(judged_same), "counts": counts, "threshold": decided}
    (build_dir / "threshold_check_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for name, count in counts.items():
        print(f"{name:20s} {count['judged_same']:3d} of {count['pairs']:3d} judged the same scene")
    print(f"cosine threshold: {threshold} -> {decided}")
    return 0


def parse_marks(entries: list[str]) -> dict[int, str]:
    """Items given as NUMBER=REASON, for example 17=R8."""
    marks = {}
    for entry in entries:
        number, sep, reason = entry.partition("=")
        if not sep or not number.isdigit():
            raise ValueError(f"{entry!r} is not NUMBER=REASON")
        if int(number) in marks:
            raise ValueError(f"item {number} is given twice")
        marks[int(number)] = reason
    return marks


def run_review_command(config: BuildConfig, root: Path, args: argparse.Namespace) -> int:
    build_dir = root / "builds" / args.build
    review = Review(build_dir, config)
    categories = {c.slug: c.name for c in config.categories}
    out_dir = build_dir / "review"
    try:
        if args.step == "start":
            review.start()
            print(f"round 1: {len(review.current['items'])} items in {review.batches()} batches")
        elif args.step == "show":
            number = len(review.state["rounds"])
            lines = [line for batch in range(1, review.batches() + 1) for line in review.show(batch, categories)]
            path = out_dir / f"round{number}.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            print(f"round {number}: {len(lines)} items in {path}")
        elif args.step == "prescreen":
            return run_prescreen(config, root, review)
        elif args.step == "sample":
            size = SPOT_CHECK_SIZE if len(review.state["rounds"]) == 1 else 0
            print(f"spot-check sample: {len(review.spot_check(size))} items")
        elif args.step == "sheet":
            path = out_dir / "author_sheet.txt"
            path.write_text("\n".join(review.sheet(categories)) + "\n", encoding="utf-8", newline="\n")
            print(f"reading list: {path}")
        elif args.step == "decide":
            review.decide(parse_marks(args.remove))
            print("spot-check result recorded")
        elif args.step == "close":
            size = review.close()
            print(f"next round: {size} items" if size else "review complete")
        elif args.step == "status":
            for problem in review.coverage_problems():
                print(problem)
            print(json.dumps(review.summary(), indent=2))
    except ValueError as error:
        print(error)
        return 1
    return 0


def run_prescreen(config: BuildConfig, root: Path, review: Review) -> int:
    settings = config.prescreen
    if settings is None:
        raise ValueError("config.toml has no [prescreen] section")
    log = log_path(review)
    round_number = len(review.state["rounds"])
    manifest = build_manifest(root, settings.model, None, {"review_round": round_number, "base_url": settings.base_url})
    if manifest["working_tree_clean"] is not True:
        print("The working tree has uncommitted changes, or git could not be run. Commit first.")
        return 1
    differences = check_or_write(log.with_suffix(".manifest.json"), manifest)
    if differences:
        print("This pre-screen was started with different inputs and cannot be resumed:")
        for difference in differences:
            print(f"  {difference}")
        return 1
    load_env_file(ENV_FILE)
    client = ChatClient(
        base_url=settings.base_url,
        api_key=os.environ.get("PRIME_API_KEY", ""),
        timeout=settings.timeout_seconds,
        team_id=os.environ.get("PRIME_TEAM_ID") or None,
    )
    template = Template((root / "templates" / "prescreen.txt").read_text(encoding="utf-8"))
    outcome = prescreen_round(review, settings, template, config.categories, client.complete)
    print(f"requests sent: {outcome.sent}; log: {log}")
    print(f"totals over the whole log: {json.dumps(outcome.totals)}")
    for reason in outcome.stopped:
        print(f"stopped: {reason}")
    if outcome.flags is None:
        print("Some items have no valid verdict yet. Fix the cause, then run the step again.")
        return 1
    flags = {n: v for n, v in outcome.flags.items() if v.flag}
    review.flag({n: v.flag for n, v in flags.items()}, {n: v.reason for n, v in flags.items()})
    print(f"{len(flags)} model flags recorded")
    return 0


def run_refill_command(config: BuildConfig, root: Path, args: argparse.Namespace) -> int:
    if args.stratum:
        shortfalls = {args.stratum: args.missing}
    else:
        summary = json.loads((root / "builds" / args.build / "summary.json").read_text(encoding="utf-8"))
        shortfalls = summary["shortfalls"]
    if not shortfalls:
        print("The build has no short strata.")
        return 0
    manifest = build_manifest(root, config.generator.model, None)
    if manifest["working_tree_clean"] is not True:
        print("The working tree has uncommitted changes, or git could not be run. Commit first.")
        return 1
    differences = check_or_write(root / "runs" / args.run / "manifest.json", manifest)
    if differences:
        print(f"Run {args.run!r} was started with different inputs and cannot be resumed:")
        for difference in differences:
            print(f"  {difference}")
        return 1

    strata = build_strata(config)
    by_key = {s.key: s for s in strata}
    generation = make_generation(config, root, args.run, None, strata)
    logs = {name: RunLog(root / "runs" / name / "requests.jsonl").read() for name in (*args.runs, args.run)}
    earlier = [r for records in logs.values() for r in records]
    earlier_refills = [r for name, records in logs.items() if name != args.runs[0] for r in records]
    lexicon = load_lexicon(root / "terms")
    met = True
    for key, missing in sorted(shortfalls.items()):
        outcome = refill_stratum(generation, by_key[key], missing, earlier, earlier_refills, lexicon)
        met &= outcome.met
        print(f"{key}: missing {missing}, usable {outcome.usable}, batches {outcome.batches}")
    if generation.abort.is_set():
        for reason in generation.abort_reason:
            print(f"  stopped: {reason}")
    print(f"Log: {generation.log.path}")
    return 0 if met and not generation.abort.is_set() else 1


def run_release_command(config: BuildConfig, root: Path, build: str) -> int:
    build_dir = root / "builds" / build
    problems, digests = release_build(config, root, build_dir)
    if problems:
        print(f"final checks: {len(problems)} problem(s)")
        for problem in problems[:20]:
            print(f"  {problem}")
        return 1
    config = build_settings(config, build_dir)
    review = Review(build_dir, config)
    provenance = build_manifest(root, config.generator.model, None)
    inputs = [build_dir / "candidates.jsonl", build_dir / "review" / "state.json",
              build_dir / "review" / "late_candidates.jsonl"]
    release_record = {
        "spec_version": config.spec_version,
        "code_commit": provenance["code_commit"],
        "working_tree_clean": provenance["working_tree_clean"],
        "review": review.summary(),
        "inputs": {path.relative_to(build_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in inputs if path.exists()},
        "files": digests,
    }
    (build_dir / "release" / "release.json").write_text(
        json.dumps(release_record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print("final checks: all passed")
    for name, digest in sorted(digests.items()):
        print(f"{name}  {digest[:16]}")
    return 0


def run_reserves_command(config: BuildConfig, root: Path, args: argparse.Namespace) -> int:
    added = add_late_reserves(config, root, root, root / "builds" / args.build, args.stratum, args.run)
    print(f"{args.stratum}: {added} new reserves")
    return 0 if added else 1


def run_export_command(root: Path, args: argparse.Namespace) -> int:
    out = args.out.resolve()
    manifest = export_public(root, out, args.build, args.threshold_build, args.pool_runs, args.threshold_runs)
    total = sum(f["bytes"] for f in manifest["files"].values())
    print(f"{len(manifest['files'])} files, {total / 1e6:.1f} MB, in {out}")
    removed = ", ".join(f"{f['field']} ({f['count']})" for f in manifest["removed_fields"])
    print(f"fields removed: {removed or 'none'}")
    hits = scan(out)
    print(f"scan: {len(hits)} line(s) to check")
    for hit in hits:
        print(f"  {hit}")
    return 0


def run_rebuild_command(config: BuildConfig, root: Path, args: argparse.Namespace) -> int:
    problems = rebuild(config, root, args.raw.resolve(), args.out.resolve())
    for problem in problems:
        print(f"  {problem}")
    print("rebuild: identical to the release" if not problems else f"rebuild: {len(problems)} difference(s)")
    return 1 if problems else 0


def run_report(config: BuildConfig, root: Path, run_name: str) -> int:
    run_dir = root / "runs" / run_name
    records = RunLog(run_dir / "requests.jsonl").read()
    if not records:
        print(f"No records in {run_dir}")
        return 1
    lexicon = load_lexicon(root / "terms")
    for line in format_report(build_report(records, lexicon)):
        print(line)
    seed = config.base_seed + config.seed_offsets["hand_check"]
    sample_path = run_dir / "hand_check.tsv"
    sample = hand_check_sample(records, seed)
    sample_path.write_text("\n".join(sample) + "\n", encoding="utf-8", newline="\n")
    print(f"Hand-check sample: {sample_path}")
    return 0


def run_diagnostic(config: BuildConfig, root: Path, args: argparse.Namespace) -> int:
    overrides = {}
    if args.model:
        overrides["model"] = args.model
    if args.json_schema:
        overrides["use_json_schema"] = args.json_schema == "on"
    if args.reasoning_effort:
        overrides["reasoning_effort"] = "" if args.reasoning_effort == "none" else args.reasoning_effort
    run_name = f"diagnostic-{args.name}"
    model = overrides.get("model", config.generator.model)
    manifest = build_manifest(root, model, None, overrides)
    differences = check_or_write(root / "runs" / run_name / "manifest.json", manifest)
    if differences:
        print(f"Run {run_name!r} was started with different inputs and cannot be resumed:")
        for difference in differences:
            print(f"  {difference}")
        return 1

    pilot = {s.key: s for s in pilot_strata(build_strata(config), config.split_categories("train"))}
    strata = [
        Stratum(i, s.category, s.motion, s.distractor, s.kind, s.quota)
        for i, s in enumerate(pilot[key] for key in DIAGNOSTIC_KEYS)
    ]
    generation = make_generation(
        config, root, run_name, None, strata, diagnostic=True, overrides=overrides
    )
    print(f"settings: {json.dumps(overrides) if overrides else 'as configured'}")
    for line in run_pilot(generation, strata):
        print(line)
    if generation.abort.is_set():
        for reason in generation.abort_reason:
            print(f"  stopped: {reason}")
    print(f"Log: {generation.log.path}")
    return 1 if generation.abort.is_set() else 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m prompt_build")
    parser.add_argument("--root", type=Path, default=DATA_ROOT, help="data folder, if not this one")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="validate config, word lists and templates")
    commands.add_parser("plan", help="show strata and request counts without sending anything")
    pilot = commands.add_parser("pilot", help="send the pilot requests for one round")
    pilot.add_argument("--round", type=int, default=1, help="pilot round, from 1")
    diagnose = commands.add_parser("diagnose", help="send six fixed test requests")
    diagnose.add_argument("--name", required=True, help="diagnostic name, used in the log folder")
    diagnose.add_argument("--model", help="model ID, if not the configured one")
    diagnose.add_argument(
        "--json-schema", choices=("on", "off"), help="use strict JSON schema output"
    )
    diagnose.add_argument(
        "--reasoning-effort", help='reasoning effort to send, or "none" to leave it out'
    )
    build = commands.add_parser("build", help="clean the pool and assign splits (offline)")
    build.add_argument("--runs", nargs="+", default=["main"], help="run folders to read")
    build.add_argument("--name", default="main", help="build name, used as the output folder")
    build.add_argument(
        "--cosine-threshold", type=float, help="embedding threshold, if not the configured one"
    )
    check = commands.add_parser("threshold-check", help="apply the threshold-check decision rule")
    check.add_argument("--build", required=True, help="build whose list was read")
    check.add_argument("--same", type=int, nargs="*", default=[], help="pairs judged the same scene")
    review = commands.add_parser("review", help="run one step of the review")
    review.add_argument("--build", default="main", help="build under review")
    steps = review.add_subparsers(dest="step", required=True)
    steps.add_parser("start", help="open round 1")
    steps.add_parser("show", help="write the current round as a numbered list")
    steps.add_parser("prescreen", help="send the round to the pre-screen model and record its flags")
    steps.add_parser("sample", help="draw the seeded spot-check sample of unflagged items")
    steps.add_parser("sheet", help="write the author's reading list")
    decision = steps.add_parser("decide", help="record the author's spot-check result")
    decision.add_argument("--remove", nargs="*", default=[], help="sampled items to remove, NUMBER=REASON")
    steps.add_parser("close", help="close the round and replace its removals")
    steps.add_parser("status", help="show removals, agreement and coverage")
    refill = commands.add_parser("refill", help="send further requests for short strata")
    refill.add_argument("--build", default="main", help="build whose shortfalls to fill")
    refill.add_argument(
        "--runs", nargs="+", default=["main"], help="runs already logged, main run first"
    )
    refill.add_argument("--run", default="refill-1", help="name of the refill run")
    refill.add_argument("--stratum", help="one stratum to refill, instead of the build's shortfalls")
    refill.add_argument("--missing", type=int, default=1, help="prompts needed with --stratum")
    release = commands.add_parser("release", help="check the reviewed splits and write the files")
    release.add_argument("--build", default="main", help="reviewed build")
    reserves = commands.add_parser("reserves", help="add a refill run's candidates as reserves")
    reserves.add_argument("--build", default="main", help="build under review")
    reserves.add_argument("--stratum", required=True, help="stratum to add reserves to")
    reserves.add_argument("--run", required=True, help="refill run to screen")
    export = commands.add_parser("export-public", help="copy the rebuild inputs, without private fields")
    export.add_argument("out", type=Path, help="new or empty folder to write to")
    export.add_argument("--build", default="main", help="released build")
    export.add_argument("--threshold-build", default="threshold-check", help="build whose pairs were read")
    export.add_argument("--pool-runs", nargs="+", default=list(POOL_RUNS), help="runs the build read")
    export.add_argument(
        "--threshold-runs", nargs="+", default=list(THRESHOLD_RUNS), help="runs the threshold-check build read"
    )
    rebuild_ = commands.add_parser("rebuild", help="rebuild the released splits from the public raw data")
    rebuild_.add_argument("--raw", type=Path, required=True, help="folder written by export-public")
    rebuild_.add_argument("--out", type=Path, required=True, help="new or empty folder for the rebuild")
    report = commands.add_parser("report", help="score a run against the pilot criteria")
    report.add_argument("run", help="run folder name, for example pilot-1")
    generate = commands.add_parser("generate", help="send the main-run requests")
    generate.add_argument("--run", default="main", help="run name, used as the log folder")
    generate.add_argument(
        "--allow-dirty", action="store_true", help="run even with uncommitted changes"
    )
    args = parser.parse_args()

    root = args.root.resolve()
    problems = run_check(root)
    for problem in problems:
        print(problem)
    if problems:
        print(f"{len(problems)} problem(s) found.")
        return 1
    if args.command == "check":
        print("All checks passed.")
        return 0

    config = load_config(root / "config.toml")
    if args.command == "plan":
        show_plan(config)
        return 0
    if args.command == "report":
        return run_report(config, root, args.run)
    if args.command == "build":
        if args.cosine_threshold is not None:
            config = dataclasses.replace(
                config, build=dataclasses.replace(config.build, cosine_threshold=args.cosine_threshold)
            )
        return run_build_command(config, root, args.runs, args.name)
    if args.command == "threshold-check":
        return run_threshold_check(config, root, args)
    if args.command == "review":
        return run_review_command(config, root, args)
    if args.command == "refill":
        return run_refill_command(config, root, args)
    if args.command == "reserves":
        return run_reserves_command(config, root, args)
    if args.command == "release":
        return run_release_command(config, root, args.build)
    if args.command == "export-public":
        return run_export_command(root, args)
    if args.command == "rebuild":
        return run_rebuild_command(config, root, args)
    if args.command == "diagnose":
        return run_diagnostic(config, root, args)

    strata = build_strata(config)
    pilot_round = args.round if args.command == "pilot" else None
    run_name = f"pilot-{pilot_round}" if pilot_round is not None else args.run
    manifest = build_manifest(root, config.generator.model, pilot_round)
    if pilot_round is None and manifest["working_tree_clean"] is not True and not args.allow_dirty:
        print("The working tree has uncommitted changes, or git could not be run. Commit")
        print("first, so the run can be traced to a commit, or pass --allow-dirty.")
        return 1
    differences = check_or_write(root / "runs" / run_name / "manifest.json", manifest)
    if differences:
        print(f"Run {run_name!r} was started with different inputs and cannot be resumed:")
        for difference in differences:
            print(f"  {difference}")
        return 1

    if pilot_round is not None:
        strata = pilot_strata(strata, config.split_categories("train"))
    generation = make_generation(config, root, run_name, pilot_round, strata)
    if pilot_round is not None:
        lines = run_pilot(generation, strata)
    else:
        lines = run_main(generation, strata)
    for line in lines:
        print(line)
    if generation.abort.is_set():
        print("Stopped early:")
        for reason in generation.abort_reason:
            print(f"  {reason}")
    print(f"Log: {generation.log.path}")
    return 1 if generation.abort.is_set() else 0


if __name__ == "__main__":
    sys.exit(main())
