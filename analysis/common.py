"""Shared loading and statistics for the analysis scripts.

Every script reads the run's results from DATA_DIR, laid out as the results repository
on Hugging Face (see download.py), and writes to OUT_DIR. Both default to folders next
to this file and can be changed with the ANALYSIS_DATA and ANALYSIS_OUT environment
variables.
"""

import importlib.util
import io
import json
import math
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ANALYSIS_DATA", HERE / "data"))
OUT_DIR = Path(os.environ.get("ANALYSIS_OUT", HERE / "results"))
ENV_DIR = HERE.parent / "environments" / "isometric_scenes"

MODELS = ("base", "trained")
SPLITS = ("test", "test_unseen")
SEEDS = (0, 1, 2, 3)
EVAL_JUDGE = "google_gemini-3.1-pro-preview"
RESAMPLES, BOOT_SEED = 10_000, 0


def rescore_rows(judge: str = EVAL_JUDGE) -> list[dict]:
    """Every evaluation-judge row of the final evaluation: one per (model, split, row_id, seed)."""
    rows = []
    for model in MODELS:
        for seed in SEEDS:
            path = DATA_DIR / "final_eval" / model / f"seed{seed}" / "rescore" / f"{judge}.jsonl"
            with open(path, encoding="utf-8") as f:
                rows += [json.loads(line) for line in f if line.strip()]
    return rows


def rescore_summaries(judge: str = EVAL_JUDGE) -> list[dict]:
    """The rescore summary of each final-evaluation run (one per model and seed)."""
    return [json.loads((DATA_DIR / "final_eval" / m / f"seed{s}" / "rescore" / f"{judge}.summary.json")
                       .read_text(encoding="utf-8")) for m in MODELS for s in SEEDS]


def trace_records(model: str, seed: int):
    """The saved final-evaluation episodes of one model and seed (zstd JSONL chunks)."""
    import zstandard

    stream = DATA_DIR / "final_eval" / model / f"seed{seed}" / "monitors" / "file" / "traces" / "stream"
    for chunk in sorted(stream.glob("*.jsonl.zst")):
        with open(chunk, "rb") as f:
            text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8").read()
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)


def longest_reply(record) -> str:
    """The longest assistant message in an episode record, taken as the policy's answer.
    Judge calls, stored under "info", are skipped."""
    replies, stack = [], [record]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if item.get("role") == "assistant" and isinstance(item.get("content"), str):
                replies.append(item["content"])
            stack.extend(v for k, v in item.items() if k != "info")
        elif isinstance(item, list):
            stack.extend(item)
    return max(replies, key=len) if replies else ""


def load_gates():
    """svg_scoring/gates.py from the environment, loaded by path (standard library only),
    so the verifiers-dependent package is not imported."""
    spec = importlib.util.spec_from_file_location("gates", ENV_DIR / "svg_scoring" / "gates.py")
    gates = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gates)
    return gates


def bootstrap(items, stat, resamples=RESAMPLES, seed=BOOT_SEED):
    """Percentile 95% interval of stat over resamples of items, drawn with replacement.
    Resamples where stat returns None are left out."""
    rng = random.Random(seed)
    stats = [stat([rng.choice(items) for _ in items]) for _ in range(resamples)]
    stats = sorted(s for s in stats if s is not None)
    return [stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats))]]


def mean(values):
    return sum(values) / len(values)


def ratio(clusters):
    """Pooled share of (successes, trials) clusters, or None if there are no trials."""
    n = sum(c[1] for c in clusters)
    return sum(c[0] for c in clusters) / n if n else None


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def write_json(name: str, data) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
