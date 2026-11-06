"""What improved: the share of answers that earn each part of the reward, base vs trained,
J5 (figure present) on distractor prompts, and the mean reward by prompt category.
Evaluation judge (Gemini), all four seeds, scored answers only (unscored answers have
no scoring record and are counted in the output).

Reward parts, as in isometric_scenes/taskset.py (SceneTask.scene), and their denominators:
  code checks   valid SVG (passes the gates), renders not blank, animates (moves between
                frames); each one requires the ones before it. Denominator: all answers.
  judge         scene, objects, isometric, composition, no figure; and motion, for motion
                prompts only. Denominator: answers that reached the judge (passed the gates
                and rendered not blank).
Distractor prompts: the share of judged answers with a figure present, and the share of
all answers, per model.
Categories: in-scope prompts only; a prompt's value is its mean reward over seeds, and a
category's value is the mean over its prompts.

Intervals are 95% cluster bootstraps over prompts (each prompt's 4 seeds stay together),
10,000 resamples, seed 0. Writes OUT_DIR/per_item.json and .md.
"""

import sys
from collections import defaultdict

from common import MODELS, OUT_DIR, SPLITS, bootstrap, mean, ratio, rescore_rows, write_json


def judge(s):
    return s.get("judge") or {}


def renders(s):
    return not s.get("gate_failures") and s.get("not_blank") is True


def judged(s):
    return renders(s) and bool(s.get("judge"))


ALL = "all scored answers"
JUDGED = "answers that reached the judge"
JUDGED_MOTION = "answers that reached the judge, motion prompts"

# name, kind, denominator, applies(row), passes(scoring)
PARTS = [
    ("valid SVG", "code", ALL, lambda r: True, lambda s: not s.get("gate_failures")),
    ("not blank", "code", ALL, lambda r: True, renders),
    ("moves", "code", ALL, lambda r: True, lambda s: renders(s) and s.get("moves") is True),
    *[(name, "judge", JUDGED, lambda r: judged(r["scoring"]), test) for name, test in [
        ("scene", lambda s: judge(s).get("scene") is True),
        ("objects", lambda s: judge(s).get("objects") is True),
        ("isometric", lambda s: judge(s).get("isometric") is True),
        ("composition", lambda s: judge(s).get("composition") is True),
        ("no figure", lambda s: judge(s).get("figure_present") is False)]],
    ("motion", "judge", JUDGED_MOTION, lambda r: judged(r["scoring"]) and r["motion_in_prompt"],
     lambda s: judge(s).get("motion") is True),
]


def share(rows, applies, passes):
    """Pooled share over rows, with a cluster bootstrap over prompts."""
    clusters = defaultdict(lambda: [0, 0])
    for r in rows:
        c = clusters[r["row_id"]]
        if applies(r):
            c[0] += bool(passes(r["scoring"]))
            c[1] += 1
    clusters = [tuple(c) for c in clusters.values()]
    k, n = sum(c[0] for c in clusters), sum(c[1] for c in clusters)
    if n == 0:
        return {"share": None, "ci95": [None, None], "k": 0, "n": 0}
    return {"share": round(k / n, 4), "ci95": [round(v, 4) for v in bootstrap(clusters, ratio)], "k": k, "n": n,
            "prompts": sum(c[1] > 0 for c in clusters)}


def main() -> int:
    every = rescore_rows()
    rows = [r for r in every if r["status"] == "scored" and r.get("scoring")]
    pool = {(m, s): [r for r in rows if r["model"] == m and r["split"] == s] for m in MODELS for s in SPLITS}

    parts = []
    for split in SPLITS:
        for name, kind, denominator, applies, passes in PARTS:
            parts.append({"split": split, "part": name, "kind": kind, "denominator": denominator,
                          **{m: share(pool[(m, split)], applies, passes) for m in MODELS}})

    figure = lambda s: judge(s).get("figure_present") is True  # noqa: E731
    distractor = []
    for split in SPLITS:
        for denominator, applies in ((JUDGED, lambda r: judged(r["scoring"])), (ALL, lambda r: True)):
            distractor.append({"split": split, "denominator": denominator,
                               **{m: share([r for r in pool[(m, split)] if r["distractor"]], applies, figure)
                                  for m in MODELS}})

    prompt_rewards = defaultdict(list)
    for r in rows:
        if not r["distractor"]:
            prompt_rewards[(r["category"], r["model"], r["row_id"])].append(r["reward"])
    categories = defaultdict(lambda: {m: [] for m in MODELS})
    for (category, model, _), values in prompt_rewards.items():
        categories[category][model].append(mean(values))
    by_category = sorted(({"category": c, **{m: round(mean(v[m]), 4) for m in MODELS}, "prompts": len(v["trained"])}
                          for c, v in categories.items()), key=lambda e: -e["trained"])

    write_json("per_item.json", {
        "answers": {m: sum(r["model"] == m for r in every) for m in MODELS},
        "unscored_answers": {m: sum(r["model"] == m for r in every) - sum(r["model"] == m for r in rows)
                             for m in MODELS},
        "intervals": "cluster bootstrap over prompts, 10,000 resamples, seed 0, 95%",
        "parts": parts, "distractor_figure_present": distractor,
        "by_category": by_category, "by_category_method": "in-scope prompts; mean over prompts of the per-prompt mean"})

    def cell(e):
        return f"{e['share']:.0%} [{e['ci95'][0]:.0%}, {e['ci95'][1]:.0%}] ({e['n']})"

    lines = ["Share earning each reward part; 95% cluster bootstrap over prompts; answers in brackets.", "",
             "| Split | Part | Denominator | Base | RL |", "|---|---|---|---|---|"]
    lines += [f"| {e['split']} | {e['part']} | {e['denominator']} | {cell(e['base'])} | {cell(e['trained'])} |"
              for e in parts]
    lines += ["", "Distractor prompts: share with a figure present (J5).", "",
              "| Split | Denominator | Base | RL |", "|---|---|---|---|"]
    lines += [f"| {e['split']} | {e['denominator']} | {cell(e['base'])} | {cell(e['trained'])} |" for e in distractor]
    lines += ["", "Mean reward by category: in-scope prompts, mean over prompts of each prompt's mean over seeds.", "",
              "| Category | Prompts | Base | RL |", "|---|---|---|---|"]
    lines += [f"| {e['category']} | {e['prompts']} | {e['base']:.3f} | {e['trained']:.3f} |" for e in by_category]
    (OUT_DIR / "per_item.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
