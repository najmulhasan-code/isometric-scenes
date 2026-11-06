"""Headline results of the final evaluation, by the analysis method fixed before the run.

- Primary: the evaluation judge's (Gemini) reward on `test`, base vs trained; the same
  on `test_unseen`, the held-out category `space`.
- Pairs are matched by (split, row_id, seed); a pair is kept only if both models'
  answers were scored. A prompt's value is the mean over its kept seeds, and the
  difference gets a paired bootstrap over prompts (10,000 resamples, seed 0, 95%).
  Each model's mean gets a bootstrap interval over the same prompts.
- Sensitivity: unscored answers counted as 0.
- Judge comparison: the training judge (Gemma) and the evaluation judge on the pairs
  that both judges scored for both models. The judge gap is Gemma's gain minus Gemini's
  gain, with a paired bootstrap over prompts; a positive gap is a sign of reward hacking.
- Thinking: whether every final-evaluation run had thinking off (rescore summaries).

Writes OUT_DIR/headline.json and OUT_DIR/headline.md.
"""

import sys
from collections import defaultdict

from common import (BOOT_SEED, MODELS, OUT_DIR, RESAMPLES, SPLITS, bootstrap, mean, rescore_rows,
                    rescore_summaries, write_json)

GEMINI, GEMMA = "reward", "run_judge_reward"


def per_prompt(rows, split, fields, unscored_as_zero=False):
    """Per-prompt means of each field for each model, over the pairs kept.
    A pair is kept if every field is scored for both models."""
    by_key = {(r["model"], r["row_id"], r["seed"]): r for r in rows if r["split"] == split}
    keys = sorted({(r, s) for (_, r, s) in by_key})
    values = {m: {f: defaultdict(list) for f in fields} for m in MODELS}
    unscored = dict.fromkeys(MODELS, 0)
    kept = 0
    for row_id, seed in keys:
        pair = {m: by_key.get((m, row_id, seed)) for m in MODELS}
        ok = {m: r is not None and r["status"] == "scored" and all(r.get(f) is not None for f in fields)
              for m, r in pair.items()}
        for m in MODELS:
            unscored[m] += not ok[m]
        if not (unscored_as_zero or all(ok.values())):
            continue
        kept += 1
        for m, r in pair.items():
            for f in fields:
                values[m][f][row_id].append(r[f] if ok[m] else 0.0)
    prompts = sorted(values[MODELS[0]][fields[0]])
    means = {m: {f: [mean(values[m][f][p]) for p in prompts] for f in fields} for m in MODELS}
    return means, {"prompts": len(prompts), "pairs": len(keys) if unscored_as_zero else kept,
                   "unscored_answers": unscored}


def interval(values):
    return [round(v, 4) for v in bootstrap(values, mean)]


def summarise(rows, split, field, fields=None, **kw):
    means, counts = per_prompt(rows, split, fields or (field,), **kw)
    base, trained = means["base"][field], means["trained"][field]
    diffs = [t - b for b, t in zip(base, trained)]
    return {"split": split, **counts,
            "base": round(mean(base), 4), "base_ci95": interval(base),
            "trained": round(mean(trained), 4), "trained_ci95": interval(trained),
            "difference": round(mean(diffs), 4), "ci95": interval(diffs)}


def judge_gap(rows, split):
    means, counts = per_prompt(rows, split, (GEMINI, GEMMA))
    gain = {f: [t - b for b, t in zip(means["base"][f], means["trained"][f])] for f in (GEMINI, GEMMA)}
    gaps = [a - b for a, b in zip(gain[GEMMA], gain[GEMINI])]
    return {"split": split, **counts,
            "gemini_gain": round(mean(gain[GEMINI]), 4), "gemma_gain": round(mean(gain[GEMMA]), 4),
            "gap": round(mean(gaps), 4), "ci95": interval(gaps)}


def main() -> int:
    rows = rescore_rows()
    summaries = rescore_summaries()
    out = {
        "method": {"pairs": "(split, row_id, seed)", "bootstrap_resamples": RESAMPLES, "bootstrap_seed": BOOT_SEED,
                   "test_unseen": "held-out category: space"},
        "thinking_off_all_runs": not any(s["thinking_on"] for s in summaries),
        "runs_with_thinking_on": [f"{s['model']}/seed{s['seed']}" for s in summaries if s["thinking_on"]],
        "evaluation_judge": [summarise(rows, s, GEMINI) for s in SPLITS],
        "sensitivity_unscored_as_zero": [summarise(rows, s, GEMINI, unscored_as_zero=True) for s in SPLITS],
        "evaluation_judge_both_scored": [summarise(rows, s, GEMINI, fields=(GEMINI, GEMMA)) for s in SPLITS],
        "training_judge": [summarise(rows, s, GEMMA, fields=(GEMINI, GEMMA)) for s in SPLITS],
        "judge_gap": [judge_gap(rows, s) for s in SPLITS],
        "row_status": {m: dict(sorted({st: sum(r["model"] == m and r["status"] == st for r in rows)
                                       for st in {r["status"] for r in rows}}.items())) for m in MODELS},
    }
    write_json("headline.json", out)

    lines = ["| Judge | Split | Prompts | Pairs | Unscored (base / RL) | Base | RL | Difference (95% CI) |",
             "|---|---|---|---|---|---|---|---|"]
    for label, key in [("Evaluation (Gemini)", "evaluation_judge"),
                       ("Evaluation, unscored = 0", "sensitivity_unscored_as_zero"),
                       ("Evaluation (Gemini), both judges scored", "evaluation_judge_both_scored"),
                       ("Training (Gemma), both judges scored", "training_judge")]:
        for r in out[key]:
            u = r["unscored_answers"]
            lines.append(f"| {label} | {r['split']} | {r['prompts']} | {r['pairs']} | {u['base']} / {u['trained']} | "
                         f"{r['base']:.3f} | {r['trained']:.3f} | "
                         f"{r['difference']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] |")
    lines += ["", "Judge gap: Gemma gain minus Gemini gain, on pairs both judges scored for both models.", "",
              "| Split | Prompts | Pairs | Unscored (base / RL) | Gemini gain | Gemma gain | Gap (95% CI) |",
              "|---|---|---|---|---|---|---|"]
    for r in out["judge_gap"]:
        u = r["unscored_answers"]
        lines.append(f"| {r['split']} | {r['prompts']} | {r['pairs']} | {u['base']} / {u['trained']} | "
                     f"{r['gemini_gain']:+.3f} | {r['gemma_gain']:+.3f} | "
                     f"{r['gap']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] |")
    lines += ["", f"Thinking off in every final-evaluation run: {'yes' if out['thinking_off_all_runs'] else 'no'}."]
    (OUT_DIR / "headline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
