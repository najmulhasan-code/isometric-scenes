"""Score the author's blind pairwise comparison against its answer key.

blind/picks.json holds the author's pick for each of the 50 items ("A", "B" or "tie"),
made without seeing which side was which; blind/answer_key.json says which model was
on each side. The 50 `test` prompts (distractors included) were sampled with seed 2026,
the sides ordered with seed 7, and the answers are the seed-0 final-evaluation answers
(blind_set.py rebuilds the key).

Reports the trained model's wins over wins plus losses, with a Wilson 95% interval, and
the ties: over all items, and over the items where both answers reached the judge
(passed the code checks and rendered not blank), from the seed-0 evaluation-judge rows.
Writes OUT_DIR/blind.json.
"""

import json
import sys

from common import HERE, MODELS, rescore_rows, wilson, write_json


def tally(items, picks):
    wins = sum(picks[str(i["n"])] in ("A", "B") and i[picks[str(i["n"])]] == "trained" for i in items)
    losses = sum(picks[str(i["n"])] in ("A", "B") and i[picks[str(i["n"])]] == "base" for i in items)
    decided = wins + losses
    return {"items": len(items), "trained_wins": wins, "base_wins": losses, "ties": len(items) - decided,
            "win_rate": round(wins / decided, 4) if decided else None, "ci95": wilson(wins, decided)}


def main() -> int:
    picks = json.loads((HERE / "blind" / "picks.json").read_text(encoding="utf-8"))
    key = json.loads((HERE / "blind" / "answer_key.json").read_text(encoding="utf-8"))["items"]
    if set(picks) != {str(i["n"]) for i in key}:
        sys.exit("picks.json items do not match answer_key.json items")
    if not set(picks.values()) <= {"A", "B", "tie"}:
        sys.exit("a pick is not A, B or tie")

    seed0 = {(r["model"], r["row_id"]): r for r in rescore_rows() if r["seed"] == 0 and r["split"] == "test"}

    def judged(row_id):
        rows = [seed0.get((m, row_id)) for m in MODELS]
        return all(r is not None and r["status"] == "scored" and (r.get("scoring") or {}).get("judge")
                   for r in rows)

    out = {"all": tally(key, picks), "both_judged": tally([i for i in key if judged(i["row_id"])], picks)}
    write_json("blind.json", out)
    for name, t in out.items():
        rate = f"{t['win_rate']:.3f} [{t['ci95'][0]:.3f}, {t['ci95'][1]:.3f}]" if t["win_rate"] is not None else "n/a"
        print(f"{name} ({t['items']} items): trained {t['trained_wins']}, base {t['base_wins']}, ties {t['ties']}; "
              f"win rate {rate} over {t['trained_wins'] + t['base_wins']} decided pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
