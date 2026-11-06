"""Rebuild the blind comparison's answer key and check it against blind/answer_key.json.

The set is 50 `test` prompts (distractors included) sampled with seed 2026 from the
sorted row_ids that have a seed-0 final-evaluation answer from both models. Items are
taken in sampled order; for each, Random(7).random() < 0.5 puts the trained answer on
side A. Writes OUT_DIR/blind_key.json and exits with an error if it differs from the
key used for the picks.
"""

import json
import random
import sys

from common import HERE, MODELS, trace_records, write_json

SAMPLE_SEED, ORDER_SEED, N = 2026, 7, 50


def main() -> int:
    answered = {m: {r["task"]["data"]["row_id"] for r in trace_records(m, 0) if r["task"]["data"]["split"] == "test"}
                for m in MODELS}
    ids = sorted(answered["base"] & answered["trained"])
    chosen = random.Random(SAMPLE_SEED).sample(ids, N)
    order = random.Random(ORDER_SEED)
    items = []
    for n, row_id in enumerate(chosen, 1):
        trained_a = order.random() < 0.5
        items.append({"n": n, "row_id": row_id, "A": "trained" if trained_a else "base",
                      "B": "base" if trained_a else "trained"})
    key = {"sample_seed": SAMPLE_SEED, "order_seed": ORDER_SEED, "answers_seed": 0, "items": items}
    write_json("blind_key.json", key)

    used = json.loads((HERE / "blind" / "answer_key.json").read_text(encoding="utf-8"))
    if used != key:
        sys.exit("rebuilt key differs from blind/answer_key.json")
    print(f"rebuilt {N} items ({sum(i['A'] == 'trained' for i in items)} with the trained answer on A); "
          "matches blind/answer_key.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
