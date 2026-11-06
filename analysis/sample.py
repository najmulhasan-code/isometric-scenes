"""A random, unselected sample of base/trained pairs from the final evaluation.

Draws N in-scope `test` prompts (distractors excluded) with a fixed seed and takes the
seed-0 answer of each model, whatever its score. Shows readers typical output next to
any selected examples.

Each answer goes through the reward's code gates (svg_scoring/gates.py in
environments/isometric_scenes). Its SVG is saved only if the answer holds one SVG
document with no security failure (forbidden element, event handler, external
reference, DOCTYPE or ENTITY, oversize); other gate failures, such as malformed XML, do
not stop it being saved. index.json records every answer's gate failures.

Clears and writes OUT_DIR/random_sample/: the saved SVGs (<row_id>_<model>.svg),
index.json, and index.html, a side-by-side page that shows the SVGs through <img>, so no
script in an answer runs.
"""

import argparse
import base64
import html
import json
import random
import shutil
import sys

from common import MODELS, OUT_DIR, load_gates, longest_reply, trace_records

SECURITY = ("not exactly one", "larger than", "has a DOCTYPE", "forbidden element", "event handler",
            "external reference", "external url", "@import", "animation targets an href")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    gates = load_gates()

    answers, prompts = {}, {}
    for model in MODELS:
        for record in trace_records(model, 0):
            d = record["task"]["data"]
            if d["split"] == "test" and str(d["distractor"]) != "True":
                answers[(model, d["row_id"])] = longest_reply(record)
                prompts[d["row_id"]] = d["prompt"]
    row_ids = sorted(r for r in prompts if all((m, r) in answers for m in MODELS))
    if args.n > len(row_ids):
        sys.exit(f"--n {args.n} is larger than the {len(row_ids)} eligible prompts")
    chosen = random.Random(args.seed).sample(row_ids, args.n)

    out = OUT_DIR / "random_sample"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    index, cells = [], []
    for row_id in chosen:
        entry = {"row_id": row_id, "prompt": prompts[row_id]}
        pair = []
        for model in MODELS:
            gate = gates.check(answers[(model, row_id)], prompts[row_id])
            unsafe = gate.svg is None or any(f.startswith(SECURITY) for f in gate.failures)
            entry[model] = None if unsafe else f"{row_id}_{model}.svg"
            entry[f"{model}_gate_failures"] = gate.failures
            if unsafe:
                pair.append('<div class="none">Not shown: ' + html.escape("; ".join(gate.failures)) + "</div>")
                continue
            (out / entry[model]).write_text(gate.svg, encoding="utf-8")
            data = base64.b64encode(gate.svg.encode("utf-8")).decode("ascii")
            pair.append(f'<img alt="" src="data:image/svg+xml;base64,{data}">')
        index.append(entry)
        cells.append(f'<section><p>{html.escape(prompts[row_id])}</p><div class="pair">'
                     f'<figure><figcaption>Base</figcaption>{pair[0]}</figure>'
                     f'<figure><figcaption>RL</figcaption>{pair[1]}</figure></div></section>')
    (out / "index.json").write_text(json.dumps({"seed": args.seed, "n": args.n, "split": "test", "answers_seed": 0,
                                                "items": index}, indent=2) + "\n", encoding="utf-8")
    (out / "index.html").write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Random sample</title><style>'
        'body{margin:24px;font:16px/1.4 Georgia,serif;color:#222;background:#fff}section{margin:0 0 28px}'
        '.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;max-width:760px}figure{margin:0}'
        'figcaption{font-weight:600;margin-bottom:4px}img,.none{width:100%;aspect-ratio:1;border:1px solid #d9d9d9;'
        'object-fit:contain;display:flex;align-items:center;justify-content:center;color:#595959}</style></head><body>'
        + "".join(cells) + "</body></html>\n", encoding="utf-8")
    saved = sum(bool(e[m]) for e in index for m in MODELS)
    print(f"sampled {args.n} prompts (seed {args.seed}), saved {saved} of {2 * args.n} SVGs "
          f"into {out.relative_to(OUT_DIR.parent).as_posix()}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
