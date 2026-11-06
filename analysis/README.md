# Analysis

Scripts that turn the training run's saved outputs into the reported results and figures.

## Setup

Python 3.12:

```
pip install -r requirements.txt
playwright install chromium
```

`render.py` needs a Chromium build; instead of `playwright install chromium` you can set `PLAYWRIGHT_CHANNEL=msedge` to use an installed Edge. `sample.py` and `render.py` use the reward's code gates, `environments/isometric_scenes/svg_scoring/gates.py` in this repository, so run them from a full checkout.

## Commands

Run from this folder, in order:

```
python download.py
python results.py
python per_item.py
python blind.py
python blind_set.py
python sample.py
python render.py
python figures.py
```

`download.py` fetches the run's results from the [model repository](https://huggingface.co/najmulhasan/qwen3.5-4b-isometric-scenes-lora) into `data/` at a pinned revision. `RESULTS_REPO` and `RESULTS_REVISION` override the repository and the revision.

## Outputs

All outputs are written to `results/`.

| File | Contents |
|---|---|
| `headline.json`, `headline.md` | Mean reward of the base and the trained model on `test` and `test_unseen` (the held-out category `space`), with paired bootstrap 95% intervals; the unscored-as-0 sensitivity; both judges on the pairs they both scored, and the judge gap (training judge's gain minus the evaluation judge's gain); whether every run had thinking off |
| `per_item.json`, `per_item.md` | Share of answers earning each part of the reward (code checks over all answers, judge parts over answers that reached the judge), figures drawn on distractor prompts, and mean reward by category |
| `blind.json` | Blind pairwise comparison on 50 `test` prompts: wins, losses, ties and a Wilson 95% interval, over all items and over items where both answers reached the judge |
| `blind_key.json` | The blind set's answer key, rebuilt by `blind_set.py` and checked against `blind/answer_key.json` |
| `random_sample/` | 12 unselected in-scope `test` pairs; open `index.html` to view them side by side rather than the raw SVGs |
| `random_sample/tiles/` | Each sampled SVG rendered at 1 second into its animation, offline |
| `figures/` | `validation`, `headline`, `parts` and `examples`, as PNG |

## Method

The final evaluation used 360 prompts not used in training (`test` and `test_unseen`). Both models answered each 4 times with fixed seeds 0 to 3, and the evaluation judge, `google/gemini-3.1-pro-preview`, scored the answers.

A pair is the same prompt and seed for both models, and it is kept only if both answers were scored. Comparisons between judges use only pairs that both judges scored for both models.

A prompt's value is its mean over seeds. Differences get a paired bootstrap over prompts, and shares a cluster bootstrap over prompts, so a prompt's 4 answers stay together. Both use 10,000 resamples with seed 0.

`blind/` holds the 50 picks of the blind comparison and its answer key, which was kept unseen until all picks were made. The blind set includes distractor prompts, while `sample.py` excludes them.

In the random sample, an SVG is saved only if it passes the gates' security checks, and `index.json` lists every answer's gate failures.
