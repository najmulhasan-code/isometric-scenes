# isometric-scenes

A verifiers v1 environment: the model writes one animated isometric SVG for a scene of places and objects.

- Code: [github.com/najmulhasan-code/isometric-scenes](https://github.com/najmulhasan-code/isometric-scenes)
- Model trained on it: [najmulhasan/qwen3.5-4b-isometric-scenes-lora](https://huggingface.co/najmulhasan/qwen3.5-4b-isometric-scenes-lora)
- Prompts: [najmulhasan/isometric-scenes](https://huggingface.co/datasets/najmulhasan/isometric-scenes)

## Scoring

A rollout is scored in four stages.

1. Code gates check that the answer is one well-formed, self-contained SVG of at most 64 KB, with no scripts, external references, event handlers or animation that starts or ends on an event, and with text only from quoted spans of the prompt (`svg_scoring/gates.py`).
2. The SVG is rendered in its own Prime sandbox with all outbound traffic denied. Animations are paused at four times, t = 0 and three times seeded from the SVG text, and the frames are measured in the browser (`isometric_scenes/render.py`, `svg_scoring/render_frames.py`).
3. Code checks on the frames test that no single colour covers more than 98% of the first frame (a blank frame scores 0), and that something moves locally, also between two of the later frames, without a full-frame flash (`svg_scoring/motion.py`).
4. A vision judge answers six yes/no items about the frames (`isometric_scenes/judge.py`). The default judge is `google/gemma-3-27b-it`, the training judge. Change it with `--taskset.task.judge.model`, or `env.taskset.task.judge.model` in a prime-rl config.

A failed gate, a failed render or a blank frame scores 0, and so does a person, animal, face or figure seen by the judge. Otherwise the reward is the mean of the judge's scene, objects, isometric and composition items, the motion check, and, for prompts that name a moving element, the judge's motion item.

Sandbox and judge failures raise an error instead of returning a score. `verifiers` records such rollouts with the error, and they were left out of training and evaluation. The analysis reports how many final-evaluation answers this affected.

## Setup

Install the environment from the Environments Hub with `prime env install najmulhasan-code/isometric-scenes`, or from a checkout of this folder with `uv pip install -e .`.

Rendering needs a sandbox image. Build it once from this folder with `prime images push <name>:<tag> --context . --dockerfile render_image/Dockerfile`, and pass the reference it prints as `RENDER_IMAGE` or as `--taskset.task.render_image`.

`PRIME_API_KEY` must allow inference and sandboxes. It is read on the evaluator host and never enters a render sandbox.

`isometric_scenes/data/` holds the released splits and their `SHA256SUMS`. Choose one with `--taskset.split`: `train`, `val`, `test` or `test_unseen`. The validation split is called `val` here and `validation` on the [Hugging Face dataset](https://huggingface.co/datasets/najmulhasan/isometric-scenes).

## Evaluate

The final evaluation used prime-rl's `eval` command, at prime-rl commit `35fcb52`, with the model served by vLLM. [`training/final_eval.toml`](https://github.com/najmulhasan-code/isometric-scenes/blob/main/training/final_eval.toml) is the full configuration. From a prime-rl checkout, a run looks like this:

```
uv run eval @ final_eval.toml --model <served model name> --client.base-url http://localhost:8000/v1
```

## Tests

The code checks run with the standard library on any machine:

```
python -m unittest discover -s tests -t .
```

`scripts/check_renderer.py` renders the fixtures in one sandbox and compares the decisions with `tests/fixtures/expected.json`.
