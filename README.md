# Isometric Scenes

RL post-training of [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) to write animated isometric SVG scenes of places and objects.

- Environment: [najmulhasan-code/isometric-scenes](https://app.primeintellect.ai/dashboard/environments/najmulhasan-code/isometric-scenes) on the Prime Intellect Environments Hub
- Model: [najmulhasan/qwen3.5-4b-isometric-scenes-lora](https://huggingface.co/najmulhasan/qwen3.5-4b-isometric-scenes-lora)
- Dataset: [najmulhasan/isometric-scenes](https://huggingface.co/datasets/najmulhasan/isometric-scenes)

https://github.com/user-attachments/assets/4612843a-91f4-4096-815a-66bd7148d020

Selected examples from the final evaluation, with the same prompt and seed for both models. The figure below is a random sample of test prompts without distractors.

![Base model and RL model on random test prompts](analysis/results/figures/examples.png)

## Results

Both models answered every test prompt 4 times, with seeds 0 to 3, at temperature 1.0 with up to 8,192 tokens and thinking off. The evaluation judge, `google/gemini-3.1-pro-preview`, a preview model that was not used in training, scored the answers at temperature 0 with up to 8,192 tokens. Differences have paired bootstrap intervals over prompts, shares have cluster bootstrap intervals over prompts, and the blind comparison has a Wilson interval.

| Prompts | Base | RL | Difference (95% CI) |
|---|---|---|---|
| Test (240) | 0.078 | 0.489 | +0.411 [0.387, 0.435] |
| Held-out category, `space` (one category, 120) | 0.082 | 0.561 | +0.479 [0.448, 0.511] |

The judge request failed for 2 of the 960 RL answers on test. Scoring them as 0 instead of leaving them out gives +0.409 [0.385, 0.432].

In a blind pairwise comparison by the author on 50 test prompts, the RL answer was preferred 40 times and the base answer 6 times, with 4 ties (win rate 0.87 excluding ties, Wilson 95% CI [0.74, 0.94]). On the 30 pairs where both answers reached the judge, the counts were 22, 5 and 3 (0.81 [0.63, 0.92]). The RL outputs have a recognisable style, so the blinding is weak.

The gain is larger under the training judge, `google/gemma-3-27b-it`. On the same pairs, Gemma scores the base model at 0.25 and the RL model at 0.81 on test, against 0.08 and 0.49 under Gemini, so its gain exceeds Gemini's by 0.15 [0.12, 0.18] on test and by 0.09 [0.04, 0.14] on `space`. Gemma is more lenient overall, and part of the larger gain may come from optimising against it.

The [model card](https://huggingface.co/najmulhasan/qwen3.5-4b-isometric-scenes-lora) gives the details, and [`analysis/`](analysis/) reproduces every evaluation number.

## Method

The model was trained with prime-rl and LoRA (rank 32) for 150 steps of 128 rollouts (16 prompts, 8 answers each) on 2 × H100. The reward first runs code checks: the answer must be one valid, safe SVG that renders to a non-blank frame. A code check for motion and the training judge's yes/no answers about the rendered frames are then averaged, and a figure seen by the judge scores 0. The [environment README](environments/isometric_scenes/README.md) gives the details. A checkpoint was saved every 15 steps, and the one with the best validation reward under the training judge was selected: step 150, the last, at 0.813 (next best 0.773 at step 120, on 88 validation prompts with 4 answers each).

## Reproduce

```
git clone https://github.com/najmulhasan-code/isometric-scenes
```

Each folder's README gives the commands.

1. [`prompt_data/`](prompt_data/) builds the prompt dataset.
2. [`judge_selection/`](judge_selection/) probes the candidate judges.
3. [`environments/isometric_scenes/`](environments/isometric_scenes/) defines the task and the reward.
4. [`training/`](training/) runs training and the final evaluation.
5. [`analysis/`](analysis/) computes the results and draws the figures.

## Limitations

- Few answers are isometric: 10% of the RL model's answers that reached the judge on test, up from 2%.
- Neither judge was checked against human ratings. The evaluation judge is a preview API model, so rescoring later may give different scores, and the gain is larger under the training judge, so part of it may be specific to that judge.
- The RL model did not clearly learn to leave out the figure on distractor prompts: on test, 27 base and 29 RL answers were judged to show one.
- Evidence of generalisation is limited: the held-out set is one category, and all prompts come from the same generator.
- There was one training run and one baseline, the base model, and the blind comparison had one rater, the author.

## Citation

```bibtex
@misc{hasan2026isometricscenes,
  author = {Najmul Hasan},
  title = {Isometric Scenes: RL post-training Qwen3.5-4B to write animated isometric SVG},
  year = {2026},
  url = {https://github.com/najmulhasan-code/isometric-scenes}
}
```

## License

The code and the adapter are released under [Apache-2.0](LICENSE). The dataset, including its copy in `environments/isometric_scenes/isometric_scenes/data/`, is released under CC BY 4.0.
