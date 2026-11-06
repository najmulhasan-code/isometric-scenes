# Isometric Scenes

RL post-training of [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) to write animated isometric SVG scenes of places and objects.

- Environment: [najmulhasan-code/isometric-scenes](https://app.primeintellect.ai/dashboard/environments/najmulhasan-code/isometric-scenes) on the Prime Intellect Environments Hub
- Model: [najmulhasan/qwen3.5-4b-isometric-scenes-lora](https://huggingface.co/najmulhasan/qwen3.5-4b-isometric-scenes-lora)
- Dataset: [najmulhasan/isometric-scenes](https://huggingface.co/datasets/najmulhasan/isometric-scenes)
  
https://github.com/user-attachments/assets/4612843a-91f4-4096-815a-66bd7148d020

![Base model and RL model on random test prompts](analysis/results/figures/examples.png)

## Results

Both models answered each test prompt 4 times with the same seeds. The evaluation judge, `google/gemini-3.1-pro-preview`, which was not used in training, scored the answers.

| Prompts | Base | RL | Difference (95% CI) |
|---|---|---|---|
| Test (240) | 0.078 | 0.489 | +0.411 [0.387, 0.435] |
| Held-out category, `space` (one category, 120) | 0.082 | 0.561 | +0.479 [0.448, 0.511] |

The author, one rater who built the project, compared 50 test pairs without knowing which model made which. RL was preferred 40 times, base 6, ties 4 (win rate 0.87, Wilson 95% [0.74, 0.94]).

The gain is larger under the training judge, `google/gemma-3-27b-it`, by 0.15 [0.12, 0.18] on test and 0.09 [0.04, 0.14] on `space`. Of the RL model's answers that rendered, 10% were judged isometric by Gemini. The [model card](https://huggingface.co/najmulhasan/qwen3.5-4b-isometric-scenes-lora) gives the details, and [`analysis/`](analysis/) reproduces every number.

## Method

The model was trained with prime-rl and LoRA (rank 32) for 150 steps of 128 rollouts (16 prompts, 8 answers each) on 2 × H100. The reward first runs code checks: the answer must be a valid, safe SVG that renders and moves. A vision judge then answers six yes/no questions about the rendered frames; see the [environment README](environments/isometric_scenes/README.md). The checkpoint was chosen on the validation split. It was the last one, step 150, where validation reward was still rising.

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

- Neither judge was checked against human labels, and the training judge scores both models far higher than the evaluation judge.
- Over all answers, RL did not reduce figures on distractor prompts (17% base, 18% RL on test). The drop among rendered answers comes from the larger number of RL answers that rendered.
- The blind comparison had one rater, the author, and the RL outputs share a recognisable style, so the blinding is weak.
- The held-out result covers one category, and the test prompts come from the same generator as the training prompts.
- There was a single training run, so the intervals cover prompt sampling only. Longer training and output diversity were not examined.

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

The code and the adapter are released under [Apache-2.0](LICENSE), and the dataset under CC BY 4.0.
