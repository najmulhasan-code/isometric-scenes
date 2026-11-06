# Training

The RL job for Qwen3.5-4B on the [isometric-scenes](https://app.primeintellect.ai/dashboard/environments/najmulhasan-code/isometric-scenes) environment, with prime-rl and LoRA. It runs unattended on a Prime GPU pod with 2 GPUs, and the pod shuts itself down when the job ends.

## Run

You need the [Prime CLI](https://github.com/PrimeIntellect-ai/prime-cli), logged in, with an SSH key set by `prime config set-ssh-key-path` and its public half added on the Prime dashboard. `launch.sh` also uses ssh, scp, tar and Python 3.

Create `training/secrets.env` with `PRIME_API_KEY`, `HF_TOKEN`, `HF_REPO`, `RENDER_IMAGE` and `WANDB_API_KEY` as `KEY=value` lines, and give it mode 600 (`chmod 600 training/secrets.env`). The job stops if the mode is different. `HF_REPO` must be a private Hugging Face model repository, which the job creates if needed and checks before training. `RENDER_IMAGE` is the renderer image reference that `prime images push` prints (see the [environment README](../environments/isometric_scenes/README.md)). The file is ignored by git. Then, from the repository root:

```
bash training/launch.sh
```

`launch.sh` shows the GPU offer and its price, and asks before creating the pod. On the pod, the job runs these steps:

1. A short smoke test of every stage.
2. Training (`rl.toml`): 150 steps, with validation every 15 steps.
3. Choice of the checkpoint with the best validation reward.
4. A final evaluation of the base and trained models on the test splits (`final_eval.toml`), with the answers rescored by the evaluation judge (`rescore.py`).
5. Upload of the adapter and results to `HF_REPO`.

`pod_setup.sh` installs [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) at commit `35fcb52` and checks that its bundled [verifiers](https://github.com/PrimeIntellect-ai/verifiers) is at commit `540f40f`.

## Generate with the trained model

Serve the base model with the [adapter](https://huggingface.co/najmulhasan/qwen3.5-4b-isometric-scenes-lora) in vLLM:

```
vllm serve Qwen/Qwen3.5-4B --max-model-len 10240 --enable-lora --max-lora-rank 32 --lora-modules trained=<adapter folder>
```

Then write one scene description per line in `prompts.txt` and run:

```
python training/generate.py --prompts prompts.txt --out scenes/ --model trained
```

The script uses the training system prompt and sampling, with thinking off, and saves each answer's SVG.

## Tests

```
python -m unittest discover -s training/tests
```
