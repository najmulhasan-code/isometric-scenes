"""Generate animated isometric SVG scenes with a served model.

Sends each prompt to an OpenAI-compatible chat endpoint (for example vLLM serving
Qwen/Qwen3.5-4B with the trained LoRA adapter), with the system prompt and sampling
used in training and evaluation, and saves every answer's SVG.

  python training/generate.py --prompts prompts.txt --out scenes/ \
      --base-url http://localhost:8000/v1 --model trained --samples 4

prompts.txt holds one scene description per line. Writes OUT/answers.jsonl (one row per
answer) and OUT/<prompt number>_<sample>.svg for each answer that holds one SVG
document. Standard library only, plus the isometric-scenes package for the system
prompt and the SVG extraction.
"""

import argparse
import importlib.util
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENV = Path(__file__).resolve().parent.parent / "environments" / "isometric_scenes"


def load(name: str, path: Path):
    """Load one module file, without importing its package (which needs verifiers)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SYSTEM_PROMPT = load("scene_prompts", ENV / "isometric_scenes" / "prompts.py").SYSTEM_PROMPT
extract_svg = load("scene_gates", ENV / "svg_scoring" / "gates.py").extract_svg

# As in training and evaluation (training/rl.toml, training/final_eval.toml).
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": False}}


def ask(base_url: str, model: str, prompt: str, seed: int, timeout: float) -> str:
    body = {"model": model, "seed": seed, **SAMPLING,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]}
    request = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["choices"][0]["message"]["content"] or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="trained")
    parser.add_argument("--samples", type=int, default=4, help="answers per prompt, with seeds 0..samples-1")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    prompts = [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(n, prompt, seed) for n, prompt in enumerate(prompts, 1) for seed in range(args.samples)]

    def run(job):
        n, prompt, seed = job
        try:
            answer, error = ask(args.base_url, args.model, prompt, seed, args.timeout), None
        except Exception as exc:  # recorded per answer; the run goes on
            answer, error = "", f"{type(exc).__name__}: {exc}"
        svg = extract_svg(answer) if answer else None
        if svg is not None:
            (args.out / f"{n:02d}_{seed}.svg").write_text(svg, encoding="utf-8")
        return {"n": n, "prompt": prompt, "seed": seed, "model": args.model, "has_svg": svg is not None,
                "error": error, "answer": answer}

    with ThreadPoolExecutor(args.concurrency) as pool, open(args.out / "answers.jsonl", "w", encoding="utf-8") as log:
        done = 0
        for row in pool.map(run, jobs):
            log.write(json.dumps(row) + "\n")
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"{done} of {len(jobs)} answers", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
