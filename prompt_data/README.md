# Prompt data

This folder holds the code that builds the prompt dataset for isometric scenes: the text prompts in `train`, `val`, `test` and `test_unseen`. The dataset card describes the rules, the splits and the known limitations: https://huggingface.co/datasets/najmulhasan/isometric-scenes

The prompts were written by a language model, `openai/gpt-oss-120b`, and every request and reply was logged. Every step after that is code: rule checks, deduplication, the split assignment, a review in rounds by a review model, with the author's spot check of 50 seeded items in the first round, and the release checks. The released files can therefore be rebuilt exactly from the logs and the recorded review decisions.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). From this folder:

```
uv sync
```

Deduplication uses the sentence encoder `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, on CPU. The code runs it with the Hugging Face hub offline, so download it once:

```
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41')"
```

Only the steps that call a model need an API key. They read `PRIME_API_KEY` from the environment, or from a `.env` file at the repository root. `PRIME_TEAM_ID` is optional; set it to send the requests under a Prime team account.

## Rebuild the released splits

The raw data is published with the dataset in the `raw/` folder. It holds the generator logs of the main run and the refill runs, the review model's replies in each round, the author's decisions (`review/decisions.json`), and the published checksums. Provider request IDs were removed from the logs, and no step reads them.

```
uv run python -m prompt_build rebuild --raw path/to/raw --out path/to/rebuild
```

This runs offline and repeats every step after generation, then compares the result with the published checksums. It prints `rebuild: identical to the release` when every file matches. The split files are written to `path/to/rebuild/main/release/`.

Recorded rather than recomputed: the generator's replies, the review model's replies, and the author's decisions (no pairs judged the same scene in the threshold check, and no removals in the round 1 spot check).

The raw folder is written from the full working records by:

```
uv run python -m prompt_build export-public path/to/raw
```

## Generate from scratch

These steps send requests and need `PRIME_API_KEY`. Model replies are not byte-reproducible, even with fixed seeds, so a new run gives a different pool and different splits; it follows the same method.

```
uv run python -m prompt_build check
uv run python -m prompt_build plan
uv run python -m prompt_build pilot --round 1
uv run python -m prompt_build report pilot-1
uv run python -m prompt_build generate
uv run python -m prompt_build build --runs main --name threshold-check --cosine-threshold 0.85
uv run python -m prompt_build threshold-check --build threshold-check --same
uv run python -m prompt_build build --runs main
uv run python -m prompt_build refill --build main --run refill-1
uv run python -m prompt_build build --runs main refill-1
```

`threshold-check` takes the numbers of the pairs a reader judged to be the same scene, from `builds/threshold-check/threshold_check.txt`. Refill until the build reports no short strata. Then review, one round at a time:

```
uv run python -m prompt_build review start
uv run python -m prompt_build review prescreen
uv run python -m prompt_build review sample
uv run python -m prompt_build review sheet
uv run python -m prompt_build review decide --remove 17=R8
uv run python -m prompt_build review close
```

`decide` lists the items of the spot-check sheet that the reader removes, each with its rule. The spot check has 50 items in round 1 and none in later rounds. Repeat `prescreen`, `sample`, `decide` and `close` until `close` reports that review is complete. If a stratum runs out of reserves, refill it and add the new candidates:

```
uv run python -m prompt_build refill --stratum in_scope/space/motion --missing 10 --runs main refill-1 --run refill-2
uv run python -m prompt_build reserves --stratum in_scope/space/motion --run refill-2
```

Finally, check the splits and write the files:

```
uv run python -m prompt_build release
```

Logs go to `runs/`, builds to `builds/`. The networked steps refuse to start with uncommitted changes, so each run can be traced to a commit.

## Tests

```
uv run python -m unittest discover -s tests -t .
```
