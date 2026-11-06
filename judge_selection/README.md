# Judge selection

`probe.py` checks which candidate vision judges can read several images in one request. Each candidate gets four PNG squares of known colours and must name the colours in order, and the script also records whether the API accepts a JSON schema for the answer.

Run it from the repository root with `PRIME_API_KEY` set, either in the environment or in a `.env` file at the root:

```
python judge_selection/probe.py
```

Every request and reply is appended to `judge_selection/runs/probe/results.jsonl`, which is ignored by git.
