| Judge | Split | Prompts | Pairs | Unscored (base / RL) | Base | RL | Difference (95% CI) |
|---|---|---|---|---|---|---|---|
| Evaluation (Gemini) | test | 240 | 958 | 0 / 2 | 0.078 | 0.489 | +0.411 [+0.387, +0.435] |
| Evaluation (Gemini) | test_unseen | 120 | 480 | 0 / 0 | 0.082 | 0.561 | +0.479 [+0.448, +0.511] |
| Evaluation, unscored = 0 | test | 240 | 960 | 0 / 2 | 0.079 | 0.488 | +0.409 [+0.385, +0.432] |
| Evaluation, unscored = 0 | test_unseen | 120 | 480 | 0 / 0 | 0.082 | 0.561 | +0.479 [+0.448, +0.511] |
| Evaluation (Gemini), both judges scored | test | 240 | 921 | 13 / 26 | 0.077 | 0.488 | +0.411 [+0.386, +0.435] |
| Evaluation (Gemini), both judges scored | test_unseen | 120 | 466 | 3 / 11 | 0.081 | 0.562 | +0.480 [+0.448, +0.513] |
| Training (Gemma), both judges scored | test | 240 | 921 | 13 / 26 | 0.248 | 0.812 | +0.564 [+0.538, +0.591] |
| Training (Gemma), both judges scored | test_unseen | 120 | 466 | 3 / 11 | 0.270 | 0.841 | +0.571 [+0.525, +0.617] |

Judge gap: Gemma gain minus Gemini gain, on pairs both judges scored for both models.

| Split | Prompts | Pairs | Unscored (base / RL) | Gemini gain | Gemma gain | Gap (95% CI) |
|---|---|---|---|---|---|---|
| test | 240 | 921 | 13 / 26 | +0.411 | +0.564 | +0.153 [+0.123, +0.184] |
| test_unseen | 120 | 466 | 3 / 11 | +0.480 | +0.571 | +0.091 [+0.042, +0.139] |

Thinking off in every final-evaluation run: yes.
