Share earning each reward part; 95% cluster bootstrap over prompts; answers in brackets.

| Split | Part | Denominator | Base | RL |
|---|---|---|---|---|
| test | valid SVG | all scored answers | 54% [51%, 58%] (960) | 97% [96%, 98%] (958) |
| test | not blank | all scored answers | 53% [50%, 57%] (960) | 97% [95%, 98%] (958) |
| test | moves | all scored answers | 25% [22%, 28%] (960) | 91% [89%, 93%] (958) |
| test | scene | answers that reached the judge | 6% [4%, 9%] (510) | 46% [42%, 50%] (926) |
| test | objects | answers that reached the judge | 14% [11%, 17%] (510) | 57% [54%, 61%] (926) |
| test | isometric | answers that reached the judge | 2% [1%, 3%] (510) | 10% [8%, 12%] (926) |
| test | composition | answers that reached the judge | 6% [4%, 8%] (510) | 53% [49%, 57%] (926) |
| test | no figure | answers that reached the judge | 90% [87%, 93%] (510) | 96% [94%, 98%] (926) |
| test | motion | answers that reached the judge, motion prompts | 25% [21%, 30%] (360) | 57% [53%, 62%] (661) |
| test_unseen | valid SVG | all scored answers | 50% [45%, 56%] (480) | 97% [96%, 98%] (480) |
| test_unseen | not blank | all scored answers | 49% [43%, 54%] (480) | 97% [95%, 98%] (480) |
| test_unseen | moves | all scored answers | 26% [22%, 31%] (480) | 92% [89%, 94%] (480) |
| test_unseen | scene | answers that reached the judge | 10% [6%, 15%] (233) | 59% [54%, 65%] (465) |
| test_unseen | objects | answers that reached the judge | 14% [9%, 19%] (233) | 64% [59%, 69%] (465) |
| test_unseen | isometric | answers that reached the judge | 3% [1%, 5%] (233) | 8% [6%, 11%] (465) |
| test_unseen | composition | answers that reached the judge | 6% [4%, 10%] (233) | 67% [62%, 72%] (465) |
| test_unseen | no figure | answers that reached the judge | 93% [89%, 96%] (233) | 98% [96%, 99%] (465) |
| test_unseen | motion | answers that reached the judge, motion prompts | 22% [15%, 30%] (160) | 63% [57%, 69%] (325) |

Distractor prompts: share with a figure present (J5).

| Split | Denominator | Base | RL |
|---|---|---|---|
| test | answers that reached the judge | 35% [24%, 45%] (78) | 19% [10%, 28%] (156) |
| test | all scored answers | 17% [11%, 23%] (160) | 18% [10%, 27%] (159) |
| test_unseen | answers that reached the judge | 31% [20%, 43%] (35) | 10% [4%, 18%] (79) |
| test_unseen | all scored answers | 14% [8%, 20%] (80) | 10% [4%, 18%] (80) |

Mean reward by category: in-scope prompts, mean over prompts of each prompt's mean over seeds.

| Category | Prompts | Base | RL |
|---|---|---|---|
| nature_islands | 25 | 0.128 | 0.613 |
| space | 100 | 0.090 | 0.571 |
| food_drink | 25 | 0.065 | 0.554 |
| single_objects | 25 | 0.075 | 0.549 |
| transport | 25 | 0.096 | 0.536 |
| home_interiors | 25 | 0.100 | 0.490 |
| workspaces | 25 | 0.073 | 0.469 |
| shops_food_service | 25 | 0.080 | 0.457 |
| streets_rooftops | 25 | 0.078 | 0.452 |
