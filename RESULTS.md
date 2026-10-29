# Marvijo Web Search results

Exact-match plans (URL and field list both right), scored by the JavaScript runtime on the laptop CPU.
The shipped model is chosen on the independent dev set only. The blind test was committed before any v2 result and is reported, never tuned on.

Checkpoint names for the current model: `marvijo-web-search-1.1-2.7m` (lane `v4-small`),
`marvijo-web-search-1.1-copy-2.7m` (`v4-copy`), `marvijo-web-search-1.1-mix-2.7m` (`v4-mix`) and
`marvijo-web-search-1.1-mix2-2.7m` (`v4-mix2`, shipped, Hugging Face repo id
`marvijo/marvijo-web-search-1.1-mix2-2.7m`). Lanes `v1` to `v3` predate the 1.1 name and keep
their old labels in this table.

| Model | Data | Params | Train min | Hand-written (50) | Independent dev (120) | Final blind (150) | Blind ms/request |
|---|---|---|---|---|---|---|---|
| Regex baseline | - | - | - | 100.0% | 90.8% | 85.3% | - |
| small | v1 | 2.68M | 55 | 96.0% | 64.2% | 60.0% | 1414 |
| v2-small | v2 | 2.68M | 55 | 96.0% | 77.5% | 73.3% | 1147 |
| v2-wide | v2 | 6.35M | 60 | 94.0% | 74.2% | 67.3% | 3751 |
| v3-small | v3 | 2.68M | 55 | 100.0% | 95.8% | 92.7% | 1652 |

## v4: the agent (step-by-step replay)

v4 writes one tool call at a time and reads each tool result. Scored by `evaluate.py` with teacher forcing: at every step the model sees the request plus the gold calls and results so far, and must write the next call. Not comparable with the v3 table above.

| Model | Data | Params | Train min | Set | Episode exact | Step acc | First call | JSON valid | ms/step (PyTorch, 4 vCPU) |
|---|---|---|---|---|---|---|---|---|---|
| v4-small | v4 (200k episodes) | 2.68M | 90 | val (500) | 90.2% | 96.8% | 95.6% | 99.9% | 108 |
| v4-small | v4 | 2.68M | 90 | unseen topics (600) | 38.8% | 79.4% | 41.7% | 99.8% | 107 |
| v4-small | v4 | 2.68M | 90 | unseen phrasings (600) | 75.5% | 92.2% | 83.7% | 99.9% | 97 |
| v4-copy | v4 --copy 0.4 | 2.68M | 90 | val (500) | 87.6% | 96.1% | 93.8% | 99.9% | 94 |
| v4-copy | v4 --copy 0.4 | 2.68M | 90 | unseen topics (600) | 68.2% | 89.6% | 74.7% | 99.8% | 95 |
| v4-copy | v4 --copy 0.4 | 2.68M | 90 | unseen phrasings (600) | 65.3% | 88.5% | 73.5% | 99.7% | 88 |
| v4-mix | v4 --copy 0.4 --mix 0.5 | 2.68M | 90 | val (500) | 84.2% | 95.0% | 91.6% | 99.9% | 87 |
| v4-mix | v4 --copy 0.4 --mix 0.5 | 2.68M | 90 | unseen topics (600) | 66.7% | 89.2% | 74.0% | 99.9% | 95 |
| v4-mix | v4 --copy 0.4 --mix 0.5 | 2.68M | 90 | unseen phrasings (600) | 77.5% | 93.2% | 81.7% | 99.9% | 98 |
| **v4-mix2** | v4 --copy 0.3 --mix 0.5 --typo 0.15 | 2.68M | 150 | val (500) | **94.0%** | **98.0%** | **95.8%** | 99.9% | 99 |
| **v4-mix2** | v4 --copy 0.3 --mix 0.5 --typo 0.15 | 2.68M | 150 | unseen topics (600) | **70.2%** | **90.5%** | 72.2% | 99.9% | 94 |
| **v4-mix2** | v4 --copy 0.3 --mix 0.5 --typo 0.15 | 2.68M | 150 | unseen phrasings (600) | **82.8%** | **94.6%** | **88.3%** | 99.9% | 91 |

Final val loss 0.00283 (step 2805, under one epoch, bf16). JS runtime parity: PARITY OK.
Main weakness: unseen topics. The model does not copy words it never saw in training; it substitutes known ones
("polars" -> `pypi.org/project/python/`, "nvidia reviews" -> `"q":"bernet mes"`). Training topics come from small closed lists.

v4-copy rows are scored on the v4-small test files (`runs/v4-copy/eval_on_v4-small_sets.json`), so both models see the
same rows. On its own lane's files it scores 86.8 / 68.0 / 65.3 and v4-small scores 89.8 / 38.3 / 75.5 there
(`runs/v4-small/eval_on_v4-copy_sets.json`): the same picture. Final val loss 0.00492 (step 3140, bf16). PARITY OK.

`--copy 0.4` invents 40% of training topic, repo, package and subreddit names. Copying unseen topics improves by
about 30 points, but unseen phrasings drop about 10: the model now treats any word as a possible topic, so
filler from templates it never saw leaks into the query ("please bring me space x updates" -> `"q":"bring me space x"`,
"i'm curious about psl schedule" -> `"q":"curisous about psl schedule"`). The likely fix is more varied filler
phrasing around invented topics, so the model learns where a topic starts and ends.

`--mix 0.5` phrases half of the training searches with generated filler (v4-mix, scored on the v4-small files,
`runs/v4-mix/eval_on_v4-small_sets.json`). Unseen phrasings recover to 77.5%, above v4-small, and unseen topics stay
near v4-copy. Val drops to 84.2%: the model now copies typos of known words into the query ("lsibon", "fixturse",
"upblic") because an unknown string may just as well be a new name. On its own lane files: 87.6 / 66.5 / 77.5.

`--typo 0.15` forces a typo into the real topic name of 15% of training searches and recipe requests (the gold
keeps the right spelling), so correcting known words keeps a strong signal next to copying invented ones.
v4-mix2 also trains for 150 minutes instead of 90 (4916 steps, into the second epoch; final val loss 0.00164,
train loss about 0.001, no sign of overfitting). Both changes moved at once, so part of the gain is the longer run.
On its own lane files: 94.6 / 70.2 / 82.8. PARITY OK.

### v4 champion: v4-mix2

Chosen on val episode exact plus the mean of the two unseen sets, all on the v4-small test files:

| Model | Val | Unseen topics | Unseen phrasings | Val + mean(unseen) |
|---|---|---|---|---|
| v4-small | 90.2% | 38.8% | 75.5% | 1.474 |
| v4-copy | 87.6% | 68.2% | 65.3% | 1.544 |
| v4-mix | 84.2% | 66.7% | 77.5% | 1.563 |
| **v4-mix2** | **94.0%** | **70.2%** | **82.8%** | **1.705** |

v4-mix2 is the first model at or above the best earlier score on every set. Remaining failures: typos in
words it never saw ("anthropic blog posst" -> `"q":"natheric blog"`, "polars" -> `polart`), field lists with three
or more fields, and an extra field on unseen phrasings ("harvest h1-h3" -> `["headings","links"]`).
