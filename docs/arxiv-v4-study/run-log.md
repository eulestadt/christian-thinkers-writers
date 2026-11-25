# v4 run log

Session name: `ctw-v4`. Account: `eulestadt@gmail.com`. Hardware: one L4. H100 was not used. Soft stop if the balance would fall below **255.92**. Starting balance **495.92**. Ending balance **478.58**. **CU spent: 17.34.** The session was stopped after the scores were copied off.

All times are UTC and US/Eastern.

## 2026-11-24 20:57 UTC / 15:57 EST — first L4

`colab new -s ctw-v4 --gpu L4` returned READY. Runtime `gpu-l4-s-ass1a0-3njjjhykzh74`. Rate **2.33 CU/hr** (the v3 L4 had reported 1.94). NVIDIA L4, 23034 MiB, confirmed 21:00 UTC / 16:00 EST.

## Generation

DeepSeek `deepseek-flash`, thinking off, only outside weekday UTC 01:00–04:00 and 06:00–10:00. Open and Sealed share the v3 treatment dilemmas. Neighbor uses the v3 control dilemmas. The 300 format-replay rows are the v3 rows, unchanged.

- 1,000 of 1,001 treatment rows kept. Row `36939f9f96df824e` kept copying the private trace and was dropped, not invented.
- Neighbor: 1,001 of 1,001.
- Open verdicts: 82 condemn, 918 allow. The same verdicts on Sealed. Neighbor: 231 condemn, 770 allow.
- Rejected: Open and Neighbor split shrug/sermon. Sealed rejects only the shrug.
- Second turns: 48 Open, 48 Sealed, 46 Neighbor. Two Neighbor rows were lost when concurrent writes collided. Held-out second turns: 36, not trained on.
- Logged generation calls about 2,132, roughly 1.78M prompt tokens and 0.82M completion tokens. A later labeling pass on the 684 held-out replies used about 198k prompt tokens and 4k completion tokens. Account balance **10.86 → 9.92 USD**.

## Re-score of the v3 adapters

Started 21:25 UTC / 16:25 EST on the same L4. The nine v3 adapters were read, not written. Output only under `results/llama32_3b/v3_rescored`. Fixes: the ETHICS question matches the subset; a reply with no digit is no-answer; balanced accuracy and the positive-label rate; MMLU options rotated four ways; TruthfulQA not scored. Finished 22:07 UTC / 17:07 EST. 32,000 scored generations. The commonsense gain survived. The old drops on excuses and justice reversed once the question matched the key. The virtue rise did not survive. The three old arms stayed tied.

## Training

Recipe: QLoRA r=16 on seven projections, one SFT epoch at 1e-4, then R-DPO at 5e-6, α=0.01, β=0.1. Seeds 0, 1, 2. Weights all 1. Writes only under `adapters/llama32_3b/v4`.

Open seed 0 finished 22:35 UTC. Sealed seed 0 at 22:51. Neighbor seed 0 at 23:08. Open seed 1 at 23:26. Sealed seed 1 at 23:43. Each adapter was copied to the store as it finished.

## 2026-11-25 00:19 UTC / 2026-11-24 19:19 EST — first machine gone

The L4 disappeared after sealed seed 1. Five adapters were already local: open 0, sealed 0, neighbor 0, open 1, sealed 1. A stray SSH client then created an empty CPU session under the same name. That CPU session was stopped.

## 2026-11-25 00:27 UTC / 19:27 EST — second L4

Runtime `gpu-l4-s-ass1a1-1r399ixi0szmf`. The five saved adapters were put back so training would skip them. Neighbor seed 1 finished 00:48 UTC and was copied off. That machine also disappeared before seed 2 was fully copied.

## 2026-11-25 01:45 UTC / 20:45 EST — third L4

Runtime `gpu-l4-s-ass1a1-2gafhdxag13zh`. Six adapters restored. Open seed 2, sealed seed 2, and neighbor seed 2 were trained and copied off. All nine weights are saved.

## Eval

A fourth L4, `gpu-l4-s-ass1a0-x86wnwya5b13`, started 04:02 UTC / 23:02 EST the previous evening, after an earlier eval machine died mid-run. Scoring resumed from the 12,950 already-finished item groups (the backup had 23,150 complete rows). Full eval finished **05:15 UTC / 00:15 EST**. 61,484 generative rows and 950 style judgments were copied off, and the session was stopped. Final balance **478.58**. Nothing was running.

## What was not done

No fifth seed. No model larger than 3B. No BBQ. TruthfulQA dropped. The three earlier papers were not edited. Nothing was pushed to GitHub. v1, v2, and v3 files were not rewritten.
