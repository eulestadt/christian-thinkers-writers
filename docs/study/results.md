# v3 results so far

There are still no evaluation scores. This page is not a copy of a Colab results table, because that table has not been written. Do not fill the blanks with v1 or v2 numbers. Do not add Mistral 24B or Qwen2-VL-72B rows. When scores exist, the reported metric is 8-token generation plus `parse_letter`, averaged across seeds 0, 1, and 2 at WiSE-FT λ=0.5, with each seed listed beside the mean, and with paired-bootstrap 95% intervals on Arm A minus Arm B and Arm A minus Arm C.

What has happened is training, not scoring. On the NVIDIA L4 session `ctw-v3`, weighted seed 0 finished in about 16.2 minutes (82 SFT steps and 82 R-DPO steps). Uniform seed 0 finished after that. Control seed 0 started at 9:06 p.m. Eastern on 23 November 2026 and is still in progress in the last log copy. Cumulative compute spent from the 506.04 starting balance is about 1.23 of the 240-unit cap; the meter read 504.81 at 1.94 units per hour. The step-by-step account is [`v3-run-log.md`](v3-run-log.md).

The two finished adapters live in the private study artifact store, not in this git tree. Each `adapter_model.safetensors` is about 93 MB, and Git LFS in this repository tracks only `docs/christian-writers-database.jsonl` and `docs/intrinsic-motivation-scores.sqlite`. The store paths are `docs/study/artifacts/adapters/llama32_3b/v3/weighted_seed0_final_lora/` and `docs/study/artifacts/adapters/llama32_3b/v3/uniform_seed0_final_lora/`. Control seed 0 has no downloaded adapter yet.

The four comparisons this study will report, when they exist, are: the Llama 3.2 3B Instruct base with no fine-tune (λ=0); the size-matched citation-control at λ=0.5; the uniform high-intrinsic arm at λ=0.5, which is the same 1,001 rows as the weighted arm with every weight set to 1; and the weighted high-intrinsic arm at λ=0.5, with `w = s / mean` and no clip. Arm A at λ=1 is a reference row, not a fifth experimental arm. Until Colab writes those numbers, every score cell below is pending.

| Arm | What it is | λ | Seed mean | Seed 0 | Seed 1 | Seed 2 |
|---|---|---:|---:|---:|---:|---:|
| Base | Llama 3.2 3B Instruct, no fine-tune | 0 | _pending_ | — | — | — |
| C, citation-control | size-matched, weights 1 | 0.5 | _pending_ | _pending_ | _pending_ | _pending_ |
| B, uniform high-intrinsic | same 1,001 rows as A, weights 1 | 0.5 | _pending_ | _pending_ | _pending_ | _pending_ |
| A, weighted high-intrinsic | `w = s / mean`, no clip | 0.5 | _pending_ | _pending_ | _pending_ | _pending_ |

The mix that these adapters were trained on is already measured, and is not an evaluation result: 1,001 CAI rows, 300 format-replay rows, Kish ESS 789.94, max weight 2.248. Style-drift and dogmatic-versus-reflective judgments on fifty held-out contemporary dilemmas, at λ=0.5, are also pending. JSON and CSVs from Colab should land under `colab/results/` and be copied next to this page when they exist.
