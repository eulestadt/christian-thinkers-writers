# v3 results so far

Evaluation scores are still pending. When they exist, the reported metric is 8-token generation plus `parse_letter`, averaged across seeds 0, 1, and 2 at WiSE-FT λ=0.5. The mean, not the best seed, is the result because seed scatter can mimic a treatment effect (Dodge et al., 2020). λ=0.5 is a blend of adapter and base, not a hedge (Wortsman et al., 2022). Paired-bootstrap 95% intervals (Dror et al., 2018) will cover Arm A minus Arm B (weights, filter held fixed) and Arm A minus Arm C (corpus, size held fixed).

What has happened is training. On the NVIDIA L4 session `ctw-v3`, weighted seed 0 finished in about 16.2 minutes (82 SFT steps and 82 R-DPO steps). Uniform seed 0 finished after that: same rows, weights 1, so a later A-minus-B contrast can be read as the Byrd and Lipton weight effect rather than as “we also changed which dialogues were in the mix.” Control seed 0 started at 9:06 p.m. Eastern on 23 November 2026 and is still in progress in the last log copy. Cumulative compute spent from the 506.04 starting balance is about 1.23 of the 240-unit cap; the meter read 504.81 at 1.94 units per hour. The step-by-step account is [`v3-run-log.md`](v3-run-log.md).

The two finished adapters live in the private study artifact store. Each `adapter_model.safetensors` is about 93 MB, and Git LFS in this repository tracks the census jsonl and the scores sqlite. The store paths are `docs/study/artifacts/adapters/llama32_3b/v3/weighted_seed0_final_lora/` and `docs/study/artifacts/adapters/llama32_3b/v3/uniform_seed0_final_lora/`. Control seed 0 has no downloaded adapter yet.

The comparisons this study will report, when Colab writes them, are the four that the design can actually support: the Llama 3.2 3B Instruct base (λ=0); the size-matched citation-control at λ=0.5; the uniform high-intrinsic arm at λ=0.5 (same 1,001 rows as weighted, weights 1); and the weighted high-intrinsic arm at λ=0.5 (`w = s / mean`). Arm A at λ=1 is a WiSE-FT reference for the unblended adapter, so a reader can see what the blend changed. Until those numbers exist, every score cell below is pending.

| Arm | What it is | λ | Seed mean | Seed 0 | Seed 1 | Seed 2 |
|---|---|---:|---:|---:|---:|---:|
| Base | Llama 3.2 3B Instruct, no fine-tune | 0 | _pending_ | — | — | — |
| C, citation-control | size-matched, weights 1 | 0.5 | _pending_ | _pending_ | _pending_ | _pending_ |
| B, uniform high-intrinsic | same 1,001 rows as A, weights 1 | 0.5 | _pending_ | _pending_ | _pending_ | _pending_ |
| A, weighted high-intrinsic | `w = s / mean`, no clip | 0.5 | _pending_ | _pending_ | _pending_ | _pending_ |

The mix those adapters were trained on is already measured, and is not an evaluation result: 1,001 CAI rows, 300 format-replay rows, Kish ESS 789.94, max weight 2.248. Style-drift and dogmatic-versus-reflective judgments on fifty held-out contemporary dilemmas, at λ=0.5, are also pending. JSON and CSVs from Colab should land under `colab/results/` and be copied next to this page when they exist. Why each arm exists is [`v3-methods.md`](v3-methods.md).
