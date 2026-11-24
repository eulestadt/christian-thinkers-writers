# v3 3B results

Arms A, B, and C are tied on the primary multiple-choice suite. That is the finding, not a failed contrast. The uniform arm exists so a gap could be read as I/E-R weights rather than as the top-twenty filter. The size-matched control exists so a gap could be read as corpus. Neither gap appeared.

Pooled over 1,600 primary items, item means first across seeds 0, 1, and 2 at WiSE-FT λ=0.5, then a paired bootstrap over items (Dror et al., 2018): **A−B 0.000 pp, 95% CI [−0.417, 0.438]**; **A−C −0.312 pp, 95% CI [−0.917, 0.271]**. Both intervals cover 0. A naive reading would take “the high-intrinsic books were weighted” as the cause of whatever moved. The design was built to test that, and the weights did not move this metric. Kish ESS was 789.94 and the heaviest CAI weight was 2.248, so the weights were doing real work in the loss; they still did not separate A from B.

What did move is shared across A, B, and C relative to the Llama 3.2 3B Instruct base. All three arms rise on ETHICS commonsense and virtue and on the MMLU philosophy and world-religions slices, and fall on ETHICS deontology and justice and on TruthfulQA MC1. That common shift is the training signal — Constitutional AI dialogues plus format replay plus R-DPO on this 3B QLoRA setup — not I/E-R weights, and not the treatment-versus-control cut.

The reported number is the mean of seeds 0, 1, and 2 at λ=0.5, 8-token generation plus `parse_letter` (`parse_01` on ETHICS binary). The mean, not the best seed, is the result because seed scatter can mimic a treatment effect (Dodge et al., 2020). λ=0.5 is a blend of adapter and base (Wortsman et al., 2022). I/E-R scores stayed trainer metadata. ETHICS, TruthfulQA, and MMLU were evaluation suites.

## Reported means (λ=0.5, mean of three seeds)

Accuracy %, 8-token generative scorer. Source: [`v3_reported_means_lambda0.5.csv`](v3_reported_means_lambda0.5.csv).

| Benchmark | base | weighted (A) | uniform (B) | control (C) |
|---|---:|---:|---:|---:|
| ETHICS_commonsense | 50.50 | 59.83 | 59.67 | 59.50 |
| ETHICS_deontology | 48.00 | 40.83 | 40.33 | 41.17 |
| ETHICS_justice | 55.00 | 38.67 | 39.00 | 38.83 |
| ETHICS_virtue | 29.00 | 36.83 | 37.50 | 37.17 |
| MMLU_moral_disputes | 52.67 | 55.34 | 55.34 | 54.89 |
| MMLU_moral_scenarios | 48.00 | 43.55 | 43.56 | 45.33 |
| MMLU_philosophy | 60.67 | 63.33 | 62.67 | 62.67 |
| MMLU_world_religions | 68.67 | 71.33 | 71.56 | 72.22 |
| TruthfulQA_MC1 | 53.00 | 45.67 | 45.67 | 46.50 |

A, B, and C sit within about one percentage point on every row. Per-benchmark intervals are in [`paired_bootstrap.json`](paired_bootstrap.json). Per-seed cells, which should not be cherry-picked, are in [`objective_by_seed.csv`](objective_by_seed.csv). Example, ETHICS commonsense at λ=0.5: weighted 60.5 / 59.5 / 59.5; uniform 58.0 / 61.0 / 60.0; control 61.0 / 56.0 / 61.5.

## Arm A at λ=1 (reference, not the reported result)

Mean of seeds 0, 1, and 2 at unblended adapter scale 1. λ=1 lifts ETHICS virtue a lot versus λ=0.5 (52.17 vs 36.83). It does not change the A/B/C near-tie at the reported operating point.

| Benchmark | weighted λ=1 mean |
|---|---:|
| ETHICS_commonsense | 60.00 |
| ETHICS_deontology | 48.17 |
| ETHICS_justice | 39.83 |
| ETHICS_virtue | 52.17 |
| MMLU_moral_disputes | 56.89 |
| MMLU_moral_scenarios | 44.67 |
| MMLU_philosophy | 63.33 |
| MMLU_world_religions | 72.67 |
| TruthfulQA_MC1 | 43.00 |

## Style (50 held-out dilemmas, λ=0.5, mean of seeds)

Regex cues, not an LLM judge. Archaic-pronoun drift was 0 on the base and on every arm. Reflective-cue rate was 0: the held-out replies did not match the “prior / candidate / changes / stays” patterns. Dogmatic-command rate: base 0.10, weighted 0.0067, uniform 0.0133, control 0.0067. Tables: [`style_summary.csv`](style_summary.csv), [`style_summary.json`](style_summary.json).

## Run

Session `ctw-v3` on an NVIDIA L4, account `eulestadt@gmail.com`. All nine QLoRA trains finished (weighted, uniform, and control × seeds 0, 1, and 2), then primary eval and the fifty-item style set. Start balance 506.04 CU at 8:24 p.m. Eastern on 23 November 2026; stop 495.92 CU at 1:40 a.m. Eastern on 24 November 2026. **10.12 compute units** spent of the ~240 cap, at 1.94 units per hour. Mix: 1,001 CAI rows plus 300 format replay, Kish ESS 789.94, max weight 2.248. Chronology: [`v3-run-log.md`](v3-run-log.md). Machine-readable summary: [`v3-results.json`](v3-results.json). Executed notebook: [`cai_3way_experiment_and_eval_v3_executed.ipynb`](cai_3way_experiment_and_eval_v3_executed.ipynb).

Adapter safetensors (~93 MB each) stay in the private study store; this repository’s Git LFS tracks the census jsonl and the scores sqlite. Trainer-state JSON for the nine adapters is under [`artifacts/adapters/llama32_3b/v3/`](artifacts/adapters/llama32_3b/v3/). Eval tables are also under [`artifacts/results/llama32_3b/v3/`](artifacts/results/llama32_3b/v3/). Why the arms were built this way is [`v3-methods.md`](v3-methods.md).
