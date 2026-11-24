# v3 methods (vs v1 and v2)

v3 is the current training run. It is **not** a five-arm new-age / virtue-across-religions design. It is **not** a Mistral 24B or Qwen2-VL-72B run. Llama 3.2 3B Instruct only, QLoRA 4-bit.

I/E-R scores are trainer metadata. They are not written into prompts, constitutions, or judges. Critique and revision stay on author-text vocabulary: no VanderWeele domains, FAI, or other flourishing taxonomies. Rank by I/E-R intrinsic only; do not require I−E > 0. Do not train on ETHICS, TruthfulQA, or MMLU. Writes go only to `adapters/{ACTIVE_MODEL}/v3/` and `results/{ACTIVE_MODEL}/v3/`. Prior run namespaces stay on disk.

## What changed

| | v1 | v2 | **v3 (this run)** |
|---|---|---|---|
| Notebook | `colab/cai_3way_experiment_and_eval.ipynb` | `…_v2.ipynb` | `…_v3.ipynb` |
| Books | 40 treatment + 40 control | same 40 + 40 | **top 20** treatment books by `ier_intrinsic`; control size-matched from the 40 |
| Dialogue | book-elicited traces | those traces plus contemporary dilemmas | **one weighed turn**: rewrite existing v2 `draft_0` / `revision_3` traces into one user dilemma and one assistant completion |
| UltraChat | 500 rows in the 2,500-prompt set | 500 rows in the 3,500-row mix | **none** |
| Format replay | not the v1 mix | ~300 rows | **300** rows, weight 1 (ARC-Easy train + synthetic letter-format items) |
| Train arms | treatment CAI vs citation-control CAI | same two arms | **three**: weighted, uniform (same rows), size-matched control |
| CAI rows per arm | 2,500 | 3,500 mix | **1,001** + 300 replay = 1,301 SFT / 1,301 DPO |
| Weights | not the v3 formula | clip `(0.25, 2.0)`, mean ~1.08, draw with replacement | **Arm A** `w = s / mean(s_kept)`, no clip, mean CAI weight 1; **Arm B** same rows, `w = 1`; **Arm C** `w = 1` |
| Kish ESS (CAI slice) | — | — | **789.94** (`(∑w)² / ∑w²`) |
| Max / min CAI weight | — | 2.0 / 0.25 by clip | **2.248** / 0.571 |
| Loss | DPO | DPO | **R-DPO α=0.01, β=0.1** on every arm (Park et al. 2024, dialogue setting) |
| Seeds | one (`3407`) | one | **0, 1, 2** (mean, not best) |
| Reported eval | adapter as trained | adapter as trained | **WiSE-FT λ=0.5** (λ=1 on Arm A as reference; λ=0 is the base model, scored once) |
| Adapter path | `adapters/{model}/` | `adapters/{model}/v2/` | `adapters/{model}/v3/` only |

Published mix: [`colab/data/v3/mix_summary.json`](../../colab/data/v3/mix_summary.json). Book list: ranks 1–20 of [`four-top40-lists.md`](../four-top40-lists.md) (Brother Lawrence through John Taylor). Notebook coverage: [`cai-3way-colab.md`](../cai-3way-colab.md).

## Four honest eval arms

The letter and the result tables report these four, if they exist after Colab:

1. **Base** — Llama 3.2 3B Instruct, no fine-tune (WiSE-FT λ=0).
2. **Citation-control, size-matched** — Arm C: 1,001 control dialogues, stratified by book, weights 1, plus 300 format replay.
3. **Uniform high-intrinsic** — Arm B: the 1,001 kept treatment dialogues, every weight 1, plus the same 300 replay rows.
4. **Weighted high-intrinsic** — Arm A: the same 1,001 rows and row ids as Arm B, `w = s / mean`, plus the same replay.

Arm A minus Arm B isolates the weights from the top-20 filter. Arm A minus Arm C isolates the high-intrinsic corpus from a citation-matched public-domain control of the same size. λ=1 on Arm A is a reference row, not a fifth experimental arm.

## Trainer (every trained arm, every seed)

- Model: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`. LoRA on all seven projections, `r=16`, `alpha=32`. SFT LR `1e-4`, 1 epoch. DPO LR `5e-6`.
- Per-sequence SFT loss and per-pair R-DPO loss multiplied by `w_i` once per epoch. No sampling with replacement.
- R-DPO: `L = -w * log σ( β log π(y_w)/π_ref(y_w) − β log π(y_l)/π_ref(y_l) + α(|y_w| − |y_l|) )` with α=0.01, β=0.1. `|y|` is tokenizer length of the completion.
- Rejected answers alternate between the unrevised draft (501) and a dogmatic rewrite (500) on every arm.
- Nine trains: `{weighted, uniform, control} × {0, 1, 2}`. Arm A and Arm B share the shuffle at a given seed.

## What is not in this method

- No flourishing taxonomy, VanderWeele domains, or FAI labels in extract / critique / revision / judge prompts.
- No I−E > 0 filter.
- No UltraChat.
- No new-age wisdom arm and no virtue-across-religions arm.
- No Mistral 24B and no Qwen2-VL-72B in this run.
- No overwrite of v1, v1_fixed, or v2 adapters or notebooks.

## Scores still pending

Eval (8-token generation + `parse_letter`, paired-bootstrap 95% intervals on A−B and A−C, 50-item style / dogmatic-vs-reflective set at λ=0.5) has not been written to GitHub. Fill [`results.md`](results.md) from Colab, then rewrite [`email-vanderweele.md`](email-vanderweele.md) from those numbers.
