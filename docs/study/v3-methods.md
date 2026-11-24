# How v3 is trained, and how that differs from v1 and v2

The current experiment is a small, honest comparison on one open-weight model: Llama 3.2 3B Instruct, loaded in 4-bit with Unsloth, LoRA on all seven projection matrices. It is not a five-arm design with a new-age corpus or a “virtue across religions” corpus. Those sketches exist in older notes; they were not trained in this run. It is also not a 24B or 72B study. If those models are trained later, they will get their own namespace. This page describes only v3.

The scientific question is whether a language model that sees a high-intrinsic Christian corpus, scored with Gorsuch and McPherson I/E-R intrinsic and not filtered on I minus E, answers held-out dilemmas any differently than the same recipe on a citation-matched public-domain control of the same size. I/E-R scores are trainer metadata. They are used to pick books and to set row weights. They are not written into prompts, constitutions, or judges. Critique and revision stay on the authors’ own vocabulary. External flourishing taxonomies are not a training target and are not injected into the loop.

## What v1 and v2 already did

v1, in [`colab/cai_3way_experiment_and_eval.ipynb`](../../colab/cai_3way_experiment_and_eval.ipynb), trained two arms on the full book-dialogue sets: treatment Constitutional AI versus citation-control Constitutional AI. It used one seed (`3407`), a higher LoRA rank, and ordinary DPO. Adapters landed directly under the model name. That run is frozen.

v2, in [`colab/cai_3way_experiment_and_eval_v2.ipynb`](../../colab/cai_3way_experiment_and_eval_v2.ipynb), kept all forty treatment books and all forty control books. It mixed in contemporary dilemmas and a 500-row UltraChat slice, clipped treatment weights into the interval 0.25 to 2.0, and drew rows with replacement. It still trained two arms with one seed, and it wrote under `adapters/{model}/v2/` so that v1 would not be overwritten. The v2 traces (`draft_0` and `revision_3`) are the raw material v3 rewrites. The v2 mix itself is not what v3 trains.

## What v3 changes

v3 keeps only the twenty books with the highest `ier_intrinsic` scores — ranks 1 through 20 of the I/E-R list in [`four-top40-lists.md`](../four-top40-lists.md), from Brother Lawrence’s *The Practice of the Presence of God* through John Taylor’s *The Government of God*. There is no I−E > 0 requirement. Each kept v2 trace is rewritten, with `deepseek-flash` and thinking off, into a single user dilemma and a single weighed assistant reply. The completion is supposed to state a prior, name a principle as a candidate, and say what changes and what does not. Loss is on that completion only. UltraChat is out of the mix.

What stays is a shared 300-row format-replay slice (ARC-Easy train items and synthetic letter-format items, not MMLU, ETHICS, or TruthfulQA), every row at weight 1. That slice exists so the model still answers multiple-choice items with a letter in the first eight tokens. The published mix is in [`colab/data/v3/mix_summary.json`](../../colab/data/v3/mix_summary.json): 1,001 CAI rows per arm plus those 300 replay rows, so 1,301 SFT rows and 1,301 DPO pairs. After the top-twenty filter, Arm A’s CAI weights are `w = s / mean(s_kept)` with no clip, renormalized so the mean CAI weight is 1. The Kish effective sample size of that weighted CAI slice is 789.94. The heaviest row is 2.248; the lightest is 0.571.

## Three trained arms, four things we will score

There are three things we actually fine-tune, and a fourth we score without training.

Arm A, weighted high-intrinsic, uses the 1,001 kept treatment dialogues with those uncapped weights, plus the shared replay. Arm B, uniform high-intrinsic, uses the same 1,001 rows and the same row ids, every weight set to 1, plus the same replay. At a given seed, Arm A and Arm B share the shuffle, so the only difference between them is `w`. Arm C, citation-control, draws 1,001 dialogues from the control pool, stratified by book so one author cannot fill the set, all weights 1, plus the same replay. If the control pool had been smaller than 1,001 we would have stopped; it was not.

When evaluation exists, the honest comparison is four-way: the untouched base model (WiSE-FT λ=0, scored once); Arm C at λ=0.5; Arm B at λ=0.5; and Arm A at λ=0.5. The mean of seeds 0, 1, and 2 is the reported number, not the best seed. λ=1 on Arm A is a reference row that shows the unblended adapter; it is not a fifth experimental arm. Arm A minus Arm B isolates the weights from the top-twenty filter. Arm A minus Arm C isolates the high-intrinsic corpus from a size-matched citation control.

Those scores have not been written yet. This page does not invent them.

## The shared trainer

Every trained arm, every seed, uses the same optimizer. The base is `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`. LoRA rank 16, alpha 32, all seven projections. Supervised fine-tuning runs for one epoch at learning rate `1e-4`. Preference training then uses R-DPO (Park, Rafailov, Ermon, and Finn, 2024) with α=0.01 and β=0.1, the dialogue-setting values from that paper, at learning rate `5e-6`. The length term sits inside the logistic: when the chosen completion is longer than the rejected one, that pair’s gradient shrinks. The row weight multiplies the whole per-sequence SFT loss and the whole per-pair R-DPO loss, once per epoch. Rows are not drawn with replacement.

Rejected answers alternate between the unrevised draft (501 pairs) and a dogmatic rewrite that drops the user’s constraints (500 pairs), on every arm, including uniform and control. Nine trains are planned: three arms times seeds 0, 1, and 2. Adapters and result files go only under `adapters/{ACTIVE_MODEL}/v3/` and `results/{ACTIVE_MODEL}/v3/`. v1, v1_fixed, and v2 folders stay untouched. ETHICS, TruthfulQA, and MMLU are evaluation surfaces, not training text.

## What we will report, when Colab finishes

The primary metric remains 8-token generation plus `parse_letter`. Paired-bootstrap 95% intervals, resampling evaluation items and keeping arms paired, will cover Arm A minus Arm B and Arm A minus Arm C. A small held-out set of about fifty contemporary dilemmas will be scored at λ=0.5 for style drift and for dogmatic versus reflective replies. None of that is on GitHub yet. Fill [`results.md`](results.md) from Colab when it exists; do not backfill from older runs.
