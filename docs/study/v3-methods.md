# Why v3 is trained this way

v3 fine-tunes Llama 3.2 3B Instruct with 4-bit QLoRA on a Constitutional AI mix drawn from the twenty English public-domain books with the highest Gorsuch and McPherson I/E-R intrinsic scores. The question is whether that high-intrinsic slice, versus a citation-matched public-domain control of the same size, moves how the model answers held-out dilemmas. I/E-R scores are trainer metadata: they pick books and set row weights. They are not written into prompts. Critique and revision stay on the authors’ own vocabulary, in the Constitutional AI sense of Bai et al. (2022, arXiv:2212.08073): principles are instructions for a critique, not text the model is asked to imitate.

## A small high-scoring slice, then one deliberative turn

v1 trained on the full forty-book dialogues. v2 kept those forty and mixed in extra contemporary rows. v3 instead keeps ranks 1–20 of the I/E-R list in [`four-top40-lists.md`](../four-top40-lists.md), from Brother Lawrence through John Taylor. Zhou et al. (2023, LIMA, arXiv:2305.11206) and Chen et al. (2023, AlpaGasus, arXiv:2307.08701) are the reason: a curated high-quality subset beat training on the larger, noisier set. Ranking is by I/E-R intrinsic; I−E is not a second gate, because the published scales treat intrinsic as its own construct.

Each kept v2 `draft_0` / `revision_3` trace is rewritten, with `deepseek-flash`, into one user dilemma and one weighed assistant reply. Guan et al. (2024, arXiv:2412.16339) is why the saved completion is deliberative: it states a prior, names the principle as a candidate, and says what it changes and what it does not. Loss is on that completion only, so the model is not trained to imitate the book’s diction. The generation may reason in a short exchange; that draft is discarded.

The mix that actually enters the trainer is those rewritten CAI rows plus a shared 300-row format-replay slice (ARC-Easy train and synthetic letter-format items), every replay row at weight 1. That is the InstructGPT pattern (Ouyang et al., 2022): a little format-following data so the model still emits the output shape the scorer expects — here, a multiple-choice letter in the first eight tokens. The published files in [`colab/data/v3/mix_summary.json`](../../colab/data/v3/mix_summary.json) are 1,001 CAI rows per arm plus those 300 replay rows (1,301 SFT and 1,301 DPO). 1,001 is simply how many v2 traces survived the top-twenty cut.

## Weights, and a uniform arm on the same rows

After the filter, Arm A sets `w = s / mean(s_kept)` with no clip, then renormalizes so the mean CAI weight is 1. Byrd and Lipton (2019) and Cui et al. (2019) are why the score is a loss weight rather than a sampling probability, and why the weights stay uncapped: clipping would hide how uneven the kept slice is. Uncapped weights raise variance, so the run logs Kish effective sample size `(∑w)² / ∑w²`. For this mix that ESS is 789.94 on 1,001 rows; the heaviest CAI weight is 2.248 and the lightest is 0.571. Those numbers are a check that the weights are doing real work without collapsing the slice to a handful of books.

Arm B is the same 1,001 rows and the same row ids, every weight 1. That is the ablation Byrd and Lipton imply: if Arm A beats Arm B, the gain is from the weights, not from having already thrown out ranks 21–40. At a given seed, A and B share the shuffle so `w` is the only difference. Arm C draws 1,001 control dialogues, stratified by book so one author cannot fill the set, all weights 1, plus the same 300 replay rows. Size-matching is what makes A minus C a corpus contrast rather than a contrast of “more rows versus fewer.”

When evaluation exists, the reported comparison is the base model, Arm C, Arm B, and Arm A. WiSE-FT λ=0.5 is the reported operating point for the three trained arms; λ=0 is the base, scored once; λ=1 on Arm A is a reference for the unblended adapter.

## Length, both kinds of rejection, and three seeds

Preference training is R-DPO (Park, Rafailov, Ermon, and Finn, 2024, arXiv:2403.19159) with α=0.01 and β=0.1, the values they report for the Helpful-Harmless dialogue setting at this β. Ordinary DPO’s implicit reward tracks completion length on samples the model itself writes; the length term sits inside the logistic so a longer chosen answer shrinks that pair’s gradient. β=0.1 is also the DPO default for staying near the reference (Rafailov et al., 2023, arXiv:2305.18290). The same R-DPO loss is used on every arm so length is not a fourth experimental factor.

Rejected answers alternate between the unrevised draft (501 pairs) and a dogmatic rewrite that drops the user’s constraints (500 pairs), on every arm. Sharma et al. (2023, arXiv:2310.13548) is why both are there: one rejected type is “ignore the principle,” the other is “parrot it and run over the user.” Chosen is always the weighed, deliberative completion.

Dodge et al. (2020, arXiv:2002.06305) is why there are three seeds, 0, 1, and 2, and why the reported number is their mean. Initialization and data order each move fine-tuning results by as much as a real treatment effect; picking the best seed would write that noise down as a finding.

Every arm, every seed, uses the same QLoRA setup (Dettmers et al., 2023; Hu et al., 2021): all seven projections, rank 16, alpha 32, one SFT epoch at `1e-4`, then R-DPO at `5e-6`. The row weight multiplies the whole per-sequence SFT loss and the whole per-pair R-DPO loss once per epoch. Adapters write under `adapters/{ACTIVE_MODEL}/v3/`.

## How scores will be read

Wortsman et al. (2022, WiSE-FT, arXiv:2109.01903) is why the reported evaluation blends the adapter with the base at λ=0.5: fine-tuning improves the training distribution and often hurts the rest; interpolating recovers the base’s robustness while keeping most of the gain. The primary metric remains 8-token generation plus `parse_letter`. Dror et al. (2018) is why the interval will be a paired bootstrap over items on Arm A minus Arm B and Arm A minus Arm C, using the per-item mean across the three seeds. A fifty-item contemporary set at λ=0.5 will watch style drift and dogmatic versus reflective replies.

Those scores are not on GitHub yet. [`results.md`](results.md) stays empty until Colab writes them. The notebook is [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../../colab/cai_3way_experiment_and_eval_v3.ipynb).
