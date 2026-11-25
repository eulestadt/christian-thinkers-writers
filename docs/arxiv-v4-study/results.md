# v4 results

Llama 3.2 3B Instruct, QLoRA, one SFT pass then R-DPO (α=0.01, β=0.1). Three arms, seeds 0, 1, and 2. The reported number is the mean of the three seeds at WiSE-FT λ=0.5. Intervals are 95% paired bootstraps over items. Ethics numbers are balanced accuracy. MMLU numbers are accuracy averaged over four rotations of the options. TruthfulQA was not scored.

Open and Sealed used the same 1,000 dilemmas and the same verdicts (82 plain refusals, 918 allows). Neighbor used the 1,001 citation-matched dilemmas, written in the open way. Every weight was 1. Each arm also had the unchanged 300-row format replay, plus 48 second-turn rows for Open and Sealed and 46 for Neighbor.

## Headline

Open − Sealed on ETHICS commonsense balanced accuracy is **−0.8 points, 95% interval [−2.8, +1.3]**. Open − Neighbor is **+0.0 [−2.4, +2.5]**. Every other Open−Sealed and Open−Neighbor interval in the suite covers zero. The way of reading did not move these tests. The training did.

| Test | Base | Open | Sealed | Neighbor | Open−Sealed | Open−Neighbor |
|---|---:|---:|---:|---:|---|---|
| Commonsense | 48.6 | 58.8 | 59.6 | 58.8 | −0.8 [−2.8, +1.3] | +0.0 [−2.4, +2.5] |
| Excuses | 53.1 | 58.3 | 58.2 | 59.1 | +0.1 [−1.3, +1.5] | −0.8 [−2.3, +0.7] |
| Justice | 52.2 | 68.3 | 66.9 | 67.0 | +1.4 [−0.2, +3.2] | +1.3 [−0.7, +3.3] |
| Virtue | 67.5 | 70.3 | 69.4 | 70.2 | +1.0 [−1.1, +3.1] | +0.1 [−1.8, +1.8] |
| Moral scenarios | 42.7 | 47.6 | 47.3 | 47.6 | +0.3 [−1.2, +1.8] | 0.0 [−1.3, +1.3] |
| Moral disputes | 48.8 | 54.8 | 54.1 | 53.9 | +0.7 [−0.7, +2.0] | +0.9 [−0.3, +2.1] |
| Philosophy | 56.8 | 60.8 | 61.1 | 60.7 | −0.3 [−1.3, +0.7] | +0.1 [−1.3, +1.3] |
| World religions | 68.8 | 72.6 | 72.7 | 71.8 | −0.1 [−0.9, +0.7] | +0.8 [−0.5, +2.1] |

Commonsense positive-label rate (share of answers that are 1, "unacceptable"): base 94.5%, Open 58.7%, Sealed 62.8%, Neighbor 68.2%. Open's three seeds on commonsense balanced accuracy were 57.5, 61.0, and 58.0. Sealed's were 58.7, 61.0, and 59.0. Neighbor's were 59.9, 59.5, and 57.0. An earlier note listed a different triple; those were the full-strength (λ=1) seeds, not the reported λ=0.5 ones. The seed spread is larger than the arm gap.

At λ=1 the tie remains. Commonsense balanced accuracy is 61.1 / 61.7 / 60.5 for Open / Sealed / Neighbor, and Open−Sealed is −0.6 [−2.7, +1.4]. Virtue gets worse as the adapter is mixed in further: the models say the trait applies more often (Open 68.5%) on a key that is mostly "does not apply," and balanced accuracy slips below base.

## Commands and second turns

Command-phrasing rate on 50 held-out dilemmas, λ=0.5: base 0.08, Open 0.02, Sealed 0.073, Neighbor 0.033. Open−Sealed is −5.3 percentage points [−10.0, −1.3]. At λ=1 that gap flips (Open 0.093, Sealed 0.053), so I do not treat it as a stable finding.

On 36 held-out second turns, discernment is the rate of changing for a real reason minus the rate of changing under pressure only. Base 61.1, Open 14.8, Sealed 13.0, Neighbor 11.1. Open−Sealed is +1.9 [−15.7, +20.0]. The trained models almost never update. The untouched model does.

## What was dropped

TruthfulQA, because the old harness always listed the correct option first. One treatment dilemma the generator would not stop copying. Two Neighbor second-turn rows lost to a file collision. No fifth seed, no larger model, no BBQ. The first L4 died after five adapters were already saved; the missing four were trained on a later L4. All nine finished.
