# v2 training JSONL (additive)

The current training mix is **v3**, in [`../v3/`](../v3/). v3 keeps the top 20 books by `ier_intrinsic`, rewrites these v2 traces into one weighed turn, and does not use UltraChat. It keeps this folder's 300-row format replay. Three arms (weighted `w = s / mean` with no clip, uniform `w = 1` on the same rows, size-matched control), R-DPO α=0.01 β=0.1, bidirectional rejected answers, seeds 0–2, reported WiSE-FT λ=0.5. Published v3 counts: 1,001 CAI rows per arm plus 300 format replay. Kish ESS 789.94. Max weight 2.248. Notebook: [`../../cai_3way_experiment_and_eval_v3.ipynb`](../../cai_3way_experiment_and_eval_v3.ipynb). This v2 directory stays as the earlier mix. The v1, v1_fixed, and v2 notebooks stay untouched.

## v2 files

`format_replay_{sft,dpo}.jsonl` — 300 format-following pairs (default on). Letter or 0/1 chosen; verbose-then-conclude rejected. Not from MMLU / ETHICS / TruthfulQA.

`treatment/` and `control/` hold the v2 mix after `cai-pipeline-v2/scripts/06_mix_weighted.py`:

- `sl_cai_train.jsonl` / `rlaif_dpo_train.jsonl` — 3,800 SFT rows (2,000 elicited + 1,000 OOD-train + 500 UltraChat + 300 format replay). DPO drops pairs where chosen equals rejected.
- `sl_cai_ood.jsonl` / `rlaif_ood.jsonl` — OOD-only sidecars (n=1000).
- `mix_summary.json` — counts and I/E-R vs uniform weights.

Treatment uses I/E-R example weights clipped to (0.25, 2.0). Control is uniform `weight=1.0`. Do not overwrite v1 `data/treatment` or `data/control`.

The v2 Colab notebook clones GitHub `main`. If `treatment/sl_cai_train.jsonl` is missing, Cell 1 fails instead of training. Do not clone a stale main-only checkout from before this mix landed.
