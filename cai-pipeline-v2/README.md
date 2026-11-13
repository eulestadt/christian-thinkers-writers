# v2 CAI pipeline (additive)

This folder builds the **v2** mix. The current training run is **v3**: [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../colab/cai_3way_experiment_and_eval_v3.ipynb), mix in [`colab/data/v3/`](../colab/data/v3/), builder [`colab/scripts/build_v3_dialogues.py`](../colab/scripts/build_v3_dialogues.py).

v1 trains two arms on the book dialogues with one seed. v2 keeps all 40 books, adds contemporary dilemmas and UltraChat, clips weights to 0.25–2, and draws rows with replacement. v3 keeps the top 20 books by `ier_intrinsic`, rewrites the v2 `draft_0` / `revision_3` traces into one weighed turn, and drops UltraChat. The 300-row format replay stays. Three arms: weighted `w = s / mean` with no clip, uniform `w = 1` on the same rows, and a size-matched control. R-DPO α=0.01, β=0.1. Rejected answers alternate between the unrevised draft and a dogmatic rewrite. Seeds 0–2. Reported score: WiSE-FT λ=0.5. Published mix: 1,001 CAI rows per arm plus 300 format replay. Kish ESS 789.94. Max weight 2.248. The v1, v1_fixed, and v2 notebooks stay untouched.

The scripts below still build the v2 mix for
`colab/cai_3way_experiment_and_eval_v2.ipynb`. They do not overwrite v1
`cai-pipeline/` or `cai-pipeline-control/` artifacts, and they do not write `colab/data/v3/`.

Default model for the Colab train cell is **Llama 3.2 3B Instruct**. Mistral
24B is an optional later run, not part of this PR.

## Mix

- 2,000 book-elicited + 1,000 OOD-train + 500 UltraChat = 3,500
- Plus ~300 format-following SFT/DPO pairs (`colab/data/v2/format_replay_*.jsonl`)
  default on: letter or 0/1 chosen; verbose-then-conclude rejected; **not**
  from MMLU / ETHICS / TruthfulQA (ARC-Easy train + synthetic)
- Treatment weights: clip(s_bar / mean(s), 0.25, 2.0)
- Control: same OOD prompts, uniform weights

## Scripts

| Script | Role |
|---|---|
| `scripts/01_build_ood_train.py` | 1,000 OOD-train prompts, MiniLM near-dup filter |
| `scripts/03_stage1_sl_cai_ood.py` | SL-CAI on OOD-train (`--arm treatment\|control`) |
| `scripts/04_stage2_rlaif_ood.py` | RLAIF DPO pairs |
| `scripts/06_mix_weighted.py` | Mix elicited + UltraChat + OOD + format replay |

Do not launch GPU training from the agent VM. The current notebook to run on Colab is the v3 notebook. The v2 notebook remains available for the mix this folder builds.
