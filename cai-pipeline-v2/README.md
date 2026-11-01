# v2 CAI pipeline (additive)

Builds the OOD-train mix and I/E-R-weighted SL-CAI / RLAIF JSONL for
`colab/cai_3way_experiment_and_eval_v2.ipynb`. Does not overwrite v1
`cai-pipeline/` or `cai-pipeline-control/` artifacts.

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

Do not launch GPU training from the agent VM; run the v2 notebook on Colab.
