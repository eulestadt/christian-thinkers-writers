# Cursor specification: 3-arm CAI fine-tune and multi-benchmark Colab notebook

## Current notebook (v3)

The notebook to open is `colab/cai_3way_experiment_and_eval_v3.ipynb`. The v1 notebook, the v1_fixed notebook, and the v2 notebook stay untouched. The sections below the v3 notes are the earlier specifications. They are not the current run.

v1 trains two arms on the book dialogues with one seed. v2 keeps all 40 books, adds contemporary dilemmas and UltraChat, clips weights to 0.25–2, and draws rows with replacement.

v3 keeps the top 20 books by `ier_intrinsic` (no I−E filter) and rewrites the existing v2 `draft_0` / `revision_3` traces into one user message and one weighed assistant reply. UltraChat is not in the mix. The shared 300-row format replay stays, at weight 1.

Three arms:

| Arm | Rows | Weight |
|---|---|---|
| A, weighted | 1,001 kept treatment dialogues | `w = s / mean(s)`, no clip, mean CAI weight 1 |
| B, uniform | the same rows and row ids | every weight 1 |
| C, control | 1,001 control dialogues, stratified by book | every weight 1 |

Each arm also has the 300 format-replay rows. Kish ESS of the weighted CAI slice is **789.94**. Max weight is **2.248**.

Every arm uses R-DPO with α=0.01 and β=0.1. Rejected answers alternate between the unrevised draft and a dogmatic rewrite. Seeds are 0, 1, and 2. The reported score is the mean of those seeds at WiSE-FT λ=0.5. λ=1 on arm A is a reference. λ=0 is the base model, scored once. Llama 3.2 3B only. Writes only `adapters/{ACTIVE_MODEL}/v3/` and `results/{ACTIVE_MODEL}/v3/`. Scores are trainer metadata and never appear in prompts. Do not train on ETHICS, TruthfulQA, or MMLU. The nine QLoRA trains run in Colab.

## Objective (v1, kept for the original notebook)

Self-contained Jupyter notebook `cai_3way_experiment_and_eval.ipynb` for Google Colab (L4 24GB or A100 40GB). Primary model: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`. Toggle `ACTIVE_MODEL = "mistral_small_24b"` for Mistral Small 24B Instruct.

## Strict constraints

1. **Read-only treatment/control data.** Never modify `data/treatment/`, `data/control/`, or any `.sqlite`. Open input JSONL with `"r"` only. Copy JSONL to chmod `0444` scratch. Copy sqlite to `/content/local_db_cache`, chmod `0444`, open `file:...?mode=ro&immutable=1`. GitHub LFS is the DB source of truth; Drive FUSE is never the live SQLite filesystem. Training stays on JSONL.
2. **Identical hyperparameters.** Both arms use `seed=3407`, LoRA `r=32`, `lora_alpha=64`, SFT LR `2e-4`, DPO LR `5e-6`, same batch/accum as the active model preset.
3. **Do not launch GPU** until control `sl_cai_train.jsonl` and `rlaif_dpo_train.jsonl` exist (2,500 rows). Skip training if `adapters/{ACTIVE_MODEL}/{arm}_final_lora` already exists.
4. **No merged 6–48GB dumps.** Keep the 4-bit base in VRAM; attach/detach LoRA adapters.
5. **Writes** only under `adapters/{ACTIVE_MODEL}/` and `results/{ACTIVE_MODEL}/`.

## Drive layout (`/content/drive/MyDrive/CAI_Experiment/`)

```
db/audit.sqlite                            # OPTIONAL rclone overlay; GitHub LFS is SoT
data/treatment/sl_cai_train.jsonl          # 2,500  READ-ONLY
data/treatment/rlaif_dpo_train.jsonl       # 2,500  READ-ONLY
data/control/sl_cai_train.jsonl            # 2,500  READ-ONLY
data/control/rlaif_dpo_train.jsonl         # 2,500  READ-ONLY
data/eval/open_ended_dilemmas.jsonl        # 150 held-out
data/eval/psychometric_kept_works.csv      # 527 census overlay
adapters/{ACTIVE_MODEL}/{experimental,control}_final_lora
results/{ACTIVE_MODEL}/objective_alignment_benchmarks.csv
results/{ACTIVE_MODEL}/blind_3way_arena_outputs.jsonl     # includes mapping (researcher)
results/{ACTIVE_MODEL}/blind_3way_arena_ballots.jsonl     # no mapping (judge)
results/{ACTIVE_MODEL}/cai_3arm_master_diagnostics.png
results/{ACTIVE_MODEL}/qualitative_3arm_inspector.html
```

## Notebook cells

1. Install Unsloth/TRL; mount Drive; verify read-only JSONL; cache sqlite to local NVMe (`0444`, immutable URI).
2. `train_cai_arm` for experimental then control (SFT → DPO).
3. Load base once; `PeftModel` multi-adapter; `generate_with_arm`.
4. ETHICS (4 subsets × 200), TruthfulQA MC1 (200), MMLU moral/philosophy (≤150).
5. 150-dilemma blind 3-way arena (temperature 0.3, shuffled Option_1/2/3).
6. Four-panel 300 DPI figure (radar, Δ bars, DPO margins, psychometric KDE).
7. Lorenz, SFT loss overlay, paired bootstrap CIs, optional UMAP (t-SNE fallback), win-rate.
8. Optional DeepSeek judge (`RUN_JUDGE=False`; `deepseek-flash` = V4.1-Flash-1017, not retired `deepseek-chat`).
9. HTML qualitative inspector.

## v2 notebook (additive; not the current run)

`cai_3way_experiment_and_eval_v2.ipynb` is an earlier copy. The current run is v3, described at the top of this file. v2 default `ACTIVE_MODEL = llama32_3b` (Mistral 24B is optional later, not the v2 PR). v2 writes:

- `adapters/{ACTIVE_MODEL}/v2/`
- `results/{ACTIVE_MODEL}/v2/` (primary generative CSVs + `logprob/` side files + `error_direction.csv`)
- `CAI_Experiment/v2/data/` for the mix JSONL

Never write `adapters/{arm}_final_lora` or `adapters/{ACTIVE_MODEL}/{arm}_final_lora` from v2. Load those v1 LoRAs read-only if v2 adapters are absent. Extra cells: OOD scoring of `data/eval/ood_contemporary_dilemmas.jsonl`; test-time retrieve-and-revise on frozen constitutions; log-prob side scores; error-direction. Forgetting knobs: LoRA r=16 (32 optional), SFT LR 1e-4, optional attention-only modules, format-following replay (~300, not from MMLU/ETHICS/TQA), I/E-R weight clip (0.25, 2.0). Primary MC scorer remains 8-token generation.

## Conditions

| Condition | Weights | Scientific purpose |
|---|---|---|
| Base | unmodified Instruct | Corporate RLHF baseline |
| Control CAI | `control` LoRA | CAI mechanic, pre-1930 prose, distillation artifacts |
| Experimental CAI | `experimental` LoRA | Intrinsic/psychometric corpus worldview |
