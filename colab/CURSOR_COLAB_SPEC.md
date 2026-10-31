# Cursor specification: 3-arm CAI fine-tune and multi-benchmark Colab notebook

## Objective

Self-contained Jupyter notebook `cai_3way_experiment_and_eval.ipynb` for Google Colab (L4 24GB or A100 40GB). Primary model: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`. Toggle `ACTIVE_MODEL = "mistral_small_24b"` for Mistral Small 24B Instruct.

## Strict constraints

1. **Read-only treatment/control data.** Never modify `data/treatment/`, `data/control/`, or any `.sqlite`. Open input JSONL with `"r"` only. The notebook copies JSONL into a chmod `0444` scratch dir before training.
2. **Identical hyperparameters.** Both arms use `seed=3407`, LoRA `r=32`, `lora_alpha=64`, SFT LR `2e-4`, DPO LR `5e-6`, same batch/accum as the active model preset.
3. **Do not launch GPU** until control `sl_cai_train.jsonl` and `rlaif_dpo_train.jsonl` exist (2,500 rows). Skip training if `adapters/{arm}_final_lora` already exists.
4. **No merged 6–48GB dumps.** Keep the 4-bit base in VRAM; attach/detach LoRA adapters.
5. **Writes** only under `adapters/` and `results/`.

## Drive layout (`/content/drive/MyDrive/CAI_Experiment/`)

```
data/treatment/sl_cai_train.jsonl          # 2,500  READ-ONLY
data/treatment/rlaif_dpo_train.jsonl       # 2,500  READ-ONLY
data/control/sl_cai_train.jsonl            # 2,500  READ-ONLY
data/control/rlaif_dpo_train.jsonl         # 2,500  READ-ONLY
data/eval/open_ended_dilemmas.jsonl        # 150 held-out
data/eval/psychometric_kept_works.csv      # 527 census overlay
adapters/{experimental,control}_final_lora
results/objective_alignment_benchmarks.csv
results/blind_3way_arena_outputs.jsonl     # includes mapping (researcher)
results/blind_3way_arena_ballots.jsonl     # no mapping (judge)
results/cai_3arm_master_diagnostics.png
results/qualitative_3arm_inspector.html
```

## Notebook cells

1. Install Unsloth/TRL; mount Drive; verify read-only row counts.
2. `train_cai_arm` for experimental then control (SFT → DPO).
3. Load base once; `PeftModel` multi-adapter; `generate_with_arm`.
4. ETHICS (4 subsets × 200), TruthfulQA MC1 (200), MMLU moral/philosophy (≤150).
5. 150-dilemma blind 3-way arena (temperature 0.3, shuffled Option_1/2/3).
6. Four-panel 300 DPI figure (radar, Δ bars, DPO margins, psychometric KDE).
7. Lorenz, SFT loss overlay, bootstrap CIs, optional UMAP, win-rate.
8. Optional DeepSeek judge (`RUN_JUDGE=False` by default).
9. HTML qualitative inspector.

## Conditions

| Condition | Weights | Scientific purpose |
|---|---|---|
| Base | unmodified Instruct | Corporate RLHF baseline |
| Control CAI | `control` LoRA | CAI mechanic, pre-1930 prose, distillation artifacts |
| Experimental CAI | `experimental` LoRA | Intrinsic/psychometric corpus worldview |
