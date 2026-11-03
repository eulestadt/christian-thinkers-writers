# 3-arm CAI Colab experiment

Open [`cai_3way_experiment_and_eval.ipynb`](cai_3way_experiment_and_eval.ipynb) in Google Colab with an **L4 24GB** or **T4 16GB** GPU.

Do **not** start GPU training until `data/control/sl_cai_train.jsonl` and `data/control/rlaif_dpo_train.jsonl` exist (this archive already has 2,500 rows each). Runtime → Run all is ~75–90 minutes for Llama 3.2 3B Instruct (two SL-CAI+DPO arms + 4-benchmark eval). Toggle `ACTIVE_MODEL` in the config cell to `mistral_small_24b` only on L4/A100.

## Drive layout (`CAI_Experiment/`)

```
CAI_Experiment/
├── cai_3way_experiment_and_eval.ipynb
├── data/treatment/     # READ-ONLY copies of cai-pipeline JSONL
├── data/control/       # READ-ONLY copies of cai-pipeline-control JSONL
├── data/eval/          # 150 held-out dilemmas + psychometric overlays
├── adapters/{ACTIVE_MODEL}/  # LoRA checkpoints, namespaced so Llama is not loaded onto Mistral
└── results/{ACTIVE_MODEL}/   # CSVs, figures, arena JSONL, HTML inspector
```

In Colab, add a Drive shortcut named `CAI_Experiment` under My Drive (the empty folder tree already lives in shared **Alignment Research**), or set `DRIVE_CANDIDATES`. A Google service account **cannot** upload JSONL into My Drive (quota 0); the notebook clones this public GitHub repo and copies training/eval files into your mounted Drive on first run (your quota). Adapters and results are written only under `adapters/{ACTIVE_MODEL}/` and `results/{ACTIVE_MODEL}/`.

The notebook copies JSONL to `/content/readonly_cai/` with mode `0444` and never writes `data/treatment`, `data/control`, or sqlite files. SQLite is copied to `/content/local_db_cache` and opened with `mode=ro&immutable=1` (GitHub LFS is the source of truth; Drive FUSE is not a SQLite disk). Adapters stay in VRAM and are hot-swapped (`disable_adapter` / `set_adapter`). Training reads JSONL, not SQL.

## v1_fixed (ETHICS MC scorer; same 2,500-row train)

Open [`cai_3way_experiment_and_eval_v1_fixed.ipynb`](cai_3way_experiment_and_eval_v1_fixed.ipynb) ([Colab](https://colab.research.google.com/github/eulestadt/christian-thinkers-writers/blob/main/colab/cai_3way_experiment_and_eval_v1_fixed.ipynb)). Assemble with `python build_ipynb_v1_fixed.py` (never `build_ipynb.py` / `build_ipynb_v2.py`). Does **not** replace v1 or v2.

The only v1 vs v2 **training** difference is dataset volume (2,500 vs 3,800 rows; ~156 vs ~237 steps). v1_fixed does **not** change v1 HPs, batch, epochs, warmup, LoRA r, or train size, and does **not** pull `colab/data/v2/`. ETHICS eval uses the v2-style MC prompt `(A) No` / `(B) Yes` and `parse_letter` (a missing letter is None, not `parse_01` default 0). TruthfulQA and MMLU stay on the original v1 letter prompts (MMLU **test** only; no validation/dev). Default `RUN_TRAINING=False` loads the existing 2,500-row v1 LoRAs and writes scores to `results/{ACTIVE_MODEL}/v1_fixed/`. If training is turned on, adapters write to `adapters/{ACTIVE_MODEL}/v1_fixed/` only.

Original v1 (unchanged): [Colab](https://colab.research.google.com/github/eulestadt/christian-thinkers-writers/blob/main/colab/cai_3way_experiment_and_eval.ipynb).

## Held-out eval

[`data/eval/open_ended_dilemmas.jsonl`](data/eval/open_ended_dilemmas.jsonl) — 150 original first-person dilemmas (no overlap with treatment or control training prompts). Judge-facing ballots omit the arm mapping.

## v2 (OOD + test-time CAI; does not replace v1)

Open [`cai_3way_experiment_and_eval_v2.ipynb`](cai_3way_experiment_and_eval_v2.ipynb). Assemble with `python build_ipynb_v2.py` (never `build_ipynb.py`, which still writes only the v1 notebook).

```
CAI_Experiment/
├── data/                              # shared READ-ONLY JSONL (v1 + v2)
├── v2/data/{treatment,control}/       # 3,500-row mix + format_replay copies
├── adapters/{ACTIVE_MODEL}/           # v1 LoRAs — do not overwrite
├── adapters/{ACTIVE_MODEL}/v2/        # v2 LoRA writes only
├── results/{ACTIVE_MODEL}/            # v1 CSVs/arena — do not overwrite
└── results/{ACTIVE_MODEL}/v2/         # ood_*, generative_raw, logprob/, error_direction.csv
```

[`data/eval/ood_contemporary_dilemmas.jsonl`](data/eval/ood_contemporary_dilemmas.jsonl) — 150 contemporary first-person prompts (workplace, civic, interpersonal, medical-adjacent, online). Deduped against both `prompts_dataset.jsonl` files, both SFT/DPO JSONL files, and the held-out dilemmas. [`data/eval/ood_train_contemporary_dilemmas.jsonl`](data/eval/ood_train_contemporary_dilemmas.jsonl) is a disjoint 1,000-row train set. v2 `RUN_TRAINING` defaults True and writes only under `adapters/{ACTIVE_MODEL}/v2/`. Default model is Llama 3.2 3B; Mistral 24B stays in `MODEL_PRESETS` as an optional later run. Primary MC metric is still 8-token generation; log-prob scores are side files under `results/{ACTIVE_MODEL}/v2/logprob/`. Test-time retrieve-and-revise uses the frozen 40-book constitutions (`all-MiniLM-L6-v2`, k=3). Optional `IER_RETRIEVAL_WEIGHT` (default 0) can reweight experimental retrieval by I/E-R; it does not require I−E>0.
