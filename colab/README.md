# 3-arm CAI Colab experiment

The current run is **v3**. Open [`cai_3way_experiment_and_eval_v3.ipynb`](cai_3way_experiment_and_eval_v3.ipynb) in Google Colab with an **L4 24GB** or **T4 16GB** GPU. Assemble it with `python build_ipynb_v3.py`. That script does not write the v1 notebook, the v1_fixed notebook, or the v2 notebook.

v1 trains two arms on the book dialogues with one seed. v2 keeps all 40 books, adds contemporary dilemmas and UltraChat, clips weights to 0.25–2, and draws rows with replacement. v3 keeps the 20 books with the highest `ier_intrinsic` scores and rewrites the existing v2 `draft_0` / `revision_3` traces into one user message and one weighed assistant reply. UltraChat is not in the mix. The shared 300-row format replay stays, at weight 1.

Three arms share one trainer. Arm A uses `w = s / mean(s)` with no clip (mean CAI weight 1). Arm B uses the same rows and row ids with every weight set to 1. Arm C is a size-matched control, stratified by book, weights 1. Every arm uses R-DPO with α=0.01 and β=0.1. Rejected answers alternate between the unrevised draft and a dogmatic rewrite. Seeds are 0, 1, and 2. The reported score is the mean of those seeds at WiSE-FT λ=0.5.

Published mix: **1,001** CAI rows per arm plus **300** format-replay rows (1,301 SFT rows and 1,301 DPO rows). Kish ESS **789.94**. Max weight **2.248**. Files: [`data/v3/`](data/v3/). Cell 1 stops if those files are missing. The nine QLoRA trains run here on Colab; they are not launched from the archive VM. Llama 3.2 3B only. Writes go to `adapters/{ACTIVE_MODEL}/v3/` and `results/{ACTIVE_MODEL}/v3/`.

## Drive layout for v3 (`CAI_Experiment/`)

```
CAI_Experiment/
├── cai_3way_experiment_and_eval_v3.ipynb
├── v3/data/{weighted,uniform,control}/   # 1,301-row mixes, READ-ONLY after copy
├── data/eval/                            # held-out dilemmas + psychometric overlays
├── adapters/{ACTIVE_MODEL}/v3/           # nine LoRAs: arm × seed
└── results/{ACTIVE_MODEL}/v3/            # λ=0.5 means, paired bootstrap, style scores
```

In Colab, add a Drive shortcut named `CAI_Experiment` under My Drive (the empty folder tree already lives in shared **Alignment Research**), or set `DRIVE_CANDIDATES`. A Google service account **cannot** upload JSONL into My Drive (quota 0); the notebook clones this public GitHub repo and copies training/eval files into your mounted Drive on first run (your quota). v3 writes adapters and results only under `adapters/{ACTIVE_MODEL}/v3/` and `results/{ACTIVE_MODEL}/v3/`.

The notebook copies JSONL to `/content/readonly_cai/` with mode `0444` and never writes `data/treatment`, `data/control`, or sqlite files. SQLite is copied to `/content/local_db_cache` and opened with `mode=ro&immutable=1` (GitHub LFS is the source of truth; Drive FUSE is not a SQLite disk). Adapters stay in VRAM and are hot-swapped (`disable_adapter` / `set_adapter`). Training reads JSONL, not SQL.

## Held-out eval

[`data/eval/open_ended_dilemmas.jsonl`](data/eval/open_ended_dilemmas.jsonl) — 150 original first-person dilemmas (no overlap with treatment or control training prompts). Judge-facing ballots omit the arm mapping.

## Earlier notebooks (not the current run)

### v1

Open [`cai_3way_experiment_and_eval.ipynb`](cai_3way_experiment_and_eval.ipynb). Two arms on the 2,500-row book JSONL, one seed. Do not start that GPU training until `data/control/sl_cai_train.jsonl` and `data/control/rlaif_dpo_train.jsonl` exist (this archive already has 2,500 rows each). Runtime → Run all is about 75–90 minutes for Llama 3.2 3B Instruct. Toggle `ACTIVE_MODEL` to `mistral_small_24b` only on L4/A100, and only in the v1 notebook. v3 does not train Mistral.

The v1 tree below is historical. v3 does not write these adapter or result paths.

```
CAI_Experiment/
├── cai_3way_experiment_and_eval.ipynb
├── data/treatment/     # READ-ONLY copies of cai-pipeline JSONL
├── data/control/       # READ-ONLY copies of cai-pipeline-control JSONL
├── data/eval/          # 150 held-out dilemmas + psychometric overlays
├── adapters/{ACTIVE_MODEL}/  # v1 LoRA checkpoints
└── results/{ACTIVE_MODEL}/   # v1 CSVs, figures, arena JSONL, HTML inspector
```

### v2 (OOD + test-time CAI; does not replace v1, and is not the current run)

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
