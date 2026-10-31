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

## Held-out eval

[`data/eval/open_ended_dilemmas.jsonl`](data/eval/open_ended_dilemmas.jsonl) — 150 original first-person dilemmas (no overlap with treatment or control training prompts). Judge-facing ballots omit the arm mapping.
