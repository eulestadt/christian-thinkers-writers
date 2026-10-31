# 3-arm CAI Colab notebook (Step 5)

Generated for the Llama 3.2 3B Instruct full-matrix run. **GPU training was not launched from this snapshot.** Control and treatment SL-CAI / DPO JSONL are already complete (2,500 rows each), so the notebook is ready to run on Colab.

Empty `CAI_Experiment/{data,adapters,results}` folders were created in the shared Drive **Alignment Research** folder. File upload via the research service account failed (`403` service-account storage quota = 0 — My Drive files would be owned by the SA). Colab runs as Phoenix: the notebook clones **main** (which includes `colab/`) and copies JSONL into those folders (Phoenix's quota), then writes LoRA adapters and CSVs there. SQLite cache paths are always wrapped in `Path(...)` before `/`.

## What to open

- Notebook: [`colab/cai_3way_experiment_and_eval.ipynb`](../colab/cai_3way_experiment_and_eval.ipynb)
- Spec: [`colab/CURSOR_COLAB_SPEC.md`](../colab/CURSOR_COLAB_SPEC.md)
- Colab folder README: [`colab/README.md`](../colab/README.md)

Primary weights: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` on one L4 (24GB) or T4 (16GB). Expected wall time for two SL-CAI+DPO pipelines plus the 4-benchmark suite: **75–90 minutes**. Change one string, `ACTIVE_MODEL = "mistral_small_24b"`, for Mistral Small 24B on L4/A100.

## Three conditions

1. **Base** — unmodified Instruct (no extra SFT/DPO).
2. **Control CAI** — citation-control 40 most-cited English PD works, 1500–1930 ([coverage](cai-control-coverage.md)).
3. **Experimental CAI** — I/E-R intrinsic top 40 ([coverage](cai-deepseek-coverage.md)).

Identical seed `3407`, LoRA rank 32 / alpha 64, SFT LR `2e-4`, DPO LR `5e-6`. Differences in scores are therefore attributable to corpus content, not training mechanics.

## Eval suite

| Benchmark | What it tests |
|---|---|
| ETHICS (`hendrycks/ethics`, 200/subset) | Commonsense, deontology, justice, virtue classification |
| TruthfulQA MC1 (200) | Epistemic honesty |
| MMLU `moral_scenarios`, `moral_disputes`, `philosophy`, `world_religions` | Knowledge held after CAI |
| 150 held-out open-ended dilemmas | Blind 3-way arena; judge ballots omit arm labels |

Training constitutions remain corpus-isolated (no VanderWeele/FAI taxonomies in critique/revision). The eval harness is the external alignment suite.

## Read-only / write split

- Inputs: `data/treatment/*`, `data/control/*`, `data/eval/*`, scoring sqlite (never opened for write).
- Outputs: `adapters/{ACTIVE_MODEL}/` (LoRA only, not merged 6–48GB models) and `results/{ACTIVE_MODEL}/`. Namespacing keeps Llama adapters from being loaded onto Mistral or from skipping a Mistral train.

### SQLite in Colab: GitHub source, local-NVMe open

Drive FUSE is a bad SQLite disk (`database is locked`, metadata touches). Cell 1b **copies** the DB onto `/content/local_db_cache`, `chmod 0444`, and opens `file:...?mode=ro&immutable=1` plus `PRAGMA query_only=ON`. The Drive/git original is never used as the live connection.

**GitHub is the source of truth**, not Drive:

| Why GitHub | Why not Drive-as-SoT |
|---|---|
| `docs/intrinsic-motivation-scores.sqlite` is already LFS-versioned and citable | Service account cannot upload files (My Drive quota 0) |
| Reviewers reproduce the exact SHA | rclone/Desktop copies drift from the experiment pin |
| Training pairs are frozen JSONL | `work_subscores` is long-format (`subscale`,`score`) — it is not an SFT table |

Optional `CAI_Experiment/db/audit.sqlite` is a convenience overlay if you `rclone` it; the notebook still prefers the GitHub LFS file when present. **Do not** load Unsloth SFT/DPO from SQL. Use JSONL. SQLite is for Hoge / I/E-R diagnostics (and optional `state.db` table inspection).

Hot-swap: keep the 4-bit base in VRAM; `model.disable_adapter()` for Base, `model.set_adapter("control"|"experimental")` for the CAI arms.

## Held-out dilemmas

[`colab/data/eval/open_ended_dilemmas.jsonl`](../colab/data/eval/open_ended_dilemmas.jsonl) — 150 original first-person prompts, 0 exact overlap with treatment or control training JSONL.

## Psychometric overlay

[`colab/data/eval/psychometric_kept_works.csv`](../colab/data/eval/psychometric_kept_works.csv) is a read-only export of 527 scored census works (Hoge vs I/E-R intrinsic). The experimental top 40 sit inside that space. The citation-control 40 are mostly *outside* the Christian-writers census by design; title matches inside the 527 are tagged `control_in_census`.
