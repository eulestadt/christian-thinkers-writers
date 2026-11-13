# 3-arm CAI Colab notebook

The current run is **v3**. Open [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../colab/cai_3way_experiment_and_eval_v3.ipynb) on a Colab GPU (L4 24GB or T4 16GB). The nine QLoRA trains stay there. This archive does not launch them. Cell 1 stops if `colab/data/v3` is missing.

The v1 notebook ([`colab/cai_3way_experiment_and_eval.ipynb`](../colab/cai_3way_experiment_and_eval.ipynb)), the v1_fixed notebook, and the v2 notebook ([`colab/cai_3way_experiment_and_eval_v2.ipynb`](../colab/cai_3way_experiment_and_eval_v2.ipynb)) stay in the repo. They are not the run to open.

- Spec: [`colab/CURSOR_COLAB_SPEC.md`](../colab/CURSOR_COLAB_SPEC.md)
- Colab folder README: [`colab/README.md`](../colab/README.md)
- Mix: [`colab/data/v3/`](../colab/data/v3/)

Primary weights: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`. v3 trains Llama 3.2 3B only.

## How v3 differs from v1 and v2

v1 trains two arms on the book dialogues with one seed. Adapters go under the model name. The hyperparameters in the v1 notebook are seed `3407`, LoRA rank 32 / alpha 64, SFT LR `2e-4`, DPO LR `5e-6`.

v2 keeps all 40 books, adds contemporary dilemmas and UltraChat, clips weights to 0.25–2, and draws rows with replacement. It still trains two arms with one seed, and it writes under a v2 folder.

v3 keeps the **20 books with the highest `ier_intrinsic` scores** (ranks 1–20 of the I/E-R list in [`four-top40-lists.md`](four-top40-lists.md); no I−E filter). It rewrites the existing v2 `draft_0` / `revision_3` traces into one user message and one weighed assistant reply. UltraChat is not in the mix. The shared **300-row format replay** stays, at weight 1. That slice is ARC-Easy train and synthetic letter-format items, not MMLU, ETHICS, or TruthfulQA.

Three arms share one trainer:

| Arm | What it is | Weight |
|---|---|---|
| A, weighted | the kept treatment dialogues | `w = s / mean(s)`, no clip, mean CAI weight 1 |
| B, uniform | the same rows and the same row ids as A | every weight 1 |
| C, control | the same CAI count, stratified by book, no score filter | every weight 1 |

Every arm uses R-DPO with **α=0.01** and **β=0.1** after one epoch of weighted SFT. The length term sits inside the logistic, and the row weight multiplies the whole per-sequence SFT loss and the whole per-pair R-DPO loss. Rows are not drawn with replacement. Rejected answers alternate between the unrevised draft and a dogmatic rewrite (bidirectional negatives) on every arm, including uniform and control. Seeds are **0, 1, and 2**. Arm A and arm B use the same shuffle at a given seed, so the only difference between them is `w`.

The reported score is the mean of seeds 0, 1, and 2 at **WiSE-FT λ=0.5**. Each seed is listed beside the mean. λ=1 on arm A is a reference row. λ=0 is the base model, scored once. The primary metric is still 8-token generation plus `parse_letter`. Paired bootstrap 95% intervals cover arm A minus arm B and arm A minus arm C. Style drift and dogmatic-versus-reflective judgments use 50 held-out contemporary dilemmas at λ=0.5.

Scores stay trainer metadata. They are not written into prompts. Critique and revision text stays on author vocabulary: no VanderWeele, FAI, or flourishing terms.

## Published mix

| Arm | CAI rows | Format replay | SFT / DPO rows |
|---|---:|---:|---:|
| weighted | 1,001 | 300 | 1,301 |
| uniform | 1,001 | 300 | 1,301 |
| control | 1,001 | 300 | 1,301 |

Kish ESS of the weighted CAI slice: **789.94**. Max weight: **2.248**. Min weight: 0.571. Mean CAI weight: 1. Rejected CAI pairs: 501 unrevised drafts and 500 dogmatic rewrites on each arm. Summary: [`colab/data/v3/mix_summary.json`](../colab/data/v3/mix_summary.json).

Writes go only to `adapters/{ACTIVE_MODEL}/v3/` and `results/{ACTIVE_MODEL}/v3/`.

## Read-only / write split

- Inputs: `colab/data/v3/{weighted,uniform,control}/`, the shared format-replay files, `data/eval/*`, and the scoring sqlite. Never opened for write.
- Outputs: `adapters/{ACTIVE_MODEL}/v3/` (LoRA only) and `results/{ACTIVE_MODEL}/v3/`.

Empty `CAI_Experiment/{data,adapters,results}` folders live in the shared Drive **Alignment Research** folder. A research service account cannot upload into My Drive (quota 0). Colab clones this repo and copies JSONL onto the mounted Drive (the Colab user's quota). SQLite paths are wrapped in `Path(...)` before `/`.

### SQLite in Colab: GitHub source, local-NVMe open

Drive FUSE is a bad SQLite disk (`database is locked`, metadata touches). The notebook **copies** the DB onto `/content/local_db_cache`, `chmod 0444`, and opens `file:...?mode=ro&immutable=1` plus `PRAGMA query_only=ON`. The Drive or git original is never the live connection.

**GitHub is the source of truth**, not Drive:

| Why GitHub | Why not Drive-as-SoT |
|---|---|
| `docs/intrinsic-motivation-scores.sqlite` is already LFS-versioned and citable | Service account cannot upload files (My Drive quota 0) |
| Reviewers reproduce the exact SHA | rclone/Desktop copies drift from the experiment pin |
| Training pairs are frozen JSONL | `work_subscores` is long-format (`subscale`,`score`) — it is not an SFT table |

Optional `CAI_Experiment/db/audit.sqlite` is a convenience overlay if you `rclone` it; the notebook still prefers the GitHub LFS file when present. **Do not** load Unsloth SFT/DPO from SQL. Use JSONL. SQLite is for Hoge / I/E-R diagnostics (and optional `state.db` table inspection).

At eval, the 4-bit base stays in VRAM. WiSE-FT λ=0.5 scales the active v3 adapter. λ=0 disables the adapter and is the base model.

## Held-out dilemmas

[`colab/data/eval/open_ended_dilemmas.jsonl`](../colab/data/eval/open_ended_dilemmas.jsonl) — 150 original first-person prompts. The style cell scores 50 rows from [`colab/data/eval/ood_contemporary_dilemmas.jsonl`](../colab/data/eval/ood_contemporary_dilemmas.jsonl) and refuses the cell if a prompt is also in the v3 training mix.

## Psychometric overlay

[`colab/data/eval/psychometric_kept_works.csv`](../colab/data/eval/psychometric_kept_works.csv) is a read-only export of 527 scored census works (Hoge vs I/E-R intrinsic). The v3 treatment books are ranks 1–20 of that I/E-R list. The citation-control 40 are mostly *outside* the Christian-writers census by design; title matches inside the 527 are tagged `control_in_census`. v3 subsamples the control dialogues to 1,001 rows, stratified across those 40 books.

## Earlier notebooks

These stay on disk. Do not treat them as the current run, and do not overwrite their adapter folders from the v3 notebook.

**v1.** [`colab/cai_3way_experiment_and_eval.ipynb`](../colab/cai_3way_experiment_and_eval.ipynb). Base, citation-control CAI, and experimental CAI on the full 2,500-row JSONL. One seed. Writes `adapters/{ACTIVE_MODEL}/` and `results/{ACTIVE_MODEL}/`.

**v2.** [`colab/cai_3way_experiment_and_eval_v2.ipynb`](../colab/cai_3way_experiment_and_eval_v2.ipynb). All 40 books, UltraChat, OOD-train rows, weight clip (0.25, 2.0), sampling with replacement. Writes `adapters/{ACTIVE_MODEL}/v2/` and `results/{ACTIVE_MODEL}/v2/`. Mix notes: [`colab/data/v2/README.md`](../colab/data/v2/README.md).
