# Christian Thinkers Writers

Public archival corpus for the **Christian Thinkers Writers** project: a versioned, citable backup of the Wikipedia Christian-writers census, I/E-R intrinsic-motivation scores, and Constitutional AI (CAI) **treatment vs citation-control** datasets.

This repository is an **archive**, not a hosted app or database. It exists so Phoenix (and later arXiv reviewers) can open the artifacts on GitHub.

**Start here — coverage pages (in-repo):**

- [Treatment CAI coverage (Steps 0–4, `deepseek-flash`)](docs/cai-deepseek-coverage.md)
- [Control CAI coverage (citation-control, Steps 0–4, `deepseek-flash`)](docs/cai-control-coverage.md)
- [3-arm Colab notebook (Step 5, Llama 3.2 3B)](docs/cai-3way-colab.md)
- [CAI protocol — I/E-R intrinsic top 40](docs/cai-protocol-intrinsic-subset.md)
- [Christian writers census coverage](docs/christian-writers-coverage-report.md)
- [Intrinsic-motivation scoring coverage](docs/intrinsic-motivation-coverage.md)
- [Scoring method (Hoge + Gorsuch I/E-R)](docs/intrinsic-motivation-scoring.md)

Related notes: [project context](docs/project-context.md), [legacy Gemini coverage](docs/cai-gemini-coverage.md) (discarded; regenerated with DeepSeek), [scoring plan](docs/intrinsic-motivation-scoring-plan.md).

## Research question

Does aligning language models with a **Christian, intrinsically motivated virtue ethic** — texts ranked by Gorsuch & McPherson **I/E-R intrinsic** (not an I−E filter) — reduce alignment failures relative to a **citation-matched public-domain control** that is *not* restricted to Christian writers?

Constitution extraction, critique, revision, and judge prompts use **author-text vocabulary only** (no VanderWeele domains, FAI, or other external flourishing taxonomies).

## Treatment vs control

| | Treatment | Control |
|---|---|---|
| Tree | [`cai-pipeline/`](cai-pipeline/) | [`cai-pipeline-control/`](cai-pipeline-control/) |
| Coverage | [`docs/cai-deepseek-coverage.md`](docs/cai-deepseek-coverage.md) | [`docs/cai-control-coverage.md`](docs/cai-control-coverage.md) |
| Book set | I/E-R intrinsic **top 40** English public-domain works from the Christian-writers census (one scored PD work per author) | **40 most-cited** English PD texts in the **same publication-year window (1500–1930)**, excluding those 40 treatment works **and their authors** |
| Model | `deepseek-flash` (thinking off) | same |
| Prompt set | 2,000 elicited + 500 UltraChat = **2,500** | same size, same protocol |
| SL-CAI / RLAIF | 2,500 / 2,500 rows | 2,500 / 2,500 rows |

Treatment examples: Brother Lawrence, *The Practice of the Presence of God*; Ignatius of Loyola, *Spiritual Exercises*; Julian of Norwich, *Revelations of Divine Love*.

Control examples: William James, *The Principles of Psychology*; Darwin, *The Expression of the Emotions*; Shakespeare, *Hamlet* (citation-ranked PD, not a Christian-writers filter).

Step 5 notebook is in [`colab/cai_3way_experiment_and_eval.ipynb`](colab/cai_3way_experiment_and_eval.ipynb) (Base vs Control CAI vs Experimental CAI, adapter hot-swap, ETHICS / TruthfulQA / MMLU / 150-dilemma arena). **GPU training was not executed in this snapshot**; open that notebook on Colab L4/T4 after mounting `CAI_Experiment/`. Coverage: [`docs/cai-3way-colab.md`](docs/cai-3way-colab.md).

## Model (`deepseek-flash`)

- Client: OpenAI-compatible `https://api.deepseek.com`
- Thinking disabled on every call (`thinking: {type: disabled}`)
- Gemini-generated principles, prompts, SL-CAI, and RLAIF labels were discarded and regenerated

## Public-domain texts (PD)

`cai-pipeline/books_raw/` and `cai-pipeline-control/books_raw/` are English **public-domain** full texts (Project Gutenberg, Wikisource, Internet Archive `pdusa`). US PD cutoff used for eligibility is **1930**. Gutenberg headers, where present, remain as distributed.

## Layout

```
README.md
LICENSE                 # PD texts stay PD; code MIT; original docs/artifacts CC BY 4.0
CITATION.cff
docs/                   # markdown coverage + sqlite/csv/jsonl census and scores
cai-pipeline/           # treatment CAI (I/E-R top 40)
cai-pipeline-control/   # citation-control CAI
colab/                  # 3-arm Unsloth notebook + held-out eval prompts
```

Large files (>50 MB) are stored with **Git LFS**:

- `docs/intrinsic-motivation-scores.sqlite` (~89 MB)
- `docs/christian-writers-database.jsonl` (~60 MB)

Clone with Git LFS installed (`git lfs install` then `git clone`).

## Databases and tables

| File | What it is |
|---|---|
| [`docs/christian-writers.sqlite`](docs/christian-writers.sqlite) | Wikipedia `Category:Christian writers` census: 14,128 writers, 163,063 works |
| [`docs/intrinsic-motivation-scores.sqlite`](docs/intrinsic-motivation-scores.sqlite) | Closed-form Hoge + I/E-R scores on one English PD work per kept author (527 scored) |
| [`cai-pipeline/data/state.db`](cai-pipeline/data/state.db) | Treatment CAI checkpoint (extract / SL-CAI / RLAIF) |
| [`cai-pipeline-control/data/state.db`](cai-pipeline-control/data/state.db) | Control CAI checkpoint |

CSV/JSONL exports live next to the census sqlite. A gzip of the scores sqlite is at [`docs/intrinsic-motivation-scores.sqlite.gz`](docs/intrinsic-motivation-scores.sqlite.gz).

## Pipeline outputs (each tree)

- `books_raw/` — 40 source texts
- `data/corpus_manifest.json`
- `data/constitution.json` — clustered critique/revision pairs
- `data/prompts_dataset.jsonl` — 2,500 prompts
- `data/sl_cai_train.jsonl` — Stage 1 SFT
- `data/rlaif_dpo_train.jsonl` — Stage 2 DPO pairs
- `scripts/` — Steps 0–4 (`deepseek-flash`); `05_colab_train.py` points at the 3-arm notebook

## License

- **Public-domain source texts** in `books_raw/` stay in the public domain.
- **Code** (`scripts/`, `requirements.txt`) is [MIT](LICENSE).
- **Original documentation and generated research artifacts** (coverage pages, constitutions, CAI JSONL, census/score tables) are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

No API keys, `.env` files, or agent-only notes are included.
