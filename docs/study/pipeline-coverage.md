# Pipeline coverage (v3 study)

How the archive pieces fit the current run, and what is still missing.

```text
Wikipedia Category:Christian writers
        │
        ▼
docs/christian-writers.sqlite          14,128 writers, 163,063 works
        │  closed-form Hoge + Gorsuch I/E-R (no LLM)
        ▼
docs/intrinsic-motivation-scores.sqlite   527 scored English PD works
        │  rank by ier_intrinsic only (no I−E filter)
        ▼
┌───────────────────────┬──────────────────────────┐
│ Treatment top 40      │ Citation-control 40      │
│ cai-pipeline/         │ cai-pipeline-control/    │
│ deepseek-flash        │ same protocol            │
│ 2,500 prompts each    │ 2,500 prompts each       │
└───────────┬───────────┴────────────┬─────────────┘
            │ v2 traces (kept)       │
            ▼                        ▼
     top 20 books              size-match to N=1,001
     rewrite → one weighed turn     stratified by book
            │                        │
            └──────────┬─────────────┘
                       ▼
            colab/data/v3/{weighted,uniform,control}/
            1,001 CAI + 300 format replay  (no UltraChat)
                       │
                       ▼
            Colab: Llama 3.2 3B QLoRA × 3 arms × 3 seeds
            R-DPO α=0.01 β=0.1; report WiSE-FT λ=0.5
                       │
                       ▼
            docs/study/   ← this folder (log, methods, results, email)
```

## Covered now (on GitHub)

| Stage | Artifact | Coverage page |
|---|---|---|
| Census | [`docs/christian-writers.sqlite`](../christian-writers.sqlite) | [`christian-writers-coverage-report.md`](../christian-writers-coverage-report.md) |
| I/E-R scoring | [`docs/intrinsic-motivation-scores.sqlite`](../intrinsic-motivation-scores.sqlite) | [`intrinsic-motivation-coverage.md`](../intrinsic-motivation-coverage.md), [`intrinsic-motivation-scoring.md`](../intrinsic-motivation-scoring.md) |
| Four top-40 lists | ranks 1–20 used by v3 | [`four-top40-lists.md`](../four-top40-lists.md) |
| Treatment CAI (Steps 0–4) | [`cai-pipeline/`](../../cai-pipeline/) | [`cai-deepseek-coverage.md`](../cai-deepseek-coverage.md) |
| Control CAI (Steps 0–4) | [`cai-pipeline-control/`](../../cai-pipeline-control/) | [`cai-control-coverage.md`](../cai-control-coverage.md) |
| Whole-experiment map | mermaid + PNG | [`pipeline-flowchart.md`](../pipeline-flowchart.md) |
| v3 mix | [`colab/data/v3/`](../../colab/data/v3/) — 1,001 CAI rows, 300 replay, Kish ESS **789.94**, max weight **2.248** | [`colab/data/v3/README.md`](../../colab/data/v3/README.md), [`mix_summary.json`](../../colab/data/v3/mix_summary.json) |
| v3 notebook (unexecuted) | [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../../colab/cai_3way_experiment_and_eval_v3.ipynb) | [`cai-3way-colab.md`](../cai-3way-colab.md) |
| Isolation | author-text vocabulary only in extract / critique / revision / judge | README, protocol, this study |
| Methods vs v1 / v2 | this study | [`v3-methods.md`](v3-methods.md) |

v2 mix files stay in [`colab/data/v2/`](../../colab/data/v2/) and [`cai-pipeline-v2/`](../../cai-pipeline-v2/). They are inputs to the v3 rewrite, not the run to train.

## Not covered yet (Colab)

| Stage | Expected landing place | Status |
|---|---|---|
| Session provision (L4 or T4, `ctw-v3`) | [`v3-run-log.md`](v3-run-log.md) | pending |
| Nine QLoRA trains (`{weighted,uniform,control}_seed{0,1,2}_final_lora`) | `adapters/{ACTIVE_MODEL}/v3/` (Drive + later git if published) | pending |
| Eval at λ=0.5 (all seeds), λ=1 Arm A, λ=0 base | [`results.md`](results.md), `colab/results/` | pending |
| Executed notebook | `colab/` or this folder | pending |
| Compute-unit log | [`v3-run-log.md`](v3-run-log.md) | pending |
| Email from measured numbers | [`email-vanderweele.md`](email-vanderweele.md) | skeleton only |

## Isolation (hard)

- No VanderWeele, FAI, or flourishing strings in training prompts or judges.
- No I−E > 0 requirement.
- Do not overwrite v1, v1_fixed, or v2 notebooks or adapter folders.
