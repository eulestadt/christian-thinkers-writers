# How the archive feeds the v3 run

The GitHub repository is an archive of several stages that were finished before this Colab session started. This page is the map from those stages to the files a reviewer can actually open, and a frank list of what Colab has not produced yet.

The census begins with Wikipedia’s `Category:Christian writers`. That walk is stored in [`docs/christian-writers.sqlite`](../christian-writers.sqlite): 14,128 writers and 163,063 works. Coverage for the crawl itself is [`christian-writers-coverage-report.md`](../christian-writers-coverage-report.md). Closed-form Hoge and Gorsuch I/E-R scoring, with no LLM in the loop, then scores one English public-domain work per kept author. That database is [`docs/intrinsic-motivation-scores.sqlite`](../intrinsic-motivation-scores.sqlite) (527 works). Ranking is by I/E-R intrinsic only; there is no I−E filter. Scoring coverage and the formula are [`intrinsic-motivation-coverage.md`](../intrinsic-motivation-coverage.md) and [`intrinsic-motivation-scoring.md`](../intrinsic-motivation-scoring.md). The four ranked lists, including the I/E-R top forty that v3 cuts to twenty, are in [`four-top40-lists.md`](../four-top40-lists.md).

From that ranking, two Constitutional AI trees were built with `deepseek-flash` and thinking off. Treatment, in [`cai-pipeline/`](../../cai-pipeline/), uses the I/E-R top forty books, 2,500 prompts, SL-CAI, and RLAIF; see [`cai-deepseek-coverage.md`](../cai-deepseek-coverage.md). Control, in [`cai-pipeline-control/`](../../cai-pipeline-control/), uses the forty most-cited English public-domain works in the same 1500–1930 window, excluding the treatment works and their authors, and runs the same protocol; see [`cai-control-coverage.md`](../cai-control-coverage.md). Extract, critique, revision, and judge prompts stay on author-text vocabulary. They do not use external flourishing taxonomies as a training target.

v2 took those traces, mixed in contemporary dilemmas and UltraChat, and wrote a two-arm mix under [`colab/data/v2/`](../../colab/data/v2/) and [`cai-pipeline-v2/`](../../cai-pipeline-v2/). v3 does not train that mix. It keeps the v2 `draft_0` / `revision_3` traces as source, keeps only the top twenty treatment books, rewrites each kept trace into one weighed turn, drops UltraChat, and size-matches the control to 1,001 rows, stratified by book. The published files are [`colab/data/v3/`](../../colab/data/v3/): 1,001 CAI rows per arm, 300 format-replay rows, Kish ESS 789.94, max weight 2.248. The unexecuted notebook is [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../../colab/cai_3way_experiment_and_eval_v3.ipynb), with notebook notes in [`cai-3way-colab.md`](../cai-3way-colab.md). The whole-experiment drawing is [`pipeline-flowchart.md`](../pipeline-flowchart.md). How v3 differs from v1 and v2 is written out in [`v3-methods.md`](v3-methods.md).

```text
Wikipedia Category:Christian writers
        │
        ▼
docs/christian-writers.sqlite          14,128 writers · 163,063 works
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
            docs/study/   (methods, run log, results)
```

Colab is the stage that is still moving. Session `ctw-v3` is on an NVIDIA L4. Weighted seed 0 and uniform seed 0 have trained. Control seed 0 is in progress. About 1.23 of 240 compute units have been spent. The remaining seven trains, the λ=0.5 evaluation of every seed, the λ=1 Arm A reference, the λ=0 base pass, the fifty-item style set, and an executed notebook are not on GitHub. Those belong in [`v3-run-log.md`](v3-run-log.md), [`results.md`](results.md), and `colab/results/` when they exist. Adapter safetensors stay in the private study artifact store unless Git LFS is later extended to them.

Hard limits that do not change when scores arrive: do not require I−E > 0; do not overwrite v1, v1_fixed, or v2 notebooks or adapter folders; do not train 24B or 72B in this namespace.
