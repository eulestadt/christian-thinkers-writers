# How the archive feeds the v3 run

The GitHub repository is an archive of stages that were finished before this Colab session. This page is the map from those stages to the files a reviewer can open, and a note on what Colab is still producing.

The census begins with Wikipedia’s `Category:Christian writers`, stored in [`docs/christian-writers.sqlite`](../christian-writers.sqlite) (14,128 writers, 163,063 works); crawl coverage is [`christian-writers-coverage-report.md`](../christian-writers-coverage-report.md). Closed-form Hoge and Gorsuch I/E-R scoring, with no LLM in the loop, then scores one English public-domain work per kept author into [`docs/intrinsic-motivation-scores.sqlite`](../intrinsic-motivation-scores.sqlite) (527 works). Ranking is by I/E-R intrinsic, because that is the construct the published scale measures; see [`intrinsic-motivation-coverage.md`](../intrinsic-motivation-coverage.md), [`intrinsic-motivation-scoring.md`](../intrinsic-motivation-scoring.md), and the four lists in [`four-top40-lists.md`](../four-top40-lists.md).

Two Constitutional AI trees were built from that ranking with `deepseek-flash`, thinking off, following Bai et al. (2022): principles as critique instructions, author-text vocabulary only. Treatment, in [`cai-pipeline/`](../../cai-pipeline/), is the I/E-R top forty; see [`cai-deepseek-coverage.md`](../cai-deepseek-coverage.md). Control, in [`cai-pipeline-control/`](../../cai-pipeline-control/), is the forty most-cited English public-domain works in the same 1500–1930 window, excluding treatment works and authors, so the contrast is corpus rather than “any old text”; see [`cai-control-coverage.md`](../cai-control-coverage.md).

v2 kept those traces as a forty-book mix. v3 uses the v2 `draft_0` / `revision_3` traces as source and then applies the LIMA / AlpaGasus cut: ranks 1–20 only, each trace rewritten into one deliberative turn (Guan et al.), control size-matched to 1,001 rows stratified by book. The files in [`colab/data/v3/`](../../colab/data/v3/) are 1,001 CAI rows per arm plus 300 format-replay rows (InstructGPT-style letter format; Kish ESS 789.94, max weight 2.248). The notebook is [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../../colab/cai_3way_experiment_and_eval_v3.ipynb); notebook notes are [`cai-3way-colab.md`](../cai-3way-colab.md); the drawing is [`pipeline-flowchart.md`](../pipeline-flowchart.md). Why each of those cuts is there is [`v3-methods.md`](v3-methods.md).

```text
Wikipedia Category:Christian writers
        │
        ▼
docs/christian-writers.sqlite          14,128 writers · 163,063 works
        │  closed-form Hoge + Gorsuch I/E-R (no LLM)
        ▼
docs/intrinsic-motivation-scores.sqlite   527 scored English PD works
        │  rank by ier_intrinsic
        ▼
┌───────────────────────┬──────────────────────────┐
│ Treatment top 40      │ Citation-control 40      │
│ cai-pipeline/         │ cai-pipeline-control/    │
│ deepseek-flash        │ same protocol            │
│ 2,500 prompts each    │ 2,500 prompts each       │
└───────────┬───────────┴────────────┬─────────────┘
            │ v2 traces as source    │
            ▼                        ▼
     top 20 books (LIMA / AlpaGasus)  size-match to N=1,001
     one deliberative turn            stratified by book
            │                        │
            └──────────┬─────────────┘
                       ▼
            colab/data/v3/{weighted,uniform,control}/
            1,001 CAI + 300 format replay
                       │
                       ▼
            Colab: Llama 3.2 3B QLoRA × 3 arms × 3 seeds
            R-DPO α=0.01 β=0.1; report WiSE-FT λ=0.5
                       │
                       ▼
            docs/study/   (methods, run log, results)
```

Colab is still moving. Session `ctw-v3` is on an NVIDIA L4. Weighted seed 0 and uniform seed 0 have trained. Control seed 0 is in progress. About 1.23 of 240 compute units have been spent. Remaining trains, λ=0.5 evaluation, the λ=1 Arm A reference, the λ=0 base pass, and the fifty-item style set belong in [`v3-run-log.md`](v3-run-log.md) and [`results.md`](results.md) when they exist. Adapter safetensors stay in the private study store because Git LFS here tracks the census jsonl and the scores sqlite.
