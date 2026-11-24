# v3 study (Llama 3.2 3B)

This folder is the living write-up of the **v3** Colab run. Phoenix can watch it grow: mix and methods are here now; the run log, result tables, executed notebook, and email fill in as Colab finishes.

**Status: L4 session `ctw-v3` is running. Weighted seed 0 and uniform seed 0 have trained (~1.23 of 240 CU). Eval scores are not on GitHub yet. Do not treat pending cells as results.**

| File | What it is |
|---|---|
| [`v3-methods.md`](v3-methods.md) | What is unique vs v1 and v2 |
| [`pipeline-coverage.md`](pipeline-coverage.md) | Census → scoring → CAI → mix → Colab → this folder |
| [`v3-run-log.md`](v3-run-log.md) | Provision, clone, each train, each eval, compute units |
| [`results.md`](results.md) | Measured tables (empty until Colab writes them) |
| [`email-vanderweele.md`](email-vanderweele.md) | Draft to Dr. Tyler VanderWeele (results pending; do not send) |

Notebook: [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../../colab/cai_3way_experiment_and_eval_v3.ipynb). Mix: [`colab/data/v3/`](../../colab/data/v3/). The v1, v1_fixed, and v2 notebooks stay untouched.

When Colab artifacts land, they copy here (and under `colab/results/` / `colab/` for the executed notebook). The email is then rewritten from the measured numbers.
