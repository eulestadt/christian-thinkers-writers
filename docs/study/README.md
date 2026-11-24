# v3 study notes

This folder is the public write-up of the current training run. The run uses Llama 3.2 3B Instruct, fine-tuned with 4-bit QLoRA, on a Constitutional AI mix built from the twenty English public-domain books with the highest Gorsuch and McPherson I/E-R intrinsic scores. Nothing larger is being trained here: there is no Mistral 24B run and no Qwen2-VL-72B run in this study.

The mix itself is already on GitHub, under [`colab/data/v3/`](../../colab/data/v3/). The notebook that trains and scores it is [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../../colab/cai_3way_experiment_and_eval_v3.ipynb). The v1 notebook, the v1_fixed notebook, and the v2 notebook stay in the repository so earlier work can still be read; they are not the run to open, and this study does not overwrite their adapters.

As of the last log copy (23 November 2026, 9:06 p.m. Eastern), a Colab session named `ctw-v3` is on an NVIDIA L4. Weighted seed 0 and uniform seed 0 have finished. Control seed 0 has started and has not finished. About 1.23 of the 240 compute-unit budget has been spent. There are still no evaluation scores. Empty cells in the result tables are empty on purpose; they are not zeros, and they are not numbers borrowed from v1 or v2.

Read these pages in this order if you are coming in cold:

[`v3-methods.md`](v3-methods.md) explains how v3 differs from v1 and v2, including the top-twenty cutoff, the one-turn rewrite, the three arms, R-DPO, the three seeds, and WiSE-FT λ=0.5. [`pipeline-coverage.md`](pipeline-coverage.md) maps the census, the scoring database, the treatment and control CAI trees, and this Colab run. [`v3-run-log.md`](v3-run-log.md) is the live session log: provision, clone, each train, compute units remaining. [`results.md`](results.md) will hold measured tables when Colab writes them; today it only records that two adapters exist in the private artifact store and that scores do not.

When later Colab artifacts appear (the rest of the nine trains, eval CSVs, an executed notebook), they belong in this folder and under `colab/results/`. Adapter weight files are about 93 MB each and are not committed here, because this repository’s Git LFS tracks only the census jsonl and the scores sqlite.
