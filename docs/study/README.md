# v3 study notes

This folder is the public write-up of the Llama 3.2 3B Instruct run: 4-bit QLoRA on a Constitutional AI mix from the twenty English public-domain books with the highest Gorsuch and McPherson I/E-R intrinsic scores. The mix on GitHub, under [`colab/data/v3/`](../../colab/data/v3/), is 1,001 of those rewritten CAI dialogues plus 300 format-replay rows. The notebook that trains and scores it is [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../../colab/cai_3way_experiment_and_eval_v3.ipynb). Earlier notebooks remain in the repository so that work can still be read; v3 writes only under the `v3/` adapter and result folders.

The choices that look like defaults — twenty books, one deliberative turn, a uniform arm, R-DPO, three seeds, WiSE-FT λ=0.5, the 300-row format replay — each exist because the obvious alternative fails in a particular way. [`v3-methods.md`](v3-methods.md) is that argument, tied to the papers (LIMA and AlpaGasus, Guan, Byrd and Lipton, Park, Dodge, Wortsman, InstructGPT, Sharma). [`pipeline-coverage.md`](pipeline-coverage.md) maps the census and the two CAI trees onto the mix. [`v3-run-log.md`](v3-run-log.md) is the live Colab log. [`results.md`](results.md) will hold measured tables when they exist.

As of the last log copy (23 November 2026, 9:06 p.m. Eastern), session `ctw-v3` is on an NVIDIA L4. Weighted seed 0 and uniform seed 0 have finished. Control seed 0 has started. About 1.23 of the 240 compute-unit budget has been spent. There are no evaluation scores yet; empty cells in the result tables are empty because Colab has not written them.

Adapter weight files are about 93 MB each. They stay in the private artifact store because this repository’s Git LFS tracks the census jsonl and the scores sqlite.
