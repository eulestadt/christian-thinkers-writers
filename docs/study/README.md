# v3 study notes

This folder is the public write-up of the Llama 3.2 3B Instruct run: 4-bit QLoRA on a Constitutional AI mix from the twenty English public-domain books with the highest Gorsuch and McPherson I/E-R intrinsic scores. The mix on GitHub, under [`colab/data/v3/`](../../colab/data/v3/), is 1,001 of those rewritten CAI dialogues plus 300 format-replay rows. The notebook that trains and scores it is [`colab/cai_3way_experiment_and_eval_v3.ipynb`](../../colab/cai_3way_experiment_and_eval_v3.ipynb). Earlier notebooks remain in the repository so that work can still be read; v3 writes only under the `v3/` adapter and result folders.

The choices that look like defaults — twenty books, one deliberative turn, a uniform arm, R-DPO, three seeds, WiSE-FT λ=0.5, the 300-row format replay — each exist because the obvious alternative fails in a particular way. [`v3-methods.md`](v3-methods.md) is that argument, tied to the papers (LIMA and AlpaGasus, Guan, Byrd and Lipton, Park, Dodge, Wortsman, InstructGPT, Sharma). [`pipeline-coverage.md`](pipeline-coverage.md) maps the census and the two CAI trees onto the mix. [`v3-run-log.md`](v3-run-log.md) is the Colab log. [`results.md`](results.md) has the measured tables.

Session `ctw-v3` finished on an NVIDIA L4: all nine trains, then eval. **10.12 of ~240 compute units** spent. Arms A, B, and C are tied on the primary suite (pooled A−B 0.000 pp, 95% CI [−0.417, 0.438]; A−C −0.312 pp [−0.917, 0.271]). The shared shift versus the base is the training signal, not I/E-R weights.

Adapter weight files are about 93 MB each. They stay in the private artifact store because this repository’s Git LFS tracks the census jsonl and the scores sqlite.

## v4

v4 asks the question this run left open: the same size mix, read open, read sealed, or read from the neighbor shelf. The write-up is [`../arxiv-v4-study/`](../arxiv-v4-study/) (paper, results, run log). Mixes and analysis JSON are in [`v4/`](v4/). The v3 notes above were not rewritten.
